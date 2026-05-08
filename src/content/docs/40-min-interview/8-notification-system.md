---
title: "Notification System"
description: "Notification System"
---

“Design a notification system that delivers messages to users across web, mobile push, email, SMS. 100M users, support for personalized batching (‘don’t send more than 3 notifications/hour to a user’), preference management, retry.”

---


## 1. Reframing

The mid-level read of this question is "fan-out to four channels with a Kafka queue and per-user prefs in Postgres." That answer is wrong because it locates the complexity in the wrong layer. **The hard problem is not delivering a message to FCM; it is deciding what to send, to whom, on which channel, at what time, batched with what other notifications, under whose consent.** The channel adapters are commodity glue. The orchestration layer — the per-recipient decision engine that consumes events and emits send commands — is where staff-level design lives. **[STAFF SIGNAL: orchestration-as-central]**

Two architectural facts dominate everything downstream:

**Fact 1: per-user state is the scaling axis, not events/sec.** Every send decision requires reading preferences, recent delivery history, current batch state, presence, device tokens, and consent flags for the recipient. At 100M users, that's hundreds of GB of hot state, and every event multiplied by fan-out hits it. **[STAFF SIGNAL: per-user-state-as-scaling-axis]**

**Fact 2: channels are not interchangeable.** Push has <1s latency but no delivery guarantee. Email tolerates minutes but has deliverability/reputation concerns and bounce semantics. SMS costs $0.005–0.05 per message and is regulated under TCPA. Web push only works when the browser is open. In-app is real-time when online, queued otherwise. A design that treats them as a uniform "send" interface hides the entire problem. **[STAFF SIGNAL: channel-heterogeneity discipline]**

The third fact, which the prompt's batching requirement forces: **"don't send more than 3/hour to a user" cannot be enforced by a stateless check before each send** — by the time you check, more notifications have been generated and concurrent workers race. It requires a stateful per-user scheduler with serialized decisions per user. This shifts the architecture from stateless workers consuming a queue to stateful per-user orchestration. **[STAFF SIGNAL: stateful-scheduler-for-batching]**

## 2. Scoping

I'm committing to the following before designing: **[STAFF SIGNAL: scope negotiation]**

- **Notification mix**: ~70% social (someone followed/liked/replied), ~25% transactional (security alerts, order status, password resets), ~5% marketing. Different SLOs and consent rules per class.
- **Workload shape**: bimodal. Steady stream of per-user social/transactional events (~hundreds to low thousands/sec) plus occasional viral fan-out bursts (one event → 10M–100M recipients) that dominate peak load.
- **Build vs buy on channel adapters**: I'm building the orchestration, scheduler, and preference layer in-house, and using third-party providers (FCM, APNS, SES/SendGrid, Twilio) for the actual channel delivery. Building APNS connection pooling at scale is a solved problem and not where engineering effort earns return.
- **Geographic footprint**: multi-region (US, EU, APAC) for latency and data residency. Preferences and delivery state are regionally partitioned by user home region. GDPR forces EU isolation.
- **Latency SLOs by class**: transactional p95 < 30s end-to-end; social p95 < 2 min; marketing best-effort within hours.
- **Out of scope**: notification content authoring/CMS, A/B test experiment definition (we integrate with an existing experimentation platform), user-facing notification feed UI, real-time chat (different system).

What I'm explicitly **rejecting** as scope: this is not a feed system. Lossy delivery is not acceptable for transactional. Quality (relevance, fatigue) matters more than raw throughput.

## 3. Capacity Math

**[STAFF SIGNAL: capacity math]**

Assumptions: 100M registered users, 30M DAU, average 5 notifications/user/day (suppressed/delivered combined).

```
Metric                                Value             Notes
------------------------------------- ----------------- ----------------------------------
Notifications generated/day           150M              30M DAU * 5
Avg generation rate                   ~1,700/sec        150M / 86,400
Peak (3x avg)                         ~5,000/sec        diurnal + bursts
Viral fan-out burst                   10M–100M          one event, processed over 5–10 min
Decisions/sec at peak with fan-out    50,000–100,000    each recipient = one decision
State reads per decision              3–5               prefs, freq counter, presence, tokens
Cache reads/sec at peak               ~300,000          mandatory cache; DB cannot absorb
Per-user preference state             ~1 KB             100M * 1KB = 100 GB
Per-user delivery state               ~500 B            counters, last-N timestamps; 50 GB
Per-user pending batch                ~200 B avg        50M active * 200B = 10 GB
Total hot state                       ~160 GB           fits in a sharded Redis cluster
Device tokens                         ~3 per user       300M tokens; ~30 GB
Push delivery cost                    ~$0               FCM/APNS free
Email cost                            $0.0001/msg       SES; ~$15/day at 150M*0.7 channel mix
SMS cost                              $0.01/msg         expensive; restrict to txn + opted-in
SMS daily cost if 5% of notifs        $75K/day          forces aggressive gating policy
```

The SMS line is the one that drives a real architectural constraint: **SMS is expensive enough that it must be gated by class (transactional only by default), by per-user opt-in, and by a hard global daily budget circuit breaker.** A bug that sends marketing via SMS to all DAU is a six-figure incident. **[STAFF SIGNAL: cost-explicit]**

Fan-out burst arithmetic: one event with 50M recipients, at 50K decisions/sec sustained, is 1,000 seconds (~17 minutes). I will spread non-urgent fan-out over 5–10 minutes deliberately rather than try to do it instantly — saves capacity and is invisible to users.

## 4. High-Level Architecture

```
                         ┌─────────────────────────┐
   producer services →→→ │   Event Ingest (gRPC)   │  validates, classifies (txn/social/mkt),
   (post created,        └────────────┬────────────┘  assigns event_id, persists for replay
    order shipped, etc.)              ↓
                         ┌─────────────────────────┐
                         │   Fan-out Service       │  expands 1 event → N recipients
                         │   (sharded by event)    │  batches (1000 recipients/msg)
                         └────────────┬────────────┘  applies push-vs-pull policy
                                      ↓
                    ┌─────────────────────────────────┐
                    │ Per-User Work Queue             │  Kafka, partitioned by user_id
                    │ (key = user_id, ordered)        │  ordering per user matters
                    └────────────┬────────────────────┘
                                 ↓
               ┌──────────────────────────────────────────────┐
               │ Orchestration / Per-User Scheduler           │
               │ (sharded by user_id, one shard owner per     │
               │  user; in-memory state for active users)     │
               │  - reads prefs (cache → DB)                  │
               │  - applies frequency cap & batching          │
               │  - applies quiet hours, presence, consent    │
               │  - selects channel(s)                        │
               │  - emits per-channel SendCommand             │
               └────────────┬─────────────────────────────────┘
                            ↓
        ┌───────────────────┼───────────────────┬──────────────┐
        ↓                   ↓                   ↓              ↓
  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐  ┌─────────────┐
  │ Push Queue  │    │ Email Queue │    │  SMS Queue  │  │ In-App Queue│
  └──────┬──────┘    └──────┬──────┘    └──────┬──────┘  └──────┬──────┘
         ↓                  ↓                  ↓                ↓
  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐  ┌─────────────┐
  │FCM/APNS Adp │    │ SES Adapter │    │Twilio Adptr │  │WS Push Adptr│
  │ + circuit   │    │ + circuit   │    │ + circuit   │  │ + presence  │
  │   breaker   │    │   breaker   │    │   breaker   │  │   layer     │
  └──────┬──────┘    └──────┬──────┘    └──────┬──────┘  └──────┬──────┘
         ↓                  ↓                  ↓                ↓
       FCM/APNS            SES              Twilio         WebSocket fleet
         │                  │                  │                │
         └──────────────────┴────────┬─────────┴────────────────┘
                                     ↓
                       ┌──────────────────────────┐
                       │ Status / Feedback Stream │   delivery confirmed,
                       │ (callbacks from providers│   bounce, complaint,
                       │  + provider FBLs)        │   FBL spam reports
                       └────────────┬─────────────┘
                                    ↓
                       ┌──────────────────────────┐
                       │ Preference & State Store │   updates prefs on FBL,
                       │  (sharded Postgres SoT + │   persists delivery history,
                       │   Redis hot cache)       │   feeds analytics/audit
                       └──────────────────────────┘
```

Three persistence tiers: **(1)** Postgres (sharded by user_id) as source of truth for preferences, consent, audit log, device tokens. **(2)** Redis cluster (sharded by user_id, same partitioning as orchestration) for hot per-user state — frequency counters, current batch, presence, device-token cache. **(3)** Kafka for inter-stage queues with per-user ordering and replay.

**Rejected alternatives:** **[STAFF SIGNAL: rejected alternative]**
- *Stateless worker pool consuming an unkeyed queue*: rejected because frequency caps and batching require per-user serialization. Concurrent workers for the same user race on the cap counter and produce 4-in-an-hour bugs.
- *Single Postgres for all hot state*: rejected — at 300K reads/sec, a relational DB on the hot path collapses or costs absurdly. Cache-aside is mandatory.
- *Stream processor (Flink) for orchestration*: viable alternative, considered. Rejected for primary path because operational maturity at this org is lower than for sharded services + Redis, and the per-user-actor abstraction is cleaner. Flink is used for analytics aggregation downstream.

## 5. The Orchestration Layer

This is the central engine. For each (event, recipient) pair, it answers four questions and emits zero or more channel-specific send commands.

```
Input: (event, recipient_user_id)
       │
       ↓
┌─────────────────────────────────────────────────────────┐
│ 1. Should this user be notified at all?                 │
│    - Hard consent check  (GDPR/TCPA: opt-in present?)   │
│    - Topic subscribed?   (e.g., "replies" enabled?)     │
│    - Block list?         (recipient blocked sender?)    │
│    - Mute on this thread/source?                        │
│    → if NO at any check: emit Suppressed audit event,   │
│      return (no send)                                   │
└─────────────────────────────────────────────────────────┘
       │ YES
       ↓
┌─────────────────────────────────────────────────────────┐
│ 2. Which channel(s)?                                    │
│    - User channel preference for this topic             │
│    - Channel availability (device token? verified email?│
│      phone verified? in-app session?)                   │
│    - Class policy (SMS only for txn, marketing not SMS) │
│    - Cross-channel rule (one of {push, email}, not both)│
│    → produces channel candidate set                     │
└─────────────────────────────────────────────────────────┘
       │
       ↓
┌─────────────────────────────────────────────────────────┐
│ 3. When?                                                │
│    - Quiet hours? (user-local TZ; transactional bypass) │
│    - Frequency cap budget remaining for this hour?      │
│    - Currently batching? (next-window scheduled?)       │
│    → "now" | "deferred to T" | "merge into pending"     │
└─────────────────────────────────────────────────────────┘
       │
       ↓
┌─────────────────────────────────────────────────────────┐
│ 4. In what form?                                        │
│    - Standalone vs merged ("3 new replies" vs 3 sends)  │
│    - Full content vs preview (privacy/locked-screen)    │
│    - Localization (user language pref)                  │
│    → render channel-specific payloads                   │
└─────────────────────────────────────────────────────────┘
       │
       ↓
   emit SendCommand(s) to channel queues
```

**Data model on the read path** (per recipient, every event):

| Read | Source | Latency target | Hit rate target |
|---|---|---|---|
| Preferences (channels, topics, quiet hours, lang) | Redis → Postgres | <2 ms | 99.5% |
| Frequency counter (sliding-window per hour) | Redis (per-user key) | <1 ms | 99.9% |
| Pending batch state | Redis | <1 ms | 99.9% |
| Presence (online sessions) | Redis pub/sub digest | <1 ms | 99% |
| Device tokens / verified contacts | Redis → Postgres | <2 ms | 99% |
| Block list (against sender) | Redis (bloom + set) | <1 ms | 99.9% |

A single decision is 5–10 ms wall clock, dominated by Redis round-trips. Pipelined: one Redis MULTI/pipeline call returns all state for one user in ~1 ms.

**Sharding**: orchestrator nodes own user_id ranges via consistent hashing. The per-user Kafka partition (also user_id keyed) routes that user's events to the same orchestrator instance, which then holds in-memory the active per-user scheduler state. Failover: another node takes the partition, rehydrates state from Redis (the source of hot truth) within seconds. The user's brief "outage" surfaces as a few-second latency bump, not lost notifications (Kafka retains the events).

**Invariant**: preferences are read from cache for the orchestrator decision **and re-checked at the channel adapter immediately before send**. Defense in depth — a stale cache must never cause a consent violation. Cache invalidation happens on preference write via Postgres → Redis synchronous invalidation (write-through), with a TTL backstop. **[STAFF SIGNAL: invariant-based thinking]**

## 6. Per-User Batching and Frequency Caps

The naïve "check counter, then send" is broken under concurrency. The correct primitive is a **per-user actor** — a logical owner that processes notifications for one user serially. **[STAFF SIGNAL: stateful-scheduler-for-batching]**

State machine per user:

```
                                     ┌─────────────────────────┐
                                     │   IDLE (no pending)     │
                                     │   sent_this_hour = k    │
                                     │   window_resets_at = T  │
                                     └────────┬────────────────┘
                            new notif arrives │
                                              ↓
                       ┌──────────────────────────────────────┐
                       │  Decide deliverability for this notif│
                       │  (consent, channel, class, presence) │
                       └────┬──────────────┬──────────────────┘
                  high-pri  │              │ normal
                  txn       ↓              ↓
        ┌─────────────────────────┐  ┌────────────────────────────┐
        │ Priority Breakthrough   │  │ k < cap?                   │
        │ - bypasses freq cap     │  │  YES → deliver now,        │
        │ - own per-hour budget   │  │        k++                 │
        │   (max 1 breakthrough/h)│  │  NO  → enqueue in pending; │
        │ - bypasses quiet hours  │  │        schedule wake @ T+ε │
        │   only if class=critical│  │        if not scheduled    │
        └─────────┬───────────────┘  └────────────┬───────────────┘
                  ↓                               │
              deliver                             ↓
                                          ┌─────────────────────┐
                                          │  BATCHING           │
                                          │  pending = [n1, n2] │
                                          │  wake_at = T_window │
                                          └────────┬────────────┘
                                wake fires (or     │
                                user comes online) ↓
                                          ┌─────────────────────┐
                                          │ Window tick:        │
                                          │  - merge if same    │
                                          │    source ("3 new   │
                                          │    replies from X") │
                                          │  - emit up to (cap- │
                                          │    k) sends         │
                                          │  - rest stay queued │
                                          │    or are dropped   │
                                          │    per class policy │
                                          └────────┬────────────┘
                                                   ↓
                                                 IDLE
```

**Cap semantics**: sliding hour window, not fixed top-of-hour, to prevent the 6-in-2-minutes pathology at hour boundaries (3 at 10:59, 3 at 11:00). Implemented as a sorted set in Redis: one entry per delivery with the timestamp; eviction-on-read removes entries older than 1 hour; cardinality is the current count. ZADD/ZREMRANGEBYSCORE/ZCARD in a single pipeline.

**Priority breakthrough policy**: notification class is one of {critical, transactional, social, marketing}. **[STAFF SIGNAL: priority-breakthrough policy]**
- *critical* (security alert, account compromise): bypasses cap and quiet hours; capped at 2/hour against an independent budget to prevent abuse.
- *transactional* (order shipped, password reset): bypasses cap up to 1/hour; respects quiet hours unless time-sensitive flag set.
- *social*, *marketing*: hard cap; quiet hours respected.

The two budgets — cap budget and breakthrough budget — are independent counters. A bug that misclassifies marketing as critical is contained by the breakthrough cap.

**Presence-aware early flush**: when a user's app comes to foreground, the presence service publishes a "user online" event keyed by user_id. The orchestrator instance owning that user's actor wakes the scheduler immediately, draining pending batched notifications via in-app delivery (no push needed) and possibly merging them into a single "you have 4 new things" summary. **[STAFF SIGNAL: presence-aware-delivery]**

**Scale**: 100M users, ~10M with at least one pending notification at any moment. Each active actor's hot state is ~500 bytes in Redis plus negligible in-memory orchestrator state for currently-scheduled wakes. With 100 orchestrator nodes, ~100K active actors per node — trivial.

**Rejected alternatives**: *Per-user scheduled jobs in a job system* (Sidekiq-style): rejected because 10M scheduled jobs/hour stresses the job system and loses the in-memory locality benefit. *Stateful Flink with user-keyed state*: viable, considered; rejected because debugging Flink state in a 3am incident is painful and the per-user-actor pattern is more direct.

## 7. Fan-out Burst Handling

A celebrity post fans out to 50M followers. **[STAFF SIGNAL: fan-out burst handling]** **[STAFF SIGNAL: blast radius reasoning]**

```
                     ┌──────────────────────────────┐
                     │ Event: user_X posted         │
                     │ follower_count = 50M         │
                     └──────────────┬───────────────┘
                                    ↓
                   ┌────────────────────────────────────┐
                   │ Fan-out Policy Decision            │
                   │  fanout_size > PUSH_THRESHOLD (1M)?│
                   │   YES → hybrid push+pull           │
                   │   NO  → full push                  │
                   │  also: urgency class = social      │
                   │   → spread over T_spread (5 min)   │
                   └──────────────┬─────────────────────┘
                                  ↓
                   ┌────────────────────────────────────┐
                   │ Recipient Iterator                 │
                   │  - reads follower index in chunks  │
                   │    of 10K (paginated)              │
                   │  - for each chunk, computes         │
                   │    target enqueue time uniformly   │
                   │    across [now, now + T_spread]    │
                   │  - active-followers-only filter    │
                   │    (skip 90-day inactive)          │
                   └──────────────┬─────────────────────┘
                                  ↓
                   ┌────────────────────────────────────┐
                   │ Batched writes to user-keyed queue │
                   │ 1000 recipients per Kafka message  │
                   │ → 50M / 1000 = 50K messages        │
                   │ Consumer expands batch into 1000   │
                   │ per-user records before processing │
                   └──────────────┬─────────────────────┘
                                  ↓
                          per-user orchestration
                          (existing path)
                                  │
                                  ↓
                   ┌────────────────────────────────────┐
                   │ For very-inactive followers:       │
                   │  PULL MODEL — no push at all.      │
                   │  Notification materialized into    │
                   │  in-app feed only, surfaced when   │
                   │  user opens the app.               │
                   │  Saves 30%+ of fan-out work for    │
                   │  the long tail.                    │
                   └────────────────────────────────────┘
```

**Spreading over T_spread**: instead of trying to push 50M notifications in 1 second, the fan-out service distributes them uniformly across 5 minutes. From a user perspective indistinguishable; from a capacity perspective the difference between "system melts" and "system runs at 1.5x baseline for 5 minutes."

**Push-vs-pull threshold**: for fan-out > 10M with notification class = social/marketing, the long-tail (followers inactive >30 days) is moved to pull-only — they will see it in their in-app feed when they next open the app, no push, no email. For followers active in last 24h, full push. This is a deliberate quality decision: a notification has near-zero value if delivered to a dormant user, so spending capacity on them is waste.

**Rejected**: *push to all 50M synchronously*. At our per-decision cost it's 17 minutes of full-fleet saturation per celebrity post. *No spreading, queue-and-let-it-drain*. Works but produces unpredictable latency for non-celebrity events queued behind. Spreading bounds the impact.

**Backpressure**: the Fan-out Service monitors per-user-queue depth. If queue depth > threshold, it pauses or further-spreads new fan-outs. Producers (event ingest) are not directly throttled — the queue absorbs the burst, the spreading is the throttle.

**Containment of runaway fan-out**: hard ceiling per event — no event can fan out more than CONFIG_MAX_FANOUT recipients (set to ~100M, the entire user base). Per-actor (sender) fan-out budget per hour: a sender producing >5 fan-out events of >10M each gets throttled — protects against a compromised celebrity account spamming.

## 8. Channel Adapters and Channel-Specific Concerns

Each adapter is a stateless worker pool consuming from its channel queue. The work is mostly: render the channel-specific payload, call the provider with idempotency key, handle the response, post a status event back. The interesting parts are channel-specific:

**Push (FCM + APNS)**:
- Connection multiplexing: APNS requires HTTP/2 with persistent connections; pool ~100 connections per APNS region per worker. FCM via batched HTTP.
- Token lifecycle: tokens expire silently; a 410/404/`Unregistered` from FCM/APNS triggers token deletion from user state. A user with all tokens expired has push automatically disabled until a fresh registration arrives from the app.
- No real delivery confirmation. APNS will tell you "I accepted it"; nothing tells you the user saw it. Status callbacks here are about acceptance, not delivery.
- Per-user device set: notifications go to all of a user's devices (matches user expectation), but this multiplies cost for high-fan-out users.

**Email (SES primary, SendGrid fallback)**:
- Bounces (hard/soft) and complaints (FBL) flow via SNS callbacks → status stream → preference store. Hard bounce auto-disables email channel for that address. Complaint also auto-disables and logs for compliance.
- Per-recipient-domain throttling: gmail.com can take 1000s/sec from us, but a small corporate domain might rate-limit at 5/sec. The adapter shapes traffic per destination domain.
- Reputation: bounce rate >5% or complaint rate >0.1% triggers SES suppression. We monitor and proactively suppress sends to risky addresses.
- Built-in retry: provider handles transient retry; we only retry on our own ingestion failures.

**SMS (Twilio)**:
- Cost-gated: SMS is restricted to class=critical and class=transactional, plus explicit user opt-in for any other class. Hard global daily budget; circuit-breaks at threshold.
- TCPA compliance: prior express consent recorded with timestamp + source. STOP/UNSUBSCRIBE keywords processed by Twilio webhook → preference store update within seconds; we will never send another SMS to that number.
- Time-of-day: many US states restrict SMS marketing to 8am–9pm local; the orchestrator's quiet-hours logic enforces this with destination phone number's area-code TZ.
- Per-message idempotency via Twilio's `X-Idempotency-Key`-equivalent (MessagingServiceSid + ProviderMessageId).

**Web Push**:
- Subscription managed in user state; expires; refresh on page load.
- VAPID-signed; payload < 4 KB.
- Best-effort; browser must be running.

**In-App**:
- Real-time when user is online via the WebSocket presence layer. Otherwise persisted in the user's notification feed (Postgres + Redis recent cache) and surfaced on next app open. The most reliable channel because we own it end-to-end.

All adapters share an identical interface (`SendCommand` → `DeliveryStatus`), per-channel circuit breaker, per-channel rate limiter (protects provider), and idempotency by `notification_id`.

## 9. Preferences, Consent, and Regulatory

**[STAFF SIGNAL: regulatory awareness]**

Data model (hierarchical):

```
user_preferences
├── global
│   ├── enabled_channels: {push, email, sms, web, in_app}
│   ├── language: en-US
│   └── quiet_hours: 22:00–07:00 local
├── per_topic[]
│   ├── topic: "replies" | "marketing" | "security" | ...
│   ├── enabled: bool
│   ├── channel_override: optional channel set
│   └── frequency_cap_override: optional int
└── consent_record[]
    ├── channel: sms | email | ...
    ├── source: "signup_form_v3" | "settings_toggle" | ...
    ├── timestamp: ...
    └── ip_address / user_agent (for SMS / regulated channels)
```

**Source of truth**: Postgres, sharded by user_id, per-region (EU users in EU shard for GDPR). Cache: Redis. Write-through invalidation: every preference write commits to Postgres and synchronously invalidates Redis. Cache miss falls through to Postgres, but **never fails-open** — if both are unreachable, the orchestrator suppresses non-critical sends rather than risking a consent violation. **[STAFF SIGNAL: invariant-based thinking]**

**Defense in depth**: the channel adapter re-checks consent for the specific channel against its own cache before calling the provider. A drift between orchestrator decision and reality at send time is caught here. Adapter-level suppression is logged.

**Audit log**: every decision (sent / suppressed-for-reason / failed) is written to an append-only audit stream (Kafka → S3 + searchable index) with retention >7 years for regulated classes. This is the answer to "why did I get / not get this notification?" — both for users and regulators.

**Specific obligations the architecture supports**:
- **GDPR**: explicit consent for marketing recorded; right-to-be-forgotten erases user state across all stores via a coordinated tombstone (replicated to all caches and the audit log marks the user as erased while retaining minimal compliance metadata).
- **TCPA**: SMS opt-in stored with provenance; STOP keyword handled within seconds via Twilio webhook → preference write.
- **CAN-SPAM**: every marketing email contains unsubscribe link; click writes to preference store within 10 days (in practice, seconds).

## 10. Retry, Dead-Lettering, Idempotency

**[STAFF SIGNAL: retry-without-amplification]**

The retry-amplification trap: a transient FCM 503 should be retried; a "user marked us as spam" feedback should *never* be retried; an "invalid email" must immediately disable the channel.

```
   ┌─────────────────────────────────────┐
   │       SendCommand → Adapter         │
   └────────────────┬────────────────────┘
                    ↓
            Provider call w/ idempotency_key=notif_id
                    │
        ┌───────────┼────────────┬─────────────┐
        ↓           ↓            ↓             ↓
     Success    Transient     Hard fail    Spam/FBL
        │       (5xx, net)  (bad addr,     (complaint)
        │           │        bounce)            │
        ↓           ↓            ↓              ↓
    audit:OK   retry budget    audit:FAIL   disable channel
    update     remaining?       update       for user;
    counters    YES → backoff    user state   audit:SPAM;
                  enqueue        (token       no retry;
                  retry          delete,      alert if rate
                NO → DLQ         email        elevated
                                 disable)
```

Per-channel retry budgets:

| Channel | Max retries | Backoff | Retry on hard fail |
|---|---|---|---|
| Push | 2 | 30s, 2m | No (likely offline; not worth it) |
| Email | 3 | 1m, 5m, 30m | No (provider already retried) |
| SMS | 2 | 1m, 10m | No (carrier failures often permanent) |
| Web Push | 1 | 30s | No |
| In-App | 5 (own infra) | exp backoff | n/a (we own it) |

After the budget, the message goes to a per-channel **dead-letter queue**. DLQ items are inspected by an alerting system: a sudden spike in DLQ for one channel is a provider/regional issue and pages.

**Idempotency**: every notification has a globally unique `notification_id = hash(event_id, recipient_user_id, channel)`. Adapters pass this as the provider's idempotency key where supported (Twilio, modern email providers). For providers without native support, the adapter maintains a short-TTL Redis dedup set (`SET notif_id NX EX 3600`) — if the key already exists, skip the call.

**Specific scenario the design must survive**: a bug enqueues the same notification 10 times for 100M users. **[STAFF SIGNAL: blast radius reasoning]** Defenses, layered:
1. Orchestrator dedup on `(user_id, notification_id)` within a 24h Redis set.
2. Per-channel adapter dedup as above.
3. Hard per-user per-channel rate ceiling (e.g., 100 sends/user/hour, regardless of class) at the adapter — defense beyond all preference-level limits.
4. Global anomaly detector on per-user send rate; auto-throttles if a user's send rate exceeds 5σ.

## 11. Cross-Channel Coordination

Three patterns, each appropriate for different classes:

```
Pattern A: Single-channel with fallback (transactional default)
  ─────────────────────────────────────────────────────────────
   Orchestrator → push → wait T_fb (e.g., 60s) for ack
                          │
              ack received? ─── YES → done
                          │
                          NO → orchestrator emits email send
                               (idempotency-keyed; if push
                                actually delivered late, the
                                duplicate is on the user's view,
                                not the system's)

Pattern B: Channel selection by presence (social default)
  ─────────────────────────────────────────────────────────────
   Orchestrator reads presence:
     user is active in web app NOW  → in-app only, no push
     user has no recent web session → push to mobile
     user has neither and topic=high → digest email at next window

Pattern C: Parallel multicast (rare; only critical security)
  ─────────────────────────────────────────────────────────────
   Push + email + SMS in parallel; user expected to see all,
   acceptable for "your password was changed" or "new login".
   Marked clearly as security event.
```

**Presence-aware suppression**: if the user is currently active in the in-app session, push to the same device is suppressed (the user already saw it in-app via the WebSocket). This requires a presence service maintaining current session state per user, queried in the orchestration decision. **[STAFF SIGNAL: presence-aware-delivery]**

**Cross-channel dedup**: the orchestrator emits at most one channel-set per `(user_id, notification_id)`. If the same logical notification arrives from two upstream paths (e.g., a re-published event), the dedup key collapses them.

**Rejected**: parallel-to-all-channels-by-default. Produces visible duplication ("got a push *and* an email about the same like"), which is the #1 user complaint that drives unsubscribes. Pattern A is the default for a reason — quality over coverage.

## 12. Failure Modes and Graceful Degradation

**[STAFF SIGNAL: failure mode precision]**

| Failure | Detection | Response | Containment |
|---|---|---|---|
| FCM down | error rate spike on push adapter; circuit breaker opens | push queue accumulates; if class=transactional, orchestrator failover-emits email after timeout; class=social held until recovery (with TTL) | per-channel breaker isolates from email/SMS |
| SES down | bounce on adapter | failover to SendGrid (warm secondary); if both, queue and alert | reputation damage if held too long → drop class=marketing first |
| Preference DB down | Postgres errors on cache miss | serve from Redis only; **suppress non-critical** if both miss; allow critical with most-recent-known consent | never send unconsented; better to delay than violate |
| Redis cluster partition | latency / unavailability on shard | degrade to direct Postgres reads, drop throughput by 10x; shed load by class (drop marketing, defer social) | consistent-hash failover on shard; per-shard isolation |
| Orchestrator overload | per-user queue depth grows | backpressure to fan-out; spread non-urgent fan-outs further; class-priority shedding | shedding order: marketing → social → transactional → critical |
| Massive fan-out (Taylor Swift) | fan-out service watermark | spread, push-vs-pull policy kicks in, long-tail goes pull-only | bounded peak load |
| Runaway notification bug (10x to 100M users) | global anomaly detector on send rate per (user, source) | per-user hard cap (defense #3) blocks the spam; DLQ accumulates; on-call paged | bug is contained at adapter ceiling, not at orchestrator logic |
| Corrupted preferences for a user | audit log + checksum on read | restore from latest snapshot; conservative defaults (no marketing, only transactional) until restored | per-user blast radius |
| FBL spike | complaint rate watcher | auto-disable channel for affected users; investigate content/source | avoids reputation damage |
| Provider idempotency-key collision | dedup set hit | skip; log; verify at audit | rare; design uses globally unique IDs |

**Shed order** under load is explicit: marketing first, then social non-essential, then social essential, then transactional, never critical. This is configured policy, not ad-hoc.

## 13. Operational Reality

**Metrics that matter** (and what they predict):
- *End-to-end delivery latency p50/p95/p99 by channel, by class*. Tail latency growth is the leading indicator of saturation.
- *Per-channel delivery success rate*. Drop signals provider issue or content-quality issue.
- *Queue depth at each stage*. Primary capacity signal; pages on sustained growth.
- *Suppression rate by reason* (consent, cap, quiet-hours, presence). Sudden change signals upstream behavior change.
- *Unsubscribe rate by source/class*. **The most important quality metric.** Rising unsubscribes mean we are over-notifying — architecture should support cutting volume, not just throughput. **[STAFF SIGNAL: quality-over-throughput]**
- *Per-channel cost* (especially SMS) vs. budget. Daily reconciliation.
- *FBL/spam rate*. Reputation early-warning.

**Audit log**: every decision indexed by user_id and notification_id. Customer support self-serve — "why didn't I get this?" answered without engineering. This is operationally cheap and saves enormous toil.

**A/B testing integration**: notifications integrate with the experimentation platform via a `treatment` field on the SendCommand. Send-time, content, channel choice are all standard A/B'd. The experimentation system subscribes to delivery + downstream outcome events (open, click, conversion).

**Cost attribution**: each notification carries a `source_team` tag; daily cost rollups attribute SMS/email spend. Internal teams have budgets — bad citizens are visible.

**Runbooks**: per failure mode above, an explicit runbook with the shed-order, the failover toggles, and the metrics to watch.

## 14. Tradeoffs Taken and What Would Change Them

- **Per-user actor with sharded ownership**, not stateless workers. Changes if frequency caps and batching are dropped from the requirements; then a stateless fanout suffices and is simpler.
- **Build orchestration, buy delivery**. Changes if channel volume justifies dedicated ESP relationships.
- **Spread fan-outs over minutes** for non-urgent. Changes if product demands simultaneity (live events).
- **Hybrid push+pull at extreme fan-out**. Changes if active-rate is high or universal push is expected.
- **Preferences in regional Postgres**. Changes under stricter data-residency regimes needing isolation.

## 15. What I Would Push Back On

**[STAFF SIGNAL: saying no]**

1. **"100M users" is the wrong scaling unit.** Peak events/sec, max fan-out per event, and per-user state size predict cost and design far better. I'd ask for those numbers before sizing capacity.
2. **"Support all four channels equally."** In practice, push + in-app dominate volume, email is steady-state, SMS is rare and expensive. Designing for channel parity wastes engineering on SMS-at-scale features that don't exist in the workload.
3. **The implicit framing that more notifications = better.** **[STAFF SIGNAL: quality-over-throughput]** The highest-leverage notification system sends *fewer*, more relevant notifications. The architecture I described — frequency caps, batching, presence-aware suppression, unsubscribe-as-first-class-metric — is built to enable saying "no" to sends, not just enabling more of them. If the product team measures success by notifications-per-day-per-user going up, the system is pointed in the wrong direction.
4. **"Build vs buy" deserves explicit discussion.** For many companies the right answer is integrate Braze/OneSignal/Customer.io and own only the upstream event layer; the conversation about what we are *actually* differentiating on changes the design entirely.