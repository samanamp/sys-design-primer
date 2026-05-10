---
title: "Webhook delivery system"
description: "Webhook delivery system"
---

"Design the system that delivers webhooks to customer endpoints — at-least-once delivery, exponential backoff, dead-lettering, isolation between fast and slow customers, signed payloads. Walk me through it."

---

## 1. Reframing

Before architecture, the framing that shapes every decision: **the customer endpoint is the adversary**. Not malicious — but untrusted, slow, and unreliable in every dimension that matters. [STAFF SIGNAL: untrusted-endpoint-as-central]

Real customer endpoints, observed from production at scale: time out at 30 seconds; return `200 OK` in 1ms but never actually process; return 500s for hours during incidents; sit behind rate limiters that throttle us silently; redirect to other hosts; present invalid TLS certificates; resolve to `127.0.0.1` or `10.0.0.0/8` (intentionally or via DNS rebinding); respond at p50=80ms but p99.9=28s. The wire protocol is trivial — `POST` with an HMAC header. **The system is a long-tail-latency, multi-tenant scheduling problem against an adversarial wall.** Mid-level designs collapse because they treat the endpoint as cooperative.

Two more framings, both load-bearing:

**Slow-customer isolation is the central architectural problem.** [STAFF SIGNAL: per-customer-isolation-as-architecture] If one customer's endpoint goes from 50ms to 30s, a shared worker pool fills with workers blocked on their socket reads. Throughput collapses for everyone. The entire architecture exists to make the invariant *"a slow customer cannot block fast customers"* enforceable, not aspirational.

**At-least-once is the only honest contract.** [STAFF SIGNAL: at-least-once honesty] Exactly-once across an untrusted network is a marketing claim. A customer endpoint returns `200 OK`, the response packet is dropped, our worker times out and retries — they receive the event twice. There is no protocol fix. The contract is: **we deliver each event ≥1 times; you implement an idempotent receiver keyed on `event_id`**. The whole design assumes this.

Every subsequent decision flows from these three.

## 2. Scoping

I'll commit. [STAFF SIGNAL: scope negotiation]

- **Volume**: 500M deliveries/day. Average ~6K/sec; bursts to 60K/sec (e.g., end-of-month billing events fan out simultaneously).
- **Latency target**: p50 < 1s from event-creation to first-attempt-completion; p99 < 5s. No hard SLA on retried deliveries — those are best-effort within the retry budget.
- **Customer count**: 100K active customers, ~2 endpoints each on average, long-tailed (largest customers have 50+ endpoints).
- **Payload size**: median 2 KB, p99 50 KB, hard cap 1 MB (we reject larger at the producer).
- **Ordering**: **default unordered**, with opt-in per-resource ordering for customers who need it (e.g., subscription state machines). Most customers do not need it; we don't pay the throughput cost by default.
- **Retention**: 30 days for delivery audit log, 90 days for DLQ payloads, 7 days for in-flight retry queue.
- **Tier-aware SLA**: enterprise tier sub-second p99 first attempt; free tier best-effort with minute-class latency acceptable.

If the interviewer pushes back ("what if it's 10B/day?"), the design scales linearly on partition count and worker count up to the point where the global metadata DB becomes the bottleneck — at which point we shard the metadata DB by customer ID. Nothing else changes.

## 3. Capacity Math

[STAFF SIGNAL: capacity math]

```
Deliveries/sec average:    500M / 86400  ≈  5,800/sec
Peak burst (10x):                         ≈ 58,000/sec
Avg endpoint latency (p50):                  200 ms
p99 endpoint latency:                          3 s
p99.9 (long-tail customers):                  28 s

Worker concurrency (Little's Law):
  avg:   5,800/s × 0.2s   =    1,160 concurrent
  peak:  58,000/s × 0.2s  =   11,600 concurrent
  + p99 tail headroom (3x):    35,000 concurrent

Worker fleet sizing:
  ~3,000 workers @ 12 concurrent in-flight each
  (async I/O; one TCP connection per delivery)

Queue storage (in-flight, 7-day retention worst case):
  Avg payload + metadata: ~5 KB
  At 5,800/s × 7d × 86400s × 5KB ≈ 17.5 TB hot
  Realistically: <1% retried >24h, so ~2–3 TB hot

DLQ storage (90 days):
  Assume 0.1% terminal failure rate
  500M/d × 0.001 × 90d × 5 KB     ≈ 225 GB

Audit log (30 days):
  Every attempt logged: ~1.5x event volume due to retries
  500M × 1.5 × 30d × 1 KB         ≈  22.5 TB
```

Note that **worker fleet size is dominated by tail latency, not average load**. If we sized to p50 we'd have ~1,200 workers and the system would lock up the first time 5% of customers got slow simultaneously. Sizing to ≥3x p99 headroom is non-negotiable.

## 4. High-Level Architecture

```
                        ┌─────────────────────┐
   Producers  ────►     │   Event Bus (Kafka) │   (durable, source of truth)
   (other svcs)         │  topic: events.raw  │
                        └──────────┬──────────┘
                                   │
                                   ▼
                       ┌────────────────────────┐
                       │  Dispatcher / Fanout   │  resolves event → endpoints,
                       │   (stateless workers)  │  produces 1 delivery per endpoint
                       └────────────┬───────────┘
                                    │
              ┌─────────────────────┼─────────────────────┐
              ▼                     ▼                     ▼
     ┌──────────────┐      ┌──────────────┐      ┌──────────────┐
     │ Per-endpoint │      │ Per-endpoint │      │ Per-endpoint │  …100K virtual queues
     │   queue A    │      │   queue B    │      │   queue Z    │  on Kafka partitions +
     └──────┬───────┘      └──────┬───────┘      └──────┬───────┘  Redis Streams overlay
            │                     │                     │
            └────────────┬────────┴──────────┬──────────┘
                         ▼                   ▼
              ┌─────────────────────────────────────┐
              │   Worker Fleet (3K async workers)   │
              │   - fair-share scheduler            │
              │   - per-endpoint concurrency cap    │
              │   - signs payload (HMAC)            │
              │   - SSRF guard (DNS verify)         │
              └────────────┬────────────────────────┘
                           │  HTTPS POST
                           ▼
                   ┌───────────────┐
                   │  Customer     │   (untrusted)
                   │  Endpoint     │
                   └───────┬───────┘
                           │ status
                           ▼
              ┌────────────────────────┐      ┌──────────────────┐
              │  Result Reconciler     │─────►│  Audit Log       │ (30d, queryable)
              │  - schedules retries   │      └──────────────────┘
              │  - emits to DLQ        │      ┌──────────────────┐
              │  - updates customer    │─────►│  DLQ + Replay    │ (90d)
              │    health metrics      │      └──────────────────┘
              └────────────────────────┘
                           │
                           ▼
              ┌────────────────────────┐
              │  Customer Dashboard    │  (success rates, replay UI,
              │  + Webhook API         │   per-event audit)
              └────────────────────────┘
```

**Design choices, with rejected alternatives:** [STAFF SIGNAL: rejected alternative]

- **Kafka as event bus, not direct DB writes from producers**: producers commit to their own DB, emit to Kafka via outbox. *Rejected*: producers calling webhook service synchronously — couples producer latency to webhook latency.
- **Dispatcher fans out 1 event → N deliveries**: each (event, endpoint) is its own delivery row, independently retried. *Rejected*: single delivery row tracking N endpoint statuses — turns every retry into a partial-success bookkeeping nightmare.
- **Per-endpoint queues via Kafka partitions + Redis Streams overlay**: Kafka gives durability and ordered partitions; Redis Streams gives us cheap virtual sub-queues per endpoint with fast head-of-line skip. *Rejected*: pure Kafka with one partition per endpoint — 200K+ partitions strains the broker. *Rejected*: pure Redis — durability and replay become expensive.
- **Result reconciler as separate service**: workers do the HTTP, reconciler decides next state. *Rejected*: workers self-scheduling retries — workers crash mid-delivery and the retry is lost; centralizing state transitions in a reconciler with idempotent updates is safer.

## 5. Per-Customer / Per-Endpoint Isolation

The central architectural concern. We isolate at the **endpoint** level, not customer level — a customer with one slow endpoint shouldn't have their other 9 endpoints suffer. [STAFF SIGNAL: per-customer-isolation-as-architecture]

**Mechanism**: each endpoint has a logical queue and a **concurrency cap** (default `N=5` simultaneous in-flight deliveries per endpoint). Workers pull work via a fair-share scheduler that respects the cap.

```
Endpoint A (healthy, 50ms)            Endpoint B (slow, 30s timeout)
┌─────────────────────────┐           ┌─────────────────────────┐
│ queue depth: 12         │           │ queue depth: 8,400      │  (growing)
│ in-flight: 5/5          │           │ in-flight: 5/5 (stuck)  │
│ throughput: 100/s       │           │ throughput: ~0.17/s     │  (5÷30s)
└─────────────────────────┘           └─────────────────────────┘
       │                                       │
       └──── workers pull ─────────┬───────────┘
                                   ▼
                  ┌─────────────────────────────┐
                  │ Worker fleet (3000 workers) │
                  │ At most 5 workers blocked   │
                  │ on Endpoint B at any time.  │
                  │ Other 2,995 keep working.   │
                  └─────────────────────────────┘
```

**The invariant**: *the number of workers blocked on any single endpoint is bounded by `N`.* [STAFF SIGNAL: invariant-based thinking] Slow endpoint B's queue grows; that's the *expected* failure mode. But endpoint A's deliveries are unaffected.

**Implementation**: a global scheduler service (multiple replicas, leader-elected per shard of endpoint-space) maintains in-memory state `{endpoint_id → (queue_head_offset, in_flight_count)}` backed by Redis. Workers `dequeue(worker_id)` returns the next eligible delivery — eligible means the endpoint's `in_flight_count < N`. When worker finishes (success or terminal failure), it `release(endpoint_id)`, decrementing the counter. Reconciler increments on retry-scheduled.

**Fair scheduling across endpoints**: deficit round-robin (DRR) over endpoint queues, weighted by tier. Enterprise endpoints get weight 4, paid get weight 2, free get weight 1. Within a tier, oldest-deferred-first. This prevents a high-volume free customer from starving paid traffic. [STAFF SIGNAL: tier-aware fairness]

**Adaptive concurrency cap**: `N` is not fixed at 5. If an endpoint's recent error rate exceeds a threshold (say 50% over last 100 attempts), `N` drops to 1 — circuit-breaker style. We don't pile 5 concurrent deliveries onto an endpoint that's already failing; we probe with one until it recovers. This is also the **thundering-herd-on-recovery** defense [STAFF SIGNAL: thundering-herd-on-recovery]: when an endpoint that's been down for hours recovers, it doesn't get hit with all 8,400 queued deliveries at once — `N=1` with success → ramp `N` to 2, 3, 5 over minutes as success accumulates. The queue drains, but at a rate the customer's recovery can absorb.

**Why per-endpoint and not per-customer**: a customer with 10 endpoints, one slow, would otherwise have all 10 share a concurrency budget. The slow one would absorb all the budget; the other 9 would block. Per-endpoint isolation is finer-grained and worth the extra metadata cost (200K endpoint records vs 100K customer records — trivial).

**Why not sticky workers**: tempting to assign workers to specific endpoints to keep TCP connections warm. *Rejected*: sticky workers concentrate failure (worker crash drops one endpoint entirely); load balancing becomes harder; warm connections are a real but small win we get cheaper via HTTP keep-alive pools per worker indexed by hostname.

**Scaling reality**: 200K virtual queues is fine. At 1M+ endpoints, the scheduler's state grows; we shard the scheduler by `hash(endpoint_id) % S`. Each shard owns ~20K endpoints. The scheduler is stateless aside from Redis — failover is fast.

## 6. Retry Policy

[STAFF SIGNAL: per-status-code retry policy]

**Backoff schedule** (with ±20% jitter): `1m, 2m, 5m, 15m, 1h, 4h, 12h, 24h, 72h`. Nine attempts over ~5 days, then DLQ. Total retry budget is ~5 days, not unbounded.

**Per-status-code decisions**:

| Outcome | Action | Reasoning |
|---|---|---|
| `2xx` | Success, mark delivered | Done |
| `3xx` | Follow once if same-origin & HTTPS | Don't follow redirects to internal IPs (SSRF) |
| `400 Bad Request` | Retry once, then DLQ + alert customer | Likely *our* serialization bug or schema mismatch — alert engineering |
| `401/403` | 3 retries with shorter backoff, then alert customer | Customer rotated their secret/auth; surface fast |
| `404` | 2 retries, then DLQ | Endpoint moved or wrong URL |
| `410 Gone` | Stop immediately, mark endpoint disabled | Customer told us explicitly to stop |
| `413 Payload Too Large` | DLQ immediately | Won't fit ever — customer must reconfigure |
| `429 Too Many Requests` | Honor `Retry-After`; full retry cycle | Customer's rate limit |
| `5xx` | Full retry cycle | Transient |
| Network timeout | Full retry cycle, but reduce endpoint `N` | Probably overloaded |
| DNS failure | Brief retry (3 attempts in 5 min); then longer cycle | Could be transient; could be misconfig |
| TLS failure | 2 retries; then alert customer + DLQ | Their cert is broken; they need to know |

**Retry state machine for one delivery**:

```
                  ┌──────────┐
                  │  PENDING │
                  └────┬─────┘
                       ▼
         ┌──────► IN_FLIGHT ◄──────┐
         │            │            │
         │            ▼            │
         │   ┌─────────────────┐   │
         │   │  HTTP response  │   │
         │   └────────┬────────┘   │
         │            │            │
         │  ┌─────────┼─────────┐  │
         │  │         │         │  │
         ▼  ▼         ▼         ▼  │
      2xx   401/403 5xx/timeout 410
       │     │       │           │ │
       ▼     ▼       ▼           ▼ │
   DELIVERED RETRY  RETRY    TERMINAL_DISABLED
             │ (max 3)        │
             ▼                │
         ┌─────────┐          │
         │SCHEDULED│          │
         │for next │          │
         │attempt  │──────────┘ (back to IN_FLIGHT after backoff)
         └────┬────┘
              │
              │ exceeded budget
              ▼
            ┌─────┐
            │ DLQ │
            └─────┘
```

**Customer-history adjustment**: an endpoint with 99.9% success over the last 10K attempts gets *aggressive* first retry (start at 30s instead of 1m) — transient blips clear quickly. An endpoint with <50% success over the last hour gets *longer* backoffs (start at 5m, lengthened) — they're broken; don't pile on. The retry policy is a function of `(status_code, endpoint_health_history, attempt_number)`, not just `attempt_number`.

**Adaptive backoff during partial recovery**: when an endpoint transitions from failing → succeeding, we don't drain the backlog at full rate. Drain rate is gated by the adaptive concurrency cap (Section 5). Backoff for already-scheduled retries to that endpoint *shortens* (we want to drain), but new failures during drain immediately re-extend backoff.

**The "stop retrying after the customer fixes it" optimization**: when a delivery to endpoint E succeeds after a failure streak, all other pending retries to E that are scheduled for >5min in the future get their `next_attempt_at` advanced to `now + 30s`. We learned the endpoint is healthy; respect that signal.

**Don't retry hard failures**: this is where mid-level answers leak resources. A `404` retried 9 times over 5 days is just spam — the URL is wrong. `410` is the customer's explicit "stop." `413` will never succeed with a smaller payload (we don't shrink it). Each of these short-circuits the retry cycle.

## 7. Dead-Lettering and Replay

[STAFF SIGNAL: DLQ as customer-facing surface]

**DLQ is not an internal ops concern; it's a product surface.** A customer's DLQ is the answer to "what events did I miss and why."

**DLQ entry**: `{event_id, endpoint_id, customer_id, payload (in object store), final_status, attempt_history[], first_attempt_at, last_attempt_at, dlq_reason}`. Metadata in a queryable store (Postgres partitioned by customer + month); payloads in S3-class object storage keyed by `customer_id/yyyy-mm-dd/event_id`.

**Customer-facing DLQ API and UI**:
- `GET /v1/webhooks/{endpoint}/dlq?since=...&until=...&event_type=...` — paginated list with filters.
- `GET /v1/webhooks/dlq/{dlq_id}` — full payload + attempt history + last error.
- `POST /v1/webhooks/dlq/{dlq_id}/replay` — re-enqueue to current endpoint. Payload sent with header `X-Webhook-Replay: true` and original `X-Webhook-Timestamp` preserved (so signature still validates if customer kept their old secret).
- `POST /v1/webhooks/dlq/replay-batch` — replay range (e.g., "all DLQ entries for endpoint E from 2026-04-01 to 2026-04-03").

**Replay protocol**:

```
Customer    DLQ API                 Webhook System         Endpoint
   │           │                          │                    │
   │ replay 50 ├─► validate (auth,        │                    │
   │           │   ownership, rate)       │                    │
   │           │                          │                    │
   │           ├─► enqueue 50 deliveries ►│                    │
   │           │   tag: replay=true       │                    │
   │           │   priority: low          │ (separate queue,   │
   │           │                          │  doesn't compete   │
   │           │                          │  with live events) │
   │           │                          │                    │
   │           │                          ├──── POST ────────►│
   │           │                          │   X-Replay: true   │
   │           │                          │   X-Original-TS    │
```

**Key replay design choices**:
- Replays go through a *separate, lower-priority* queue. A panicked customer replaying 100K events doesn't starve their live event delivery.
- Replays carry the original timestamp so the customer's idempotency layer (keyed on `event_id`) deduplicates correctly if they already processed the event via a successful delivery they didn't realize.
- Rate-limited per customer to prevent replay-storm self-DoS.

**Retention**: 90 days for DLQ payloads. After 90 days: payload deleted, metadata stub kept for 1 year ("event X was DLQ'd on date Y, payload no longer available, last error was Z"). Auto-archive notification when DLQ exceeds 1K events for a customer — they need to know their integration is broken.

**Permanent failures bypass retry**: 410, 413, schema-rejected — straight to DLQ on first attempt. Don't waste the retry budget.

## 8. Security: Signing, SSRF, Key Rotation

**Payload signing**: HMAC-SHA256 over `timestamp + "." + payload`. Header: `X-Webhook-Signature: t=<unix_ts>,v1=<hex_hmac>`. Customer recomputes; rejects if mismatch or `|now - timestamp| > 5min` (replay-window protection).

**Per-customer secret**: generated at endpoint registration, shown once, stored encrypted (KMS-wrapped, per-customer DEK). Compromise → customer rotates.

**Key rotation with overlap window**: [STAFF SIGNAL: signing key rotation]

```
   t0          t1                  t2 (rotation)         t3 (cutover)
   │           │                     │                      │
   │  v1 only  │  v1 + v2 (overlap)  │   v2 only            │
   ├───────────┼─────────────────────┼──────────────────────┤
   │           │ 24h: customer       │ Old key retired      │
   │           │ updates verifier    │
   │           │ to accept both      │
```

We send `X-Webhook-Signature: t=...,v1=<old_hmac>,v1=<new_hmac>` — both signatures present during the overlap window. Customer accepts a delivery if *any* of the listed signatures verify. After overlap, we drop the old signature. **Never** rotate without overlap; instant cutover guarantees deliveries fail during clock skew or in-flight verification.

**SSRF defense**: [STAFF SIGNAL: SSRF and DNS-rebinding awareness]

The threat: customer registers `https://internal-vault.our-corp.local/secrets` as their webhook URL. Or registers `https://harmless.example.com` which DNS-resolves to a public IP at registration but to `10.0.0.5` at delivery time (DNS rebinding). Either leaks our internal data to them.

Defenses, layered:

1. **Registration-time URL validation**: scheme must be `https://`; hostname is parsed and rejected if it's an IP literal in a private range, our internal hostnames, `localhost`, or `metadata.google.internal`-class endpoints. DNS resolved at registration; if it resolves to private IP, reject.
2. **Delivery-time DNS resolution + IP validation**: every delivery, resolve the hostname *ourselves* (not via OS resolver that may cache stale public IPs), check every returned A/AAAA record against the private-IP blocklist. **If any resolved IP is private, abort the delivery.** This is the DNS rebinding defense.
3. **Connect to the resolved IP, not the hostname**: after validating, we open the TCP connection to the validated IP, then send `Host: <hostname>` header and SNI `<hostname>`. The DNS resolver can't change the IP between validation and connection because we already have it.
4. **Network-level egress controls**: webhook delivery workers run in a network segment with only public-internet egress. Internal services unreachable at the network layer regardless of DNS. Defense in depth.
5. **TLS strict by default**: cert must validate against system trust store. Customer can opt out only with a per-endpoint `allow_insecure_tls=true` flag, gated on tier and acknowledged in writing — and we never enable it for endpoints carrying PII or financial data.
6. **Redirect handling**: at most one redirect; redirect target re-validated through (1)–(3). No redirect chains.

**Data-leak defense beyond SSRF**: [STAFF SIGNAL: data-leak-defense] every delivery worker, before send, asserts `delivery.customer_id == endpoint.customer_id == event.customer_id`. If any mismatch, abort, page on-call, never retry. A bug in the dispatcher fanout that crosses customers is a data-breach-grade incident; the worker is the last line of defense.

## 9. At-Least-Once Contract and Idempotency

The contract, written down: [STAFF SIGNAL: at-least-once honesty]

> *We will deliver each event one or more times. Your endpoint must be idempotent, keyed on the `event_id` field of the payload. Process each `event_id` exactly once on your side; ignore subsequent deliveries.*

The payload always includes:
```json
{
  "event_id": "evt_01HQ8...",       // ULID, unique, signed
  "type": "subscription.updated",
  "created_at": "2026-05-09T...",
  "delivery_attempt": 3,             // hint to customer
  "data": { ... }
}
```

`delivery_attempt` is a courtesy hint, not security. Customer's idempotency must work regardless: a network split could deliver attempt 1 and attempt 3 in either order.

**Why not exactly-once**: between our worker and the customer's database lies the public internet, the customer's load balancer, possibly a queue, possibly multiple replicas of their handler. There is no end-to-end transaction we can participate in. Our worker times out at 30s; customer's request actually completed at 31s and committed to their DB. We retry; they receive it again. *Any* sufficiently honest webhook system has duplicates. Anyone selling exactly-once is selling at-least-once with extra steps and worse failure modes.

The honest framing is also a *better* framing for the customer: their idempotent receiver protects them against their own retries (mobile app firing twice, their internal queue replaying), not just ours.

## 10. Ordered Delivery (Opt-In)

Default is unordered for throughput. For customers who need ordering — typically state-machine events (`subscription.created` before `subscription.activated`) — we offer **per-resource ordering** via an opt-in flag at the endpoint level, with a partition key field name (e.g., `partition_key: "subscription_id"`). [STAFF SIGNAL: ordered-delivery tradeoff]

```
                    Unordered (default)        Ordered per subscription_id
                    ────────────────────       ─────────────────────────────
   sub_42 events:    [e1] [e2] [e3]            [e1]──►[e2]──►[e3]  (serial)
                      │    │    │                │
                      ▼    ▼    ▼                ▼ (must complete before next)
   workers handle:  parallel, any order        worker pool sized 1 per resource
```

Implementation: events for an ordered endpoint are partitioned by `hash(partition_key)`; each partition has effective concurrency 1. Throughput per resource is bounded by the slowest delivery to that endpoint.

**Skip-ahead policy on prolonged failure**: if event 42 for `sub_X` fails for >1 hour, do we hold events 43, 44, 45? Two modes, customer-selectable:

- **Strict ordering** (default for ordered endpoints): yes, hold. Eventually all dead-letter together.
- **Skip-ahead**: after `max_block_duration` (default 1h), 42 goes to DLQ and 43, 44, 45 proceed. Customer's payload includes a `sequence_number` so they can detect the gap.

State the tradeoff clearly to the customer at registration: strict ordering means one stuck event blocks all subsequent events for that resource; skip-ahead means events arrive out of order on prolonged failure. **Most production systems offer both per endpoint.**

## 11. Multi-Tenant Fairness and Quotas

[STAFF SIGNAL: tier-aware fairness]

**Fair scheduler weights** (DRR, Section 5) are tier-driven. Within a tier, secondary tie-breaker is age (oldest pending first).

**Per-customer rate limits** at the dispatcher (events accepted into the system) and at the worker (deliveries dispatched per customer per second):

| Tier | Accept rate | Deliver rate | Queue depth cap |
|---|---|---|---|
| Free | 100/s | 50/s | 100K events |
| Paid | 1K/s | 500/s | 1M events |
| Enterprise | 10K/s | 5K/s | 10M events |

When a customer exceeds their accept rate, we 429 the producer (or buffer with backpressure). When their queue depth hits the cap, we alert them and stop accepting new events (drop with audit log entry; their producer should retry later or we lose events — we choose lose-with-loud-alarm over silently filling our infra).

**Abuse detection**: anomaly detection on per-customer event rate (z-score against their 7-day baseline). 50x baseline → page customer's account team + auto-throttle to baseline×3 until human reviews. The "customer accidentally fired a webhook for every API call" failure is real and has cost real platforms real money.

## 12. Failure Modes and Graceful Degradation

[STAFF SIGNAL: failure mode precision]

**Slow-customer cascade**: prevented by per-endpoint concurrency cap (§5). The invariant.

**Regional outage hits 30% of customers simultaneously**: 30% of endpoints fail; their queues grow; retry storms compound. Mitigations:
- Aggregate failure-rate detector: if global delivery success drops below 80%, the retry cadence shifts to "regional outage mode" — retries spread further, backoffs lengthen, retry budget extended (avoid premature DLQing perfectly good customers whose cloud provider is having an hour).
- Worker fleet not under pressure (most workers idle waiting); only the affected queues grow.
- Storage pressure managed: in-flight queue can hold ~3 days of failed events for affected customers without overflow.

**Kafka cluster down** (event bus): producers fail; events buffer in producer-side outbox table (durability handled by producer). Webhook system is idle. When Kafka returns, drain outbox. Acceptable for <30 min; alarming beyond.

**Redis (scheduler state) down**: scheduler falls back to "open mode" — no per-endpoint concurrency cap enforcement, just a global cap. Worse isolation, but deliveries continue. Failure mode is degraded, not stopped.

**Worker fleet 50% loss** (e.g., bad deploy, AZ failure): surviving workers absorb load up to capacity; queue depth grows; retry budget extends are automatic. Auto-scaling brings new workers; full recovery in minutes.

**Customer endpoint silently drops events** (returns 200 OK, never processes): undetectable from our side. Mitigations:
- **Heartbeat events**: optional opt-in synthetic events (`webhook.heartbeat`) sent every 5 minutes; the customer's dashboard can show "we sent N heartbeats; your system reported processing M." Mismatch is a customer-side bug.
- Customer-side reconciliation tooling: we expose `GET /v1/events?since=...` so customer can pull and compare against their processed set. Belt-and-suspenders.

**Dispatcher delivers wrong event to wrong customer** (catastrophic bug): worker-level customer-ID assertion (§8) aborts the delivery before send. Pages on-call. Audit log captures the attempt for forensics.

**Signing key compromise**: customer rotates via API; old key kept valid for 24h overlap (can be expedited to immediate cutover on customer request). Already-delivered events with old key are still cryptographically valid — that's fine, the secret was for authentication of the sender, not forward secrecy.

**Source event store loses events**: webhook system has nothing to deliver. Not our problem to *fix*, but our problem to *detect*. Per-producer event-rate monitoring; if a producer's event rate drops 90% with no deploy correlation, page their team.

**Worker crash mid-delivery**: the delivery is in `IN_FLIGHT` state with a lease (60s default). Reconciler reaps expired leases and re-enqueues. **At-least-once is preserved.** This is also why exactly-once is impossible: the customer may have processed the event before the lease expired.

## 13. Observability and Customer Surface

**Internal metrics** (Prometheus + per-customer high-cardinality store):
- Delivery success rate (overall, per customer, per endpoint, per status-code class)
- p50/p99/p99.9 first-attempt latency, per-tier
- Queue depth (overall, per customer, per endpoint), histogram
- Retry rate, DLQ rate, replay rate
- Signing latency, DNS resolution latency, TLS handshake latency
- SSRF block events (any block is an alert — could be misconfig or attack)

**Customer-facing dashboard** [STAFF SIGNAL: customer-facing observability] — this is the differentiator. Stripe, Twilio, Shopify all do this; the ones that don't get a perpetual support burden of "I think I'm missing webhooks." For each endpoint:

- Live success rate (last 1h, 24h, 7d) with timeseries
- Recent failures: last 50 with status code, error, retry status, "view payload" link
- p50/p99 their endpoint's response latency (so they see *their* slowness)
- Current queue depth ("12 events waiting to be sent to you")
- DLQ size + replay UI
- Heartbeat status (if enabled)
- "Test webhook" button: sends a synthetic event, shows the request/response live

**Per-event audit**: customer can query by `event_id` and see every attempt: timestamp, our outbound IP, request headers (signature shown), customer's response code, response body (truncated), latency. Indispensable for "we're missing event evt_X" support tickets — answer is self-serve.

## 14. Operational Reality

**Deployment**: rolling, with graceful drain. Worker receives `SIGTERM`, stops accepting new deliveries, finishes in-flight (max 60s; longer in-flight ones get their lease handed off to a reconciler entry for re-pickup by a new worker). Full deploy takes ~10 min for 3K workers.

**Support tooling**: ops engineers have a "customer cockpit" with view of customer's queue, ability to pause delivery (e.g., customer requests during their incident), force-deliver a specific event, manually DLQ or replay. **Every internal action is audit-logged with engineer ID and reason.** Granular permissions: most engineers read-only; mutation requires elevated role + ticket reference.

**Upstream dependency**: webhook system durability is bounded by the event bus's durability. If Kafka loses a message before it's consumed, the delivery never happens. We replicate Kafka 3x across AZs; producers wait for ack from majority. Beyond that, the failure mode is a regional disaster — and we have bigger problems.

**Capacity headroom**: provisioned for 3x peak (so 60K/sec sustained). Auto-scale beyond on queue-depth alarms.

## 15. Tradeoffs Taken and What Would Change Them

- **Per-endpoint queues over per-customer queues**: more metadata, finer isolation. If endpoint count exceeded ~10M, we'd reconsider — virtualize per-endpoint within per-customer queues.
- **5-day retry budget**: suits SaaS norms. For payments-class events, we'd extend to 30 days with archival.
- **Default unordered, opt-in ordered**: throughput optimization. If most customers needed ordering, default would flip and we'd partition aggressively from the start.
- **Kafka + Redis hybrid**: gives durability + cheap fanout. Pure Pulsar (with sub-partitions) could replace both at cost of operational complexity.
- **5-min replay window for signatures**: tight enough to limit replay attacks, loose enough to absorb network/clock jitter. Customers with strict requirements can shorten.

## 16. What I'd Push Back On

[STAFF SIGNAL: saying no]

- **The implicit "exactly-once" expectation**, even if unstated by the interviewer. It is the single most common mistaken framing for webhook systems. If the prompt secretly wants exactly-once, the answer is: "you cannot have it across an untrusted network; here is at-least-once with strong idempotency guidance, which is what you actually want."
- **"Customers will implement signature verification correctly"** — many won't. We provide reference implementations in 8 languages, a verification-test endpoint, and we surface "your last 100 deliveries had 0% signature failures" as a positive health metric so customers notice when verification breaks.
- **"All customers are equal"** — they're not, and pretending otherwise produces a system where free-tier abuse degrades enterprise SLA. Tier-aware fair scheduling is non-optional at this scale.
- **"We'll just retry forever"** — retry budget exists precisely so terminal failures fail loudly. A `404` retried for a year is a bug, not robustness.
- **"Webhook delivery is the customer's reliability problem after we POST"** — partly true, but customer-facing observability is a product feature, not a courtesy. Without it, every "missing webhook" becomes a support ticket and the platform's reputation absorbs the cost.