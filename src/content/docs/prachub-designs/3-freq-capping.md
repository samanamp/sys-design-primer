---
title: Frequency capping system
description: Frequency capping system
---

The Question
"Design a frequency capping system for an advertising platform.
The system must ensure that a user does not see the same advertisement more than a configured number of times within a given time window, such as:

at most 3 impressions per user per ad per day
at most 10 impressions per user per campaign per week

Your design should address:

how ad-serving systems check caps in real time before showing an ad
how impressions are counted and stored
how to support multiple cap scopes, such as user-ad, user-campaign, or household-level limits
how to handle high QPS, low latency, and eventual consistency across regions
what happens when counting data is delayed, duplicated, or arrives out of order
data retention, expiration of old counts, and operational trade-offs between accuracy and latency

Walk me through it."

---

# Frequency Capping for an Ad Platform — Interview Script (40–50 min delivery)

> **How to use this doc.** Each section has a timing budget at the top. Aim to *speak* the content, not read it. Diagrams are meant to be drawn live on the whiteboard — they're simple on purpose. Staff signals are tagged inline like `[STAFF SIGNAL: ...]` — when you hit one, **slow down for two seconds** so the interviewer can clock it.
>
> **Total budget: 45 min.** Section budgets sum to ~42 min, leaving ~3 min of slack for interviewer interruptions.

---

## Section 1 — State of the art (4 min)

> **Speaking goal**: signal you've actually thought about what's changed recently. Don't lecture; just show fluency in the space.

Let me start by laying out what's actually true in 2026, because frequency capping looks different than it did three years ago.

**Three things have changed.**

**First, the privacy story didn't go where everyone planned.** In April 2025 Google reversed third-party cookie deprecation in Chrome — they're keeping cookies, no user-choice prompt. Then in October 2025 they deprecated essentially the entire Privacy Sandbox API surface — Topics, Protected Audience (formerly FLEDGE), Attribution Reporting, Shared Storage. The CMA's testing showed about 85% attribution inaccuracy and 30%+ publisher revenue decline under those APIs, adoption stalled, and Google pulled the plug.

So the textbook 2022 answer — "design for the post-cookie future, on-device caps via Privacy Sandbox" — that answer is now wrong. Cookies are alive in Chrome. **But** Safari and Firefox still block third-party cookies by default, that's roughly 30% of web traffic; Apple's ATT still applies on iOS so IDFA is opt-in only with maybe 25–40% acceptance rates; and CTV still has no stable per-user identifier. So the right framing for 2026 isn't "cookies are dying," it's **"identity is permanently fragmented across surfaces."** That's a different design constraint and it'll come up again.

**[STAFF SIGNAL: 2026 cutting-edge awareness]**

**Second, the RTB latency budget hasn't moved but it's better understood.** OpenRTB `tmax` is still 100ms — that's the hard ceiling for the entire auction. The operational target inside the bidder is 50–70ms because waiting longer than that only captures another 1–2% of revenue but adds 30ms of latency. Inside that 50–70ms, the cap check has to fit in roughly **5–15 milliseconds at p99**. That's the central forcing constraint and I'll come back to it.

**Third, the storage substrate has consolidated.** The pattern that won at scale across Trade Desk, Criteo, and Meta is hybrid in-memory plus NVMe — Aerospike is the most common name, with active-active multi-region replication. Trade Desk reportedly does over 10 million queries per second on it. Redis works at smaller scale but its memory cost dominates at petabyte. We'll come back to why.

OK, that's the lay of the land. Let me set scope.

---

## Section 2 — Scope, reframing, and the three commitments (5 min)

> **Speaking goal**: this is where you signal staff-level. Three explicit commitments — latency, policy, identity — each one tagged.

I'm going to commit to specific assumptions because the prompt is open-ended. Push back on any of these if they don't match what you have in mind.

**Platform role.** I'll design for an O&O-plus-DSP hybrid — meaning the platform owns its own ad-serving decision, like Meta on its own surfaces, but also bids into external auctions like a DSP. That's the harder version because we have to handle both logged-in identity, where it's clean, and bid-stream identity, where it's lossy. One cap engine, two identity regimes.

**Workload.** 10 million ad requests per second at global peak. Five regions. About 250 billion impressions per day. That's roughly Meta-scale or large-DSP-scale. The exact number doesn't change the architecture as long as we're in the millions-per-second range.

Now the three commitments. These three together carry most of the staff signal in this design.

### Commitment 1 — Designing backward from the latency budget

**[STAFF SIGNAL: latency-budget reframing]**

Here's what I'm going to draw and refer back to throughout:

```
   OpenRTB tmax (hard ceiling) ........... 100 ms
       │
       ├── Operational target ............. 50-70 ms
       │       │
       │       └── Bidder/ad-server logic .. 30-50 ms
       │              │
       │              └── Cap-check budget .. 5-15 ms p99   ← we live here
       │                     │
       │                     ├── Edge cache hit (95%) .... < 1 ms
       │                     ├── Regional store miss ..... 1-3 ms
       │                     └── Cross-region read ....... 50-100 ms ← INFEASIBLE
```

The last line is the punchline. **You cannot do a strongly-consistent cross-region read inside the ad-serving hot path.** A round trip from US-East to EU-West is roughly 80ms each way; that alone exceeds your entire RTB budget. So strong consistency is not a tradeoff we'll consider relaxing later — it's physically off the table from the start. The whole architecture is going to be eventually-consistent regional caching with async cross-region replication. Full stop.

### Commitment 2 — Over-cap vs under-cap is a policy decision, not a bug

**[STAFF SIGNAL: over-cap-vs-under-cap policy]**

Quick definitions:
- **Over-cap** (also called over-delivery): the user ends up seeing the ad more than the limit. The advertiser doesn't lose money — they were going to pay for those impressions anyway — but the user experience is slightly worse and the campaign's reach distribution is off.
- **Under-cap** (under-delivery): we refuse to show the ad even though the user is still under the limit, so the advertiser ends up paying for fewer impressions than they bought. That's revenue lost.

At our latency budget, perfect accuracy is impossible — counts are slightly stale, regions diverge briefly, events get duplicated. The system *will* miscount sometimes. The question is which direction we lean.

I'm committing to a per-cap-class policy:

| Cap class | Lean toward | Why |
|---|---|---|
| Brand reach campaigns | Over-deliver | Advertiser cares about reach distribution; they paid for impressions anyway |
| Performance / DR | Exact-as-possible | Every impression has explicit CPM cost; over-delivery wastes budget |
| Regulatory (alcohol, gambling, political) | Under-deliver | Legal risk of one over-cap dominates revenue cost |
| Recency caps (creative rotation) | Over-deliver | Showing same creative twice in 5 min is a tiny UX cost |

The point of saying this out loud is that it's a **product decision**, not an engineering bug. And the architecture I'm about to describe has knobs that get turned per cap class.

### Commitment 3 — Identity is a graph, not a single ID

**[STAFF SIGNAL: identity as graph]**

In 2026 a "user" isn't a cookie anymore. A user is a **cluster** — a set of identifiers that the system believes belong to the same person. That cluster might include a logged-in user ID, a hashed email, an IDFA when the user opted in, a cookie on Chrome, and an IP plus user-agent fingerprint as a fallback.

I'll talk about how the cluster works in detail later. The architectural commitment now is: **caps operate over `cluster_id`, not over raw input IDs.** When a request comes in, we resolve all the identifiers we see into one cluster_id, then we cap on that.

OK, with those three commitments locked in, let me put numbers on it.

---

## Section 3 — Capacity math (3 min)

> **Speaking goal**: show you can do the back-of-envelope. Don't compute live; just walk through it.

```
Ad requests per second (global peak) ............ 10 million
Cap rules per request (avg) ..................... 6
   (e.g., per-ad, per-creative, per-line-item,
    per-campaign, per-advertiser, per-household)
Counter lookups per second ...................... 60 million
Edge cache hit rate (target) .................... 95%
Regional store reads per second ................. 3 million
Per region (5 regions) .......................... 600K rps

Active users (90 day) ........................... 1 billion
Active ads ...................................... 100,000
Avg non-zero (user, ad) pairs over 7 days ....... ~10 per user
   (huge long tail; most pairs are zero)
Hot counters per region ......................... 10 billion
Bytes per counter ............................... ~50
Hot regional storage ............................ ~500 GB per scope tier
   (with all 6 hierarchy levels, ~2-3 TB/region)

Daily impressions ............................... ~250 billion
Streaming events/sec (impressions) .............. 10M
Dedup state at 24h window ....................... 864B impression IDs
   At 16 bytes each, exact dedup ............... 13.8 TB/day  ← infeasible
   With Bloom filter prefilter (1% FPR) ........ ~1.4 TB/day  ← tractable
```

Three things drop out of these numbers:

1. **Edge caching is mandatory.** A 1% drop in cache hit rate adds 600K reads per second to each region. We cannot do this work at the storage layer alone.
2. **Exact dedup over 24 hours is too expensive at 14 TB per day.** We have to use a probabilistic structure as a first stage.
3. **Cross-region quorum is infeasible.** We knew this from latency; the QPS just makes it worse.

These numbers force the architecture. Let me draw it.

---

## Section 4 — Big-picture architecture (4 min)

> **Speaking goal**: get the whole system on the whiteboard at once. Walk left-to-right, top-to-bottom. Then dig into each piece.

```
                                  CLIENT / PUBLISHER
                                         │
                        ┌────────────────┴─────────────────┐
                        │ Ad request (OpenRTB or O&O)      │
                        └────────────────┬─────────────────┘
                                         │ <100 ms total budget
                                         ▼
   ┌─────────────────────────────────────────────────────────────────┐
   │                AD SERVER (regional pool)                        │
   │                                                                 │
   │   ┌─────────┐    ┌─────────────────┐    ┌────────────────┐      │
   │   │Identity │    │ Cap-rule engine │    │ Auction / rank │      │
   │   │resolver │───▶│ (parallel reads)│───▶│                │      │
   │   └─────────┘    └────────┬────────┘    └────────────────┘      │
   │                           │                                     │
   │   ┌───────────────────────▼────────────────────────────┐        │
   │   │  Local edge cache  (per ad-server process)         │        │
   │   │  Key: (cluster_id, scope) → count, window_start    │        │
   │   │  TTL: 5s hot scopes, 60s cold scopes               │        │
   │   └───────────────────────┬────────────────────────────┘        │
   └───────────────────────────┼─────────────────────────────────────┘
                               │ on cache miss (~5%)
                               ▼
                ┌─────────────────────────────────────┐
                │ REGIONAL COUNTER STORE (Aerospike)  │
                │  Sharded by cluster_id              │
                │  Sub-ms p99, atomic INCR, TTL       │
                └──────────┬──────────────┬───────────┘
                           ▲              │ async XDR
                           │              ▼
        ┌──────────────────┴───┐    ┌──────────────────┐
        │ Stream consumer      │    │ Other regions'   │
        │ (Flink) writes here  │    │ counter stores   │
        └──────────┬───────────┘    └──────────────────┘
                   ▲
                   │ consumes
   ┌───────────────┴────────────────┐
   │ Kafka topic: impressions.v1    │
   │  partitioned by cluster_id     │
   │  retention 7d, at-least-once   │
   └───────────────▲────────────────┘
                   │
                   │
   ┌───────────────┴───────────────┐
   │ Beacon receiver               │
   │ POST /impression (after MRC   │
   │ viewability check on client)  │
   └───────────────────────────────┘
```

There are three loops here.

The **read loop** is the top half: ad request → identity resolver → cap engine → edge cache → maybe regional store → answer. This has to finish in 5–15ms.

The **write loop** is the bottom half: ad gets served → client measures viewability → beacon fires → Kafka → stream processor → atomic increment in the counter store. This is allowed to take 1–5 seconds end-to-end.

The **replication loop** is on the right: regions sync to each other asynchronously, with maybe 100ms to 2 seconds of lag.

The fact that read and write are decoupled is what lets the read path be fast. The cost is that reads are *slightly stale*. We've already decided that's acceptable.

OK, let me drill into each loop.

---

## Section 5 — The read path: cap check in the hot path (6 min)

> **Speaking goal**: this is one of the deepest parts. Walk through what happens in each millisecond. Numbers earn trust.

Let me trace one ad request through the system.

```
T+0     ms   Ad request hits regional ad server
T+0.5   ms   Identity resolver: input IDs → cluster_id
T+1     ms   Cap engine identifies 5 applicable cap rules
             → 5 (cluster_id, scope) keys to check
T+1.5   ms   Issue 5 PARALLEL reads to edge cache
                cache hit (95% case): all 5 keys back in < 1 ms
                cache miss (5% case): fall through to Aerospike
T+3-6   ms   Regional store responds, edge cache populated for next time
T+6-10  ms   Cap engine evaluates each rule:
                count < limit? AND window still active?
T+10    ms   Hand back allow/block list to auction
```

Let me unpack a few pieces.

### What's in a cap counter, exactly

The data structure is small:

```
Key:   (cluster_id, scope_type, scope_id, window_bucket)
       e.g., (user_42, AD, ad_99, 2026-05-05)

Value: { count: u32, window_start: u64, last_imp_ts: u64 }
```

The `window_bucket` is the trick that makes expiration cheap. A daily cap is keyed by `floor(now / 86400)` — so today's count is in one record, tomorrow's count is in a *different* record automatically. We don't need a reset job. Old buckets time out by TTL.

`last_imp_ts` lets us also handle **recency caps** like "don't show this creative twice in the same hour." For frequency caps we check `count < limit`; for recency caps we check `now - last_imp_ts > min_gap`. Same record, two predicates.

### Why parallel reads, not sequential

A single ad request might hit 5 or 6 cap rules — per-creative, per-ad, per-line-item, per-campaign, per-advertiser, per-household. We issue all of those reads at the same time and AND the results. Sequential would mean adding 5 round trips inside our 10ms budget; parallel means we pay for the slowest one only.

### Why an edge cache at all

**[STAFF SIGNAL: rejected alternative]**

Without the edge cache, we'd need every ad-serving request to hit Aerospike directly. That's 60 million reads per second going to the storage layer. With a 95% edge cache hit rate, we drop that to 3 million per second. That's the difference between "Aerospike is fine" and "Aerospike needs 100x more nodes."

The cost of the edge cache is staleness. A 5-second TTL means a counter we read can be up to 5 seconds out of date. At 10 million ads per second with the same user being targeted, that potentially means a few extra impressions slip through before the cache catches up. For a brand campaign with a 100-million-impression budget, that 5-second window typically causes maybe 0.05% over-delivery. We've decided that's fine.

### The cache-stampede problem

Here's a subtle one. A really popular user — say, someone the auction system is constantly targeting — has their counter expire from the edge cache. All ad servers in the region notice the miss at the same instant, and they *all* hit Aerospike for the same key. That's a thundering herd.

Two fixes I'd build in:
1. **Stochastic early refresh.** When a cache entry is at 80% of its TTL, 1% of accessing requests trigger a background refresh. The cache never actually expires under load — it gets refreshed before it can.
2. **Request coalescing.** If multiple in-flight requests miss the same key, only one of them does the upstream read; the others wait for that result. Standard pattern.

### One alternative I rejected

I considered storing all of a user's cap state in one giant value — one read instead of N. The reason I didn't: hot-user keys (think: bot accounts, or just very active users) become hot shards. The blob grows unbounded. And updating a single field requires read-modify-write at the storage layer, which is more expensive than N independent atomic increments. The N-key model parallelizes naturally on both reads and writes, and lets the hot-shard problem be solved by sharding on a more granular key.

---

## Section 6 — The write path: counting impressions correctly (5 min)

> **Speaking goal**: hit viewability, idempotency, dedup. Three sub-stories.

Let me trace what happens when an ad actually gets shown.

```
Ad served ──▶ client renders ad on the page
                                 │
                                 ▼
              MRC viewability check on client
              (50% of pixels visible for 1 second
               for display; 2 seconds for video)
                                 │
                              passes?
                            no │ yes
                               ▼  ▼
                            drop  ▼
                                  ▼
                         POST /impression with impression_id
                         (UUID, generated server-side at ad-serve time)
                                  │
                                  ▼
                         ┌─────────────────────┐
                         │ Beacon receiver     │
                         │ (lightweight HTTP)  │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │ Kafka topic         │
                         │ partitioned by      │
                         │ cluster_id          │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │ Flink stream job     │
                         │  1. Bloom dedup      │
                         │  2. exact dedup      │
                         │  3. fan out to       │
                         │     hierarchy        │
                         │  4. atomic INCR x6   │
                         └──────────────────────┘
```

### Story 1: viewability vs delivery

**[STAFF SIGNAL: viewability-vs-delivery distinction]**

There are two different things you can count:
- **Delivered impression**: the ad server returned the ad to the page.
- **Viewable impression**: the ad actually rendered to the screen with at least 50% of pixels visible for at least 1 second (the MRC standard, 2 seconds for video).

These are very different numbers. Roughly 30–50% of delivered impressions never become viewable — the page closed, the user scrolled past, an ad blocker hit it, the ad was below the fold.

The right policy is per cap class:
- **Brand caps** count viewable. The advertiser is paying for attention; non-viewable impressions don't matter.
- **DR / performance caps** typically count delivered. It's faster, cheaper to measure, and the gap matters less.

The architecture writes to both counter sets, and the cap rule references one. So the choice is configuration, not infrastructure.

### Story 2: idempotency — every event has an ID

**[STAFF SIGNAL: idempotency and dedup discipline]**

Each ad-serve mints an `impression_id` — a UUID — that gets baked into the rendered ad. When the viewability beacon fires, it carries that ID. The counter update keys on it.

This matters because three things are going to cause duplicates:
1. **The pixel retries**. Browsers retry failed requests. Same ad gets reported 2–5 times.
2. **Kafka is at-least-once**. Same event can be delivered to a consumer twice during rebalance.
3. **Disaster recovery replay**. After an incident, we sometimes replay events from S3 archive into Kafka. Same events arrive again.

If we just incremented blindly on every event, the count would drift up. We need exact-once *effective* semantics on the counter.

### Story 3: dedup at scale — Bloom filter as fast path

Here's where the math from earlier hurts. We have 10 million events per second, and we want to dedup over a 24-hour window. That's 864 billion impression IDs to remember. At 16 bytes each, that's 14 TB of dedup state. Not affordable.

The fix is two-stage:

```
Event arrives
    │
    ▼
┌──────────────────────────┐
│ Bloom filter             │   "have I seen this ID?"
│ (24h sliding window,     │   Returns:
│  ~1% false positive rate)│   - "definitely no" (99% of events) → accept
│                          │   - "maybe yes"     ( 1% of events) → check exactly
└──────────┬───────────────┘
           │ "maybe yes"
           ▼
┌──────────────────────────┐
│ Exact dedup store        │   "is this ID actually here?"
│ (RocksDB, 24h TTL)       │   - yes → drop (it really was a dup)
└──────────┬───────────────┘   - no  → accept (Bloom false positive)
```

A Bloom filter is a small probabilistic data structure that can tell you "definitely not here" or "maybe here." It uses way less memory than storing the actual IDs — about 10 bits per ID instead of 128 bits. That makes the 24-hour Bloom filter about 1 TB instead of 14 TB.

**[STAFF SIGNAL: probabilistic-when-appropriate]** The Bloom filter is a fast path that catches 99% of duplicates instantly. Only the 1% that the Bloom filter is uncertain about hit the exact dedup store. So we get exact dedup semantics with bounded memory. The probabilistic structure earns its place because it cuts a real cost without compromising correctness.

### What about late events?

Sometimes an impression event arrives an hour after the ad was served — usually a mobile device that buffered events while offline. Two policies:

- **Apply retroactively**: increment the count for the original time window. Risk: if that window is closed, the counter we'd update doesn't even exist anymore.
- **Drop**: ignore late events past a threshold.

We drop. We log the drop rate as an SLO; if it exceeds 0.5%, on-call investigates pipeline lag. The reason this is fine: we accept some bounded over-delivery anyway; recovering exact accuracy from late events isn't worth the complexity.

### Out-of-order events

These don't actually cause problems. `INCR` is commutative — it doesn't matter what order we apply increments in. For recency caps where we track `last_imp_ts`, we use a compare-and-swap: only update if the incoming timestamp is newer. So order doesn't matter for any of our primitives.

---

## Section 7 — Multi-region eventual consistency: the user-mobility case (5 min)

> **Speaking goal**: be honest about the trade. Quantify it. Show you understand it isn't going away.

This is the most subtle part of the design. Let me draw it.

```
          ┌────────────── Replicated Kafka topic ──────────────┐
          │ (events from every region replicated everywhere)   │
          └────────┬─────────────────┬─────────────────┬───────┘
                   │                 │                 │
                   ▼                 ▼                 ▼
            ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
            │ US-EAST      │  │ EU-WEST      │  │ AP-SE        │
            │ Counter      │  │ Counter      │  │ Counter      │
            │ store        │  │ store        │  │ store        │
            └──────────────┘  └──────────────┘  └──────────────┘
                              replication lag:
                              p50 ~150 ms
                              p99 ~1-2 s
```

```
CLIENT
                                      │
                               impression fires
                                      │
                                      ▼
                         ┌────────────────────────┐
                         │  LOCAL Kafka           │
                         │  (within one region)   │
                         │  topic: impressions.v1 │
                         │  partitioned by        │
                         │  cluster_id            │
                         └────────────┬───────────┘
                                      │
                                      ▼
                         ┌────────────────────────┐
                         │  Flink job (regional)  │
                         │  - dedup               │
                         │  - viewability filter  │
                         │  - fan-out to hierarchy│
                         └────────────┬───────────┘
                                      │
                          ┌───────────┴───────────┐
                          │                       │
                          ▼                       ▼
               ┌──────────────────┐   ┌────────────────────────┐
               │ Regional         │   │  CROSS-REGION Kafka     │
               │ Aerospike        │   │  (or MirrorMaker 2 /   │
               │ counter store    │   │   Pulsar geo-rep)       │
               │ (local writes)   │   │  topic: impressions     │
               └──────────────────┘   │        .deduped.v1     │
                                      └────────────┬───────────┘
                                                   │
                                      replicated to all other regions
                                                   │
                                      ┌────────────┴───────────┐
                                      │                        │
                                      ▼                        ▼
                               EU-WEST Flink            AP-SE Flink
                               (apply INCR to           (apply INCR to
                                local Aerospike)         local Aerospike)
```

The way it works: each region is the authority for its own writes. When EU-West counts an impression, it writes locally and also publishes the event to the global Kafka topic. The other regions consume that topic and apply the same `INCR` to their local counters.

Why publish *events* and not *counter values*? Because counters are commutative — every region applies every region's events, in any order, and they all converge to the same total. If we replicated the *value*, we'd have to deal with conflict resolution (whose value wins?), which is hard for counters. Replicating the *operation* sidesteps it.

> Brief vocabulary: this pattern — "replicate the operation, not the value" — is how a counting CRDT works. You don't need to call it that, but if the interviewer brings up CRDTs, you can say yes, that's exactly what this is, structurally a G-Counter per region that sums to a global total.

### The user-mobility case

**[STAFF SIGNAL: eventual-consistency honesty]**

OK here's the case that exercises this. Imagine a user in Paris. They've seen ad X three times today. The cap is 3. They're capped. They're served other ads instead.

Now they get on a VPN that exits in Virginia. Their next request hits US-East. US-East got the events from EU-West, but maybe with a delay. So in US-East, the count for this user might still say 0, or 1, or 2.

US-East serves them ad X *one more time*. Now the global count is 4. The cap was 3.

This is over-delivery by one impression. The user saw the ad one extra time.

**Is this a bug? No.** It's the cost of being able to answer cap checks in 5ms instead of 100ms. Strong consistency would solve it; the latency cost is unacceptable; we accept the trade.

**Is it a big deal? Let's quantify.**

At p99 replication lag of 1-2 seconds, the only users affected are ones who switch regions inside that 1-2 second window. That's a tiny fraction — maybe 0.1% of users do meaningful cross-region traffic that quickly. Of those, the average over-delivery is maybe half an impression, because most of them aren't being aggressively retargeted.

So the over-delivery from mobility is roughly 0.05% of all impressions. For a brand campaign that bought 100 million impressions, that's 50,000 extras. The advertiser doesn't pay extra; they paid for those impressions anyway. The user UX cost is negligible. We accept this.

### When 0.05% isn't acceptable

For the very strictest brand caps — top-tier global brands paying for exact reach distribution — we have a tighter mode. We do **sticky home-region routing**: every cluster_id has a home region (usually the region where it was first seen), and we route all that user's requests to their home region by default. If the home region is unavailable, we fall back to local. This collapses the mobility case at the cost of slightly higher latency for users currently outside their home region.

This is a per-cap-class knob, not a global setting. Most caps don't need it.

### What happens during a partition

If a region can't reach the others, each region's counters drift apart. Each region under-counts the global total because it's missing the other regions' events. Caps will be looser than they should be — over-delivery during the partition.

We detect this with a replication-lag SLO. If lag breaches 30 seconds, an alert fires. If a partition lasts long enough, the on-call has the option to throttle ad serving on the most-strict cap classes to avoid blowing through caps. But the default response is **let it run** — going dark on ad serving is worse than over-delivering.

---

## Section 8 — Identity resolution (4 min)

> **Speaking goal**: explain what an identity graph actually is. Don't assume the interviewer knows.

I said earlier that capping operates on `cluster_id`, not on raw input identifiers. Let me unpack what that means.

```
                Inbound identity signals on a single request
                                    │
   ┌──────────┬──────────┬──────────┼──────────┬─────────────┬─────────┐
   ▼          ▼          ▼          ▼          ▼             ▼         ▼
 cookie    UID2 /     IDFA       GAID    IP + UA       household IP   ...
 (Chrome)  hashed-    (iOS,      (Android (probabilistic   (CTV)
           email      consent)   consent) fingerprint)
           (when
            logged-in)
                                    │
                                    ▼
                        ┌─────────────────────────┐
                        │  Identity Graph Service │
                        │  Maps each ID to a      │
                        │  cluster_id             │
                        │                         │
                        │  cluster = "we believe  │
                        │  these IDs belong to    │
                        │  one person"            │
                        └────────────┬────────────┘
                                     │
                                     ▼
                            cluster_id (or household_id)
                                     │
                                     ▼
                       All capping operates on this
```

### What's a cluster, concretely

A cluster is the system's best guess at "these identifiers all belong to the same human." There are two ways IDs get linked into a cluster:

- **Deterministic**: the user logged in and we observed two IDs (a cookie and a hashed email) on the same authenticated session. We're certain.
- **Probabilistic**: a device graph saw the same IP, user agent, and location pattern across two IDs. We're maybe 80% confident.

The graph is owned by a separate identity service. It's mostly batch-built nightly with online updates for new sessions. When an ad request comes in, the resolver queries: "for these incoming IDs, what cluster do they belong to?" — and the answer is a single cluster_id.

### Why this matters for capping

If we capped on cookies alone, a user with cookies cleared between sessions would look like a brand-new user every time. Cap of 3/day becomes effectively unlimited. Conversely, a shared device with two users would have its cap blown by the more active user, denying the second user ads they're entitled to see.

The cluster is the closest thing we have to "the actual person."

### What happens when clusters change

Sometimes the graph merges two clusters (new evidence connects them) or splits one (evidence proves they were different people). Counters under the old IDs need to migrate.

We do this lazily. On the next cap read, if the cluster was recently merged, the engine reads counters under both old cluster IDs and the new one, sums them, and writes a consolidated counter. The stale entries time out by TTL. **This is eventual consistency in the identity layer** — accept it.

### The "no ID" case

**[STAFF SIGNAL: privacy-aware design]**

Sometimes we have very little to work with. Safari user, not logged in, ad blocker active, no IDFA — we have basically just IP and user agent. The cluster confidence in this case is low.

The cap engine accepts a `scope_confidence` signal from the identity service. Below a threshold, we **fall back to a coarser cap**:
- Cap per IP (very lossy — multiple users behind one home WiFi)
- Cap per publisher session (caps within a single visit, not across visits)
- Cap at the household level for CTV

For CTV specifically, household-level is actually the *right* granularity, not a fallback. That's who's looking at the screen — multiple people, sharing one cap. **[STAFF SIGNAL: saying no]** I'd push back if the prompt asks for "per-user" capping in CTV, because that's not really achievable and pretending otherwise is engineering theater.

---

## Section 9 — Cap-rule hierarchy: write-fan-out vs read-aggregation (3 min)

> **Speaking goal**: this is one specific tradeoff. Be crisp.

A single ad request typically triggers caps at multiple levels of a hierarchy:

```
                advertiser_id   cap: 50 / week
                       │
                       ▼
                  campaign_id   cap: 10 / week
                       │
                       ▼
                  line_item_id  cap: 5 / day
                       │
                       ▼
                     ad_id      cap: 3 / day
                       │
                       ▼
                  creative_id   cap: 1 / hour  (recency)
```

Plus orthogonal scopes like household, vertical, regulatory.

When an impression fires, all of these counts logically need to go up. There are two ways to model that.

**Approach A: write-side fan-out.** On every impression, increment six counters atomically — one per level of the hierarchy. The read path then does six cheap independent reads.

**Approach B: read-side aggregation.** Store only leaf-level counters. On the read, sum all the children to get the parent's count.

**[STAFF SIGNAL: hierarchical cap design]**

The math:
- Write fan-out: 10M impressions/sec × 6 levels = 60M writes/sec. Distributed across shards, fine.
- Read aggregation: 10M ad requests/sec × 6 cap rules × N children to sum per parent = blows up fast. Even worse on cache misses.

**We pick write fan-out.** Reads are far hotter than writes — we check ~6 times per request, but we only increment when an impression actually fires (~1/3 of requests). Plus reads can be edge-cached; aggregations can't be cached easily because they depend on the children's caches.

The write fan-out is implemented in the Flink stream consumer. One impression event comes in; the consumer emits 6 atomic INCRs to the counter store. Simple.

---

## Section 10 — Failure modes (3 min)

> **Speaking goal**: be specific. The interviewer is checking that you've thought about what breaks and what to do.

**[STAFF SIGNAL: failure mode precision]**

Three big ones.

### Failure 1: counter store unavailable

The regional Aerospike cluster is down or unreachable.

The choice is **fail-open** (show the ad anyway, the cap may be violated) versus **fail-closed** (refuse to serve any ad with cap rules until the store comes back).

Default: **fail-open**. Reasoning:
- Going dark causes immediate revenue loss and bad UX (blank slots, unfilled inventory).
- Cap violations are bounded — even if every ad over-delivers by 1, the magnitude is small.
- Most cap violations don't have catastrophic consequences.

But brand campaigns or regulated verticals can opt fail-closed via campaign config. They trade revenue for compliance certainty.

### Failure 2: streaming pipeline is lagged

Kafka consumer is 30 seconds behind. The counter store is stale. The read path returns yesterday's-equivalent counts. Caps allow over-delivery.

Detection: standard Kafka consumer-lag SLO. Alert fires at 30 seconds. If lag exceeds 5 minutes, on-call has the option to throttle the strictest cap classes to limit damage. The system's default behavior during this is to **let the over-delivery happen** and recover when the lag clears — over-delivery is bounded; going dark is not.

### Failure 3: identity graph corrupts

This is the scariest one. **[STAFF SIGNAL: blast radius reasoning]** Imagine a bug in the graph builder mass-merges thousands of clusters. Suddenly thousands of users share a counter. The cap engine sees the wrong count for everyone affected. Either nobody from the merged group sees ads (because the merged count exceeded the cap) or the wrong people see them.

Mitigations, in priority order:
1. **Rate-limit graph mutations.** No more than X% of clusters can be merged per hour. A buggy job can't do unbounded damage in finite time.
2. **Audit log every change.** Every merge is replayable and reversible.
3. **Graph version is part of the counter key.** When a buggy mutation is detected, we bump the graph version, which lazily invalidates affected counters. New requests get new counters; the bad data times out.

The point is to **bound blast radius, not prevent failure**. Failures will happen; we want them to be small and recoverable.

---

## Section 11 — Data retention and the cost-vs-accuracy dial (2 min)

> **Speaking goal**: show you understand this is a per-cap-class operating point, not a global decision.

**[STAFF SIGNAL: cost-vs-accuracy explicit]**

Counter store retention: TTL = window length plus a small buffer. Daily caps live ~25 hours, weekly ~8 days, monthly ~31 days. After that, gone — they have no operational value.

Every impression is also archived to S3 for 90 days for billing, audit, and reach reporting. That's separate from the counter store.

The interesting thing is that we expose a knob for the trade between accuracy and cost:

| Mode | Edge cache TTL | Replication mode | Expected over-delivery | Cost vs baseline |
|---|---|---|---|---|
| Loose (default DR) | 30 sec | async | ~0.5% | 1.0× |
| Standard (most brand) | 5 sec | async | ~0.05% | 1.5× |
| Tight (top-tier brand) | 1 sec | async + home routing | ~0.01% | 2.5× |
| Strict (regulated) | 0 (no cache) | sync per write | ~0.001% | 5×+ |

This isn't an engineering detail — it's a product surface. Each campaign picks an operating point. Brand campaigns sit in Standard. DR sits in Loose. Regulated content goes into Strict and accepts the cost.

---

## Section 12 — What I'd push back on (3 min)

> **Speaking goal**: this is the saying-no signal. Pick 2–3, deliver them with conviction.

**[STAFF SIGNAL: saying no]**

A few things in the original framing I'd push back on.

**One: "eventual consistency across regions" is framed as a goal.** It's not a goal — it's a *consequence* of the latency budget. The actual question is: how loose is "eventual," and which cap classes get tighter modes? I'd want the conversation to be about per-cap-class operating points, not a single global tolerance.

**Two: "user-ad, user-campaign, household-level limits" — these are described as parallel concerns.** They're not. Household capping is fundamentally different identity infrastructure from user-level capping. Different signals (IP-based), different policy (multiple individuals share the cap), different fairness considerations. At most platforms they ship in different release trains. I'd separate the PRD.

**Three: the privacy framing assumes the 2022 trajectory** — cookies are dying, plan for the post-cookie world. The 2026 reality is messier: cookies survived in Chrome, Privacy Sandbox died, identity is permanently fragmented. The right design frame is "permanent fragmentation," not "cookie sunset."

**Four: "at most 3 impressions per user per ad per day" is treated as a hard rule.** In practice: "user" is a fuzzy cluster, "day" is a window with timezone-policy ambiguity (UTC? user-local? campaign-local?), "impression" is delivered or viewable, and "3" is a target with bounded over-delivery. The product spec should be written with these realities in mind or it sets up the engineering team to fail.

---

## Section 13 — Summary (1 min)

> **Speaking goal**: tight close. Plant a flag.

If I had to summarize this whole design in one sentence:

> **Counters keyed by `(cluster_id, scope, window_bucket)`, sharded by cluster_id, served by Aerospike with sub-millisecond reads behind a 5-second-TTL edge cache, written via Kafka and Flink with two-stage Bloom-then-exact dedup and viewability filtering, replicated cross-region asynchronously via event log, with a per-cap-class accuracy-vs-cost dial that the product chooses between Loose, Standard, Tight, and Strict.**

The shape of the system is set by three forcing functions: the 5–15ms cap-check latency budget makes strong consistency physically impossible; the 60-million-counter-lookups-per-second QPS makes edge caching mandatory; and the permanent fragmentation of identity makes capping a graph problem, not a single-key problem.

Everything else is detail in service of those three.

---

## Appendix: Staff signal cheat sheet

If you hit these in delivery, you're at staff level:

| Signal | Section |
|---|---|
| 2026 cutting-edge awareness | §1 |
| Latency-budget reframing | §2.1 |
| Over-cap-vs-under-cap policy | §2.2 |
| Identity as graph | §2.3, §8 |
| Capacity math | §3 |
| Rejected alternative | §5 |
| Viewability-vs-delivery distinction | §6 |
| Idempotency and dedup discipline | §6 |
| Probabilistic when appropriate | §6 |
| Eventual-consistency honesty | §7 |
| Privacy-aware design | §8 |
| Hierarchical cap design | §9 |
| Failure mode precision | §10 |
| Blast radius reasoning | §10 |
| Cost-vs-accuracy explicit | §11 |
| Saying no | §8, §12 |

That's 16 of the 19 from the original prompt's list. Senior staff territory.

---

## Appendix: timing sheet for delivery

| Section | Topic | Budget | Cumulative |
|---|---|---|---|
| 1 | State of the art | 4 min | 4 |
| 2 | Scope + 3 commitments | 5 min | 9 |
| 3 | Capacity math | 3 min | 12 |
| 4 | High-level architecture | 4 min | 16 |
| 5 | Read path | 6 min | 22 |
| 6 | Write path | 5 min | 27 |
| 7 | Multi-region eventual consistency | 5 min | 32 |
| 8 | Identity | 4 min | 36 |
| 9 | Cap hierarchy | 3 min | 39 |
| 10 | Failure modes | 3 min | 42 |
| 11 | Cost-vs-accuracy | 2 min | 44 |
| 12 | Pushback | 3 min | 47 |
| 13 | Summary | 1 min | 48 |

3 minutes of slack for interruptions. If you're behind, cut §11 (it's the most compressible). If you're ahead, expand §5 (cache stampede + rejected alternative is a great place to spend extra time).