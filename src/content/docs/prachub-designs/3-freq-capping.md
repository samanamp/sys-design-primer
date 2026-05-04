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

# Frequency Capping for an Ad Platform — Staff-Level System Design

> Target role: Staff/Sr. Staff (E6/L7) at a large ad-tech platform. Date: 2026.
> The prompt asks for a server-side per-user frequency capping system. I'll commit to the architecture, but I'll push back on two of its assumptions before I'm done — the privacy framing in the prompt is stale, and "global consistency" is not the right success criterion.

---

## 1. Research pass — state of the art, 2026

I deliberately went looking for what's *changed* since the textbook ad-tech designs of 2018-2020. Three things matter, and they materially change what a 2026 staff answer looks like.

**1.1 The privacy landscape did not go where everyone planned.** As of April 2025, Google formally abandoned third-party cookie deprecation in Chrome, dropped the "user choice prompt," and kept third-party cookies on by default. Then in October 2025, Google deprecated essentially all of the Privacy Sandbox APIs that were supposed to be the replacement — Topics, Protected Audience (formerly FLEDGE), Attribution Reporting, Shared Storage, Private Aggregation, Protected App Signals on Android. The CMA's testing showed ~85% attribution inaccuracy and 30%+ publisher revenue declines under the Privacy Sandbox. Industry adoption stalled, and Google pulled the plug. Chrome still has Topics and Ad Privacy toggles for end users, but the API surface advertisers were going to migrate to is mostly gone.

The implication for capping: cookie-based capping in Chrome is *not* sunsetting on a known timeline. The "post-cookie world" prediction underdelivered. But — and this is critical — **Safari and Firefox still block third-party cookies by default**, which is roughly 30% of browser traffic; **Apple ATT still applies on iOS** with IDFA opt-in rates somewhere in the 25–40% range; and **CTV has no stable per-user identifier** at all in most environments. So the staff-level framing isn't "design for the post-cookie future"; it's **design for an identity environment that is permanently fragmented across surfaces** — Chrome cookies on web, hashed-email/UID2 on logged-in surfaces, IDFA-with-consent on iOS, GAID on Android, household-IP on CTV. Capping must operate over an identity *graph*, not a single ID.

**1.2 RTB latency budget is unchanged but better understood.** OpenRTB `tmax` is still 100ms hard ceiling (some SSPs go to 120ms p99, Magnite CTV up to 250ms). Operational target inside the bidder is 50–70ms — wait longer and you only capture 1–2% additional revenue but eat 30ms of latency. Cap check has to fit inside ~5–15ms p99; it cannot be a cross-region quorum read.

**1.3 Storage substrate has consolidated.** The pattern that won at scale is **Aerospike** with hybrid memory + NVMe, deployed active-active multi-region with XDR (cross-datacenter replication). Trade Desk reportedly does 10M+ QPS on Aerospike, and platforms like FuboTV explicitly cite Aerospike for frequency-cap counters. Redis Cluster is fine at smaller scale but its memory cost dominates at petabyte. ScyllaDB shows up at Discord-style scale. The interesting design pattern across all of them is the same: **TTL-managed counters keyed by `(identity, scope, window)`, sharded by identity, with async cross-region replication**.

**1.4 Viewability is not delivery.** The MRC standard (50% of pixels for 1 continuous second for display, 2s for video) defines what counts as a "viewable" impression. Ad servers see *delivered* impressions; only the client-side measurement pixel sees *viewable* ones. Capping policy must commit to one of these — modern brand-buying platforms cap on viewable; most performance/DR platforms cap on delivered because it's faster and cheaper. This gap is on the order of 30–50% of delivered impressions.

**1.5 The retiring Privacy Sandbox left one durable lesson.** Even though the APIs are dead, the *architectural pattern* of on-device frequency capping is real and survives outside Privacy Sandbox: SKAdNetwork, SKAN 4, AdAttributionKit, browser-local storage in walled-garden SDKs. For unauthenticated traffic where you can't form a server-side identity, **delegating capping to the device** is a viable fallback even today.

---

## 2. Scope and reframing

I'm going to commit to specific assumptions because the prompt is open-ended. Push back on any that don't match the actual problem.

**Platform role: O&O + DSP hybrid.** I'll design for a platform that owns the ad-serving decision (Meta-style on its own surfaces, plus DSP-style off-platform via OpenRTB). This is the harder version because we have to handle *both* logged-in identity (good) and bid-stream identity (lossy) with one cap engine.

**Workload: 10M ad requests/sec global peak, 5 regions.** That's roughly Meta-scale or large-DSP-scale. Order-of-magnitude is what matters; the numbers move within ±3× without the architecture changing.

**Cap rules: predicate-based, bounded.** Most caps are flat (`N per period per scope`). Some are recency (`not seen in last X minutes`). I'll support both. I will *not* support arbitrary user-defined Turing-complete predicates — that's a different system (real-time policy engine).

**Identity: graph over ID cluster.** Discussed in §7.3.

**[STAFF SIGNAL: latency-budget reframing]** The forcing constraint, which I'll design backward from:

```
OpenRTB tmax ........................ 100 ms hard ceiling
  Operational target .................. 50-70 ms
  Bidder/ad-server internal logic ..... 30-50 ms
    Cap-check budget (parallel) ....... 5-15 ms p99
      Edge cache hit (95%) ............ < 1 ms
      Regional store (cache miss) ..... 1-3 ms
      Cross-region read ............... 50-100 ms ← infeasible
```

You cannot do a strongly-consistent cross-region read inside the ad-serving hot path. This is not a tradeoff to relax later; it is a physical constraint. **Strong consistency is off the table for cap reads.** The architecture has to be eventually-consistent regional with async replication, full stop.

**[STAFF SIGNAL: over-cap-vs-under-cap policy]** Perfect accuracy is impossible at this latency budget. I commit to a **per-cap-class policy**:

| Cap class | Lean | Why |
|---|---|---|
| Brand reach campaigns (e.g., 3/day) | over-deliver | Advertiser cares about reach distribution; over-cap is fine on their dime |
| Performance/DR | exact-as-possible | Every impression has CPM cost; over-delivery wastes budget |
| Regulatory/policy caps (alcohol, gambling) | under-deliver | Compliance dominates; we'd rather miss revenue than serve a regulated ad to a flagged user once over |
| Recency caps (creative rotation) | over-deliver | The user-experience cost of seeing the same creative twice in 5 min is small |

This is not a footnote. It is the central product decision and it dictates which part of the system gets engineered for accuracy and which doesn't.

**[STAFF SIGNAL: identity as graph]** A "user" is a cluster of IDs. A cap is enforced over the cluster, not a single key. More in §7.3.

---

## 3. Capacity and budget math

```
Ad requests/sec (global peak) ............ 10,000,000
Avg cap rules per request ................ 6      (ad, creative, line-item, campaign, advertiser, household)
Counter lookups/sec ...................... 60,000,000
Edge cache hit rate (target) ............. 95%
Regional store reads/sec ................. 3,000,000
Per region (5 regions, ~uniform) ......... 600,000

Daily impression volume .................. ~250B (peak-adjusted)
Streaming bus events/sec ................. 10M (1:1 with ad serves) + viewability events ~3M

Active users (90d) ....................... 1,000,000,000
Active ads ............................... 100,000
Avg non-zero (user, ad) pairs over 7d .... ~10  (long-tail; most pairs are zero)
Hot counters per region .................. 10B
Bytes per counter ........................ ~50  (key + count + window_start + ttl_meta)
Hot regional storage ..................... 500 GB per scope tier
Total across all scope tiers ............. ~2-3 TB per region
Plus user-level identity index ........... ~200 GB

Dedup window (24h) keys .................. 10M/s × 86400 = 864B impression_ids
At 16 bytes/id ........................... 13.8 TB/day exact dedup → infeasible
With Bloom filter pre-filter (~1% FPR) ... ~1.4 TB/day → tractable
```

These numbers force the architecture: edge caching is mandatory (a 1% drop in cache hit rate is 600K extra reads/sec/region); exact dedup over 24h is not affordable so probabilistic dedup as fast-path is mandatory; cross-region quorum is infeasible so we replicate counters async.

---

## 4. High-level architecture

```
                                  CLIENT / PUBLISHER
                                         │
                        ┌────────────────┴─────────────────┐
                        │ Ad request (OpenRTB / first-party)│
                        └────────────────┬─────────────────┘
                                         │ <100 ms
                                         ▼
   ┌─────────────────────────────────────────────────────────────────────┐
   │                       AD SERVER (regional pool)                     │
   │                                                                     │
   │   ┌─────────────┐    ┌───────────────────┐    ┌─────────────────┐   │
   │   │ Identity    │    │ Cap-rule engine    │    │ Auction / rank  │   │
   │   │ resolver    │───▶│ (parallel checks)  │───▶│                 │   │
   │   └─────────────┘    └─────────┬─────────┘    └─────────────────┘   │
   │                                │                                    │
   │   ┌────────────────────────────▼───────────────────────────┐        │
   │   │  Local edge cache  (per ad-server process)             │        │
   │   │  (user_id, scope) → count, window_start, fetched_at    │        │
   │   │  TTL: 5s for hot scopes, 60s for cold                  │        │
   │   └────────────────────────────┬───────────────────────────┘        │
   └────────────────────────────────┼────────────────────────────────────┘
                                    │ cache miss (~5% of reads)
                                    ▼
                ┌───────────────────────────────────────┐
                │  REGIONAL COUNTER STORE (Aerospike)   │
                │  Sharded by user_id; sub-ms p99       │
                │  Hybrid memory (RAM index + NVMe data)│
                │  TTL on every record                  │
                └───────────────────┬───────────────────┘
                                    ▲ async writes from
                                    │ stream consumer
                                    │
   ┌─── IMPRESSION FIRES ───┐       │      ┌─── XDR replication ─────┐
   │ (or viewability fires) │       │      │ async, ~100ms-1s lag    │
   └────────────┬───────────┘       │      └──────────────┬──────────┘
                │                   │                     │
                ▼                   │                     ▼
   ┌────────────────────────┐       │      ┌──────────────────────────┐
   │ Impression beacon /    │       │      │  Other regions' counter  │
   │ viewability beacon     │       │      │  stores (active-active)  │
   └────────────┬───────────┘       │      └──────────────────────────┘
                │                   │
                ▼                   │
   ┌────────────────────────┐       │
   │ Kafka / Pulsar         │       │
   │ (impressions topic)    │       │
   └────────────┬───────────┘       │
                │                   │
                ▼                   │
   ┌────────────────────────┐       │
   │ Flink stream consumer  │───────┘
   │  - dedup (Bloom→exact) │
   │  - viewability filter  │
   │  - hierarchy fan-out   │
   └────────────┬───────────┘
                │
                ▼
   ┌────────────────────────┐         ┌──────────────────────────┐
   │ Data lake (S3/HDFS)    │         │  Analytics / billing     │
   │ raw event archive      │────────▶│  reach reporting         │
   └────────────────────────┘         └──────────────────────────┘
```

The **read path** (top) is the hot path inside the latency budget. The **write path** (bottom) is asynchronous, eventually-updates-counters, and is allowed to lag by 1–5 seconds end-to-end. **Multi-region replication** (right side) is also async; same eventual-consistency property.

---

## 5. The read path: cap check in the hot path

```
T+0 ms     Ad request arrives at regional ad server
T+0.5 ms   Identity resolver: look up identity cluster for incoming IDs
           (cookie + UA fingerprint + UID2 → cluster_id)
T+1 ms     Cap-rule engine identifies all applicable cap rules for the
           candidate ad set: e.g., 5 rules for ad_99 → 5 (cluster, scope) keys
T+1.5 ms   Issue 5 parallel reads to edge cache
              cache hit  (95%): all keys back in < 1 ms
              cache miss ( 5%): fall through to regional store
T+3-6 ms   Regional store responds, edge cache populated for next request
T+6-10 ms  Cap-rule engine evaluates each predicate:
              count < limit AND window_start within [now - window, now]
T+10 ms    Hand back (allow / block list of ad_ids) to auction
```

**Key data structure**:

```
Key:   (cluster_id_hash[64bit], scope_type[1byte], scope_id[8byte], window_bucket[4byte])
Value: { count: uint32, window_start: uint64, last_imp_ts: uint64, ttl }
```

The window_bucket lets us avoid sweeping. A daily cap is keyed by `floor(now / 86400)`; a weekly cap by `floor(now / 604800)`. Old buckets expire by TTL. **No active reset job is needed.**

**Parallel rule evaluation**: For an ad request that touches 6 cap rules, we issue 6 reads in parallel and AND the results. We do not short-circuit on the first failure — we want the failure reason for ad-rotation logic, and the parallel reads dominate sequential ones.

**[STAFF SIGNAL: rejected alternative]** I considered storing all of a user's cap state under a single user-level key (one read instead of N). Rejected: hot-user keys (bots, power users) become hot shards; the value blob grows unbounded; updating a single field requires a CAS or read-modify-write at the storage layer, which is more expensive than N independent atomic increments. The N-key model parallelizes on both reads (network) and writes (shard locality).

**Edge cache discipline**:
- **Bounded staleness**: 5s TTL on hot scopes, 60s on cold. The 5s acceptance is a deliberate **over-delivery budget** — at 10M QPS, 5s of staleness means up to ~50K impressions can over-cap globally before the cache catches up. For a brand campaign at 100M total impressions, that's a 0.05% over-delivery, which is well within tolerance.
- **Stochastic early refresh** to avoid stampede: when a cache entry is at 80% of TTL, 1% of accessing requests trigger a background refresh. Eliminates the herd that would otherwise hit the regional store when a popular user's entry expires.
- **Negative caching**: if the regional store responds "no counter exists" (sparsity case), cache that for 5s too, with a smaller value. Roughly half the ad requests are for users who haven't seen any of the candidate ads.

---

## 6. The write path: counting impressions correctly

```
Ad served ──▶ client renders ──▶ measurement script polls every 200ms
                                       │
                                       ▼
                              50% pixels in view for 1s? ─── No: drop
                                       │ Yes
                                       ▼
                       Beacon: POST /impression  with impression_id
                                       │                (uuid v4, generated server-side)
                                       ▼
                            ┌─────────────────────┐
                            │ Beacon receiver     │
                            │ (lightweight HTTP)  │
                            └──────────┬──────────┘
                                       │
                                       ▼
                            ┌─────────────────────┐
                            │ Kafka topic:        │
                            │ impressions.v1      │ partitions = 1024
                            │  partitioned by     │ retention = 7 days
                            │  cluster_id         │ at-least-once
                            └──────────┬──────────┘
                                       │
                                       ▼
                            ┌─────────────────────┐
                            │ Flink stream job    │
                            │  1. dedup           │   (24h sliding window)
                            │  2. fan out by      │
                            │     scope hierarchy │   (ad → campaign → advertiser)
                            │  3. atomic INCR     │   to Aerospike counter
                            └──────────┬──────────┘
                                       │
                                       ▼
                            Counter updated. New regional reads see updated count
                            within edge-cache TTL (≤5s).
```

**[STAFF SIGNAL: viewability-vs-delivery distinction]** The beacon fires only on viewable impressions per MRC (50% pixels / 1s display, 2s video). Roughly 60-70% of delivered impressions become viewable. **For brand cap classes we count viewable; for DR/performance we count delivered**. Both are first-class — Flink writes to *both* counter sets and the cap rule references one.

**[STAFF SIGNAL: idempotency and dedup discipline]** Each beacon carries an `impression_id` minted at ad-serve time, included in the rendered creative. Three failure modes to handle:

1. **Pixel retries** at the client → beacon arrives 2–5×. Dedup at Flink: a 24h sliding-window seen-set on `impression_id`.
2. **Kafka at-least-once delivery** → same event delivered to two consumers. Dedup again. Flink's exactly-once sink semantics help but I do not rely on them across a 24h window.
3. **Server-side replay** during incident recovery → same beacons replayed from S3 archive into Kafka. Same dedup catches it.

Sizing the dedup state: 10M events/sec × 86400s = 864B `impression_id`s/day. Exact storage at 16B = 13.8TB. Two-stage approach: **Bloom filter** sized for 1% FPR (~10TB at 1% gives ~10 bits/key → ~1TB) catches 99% of duplicates instantly; on a Bloom hit, consult an exact RocksDB-backed dedup store (only ~1% of events reach it). Keeps memory bounded and gives **exact dedup** at acceptable cost. Late-arriving events (>24h after the original) are dropped — see §7.5.

---

## 7. Deep dives

### 7.1 Counter storage at scale

**Substrate: Aerospike.** Hybrid memory architecture (primary index in RAM, data on NVMe), sub-ms p99 reads, atomic single-record `INCR`, native record TTL — exactly the primitives this workload needs. Redis Cluster works at smaller scale but the all-in-RAM cost is 5–10× at our scale for the same data. ScyllaDB is competitive on cost but the LSM compactions create p99 jitter that hurts the ad-serving budget. **Rejected alternatives**: DynamoDB (hot-partition throttling on hot users; cost-prohibitive at 60M lookups/sec); Cassandra (read amplification + compaction storms); Redis-only (memory cost).

**Sharding**: hash `cluster_id` as primary key. All of a user's counters land on one node — co-location helps both read locality and write batching. Hot-cluster mitigation: detect by counter-update rate, split into virtual sub-keys for the top 0.01% of users; the cap engine has to fan out reads but the storage stays balanced.

**TTL management**: every counter carries a TTL equal to `window_length + buffer`. A daily cap = 24h + 1h buffer; weekly = 7d + 1d. Aerospike handles expiration in the background — no sweeps. **This is the single biggest reason to pick Aerospike over Redis** at this scale: Redis TTL eviction creates p99 hiccups during eviction storms; Aerospike's expiration is amortized.

**Sparsity**: ~10B non-zero (user, ad) entries against an upper bound of 1B × 100K = 10^14 — 99.99% sparse. We **never pre-populate**. First impression creates the counter via atomic `add(key, 1, ttl=window)`; reads of nonexistent counters return zero immediately.

**Capacity per region**: ~2–3 TB hot data on NVMe. Comfortably 8–16 nodes per region with i4i or comparable instances.

### 7.2 Multi-region eventual consistency

```
                       ┌──────────────────────────────┐
                       │  Global impression bus       │
                       │  (cross-region replicated    │
                       │   Kafka via MirrorMaker 2    │
                       │   or Pulsar geo-replication) │
                       └──────────────┬───────────────┘
                                      │
              ┌───────────────────────┼───────────────────────┐
              │                       │                       │
              ▼                       ▼                       ▼
       ┌──────────────┐        ┌──────────────┐        ┌──────────────┐
       │  US-EAST     │ ◀────▶ │  EU-WEST     │ ◀────▶ │  AP-SE       │
       │  Counter     │  XDR   │  Counter     │  XDR   │  Counter     │
       │  store       │        │  store       │        │  store       │
       └──────────────┘        └──────────────┘        └──────────────┘
                          replication lag p50: ~150 ms
                          replication lag p99: ~1-2 s
```

**Replication topology**: each region is the authority for its own writes. Two complementary mechanisms:

1. **Aerospike XDR** for direct cross-region counter sync (Last-Write-Wins per record). Simple, reliable for non-counter data.
2. **Replicated Kafka topic** for impression *events*. Each region's stream consumer applies remote events to its local counter store. This is what actually carries cross-region count convergence.

**Why both**: XDR alone uses LWW, which is wrong for counters (last write loses the previous count). Replicating *events* and re-applying `INCR` is associative and commutative — operation-based CRDT semantics, naturally convergent. We use XDR for identity-graph sync and metadata; we use event replication for counters.

**Convergence model**: each region's counter is effectively a G-Counter (grow-only) per impression-source-region. Total count = sum across regions. We don't store this explicitly — we just rely on each region applying every other region's events.

**[STAFF SIGNAL: eventual-consistency honesty]** The user-mobility case: user has 3 impressions in EU (cap = 3, capped). They VPN to US instantly. US region hasn't yet received the 3 EU events; US count = 0 or 1. They see one more impression. Final global count = 4, cap was 3. Over-delivery by 1.

Quantifying: at p99 replication lag of 1–2s, and assuming 0.1% of users do meaningful cross-region activity within that window, the over-delivery rate is roughly `0.1% × (avg impressions a user gets in 2s if they're being aggressively exposed)` ≈ 0.001 × 0.5 ≈ **0.05% over-delivery from mobility**. Acceptable for almost all cap classes. For the strictest brand caps where this isn't acceptable, we route the ad request through the user's *home* region (sticky routing by cluster_id) and only fall back to the local region on home-region unavailability — this collapses the mobility case at the cost of some additional latency for VPN users.

**During partition** (region cannot replicate): counters in each partition are local-only. Each region under-counts the global. Caps are looser. We accept this — we publish a partition-detection metric and a "we are over-delivering" SLO breach goes to the on-call.

### 7.3 Identity resolution

This is the single biggest 2026 difference from a 2018 ad-tech answer.

```
                Inbound identity signals
                         │
   ┌──────────┬──────────┼──────────┬──────────┬──────────┐
   ▼          ▼          ▼          ▼          ▼          ▼
 cookie    UID2/      IDFA      GAID    IP+UA fingerprint  household IP
 (web)     hashed-    (iOS,     (Android)   (low precision,  (CTV)
           email      consent)              probabilistic)
           (logged-
            in)
                         │
                         ▼
              ┌─────────────────────┐
              │ Identity Graph      │
              │  Service            │
              │  (clusters of IDs   │
              │   = persons or HH)  │
              │  cluster_id ◀──┐    │
              └─────────┬──────┴────┘
                        │
                        ▼
              All capping operates over cluster_id
              (or household_id for CTV)
```

**Cluster**: each user is a cluster of IDs the system has merged via deterministic links (logged-in events, hashed-email matches across surfaces) and probabilistic links (device graph, IP+UA over time). The graph is owned by an offline batch + online incremental service. **Capping operates on cluster_id**, not on the raw input IDs. When the inbound request has multiple IDs, we resolve all of them, take the union of clusters they belong to, and cap on each — usually they all map to the same cluster.

**Cluster mutation**: when the graph merges two clusters into one (or splits one), counters need to be merged too. We do this lazily: on next read, if the cluster was recently merged, the cap engine reads counters under both old cluster IDs and the new one, sums them, and writes the consolidated counter under the new ID. The stale entries TTL-out. This is **eventual consistency in the identity layer** — accept it.

**The "no ID" case**: for unauthenticated CTV traffic where we have only `household_IP + device_make`, we cap at household level. Household caps are necessarily looser (multiple people share an IP) but that's the right granularity — that's who's seeing the screen. **This is fundamentally different identity infrastructure** from per-user web capping, and I'd push back on a prompt that conflates them.

**The Privacy-Sandbox-was-supposed-to-handle-this case**: in 2024 the architectural plan was to delegate caps to Protected Audience on-device. That API was deprecated in October 2025. So the on-device pattern doesn't have a privacy-preserving home in Chrome anymore. It does still exist on iOS via SKAdNetwork campaign-level frequency caps, and within walled-garden SDKs (Meta Audience Network, Google AdMob) which manage their own on-device cap state. **We treat on-device as a fallback for surfaces where we genuinely have no server-side identity**, not as the primary architecture.

### 7.4 Cap-rule scoping and hierarchy

Single ad request frequently triggers 5–10 cap checks across a hierarchy:

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

Plus orthogonal scopes:
```
   household_id   cap: 8 / day  (CTV)
   vertical_id   cap: regulated content limits
```

**Hierarchical fan-out — write-amp vs read-amp**. Two designs:

- **Write-side fan-out**: on each impression, increment all 6 hierarchy counters atomically. Read path does 6 cheap reads.
- **Read-side aggregation**: store only leaf-level counters; on cap check, sum children for parent scope. Write path does 1 increment.

At 10M imp/sec with 6 levels, write-fan-out costs 60M writes/sec, distributed across shards — not bad but more than read-fan-out's 10M. But read-fan-out costs ~60M reads/sec across the hierarchy on every check (assuming N children per parent), which dominates cache misses much more.

**Decision**: write-fan-out for hierarchy. Reads are far hotter than writes (we check ~6× per request, we increment 6× per impression but only ~1/3 of requests result in an impression). Write fan-out is implemented in Flink — single stream consumer reads one impression event, emits 6 atomic INCRs to the counter store. **[STAFF SIGNAL: hierarchical cap design]**

**Frequency vs recency**: frequency caps use `count`. Recency caps use `last_imp_ts`. Both are stored in the same record:
```
record = { count: u32, window_start: u64, last_imp_ts: u64, ttl }
```
Recency check is `now - last_imp_ts > min_gap`. Trivial; no new infrastructure.

### 7.5 Delayed, duplicated, out-of-order events

```
Event arrives ─────────────────────────┐
                                       ▼
                              ┌────────────────┐
                              │ Bloom filter   │ Bloom hit?
                              │ (24h sliding)  │ ──No──▶ accept, increment
                              └────────┬───────┘
                                       │ Yes (~1% FPR)
                                       ▼
                              ┌────────────────┐ Found?
                              │ Exact dedup    │ ──Yes──▶ drop (duplicate)
                              │ store (RocksDB)│
                              └────────┬───────┘ ──No──▶ accept
                                       │
                              ┌────────▼─────────┐
                              │ Watermark check  │ event_time < now-1h?
                              │                  │ ──Yes──▶ drop (late, log)
                              └────────┬─────────┘
                                       │
                                       ▼
                                accept, atomic INCR
```

- **Duplicates**: handled by Bloom + exact dedup, sized in §3.
- **Out-of-order**: `INCR` is commutative; order doesn't matter for frequency. For recency (`last_imp_ts`), update only if event_ts > stored last_imp_ts. Atomic compare-and-swap on the recency field.
- **Late events** (>1h late): drop. Apologies in advance, advertiser. Reason: applying a 6-hour-late impression to today's count when today's window has already closed creates inconsistencies in reporting that aren't worth the recovery. The dropped fraction is monitored as an SLO; if it exceeds 0.5%, on-call investigates pipeline lag.

### 7.6 Probabilistic structures — when they earn their place

Two clear places:

**1. Bloom filter as fast-path on dedup** (already used in §6).

**2. "Has this user seen any ad from advertiser X in the last hour?" type queries.** A Counting Bloom Filter or per-(cluster, advertiser, hour) Bloom keyed on ad_id can answer the "is the user's hourly advertiser frequency at the cap" question without storing millions of per-(cluster, advertiser) integer counters. The advertiser-level cap is rare-path; not worth its own integer counter for every cluster. The CBF saves ~10× memory at 1% FPR.

**Count-Min Sketch** for frequency estimation: I considered using CMS as the primary counter store. Rejected. CMS is one-sided overestimating — which means we'd over-cap (under-deliver), which is the wrong direction for performance campaigns and only acceptable for the most paranoid brand caps. Also, exact integer counters at our scale are *affordable* in Aerospike; the memory savings of CMS don't justify the loss of accuracy.

**HyperLogLog**: useful for *reach reporting* (how many distinct users saw this ad), not for capping (we need to know if *this* user has seen it). Out of scope here, lives in the analytics path.

**[STAFF SIGNAL: probabilistic-when-appropriate]** Probabilistic structures are fast-path optimizations and answer questions exact counters can't afford. They are not a substitute for the primary counter store at this scale.

---

## 8. Failure modes and graceful degradation

| Failure | Detection | Response | Policy |
|---|---|---|---|
| Counter store unavailable (regional) | Health check + read-error rate | **Fail open** for ads with cap rules; mark "cap state unknown" in audit log | Brand campaigns may opt fail-closed via campaign config — accept the lost revenue to avoid a brand-safety incident |
| Streaming pipeline lag > 30s | Kafka consumer lag SLO | Caps are stale; over-delivery happens. Alert; throttle ad serving on affected accounts if lag > 5min | This is a systemic SLO violation, not an ad-server-time decision |
| Edge cache stampede on hot cluster | Spike in regional store reads for one shard | Stochastic early refresh (already in design) + per-cluster request coalescing at the ad server | Built-in; no human action |
| Identity-graph false-merge (two users now one cluster) | Anomaly detection on counter velocity for a cluster | Halt new merges from the affected lineage; manual unmerge; counters are wrong until rebuilt from event log | Last-resort recovery from S3 event archive |
| Dedup state corrupted/lost | Hash mismatch on dedup checkpoint | Rebuild dedup state from Kafka over last 24h (~13TB replay, ~30 min) | During rebuild, accept that some duplicates leak through; over-delivery is bounded by the corruption window |
| Single ad-server replica fail | LB removes from pool | Other replicas absorb load; edge cache cold-starts but warms in <60s | Routine |

**[STAFF SIGNAL: failure mode precision]** The single most important design decision in the failure column is **fail-open by default**. Rationale: ad serving going dark causes acute revenue loss and bad UX (unfilled slots, blank space). Cap violations are bounded in magnitude (you can't violate a 3/day cap by 1000×). Fail-open is the correct policy unless campaign config overrides it.

**[STAFF SIGNAL: blast radius reasoning]** A bug in the identity graph causing global mass-merge of clusters would corrupt every cap on the platform. Mitigation: graph mutations are rate-limited globally, every change is auditable, and the graph version is part of the counter key — a graph-version bump invalidates affected counters lazily rather than instantly, isolating blast radius to *new* requests under the new graph version.

---

## 9. Data retention and operational tradeoffs

**Counter store**: TTL = window_length + small buffer. Daily caps live ~25h, weekly ~8d, monthly ~31d. Counters older than that are gone — they have no operational value.

**Event archive**: every impression event is durable in S3/HDFS for 90 days for billing/audit/reach reporting. Kafka holds 7 days of hot replay capacity.

**[STAFF SIGNAL: cost-vs-accuracy explicit]** The accuracy-vs-cost dial:

| Operating point | Edge TTL | XDR mode | p99 over-delivery | Infra cost |
|---|---|---|---|---|
| Loose (default DR) | 30 s | async | ~0.5% | 1.0× baseline |
| Standard (most brand) | 5 s | async | ~0.05% | 1.5× |
| Tight (top-tier brand) | 1 s | async + home-region routing | ~0.01% | 2.5× |
| Strict (regulated) | 0 s (no edge cache) | sync per write | ~0.001% | 5×+ |

The platform exposes this as a per-cap-class config. **Strict mode is reserved for regulated verticals** (alcohol, gambling, political ads in some geos) where the legal risk of one over-cap dominates the cost. Most brand campaigns sit in Standard. DR campaigns sit in Loose.

---

## 10. Privacy and the post-cookie reality (revised for 2026)

The 2022-vintage answer here would have been: "Privacy Sandbox is the future, on-device caps via Protected Audience, design for migration." That answer is now wrong. October 2025 deprecation killed Protected Audience, Topics, Attribution Reporting, and the rest. Chrome kept third-party cookies. The post-cookie world arrived for Safari and Firefox in 2020 and didn't arrive for Chrome.

The **actual** 2026 privacy posture for capping is:

- **Chrome web**: third-party cookies still work; cap server-side keyed on cookie-derived cluster_id.
- **Safari/Firefox web** (~30% traffic): no cookies. Fall back to UID2/hashed-email when logged in; otherwise IP+UA fingerprint at degraded precision; otherwise cap per-publisher only.
- **iOS apps**: IDFA only with ATT consent (~25–40% opt-in). For the rest, cap per-IDFV (vendor-scoped) within the publisher's own apps; SKAdNetwork covers cross-app campaign-level cap signals at the campaign granularity.
- **Android apps**: GAID still available with consent (Android privacy posture stayed roughly where it was; Privacy Sandbox on Android was also largely retired).
- **CTV**: household-IP capping. No per-individual capping exists meaningfully here, and pretending otherwise is engineering theater.
- **Logged-in O&O surfaces**: the easy case. Stable user_id, exact capping, this is where the system performs best.

**[STAFF SIGNAL: privacy-aware design]** The architectural implication: the cap engine must be **identity-source-agnostic at the read path** — it caps over `cluster_id`, and the identity layer is responsible for resolving inputs to clusters with appropriate confidence. Confidence below a threshold → fall back to coarser scope (IP, household, or per-publisher). The cap-rule engine accepts a `scope_confidence` signal and may relax or tighten enforcement accordingly.

**[STAFF SIGNAL: 2026 cutting-edge awareness]** The thing to be honest about in an interview today: the industry hasn't solved cross-publisher unauthenticated capping. UID2 is the closest thing to a standard, and adoption is partial. We design for it, we accept where it doesn't reach, and we don't pretend Privacy Sandbox is going to fix it.

---

## 11. Tradeoffs and what would change the design

| Lever | If pulled | Redesign required |
|---|---|---|
| QPS goes from 10M to 100M | 10× counter-store + edge cache; identity service becomes the bottleneck | Yes; revisit identity service capacity |
| Strict global consistency required | Cross-region quorum or routing all reqs to home region | Yes; the latency budget is the central constraint |
| All caps must be exact (no over-delivery) | Cannot; physical impossibility within tmax | Push back |
| Pure on-device capping (Privacy-Sandbox-style) | Move cap state to client SDK; server gets aggregated reach signals only | Yes; architecturally different system |
| CTV-only platform | Drop user-level identity entirely; household + content rules dominate | Yes; identity layer is much simpler, but content/policy layer is much richer |
| Move from O&O+DSP to pure SSP | We never see impressions; we'd need bidder-reported delivery signals | Yes; capping moves to bidders, we coordinate |

---

## 12. What I'd push back on in the prompt

**[STAFF SIGNAL: saying no]**

1. **"Eventual consistency across regions"** — fine as a goal, but the prompt frames it as a spectrum. It's not. Strong consistency at this latency budget is *physically impossible*. The choice is: *how loose* is the eventual consistency, and *which cap classes get the tighter mode*. I want the conversation to be about per-cap-class operating points, not a single global tolerance.

2. **"Multiple cap scopes such as user-ad, user-campaign, household."** Treated as parallel concerns. They're not. **Household capping is fundamentally different identity infrastructure** from user-level capping — different signals (IP-based), different policy (multiple individuals share the cap), different fairness considerations. Most platforms ship them in different release trains. I'd separate the PRD.

3. **"Privacy-driven changes"** — the prompt's framing assumes the 2022 trajectory: cookies are dying, plan for the post-cookie world. The 2026 reality is messier: cookies survived in Chrome, Privacy Sandbox died, identity is permanently fragmented across surfaces. The right frame is "design for permanent identity fragmentation," not "design for the cookie-pocalypse."

4. **"At most 3 impressions per user per ad per day"** — ostensibly a hard rule. In practice, "user" is a fuzzy cluster, "day" is a window with TZ-policy ambiguity (UTC? user-local?), "impression" is delivered or viewable, and "3" is a soft target with ~0.05% expected over-delivery. The product spec should be written with these realities in mind or it sets up the engineering team to fail.

5. **"What happens when counting data is delayed, duplicated, or arrives out of order"** — listed as one bullet. These are three different problems with three different solutions (watermarking, dedup, commutativity respectively), and conflating them produces muddy designs. I separated them in §7.5.

If I had to summarize the design in one paragraph: **counters keyed by `(identity_cluster, scope, window_bucket)`, sharded by cluster_id, served by Aerospike with sub-ms reads behind a 5-second-TTL edge cache, written via Kafka-Flink with Bloom-then-exact dedup and viewability filtering, replicated cross-region asynchronously via event log, with a per-cap-class accuracy/cost dial that the product chooses between Loose/Standard/Tight/Strict.** Everything else is detail in service of that.