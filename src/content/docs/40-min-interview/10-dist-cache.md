---
title: "9-Distributed Cache"
description: Distributed Cache
---
"Design a distributed cache serving 10M ops/sec with sub-ms latency. Eviction, replication, hot-key mitigation, consistency model."

---

## 1. Reframing

A cache is not a performance optimization. A cache is a **consistency contract** — a promise about how stale the data it returns may be relative to the source of truth — that we are willing to honor in exchange for serving reads from RAM instead of from a disk-backed system. Every other decision (eviction, replication, sharding, invalidation) is a downstream consequence. **[STAFF SIGNAL: consistency-model-first]**

I am committing up front to the contract this cache will honor:

> **Bounded staleness, default 30s, per-key.** A read for key `K` returns a value that was the committed-DB value for `K` at some point within the last 30 seconds, *or* the most recent value if invalidation has propagated faster. A small, application-tagged class of keys (`strong:*` namespace — account balances, idempotency tokens) bypasses the cache for writes and uses a synchronous write-through path with version fencing. Read-your-writes is provided **opt-in** via a per-request `min_version` header.

This is not "eventual consistency" hand-waving. The 30s bound is enforced by two independent mechanisms — TTL ceiling and CDC-driven invalidation with a deadline alarm — so a single mechanism failing does not silently extend staleness.

Two problems will dominate the rest of the design, treated as primary rather than as footnotes:

1. **Hot keys.** With Zipfian load, the top 0.1% of keys carry ~50% of traffic. Naive single-shard placement of the hottest key receives ~500K ops/sec — no single Redis-class instance survives that. Hot-key handling is architectural, not operational. **[STAFF SIGNAL: hot-key as central]**
2. **The cache is load-bearing.** At 10M ops/sec, the DB cannot serve 10M ops/sec — probably not 1M, not 500K. If the cache disappears, the DB melts and takes the product down. The cache is part of the durability story for *availability*, even though it has no durability story for *data*. Designs that fall through to the DB on miss must include a budget. **[STAFF SIGNAL: cache-as-load-bearing]**

## 2. Scoping

**[STAFF SIGNAL: scope negotiation]** Committed assumptions:

- **Workload mix:** 95% reads / 5% writes. 10M ops/sec aggregate ⇒ 9.5M reads/sec, 500K writes/sec.
- **Access pattern:** Zipfian with α ≈ 1.0. Top 0.1% of keys ≈ 50% of read traffic. This is the realistic case (social feeds, product catalogs, profile lookups). Uniform-distribution designs are wrong here.
- **Value size:** median 1KB, p99 8KB, hard cap 64KB. Larger blobs go to a separate object store with a cache-of-pointers pattern.
- **Working set:** ~1B keys hot enough to live in cache. At 1KB/value + ~200B overhead per entry ⇒ ~1.2TB logical, ~3.6TB with replication factor 3.
- **Side-cache, not primary store.** A relational + columnar DB is the source of truth. The cache holds no data the DB does not have. We can lose the entire cache and not lose data — we will lose availability for ~minutes while it warms.
- **Regional, not global.** One cache fleet per region (US-East, US-West, EU). Cross-region invalidation is in scope as an eventual-consistency mechanism with seconds-of-RPO; cross-region reads are out of scope. Sub-ms p99 across continents is physically impossible (TCP RTT US↔EU ≈ 70ms minimum).
- **Durability of writes:** writes are durable in the DB *before* cache acknowledgment for the write-through path; writes to cache-aside paths are durable as soon as the DB write commits — cache failure cannot lose writes.

Anything not scoped above (e.g. transactional cross-key reads, multi-key atomic writes) the cache does not provide. The DB does.

## 3. Capacity and latency math

**[STAFF SIGNAL: capacity math]**

```
Aggregate read bandwidth:   9.5M ops/s × 1KB     =  9.5 GB/s
Aggregate write bandwidth:  500K ops/s × 1KB × 3 =  1.5 GB/s  (replication factor 3)
Total wire bandwidth:                            ≈ 11  GB/s

Single 100GbE NIC saturates at ~12.5 GB/s ⇒ multi-server FORCED.
```

Per-node sizing (modern Redis-class: shard-threaded engine, e.g. KeyDB / Redis-7 IO threads / Dragonfly):

| Resource           | Per node            | Per fleet (40 nodes) |
| ------------------ | ------------------- | -------------------- |
| Sustained ops/sec  | 350K (mixed RW)     | 14M (40% headroom)   |
| Memory             | 96 GB usable        | 3.84 TB              |
| NIC                | 25 GbE × 2 bonded   | 100 GB/s aggregate   |
| CPU                | 32 vCPU             | —                    |

40 nodes × 3 replicas-per-shard logical = ~14 shards × 3 replicas, plus 2-node hot-key spillover capacity. Cost (RAM-only, AWS r7g pricing ≈ $4/GB/month for memory-optimized): **~$15K/month RAM, ~$60K/month all-in incl. compute and bandwidth.** **[STAFF SIGNAL: cost-explicit]**

**Latency budget for sub-ms p99** **[STAFF SIGNAL: latency-budget decomposition]**:

```
Component                           Budget (μs)   Mechanism
─────────────────────────────────────────────────────────────────────
Client serialization                       30     Pre-allocated buffers, zero-copy
TCP send + intra-rack RTT                 180     Same-rack placement, persistent conns
Server-side parse + lookup                 50     Hash table, no GC pause
Server serialize + send                    40     RESP3 wire protocol
Client deserialize                         30     Zero-copy on recv path
                                       ─────
Median path                               330
Tail buffer (GC, scheduler, queue)        500
                                       ─────
p99 budget                                830 μs   ✓ under 1 ms
```

**Tail-latency mitigations** (separate from median):
- Hedged reads to a second replica at the p95 latency mark (~400μs); take first response. Costs +5% bandwidth, cuts p99.9 by ~2x.
- Same-rack placement for clients ↔ cache via topology-aware shard map. Cross-rack adds 200μs and blows the budget at p99.
- No TLS on the cache-internal hop (the network is private); TLS adds 50–100μs handshake amortized + per-record overhead.

Cross-rack p99 ≥ 1.2ms. Cross-AZ p99 ≥ 2ms. **The sub-ms p99 promise is a same-rack promise; I will state this explicitly in the SLA.** **[STAFF SIGNAL: saying no]**

## 4. High-level architecture

```
                ┌──────────────────────────────────────────────┐
                │            Application tier (1000s)           │
                │  ┌──────────────────────────────────────────┐ │
                │  │  Cache client SDK                        │ │
                │  │  - Shard map (versioned, gossip-refresh) │ │
                │  │  - Single-flight coalescer (per-process) │ │
                │  │  - Hot-key micro-cache (LRU, 10K, 200ms) │ │
                │  │  - Hedged read scheduler                 │ │
                │  └──────────┬───────────────────────────────┘ │
                └─────────────┼────────────────────────────────┘
                              │ persistent conns, 50 per app→shard
              ┌───────────────┼───────────────────┐
              ▼               ▼                   ▼
    ┌──────────────┐  ┌──────────────┐    ┌──────────────┐
    │   Shard 0    │  │   Shard 1    │... │   Shard 13   │
    │  P  R1  R2   │  │  P  R1  R2   │    │  P  R1  R2   │
    └──────┬───────┘  └──────┬───────┘    └──────┬───────┘
           │                 │                   │
           └─────── async replication ───────────┘
                              │
              ┌───────────────┼───────────────┐
              ▼                               ▼
    ┌──────────────────┐              ┌──────────────────┐
    │  Hot-Key Tier    │              │  Invalidation    │
    │  (separate fleet,│              │  Bus (Kafka)     │
    │   keys replicated│              │  - DB CDC events │
    │   N=8 ways,      │◄─────────────┤  - Explicit inv. │
    │   read-only)     │              └──────────────────┘
    └──────────────────┘                       ▲
                                               │
                                       ┌───────┴───────┐
                                       │  Source of    │
                                       │  Truth (DB)   │
                                       └───────────────┘
```

The architecture has four tiers, not the typical two:

1. **Client SDK is a participant**, not a thin RPC stub. Hot-key micro-cache, single-flight, and hedged-read logic live here — work done before the network is free relative to the round-trip.
2. **Main cache fleet**: ~14 shards × 3 replicas. Holds the bulk of the working set under bounded-staleness contract.
3. **Hot-key tier**: separate, smaller fleet of read-only replicas of identified hot keys. Decouples hot-key load from main-fleet CPU. Justified in §7.
4. **Invalidation bus**: Kafka topic fed by DB CDC (Debezium-style) plus explicit application invalidations. Cache nodes tail the bus and apply invalidations. The deadline alarm here is the staleness-bound enforcer.

The DB is *behind* the cache, never the front door. Application code reaches the DB only via the cache's miss-fall-through path, which is *budgeted* (see §10).

## 5. Consistency model and invalidation pattern

**Committed pattern: cache-aside with CDC-driven invalidation, plus a write-through subset for the `strong:*` namespace.** **[STAFF SIGNAL: invalidation policy commitment]**

Rationale:
- **Cache-aside** for the 99% of keys: writes go to the DB regardless; the cache is purely an acceleration layer. The application must work without the cache (and does, during cold start), simplifying the failure model.
- **CDC-driven invalidation** (rejected: pure-TTL, write-through, write-back) gives bounded staleness by *mechanism*, not hope. TTL alone makes worst-case staleness = TTL; CDC drops it to replication-lag + propagation, typically <1s. **Rejected pure-TTL** because 30s is a poor UX for "I just changed my profile picture." **Rejected write-through-everywhere** because it puts the cache on the write critical path — a degraded cache becomes a write-availability problem. **Rejected write-back** because losing writes on cache failure violates the side-cache assumption.
- **Write-through for `strong:*`** because some keys (idempotency tokens, lock state, transactional balance reads) cannot tolerate even 1s of staleness. <1% of keys; we eat the write latency for them.

**Write path** (cache-aside with CDC):

```
Application                Cache (shard P)            DB              Kafka (CDC)        All other shards
    │                            │                     │                    │                     │
    │── PUT user:42 ─────────────────────────────────► │                    │                     │
    │                            │                     │── commit ─────────►│                     │
    │◄── 200 OK ─────────────────────────────────────  │                    │                     │
    │                            │                     │                    │                     │
    │── DELETE cache user:42 ───►│ (best-effort, sync) │                    │                     │
    │                            │── tombstone, v=t ──►│                    │                     │
    │                            │                     │                    │                     │
    │                            │                     │  CDC event ───────►│                     │
    │                            │                     │                    │── invalidate ──────►│
    │                            │                     │                    │                     │ (delete or
    │                            │                     │                    │                      version-fence)
```

Two invalidation paths run in parallel:
1. **Application-driven** (sync on hot path): writer issues best-effort `DELETE` after DB commit. May fail. **Failure here does not fail the write** — CDC is the safety net.
2. **CDC-driven** (async, authoritative): the DB's WAL is tailed; every committed mutation produces an invalidation event with a logical version number; every cache shard applies it. **This enforces the 30s bound.**

**The version fence** is what makes the dual-path safe. Every entry stores `(value, version)`. The CDC event carries the post-write version. Invalidation sets a tombstone-with-version: subsequent cache writes for `K` are accepted only if version ≥ tombstone. This prevents the classic race:

```
Bad ordering without version fence:
  T0  app reads K, miss, fetches v=1 from DB
  T1  app writes K=v2 to DB
  T2  app deletes K from cache
  T3  app from T0 finally writes (K, v=1) to cache  ← stale forever
```

With versioning, the T3 write is rejected because tombstone.version ≥ 2.

**Failure modes of this scheme:**

- **CDC lag spike (Kafka backed up):** staleness bound exceeded. *Mitigation:* per-shard CDC-lag alarm at 10s; if lag > 30s, the shard refuses reads (returns "cache unavailable") rather than serving silently-stale data. **[STAFF SIGNAL: invariant-based thinking]** — the invariant is *bounded staleness*, and we'd rather fail closed than violate it.
- **Synchronous DELETE succeeds, CDC fails permanently:** unlikely (Kafka is replicated), but the TTL ceiling (30s) is the ultimate backstop. Every key's TTL is hard-capped.
- **DB write commits, app crashes before sync DELETE *and* before CDC catches up:** TTL backstop ensures eventual convergence within 30s.

The contract a staff engineer commits to: *staleness is bounded by `max(CDC_lag, TTL_ceiling)`; if either mechanism degrades, the system fails closed on reads of likely-stale keys, not silently.*

## 6. Sharding and replication

**Sharding: rendezvous hashing (HRW) with explicit shard-map override for hot keys.** **[STAFF SIGNAL: rejected alternative]**

Rejected: **hash-mod-N** (catastrophic on resize). Rejected: **consistent hashing with virtual nodes** — fine, what most candidates propose. Rendezvous gives slightly better load distribution under heterogeneous server sizes, simpler reasoning about replica selection (HRW naturally yields a *ranked list* of N servers per key — exactly what we need for replica placement), and no ring data-structure to maintain. Tradeoff: O(N) hash computations vs O(log N); at N=14 this is irrelevant.

**Hot-key override:** the top-K hottest keys (~10K keys) live in an explicit shard map maintained by the control plane and pushed via gossip to clients. These keys are pinned to the hot-key tier (§7), bypassing HRW.

**Ring sketch (HRW logical view, replicas-per-key=3):**

```
    key="user:42" → hash with each of {S0..S13} → rank descending
                  → top-3: [S7, S2, S11]
                  → primary=S7, replicas=S2, S11

    Adding S14: ~1/15 of keys re-rank (their new top-3 includes S14).
    Removing S7: keys whose top-3 included S7 promote rank-4 to rank-3.
                 Data movement: ~1/14 × 1.2TB ≈ 86 GB.
                 At 25 Gbps backfill: ~30 seconds per departing shard.
```

**Replica topology: primary + 2 async replicas, all-replicas-serve-reads.**

- **Writes** go to primary, ack on primary commit, replicate async with bounded lag (target <50ms p99).
- **Reads** hit any of 3. SDK round-robins weighted by recent latency. Triples read throughput per shard; natural failover.
- **Read-your-writes:** opt-in via `min_version`. Replicas with `local_version < min_version` either wait briefly (≤5ms) or redirect to primary. Default reads do not pay this cost.

**Replication mechanism: async with quorum-on-demand.** Pure async risks losing un-replicated cache state on primary failure — but this is a side cache, so losing recent cache writes just costs a re-fetch from DB. Acceptable. For `strong:*` keys, replication is **sync to ≥2 of 3** before primary acks; ~200μs cost, error on quorum failure.

**[STAFF SIGNAL: rebalancing discipline]** Topology change handling: when a shard joins or leaves, the shard map version increments. Clients with stale shard maps may target the wrong shard; the wrong shard returns a `MOVED` redirect (Redis-Cluster-style) carrying the new map version. Data migration uses **warm handoff**: the new shard pre-populates its key-range from the displaced shard's primary (streaming, ~30s for 86GB at 25 Gbps); during this window both shards serve reads, the displaced one is authoritative for writes; cutover is atomic on shard-map version increment. **No request fails during rebalancing.** Hit-rate degrades modestly during the warm-up window because fresh keys arrive on the new shard with empty CDC backfill.

## 7. Hot-key mitigation

**[STAFF SIGNAL: hot-key as central]** This is the section that separates a working design from a failing one.

**The math, sharpened.** With 9.5M reads/sec and Zipfian α=1, the single hottest key receives ~9.5M / H_N where H_N is the harmonic number over the keyspace. For N=10⁹, H_N ≈ 21, so the hottest key receives ~450K reads/sec. The top 10 collectively serve ~25% ≈ 2.4M/sec; top 100 ≈ 35%; top 10K ≈ 60%.

A single Redis-class shard handles ~350K ops/sec sustained. **A single key can require more throughput than a single shard.** No amount of horizontal scaling helps if the hot key lives on one shard. The architectural answer must replicate hot keys across multiple shards — the question is how.

### Detection

```
        Cache shard
        ┌─────────────────────────────────────────┐
        │  Per-shard Count-Min Sketch (CMS)       │
        │  - 4 hash functions × 65K counters       │
        │  - Decay: halve all counters every 10s  │
        │  - Memory: ~1 MB                        │
        │                                         │
        │  Top-K Heavy Hitters (Space-Saving)     │
        │  - Tracks top-1000 keys per shard       │
        │  - Memory: ~50 KB                       │
        └────────────────┬────────────────────────┘
                         │
                         ▼
                  Hot-Key Controller
            (aggregates from all shards every 5s)
                         │
                         ▼
              Globally hot-key set (top-10K)
                         │
                         ▼
                  Pushed to clients via
                shard-map gossip channel
```

CMS gives ε-approximate counts cheaply; Space-Saving gives top-K; the aggregator fuses per-shard results into a global ranking. Promotion threshold: **5K ops/sec** (≈1.5% of single-shard capacity); demotion at 1K (hysteresis prevents flapping).

### Mitigation, in layers

The four mitigations stack — each catches the load the previous one missed.

**Layer 1: Client-side micro-cache.** Each app process keeps an LRU of ~10K entries with 200ms TTL for hot-flagged keys. A read for a hot key hits process memory in ~200ns — zero network. Trade: 200ms additional staleness on hot keys. Absorbs ~40% of hot-key reads before they reach the network. Memory: ~10MB per process; 10GB across 1000 processes — cheap.

**Layer 2: Single-flight coalescing.** Concurrent in-process reads for the same missing-or-stale key collapse to one upstream fetch via a per-key promise map. Caps any one key's upstream load at ~1 per app process per fetch latency. With 1000 processes × 1ms fetch, hot-key upstream is bounded at ~1M ops/sec — still too much, but a 10x reduction.

**Layer 3: Hot-key tier (separate fleet, replicate-on-detection).** Identified hot keys are replicated to **N=8 dedicated read-only nodes** holding *only* hot keys (~10K × ~1KB = 10MB, fits in CPU cache). Reads hash to a random one of 8, spreading 450K ops/sec to ~56K per node — within budget. Writes/invalidations fan out to all 8.

```
   Hot-key reads (post-detection)
                │
        ┌───────┼───────┬────────┬────────┐
        ▼       ▼       ▼        ▼        ▼
      H0      H1      H2  ...   H6       H7    (8 hot-key nodes)
        ▲       ▲       ▲        ▲        ▲
        └───────┴───┬───┴────────┴────────┘
                    │
              Invalidation bus
              (CDC + explicit)
```

The hot-key tier is **physically separate** — different machines, different failure domain. A hot-key tier outage does not melt the main fleet; clients fall back to the main fleet's primary copy with degraded capacity but warm cache (CDC keeps it current).

**Layer 4: Probabilistic early refresh.** Reads near expiry probabilistically refresh proactively (XFetch); decorrelates expiry-driven misses across clients. Eliminates the "10K clients all miss at second 60" pattern.

### Rejected alternatives

- **Pure client-side caching of all keys:** memory cost across N clients (N×working-set) explodes; staleness becomes per-client and unbounded.
- **Replicating the entire keyspace N-ways:** cost-prohibitive (3.6TB × additional N).
- **Sharding hot keys by suffix (e.g. `user:42:shard0..7`):** breaks API (clients must know which shard to read); pushes complexity to every consumer.
- **Just adding more shards:** does not help — the issue is per-key, not aggregate.

### Quantified outcome

```
Scenario: hottest key, 450K reads/sec.

  Without mitigation:        450K hits one shard. Shard CPU pegged at 130%. p99 explodes.
  With L1 (client micro):    270K reach network    (40% absorbed at client)
  With L2 (single-flight):   ~5K hit upstream from any one app process
                             Aggregated across 1000 apps: ~270K still possible at peak
                             (single-flight only collapses concurrent in-flight)
  With L3 (hot-key tier):    270K spread across 8 nodes = 34K/node. Comfortable.
  With L4 (early refresh):   eliminates expiry-spike contribution.

Result: hottest-key load on any single node ≤ 35K ops/sec. 10x headroom.
```

## 8. Eviction policy

**Committed: W-TinyLFU with size-aware admission (Greedy-Dual-Size weighting).** **[STAFF SIGNAL: eviction-policy modernity]**

W-TinyLFU (the policy from Caffeine, also present in modern Redis variants and CDN caches) maintains a small *frequency sketch* (a Count-Min) and an LRU window. Admission to the main cache is gated: a candidate is admitted only if its sketched frequency exceeds the eviction victim's frequency. This solves the **scan-resistance** problem that vanilla LRU fails (a single full-table scan flushes a vanilla LRU; W-TinyLFU rejects the scan's one-touch entries on admission).

**Rejected:**
- **LRU:** scan-vulnerable. Hit rate ~5–15 percentage points lower than W-TinyLFU on Zipfian workloads — measured on production traces.
- **LFU:** stale-set problem (yesterday's hot key squats forever). Slow to adapt.
- **ARC:** patented historically, less library support, marginal gains over W-TinyLFU.
- **Random/FIFO:** surprisingly competitive on uniform workloads, but loses badly on Zipfian. Not our workload.
- **2Q / SLRU:** approximations of the same idea W-TinyLFU does better.

**Size-awareness:** values range 200B → 64KB. Naive frequency-based eviction may evict a 200B value to make room for a 64KB value of equal frequency — a 320× space loss for parity. **Greedy-Dual-Size** weights eviction priority by `frequency / size`. A 64KB candidate must be 320× more frequent than a 200B incumbent to be admitted. This is non-negotiable when value sizes vary by >10×.

**TTL is orthogonal to eviction.** Every entry has a TTL ceiling (30s default, 300s max for `cold:*`-tagged entries). TTL expiry removes entries; eviction-on-pressure removes entries when memory is tight. Both can fire on the same key.

**Memory-pressure threshold:** eviction triggers at 85% memory utilization, not at OOM. The 15% headroom absorbs traffic spikes and replication backlog without entering the slow-path of synchronous eviction during reads. Crossing 95% triggers the **emergency-shed mode**: writes are rejected (cache becomes read-only), TTLs are aggressively shortened, and an alarm fires.

## 9. Cache stampede / thundering herd

**[STAFF SIGNAL: stampede-discipline]** Three named mechanisms, layered:

**Mechanism 1: Single-flight (request coalescing) at two levels.**

```
   App process                                  Cache shard
   ┌──────────────────────────┐                ┌──────────────────────┐
   │                          │                │                      │
   │  read("foo") ─┐          │                │  read("foo") ─┐      │
   │  read("foo") ─┤          │                │  read("foo") ─┤      │
   │  read("foo") ─┼─► coalesce single fetch ──►  ...          ├──► coalesce │
   │  read("foo") ─┘          │                │  read("foo") ─┘      │ DB fetch │
   │                          │                │                      │
   └──────────────────────────┘                └──────────────────────┘
        per-process map                              per-shard map
        of in-flight keys                            of in-flight keys
```

A miss for `K` registers a future in the per-shard in-flight map. Concurrent misses for `K` await the same future. Only one upstream DB fetch per shard per missing key. With 14 shards and a hot missing key, max DB load = 14 concurrent fetches per miss event — manageable.

**Mechanism 2: Probabilistic early expiration (XFetch algorithm).**

For a key with TTL `T_total` and expected fetch latency `δ`, each read computes:

```
  if (time_remaining < δ × β × ln(rand())):
      refresh_async(K)      // serve current value; refresh in background
```

Where `β` controls aggressiveness (β=1 typical). This makes the *earliest* refresh probabilistically distributed in time — the first reader near expiry triggers an async refresh; subsequent readers see the still-fresh value or the just-refreshed one. **No discontinuity at TTL = 0.**

**Mechanism 3: Stale-while-revalidate.**

When a read finds an expired-but-recent (within `2 × TTL_total`) value, return the stale value immediately and asynchronously refresh. The stale flag is propagated as a response header so the application can decide; defaults to "accept stale." This gives users the latency of a hit during refresh, at the cost of brief staleness — which is exactly the contract we already promised (bounded staleness).

**The post-invalidation stampede sibling.** When CDC invalidates a hot key, every shard's copy is deleted simultaneously. The next read on each replica misses, and 14 shards × ~5 concurrent reads each = ~70 concurrent DB fetches arrive at the DB at once. Mitigation: invalidation **stagger** — the invalidation bus jitters delivery to each shard within a 50ms window, smoothing the miss event. Combined with single-flight, post-invalidation DB load is bounded at ~14 concurrent reads per invalidated key, smoothed across 50ms ≈ ~280 ops/sec per hot-key invalidation.

## 10. Failure modes and graceful degradation

**[STAFF SIGNAL: failure mode precision]** **[STAFF SIGNAL: blast radius reasoning]**

**Single shard down (primary lost).** Clients receive timeout. SDK promotes a replica to primary via control-plane election (~3s). During the window, reads continue from remaining replicas; writes return `503` for that key range. Application is expected to handle write-503 (retry with backoff). Hit rate locally drops to ~0% on the failed shard's keys for ~3s.

**Single shard slow (gray failure).** The killer scenario — node not down, just degraded (GC pause, NIC flap, noisy neighbor). SDK detects via per-shard latency p99 > 5× rolling-baseline; trips a circuit breaker that diverts new reads to replicas. Hedged reads (issued at p95-mark) catch in-flight slow requests. Without hedging, gray failure dominates p99.9 incidents.

**Stale shard map at client (during topology change).** Wrong-shard request → shard responds `MOVED v=N+1`. Client refreshes map and retries. Bounded one-extra-hop cost. Old maps eventually expire.

**Network partition between cache and DB (CDC bus down).** Cache continues serving reads from its current state; staleness-bound enforcement activates: as CDC lag exceeds 30s, the affected shards return `503 STALE` rather than serving silently-stale data. Writes to the cache itself proceed, but invalidations queue. When connectivity restores, queued invalidations replay (with version fence preventing out-of-order corruption).

**Cache fully unavailable (the cascading-failure case).** **[STAFF SIGNAL: cache-as-load-bearing]** This is the scenario most designs ignore. If clients naively fall through to the DB on every miss, the DB receives 10M ops/sec — ~100x its survivable load. The DB melts. Recovery is slow because the DB cannot serve traffic *and* warm the cache simultaneously.

The architectural protection is a **per-client fall-through budget**:

```
  If cache_miss_rate > 30%  AND  cache_p99 > 10ms:
      enter "brownout mode":
        - Fall-through is rate-limited per client to N ops/sec/key
        - Excess requests return application-layer-degraded responses
          (last-known-good values cached at the client / static defaults)
        - "Loss-of-service" responses for non-critical reads
```

Brownout is a **product decision encoded in infrastructure** — the application and product team must agree on which reads can degrade and to what. Login session reads degrade differently than a homepage banner.

The cache is also **warm-restartable**: shard memory snapshots to local SSD every 5 minutes (background fork-and-dump, no read-path impact). On planned restart, the shard reloads from snapshot — cold-start hit rate is ~80% rather than 0%. **[STAFF SIGNAL: cold-cache problem]**

**Cache poisoning.** A bad write — application bug, malformed CDC event, attack — propagates a wrong value. Mitigations: validation hooks on write (schema check, size check, suspicious-content rules); ability to flush by key prefix from a control-plane API; per-write audit log with attribution. Recovery from poisoning is a flush-by-prefix and let CDC repopulate.

**The replica-divergence case.** Async replication can diverge if a primary commits writes the replica never sees (primary crash with unreplicated tail). On primary recovery / replica promotion, divergence is detected via version vector compare; the divergent entries are **invalidated rather than reconciled** — the DB is the source of truth, and any cache divergence is resolved by re-fetch.

## 11. Operational reality

**[STAFF SIGNAL: observability discipline]** Metrics that drive decisions, not vanity metrics:

```
   Metric                              Decision it drives
   ──────────────────────────────────  ──────────────────────────────────────
   Hit rate (per-shard, per-prefix)    Sizing; eviction policy effectiveness
   Per-shard QPS skew (max/median)     Hot-key detector calibration
   Per-key top-K rate                  Hot-key promotion to hot-key tier
   p50/p99/p99.9 read latency          Same-rack placement, hedging tuning
   Eviction rate per shard             Memory pressure; sizing
   Replication lag (primary→replica)   Strong-read failover, RYW correctness
   CDC lag (DB→cache)                  Staleness-bound enforcement
   Fall-through budget consumption     Cache-as-load-bearing alarm
   Single-flight coalesce ratio        Stampede-control effectiveness
```

**Deployment.** Rolling restart shard-by-shard with snapshot-restore. Each shard restart: ~30s downtime for that shard's range; reads served by replicas; writes return retryable 503 for ~3s during primary failover. Full-fleet rollout: ~90 minutes for 40 nodes, sequenced by shard topology.

**Capacity planning loop.** Hit rate < 95% → working set exceeds memory → grow. Per-shard p99 > budget → CPU/NIC bound → shard-split. Hot-key skew > 50× → tune detector or expand hot-key tier.

**Cost attribution.** $60K/month/region. Three regions ⇒ $180K/month ⇒ ~$2.2M/year. Compared to the alternative of running the DB at 10M ops/sec capacity (~$10M+/year for the storage tier alone), the cache pays for itself ~5x over. The conversation with finance is explicit: cache cost vs DB cost vs latency / hit-rate target. Tuning the TTL ceiling, the hot-key threshold, and the working-set headroom are all cost levers.

## 12. Multi-region considerations

**[STAFF SIGNAL: multi-region honesty]** Scoped: each region runs an independent cache fleet. Cross-region reads are out of scope. Cross-region invalidation is in scope.

**Mechanism.** The CDC bus is global (Kafka MirrorMaker or equivalent), with bounded cross-region replication lag (target p99 < 5s, hard cap with alarm at 30s). Every regional cache subscribes to its own region's invalidation topic *and* a global cross-region invalidation topic. A write in region A produces (a) a local invalidation immediately, (b) a global invalidation that propagates to regions B and C within seconds.

**The user-mobility staleness case.** User updates profile in EU; flies to US, opens app. The US cache may still have the pre-update value for ≤5s p99, ≤30s worst case. This is the cost of regional caching, and it must be communicated to the product team. For data classes where this is unacceptable (e.g. payment state), the `strong:*` namespace bypasses the regional cache and reads from the global DB primary directly — paying the cross-region latency once, in exchange for correctness.

**No multi-region active-active write coordination.** Last-writer-wins on conflicts at the cache layer; the DB layer is the conflict-resolution authority. The cache merely reflects whichever write the DB committed last.

## 13. Tradeoffs taken and what would change them

- **Bounded staleness over strong consistency:** chosen because the read/write ratio is 95/5 and the workload tolerates seconds of staleness. *Would change* if the workload were trading-platform-class (millisecond-meaningful state changes) — would shift to read-through cache with synchronous invalidation, accepting lower throughput.
- **Cache-aside over write-through:** chosen because the application already handles DB writes; cache should not be on the write critical path. *Would change* if the cache were a primary store (e.g. session storage with no DB), in which case write-through is mandatory.
- **HRW over consistent hashing:** chosen for ranked-replica selection and heterogeneous-server flexibility. *Would change* to consistent hashing if the team's existing libraries / observability assumed it — operational familiarity matters.
- **Hot-key tier as separate fleet:** chosen because the load-isolation property is worth the operational cost. *Would change* (collapse hot-key tier into main fleet with per-shard hot-key replicas) if hot-key cardinality were tiny (<100) or skew were moderate.
- **W-TinyLFU + Greedy-Dual-Size:** chosen for Zipfian + variable-size workload. *Would change* to plain LRU if value sizes were uniform and access pattern were close to recency-driven.

## 14. What I would push back on

**[STAFF SIGNAL: saying no]**

1. **"Sub-millisecond p99 at 10M ops/sec" — qualified accept.** Sub-ms p99 is achievable *same-rack* with the design above. Cross-rack adds ~200μs and pushes p99 over budget under load. Cross-region is physically impossible (RTT alone exceeds the budget). The honest SLA is **"<1ms p99 same-rack, <2ms p99 same-AZ, best-effort cross-region with 50ms+ p99."** Anyone promising sub-ms cross-region is wrong.

2. **"Distributed cache" implies one tier — push back.** The right architecture has **four tiers**: client micro-cache, main fleet, hot-key fleet, invalidation bus. A single-tier "distributed cache" cannot survive Zipfian load at 10M ops/sec. The interview's framing collapses this; the answer should not.

3. **The implicit "use Redis" answer.** Redis is *an* implementation; the design is independent of vendor. For very high write throughput, Dragonfly or similar shard-threaded engines outperform Redis 5x per node. For very large values, a different architecture entirely (object store + pointer cache) is correct. The vendor choice follows from the workload, not the other way around.

4. **The implicit assumption that the cache is a performance tool.** At 10M ops/sec, the cache is **load-bearing for the DB's availability**. Its outage is a product outage, not a latency regression. This reframes the operational priority — the cache deserves the same SRE rigor as the DB itself, possibly more.

5. **The "one consistency model" assumption.** Different keys deserve different contracts. The `strong:*` namespace is not a complication; it is the correct shape of the answer. A staff engineer pushes for *per-key consistency declarations as a first-class API*, not a single global setting.