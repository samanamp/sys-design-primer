---
title: Caching & In-Memory Systems
description: Caching & In-Memory Systems
---

## 1. What a Staff Engineer Actually Needs to Know

**What matters in interviews:**
- Knowing **when** to add a cache and **why**, not just that you can.
- Naming the **pattern** (cache-aside, write-through, etc.) and its failure modes.
- Reasoning about **consistency, invalidation, and hot keys** without hand-waving.
- Placing cache at the **right layer** (client → CDN → app → distributed → DB buffer).
- Quantifying: hit rate, TTLs, memory cost, P99 impact.

**Expected depth:**
- You should speak fluently about cache-aside vs write-through, TTL strategy, stampede mitigation, hot-key handling, and local vs distributed tradeoffs.
- You should be able to sketch a system where cache is one layer among many, and explain what breaks when it fails.

**What does NOT matter (unless role is cache-infra-focused):**
- Redis cluster gossip protocol internals.
- Memcached slab allocator specifics.
- Exact LRU/LFU implementation (CLOCK vs TinyLFU vs W-TinyLFU).
- Consistent hashing math beyond "virtual nodes smooth distribution."
- CRDT internals.

The bar: **make correct design decisions and defend them**, not recite implementations.

---

## 2. Core Mental Model

**A cache is a bet.** You're betting that:
1. Reads dominate writes for this data.
2. The same data is read repeatedly within some window.
3. The staleness window is acceptable to the business.

If any of those three is false, cache is the wrong tool.

**Three things caches actually do:**

| Goal | What it buys you | Example |
|---|---|---|
| **Latency reduction** | P99 drops from 50ms → 1ms | User profile lookup |
| **Throughput protection** | DB QPS drops 10–100x | Hot product page during flash sale |
| **Cost reduction** | Fewer expensive compute/IO calls | LLM inference results, complex joins |

**Cache is never the source of truth.** It's a derived, lossy, expirable projection of the SoT. If the cache disappears, the system must still be correct — just slower. State this explicitly in interviews.

**In-memory systems ≠ caches.** Redis is also used as:
- Session store (ephemeral but authoritative during session).
- Rate-limit counters (authoritative, but bounded-loss acceptable).
- Leaderboards (sorted set is the actual data structure the app needs).
- Ephemeral coordination (locks, job queues).

When Redis is SoT for a workload, you accept losing some data on failure — it's a business decision, not a design mistake.

**When cache hurts:**
- Write-heavy workloads (invalidation cost > read savings).
- Low reuse / long-tail access (cache churns, no hit rate).
- Correctness-critical data with no staleness tolerance (ledgers, auth tokens).
- Tiny datasets that fit in DB buffer cache already.

```
        READ-HEAVY, REUSED,    WRITE-HEAVY, UNIQUE,
        STALENESS OK           STALENESS BAD
        ┌──────────────┐       ┌──────────────┐
        │  CACHE WINS  │       │ CACHE HURTS  │
        └──────────────┘       └──────────────┘
```

---

## 3. The Essential Cache Patterns

### 3.1 Cache-Aside (Lazy Loading)

**App manages the cache.** Most common pattern. Default answer for most interviews.

```
READ:                              WRITE:
  app → cache.get(k)                 app → db.write(k, v)
   │                                  │
   ├── HIT → return                   └── cache.delete(k)   (or skip)
   │
   └── MISS → db.read(k)
             cache.set(k, v, ttl)
             return v
```

**Strengths:** Simple. Cache only holds what's actually read. Resilient to cache outage (degrades to DB reads).

**Weaknesses:** First read is always a miss (cold). Race condition on concurrent write+read can cache stale data. Every cache miss = DB hit → stampede risk.

**Failure behavior:** Cache down → all reads hit DB. Must have DB capacity headroom or a fallback.

**Consistency:** Eventually consistent. Stale reads possible until TTL expires or explicit invalidation.

**What interviewers want to hear:** "I'd use cache-aside with a TTL and delete-on-write. I'd add request coalescing to prevent stampedes on hot keys."

---

### 3.2 Read-Through

**Cache manages the DB fetch.** App only talks to cache; cache loads from DB on miss.

```
app → cache.get(k)
       │
       ├── HIT → return
       └── MISS → cache loads from DB, stores, returns
```

**Strengths:** Simpler app code. Centralized loading logic.

**Weaknesses:** Requires cache that supports loader hooks (not raw Redis/Memcached — typically a library like Caffeine, or a cache service). Tight coupling between cache and DB schema.

**Use case:** Mostly in-process caches (Caffeine, Guava). Less common for distributed cache in interview contexts.

---

### 3.3 Write-Through

**Writes go to cache AND DB synchronously.**

```
WRITE:  app → cache.set(k, v) → db.write(k, v) → return
        (or the cache does both atomically)
```

**Strengths:** Cache never stale relative to DB. Reads after write always hit.

**Weaknesses:** Every write pays cache + DB latency. Cache holds data that may never be read (wasted memory). Still doesn't solve consistency under failure — if DB write fails after cache write, you're inconsistent.

**Failure behavior:** Atomic write is hard without 2PC. Usually: write DB first, then cache; if cache write fails, accept stale until TTL.

**What interviewers want to hear:** "Write-through trades write latency for read consistency. I'd use it when writes are rare and reads must be fresh — e.g., feature flags, config."

---

### 3.4 Write-Behind (Write-Back)

**Writes go to cache; cache flushes to DB asynchronously.**

```
WRITE:  app → cache.set(k, v) → return (fast!)
                │
                └── async worker → db.write(k, v)
```

**Strengths:** Very low write latency. Batches writes to DB (huge throughput win).

**Weaknesses:** **Data loss on cache failure** before flush. Ordering/consistency across keys is tricky. DB can lag significantly.

**Use case:** Metrics, counters, high-volume event ingestion where some loss is OK. Rarely the right answer for user-facing data unless you can reconstruct.

**What interviewers want to hear:** "Write-behind is for write-heavy workloads where some loss is acceptable. I'd use a durable queue (Kafka) in front instead of trusting cache as write buffer for critical data."

---

### 3.5 Refresh-Ahead

**Cache proactively refreshes entries before they expire**, based on access patterns.

**When to mention:** Predictable hot keys with tight freshness requirements (e.g., top-10 trending, homepage data).

**Tradeoff:** Extra load on DB for items that might not be read again. Only worth it if hit rate on refreshed key is very high.

---

### Pattern Quick Reference

| Pattern | Read | Write | Best For |
|---|---|---|---|
| Cache-aside | Miss → load | Delete cache | **Default choice** |
| Read-through | Cache loads | (separate) | In-process caches |
| Write-through | Normal | Cache + DB sync | Read-after-write freshness |
| Write-behind | Normal | Cache, async DB | High-write, loss-tolerant |
| Refresh-ahead | Normal | Normal | Predictable hot keys |

---

## 4. Must-Know Concepts

**TTL (Time-To-Live)** — How long an entry is valid. Primary invalidation mechanism for eventually-consistent caches. Shorter TTL = fresher data + higher miss rate + more DB load. Longer TTL = staler data + better hit rate.

**Expiration** — Entries can expire lazily (checked on access) or actively (background sweeper). Redis does both. Matters because with lazy-only expiration, dead entries consume memory until touched.

**Eviction** — What happens when cache is **full** and needs to admit new entries. Distinct from expiration.

**LRU (Least Recently Used)** — Evict the entry not accessed for the longest time. Good general-purpose default. Fails on scan workloads (one-time large reads pollute the cache).

**LFU (Least Frequently Used)** — Evict the entry with fewest accesses. Better for workloads with stable popularity distributions. Classic LFU doesn't age counters; modern variants (TinyLFU) do.

**Interview depth:** Know LRU vs LFU, know when each wins. Don't go deeper unless asked.

**Cache hit rate** — `hits / (hits + misses)`. A hit rate below 80% for a hot path usually means the cache is miscalibrated (wrong key granularity, too-short TTL, too-small size, or wrong workload). A cache at 50% hit rate is adding a network hop for little gain.

**Hot keys** — A small number of keys getting a disproportionate share of traffic. Single cache shard becomes the bottleneck. Mitigation below.

**Cache invalidation** — Keeping cache consistent with SoT. Famously hard (Phil Karlton's "two hard things" line). Primary strategies: TTL, explicit delete on write, versioned keys.

**Stale data** — Cache value is older than SoT. Sometimes acceptable, sometimes not. Business decides.

**Negative caching** — Caching "not found" / error results. Prevents repeated lookups for missing keys (especially important for DoS-style attacks where attacker probes non-existent keys). Use short TTL.

**Request coalescing / single-flight** — When many requests miss for the same key simultaneously, only one goes to the DB; others wait for its result. Critical for stampede prevention.

```
Without coalescing:              With coalescing (single-flight):
  req1 ─► MISS ─► DB              req1 ─► MISS ─► DB
  req2 ─► MISS ─► DB              req2 ─► MISS ─► wait ─┐
  req3 ─► MISS ─► DB              req3 ─► MISS ─► wait ─┼─► result
  req4 ─► MISS ─► DB              req4 ─► MISS ─► wait ─┘
  (4 DB calls)                    (1 DB call)
```

**Cache stampede / dogpile** — Popular key expires; N concurrent requests all miss and hammer DB. Mitigation: coalescing, probabilistic early expiration, locked refresh.

**Warmup / cold start** — Empty cache after deploy/restart. All requests miss → DB overload. Mitigation: prewarm from known hot keys, staged rollout, shadow traffic.

**Local cache vs distributed cache** — Local: in-process, nanosecond access, no network, but per-instance (N copies, N× memory, consistency drift across replicas). Distributed: shared, consistent view, one network hop (~0.5–1ms), scales memory.

**In-memory store vs cache** — A *cache* can be thrown away without losing data. An in-memory *store* (Redis used for sessions, rate limits, sorted sets) holds authoritative data you'd lose on failure. Different durability contract, different design choices (persistence, replication).

---

## 5. Cache Placement and Layers

```
┌────────────┐  browser/app cache: HTTP cache, IndexedDB, SW cache
│  CLIENT    │  Good for: static assets, per-user data
└─────┬──────┘
      │
┌─────▼──────┐  CDN / edge cache: geographically distributed
│   EDGE     │  Good for: static content, public GET responses,
│  (CDN)     │  API responses safely cacheable by URL
└─────┬──────┘
      │
┌─────▼──────┐  in-process: Caffeine, Guava, sync.Map
│   APP      │  Good for: small hot data, per-pod, ns-latency
│  (local)   │
└─────┬──────┘
      │
┌─────▼──────┐  Redis / Memcached cluster
│DISTRIBUTED │  Good for: shared hot data, cross-pod consistency,
│  CACHE     │  session store, counters
└─────┬──────┘
      │
┌─────▼──────┐  DB buffer pool: pages in RAM managed by DB
│  DB BUFFER │  Good for: working set that fits RAM; automatic
└─────┬──────┘
      │
┌─────▼──────┐
│   DISK     │
└────────────┘
```

**When each layer helps:**

- **Client cache:** Avoid network entirely. Best wins. Bound by cacheability (user-specific data, auth).
- **CDN/edge:** Massive win for any content that's the same for many users. Also protects origin during spikes.
- **Local in-process:** Sub-microsecond hits, no network. Use for small, hot, somewhat-stale-tolerant data (config, feature flags, small lookup tables). Key limit: memory per pod, staleness across pods.
- **Distributed cache:** The workhorse for shared hot data. One network hop but consistent view across all app instances.
- **DB buffer cache:** Free if working set fits. Interviewers sometimes forget this exists — mentioning it signals seniority ("do we even need Redis? the working set is 2GB and Postgres has 16GB shared_buffers").

**Multi-layer rule:** Each layer should have a higher hit rate at lower cost than the layer below. If your local cache has a 20% hit rate, the distributed cache it fronts should have >90% — otherwise local is just added complexity.

---

## 6. Invalidation and Consistency

The hardest part. Staff-level answers differentiate here.

**The core problem:** cache and DB are two stores. Any write sequence across them has a window where they disagree. You pick which anomalies you tolerate.

### Strategies

**TTL-based** — Set expiration; accept staleness up to TTL. Simple, robust, eventually consistent. Default choice when business can tolerate bounded staleness.

**Explicit invalidation (delete-on-write)** — On DB write, delete the cache key. Next read repopulates. Simple but races exist.

**Write-through** — Update cache atomically with DB. Reduces staleness window but doesn't eliminate it under partial failures.

**Versioned keys** — Include a version in the key (`user:123:v42`). Bumping version effectively invalidates all cached variants. Old entries age out via TTL. Great for schema changes, bulk invalidation.

### Delete vs Update on Write

**Delete-on-write (preferred):** After DB write, `DELETE cache_key`. Next read misses, re-reads DB, repopulates. Safe against stale caches.

**Update-on-write:** After DB write, `SET cache_key = new_value`. Faster on next read, but has a **classic race**:

```
T1: read DB (old)
T1: [network delay]
T2: write DB (new)
T2: SET cache = new
T1: SET cache = old    ← cache now stale, no TTL until next write
```

**Delete-on-write dodges this** because T1's stale read just repopulates into an empty slot; and if T2's delete happened after T1's set, T1's entry gets cleared.

The cleanest pattern: **DB write first, then cache delete, with a TTL as backstop**.

### Stale Reads & Eventual Consistency

State these plainly in interviews:
- Cache is eventually consistent with DB.
- Staleness window = max(TTL, replication lag + invalidation delivery time).
- For correctness-sensitive paths (balances, permissions, auth), **bypass the cache** or use very short TTLs with strong invalidation.

### Why Invalidation Is Hard

1. **Multiple writers** can race.
2. **Multi-region** cache replication adds lag.
3. **Related keys**: updating user profile might need to invalidate "user_posts:123", "friends_of:123", etc. — fan-out invalidation is error-prone.
4. **Failure during invalidation**: DB write succeeds, cache delete fails → stale until TTL.

### What a Staff-Level Candidate Should Say

"For correctness-sensitive data (auth, money, permissions), I read from SoT or use very short TTLs with explicit invalidation. For everything else, I use cache-aside with delete-on-write plus a TTL backstop, and I accept a bounded staleness window that the product can tolerate. I identify related-key invalidation explicitly in the design and either use versioned keys or a coarser invalidation event (e.g., pub/sub to invalidate a group)."

---

## 7. Failure and Hotspot Scenarios

### Hot Key

One key gets disproportionate traffic. The shard holding it saturates while others idle.

**Mitigations:**
- **Local cache in front** of distributed cache (absorbs 90%+ of reads locally).
- **Key splitting / replication**: store `hot_key:0..N`, clients pick randomly, read from any copy. Writes fan out.
- **Read replicas** for cache: route reads to replicas.
- **Consistent hashing with virtual nodes** helps distribution of *many* keys, not a single hot key.

### Cache Stampede / Thundering Herd

Hot key expires → N concurrent misses → N DB calls.

**Mitigations:**
- **Single-flight / request coalescing**: only one in-flight fetch per key.
- **Probabilistic early expiration**: refresh with small probability as TTL approaches so you don't hit a cliff. (XFetch algorithm.)
- **Distributed lock**: one worker fetches, others wait/retry.
- **Stale-while-revalidate**: serve stale value while async refresh runs.

### Cache Node Failure

- If cache is a performance layer: failover, accept higher DB load until recovery. DB must have headroom (2–3× normal traffic, typically).
- If cache is SoT (sessions, counters): replication matters. Redis replica + sentinel/cluster for failover; accept some data loss window.

### Cold Start

After restart/deploy, cache is empty. All traffic misses → DB overload → failures → retries → worse.

**Mitigations:**
- **Prewarm** from known hot keys (top-N by past access).
- **Staged rollout**: ramp traffic gradually.
- **Shadow traffic**: warm new cache from prod traffic before cutover.
- **Per-key rate limit to DB**: cap miss-driven DB QPS.

### Uneven Key Distribution

Bad hashing, small key space, or genuine skew. Symptom: one shard at 80% CPU, others idle.

**Mitigations:** consistent hashing with virtual nodes, rebalancing, or key splitting for individual hotspots.

### Stale Cache After SoT Update

Write committed to DB, cache invalidation lost (network partition, bug, async delivery failure).

**Mitigations:** TTL as backstop (always), idempotent invalidation events, monitoring for cache/DB divergence on sampled keys.

### Retry Storms

When DB is slow, clients retry → more load → slower → more retries. Cache misses amplify this.

**Mitigations:** exponential backoff + jitter, circuit breakers, rate limiting at the cache miss boundary, token buckets per key.

### Mitigation Toolkit Summary

```
Problem              | Tool
─────────────────────┼────────────────────────────────────
Hot key              | Local cache, key splitting, replicas
Stampede             | Single-flight, jittered TTL, SWR
Cold start           | Prewarming, staged rollout
Node failure         | Replication, graceful degradation
Retry storms         | Backoff+jitter, circuit breaker
Stale data           | TTL backstop, versioned keys
Uneven shards        | Virtual nodes, rebalancing
```

---

## 8. Distributed Cache Design Tradeoffs

### Sharding / Partitioning

Keys partitioned across nodes by hash. **Consistent hashing** with virtual nodes minimizes rebalancing on node add/remove (k/N keys move instead of nearly all). Say this once in interviews and move on — don't whiteboard the math.

### Replication

- **Read replicas** for throughput and hot-key relief.
- **Primary-replica** for HA: replica takes over on primary failure.
- **Async replication** = risk of small data loss window.

Redis: primary-replica with Sentinel (HA) or Cluster (sharding + replication). Memcached: client-side sharding, no replication — simpler, less durable.

### Consistency

Caches are almost always eventually consistent. Strong consistency in a distributed cache is possible (consensus protocols) but **expensive and usually the wrong tool** — if you need strong consistency, you probably want a database, not a cache.

### Memory Cost vs Hit Rate

More memory → more entries → higher hit rate — with **diminishing returns**. Plot hit rate vs cache size; the knee tells you budget. Going from 10% to 20% of working set often gets you from 50% to 85% hit rate; from 80% to 95% often requires 5× more memory.

### Serialization / Deserialization Cost

Often overlooked. Serializing a 100KB object to JSON/protobuf costs real CPU. If you cache large objects and deserialize on every hit, you may not be saving much vs. DB. Options: cache already-serialized bytes, cache smaller projections, cache computed results rather than raw data.

### Network Hop Cost

Distributed cache ≈ 0.5–1ms RTT intra-DC. Local memory ≈ 100ns. That's **~10,000×**. For very small, very hot data, local cache is dramatically faster — even if distributed cache would hit.

### Local vs Remote: When Each Wins

**Local cache wins when:**
- Data is small and fits comfortably in-process.
- High read rate per instance.
- Staleness across instances is tolerable.
- You want sub-microsecond latency.

**Distributed cache wins when:**
- Shared state across instances matters (sessions, rate limits).
- Data too large to replicate to every pod.
- You need a single source of invalidation.
- Cross-instance consistency matters more than absolute latency.

**Common pattern:** both. Local L1 in front of distributed L2. L1 catches the very hottest keys; L2 catches the warm tail; DB catches the cold.

---

## 9. In-Memory Systems Beyond Caching

Redis is often used as an in-memory **data structure store**, not a cache. Interview-useful cases:

**Session storage** — User sessions keyed by session ID. TTL = session length. Redis handles millions of sessions easily. Durability requirement is low (user re-logs in on loss), but replication avoids that UX hit.

**Rate limiting counters** — `INCR user:123:minute_42` with TTL. Token buckets, sliding windows, fixed windows all trivial. Atomic ops are the key property — you can't implement this correctly against most SQL DBs at scale.

**Leaderboards / sorted sets** — Redis ZSET: `ZADD leaderboard score member`, `ZRANGE` for top-K, `ZRANK` for position. O(log N). This is Redis being used because its **data structure** is the right tool, not because it's fast.

**Ephemeral coordination** — Distributed locks (with caveats — Redlock is controversial; for correctness-critical locks use Zookeeper/etcd). Job deduplication. Short-lived flags.

**Queues / pub-sub** — Redis Streams, LPUSH/BRPOP, PUB/SUB. Fine for low-to-medium-scale internal messaging. For durable, high-throughput, ordered, or replay-required queues, use Kafka.

**Key interview insight:** When someone says "use Redis," ask **why**. Cache? Data structure store? SoT for ephemeral data? The answer changes the design. A staff engineer makes this distinction explicitly.

---

## 10. Interview Reasoning Patterns

**When should you add a cache?**
"When reads dominate, the same data is read repeatedly, and the product tolerates some staleness. I'd measure hit rate projections from access patterns before committing — a cache with 30% hit rate is often worse than no cache. I'd also check if a DB buffer cache or index tuning already solves the problem."

**What layer should you cache at?**
"Cache at the highest layer where the data is still cacheable. Public static data: CDN. Per-user cacheable data: local + distributed. Hot shared state: distributed cache. Small, hot, low-staleness-cost data: local cache. Multi-layer if the cost/hit-rate math works."

**When is Redis a bad answer?**
"When the data is write-heavy, unique per access, or correctness-critical. When the working set doesn't fit in memory economically. When the real need is durable storage (use a DB). When I'm adding it reflexively without identifying what I'd cache or why — that's where junior answers fail."

**How do you keep cache and DB consistent enough?**
"Cache-aside with DB-write-then-cache-delete, plus a TTL backstop. For correctness-sensitive paths, bypass cache or use short TTLs. Use versioned keys when I need bulk invalidation. Accept eventual consistency — strong consistency in a cache usually means I chose the wrong tool."

**How do you handle hot keys?**
"Detect first via key-level metrics. Then: local cache in front to absorb reads; key splitting with N replicas if the key is genuinely burning a shard; read replicas for distributed cache. Consistent hashing doesn't solve a single hot key — it only helps overall distribution."

**How do you prevent cache stampede?**
"Single-flight coalescing so only one miss goes to DB per key. Jittered TTLs so keys don't expire in sync. Probabilistic early refresh for very hot keys. Stale-while-revalidate if the product tolerates it."

**When is local cache better than distributed cache?**
"When the data is small, hot, and staleness across instances is tolerable. Sub-microsecond access vs. sub-millisecond. For session data or shared counters, distributed wins because the local option is wrong, not slow."

**When should cache be bypassed?**
"For correctness-critical reads (after-write reads in money flows, auth checks with recent revocation). For long-tail keys where cache hit rate would be near zero. When debugging consistency issues. Expose a 'skip cache' path explicitly."

---

## 11. Common Mistakes Candidates Make

- **"I'll add Redis"** without specifying *what* is cached, at what granularity, with what TTL, and what the invalidation strategy is.
- **Treating cache as SoT** — designing a system that loses data if cache goes down.
- **Ignoring invalidation** — setting a TTL and calling it a day even for write-heavy data.
- **Ignoring hot keys** — assuming uniform distribution across shards.
- **Hand-waving consistency** — "eventually consistent" without specifying the window or what anomalies the user sees.
- **Assuming higher TTL is always better** — ignoring staleness cost to business.
- **Ignoring cold start** — presenting a design that can't actually handle a deploy.
- **Ignoring network latency of remote cache** — treating Redis as free. It's ~1ms. If you're chasing P50 of 5ms, that's 20%.
- **Confusing cache with durable datastore** — using Redis for data that must survive failure, without persistence/replication config, without acknowledging the risk.
- **Over-caching** — layers of cache where a single layer would do, or caching data that's rarely re-read.
- **Not mentioning DB buffer cache** — sometimes the "add a cache" answer is "the DB already caches this."
- **Caching at wrong granularity** — caching whole objects when only a field is hot, or caching fields when the whole object is always read together.

---

## 12. Final Cheat Sheet

### Table 1: Local vs Distributed Cache vs DB Read

| Dimension | Local Cache | Distributed Cache | DB Read |
|---|---|---|---|
| Latency | ~100ns | ~0.5–1ms | 1–50ms |
| Consistency across instances | Drifts per-pod | Single shared view | Authoritative |
| Memory cost | N × (per-pod size) | Cluster size | (already paid) |
| Blast radius on failure | Per-pod | Cluster-wide | System-wide |
| Best for | Small, hot, staleness-OK | Shared hot, sessions | SoT, cold data |
| Scalability of memory | Limited to pod RAM | Horizontal | Horizontal |
| Cross-instance invalidation | Hard (pub/sub hacks) | Easy | N/A |

### Table 2: Cache-Aside vs Write-Through vs Write-Back

| Dimension | Cache-Aside | Write-Through | Write-Back |
|---|---|---|---|
| Read path | Miss → load → cache | Always hits (after first write) | Always hits |
| Write path | DB write, delete cache | Cache + DB sync | Cache now, DB async |
| Write latency | DB latency | Cache + DB | Cache only (fast) |
| Read-after-write freshness | Miss on next read | Hit with fresh data | Hit with fresh data |
| Data loss on cache failure | None (DB is SoT) | None | **Yes** (pre-flush writes) |
| Best for | General read-heavy | Read-after-write matters | Write-heavy, loss-tolerant |
| Complexity | Low | Medium | High |

### Decision Framework for Interviews

```
1. Do reads dominate writes?               No  → probably skip cache
   Yes ↓
2. Is data reused within a time window?    No  → probably skip cache
   Yes ↓
3. Is staleness tolerable?                 No  → bypass cache or short TTL
   Yes ↓
4. Shared state across instances needed?   No  → local cache (+ maybe distributed L2)
   Yes ↓
5. Distributed cache (Redis/Memcached).
6. Pick pattern: cache-aside by default.
7. Set TTL based on staleness tolerance.
8. Plan: hot keys, stampede, cold start, invalidation.
9. Identify what bypasses cache (correctness paths).
10. State the failure mode: cache down → DB handles N× load.
```

### 10 Likely Interview Questions + Strong Short Answers

**1. "How would you cache user profiles?"**
Cache-aside, key = `user:{id}`, value = profile blob, TTL 5–15 min. Delete on write. Local L1 for very hot users (celebrities, admins). Bypass cache for auth-sensitive reads.

**2. "How do you handle cache stampede?"**
Single-flight coalescing at the app layer, jittered TTLs to desynchronize expirations, and probabilistic early refresh for the hottest keys. Stale-while-revalidate if product allows.

**3. "Redis goes down. What happens?"**
If cache is performance layer: DB load spikes; needs headroom (design for 2–3× normal). If cache is SoT (sessions): users logged out or counter state lost unless replicated. I design assuming a cache-down period is survivable and add replication where data loss is unacceptable.

**4. "How do you keep cache consistent with DB?"**
DB write first, then cache delete. TTL as backstop. Versioned keys for bulk invalidation. For correctness-critical reads, bypass cache. Acknowledge a bounded staleness window — don't promise strong consistency.

**5. "How do you detect and handle a hot key?"**
Key-level metrics (sampled access counts per shard). Mitigations: local cache in front absorbs most reads; replicate the key as `hot_key:0..N` and randomize client reads; route reads to replicas. Consistent hashing alone doesn't solve this.

**6. "When would you NOT use a cache?"**
Write-heavy workloads, unique-per-access data, correctness-critical paths, or when the DB's own buffer cache already handles the working set. Also when hit rate projections are under ~70% — the added complexity isn't worth it.

**7. "How does Redis differ from Memcached?"**
Redis: rich data structures (lists, sets, sorted sets, streams), persistence options, replication, pub/sub. Memcached: simple KV, multithreaded, client-side sharding, no replication. Use Redis if you need data structures or durability; Memcached if you want a pure, horizontally-sharded KV cache.

**8. "How would you design session storage?"**
Redis keyed by session ID, value = session blob, TTL = session length (with sliding expiration on activity). Replicate for HA; accept tiny loss window. Hash-partitioned across the cluster. This is Redis-as-SoT, not cache.

**9. "LRU vs LFU — which would you pick?"**
LRU for general workloads where recency predicts future access. LFU for stable-popularity distributions where frequency matters more than recency (e.g., content recommendations). Modern caches often use TinyLFU-style hybrids. For an interview, pick LRU as default and explain when LFU wins.

**10. "Design a rate limiter."**
Redis with atomic `INCR` on a key like `rate:{user_id}:{window}` with TTL = window length. For sliding windows, use sorted sets (`ZADD` with timestamp scores, `ZREMRANGEBYSCORE` to trim). Hash-partitioned; hot users get distributed load. This is Redis being used for its data structures and atomicity, not as a cache.

---

**Final note for interviews:** When in doubt, say cache-aside + TTL + delete-on-write + plan for stampede + identify hot keys + acknowledge the staleness window. That's the staff-level skeleton. Everything else is justification.