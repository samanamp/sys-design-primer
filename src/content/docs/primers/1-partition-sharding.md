---
title: Partitioning & Sharding
description: Partitioning & Sharding
---

## 1. What a staff engineer actually needs to know

In a staff-level interview, **interviewers are not testing whether you can build a database**. They are testing whether you can:

- Pick a shard key and _defend the choice against the workload_.
- Identify where the design breaks (hotspots, cross-shard queries, rebalancing).
- Explain what changes when the system grows 10x or 100x.
- Distinguish **partitioning** (scaling) from **replication** (availability/read scaling) — confusing these is an instant signal of weakness.

**Depth expected:**

- Can you pick hash vs range vs directory and justify it? **Yes.**
- Can you describe consistent hashing well enough to explain how nodes are added without a mass rehash? **Yes.**
- Can you describe a no-downtime resharding plan in ~5 sentences? **Yes.**
- Can you describe B-tree split logic or Dynamo's Merkle tree gossip? **Only if the role is storage-heavy.**

**What does not matter** (unless you're interviewing for a database team): internal page layout, log-structured merge tree write amplification, specific product version behavior, paxos vs raft mechanics.

Rule of thumb: **know the design decisions, not the implementation details.**

---

## 2. Core mental model

**Partitioning / sharding = splitting a logically single dataset or workload across multiple independent units (shards) so that each unit handles a subset.** "Partition" and "shard" are used interchangeably in interviews; some people reserve "partition" for logical split and "shard" for physical placement. Don't fight over the terminology.

**Why shard:**

1. **Data volume** — a single node's disk/RAM can't hold the dataset.
2. **Write throughput** — a single primary can't absorb the write rate.
3. **Read throughput at the key level** — replication fixes read scaling for _different_ keys, but not the bottleneck of a single very hot key.
4. **Blast radius** — isolate failure and noisy tenants.
5. **Geographic locality** — pin data near users for latency/compliance.

**Three things get partitioned, and they are not the same:**

```
  DATA           TRAFFIC          WORK
  ----           -------          ----
  rows, docs,    requests,        jobs, tasks,
  keys          queries          background compute

  sharded DB     LB / gateway     job queue / scheduler
```

A system design often needs to partition all three, and they can use **different** partitioning schemes. Example: shard user data by `user_id` (hash), but route traffic by geography (directory), and partition background jobs by `job_type` (range/directory).

**Workload shape drives everything.**

```
Read-heavy, point lookups          → any scheme works; optimize cache
Write-heavy, key-distributed       → hash
Write-heavy, time-series           → range on time + another dim, OR hash
Heavy range scans                  → range
Multi-tenant with one huge tenant  → directory, maybe per-tenant shards
```

**Partitioning vs replication — internalize this.**

```
PARTITIONING          REPLICATION
------------          -----------
splits the dataset    duplicates the dataset
scales capacity       scales read throughput
                       + provides availability
orthogonal concerns; real systems do both
```

A sharded system where each shard has replicas is the norm. Conflating them ("I'll shard by user_id for high availability") is the #1 staff-interview tell that the candidate is shallow.

---

## 3. The main shard strategies

### 3.1 Hash partitioning

Apply a hash function to the shard key; assign based on `hash(key) mod N` or (better) consistent hashing.

```
            hash(user_id) mod 4
              │
  user_id=42  │→  2  →  Shard 2
  user_id=17  │→  1  →  Shard 1
  user_id=99  │→  3  →  Shard 3
```

- **Routing:** client or router computes hash; cheap and stateless.
- **Strengths:** near-uniform distribution; predictable load; good for write-heavy point-lookup workloads.
- **Weaknesses:** destroys locality — **range queries become scatter-gather** across all shards. Ordered scans and "top N most recent" are painful.
- **Hotspot risk:** low _in the average case_, but a **single hot key** (celebrity) still lands on one shard.
- **Rebalancing:** naive `mod N` requires massive reshuffling when N changes. **Consistent hashing** or **virtual buckets** fix this.
- **What interviewers want to hear:** "Hash by X, using consistent hashing with virtual nodes so adding a shard only moves ~1/N of the data. Range queries will be scatter-gather, which is fine because my workload is point lookups."

### 3.2 Range partitioning

Assign contiguous key ranges to shards.

```
  Shard A:  keys [0000 ─── 2499]
  Shard B:  keys [2500 ─── 4999]
  Shard C:  keys [5000 ─── 7499]
  Shard D:  keys [7500 ─── 9999]
```

- **Routing:** a small routing table maps ranges to shards; clients or a router do lookup.
- **Strengths:** **range scans are cheap** (ordered, local); pagination is sane; good for time-series when combined with another dimension.
- **Weaknesses:** prone to skew. Sequential keys (timestamps, autoincrement IDs) concentrate writes on the tail shard.
- **Hotspot risk:** **high** for monotonic keys. The "latest" range gets all traffic. This is the classic write hotspot.
- **Rebalancing:** split hot ranges; merge cold ones. Requires coordination but is conceptually simple.
- **What interviewers want to hear:** "Range sharding is great for ordered scans, but if the key is time-ordered, writes pile up on one shard. I'd either add a hash prefix to break temporal locality, or use a composite key like `(user_id, timestamp)` so each user gets their own time-ordered partition."

### 3.3 Directory / lookup-based partitioning

Maintain an explicit mapping from key (or key group) to shard, stored in a routing service.

```
  Lookup service
  ┌──────────────────────────┐
  │ tenant_A → Shard 1       │
  │ tenant_B → Shard 1       │
  │ tenant_big → Shard 3     │   (isolated because heavy)
  │ tenant_D → Shard 2       │
  └──────────────────────────┘
              │
              ▼
          shards 1..N
```

- **Routing:** look up the mapping; cache aggressively.
- **Strengths:** **maximum flexibility**. You can move a heavy tenant to its own shard, rebalance at any granularity, and handle non-uniform workloads.
- **Weaknesses:** the lookup service is a **dependency and potential bottleneck**; must be highly available and consistent-enough. Stale routing causes misrouted requests.
- **Hotspot risk:** **you can engineer around it** — that's the point.
- **Rebalancing:** easiest of the three; update the map and migrate the data.
- **What interviewers want to hear:** "When tenants are wildly uneven or I need fine-grained control — e.g., SaaS with power users — directory-based sharding lets me isolate heavy tenants. The trade-off is the routing layer, which I'd make a replicated, cached service."

---

## 4. Must-know concepts

**Shard key** — the field (or tuple) used to decide which shard owns a record. Everything flows from this choice.

**Cardinality** — number of distinct values of the shard key. Low cardinality = few possible buckets = can't scale out. Sharding by `country_code` (~200 values) caps you at 200 shards and guarantees skew.

**Uniformity vs locality** — usually in tension:

- **Uniformity** (even distribution) favors hash.
- **Locality** (related data on the same shard, cheap scans) favors range or directory.
- Pick based on which the workload needs more.

**Hot key / hot partition** — a single key or partition that receives disproportionate traffic. Pure hash distribution does not save you from a single celebrity key.

**Skew** — uneven distribution of data _or_ traffic across shards. Data skew = one shard is bigger. Traffic skew = one shard is busier. They often don't coincide.

**Consistent hashing (interview depth)**

Nodes and keys are both hashed onto a ring. Each key is owned by the next node clockwise. Adding/removing a node only moves keys in that node's arc — roughly `1/N` of data, not all of it.

```
               0°
        ┌──────N1──────┐
        │              │
      N4│              │N2
        │              │
        └──────N3──────┘
              180°

  key→hashes to 85° → owned by N2 (next clockwise)
```

**Virtual nodes (vnodes):** each physical node is placed on the ring at many points (e.g., 100–200). This smooths distribution and makes adding heterogeneous-sized nodes easy. Always mention vnodes — bare consistent hashing has measurable skew.

**Routing metadata** — the mapping of `key → shard`. Can be:

- Computed (pure hash) — no metadata.
- Small table (range partitioning) — replicate and cache.
- Full lookup service (directory) — needs its own HA design.

Clients either know the mapping (smart client) or hit a router/proxy (smart router). Mention both and the trade-off: smart clients are fast but harder to upgrade; routers add a hop but centralize changes.

**Resharding / rebalancing** — see §7.

**Scatter-gather queries** — a query that can't be answered by one shard, so the router fans out to many and merges. Cost grows with shard count; tail latency hurts. **Latency = slowest shard + merge overhead.**

**Tenant isolation / noisy neighbors** — in shared infrastructure, one heavy tenant can starve others on the same shard. Mitigations: per-tenant rate limits, quota enforcement, physical isolation for biggest tenants.

---

## 5. Choosing the shard key

**Properties of a good shard key:**

1. **High cardinality** — enough distinct values to spread across current + future shards.
2. **Even distribution** — no single value dominates.
3. **Aligned with the dominant access pattern** — most queries should hit one shard.
4. **Stable** — doesn't change after creation (changing a shard key = moving the record).
5. **Available at query time** — if reads don't have the shard key, every read is scatter-gather.

**`user_id` is not automatically the right answer.**

It passes cardinality and distribution. But it's wrong if:

- The dominant query is "get all posts in city X" (now every read is scatter-gather).
- One user is 1000x the others (Justin Bieber problem).
- The workload is multi-tenant and "tenant" ≠ "user."

Interviewers watch for candidates who reach for `user_id` without asking about access patterns. **Always ask: "What are the dominant queries? Read:write ratio? Largest-user skew?"**

**Monotonic keys and write hotspots.**

Sharding by timestamp, auto-increment ID, or UUIDv1 on a range scheme means **all writes hit the last shard**. Fixes:

- Hash-prefix the key: `hash(user_id) || timestamp` spreads writes but preserves per-user ordering.
- Use a hash scheme instead of range.
- Bucket by time (`day_bucket`) + another dimension.

**Composite keys.**

Combine two fields to get both distribution and locality:

```
  Shard key:  (user_id, timestamp)
  Partition by:  hash(user_id)
  Sort within shard by:  timestamp

  → Writes spread across shards by user
  → "last 100 events for user X" is a cheap local scan
```

This is the Cassandra/DynamoDB pattern (partition key + sort key) and is a strong answer in many interviews.

**Tenant-based sharding.**

For B2B SaaS: shard by `tenant_id`. Pros: strong isolation, simple per-tenant operations (backup, delete, migrate). Cons: **tenant size variance is enormous** — handle with directory-based routing and per-tenant shards for the heaviest.

**Geo-based sharding.**

Shard by region for latency and data-residency (GDPR). Watch for: users who travel, cross-region queries, and the "one region has 10x the traffic" problem. Geo sharding is often **the outer layer**, with hash/range inside each region.

**Query-pattern-driven shard choice.**

```
Dominant query                  → Shard key choice
------------------------------    -----------------------
"get user X's data"             → hash(user_id)
"get all orders in range [t1,t2]" → range(time) — but watch hotspots
"get tenant X's everything"     → tenant_id (directory if sizes vary)
"nearby items"                  → geo key (geohash / S2)
"global top K"                  → scatter-gather; consider precomputed aggregate
```

---

## 6. Hotspots and skew

**Celebrity problem** — one key gets vastly more traffic than the mean. Hash distribution doesn't help because _all requests go to that one key_.

Mitigations:

- **Read caching** — CDN / edge cache / in-memory cache in front of the shard. Usually the first and best answer for read hotspots.
- **Read replicas for the hot shard** — scale reads horizontally for that specific shard.
- **Write fan-out with salted keys** — for writes to a hot key, split into N sub-keys (`celebrity:0..N-1`), write to a random one, read by scatter-gather across N. Trades read complexity for write capacity.
- **Write batching / coalescing** — aggregate counter updates; flush periodically.

**Heavy tenants** — one customer is 100x bigger than the next.

- Directory-based routing to dedicated shards.
- Split the tenant internally (sub-shard by `user_id` within the tenant).
- Per-tenant rate limits.

**Temporal hotspots** — writes concentrate on the current time range.

- Hash-prefix the shard key.
- Pre-split ranges ahead of time.
- Rolling time buckets with a second dimension.

**Read vs write hotspots are different problems.**

```
Read hotspot                  Write hotspot
------------                  -------------
cache                         salting / bucketing
read replicas                 batching / coalescing
CDN                           switch to hash scheme
                              queue + async write
```

Interviewers love to hear you distinguish these.

**Other mitigations worth naming:**

- **Queue smoothing** — buffer bursty writes in a queue to flatten the load curve into the shard.
- **Admission control / rate limiting** — protect shards from noisy neighbors with per-tenant quotas.
- **Tenant splitting** — physically move the biggest tenant to its own shard.

---

## 7. Resharding and rebalancing

**When you reshard:**

- Data volume exceeds shard capacity.
- Write throughput exceeds shard capacity.
- Persistent hot partition that can't be mitigated in place.
- Adding or removing hardware.

**Moving data ≠ moving traffic.**

You can change the routing _first_ (send new writes to new shards) while backfilling old data asynchronously. This decouples the high-throughput traffic switch from the slow data copy.

**Online migration, high level (this is the template answer):**

```
1. Dual-write:  write to both old and new shards for a key range.
2. Backfill:    copy historical data from old to new in the background.
3. Verify:      checksums / shadow reads to confirm parity.
4. Cutover:     flip reads to new shards (behind a feature flag).
5. Decommission: stop writes to old, after a safety window.
```

Say each of these steps out loud in an interview. Staff-level candidates get credit for naming **verify** and **decommission** — juniors skip them.

**Metadata updates** — the routing table changes. Must be:

- Versioned (so stale clients can be rejected or forwarded).
- Propagated atomically (all routers see the same view at cutover, or a brief inconsistency is tolerated with retries).
- Backed by a consistent store (etcd, ZooKeeper, Spanner, etc.) if exact consistency matters.

**Failure handling during migration.**

- Dual-write failures → use a durable log (outbox / CDC) so a failed write to one side can be retried without losing the other.
- Partial cutover → feature-flag per shard or per key range so you can roll back granularly.
- Data divergence → shadow reads and periodic checksums to detect drift early.

**What a staff-level answer should mention:**

- Use consistent hashing / vnodes so rebalancing moves ~`1/N` of data, not everything.
- Migrate at the range or vnode granularity, not one row at a time.
- Have a rollback path at every step.
- Expect 2x peak capacity during migration (dual-write + backfill IO).
- Monitor tail latency, not just averages, during cutover.

---

## 8. Cross-shard pain

After you shard, previously cheap operations get expensive.

**Fanout / scatter-gather.**

```
                  ┌─ Shard 1 ─┐
  Query  ──router─┼─ Shard 2 ─┼──merge──→ response
                  ├─ Shard 3 ─┤
                  └─ Shard 4 ─┘

  latency = max(per-shard latency) + merge
  error rate ≈ 1 − (1 − p)^N
```

Adding shards _increases_ tail latency and failure probability. Mitigations: **hedged requests**, **partial results with a deadline**, **backup requests**, or **precomputed aggregates** so you don't fanout at read time.

**Cross-shard joins.**

The classic "join users with orders" fails when they're on different shards. Options:

- **Denormalize** — duplicate the join target into the querying shard.
- **Co-locate** — pick a shard key that puts related data on the same shard (e.g., both users and their orders by `user_id`).
- **Application-side join** — fetch from both, merge in app code (acceptable for small result sets only).
- **Materialized views / secondary indexes** — shard-local indexes kept in sync via CDC.

**Aggregations.**

- Precompute when possible (counters, rollups updated on write).
- Use approximation (HyperLogLog for distinct counts, t-digest for quantiles) when exactness isn't required.
- Run async analytical queries against a replicated OLAP store, not the live shards.

**Pagination after sharding.**

Offset pagination (`LIMIT 100 OFFSET 500`) across shards forces each shard to return `offset + limit` rows for the merge — cost grows linearly with page depth. Use **cursor-based pagination** (`WHERE ts < last_ts ORDER BY ts DESC LIMIT N`) per shard, merge at the router. This is a classic staff-interview follow-up, so have the answer ready.

**Transactions.**

Single-shard transactions are fine. Cross-shard transactions require 2PC or a similar protocol, which is expensive and failure-prone. The strong staff answer: **design the shard key so that transactional units fit within one shard**. If you can't, consider Sagas (compensating transactions) for cross-shard workflows.

---

## 9. Interview reasoning patterns

### "How do you choose a shard key?"

1. Ask about dominant queries and read:write ratio.
2. Ask about tenant/user size distribution.
3. Ask about range/ordering requirements.
4. Propose a key; explicitly check: cardinality, distribution, query alignment, stability.
5. Acknowledge what becomes expensive (the queries that don't align).

### "When is hash sharding a bad fit?"

- Range scans and ordered pagination dominate.
- You need locality between related items (e.g., all of a user's events together, ordered).
- Low-cardinality shard key.
- Single hot key (hash doesn't help here — a single key always lands on one shard).

### "When is range sharding a bad fit?"

- Monotonic keys (timestamps, autoincrement) → write hotspots.
- Uneven key distribution causing skew.
- No natural ordering in the workload.

### "How do you handle a hot partition?"

1. Identify whether it's read or write hotspot.
2. Reads → cache, then read replicas.
3. Writes → salting / sub-bucketing, batching, or switch to a hash scheme.
4. If persistent, isolate the hot entity on its own shard (directory routing).
5. Mention observability: you need per-shard metrics to detect this.

### "How do you reshard without downtime?"

Template: dual-write → backfill → verify → cutover → decommission, behind a feature flag, with rollback at every step. Use consistent hashing / vnodes to minimize moved data.

### "What breaks after sharding that worked before?"

- Cross-shard joins.
- Cross-shard transactions.
- Global aggregates and ordering (top-N, global count).
- Pagination with large offsets.
- Secondary-index queries that don't use the shard key → scatter-gather.
- Unique constraints that span the partition key.

---

## 10. Common mistakes candidates make

- **"Shard by user_id"** with no discussion of access patterns, tenant skew, or hot keys.
- **Ignoring hotspots** — assuming hash distribution = uniform load.
- **Confusing partitioning with replication** — e.g., "I'll shard for high availability."
- **Hand-waving rebalancing** — "we'll just add shards." Interviewers ask _how_.
- **Ignoring routing metadata** — no mention of how clients know which shard owns a key.
- **Assuming consistent hashing solves everything** — it handles node-count changes, not hot keys or skew.
- **Forgetting cross-shard queries** — choosing a shard key that makes the dominant query a scatter-gather.
- **Not mentioning vnodes** when they come up with consistent hashing.
- **Picking a low-cardinality shard key** (country, status, category).
- **Ignoring that a shard key change = moving the row** — treating shard key as mutable.
- **No observability plan** — can't detect hotspots without per-shard metrics.

---

## 11. Final cheat sheet

### Comparison table

|Dimension|Hash|Range|Directory|
|---|---|---|---|
|Distribution|Near-uniform (with good key)|Depends on key — skew-prone|Fully controllable|
|Routing|Compute `hash(key)`|Small range table|Explicit lookup service|
|Range scans|Bad (scatter-gather)|Excellent|Depends on design|
|Point lookups|Excellent|Good|Good (plus lookup hop)|
|Hotspot risk|Hot single key still hurts|Monotonic keys cause write hotspots|Engineered around; best of three|
|Rebalancing|Easy with consistent hashing|Split / merge ranges|Easiest (just update map)|
|Metadata overhead|None|Small|Significant (HA routing service)|
|Best for|Write-heavy point lookups|Time-series, ordered scans|Multi-tenant, uneven workloads|
|Worst for|Ordered queries, pagination|Monotonic keys, skewed data|Simple uniform workloads (overkill)|

### Shard-key selection checklist

- [ ] High cardinality (>> shard count, with room to grow)?
- [ ] Even distribution of values?
- [ ] Dominant query includes the shard key?
- [ ] Stable (doesn't mutate)?
- [ ] Available at read time?
- [ ] Transactional units fit within one shard?
- [ ] No single value is a significant fraction of traffic?
- [ ] Does it align with tenant isolation boundaries?

If most are "yes," defend it. If not, name the trade-off explicitly.

### Decision framework for interviews

```
1. Clarify workload
   - dominant queries? read:write ratio? ordering needs?
   - tenant / user distribution? largest entity?
   - growth trajectory?

2. Pick scheme
   - multi-tenant with heavy skew  →  directory
   - ordered scans / time-series   →  range (beware monotonic writes)
   - uniform point lookups         →  hash + consistent hashing + vnodes
   - geo or residency driven       →  geo outer + hash/range inner

3. Stress-test the choice
   - hot keys? hot partitions? scatter-gather queries?
   - what breaks at 10x? at 100x?

4. Mitigate explicitly
   - caching, replicas, salting, directory for heavy tenants
   - precomputed aggregates for global queries

5. Describe rebalancing
   - consistent hashing / vnodes
   - dual-write → backfill → verify → cutover → decommission
```

### 10 likely interview questions with strong short answers

1. **"How would you shard this?"** "Depends on the workload. Dominant query, read:write ratio, tenant-size distribution, and ordering needs drive the choice. Let me ask about those first, then pick."
    
2. **"Why not just `hash(user_id)`?"** "It works for uniform point lookups, but if the workload has celebrity users, ordered queries, or tenant isolation needs, it breaks. I'd only pick it after confirming those aren't in play."
    
3. **"How does consistent hashing help?"** "Adding or removing a node only moves about `1/N` of the keys instead of rehashing everything. Pair it with virtual nodes so distribution is smooth and nodes can be heterogeneously sized."
    
4. **"You have a hot partition. What do you do?"** "First, separate read vs write. Reads → cache, then per-shard replicas. Writes → salt the key into sub-buckets, or batch writes. If it's a persistent heavy tenant, isolate it on its own shard via directory routing."
    
5. **"How do you reshard without downtime?"** "Dual-write new keys to old and new shards, backfill history asynchronously, verify with checksums and shadow reads, cut over reads behind a feature flag, then decommission the old shard after a safety window."
    
6. **"What about cross-shard joins?"** "Design them out. Either co-locate related data via a composite shard key, denormalize the join target, or maintain a shard-local secondary index kept in sync via CDC. Real cross-shard joins at request time don't scale."
    
7. **"Monotonic writes on a range-sharded table?"** "Classic tail-shard hotspot. Fix by hash-prefixing the key to break temporal locality, pre-splitting ranges, or switching to a hash scheme if ordering isn't essential."
    
8. **"Partitioning vs replication — same thing?"** "No. Partitioning splits the dataset to scale capacity. Replication duplicates it for read throughput and availability. Real systems do both: shards with replicas. Conflating them is a common design error."
    
9. **"What about cross-shard transactions?"** "Avoid if possible — pick a shard key so transactional units fit on one shard. When you can't, use Sagas with compensating actions. 2PC exists but it's a tail-latency and availability tax."
    
10. **"Scatter-gather is fine, right?"** "Fine until it isn't. Tail latency is `max` across shards, error rate compounds with shard count. Mitigate with hedged requests, deadlines, and precomputed aggregates so common queries don't fanout at read time."