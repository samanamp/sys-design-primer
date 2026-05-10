---
title: "Time-series / observability database"
description: "Time-series / observability database"
---

“Design the storage and query layer for a metrics system: 10M points/sec ingest, 100M active series, queries from ‘last 5 minutes one series’ to ‘last 30 days, sum 10K series.’”

---

## 1. Reframing

Three things must be established before any boxes-and-arrows begin, because if they are not established the design is wrong from sentence one.

**First: cardinality is the existential threat, not throughput.** 10M points/sec is large but tractable — roughly steady-state of a large Prometheus fleet, parallelizes cleanly. 100M *active series* is the number that determines whether the inverted index fits in memory, whether compaction keeps up, and whether one bad customer destroys the system. A single deploy of `http_requests{user_id="..."}` can mint 50M new series in a day. Any design that does not lead with cardinality detection, enforcement, and customer-contract framing has already failed. **[STAFF SIGNAL: cardinality-as-existential]**

**Second: query workload diversity is not a tuning concern, it is an architectural fork.** "Last 5 min, one series" wants p99 < 10 ms, served from memtable, single shard, single posting list. "Last 30 days, sum across 10K series" wants throughput, scans tens of billions of points pre-rollup, fans out across shards, spans cold tiers. Different systems wearing the same coat. Storage layout, index path, and execution model must accommodate both. **[STAFF SIGNAL: query-diversity reframing]**

**Third: time-series compression is not generic compression.** gzip on float64 timestamps + values leaves 5–10× on the table because it does not exploit fixed-interval timestamps and slowly-varying values. Delta-of-delta on timestamps and Gorilla XOR on values reach ~1.5 bytes/point on real metrics versus 16 bytes raw. This 10× ratio drives storage cost, ingest IOPS, page cache footprint, and read throughput simultaneously. **[STAFF SIGNAL: compression specificity]**

These three are the spine. Everything below is consequences.

## 2. Scoping

**In-scope:**
- Numeric metrics: counters, gauges, histograms (pre-bucketed and t-digest), summaries.
- PromQL-like query semantics: instant queries (point-in-time) and range queries (matrix over `[start, end, step]`), with `sum/avg/min/max/quantile/rate` aggregations and label-set selectors with regex.
- Multi-tenant SaaS posture: hundreds to thousands of tenants on shared infrastructure, with the largest tenants on dedicated shards.
- Retention: raw 15 days, 1-min rollup 90 days, 1-hour rollup 13 months, 1-day rollup 5 years.

**Explicitly deferred:**
- Logs, traces, profiles, events. Different data models, different storage, different query languages. They share *identity* (resource labels, trace IDs) with metrics through a common metadata service.
- Exemplars (point-to-trace links): noted as a 30-byte sidecar per point, not designed.
- Streaming alerting evaluation: a separate read-side consumer of the ingest stream.
- Recording rules: implemented as a query-driven write-back, designed last.

**Assumptions stated:**
- Server-side timestamping by default; client timestamps accepted only within a 15-min grace window.
- Dimensionality target: median ~15 labels per series, p99 ~30 labels, mean label value length ~25 bytes.
- Customer-facing limit: 1M active series per tenant on shared shards; larger requires a dedicated cell. **Cardinality is a billing axis, not a hidden internal limit.** **[STAFF SIGNAL: scope negotiation]**

## 3. Capacity math

```
Ingest:        10,000,000 pts/sec  -> 864B pts/day
Raw point:     16 B (8B ts + 8B val)
Raw/day:       ~14 TB/day
Gorilla:       ~1.5 B/point on real metrics  (8–11x)
Compressed/day:~1.3 TB/day
+2x replica:   ~2.6 TB/day
+ index + WAL: ~3.0 TB/day hot
30d hot raw:   ~90 TB across fleet
```

```
Series metadata:   100M series x ~15 labels x ~30B avg = ~45 GB labels
Inverted index:    posting lists; per (name,value) pair ~ 4–8B per series ID,
                   compressed via roaring bitmaps to ~1–2 bits amortized for
                   dense posts. Total in-mem index target: 30–50 GB / shard
                   group, partitioned across ~32 shards.
```

| Tier      | Resolution | Retention | Approx footprint | $/GB-mo | Monthly $   |
|-----------|------------|-----------|------------------|---------|-------------|
| Hot SSD   | raw 1–10s  | 15 d      | ~45 TB           | $0.10   | ~$4,500     |
| Warm SSD  | 1-min      | 90 d      | ~90 TB           | $0.05   | ~$4,500     |
| Cold S3   | 1-hr       | 13 mo     | ~120 TB          | $0.02   | ~$2,400     |
| Cold S3   | 1-day      | 5 yr      | ~30 TB           | $0.02   | ~$600       |
| Replicas  | 2x hot/warm| —         | ~135 TB          | mixed   | ~$9,000     |

```
Query scan math:
 - 30-day query, 10K series, raw 1s:  10,000 * 30 * 86,400  = 26 B points
   @ 50 ns/point decompress           = 1,300 sec = 22 min   <-- IMPOSSIBLE
 - Same query at 1-min rollup:                    = 432 M points
   @ 50 ns/point                                  = 22 sec
 - Same query at 1-hour rollup:                   = 7.2 M points
                                                  = 360 ms   <-- target
```

The 30-day query is not solvable on raw at any cluster size; rollups are not an optimization, they are the only way the query terminates. **[STAFF SIGNAL: capacity math]**

## 4. High-level architecture

```
                 ┌──────────────────────────────────────────────┐
                 │              Tenants / Agents                 │
                 │   (Prometheus remote_write, OTLP, statsd)     │
                 └───────────────────┬──────────────────────────┘
                                     │
                            ┌────────▼────────┐
                            │   Distributor    │  per-tenant rate limit
                            │  (stateless)     │  cardinality pre-check
                            │  series-id hash  │  shard routing
                            └────────┬─────────┘
                                     │ (consistent hash on series_id)
            ┌───────────┬────────────┼────────────┬────────────┐
            ▼           ▼            ▼            ▼            ▼
        ┌──────┐   ┌──────┐     ┌──────┐     ┌──────┐     ┌──────┐
        │Ing-0 │   │Ing-1 │ ... │Ing-N │     │      │     │      │
        │ WAL  │   │ WAL  │     │ WAL  │     │ ...  │     │ ...  │
        │ Mem- │   │ Mem- │     │ Mem- │     │      │     │      │
        │ table│   │ table│     │ table│     │      │     │      │
        └──┬───┘   └──┬───┘     └──┬───┘                          
           │ flush   │             │  every ~2h or size threshold
           ▼          ▼             ▼
        ┌──────────────────────────────────┐
        │       Object Storage (S3)         │
        │   sealed blocks, immutable        │
        │   per-shard prefix, per-2h chunks │
        └──────────┬───────────────────────┘
                   │ compaction (LSM-style)
                   ▼
        ┌──────────────────────────────────┐
        │   Compactor (separate fleet)      │
        │   merges 2h -> 8h -> 24h blocks   │
        │   builds rollups: 1m, 1h, 1d      │
        │   tombstones for deletes          │
        └──────────────────────────────────┘

Read path (independent fleet):

    Query  ──► Query Frontend ──► Query Planner ──► Querier(s) ──► Storage
              (caching, splitting) (cost estimate)   (per-shard)    (mem+S3)
                    │                  │                │
                    ▼                  ▼                ▼
              result cache       admission ctl    inverted index +
                                                  block decompression
```

A **read/write split LSM tree with separate compaction and query fleets, fronted by a stateless distributor and a planning query frontend.** The split exists because ingest, compaction, and query have different scaling axes and failure tolerances. Distributor: stateless, scales with QPS. Ingester: stateful (WAL + memtable), scales with active series; consistent-hash on `series_id = hash(metric, sorted_labels)` so a series lives on a deterministic pair (RF=2). Compactor: scales with bytes/day flushed; produces rollups during compaction, sharing I/O. Querier: stateless; fetches blocks from S3 (local SSD cache) and queries ingesters for the unflushed tail. Each shard has its own inverted index; cross-shard queries fan out and merge.

**Rejected: ClickHouse/general OLAP.** Handles cardinality fine, query engine is excellent, but column compression is generic LZ4/ZSTD; we lose 4–5× vs Gorilla on float metrics. At 1.3 PB/year hot+warm, that's ~$200K/yr storage plus the cache-footprint multiplier on read throughput. Generic OLAP also lacks the per-series memtable model that makes point queries microsecond-cheap. **[STAFF SIGNAL: rejected alternative]**

**Rejected: in-memory-only index fleet-wide.** Index is ~200 GB total — fits in RAM today, but mmap'd disk for older blocks prevents a series-count spike from OOM-ing ingesters.

## 5. Cardinality defense

This is the most important section. Cardinality is *the* failure mode.

**The threat model.** Customer ships `http_requests{user_id, request_id, trace_id}`. Each request mints a series. In 10 min, 5M series. Cascade: (1) memtables explode; (2) posting-list table grows, new-series insert becomes a write hotspot; (3) storage/day inflates, per-block header dominates; (4) `sum(http_requests)` scans 5M posting lists instead of 50K, latency goes ms→minutes; (5) compaction never finishes; lag grows monotonically.

**Detection points** (annotated on the ingest path):

```
client ──► Distributor ──► Ingester ──► Memtable
            │                │
            │                ├── (D3) per-shard active-series counter (HLL)
            │                ├── (D4) new-series-per-minute rate
            │                └── (D5) per-metric series count (HLL)
            │
            ├── (D1) per-tenant active-series HLL (HyperLogLog, ~16 KB/tenant)
            ├── (D2) per-tenant series-creation-rate counter
            └── (D6) regex-fanout pre-check for known abusive patterns
```

- **D1** is canonical, kept in the distributor with HyperLogLog (~16 KB/tenant for ~1% error). 1000 tenants × 16 KB = 16 MB — trivial.
- **D2** triggers burst alerts: "tenant X created 100K new series in 60s" is almost always a deploy bug. Page the tenant, not us.
- **D3–D5** are local for shard-level enforcement and forensics.

**Enforcement mechanisms.**

| Mechanism                | Trigger                                  | Action                                                            |
|--------------------------|------------------------------------------|-------------------------------------------------------------------|
| Soft warn                | 80% of tenant series limit               | Email + dashboard banner; metric `cardinality_warn` exposed       |
| Hard reject              | 100% of tenant series limit              | Reject **only new** series; existing series continue to ingest    |
| Per-metric cap           | Single metric > 10% of tenant limit      | Reject; require explicit override                                 |
| Burst rejection          | >50K new series/min for >2 min           | Reject new series for that metric; 5-min cooldown                 |
| Regex query refusal      | Predicate matches >100K series           | Query rejected with `series_match_too_broad`                      |

**Why reject over sample.** Sampling silently distorts data — a sampled `count` is wrong, an averaged `avg` of 1% of series misrepresents the population. Rejection is loud, attributed (the error names the offending metric and limit), and recoverable. Sampling looks like "the system is fine but the dashboards are wrong" — the worst possible failure for a system whose entire purpose is fidelity. **[STAFF SIGNAL: invariant-based thinking]** — invariant: rejection is observable, sampling is not.

**Cleanup.** When a customer accidentally creates 10M bad series, deletion is *not* a `DELETE` — it is tombstoning during compaction. Operator runs `delete_series --tenant X --matcher '{user_id=~".+"}'`; tombstone records written per-shard; next compaction omits matching series from rebuilt blocks and removes posting list entries. End-to-end: 2–6 hours depending on compaction lag. The API exposes a job ID and progress; the customer must understand this.

**Customer contract.** Cardinality is published in the SaaS plan: "Tier 1 = 1M series included, $X per additional 100K." This makes the limit a billing axis rather than a hidden cliff, aligns customer incentives with system health, and removes the "why is my data being rejected" support load. **[STAFF SIGNAL: cardinality-as-existential]**

## 6. Compression and storage layout

**Per-series block, on disk:**

```
┌──────────────────────────────────────────────────────────────────┐
│  Block header                                                     │
│  ├── series_id (8B)                                               │
│  ├── min_ts, max_ts (16B)                                         │
│  ├── point_count (4B)                                             │
│  ├── timestamp_block_offset, len                                  │
│  ├── value_block_offset, len                                      │
│  └── checksum (CRC32C)                                            │
├──────────────────────────────────────────────────────────────────┤
│  Timestamp block (delta-of-delta encoded)                         │
│                                                                   │
│   t0                  : 64-bit literal                            │
│   t1 - t0 (delta)     : 14-bit varint                             │
│   for k>=2:                                                       │
│     dod = (tk - tk-1) - (tk-1 - tk-2)                             │
│     if dod == 0 : 1 bit  '0'           ◄── steady interval        │
│     elif |dod|<=63: '10' + 7-bit signed                           │
│     elif |dod|<=255:'110'+ 9-bit signed                           │
│     elif |dod|<=2047:'1110'+12-bit signed                         │
│     else:           '1111'+ 32-bit signed                         │
│                                                                   │
│   Typical 1s scrape: 99% of points encode in 1 bit.               │
├──────────────────────────────────────────────────────────────────┤
│  Value block (Gorilla XOR-encoded float64)                        │
│                                                                   │
│   v0          : 64-bit literal                                    │
│   x = vk XOR vk-1                                                 │
│   if x == 0   : 1 bit '0'        ◄── repeated value               │
│   else:                                                           │
│     '1' + control-bit:                                            │
│       same leading/trailing zero range as prev: '0' + body bits   │
│       new range: '1' + 5b leading + 6b body_len + body bits       │
│                                                                   │
│   Typical gauge: 1.2–1.6 B/point.                                 │
└──────────────────────────────────────────────────────────────────┘
```

**Why delta-of-delta works.** Scrape intervals are nearly constant (1s, 10s, 60s). The first delta is the interval; the delta-of-delta is zero except at boundaries (process restart, scrape skip). Encoding zero in 1 bit means timestamps cost ~0.1–0.2 B/point amortized. **[STAFF SIGNAL: compression specificity]**

**Why Gorilla XOR works.** Most metrics are continuous: CPU drifts, counter rates are smooth, even gauges share most mantissa bits with neighbors. The XOR is dominated by leading/trailing zeros — encode the meaningful middle bits, run-length the zeros. Counters specifically emit monotonically increasing values, where high-order bits change rarely.

**Block-size tradeoff.**

| Block duration | Bytes/point | Point-query cost          | Index entries  |
|----------------|-------------|---------------------------|----------------|
| 30 min         | ~2.5 B      | <1 KB read               | 48/series/day  |
| **2 hr**       | **~1.5 B**  | **~3 KB read**           | **12/series/day** |
| 8 hr           | ~1.3 B      | ~12 KB read              | 3/series/day   |

2-hour blocks are the production sweet spot: Gorilla amortizes its preamble, index entries are manageable, a recent-time point query reads at most ~3 KB per series.

**Column layout.** Timestamps and values are separate contiguous compressed streams within each block. Aggregation queries computing `rate(counter)` need only values; tag-pivoting can skip values entirely. Column separation also enables SIMD decompression: AVX-512 decodes 8 timestamps/cycle once the bitstream is unpacked.

**Decompression cost.** ~50 ns/point on modern x86 with vectorized decoders. A single core decodes 20M points/sec sustained. A 1B-point query needs 50 core-seconds. **Rollups are mathematically necessary, not optional.** **[STAFF SIGNAL: capacity math]**

## 7. Inverted index and tag queries

**Posting list structure.**

```
Per shard, in memory + mmap:

  Symbol table (label name/value -> uint32 ID)
  ┌──────────────┬────────────┐
  │ "region"     │   17       │
  │ "us-east-1"  │   42       │
  │ "service"    │   18       │
  │ "checkout"   │   91       │
  └──────────────┴────────────┘

  Posting lists (key = (label_name_id, label_value_id))
  ┌─────────────────────────┬───────────────────────────────────┐
  │ (17, 42)  region=us-... │ Roaring Bitmap [s_19, s_20, ...]  │
  │ (18, 91)  service=ch... │ Roaring Bitmap [s_20, s_44, ...]  │
  └─────────────────────────┴───────────────────────────────────┘

Query:  region="us-east-1" AND service="checkout"
        = AND(bitmap1, bitmap2)   ◄── SIMD'd, ~ns/64 series
```

**Roaring bitmaps** are the standard. They split the 32-bit series-ID space into 2^16 chunks; each chunk is a bitmap, a sorted array, or a run-length list, picked dynamically. AND/OR/NOT are SIMD-friendly. For dense posting lists (millions of series), they reach ~1–2 bits/series amortized; for sparse lists, ~16 bits/series. **[STAFF SIGNAL: index discipline]**

**Memory vs. disk.** Recent (~24h) postings live in RAM. Older postings live on mmap'd files; kernel page cache handles working set. Cross-shard fanout means each shard answers locally; the querier merges by `metric, labels` (IDs are shard-local).

**Intersection cost.** For `metric=foo{a="x", b="y"}`: lookup three posting lists, AND the bitmaps. O(min posting × log fanout). Even at 1M series in the smallest list, sub-millisecond.

**The high-fanout regex problem.** `metric=~".*errors.*"` expands to every metric matching the regex — possibly thousands of separate posting lists. Pre-filter:
- Cost-estimate: `expanded_terms × avg_posting_size × 2 bits`. Reject if > threshold (e.g., 1B series-id-bits).
- Force prefix anchoring (`metric=~"http_.*"`) when fanout is high; reject pure unanchored regex on `__name__` for shared-shard tenants.
- The error names the offending matcher.

**Index update on new series.** Symbol-table lookup/insert (locked B-tree per shard); insert series_id into N posting lists (N≈15); append symbol-table delta to WAL. At burst series creation, this is the actual bottleneck — not the data path. Cardinality enforcement gates upstream.

**Per-shard, not global.** A global secondary index would be a SPOF and a write hotspot. Per-shard means cross-shard queries fan out, but for almost all queries this is parallelism, not overhead.

## 8. Downsampling and rollups

**Tier hierarchy:**

```
                    Resolution      Retention    Storage    Use
                    ───────────     ─────────    ───────    ───
   Raw (scraped)    1s / 10s / 60s  15 d          ~45 TB    debugging,
                                                            recent dashboards
        │ aggregate during compaction
        ▼
   1-min rollup     60s             90 d          ~7 TB     7-day dashboards,
                                                            alerting
        │
        ▼
   1-hour rollup    3600s           13 mo         ~6 TB     month/quarter
                                                            comparisons
        │
        ▼
   1-day rollup     86400s          5 yr          ~1 TB     YoY trends, capacity
```

**What is pre-computed per (series, window):** `sum`, `count`, `min`, `max`, `last`. From these, `avg` is recoverable (sum/count). Six 8-byte values per window per series.

**The quantile problem.** `quantile(0.99, http_request_latency_seconds)` cannot be computed from per-window quantiles because quantiles do not compose: `quantile_p99(merge(W1, W2)) ≠ f(p99(W1), p99(W2))` for any aggregation `f`. This is a fundamental limitation, not a tuning issue.

**Solution: t-digest.** Each window stores a t-digest (Dunning) — a compressed sketch of the distribution, ~100 centroids = ~1.6 KB per window per series. t-digests merge associatively and exactly: `merge(td1, td2)` is a valid t-digest of the union. Quantiles read with ~1% error at p99, ~0.1% at p50. Alternative: HDR histograms (fixed-bucket, larger). t-digest wins for skewed latency; HDR wins where all buckets matter for SLOs. We use t-digest natively, HDR-style fixed-bucket histograms when producers (Prometheus) emit them. **[STAFF SIGNAL: rollup-and-quantile precision]**

**Storage cost of rollups.** ~20–30% of raw at the same retention. We do *not* keep rollups at raw retention — we age out raw at 15 days, keep 1-min for 90 days, etc. Net rollup storage is small relative to raw because of the cascading retention.

**Query-time tier selection** (planner logic):

```
selected_tier = argmin(tier) such that
    tier.resolution * tier.points_per_series_in_range <= QUERY_BUDGET
    AND tier.resolution <= user_step

QUERY_BUDGET = 10M points scanned for interactive queries
             = 100M for "expand details" queries (slower SLO)
```

A 30-day query with `step=1h` automatically selects the 1-hour rollup. A 5-min query with `step=15s` reads raw. A 7-day query with `step=1m` reads the 1-min rollup. This is mechanical; the planner does not guess.

**The recent-data hole.** Rollups lag ingestion. The 1-min rollup for the current minute does not exist until the minute closes plus ~30s propagation. Queries for `now()` fall through to raw automatically; `now() - 1h to now()` reads raw at the recent edge and rollup for the rest.

## 9. Ingest pipeline

```
                          ┌─────────────────────┐
   write request  ───►   │  Distributor         │
                         │  - auth              │
                         │  - cardinality check │
                         │  - shard route       │
                         └──────┬──────────────┘
                                │ (replication factor 2)
                       ┌────────┴────────┐
                       ▼                 ▼
                 ┌──────────┐      ┌──────────┐
                 │Ingester A│      │Ingester B│
                 │          │      │          │
                 │  WAL  ◄──── append, fsync (group commit, 50 ms window)
                 │   │      │      │          │
                 │   ▼      │      │          │
                 │ Memtable │      │ Memtable │
                 │ (per-   │      │ (per-    │
                 │  series  │      │  series  │
                 │  buffer) │      │  buffer) │
                 │   │      │      │          │
                 └───┼──────┘      └──────────┘
                     │ flush every 2h or 1GB
                     ▼
                  ┌──────────────┐
                  │ Sealed block │ ──► S3
                  └──────────────┘
                     │
                     │ async
                     ▼
                  ┌──────────────┐
                  │  Compactor   │  merges 2h blocks into 8h, 24h
                  │              │  produces 1m, 1h, 1d rollups
                  │              │  applies tombstones
                  └──────────────┘
```

**WAL.** Append-only log per ingester, fsync'd in group-commit windows of ~50 ms (trades 50 ms durability latency for ~1000× IOPS reduction). Format: `[len][crc][series_ref][ts][val]`. `series_ref` is a shard-local 4-byte ID; symbol-table additions are themselves WAL records. Recovery: replay WAL on crash, rebuild memtable.

**Memtable.** Per-series ring buffer holding the last ~2 hours. ~1 KB/series average; 100M series across the fleet ≈ 100 GB (~32 ingesters × ~3 GB each). Flush triggers: 2-hour boundary, 1 GB sealed-block size, or operator command.

**Flush.** Encodes memtable into the on-disk block format, writes to local disk, uploads to S3, marks WAL segment for truncation. Flushed blocks are immutable.

**Compaction.** Separate fleet reads small recent blocks and produces: merged blocks at coarser time grain (2h→8h→24h), rollup blocks (1m, 1h, 1d) co-located with sources, and tombstone-applied output.

**Compaction throughput as a planning constraint.** At 10M pps × 1.5 B/point compressed, ingest is ~15 MB/s/shard; compaction must process this rate sustained, plus 2× LSM write amplification, plus rollup computation. A compactor falling behind = unbounded S3 cost growth + degrading query performance (more blocks = more index lookups + scattered reads). **Capacity planning is sized for compaction throughput, not ingest.** **[STAFF SIGNAL: compaction-as-bottleneck]**

## 10. Query planner

**The two query classes diverge at the planner.**

```
Query path A: point query   "rate(http_requests{pod='x'}[5m])"
─────────────────────────────────────────────────────────────────
  Frontend  ─►  Planner  ─►  cost = 1 series × 300s × 1B/pt = 300 B
                       │
                       ├─► route to single shard (series_id hash)
                       │
                       ▼
  Querier  ─►  Ingester (memtable) ─►  return 300 points
                  │
                  └─► block fetch only if memtable cold-start
  p99 target: < 10 ms

Query path B: aggregate     "sum by (region) (http_requests[30d])"
─────────────────────────────────────────────────────────────────
  Frontend  ─►  Planner  ─►  estimate: 10K series × 30d × 1m rollup
                       │            = 432 M points
                       │            > 10M budget? NO at 1h rollup (7M)
                       │            select 1-hour rollup
                       │
                       ├─► fan-out: 32 shards in parallel
                       ├─► per-shard: posting list intersect for region;
                       │   group-by-region partial sums
                       │
                       ▼
  Querier  ─► [shard 0] partial: {us-east: sum_0, eu: sum_0, ...}
              [shard 1] partial: {us-east: sum_1, eu: sum_1, ...}
              ...
                  │
                  ▼
            Merge: sum_total[region] = Σ partials
            p99 target: < 2 s
```

**Planner responsibilities:**

| Responsibility           | Mechanism                                                          |
|--------------------------|---------------------------------------------------------------------|
| Resolution selection     | argmin tier with `points <= budget` and `step <= user_step`        |
| Shard fan-out            | inverted-index pre-resolve to shard set; skip non-matching shards   |
| Predicate push-down      | label matchers run on each shard's index, not at the merge step    |
| Cost estimation          | bytes-to-scan estimate; reject above per-tenant budget              |
| Admission control        | per-tenant query queue; preempt low-priority below threshold        |
| Result caching           | hash(query, time-bucket-aligned) → result; 30s–5min TTL             |

**Cost estimation.** From the index, the planner knows how many series match. Multiply by step count, tier point density, ~16 B working memory per point. Reject queries exceeding the tenant's per-query budget (typical: 4 GB working set, 60s wall time). The error names the limit and suggests a coarser step or narrower matcher. **[STAFF SIGNAL: query planner discipline]**

**Result caching.** Dashboard queries are pathologically repetitive — the same `sum(rate(...))` runs every 30s. Time-aligned queries (start/end snapped to step boundaries) hit a result cache keyed on matcher + step + window. Often a 10× backend traffic reduction.

**Admission control.** Per-tenant concurrency limit and cost budget per minute. Heavy tenants hit budget and queue; small-query tenants are served from a separate priority lane. Global concurrency cap on the querier fleet prevents fleet-wide saturation.

## 11. Multi-tenancy and isolation

**Five isolation axes:**
1. **Ingest cardinality** — per-tenant active series, enforced at distributor (Section 5).
2. **Ingest rate** — per-tenant points/sec; token bucket; reject excess with retry-after.
3. **Query concurrency** — per-tenant max concurrent queries; queued beyond.
4. **Query cost** — per-tenant cost budget per minute (estimated bytes scanned).
5. **Storage** — per-tenant retention is a plan parameter; series count is billed.

**Shared vs. dedicated shards.** Most tenants share. Tenants exceeding 5M active series, OR with strict noisy-neighbor SLOs, get a dedicated cell — same software, separate storage. Sold at higher tiers. Migration is via a series-by-series double-write window. **[STAFF SIGNAL: multi-tenant isolation]**

**Noisy neighbor at query time.** Per-cost budgets aside, a heavy tenant on a shared shard saturates that shard's CPU. Mitigation: per-shard fair-share scheduler weighted by tenant tier. Persistent saturation triggers migration to dedicated.

**Per-tenant pipelines.** A tenant emitting malformed metrics (missing labels, NaNs, bad timestamps) gets its own ingest queue so parse errors don't block other tenants' writes through shared distributor goroutines. Bounded buffer per tenant; saturation throws back-pressure to that tenant only.

## 12. Hot/cold tiering

**Tier transitions.**

```
   t=0           t=24h        t=15d          t=90d            t=13mo
   ──────────────►──────────────►──────────────►──────────────►
   Memtable      Local SSD     S3 (raw)       S3 (1m rollup)   S3 (1h)
   (ingester)    (querier      delete after   delete after     ...
                  cache)       15d            90d
```

A block transitions when it crosses age boundaries:
- Days 0–1: served from memtable + recent-block local cache.
- Days 1–15: served from S3 with local SSD cache (~LRU, ~10% hot).
- Days 15+: raw is deleted; only rollups remain.

**Querying across tiers.** A 7-day query reads recent days from local cache, older from S3. The querier issues parallel range reads (HTTP/2 multiplex); typical S3 GET latency 30–80 ms first-byte, throughput limited by parallelism. With 32 parallel reads/shard, a 7-day query completes in ~1s of S3 latency.

**Snapshot-and-promote during transitions.** A block being moved is dual-located briefly; the block index records both, a flag flips when the new location is durable, the old is GC'd. Queries during transition read from either side. **[STAFF SIGNAL: tiering economics]**

**Cost.** SSD ~$0.10/GB-mo, S3 standard ~$0.023, S3 IA ~$0.0125. Warm SSD dominates monthly cost at our footprint. Aggressive rollup-and-delete keeps raw on SSD only 15 days.

## 13. Late-arriving data

**Watermark / grace window.** Accept points up to 15 minutes older than current memtable time; reject earlier as `out_of_window`. 15 min covers normal scrape-and-buffer delays and short partitions while bounding memtable memory and rollup invalidation scope.

**Out-of-order within window.** Memtable accepts unsorted appends; on flush, points are sorted by timestamp before encoding. Compression efficiency preserved.

**Late point landing in an already-rolled-up window.** Three options: (1) mark affected rollup window dirty, recompute on next compaction — **default for 1-min and 1-hour tiers**; (2) accept inconsistency: the rollup is from in-flight data, late points lost from rollup but present in raw — **default for 1-day** (recomputation cost not worth rare correction); (3) block ingestion of late point — we don't, data loss is worse than minor inconsistency.

**Client-clock skew.** Server-side timestamping by default; client `event_time` is metadata, not the primary index. Devices with skewed clocks were the source of half the worst incidents at scale; trust no client clock unless the customer explicitly opts in. **[STAFF SIGNAL: late-data discipline]**

## 14. Failure modes and observability-of-observability

**The meta-problem.** This system tells you when other systems are down. If it is down, the failure is invisible from inside. Cross-region / out-of-band monitoring is mandatory: a small separate deployment in a second cloud, status page driven from there.

**Concrete scenarios:**

| Scenario                          | Detection                                  | Response                                                              |
|-----------------------------------|--------------------------------------------|-----------------------------------------------------------------------|
| Compaction stalls on a shard      | `compaction_lag_seconds` > threshold       | Pause new flushes to shard; redirect ingest replicas; page on-call    |
| Querier OOM                       | Query process killed by OOM-killer         | Querier restart; storage daemon unaffected (separate process)         |
| Ingester WAL disk full            | Disk usage alert                           | Reject writes; ingester sheds load; replication compensates          |
| Bad-metric ingest blows up parser | Parse error rate per tenant spikes         | Per-tenant pipeline isolates blast radius; tenant gets the alert     |
| Block checksum mismatch           | CRC fail on read                           | Serve from replica; mark block for re-fetch from S3; alert            |
| S3 region degraded                | S3 GET latency spike                       | Querier serves from local cache; degraded-mode banner; eventual reads recover |
| Region failure                    | Health-check loss across many ingesters    | Cross-region replica becomes primary; ingest fails open in replica region |
| Cardinality bomb (live)           | New-series rate > 100K/min for tenant      | Auto-reject; page tenant; no operator action needed for resolution    |

**Blast-radius design choices:** querier is a separate process from ingester (querier OOM cannot crash ingest); per-tenant pipelines (one bad tenant cannot block others); per-shard quota (one bad shard cannot saturate the fleet); cross-region replication for ingest WAL.

**Gap markers.** When we know we lost data (region down, ingester crashed past WAL durability), the query layer returns explicit `gap` markers so dashboards distinguish "no data because nothing happened" from "no data because we lost it." A silent gap masquerading as zero traffic is the worst possible failure for a metrics product. **[STAFF SIGNAL: failure mode precision]** **[STAFF SIGNAL: blast radius reasoning]**

## 15. Metrics vs. wide-events

The architecture above commits to the metrics-TSDB philosophy: pre-shaped series, label-set identity, pre-aggregated rollups. It loses raw event detail; it is exquisitely efficient for queries you knew to ask.

The contrast is the **wide-events / column store** approach (Honeycomb, ClickHouse over raw events). Keep every event as a row with all attributes; no series concept; arbitrary `GROUP BY` at query time. Pros: no cardinality limit (every row is a row), arbitrary slice-and-dice, perfect for unknown-unknowns. Cons: ~10–50× more storage; queries are seconds to tens of seconds; quantile-over-large-window means scanning millions of events.

**Empirical reality.** Dashboards, alerting, and SLOs are well-served by metrics. Debugging novel incidents is well-served by wide events. Mature stacks have both, sharing identity (trace IDs, resource attributes). This design solves the prompt but is missing half the picture; production should pair it with a wide-event store and a unified query layer that joins by trace ID and resource label set. **[STAFF SIGNAL: metrics-vs-wide-events awareness]**

## 16. Tradeoffs taken and what would change them

- **Per-shard inverted index** over global. Simpler, no write hotspot, but cross-shard queries fan out. Change if: cross-shard query latency dominates SLO and a global secondary index is feasible (it almost never is at this scale).
- **Reject over sample on cardinality** breach. Honest but causes loud failures. Change if: a tenant explicitly opts in to sampling for dev/test environments.
- **2-hour blocks**. Compression sweet spot, manageable index. Change if: SSD throughput becomes the bottleneck (smaller blocks = better page-cache locality) or extreme query latency targets push us to memory-mapped only.
- **t-digest for native quantiles**. ~1% error at p99. Change if: SLO computation requires exact buckets; switch to fixed-bucket histograms at storage cost.
- **Server-side timestamps** by default. Change only with explicit customer commitment to clock discipline.

## 17. What I would push back on

The prompt accepts "100M active series" as a system property without naming it as a constraint. **A staff engineer rejects this framing.** 100M series is not free; it is a budget with a price, and the price must be visible to the customer. The first design move is making cardinality a billing dimension and a customer contract — not an internal limit, not a tuning parameter. **[STAFF SIGNAL: saying no]**

The prompt also implies a single backend serves both query types. At extreme query diversity (sub-10 ms point queries and 30-day fanouts), the right answer is two storage representations sharing an ingest path: the LSM I described, plus an explicit recency cache (in-memory ring per series) for point queries that bypasses block decoding. Production systems do this implicitly via memtable hot reads; I would make it explicit.

Finally, the prompt frames "metrics" in isolation. A metrics system without integration to traces, logs, and wide events is half a product. The query layer should expose `metric → exemplar → trace_id → log` joins as a first-class capability. **[STAFF SIGNAL: saying no]**