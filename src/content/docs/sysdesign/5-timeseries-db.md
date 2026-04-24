---
title: Time-Series Database for Observability
description: Time-Series Database for Observability
---

```
Design the storage and query layer for a metrics system handling 100M active series, 10M points/sec ingest, with queries ranging from 'last 5 minutes, one series' to 'last 30 days, sum across 10K series'. You're building the backend that competes with something like Prometheus remote storage or Datadog's metrics store. Walk me through it.
```

# Metrics Storage & Query Layer — Staff-Level Design Answer

## 1. Scope, Reframing, and What I'm Refusing to Solve

Before I draw a single box I want to push on the prompt, because the way it's stated conflates three workloads that want different backends.

**[STAFF SIGNAL: query-diversity reframing]** The two named queries are not two points on a continuum — they are two separate systems wearing a trench coat. "Last 5 minutes, one series" is a **point lookup** against the write path's hot memory: p99 target ~10ms, one shard, ~30 points returned, never touches disk if we do it right. "Last 30 days, sum across 10K series" is an **aggregation scan**: p99 target ~2s is acceptable, spans multiple shards, spans rollup tiers, and at raw 10-second resolution would touch ~2.6B points (10K × 30 × 86400/10) — which is a non-starter, so it *must* be served from pre-aggregated rollups. Every architectural decision from here down — storage layout, compression, index, downsampling, query planner — is driven by the fact that one backend must answer both. If you don't fork the read path at the query planner, you build a system that is either slow for dashboards or expensive for debugging, and usually both.

**[STAFF SIGNAL: cardinality as first-class]** The existential threat is not 10M points/sec; 10M points/sec is ~1.3 TB/day compressed and is tractable on a few dozen well-tuned machines. The existential threat is **cardinality**. If a user ships `http_request_count{trace_id="..."}` once, we gain 10M new series in an hour, the in-memory head block doubles, the inverted index doubles, compaction falls behind, query latency collapses, and the blast radius hits every tenant on the shard. Every mature TSDB operator has a war story about this. Cardinality defense is not a feature — it is a load-bearing wall.

**[STAFF SIGNAL: scope negotiation]** What I'm committing to:
- **Data type**: numeric metrics — counters, gauges, and pre-bucketed histograms (Prometheus-style `_bucket`/`_count`/`_sum` triplets, plus native histograms / sparse exponential histograms à la OpenTelemetry). Summaries are explicitly out because client-side quantiles don't compose.
- **Query semantics**: a PromQL-shaped surface (label matchers + range vectors + a fixed set of aggregation operators). No arbitrary JOINs, no user-defined functions at query time.
- **Data model**: (metric_name, sorted label set) → series; (series, timestamp) → f64 value. One writer-of-truth per series within a scrape interval.
- **Explicitly out**: logs, traces, profiles, and — critically — **ad-hoc analytics on raw unaggregated events**. That workload belongs in a columnar event store (the Honeycomb / Scuba / ClickHouse philosophy I'll contrast at the end). Trying to serve it from a metrics TSDB is the #1 way these systems get driven off a cliff.

**What I'll push back on at the end** (deferring specifics to §10): the "100M active series" number is meaningless without a per-tenant cardinality policy; "10M points/sec" is meaningless without a burst profile; "competes with Prometheus remote storage *or* Datadog's metrics store" is two very different products.

---

## 2. Capacity Math and Budget

I'll do this before architecture because the numbers drive the structure.

| Quantity | Formula | Value |
|---|---|---|
| Ingest rate | given | 10M points/sec |
| Points/day | 10M × 86,400 | 864B |
| Raw point size (ts8 + val8) | — | 16 B |
| Raw volume/day | 864B × 16 | **~13.8 TB/day** |
| Gorilla-compressed point | empirical, typical metrics | ~1.37 B |
| Compressed volume/day | 864B × 1.37 | **~1.18 TB/day** |
| Active series | given | 100M |
| Points per series per sec | 10M / 100M | 0.1 (≈ 10s scrape) |
| Per-series head buffer (target 2h, ~720 points × ~2B compressed + overhead) | ~2 KB | 2 KB |
| Total head-block RAM | 100M × 2 KB | **~200 GB RAM** (sharded across ingesters) |
| Avg labels/series | typical | 20 |
| Avg bytes/label (name+value, before dict) | — | ~50 B |
| Label metadata naive | 100M × 20 × 50 | ~100 GB |
| Label metadata with dictionary encoding (labels repeat massively) | empirically 5–10× compression | **~15 GB** |
| Series ID → label-set map | 100M × (16B id + ~32B dict ptrs) | ~5 GB |
| Inverted index (roaring bitmaps over series IDs) | empirically ~0.5–1 bit per series per posting | **~5–10 GB** |
| 1m rollup volume/day (4 aggs: sum/count/min/max) | 100M × 1440 × 4 × 1.5B | ~830 GB/day |
| 5m rollup/day | 100M × 288 × 4 × 1.5B | ~166 GB/day |
| 1h rollup/day | 100M × 24 × 4 × 1.5B | ~14 GB/day |
| 1d rollup/day | 100M × 1 × 4 × 1.5B | ~0.6 GB/day |

**Retention tier math (one year of storage):**

| Tier | Resolution | Retention | Volume |
|---|---|---|---|
| Hot (SSD) | 10s raw | 6h | 1.18 TB / 4 ≈ 295 GB |
| Warm (SSD) | 1m | 3d | 2.5 TB |
| Warm (SSD) | 5m | 30d | 5 TB |
| Cold (object store) | 1h | 1y | 5 TB |
| Cold (object store) | 1d | 5y | 1.1 TB |
| **Total active** | | | **~14 TB** (vs. 430 TB raw @ 1y — a **30×** reduction from rollups alone, before compression) |

**Query scan math:**

- **Q1 (5 min, 1 series):** 30 points × ~2B compressed = **60 B** of storage read. This must hit memtable or head block. Anything else is a bug.
- **Q2 (30d, sum over 10K series) at raw:** 10K × 30 × 8640 = **2.6B points** ≈ 3.5 GB compressed scan. Unacceptable cost and latency.
- **Q2 at 5m rollup:** 10K × 30 × 288 = **86M points** ≈ 120 MB scan. Acceptable for a 2s query.
- **Q2 at 1h rollup:** 10K × 30 × 24 = **7.2M points** ≈ 10 MB scan. This is what a dashboard uses.

**[STAFF SIGNAL: capacity math]** The compressed-vs-raw factor (10×), the rollup-vs-raw factor (360× for 1h rollup), and the ~200 GB of head-block RAM are the three numbers that force the rest of the design.

---

## 3. High-Level Architecture

```
                                    ┌───────────────────────────────────┐
                                    │          QUERY PLANNER            │
                                    │  - resolution selection           │
                                    │  - cost estimation / admission    │
                                    │  - tenant quota enforcement       │
                                    │  - shard fan-out & merge plan     │
                                    └──┬─────────────────────────┬──────┘
                                       │  hot path               │  cold path
                                       ▼                         ▼
Producers                  ┌─────────────────────────┐  ┌─────────────────────┐
(OTLP/                     │  INGESTER SHARD (N=...)│  │  QUERIER (stateless)│
 Prom remote-write) ──┐    │                         │  │  reads blocks from  │
                     │    │  ┌──────────────────┐  │  │  object store via   │
                     ▼    │  │ WAL (append log) │  │  │  block-range index  │
             ┌──────────┐ │  └──────────────────┘  │  └──────────┬──────────┘
             │DISTRIBUTOR│─┤           │            │             │
             │ (routing, │ │           ▼            │             │
             │  de-dup,  │ │  ┌──────────────────┐  │             │
             │  quota,   │ │  │ HEAD BLOCK (2h)  │◄─┼── point lookups (Q1)
             │ cardinal- │ │  │  ├ series map    │  │             │
             │  ity gate)│ │  │  ├ per-series    │  │             │
             └──────────┘ │  │  │   chunk buffer │  │             │
                          │  │  └ inverted index │  │             │
                          │  └──────────────────┘  │             │
                          │           │            │             │
                          │           ▼  flush (2h)│             │
                          │  ┌──────────────────┐  │             │
                          │  │ PERSISTENT BLOCK │  │             │
                          │  │ (immutable,      │  │             │
                          │  │  columnar, per-  │──┼─────────────┤
                          │  │  shard on SSD)   │  │             │
                          │  └──────────────────┘  │             │
                          └───────────┼────────────┘             │
                                      │ compaction +             │
                                      │ downsampling pipeline    │
                                      │ (streaming rollup        │
                                      │  service consumes WAL)   │
                                      ▼                          │
                           ┌─────────────────────────┐           │
                           │   OBJECT STORE (S3)     │◄──────────┘
                           │ - raw blocks (6h TTL)   │
                           │ - 1m rollup blocks      │
                           │ - 5m rollup blocks      │
                           │ - 1h / 1d rollup blocks │
                           │ - per-block index files │
                           └─────────────────────────┘
```

**[STAFF SIGNAL: rejected alternative]** Before moving on, a few backend choices I considered and rejected:

- **ClickHouse / general columnar OLAP.** Real systems (Uber's M3 successor work, some Grafana Loki-adjacent experiments) have done this and it works for point-in-time *analytics*. But ClickHouse stores every point as a row with full labels (or dictionary IDs), which loses the per-series locality that time-series compression needs, and its MergeTree sorting is tuned for analytical scans not for the "last 5 min of one series" point lookup. Gorilla-style compression on co-located per-series timestamps is 5–10× denser than ClickHouse's general codecs on the same data. For a general-purpose "I also want to JOIN with dimension tables" use case ClickHouse wins; for metrics specifically it loses on cost-per-point stored and point-lookup latency.
- **Cassandra wide-rows (the original Netflix Atlas design).** Write throughput is great. But the tag index has to live somewhere else (Atlas used in-memory and rebuilt on restart — brutal at our scale), and Cassandra's storage format gives you nothing for free on time-series compression. You end up running two systems, one of which is an index Cassandra doesn't know exists.
- **Postgres + TimescaleDB.** Fine up to ~1M series on a single node. At 100M the B-tree index on (series_id, ts) becomes the bottleneck, chunk exclusion degrades, and the tag index (a GIN on hstore/jsonb) scales even worse than the primary data. Wrong tool.
- **General LSM (RocksDB) as the primitive.** Tempting, and Prometheus's TSDB is morally similar. But RocksDB's per-key overhead (tens of bytes of metadata per key, assuming one key per point) destroys compression, and running compaction on 864B keys/day is a losing battle. What we actually want is an LSM over **chunks** (a chunk = one series × ~2h of compressed points), not over individual points. That's what we build ourselves.

We build a purpose-built TSDB. Commit.

---

## 4. Data Model and Per-Series Storage Layout

**Series identity.** A series is identified by the sorted, canonicalized tuple `(metric_name, [(label_key, label_value), ...])`. We compute a 128-bit FNV-1a or xxHash128 of the canonical form to produce a `series_id`. The full label set lives in a shard-local **postings dictionary** (FST-based, like Prometheus or Lucene) that maps label-key/value strings to compact IDs; the series-ID → label-set mapping uses those IDs, not raw strings. This is what collapses the 100 GB label metadata down to ~15 GB.

**On-disk block layout.** A **block** is a 2-hour immutable unit per shard containing:

```
   BLOCK ON DISK
   ┌──────────────────────────────────────────────────────────┐
   │ meta.json  (min_ts, max_ts, series_count, compaction_lvl)│
   ├──────────────────────────────────────────────────────────┤
   │ chunks/                                                   │
   │   Each chunk = one series, ~120 min of points, compressed│
   │   Layout of ONE chunk:                                    │
   │   ┌───────────────────────────────────────────────┐      │
   │   │ header: series_id, encoding, n_points, min_ts │      │
   │   ├───────────────────────────────────────────────┤      │
   │   │ timestamps: delta-of-delta, bit-packed         │      │
   │   │   ts₀ (64b), Δts₁ (varint),                    │      │
   │   │   δδ₂, δδ₃, ... (mostly 0 → 1 bit each)       │      │
   │   ├───────────────────────────────────────────────┤      │
   │   │ values: Gorilla XOR encoding                   │      │
   │   │   v₀ (64b), then XOR(vᵢ, vᵢ₋₁) with          │      │
   │   │   leading-zero / meaningful-bits control bits  │      │
   │   └───────────────────────────────────────────────┘      │
   │   Chunks packed contiguously, offset table at end.        │
   ├──────────────────────────────────────────────────────────┤
   │ index/  (inverted index — see §5.3)                       │
   │   - symbol table (all unique label strings, FST)          │
   │   - postings: label_key=label_value → [series_ids]        │
   │   - series table: series_id → [chunk_offsets in file]     │
   └──────────────────────────────────────────────────────────┘
```

**Why this layout.** Points for one series are physically contiguous — timestamp stream separate from value stream — because the compressors for the two have completely different regularity assumptions. Timestamps in a regular-scrape world are effectively constant (10s, 10s, 10s…), which is delta-of-delta's sweet spot: after the second point, every Δ is 0, costing one bit. Values are 64-bit floats where consecutive samples usually share most leading bits; Gorilla's XOR encoding plus leading-zero/meaningful-bits control exploits this to ~1.37 bytes/value on real metrics. The two are interleaved only at read time, never at write time.

**[STAFF SIGNAL: compression/encoding precision]** The empirical number — Facebook's 2015 Gorilla paper reports 1.37 bytes/point, Prometheus TSDB reports ~1.3, VictoriaMetrics reports ~0.8 with its additional codec layer — is earned on *regular* data. On irregular data (flaky scrapes, varying intervals, high-entropy float values like per-request latencies that are never repeated) it degrades to 3–6 bytes/point. This matters for capacity planning: a customer shipping a lot of `histogram_bucket{le="..."}` with well-behaved counter behavior compresses wonderfully; one shipping `gauge_with_random_float` does not. Capacity model must have a slack factor of ~2× on this, and detecting "this tenant's compression ratio just degraded" is an SLI.

---

## 5. Deep Dives

### 5.1 Cardinality Defense (the one that matters most)

```
INGEST PATH WITH CARDINALITY ENFORCEMENT
Producer ──► Distributor ──────────────────► Ingester Shard
                │                                    │
                │  ┌────────────────────────────┐    │
                ├─►│ (A) per-tenant series      │    │  ┌──────────────────────┐
                │  │     counter (HLL, cheap)   │    ├─►│ (C) head-block insert│
                │  │     reject if > quota      │    │  │     if new series:   │
                │  └────────────────────────────┘    │  │      - inc series    │
                │  ┌────────────────────────────┐    │  │        counter       │
                ├─►│ (B) per-(tenant, metric)   │    │  │      - write to WAL  │
                │  │     "suspect label" check: │    │  │      - add to index  │
                │  │     if a single label has  │    │  └──────────────────────┘
                │  │     >100K unique values in │    │
                │  │     last 1h → alert/block  │    │  ┌──────────────────────┐
                │  └────────────────────────────┘    └─►│ (D) real-time card   │
                │                                       │     tracker (streams │
                └──► sampled  ────────────────────────►│     to control plane)│
                     new-series log                     └──────────────────────┘
```

**[STAFF SIGNAL: cardinality as first-class] [STAFF SIGNAL: invariant-based thinking]** The invariants the system preserves: (i) per-tenant active-series ≤ quota, enforced at ingest — not at compaction, because by compaction the damage is done; (ii) no single label value may cause >N new series creations per minute without triggering a circuit breaker; (iii) the cardinality tracker itself must be bounded — we use HyperLogLog sketches per (tenant, metric_name), updated in the distributor, giving ~1% accuracy at ~12 KB per sketch.

**Reasonable defaults.** Per-tenant hard cap: 10M series (with quota increases via explicit approval — this is a revenue conversation, not a technical one). Per-metric soft cap: 1M series with auto-alert. Per-label "explosion detector": if a label's unique-value cardinality is growing faster than 10× in an hour and it's a label we've never seen before on this metric, *quarantine* new series containing that label and surface to the user's dashboard: "we are dropping series on metric `http_request_count` because label `user_id` appears to be unbounded. Accept or reject?" This user communication is the hardest part. Silently dropping data is how you lose customers; hard-rejecting without context is how you lose them more slowly.

**Post-incident cleanup.** When a tenant has emitted 50M bad series and wants them gone: we support a `forget_series(matcher)` operation that (a) marks series tombstones in the index, (b) excludes them from queries immediately, and (c) lets compaction physically drop them at the next rewrite. We do *not* attempt to rewrite in place — that's a day-long job on a live shard. The operation is async, idempotent, and auditable.

**What mid-level answers miss.** The detection problem is fine-grained. "100M total series" is a useless limit because by the time you hit it, one tenant has eaten everyone else's budget. The only limit that matters is per-tenant, and it has to be enforced at the *distributor* (the first-touch point that still has the full label set in memory), not at the shard.

### 5.2 The Tag Index: Inverted Index at 100M Series

```
INVERTED INDEX STRUCTURE (per-block)

SYMBOL TABLE (FST — finite state transducer)
   "region"     → 0x01
   "us-east-1"  → 0x02
   "us-west-2"  → 0x03
   "method"     → 0x04
   "GET"        → 0x05
   "POST"       → 0x06
   ... ~500K unique symbols at 100M series ...

POSTINGS (label_id=value_id → roaring bitmap of series IDs)
   0x01=0x02 (region=us-east-1) ─► RoaringBitmap{s₀, s₁, s₄, s₇, ...}  ~40M series
   0x01=0x03 (region=us-west-2) ─► RoaringBitmap{s₂, s₃, s₅, s₆, ...}  ~60M series
   0x04=0x05 (method=GET)       ─► RoaringBitmap{s₀, s₂, s₄, s₆, ...}  ~70M series
   0x04=0x06 (method=POST)      ─► RoaringBitmap{s₁, s₃, s₅, s₇, ...}  ~30M series

QUERY: {region="us-east-1", method="POST"}
   → intersect(posting(0x01=0x02), posting(0x04=0x06))
   → roaring AND  ~12M series  → stream series_ids to chunk reader
```

**[STAFF SIGNAL: index architecture precision]** Why roaring bitmaps over alternatives:
- **Raw sorted int arrays**: 4B/series ID × 40M = 160 MB per posting. Too big.
- **Elias-Fano** (VictoriaMetrics-style): denser than roaring on uniformly distributed IDs, but lookup is slower and ANDs are more complex.
- **Roaring**: chunks series IDs into 2¹⁶ buckets and picks dense-array, bitmap, or run-length representation per bucket. Gets ~0.5–1 bit/series on typical metric postings and supports SIMD-accelerated AND/OR/ANDNOT. It's the right default, and it's what Lucene and Prometheus TSDB both land on.

The FST for symbols gives us ~5× compression on the label-string dictionary vs. a flat hash map and supports ordered iteration (critical for prefix queries like `label=~"us-.*"`, which we expand to a prefix scan on the FST followed by a posting-list OR).

**Memory vs disk tradeoff.** Postings for the 2h **head block** live in RAM: this is the fast path for recent queries. Postings for persistent blocks live on disk with mmap, relying on the page cache; hot postings stay resident, cold ones page in at query time. At 100M series the index is ~10 GB per shard (we'll run 10–20 shards, so ~500 MB – 1 GB per shard), easily in memory.

**The query that melts the index.** Two queries are pathological:
1. A high-selectivity regex across a high-cardinality label: `{trace_id=~".*abc.*"}`. We must scan every posting for that label — if `trace_id` has 10M values, we iterate 10M postings. Mitigation: we **refuse** unanchored regexes on labels with cardinality above 10K (the query planner rejects with an actionable error). Trigram indexing for regex is on the roadmap but not in v1 because the memory cost triples the index.
2. A giant OR across many equalities: `{service=~"a|b|c|...|z_1000"}`. The posting-list union is linear in the number of terms. We bound the alternation count at 128 and require the user to use multi-label selectors instead.

These are policy decisions. **[STAFF SIGNAL: saying no]** Saying no to arbitrary regex is how the system stays fast.

### 5.3 Downsampling and Rollups

**Streaming rollup, not batch.** We compute 1m, 5m, 1h, 1d rollups in a streaming rollup service that tails the WAL. At flush time (every 2h) the 1m rollup block for that window is finalized. This avoids the "compaction also has to recompute rollups" coupling that makes Prometheus + Thanos's downsampler a separate, slow, eventually-consistent service. Streaming gives us rollup availability within seconds of ingest.

**Pre-computed aggregations.** For each rollup interval per series we store: `sum`, `count`, `min`, `max`. From these, the query planner can compose: `avg = sum/count`, `rate = delta(sum)/interval`. We do **not** store `avg` directly because averages don't compose across time windows (averaging 1m averages weighted-vs-unweighted is a bug factory).

**[STAFF SIGNAL: rollup precision]** The quantile problem. `histogram_quantile(0.99, ...)` is the query that breaks naive rollups. You cannot average p99s across series or across time — it's mathematically wrong, and the error can be enormous for heavy-tailed distributions. Two solutions, and we use both:

1. **Prometheus-style pre-bucketed histograms**: the user ships `http_latency_bucket{le="0.5"}`, `le="1.0"`, etc. These are counters and compose additively across series and across time. p99 is computed at query time from the bucket counts. Rollup works natively — we just sum the bucket counters.
2. **Sparse exponential histograms (OTel native histograms)**: each series carries a compact sketch (exponential bucket scheme with auto-adjusting scale). These compose via sketch merge. At rollup time we merge the sketches, not the quantiles.
3. **t-digest** as a fallback for user-submitted raw-value metrics where we want p99 support and they haven't pre-bucketed. A t-digest with ~100 centroids is ~2 KB, merges associatively, and gives ~1% quantile error. We store one per series per rollup interval. Storage cost: 100M series × 1440 intervals/day × 2 KB × compression ≈ extra 30 GB/day for 1m rollups that include t-digest. Not free. We make it opt-in per metric.

**Query-time resolution selection.** The planner picks the coarsest resolution whose step size is ≤ `query_range / target_num_points` (typically target = 1000 points for a dashboard, 10K for a deeper investigation). For Q2 (30d, 10K series): 30 days / 1000 points = ~43 min step → pick 1h rollup. This is what drops the scan from 2.6B points to 7.2M points.

### 5.4 Query Planner and Execution

**[STAFF SIGNAL: query planner discipline]** The planner owns three things the backend storage can't:

**Cost estimation before execution.** Given a query's matchers and range, the planner estimates: (number of matching series × points in range at chosen resolution × bytes/point). This is done by consulting the inverted index's posting-list sizes (cheap — it's just the bitmap cardinalities, which roaring stores in the header). If the estimate exceeds a tenant's query budget, we reject with an error message telling them what would fit: "your query would scan 500 GB; try narrowing to `region="..."` or reducing range to 7d."

**Admission control and fair share.** Each tenant has a concurrent-query quota and a CPU-seconds-per-minute budget. A tenant that submits 100 heavy queries doesn't deadlock the system — excess queries queue with a bounded wait, then fail. Within a tenant, queries are fair-shared. Cross-tenant, the scheduler uses weighted fair queuing so a whale doesn't starve everyone.

**Shard fan-out and merge.** A cross-series query fans out to every shard whose tenant + time range is relevant, each shard locally matches posting lists, pulls chunks, decodes, does partial aggregation (sum/count/min/max per output bucket), streams back to the querier, which does the final merge. The merge is commutative for sum/count/min/max, which is why those are the rollup primitives.

**Execution walkthrough, Q2 (30d, sum(http_requests) by (region), 10K matching series):**
1. Planner matches inverted index across shards: 10K series, distributed across 16 shards (~625/shard).
2. Range is 30 days; step for ~500 output points is ~1.5h → picks 1h rollup.
3. Cost estimate: 10K × 720 points × 16B ≈ 115 MB scan. Within budget. Admit.
4. Fan-out to 16 shards. Each: open rollup blocks for the time range (object store), read the relevant chunks (one per (series, 24h block)), decode Gorilla, group by `region` label (streamed from the in-memory label-set map), accumulate per-region partial sums per time bucket.
5. Shards stream partials back (small: 720 buckets × ~10 regions × 16B ≈ 115 KB/shard).
6. Querier sums partials across shards, returns 500 output points × 10 regions.
7. p99 target: 2s. Dominant cost: object-store GET latency on cold blocks. Mitigation: per-block result caching keyed by (block_id, matcher_hash, agg_spec); 80%+ hit rate on dashboard-driven queries.

### 5.5 Late-Arriving Data

**[STAFF SIGNAL: invariant-based thinking]** The invariant: a point's placement is determined by its timestamp, not its arrival time. We define a **grace window** of 1 hour. Points arriving within 1 hour of their timestamp go into the active head block (which holds 2 hours). Points arriving older than that go to a **backfill block** that triggers a targeted rewrite.

The ugly case: a rollup for 15:00 was finalized at 17:00 (head block flushed). A point with timestamp 15:30 arrives at 18:00. We have three options:
1. **Drop**, emit `metric_points_dropped_too_old`. Honest, simple, user is upset.
2. **Append to backfill block and schedule a rollup rewrite**. Correct, costly. We limit rewrite frequency to once/hour per block.
3. **Accept but don't recompute rollup**. Silently wrong. Never.

We pick (2) for points within 24h of their timestamp, (1) after that. The metric emitted on drop tells users when their collectors are misbehaving. This is explicit in our SLO dashboard.

---

## 6. Failure Modes and the Observability-of-Observability Problem

**[STAFF SIGNAL: failure mode precision]** Concrete scenarios:

- **Cardinality spike**: distributor's HLL counter detects the rate, auto-applies a per-tenant ingest circuit breaker within 30s. Tenant's dashboard shows a banner: "ingestion rate-limited on `metric_X` due to cardinality spike on label `label_Y`." SRE paged only if the spike spans >3 tenants simultaneously (platform-level issue).
- **Compaction backlog**: if L0 blocks accumulate beyond 3× target, ingest slows (WAL keeps going, head block grows, query latency for recent data degrades but queries still work). If backlog hits 10×, we auto-add capacity or shed new writes for the backlogged tenant. Compaction progress is an SLI, with pages at specific thresholds.
- **Query OOM**: each query runs with a memory cap (typically 4 GB). Exceeding it returns a structured error with the estimated required memory. The querier process is cgroup-isolated so one bad query can't kill neighbors; if it does escape, the querier restarts and in-flight queries for other tenants retry transparently (queriers are stateless).
- **Block corruption**: detected at read time via per-chunk CRC. Response: return partial results with a `partial=true` flag, emit `metric_block_corruption_total`, trigger a rewrite from adjacent replica (we maintain 2× replication on hot/warm, single-copy with S3 durability on cold).
- **Region failure**: each region is independent — writes land locally, are replicated async to a secondary region. On failover: queries for the failed region return `status=unavailable` for that region's data, *not* interpolated values, *not* gaps. **[STAFF SIGNAL: blast radius reasoning]** The observability-of-observability rule: when the metrics system itself is degraded, the degradation must be *visible* to operators, not hidden behind interpolated charts. Interpolating gaps is the cardinal sin — it's how you hide your own outage from the on-call.
- **The meta-problem**: our own system's metrics must not be served by itself as sole source of truth. We ship a second, minimal telemetry path (low-cardinality gauges to a separate Prometheus instance, and critical logs to a separate logging stack) for the on-call to see "is the big system up?" when the big system is down. This is the "phone-a-friend" oncall pattern and it is non-negotiable.

---

## 7. Multi-Tenancy and Isolation

**[STAFF SIGNAL: multi-tenant isolation]** Isolation at each layer:

- **Ingest**: per-tenant rate limit (points/sec) and cardinality limit (series count), enforced at distributor via leaky-bucket in Redis or embedded token bucket.
- **Storage**: tenants are *logically* isolated by prefixing series IDs with tenant ID, but physically share shards up to a threshold. Whales (>1M series or >100K points/sec) get dedicated shards. This is cheaper than dedicated-everything and bounds blast radius.
- **Query**: per-tenant concurrent-query quota, CPU-seconds budget, memory cap. Weighted fair queuing across tenants.
- **Compaction**: per-tenant compaction budget so a tenant whose blocks are slow to compact doesn't starve others.

The cost of over-isolation: dedicated shards are ~3× the per-series cost of shared shards (lower utilization, replication overhead). So we shard by default and only dedicate when a tenant's noise profile justifies it. Measured, not assumed.

---

## 8. Recent Developments — What We Borrow From

**[STAFF SIGNAL: modern awareness]**

- **Gorilla (Facebook, 2015)**: the compression foundation. Every serious TSDB since has used delta-of-delta + XOR with minor variations.
- **Prometheus TSDB (Fabian Reinartz et al.)**: the head/WAL/blocks architecture we're borrowing directly. Well-proven at single-node scale; our contribution is horizontal sharding on top.
- **Thanos / Cortex / Mimir**: the "scale Prometheus horizontally" lineage. Mimir's blocks-storage design (query frontend, store gateway, ingester separation) is closest to what we've sketched. Worth stealing from.
- **VictoriaMetrics**: an alternative design point with its own compression (MetricsQL + additional encoding layer), a mergeset-based index, and a well-known high-cardinality story. Their approach to not using a separate inverted index (they use a sorted-key store with prefix scans) is elegant; we've chosen the Lucene-style inverted index instead because it's better for complex matchers at the cost of higher memory.
- **InfluxDB IOx**: abandoned the bespoke TSM file format in favor of Apache Arrow / DataFusion / Parquet. An important bet on "columnar file formats are general enough," and it does give great ad-hoc analytics, but loses some specialized compression. We've explicitly *not* taken this route for the core path because our compression ratio matters more than analytical flexibility; but a Parquet-based *export* path for users who want to run ad-hoc SQL against their own metrics is a natural extension.
- **Monarch (Google)**: multi-region, hierarchical aggregation, push-based ingest, in-memory-first. The hierarchical rollup is worth studying for our cross-region story.
- **Honeycomb (and the wide-event philosophy)**: covered in §9.
- **OpenTelemetry**: the ingest protocol converging toward OTLP, and the native histogram format (sparse exponential) that finally gives us composable quantiles. Our ingest path must speak both Prometheus remote-write and OTLP from day one.

---

## 9. Tradeoffs Taken, What Would Change Them, and the Wide-Event Philosophical Fork

**What would force a redesign?**

- **If queries must support arbitrary expressions over raw events** (joins, arbitrary grouping by high-cardinality attributes, ad-hoc investigation): this design is wrong. We'd lose on flexibility. The right system is a columnar event store — ClickHouse, or a Honeycomb-style column-per-attribute layout with segment-level indices. Metrics pre-aggregation throws away the raw cardinality that ad-hoc debugging needs. The Honeycomb argument is that for *debugging production issues* you don't want metrics at all; you want wide events (one row per request, ~100 attributes), queryable at ingest resolution forever. They're right for that use case.
- **If retention must be 10 years raw**: rollups become insufficient. We'd need a columnar cold-storage format (Parquet with Arrow) and accept that historical queries are analytical, not dashboard-speed.
- **If we need real-time alerting <1s from ingest**: the current design has a ~5s ingest-to-query window (distributor batching + head-block insert + index update). To go below 1s we'd need a separate in-memory streaming alert path that bypasses the storage entirely.

**The philosophical fork.** Metrics-TSDB pre-aggregates: you commit at instrumentation time to the dimensions you'll slice by. Wide-event stores keep it all: you figure out the slice at query time. Metrics are cheaper, faster for known dashboards, and hit a cardinality wall. Events are more expensive, slower, and don't hit that wall. The *right* observability stack has both — metrics for SLO dashboards and alerting (what this design serves), events for debugging (a separate system). Shipping one backend that claims to do both is how you get a product that's mediocre at each. I'd be explicit with the interviewer that my answer covers the metrics half and would architect the events half as a separate service.

---

## 10. What I'd Push Back On

**[STAFF SIGNAL: saying no] [STAFF SIGNAL: operational longevity]**

1. **"100M active series"** without a per-tenant policy is not a real requirement. The right requirement is "support up to X tenants with per-tenant caps totaling 100M aggregate, with the cardinality enforcement story below." Otherwise "100M series" means "one bad customer away from outage."
2. **"Competes with Prometheus remote storage or Datadog's metrics store"** is two products. Prometheus remote storage is a backend bolted to an existing scrape protocol, and users expect PromQL fidelity. Datadog is a SaaS with its own agent, custom metric types, APM integration, and a sprawling query language. Which one?
3. **"Queries ranging from 5 minutes to 30 days"** doesn't bound the cross-product with series count. A "last 30 days, 1 series" query is trivial; a "last 30 days, all series" query is infeasible at any budget. I'd insist on a query-cost policy negotiated with the product team.
4. **Uniform retention** — as drafted above, I'd push back on any requirement that all metrics have the same retention. Retention must be metric-class-based (SLO metrics: 1y at 1m rollup; debug metrics: 7d raw; cost-control metrics: 1h raw). Otherwise you're paying to store data nobody queries.
5. **Year-3 realities.** This design is correct for year 0–1. By year 3: the compression format will have evolved (we'll want native histograms everywhere, not in an opt-in path), the index memory budget will have shifted (trigram regex index if trace_id-style queries become common), at least one tenant will have exfiltrated data via a clever label construction, and schema evolution will be the dominant eng cost — which argues for versioning block formats from day one and having an online block-rewriter as a first-class service, not a migration tool.

---

## Close

The design is a purpose-built TSDB: distributor with hard cardinality gates → sharded ingesters with WAL + head block + per-series Gorilla-compressed chunks + roaring-bitmap inverted index → streaming rollup pipeline into 1m/5m/1h/1d tiers → object-store tiered long-term storage → a stateless querier fronted by a cost-aware planner that picks resolution, enforces admission, and fans out per-tenant fair-share. The two named queries are served by two paths: Q1 hits the head block in memory and returns in <10ms; Q2 hits 1h rollup blocks from object store via block-range index, fans out across shards, partial-aggregates, merges, and returns in <2s. Cardinality is the existential threat, enforced at the distributor with HLL tracking and per-tenant caps; quantile rollups use pre-bucketed histograms and exponential native histograms, never averaged quantiles; the observability-of-observability problem is handled with a separate minimal telemetry path for operator visibility during our own outages. For workloads that need arbitrary raw-event queries, this is the wrong system and I'd architect a separate wide-event store alongside it.