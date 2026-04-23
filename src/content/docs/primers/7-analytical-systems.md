---
title: Analytical systems
description: Analytical systems
---

## 1. What a staff engineer actually needs to know

**Interviewers probe depth in three places:**

1. Can you correctly decide *when* an analytical system is needed vs when OLTP + cache suffices.
2. Can you reason about storage layout (row vs column), partitioning, sorting, and indexing as *tradeoffs* — not as "I add an index."
3. Can you articulate ingestion/freshness/query-latency tradeoffs without handwaving.

**Expected depth:**
- Explain columnar layout in 60 seconds, with compression and pruning reasoning.
- Pick partitioning + sort keys from a workload description and justify them.
- Name at least three index types, when each helps, when each hurts.
- Design an ingestion path that doesn't block the query path.
- Describe why joins are usually denormalized or precomputed in analytics.

**Ignore for most staff interviews:**
- MVCC internals, WAL group commit, buffer pool replacement policies.
- Specific codec internals (Gorilla, FOR-delta details, ZSTD dictionary training).
- SIMD/vectorized execution internals beyond "it's batched and cache-friendly."
- Database-implementer depth of join algorithms (enough: hash, sort-merge, broadcast, shuffle).
- Specific product feature matrices. You need the *shape* of Pinot/Druid/ClickHouse, not their release notes.

---

## 2. Core mental model

### 2.1 What an analytical data system is

A system optimized for reading **large ranges of many rows** to compute **aggregates** over **few columns**, with **bulk ingestion** and **rare row-level updates**. Reads dominate; writes arrive in batches or streams as append-only.

### 2.2 OLTP vs OLAP — the only distinction that matters

```
OLTP (e.g. Postgres, MySQL)           OLAP (e.g. ClickHouse, Druid, Pinot, BigQuery)
─────────────────────────────────      ────────────────────────────────────────────
Workload: point lookups + updates      Workload: scans + aggregations
Query touches: 1–100 rows              Query touches: 10^6 – 10^12 rows
Columns read: most of the row          Columns read: 3–20 of 200
Concurrency: thousands of small txns   Concurrency: tens to hundreds of large queries
Writes: random, row-level, ACID        Writes: bulk append, immutable segments
Latency target: ms per query           Latency target: ms–seconds per query, but
                                        over vastly more data
Storage layout: row-oriented            Storage layout: column-oriented
Index: B-tree on PK + secondary         Index: partitions, zone maps, bitmap, inverted
```

### 2.3 Why analytical systems are optimized differently

Three forcing functions:

1. **I/O is the bottleneck, not CPU.** A 100-column table where you read 3 columns means row storage wastes 97% of bytes read. Column storage doesn't.
2. **Values within a column are homogeneous**, so compression ratios are 5–20×. Row storage fights you here — adjacent bytes are unrelated types.
3. **Queries are read-mostly and bulk**, so you can make writes slow/batched to make reads fast, and you can make segments immutable (no per-row locking, no MVCC tax on reads).

### 2.4 Typical OLAP workloads

- **Dashboards:** pre-defined queries, p95 < 1s, high QPS, last N days.
- **Ad hoc exploration:** arbitrary slice/dice, p95 seconds to minutes, low QPS.
- **ETL/reporting:** nightly batch, minutes-to-hours acceptable.
- **User-facing analytics:** p95 < 200ms, high QPS, bounded query shape. (Pinot/Druid territory.)

### 2.5 "Just use Postgres" — when it breaks

Postgres handles analytics fine until one of:

- Working set stops fitting in RAM (hot data ≫ `shared_buffers`).
- Scan volume per query > ~10 GB and p95 matters.
- You need to aggregate billions of rows in seconds.
- Ingest rate exceeds what random-I/O row storage can sustain.
- Dashboard QPS × scan-per-query exceeds disk bandwidth.

At that point row storage + B-trees becomes structurally wrong — not "tune it harder" wrong. You need a columnar engine.

---

## 3. Essential design ideas

The seven concepts below are the vocabulary. You must speak all of them fluently.

### 3.1 Columnar storage

- **What:** values of one column stored contiguously on disk, separate file/stream per column.
- **Why:** scans read only needed columns; homogeneous data compresses well; vectorized execution over tight loops.
- **Tradeoff:** row reconstruction costs more; row-level updates are expensive (touch N column files); wide-schema OLTP kills it.
- **Say in interview:** "Columnar cuts scan bytes proportional to columns-read / columns-total, and compression gives another 5–20× on top."

### 3.2 Partitioning

- **What:** physical split of data by a key, typically time (`day`) or a low-cardinality categorical (`region`, `tenant_id`).
- **Why:** partition pruning — queries with a predicate on the partition key skip whole partitions without reading them.
- **Tradeoff:** too fine-grained → many small files, metadata overhead; too coarse → no pruning benefit. Wrong key → no pruning at all.
- **Say:** "Partition by the dimension that's in 90%+ of WHERE clauses, usually time."

### 3.3 Sorting / clustering

- **What:** within a partition, rows sorted by a key or a composite key.
- **Why:** range scans, zone maps, and compression all benefit from locality. Sorted data compresses far better (RLE, delta).
- **Tradeoff:** only one sort order per storage copy; ingestion must sort or compaction must re-sort.
- **Say:** "Sort key = most selective frequent filter after the partition key."

### 3.4 Indexing

- **What:** auxiliary structures (zone maps, bitmaps, inverted lists) that skip data.
- **Why:** avoid scanning what you don't need.
- **Tradeoff:** storage overhead, write amplification, must be maintained during compaction.
- **Say:** "Indexes in OLAP are pruning aids, not lookup structures — selectivity and write cost decide."

### 3.5 Pre-aggregation / materialization

- **What:** precompute aggregate tables or cubes (rollups, materialized views, star-trees).
- **Why:** trade storage + write-time compute for dramatically cheaper reads.
- **Tradeoff:** staleness, storage cost, query must match the cube's dimensions or fall back to raw.
- **Say:** "If dashboards hit a fixed query shape at high QPS, pre-aggregation converts O(rows) reads into O(dimensions) reads."

### 3.6 Compression

- **What:** dictionary, RLE, delta, bit-packing, frame-of-reference, plus general-purpose (ZSTD, LZ4) on top.
- **Why:** smaller files = less I/O = faster scans. On modern hardware, decompression is faster than reading uncompressed bytes from disk.
- **Tradeoff:** CPU cost on decode, but almost always worth it. Encodings constrain update patterns.
- **Say:** "Compression isn't about disk cost — it's scan speed. I/O is the bottleneck."

### 3.7 Append-heavy ingestion

- **What:** writes arrive as new immutable segments; updates/deletes are handled via tombstones + compaction or via eventual rewrite.
- **Why:** lets you build dense, sorted, compressed, indexed segments once; reads never fight writes.
- **Tradeoff:** updates are expensive; freshness bounded by segment-seal latency; need background compaction.
- **Say:** "Immutable segments are what lets columnar layouts exist at all — you can't maintain 200 compressed column streams under random row updates."

---

## 4. Columnar systems and data layout

This section is the one that separates strong candidates. Get it precise.

### 4.1 Row vs column layout — the picture

```
Row store (Postgres page):
┌────────────────────────────────────────────────────────┐
│ row1: (user_id, ts, country, amount, status, ...)      │
│ row2: (user_id, ts, country, amount, status, ...)      │
│ row3: (user_id, ts, country, amount, status, ...)      │
│ row4: ...                                              │
└────────────────────────────────────────────────────────┘
   ↑ Read one row = one page. Read SUM(amount) = read everything.

Column store (Parquet / ORC / native columnar):
┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
│  user_id     │ │      ts      │ │   country    │ │    amount    │
│  [101,102,   │ │ [t1,t2,t3,   │ │ [US,US,FR,   │ │ [12.0,7.5,   │
│   103,...]   │ │   t4,...]    │ │   US,...]    │ │   3.2,...]   │
└──────────────┘ └──────────────┘ └──────────────┘ └──────────────┘
   ↑ Read SUM(amount) = read one file. Read whole row = read N files + reconstruct.
```

### 4.2 Why columnar helps scans and aggregation

- **Bytes read:** `bytes_read ≈ rows × sum(size(c) for c in projected_columns)` vs row store's `rows × sum(size(c) for c in all_columns)`.
- **Cache & SIMD:** tight loop over a packed, type-homogeneous array → vectorized kernels, branch-predictable, prefetchable.
- **Column-local predicates:** filter on `country = 'US'` by scanning one file, producing a bitmap, then materializing other columns only for matching rows (late materialization).

### 4.3 Why compression is better in columnar layouts

Column values share **type, distribution, and often sort order**. That unlocks:

```
Dictionary:       [US, US, FR, US, DE, US]   →  dict = [US:0, FR:1, DE:2]
                                                data = [0,0,1,0,2,0]  (2-bit packed)

RLE:              [0,0,0,0,1,1,2,2,2,2]      →  [(0,4),(1,2),(2,4)]

Delta:            [1700, 1703, 1710, 1712]   →  [1700, +3, +7, +2]

Frame-of-ref:     [1000003, 1000007, 1000012] → base=1000003, [0,4,9]  (bit-packed)
```

Row stores can't use any of these well — adjacent bytes belong to unrelated fields.

### 4.4 Why columnar is bad for point writes / row updates

Updating one row in a 200-column table means touching 200 column streams, each compressed, often RLE'd across many rows. So OLAP systems:

- Make segments immutable.
- Handle updates via: delete-vector + re-insert, or merge-on-read, or periodic rewrite via compaction.
- Expose them at read time via a merge step (adds query cost).

This is why you don't run OLTP on columnar engines.

### 4.5 How partitioning and sort order affect pruning and locality

```
Query: SELECT SUM(amount)
       FROM events
       WHERE day = '2026-04-20' AND country = 'US' AND user_id = 42;

Partition pruning:    day = '2026-04-20'         → skip all other day partitions
Zone-map pruning:     country = 'US'             → skip blocks whose min..max
                                                    doesn't overlap 'US'
Sort-key skip:        user_id = 42 (if sorted)   → seek into sorted run
Column scan:          read `amount` column only  → tiny I/O

I/O read: one day partition × surviving blocks × one column   <<   full table scan
```

Partitioning is coarse pruning (directories/files). Sort + zone maps are fine pruning (blocks within a file).

### 4.6 What staff-level candidates say about scan reduction

Frame every design choice as **how it reduces scan bytes**:

1. Partitioning → skip partitions.
2. Sort + zone maps → skip blocks within partition.
3. Column projection → skip columns.
4. Compression → fewer bytes per surviving block.
5. Pre-aggregation → skip the raw table entirely.

This layered pruning story is what interviewers want to hear.

---

## 5. Must-know indexing concepts

In OLAP, "index" mostly means "data-skipping structure." Point lookups aren't the common case.

### 5.1 Partition pruning

- **Problem solved:** avoid opening irrelevant files.
- **When it helps:** queries filter on the partition key (usually time).
- **When it doesn't:** queries don't filter on that key, or filter on a high-cardinality key you partitioned on (too many small files).
- **Tradeoffs:** essentially free at query time, but fine-grained partitioning causes metadata explosion and slow planning.

### 5.2 Zone maps / min-max metadata

- **What:** per block (e.g. per 64K-row chunk), store `min`, `max`, null count for each column.
- **Problem solved:** skip blocks whose value range can't match the predicate.
- **When it helps:** predicate column is **correlated with sort order**. On sorted data: skips are dramatic.
- **When it doesn't:** random/unsorted data → min/max spans the whole domain → no skipping.
- **Tradeoff:** cheap to store, cheap to maintain, useless without sort correlation.

```
Sorted by user_id:           Unsorted:
Block 1: min=1,   max=1000   Block 1: min=1, max=999999
Block 2: min=1001,max=2000   Block 2: min=2, max=999998
Block 3: min=2001,max=3000   Block 3: min=1, max=999997
        ↑                              ↑
WHERE user_id=1500 skips 2/3    WHERE user_id=1500 skips 0/3
```

### 5.3 Inverted index (high level)

- **What:** term → list of doc/row IDs containing it. Same idea as search engines.
- **Problem solved:** highly selective equality/IN predicates on high-cardinality columns (e.g. `user_id = X`, `trace_id = Y`).
- **When it helps:** needle-in-haystack queries; text search; log lookups.
- **When it doesn't:** broad scans, aggregations over most rows.
- **Tradeoff:** large storage overhead (can rival data size), expensive to build and maintain on updates. Common in Elasticsearch, Pinot text indexes, Druid.

### 5.4 Bitmap index (high level)

- **What:** per distinct value of a column, a bitmap of rows where that value appears.
- **Problem solved:** fast AND/OR over multiple low/medium-cardinality predicates.
- **When it helps:** `WHERE country='US' AND device='mobile' AND status='paid'` — intersect three bitmaps.
- **When it doesn't:** high-cardinality columns (a bitmap per value explodes), heavy updates.
- **Tradeoff:** compressed bitmaps (Roaring) make this cheap for moderate cardinality. Great for faceted filtering.

```
country:  US -> 1110010110
          FR -> 0001101001
          DE -> 0000000000...

device:   mobile  -> 1011010110
          desktop -> 0100101001

AND:      US & mobile -> 1010010110  (scan only these rows)
```

### 5.5 Star-tree / pre-aggregation indexes (high level)

- **What:** a tree that materializes aggregates along a set of dimensions, with "star" nodes that represent "any value."
- **Problem solved:** fixed-shape dashboard queries over many dimensions at high QPS with bounded latency.
- **When it helps:** you know the dimension set, slices are predictable, ingest rate is manageable.
- **When it doesn't:** ad-hoc exploration over arbitrary columns; cube explosion.
- **Tradeoff:** storage blow-up proportional to dimension combinations; ingestion cost; staleness vs rebuild frequency. Pinot's differentiator.

### 5.6 Dictionary encoding

- **What:** replace string/large values with integer codes; store a dictionary per column or per block.
- **Problem solved:** compression + fast equality/GROUP BY (integer compares, not string compares).
- **When it helps:** low-to-medium cardinality string columns (country, device, status).
- **When it doesn't:** high-cardinality unique strings (URLs, user agents verbatim) — dictionary grows without bound.
- **Tradeoff:** small CPU cost for dict lookup vs huge wins for aggregation and joins.

### 5.7 Clustering / sorted order as indexing aid

- **What:** sort the table (or sub-groups) on a key.
- **Problem solved:** makes zone maps effective, improves compression, enables range scans.
- **When it helps:** predictable hot filter columns (time + tenant_id is classic).
- **When it doesn't:** evenly-accessed dimensions — pick one sort order and you lose on the others.
- **Tradeoff:** one physical sort per copy. Some systems support secondary sort via projections/skip-indexes.

### 5.8 When is full indexing worth it vs too expensive?

```
                   │  Low write rate   │  High write rate
                   │  + High QPS       │  + Low QPS
───────────────────┼───────────────────┼───────────────────
High selectivity   │  BUILD INDEX      │  Maybe build zone
(needle/haystack)  │  (inverted,bitmap)│  maps only
───────────────────┼───────────────────┼───────────────────
Low selectivity    │  Scan is fine;    │  Definitely no
(broad aggregate)  │  use compression  │  index; relies on
                   │  + sort           │  partitioning alone
```

Default position: **partition + sort + zone maps + dictionary**. Add bitmap/inverted only when a specific query pattern demands it.

---

## 6. Ingestion vs query tradeoffs

### 6.1 Batch vs real-time

- **Batch:** hourly/daily load, large sorted compressed segments, maximum read performance, staleness = batch window.
- **Real-time:** streaming ingest, small segments, often less compressed/sorted, queryable in seconds, higher per-row overhead.
- **Staff move:** combine both. Lambda-flavored.

```
            Stream (Kafka)                          Batch (S3 / warehouse)
                 │                                           │
                 ▼                                           ▼
        ┌──────────────────┐                      ┌────────────────────┐
        │ Real-time segs   │                      │ Historical segs    │
        │ (mutable-ish,    │                      │ (immutable, sorted,│
        │  small, recent)  │                      │  compressed, pre-  │
        │                  │                      │  aggregated)       │
        └────────┬─────────┘                      └──────────┬─────────┘
                 │                                           │
                 └──────────────────┬────────────────────────┘
                                    ▼
                            Query layer merges
                            results from both,
                            dedupes on PK/version
```

### 6.2 Mutable vs immutable segments

- **Immutable:** append-only, compressed, indexed once, never rewritten except by compaction. Fast to read.
- **Mutable (real-time):** recent window, in-memory or lightly persisted, supports per-row writes, slower per-row reads but tiny data volume.
- **Transition:** real-time segment *seals* (at size/time threshold) → becomes an immutable segment → eventually merged by compaction.

### 6.3 Freshness vs query speed

Three dials, pick your spot:

```
FRESHNESS            │  SEGMENT SIZE      │  QUERY SPEED
High (seconds)       │  Small             │  Slower (many segs, less compressed)
Low (hours)          │  Large             │  Fast (few big sorted segs)
```

### 6.4 Compaction / merge (high level)

- Background process merges small segments into larger ones, re-sorts, re-compresses, rebuilds indexes, applies tombstones.
- Reduces segment count (read amplification) at the cost of write amplification.
- LSM-ish. Tuning knobs: trigger threshold, concurrency, resource isolation from query path.

```
Before:  [s1][s2][s3][s4][s5][s6][s7][s8]   (8 small segments, lots of metadata)
            │   │   │   │   │   │   │   │
            └───┴───┴───┴───┼───┴───┴───┴──── compaction
                            ▼
After:   [  S1         ][  S2         ]      (2 large segments, sorted, compressed)
```

### 6.5 Why separate ingest path from serving path

- Ingestion bursts should not crash queries. Queries under load shouldn't block ingestion.
- Separate workers (different machines or at least isolated CPU/mem).
- Shared storage (object store or distributed FS) as handoff.
- Lets you scale each independently and do rolling upgrades safely.

### 6.6 Operational implications of low-latency analytics

- **Hot tier** for recent data (local NVMe, replicated); **cold tier** on object store.
- **Tiered storage** with async demotion.
- **Query-time freshness** depends on segment handoff latency — watch it.
- **Compaction as a first-class resource consumer** — plan its budget.
- **Rebalancing/replication** when nodes come and go; segment assignment algorithms matter.

---

## 7. Tiered storage and decoupled storage-compute

Two linked architectural shifts worth fluency on: **separating storage from compute**, and **tiering data by access frequency**. Both show up in every modern analytics system, and the same ideas are now being imported into inference infrastructure.

### 7.1 Shared-nothing MPP vs decoupled architecture

Classical MPP (Vertica, early Redshift, Greenplum) co-located storage and compute on each node. Scaling meant restriping data across new nodes. Compute ran 24/7 to keep storage online.

Decoupled architectures (Snowflake, BigQuery, Databricks, Trino-over-Iceberg) put durable data in object storage (S3, GCS, Azure Blob) and run stateless compute clusters over it. A catalog/metadata service mediates (file lists, schemas, snapshots, stats).

```
Shared-nothing MPP                     Decoupled storage-compute
──────────────────                     ─────────────────────────
┌─────┐ ┌─────┐ ┌─────┐                ┌───────┐ ┌───────┐ ┌───────┐
│ C+S │ │ C+S │ │ C+S │                │compute│ │compute│ │compute│
└─────┘ └─────┘ └─────┘                │  W1   │ │  W2   │ │  W3   │
   ↑       ↑       ↑                   └───┬───┘ └───┬───┘ └───┬───┘
 scale = restripe                          │         │         │
 data across nodes                         └─────────┼─────────┘
                                                     │
                                              ┌──────▼──────┐  metadata:
                                              │  catalog    │  files, stats,
                                              │  (Iceberg / │  snapshots,
                                              │   Snowflake │  schema
                                              │   meta svc) │
                                              └──────┬──────┘
                                                     │
                                              ┌──────▼──────┐  durable, infinite,
                                              │ object store│  cheap,
                                              │  (S3/GCS)   │  high-latency
                                              └─────────────┘
```

**Wins from decoupling:**
- **Elastic compute:** spin up a 100-node cluster for a heavy query, tear it down. Pay per second.
- **Multi-tenant isolation:** independent compute clusters over the same data; no resource contention.
- **Scale to zero:** compute cost → $0 when idle; storage cost stays linear in data volume.
- **Storage is someone else's problem:** S3 gives 11 nines durability at commodity price; don't reinvent it.
- **Independent scaling:** storage and compute grow at different rates. Storage usually grows faster.

**Costs:**
- **Network is the new bottleneck.** Object store GET is ~20–100ms vs local NVMe ~100µs. Orders-of-magnitude worse tail latency.
- **Cold start.** Warming caches on a fresh warehouse takes seconds to minutes.
- **Consistency model.** Historically object stores were eventually consistent (S3 is stronger now); table formats (Iceberg/Delta/Hudi) layer snapshot isolation on top.

### 7.2 Open table formats as the enabler

Iceberg / Delta Lake / Hudi turned "a bunch of Parquet files in S3" into a queryable table with schema evolution, snapshots, and ACID-ish semantics. Critical because they let multiple engines (Trino, Spark, Flink, ClickHouse) share the same data without a proprietary format.

What they give you:
- **Manifest files:** list of data files + per-file stats (min/max per column).
- **Snapshots:** atomic commit unit; readers see a consistent view.
- **Schema evolution** without full rewrite.
- **Time travel:** query as-of a snapshot.
- **Partition evolution:** change partition scheme without full rewrite.

Interview relevance: if someone asks about "data lake" design, Iceberg/Delta is the modern answer. Raw Parquet in S3 without a table format is a pre-2020 design.

### 7.3 Hot / warm / cold tiering

Within a given system, data is classified by access frequency and placed on storage with matching latency/cost.

```
Tier      Latency              Cost      Typical use
──────    ───────────────      ─────     ─────────────────────────────
Hot       100µs  (NVMe)        $$$$      Last 1–24 hours; real-time
                                         dashboards; low-latency serving
Warm      1–10ms (SSD, remote  $$        Last 1–30 days; frequently
          block storage)                 queried historical
Cold      50–200ms (S3/GCS)    $         Months to years; rare ad-hoc
                                         or compliance queries
Archive   seconds–minutes      ¢         Years; audit, recovery only
          (Glacier, tape)
```

Placement rules:
- **By age:** time-based demotion (hot → warm → cold as data ages).
- **By access frequency:** LRU-ish promotion/demotion based on recent query hits.
- **By explicit label:** user-declared hot datasets (Snowflake, Pinot support this).

### 7.4 Caching is the whole game

When the authoritative store is remote, cache hit rate is the dominant performance lever. Typical layers:

```
┌─────────────────────────────────────────┐
│ Query result cache (query text → result)│  p99 µs; low hit rate for
│                                         │  parameterized dashboards
├─────────────────────────────────────────┤
│ Fragment / partial-aggregation cache    │  reuse across similar queries
├─────────────────────────────────────────┤
│ Block / page cache (local NVMe)         │  the workhorse; column-block LRU
├─────────────────────────────────────────┤
│ Metadata cache (file lists, stats)      │  critical; planning can take
│                                         │  longer than execution if this
│                                         │  misses against S3
├─────────────────────────────────────────┤
│ Remote object store (S3)                │  authoritative, slow
└─────────────────────────────────────────┘
```

Interview points:
- Query planning hits the metadata cache first — raw S3 file listing is slow; with 10K+ partitions it can dominate latency.
- Block cache hit rate on recent data is typically high; cold historical queries are where tail latency blows up.
- Pre-warm caches on node addition ("cache hydration") rather than cold-starting.

### 7.5 Inference-adjacent analogs

Same pattern, different scale:

- **Feature stores** (Feast, Tecton): offline store on object storage for training; online store on Redis/DynamoDB for serving. Same data, two tiers for two latency regimes.
- **Telemetry / request-log lakes:** inference request/response logs in Parquet on S3; ClickHouse or Trino for ad-hoc. SLO is "queryable within minutes," not milliseconds.
- **Model artifact serving:** weights in S3/GCS (cold) → warm cache on shared NVMe → hot copy in GPU HBM per worker. Same tiering principle, different latency scales.
- **KV cache tiering for LLM inference:** HBM (µs) → host DRAM (10s of µs) → local NVMe (100s of µs) → distributed object store (ms). Same eviction/promotion questions, same decoupled-compute insight (serve a request from any node that can pull the prefix fast enough). This is active research territory — prefix cache offloading, paged/tiered KV.

Honest framing for an inference-infra interview: *analytical systems pioneered decoupled storage/compute a decade ago; inference systems are importing the ideas now because HBM is the new "RAM" and NVMe + object store are the new "disk."*

### 7.6 Interview reasoning — when do I pick decoupled?

- **Pick decoupled** when: workload is bursty or bimodal, multiple teams share data, storage ≫ sustained compute, cost elasticity matters, or you're on a cloud with cheap durable object storage.
- **Stick with shared-nothing** when: p95 at single-digit ms is non-negotiable, workload is steady-state (always-on high-QPS dashboards), network to object store would dominate, or on-prem without a good object store.
- **Hybrid is the common reality:** hot tier on shared-nothing for user-facing queries; cold tier in object storage queried by a separate analytics cluster. Ingestion writes to both.

### 7.7 First-order bottlenecks in decoupled systems

1. **Object store throughput/latency** — you're GET-limited on scans. Parallelize aggressively; use ranged GETs, not whole-file fetches.
2. **Cache hit rate** — one miss on a 1 GB file blows your latency budget.
3. **Metadata planning** — listing 100K partitions from S3 is slow. Manifest files (Iceberg) fix this.
4. **Cold start** — warming a fresh cluster's caches from S3 on first queries.
5. **Request amplification** — each Parquet file = header + footer + column chunks, i.e. multiple round trips if you're not careful. Coalesce reads.

---

## 8. Query patterns and bottlenecks

### 8.1 The five query shapes

| Pattern | What dominates | What to exploit |
|---|---|---|
| Selective filter (1 in 10⁶) | index probe | inverted/bitmap index, sort + zone map |
| Broad scan | sequential I/O | columnar, compression, projection, parallelism |
| GROUP BY aggregation | hash aggregation, memory | dictionary encoding, pre-aggregation, partition-local agg |
| Top-K | partial heap | compute top-K per partition, merge |
| COUNT DISTINCT | memory/exactness | HyperLogLog (approx), sort+stream (exact, expensive) |

### 8.2 Distinct counts (high level)

- Exact distinct = keep a hash set. Memory = O(distinct), which can be huge, and doesn't parallelize without shuffle.
- Approx distinct (HLL, Theta) = fixed ~16KB sketch, mergeable, 1–2% error. Default to it for dashboards.
- Interview answer: "Unless the PM requires exactness, use HLL. It merges across partitions and fits in memory."

### 8.3 Joins at interview depth

OLAP joins are constrained relative to OLTP. You need to know:

- **Broadcast join:** small side replicated to all workers, big side scanned locally. Fast when one side is small (< few GB). No shuffle.
- **Shuffle (hash) join:** both sides repartitioned on join key. Expensive network; skew kills you.
- **Sort-merge join:** both sides sorted on join key. Good when data is already sorted or must be sorted anyway.
- **Co-located join:** both tables partitioned by the same key on the same nodes → local join, no shuffle. The one you design for when possible.

```
Broadcast:                     Shuffle:
   small                          left              right
     │                              │                 │
 replicate                     hash-partition    hash-partition
     │ │ │                         │ │ │            │ │ │
     ▼ ▼ ▼                         ▼ ▼ ▼            ▼ ▼ ▼
 [W1][W2][W3]                    [W1][W2][W3]  ←←←←  network
 scan big locally                 local hash join on each worker
 join + emit                      (watch skew!)
```

### 8.4 Why joins are constrained or precomputed in analytics

- Shuffles move terabytes; bandwidth and skew dominate latency.
- Star schema dim tables are usually small → broadcast joins are fine.
- When dims are huge, systems often **denormalize** (flatten at ingest) or **pre-join** into a wide table. Pinot/Druid famously restricted joins for this reason (loosening over time).
- Answer: "In analytics, you design the physical layout to avoid shuffles. Either co-locate on the join key, broadcast small dims, or denormalize."

### 8.5 First-order bottlenecks

In rough order of what kills queries:

1. **Scan volume** — too many bytes, not enough pruning.
2. **Shuffle** — cross-node data movement on joins and high-cardinality GROUP BY.
3. **Skew** — one partition or key has 80% of data; one worker becomes the critical path.
4. **Memory pressure** — hash aggregation spilling to disk.
5. **Query planning overhead** — thousands of tiny partitions/files.
6. **Compaction interference** — background work starving foreground queries.

Your design should name the bottleneck and kill it explicitly.

---

## 9. Interview reasoning patterns

These are the reusable scripts.

### 9.1 Do I need an analytical store vs OLTP?

```
Start
  │
  ├── Query touches < 10K rows, needs ACID on row updates? ─── OLTP
  │
  ├── Query scans millions+, aggregations, read-mostly? ───── OLAP
  │
  ├── Mixed? Small hot (last hour) + big historical? ──────── OLTP for hot,
  │                                                           OLAP for cold,
  │                                                           CDC between them
  │
  └── User-facing, high QPS, bounded query shape, fresh? ──── Pinot/Druid class
                                                              (or pre-agg + cache)
```

### 9.2 Does columnar help here?

Yes if: wide schema, queries project few columns, aggregation-heavy, read-mostly, bulk ingest.
No if: row updates are frequent, queries need full rows, narrow schema.

### 9.3 Are indexes worth it?

- Default: no index beyond sort + zone maps + dictionary.
- Add bitmap when you have 3+ low-cardinality filters ANDed together frequently.
- Add inverted when you have very selective equality on high-cardinality keys that are hot.
- Reject: "I'll index everything." Indexes cost storage, write time, and compaction work.

### 9.4 Should I pre-aggregate?

- Yes if: query shape is fixed, QPS is high, latency SLO is tight, dimension set is bounded.
- No if: ad-hoc exploration, unbounded dimension combinations, low QPS.
- Middle ground: materialize rollups for the top-N known queries; fall back to raw for the rest.

### 9.5 Freshness vs latency

Three levers:

1. **Segment size** — smaller = fresher, slower queries.
2. **Compaction cadence** — more frequent = fresher sealed state, more background load.
3. **Real-time tier size** — bigger RT window = more fresh data queryable, but slower since RT is less optimized.

Name the SLO first (e.g., p95 < 500ms, data ≤ 30s old), then pick the levers.

### 9.6 Ingestion + fast queries simultaneously

- Separate write path and read path processes.
- Immutable segments so readers never lock against writers.
- Object store as durable landing zone; hot tier on local disk for serving.
- Background compaction with its own resource budget, isolated from query workers.

### 9.7 When is Pinot/ClickHouse/Druid thinking appropriate?

- **User-facing analytics** with high QPS, sub-second p95, bounded queries → Pinot/Druid.
- **Internal analytics** with broad ad-hoc queries, tolerant of seconds-to-minutes → ClickHouse, Snowflake, BigQuery, Trino over Iceberg/Delta.
- **Logs / traces at scale** → columnar + some text indexing; ClickHouse, Pinot, or Elasticsearch depending on shape.

### 9.8 First bottlenecks / operational pains

- Partition key selection gone wrong → no pruning → everything scans full table.
- Skew on a tenant_id or user_id causing one hot shard.
- Compaction backlog causing segment count to explode, metadata planning time to spike.
- Real-time segment seal latency defining your freshness SLO.
- Tiered storage cache misses on cold data when someone runs a wide historical query.

---

## 10. Common candidate mistakes

1. **"Use a data warehouse"** without specifying latency, freshness, or QPS. Snowflake is not a dashboard engine out of the box.
2. **Ignoring row vs column** — proposing scans over Postgres for analytics at TB scale.
3. **Handwaving "add an index"** without naming the index type, what it indexes, its update cost, or its selectivity assumption.
4. **Skipping partition pruning** — proposing a big table without saying what's partitioned and how.
5. **Assuming indexes always help** — indexes on low-selectivity columns are strictly negative.
6. **Ignoring ingest/query tradeoff** — proposing low-latency fresh queries without saying how writes don't collide with reads.
7. **Assuming OLTP scales to dashboards** — Postgres with read replicas does not solve 10 TB scan dashboards.
8. **Ignoring skew and scan cost** — uniform workload assumption breaks in real systems; tenant-based skew is the norm.
9. **Proposing heavy joins** without discussing broadcast vs shuffle, co-location, or denormalization.
10. **Over-engineering** — building a Pinot clone when a Postgres read replica + materialized view would suffice.
11. **Talking about "caching" as a primary strategy** — for dashboards with varied parameters, cache hit rate is often low; fix the storage layout instead.
12. **Forgetting compaction** — systems need it; ignoring it means ignoring half the operational pain.

---

## 11. Final cheat sheet

### 11.1 OLTP row store vs OLAP column store

| Axis | OLTP row store | OLAP column store |
|---|---|---|
| Layout | Rows contiguous | Columns contiguous |
| Read pattern | Point / small range | Large scan on few columns |
| Write pattern | Random, per-row, ACID | Bulk append, immutable segs |
| Compression | Modest | 5–20× |
| Index default | B-tree PK + secondary | Partition + sort + zone map |
| Concurrency | Thousands of small txns | Tens of large queries |
| Latency | ms per query | ms–seconds over 10⁹ rows |
| Update cost | Cheap | Expensive (tombstone + compact) |
| Canonical fit | Transactions | Analytics, dashboards, logs |
| Products | Postgres, MySQL, Spanner | ClickHouse, Druid, Pinot, BigQuery, Snowflake |

### 11.2 Partitioning vs sorting vs indexing vs pre-aggregation

| Technique | Granularity | Kills | Write cost | Storage cost | Freshness impact |
|---|---|---|---|---|---|
| Partitioning | Directory / file | Whole-file scans | Low | None | None |
| Sorting / clustering | Within partition | Block scans (via zone map) | Medium (must sort) | None | Sort may delay seal |
| Zone map / min-max | Block-level | Block reads | Trivial | Trivial | None |
| Bitmap index | Column-level | Multi-predicate scan | Medium | Moderate | Must rebuild on update |
| Inverted index | Term-level | Needle queries | High | High (rivals data) | High — costly to update |
| Dictionary encoding | Column-level | Decompression cost; speeds GROUP BY / eq | Low | Negative (saves storage) | None |
| Pre-aggregation | Query-level | Raw table scan | High (precompute) | High (cube blowup) | Rebuild window = staleness |

### 11.3 Decision framework

```
1. Workload:  OLTP? OLAP? mixed?
       ↓
2. Query shape:  scans / aggregations / selective / joins / top-K?
       ↓
3. Freshness SLO:  seconds? minutes? hours?
       ↓
4. QPS:  user-facing high QPS?  internal low QPS?
       ↓
5. Storage layout:  row or columnar?
       ↓
6. Partition key:  what's in 90% of WHERE clauses?  (usually time)
       ↓
7. Sort key:  next most selective frequent filter
       ↓
8. Indexes:  zone map by default;  bitmap/inverted only if query demands it
       ↓
9. Pre-aggregation:  fixed query shape + high QPS + tight SLO → yes
       ↓
10. Ingest path:  immutable segments, background compaction,
                   isolated from query path
       ↓
11. Name bottlenecks:  scan volume, shuffle, skew, memory, compaction
       ↓
12. Operational:  tiered storage, rebalancing, compaction budget,
                   monitoring for segment count & skew
```

### 11.4 Ten likely interview questions with short strong answers

**Q1. When would you not use Postgres for analytics?**
A. When working set exceeds RAM and per-query scans exceed ~10 GB with sub-second SLOs, or when ingest rate plus dashboard QPS saturates random-I/O row storage. At that point columnar + partitioning is structurally right, not just a tuning issue.

**Q2. Why is columnar better for analytics?**
A. Scans read only projected columns (bytes_read ∝ cols_read/cols_total), column-homogeneous data compresses 5–20× better, and execution vectorizes cleanly. Tradeoff: row updates and row reconstruction are expensive, so segments are immutable.

**Q3. How do you pick a partition key?**
A. The dimension that appears in 90%+ of WHERE clauses, usually time (day or hour). Low enough cardinality to avoid small-file explosion, high enough to give meaningful pruning. For multi-tenant, sometimes time + tenant_id, but watch skew.

**Q4. When is a bitmap index worth it?**
A. Low-to-medium cardinality columns used in ANDed predicates with moderate-to-high selectivity. Classic case: `country AND device AND status` dashboard filters. Not worth it on high-cardinality or write-heavy columns.

**Q5. How do you handle updates in a columnar store?**
A. Immutable segments + delete vectors (or tombstones) + re-insert, merged at read time, with periodic compaction to rewrite. The system is append-heavy by design; true row-level UPDATE at high rate is a signal you picked the wrong store.

**Q6. Freshness vs query latency — how do you balance them?**
A. Real-time tier holds recent data in small, less-optimized segments for low freshness latency. Historical tier holds sealed, sorted, compressed, indexed segments for fast queries. Queries merge both. Tune segment-seal cadence and compaction frequency to meet the SLO.

**Q7. Why are joins constrained in analytics systems?**
A. Shuffles move huge data volumes across the network, and skew creates stragglers. Designs avoid shuffles via (a) co-located partitioning on the join key, (b) broadcast joins for small dims, or (c) denormalization at ingest into wide tables. Many early OLAP systems forbade joins for exactly this reason.

**Q8. When would you pre-aggregate?**
A. Fixed query shape, bounded dimension set, high QPS, tight latency SLO. Pre-aggregation converts per-query work from O(rows) to O(dimensions). Don't pre-aggregate when exploration is ad-hoc or the dimension cube would explode.

**Q9. What's the first bottleneck you'd watch for?**
A. Scan volume not reduced by partition pruning — usually a bad partition key. Then shuffle volume and skew on joins and high-cardinality GROUP BYs. Then compaction interference on the query path.

**Q10. Pinot/Druid vs ClickHouse vs Snowflake — when does each fit?**
A. Pinot/Druid for user-facing, high-QPS, sub-second analytics with bounded query shapes. ClickHouse for internal analytics and logs with heavy ad-hoc SQL at seconds-latency. Snowflake/BigQuery for warehouse-scale ad-hoc analytics where minutes are acceptable and ease-of-use and decoupled storage matter. The axes are QPS, freshness, query flexibility.

---

## One-page summary

```
Decide:  OLTP or OLAP?
         └─ OLAP →
              Storage:     columnar, immutable segments
              Partition:   the filter in 90%+ of queries (usually time)
              Sort:        next most selective frequent filter
              Index:       zone map + dict by default;
                           bitmap/inverted only if specific query demands
              Pre-agg:     if fixed shape + high QPS + tight SLO
              Ingest:      separate from query path;
                           real-time tier + sealed historical tier;
                           background compaction
              Bottlenecks: scan volume, shuffle, skew, memory, compaction

Always frame choices as:  how does this reduce scan bytes?
```