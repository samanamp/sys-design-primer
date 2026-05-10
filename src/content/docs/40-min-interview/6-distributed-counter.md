---
title: "Real-time metrics / counting system"
description: "Real-time metrics / counting system"
---

Design a system that counts events at scale — think view counts, like counts, ad impressions. 100B events/day, near-real-time read freshness, queries that aggregate across counters. Walk me through it.

---

## 1. Reframing

**This is not one problem. It is two read paths fed by one stream pipeline, with a hot-key adversary and an integrity contract.**

Most candidates hear "counters" and reach for sharded Redis. That answer fails the moment the interviewer asks "now sum 10K counters in the last hour." A KV store cannot serve cross-counter aggregations at request time — 10K random reads is 100ms+ p99 even with perfect caching, and the query pattern fans out in ways no caching layer absorbs. **[STAFF SIGNAL: two-output-architecture]** The right architecture has *two* serving stores fed by one streaming pipeline:

1. **Per-counter KV store** — sub-10ms point reads ("views for video X").
2. **OLAP rollup store** — pre-aggregated time-bucketed materializations for cross-counter queries ("top 100 in last 5 min", "sum impressions by country").

**[STAFF SIGNAL: stream-pipeline-as-source-of-truth]** The pipeline is the source of truth; both stores are derived materializations optimized for opposite access patterns. An audit log of raw events sits beneath everything — *that* is the canonical truth, against which the live stores are reconciled.

The operational test is the **hot counter** — a viral video at 1M events/sec for hours. Naive sharding melts a single shard. Real mitigation is layered: producer-side batching, key splitting, two-phase combiners, read-side caching, with concrete activation thresholds.

**Freshness must be defined.** I'll commit: **p95 freshness ≤ 30 seconds for the per-counter path, ≤ 60 seconds for OLAP rollups at minute granularity, with a billing-grade tier at exactly-once and end-of-hour close.** Sub-second freshness for everyone is a 10x cost tier I would refuse to design without product justification. **[STAFF SIGNAL: freshness-as-product-tier]**

## 2. Scoping

Before architecture, commit to assumptions. **[STAFF SIGNAL: scope negotiation]**

| Axis | Commitment |
|---|---|
| Counter types | Views, likes, impressions. Append-only in normal operation. |
| Decrements | Yes, but rare (un-like). Modeled as `delta = -1` events through the same pipeline, not as direct mutations. |
| Integrity | Two tiers. **Default tier**: 0.01% error budget, effectively-once via dedup. **Billing tier** (ad impressions): exactly-once via transactional commits + nightly reconciliation against an audit log. ~2% of counter volume, ~30% of cost. |
| Freshness | Default 30s p95 per-counter, 60s OLAP. Billing tier closes hourly. |
| Read mix | Heavily Zipfian. Top 0.01% of counters take ~30% of reads. Cross-counter queries are 5% of QPS but 50% of compute cost. |
| Cardinality cap | 1B distinct counters live; 10B with cold-tier offload. Per-tenant cardinality budget enforced. |
| Late data | Mobile replay arrives up to 24h late. Watermark at 1h; past-watermark events go to a late-update path that mutates historical OLAP buckets. |

**Out of scope by my call**: real-time anomaly detection, fraud/bot filtering (assume upstream), full SQL OLAP — only the pre-defined query shapes plus a controlled escape hatch.

## 3. Capacity Math

**[STAFF SIGNAL: capacity math]**

```
INGEST
─────────────────────────────────────────────
events/day:       100B
events/sec avg:   ~1.16M
peak burst (5x):  ~6M events/sec
event size:       200 B raw, ~80 B compressed
ingest BW peak:   ~480 MB/s compressed, ~1.2 GB/s raw
raw/day:          ~20 TB raw, ~5 TB compressed

COUNTERS
─────────────────────────────────────────────
distinct counters live:    1B
per-counter row:            ~64 B (key + value + metadata)
KV store hot working set:   ~64 GB live + 3x replication = ~200 GB
KV store cold tier:         ~640 GB on SSD-backed tier

HOT KEYS
─────────────────────────────────────────────
top-100 counters peak QPS:  1M+ events/sec each
top-1 counter (viral peak): 5M events/sec sustained 1h

OLAP ROLLUPS
─────────────────────────────────────────────
1-min buckets:  1B counters × 1440/day × ~32 B = 46 TB/day raw
                Most counters have 0 events/min → sparse storage:
                ~5% of counters active per minute → ~2.3 TB/day
1-hr rollup:    ~100 GB/day
1-day rollup:   ~5 GB/day
retention:      1-min for 7d, 1-hr for 90d, 1-day forever

QUERY SCAN VOLUMES
─────────────────────────────────────────────
"sum 10K counters last 1h": 
  at 1-sec resolution:   36M points  → infeasible
  at 1-min resolution:   600K points → ~200ms
  at 1-hr resolution:    10K points  → <10ms ✓

AUDIT LOG
─────────────────────────────────────────────
5 TB compressed/day × 365d = 1.8 PB/year
S3 standard: ~$45K/month — non-trivial, tractable
```

The 1-hr rollup is the answer for almost all cross-counter queries. The 1-sec rollup is what a junior engineer designs and what a staff engineer refuses to build.

## 4. High-Level Architecture

```
                    PRODUCERS (web, mobile, ad servers)
                              │
                              │  local pre-agg (1s window for hot keys)
                              ▼
                  ┌────────────────────────────┐
                  │   Kafka  (partitioned by   │
                  │   counter_key hash)        │
                  │   1024 partitions, RF=3    │
                  └────────┬───────────────────┘
                           │
              ┌────────────┴────────────┐
              ▼                         ▼
   ┌────────────────────┐    ┌────────────────────┐
   │  Audit Log Sink    │    │  Flink Stream      │
   │  → S3 (Parquet,    │    │  Processor         │
   │    partitioned     │    │  - per-key state   │
   │    by date/source) │    │  - windowed agg    │
   │  CANONICAL TRUTH   │    │  - dedup           │
   └────────────────────┘    │  - hot-key detect  │
                             │  - top-K maint.    │
                             └────┬───────────┬───┘
                                  │           │
                  per-counter Δ   │           │  windowed rollups
                  + top-K updates │           │  (1-min granularity)
                                  ▼           ▼
                        ┌──────────────┐  ┌──────────────┐
                        │  KV Store    │  │  OLAP Store  │
                        │  (Scylla/    │  │  (ClickHouse │
                        │   Aerospike) │  │   /Pinot)    │
                        │  point reads │  │  rollup tier │
                        │  <10ms p99   │  │  hierarchy   │
                        └──────┬───────┘  └──────┬───────┘
                               │                 │
                        ┌──────┴─────────────────┴──────┐
                        │   Query API (router)          │
                        │   - point read → KV           │
                        │   - aggregate → OLAP          │
                        │   - top-K → OLAP top-K MV     │
                        └───────────────────────────────┘
                                       ▲
                                       │
                        ┌──────────────┴──────────────┐
                        │  Reconciliation (nightly)   │
                        │  audit log → recompute →    │
                        │  diff vs OLAP, alert >0.01% │
                        └─────────────────────────────┘
```

The pipeline forks at Kafka: **everything** lands in S3 first (the audit log is a sink, not an afterthought), and Flink consumes the same partitions to produce live materializations. **[STAFF SIGNAL: invariant-based thinking]** Invariants enforced: (a) every event has exactly one row in S3; (b) KV and OLAP are eventually consistent with S3 within freshness SLO; (c) every event ID is processed effectively-once into the live stores.

## 5. The Two-Output Split

The defining decision. **Why not one store**:

- **KV store doing aggregation**: "sum 10K counters" = 10K random reads. Even at 0.5ms each with batching, ~50ms p50 and 200ms+ p99 due to tail amplification. Worse, it pins read capacity that should be serving point reads. Caching helps zero — the query set is open.
- **OLAP store doing point reads**: ClickHouse / Pinot serving "view count for video X" at 1M QPS is grotesque. Columnar stores have ~10ms minimum query overhead from segment scanning. Wrong tool.

**What goes where**:

```
┌───────────────────────────────┬───────────────────────────────┐
│  PER-COUNTER KV STORE         │  OLAP ROLLUP STORE            │
├───────────────────────────────┼───────────────────────────────┤
│ schema:                       │ schema:                       │
│   key = counter_id            │   (counter_id, time_bucket,   │
│   value = current_count       │    dim_country, dim_device,   │
│           + last_updated      │    count, sum, hll_users)     │
│           + version           │                               │
│                               │ buckets: 1-min, 1-hr, 1-day   │
│ access:                       │ access:                       │
│   point GET <10ms p99         │   range scan + group by       │
│   batch GET (multi-get)       │   100ms p95 typical           │
│                               │                               │
│ workload:                     │ workload:                     │
│   95% of read QPS             │   5% of read QPS, 50% compute │
│   1M reads/sec peak           │   complex aggregations        │
│                               │                               │
│ writes:                       │ writes:                       │
│   merge-by-counter from Flink │   bulk insert windowed agg    │
│   batched ~100ms commit       │   from Flink, 1-min cadence   │
│                               │                               │
│ engine: Scylla (CAS counters) │ engine: ClickHouse            │
│         or Aerospike          │   ReplacingMergeTree for late │
│                               │   updates; AggregatingMergeT- │
│                               │   ree for HLL                 │
└───────────────────────────────┴───────────────────────────────┘
```

**Top-K is not a query, it's a materialization.** "Top 100 most-viewed in last 5 min" cannot be served by scanning all counters. Flink maintains a sliding-window top-K (heap-based, per dimension slice) and writes it to the OLAP store as a first-class table. Query latency: O(K), not O(N).

**[STAFF SIGNAL: rejected alternative]** Considered: (a) one OLAP store with a hot-row cache for point reads — rejected, the point-read SLO requires sub-10ms which OLAP stores cannot guarantee under load; (b) Druid for both — same problem, plus Druid's mutability story is weak for late updates; (c) Pinot with star-tree indices — viable, kept as a swap candidate if ClickHouse sharding becomes painful.

## 6. Pipeline Design

```
PRODUCERS ──[event_id, counter_key, ts, dims, delta]──▶  KAFKA
                                                         │
                                                  partition = hash(counter_key) % 1024
                                                         │
                                                         ▼
   ┌──────────────────────────────────────────────────────────────┐
   │  FLINK (1024 parallel tasks, one per partition)              │
   │                                                              │
   │  per-task state (RocksDB-backed, checkpointed every 30s):    │
   │    ┌──────────────────────────────────────────────┐          │
   │    │ dedup: bloom filter (24h) + exact LRU (1h)   │          │
   │    │ counters: hashmap<counter_key, {count, win}> │          │
   │    │ minute_buckets: per-key sliding window       │          │
   │    │ topK: dimension-sliced bounded heaps         │          │
   │    │ hot_detector: count-min sketch over keys     │          │
   │    └──────────────────────────────────────────────┘          │
   │                                                              │
   │  per-event: dedup → update counter → update window           │
   │  every 1s: emit per-counter delta batch to KV writer         │
   │  every 60s: emit minute rollup batch to OLAP writer          │
   │  every 60s: emit topK snapshot to OLAP writer                │
   └──────────────────────────────────────────────────────────────┘
                       │                            │
                       ▼                            ▼
                  KV writer pool              OLAP writer pool
                  (batch 1000 ops)            (bulk insert)
                       │                            │
                       ▼                            ▼
                    Scylla                     ClickHouse
```

**Partitioning**: hash by `counter_key`. All events for a counter pin to one task — eliminates cross-task coordination, makes per-counter increments deterministic. Throughput: 1.2M events/sec / 1024 tasks = ~1200 events/sec/task average, peaks ~6K/sec.

**Why Flink over Kafka Streams or Spark**: Flink's exactly-once via two-phase commit + RocksDB state backend handles the dedup window (1h exact + 24h bloom) at scale. Kafka Streams' state-store story is weaker; Spark Structured Streaming's micro-batch model adds 5-10s of latency we can't absorb. **[STAFF SIGNAL: rejected alternative]**

**Batched commits**: writing every event to the KV store at 1.2M writes/sec is impossible — Scylla shards would saturate. Batching at 1s windows reduces write IOPS by ~1000x (most counters get many events/window). Trade: 1s of additional freshness lag. Worth it.

**State size per task**: ~1M live counters / 1024 tasks ≈ 1K counters/task active, plus dedup window (~3.6M event IDs/hour exact, ~86M in bloom). Comfortable in RocksDB at ~500MB/task.

**Checkpointing**: every 30s to S3. Recovery time on task failure: ~60s (replay from last checkpoint + re-consume from Kafka). Within freshness SLO.

## 7. Hot-Counter Mitigations

**The adversary**: a viral video at 5M events/sec. With hash-partitioning, all 5M land on one Flink task. That task's CPU melts; its checkpoint balloons; backpressure stalls upstream.

**[STAFF SIGNAL: hot-counter discipline]** Five layered mitigations with thresholds:

```
LAYER 1: PRODUCER-SIDE PRE-AGGREGATION  (always on)
─────────────────────────────────────────────────────
At edge / web server: 1s window, per-(counter, source).
Emits 1 batched event with delta=N instead of N events.
Reduces 5M events/sec to ~5K events/sec for hot keys
(assuming ~1000 producer instances).

[STAFF SIGNAL: producer-side optimization]


LAYER 2: HOT-KEY DETECTION  (count-min sketch in Flink)
─────────────────────────────────────────────────────
Per-task CMS tracks per-key event rate.
Threshold: >10K events/sec/key → flag as hot.
Hot-key set published to a control topic; producers
subscribe and apply Layer 3.


LAYER 3: KEY SPLITTING  (>10K events/sec)
─────────────────────────────────────────────────────
For hot keys, producer rewrites:
  counter_key = "video_42"
  → counter_key = "video_42:shard_" + (rand() % N)

Where N scales with rate: 10K/sec → N=4, 100K → N=16,
1M+ → N=64. Spreads writes across N Flink tasks.

Read path: KV store has a hot-key registry; on read of
"video_42", fan out to N sub-keys, sum, return.
Read fanout cost: N point reads. Cached at edge with 1s
TTL — for hot keys, every read is from cache.

   PRE-SPLIT (melts):                POST-SPLIT:
   ───────────────                   ──────────
   all events ──▶ task_42            shard_0 ──▶ task_42
                  💀                 shard_1 ──▶ task_137
                                     shard_2 ──▶ task_891
                                     ...
                                     shard_15 ──▶ task_512


LAYER 4: TWO-PHASE AGGREGATION  (>100K events/sec)
─────────────────────────────────────────────────────
For super-hot keys, in addition to splitting:

  Stage 1 (combiner): each upstream task aggregates
    locally over 1s windows, emits partial sum.
  Stage 2 (reducer): single task per logical counter
    sums partial sums.

  Hot key load:
    Stage 1: 1024 tasks × ~5K ev/s/task = full parallelism
    Stage 2: receives 1024 partial sums per second per
             hot key — trivial load.

           events
             │
   ┌─────────┼─────────┐
   ▼         ▼         ▼
 task_0   task_1 ... task_1023      ← Stage 1 combiner
   │         │         │              local 1s sums
   └─────────┼─────────┘
             ▼
        reducer_42                    ← Stage 2 reducer
             │                          authoritative count
             ▼
        KV store + OLAP


LAYER 5: READ-SIDE CACHING  (always on for hot keys)
─────────────────────────────────────────────────────
Hot keys cached at API tier (1s TTL) + CDN edge (5s TTL).
For top-1000 counters, ~99% of reads served from cache.
Stale-while-revalidate to absorb thundering herd.
```

**Activation thresholds (concrete)**:
- < 10K ev/s: vanilla path, no mitigation.
- 10K–100K: Layer 3 (key splitting, N=4–16).
- 100K–1M: Layers 3 + 4 (splitting + two-phase agg).
- > 1M: Layers 1+3+4+5 mandatory; Layer 1 is enforced via producer SDK.

**Failure modes**: hot-key detection lags reality by ~10s (CMS staleness). During the lag, one task absorbs the load — handled by per-task headroom (5x avg capacity). If that's exceeded, backpressure stalls Kafka consumption on that partition only — other counters unaffected. **[STAFF SIGNAL: blast radius reasoning]**

## 8. Cross-Counter Aggregation Path

```
QUERY: "sum views, top 100 videos, last 5 min, group by country"
                          │
                          ▼
                ┌──────────────────────┐
                │  Query Planner       │
                │  - parse time range  │
                │  - pick rollup tier  │
                │  - check top-K MV    │
                └──────┬───────────────┘
                       │
       ┌───────────────┼───────────────────┐
       ▼               ▼                   ▼
   exact span    span < 1h →        top-K query →
   < 5 min       use 1-min          read top-K MV
   use 1-min     buckets             directly
   buckets       (~600K rows         (~100 rows
                  to scan)            to scan)


ROLLUP HIERARCHY (in OLAP store)
─────────────────────────────────────────────
counter_minute  : (counter_id, minute, dims, count, hll)
counter_hour    : rollup of counter_minute (60→1)
counter_day     : rollup of counter_hour (24→1)

topk_minute_5m  : top-1000 counters per dim slice,
                  5-min sliding window, refreshed every 60s
topk_hour       : top-1000 per dim slice, 1h window
topk_day        : top-1000 per dim slice, 1d window


APPROX AGGREGATIONS  [STAFF SIGNAL: approximate-when-appropriate]
─────────────────────────────────────────────
Distinct users per video:    HyperLogLog (1.5KB/HLL,
                              error ~2%)
Frequency estimates:          Count-Min Sketch
HLLs are mergeable: hour HLL = merge of 60 minute HLLs.
This is why HLL is mandatory — exact distinct counts at
1B counters × 1440 buckets × billions of users is
infeasible storage.
```

**[STAFF SIGNAL: rollup-hierarchy precision]** The planner picks the coarsest tier whose buckets fully cover the query range. "Last 5 min" → 1-min tier. "Last 24h" → 1-hr tier. "Last 30 days" → 1-day tier. This makes scan cost bounded by ~1500 rows for any time range from 1-min to forever.

**Top-K maintenance** is the non-obvious staff move. A naive query plan for "top 100 last 5 min" scans all counters with activity, sorts, takes top-100 — at 50M active counters in a 5-min window, that's 50M rows scanned per query. Maintaining top-K as a stream-state structure in Flink (one bounded heap per dimension slice, updated incrementally) reduces query cost to O(K) and shifts work to write-time where it amortizes. Trade: top-K is approximate at the boundary — counters churning around rank 100 may flicker. Acceptable.

**Ad-hoc query escape hatch**: queries outside the pre-aggregated dimensions (e.g., "group by user_agent_string", which we didn't roll up) are routed to a wide-event store (the audit log queryable via Athena/BigQuery). Latency: minutes. Cost: per-query. Rate-limited per tenant. The contract is: pre-aggregated dimensions are SLO'd; ad-hoc is best-effort.

## 9. Idempotency and Effectively-Once

**[STAFF SIGNAL: effectively-once honesty]** True exactly-once across a distributed system with retries and partial failures is unachievable in the general case — FLP, two generals, the usual. What we *can* achieve is **effectively-once**: every event contributes exactly once to the final counter value, regardless of retries.

```
EVENT ID FORMAT
─────────────────────────────────────────────
event_id = hash(producer_id || session_id || event_seq)
         128 bits, globally unique with negligible
         collision probability

DEDUP IN FLINK (per task, per counter_key partition)
─────────────────────────────────────────────
Stage 1: bloom filter, 24h window
   - 1.2M events/sec × 86400s / 1024 tasks ≈ 100M ids/task
   - bloom: 1% FP, ~120 MB/task — fits
   - bloom HIT → drop event (false positives 1% drop = OK
     because Stage 2 confirms)

Stage 2: exact LRU set, 1h window
   - ~4M ids/task in 1h
   - ~64 MB/task with hashed keys
   - exact dedup for events near-in-time
   
Past 24h: assume duplicate impossible (producer SLA);
if it happens, late-data path handles it.


WRITE IDEMPOTENCY TO KV
─────────────────────────────────────────────
Writes are increment-by-delta, not set-to-value.
Each write batch carries a batch_id. KV store records
last-applied batch_id per shard; rejects re-applies.

Flink's two-phase commit:
  1. pre-commit: batch staged with batch_id
  2. commit: on Flink checkpoint success, batch applied
  3. on failure during commit: replay; KV rejects dup
     by batch_id


BILLING TIER (ad impressions)  
─────────────────────────────────────────────
Different SLA. The advertiser pays per impression;
double-counting = fraud risk; under-counting = revenue
loss.

- Producer: synchronous ack from Kafka before serving
  the ad — no fire-and-forget.
- Pipeline: separate Flink job, exactly-once sink to
  a transactional store (e.g., CockroachDB or
  Spanner-class).
- Reconciliation: hourly job re-derives counts from
  audit log, diffs against live store; >0.001%
  variance pages on-call.
- Cost: ~10x the default tier. Customer pays.
```

The default tier accepts ~0.01% error (bloom FPs, late events past watermark, rare dropped writes). The billing tier pays for the difference. Pretending the default tier is exactly-once is dishonest; pretending the billing tier is the same as the default is negligent.

## 10. Late-Arriving Data and Backfill

```
WATERMARK STRATEGY
─────────────────────────────────────────────
event_time = ts on event
processing_time = wall clock at Flink

watermark = max(event_time seen) - 5s slack
window_close = watermark + 1h grace period

  ───────── time ──────────▶
   [─────── window ───────] [──── next window ────]
                          │ │
              window event│ │watermark advances 1h
                       arrives                    after
                       "on time"                  window
                                                  closes


LATE-DATA POLICY (3 zones)
─────────────────────────────────────────────
Zone A: event_time within current open window
  → normal aggregation

Zone B: event_time within last 1h, window closing
  → still aggregated; window emit is after grace

Zone C: event_time > 1h late (up to 24h)
  → "late path": event still goes to KV (counter is
    monotonic, can be incremented anytime)
  → OLAP minute-bucket UPSERTed via ReplacingMergeTree
  → metric: late_event_count, alert if anomaly

Zone D: event_time > 24h late
  → drop with metric. Audit log still has it for
    potential backfill.


BACKFILL FLOW (the bug-correction scenario)
─────────────────────────────────────────────

  Day 0: bug deployed; counts inflated 2x for 3h
  Day 14: bug discovered

  ┌──────────────────────────────────────────────┐
  │ 1. AUDIT LOG (S3, immutable)                 │
  │    raw events from Day 0 — UNAFFECTED        │
  │    by the bug (bug was in pipeline, not      │
  │    producer)                                 │
  └──────────────────┬───────────────────────────┘
                     │
                     ▼
  ┌──────────────────────────────────────────────┐
  │ 2. SHADOW PIPELINE                           │
  │    spin up clean Flink job (fixed code)      │
  │    consume from S3 (audit log) for affected  │
  │    time range only                           │
  │    write to SHADOW tables (kv_v2, olap_v2)   │
  └──────────────────┬───────────────────────────┘
                     │
                     ▼
  ┌──────────────────────────────────────────────┐
  │ 3. VALIDATE                                  │
  │    diff shadow vs live for unaffected hours  │
  │      → should match within 0.01%             │
  │    diff shadow vs live for affected hours    │
  │      → expect ~50% (the 2x bug)              │
  └──────────────────┬───────────────────────────┘
                     │
                     ▼
  ┌──────────────────────────────────────────────┐
  │ 4. ATOMIC SWAP (versioned)                   │
  │    OLAP: rename tables (v1 → v_old, v2 → v1) │
  │    KV: per-key correction = (v2 - v1)        │
  │      applied via increment events            │
  │    publish "correction event" with version   │
  │      so dashboards can flag                  │
  └──────────────────┬───────────────────────────┘
                     │
                     ▼
  ┌──────────────────────────────────────────────┐
  │ 5. COMMUNICATE                               │
  │    PMs get alert: "counts for window X       │
  │    corrected at time Y, magnitude Z"         │
  │    historical dashboards show a delta marker │
  └──────────────────────────────────────────────┘
```

**[STAFF SIGNAL: backfill-as-architectural]** Backfill is not "we'll figure it out" — it's a designed protocol with a shadow pipeline, validation gates, atomic swap, and product comms. The audit log retention (1y) is what makes this possible at all. Without it, a 2-week-old bug means permanently wrong numbers.

**[STAFF SIGNAL: late-data discipline]** Watermarks are committed; the policy by zone is explicit; the 24h cutoff is justified by storage and product needs (mobile replay is the dominant late-data source).

## 11. Audit Log and Reconciliation

**[STAFF SIGNAL: audit-log discipline]** The audit log is not a backup — it's the canonical record. The live stores are derived materializations.

```
AUDIT LOG
─────────────────────────────────────────────
storage:    S3, Parquet, partitioned by
            (date, source, hour)
schema:     event_id, ts_event, ts_ingest,
            counter_key, dims, delta, source
write path: Kafka → S3 sink (Flink secondary job)
            independent of the main pipeline
volume:     5 TB compressed/day → 1.8 PB/year
cost:       ~$45K/month S3 standard
            archive Day 90+ to Glacier: ~$8K/month
queryable:  Athena, Spark, BigQuery


RECONCILIATION (nightly, billing-tier hourly)
─────────────────────────────────────────────
job: re-aggregate counters from audit log for
     window W (yesterday)
diff: re-aggregated vs live OLAP

variance bands:
  < 0.001%   : pass silently  (billing tier)
  < 0.01%    : pass with metric
  < 0.1%     : warn, ticket
  > 0.1%     : page on-call, freeze that counter
               class until investigated

reconciliation as an SLI: "X% of counter-hours
reconciled to within 0.01%"  
target: 99.9%
```

The reconciliation job is the **answer to "is the count correct?"** — without it, the system is unauditable and the on-call has no story when a customer disputes a number. With it, every reported count carries an implicit confidence interval bounded by the last successful reconciliation.

## 12. Multi-Tenancy and Isolation

**[STAFF SIGNAL: multi-tenant isolation]**

```
PER-TENANT LIMITS
─────────────────────────────────────────────
ingest:        N events/sec, sliding window
               enforced at Kafka producer (auth token
               carries tenant; rate limiter at edge)
cardinality:   M distinct counters live
               enforced in Flink: new-counter creation
               against tenant budget; reject + metric
queries:       cost-based budget (rows scanned × seconds)
               per-tenant queue with weighted-fair
               scheduling in OLAP query layer
storage:       per-tenant retention tier; non-paying
               tenants get 7d 1-min, 30d 1-hr; premium
               gets full retention

NOISY-NEIGHBOR DEFENSES
─────────────────────────────────────────────
- Flink: tenant-tagged events; per-tenant operator
  state size capped; over-budget tenants get
  shedding before others
- Kafka: per-tenant quotas at broker
- OLAP: query queues per tenant; heavy queries
  routed to a separate replica set
```

The cardinality limit is the one that bites in practice — a tenant emitting metrics with `user_id` as a counter dimension can create 10⁹ counters in a day, blowing past the entire system's budget. Hard cap with clear errors back to the producer.

## 13. Failure Modes and Freshness Tiers

**[STAFF SIGNAL: failure mode precision]**

| Failure | Detection | Response | Blast Radius |
|---|---|---|---|
| Flink task lag | per-task watermark vs wall-clock | scale out parallelism; if >5min lag, page | one partition's counters slow; others fine |
| KV store write rejection | sink error rate >0.1% | backpressure into Kafka (bounded); shed low-priority counters | freshness lag, no data loss (Kafka retains 24h) |
| OLAP ingest behind | OLAP table watermark | non-real-time dashboards degrade gracefully; real-time queries fall back to "as of T-N min" | aggregate freshness only; per-counter unaffected |
| Hot counter melts task | per-task CPU / queue depth | auto-trigger Layer 3+4 mitigations | one counter; mitigation kicks in within 30s |
| Pipeline bug → wrong counts | reconciliation alert | shadow pipeline + atomic swap (§10) | scoped to bug window; protocol contains it |
| S3 audit log write failure | sink lag | dual-region write; if both fail, halt main pipeline (audit log is non-negotiable invariant) | total ingest stop is preferable to losing canon |
| Kafka outage | broker health | producer-side local buffer (10min); past that, drop with audit at producer | data loss for outage > 10min; rare |

**Freshness tiers** as a product feature, not a uniform default:

```
TIER          FRESHNESS    COST INDEX   USE CASE
─────────────────────────────────────────────────
real-time     1-5s         10x          live ops dashboards, anti-abuse
default       30-60s       1x           product UI counts
analytics     5-15min      0.3x         BI dashboards
billing       hourly close 3x           ad billing, audit
```

The same pipeline produces all tiers via different commit cadences. Real-time is reserved for the top-1000 counters by configuration — designing real-time for all 1B is the 10x cost mistake.

## 14. Operational Reality

**Metrics that matter**:
- per-partition Flink lag (the canary — high lag = freshness broken)
- dedup rate (sanity — if drops to 0, dedup is broken; if spikes, producer retry storm)
- reconciliation variance distribution (the trust metric)
- top-K churn rate (high = data quality issue or genuine virality)
- hot-key activation rate (how often Layer 3/4 trigger)
- audit log write success rate (must be 100% — invariant)

**Deployment**: Flink job upgrades use savepoints — graceful drain, restore on new version, watermark continuity preserved. Schema migrations to OLAP go through a shadow table with double-write for one tier-window before cutover. KV store changes use the per-shard versioning to avoid cross-shard inconsistency.

**The "we re-ran the pipeline" credibility problem**: every backfill emits a versioned correction. Dashboards display a small badge: "value corrected at T". Without explicit comms, finance teams see numbers move and lose trust permanently. The product surface for corrections is part of the system, not an afterthought.

## 15. Tradeoffs Taken and What Would Change Them

- **Two stores instead of one**: chosen for workload isolation. Would consolidate to one (Pinot or Druid with hot caching) only if point-read QPS dropped 10x or if operational cost of two systems exceeded engineering value — neither likely.
- **Effectively-once default + exactly-once billing tier**: chosen because billing-grade everywhere is 10x cost for ~2% of value. Would change if regulatory regime forced uniform billing-grade integrity (e.g., a healthcare counting use case).
- **1-min as the smallest OLAP granularity**: chosen because 1-sec rollups are 60x storage and the product cases I've seen don't justify it. Would add 10-sec rollups for the real-time tier if a specific top-1000-counter use case demanded sub-minute aggregation.
- **24h dedup window**: bounded by mobile-replay reality. Would extend to 72h if the producer SLA loosened, at proportional state cost.

## 16. What I Would Push Back On

**[STAFF SIGNAL: saying no]**

1. **"Near-real-time"** is underspecified by 100x in cost. Before designing, I'd force the conversation to a numeric SLO per tier — and resist a uniform 1-second SLO across all 1B counters.
2. **"Aggregate across counters"** without a query catalogue. Pre-aggregating dimensions you never query is waste; not pre-aggregating the ones you do is failure. I'd insist on the top 20 query shapes before committing to rollup schemas.
3. **The implicit "exactly-once"** expectation. Most workloads tolerate 0.01% error; only billing-grade pays the cost. Pushing every counter into the billing tier is a 10x cost mistake masquerading as rigor.
4. **The "one architecture" assumption**. Per-counter reads and cross-counter aggregations are fundamentally different workloads. A single store serving both forces one to lose.
5. **"100B events/day"** as a static target. Capacity planning needs a 12-month growth curve and a cardinality growth curve — which usually grows faster than event volume and is the actual constraint.

The honest answer to most "real-time counting" requests is: build the 30s-freshness default tier well, build the audit log and reconciliation, and resist scope creep into sub-second freshness or exact-everywhere integrity until a specific product justifies the 10x.