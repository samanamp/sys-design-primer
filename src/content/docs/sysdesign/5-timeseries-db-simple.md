---
title: Time-Series Database for Observability (simplified)
description: Time-Series Database for Observability
---

```
Design the storage and query layer for a metrics system handling 100M active series, 10M points/sec ingest, with queries ranging from 'last 5 minutes, one series' to 'last 30 days, sum across 10K series'. You're building the backend that competes with something like Prometheus remote storage or Datadog's metrics store. Walk me through it.
```

# Metrics Storage & Query Layer — Plain-English Version

## Quick vocabulary before we start

A few terms will appear a lot. I'll use plain words after this.

- **Metric**: a named number that changes over time, like `http_requests` or `cpu_usage`.
- **Labels**: key/value tags attached to a metric, like `region=us-east` or `service=checkout`.
- **Series**: one specific combination of (metric name + labels). For example, `http_requests{region=us-east, service=checkout}` is one series. Change any label and it's a different series.
- **Point**: one (timestamp, value) pair belonging to a series. A series collects many points over time — one every 10 seconds, say.
- **Cardinality**: how many unique series exist. This is the number that kills metrics systems.
- **Query**: asks the system for points back. Either one series over a short time (a point lookup), or many series over a long time (an aggregation).

---

## 1. First, let me push back on the question

Before designing, I want to reframe what's actually being asked.

**[STAFF SIGNAL: query-diversity reframing]** The prompt says "queries ranging from 5 minutes, one series to 30 days, 10,000 series" as if that's a range. It isn't. Those are two different workloads that happen to hit the same API:

- **"Last 5 minutes, one series"** — wants an answer in under 10 milliseconds. Reads maybe 30 points. Should never touch a disk.
- **"Last 30 days, 10,000 series"** — wants an answer in a couple of seconds. At the natural 10-second resolution this would read 2.6 *billion* points, which is hopeless, so we *must* have pre-computed summaries to serve it.

Every later decision — how we lay data on disk, what we compress, what we pre-compute, how we route queries — is driven by serving both of these cheaply. If you don't split your read path into a fast path and a cheap path, you build a system that is slow for one and expensive for the other.

**[STAFF SIGNAL: cardinality as first-class]** The scary number is not "10 million writes per second." That's ~1.3 TB per day after compression, which is fine. The scary number is "100 million unique series." Because: if a user accidentally puts a `user_id` or `trace_id` label on a metric, their series count doubles overnight, our in-memory indexes double, our background jobs fall behind, and queries slow down *for everyone else on the same machine*. Protecting the system from this is not a feature. It is the whole design.

**[STAFF SIGNAL: scope negotiation]** I'm committing to:
- Numeric metrics only — counters (always-increasing), gauges (go up or down), histograms (pre-bucketed latency distributions).
- A Prometheus-style query language. No joins, no user-defined functions.
- Series identity = (metric name, sorted labels). Same combination always hashes to the same ID.
- **Explicitly out of scope**: logs, traces, profiles, and ad-hoc analytics on raw unaggregated events. That last one matters. There's a whole school of thought (Honeycomb-style "wide events") that says metrics are the wrong abstraction for debugging. I'll come back to this at the end.

---

## 2. The math that drives the design

| What | How | Number |
|---|---|---|
| Writes per second | given | 10M |
| Writes per day | × 86,400 | 864 billion points |
| Raw point size (8-byte timestamp + 8-byte value) | — | 16 bytes |
| Raw daily volume | 864B × 16B | ~14 TB/day (too much) |
| Compressed point size (see §4) | empirical | ~1.4 bytes |
| Compressed daily volume | | **~1.2 TB/day** (manageable) |
| Active series | given | 100M |
| Avg points per series per second | 10M / 100M | 0.1 (one point every 10 seconds) |
| In-memory buffer per active series | ~2 KB | |
| Total RAM just for write buffers | 100M × 2 KB | **~200 GB** across the fleet |
| Inverted index size (see §5.2) | empirical | ~5–10 GB per region |

**The two queries, costed out:**

- **Query 1 (5 min, 1 series)**: 30 points × ~2 bytes = 60 bytes read. If this takes longer than memory access, we've done something wrong.
- **Query 2 (30 days, 10K series) at full resolution**: 10K × 30 × 8,640 = 2.6 *billion* points = ~3.5 GB scanned. Unacceptable.
- **Query 2 using per-hour summaries**: 10K × 30 × 24 = 7.2 *million* points = ~10 MB scanned. Fine.

**[STAFF SIGNAL: capacity math]** The 360× reduction from hourly summaries is what makes the long query practical. Without pre-computed summaries, this design doesn't work at any price.

---

## 3. The overall architecture

```
                        ┌─────────────────────────────────┐
                        │        QUERY PLANNER            │
                        │  - which summary level to use?  │
                        │  - how much will this scan?     │
                        │  - is the user over budget?     │
                        └──┬──────────────────┬───────────┘
                           │ fast path        │ cheap path
                           ▼                  ▼
Producers           ┌──────────────────┐  ┌────────────────┐
(app sends          │ INGESTER SHARDS  │  │    QUERIER     │
 metrics) ──► [DISTRIBUTOR]──►          │  │  (reads old    │
                    │  - write-ahead   │  │   data from    │
             (checks quotas,            │  │   object store)│
              routes writes)            │  │                │
                    │    log           │  └───────┬────────┘
                    │  - recent data   │          │
                    │    held in RAM   │──────────┤
                    │  - flushes to    │          │
                    │    disk every 2h │          │
                    └────────┬─────────┘          │
                             │                    │
                             │ background         │
                             │ summary job        │
                             │ + compaction       │
                             ▼                    │
                    ┌─────────────────────┐       │
                    │   OBJECT STORE (S3) │◄──────┘
                    │ - raw data (6h)     │
                    │ - 1-min summaries   │
                    │ - 5-min summaries   │
                    │ - 1-hour summaries  │
                    │ - 1-day summaries   │
                    └─────────────────────┘
```

**[STAFF SIGNAL: rejected alternative]** Why not just use an existing database?

- **ClickHouse** (a fast columnar OLAP database): good for one-off analytical queries, but it stores each point as a row with full labels, which loses the compression trick that makes metrics cheap. Metrics-specific compression beats ClickHouse's general compression by roughly 5–10× on the same data.
- **Cassandra with wide rows** (what Netflix's original Atlas tried): good write throughput, but there's no place for the label index to live — you'd have to run a separate index system next to it, which is painful.
- **Postgres + TimescaleDB**: fine up to about 1 million series. At 100 million, the indexes become the bottleneck.
- **RocksDB as the primitive**: tempting, but if you store one key per point, the per-key overhead destroys your compression ratio. You want to store *chunks* (one series × 2 hours of points, compressed together), not individual points.

Conclusion: the compression ratios and access patterns of metrics are specialized enough that we build a purpose-built system, borrowing ideas from Prometheus TSDB and its scaled-out cousins (Mimir, Thanos, Cortex).

---

## 4. How one series is stored

Each series' points get collected together in time windows (say, 2 hours of points per **chunk**). The chunks are stored with specialized compression.

```
ONE CHUNK ON DISK
┌─────────────────────────────────────────────┐
│ header: which series, how many points, etc  │
├─────────────────────────────────────────────┤
│ TIMESTAMPS (compressed separately)           │
│   First timestamp: full 8 bytes             │
│   Next: the gap (usually 10 sec)            │
│   Next: the change in the gap               │
│         (almost always 0 → 1 bit!)          │
├─────────────────────────────────────────────┤
│ VALUES (compressed separately)               │
│   First value: full 8 bytes                 │
│   Next: XOR with previous value             │
│         → most bits usually match           │
│         → store only the differing ones     │
└─────────────────────────────────────────────┘
```

**Why separate the timestamp and value streams?** Because they compress using different patterns. Timestamps in a regular scrape are nearly constant — the gap between them doesn't change — so "store how the gap changed" compresses to 1 bit per point in the common case. Values, meanwhile, are 64-bit floats where consecutive samples usually share most of their bits, so "store only the bits that differ" is what you want there.

**[STAFF SIGNAL: compression/encoding precision]** This scheme is called **Gorilla compression** (Facebook, 2015). On real-world metrics it averages about 1.37 bytes per point. General-purpose compression like gzip gets nowhere near that. But: on *irregular* data — flaky scrapes, varying intervals, high-entropy values — Gorilla degrades to 3–6 bytes per point. So "a customer's compression ratio just got worse" is something we need to detect and alarm on.

---

## 5. The deep parts

### 5.1 Cardinality defense (the most important thing)

The threat: a user adds a label like `user_id` to a metric. Each unique value creates a new series. 10 million unique users = 10 million new series = system possibly on fire.

```
INGEST PATH WITH CARDINALITY PROTECTION

 Producer → Distributor ──────────────────→ Ingester Shard
              │                                  │
              │  ┌──────────────────────┐       │   ┌────────────────┐
              ├─►│ Check: is this tenant │      ├──►│ On new series: │
              │  │ under their series   │       │   │  - write to log│
              │  │ quota?               │       │   │  - add to index│
              │  │ (uses a tiny counter │       │   └────────────────┘
              │  │  called HyperLogLog) │       │
              │  └──────────────────────┘       │
              │                                  │   ┌────────────────┐
              │  ┌──────────────────────┐       └──►│ Stream count to│
              ├─►│ Check: is any one    │           │ control plane  │
              │  │ label growing too    │           │ for alerting   │
              │  │ fast? (an hour's     │           └────────────────┘
              │  │ worth of new values  │
              │  │ over 100K → block)   │
              │  └──────────────────────┘
```

**[STAFF SIGNAL: cardinality as first-class] [STAFF SIGNAL: invariant-based thinking]** The rules the system always enforces, no matter what:

1. Per-customer series count ≤ their quota. Enforced at the front door, not at the storage shard — by the time data reaches the shard, the damage is done.
2. No single label may produce more than N new series per minute without tripping a breaker.
3. We use **HyperLogLog** (a tiny probabilistic counter — think of it as a 12 KB gadget that can estimate "how many unique things have I seen?" within 1%) to track this cheaply.

**Reasonable defaults**: 10 million series per customer by default; increases require a conversation (it's a billing matter as much as a technical one). Per-metric soft cap 1 million. A "this label is exploding" detector quarantines new series and tells the user: *"We are dropping series on metric http_request_count because label user_id appears to be unbounded. Accept the drop, or fix the label?"* Silently dropping data loses customers. Rejecting without explanation loses them faster.

**Cleanup**: when a customer has already emitted 50 million bad series, we support a `forget_series` operation that immediately hides them from queries and lets background compaction physically delete them over the next day.

### 5.2 The tag index (reverse lookup)

The problem: given a query like "sum http_requests by region," we need to find every series whose labels match. This is a reverse lookup — "which series contain the label `region=us-east`?" — identical in structure to how search engines work.

```
REVERSE LOOKUP STRUCTURE

DICTIONARY (all unique label strings, compactly stored)
   "region"     → ID 1
   "us-east-1"  → ID 2
   "us-west-2"  → ID 3
   "method"     → ID 4
   "GET"        → ID 5
   "POST"       → ID 6

POSTINGS (for each label=value, the list of series that have it)
   region=us-east-1  →  series #0, #1, #4, #7 ...  (~40M series)
   region=us-west-2  →  series #2, #3, #5, #6 ...  (~60M series)
   method=GET        →  series #0, #2, #4, #6 ...  (~70M series)
   method=POST       →  series #1, #3, #5, #7 ...  (~30M series)

QUERY: {region="us-east-1", method="POST"}
  → take the intersection of two lists  →  ~12M series
  → read only those series' data
```

**[STAFF SIGNAL: index architecture precision]** Storing those lists of series IDs naively (4 bytes per ID × 40 million = 160 MB per list) is too expensive. We use **roaring bitmaps**: a clever representation that splits the series IDs into chunks and picks the cheapest encoding for each chunk — a dense bitmap, a sorted array, or a run-length code. In practice this gets us to under 1 bit per series in many postings, and supports fast AND/OR operations using CPU SIMD instructions. Prometheus and Lucene both use this approach for the same reasons.

**The memory-vs-disk tradeoff**: recent data's index (the last 2 hours) lives in RAM — that's the fast path. Older data's index lives on disk and is memory-mapped, so the operating system pages hot parts into RAM automatically. Total index RAM: about 10 GB per region.

**Queries that break the index**:
- A regex pattern on a high-cardinality label like `{trace_id=~".*abc.*"}` forces us to scan every value of that label — 10 million operations for a single query. **[STAFF SIGNAL: saying no]** We refuse those queries and tell the user why. Supporting them would require a much more expensive index type (trigram) that triples our memory cost.
- Giant OR lists like `{service=~"a|b|...|z_1000"}` — we bound these at 128 alternatives.

### 5.3 Pre-computed summaries (rollups)

We pre-compute sums, counts, mins, and maxes at 1-minute, 5-minute, 1-hour, and 1-day resolutions. This happens continuously, as data streams in — not in a batch job afterward. That way, summaries are available within seconds of the raw data.

**[STAFF SIGNAL: rollup precision]** The catch: we store `sum`, `count`, `min`, `max`, but **not** `avg`. Why? Because averages don't combine correctly across intervals. If you average 60 one-minute averages, you're not getting the correct hourly average unless each minute had the same number of points. So we let the query compute `avg = sum/count` when it needs to.

**The percentile problem**: a user wants p99 latency over 30 days. You cannot average p99s. You cannot take a 1-minute p99 and then combine 60 of them into an hourly p99 — it's mathematically wrong, and the error can be huge. Three solutions, and we use all three:

1. **Pre-bucketed histograms** — the user instruments their app to emit `latency_bucket{le="0.5"}`, `le="1.0"`, etc. These are just counters, and counters *do* combine correctly. p99 is computed at query time from the buckets.
2. **Exponential histograms** (the new OpenTelemetry format) — a compact sketch per series that merges correctly across time.
3. **t-digest** — another small sketch (~2 KB per series per interval) that approximates percentiles within 1% and merges. Costs ~30 GB/day extra storage at 1-minute resolution. We make it opt-in per metric.

**Query resolution selection**: for a query covering 30 days wanting ~1000 points back, step size = 43 min → pick 1-hour summaries. That's what turns the 2.6-billion-point scan into a 7.2-million-point scan.

### 5.4 The query planner

**[STAFF SIGNAL: query planner discipline]** The planner is a real component with real responsibilities:

1. **Cost estimation before executing**. Given the matchers and range, estimate how many series × how many points × how many bytes. This is cheap because the roaring bitmaps already know their cardinality. If the estimated cost exceeds the tenant's budget, reject with a useful message: *"This query would scan 500 GB. Try narrowing to a region or reducing the range to 7 days."*

2. **Admission control**. Each tenant has quotas on concurrent queries and CPU-seconds per minute. Heavy queries don't deadlock the system — they queue or fail. Across tenants, weighted fair queuing so one whale doesn't starve everyone.

3. **Fan-out and merge**. A cross-series query goes to every relevant shard. Each shard finds matching series, reads chunks, decodes, does partial aggregation (sum/count/min/max per output bucket), and streams small partials back. The final merge happens in one place. This works because sum/count/min/max are commutative.

**Walkthrough of Query 2** (30 days, sum by region, 10K series):
1. Planner matches the tag index across shards → 10K series, ~625 per shard.
2. 30-day range wants ~500 output points → step size ~1.5 hours → use 1-hour summaries.
3. Cost estimate: 10K × 720 points × 16 bytes ≈ 115 MB scan. Approve.
4. Each shard reads its chunks, groups by `region`, partial-sums per region per time bucket, streams back ~115 KB.
5. Central querier adds up partials. Returns ~500 × 10 regions of data.
6. Target: 2 seconds. Main cost: the object-store read latency for cold data. Mitigation: cache results keyed by (block ID, matcher, aggregation). Dashboard queries hit this cache 80%+ of the time.

### 5.5 Late-arriving data

**[STAFF SIGNAL: invariant-based thinking]** The rule: a point's placement is based on its timestamp, never on when it arrived. If a scraper is delayed or a device's clock is wrong, points arrive with old timestamps.

Our grace window: 1 hour. Points arriving within 1 hour of their real timestamp go into the current active buffer. Points older than that go into a "backfill" path that triggers a targeted rewrite of the affected time window.

The ugly case: a 3:00 PM point shows up at 6:00 PM, but the rollup for 3:00 was finalized at 5:00. Three options:
1. Drop the point, emit a metric so the user can see their collector is broken.
2. Accept it and rebuild the affected rollup. Correct but expensive.
3. Accept it and *don't* rebuild the rollup. Silently wrong. Never.

We pick option 2 for points within 24 hours, option 1 after that. Operators need to see when their collectors are misbehaving, which means option 1 must be observable, not silent.

---

## 6. What happens when things break

**[STAFF SIGNAL: failure mode precision]** Specific scenarios with specific responses:

- **Cardinality spike**: detector trips within 30 seconds, auto-applies a rate limit on that tenant, banner on their dashboard explaining why.
- **Compaction falling behind**: first response is a warning; if it gets worse, we slow ingest for the affected tenant; if it gets severe, we add capacity or shed writes. Compaction progress is an alarmable metric.
- **A query uses too much memory**: each query has a cap (say 4 GB). Over it, we return an error that says what the query would have needed. The querier process is isolated so one bad query can't crash others.
- **Data on disk is corrupted**: we check CRCs on every read. On corruption, return partial results with a flag, log the corruption, rebuild from a replica.
- **Whole region dies**: the other region serves writes locally. Queries against the dead region return "unavailable" for that range — **we never interpolate fake points and we never silently hide gaps**. The user must be able to tell the difference between "nothing happened" and "we don't know what happened."

**[STAFF SIGNAL: blast radius reasoning]** The meta-problem: this system is what tells operators when *other* systems are broken. When *this* system breaks, its failure is invisible, which is the worst case. Our own health metrics must not be served only by ourselves — we ship a small, separate telemetry path (a minimal Prometheus instance and a separate log stream) that the on-call team uses when the main system is down. This is the "phone a friend" pattern and it is non-negotiable.

---

## 7. Multi-tenancy (many customers sharing the system)

**[STAFF SIGNAL: multi-tenant isolation]** Isolation at every layer:

- **At ingest**: per-tenant rate limits, per-tenant series quotas.
- **In storage**: tenants share shards by default; a whale (over 1M series or 100K points/sec) gets dedicated shards. Dedicated everywhere is ~3× more expensive, so we reserve it for tenants whose noise profile justifies it.
- **At query**: per-tenant concurrent-query quotas, CPU-seconds budgets, weighted fair queuing across tenants.
- **At compaction**: per-tenant budgets, so one tenant's slow compactions don't starve others.

The principle: **bound the blast radius**. One noisy customer must not be able to degrade another.

---

## 8. What's been happening in this space (so you know I'm current)

**[STAFF SIGNAL: modern awareness]**

- **Gorilla paper (Facebook, 2015)**: the compression foundation. Every serious time-series DB since has used variants.
- **Prometheus TSDB**: the memory-buffer + write-ahead-log + 2-hour-blocks architecture we're borrowing. Great for single-node. We add horizontal sharding.
- **Cortex / Thanos / Grafana Mimir**: three projects that scaled Prometheus horizontally. Mimir's design (separate ingester, querier, compactor, store gateway) is closest to ours.
- **VictoriaMetrics**: an alternative that's very good at high cardinality. Uses a sorted-key approach instead of an inverted index. Elegant, but we chose the Lucene-style approach for better handling of complex label matchers.
- **InfluxDB IOx**: rewrote their engine on top of Apache Arrow / DataFusion / Parquet. Great for analytics but sacrifices some compression. We haven't gone that way for the hot path, but a Parquet *export* feature would be natural.
- **Google Monarch**: hierarchical multi-region aggregation. Worth studying for our cross-region story.
- **Honeycomb and wide events**: see the next section.
- **OpenTelemetry**: the ingest protocol is converging on OTLP. The native histogram format finally gives us composable percentiles. Day-one support required.

---

## 9. Tradeoffs, and when this whole design is wrong

**What would force a redesign?**

- **If users need arbitrary queries on raw, unaggregated events** — this design is wrong, and the right one is a columnar event store. Metrics pre-aggregate: you commit upfront to the dimensions you'll slice by. Wide-event stores keep everything raw and let you slice later. Metrics are cheaper and faster for pre-known dashboards; events are more flexible and don't have a cardinality ceiling. **The honest answer** is that a mature observability stack has *both* — metrics for dashboards and alerts, a separate event store for debugging. Any vendor claiming one system does both well is oversimplifying.
- **If retention is 10 years of raw data**: summaries aren't enough. We'd move to a columnar cold-storage format.
- **If we need sub-second alerting**: current design has ~5 seconds from ingest to query. Lower requires a separate in-memory alerting path.

---

## 10. What I'd push back on in the requirements

**[STAFF SIGNAL: saying no] [STAFF SIGNAL: operational longevity]**

1. **"100 million series"** without a per-tenant policy is not a useful requirement — you're one bad customer away from an outage. The real requirement should be "N tenants with per-tenant caps that sum to 100M, plus the cardinality enforcement we discussed."
2. **"Compete with Prometheus remote storage OR Datadog"** — those are two very different products. Which?
3. **"Queries ranging from 5 min to 30 days"** — doesn't bound the cross-product with series count. A 30-day, 1-series query is trivial; a 30-day, all-series query is infeasible at any price.
4. **Uniform retention** for all metrics is probably wasteful. SLO metrics want long retention; debug metrics can expire in a week. This needs a policy, not a one-size answer.
5. **Year-3 reality**: the compression format will evolve, at least one customer will find a clever way to abuse labels, schema changes will be the dominant engineering cost. Which means: version the block format from day one, and make an online block-rewriter a first-class service, not a migration tool.

---

## Close

The design in one paragraph: a purpose-built TSDB with (a) a **distributor** that enforces per-tenant quotas and cardinality caps at the front door; (b) **ingester shards** that hold recent data in RAM with a write-ahead log for durability and compress older data using Gorilla into 2-hour chunks on disk; (c) a **streaming summary pipeline** producing 1-minute, 5-minute, 1-hour, and 1-day rollups into tiered object storage; (d) a **cost-aware query planner** that picks the right resolution, rejects expensive queries before running them, and fans out across shards with fair-share scheduling. Query 1 (5 min, 1 series) hits RAM and returns in under 10 ms. Query 2 (30 days, 10K series) hits 1-hour summaries and returns in under 2 seconds. Cardinality is the existential threat, not volume — enforced at the front door using HyperLogLog. Percentiles use pre-bucketed histograms or sketches, never averaged. The observability-of-observability problem is handled with a separate minimal telemetry path. And if users need arbitrary raw-event queries, this is the wrong system — that's a wide-event store, which should live alongside this one.