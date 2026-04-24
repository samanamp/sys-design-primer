---
title: Online Feature Store
description: Online Feature Store for Recommendation Ranking
---
```
"Design the online feature store for a recommendation system. 50K feature lookups per request, 1M requests/sec, p99 budget for feature fetch is 20ms. Features include real-time counters, embeddings, and historical aggregates. Walk me through it."
```
---
# Online Feature Store for Recommendation Ranking — Staff-Level Design

## 1. Scope and Reframing

Before architecture: the question covers too much ground as posed. 50K feature lookups per request places us squarely in **fine ranking** — not retrieval (thousands of candidates with tens of features, ANN-driven), and not re-ranking (tens of items, diversity/business-rule focused). Fine ranking scores O(100–1000) candidates with O(50–500) features each. I'll commit to **~1000 candidates × 50 feature lookups per candidate**, where "lookup" means a read of one feature value for one (user, item, context) tuple. User-side features are shared across candidates (~50 user features fetched once), item-side multiplies by candidate count (~30 × 1000 = 30K), cross features per pair (~15 × 1000 = 15K). Net: ~45K unique reads, ~50K lookups counting the tensor-shape redundancy. [STAFF SIGNAL: scope negotiation]

The math is the forcing function. **50B feature reads/sec** at the fleet level. **20ms p99** for the feature-fetch phase (not the whole ranking request — model inference eats its own budget). If we naively place features behind a KV RPC at 1ms median / 5ms p99 per hop, 50K serial lookups is 250 seconds. Even 1000-way parallelized it's 250ms and blows the budget by 12×. The answer cannot be "faster Redis." The answer is **colocation of the hot working set with the ranker, specialization of the read path per feature type, and ruthless batching**. [STAFF SIGNAL: latency budget reframing]

**Training/serving skew is the principal architectural risk, not a test-harness concern.** [STAFF SIGNAL: skew as first-class] A fine ranker trained on features computed in Spark and served from an online store computed in Flink will silently degrade as definitions drift. I'll treat the skew invariant — *the feature value the model trained on is byte-equivalent to the feature value served* — as a property of the system, not an assertion checked after the fact. The dominant design choice: **log-and-train** (we log served features and train on those logs) rather than recomputing training features from raw data.

**What I'd push back on up front**: [STAFF SIGNAL: saying no] 50K features per request is almost certainly excessive. In every large recsys I've audited, 60–80% of features contribute sub-basis-point NE lift. A feature-pruning initiative (SHAP-based, or drop-and-retrain with online A/B) typically removes 30–50% of features with no measurable quality regression and buys back half the latency budget. I'd land the infra design for 50K but fund pruning in parallel — it's a larger win than any single infra optimization.

## 2. Capacity Math and Latency Budget

Per-request wire math:

```
Component              | Min wire       | Realistic
-----------------------|----------------|------------
Counters (~50)         |  50 × 4B = 200B|  50 × 8B = 400B
Scalar aggregates (5K) | 5K × 4B = 20KB | 5K × 8B = 40KB
Dense numeric (~10K)   |10K × 4B = 40KB |10K × 4B = 40KB
Embeddings (2 user +   |(2+1K) × 64B    |(2+1K) × 96B
  1K item, 64-d int8)  |      = 64KB    |      = 96KB
Cross features (~15K)  |15K × 4B = 60KB |15K × 4B = 60KB
-----------------------|----------------|------------
Total per request      | ~184 KB        | ~236 KB
At 1M rps              | ~184 GB/s      | ~236 GB/s aggregate
```

~250 GB/s aggregate wire from feature substrate to rankers. If features are remote, that's 250 GB/s of NIC budget split across ~2000 ranker boxes plus the serving side. Feasible with 100 Gbps NICs (12.5 GB/s each) but tight once protocol overhead is counted. **If features are colocated in-process, this bandwidth is DRAM bandwidth (~200 GB/s per socket, free).** That alone argues for colocation.

Embedding table sizes (hot path):

```
User embedding       : 1B users × 64 dim × int8 × 3 models = ~192 GB
Item embedding       : 300M items × 64 dim × int8 × 3 models = ~57 GB
Cross-surface embed  : 5 tables × ~50 GB avg = ~250 GB
Category/taxonomy    : 20 small tables × 1 GB = ~20 GB
                                                 --------
Total active (int8)                              ~520 GB
Equivalent at fp32                               ~2 TB
```

Historical aggregates (7d/30d windows, ~500 features): ~100M active users × 500 × 8B ≈ 400 GB.

**Working set at 1M rps, Zipfian α≈1.0:** top 10% of users cover ~80% of traffic. Hot user set ≈ 100M; hot item set ≈ 50M (catalogs churn faster than users). Hot embedding working set ≈ 100M × 64 × int8 × 3 tables ≈ 20 GB. **That fits in RAM on every ranker box — dominant implication for architecture.** [STAFF SIGNAL: capacity math]

Latency budget decomposition (20ms p99):

```
Stage                               | Budget (p99) | Mechanism
------------------------------------|--------------|-----------------------
Request fan-out / batching wait     |  1.0 ms      | 1ms batch window
Ranker → local feature orchestrator |  0.2 ms      | in-process call
In-process lookup (hot tier)        |  1.0 ms      | 80% hit rate
On-box sidecar (warm tier)          |  2.0 ms      | 18% hit rate, UDS
Remote embedding shard fetch        |  6.0 ms      | 2% cold, HDR-p99
Deserialization / tensor assembly   |  3.0 ms      | zero-copy where possible
Remote aggregate fetch (rare)       |  4.0 ms      | Scylla fallback
Slack / GC / tail                   |  2.8 ms      | everything else
------------------------------------|--------------|-----------------------
Total                               | 20.0 ms      |
```

The in-process layer must deliver 80% hit or we miss the budget. That's the concrete design pressure.

## 3. High-Level Architecture

```
┌───────────────────────────────────────────────────────────────────────┐
│                        OFFLINE SUBSTRATE                              │
│  ┌─────────────┐    ┌────────────────┐    ┌──────────────────────┐    │
│  │ Event Log   │ →  │ Spark / Flink  │ →  │ Feature Warehouse    │    │
│  │ (Kafka→HDFS)│    │ batch + stream │    │ (Iceberg, partitioned│    │
│  └─────────────┘    │ feature jobs   │    │  by feature + date)  │    │
│         │           └────────────────┘    └──────────────────────┘    │
│         │                   ▲                   │                     │
│         │                   │                   ▼                     │
│         │           ┌───────┴────────┐  ┌───────────────────────┐     │
│         │           │ Feature DSL    │  │ Training snapshot     │     │
│         │           │ registry       │  │ (point-in-time join   │     │
│         │           │ (single defn)  │  │  only for cold-start) │     │
│         │           └────────────────┘  └───────────────────────┘     │
│         │                   │                    │                    │
│         │                   │                    ▼                    │
│         │                   │            ┌───────────────────┐        │
│         │                   │            │ Training loop     │        │
│         │                   │            │ (trains on LOGGED │        │
│         │                   │            │  served features) │        │
│         │                   │            └───────────────────┘        │
├─────────┼───────────────────┼───────────────────────────────────────  │
│         │                   │         ONLINE SUBSTRATE                │
│         ▼                   ▼                                         │
│  ┌─────────────┐    ┌────────────────┐                                │
│  │ Kafka topic │ →  │ Flink feature  │ ───────────────┐               │
│  │ (events)    │    │ jobs (codegen  │                │               │
│  └─────────────┘    │ from DSL)      │                │               │
│                     └────────────────┘                │               │
│                             │                         │               │
│                             ▼                         ▼               │
│            ┌────────────────────────────┐  ┌──────────────────────┐   │
│            │ Online Feature Store       │  │ Embedding Service    │   │
│            │ ┌──────────────────────┐   │  │ (specialized; shard  │   │
│            │ │ Remote KV (ScyllaDB) │   │  │  by entity-id hash;  │   │
│            │ │ counters + agg tail  │   │  │  int8, row-major)    │   │
│            │ └──────────────────────┘   │  └──────────────────────┘   │
│            └────────────────────────────┘         │                   │
│                  │              │                 │                   │
│                  ▼              ▼                 ▼                   │
│            ┌──────────────────────────────────────────┐               │
│            │ Ranker box (colocation)                  │               │
│            │  ┌────────────────────────────────────┐  │               │
│            │  │ In-process tier (RAM, hot 20 GB)   │  │               │
│            │  │  - hot user counters/embeddings    │  │               │
│            │  │  - recent aggregates               │  │               │
│            │  ├────────────────────────────────────┤  │               │
│            │  │ On-box sidecar (UDS; RocksDB +     │  │               │
│            │  │  page cache; warm 200 GB SSD)      │  │               │
│            │  ├────────────────────────────────────┤  │               │
│            │  │ Feature logger (writes served      │──┼──→ Kafka      │
│            │  │  snapshots to log topic)           │  │  (train loop) │
│            │  └────────────────────────────────────┘  │               │
│            │                                          │               │
│            │   Ranker GPU(s) ← tensor assembled       │               │
│            └──────────────────────────────────────────┘               │
└───────────────────────────────────────────────────────────────────────┘
```

Commitments: [STAFF SIGNAL: storage tiering]

1. **Three-tier online read path** — in-process (hot, 20 GB, 80% hit), on-box sidecar over UDS (warm, 200 GB, 18% hit), remote shards (cold, 2%). No uniform "KV for everything" — that blows the budget.
2. **Embedding service is separate** from the main online store. Embedding access is batched, row-oriented, bandwidth-dominated; counters and aggregates are point reads. Mixing them penalizes both.
3. **Logged-features training loop.** Rangers write the exact served feature vector to Kafka; training consumes those logs as source of truth. Eliminates offline/online computation skew by construction for served features. Warehouse-computed features still exist for exploration and new-model cold-start, but production training runs on logs.
4. **Rejected alternatives** [STAFF SIGNAL: rejected alternative]:
   - *Single global Redis/Scylla for all features.* 50K × 1ms serial = 50s; even heavily parallelized, fan-out tail ruins p99.
   - *Off-the-shelf Feast as serving path.* Feast is a control plane (registry, lineage, offline joins), not a hot-path substrate. Useful for the offline side; not a live store at 1M rps × 50K.
   - *Tecton-style unified offline+online execution.* The DSL cannot express 30-day windowed aggregates efficiently in the online path without incurring training-path constraints. Use unified **definition** (DSL → codegen for both Flink and Spark); retain separate execution substrates. [STAFF SIGNAL: invariant-based thinking]
   - *Pure in-memory, no remote tier.* Long tail of cold embeddings (inactive users, new items) pushes working set beyond per-box RAM; remote tier exists for completeness, not latency.

## 4. Feature-Type-Specific Design

[STAFF SIGNAL: feature-type differentiation] Three systems, not one.

**Real-time counters** (e.g., `user_clicks_last_5min`, `item_impressions_last_1h`). Writes: ~500K events/sec. Reads: 1M rps × ~50 counters = 50M reads/sec. Ingest: Kafka → Flink (RocksDB state) → sliding-window aggregation → **ScyllaDB** sink + changelog topic for sidecar cache invalidation. Freshness target: event-to-served p95 under **10 seconds**. Exactly-once unnecessary for most counters; at-least-once with 5-minute Bloom-filter dedupe suffices. A whitelist of fraud-style counters runs an exactly-once pipeline at ~10× the cost.

**Embeddings** (user, item, query, cross-surface). Writes: tables retrained on 6–24h cycles; incremental streaming embeddings from a two-tower run as a separate substream at minute-scale staleness. Storage: **dedicated embedding service**, sharded by entity-id hash, row-major within shard, int8-quantized with per-row scale + zero-point. Candidate-batch fetch: one RPC per shard carrying up to 1000 IDs, returns a dense matrix. **Critical invariant: the quantized form stored online is the form trained on** — quantization happens before feature logs are emitted, never as a post-hoc deploy-time step.

**Historical aggregates** (e.g., `purchases_7d`, `sessions_30d`, `avg_dwell_90d`). Long windows make streaming infeasible (30-day state per user at 100M users is prohibitive in Flink). Writes: daily batch backfill from warehouse. Storage: compact columnar blobs per entity, loaded to each ranker's warm tier on a daily cycle. Reads: 5K point lookups per request, >99% on-box hit rate since values change daily.

## 5. Deep Dives

### 5.1 Latency Budget Decomposition

Read path with per-stage timing:

```
t=0ms   Request arrives at ranker box
        │
        │ ┌── Feature IDs derived from candidate-gen result ───────┐
        │ │   (1000 item IDs + user context; no feature fetch yet) │
        │ └─────────────────────────────────────────────────────── ┘
        │
t=0.2   ├─► Feature orchestrator batches lookups by tier × type
        │     ┌────────────────┬────────────────┬──────────────────┐
        │     │                │                │                  │
t=0.3   ├────►│ In-process     │ On-box sidecar │ Remote fan-out   │
        │     │ (≤1.0ms p99)   │ (≤2.0ms p99)   │ (≤6.0ms p99)     │
        │     │ SwissTable +   │ UDS + shmem    │ gRPC, batched    │
        │     │ cacheline-opt  │ zero-copy emb  │ per-shard         │
        │     │ 40K lookups    │ 9K lookups     │ 1K lookups       │
        │     └────────┬───────┴────────┬───────┴────────┬─────────┘
        │              │                │                │
t=1.3   ├──────────────┴── in-process hits returned
t=2.3   ├───────────────────────── on-box hits returned
t=6.3   ├────────────────────────────────────────── remote hits returned
        │                                            (embedding cold + 
        │                                             long-tail counters)
        │
t=6.3   ├─► Tensor assembly: scatter into model-expected layout
        │   (zero-copy for embeddings from on-box shmem)
        │
t=9.3   ├─► Padding + missing-value fill with learned per-feature defaults
        │
t=10.3  └─► Hand to GPU ranker
                                          ┌── remaining budget ~9.7ms ──┐
                                          │  for inference + tail slack │
                                          └─────────────────────────────┘
```

The in-process layer is the linchpin. It isn't a cache in the eviction sense — it's a **curated mirror of the predicted hot set**, refreshed by a background controller that reads the last hour of traffic, computes per-entity QPS, and pushes the top-N (20 GB worth) to every ranker box. [STAFF SIGNAL: invariant-based thinking] Invariant: **any entity served >1000 QPS globally must be in-process on every ranker**. Enforcement: the hot-set controller recomputes every 5 minutes; ranker boxes pull deltas. Cold-start of a new ranker box takes 5 min; it stays out of the LB until warm.

On-box sidecar: a separate process per ranker box, UDS-connected, 200 GB warm features in RocksDB with `dirty_ratio=5`, `readahead` disabled (random access dominates). UDS + shared-memory ring for zero-copy embedding reads — avoids the ~200μs of protobuf overhead that would otherwise dominate small-value reads. At 50K lookups × 200μs = 10 seconds of overhead without this, so the optimization is not cosmetic.

Remote fan-out: gRPC with pooled connections, per-shard concurrency capped at 32 (higher caps worsen tail from head-of-line blocking). Shards: 128-MB RocksDB nodes on Scylla for counters/aggregates; specialized embedding servers for embedding tables.

**Batching vs latency**: a 1ms batch window at the ranker. At 1M rps ÷ 2000 boxes ≈ 500 rps/box, 1ms captures 0.5 requests on average — too little for effective cross-request micro-batching at that level. Instead we batch **within** a single request's 50K lookups. Cross-request batching only pays on the remote path and is handled by the sidecar transparently.

### 5.2 Embedding Storage

```
 Shard S (of 256; sharded by hash(entity_id) % 256)
 ┌─────────────────────────────────────────────────────────┐
 │ Header: {version, row_count, dim, dtype, layout}        │
 ├─────────────────────────────────────────────────────────┤
 │ Row 0: [scale(fp16) | zero_pt(int8) | 64 × int8 values] │
 │ Row 1: [scale       | zero_pt       | 64 × int8 values] │
 │ ...                                                     │
 │ Row N: [scale       | zero_pt       | 64 × int8 values] │
 ├─────────────────────────────────────────────────────────┤
 │ ID → row offset index (perfect hash, built at load)     │
 └─────────────────────────────────────────────────────────┘

 Row size: 2 (scale) + 1 (zero_pt) + 64 = 67B → padded to 72B
 
 Candidate-batch fetch:
   Request: fetch([id_1 ... id_1000]) for T_item
     │
     ├─ Partition by shard: { s_0: [...], s_1: [...], ..., s_255: [...] }
     ├─ Fan out 256-way (most shards see few IDs; p99 shard is the 
     │                    tail; over-provision slots per shard)
     │   Each shard: perfect-hash lookup → gather 72B rows into a 
     │   contiguous response buffer, one memcpy per ID.
     └─ Assemble at client: [1000 × 72B] → GPU layout [1000 × 64 int8]
        Dequantize on GPU using per-row scale/zero_pt (fused kernel).
```

**Compression choice: int8 per-row scale.** Evaluated:

- fp32: 256B/row — 4× storage, 4× wire, no quality gain vs int8 w/ calibration.
- fp16: 128B/row — 2× savings, rejected because int8 per-row QAT matches it offline.
- int8 per-row scale (**chosen**): 72B/row — ~0.01% NE regression on ads ranking with QAT.
- int4 groupwise scale: 40B/row — 0.05–0.1% regression; reserved for the largest tables (>1 TB) where wire savings dominate quality.
- PQ-8×16: 8B/row — 0.5%+ regression at ranking resolution; fine for retrieval, not ranking. [STAFF SIGNAL: embedding-specific thinking]

**Training-serving equivalence for embeddings**: the model trains on dequantized int8 (with the scale/zero_pt that will exist at serving) — quantization-aware training. Train on fp32, deploy int8, and a ~0.1% NE gap materializes silently. QAT costs ~15% extra training FLOPs and is non-negotiable. [STAFF SIGNAL: invariant-based thinking]

**Sharding by entity-id hash** (not popularity): popularity-based sharding concentrates hot entities on dedicated shards, creating a single point of hotspot load. Hash sharding spreads load; hot-entity handling is a **replication** concern (section 6), not a sharding concern. Consistent hashing with 256 virtual per physical shard for rebalance without full reshuffle.

### 5.3 Training/Serving Skew Prevention

```
   ┌──────────────────────────────────────────────────────────────┐
   │  FEATURE DEFINITION (single source of truth)                 │
   │  ┌────────────────────────────────────────────────────────┐  │
   │  │  DSL file per feature:                                 │  │
   │  │    feature user_clicks_last_5min:                      │  │
   │  │      window: 5m sliding                                │  │
   │  │      source: event_stream                              │  │
   │  │      agg:    count                                     │  │
   │  │      key:    user_id                                   │  │
   │  │      default: 0     # explicit, single-source          │  │
   │  └────────────────────────────────────────────────────────┘  │
   │         │                                 │                  │
   │         ▼                                 ▼                  │
   │  ┌──────────────┐                  ┌──────────────┐          │
   │  │ Flink codegen│                  │ Spark codegen│          │
   │  │ (online path)│                  │ (offline)    │          │
   │  └──────────────┘                  └──────────────┘          │
   └────────│───────────────────────────────────│─────────────────┘
            │                                   │
            ▼                                   ▼
     ┌────────────┐                       ┌──────────────┐
     │ Online FS  │ ──── served ────────► │ Feature log  │ ← source of
     │ (Scylla,   │      values           │ (Kafka;      │   truth for
     │  emb svc)  │      logged ─────────►│  immutable)  │   training
     └────────────┘                       └──────────────┘
            │                                   │
            ▼                                   ▼
    ┌─────────────┐                      ┌──────────────┐
    │  Ranker     │ ─── outcomes ──────► │ Join(logged  │
    │  (serves)   │     labels           │  feats,      │
    └─────────────┘                      │  labels)     │
                                         └──────────────┘
                                                │
                                                ▼
                                         ┌──────────────┐
                                         │ Training data│
                                         └──────────────┘
```

Five skew sources; defense per:

1. **Code-path skew** (different logic computing the same feature offline vs online). Defense: single DSL with codegen. Tradeoff: DSL can't express every feature; ~5% need custom code with both paths implemented plus a parity CI that diff-checks on a daily sampled workload (alerts on >0.01% value mismatch).
2. **Time-travel skew** (training uses end-of-day snapshot; serving sees as-of-now). Defense: **log-and-train**. The training row literally *is* the feature vector that was served, labeled after the fact. This is the single most important defense and eliminates point-in-time-correctness bugs by construction. Storage cost: ~500 GB/day at 1M rps × 50 sampled features × 100B × 10% sample — acceptable.
3. **Freshness skew** (online is seconds stale; training assumes zero staleness). Defense: feature logs capture *as-served* staleness, not ideal staleness. Training sees the same staleness distribution as serving.
4. **Missing-value skew** [STAFF SIGNAL: missing-value discipline]. Defense: DSL declares default; both codegens use it; feature log records whether the value was a default (sentinel bit). Distinguish "missing because new entity" (normal; model should handle) from "missing because pipeline broken" (incident; mask feature, do not train on garbage). The sentinel is two bits: `{present, new_entity, pipeline_broken}`. Model architecture can optionally consume the sentinel as a feature — lets the model learn to ignore broken-pipeline values during inference.
5. **Distribution drift** (world changes; model stays fixed). Defense: KS-test / PSI on serving distribution vs training distribution, per feature, hourly. Alert on KL divergence above threshold. Not a skew fix — a detection layer for model staleness.

Invariant: `feature_value_served(t) == feature_value_used_in_training(log_row_at_t)`. Enforced by log-and-train for 95% of features; weaker invariant (within ε) for the 5% custom-code features, enforced by parity CI.

### 5.4 Real-Time Features and Freshness

```
Event → Kafka → Flink (keyed stream, RocksDB state) 
                  │
                  ├──► (a) Scylla sink (authoritative)
                  ├──► (b) changelog Kafka ─► ranker sidecar refresh
                  └──► (c) feature-logger tap ─► training log
```

End-to-end budget: event-to-served p95 under 10s.
- Event → Kafka ingest: 50ms
- Kafka → Flink consume: 100ms
- Flink windowed agg + sink: 200ms
- Scylla write + replication: 50ms
- Changelog → ranker sidecar propagation: 2–8s (dominant term)

**Freshness-cost tradeoff** [STAFF SIGNAL: freshness-cost tradeoff]: sub-second freshness requires the ranker to bypass sidecar and hit Scylla directly, ~4–6ms per lookup. At 50 counters per request that's 200–300ms of budget — infeasible fleet-wide. 5–10s via changelog propagation fits the budget. Achieving 1s would require a separate super-fresh path just for counters that demonstrably need it (fraud signals) — roughly 10× infra cost per such feature. **Committed position: 10s freshness default; 1s path reserved for a whitelist of ~5% of features.**

**Read-after-write for a user's own session**: a user who just clicked expects the click reflected in next-request features. If session-affine to one ranker box (usually true), we tap the event stream at the ranker and locally update in-process counters on the way to Flink. The authoritative value is still what Flink produces; the local tap is a low-confidence prediction reconciled within seconds. Deliberate weakening of consistency for latency — documented, invariant-bounded (bound: local value ≤ global value + 1).

### 5.5 Cardinality Defense

[STAFF SIGNAL: cardinality defense] A team ships `click_ctr_by_user_x_item_x_context` keyed on the triple. Expected cardinality: 100M × 10M × 1K = 10^18 keys. Storage and serving cost go to infinity.

Defenses at five layers:

1. **Definition-time (DSL)**: every feature declares `max_cardinality`. Features above 10^9 keys require architecture-review signoff. The DSL compiler rejects undeclared cardinality.
2. **Ingest-time**: the feature registry tracks per-feature unique-key growth rate. Alert fires on >10× WoW growth. Flink job is auto-paused pending owner review.
3. **Storage-time**: per-feature shard quota. Exceeding quota triggers sample-on-write (1% of keys retained); old keys TTL. Feature degrades to "sampled" status, visible to models as metadata.
4. **Query-time**: missing keys return feature-level default. The ranker sees "no signal" for the long tail, which is correct.
5. **Cleanup**: every feature has `owner` and `last_queried_by_active_model` annotations. After 90 days with no active reader, a weekly reaper proposes deletion; owner has 14 days to object. No objection → delete.

Without these, the feature store fills with abandoned features — I've seen 40% storage waste from dead features in two prior orgs. This is the same class of problem as unbounded-cardinality metrics in observability TSDBs; the governance model transfers.

## 6. Failure Modes and Graceful Degradation

[STAFF SIGNAL: failure mode precision] [STAFF SIGNAL: blast radius reasoning]

**Single embedding shard down (1/256 = 0.4% of rows unavailable).** Policy: cold-tier fetches that fail return a **learned-default embedding** (not zeros — zeros may be in-distribution for some rows and cause silent bias). Quality impact: 0.4% of candidates get defaults, estimated ~0.1% NE loss on ads ranking. Alerting: shard-availability metric. Recovery: shard has hot standby; promotion <30s.

**Flink job lag on counters (5 min behind real time).** Sidecar keeps serving last-known-good. If lag > 30 min per-counter staleness monitor fires; the ranker is told to mask affected features (replaced with default + a "stale" sentinel bit — the model was trained to consume this). Quality: ~0.05% CTR loss while degraded.

**Hot-entity replication staleness**: embeddings update on 6h cadence, not second-to-second, so replica drift is bounded by replication SLA. Not a realistic failure at this time resolution.

**5% of features stale or missing simultaneously.** The yellow-zone scenario. Empirically on two prior systems: 0.2–0.4% NE regression — noticeable, not catastrophic. Runbook: shift 10% traffic to a fallback simpler model using only the core 50 features, monitor. If quality gap > 1%, failover simple model to 100% traffic while issue resolves. GameDay quarterly for this.

**Feature store under DDoS / runaway internal client.** Per-tenant rate limits at the gateway. Dedicated shards for critical tenants (separate QPS domains). Shed-load ordering: non-critical readers get 503 before critical ones.

**Ranker loses in-process tier (restart).** 5-minute sidecar-driven warmup before box enters the LB. Cold boxes have 3–5× tail latency; keeping them out until warm preserves p99.

**Hot-entity surge (viral item hits 100K QPS).** Detection: per-key QPS counter in the embedding service. Mitigation (layered): (a) replicate the hot row to all shards (cost: +size × shard_count, negligible for a few hot rows); (b) client-side caching in ranker in-process tier (already happening via hot-set controller); (c) if sustained, pin the row to in-process on every box. Consistency: hot rows update via gossip broadcast rather than polling; update latency bounded by gossip epoch (~1s).

## 7. Multi-tenancy, Versioning, Lifecycle

[STAFF SIGNAL: operational longevity] Features versioned by content-hash of DSL definition + input schema. An edit produces a new version; old version still served until readers migrate. Namespace: `team.feature_group.feature@vHash`. One version per job; a DSL edit spawns a parallel Flink job.

**Storage overhead from versioning**: typically 3–5 live versions per feature. 2K features × 4 versions × 2 GB avg = 16 TB overhead — acceptable.

**Lifecycle states**: `experimental` (sampled writes, no prod readers), `production` (full writes, any reader), `deprecated` (writes continue, no new readers, 90-day sunset). A deprecated feature with zero readers for 14 days auto-deletes.

**The hard part is not versioning but migration.** Without an SLA — "a feature version is supported for at most 12 months past its successor's release" — versions accumulate forever. I'd own this SLA at the feature-platform level with automated PRs to readers of deprecated features, owner escalation after 60 days, and automatic reader-side fallback to the new version (with correctness gating) at 90 days.

## 8. Recent Developments

[STAFF SIGNAL: modern awareness]

- **Uber Michelangelo / Palette**: pioneered log-and-train as the canonical skew defense. Worth borrowing wholesale. Only reason not to: training on historical data not yet served (new-model cold start on a surface that hasn't run). For that, point-in-time batch joins are the fallback — but they should be the minority path.
- **Meta's embedding infrastructure (ZionEX, TorchRec)**: the largest embedding tables (>100 TB) use disaggregated memory pools with RDMA from trainers. Same pattern applies on the serving side when tables outgrow per-box RAM. Not needed for a 500 GB hot working set but the right pattern at 10 TB+.
- **Matryoshka embeddings (Kusupati et al., 2022)**: train embeddings so the first-k dims are themselves a valid (lower-quality) embedding. Lets you serve 32-dim to cold candidates and 128-dim to top-k, cutting wire cost for the long tail. Worth piloting on item embeddings specifically.
- **Feast / Tecton**: Feast is a control plane (registry, lineage, offline joins), not a serving substrate at 1M rps. Tecton's unified substrate is production-viable to ~100K QPS; past that the abstraction taxes — DSL constraints on window types, aggregation functions — become real.
- **Flink 1.17+ with async I/O + tuned RocksDB**: modern default for streaming features. Kafka Streams is a sibling for small topologies.
- **Published skew postmortems**: Meta's 2022 IG ads training-serving skew (~1% revenue over 3 weeks before detection) reshaped the industry's monitoring posture. Continuous serving-distribution vs training-distribution comparison, per feature, is now a required runtime invariant in serious orgs.
- **Vector DBs (Milvus, Qdrant, Pinecone)**: optimized for ANN search, not exact-ID batch fetch. At ranking scale we want the latter — a specialized embedding service outperforms vector DBs on candidate-batch fetch latency. Vector DBs are the right substrate for candidate generation, not ranking.
- **Streaming feature platforms (Chronon @ Airbnb, Zipline @ Stripe)**: both publish the "single definition, dual execution" pattern and log-and-train as standard practice. Confirms the direction.

## 9. Tradeoffs Taken and What Would Change Them

- **If embedding tables exceed 10 TB hot**, on-box tier becomes infeasible. Redesign: RDMA fabric with disaggregated embedding pool; ranker boxes hold only a prediction cache. Meta ZionEX-style.
- **If all features required sub-1s freshness**, the sidecar changelog path is too slow. Redesign: direct Scylla hits for fresh features, probably with per-feature QoS classes and a smaller total feature count (freshness bought from feature count at fixed budget).
- **If we served 10× more concurrent models**, namespace isolation and per-tenant shard quotas become critical; the single-registry design creates a control-plane hotspot. Redesign: federated registry with per-tenant control planes.
- **If log storage becomes prohibitive** (>10 PB/yr), sampled logging per user with reservoir sampling; trainer weights samples. Measurable quality impact but bounded; worth it for storage cost.
- **If the business wins on retrieval rather than ranking**, the 50K-per-request assumption dissolves; architecture shifts to ANN-first with lighter feature sets at ranking time — likely a better overall product win, worth exploring before committing to 50K-feature infra.
- **If skew postmortems reveal log-and-train misses something structural** (e.g., the feature log itself has bugs), add parity checks between warehouse recomputes and logged features on a 1% sample daily. Catches log-pipeline bugs that would otherwise be invisible.

## 10. What I'd Push Back On

[STAFF SIGNAL: saying no]

1. **50K features per request is likely excessive.** Feature-pruning studies consistently find 30–50% of features are pass-through noise. Before scaling infra, fund a pruning initiative. It's cheaper, reduces operational surface area, and buys back latency headroom.
2. **20ms uniform budget is the wrong allocation.** A small number of features (top embeddings, a handful of counters) drive most discrimination. Budget should be allocated by feature importance, not uniformly. Unimportant features live in cheapest tier with loosest freshness; important features get in-process treatment regardless of raw hit rate. Probably a 20% efficiency win.
3. **"Unified offline/online feature store" is often an X-Y problem.** The real goal is skew elimination. Unified **definition** gets 95% of that; unified **execution** is expensive and constraining. Insist on the former; resist the latter unless explicitly better than log-and-train.
4. **Freshness requirements are probably over-specified.** For the 80% of features that are historical aggregates on 7–30d windows, sub-minute freshness is pointless — the signal moves on a day timescale. Feature-by-feature freshness SLAs unlock 10× cost savings on the long tail.
5. **"Design the online feature store" treats the feature store as the system.** The actual system is model + features + ranker. If the model can tolerate lossy compression, defaulted missing values, and a smaller feature count, infra gets dramatically cheaper. The cross-team conversation with modeling — "what degradation tolerances can you absorb?" — is worth 10× any internal infra optimization. A staff engineer owns that conversation, not just the infra.