---
title: Distributed counter at scale
description: Distributed counter at scale
---
```
"Design a system that counts events (think: video views, like counts, ad impressions) with eventual consistency. 200B events/day, counters need to be readable within a few seconds of write. Tail latency on reads matters."
```
---
# Counter System at 200B events/day
 
## 1. Scope and requirements

Before committing to a design: "video views, likes, ad impressions" are three different systems. They share infra but not correctness contracts.

**[STAFF SIGNAL: scope negotiation]** The three are:

- **View counter**: display number. Off-by-1% is invisible. Drift-tolerant. Dimensional rollups (country, device, referrer) are the product. This is the cheapest semantic.
- **Like counter**: per-user idempotency is the whole problem. A user liking twice must collapse to one. The hot key here is the *creator*, aggregated across their content. Scale is 10–100x smaller than views but the dedup set is user × content, which does not fit in RAM.
- **Ad impression counter**: billing-grade. Undercounting costs advertisers trust, overcounting costs the platform millions per day. Audited against SOX-style controls. Exactly-once is not a lie here — it's a contract. This is a completely different system and if the interviewer is asking for this under "eventual consistency," they've mis-specified.

I'll design for **video view counters** as primary, with a clear note on what changes if it were ads. View counter semantics: at-least-once ingestion, best-effort dedup within a short window, display counts eventually correct to within ~0.1% of the "truth" (which is itself ill-defined), dimensional rollups supported, no billing linkage. **[STAFF SIGNAL: correctness semantics precision]**

Requirements I'm committing to:
- 200B events/day globally. Peak burst factor 3–5x (Super Bowl, World Cup, K-pop drops). Peak = ~10M events/sec.
- Read freshness: "few seconds." I will push back on this in §9 — the cost delta between 1s and 10s is large and "few seconds" is almost certainly not a real product requirement.
- Read latency: p99 < 10ms on counter lookup, p999 < 50ms. This one is real; it's on the video playback hot path.
- Dimensional rollups: `count by (video, country, device, hour)`. Cardinality on video is ~1B lifetime, ~100M active (read in last 30d).

## 2. Capacity math

```
Writes
  Avg:      200B / 86400 = 2.31M events/sec
  Peak:     ~10M events/sec (5x burst)
  Event size: 250B on wire after enrichment
             (video_id 8B, user_id 8B, ts 8B, geo 2B, device 4B,
              ref_source 8B, session 16B, headers/framing ~196B)
  Raw bandwidth peak: 2.5 GB/s ingest, 200 Gbps aggregate
  Raw storage:   200B × 250B = 50 TB/day uncompressed,
                 ~12 TB/day with zstd (4x ratio on these fields)
                 → 4.4 PB/year raw archive
Reads
  Video playback emits 1 count read per play. Global plays ≈ video events
  → read QPS ≈ write QPS in steady state ≈ 2.3M QPS avg, 10M peak.
  Mitigated heavily by caching; serving tier QPS after 95% edge hit
  ≈ 115k QPS (avg), 500k QPS (peak).
Hot set
  Zipfian: top 0.1% of videos = ~100k videos get ~50% of traffic.
  Keep top 10M videos (active counters for last 7d) in RAM.
  Per-counter state: video_id(8) + K shard counts (K avg = 4, max 256)
    + last_update_ts(8) + tombstone bits
    ≈ 80B average, worst case 2 KB for hot shards.
  RAM for hot set: 10M × 80B = 800 MB base counters,
    + 1000 hot-sharded videos × 2KB = 2 MB negligible
    → one Redis shard per region (16GB box) holds hot set with headroom
Aggregation ratio needed
  If read path hits Redis for every count, we need commits per key <= 1/s.
  Input: 10M writes/sec across ~100M active keys over a second
    → most keys 0-1 writes/sec, hot keys 1M/sec
  Aggregation job must compress 10M input writes/sec into ~10M output
    key-updates/sec (1:1 naive), OR ~100k/sec if we batch 100ms windows.
  Target: 100:1 input-to-commit ratio via 100ms micro-batch per subtask.
    → serving-tier write load = 100k/s per region. Redis handles this
    at one shard.
```

**[STAFF SIGNAL: capacity math]**

## 3. High-level architecture

```
   clients (web/mobile/TV)
       │
       ▼  HTTPS batch POST, 10 events/req typical
  ┌─────────────────────┐
  │  Edge Collectors    │  stateless, regional, 500 nodes global
  │  (Envoy + lua/Rust) │  auth, schema validate, enrich (geo from IP)
  └──────────┬──────────┘
             │ produce, partitioned by hash(video_id) mod 4096
             ▼
  ┌─────────────────────┐
  │  Kafka: raw_events  │  4096 partitions, RF=3, retention=72h
  │  (or Redpanda)      │  compressed=zstd, batch=64KB, linger=20ms
  └──────────┬──────────┘
             │
             ▼
  ┌─────────────────────┐           ┌────────────────────────┐
  │ Flink: preaggregator│──────────▶│ S3 raw archive (iceberg)│
  │ tumbling 1s, EvTime │           │ 1h partitions, parquet  │
  └──────────┬──────────┘           │ for backfill + audit    │
             │ partial_aggs topic   └────────────────────────┘
             │ keyed by video_id          (trust contract / §6)
             ▼
  ┌─────────────────────┐
  │ Flink: finalizer    │  merges partials across edge preaggs
  │ tumbling 1s, 10s    │  two-phase: hot keys split, see §5.2
  │ allowed lateness    │
  └──────────┬──────────┘
             │ commit path splits:
       ┌─────┴──────────────┬───────────────────┐
       ▼                    ▼                   ▼
   Redis cluster       Pinot (OLAP)        Kafka: commits_log
   (current counts)    (dim rollups)       (downstream fanout)

    reads
   ──────▶ [edge cache] ──▶ [CountService] ──▶ [L1 in-proc LRU]
                                             ──▶ Redis
                                             ──▶ Pinot (cold/dim)
```

I'm using Flink + Kafka + Redis + Pinot. I'll justify each against a specific alternative in §5.

## 4. Pre-aggregation topology

```
  raw_events partition P (single Kafka partition, ~2.5k ev/s avg, 25k peak)
             │
             ▼
  ┌──────────────────────────┐
  │ Preagg Task (Flink op)   │  state: HashMap<video_id, count>
  │                          │  flush: every 100ms OR 10k unique keys
  │  event → localMap[vid]++ │  evict: LRU on 10k cap → emit partial
  └──────────┬───────────────┘
             │
             ▼  partial record: (vid, shard_hint, count, window_start, window_end)
  ┌──────────────────────────┐
  │ keyBy(vid)               │  <-- shuffle boundary, network cost here
  │  + hot-key rebalance     │      hot keys get split; see §5.2
  └──────────┬───────────────┘
             │
             ▼
  ┌──────────────────────────┐
  │ Finalizer Task           │  state: Map<vid, running_count>
  │  window: tumbling 1s     │  backed by RocksDB (Flink state backend)
  │  trigger: every 1s       │  checkpoint: 30s, incremental, S3
  │  allowed lateness: 10s   │  late events → side output → reconciler
  └──────────┬───────────────┘
             │
             ▼ committed delta: (vid, window, delta_count)
         serving tier
```

Windowing model: tumbling 1s event-time windows with a 10s lateness allowance, early-firing trigger every 1s. Late events past 10s go to a side output that feeds the reconciler (§6.3). Watermark = min across sources, which in practice is bounded by the slowest edge collector clock skew (~2s).

## 5. Deep dives

### 5.1 Pre-aggregation: partial aggregation, aggregation ratio, shuffle cost

The 100:1 aggregation target in §2 is a design constraint that drives everything downstream. The naive "emit one record per event to keyBy" pushes 10M recs/sec across the shuffle boundary, which is 200Gbps of intra-cluster traffic just for the shuffle — infeasible on most clusters.

The preagg task keeps a local hashmap, no state backend, not checkpointed — it's deliberately volatile. On a 100ms flush tick or 10k-key cap, it emits partials. For Zipfian key distributions, a single preagg task sees ~5k unique keys per 100ms from 25k events → 5:1 local reduction. Not enough. Extend the flush to 500ms and we get 20:1, which is workable. Tradeoff: 500ms of additional p99 latency on the write path.

Why not stateful preagg with Flink-managed state? Because the preagg is pre-shuffle, so it's running on partitions keyed by ingestion order, not by video_id. Keeping Flink-managed state here adds checkpointing cost for state that's fundamentally transient — we'll re-aggregate downstream anyway. The local volatile hashmap is the right abstraction. **[STAFF SIGNAL: rejected alternative]** Cassandra counter columns — rejected because a single counter column write at 1M ops/sec to a hot partition causes SSTable write amplification that cascades into compaction storms; we observed this pattern at Meta on analogous workloads and it's documented in the Cassandra 4.x release notes around counter accuracy improvements. Counters in Cassandra are also non-idempotent on retry, which makes the exactly-once story worse, not better.

Shuffle optimization: the partial records have a `shard_hint` that lets the keyBy partitioner use `(video_id, shard_hint)` for hot keys and just `video_id` for cold keys. This requires a sideband hot-key registry — see §5.2.

### 5.2 The hot counter problem

This is the problem. Every other decision is downstream of solving this.

Observation: for a Super Bowl ad running globally, a single `video_id` can see 1M events/sec sustained for 30s. A single Flink task can commit roughly 100k state updates/sec to RocksDB with incremental checkpoints. So the naive keyBy(video_id) → single-task-per-key pattern melts at 10x below the peak.

**Solution: adaptive key splitting with a write-side shard registry and read-side fan-in.**

```
  Write path for key V with current shard count K_V:

  event(V)
    │
    ▼
  lookup K_V from shard registry (Zookeeper/etcd, 100ms-stale OK)
    │
    ├── K_V = 1  ──▶ emit (V, 0, +1)            [cold key]
    ├── K_V = 8  ──▶ emit (V, rand%8, +1)       [warm]
    ├── K_V = 64 ──▶ emit (V, rand%64, +1)      [hot]
    └── K_V =256 ──▶ emit (V, rand%256, +1)     [scorching]
                     + route to dedicated
                       "celebrity" worker pool

  Storage layout in Redis:
    HSET video:V shard:0 <count>
    HSET video:V shard:1 <count>
    ...
    HSET video:V K <K_V>   (metadata)
    HSET video:V updated <ts>

  Read path:
    HGETALL video:V  (single round trip, pipelined)
    count = sum(shard:i values)
    latency: ~1 ms for K<=64, ~3 ms for K=256

  Promotion policy (runs every 10s):
    observed_qps(V) > 50k/s  AND K_V < 8     → K_V := 8
    observed_qps(V) > 500k/s AND K_V < 64    → K_V := 64
    observed_qps(V) > 5M/s   AND K_V < 256   → K_V := 256

    demotion: observed_qps(V) < K_V * 1000/s for 1h → halve K_V
    (halving consolidates shards to shard:i for i < K_V/2)
```

Why shard counts in powers of 2 with explicit registry rather than consistent-hash-style uniform splitting? Two reasons. First, reads cost O(K) network ops — we don't want every cold key paying 256x read cost. Second, promotion requires a rewrite of historical data when K increases; with discrete tiers we do this 3 times per key max, not continuously.

The **two-phase aggregation** fallback for anything not yet promoted:
- Phase 1 (local): preagg task aggregates `(V, shard_hint=null)` locally.
- Phase 2 (keyed): finalizer receives `(V, count_delta)` and commits.

When phase-1 detects a local key with >1000 events in a single 100ms flush, it auto-assigns a random shard_hint and emits N records. This is the hot-key "first response" before the registry catches up and promotes K_V globally. It prevents melting while the control plane is reacting.

**Failure mode**: what if the shard registry is unreachable? Edge collectors cache the last-known K_V for 10 minutes locally. A full 10-minute registry outage causes new hot videos to be under-sharded and hot-spot for that window. Acceptable because registry is replicated and this is a correlated failure the blast radius doc addresses. **[STAFF SIGNAL: failure mode precision]**

**Threshold math**: promotion at 50k/s assumes a single Flink task handles ~100k state ops/s, so we promote at 50% saturation to give headroom for the promotion itself. These numbers are hardware-specific; on newer NVMe-backed RocksDB setups I've seen 300k/s per task sustained, which would shift the promotion threshold to 150k/s.

### 5.3 Idempotency, exactly-once-ish, and the dedup problem

**[STAFF SIGNAL: correctness semantics precision]** "Exactly-once" across a distributed stream is a lie we tell ourselves when three things are all true: the producer is idempotent, the stream processor has transactional checkpoints, and the commit is idempotent at the sink. If any of those is false, you have at-least-once with small duplicate rates.

For this system:
- **Producer idempotency**: Kafka idempotent producer (per-session dedup on `<producerId, partition, sequence>`). Survives retries within a session. Does NOT survive producer restart — the session ID changes. So edge collector restarts produce duplicates.
- **Stream processor**: Flink with incremental RocksDB checkpoints every 30s, two-phase commit to Kafka commits_log. Re-processing on failure replays from the last checkpoint, and the commits_log dedup is sequence-based.
- **Sink commit**: Redis `HINCRBY` is idempotent only if deltas are tagged with `<window_id, shard_id>` and we keep a Bloom filter of seen `window_id`s (1MB per 10M windows at 1% FP). Cheap.

**Dedup window for client events**: events carry `<session_id, sequence_number>`. We keep a Redis Bloom filter per session (1 KB, TTL 60s). Client retries within 60s collapse; retries after 60s (app backgrounded, resumed) get counted as a second event. This is a conscious product decision — we are not building per-user strict dedup for views. If we were (likes, ad impressions), this would be a per-user Bloom filter with hours-to-days TTL, which pushes the dedup state from MBs to hundreds of GBs and forces a sharded architecture for it.

**Restart semantics**: Flink crash at t=T. Job restarts from checkpoint at T-30s. Replays 30s of Kafka, reconstructs partial aggregates. The commits_log has a `window_id` sequence so downstream sinks reject replays. Redis already applied some deltas pre-crash; replays with same `window_id` get rejected by the Bloom filter. Consumers see no double-counting but may see counts pause for ~30s during recovery. **[STAFF SIGNAL: failure mode precision]**

### 5.4 Read path architecture

Three tiers:
1. **Edge cache** (Varnish/envoy): 1s TTL per counter. Absorbs 90%+ of read load for hot videos. Staleness: ≤1s, which is within the "few seconds" requirement without any work from the backend.
2. **Count service** with in-process LRU L1 (100k entries per instance, 100ms TTL). Shields Redis from cache miss storms during edge deploys.
3. **Redis cluster** sharded by `hash(video_id) mod M`. 32 shards, 2 replicas each, per region. ~2M QPS per shard ceiling on modern hardware with pipelining.

Why not serve directly from Pinot/Druid/ClickHouse? **[STAFF SIGNAL: rejected alternative]** Pinot's p99 at 500k QPS on a keyed lookup is ~50ms, vs Redis at ~1ms. At 10ms p99 SLO, Pinot fails by 5x. We use Pinot for *dimensional* queries (`views by country last 24h`) where the query is not a single point lookup. For the bare counter, Redis wins on tail latency by an order of magnitude.

Why not DynamoDB or Cassandra? Again tail latency: DynamoDB p99 ~10ms but p999 can spike to 100ms during partition splits; Cassandra's counter columns have compaction-induced p999 issues noted in §5.1. Redis with a warm hot set is the right answer for the display counter. The cold tail goes to a KV fallback (DynamoDB) with a 50ms SLA — fine because by definition nobody is waiting on the count for a video with 7 views.

**Degradation under write pipeline lag**: if Flink falls behind, Redis counts go stale. Read path returns stale count + a `staleness_seconds` header. The player UI can decide to show "views updating" if staleness > 30s. The counter is still available, just drifting, which is the correct behavior under eventual consistency.

### 5.5 Late-arriving data and backfill

**[STAFF SIGNAL: precision under ambiguity]** Policy: events with event_time within 10s of watermark are included in their window. Events 10s–2h late go to a **late side output** feeding a reconciler job that emits a `count_correction` record every 10 minutes. Events >2h late are written to S3 for the next daily restate job. We do NOT drop late data silently — dropping is a correctness fiction.

**Backfill mechanic for a 3-hour double-counting bug**:
```
1. Detect via reconciler divergence (§6): live count vs batch recompute
   from S3 raw events diverges by >0.1% for window W.

2. Compute corrections: batch Flink job reads S3 raw for [T-3h, T],
   re-applies dedup, produces (vid, correct_count) for the window.

3. Compute delta: correct_count - live_count per vid.

4. Shadow apply: write corrections to Redis with key prefix
   shadow:video:V for 10 minutes. Dashboard team runs comparison.

5. Atomic swap: MULTI/EXEC rewrite video:V → shadow value.
   Publish count_correction event to downstream consumers so
   anyone who already ingested the wrong value can compensate.
```

The compensation story for downstream consumers is the hard part. If a creator-analytics pipeline already snapshotted the wrong count into their daily report, sending them a correction 3 hours later means their daily report is wrong. You need versioned counters: every counter read carries a `generation_id`, and the correction increments the generation. Downstream consumers can detect stale reads and recompute. **[STAFF SIGNAL: blast radius reasoning]**

### 5.6 Storage, cost, and the long tail

The dominant cost isn't the hot-counter RAM — it's the **billion-video tail**. Most videos get <10 views/day. Keeping them in Redis is wasteful:

```
Counter distribution (estimated):
  top 10M videos:    in Redis            800 MB × 3 replicas × 5 regions = 12 GB
  next 100M videos:  in DynamoDB         100M × 80B = 8 GB, $200/mo
  long tail 900M:    in S3 + Pinot       queried rarely, merge-rollup daily
  
Raw archive: 12 TB/day compressed × 90d = 1 PB on S3 standard
             $23k/mo at $0.023/GB. Post-90d → Glacier, $4/TB-mo.
             Annual archive cost ~$400k. Compared to ~$10M/yr cluster cost,
             archive is <5%.
```

**TTL policy**: counters idle for 30d evict from Redis to DynamoDB. Counters cold 365d go to Pinot + S3-backed rollup tables. A background job rolls per-day counters into per-week and per-month bands to shrink the tail.

### 5.7 Fan-out dimensions

Views-by-(country, device, ref_source, hour) is the product ask. Naive materialization of all dimension combinations is O(|vid| × |cntry| × |dev| × |ref| × |hr|) which is 100M × 200 × 10 × 50 × 24 = 2.4 × 10^13 cells. Infeasible.

Three-level strategy:
1. **Full cardinality base counter** in Redis: just `(video_id)`.
2. **Pre-materialized common rollups** in Pinot via a star-tree index: `(video, country)` and `(video, country, hour)`. Pinot's star-tree materializes the aggregation tree incrementally — writes are O(dims) not O(cells). Query latency p99 < 50ms for these two rollups.
3. **Query-time aggregation** for anything not pre-materialized: Pinot scans raw events with predicate pushdown. Latency 1–5s, fine for analytics UI.

The cardinality defense: we cap per-video dimensional cardinality. If a single video has >1000 unique `ref_source` values (bot traffic signature), we bucket the tail into `ref_source=other`. Protects Pinot index explosion from adversarial inputs.

### 5.8 Observability and the trust contract

At 10M events/sec, we cannot log every event's fate. The trust contract is built from three independent measurements:

1. **Live pipeline count**: sum of commits emitted by finalizer per window.
2. **Independent batch recompute**: hourly Spark job over S3 raw archive, applies same dedup logic, emits per-window counts.
3. **Sampled end-to-end probe**: synthetic events injected at edge with unique `probe_id`s, traced through to Redis commit, latency measured.

Reconciliation metric: `|live - batch| / batch` per 1h window. Alert if >0.1% for 3 consecutive windows. This is the single most important metric the system emits. **[STAFF SIGNAL: cross-cutting concern]** It's also the primitive that any downstream billing-adjacent system (ads, creator monetization) consumes to decide whether to trust a count.

Anomaly detection on counter derivatives: `d(count)/dt` for each video. A 10x jump in a minute without a corresponding trending-signal input is a bot or a double-counting bug. Feed these into the abuse-detection pipeline for creator-integrity review.

## 6. Failure modes (not an afterthought)

**Kafka raw_events partition leader election storm** (a broker dies during peak): producers buffer ~30s, then start dropping. Edge collectors fall back to local NVMe spool (100GB per node, 2 hours of events buffered). Kafka recovers, collectors drain in reverse chronological order (newest first, oldest last) so fresh counts resume quickly and history fills in. Watermark stays low during drain. **[STAFF SIGNAL: operational reality]**

**Flink finalizer OOM during a hot-key promotion**: the just-promoted K=256 video's state snapshot is 200KB instead of 2KB. Cascading snapshot costs blow through heap. Mitigation: hot keys use a separate job with dedicated heap; promotion routes the key and its state via savepoint-and-replay into the hot job. 5-minute interruption on that key's counter, consumers see staleness header.

**Redis shard failover**: 30s unavailability for the affected shard's hot set. CountService has a stale-read fallback to DynamoDB with an "approximate" marker. Display shows count from 5 minutes ago with UI indicator. Acceptable.

**Dedup Bloom filter state loss**: session dedup window resets; short window of ~1% duplicate counting. Detected by reconciler, corrected in next restate cycle. Published as a known incident.

**The "double-counting bug in production for 3 hours" scenario** (§5.5 already covered the mechanic): the key insight is that your system must have versioned counters and a correction event stream *before* the bug happens. Adding them after is twice the work and leaves corrupted historical data.

## 7. Recent developments

**[STAFF SIGNAL: modern awareness]**

- **Apache Paimon / Iceberg V3 for the S3 archive**: streaming-native table formats with primary-key merging at read. For this system, Paimon's bucketed LSM-on-S3 lets us treat the archive as a queryable table with CDC, which means the reconciler (§5.8) becomes a straight SQL join rather than a custom Spark job. Iceberg V3 added deletion vectors and row-level updates that similarly simplify corrections.
- **Pinot's upsert tables with star-tree**: Pinot post-0.12 supports upsert semantics on dimensional aggregates, so a correction event can update a rollup cell directly rather than forcing full segment rewrites. Big operational win for the backfill story.
- **Redpanda vs Kafka at this scale**: Redpanda's per-core thread-per-shard architecture and Raft-native replication cut p99 produce latency roughly in half for small-batch workloads, but the ecosystem (connectors, Flink integration maturity) still favors Kafka for a counter pipeline. For the raw ingest tier where tail latency matters less than throughput, Kafka stays. For a realtime-correction sidecar topic, Redpanda is a legit choice.
- **Flink 1.18+ with disaggregated state**: Flink's async state backend and the disaggregated state proposal (ForSt) let RocksDB live on remote storage, which reduces the checkpointing cost for hot-keyed jobs dramatically. For the hot-key finalizer, this matters — the 200KB-per-promoted-key checkpoint cost stops being a node-local disk pressure problem.
- **RisingWave / Materialize**: streaming databases with incremental materialized views. Tempting for the dimensional rollup problem, but at 10M events/sec sustained neither is production-proven at that throughput per my read as of early 2026. Watch, don't bet.
- **CRDT counters (PN-counters) at scale**: papers and systems (Riak, AntidoteDB) show that multi-region active-active counters with CRDT semantics work for up to ~10k writes/sec per counter. They do NOT solve the hot-counter problem at 1M/sec because the merge cost per replica scales with the number of active nodes. CRDT is the right answer for *geo-replication* of the Redis layer across regions, not for the write path.
- **Kafka KIP-848 (new consumer group protocol)**: reduces rebalance pauses for long-running Flink consumers, which matters because Flink rescaling was historically a 30s+ operation that manifested as visible counter stalls. Production-ready in Kafka 3.7+.

## 8. Tradeoffs I'm taking and what would force redesign

- **At-least-once ingestion with window-bounded dedup** is acceptable because views are display-only. If this were ads, I'd move to transactional producer + Flink exactly-once sink + per-impression persistent dedup keyed on `<user, ad, minute>`. That adds ~40% cost and halves throughput per node.
- **Hot-key splitting with discrete tiers** accepts O(K) read cost for hot videos. If reads became more expensive than writes (e.g., a new product feature that hammers counters from every search result card), I'd consider a read-side cache that stores the summed value with 1s TTL, pushing read cost back to O(1) per shard. Currently not worth the staleness.
- **1s commit windows** accept ~1s min staleness. If "few seconds" got tightened to sub-second (it won't, but hypothetically), I'd need to kill Flink's windowing and go to continuous aggregation with periodic snapshots, roughly tripling the state-backend pressure. Not worth it for views.
- **Single-region Redis with geo replication**: if the product requires cross-region strong consistency (it doesn't for views), the whole architecture changes. CRDT PN-counters become the answer and read latency goes up.

## 9. What I'd push back on

**[STAFF SIGNAL: saying no]**

1. **"Few seconds of write" freshness**: for a view counter, users cannot distinguish 1s from 30s. The product requirement is probably "user sees count update within a page-reload cycle," which is 30–60s. Tightening to seconds costs us the ability to buffer and pre-aggregate more aggressively, which is a 2–3x cost multiplier. I'd get the PM to defend 5s with data. If they can't, we ship with 30s and save the money.

2. **"Dimensional rollups for all dimensions from day one"**: the star-tree approach handles a few common ones cheaply; all of them costs 100x storage. I'd ship base counter + `(video, country, hour)` in v1, add dimensions as product validates demand. Resist the "future-proof all dimensions" instinct.

3. **"One system for views, likes, and ad impressions"**: tempting for engineering tidiness. Bad for correctness. Likes and ads have different dedup, audit, and billing contracts. Share the infra substrate (Kafka, Flink framework, S3 archive), split the pipelines.

4. **The interviewer probably assumed "eventual consistency" means "no correctness guarantees"**: it doesn't. Eventual consistency means "the count converges to a defined truth given enough time, and the truth is computable from the raw log." If we can't recompute truth from the archive, we don't have eventual consistency — we have best-effort with drift. The archive-as-source-of-truth story (§5.8, §6) is the thing that makes this design actually eventually consistent rather than just vaguely-approximately-correct.

---

If you want, I can drop into any of the deep dives further — the hot-key promotion/demotion control loop and the backfill + versioned-counter mechanic are the two places where this answer has more depth that I compressed. The star-tree dimensional thing is also a rabbit hole.