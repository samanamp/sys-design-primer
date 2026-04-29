---
title: Data Platform for Text-to-Video Training
description: Data Platform for Text-to-Video Training
---

# Data Platform for Text-to-Video Training — Reference Answer

> **Use case**: Reference for Staff-level system design rounds where the prompt is "design a data platform for distributed training." This is the answer for a 60-min slot, structured so you can deliver any subset in 35-40 min by skipping deep-dive sections. Mental template at the top is deployable in <2 min.
>
> **Staff+ signals** are marked `[STAFF+]` throughout. Senior candidates produce the boxes; Staff candidates produce the *mechanisms*, *trade-offs*, and *failure modes* unprompted.

---

## 0. The 2-Minute Opening Template (deploy on the whiteboard immediately)

For *any* "design an ML training data platform" question, draw this skeleton **before** reading the prompt twice. It applies whether the modality is text, image, video, audio, or multi-modal:

```
        ┌──────────────────┐
        │   Raw Storage    │   blob store, immutable, source of truth
        └────────┬─────────┘
                 │
        ┌────────▼─────────┐
        │  Catalog/Meta    │   queryable index over corpus
        └────────┬─────────┘
                 │  (recipe = filter + seed + version)
        ┌────────▼─────────┐
        │  Recipe Holder   │   defines what to read, in what order
        └────────┬─────────┘
                 │
        ┌────────▼─────────┐
        │ Shuffle / Cache  │   hot tier, sharded, sized for working set
        └────────┬─────────┘
                 │
        ┌────────▼─────────┐
        │ Decode / Preproc │   ⚠ THE HOT PATH FOR VIDEO ⚠
        └────────┬─────────┘
                 │
        ┌────────▼─────────┐
        │  Trainer (GPU)   │
        └──────────────────┘

  Cross-cutting:  Reproducibility • Fault tolerance • Multi-tenancy • Cost
```

Then ask three clarifying questions (box at 5 min), state assumptions out loud, and start specializing each box for the modality. **Modality-specific specialization is where you spend the next 25 min.**

> `[STAFF+]` The senior failure mode is to start drawing without this skeleton, get pulled into one box (usually storage), and run out of time. The Staff move is to put the skeleton up first and *manage time across boxes*.

---

## 1. Requirements & Assumptions (5 min)

### What I'd ask
1. **Modality specifics**: video resolution, fps, codec assumptions, how preprocessing produces model input (raw frames? VAE latents? patches?). This determines whether the hot path is bytes/sec or frames/sec.
2. **Training topology**: 4000 GPUs as 500 nodes × 8 GPU? DP-only or 3D parallel? Determines whether each GPU is an independent reader or whether DP groups share data.
3. **Researcher access pattern**: ad-hoc SQL-style filters, or programmatic? Read-only or curating new subsets?
4. **Concurrent jobs**: one big training run or N concurrent? Affects cache sharing strategy.
5. **SLOs**: target GPU utilization (95%+ implies <5% data-stall budget); job restart RTO.

### What I'd assume out loud and proceed
- 500M videos, 15 MB avg compressed (H.264-ish, ~5 Mbps × 25s ≈ 15 MB ✓)
- Single region, multiple AZs, **co-locate cache with trainers in same AZ** (cross-AZ at 300 GB/s would burn ~$10K/hr in transfer)
- DP=4000 with TP/PP wrapping inside a node; data identical across TP/PP ranks within a DP group → **only ~500 distinct readers** if 8-way TP, not 4000
- Preprocessing produces VAE latents; we will cache latents, not raw frames (huge implication, see §5)
- One large training job is the dominant load; multi-tenancy is a §9 concern

> `[STAFF+]` Naming the **DP-vs-TP distinction** for "how many distinct readers" is a Staff signal — it changes the throughput target by 8×. Senior candidates assume 4000 readers and over-provision.

---

## 2. Capacity Math (do this on the board, show your work)

### Throughput
```
Per-GPU consumption:   5 videos/s × 15 MB = 75 MB/s/GPU       (compressed)
Naive aggregate:       4000 × 75 MB/s     = 300 GB/s
With TP=8 dedup:       500 readers × ... = 37.5 GB/s          (compressed!)
```

But "compressed" is misleading for video. The **decoded** rate is what matters:
```
25s × 30fps × 256² × 3 bytes  = 147 MB raw per video
                                ≈ 10× expansion vs compressed
Decoded rate per GPU:           750 MB/s/GPU
Aggregate decoded:              4000 × 750 MB/s = 3 TB/s
```

> `[STAFF+]` Calling out the **decoded vs compressed expansion** unprompted is the single biggest Staff signal for video. Most candidates compute the compressed number and stop. The decoded number determines whether you decode on CPU (no, can't keep up), GPU NVDEC (maybe), or pre-decode offline (saves bandwidth at cost of preprocessing flexibility).

### Storage
```
Corpus:        500M × 15 MB         = 7.5 PB     (compressed)
Captions:      500M × 1 KB          = 500 GB     (small-object problem, see §3)
Metadata:      500M × 500 B         = 250 GB
Latent cache:  500M × ~2 MB         = 1 PB       (if pre-encoded VAE latents)
```

### Per-epoch math (this is where I had errors before — redoing it)
```
Subset for an epoch:   250M videos    (researcher recipe filters corpus down)
Throughput:            20K videos/s aggregate
Time per epoch:        250M / 20K     = 12,500 s = 3.47 hours  ✓
Bytes per epoch:       250M × 15 MB   = 3.75 PB compressed
                                     = 37.5 PB decoded  (relevance: cache sizing)
```

### Cache sizing
Working set is **not** the full epoch — it's "what we need in the next ~10 minutes" plus shuffle buffer:
```
Aggregate rate:         300 GB/s compressed
10 min working set:     300 GB/s × 600s = 180 TB
Per-shard at 50 GB/s:   need ≥ 6 shards for bandwidth
Per-shard storage:      180 TB / 6 = 30 TB  (achievable with NVMe)
Recommended:            12 shards (2× headroom, replication, straggler tolerance)
```

> `[STAFF+]` Sizing the cache as a **rolling working set, not a full-epoch cache**, is the right framing. Caching the entire epoch (3.75 PB) is wasteful and you don't need it — shuffle determinism + sequential prefetch means you only need enough buffer to hide S3 latency.

---

## 3. Storage Tier (the cold path)

```
                  ┌──────────────────────────────────────────────┐
                  │                  S3 (or equivalent)          │
                  │                                              │
                  │  videos/      packed shards of mp4/webm      │
                  │     part-00000.tar  (1000 videos, ~15 GB)    │
                  │     part-00001.tar                           │
                  │     ...                                      │
                  │                                              │
                  │  latents/     pre-encoded VAE latents        │
                  │     part-00000.tar                           │
                  │                                              │
                  │  catalog/     Iceberg + Parquet              │
                  │     metadata.parquet                         │
                  │     captions.parquet                         │
                  │                                              │
                  │  manifests/   ingestion log (append-only)    │
                  └──────────────────────────────────────────────┘
```

### Key decisions and trade-offs

**Pack videos into tar shards of ~1000 each, don't store as 500M individual objects.**
- S3 GET overhead per object is ~10 ms latency; small-object workloads are bottlenecked by request rate, not bandwidth
- 500M individual GETs/epoch at 20K req/s → 7 hours of pure S3 latency cost
- Packed: 500K GETs/epoch → 25s
- **Trade-off named**: random-access within a shard is gone, but training reads sequentially within a shuffled shard order anyway

**Captions: inline into Parquet metadata, NOT 500M × 1KB S3 objects.**
- Same small-object reasoning. 500 GB Parquet is trivially queryable and bandwidth-friendly.

**Catalog: Iceberg/Parquet + Trino, NOT Postgres.**
- Researchers writing recipes filter on length, resolution, language, source, quality scores, embeddings. Analytical workload, not OLTP.
- Postgres at 500M rows: fine for `WHERE video_id = ?`, painful for `WHERE quality > 0.8 AND lang = 'en' AND duration BETWEEN 10 AND 30`.
- Iceberg gives time-travel (recreate corpus state for old recipes) and partition pruning.
- **Postgres earns a place** for the *recipe definitions* themselves and for the *recipe holder*'s read cursor — small, transactional, HA-needed.

**Append-only ingestion log: S3 manifest files (or Kafka if streaming ingest).**
- Audit trail for "what data was added when" — needed for reproducibility and compliance.
- Cheap, immutable, queryable as Iceberg.

> `[STAFF+]` The **packed-shard format** decision is a litmus test. Senior candidates store 500M objects in S3 because that's the natural mental model. Staff candidates know request-rate is the bottleneck and pack.

---

## 4. Catalog & Recipe Holder

### Catalog (read-heavy, analytical)
```
┌──────────────────────────────────────────────────────────────┐
│  Iceberg table: corpus_v{N}                                  │
│                                                              │
│  video_id PK | shard_id | offset | duration | fps | lang     │
│              | width | height | source | quality_score       │
│              | caption | embedding[512]  | added_ts          │
│                                                              │
│  Partitioned by: source, added_date                          │
│  Sorted within partition by: video_id                        │
└──────────────────────────────────────────────────────────────┘
            │
            │  Trino/DuckDB serves researcher queries
            │
            ▼
   ┌────────────────────────────────────┐
   │  Researcher query example:         │
   │  SELECT video_id, shard_id, offset │
   │  FROM corpus_v3                    │
   │  WHERE quality_score > 0.8         │
   │    AND duration BETWEEN 10 AND 60  │
   │    AND lang IN ('en','es','fr')    │
   │  → 250M rows → save as recipe      │
   └────────────────────────────────────┘
```

### Recipe Holder (the brain of the data platform)
```
┌──────────────────────────────────────────────────────────────┐
│ Recipe Holder Service  (Postgres + small stateless API tier) │
│                                                              │
│  recipe_id:     r-2026-04-29-abc                             │
│  filter_sql:    "quality > 0.8 AND duration BETWEEN ..."     │
│  corpus_version: corpus_v3                                   │
│  seed:          0xDEADBEEF                                   │
│  shard_layout:  {video_id → cache_shard_id} hash function    │
│  read_cursor:   global step → (epoch, shuffle_idx)           │
│  created_by:    researcher@                                  │
│  created_ts:    ...                                          │
│                                                              │
│  Endpoints:                                                  │
│    GET  /recipe/{id}                  (job startup)          │
│    POST /recipe/{id}/cursor           (checkpoint write)     │
│    GET  /recipe/{id}/next_batch       (per-rank pull)        │
└──────────────────────────────────────────────────────────────┘
```

**QPS:** 4000 GPUs × 1 req/batch is misleading — DP ranks pull batches, not GPUs. With DP=500 and batches every ~1s, that's 500 QPS. Trivial. Stateless API tier with 3 replicas behind LB; Postgres in HA pair.

> `[STAFF+]` Storing the **shard layout function** (deterministic hash from video_id → cache shard) inside the recipe is the trick that lets DP ranks find their data without a coordinator on every read. The recipe is enough to deterministically compute "rank K at step S reads videos X, Y, Z from shards A, B, C." This is the Staff move that most candidates miss.

---

## 5. THE HOT PATH: Decode & Preprocess Pipeline

This is the section that distinguishes a Staff-level video data answer from a generic "I drew some boxes" answer.

### The fundamental question: where do raw frames come from?

```
   compressed mp4  ──decode──▶  raw frames  ──VAE encode──▶  latents
       15 MB                     147 MB                       2 MB
       (S3)                      (transient)                  (cacheable)
```

You have three options for **where decode happens**:

#### Option A: CPU decode at trainer node (PyAV/FFmpeg)
```
  CPU decode rate:  ~5-10 videos/s/core for 256² @ 30fps
  Per node need:    8 GPU × 5 vid/s = 40 vid/s
  Cores required:   ~5-8 cores just for decode (plus VAE on GPU)
  Verdict:          Workable but stresses CPU; bad for higher resolutions
```

#### Option B: GPU NVDEC at trainer node
```
  H100 NVDEC:       7 engines × ~50 streams = ~350 videos/s/GPU theoretical
  Realistic:        ~30-50 videos/s/GPU under load
  Per GPU need:     5 videos/s — fits with massive headroom
  Cost:             Steals GPU compute slot from training; needs careful overlap
  Verdict:          Best for online decode of fresh data
```

#### Option C: Pre-decode offline → store latents (RECOMMENDED for steady-state training)
```
  Offline pipeline:  separate decode/encode farm runs once per corpus version
  Storage:           500M × 2 MB latents = 1 PB
  Online cost:       just read latents, skip decode entirely
  Throughput online: 4000 × 5 × 2 MB = 40 GB/s — 7.5× less than compressed video
  Trade-off:         locked into one preprocessing config; experiments need re-encode
  Verdict:           Best for production training; B used for fast iteration
```

> `[STAFF+]` Naming **"latent caching" (Option C) as the production answer with B as the dev-iteration answer** is a Staff move. It demonstrates you've thought past the throughput math into what people actually do at frontier labs (Sora, Veo, MovieGen all do some form of this).

### Pipeline diagram (Option C steady state, with B fallback)

```
┌──────────────────────────────────────────────────────────────────────┐
│                       Per-Node Data Pipeline                          │
│                                                                       │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐         │
│  │  Cache   │    │  Frame   │    │  Augment │    │  Pack    │         │
│  │  client  │───▶│  sampler │───▶│  (crop,  │───▶│  into    │         │
│  │  (RDMA)  │    │  (T,H,W) │    │   flip)  │    │  batch   │         │
│  └──────────┘    └──────────┘    └──────────┘    └──────────┘         │
│       │                                                  │            │
│       │  latent tensor [T,C,H,W]                         │            │
│       │  + caption tokens                                │            │
│       ▼                                                  ▼            │
│  ┌──────────────┐                              ┌──────────────────┐   │
│  │  Pinned host │                              │  GPU HBM         │   │
│  │  memory ring │ ── async H2D copy ─────────▶ │  (training step) │   │
│  │  buffer      │                              │                  │   │
│  └──────────────┘                              └──────────────────┘   │
│                                                                       │
│  Sidecar runs N=2 batches ahead → hides cache fetch + H2D latency     │
└──────────────────────────────────────────────────────────────────────┘
```

### The sidecar prefetch math (why "10 seconds ahead" — justify the constant)
```
Per-GPU rate:           5 batches/s → 200 ms/batch
Cache fetch (RDMA):     ~50 ms for 2 MB latent
H2D copy:               ~5 ms for 2 MB
Total fetch budget:     55 ms vs 200 ms step → 27% — too high for stall-free
Solution:               prefetch N=2-4 batches; ring buffer of pinned memory
  Buffer size:          4 batches × 32 MB/batch = 128 MB host RAM/GPU = 1 GB/node
  Effective fetch:      hidden behind training step
```

10 seconds was hand-wavy. The right answer is **N batches**, sized so prefetch latency < step time, with N≥2 to absorb tail latency.

> `[STAFF+]` Showing the **stall-free condition** (`fetch_latency < step_time × prefetch_depth`) explicitly — and computing the buffer size — is Staff. Saying "prefetch 10s ahead" without justification is Senior.

---

## 6. Cache / Shuffle Layer (sharding mechanism, not just "6 nodes")

### Sharding scheme
```
  video_id ──▶ hash(video_id, recipe.shard_seed) % num_shards ──▶ shard_id

  num_shards = 12 primary
  replication factor = 2 → 24 physical nodes
  Per node: 400 Gb/s NIC, 30 TB NVMe
```

### Why deterministic + recipe-aware?
- Every DP rank, given (recipe_id, global_step), computes which shard to read from with **zero coordination**.
- No metadata service in the read hot path.
- Cache miss → fall through to S3 packed shard, populate cache, continue.

### Topology
```
                 ┌─────────────────────────────────────────┐
                 │  S3 (cold, 7.5 PB)                      │
                 └──────────────┬──────────────────────────┘
                                │  miss path (rare, after warmup)
            ┌───────────────────┼───────────────────┐
            ▼                   ▼                   ▼
     ┌────────────┐      ┌────────────┐      ┌────────────┐
     │ Cache S0   │      │ Cache S1   │ ...  │ Cache S11  │
     │ (replicas  │      │            │      │            │
     │  S0a, S0b) │      │            │      │            │
     │ 30 TB NVMe │      │            │      │            │
     │ 50 GB/s    │      │            │      │            │
     └─────┬──────┘      └─────┬──────┘      └─────┬──────┘
           │                   │                   │
           └────────RDMA / NCCL or custom──────────┘
                                │
        ┌───────────────────────┼───────────────────────┐
        ▼                       ▼                       ▼
   ┌─────────┐             ┌─────────┐             ┌─────────┐
   │ Node 0  │             │ Node 1  │  ...        │ Node 499│
   │ 8× GPU  │             │ 8× GPU  │             │ 8× GPU  │
   └─────────┘             └─────────┘             └─────────┘
```

### Cache warmup & restart
```
Cold start:    Each shard pulls from S3 at ~50 GB/s
               180 TB working set / 12 shards = 15 TB per shard
               15 TB / 50 GB/s = 5 minutes to warm
               → first 5 min of job is cache-fill bound
               → solution: lazy fill + S3 bypass for first epoch
               → or: pre-warm before job starts as part of scheduler
```

### Cache eviction
```
Policy:        Future-aware — evict samples NOT in the next epoch's prefix
               (we know the recipe's deterministic order, so we know what's next)
               LRU as fallback
Anti-pattern:  Pure LRU is wrong here — random shuffle defeats LRU
               every epoch
```

> `[STAFF+]` **Future-aware eviction using the deterministic recipe order** is a Staff insight — you're using the property that your access pattern is *predicted by the recipe*, not just observed. Most candidates default to LRU.

### Failure handling
```
Shard node dies mid-job:
  1. Replicas (factor 2) keep serving — no immediate stall
  2. Coordinator detects, spins replacement node
  3. Replacement back-fills from peer replica + S3 in parallel
  4. Catch-up time: 15 TB / 50 GB/s ≈ 5 min, single replica during this window

Cascade-prevention:
  - Per-shard rate limit on S3 fallback (don't melt S3 if cache fails wide)
  - Circuit breaker: if S3 latency > threshold, slow training (don't crash)
```

---

## 7. End-to-End Read Path Walk-Through

```
TIMELINE FOR ONE BATCH ON ONE DP RANK
═════════════════════════════════════

t=0    Training step S completes
       │
       │ Sidecar already started fetching batch S+2 at t=-200ms
       │
t=0    Trainer pulls batch S+1 from pinned ring buffer  (already there)
       │
       │ H2D copy: 5ms (overlapped with kernel launch)
       │
t=5    GPU starts forward pass on batch S+1
       │
       ▼  (meanwhile, sidecar fetches batch S+3)

SIDECAR FETCH OF BATCH S+N
═══════════════════════════

  1. Compute video_ids for batch S+N from (recipe_id, S+N, rank)
       └─ deterministic, no RPC
  2. Compute shard_ids = hash(video_id, recipe.seed) % 12
  3. RDMA read from cache shards
       ├─ HIT  → 50ms → done
       └─ MISS → fall through:
             a. Cache fetches packed-shard from S3 (~500ms)
             b. Cache extracts target latent
             c. Cache returns to client
             d. Client proceeds; cache retains for future
  4. Caption tokens already inline in Parquet → fetched once, cached locally
  5. Apply augmentation, pack into batch tensor, place in ring buffer
  6. Notify trainer: batch S+N ready
```

---

## 8. Reproducibility & Checkpointing

The recipe holder's **read cursor** is the data-side checkpoint. On trainer checkpoint:
```
  trainer_ckpt = {
    model_state, optimizer_state, scheduler_state,
    recipe_id, global_step, data_cursor_token
  }
```

On restart:
```
  1. Load model checkpoint
  2. POST recipe_holder /restore with data_cursor_token
  3. Recompute exact shuffle order from (recipe.seed, global_step)
  4. Resume — bit-exact data sequence guaranteed
```

**Why deterministic shuffle matters:** without it, restart from checkpoint at step 100K is *not the same training run*. For research reproducibility, ablations, and debugging numerical issues, this is non-negotiable.

> `[STAFF+]` Treating the **data cursor as a first-class checkpoint artifact** alongside model state is a Staff move. Senior candidates checkpoint the model and assume the dataloader will "do the right thing."

---

## 9. Multi-Tenancy (concurrent jobs)

If N concurrent training runs share infra (the realistic frontier-lab case):

### Naive approach (what I'd flag and reject)
- N separate cache fleets, each warmed independently
- Cost: N × 24 nodes × $X/month, mostly redundant data

### Better approach
```
Content-addressed shared cache:
  key = hash(video_id, preprocessing_config)
  → two jobs using identical preprocessing share cache entries
  → quality score recompute changes config_hash → cache miss → re-derive

Per-tenant quotas:
  - Bandwidth quota per job (fair-share scheduler)
  - Storage quota per job for unique latents
  - Priority tiers (research vs production)
```

> `[STAFF+]` Surfacing multi-tenancy *unprompted* — and proposing **content-addressed sharing keyed by (video_id, preprocessing_config)** — is a Staff move because it shows you understand that frontier labs run dozens of concurrent jobs and naive isolation is wasteful.

---

## 10. Cost Awareness (the conversation almost no one has)

```
S3 storage:       7.5 PB × $0.023/GB-month  = $172K/month corpus
                  Intelligent tiering → ~$80K/month for cold portion

S3 GET requests:  packed shards → ~500K GETs/epoch × $0.0004/1K
                                = $0.20/epoch (negligible — that's the WIN
                                  from packing; unpacked would be $200/epoch)

Cross-AZ egress:  300 GB/s × $0.01/GB × 3600s = $10.8K/hour
                  → CO-LOCATE cache with trainers, single AZ
                  → eat the AZ-failure risk; checkpoint to S3 cross-AZ

Cache nodes:      24 × i3en.metal-class instances
                  ~$3-5/hr × 24 × 730 = $50-90K/month

GPU stall cost:   4000 H100s × $4/hr = $16K/hr
                  1% data stall      = $160/hr = $115K/month wasted
                  → data team's KPI is GPU utilization,
                    measured in dollars not percent
```

> `[STAFF+]` Translating data-stall percentage into **GPU-hour dollars** is a Staff framing. It reframes the data team's mission from "build infra" to "minimize $115K/month of GPU stall," which changes prioritization.

---

## 11. Failure Modes Summary Table

| Failure | Detection | Mitigation | Recovery time |
|---|---|---|---|
| Cache shard node crash | health check + replica heartbeat | replication factor 2 | minutes (replacement back-fills) |
| S3 throttling / regional issue | request error rate alarm | exponential backoff, circuit breaker, fall back to local NVMe slow-path | seconds–minutes |
| Trainer rank crash | NCCL error | checkpoint restart with data cursor | minutes (deterministic resume) |
| Straggler (slow rank) | step time histogram per rank | per-rank prefetch depth bumped; long-term reschedule | continuous |
| Data corruption | checksum on read | reject sample, log to dead-letter, re-fetch from S3 | per-sample |
| Recipe holder failure | API health check | HA Postgres + stateless API replicas | seconds (failover) |
| Cache fleet collapse (cascade) | aggregate hit-rate monitor | rate-limit S3 fallback, reduce trainer rate via backpressure | hours (rebuild) |
| Hot key (one shard saturated) | per-shard NIC utilization | re-hash with epoch salt; consistent hash with bounded loads | next epoch |

> `[STAFF+]` Surfacing **cascade prevention** (cache failure → S3 stampede → S3 throttling → all-jobs-down) is a Staff move. Senior candidates list failures; Staff candidates name the *interactions between* failures.

---

## 12. Trade-offs I'd Name Out Loud

A Staff candidate doesn't just pick — they say "I'm choosing X over Y because Z, accepting trade-off W." Sample explicit trade-offs for this design:

1. **Pre-decoded latents (1 PB extra storage) vs online decode.** Choosing latents for steady-state training; accept loss of preprocessing flexibility (any aug change requires re-encode). Mitigated by keeping NVDEC online-decode path available for experiments.

2. **Iceberg + Trino vs Postgres for catalog.** Choosing Iceberg for analytical filters at 500M scale; accept higher operational complexity and Trino cluster cost. Postgres retained for recipe definitions (small, transactional).

3. **12 cache shards with replication 2 vs 24 unreplicated.** Choosing replicated; pays 2× storage but eliminates single-shard failure stall. Sized for 4000-GPU job + headroom.

4. **Single-AZ cache vs multi-AZ.** Choosing single-AZ for $10K/hr cross-AZ savings; accept AZ-failure risk for data plane (checkpoint to multi-AZ S3 mitigates total job loss).

5. **Future-aware eviction vs LRU.** Choosing future-aware (uses recipe order); accept tighter coupling between recipe holder and cache (recipe must be queryable from cache for eviction decisions).

6. **Packed shards vs individual objects.** Packed; accept loss of random sample-level GET, gain 100× on S3 request economics. Random access within shard handled by index file.

7. **Content-addressed shared cache vs per-job cache.** Shared; accept noisy-neighbor risk, gain massive cost savings under concurrent-job load. Mitigated with per-tenant bandwidth quota.

---

## 13. What I'd Skip Without Apology in 40 Min

If clock is short, drop in this order:
1. Cost section — interviewer will probe if they want it
2. Multi-tenancy — flag as future work
3. Some failure modes — keep the top 3
4. Ingestion pipeline — assume corpus exists and is curated upstream

**Never skip:** the decode/preprocess discussion, the cache sharding *mechanism*, the reproducibility story, the capacity math.

---

## 14. The Staff+ Signal Cheat Sheet

When you hear yourself about to say something, check whether it's the Senior or Staff version:

| Senior says | Staff says |
|---|---|
| "Store videos in S3" | "Pack 1000 videos per tar shard; S3 request rate is the bottleneck, not bandwidth" |
| "Add a cache" | "12 shards, deterministic hash from recipe seed, replication 2, future-aware eviction" |
| "Postgres for metadata" | "Iceberg + Trino for analytical queries at 500M; Postgres for recipe definitions only" |
| "Decode the videos" | "Pre-encode to VAE latents offline (1 PB), NVDEC online-decode path for experiments, CPU decode rejected on bandwidth grounds" |
| "Sidecar prefetches data" | "Prefetch depth N such that fetch_latency × N > step_time; ring buffer of pinned host memory; 4 batches × 32 MB = 128 MB/GPU" |
| "Save model checkpoints" | "Checkpoint includes data cursor token; restart restores bit-exact data order via deterministic shuffle from recipe seed" |
| "Use 4000 GPUs" | "DP=500 distinct readers (TP=8 dedup); aggregate consumption is 37.5 GB/s compressed, 3 TB/s decoded" |
| "Handle failures" | "Cascade prevention: rate-limit S3 fallback to avoid stampede when cache fleet degrades" |
| "Multi-tenant later" | "Content-addressed cache keyed by (video_id, preprocess_config_hash); per-tenant bandwidth quota" |
| Implicit numbers | "I'm assuming X because Y; if Z is different that changes A" |

---

## 15. Reusable Templates for Adjacent Questions

The same skeleton with different specializations answers most ML-systems design rounds:

| Question | Specialize: storage | Specialize: hot path | Specialize: reproducibility |
|---|---|---|---|
| **T2V data platform** | packed mp4 shards + latent cache | NVDEC + VAE encode | recipe seed + data cursor |
| **LLM pretraining data** | tokenized .bin shards (GPT-style) | tokenizer parallelism, document packing | seed + step counter |
| **Image generation data** | packed jpg/webp + CLIP embeddings | CPU decode + augment | seed + cursor |
| **RLHF preference data** | preference pairs in Parquet | join + reward-model scoring | seed + sample IDs |
| **Inference KV-cache management** | per-request KV in HBM | paged attention, prefix caching | not applicable (stateless) |
| **Distributed checkpointing** | sharded model state in S3 | async upload, dedup across DP | step counter + topology |

The deployable pattern: **storage → catalog → recipe → cache → preprocess → consumer**, plus **cross-cutting concerns**. Specialize each box for the domain in 25 min.

---

## 16. Time Budget for the Real Interview

```
0:00 ────────────────────────────────────────── start
0:00–0:02   Draw the skeleton (§0). DON'T skip this.
0:02–0:05   Clarifying questions (§1). Box at 5 min.
0:05–0:10   State assumptions, do capacity math (§2)
0:10–0:15   Storage tier (§3) — fast, well-understood
0:15–0:25   Decode/preprocess deep-dive (§5) — the hot path
0:25–0:30   Cache sharding mechanism (§6)
0:30–0:33   Reproducibility (§8)
0:33–0:38   Trade-offs (§12) — make these explicit
0:38–0:40   Failures (§11) + close
0:40 ────────────────────────────────────────── end
                                Buffer: 0
```

If interviewer probes a specific box, **answer in ≤2 sentences and steer back**:
> "Good question — short answer: [X]. I want to make sure we cover [Y] before we run out of time, can I park that and come back?"

This is a learnable, drillable skill. It is not the same as technical depth. It is the skill that the PI screen and the Rhoda loop both flagged.

---

## Appendix: Numbers Worth Memorizing

```
Storage hierarchy bandwidth (rough):
  S3 (per-bucket aggregate):     ~100 GB/s sustained, regional limits
  S3 (per-prefix):                ~5 GB/s before throttling
  NVMe local:                     ~5-7 GB/s read
  RDMA (200/400 Gb/s NIC):        25-50 GB/s
  HBM (H100):                     3 TB/s
  PCIe Gen5 x16:                  64 GB/s

Video math:
  H.264 typical:                  1-10 Mbps  (1080p ~5 Mbps)
  Compressed-to-raw expansion:    8-40× depending on resolution/fps
  256² × 30fps × 8-bit RGB raw:   ~5.6 MB/s per stream
  H100 NVDEC:                     ~30-50 concurrent 256² streams realistic

S3 request economics:
  GET cost:                        $0.0004 / 1K requests
  Latency:                         ~10-30 ms first byte
  Pack ratio rule:                 if avg object < 1 MB, pack

Training stall economics:
  H100 on-demand:                 ~$4/hr/GPU
  4000 GPU job:                    $16K/hr
  1% stall:                        $160/hr ≈ $1.4M/year
```