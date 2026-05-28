---
title: Data ingest system
description: Data ingest system
---

**Design data ingestion, sharding, and preprocessing for massive datasets used to train foundation models.**

---

# Q2 — Data Ingestion, Sharding, Preprocessing for FM Training

## 1. Framing & Assumptions

Before I architect anything, let me pin down what "massive" means here so we're not solving the wrong problem.

I'm assuming: 5–15 PB of raw multimodal data (heavy text bias — roughly 60% text, 25% image, 10% audio, 5% video), processed down to 1–3 PB of training-ready shards representing on the order of 30–50T tokens after dedup and filtering. Platform serves 3–5 concurrent pretraining runs, each consuming 2–8K accelerators (mix of H100/H200, TPU v5p, and Apple Silicon clusters), plus a constant tail of fine-tuning and eval workloads. Pretraining is ~90% of the GPU-hours we're feeding.

Multi-tenant by mandate. This isn't a one-team pipeline — it has to serve LLM pretraining, multimodal teams, and the on-device team whose data quality bar is the highest. That single decision changes everything downstream: I'm building a platform, not a script.

The consumer's hard requirement is MFU (Model FLOPs Utilization). If accelerators are stalled on input, nothing else matters. I'm designing backwards from "every rank gets a batch every step at 90%+ MFU, deterministically resumable, with no surprises in year one."

A few non-obvious assumptions I'm making explicit:

- Data is mostly licensed/internal/synthetic, not crawl. The web-crawl fraction is small and quarantined. Apple's posture pushes us toward provenance-tracked sources.
- We pre-tokenize ~95% of text data. Tokenization-in-the-loop is reserved for late-bound experiments where the tokenizer is changing.
- We have a working object store as source of truth. I'm not designing that; I'm designing what sits on top.
- Determinism is non-negotiable. Every training run must be bit-reproducible at the dataloader level, even after a world-size change mid-run.
- Cost matters. At this scale, S3 egress, IOPS, and CPU preprocessing genuinely move the budget — a sloppy preprocessing pass can cost mid-six-figures more than a careful one, and "we'll just rerun it" stops being a free option.

[STAFF SIGNAL: explicit assumption-stating with numbers before drawing any boxes — keeps the rest of the answer falsifiable.]

OK, with that pinned, here's the architecture.

## 2. Reference Architecture

```
┌────────────────────────────────────────────────────────────────────┐
│                      SOURCES (heterogeneous)                       │
│  licensed corpora │ internal data │ synthetic │ web (small/quar.)  │
└──────────┬──────────────────┬──────────────┬─────────────┬─────────┘
           ▼                  ▼              ▼             ▼
┌────────────────────────────────────────────────────────────────────┐
│  LANDING ZONE (S3 Standard, content-addressed, immutable)          │
│  • SHA256 keys, per-batch manifest, provenance metadata            │
└────────────────────────────────┬───────────────────────────────────┘
                                 ▼
┌────────────────────────────────────────────────────────────────────┐
│  PREPROCESSING (Ray Data + Spark, idempotent stage DAG)            │
│  ┌──────────┐ ┌──────┐ ┌───────┐ ┌─────────┐ ┌────────┐ ┌──────┐   │
│  │ validate │→│filter│→│ dedup │→│normalize│→│tokenize│→│ pack │   │
│  └──────────┘ └──────┘ └───────┘ └─────────┘ └────────┘ └──────┘   │
│  Each stage: content-hash inputs, materialized outputs in Lance    │
└────────────────────────────────┬───────────────────────────────────┘
                                 ▼
┌────────────────────────────────────────────────────────────────────┐
│  SHARD WRITER → CURATED ZONE                                       │
│  • WebDataset .tar shards (~512MB) for streaming training          │
│  • Lance tables in parallel for analytical queries / eval slicing  │
│  • Iceberg catalog tracks every dataset version                    │
└──────────┬─────────────────────────────────────────────────────────┘
           ▼
┌────────────────────────────────────────────────────────────────────┐
│  HOT TIER CACHE (WekaFS / FSx Lustre, ~100TB working set)          │
│  • Pre-warmed per scheduled training run                           │
│  • Shared across overlapping runs                                  │
└──────────┬─────────────────────────────────────────────────────────┘
           ▼
┌────────────────────────────────────────────────────────────────────┐
│  TRAINING CLUSTER (per-rank streaming dataloaders)                 │
│  shard list → deterministic permutation → local prefetch → GPU     │
└────────────────────────────────────────────────────────────────────┘
```

Walking the flow: raw data lands in an immutable, content-addressed S3 bucket with a manifest per arrival batch that captures source, license, ingestion timestamp, and a SHA256 over the bytes. Nothing in the pipeline ever mutates a landed object. Reprocessing means rerunning downstream stages, not editing inputs.

Preprocessing is a DAG of idempotent stages orchestrated by Ray Data, with Spark used for the few stages where its mature shuffle and join story still wins (whole-corpus dedup, large joins against blocklists). Each stage takes a content hash over its inputs plus its code version; if both match a prior run, it short-circuits and reuses the materialized output. This is the single biggest cost-saver at PB scale.

[STAFF SIGNAL: stage-level idempotency keyed on content+code hash means a one-line tokenizer fix doesn't reprocess 5 PB.]

Curated outputs land in a two-format layout: WebDataset `.tar` shards for streaming training, and Lance tables for analytical queries — mixture sampling, eval-set construction, dataset-card statistics. Same data, two views, one source of truth tracked in an Iceberg catalog so every shard is addressable by `(dataset_name, version, shard_id)`.

A hot-tier cache sits between the curated zone and the training cluster. Pre-warming this cache before a training run launches is treated as a first-class scheduler input — if the cache is cold, we pay an upfront 20–60 minute warm-up and surface that to the run owner. I'll come back to this in §10.

[STAFF SIGNAL: cache warmth as scheduling input rather than a side effect — difference between a script and a platform.]

## 3. Storage & Shard Format

The format choice is the most consequential decision in the stack, so I'll defend it explicitly.

**Shards: WebDataset (`.tar` of grouped samples).** Picked over MosaicML MDS, raw Parquet, and per-sample files.

- Per-sample files in S3: dead on arrival. At 50–500KB per sample, list+get latency dominates, you blow IOPS budgets, and small-object pricing is brutal. Not viable.
- Parquet as training shards: great for analytics, awkward for streaming. Row-group reads are fine for fixed-size text, but multimodal samples (variable image bytes, embedded audio) fight the columnar layout. Better as a query format than a serving format.
- MosaicML MDS: real contender. Strong determinism story, integer-indexed random access, well-engineered. The reason I default to WebDataset is the tarball is opaque-by-design — image bytes, audio bytes, JSON metadata all sit next to each other in one record with no schema enforcement. Multimodal pipelines like this. MDS works better when sample shape is regular.
- Lance for the analytical view alongside, because zero-copy random access and versioned writes solve the eval-slicing problem cleanly.

[STAFF SIGNAL: picking WebDataset with MDS as the close runner-up, discriminator being multimodal sample irregularity — not "I like WebDataset."]

**Shard size: 256MB to 1GB, target 512MB.** The math: training reader pulls 2–3 GB/s sustained per node from object store. A 512MB shard fetches in ~200ms, large enough to amortize TCP/TLS setup, small enough to hold thousands per dataset for shuffle entropy, and aligned with S3 multipart download chunking. Below 100MB you fragment shuffle; above 2GB a corrupt shard becomes a fat tail in retry latency.

**Storage tiering:**

```
┌─────────────────────────────────────────────────────────────────┐
│ HOT   │ WekaFS / FSx Lustre │ ~100 TB │ active run working set, │
│       │                     │         │ prewarmed               │
├─────────────────────────────────────────────────────────────────┤
│ WARM  │ S3 Express One Zone │ ~500 TB │ current quarter's       │
│       │                     │         │ curated shards          │
├─────────────────────────────────────────────────────────────────┤
│ COLD  │ S3 Standard         │ ~5 PB   │ historical raw +        │
│       │                     │         │ deprecated shards       │
├─────────────────────────────────────────────────────────────────┤
│ FROZEN│ S3 Glacier IR / DA  │ ~10 PB+ │ provenance archive,     │
│       │                     │         │ never deleted           │
└─────────────────────────────────────────────────────────────────┘
```

Frozen exists because raw provenance gets archived but never deleted — that's a legal and reproducibility requirement, not a storage optimization.

**Metadata catalog: Apache Iceberg.** Every curated dataset version is an Iceberg table with snapshot IDs. Training runs reference `(dataset_name, snapshot_id)`. We get time-travel queries for free, which solves "what data was the May 5 run trained on?" without extra bookkeeping. Iceberg also gives schema evolution semantics, so adding a new metadata column (say, a new quality score) doesn't break existing readers — relevant because we will absolutely add columns over time.

## 4. Ingestion & Preprocessing Pipeline

```
RAW ARRIVALS
    │
    ▼
┌─────────────┐
│ 1. VALIDATE │  schema, encoding, file integrity, license tag present
└──────┬──────┘
       ▼
┌──────────────┐
│ 2. QUARANTINE│ malware scan, PII pre-flag, source-trust score
└──────┬───────┘
       ▼
┌──────────────┐
│ 3. CAS WRITE │ content-addressed store, exact dedup at ingest
└──────┬───────┘
       ▼
┌─────────────┐
│ 4. FILTER   │ quality classifier, language ID, lexical heuristics,
│             │ image NSFW, audio SNR — emits keep/drop + reason
└──────┬──────┘
       ▼
┌─────────────┐
│ 5. DEDUP    │ MinHash LSH (text), pHash (images), embedding-based
│             │ near-dup for high-value subsets only
└──────┬──────┘
       ▼
┌──────────────┐
│ 6. NORMALIZE │ unicode NFC, whitespace, image resize policies,
│              │ audio resample 16kHz, video keyframe extraction
└──────┬───────┘
       ▼
┌──────────────┐
│ 7. TOKENIZE  │ BPE/SentencePiece, multimodal special tokens,
│              │ document boundaries preserved
└──────┬───────┘
       ▼
┌─────────────┐
│ 8. PACK     │ BFD packing into context windows, attention masks
└──────┬──────┘
       ▼
┌─────────────┐
│ 9. SHARD    │ deterministic shard assignment, WebDataset writer,
│             │ Lance writer in parallel
└─────────────┘
```

A few things worth highlighting rather than walking every stage.

**Idempotency at every stage.** Each stage is a pure function of `(input_content_hash, code_version, config_hash) → output`. Cached. A change to the filter classifier reprocesses from filter onward; tokenizer change reprocesses from tokenize onward; everything upstream is reused. This pays for itself the first time someone fixes a unicode bug and we save 80% of a reprocess. It also makes "rerun the pipeline" a safe phrase in standups — without it, rerun means "burn $200K of CPU time and hope nothing breaks."

**Synthetic data flow.** A specific note because synthetic data is increasingly load-bearing for FM training: synthetic generation runs as its own DAG that produces samples landing in the same CAS-addressed format as natural data, with provenance metadata identifying the generator model and prompt template. Downstream stages don't care; they see another data source. This is the right factoring because it lets us mix synthetic and natural at the shard-list level without special-casing.

**Compute placement per stage.** Validate/quarantine/CAS are I/O-bound — CPU farm on spot. Filter and dedup are CPU-heavy with embarrassingly parallel batch shape — Ray on spot CPU instances. Image/video decode for normalize is the one place I push to GPU (NVIDIA DALI with nvJPEG, or Intel IPP on CPU for non-GPU clusters) because JPEG decode is genuinely the bottleneck at scale; we lose roughly 30% of throughput on text-image pipelines if we leave decode on CPU. Tokenize is CPU on big-mem boxes; pack and shard are I/O-bound writers.

**Tokenization-in-the-loop vs pre-tokenized: pre-tokenized by default.** Pre-tokenization saves 15–25% of dataloader CPU work per step at the cost of dataset rigidity — a tokenizer change invalidates shards. We accept that cost because (a) tokenizers change rarely past initial design, (b) we keep the un-tokenized normalized text addressable via Lance so a retokenize is fast, and (c) at 50T tokens the per-step savings are real. Exception: rapid-iteration fine-tunes where the tokenizer is part of the experiment — those stream from the normalized Lance tables and tokenize in the dataloader.

[STAFF SIGNAL: pre-tokenize-by-default with explicit escape hatch for fine-tune iteration — names the cost paid and the recovery path.]

**Provenance threading.** Every sample at every stage carries a provenance pointer back to its CAS-addressed raw bytes. Non-negotiable in an Apple context — if a licensor revokes data, we must identify and excise every downstream artifact. Cost: ~50 bytes/sample of metadata. Worth it.

**Quarantine flow specifically.** Web-crawled or unverified data lands in quarantine until it passes (a) malware scan, (b) PII pre-flag, (c) license/provenance attestation. Quarantine is a separate bucket with restrictive IAM and an audit log. Things stay there until human-approved or auto-cleared by policy. Looks like overhead until the day legal needs it, and then it's the only thing that matters.

[STAFF SIGNAL: quarantine-as-default for untrusted ingest with explicit human-or-policy gate — anticipates a compliance failure mode rather than building it in after the audit.]

## 5. Sharding & Shuffling for Data Parallelism

This is where most pipelines quietly do the wrong thing. The interesting question isn't "how do we split files?" — it's "how do we map shards to ranks deterministically, support world-size changes mid-run, and shuffle well enough for SGD without destroying I/O patterns?"

**Logical shards, physical files.** A dataset has N logical shards, each backed by one physical `.tar` file. N is chosen so N >> world_size_max — typically 10K–100K logical shards per dataset. World size is bounded by N.

**Rank-to-shard mapping:**

```
Dataset:    [shard_0, shard_1, shard_2, ..., shard_N-1]
                │
                ▼  deterministic permutation σ(seed, epoch)
            [shard_π(0), shard_π(1), ..., shard_π(N-1)]
                │
                ▼  partition into world_size contiguous slices
   ┌────────────┬────────────┬────────────┬────────────┐
   │  rank 0    │  rank 1    │  rank 2    │  rank W-1  │
   │  N/W shards│  N/W shards│  N/W shards│  N/W shards│
   └────────────┴────────────┴────────────┴────────────┘
                │
                ▼  within each rank's slice, shuffle samples with
                   a per-shard buffer (8K–64K samples)
              [sample stream → packed batches → GPU]
```

**Why this and not alternatives.** Hash-mod assignment (shard_id % world_size → rank) is the obvious thing and is wrong: changing world size from 1024 to 2048 means every rank gets different data, you lose mid-run resumability. Random assignment loses determinism. Permutation-then-partition keeps shard ordering deterministic given a seed; a world-size change only requires repartitioning the (still-deterministic) permuted shard list, which each rank computes locally.

**Shuffling: hierarchical.** Global shuffle by permuting the shard list. Local shuffle within each rank via a sample buffer of 8K–64K samples that emits randomly. Global-only leaves intra-shard correlation; full-global sample shuffle would require either pre-shuffling samples across shards (defeats streaming I/O) or a giant shuffle buffer (eats host RAM). The hierarchical compromise costs ~0.1 perplexity vs full-random in our experience and is the right tradeoff.

[STAFF SIGNAL: naming the actual quality cost of approximate shuffling (~0.1 PPL) instead of pretending it's free — calibrated tradeoff.]

**Surviving world-size changes mid-run.** Checkpoint stores `(epoch, global_sample_step, seed)`. On restart, regardless of new world size, each rank computes the global permutation, finds the sample step, resumes. With pre-tokenized fixed-size sequences, exact. With packed sequences of variable original-document count, we checkpoint at packed-sequence boundaries — small drift, no replay.

[STAFF SIGNAL: explicit answer to mid-run reshard — most candidates skip this and it's a real production scar.]

**Multi-epoch handling.** Re-permute with seed `(base_seed, epoch_id)`. Each epoch sees a different shard order without ever needing to physically reshuffle on disk.

## 6. Sample Packing & Batch Construction

For LLM pretraining the packing problem is: documents range 100–50,000 tokens; context window is 8K, 32K, or 128K; padding wastes FLOPs. Naïve concat-and-chunk wastes nothing on padding but creates cross-document attention contamination — the model attends across unrelated documents, which has been shown to hurt long-context behavior measurably.

**Default: BFD (best-fit-decreasing) packing with document-aware attention masks.** Sort documents by length, greedily pack into context-window-sized bins, emit a block-diagonal attention mask preventing cross-document attention. Packing efficiency lands at 96–99% of context window utilization. The mask is the cheap part — FlashAttention variants support block-diagonal masks natively now.

**Alternative considered: pure concat-and-chunk.** Picked against because the cross-document leakage cost outweighs the ~2% packing efficiency gain at long context.

[STAFF SIGNAL: defending document-aware packing on quality grounds, not just hygiene — names a concrete model-quality effect rather than vague "cleaner is better."]

**Multimodal batching is a different beast.** For image-text pairs at varying resolutions, bucket by aspect ratio (~8 buckets), then by sequence length within bucket. Per-batch draw from one bucket so all images have compatible shape after a small dynamic resize. Mixing aspect ratios within a batch forces either letterboxing (wastes pixels and compute) or padding tokens (wastes compute and risks attention bias) — bucketing avoids both. For video, dominant cost is decode, so we either pre-decode keyframes to a fixed schema or use NVIDIA DALI on the GPU side for hardware-accelerated decode. Pure CPU video decode at PB scale is a non-starter — measured roughly 8x throughput improvement moving to GPU decode for a video-heavy mix. Audio batching is the easy case: pad to longest in batch with negligible overhead since sequence variance is small after resampling.

**Curriculum and mixture weights.** Mixture sampling (e.g., 60% English, 20% code, 10% multilingual, 10% multimodal) happens at the shard-list level, not the sample level. Each rank's shard list is composed of `(dataset_id, shard_id)` tuples drawn at the correct ratio. This avoids the "rank-local mixture drift" pathology where small ranks see lopsided mixtures.

## 7. Throughput Engineering: Keeping GPUs Fed

The whole pipeline lives or dies here. Bandwidth math first, design second.

```
Per H100 device target:    ~3,000 tok/s × 2 bytes (BF16 tokens)
                           ≈ 6 KB/s of TOKEN bandwidth — trivial.
But raw sample bytes:      ~50–500 KB/sample, 100–500 samples/s
                           ≈ 50–250 MB/s per accelerator
Per 8-GPU node:            ~400 MB/s – 2 GB/s
Object store → node:       2–3 GB/s sustained, achievable on S3
                           with parallel multipart GETs and a
                           dedicated network path
GPU memory PCIe budget:    32 GB/s PCIe5 × 16 lanes — never the
                           bottleneck for data loading
```

Bandwidth budget closes. The hard part is closing it consistently, not in expectation.

**Pipeline on each training node:**

```
S3 ──┬─► [shard fetcher pool, N=4 threads] ──► local shard ring buffer
     │                                              │
     │    [decode workers, M=16 procs, NUMA-pinned] │
     │             ▲                                ▼
     │             └─── samples ──► [sample queue, 2-3 batches]
     │                                              │
     │                                              ▼
     │                                  [pinned-memory batch builder]
     │                                              │
     │                                              ▼  async H2D
     └─────────────────────────────────► [GPU input queue, depth 2]
                                                    │
                                                    ▼
                                             [forward / backward]
```

**Key choices:**

- Shard fetcher decoupled from sample decode. Network stalls don't propagate immediately to GPU.
- Decode workers as separate processes (not threads) to escape Python GIL for CPU-heavy decode. NUMA-pinned to the socket attached to the GPU they feed.
- Pinned (page-locked) host memory for the batch buffer enables async H2D copy overlapping with compute. ~2x effective bandwidth vs unpinned.
- Prefetch depth of 2 batches on the GPU side. More than 2 wastes HBM with no throughput gain past steady state.
- Backpressure: queues are bounded; if the GPU stalls, decode workers stall, and we surface "input pipeline stall" as a first-class metric, not buried in nvidia-smi.

[STAFF SIGNAL: explicit backpressure with named exposed metric — turns a silent latent-killer into something an SRE can alert on.]

**GPU decode for image/video.** DALI handles JPEG, H.264, HEVC on-GPU. On text-only pipelines we skip it. On multimodal mixes we route image-bearing samples through DALI and text-only samples through the standard path. Hybrid pipeline, but the routing is one if-statement and the payoff is large.

**Caching layer interaction.** Hot tier (WekaFS) between S3 and nodes. Pre-warm before run launch pulls the expected first-epoch shard set into WekaFS at multi-TB/s aggregate. Subsequent epochs hit the cache. Cache eviction is LRU with a pinning mechanism for active runs. Hit rate >95% in steady state; the first epoch is the only one that pays cold-tier latency, and even that is hidden by sufficient prefetch depth and the fact that the first epoch's compute is also paying any startup costs the run has.

## 8. Determinism, Resumability, Versioning

Three orthogonal problems, one consistent answer: explicit state, externalized, checkpointed alongside the model.

**Determinism.** RNG state for shuffling lives in the dataloader and is included in checkpoint state. Tokenizer version is pinned by content hash in the dataset version. Numerical determinism of the data path is straightforward — no nondeterministic CUDA kernels in dataloading. The only legitimate source of nondeterminism is GPU decode (DALI), and we accept it because (a) it doesn't affect token sequence, only image decode bit-exactness, and (b) bit-exact image decode across hardware is a fool's errand. Documented explicitly.

[STAFF SIGNAL: naming a specific deliberate non-determinism (image decode) and defending it rather than pretending the whole pipeline is bit-exact.]

**Resumability.** Checkpoint state for the dataloader = `(dataset_version_id, epoch, global_sample_step, shuffle_seed, world_size_at_checkpoint, packing_state)`. On resume, the dataloader rebuilds the deterministic shard permutation under the new world size, computes its rank's slice, fast-forwards to the right sample. No replay, no skip. Fast-forward is O(shards_per_rank) not O(samples_seen) because we skip whole shards before the resume point and only stream from the partial shard.

**Versioning.** Iceberg snapshot ID. A training run records the snapshot it consumed; we can reconstruct the exact bytes any time. Dataset card auto-generated per snapshot includes token counts, mixture stats, dedup metrics, contamination scan results. Versioning extends to the preprocessing code: every shard has a sidecar with the git SHA of the preprocessing pipeline that produced it.

## 9. Quality, Dedup, Privacy

**Exact dedup at ingest** via SHA256 in the CAS step — cheap, catches duplicate uploads. **Near-dup later** via MinHash LSH for text (band-and-row params tuned for ~85% Jaccard threshold), pHash for images, and for the highest-value subsets, embedding-based semantic dedup (sentence-transformer or CLIP, FAISS index). Dedup is staged: cheap-then-expensive, drop early. At PB scale, blanket semantic dedup is infeasible cost-wise; we apply it to high-value curated subsets only.

[STAFF SIGNAL: staging dedup cheap-to-expensive rather than picking one algorithm — saves an order of magnitude in compute at PB scale.]

**Eval-set contamination.** Maintain a registry of held-out eval set hashes (n-gram, exact, and near-dup). Run every preprocessed shard through a contamination filter that drops samples matching held-out content. Non-optional, runs as a mandatory final stage before shard write.

**Quality classifiers.** Small fastText-style classifiers for text quality, language ID, perplexity-based filtering from a small reference model. Image quality via aesthetic and watermark classifiers. Audio via SNR and language-ID. None novel; what matters is they're versioned and their thresholds tracked per dataset version.

**Privacy and PII.** Two layers: (1) automated PII detection (regex + NER) flags samples for redaction or drop; (2) in Apple's context specifically, support for differential privacy at the training-data preparation level — noise added to aggregate statistics, k-anonymity for grouped data, and clear separation of on-device training data (stricter bar) from cloud training data. PCC-relevant data flows are isolated into a separate lineage namespace with stricter access controls. PII handling emits an audit log per drop/redact decision.

[STAFF SIGNAL: differentiating on-device vs cloud training data bars and namespacing them rather than treating data privacy as a single global setting.]

## 10. Multi-Tenancy & Platform Concerns

Multi-tenancy at PB scale fails in two specific ways: noisy-neighbor on the cache layer, and code drift across tokenizer/filter versions. Both are predictable.

**Cache as scheduled resource.** Pre-warm requests are submitted with the training run; the scheduler considers cache warmth alongside GPU availability. Two runs that share datasets are co-scheduled when possible to share cache. Cache quota is per-team with bursting.

**Shared registries.** One tokenizer registry, one filter registry, one packing-config registry — all versioned, all referenceable by hash. Teams don't fork; they reference. A team that needs a custom variant registers a new version, doesn't copy. This prevents the "every team has a slightly different tokenizer" disaster that shows up at month nine.

[STAFF SIGNAL: registry-with-references rather than copy-paste — a year-out scar this prevents.]

**Cost attribution.** Per-team chargeback derived from (a) S3 storage of curated shards, (b) cache occupancy time, (c) preprocessing compute. Surfaced in a dashboard the team leads see. Cost visibility drives behavior more than policy does.

**Isolation.** Quarantine namespaces, on-device-eligible namespaces, and general-cloud namespaces have separate IAM policies. A cloud-pretraining job cannot accidentally reference on-device-only data — enforced at the catalog layer, not by convention.

## 11. Failure Modes & Observability

The shortlist of things that break and how we catch them:

**Corrupt shards.** Checksum every shard at write and on every read. On mismatch, reader retries from a replica region, logs the corruption, emits a metric. Three failed reads quarantines the shard and pages the run owner — we don't silently skip. Sample loss is bounded and reported in run summary.

**Schema drift.** Every shard has a schema fingerprint. Readers reject mismatches loudly. A tokenizer-version-skew bug where dataloader and training code disagree on vocab size produces silently corrupted models for hours; making it a hard fail at startup is worth the friction.

**Slow object-store region.** Multi-region replication for hot datasets; per-request region routing with latency-based fallback. Tail latency monitored at p99 and p99.9; sustained p99 > 500ms on shard fetch triggers an alert.

**Tokenizer/code skew.** Dataset version pins the preprocessing pipeline git SHA. Training launch validates dataloader code is compatible with dataset version's pipeline SHA. Mismatch = refuse to start.

[STAFF SIGNAL: fail-loud-at-startup over fail-silent-mid-run — preserves debugging sanity at the cost of a few extra preflight checks.]

**Observability surface.** The on-call dashboard shows, in priority order: GPU input-pipeline stall % (the headline number), per-stage throughput (samples/s, MB/s), prefetch queue depth, cache hit rate, decode p50/p99, sample-loss rate, packing efficiency, shard checksum failures, region tail latency. If GPU input stall goes above 5% sustained, that's the page. Everything else is for diagnosis.

A second-tier dashboard the data-platform team owns shows preprocessing-side health: stage cache hit rates (how often we're reusing cached materializations), per-stage failure rates, quarantine bucket size and aging, dedup ratio drift over time (a sudden change in dedup ratio is often the first signal of a source format change upstream). These aren't on the training on-call rotation but they're the leading indicators for next month's training-side problems.

## 12. What I'd Build First vs Defer

With one quarter and a small team, I'd ship in this order:

1. **Object-store layout + content-addressed CAS + Iceberg catalog.** Foundation; everything else depends on it.
2. **Sharded streaming reader with WebDataset + deterministic shard permutation + mid-run reshard support.** Single highest-leverage piece for the training team.
3. **Idempotent preprocessing DAG on Ray Data** with three stages: filter, tokenize, pack. Dedup and quality classifiers come in v2.
4. **Hot-tier cache integration with pre-warm.**

Defer: multimodal decode optimization (until first multimodal customer is real), semantic dedup (until quality classifiers prove insufficient), full differential-privacy support (until on-device team formally onboards), cost-attribution dashboards (until someone's bill becomes a problem).

---

The two highest-leverage decisions in this design are (1) content-addressed, idempotent preprocessing stages — they make reprocessing cheap, and that's what dictates whether you can iterate on data quality at all — and (2) the permutation-then-partition shard-to-rank mapping with mid-run reshard support, which is the difference between a training team that can elastically scale and one that's stuck at a fixed world size. The single most likely thing to go wrong in year one is silent tokenizer-version skew producing subtly corrupted models for some run while the dataloader and the trainer disagree about vocabulary — which is why I keep harping on hard-fail-at-startup version pinning. That's the bug I'd put guardrails around before I optimized anything else.