---
title: Dedup system
description: Dedup system
---
Title: Duplicate File Detection — Single Machine to Distributed
Prompt: "Design a system to find duplicate files. Start with a single-machine version: given a large directory tree, identify groups of files with identical contents even when names and paths differ. Then extend it to a distributed storage environment — many machines, billions of files, petabytes of data. Address:

How to avoid unnecessary full file reads
How to compare files efficiently and safely (false-positive vs false-negative tolerance)
How to manage CPU, memory, and disk I/O
How to support incremental rescans when files are added, modified, or deleted
How to partition work across machines
How to aggregate and verify duplicate groups in a distributed setting

Stretch: extend the system to detect near-duplicate images and videos — files that are perceptually identical but byte-different (transcoded, resized, re-encoded). Discuss the architectural changes."

---

# Duplicate File Detection — Single Machine to Distributed

## 1. Reframing

This problem looks like a hashing question. It is not. **[STAFF SIGNAL: I/O-as-bottleneck reframing]** At billions of files and petabytes of data, the constraint is not the cost of SHA-256 (~500 MB/s/core, trivially parallel) but the cost of **reading the bytes off disk**. A single 200 MB/s SSD reads 10 TB in ~14 hours; 100 PB on 10K machines reads in ~14 hours only if every disk runs flat-out continuously, which would crater every co-tenant workload. The entire architecture is a sequence of cheap predicates that let us **avoid** opening files.

The single-machine version is a setup move. The distributed extension is the question — specifically, three sub-problems: partitioning the shuffle without hot keys, scheduling the huge-file tail, and rescanning incrementally without re-reading the unchanged 99%. The perceptual stretch is a **different system** with different mathematical structure, not an extension. **[STAFF SIGNAL: saying no]** I'll push back on that framing in the last section.

## 2. Single-Machine Version

### 2.1 The Funnel

**[STAFF SIGNAL: funnel discipline]** Each stage shrinks the candidate set with progressively more I/O cost.

```
ALL FILES: 10M files, 10 TB total
   │
   ▼ stat() only — read metadata, no file bytes
GROUP BY SIZE: ~95% of files are size-unique → drop
   │  Surviving: ~500K files in size-collision groups
   ▼ read 4 KB head + 4 KB tail of each survivor
GROUP BY CHEAP HASH (xxh64 of head+tail): 
   │  Most accidental size matches diverge in head bytes
   │  Surviving: ~50K files in cheap-hash groups (~0.5%)
   ▼ full file read + SHA-256
GROUP BY STRONG HASH:
   │  Surviving: confirmed duplicate groups
   ▼ optional byte-compare (policy choice)
CONFIRMED DUPLICATES
```

I/O budget by stage, on the 10 TB / 10 M file dataset:

| Stage | Files touched | Bytes read | Wall time @ 200 MB/s |
|---|---|---|---|
| stat() group-by-size | 10 M | ~0 (metadata) | seconds |
| Cheap hash (8 KB head+tail) | 500 K | 4 GB | ~20 s |
| Strong hash (full file) | 50 K | ~50 GB | ~4 min |
| **Naive baseline (hash everything)** | 10 M | 10 TB | **~14 hours** |

The funnel buys ~200x reduction in bytes read. **This ratio — not the algorithm — is what staff is being asked to demonstrate.**

### 2.2 Hash Choice and Collision Policy

**[STAFF SIGNAL: collision-policy commitment]** Three families:

- **Cryptographic** (SHA-256, BLAKE3): collision-resistant. SHA-256 collisions require ~2^128 work. BLAKE3 is faster (~3 GB/s/core, parallelizable within a single file via tree hashing) and equally secure — for greenfield, BLAKE3.
- **Strong non-cryptographic** (xxh3, xxh128): ~10 GB/s/core, no collision-resistance guarantee against adversaries but excellent against natural collisions.
- **Broken** (MD5): collisions are constructible. Never use for verification; OK as a cheap funnel stage if speed matters.

**Policy choice for the funnel:**
- **Cheap-hash stage:** xxh64 over 8 KB. Speed matters; false positives are fine — the strong stage handles them.
- **Strong-hash stage:** BLAKE3 of full file.
- **Byte-compare after BLAKE3 match?** Depends on the workload:
  - **Personal "find duplicates" tool, photo organizer:** trust the hash. False-positive probability is ~2^-128. Don't pay for byte-compare.
  - **Storage-layer dedup that physically merges blocks:** *byte-compare is mandatory.* A single false positive silently corrupts a file. The cost of the compare (one extra full read per confirmed group) is irrelevant against the cost of silent data loss.

I commit: for a "find your duplicates" tool, **trust BLAKE3, no byte-compare.** For a backup/dedup system, **always byte-compare on hash match.** The decision is driven by blast radius, not performance.

### 2.3 Resource Management

- **Bounded memory:** stream paths through a sized hash map keyed by size. Don't materialize the full path list. At 10 M files × 256 B per path entry, that's 2.5 GB — already too much; use a disk-backed external sort (size, path) keyed merge if needed.
- **Worker pool sized to disk concurrency, not CPU count.** SSDs benefit from queue depth ~32; rotational disks degrade past ~4 concurrent random reads. Hashing is CPU-cheap relative to read; one I/O thread feeds N hash threads via a bounded channel.
- **Page cache:** reading head+tail then later the full file — the head/tail will be cache-hot on the second pass *if* the working set fits in RAM. For 10 TB on a 64 GB machine, it won't; sequence the stages so each file is opened once when possible (read head, decide whether to continue, full hash in same fd).

## 3. Distributed Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                     CONTROL PLANE (small)                         │
│   • scan coordinator   • hash-cache GC   • verifier              │
└────────────────────────────┬─────────────────────────────────────┘
                             │ scan tickets, watermarks
       ┌─────────────────────┼─────────────────────┐
       ▼                     ▼                     ▼
┌─────────────┐       ┌─────────────┐       ┌─────────────┐
│ Crawler #1  │       │ Crawler #2  │ ...   │ Crawler #N  │   N = 10K
│ (per host)  │       │ (per host)  │       │ (per host)  │
│ reads local │       │ reads local │       │ reads local │
│ FS only     │       │ FS only     │       │ FS only     │
└──────┬──────┘       └──────┬──────┘       └──────┬──────┘
       │ {fs_id,inode,size,mtime,cheap_h,strong_h}│
       └──────────────┬──────┴────────────────────┘
                      ▼
        ┌────────────────────────────────┐
        │  HASH STORE (sharded KV)        │
        │  FoundationDB / Cassandra       │
        │  Key: (fs_id, inode)            │
        │  Indexes: by size, by strong_h  │
        │  ~1.5 TB for 10B files          │
        └────────────┬───────────────────┘
                     ▼
        ┌────────────────────────────────┐
        │  GROUPING / SHUFFLE             │
        │  Spark / Flink / Ray            │
        │  Stage 1: shuffle by size       │
        │  Stage 2: shuffle by strong_h   │
        └────────────┬───────────────────┘
                     ▼
        ┌────────────────────────────────┐
        │  VERIFIER (optional)            │
        │  byte-compare worker pool       │
        └────────────┬───────────────────┘
                     ▼
                  results
```

**Capacity math.** **[STAFF SIGNAL: capacity math]** 10B files × 150 B per record (fs_id, inode, size, mtime, cheap_h, strong_h, ts) ≈ 1.5 TB hash store. Trivial for any modern KV. The shuffle stage moves records, not file contents — at 150 B × 10B = 1.5 TB across 10K nodes, that's 150 MB of shuffle output per node. Cross-rack network is not the bottleneck.

The bottleneck is **the read step at the crawler.** 100 PB read at saturation across 10K nodes = 10 TB/node = ~14 hours per node assuming 200 MB/s sustained, with no co-tenant impact. The funnel reduces this 100x+. **Incremental rescans (Section 6) reduce it another ~100x.**

## 4. Partitioning and Shuffle

**[STAFF SIGNAL: hot-key handling]** The right partitioning is **hybrid**: data-local for the read, content-keyed for the grouping.

### 4.1 Stage 1 — Data-local crawl

Each crawler reads only files on its host's local file systems. Sending file bytes across the network for hashing is insane — it inverts the I/O ratio. The crawler emits one record per file: `{fs_id, inode, size, mtime, cheap_hash, strong_hash}`. Records are written to the hash store keyed by `(fs_id, inode)`.

### 4.2 Stage 2 — Shuffle by size

```
Crawlers ──records──▶ Shuffle by size
                      │
                      ▼
   size=0    size=4096   size=12345678 (rare)   size=2GB(rare)
   ───────   ──────────  ──────────────────     ────────────
   8B files  500M files     17 files                3 files
   HOT KEY!  HOT KEY!       OK                      OK
```

Most file sizes are unique or near-unique, so most groups have 1–2 entries and are dropped immediately. But two pathologies:

- **size = 0**: every empty file in the world. Skip these entirely at the crawler. An empty file isn't meaningfully a "duplicate" of another empty file in any system semantics that matter.
- **Common content sizes**: a popular installer binary, a shared library, a default config file. Some sizes have millions of entries. A naive `groupByKey(size)` puts them all on one reducer and stalls.

**Sub-partition popular sizes by cheap-hash prefix.** For any size with > 10K records, partition the shuffle by `(size, cheap_hash_prefix_8_bits)` — splits a hot reducer into 256 sub-reducers. The cheap-hash is already computed; this is free.

### 4.3 Stage 3 — Shuffle by strong hash

Survivors of Stage 2 (groups of size ≥ 2 with same size + same cheap hash) are re-shuffled by strong hash. By this point the candidate set is tiny (~0.5% of total), and strong-hash uniqueness is excellent — almost no hot keys survive. The output is the duplicate groups.

```
SHUFFLE STAGES — DATA VOLUME

Crawl emit:        10B records × 150B = 1.5 TB    (cross-host)
After size-dedup:    50M records ×  150B = 7.5 GB  (down 200x)
After cheap-hash:     5M records × 150B = 750 MB
After strong-hash:    duplicate groups
```

**[STAFF SIGNAL: invariant-based thinking]** Invariant the funnel preserves: **two files with identical content always end up in the same final group**. False positives are corrected by later stages; false negatives are forbidden. Every stage must be a *necessary* condition for equality, never sufficient. (Size match is necessary. Cheap-hash match is necessary. Strong-hash match is sufficient under the trust-the-hash policy; with byte-compare, only byte-compare is sufficient.)

## 5. The Huge-File Tail

**[STAFF SIGNAL: huge-file tail]** A 100 GB video at 200 MB/s takes ~8 minutes per file. A 1 TB VM image takes ~80 minutes. These dominate wall-clock if mixed into the general worker pool, because the pool blocks on its slowest task.

Three responses:

**Dedicated huge-file pool.** Files > 1 GB go to a separate worker pool sized for throughput, not parallelism. The general pool finishes the 9.99 M small files in minutes; the huge-file pool grinds through the long tail in parallel.

**Chunked, resumable hashing.** BLAKE3 supports tree hashing — hash chunks in parallel, combine at the end. More importantly, **persist the running state every N chunks** so a crawler crash 90% through a 1 TB hash doesn't restart from zero. The hash store gets `{fs_id, inode, hash_state, chunks_done, total_chunks}` for in-progress huge hashes.

**Optional cheap-stand-in for huge files.** For workloads tolerant to the rare false positive, define `huge_fingerprint = (size, BLAKE3(first_64MB), BLAKE3(last_64MB), BLAKE3(middle_64MB))`. 192 MB read instead of 1 TB. Two distinct 1 TB files agreeing on all three samples is astronomically unlikely outside adversarial inputs. Combined with byte-compare on match for paranoid use cases, this is strictly better than full hash.

I'd choose the cheap-stand-in for a "find duplicates" tool and full hash for a dedup-the-storage tool, mirroring the Section 2.2 policy.

## 6. Incremental Rescans — The Staff-Distinguishing Section

**[STAFF SIGNAL: incremental rescan as central]** Anyone can hash 10B files once. The distinguishing skill is rescanning tomorrow without re-reading the unchanged 99%.

### 6.1 The Identity Tuple

**[STAFF SIGNAL: identity-vs-path discipline]** The hash cache must be keyed by stable file identity. **Path is not identity** — a rename changes the path without changing the content. On POSIX, identity is `(fs_id, inode)`. On Windows, `(volume_id, file_id_64)`.

Cache record:
```
key:   (fs_id, inode)
value: {size, mtime, ctime, cheap_hash, strong_hash, hashed_at}
```

### 6.2 The Rescan Decision Flow

```
                   ┌──────────────────────┐
                   │ Crawler walks FS      │
                   │ stat() each file      │
                   └──────────┬───────────┘
                              ▼
                   ┌──────────────────────┐
                   │ Look up cache by      │
                   │ (fs_id, inode)        │
                   └──────────┬───────────┘
                              ▼
                ┌─────── miss ────────┐
                │                     │
                │              ┌──────▼──────┐
                │              │ hit:        │
                │              │ size match? │
                │              └──────┬──────┘
                │                     │
                │           ┌── no ───┴─── yes ──┐
                │           │                    │
                │           │            ┌───────▼──────┐
                │           │            │ mtime match? │
                │           │            │ ctime match? │
                │           │            └───────┬──────┘
                │           │                    │
                │           │          ┌── no ───┴── yes ──┐
                │           │          │                   │
                ▼           ▼          ▼                   ▼
         FULL HASH    FULL HASH    FULL HASH         CACHE HIT
         (new file)   (changed)    (changed)         (skip read)
                                                          │
                                                          ▼
                                                   re-stat after
                                                   to confirm
                                                   nothing raced
```

### 6.3 The mtime Lie and Its Workaround

mtime is unreliable in three ways:

1. **`touch -t` rewrites mtime to a forged value** without changing content. Most attackers don't bother; most users don't either. Acceptable risk for non-adversarial workloads.
2. **Atomic-write-by-rename** (`tmpfile.write; rename(tmpfile, target)`) replaces the target with a new inode. The cache entry for the old inode goes orphan; the new inode misses. **This is correct behavior** — the file truly is new from the FS's perspective. The new content gets re-hashed.
3. **Same mtime, different content** via `mtime`-preserving copy tools or filesystems with second-resolution mtime where two writes happened in the same second. Mitigation: also track `ctime` (inode change time), which is harder to forge — only root can set it on Linux. **My rule: trust the cache only if `(size, mtime, ctime)` all match.**

### 6.4 Renames

A file moved from `/a/foo` to `/b/foo`: same inode, same content. **The hash is still valid.** The path index gets updated; the hash cache entry is reused. This is the entire reason path is not the cache key — renames in a 10B-file system are common (build systems, package managers, log rotation), and re-hashing them is wasted work.

### 6.5 Cross-FS Case

Same file copied to a different file system has a different `(fs_id, inode)`. The cache misses; we re-hash. **This is correct** — the alternative (keying by path or content-prefix) risks false caching where a stale entry from FS A poisons a lookup on FS B. Re-hashing on cross-FS copy is the price of correctness.

### 6.6 Garbage Collection

Files disappear: deletion, unmount, transient FS errors. The hash store accumulates orphans. A weekly GC pass marks-and-sweeps: any cache entry whose `(fs_id, inode)` did not appear in the last successful crawl is a candidate for deletion.

**Policy:** delete with a 30-day grace period. A temporarily unmounted volume should not cost a full re-hash on remount. **[STAFF SIGNAL: blast radius reasoning]** The blast radius of a too-aggressive GC is "we re-hash unnecessarily on remount" — recoverable, just expensive. The blast radius of a too-lax GC is "stale entries take up space" — bounded and cheap. Prefer lax.

### 6.7 Race Conditions

**[STAFF SIGNAL: race-condition awareness]** A file changes between `stat()` and the read+hash. Protocol:
1. `stat()` → record `mtime_before`
2. Open + read + hash
3. `stat()` again → `mtime_after`
4. If `mtime_before == mtime_after`, commit the hash. Else discard and reschedule.

Without this check, you cache a hash that doesn't match any version of the file's content — silent corruption of the cache.

### 6.8 Quantified Win

If 99% of files are unchanged between scans (typical for storage systems with mostly-static data), the rescan reads ~1% of the bytes. **14-hour scan → ~10 minutes.** The cache hit rate is the single most important operational metric for the system. If it drops below ~95%, something is wrong (mtime resolution, atomic-write churn, or a bug).

## 7. Verification Policy

**[STAFF SIGNAL: collision-policy commitment]** Two policies, choose by failure cost:

| Policy | When | Cost | Failure mode |
|---|---|---|---|
| Trust BLAKE3 | duplicate finders, photo dedup, analytics | 0 (one read, one hash) | ~2^-128 collision = effectively never |
| BLAKE3 + byte-compare | storage dedup, backup, content-addressable storage | 1 extra read of every confirmed-group member | adversarial collision still defeated; only natural collisions caught |

The question that distinguishes the two: **what happens if we wrongly conclude two distinct files are identical?** If the answer is "the user gets confused for a moment," trust the hash. If the answer is "we deduplicate to one block and silently lose data forever," byte-compare. **No middle ground.** "We byte-compare a sample" is wrong — a sample-compare on 1% of bytes is not protection against adversarial collisions.

For the 100 PB / 10B file system: byte-compare is feasible *only* for confirmed candidate groups (the survivors of Section 4's funnel — ~0.5% of files at most). Byte-compare scheduling is co-located with the data: each member of a candidate group is read on its home host, hashed independently, and the hashes are compared. If the verifier ever re-hashes and disagrees with the cache, the cache entry is invalidated and the file is re-scanned — **[STAFF SIGNAL: failure mode precision]** treat this as a corruption signal, log it, alert.

## 8. Operational Reality

**[STAFF SIGNAL: operational reality]**

- **Failure recovery:** crawler crashes mid-scan. The hash store has partial results keyed by `(fs_id, inode)`. The restart picks up files not in the cache — same code path as incremental rescan. No special "resume" logic.
- **Throttling:** the crawler must not saturate disk I/O on production hosts. Cgroup-bound `iostat` budget per host (e.g., max 50 MB/s read, max 20% disk-time). Schedule scans to off-peak windows. **A scan that takes 24 hours instead of 14 because it's polite is the right tradeoff.**
- **Race conditions:** the stat-hash-stat protocol from §6.7. On detected race, retry up to 3 times then defer to next scan.
- **Observability:** four signals matter:
  - Per-stage funnel reduction ratio (catches bugs that break the funnel — e.g., a cheap-hash bug that suddenly drops 0% of candidates)
  - Cache hit rate on rescan (target ≥95%)
  - Tail latency of huge-file hashing (catches scheduling issues)
  - End-to-end "scan freshness" (when was the last successful walk per host)
- **Scan-completion semantics:** in a live FS, "scan complete" never means "we have a perfectly consistent snapshot." It means: "we have a consistent snapshot as of scan-start, with all files-modified-during-scan deferred to next scan." This is the only honest model. **[STAFF SIGNAL: saying no]** I'd push back if anyone asks for "exactly find all duplicates right now" semantics — it's not a meaningful spec on a live file system.

## 9. Perceptual Stretch — A Different System

**[STAFF SIGNAL: perceptual reframing]** Exact duplication forms an equivalence relation: reflexive, symmetric, **transitive**. We can partition into equivalence classes ("groups"). The whole grouping architecture above depends on this.

Perceptual duplication has none of these properties cleanly. If similarity is "cosine ≥ 0.9":
- A ≈ B (sim 0.95), B ≈ C (sim 0.92), A vs C? Could be 0.85. Not transitive.
- "Group of duplicates" is not even well-defined — it depends on threshold and traversal order.

**The right model is similarity search, not grouping.** "For each query image, return its k-nearest neighbors above threshold τ." The output is a graph of similarities, not a partition.

### 9.1 What's Changed Recently (2025)

The state of the art has moved decisively toward learned embeddings. A September 2025 TrufoAI benchmark on 10K baseline + 30K modified images found **AI fingerprinting at 67% recall @ 99% precision, vs 25% recall for PDQ and pHash**. Models like DinoHash compress DINOv2-derived embeddings (originally ~400M params) down to <30M params for production scale. ICCV 2025's ResidualViT cuts video frame encoding cost ~2.5x by exploiting temporal redundancy (I-frames vs P-frames analogous to video compression). V-JEPA 2 (Meta, mid-2025) gives strong self-supervised video embeddings without the cost of frame-by-frame ViT encoding. None of these change the architecture below; they change the encoder cost line item.

Perceptual hashes (pHash, dHash, PDQ) are still useful, but only as the **first layer** for high-recall, low-precision filtering. They miss crops, rotations beyond a few degrees, watermarks, and any "same scene, different angle" case. For real perceptual dedup at scale in 2026, embeddings are mandatory; pHash is an optimization.

### 9.2 Architecture

```
                         IMAGE / VIDEO INGEST
                                  │
                                  ▼
                  ┌─────────────────────────────┐
                  │ Stage 0: Exact dedup         │
                  │ (Sections 2-7, BLAKE3)       │
                  └────────────┬─────────────────┘
                               │ unique by bytes
                               ▼
                  ┌─────────────────────────────┐
                  │ Stage 1: pHash / PDQ         │
                  │ ~64-256 bit hash             │
                  │ Hamming distance index       │
                  │ catches transcodes,          │
                  │ resizes, recompresses        │
                  └────────────┬─────────────────┘
                               │ unique up to pHash
                               ▼
                  ┌─────────────────────────────┐
                  │ Stage 2: Embedding           │
                  │ DINOv2/SigLIP/DinoHash       │
                  │ → 512-1024 dim vector        │
                  │ (GPU-bound, expensive)       │
                  └────────────┬─────────────────┘
                               │ vectors
                               ▼
                  ┌─────────────────────────────┐
                  │ Stage 3: Vector index        │
                  │ HNSW / DiskANN / SCANN       │
                  │ approximate kNN              │
                  └────────────┬─────────────────┘
                               │ similarity graph
                               ▼
                  ┌─────────────────────────────┐
                  │ Stage 4: Threshold + review  │
                  │ τ tunable per use case       │
                  │ output: similarity edges,    │
                  │         not equivalence      │
                  │         classes              │
                  └─────────────────────────────┘
```

### 9.3 The Architectural Delta

Three things change from exact dedup:

1. **Encoder GPU compute becomes dominant.** SHA-256 was ~free relative to disk read. Embedding generation on a ViT-L is ~50–100 images/sec/GPU; at 10B images that's ~3 GPU-years. Most production systems run perceptual dedup as an opt-in tier, not on the full corpus.
2. **Hash store becomes a vector index.** A KV store with prefix scans on (size, hash) is replaced by a vector index (HNSW or DiskANN). At billions of vectors × 768 dims × 4 bytes = ~3 TB raw — fits in DiskANN; HNSW in pure RAM is harder. This is its own infrastructure problem.
3. **Output is a graph, not a partition.** Downstream consumers (UI, dedup, retrieval) must handle "near-duplicate of X with similarity 0.92" rather than "in group G." Threshold tuning becomes a product question. **[STAFF SIGNAL: perceptual reframing]**

### 9.4 Video

Video adds a temporal axis, which breaks the "one embedding per file" model. Three approaches in current production:

- **Per-keyframe embedding + temporal alignment.** Extract keyframes (one per scene change), embed each, store as a sequence. Compare via dynamic time warping or set similarity. Cheap, but loses motion.
- **Tubelet / 3D ViT (ViViT, V-JEPA).** Native 3D patches over space+time. Better quality, much more expensive — and the embedding-per-clip granularity is awkward for "is this 30-second clip a duplicate of part of that 2-hour video?"
- **ResidualViT (ICCV 2025).** Encode I-frames fully, encode P-frames as cheap residuals — ~2.5x cost reduction. Promising for the dense-sampling case.

Perceptual video dedup at petabyte scale is genuinely an open problem. Most production systems sidestep it by deduping at the manifest/asset level, not the content level.

## 10. What I'd Push Back On

**[STAFF SIGNAL: saying no]**

The prompt asks to "find duplicate files" in a distributed storage environment. Three things in that framing I'd challenge:

1. **"Find all duplicates" is not well-posed on a live FS.** Files change during the scan. The honest spec is "snapshot at scan-start, lag = scan-duration." If the requirement is real-time deduplication, this is a different system — write-time content-addressable storage (Section 7's byte-compare-on-write), not a periodic scanner.

2. **"Distributed" should not mean "ship bytes to a central hasher."** Several candidates I've seen want to fan-in file contents to a hashing cluster. This inverts the I/O ratio by 100–1000x. The data-locality of hashing is non-negotiable: hash where the bytes live.

3. **The perceptual stretch is a different product.** Bolting perceptual onto exact dedup is wrong because the output semantics differ — equivalence classes vs similarity graphs. They should share infrastructure (the crawler, the hash store as a vector store substrate) but be exposed as separate systems with separate APIs and separate SLAs. Pretending they're "the same thing with a different hash function" is the staff-level mistake that makes the perceptual system unmaintainable.