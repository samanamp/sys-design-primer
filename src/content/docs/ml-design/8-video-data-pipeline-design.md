---
title: "Training Data Pipeline for a Large Video World Model"
description: ""
---

**Scope:** 100 PB raw video+audio → curated multimodal corpus → tensors on 10k accelerators, operated continuously.
**Author stance:** written as the design I would defend at a Staff+ system design interview, with sourced 2025–26 practice tags: `[established practice]`, `[current practice 2025–26]` (with source), `[emerging/contested]`, `[inferred]`.

---

## 0. Global assumptions and the one piece of math everything hangs on

State assumptions up front; every stage's sizing derives from these.

| Quantity | Assumption | Rationale |
|---|---|---|
| Raw corpus | 100 PB video+audio | ~2 GB/hr average (mixed 480p–4K, H.264/H.265, ~2–8 Mbps) → **~50M hours raw** |
| Source mix | 40% licensed catalogs, 50% web-crawled, 10% synthetic (rendered/sim/model-generated) | licensed skews long-form high-quality; crawl skews short, noisy, duplicated |
| Image corpora | ~5B images + image-text pairs (~1–2 PB) | joint image-video training; images carry appearance diversity cheaply `[established practice]` (Movie Gen, HunyuanVideo, Cosmos all co-train images) |
| Modality token mix | ~70% video / 20% image / 10% image-text & interleaved, varying by phase | early phases image-heavier; final high-res phases video-heavier `[current practice 2025–26]` (Movie Gen progressive schedule; HunyuanVideo 1.5 unified T2I/T2V/I2V training) |
| Tokenizer | causal 3D VAE, 8×8 spatial, 4× temporal compression, 16 latent channels; DiT patchify 2×2 | matches Wan 2.x / HunyuanVideo family `[current practice 2025–26]` |
| Training cluster | 10k accelerators (H100/B200-class or TPU pod), ~100–150 day flagship run |
| Token target | 10^12–10^13 latent video tokens + image tokens |

**The token↔hours identity (memorize this; it drives everything).**
Tokens per second of 720p24 video through this tokenizer:

```
latent frame: (1280/8)/2 × (720/8)/2 = 80 × 45 = 3,600 tokens
latent fps:   24/4 = 6
→ ~21,600 tokens per video-second at 720p  (~78M tokens/hour)
At 256p:  (256/16)×(144/16)×6 ≈ 864 tokens/s   (~3.1M tokens/hour)
```

So **10^13 tokens ≈ 128K unique hours at 720p, or ≈ 3.2M hours at 256p**. With a progressive-resolution curriculum (most tokens at low res, final phase at high res), the effective unique-video demand is **~1–5M curated hours, trained for 2–5 epochs on the top-quality tier**. This is the central structural fact of the pipeline: *out of 50M raw hours you only need a few million excellent ones, and each surviving second is read many times*. Curation is not a nice-to-have; it is where the compute goes. `[established practice]` — every frontier video report (Movie Gen, Cosmos, Seedance, HunyuanVideo) describes multi-stage funnels discarding the large majority of raw footage.

A second derived fact: **online loading of latents is trivially cheap; offline processing is where the system lives.** Cluster token consumption over a 100-day 10^13-token run is 10^13 / 8.64×10^6 s ≈ **1.16M tokens/sec cluster-wide**. At 64 B/token (2×2 patch × 16 ch × fp8) that is **~75 MB/s of latent bytes across the entire 10k-accelerator cluster**. Even with 10× overhead (captions, text embeddings, mixture inefficiency, multi-resolution) it is <1 GB/s aggregate — a single storage rack. The pipeline bottlenecks are all *upstream* of training: decode, filtering, captioning, and VAE encoding at 100 PB scale. The document is organized around that asymmetry.

---

## End-to-end architecture (volumes annotated)

```
                         CONTINUOUS OPERATION (new data lands daily; recipes evolve)

  licensed (40 PB)   web crawl (50 PB)   synthetic (10 PB)
        │                  │                  │
        ▼                  ▼                  ▼
 ┌─────────────────────────────────────────────────────┐
 │ 1. INGEST & RAW STORE  (object storage, 100 PB)     │   ← provenance/legal tags,
 │    byte-level dedup, perceptual-hash dedup,         │     eval carve-out FIRST
 │    metadata catalog (~10^9 assets)                  │
 └───────────────┬─────────────────────────────────────┘
                 │ 100 PB → ~85 PB post exact/near-dup     eval vault: ~0.1 PB (frozen)
                 ▼
 ┌─────────────────────────────────────────────────────┐
 │ 2. PREPROCESS FLEET (offline, GPU+CPU, ~2–4k GPUs)  │
 │    decode → scene split → resample ladder →         │
 │    quality/motion/OCR/NSFW filters → audio path     │
 └───────────────┬─────────────────────────────────────┘
                 │ ~85 PB → ~1–3B clips, ~15–20M hours survive (~30 PB transcoded)
                 ▼
 ┌─────────────────────────────────────────────────────┐
 │ 3. CAPTION / LABEL FLEET (VLM inference, ~1–3k GPUs)│
 │    dense captions, camera motion, structured meta   │
 └───────────────┬─────────────────────────────────────┘
                 │ + ~50–100 TB text/metadata
                 ▼
 ┌─────────────────────────────────────────────────────┐
 │ 4. CURATION & MIXTURE (embeddings, semantic dedup,  │
 │    quality tiers, recipe/manifest versioning)       │
 └───────────────┬─────────────────────────────────────┘
                 │ dataset vN = frozen manifest over ~2–5M hrs top tier + long tail
                 ▼
 ┌─────────────────────────────────────────────────────┐
 │ 5. TRAINING-READY STORE (shards: latents+captions)  │
 │    VAE-encoded latents, res/duration buckets        │   ~2–6 PB latents (all ladders)
 └───────────────┬─────────────────────────────────────┘
                 │ aggregate read: <1 GB/s (!) for 10k accels
                 ▼
 ┌─────────────────────────────────────────────────────┐
 │ 6. ONLINE LOADER (host CPUs, prefetch, determinism) │──► 10k accelerators
 └─────────────────────────────────────────────────────┘    ~6–8 MB per 5s-720p sample
        ▲                                    │
        │ 8. observability, lineage,         │ 7. lifecycle: snapshots, backfill,
        │    contamination checks            ▼    tokenizer-change migration
```

---

## Stage 1 — Ingestion & raw storage

**Receives:** raw bytes from three source tiers. **Exposes:** immutable content-addressed objects + a queryable asset catalog; contract to Stage 2 is `(asset_id, uri, container/codec probe, provenance record, dedup status)`.

### 1.1 Storage layout

Object storage (S3/GCS/Blobstore-class), **content-addressed**: key = `sha256(bytes)`, with a metadata catalog (distributed OLTP + an analytics mirror in Iceberg/Delta) mapping `asset_id → {source, license, crawl_url, upload_time, probe_info, hashes}`. Content addressing gives exact dedup for free and makes every downstream artifact reproducibly traceable to bytes. `[established practice]`.

Raw assets are write-once. Never mutate; all processing produces *new* derived objects keyed by `(asset_id, recipe_version)`. At ~10^9 assets, the catalog is ~1 TB of metadata — trivial; the discipline it enforces is not.

- **Licensed:** delivered via partner drops (physical or dedicated interconnect — at 40 PB, shipping disks/Snowball-class appliances beats the network: 40 PB over a 100 Gbps link is 37 days of saturated transfer). Rich native metadata (titles, chapters, rights windows).
- **Crawled:** fetch fleet with politeness/robots handling, ~50 PB accumulated over months. Store the crawl record (URL, timestamp, page context) — page text becomes weak supervision and provenance evidence.
- **Synthetic:** rendered/sim/model-generated. Tag hard at ingest: synthetic data must be mixture-controlled and excluded from eval material, and model-generated video must be firewalled from feedback loops (see §4, §8).

### 1.2 Dedup at ingest

Two layers, in order:

1. **Exact:** sha256 on bytes. Catches mirror re-uploads. Cheap, do it inline.
2. **Perceptual/near-dup:** per-video fingerprint = sequence of frame-level perceptual hashes (pHash/videohash) at ~1 fps + an audio fingerprint (chromaprint-style). Index in an ANN store; flag matches above threshold as duplicate-of, keeping the highest-bitrate copy as canonical. This catches re-encodes, resolution variants, watermarked copies — the dominant duplication mode in web video. `[established practice]` for exact/perceptual at ingest; **semantic (embedding) dedup is deliberately deferred to Stage 4** because it needs decoded clips and good embeddings, and because Cosmos/NVIDIA-style pipelines run semantic dedup after filtering, on clips, not raw assets `[current practice 2025–26]` (Cosmos-Predict curation pipeline: split → transcode → filter → caption → semantic dedup → shard; Cosmos 3 report: embeddings + dedup before filtering/annotation — teams differ on ordering, i.e. `[emerging/contested]` where exactly dedup sits, but *that it happens at both hash and embedding level* is uncontested).

Napkin: fingerprinting 50M hours at 1 fps = 1.8×10^11 frames of tiny hash compute — CPU-bound on decode, folded into the Stage 2 decode pass for crawled data; a cheap thumbnail-track decode (many containers carry one) or keyframe-only decode makes ingest-time fingerprinting ~50× cheaper than full decode. Expect **10–25% of crawled bytes to be near-dups**; call it 100 → 85 PB effective.

### 1.3 Legal/provenance tagging

Every asset carries an immutable provenance record: `{source_tier, license_id, territory/rights window, opt_out_status, PII/faces policy class, crawl evidence}`. This is load-bearing, not compliance theater: **mixture construction (Stage 4) and takedown handling (Stage 7) are keyed on it.** A takedown must be executable as "tombstone asset_id → next manifest excludes all derived clips" without reprocessing anything, which requires lineage from Stage 1 onward. `[established practice]` in principle; the rigor level is `[current practice 2025–26]` driven by litigation environment — treat it as a hard requirement, and it is `[inferred]` that all frontier labs now maintain asset-level rights lineage since none publish details.

### 1.4 Eval carve-out at ingest — before anything else runs

**Reserve held-out eval material at ingest time, before dedup and before any filter ever sees it.** Mechanism: hash-based salt-and-split on asset_id (plus targeted curated eval sets), routing ~0.1% of each source tier into a sealed eval vault that the training pipeline cannot read.

Why before dedup/filtering, not after: two failure modes.

- **Dedup leaks eval into train.** If you carve eval *after* dedup, near-duplicates of eval clips (re-encodes, clips embedded in compilations) survive in train — dedup already collapsed them to a train-side canonical copy. Carving first lets you instead run the *reverse* check: fingerprint the eval vault, then **purge train-side near-matches of eval material** during Stage 2/4 (decontamination), which is only possible if eval membership is fixed before the corpus is touched.
- **Filters overfit the benchmark.** Filters and curation models evolve by measuring eval deltas (Stage 4 feedback loop). If eval material was itself subject to those filters, the eval distribution drifts with the recipe and you're grading yourself on paper you wrote. Frozen-at-ingest eval sets are the only stable yardstick across recipe versions. `[established practice]` in LLM land (decontamination against fixed benchmarks); applying it structurally at ingest for video is `[current practice 2025–26, inferred]` — video reports rarely detail it, which is exactly why an interviewer probes it.

**Cost/volumes:** 100 PB store at cool-tier object pricing ~$10–12/TB/mo → **~$1.0–1.2M/month**; drop the crawl tier to archive class after transcode (Stage 2) and steady-state raw storage lands ~**$0.5M/mo**. Ingest compute is negligible next to Stage 2.

---

## Stage 2 — Preprocessing fleet (offline)

**Receives:** ~85 PB deduped raw assets. **Exposes:** clip records: `(clip_id, asset_id, t_start–t_end, transcoded stream per ladder rung, audio stream, filter scores, decode health)` — ~1–3B clips, ~15–20M surviving hours, ~30 PB of normalized transcodes.

This is the largest offline compute consumer after captioning. Design center: a GPU-accelerated streaming pipeline (Ray-based orchestration with heterogeneous stages) — `[current practice 2025–26]`: NVIDIA's Cosmos-Curate/NeMo Curator processes video at 100 PB scale on exactly this shape, reporting ~89× speedup over CPU pipelines using 1k GPUs and explicitly mixing GPU types to exploit NVDEC/NVENC on L40S alongside H100/GB200 for model stages. What changed vs ~2023: pipelines like Panda-70M-era tooling were CPU-ffmpeg farms with model filters bolted on; **current systems treat curation itself as a GPU inference workload with hardware decode as a first-class resource** — this used to be done on CPU because model-filter stages were light; now filtering/captioning dominate, so colocating decode with GPUs and streaming frames GPU-resident (DALI-style) avoids paying PCIe and re-decode repeatedly.

### 2.1 Decode strategy

- **Hardware decode (NVDEC) as default** for H.264/H.265/VP9/AV1. One modern GPU's NVDEC engines sustain roughly 5–15× real-time at 1080p per engine, several engines per card; decode is "free" alongside SM-bound filter models on the same card. `[current practice 2025–26]` (NeMo Curator autobalancing across L40S NVENC/NVDEC + H100 compute).
- **CPU/ffmpeg fallback** for the codec long tail (~2–5% of web video: MPEG-2, weird profiles, broken containers) and as the arbiter for files hardware decoders reject.
- Napkin — total decode: 50M hours ≈ 1.8×10^11 s of video. At an effective 100× real-time per GPU (multiple NVDEC engines, batched), a full-corpus decode pass ≈ 1.8×10^9 GPU-s ≈ **500K GPU-hours ≈ $1–1.5M** — decode is real money but ~10× smaller than captioning; you can afford ~2–3 full decode passes over the project's life, not twenty. This is the number that later forces "store transcodes/latents" over "re-decode raw on demand."

### 2.2 Robustness to corrupt and adversarial files

At 10^9 web-sourced files, **decoder crashes are a certainty and a security surface** (codec parsers are historically the most exploited code paths in existence). Non-negotiables:

1. **Sandbox every decode**: decode workers run in locked-down containers/gVisor-class sandboxes, no credentials, no network egress. A malicious file must be able to do nothing but kill its worker. `[established practice]` for any web-content processor; `[inferred]` for video pipelines specifically since nobody writes it down, but a Staff answer that omits it is incomplete.
2. **Retry budget then quarantine**: N=2 retries (second on CPU path), then the asset gets `decode_status=quarantined` in the catalog and is *never* silently retried by future runs. At 100 PB, a 0.5% pathological-file rate without retry caps means an eternally-crashing tail that eats the fleet. Track quarantine rate per source; a spike is an ingest-quality alarm.
3. **Watchdog + poison-pill isolation**: hangs (infinite-loop bitstreams) are as common as crashes; per-file wall-clock timeout scaled to duration.
4. **Partial salvage**: decode up to first corruption, keep clean prefix clips if ≥ threshold duration.

### 2.3 Scene detection & splitting

Shot-boundary detection (histogram/feature delta, e.g. TransNetV2-class or PySceneDetect-grade for cheap first pass) → split into single-shot clips, then merge/trim into training windows (e.g. 2–20 s, favoring 5–10 s). Cosmos: "shot-aware video splitting" is stage 1 of their pipeline `[current practice 2025–26]`. Why shots: cuts inside a training window teach the model that content teleports; world models especially need intra-clip physical continuity. Long-form licensed content yields many clips per asset — keep `asset_id` lineage so semantic dedup and mixture caps can act per-source-asset (otherwise one 3-hour nature documentary becomes 1,000 highly-correlated clips flooding a bucket).

### 2.4 Resampling: the resolution ladder serves the curriculum

Transcode each surviving clip to a **fixed ladder**, e.g. `{256p, 512p, 720p}` (+1080p for the top tier only), at normalized fps (24, with the original fps recorded; high-fps sources also yield a 48 fps variant for motion-critical buckets). The ladder exists because the *training curriculum* is progressive-resolution — phase 1 trains at 256p over the widest data, later phases at 512p/720p over progressively filtered subsets `[established practice]` (Movie Gen explicitly trains with progressive resolution scaling; essentially all 2024–26 video models do low-res-first). Design consequences:

- The **same clip_id** exists at every rung; curriculum phase selects the rung via the manifest, not via re-processing.
- Ladder rungs are generated in one decode pass (decode once, downscale tree, encode each rung with NVENC) — never decode per-rung.
- Storage multiplier: ladder ≈ 1.3–1.5× the top-rung size (lower rungs are cheap); see cost roll-up in §9.
- Aspect ratio: bucket to a small set of ARs (16:9, 9:16, 1:1, 4:3) with center/smart crop recorded as metadata; Cosmos includes an explicit cropping stage `[current practice 2025–26]`.

### 2.5 Audio path

Decision: **keep audio for the top-quality tier and any clip passing an AV-sync check; store as 16 kHz mono Opus (speech/ambience) or 44.1 kHz stereo for music-tagged content; drop audio for clips with silent/broken/mismatched tracks.** Rationale: audio-video joint generation is now a live target (Movie Gen audio; MOVA-style AV pipelines filter on audio quality *and* AV alignment and caption both modalities `[current practice 2025–26]`), and audio is nearly free to store (~1% of video bytes) but expensive to re-acquire — raw may be archived/deleted. Even if the flagship model is video-only, keep audio: the marginal cost is a rounding error and it options future models. Sync: remux with PTS normalization; clips whose audio drift exceeds ~50 ms after normalization get `audio_status=unsynced` and train video-only. Audio gets its own light filter stack (silence detection, clipping, language ID, music/speech/ambience tags) feeding mixture control.

### 2.6 Quality filtering

Ordered cheap→expensive so each filter only sees survivors of the previous (classic funnel; `[established practice]`, and the specific stack below matches Cosmos/Movie Gen publicly described stages `[current practice 2025–26]` — Movie Gen: visual filtering, motion filtering, content filtering, then captioning):

| Filter | Mechanism | Typical kill rate | Cost |
|---|---|---|---|
| Technical | bitrate/resolution/duration floors, decode health | 10–20% | free (metadata) |
| Static/slideshow & motion | optical-flow magnitude or cheap motion score; kill near-static and chaotic-jitter extremes | 15–25% | light GPU |
| Text-overlay/OCR | detect burned-in subtitles, watermarks, UI captures | 10–20% (web tier) | light GPU |
| Aesthetic/quality | trained scorer (LAION-aesthetic-style lineage → modern VLM-scored quality) — **score, don't hard-kill**: score feeds tiering in Stage 4 | tiering only | medium |
| NSFW/safety/CSAM | multi-model + hash-matching (industry hash lists); hard kill + report path | 1–5% | medium |
| Face/PII policy | detect & tag density of faces, plates, documents; policy applied per deployment, tag not delete | tag only | medium |

Two Staff-level points interviewers look for: (a) **filters emit scores into the catalog; hard deletion is reserved for safety/technical failures** — everything else is a *tier*, because the optimal aggressiveness of aesthetic/motion filters is a mixture decision that changes across curriculum phases and model generations, and you cannot un-delete; (b) **filter versions are part of data lineage** — clip records carry `{filter_name: (score, model_version)}` so a filter upgrade triggers targeted backfill (§7), not amnesia.

### 2.7 The flagship decision: offline VAE/latent encoding vs decode-on-the-fly

Three candidate architectures for what the trainer reads:

**(A) Raw/transcoded video, decode+VAE-encode on the fly.**
**(B) Transcoded video stored; decode on the fly on hosts; VAE encode on the accelerator as a training-step prefix.**
**(C) Offline VAE encoding: store latents; trainer reads latents directly.**

Numbers first. VAE encode throughput: a causal-3D VAE (Wan/Hunyuan class) encodes 720p at roughly 10–20× real-time per H100-class GPU in bf16 with a tuned implementation `[inferred from published tokenizer benchmarks; order-of-magnitude]`. Training consumes ~54 video-seconds/sec at 720p cluster-wide (1.16M tok/s ÷ 21.6K tok/s — from §0), *but* multi-epoch reads mean each stored second is consumed 2–5×, and low-res phases consume vastly more video-seconds per token.

- **Option A on-the-fly encode cost:** at the 256p phase the cluster eats ~1,300 video-sec/sec (1.16M ÷ 864). At even 50× real-time per GPU for 256p encode, that's ~30 GPUs of continuous VAE encode — small — **but** it must run on accelerators (VAE is GPU work), either stealing training-cluster cycles or as a sidecar fleet with a network hop, and it re-pays the cost every epoch and every restart-replay.
- **Option C offline cost:** encode ~5M curated hours × ladder ≈ 2.5×10^10 video-sec-equivalents; at 15× real-time/GPU ≈ 1.7×10^9 GPU-s ≈ **~470K GPU-hours ≈ $1–1.5M**, once per tokenizer version.
- **Latent storage — the counterintuitive number:** 720p bf16 latents are 160×90×6 fps×16 ch×2 B ≈ **2.8 MB/s — ~4× larger than the 5 Mbps H.264 source**. Latents are *bigger than compressed video.* fp8 latent storage halves it (validate reconstruction/loss deltas first; `[emerging/contested]` — community fine-tuning stacks like musubi-tuner already cache latents in reduced precision routinely, frontier-lab precision choice unpublished). 5M hours × ladder at ~1.5 MB/s avg fp8 ≈ **~3–4 PB of latents ≈ $60–80K/mo** — noise next to raw storage.

**Decision: (C) offline latents for everything inside a dataset snapshot, with (B) retained as a fallback path for rapid tokenizer experiments.** The decisive arguments are *not* raw FLOPs (which mislead toward A):

1. **Read-many amortization + determinism.** Latents are encoded once, read 2–5×; on-the-fly encode pays per read *and* makes the sample stream depend on decoder/VAE numerics of the day. Bitwise-stable inputs make loss-spike forensics (§8) and elastic resumption (§6) tractable. `[current practice 2025–26]`: latent + text-embedding pre-caching is standard across open video training stacks (kohya/musubi-class tooling caches VAE latents and text-encoder outputs to disk precisely to free GPU cycles and memory during DiT training), and it is `[inferred]` that frontier runs do the same at scale — their throughput proofs don't work otherwise.
2. **It deletes the host decode problem.** Option B needs host CPUs to decode video at cluster rate with random access into shards — feasible at 720p rates but a persistent operational tax (codec bugs on the training hosts, decode jitter → stragglers → whole-cluster step-time inflation; one slow host gates a synchronous step across 10k chips).
3. **The re-encoding risk — priced, not avoided.** If the VAE/tokenizer changes, all latents are invalid. Mitigations: (i) full re-encode is a *known, bounded* ~$1–1.5M + ~1–2 weeks on a 2k-GPU burst — budget for 2–3 tokenizer generations up front; (ii) keep the transcoded ladder (30 PB) as the durable "source of truth" tier, so re-encode never touches raw; (iii) freeze the tokenizer *before* the flagship run and treat mid-run VAE changes as a new run (see follow-up Q1). What breaks at 10×: at 10^14 tokens / 50M curated hours, re-encode grows to ~$10–15M per tokenizer change — at that scale you either invest in tokenizer stability (train the VAE earlier, harder) or move to option B with a dedicated encode sidecar tier and accept the determinism tax.

> **Defend this in 60 seconds — offline latents.** "I store VAE latents, not pixels, as the training-facing format. Key numbers: latents at 720p are actually ~4× *larger* than the H.264 source — so this is not a compression play — but the entire 10k-chip cluster only consumes about 75 MB/s of latent bytes, so a few PB of latents make dataloading a non-problem forever. The one-time encode of ~5M curated hours costs about $1M of GPU time versus a ~$50M+ training run; in exchange I get bitwise-deterministic inputs for resumability and loss-spike forensics, and I remove video decode from the training hosts entirely. The risk is tokenizer churn: a VAE change invalidates every latent. I price that as a bounded ~$1M, 1–2 week re-encode against the transcoded tier, budget for two or three of them, and freeze the tokenizer before the flagship run."

**Preprocessing fleet sizing & cost roll-up (Stage 2):** decode+split+filters ≈ 0.7–1M GPU-hours + large CPU pool for demux/remux/IO; transcode (NVENC) rides along with decode. Order **$2–3M compute** for a full-corpus pass, plus ~30 PB transcode tier ≈ $350K/mo. Compare training: 10k GPUs × 120 days ≈ 29M GPU-hours ≈ **$60–90M** — the *entire* offline pipeline (Stages 2+3+VAE) is ~10–15% of training cost, which is exactly the ratio that makes "spend more on data, not less" the correct marginal trade. `[current practice 2025–26]` directionally: NeMo Curator's pitch is 1k GPUs for days over 20M hours, i.e., same order.

---

## Stage 3 — Labeling & captioning as an embedded inference system

**Receives:** filtered clips + ladder + audio. **Exposes:** per-clip structured record: `{captions[dense, short], camera_motion, tags, objects/tracks?, audio_caption?, embedding, label_model_versions}`.

This is **the single largest offline compute line** and should be designed like a production inference service, because it is one: continuous batch inference with SLOs on throughput and quality, not a script.

### 3.1 Fleet sizing

Caption the survivors: ~15–20M hours → at ~10 s/clip ≈ **5–7B clips**. A 7–13B VLM captioning a short clip (sampled frames or native video input) takes ~1–2 GPU-seconds with a tuned serving stack (continuous batching, fp8 KV, prefix caching of the fixed instruction — this is standard inference optimization and where an inference-systems background directly transfers). 6B clips × 1.5 GPU-s ≈ 9×10^9 GPU-s ≈ **2.5M GPU-hours ≈ $5–8M** — comparable to a mid-size model pretraining run, purely for captions. Consequences:

- **Caption-on-demand beats caption-everything**: caption in tier order (top tier first, dense; lower tiers get short captions or none until promoted). Never caption what filtering will kill — ordering filters before captioning saves ~3–5× `[established practice]`, and it's the explicit ordering in Cosmos and Movie Gen pipelines `[current practice 2025–26]`.
- **Recaptioning is a version, not an overwrite**: caption model upgrades append `caption_v2` alongside `caption_v1`; mixture recipes pick versions. Dense recaptioning with each new internal VLM generation measurably improves prompt adherence — this is the DALL-E-3-lineage insight, now uniform across video models `[established practice]`: Cosmos, HunyuanVideo, MOVA (Qwen3-Omni + MiMo-VL captioners with an LLM merging modality captions), CINEMA (Qwen2-VL) all describe VLM captioning as the core annotation step `[current practice 2025–26]`.
- **Caption density strategy**: multiple captions per clip (one dense ~100–200 words, one short, optionally style-varied) + structured fields. Training samples a caption variant per epoch — cheap augmentation against caption-style overfitting `[current practice 2025–26]` (Movie Gen uses multiple caption lengths/densities).

### 3.2 Structured metadata beyond captions

- **Camera motion classification** (static/pan/dolly/handheld/drone…) — cheap classifier or VLM field; world models and controllable generation need it as a conditioning signal, and Cosmos-class AV pipelines extract it `[current practice 2025–26]`.
- **Object tracks**: only for the subset feeding interactive/world-model objectives (detection+tracking is ~10× caption cost per clip; run on the ~5–10% of data where action/interaction supervision matters — driving, manipulation, egocentric).
- **Audio captions/tags** for the audio-kept tier; merged AV caption via LLM fusion `[current practice 2025–26]` (MOVA).
- **Embeddings** (video-level + per-clip visual embedding from an internal CLIP/SigLIP-video-class model): computed here because frames are already decoded and GPU-resident; consumed by Stage 4 for semantic dedup, clustering, retrieval, and mixture analytics.

### 3.3 Synthetic data loops

Rendered/sim data (physics-clean, action-labeled) is ingested like any source with `synthetic` provenance. **Model-generated video is used only for targeted gaps (rare camera moves, controlled counterfactuals) with hard mixture caps (~1–5%) and mandatory provenance tags** — self-training loops on generated video degrade physical fidelity in ways aesthetic filters don't catch `[emerging/contested]` — distillation-from-own-outputs is used by several teams for edit/instruct data (Movie Gen instruct-editing data), but wholesale generated-video pretraining is widely distrusted.

### 3.4 QC on labels

Sampled human review + automated checks: caption-video CLIP-score floor, hallucination spot-checks (VLM-as-judge with a *different* model family to decorrelate errors `[current practice 2025–26]` — AI-judge quality filtering is explicitly in Cosmos 3's curation), length/language distribution monitors, and per-captioner-version eval on a fixed human-labeled golden set of ~10K clips. Golden set lives in the eval vault lineage — never trained on.

---

## Stage 4 — Curation & mixture management

**Receives:** captioned, scored, embedded clips. **Exposes:** **dataset snapshots**: frozen, versioned manifests assigning every training sample to `(tier, cluster, mixture stream, weight)`.

### 4.1 Semantic dedup

Embedding-space clustering (k-means or ANN-graph over the Stage-3 embeddings); within tight clusters, keep a capped number of representatives weighted by quality score. This removes *conceptual* duplication exact/perceptual hashing can't see: 400 near-identical talking-head clips, 10K nearly-identical gameplay runs. `[current practice 2025–26]`: semantic deduplication is an explicit named stage in Cosmos-Predict 2.5 and Cosmos 3 curation. Kill/downweight rate on web tiers: another 20–40%. Compute: ANN over ~5×10^9 embeddings is a few thousand CPU-node-hours + modest GPU — noise.

### 4.2 Quality tiers and clusters

Combine filter scores into tiers (T0 pristine → T3 marginal), and cluster embeddings into a semantic taxonomy (domains: people/nature/driving/manipulation/indoor/…; refined by caption keywords). Tiers gate curriculum phases (high-res phases read T0–T1 only `[established practice]` — quality-annealed final phases are universal); clusters power mixture balancing and gap analysis ("we are short on low-light handheld indoor footage").

### 4.3 Mixture weights across sources and modalities

The mixture is a first-class, versioned artifact: a list of **streams** (source-tier × domain-cluster × modality) with sampling weights per curriculum phase. Assumed shape (stated, since the prompt requires it):

| Phase | Res | Video | Image | Image-text/interleaved | Notes |
|---|---|---|---|---|---|
| P0 warm | 256p | 40% | 45% | 15% | images bootstrap appearance/text alignment cheaply |
| P1 main | 256→512p | 65% | 25% | 10% | broadest video tiers T0–T2 |
| P2 hi-res | 720p | 80% | 15% | 5% | T0–T1 only; audio-joint stream enters |
| P3 anneal | 720/1080p | 85% | 10% | 5% | T0 + curated “finishing” sets |

`[established practice]` that image-video joint training with phase-varying proportions is how every serious video model trains (Movie Gen: billions of images + O(100M) videos; HunyuanVideo 1.5 unified T2I/T2V; Cosmos image+video curation is one pipeline); the *specific* percentages are `[inferred]` — no frontier team publishes exact mixture weights, and saying so out loud in an interview is better than fake precision.

### 4.4 Recipe versioning & the filtering-feedback loop

`recipe_vN = {filter thresholds + versions, dedup params, tier definitions, mixture table, caption version selection, tokenizer version}` — a small declarative file whose evaluation against the catalog *deterministically* materializes a manifest. Feedback loop: train small proxy models (0.5–2B DiT, ~1–2% of flagship compute) on candidate recipes, score on the frozen eval vault, promote winners. `[established practice]` in LLM data work (data-ablation proxies); `[current practice 2025–26]` for video via the same teams' reports describing filter tuning against generation quality. Guardrail: proxy-model rankings can invert at scale for mixture (not filter) decisions — validate the top-2 candidate mixtures at a 10× larger proxy before committing the flagship.

---

## Stage 5 — Training-ready storage & sharding

**Receives:** manifest vN over latent-encoded clips + captions/embeddings. **Exposes:** sharded, bucketed, shuffled physical layout + an index the loader can map `(step, rank) → samples` from, deterministically.

### 5.1 Shard format

Requirements: sequential-read-friendly, seekable to sample boundaries, tolerant of KB–tens-of-MB sample-size variance, self-describing, resumable. Options:

| Format | Seekability | Ecosystem | Verdict |
|---|---|---|---|
| WebDataset (tar) | sequential only (index add-ons exist) | huge, standard tooling | fine baseline; weak determinism story `[established practice]` |
| TFRecord | sequential | TF-centric | pass |
| MDS (MosaicML Streaming) | per-sample index, random access within shard | strong elastic-determinism features | **pick** for the loader properties (§6) |
| Energon (WebDataset++) | indexed | NVIDIA/Megatron multimodal | strong alternative, esp. on Megatron stacks `[current practice 2025–26]` |
| Parquet/Lance | columnar random access | analytics-friendly | good for catalog/metadata; awkward for big binary samples |

Pick **an MDS/Energon-class indexed shard format**: per-sample index enables sample-granular deterministic resumption and elastic resharding, which tar-only WebDataset makes painful — the exact failure MosaicML built Streaming around (OPT-era logs of burned GPU-hours replaying dataloaders after crashes) `[current practice 2025–26]`; Energon exists because NVIDIA needed the same properties for multimodal Megatron training. Contested edge: several labs run custom internal formats with identical properties — the *properties* are the answer, not the brand.

**Sample record:** `{clip_id, latents[fp8, one rung], caption variants (tokenized), text-encoder embedding (optional precomputed), metadata (bucket, tier, stream_id, lineage refs)}`. Shard size **~256–512 MB** (big enough for sequential-read efficiency and object-store request amortization; small enough that shard-granular shuffle has enough shards to mix well: 3 PB / 256 MB ≈ 12M shards).

### 5.2 Bucketing by resolution/duration — and why it's a parallelism problem

Samples are binned by (rung, AR, frame-count bucket: e.g. 1 frame(image)/33/65/121 frames) and **shards are bucket-pure**. Reason: a synchronous data-parallel step needs all ranks to see same-shaped (or same-token-count) batches; mixed shapes → padding waste or rank divergence. `[current practice 2025–26]`: Motif-Video's report describes exactly this — joint frame×resolution bucket sampling under FSDP2/HSDP where a bucket's global step can only proceed when *all* ranks have a full batch of that bucket — and solves it with an **offline bucket-balanced sampler**: precompute the per-step bucket schedule globally so every rank agrees by construction. Adopt that: the manifest materialization step emits a global `(step → bucket, shard-slice per rank)` plan. Alternative (dynamic bucket agreement at runtime via collective negotiation) adds a coordination path that can deadlock under elasticity — rejected.

### 5.3 Shuffle strategy at 100 PB→PB scale

You never globally shuffle bytes. Three-level shuffle, standard `[established practice]`:

1. **Write-time scatter:** manifest materialization assigns clips to shards pseudo-randomly *within bucket×stream* (seeded by manifest version) — breaks source locality (the 1,000-clip documentary spreads across thousands of shards).
2. **Epoch-level shard permutation:** seeded permutation of shard order per epoch per stream.
3. **Sample-level shuffle buffer** in the loader (size ~10–100 shards' worth per node) for intra-window mixing.

Quality check: autocorrelation of source_id/cluster_id over the realized sample stream must be flat beyond lag ~global-batch; monitored in §8. This is also the loss-spike-mitigation lever the Mosaic docs call out (raise shuffle strength / stratified mixing when batch composition variance destabilizes loss) `[current practice 2025–26]`.

### 5.4 Storage tier & bandwidth math

Aggregate read demand (from §0): <1 GB/s sustained for latents at 10k accelerators; even with 5× burst (restart re-prefetch across the fleet) it's ~5 GB/s. Standard object storage serves this trivially; a flash cache tier is justified only for *metadata/index* latency, not bandwidth. Contrast: if we had chosen pixel-space loading at 720p (66 MB/s per concurrent stream uncompressed, or decode-from-H.264 at ~0.6 MB/s per stream + CPU decode), object-store bandwidth still fine but host CPU/decode fleet becomes the constraint — the latent decision (§2.7) is what makes this section boring, and boring is the goal.

Placement: training-ready shards live **in-region with the training cluster** (same datacenter/az). Cross-region reads of even 1 GB/s sustained ≈ 2.6 PB/mo egress ≈ $130–250K/mo at cloud egress rates *and* adds tail latency — see §9 placement.

---

## Stage 6 — Online loading path

**Receives:** shard store + global step plan. **Exposes:** per-rank iterator delivering ready tensors into HBM ahead of compute, deterministically, under failures.

```
 object store (latent shards, in-region)
      │  ~<1 GB/s aggregate, 10k accels
      ▼
 per-node shard cache (local NVMe, LRU, ~1–4 TB/node)      ← multi-epoch reuse:
      │                                                       2nd epoch mostly cache hits
      ▼
 host loader workers (per rank): read indexed samples per global plan
      │   fp8 latents → dtype cast → (optional) latent-space aug → collate
      ▼
 pinned-memory prefetch ring (depth 2–4 steps) ──H2D copy overlapped──► HBM
```

Design points, each with the failure it prevents:

- **Decode/transform placement:** with offline latents, host work is nearly nil — index lookup, read, cast, collate. Deliberately thin: host CPU headroom is reserved for the *fallback* pixel path (option B, tokenizer experiments) where hosts decode video via NVDEC-on-host or CPU ffmpeg. Keep augmentations that must see pixels (crops/flips) *upstream at encode time* (encode augmented variants or accept latent-space-safe augs only — temporal crop of causal latents is exact; spatial flip is not, so flip is baked at encode for a fraction of copies). `[emerging/contested]`: latent-space augmentation validity varies by VAE; the conservative call is what's written.
- **Prefetch depth 2–4 steps**, sized so that p99 sample-fetch latency < step time. With 6–8 MB samples and local NVMe cache this is comfortably milliseconds vs multi-second video-DiT steps; the risk is not bandwidth but **stragglers** — one rank's cache miss to object store (~50–200 ms) must hide under prefetch depth, or a synchronous step across 10k chips inflates by the max over ranks. Monitor per-rank fetch p99.9 (§8).
- **Determinism & resumability:** the global step plan (§5.2) + indexed shards + a per-rank cursor `(step, sample_offset)` in the checkpoint gives **exact-sample resumption in seconds** with no dataloader replay — the core Streaming/Energon property `[current practice 2025–26]`. Elastic reshards (world size changes): because the plan maps *steps→samples* independent of rank count via a fixed canonical partition, re-partitioning ranks over the canonical order preserves the sample sequence (Streaming's parallelism-aware sample replication/partitioning does exactly this).
- **Multi-epoch & mixture-sampling correctness:** mixture is enforced *in the plan*, not by runtime rejection sampling — per-step stream composition is stratified (each global batch contains each stream at its target weight in expectation with bounded variance; Mosaic's `stratified` batching method is the same idea `[current practice 2025–26]`). Epoch boundaries are per-stream (a 2%-weight stream cycles many times while the main stream does 2 epochs); the plan tracks per-stream epoch counters so no stream silently over-repeats.
- **Throughput proof (the closing argument):** per accelerator, 1.16M tok/s ÷ 10k = ~116 tok/s → at 64 B/token ≈ **7.4 KB/s of latent bytes per accelerator** (~6–8 MB per sample every ~50–100 s per rank at 720p; proportionally more samples but similar bytes at low-res phases). Host budget: reading+casting 8 MB per step-interval is <1% of one CPU core. Storage budget: <1 GB/s cluster-wide vs ~10s of GB/s a single modern object-store deployment sustains. **Margin ≥ 10× on every leg; the loader cannot be the bottleneck unless the design regresses to on-the-fly pixel decode.**

> **Defend this in 60 seconds — determinism under elasticity.** "My loader is a pure function from (manifest version, seed, step) to the exact sample set on every rank — the shuffle, mixture, and bucket schedule are all precomputed into a global plan when the manifest is materialized. That buys three things: resumption after a crash is a cursor seek, not an hours-long dataloader replay — the failure MosaicML documented burning thousands of GPU-hours on OPT; elastic reshards preserve the sample sequence because the plan is defined over a canonical partition, not over ranks; and when a loss spike happens I can name the exact clips in the offending batch after the fact. The cost is that runtime mixture changes require cutting a new manifest — that's a feature: every mixture the model ever saw is a versioned artifact."

---

## Stage 7 — Lifecycle: continuous operation & versioning

**Receives:** an eternally-growing catalog + evolving recipes. **Exposes:** reproducible snapshots and bounded-cost migration paths.

- **Incremental processing:** new assets flow Stages 1–4 continuously as micro-batches; every derived artifact is keyed `(asset_id, stage_recipe_version)` so incremental runs are idempotent upserts into the catalog. The pipeline is a *materialized view maintenance* system over the catalog — that framing (declarative recipe, incremental view maintenance) is the correct mental model and is how modern lakehouse-based curation stacks behave `[current practice 2025–26]` (Iceberg/Delta incremental pipelines; Ray/Xenna streaming stages for the GPU work).
- **Dataset snapshots as frozen manifests:** "dataset v3" is physically: (a) an immutable manifest file-set (Parquet) listing every sample with its shard/offset, stream, weight, and full lineage refs; (b) a pin on the exact shard objects (object-store versioning / no-delete lease); (c) the recipe file and all model versions (filters, captioner, tokenizer) that produced it. Nothing is copied to snapshot — manifests reference shared immutable shards, so v3 and v4 dedupe storage naturally.
- **Backfill when a filter or tokenizer changes:** two regimes.
  - *Metadata-only change (filter/captioner):* lazy — new recipe reads existing scores where versions match, schedules recompute only for affected clips, materializes v(N+1) when coverage crosses threshold. Cost ∝ affected clips only.
  - *Tokenizer change:* invalidates the latent tier wholesale → **eager, full re-encode from the transcode tier** (~$1–1.5M, 1–2 weeks on burst capacity, from §2.7). Lazy latent migration is rejected for the flagship path: mixed-tokenizer latents inside one manifest would poison determinism and comparability. Lazy is allowed only for exploratory branches. 10× scale flips this: at ~$15M per re-encode you introduce a dual-format transition window and migrate stream-by-stream between runs.
- **Reproducing a months-old run:** requires exactly: manifest vK + seed + loader version + shard objects (pinned). Retention policy: pin shards for every manifest used by a *released or referenced* run indefinitely; exploratory manifests get 6-month leases. The transcode tier (30 PB) is the permanent reproducibility floor — raw crawl can go to deep archive ($1–2/TB/mo ≈ $100–200K/mo for 100 PB) or be partially deleted per legal policy, because everything training-facing is regenerable from transcodes + recipes.

---

## Stage 8 — Observability & data quality in production

**Receives:** telemetry from every stage + the live training run. **Exposes:** dashboards, lineage queries, alarms, and a rollback procedure.

- **Per-stage dashboards:** throughput (assets/hr, GPU-hrs consumed, $ burn), kill rates per filter per source (a filter's kill-rate drift is the cheapest early warning of upstream distribution shift — e.g., a crawl source silently switching codecs shows up as decode-quarantine spikes hours before anything else), queue depths, quarantine rates.
- **Sample-level lineage:** any training sample resolves in one catalog query to `clip_id → asset_id → source/license → every filter score+version → caption versions → shard/offset → every manifest and step where it was served`. This is what makes the next two bullets possible; it's ~1–2 TB of indexed metadata — trivial storage, high discipline.
- **Contamination/eval-leakage checks:** continuous job fingerprints (perceptual hash + embedding) every clip entering the training-ready tier against the sealed eval vault; matches are purged from manifests and alarmed. Run *again* at manifest materialization (defense in depth — filters change, new near-dups arrive). Same machinery handles takedowns and model-generated-content firewalls (generated-video fingerprints are registered at generation time and matched at ingest to prevent self-training loops).
- **Bad-data incident, end-to-end:** loss spike at step S → (1) the deterministic plan names the exact global batches around S; (2) lineage expands them to clips/sources/filter-versions in minutes; (3) if a cohort is implicated (e.g., a captioner-v3 backfill batch with garbled tokenization), quarantine the cohort via a manifest *patch* (v3.1 excludes cohort; loader hot-swaps manifests at a step boundary), rewind to the pre-spike checkpoint, resume — with skipped-samples accounting so the plan stays consistent; (4) postmortem adds a stage-level validator (e.g., token-distribution check on caption outputs) so the class of incident is caught offline next time. The whole procedure only works because of §6 determinism + §8 lineage — this is the systemic payoff of those choices, and it's the story to tell when an interviewer asks "how do you *know* your data is fine."

---

## 9. Cost roll-up, placement, and the top cost drivers

### 9.1 Total storage footprint including all derived artifacts

| Tier | Size | $/mo (order) | Notes |
|---|---|---|---|
| Raw (post-transcode → archive class) | 100 PB | $200–500K | deep archive for crawl; cool for licensed |
| Transcode ladder (durable source-of-truth) | ~30 PB | $300–400K | 15–20M hrs × 3-rung ladder |
| Latents (fp8, all rungs, curated tier) | ~3–4 PB | $60–80K | regenerable; pinned per manifest |
| Captions/metadata/embeddings/catalog | ~0.2 PB | <$10K | |
| Eval vault (sealed) | ~0.1 PB | <$5K | separate account/ACL boundary |
| **Total ≈ 135 PB ≈ 1.35× raw** | | **~$0.6–1.0M/mo** | **multiplication factor over raw: ~1.35×** (would be ~1.6–2× if latents were bf16 and the full corpus were encoded — controlling *what* you encode is the lever) |

### 9.2 Compute (one full build + one flagship-run's consumption)

| Item | GPU-hours | $ (order) |
|---|---|---|
| Decode/split/filter/transcode pass | ~1M | $2–3M |
| Captioning/labeling (tiered) | ~2.5M | $5–8M |
| VAE latent encode (curated × ladder) | ~0.5M | $1–1.5M |
| Embeddings, dedup, proxy-model ablations | ~0.5M | $1–2M |
| **Offline total** | **~4.5M** | **~$9–15M** |
| Flagship training (10k × 120 d) | ~29M | **$60–90M** |

**Top 3 cost drivers of the whole pipeline:** (1) **captioning inference** — dominant offline line; managed by tier-ordered caption-on-demand; (2) **raw + transcode storage carry** — ~$0.6–1M/mo forever; managed by archive-tiering raw and treating transcodes as the reproducibility floor; (3) **reprocessing events** (tokenizer/filter generation changes) — episodic $1–3M each; managed by version-keyed lazy backfill for metadata and budgeted eager re-encode for tokenizers. Explicit comparisons the prompt demands: preprocessing fleet ≈ **10–15% of training cluster cost**; store-latents ≈ **$1–1.5M once + $70K/mo** vs re-decode ≈ recurring host-fleet + determinism/ops tax (latents win at ≥1 epoch reuse); caption-everything ≈ **$15–25M** vs caption-on-demand ≈ **$5–8M**.

### 9.3 Placement / colocation — egress as a first-class line item

Topology: **preprocessing fleet colocated with object storage region R_data; training cluster in R_train (often a different region/provider — you take capacity where it exists); the only planned cross-region flow is the training-ready tier.**

- Moving 100 PB raw cross-region ≈ $2–5M egress at cloud list rates + weeks of transfer — **never move raw**. The preprocessing fleet goes to the data, not vice versa; this is why the curation fleet is provisioned in R_data even if GPUs there are a worse SKU.
- The training-facing flow is only latent shards + manifests: initial sync ~3–4 PB ≈ **$60–200K one-time egress**, then a trickle of incremental manifests. This asymmetry — *the pipeline compresses 100 PB of data gravity down to a few PB of portable artifacts* — is the placement strategy. If training capacity moves (new cluster, new provider), you re-sync 4 PB, not 130.
- Within R_train: shard replica in-region, per-node NVMe cache (§6). Within R_data: everything on the same object-store backbone; Stage-2→3 intermediate frames pass GPU-resident or via node-local NVMe, never through the object store per-frame.

---

## 10. Hard follow-ups, answered

**Q1. The VAE changes mid-project — walk me through it.**
Two cases. (a) *Between runs* (normal): freeze VAE v(K+1), eager re-encode transcode tier → new latent generation ($1–1.5M, 1–2 wk), cut manifest v(N+1) pinned to tokenizer v(K+1); old manifests remain reproducible against old latents (pinned). (b) *Mid-flagship-run* (should be an incident, not a plan): latents, positional geometry, and channel statistics all shift — the honest options are (i) don't: finish the phase, switch at a phase boundary with a re-warm, or (ii) treat as new run initialized from checkpoint with an adaptation phase. What you never do is mix tokenizer versions inside one manifest. Prevention is the real answer: the tokenizer is trained and locked *before* flagship launch, with its own eval bar (reconstruction PSNR/LPIPS + downstream proxy-DiT loss), because §2.7 shows every tokenizer change costs ~$1.5M + 2 weeks of calendar — the org must feel that price.

**Q2. How do you shuffle 100 PB?**
You don't — you shuffle *pointers*, three times (§5.3): write-time pseudo-random scatter of clips→shards within bucket×stream (breaks source locality), per-epoch seeded shard permutation, and a node-local sample buffer. Global randomness at sample granularity over PB is unnecessary: what training needs is (a) no long-range autocorrelation of source/cluster in the realized stream and (b) stratified mixture per batch — both verifiable cheaply (autocorrelation monitor, §8) and both achieved by the three-level scheme. The number: 12M shards of 256 MB means the shard-permutation alone gives per-epoch orderings drawn from 12M! arrangements; the binding constraint is scatter quality at write time, which is why scatter is seeded by manifest version and tested, not assumed.

**Q3. Why not train from raw pixels?**
Compute, not storage. A 720p 5 s clip is 121 frames × 0.92 MP; pixel-space attention over ~10^8 pixel-tokens is not a thing — you need ~200–400× token compression (8×8×4 VAE ÷ 2×2 patch) to get 5 s of 720p into ~10^5 tokens where a DiT step is feasible. `[established practice]` — latent diffusion is universal (Wan, Hunyuan, Cosmos all causal-3D-VAE); the live research axis is *how much* compression (LTX-Video pushes stronger latent compression; DCAE-style high-ratio tokenizers are `[emerging/contested]`). Secondary benefits: latents make dataloading trivial (§0) and unify image/video in one token space. The honest caveat: the VAE is a lossy ceiling on achievable detail — which is why tokenizer quality gets its own eval bar (Q1) and why super-resolution cascades exist downstream (HunyuanVideo 1.5's latent-space VSR `[current practice 2025–26]`).

**Q4. How do you keep eval sets uncontaminated?**
Structurally, not procedurally: carve at ingest before any processing (§1.4), seal in a separate ACL/account boundary the training pipeline can't read, then run *reverse* decontamination continuously — fingerprint+embedding match every train-side clip against the vault and purge hits (§8). The subtle cases: (a) near-dups that pre-date the carve (a compilation containing an eval clip) — caught by perceptual+semantic matching, which is why the vault stores fingerprints at multiple granularities; (b) eval prompts leaking via captioners (captioner trained on data containing eval-like text) — mitigated by keeping the caption-QC golden set inside the vault lineage; (c) synthetic/generated data accidentally reproducing eval content — generated-content fingerprints registered at generation time (§8). Report contamination-scan coverage as a release gate metric.

**Q5. What changes for interactive world models with action labels?**
Three deltas. (1) *Schema:* samples become `(latents, action_stream, proprio/GPS/ego-pose, timestamps)` with strict temporal alignment contracts — alignment QC (cross-correlation of action vs optical flow) becomes a filter stage; Cosmos's AV pipelines already handle multi-camera + GPS/LiDAR tracks `[current practice 2025–26]`. (2) *Mixture:* action-labeled data is scarce and precious — it becomes its own protected streams with epoch multipliers, plus sim/synthetic action data at controlled ratios (Cosmos 3 curates robot-action CoT and embodied-reasoning sets as distinct curricula `[current practice 2025–26]`). (3) *Splitting:* shot-based splitting is wrong for continuous egocentric streams — you split on episode/behavior boundaries and must keep *long* windows (30 s–minutes) for temporally-extended credit assignment, which changes bucket design (long-duration buckets, fewer tokens/frame via lower fps or stronger temporal compression) and pushes toward sequence-parallel-aware shard layout (contiguous chunks of one episode land on the ranks of one SP group).

**Q6. A loss spike is traced to data — walk me through the investigation.**
Scripted in §8, but the interview version with teeth: (0) first rule out non-data causes fast — same checkpoint, replay the batch window on a debug replica; if the spike reproduces with the same samples but not with a shifted window, it's data. (1) Deterministic plan → exact clip_ids in the offending batches. (2) Lineage pivot: cluster the clips by every axis (source, filter versions, caption version, shard write date, encode job id). The smoking gun is almost always a *cohort*, not a clip — e.g., one encode job ran with a stale VAE checkpoint, or a captioner rollout emitted mojibake for one language, or a new crawl source dodged the OCR filter. (3) Manifest patch excluding the cohort, rewind to pre-spike checkpoint, resume under vN.1 with skipped-sample accounting. (4) Add the offline validator that would have caught it (latent-statistics check per encode job; caption token-distribution check per rollout). (5) The postmortem metric: time-from-spike-to-resume — target < a few hours, which is only achievable because inputs are deterministic and lineage is one query.

**Q7. Filters are models; models drift. How do you stop the curation stack from silently reshaping the data distribution over months?**
Three controls: (a) every filter/captioner is version-pinned in lineage, so distribution change is attributable; (b) a fixed *reference slice* (random, unfiltered, frozen sample of raw) is re-scored by every new filter version — score-distribution deltas on the reference slice measure the filter's drift independent of true data drift; (c) recipe changes only ship through the proxy-model ablation gate (§4.4) scored on the frozen vault. The failure this prevents: each individually-reasonable filter tweak compounds into a corpus that's beautiful, static, and boring — visible as motion-score and cluster-entropy trends on the curation dashboard, which are first-class SLO metrics, not vanity plots.

**Q8. Your token math assumed 720p24 and one tokenizer. The model team wants 1080p, 48 fps experiments, and variable-length up to 60 s. What breaks?**
Tokens/sec scales linearly in pixels and fps: 1080p48 ≈ 2.25×2 = 4.5× tokens/sec of 720p24 → per-sample bytes and step-plan token accounting change, but the loader margin was ≥10×, so bandwidth survives (~×4.5 → still <5 GB/s aggregate). What actually breaks: (a) bucket explosion — (rungs × fps × durations × ARs) is multiplicative; cap the bucket lattice and make long-duration a separate stream with its own plan, or the offline bucket-balanced scheduler fragments into buckets too small to fill global batches (the exact pathology Motif-Video's sampler exists to manage); (b) 60 s windows at 1080p ≈ 60×97K ≈ 6M tokens/sample → sequence parallelism required, and shard layout must become SP-group-aware (Q5.3); (c) VAE encode cost for the 1080p tier ≈ ×2.25 per hour — fine because the 1080p tier is small by construction (T0 only).

> **Defend this in 60 seconds — curate hard, then over-invest in the survivors.** "The token identity says a 10^13-token run needs only ~10^5–10^6 unique hours at training resolutions — out of 50M raw hours. So the pipeline's job is to be a 30-to-1 rejection funnel and then spend generously on what survives: dense VLM captions, embeddings, multi-rung transcodes, offline latents. Total offline spend is ~$10–15M against a $60–90M training run — 15%. The asymmetry that justifies it: a 1% improvement in data quality moves the flagship's outcome more than 1% more training FLOPs, and every published frontier pipeline — Movie Gen, Cosmos, Hunyuan — has converged on the same funnel shape: split, filter, caption, semantically dedup, shard. The differentiation isn't the funnel; it's the feedback loop that tunes it with proxy models against a frozen, ingest-time eval vault."

> **Defend this in 60 seconds — eval carve-out at ingest.** "I reserve eval material the moment bytes land, before dedup or any filter runs, and seal it behind an ACL the pipeline can't read. Two reasons with teeth: dedup run before carving will collapse eval near-duplicates into train-side canonical copies — you cannot decontaminate against an eval set that didn't exist yet; and my filters are tuned by measuring eval deltas, so if eval had passed through those filters, the yardstick would bend with every recipe change. The ongoing cost is one continuous fingerprint-matching job purging train-side matches of vault content — a rounding error — and the payoff is that 'dataset v3 beats v2' is a claim I can actually defend."

---

## Appendix — source map for `[current practice 2025–26]` tags

- **NVIDIA Cosmos / Cosmos-Curate / NeMo Curator** — GPU-accelerated Ray-based curation at 100 PB / 20M-hour scale; split→transcode→crop→filter→caption→semantic-dedup→shard; heterogeneous NVDEC/NVENC + compute GPUs; AV multi-sensor pipelines; Cosmos 3 action-CoT curricula and AI-judge filtering.
- **Meta Movie Gen** — image+video joint training at billions-images/O(100M)-videos scale; visual/motion/content filter stack; progressive-resolution training; multi-density captioning.
- **HunyuanVideo 1.5 / Wan 2.x** — causal-3D-VAE latent spaces (the tokenizer assumptions here); unified T2I/T2V/I2V training; latent-space VSR cascade.
- **MOVA** — audio-video joint curation: fixed-format clip normalization, AV-quality and AV-alignment filtering, dual-modality captioning merged by an LLM.
- **MosaicML Streaming (MDS) / Megatron-Energon / WebDataset** — indexed shard formats, deterministic elastic resumption, stratified mixture batching, shuffle-strength vs loss-stability guidance.
- **Motif-Video 2B** — offline bucket-balanced sampler for frame×resolution buckets under FSDP2/HSDP; the global-agreement constraint on bucketed batching.
- Where no public source exists (frontier mixture weights, fp8 latent storage at scale, ingest-time eval carving for video, sandboxed decode), claims are marked `[inferred]` — stated as reasoned engineering defaults, not reported practice.
