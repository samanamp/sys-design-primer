---
title: Multi modal LLM inference
description: Multi modal LLM inference
---
```
"Design an inference serving system for a multi-modal model — vision + text input, text output, with optional audio input. Think GPT-4o or Claude with vision class. Target 5K concurrent users, mixed workload (single-image chat, multi-image document analysis, real-time audio conversation). Walk me through it."
```
---
# Multi-Modal Inference Serving System — Staff-Level Design (2026)

> Question: design inference serving for a vision + text + optional audio model, GPT-4o / Claude-with-vision class, 5K concurrent users, mixed workload (single-image chat, multi-image document analysis, real-time audio).

---

## 1. State of the Art (2026)

A short, opinionated summary of what frontier multi-modal serving actually looks like today, before designing anything.

**Encoder/prefill/decode disaggregation is now standard.** vLLM merged native EPD support in PR #25233 (Nov 2025, v0.11.1), exposing encoder workers as independently scalable processes that hand off encoded features to prefill workers via a "EC connector" abstraction (NIXL-RDMA, Mooncake, shared memory). vLLM-Omni (Nov 2025) generalized this into a stage-graph abstraction for any-to-any models (Qwen3-Omni, BAGEL). NVIDIA Dynamo shipped EPD with vLLM earlier. ModServe (Microsoft Research, SoCC'25, Qiu et al., arXiv:2502.00937) reports 3.3–5.5× throughput and 25–41% cost reduction vs. monolithic vLLM on production Azure traces by separating "Image Instances" from "Text Instances" and applying modality-aware routing. EPD-Serve (Huawei, arXiv:2601.11590, Jan 2026) reports 57–69% throughput improvement over PD-disaggregated baselines on Ascend with dynamic E-P-D / EP-D / E-PD / ED-P deployment topologies.

**Hybrid parallelism is the production default.** vLLM, SGLang, and ROCm all converged on **ViT data-parallel + LLM tensor-parallel**. The vision encoder is typically 1–5% of total model parameters, so TP-sharding it incurs all-reduce cost without compute benefit. The `--mm-encoder-tp-mode data` flag (vLLM batch-level DP) load-balances input batches across replicas instead of sharding weights. This is universal in 2026.

**Tile-based vision is the dominant frontier pattern.** InternVL 2.5 uses 448×448 tiles with pixel-shuffle reducing each tile to 256 visual tokens, dynamic from 1 to 40 tiles per image (up to ~10K tokens per high-res document image). Pixtral uses native-resolution 16×16 patches with 2D RoPE — variable-token-count per image. Llama-3.2 Vision uses cross-attention rather than token insertion. This matters because **token-insertion architectures put image tokens in the KV cache; cross-attention architectures put them in a separate cache injected at specific layers.** They have different cache-management consequences.

**Real-time audio uses native speech-to-speech.** Moshi (Kyutai, arXiv:2410.00037) demonstrates 160ms theoretical / 200ms practical end-to-end latency using a 7B Temporal Transformer + Depth Transformer over Mimi codec frames at 12.5 Hz. GPT-4o reports 232ms minimum / 320ms average. The pipeline approach (Whisper → LLM → TTS) is dead at frontier latency; it cost OpenAI 2.8–5.4s in their pre-4o Voice Mode. Frontier audio is **streaming-input, codec-token-output, full-duplex** — the model speaks and listens simultaneously over interleaved audio token streams.

**KV cache for image tokens is a known scaling problem.** VL-Cache (ICLR'25) reports a single batch of 4 prompts × 5 images × 2K tokens at LLaVA-1.6-34B requires 110GB HBM for the visual KV alone. VLCache (SGLang-integrated, Dec 2025) achieves 2–5% recompute with 1.2–16× TTFT speedup via position-independent KV reuse with cumulative-error-aware recomputation. Production systems (NVIDIA NIM) ship prefix caching for VLMs gated behind explicit env-var flags because cache-key correctness for image content is non-trivial.

**What surprised me / contradicts the prompt's framing.** First, the prompt implies audio is "optional" and bolts on. It doesn't. Audio at frontier latency requires architectural decisions (native speech tokens, streaming encoder, codec output, full-duplex) that are incompatible with "we'll add a Whisper sidecar." Second, the encoder is no longer cheap relative to decoder when you actually measure: a Qwen2.5-VL-3B preprocessor alone (CPU image normalization, not even GPU encoding) takes ~143ms for a 1024px image (vLLM issue #27094) — and the HydraInfer paper sets a 1-second TTFT SLO for TextVQA, which current vLLM EPD struggles to hit because of preprocessing dominance. **Preprocessing, not encoding, is often the hidden bottleneck.** Third, ModServe data shows that as image fraction increases, image preprocessing time *overtakes prefill time* in monolithic deployments — so the "decoder dominates" intuition from text serving is wrong for image-heavy traffic.

---

## 2. Scope, Reframing, and the Architectural Premise

Before I draw any boxes, two reframings.

**Reframing 1 — the architectural premise: encoder/decoder workload asymmetry.** The vision encoder is a compute-bound dense matmul workload (a ViT is essentially a transformer prefill — 50–500ms of dense MFU-bound work). The LLM autoregressive decoder is a memory-bandwidth-bound workload (loading KV cache and weights every step, MFU typically 10–30%). These are different optima on different optimal silicon. Co-locating them means decode steals encoder cycles or vice versa. **Disaggregation is not an optimization — it is the architectural premise.** Every decision below flows from this.

**Reframing 2 — the prompt is under-specified.** "5K concurrent users" without a modality mix is meaningless. A 5K-user pool that's 100% text-only chat is roughly 50 H100s of decoder. A 5K-user pool that's 50% document analysis with 30 pages each is ~10× that, and the bottleneck is the encoder pool, not the decoder pool. I'm pushing back on this and assuming a stated mix:

- **70% single-image chat** (3500 concurrent): one image per turn, average turn every ~30s during active use → ~120 images/s sustained
- **25% document analysis** (1250 concurrent): bursty, average 10–20 images per request, ~1 request every 60–120s → ~150–250 images/s sustained, with bursts to 1000+
- **5% real-time audio** (250 concurrent): continuous streams, hard SLO

If the real distribution is different, the design adjusts; the framework doesn't.

### Commits

| Decision | Choice | Rejected | Why |
|---|---|---|---|
| Vision arch | Projector-based (SigLIP-class ViT + MLP + 70B-scale LLM) | Native multi-modal (single-artifact GPT-4o-style); cross-attention (Llama-3.2 Vision) | Projector gives a clean disaggregation seam and lets us run encoders on cheaper silicon. Cross-attn has nice memory properties but breaks the unified KV-cache abstraction we want for caching. Native is operationally simpler (one artifact) but forces all serving onto the LLM tier and forfeits the cost win. |
| Audio arch | Native speech-token end-to-end (Moshi-style multi-stream, or a Talker-Thinker like Qwen3-Omni) | Whisper → text LLM → TTS pipeline | The latency budget for "feels live" excludes pipelines. A 350ms total budget cannot fit Whisper (~200ms first-pass) + LLM TTFT + TTS first-frame. |
| Modality mix priority | Real-time audio is its own SLO class, document analysis is its own SLO class, single-image chat is its own | One unified SLO | Sharing SLOs across these is malpractice. A document-analysis 5s p99 is fine; an audio 5s p99 is broken. |
| EPD topology | Encode pool (smaller GPUs) + Prefill pool + Decode pool, with encoder+prefill colocation as a tunable for low-image-fraction tenants | Monolith; encoder-as-sidecar in same worker | ModServe + EPD-Serve numbers settle this. Encoder-as-sidecar steals decode SMs. |
| Tile policy | Resolution-adaptive: 1 tile (chat), up to 12 tiles (document), capped at 40 (forced down-sample beyond) | Fixed-resolution; user-controlled only | Fixed gives bad chat latency or bad document quality; user-controlled punts the cost decision to clients who don't understand it. Auto with override is the right shape. |
| Image-token cache | Two-tier: (a) image-feature cache (post-encoder, pre-prefill) keyed by perceptual+cryptographic hash; (b) KV-prefix cache for image-prefilled segments | KV-only; encoder-output-only | Two-tier captures the two distinct reuse opportunities: same image different model (feature cache valid), same image same model + same surrounding text prefix (KV cache valid). |

---

## 3. Capacity and Budget Math

### 3.1 Encoder QPS at 5K mixed users

Assumptions: SigLIP-L-class encoder, ~400M params, BF16 on H100. One 448×448 tile ≈ 60 GFLOPs. H100 BF16 peak ~990 TFLOPs, sustained ~50% MFU = 500 TFLOPs. → ~120μs per tile compute-only at batch 1; at batch 32, ~4ms wall time per batch (amortized ~125μs/tile). With preprocessing on CPU and tensor-cast/transfer overhead, real cost is closer to **8–15ms per tile at batch 32 on H100**.

| Workload | Concurrent | Tiles/req | Req rate (Hz/user) | Tiles/sec (sustained) |
|---|---:|---:|---:|---:|
| Single-image chat | 3,500 | 1 | 1/30 | ≈117 |
| Document analysis | 1,250 | 80 (8 pages × ~10 tiles) | 1/90 | ≈1,111 |
| Real-time audio | 250 | 0 (no images) | — | 0 |
| **Total sustained** | | | | **≈1,228 tiles/s** |
| **Bursty p99 (3× sustained)** | | | | **≈3,700 tiles/s** |

At 8ms/tile amortized and batch-32 efficiency: 1,228 tiles/s requires roughly 10 H100-equivalent (or ~20 L40S). Bursty p99 needs ~30 H100-equivalents in the encoder pool. **Document analysis is 9× the chat encoder load despite being only 25% of users.** This is the variable-cost-per-request problem in numbers.

### 3.2 Image-token expansion at the LLM

InternVL-class: 256 tokens/tile. Document image at 8 tiles = 2048 tokens. 8-page document = ~16K tokens of image alone. A 50-page PDF request = ~100K tokens before the user's question.

| Workload | Image tokens per request | KV cache per request (70B-scale, FP8, 80 layers, 8 KV heads, d=128) |
|---|---:|---:|
| Chat (1 tile) | 256 | ~40 MB |
| Document (8 pages, 80 tiles) | 20,480 | ~3.2 GB |
| Document (50 pages, 500 tiles) | 128,000 | ~20 GB (will OOM single H100; needs sharding/eviction) |

A 50-page document **does not fit in a single H100 of decoder KV without paging.** This is not optional — paged-attention with disk/CPU offload is mandatory for the document tier.

### 3.3 Encoder→Decoder feature transfer

Encoded features per image, 4096 tokens × 8192 hidden dim × 2 bytes (BF16): **64 MB**.

| Path | Bandwidth | Time for 64 MB |
|---|---:|---:|
| Same-host NVLink (H100) | 900 GB/s | ~70 μs |
| Same-host PCIe Gen5 | 64 GB/s | ~1 ms |
| Cross-host RDMA (NIXL/UCX) | 50 GB/s | ~1.3 ms |
| Cross-host TCP | 10 GB/s | ~6.4 ms |

Transfer is not the bottleneck for chat-class images. For 50-page documents, 500 tiles × 64 MB = 32 GB → 640 ms over RDMA — this **is** a bottleneck and forces cross-host transfer to overlap with encoding (asynchronous prefetching, EPD-Serve calls this "asynchronous feature prefetching").

### 3.4 Real-time audio budget

Target: 350ms end-to-end (Moshi achieves 200ms, GPT-4o reports 320ms — 350ms is a defensible product floor).

| Stage | Budget | Notes |
|---|---:|---|
| User-stop-speaking → VAD detect | 50 ms | Frame-aligned, 80ms Mimi frames |
| Network ingress (audio frames) | 30 ms | Continuous, runs in parallel with encode |
| Audio encoder (streaming, finish) | 30 ms | Final frame after VAD |
| LLM TTFT (first audio-token out) | 150 ms | This is the hard one — see §6.4 |
| Audio decoder first frame | 40 ms | Codec inverse, ~1 frame |
| Network egress | 50 ms | |
| **Total** | **350 ms** | |

The LLM TTFT is the squeezed component. A 70B LLM at H100 prefill for ~1K context (typical conversation) is 80–120ms wall time; for 8K context (long conversation), 300+ ms — which **breaks the budget**. This forces aggressive prefix caching and KV reuse for audio sessions, and a smaller / faster model variant for the audio tier (e.g., MoE with sparse activation, or speculative decoding always on).

---

## 4. High-Level Architecture

```
                                ┌─────────────────────┐
   user request  ───────────►   │  L7 ingress / auth  │
                                │  rate-limit (per-   │
                                │  modality)          │
                                └────────┬────────────┘
                                         ▼
                         ┌───────────────────────────────┐
                         │   Modality-aware router       │
                         │  - inspects content types     │
                         │  - decides path & SLO class   │
                         │  - extracts image hashes      │
                         └─┬─────────────┬───────────────┘
                           │             │
                ┌──────────┘             └────────────┐
                │                                     │
                ▼                                     ▼
   ┌────────────────────────┐         ┌──────────────────────────┐
   │  Image preprocessor    │         │  Audio session manager    │
   │  pool (CPU + NVENC)    │         │  WebSocket / WebRTC term  │
   │  - decode, resize,     │         │  - VAD, jitter buffer     │
   │  - tile, normalize     │         │  - Mimi frame pipeline    │
   │  - hash for caching    │         │  - duplex stream manager  │
   └─────────┬──────────────┘         └────────────┬─────────────┘
             │ raw tiles + meta                    │ audio frames
             ▼                                     │
   ┌────────────────────────┐                      │
   │  Vision encoder pool   │                      │
   │  L40S / A100 (cheap)   │                      │
   │  ViT DP, batch=32      │                      │
   │  - feature cache write │                      │
   └─────────┬──────────────┘                      │
             │ image features (64 MB / tile-batch) │
             │ over NIXL/RDMA, async prefetch      │
             ▼                                     ▼
   ┌─────────────────────────────────────────────────────────┐
   │                     LLM core                            │
   │  ┌──────────────┐    ┌───────────────────────────────┐  │
   │  │ Prefill pool │ ─► │  Decode pool (H100/B200)      │  │
   │  │ H100 TP=4    │    │  TP=4, paged-attn, MoE expert │  │
   │  │ batch large  │    │  parallel for MoE             │  │
   │  └──────────────┘    └─────────────┬─────────────────┘  │
   │     (PD-disaggregated, KV transfer over NVLink/RDMA)    │
   └────────────────────────┬────────────────────────────────┘
                            │ token stream
                            ▼
                ┌─────────────────────────┐
                │  Output multiplexer     │
                │  - text → SSE stream    │
                │  - audio-tokens → codec │
                │  - speculative cancel   │
                └────────┬────────────────┘
                         ▼
                      user

   Side stores:
   ┌──────────────────────┐  ┌─────────────────────┐  ┌──────────────────┐
   │ Image-feature cache  │  │  KV prefix cache    │  │  Session store   │
   │ Redis + S3 (warm/cold)│  │ Mooncake/MemServe   │  │ user/conv state  │
   │ key: perceptual+sha  │  │ key: token-prefix   │  │                  │
   │       + encoder-ver  │  │  + image-hash list  │  │                  │
   └──────────────────────┘  └─────────────────────┘  └──────────────────┘
```

Key invariants enforced by this topology:

1. **Encoder pool runs on different silicon than decoder pool.** L40S/A100 for vision (~$0.50–$1.50/hr/GPU on cloud) is 2–4× cheaper than H100, and the encoder is not memory-bandwidth-bound, so the cheaper tier loses very little throughput.
2. **Image features and KV transfer over RDMA, never copied through host memory unnecessarily.** NIXL connector or Mooncake.
3. **Audio session manager is a separate path that never goes through image preprocessor.** Real-time audio cannot tolerate generic-router queueing.
4. **Cache key always includes encoder version and decoder version.** Version skew is silent corruption; the system makes it impossible.

---

## 5. Per-Workload Request Lifecycles

### 5.1 Single-image chat (the cheap, common case)

```
t=0    user sends message + 1 image
       │
       ▼
t=2    L7 + router → preprocess pool
       │ image hash computed (sha256 of bytes + perceptual hash)
       ▼
t=10   preprocess: decode JPEG, resize to 448, normalize → 1 tile
       │ check feature cache by (sha, encoder_ver) → MISS
       ▼
t=20   queue at vision encoder, batched with 30 other tiles
t=28   encoder forward → 256 features × 4096-d  (64 MB)
       │ write to feature cache (TTL 10 min, async)
       │ async transfer to LLM host
       ▼
t=29   features land at prefill worker
t=30   prefill: 256 image tokens + ~50 text tokens through 70B LLM
t=80   first text token (TTFT = 80 ms)
t=80–500  decode at ~50 tok/s for ~200 tokens
t=500  done

Total wall time: ~500 ms TTFT 80 ms.
Decoder time dominates (60% of wall) once the image is encoded.
```

### 5.2 Multi-image document analysis (the expensive, bursty case)

```
t=0       user uploads 30-page PDF + question
          │
          ▼
t=50      preprocess: PDF → 30 images (CPU-bound, parallelize on
          16 vCPU per request, ~1.5 ms/page)
          │ for each image: hash, check feature cache
          │   → 8 hits (recurring forms), 22 misses
          │ tile each image: 8 tiles × 22 = 176 new tiles
          ▼
t=100     queue 176 tiles at encoder pool
          │ encoder pool runs them across 8 GPUs in parallel
          │ batched 32 at a time → ~6 batches × 8 ms = 48 ms wall
          │ but with parallelism across 8 GPUs: 2 batches per GPU
          │ → 16 ms wall time (assuming no queueing)
          ▼
t=120     all features ready (8 cached + 22 fresh = 30 image-feature blobs)
          │ async transfer to LLM, pipelined with encoding
          ▼
t=140     prefill begins. ~7,680 image tokens + ~50 text tokens.
          │ prefill on H100 TP=4: ~1.2 ms/1K tokens at batch 1 → 9 ms
          │ but that ignores attention compute; realistic ~150 ms
          ▼
t=300     first text token (TTFT = 300 ms — within document-class SLO)
t=300–2500  decode

Total: ~2.5 s for ~400-token answer. Encoder is 10% of wall, prefill 6%, decode 80%.

Caching wins are dramatic: a follow-up question on the SAME document
re-uses 30/30 image features AND the entire image-prefilled KV span:
  TTFT drops from 300 ms to ~30 ms (just the new question prefill).
```

### 5.3 Real-time audio (the latency-bound case)

```
                Client                                Server
                  │                                     │
audio frames ─────►  WebRTC, 80ms/frame                 │
                  │  ──────────► VAD per frame          │
                  │                  │ (no end-of-turn) │
                  │  audio_in ─────► ► Mimi encode      │
                  │                  │ (streaming, runs on
                  │                  │  audio-encoder GPU
                  │                  │  during user speech)
   user pauses    │                  │
                  │  ◄── VAD detects end-of-turn        │
                  │                  │                  │
                  │                  ▼                  │
                  │            LLM is ALREADY warming up: prefill
                  │            of all-but-last-frame happens during
                  │            user speech (speculative prefill).
                  │            Final frame integrated → first
                  │            audio-out token within ~150 ms of
                  │            end-of-turn.
                  │                  │                  │
                  │                  ▼                  │
                  │  audio_out ◄── Mimi decode (streaming)
                  │  ◄────────  80ms/frame
                  │                  │                  │
   user starts speaking again        │                  │
                  │  ─────► VAD ────►│ duplex interrupt │
                  │                  │  - cancel decode │
                  │                  │  - drop pending  │
                  │                  │    audio frames  │
                  │                  │  - flip to listen│
                  │  audio_in ──────►│  (full-duplex)   │

Latency budget annotation:
  VAD detect end-of-turn       50 ms  (waits for one frame of silence)
  Network in (pipelined)        0 ms  (already arrived)
  Audio encode tail             30 ms  (one frame to finish)
  LLM TTFT (with speculative)  150 ms
  Audio decode first frame      40 ms
  Network out                   50 ms
  Buffer/jitter                 30 ms
  ─────────────────────────────────
  Total                        350 ms
```

The non-obvious win: **speculative prefill during user speech**. The LLM begins prefilling the conversation history + partial audio tokens *while the user is still speaking*, so when VAD fires, only the last ~80–160ms of audio needs to be integrated. This is a Moshi-style trick adapted to a server with prefill-decode disaggregation.

---

## 6. Deep Dives

### 6.1 Encoder/Decoder Disaggregation

The data settles this. ModServe (SoCC'25) shows 3.3–5.5× throughput and 25–41.3% cost win on a 128-GPU cluster vs. monolithic vLLM under production Azure traces. EPD-Serve (Jan 2026) reports 57–69% throughput improvement over PD-disaggregated baselines on identical hardware. vLLM-Omni measures up to 91.4% JCT reduction on any-to-any workloads.

**Why monolithic loses.** A monolithic worker has the encoder and the decoder sharing GPU SMs. When a document request arrives with 80 tiles of encoding work, the encoder consumes the GPU for ~600ms; during that window, the decoder for in-flight chat requests stalls. Tail latency on chat explodes. Conversely, when a long-context decode is running, an arriving image waits in the encoder queue. **Modal contention** is the term. The monolithic worker can't run both at peak efficiency because they need different schedules.

**The disaggregated topology.**

```
[Encoder Worker]                 [Prefill Worker]              [Decode Worker]
  ViT (DP across GPUs)     →       LLM prefill (TP=4)      →     LLM decode (TP=4)
  L40S × 8                          H100 × 8                      H100 × 32
  cost: ~$1/hr/GPU                  cost: ~$4/hr/GPU              cost: ~$4/hr/GPU
       │                                  │                            │
       │ image features                   │ KV cache pages              │
       └──────►──────► NIXL ──────►──────►┘──────►──────► NIXL ──────►─┘
                       (RDMA, async prefetch)        (Mooncake-style)
```

**Sizing.** Encoder pool sized to the burst-p99 of tiles/s (~3,700 in our workload), each encoder GPU sustaining ~150 tiles/s on L40S → ~25 L40S. LLM prefill pool sized to TTFT SLO under peak prefill load. LLM decode pool sized to total token-output throughput.

**Handoff mechanics.** EPD-Serve uses asynchronous feature prefetching: the encoder begins streaming features to the prefill worker as soon as the first batch of tiles completes, overlapping the remaining encode work with the network transfer. For a 30-page document, this hides ~80% of the network transit time inside encoding. The prefill worker holds an "encoder feature staging buffer" addressed by request_id + tile_index.

**The hidden cost — preprocessing.** vLLM issue #27094 measured Qwen2.5-VL-3B preprocessing at 143 ms for a 1024×1024 image, with 75 ms in the HF preprocessor (CPU image normalization). That's CPU-bound, not GPU-bound, and it lives on the encoder host. The encoder pool needs CPU-heavy nodes (or GPU-side preprocessing via DALI/NVENC), or preprocessing dominates the encoder critical path. **Most "encoder is fast" claims ignore this.**

**Rejected: encoder-as-sidecar.** Each LLM worker owns an encoder process on the same host, sharing GPUs. Rejected because: same modal-contention problem as monolithic, just at a finer grain; loses the cheaper-silicon win for encoders; adds operational complexity without benefit.

**Rejected: encoder-as-CPU-process.** Preprocessing yes, encoding no. ViT on CPU is ~50× slower than on a cheap GPU; not viable.

### 6.2 Variable-Request-Cost Scheduling

The scheduling unit at the encoder layer is **the tile, not the request**. A 50-image document is 500 scheduling units, not one. The encoder scheduler batches 32 tiles regardless of which request they came from, and reassembles per-request feature tensors after the batch completes. This is the only way to keep encoder MFU high under heterogeneous traffic.

**Per-request reassembly.** Each tile carries `(request_id, image_idx, tile_idx)` metadata. After the encoder batch returns, a scatter step writes each tile's feature into the correct slot of the request's feature tensor. When all tiles for an image complete, the image_features blob is shipped to the prefill worker.

**Head-of-line blocking.** The pathological case: a 500-tile document arrives, gets queued at the encoder, and a single-image chat tile queues behind it. The chat tile waits 500 × tile-time. ModServe's "modality-aware routing" addresses this by routing tiles to the encoder instance with the **fewest pending image tokens to encode**, not the fewest pending requests. RPS-Serve calls this "sand flowing through pebbles and rocks" — small image requests should not be starved by large ones. Implementation: a priority queue at each encoder where tiles from short-image-count requests get a +ε boost, plus aging to prevent starvation of large requests.

**LLM-side scheduling under heterogeneity.** A 100K-token document prefill cannot share a GPU with chat decode without destroying chat tail latency. Two solutions:

1. **PD disaggregation** (DistServe-style): prefill workers and decode workers are physically separate. A long prefill on a prefill worker doesn't affect decode tail.
2. **Chunked prefill** (vLLM default): break a 100K prefill into 2K-token chunks interleaved with decode steps. This is operationally simpler but caps the document tier's TTFT lower bound (each chunk costs ~50ms).

I'd commit to **PD disaggregation for the document tier** and **chunked prefill for the chat tier**. The audio tier needs its own prefill workers (see §6.4).

**The "user sees nothing for 5 seconds" case.** A 50-image document burns 5+ seconds of encoder before any LLM work. UX response: stream encoding progress (`{"event":"encoding","progress":23/50}`) over SSE/WebSocket, so the client can show a progress bar. This is cheap (a status message per few tiles) and removes the worst UX failure mode. **Engineering responses to UX problems are part of staff work.**

### 6.3 KV Cache for Multi-Modal Tokens

Image tokens occupy KV slots, but their economics differ from text tokens on three axes.

**Axis 1 — cacheability across requests.** Two users asking different questions about the same image can share the image's encoded features and the KV produced from prefilling those features through the LLM. **Two-tier cache:**

- **Tier 1: image-feature cache.** Keyed by `(sha256(image_bytes), encoder_version, tile_policy_version)`. Value: the encoded features (64MB per high-res tile-batch). Cheap to store (~$0.001/GB-month on S3 Glacier for cold tier; in-memory Redis for hot tier with ~1-hour TTL). Hit-rate in production is very high for document workloads (same forms, same templates) and modest for chat (people upload unique photos).
- **Tier 2: KV-prefix cache.** Keyed by the full token prefix hash, which now includes image-content hashes interspersed with text tokens. Value: the KV-cache pages produced by prefilling that prefix through the LLM. Stored on Mooncake-style distributed memory pool (CPU DRAM + NVMe).

**Axis 2 — cache key complexity.** Text-only prefix caching keys on token-hash. Multi-modal must key on (text_token_prefix, image_hashes_in_order, tile_policy, encoder_version, llm_version). This is heavier: any change to encoder or tile policy invalidates all keys for that version pair. **Versioning discipline is non-negotiable** — encoder upgrade rolls out only after the new image-feature cache is warm and the corresponding LLM has been validated against the new feature distribution.

**Axis 3 — eviction cost.** A single high-res image's KV is ~640 MB at 70B-scale FP8, vs. ~4 KB for a typical text turn. Evicting an image-KV entry costs ~640 ms of recompute (encoder + prefill); evicting a text-KV entry costs ~50 ms. Eviction must be cost-aware, not LRU. The right policy is **GreedyDual-Size with cost = recompute_time / size**, so image entries get a strong stay-bonus despite their size.

**Image-token KV size in the mixed pool.**

| Item | Size in 70B FP8 KV |
|---|---:|
| Text token | 80 layers × 8 heads × 128 dim × 1 byte × 2 (k,v) = 160 KB |
| Single-tile image (256 tokens) | 40 MB |
| 8-page document (20K tokens) | 3.2 GB |
| 50-page document (128K tokens) | 20 GB |

A single H100 80GB host running TP=4 has 320 GB of HBM, of which ~150 GB is KV-cache budget after weights. **Two concurrent 50-page documents fill it.** Paged-attention with overflow to CPU DRAM and NVMe is mandatory. Mooncake-style remote memory pool extends this to a cluster-wide cache (papers report 10–100× hit-rate improvement on long-context document workloads).

**Compression.** Image-token KV is more compressible than text-token KV — VL-Cache and ZipVL exploit per-modality sparsity to retain only 10% of image-KV with negligible quality loss. In production, this is a deferred optimization: ship the system without it, instrument it, then enable for the document tier where the savings matter most. **Don't ship vision-token-pruning in V1; the eval cost is non-trivial.**

```
KV cache layout (per LLM decode worker, paged):

  HBM (fast, expensive)
  ┌─────────────────────────────────────────────────────────┐
  │ Active decode requests: KV pages for currently           │
  │ generating tokens (text prompts, recent image tokens)    │
  │ Hot prefix cache: high-frequency document KV             │
  └─────────────────────────────────────────────────────────┘
  CPU DRAM (medium, ~1TB per host)
  ┌─────────────────────────────────────────────────────────┐
  │ Warm KV: recent doc-prefill outputs, recent chat prefixes│
  │ Mooncake remote memory pool participation                │
  └─────────────────────────────────────────────────────────┘
  NVMe (cold, large)
  ┌─────────────────────────────────────────────────────────┐
  │ Cold KV: long-tail document prefills, audit retention    │
  └─────────────────────────────────────────────────────────┘

  Tags: [ENCV=v3 LLMV=70B-v17 TILE=v2 SHA=…]
```

### 6.4 Real-Time Audio Inference

The hardest latency engineering in the system. Three architectural commits drive everything:

**Commit 1 — native speech-to-speech, not pipeline.** Mimi-style codec at 12.5 Hz (80ms frames), 8 codebooks per frame, RVQ. The LLM directly emits audio codebook tokens, interleaved with optional text tokens (Moshi's "inner monologue"). No separate Whisper, no separate TTS.

**Commit 2 — multi-stream / full-duplex from day 1.** The model has two parallel audio streams: user's audio in, system's audio out. They're both autoregressively predicted at the same time-step, conditioned on each other. This is what enables interruption: when the user starts speaking, the user-stream tokens start flowing back as new context, and the system can immediately stop generating its own audio in response.

**Commit 3 — speculative prefill during user speech.** Most of the prefill happens *while the user is still speaking*. As audio frames arrive, they're encoded into Mimi tokens and fed to the LLM as prefill — including past dialogue context. By the time VAD fires end-of-turn, only the final ~1–2 frames need to be integrated. TTFT collapses from "prefill the whole conversation" to "prefill the last 160 ms".

**The 350ms budget revisited.**

```
Stage                              Budget    What can break it
───────────────────────────────────────────────────────────────
Network in (continuous)             pipelined  packet loss, jitter
VAD detect (1 frame post-speech)    80 ms     aggressive setting → false trigger
Mimi encode tail                    30 ms     GPU contention on encoder
LLM TTFT (speculatively prefilled) 150 ms     model not warmed; KV miss; long context
Mimi decode first frame              40 ms     codec model load
Network out                          50 ms     egress bandwidth, geography
─────────────────────────────────  ────────
Total                              350 ms
```

**Failure modes and their engineering responses.**

- *"Model talks over user."* The full-duplex architecture's user-stream injection should preempt this. Engineering: VAD threshold tuned per-tenant; emergency cancel signal from VAD to decode worker that aborts the current generation (within one frame, ~80 ms).
- *"Model silent for 800 ms while thinking."* TTFT exceeded the budget. Triage: was the LLM cold (warm-pool autoscaling failed)? Was prefix-cache miss rate high (versioning issue)? Was the conversation context too long (need summarization/compression)? Each failure has a separate runbook.
- *"Model loops or rambles."* No turn-detection signal. Need an end-of-turn token in the model vocabulary, plus a max-utterance length.
- *"Network jitter eats the budget."* Client-side jitter buffer of 30 ms; server-side, opportunistic transmission of audio frames as they're decoded rather than buffering.

**Audio worker physics.** Audio sessions are stateful and long-lived (~30s–10min). They cannot be load-balanced like stateless chat requests. Each audio session pins to one decode worker for its lifetime. This means:

- Per-worker concurrency cap: ~30–60 concurrent sessions per H100 (KV cache budget; each session has ~10K tokens of context after a few minutes).
- 250 concurrent sessions → 5–10 H100s of dedicated audio decode capacity.
- Session migration on worker drain: serialize KV cache, transfer over RDMA to new worker (~100 ms), resume — but only at turn boundaries, never mid-utterance.

**Rejected: shared decode pool with chat.** Audio sessions evicting chat requests under load creates unpredictable chat tail latency. Audio gets its own pool sized for the audio SLO; the cost is real (steady-state utilization may be lower) but the alternative is broken under load.

### 6.5 Tile / Resolution Strategy

**Auto-tile policy.** Cheap end: chat images at 448×448 → 1 tile → 256 tokens, 8–15 ms encode. Expensive end: document pages at 1792×1792 → 16 tiles → 4096 tokens, 130–250 ms encode. The system inspects image dimensions and content hints (mime type, EXIF, request endpoint) and picks tile count automatically. User can override with a `detail: "low" | "auto" | "high"` parameter (OpenAI-style API).

**Cost-vs-quality knob.** At the API surface, document analysis defaults to high tile count; chat defaults to auto. Internal evals validate that "auto" doesn't degrade chart-OCR or document-QA accuracy beyond a threshold; eval is part of the model release process.

**The 50 MB image case.** Pre-validation: max 20 MB upload, max 8000×8000 pixels. Beyond → pre-downsample with high-quality Lanczos; never reject silently. The downsample target preserves aspect ratio and doesn't exceed the model's max-tile-grid (e.g., InternVL's 40-tile cap = ~2800×2800 effective pixel limit before fidelity loss).

**Document path optimization.** A 50-page PDF is 50 distinct pages, but they share cache potential — same boilerplate (logos, page numbers). Implementation: hash each page's pixel content separately, encode each separately, share feature cache hits. **Per-page cache, not per-document cache.**

### 6.6 Multi-Tenant Isolation

Per-modality rate limits, not a single TPM. A tenant's quota is a tuple:

```
{
  "text_tokens_per_minute":      <int>,
  "image_tiles_per_minute":      <int>,
  "audio_seconds_per_minute":    <int>,
  "document_pages_per_minute":   <int>,
  "encoder_flops_per_minute":    <int>  (derived budget)
}
```

`encoder_flops_per_minute` is the catch-all; a tenant with high-res-only traffic may saturate it before tile-count quotas, and a tenant uploading thumbnails won't.

**Per-pool fair share.** Encoder pool, prefill pool, decode pool, audio pool all have separate fair-share schedulers. A tenant exhausting encoder budget can still issue text-only chat against the prefill+decode pools.

**Priority classes.**

| Class | Use case | Relative priority |
|---|---|---|
| `realtime` | Audio sessions | Highest, dedicated capacity |
| `interactive` | Chat | High, latency-SLO bound |
| `batch` | Document analysis | Medium, throughput-bound |
| `bulk` | Async/background | Low, best-effort |

**Document-burst back-pressure.** A tenant submitting 1000 50-image documents in 30 seconds: detection via per-tenant token-bucket on `image_tiles_per_minute`; first response is queueing; second response is shedding to async queue with delayed result; third is HTTP 429. **Never silently degrade other tenants' chat tail.**

### 6.7 Artifact Versioning Across Encoder/Decoder

A projector-based VLM has three coupled artifacts: vision encoder, MLP projector, LLM. They share a learned embedding space; an encoder swap that moves the embedding distribution breaks the LLM's interpretation. **Version skew is silent corruption** — outputs look reasonable, but quality drops in subtle ways (OCR errors, hallucinated objects).

**Discipline.**

1. Every artifact carries a version tag: `encv3, projv2, llmv17`.
2. Compatibility manifest: only specified `(encv, projv, llmv)` triples are allowed to deploy together.
3. Cache keys include all three versions. Old cache entries are invalid and unreachable.
4. Rollouts are coordinated: encoder + projector + LLM deploy together as a unit, never independently. If only the LLM changes, the encoder/projector still re-deploy (no-op binary-identical) to bump the manifest.

**The mid-request version-skew failure.** A request arrives, encoder pool serves features from `encv3`; meanwhile prefill pool just rolled to `llmv18` which expects `encv4` features. Result: subtly wrong outputs. **Mitigation**: every feature tensor carries a version header; prefill workers reject mismatched versions and trigger a re-encode.

### 6.8 Prompt Caching with Multi-Modal Content

Anthropic and OpenAI both expose paid prompt caching for text. Multi-modal extends this with much larger savings.

**The document-Q&A pattern.** A user uploads a 50-page legal contract and asks 20 questions over an hour. Without caching:

- Per question: 500 tiles encoded (~3 s) + 128K tokens prefilled (~10 s). 13 s TTFT.
- 20 questions × 13 s = 260 s of compute, ~$2 of cost (assuming H100 hour cost ~$3).

With multi-tier caching:

- Q1: 500 tiles encoded + 128K tokens prefilled → cached. 13 s.
- Q2–Q20: feature cache hit + KV prefix hit. Only the new question (~50 tokens) prefills. ~50 ms TTFT each.
- Total: 13 s + 19 × 50 ms = 14 s. Compute cost drops ~20×.

**Cache pricing.** Mirroring Anthropic: cache writes are 25% more expensive than fresh prefill (because of the storage overhead and cache-management burden); cache hits are 90% cheaper. Tuned so users have economic incentive to write the cache for any session ≥ 2 turns.

**Cache key.** `sha256(text_prefix) + ordered_list_of_image_hashes + encv + llmv + tile_policy_v`. Order matters — the same images in different positions don't hit. **Don't try to be clever with order-independent caching for vision; it leaks correctness.**

---

## 7. Failure Modes

| Failure | Detection | Response |
|---|---|---|
| Encoder OOM on giant image | Pre-validation at preprocess | Reject with actionable error, never silently downsample beyond limit |
| Encoded-features transfer mid-request loss | Sequence number on transfer chunks; checksum | Retry from feature cache (if write completed) or re-encode |
| Encoder/LLM version skew | Version header on feature tensor | Reject with re-encode trigger; alert |
| Audio stream disconnect mid-generation | WebSocket close event | Save session KV checkpoint at last turn boundary; client reconnect resumes |
| Adversarial image content (jailbreak embedded in image) | Pre-encode safety classifier; post-encode embedding-space anomaly detection | Block + log + tenant-level rate adjustment |
| Document with poisoned text rendered as image | Same as above; OCR-then-text-classifier as second pass | Same |
| Encoder pool partial failure (1 of N down) | Health check + heartbeat | Re-route; degrade tile policy if needed; alert |
| KV cache hash collision | Cryptographic hash makes this ~impossible; but if mismatch detected via versioned check | Treat as cache miss; recompute |
| Audio worker drain during active session | Worker drain signal | Migrate at next turn boundary; never mid-utterance |
| Mooncake remote memory partition | Heartbeat between nodes | Degrade to local-only KV cache; lower hit rate but no correctness issue |

---

## 8. Observability

Per-modality is mandatory; aggregated metrics hide the failure modes that actually matter.

**Per-stage latency histograms** (p50/p95/p99/p999):

```
preprocess.duration                    by tenant, by image_size_bucket
encoder.queue_depth                    by encoder_pool_id
encoder.tile_compute_duration          by tile_count_per_request
encoder.cache_hit_rate                 by tenant, by image_class (chat|doc)
transfer.encoder_to_prefill_duration   by transfer_path (nvlink|rdma|tcp)
prefill.duration                       by image_token_count_bucket, by text_token_bucket
prefill.cache_hit_rate                 by prefix_class
decode.ttft                            by modality (chat|doc|audio)
decode.tpot                            by modality
audio.end_to_end_turn_latency          (user-stop to first-audio-out)
audio.duplex_interrupt_rate
audio.session_kv_size
```

**Cost / capacity metrics:**

```
encoder_pool.tiles_per_second.actual / .capacity
encoder_pool.gpu_mfu                      (target: 50%+ on L40S)
prefill_pool.tokens_per_second.actual / .capacity
decode_pool.kv_cache_utilization          (% of HBM budget used)
mooncake.remote_kv_hit_rate
feature_cache.hot_size_gb
feature_cache.cold_size_gb
```

**Quality metrics (sampled, offline):**

```
ocr_accuracy_eval.daily                   on held-out doc set
chart_qa_accuracy_eval.daily
audio_transcription_wer.daily
audio_response_rating.streaming_user_feedback
```

**The metric that catches version skew.** `prefill.feature_version_mismatch_rate`. Should be zero. Any non-zero value triggers paging. This is the metric that prevents the silent-corruption failure mode from becoming a quality incident.

---

## 9. Tradeoffs and What Would Change Them

- **If real-time audio drops out of scope**: the latency-budget design relaxes. Audio worker pool, native speech architecture, full-duplex infrastructure all go away. The system simplifies into a vision+text VLM serving stack. Capacity savings: ~5–10 H100s.
- **If video is in scope**: encoder pool sizes by ~10×. Video decoder (frame extraction, key-frame detection, temporal compression) becomes a new pre-stage. KV cache budget per request grows by ~30× (a 30-second clip at 1 fps × 8 tiles = 240 tiles, ~60K image tokens). The current architecture extends because we already have EPD; we'd add a video preprocessor pool and a video-encoder variant. **The fact that EPD generalizes is the future-extension value.**
- **If the model becomes unified-native (GPT-4o-style)**: encoder disaggregation goes away (no separate encoder artifact). All serving is on the LLM tier. We lose the cheap-silicon win for encoding (~30% cost increase). We gain operational simplicity (one artifact to version). Most frontier labs have not chosen this path because the cost trade is bad.
- **If the user mix shifts to 80% document analysis**: encoder pool is the bottleneck, not decode. We'd add encoder capacity and revisit feature-cache size (more hot storage). Decode pool may shrink relatively.
- **If audio mix grows to 30%**: dedicated audio pool grows substantially. Audio prefix-cache becomes a major optimization. We'd consider a smaller / faster LLM variant for the audio tier (MoE with 4-of-32 expert activation, or distilled 30B variant) to fit the TTFT budget.
- **If KV cache compression (VL-Cache, ZipVL) reaches production maturity**: image-token KV drops 5–10×. This dramatically eases the document-tier capacity problem. Worth tracking; not a V1 commit.

---

## 10. What I Push Back On

1. **"5K concurrent users" is meaningless without modality mix.** I committed to 70/25/5 and built capacity to it; if the real mix is 50/40/10, the encoder pool is ~50% under-sized and the audio pool is 2× under-sized.
2. **"Vision + text + optional audio" implies audio is bolted on.** It's not. Real-time audio at frontier latency requires a different model architecture, a different network protocol (WebRTC/WebSocket, not HTTP), a different scheduler (session-pinned, not request-routed), and a different failure model. If the team treats audio as "the same system plus Whisper," the audio product will fail SLO.
3. **Co-locating encoder and decoder is the wrong default at scale.** ModServe and EPD-Serve numbers settle this: 3–5× throughput, 25–41% cost. The only reason to monolith is operational simplicity at small scale (≤2 GPUs). At 5K users we're well past that.
4. **Real-time audio and document analysis must not share the same SLO class.** A 5-second p99 is fine for document analysis and broken for audio. Sharing pools or schedulers between them is malpractice. Separate capacity, separate alerting, separate runbooks.
5. **"GPT-4o class" is a fuzzy spec.** GPT-4o reports 320ms average audio. Moshi reports 200ms. Anthropic's vision is mostly turn-based (no real-time audio at the time of writing). Each implies a different system. I'm targeting Moshi-class audio + Pixtral-class vision; if the spec means something else, the design has to revisit.
6. **The encoder is cheap myth.** Encoder GPU compute is cheap; encoder *preprocessing* (CPU image normalization, PDF rendering, audio frame buffering) is not. Production traces (vLLM #27094) show preprocessing eating 50%+ of TTFT. The system has to invest in NVENC, DALI, GPU-side preprocessing, or it won't hit SLOs.
7. **One unified rate limit is wrong.** A tenant uploading thumbnails and a tenant uploading 1000-page PDFs cost orders of magnitude differently against the encoder pool. TPM is the right limit for text and the wrong one for vision.