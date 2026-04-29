---
title: LLM Inference Serving
description: LLM Inference Serving
---

> Target: Anthropic Inference Engineer screen, 40 min slot, "Design a high-throughput, low-latency LLM serving system." Equivalent to designing vLLM from scratch and defending every choice.

---

## 1. The 2-Minute Mental Skeleton (draw this first)

Draw this within 90 seconds of hearing the prompt. It survives any twist (multimodal, MoE, stateful, RAG, agents) because the boxes don't change — only what fills them.

```
                    ┌───────────────────────────────────────────────┐
   client ──HTTP──▶ │  FRONTEND (LB, auth, rate-limit, streaming)   │
                    └──────────────┬────────────────────────────────┘
                                   │ routed on (model, tenant, prefix-hash)
                    ┌──────────────▼────────────────────────────────┐
                    │  TOKENIZER  (CPU sidecar, batched, async)     │
                    └──────────────┬────────────────────────────────┘
                                   │ token IDs + sampling params
                    ┌──────────────▼────────────────────────────────┐
                    │  REQUEST QUEUE / ADMISSION CONTROL            │
                    │  - per-tenant fairness                        │
                    │  - KV-budget admission                        │
                    │  - SLO-aware preemption                       │
                    └──────────────┬────────────────────────────────┘
                                   │
                    ┌──────────────▼────────────────────────────────┐
                    │  SCHEDULER (iteration-level / continuous)     │
                    │  - prefill vs decode mixing (chunked prefill) │
                    │  - block-table allocator (paged KV)           │
                    │  - prefix cache lookup (radix / hash)         │
                    └──────────────┬────────────────────────────────┘
                                   │ batch = (prefill chunks ∪ decodes)
                    ┌──────────────▼────────────────────────────────┐
                    │  EXECUTOR  — TP=8 worker group on 1 node      │
                    │  ┌──────┬──────┬──────┬──────┐                │
                    │  │ GPU0 │ GPU1 │ ...  │ GPU7 │  NVLink        │
                    │  └──────┴──────┴──────┴──────┘                │
                    │  - CUDA graph cache (decode)                  │
                    │  - FlashAttention / PagedAttention kernels    │
                    │  - all-reduce per layer (NVL only)            │
                    └──────────────┬────────────────────────────────┘
                                   │ logits → sampler → token IDs
                    ┌──────────────▼────────────────────────────────┐
                    │  DETOKENIZER (CPU) → STREAM CHUNKER           │
                    └──────────────┬────────────────────────────────┘
                                   │ SSE / WebSocket
                                   ▼
                                client
```

Three things this diagram quietly commits to and which you should call out aloud:

1. **The scheduler is the product.** Everything above and below it is plumbing. Continuous batching with chunked prefill is the default.
2. **One TP group = one node.** TP across nodes is a forced move, never a chosen one.
3. **KV cache is the binding resource.** Memory is admission control; FLOPs are not the bottleneck for decode.

**[STAFF+]** Twists that only change a single box: MoE → executor adds expert-parallel and a router; multimodal → tokenizer becomes "encoder pipeline" with a vision tower; stateful (sessions) → prefix cache becomes durable + replica-affinity routing; agents/tool use → frontend loops, not a new system.

---

## 2. Clarifying Questions (5-min box, then move on)

Ask these in the first 2 minutes. State assumptions for the rest and proceed.

**Ask:**
1. Single model or multi-model fleet? (Changes routing, weight loading, MIG.)
2. Open-weight class (Llama-70B-ish) or unspecified frontier model? (Affects KV math.)
3. SLO targets: TTFT and ITL percentiles, or just throughput?
4. Long context support — what's the max? (1K, 32K, 128K, 1M change architecture.)
5. Streaming required? (Almost always yes; affects API and detokenizer.)
6. Tenancy model — public API, internal, or per-customer isolation?
7. Quantization on the table? (FP8/INT8/MXFP4 changes everything.)

**Assume (state aloud, proceed):**
- 70B-class dense transformer, GQA with 8 KV heads, 80 layers, hidden=8192, head_dim=128.
- H100 SXM 80GB nodes, 8-GPU NVLink (900 GB/s aggregate), 400 Gb/s IB inter-node.
- Workload mix: median prompt 500, p99 8K, max 128K; median gen 200, p99 2K.
- SLO: TTFT p99 < 1s, ITL p99 < 50 ms, optimize $/M-tokens subject to those.
- FP8 weights (E4M3 with per-channel scaling), BF16 KV cache.
- Streaming via SSE. Stateless API (no server-side conversation memory; client resends).

**[STAFF+] signal:** explicitly box clarifying questions and move on. Don't burn 10 minutes; the interviewer wants to see design under uncertainty.

---

## 3. Capacity Math

Show the work. Be explicit about the model card you're computing against — this is where pattern-matchers fall apart.

### 3.1 Model card (assumed)

| Param | Value |
|---|---|
| Layers `L` | 80 |
| Hidden `d_model` | 8192 |
| Heads (Q) | 64 |
| KV heads (GQA) | 8 |
| Head dim | 128 |
| FFN intermediate | 28672 (SwiGLU, 3.5×) |
| Vocab | 128K |
| Total params | ~70B |

### 3.2 Weight memory

- FP16: 70B × 2 B = **140 GB**
- FP8 (E4M3): 70B × 1 B = **70 GB**
- MXFP4 (4.25 bpw block-scaled): ~37 GB

With **TP=8** on one node:
- FP16 per-GPU weights: 17.5 GB
- FP8 per-GPU weights: 8.75 GB

### 3.3 KV-cache bytes per token

```
KV_bytes_per_token = 2 (K,V) × L × H_kv × d_head × dtype_bytes
                   = 2 × 80 × 8 × 128 × 2     (BF16)
                   = 327,680 B = 320 KiB/token
```

With TP=8 sharded over the 8 KV heads, each GPU stores **40 KiB/token**.

For sanity: a 128K-context request alone is 128K × 320 KiB = **40 GiB** of KV state across the TP group, **5 GiB per GPU**. One long request can swallow most of the budget.

### 3.4 HBM budget on 1× H100 80GB (TP=8, FP8 weights)

```
80.0 GB total
- 8.75 GB  weights (FP8, 1/8 of 70B)
- 1.5  GB  CUDA workspace (cuBLAS, NCCL buffers, kernels)
- 2.0  GB  activation buffers (peak prefill micro-batch)
- 1.5  GB  paged-allocator overhead, fragmentation, safety margin
─────────────
≈ 66.25 GB usable for KV cache
```

Per-GPU at 40 KiB/token → **~1.7M tokens of KV per GPU**. Same number across the TP group (each GPU holds a 1/8 shard of the same KV state), so **system KV capacity ≈ 1.7M tokens per replica**.

H200 (141 GB): same arithmetic, **~3.3M tokens per replica**. This is why H200 is a serving-disproportionate upgrade — bandwidth went up 1.4×, capacity went up ~2×.

### 3.5 Arithmetic intensity & roofline

H100 SXM peak: ~989 TFLOPS BF16 dense (~1979 with FP8), HBM bandwidth 3.35 TB/s.

- Compute:bandwidth ratio (BF16): **295 FLOP/B**
- Compute:bandwidth ratio (FP8): **591 FLOP/B**

A kernel needs arithmetic intensity (AI) above this ratio to be compute-bound; below, it's bandwidth-bound.

**Prefill** (sequence length `S`, batch `B`):
- Linear layers: AI ≈ `B·S` (matmul, large GEMM)
- Attention: AI ≈ `S` (FlashAttention reuses KV in SRAM)
- For S ≥ 512, B ≥ 1 → **compute-bound**

**Decode** (1 token, batch `B`, KV length `K`):
- Linear layers: each token reads all weights from HBM. AI ≈ `B` (reuses weights across batch).
- Attention: each token reads its own KV. AI ≈ 1, regardless of B.
- For typical B=64–512 → **bandwidth-bound on weights for linear layers, bandwidth-bound on KV for attention**

**[STAFF+] consequence:** decode latency = max(weights_read, KV_read) / HBM_BW. To hide weight-read latency, batch larger. To hide KV-read latency, you can't — KV grows linearly with `B × K` and there's no reuse across requests. This is why attention is the throughput ceiling for long-context decode, not matmul.

### 3.6 Per-GPU decode throughput estimate

Decode is bandwidth-bound on weights for the matmul portion.

Per token, per TP group (FP8 weights, batch ≥ 1):
- Weight bytes touched (system-wide): 70 GB
- Per GPU (TP=8): 8.75 GB
- At 3.35 TB/s: 8.75 / 3350 ≈ **2.6 ms per token (single-stream lower bound)**

Add attention (KV read per token, per request):
- Per request, K=2K context, BF16 KV: 2K × 320 KiB / 8 (TP) = 80 MB per GPU per token
- At 3.35 TB/s: **24 µs per token per request for attention KV read**

For B=128 concurrent decodes at K=2K: weights amortize across batch (still 2.6 ms), KV reads sum: 128 × 24 µs ≈ **3 ms**. Total ≈ 5–6 ms/token → **160–200 tok/s/request** at this batch size, ~25K tok/s aggregate per node.

Compare batch=1: same 2.6 ms weight read, attention negligible → ~380 tok/s. Tells you why nobody runs B=1 except for ITL-extreme requests.

**Throughput ceiling intuition:** a node does ~25K–40K decode tokens/sec depending on context length distribution. Long contexts crater this fast because attention KV reads stop amortizing across batch.

### 3.7 MoE delta (forward reference)

For a Mixture-of-Experts model (e.g., DeepSeek-V3-class, ~37B active of 671B):
- Weight memory dominated by **all** experts (671 GB FP8) — needs expert parallelism (EP) across nodes, no longer TP=8 single-node.
- Decode bandwidth read = active params only (37B → 37 GB FP8) → **decode is faster** than dense 70B.
- All-to-all in expert dispatch becomes the new latency hotspot. NVL or NVLink-Switch + IB rail-optimized.
- Capacity story flips: weights need a node, KV cache needs less per token (fewer attention heads per token's path).

---

## 4. Core Mechanisms (the meat — go deep)

### 4.a Continuous batching (iteration-level scheduling)

**Static batching** waits for a full batch, runs it to completion. **Request-level dynamic batching** adds new requests at request boundaries. **Continuous batching** rebuilds the batch at every forward pass — completed sequences leave, new ones join, decodes for in-flight requests continue.

```
time →
            t0       t1       t2       t3       t4       t5       t6
Req A  [P P P P D D D D D D D D]                                       (10 tokens)
Req B          [P P D D D D D D D D D D]                              (12 tokens, joins t1)
Req C                  [P P P P P D D]                                 (joins t2)
Req D                            [P P D D D D D D D D D D D D D D]    (joins t4)

           ───────  iteration boundaries (1 fwd pass each) ───────
batch@t2:  A.D, B.D, C.P                       <-- mixed prefill+decode!
batch@t3:  A.D, B.D, C.D
batch@t4:  A.D, B.D, C.D, D.P
batch@t5:  A.D, B.D, D.D                       <-- C finished
```

Why it wins: GPU is busy on every iteration, no padding to longest sequence, no head-of-line blocking by long generators.

**Cost:** each iteration must rebuild attention masks, KV pointers, position IDs. PagedAttention makes this cheap by indexing via block tables — no contiguous KV concatenation.

**[STAFF+] subtlety:** the "batch" Tensor is variable in shape. Linear layers see a flat token batch (sum of all queries this step); attention sees per-request KV ranges via block tables. FlashAttention-2/3 with variable-length packed inputs (`flash_attn_varlen_*`) is what makes this efficient.

### 4.b Paged KV cache

KV cache is stored in fixed-size **blocks** (pages), e.g. 16 tokens per block. Each request has a **block table**: a list of physical block IDs in HBM that hold its logical KV stream.

```
Logical view (request):  tok0..15 | tok16..31 | tok32..47 | ...
Block table:             [ #42  ] [   #17   ] [   #91   ] ...
HBM blocks (shared):     [#0][#1]...[#17]...[#42]...[#91]...
```

**Page size choice:** 16 tokens dominates industry practice (vLLM default). Smaller (8) = less internal fragmentation, more block-table overhead, more attention kernel launches; larger (32) = waste on short generations. 16 is the local optimum for the realistic workload mix.

**Fragmentation:** zero external fragmentation by construction (all blocks same size). Internal fragmentation = `(block_size − 1)/2` tokens per request average ≈ 7.5 tokens. Negligible vs the 100s–1000s of decode tokens per request.

**Copy-on-write** for parallel sampling (n>1) and beam search: shared prefix blocks are reference-counted; on the first divergent token, the block is copied. Reduces memory by `n×` for the prompt portion.

```
n=4 sampling, prompt=500 tokens, gen=200 tokens
naive:   4 × 700 = 2800 token-slots
CoW:     500 + 4 × 200 = 1300 token-slots (54% less)
```

**[STAFF+]** PagedAttention costs ~3–7% latency vs contiguous in microbenchmarks (extra indirection in attention kernel) but enables 2–4× higher batch sizes by eliminating reservation-based over-allocation. Net throughput is overwhelmingly positive.

### 4.c Prefill vs decode scheduling — chunked prefill

The interference problem:

```
Without chunked prefill (naive priority):
─────────────────────────────────────────────────────
iter:    | decode batch (B=128)  |  PREFILL 8K       |  decode...
GPU:     |  8 ms                 |  450 ms           |  8 ms
─────────────────────────────────────────────────────
                                  ↑
                 every in-flight decoder stalls 450 ms
                 → ITL p99 explodes
```

A single 8K prefill flushes ITL latency for every decoder in the batch. Naive "FIFO + admit when free" is wrong.

**Chunked prefill** (Sarathi-Serve): split a long prefill into chunks of `C` tokens (e.g. 512–2048) and mix each chunk into a decode batch.

```
With chunked prefill (C=1024):
─────────────────────────────────────────────────────
iter:    | dec(128) + pf_chunk(1024) | dec(128) + pf_chunk(1024) | ...
GPU:     |  ~16 ms                    |  ~16 ms                   |
─────────────────────────────────────────────────────
                  ↑
   ITL ~doubles vs pure decode, but never spikes; TTFT degrades
   linearly with chunk count, but predictably.
```

Choose `C` to keep iteration latency below ITL SLO. With ITL p99 = 50 ms target and pure-decode iter ≈ 8 ms, you have ~40 ms slack. At ~2 GFLOPs/token prefill on H100 ≈ 1024 tokens of prefill ≈ 8–12 ms additional → C=1024 fits.

**Alternative: prefill/decode disaggregation** (DistServe, Splitwise, Mooncake). Separate pools for prefill and decode, KV cache transferred over NVL/IB between pools. Wins when prompt/gen ratio is asymmetric or when SLOs are very tight on TTFT and ITL simultaneously. Costs: KV transfer bandwidth, more complex orchestration, replica-locality routing. **Default to chunked prefill; recommend disaggregation as the next architectural step at scale** (>10K req/s, frontier-lab surface).

### 4.d Tensor parallelism for serving

Per-layer compute: column-parallel QKV → attention → row-parallel output → all-reduce → column-parallel FFN-up → row-parallel FFN-down → all-reduce. **Two all-reduces per transformer layer** on the critical path of every forward pass.

For 80 layers × 2 = 160 all-reduces per token in decode.

**Latency budget on NVL (intra-node, NVSwitch H100 = 900 GB/s aggregate, ~450 GB/s per link):**

Per all-reduce, payload = `B × d_model × dtype` = e.g. 128 × 8192 × 2 B = 2 MiB.
Ring all-reduce on 8 GPUs: `2 × (N-1)/N × payload / BW` = 1.75 × 2 MiB / 450 GB/s ≈ **8 µs**.
Per-token total: 160 × 8 µs ≈ **1.3 ms** comm overhead. Acceptable inside a 5 ms decode step.

**Across nodes (400 Gb/s = 50 GB/s IB per NIC):** all-reduce becomes **~70 µs**, total **~11 ms/token**. **Two-thirds of the decode budget gone to comm.** Absent fancy overlap (compute-comm pipelining, hierarchical AR), TP-across-nodes is a non-starter for decode latency.

```
Decision tree:
- Model fits in 1 node weights-wise (≤8×80GB at chosen dtype)?
   → TP=8 single node. Done.
- Doesn't fit?
   → TP=8 within node + Pipeline Parallel across nodes (rare)
   → OR Expert Parallel (MoE) + smaller TP
   → OR upgrade to H200/B200, push problem one rung up
   → AVOID TP=16 across nodes for decode
```

**[STAFF+]** Sequence parallelism (SP) with TP — split LayerNorm and dropout across the sequence dim within the TP group. Saves activation memory at no comm cost. Unrelated to context parallelism (CP), which is across-node sequence sharding for very long contexts.

### 4.e Pipeline parallelism for serving

Pipeline parallel splits **layers** across devices. In training, micro-batches keep the pipeline full. In serving:

- **Decode is intrinsically sequential** (1 token at a time per request). Pipeline stages are starved unless you have enough concurrent requests to fill them, **and** you tolerate the bubble.
- **Throughput-leaning serving with relaxed ITL:** PP can be acceptable. Each request sees `pipeline_depth × per_stage_latency` ITL.
- **Latency-leaning serving:** PP is wrong. Use TP.

Empirically: PP shows up in serving as a **last resort when a model is too big for one node and TP=16-across-nodes is worse**, or in high-throughput batch inference where ITL is irrelevant.

**Recommended:** Don't use PP unless you can defend it from a specific TTFT/ITL number that breaks otherwise.

### 4.f CUDA graphs for decode

Decode is a tight Python/host loop with hundreds of small kernel launches per layer (matmul, all-reduce, attention, RMS norm, residual, sampling). CPU launch overhead (~5–10 µs/kernel) × 800+ kernels per token = **6–8 ms of pure host overhead** — most of the decode budget.

```
Without CUDA graphs:
host: [launch][launch][launch]...   <- 8 ms host time
GPU:        [k1][k2][k3]...         <- 4 ms compute
total: max(8, 4) = 8 ms (host-bound!)

With CUDA graphs:
host: [replay]                      <- 50 µs
GPU:        [graph][graph]...       <- 4 ms
total: ~4 ms (compute-bound, as it should be)
```

Capture graphs once per **batch shape** (B, padded sequence length) and cache them. Replay on hot path.

**Why prefill doesn't benefit:** prefill kernels are large GEMMs and a single FlashAttention call. Host overhead is <1% of GPU time. Capture cost (~10s of ms per shape) doesn't amortize for a one-shot prefill.

**Graph cache management:**
- Bucket batch sizes (1, 2, 4, 8, 16, 32, 48, 64, 96, 128, 192, 256...). Pad up at runtime.
- Keep ~20–30 hot graphs resident; evict LRU.
- Capture cost: 10–50 ms per graph. Pre-warm at server start, **never** capture on first hot request.

**[STAFF+]** Graphs interact badly with paged KV. Solution: capture the graph against block-table tensor *pointers*, not values; update block tables in-place each step. PagedAttention kernels are written to take block tables as inputs, so the graph topology is stable.

### 4.g Admission control, queueing, preemption

The KV cache is a finite resource. When a new request arrives or an existing one needs another block:

```
Decision matrix when KV is full:
                      │ low TTFT-slack req │ high TTFT-slack req │
─────────────────────┼─────────────────────┼─────────────────────┤
  reject (429)       │  never              │  rare (last resort) │
  queue (delay)      │  no (busts SLO)     │  yes (default)      │
  preempt + recompute│  yes (cheap reqs)   │  yes if needed      │
  preempt + swap     │  yes (long reqs)    │  yes if disk allows │
```

Two preemption strategies:
- **Recompute:** drop KV, re-prefill on resumption. O(`prefill_time`) penalty. Good when prefill is short.
- **Swap to host RAM** (vLLM does this): copy KV blocks to host (PCIe at ~25 GB/s in, ~25 GB/s out). 1 GiB swap ≈ 40 ms. Good for long-context requests where re-prefill would cost 100s of ms.

Backpressure to the LB on queue depth. Per-tenant token-bucket rate limits at the frontend, not the scheduler — scheduler should never see traffic it cannot serve given the SLO.

**[STAFF+] cascade signal:** preemption can deadlock if the freed blocks aren't enough for the next admit. Always preempt to a known **target free pool size**, not "until I have one more block." Watermark-based admission (high/low watermarks like a TCP buffer).

---

## 5. End-to-End Request Lifecycle

```
Wall clock for a "median" request: prompt=500, gen=200, p50 path
─────────────────────────────────────────────────────────────────────────
0ms   ─ Client opens HTTPS, sends JSON body
2ms   ─ Frontend (CPU): TLS, auth, rate-limit check, route
3ms   ─ Tokenizer (CPU sidecar): 500 chars → ~125 tokens, batched w/ peers
5ms   ─ Enqueue to scheduler (memory queue, no IPC)
6ms   ─ Scheduler picks request: prefix-cache hit on 80 tokens (system prompt)
        → only 45 fresh prefill tokens
       ─ Allocate block table: ⌈125/16⌉ = 8 blocks
8ms   ─ Prefill iteration (mixed with ongoing decodes)
       ─ GPU prefill on 45 tokens: ~6 ms (compute-bound, amortized in chunk)
14ms  ─ First token sampled, written to streaming queue, flushed to client
       ─── TTFT achieved: ~14 ms (well under 1s SLO) ────────────────────
14ms  ─ Decode loop begins (iteration-level batching)
       ─ Each iteration: ~5 ms (B=128 active decoders, TP=8 H100)
       ─ Token streamed to client every iteration
       ─ Detokenizer (CPU): incremental BPE merge, handles partial UTF-8
1014ms─ 200th token sampled, EOS or max_tokens
       ─── ITL p50: 5 ms (SLO 50 ms p99) ─────────────────────────────────
1015ms─ KV blocks freed back to allocator
       ─ Final SSE event flushed, connection closed

Where parallelism opportunities live:
  - Tokenization: batched across requests (CPU thread pool)
  - Prefill: multiple requests' prefills chunked + interleaved with decodes
  - Decode: continuous batching of all in-flight requests
  - Detokenization: per-request, on CPU, overlapped with next GPU iter
  - Streaming: async I/O, never blocks the GPU step
```

**[STAFF+] gotcha:** detokenization is **stateful** (BPE merges depend on history) and must be per-request. Don't try to batch it on GPU. Don't naive-decode token-by-token either — incremental decode with byte buffer for partial multi-byte chars.

---

## 6. Memory Layout on a Single GPU

```
H100 SXM 80GB, TP=8, FP8 weights, BF16 KV
═══════════════════════════════════════════════════════════════
         ▲ 80.0 GB
         │
HBM      │ ┌─────────────────────────────────────┐
addr     │ │ CUDA workspace + NCCL buffers   1.5 │
         │ ├─────────────────────────────────────┤
 high    │ │ activation buffers (peak)       2.0 │
         │ ├─────────────────────────────────────┤
         │ │ safety / fragmentation margin   1.5 │
         │ ├─────────────────────────────────────┤
         │ │                                     │
         │ │      KV CACHE (paged blocks)        │
         │ │           66.25 GB                  │
         │ │   ≈ 1.7M tokens of KV (per GPU      │
         │ │     and per TP-replica system-wide) │
         │ │                                     │
         │ ├─────────────────────────────────────┤
         │ │ FP8 weights shard (1/8 of 70B)  8.75│
 low     │ └─────────────────────────────────────┘
         │ 0
         ▼
─────────────────────────────────────────────────────────────────

Same node, FP16 weights (vs FP8):
  weights 17.5 GB instead of 8.75 GB → KV shrinks to ~57 GB
  → ~1.45M tokens KV (15% less capacity, 0% extra compute)
  Net: FP8 buys ~17% effective serving capacity FOR FREE

H200 SXM 141GB, TP=8, FP8:
  weights        8.75 GB
  workspace+act  3.5  GB
  safety         2.0  GB
  KV cache     126.75 GB → ~3.3M tokens KV per replica
  → 2× capacity vs H100 at marginally higher cost
  → H200 is the obvious pick for long-context serving today
```

---

## 7. Multi-Node Topology

```
─────────────────────────────────────────────────────────
TIER 0 — Single replica (most requests live here)

  Node A: 8× H100, NVSwitch (900 GB/s aggregate)
  ┌──────────────────────────────────────────┐
  │  GPU0  GPU1  GPU2  GPU3                  │
  │   │     │     │     │      TP group      │
  │  GPU4  GPU5  GPU6  GPU7    (NVL only)    │
  └──────────────────────────────────────────┘
                       ▲
                     NVL only, no IB on critical path

─────────────────────────────────────────────────────────
TIER 1 — DP replication for throughput

  ┌─Replica 1─┐  ┌─Replica 2─┐  ┌─Replica N─┐
  │  TP=8     │  │  TP=8     │  │  TP=8     │
  └───────────┘  └───────────┘  └───────────┘
        ▲             ▲              ▲
        └──────┬──────┴──────────────┘
               │  Load balancer
               │  routes on:
               │   - prefix-hash (locality / cache hit)
               │   - tenant (fairness, isolation)
               │   - replica load (queue depth)
        ┌──────┴──────┐
        │  frontend   │
        └─────────────┘

─────────────────────────────────────────────────────────
TIER 2 — Forced multi-node (rare for 70B; common for 400B+)

  Use ONE of:
    a) Expert Parallel (MoE only): EP across nodes via IB
    b) Pipeline Parallel: PP=2, TP=8 within each PP stage
    c) Tensor Parallel TP=16 across nodes
       (latency cost is severe; only for niche batch jobs)

  All-to-all collectives now traverse IB (~50 GB/s per NIC).
  Rail-optimized topology required: each GPU's NIC pinned to
  a switch rail, all-to-all uses NIC-direct (GPUDirect RDMA).

─────────────────────────────────────────────────────────
KV-cache locality (DP scaling)

  Prefix cache is per-replica. Routing strategy:
  - Hash request prefix prefix → consistent-hash to replica
  - Bounded migration on rebalance (rendezvous hash)
  - Fall back to least-loaded if affinity replica saturated

  [STAFF+] At fleet scale, a *global* prefix cache (Mooncake-
  style, KV stored in a tiered RAM/SSD pool, fetched over RDMA)
  beats per-replica caches when prompt reuse is high. Cost: KV
  fetch latency on cache hit (1 GB at 50 GB/s = 20 ms). Worth it
  if it avoids a 500 ms re-prefill.
```

---

## 8. Failure Modes

| # | Failure | Detection | Mitigation | Recovery time |
|---|---|---|---|---|
| 1 | GPU hardware failure mid-request | NCCL timeout, NVML ECC error | Drain TP group, restart workers from checkpointed weights, re-route in-flight to peer replica | 30–120 s replica down; per-request: rejected, client retries |
| 2 | OOM under load spike | Allocator returns null on KV block req | Watermark-based admission already prevents; if hit, preempt longest-running decoder, swap to host | Request: +40–200 ms swap penalty; system: graceful |
| 3 | Slow node / straggler in TP group | Per-rank step-time histogram, NCCL probe | Quarantine replica, drain, replace; do **not** continue serving with a 2× slower GPU | Drain in 5–30 s, replace in minutes |
| 4 | KV-cache exhaustion cascade | Watermark crossed, queue depth growing | Backpressure to LB (drop probability, 503 retry-after); preempt low-priority requests; auto-scale replicas | Seconds (if scaling provisioned); else degraded SLO until traffic ebbs |
| 5 | Client disconnect mid-generation | Async write returns broken pipe | Cancel request immediately, free KV blocks, do **not** finish generation (waste) | Immediate; reclaims ~10s of MB |
| 6 | Model weight corruption (silent) | Periodic weight checksum, output sanity probes | Halt replica, reload weights, alert | Minutes; data plane unaffected if multi-replica |
| 7 | Network partition between TP nodes | NCCL collective hang, watchdog | Force-kill TP group, restart; **do not allow TP across nodes for decode** in the first place | Why this is a tier-0 concern argues against multi-node TP |
| 8 | Tokenizer service crash | Health probe, IPC error | Local in-process fallback tokenizer; sidecar pattern with N+1 redundancy | <1 s |

### 8.1 Cascade analysis [STAFF+]

```
Load spike
  └─> KV cache fills
       └─> Admission delays new requests
            └─> Frontend queue backs up
                 └─> Client timeouts; SOME clients retry
                      └─> Effective load 1.5–2× original
                           └─> KV pressure worsens, preemption rate rises
                                └─> Preempted reqs re-prefill
                                     └─> Prefill steals decode time
                                          └─> ITL spikes for ALL in-flight
                                               └─> SLO breach across the board
                                                    └─> Auto-scaler triggers
                                                         └─> New replicas cold-start (~60 s of weight loading + graph capture)
                                                              └─> By the time they're online, original spike receded
                                                                   └─> Now 2× over-provisioned and clients have given up
```

**Mitigations:** load-shed at the frontend (return 429 with retry-after) **before** the cascade starts. Pre-warm replicas on traffic-rate derivative, not absolute level. Cap retry storms with jittered backoff in the SDK.

---

## 9. Trade-offs (state choice + why, not neutral comparison)

| Trade-off | Choice | Alternative | Why |
|---|---|---|---|
| Continuous vs static batching | **Continuous** | Static | 2–5× higher GPU utilization; static only acceptable in offline batch |
| Paged vs contiguous KV | **Paged (16-tok blocks)** | Contiguous reservation | 2–4× effective KV capacity at 3–7% kernel overhead. Contiguous wastes ~50% on overprovisioning to handle p99 lengths. |
| TP=8 single-node vs TP=16 | **TP=8** | TP=16 across nodes | All-reduce on IB destroys decode latency. Use replica DP for throughput. |
| Chunked prefill vs disaggregation | **Chunked first, disagg at scale** | Disagg from day one | Chunked prefill solves 90% of the interference problem with one moving part. Disagg adds KV-transfer plane and replica-affinity routing. Worth it at frontier-lab scale (>10K req/s) and tight SLOs. |
| FP8 vs FP16 weights | **FP8 (E4M3, per-channel scaling)** | FP16 | 2× weight memory, 2× decode throughput (bandwidth-bound), <1% quality on most evals. Per-channel JIT scaling beats per-tensor on outlier-prone layers. |
| Prefix caching vs none | **Always on, tiered** | Never | System prompts and few-shot examples give 30–80% token reduction in real workloads. Cost: hash lookup + block-table assembly. Memory cost: tunable (LRU evict). |
| In-process tokenizer vs sidecar | **Sidecar** | In-process Python | Tokenization in the request hot loop blocks the event loop; sidecar with batched dispatch isolates CPU from the scheduler. In-process Rust tokenizer (HF tokenizers) is acceptable for low-throughput. |
| Streaming vs non-streaming API | **Streaming default, both supported** | Non-streaming only | Perceived TTFT matters more than TTLT for chat; streaming makes p50 indistinguishable. Non-streaming for tool-call interior steps where the next stage parses JSON anyway. |
| BF16 vs FP8 KV cache | **BF16 default, FP8 KV optional** | FP8 KV always | FP8 KV = 2× KV capacity, but quality drops more than FP8 weights — KV outliers are heavy. Worth it for 100K+ context where capacity is binding. |
| Greedy vs speculative decoding | **Speculative for latency-critical, greedy for throughput-critical** | One-or-the-other | Spec dec gives 2–3× ITL improvement at fixed batch but lowers GPU efficiency at high batch. Toggle per route. |

---

## 10. Cost ($/M-tokens)

Reference numbers, on-demand H100 SXM cloud price ~$3.50/GPU-hr (3-yr reserved closer to $1.50; assume $2.50 blended for serving).

```
Per-node cost: 8 × $2.50 = $20/hour = $0.0056/sec
Per-node aggregate decode throughput: ~30K tok/s mixed workload (50% prefill, 50% decode budget on chunked-prefill ratio)
  → 30K tok/s × 86400s = 2.6B tok/day per node
  → cost per M tokens output: $20 × 24 / 2600 = $0.18 / M-tok output

That's the floor. Reality:
  - prompt/output ratio of 5:1 means most tokens are prefill (~free relative to decode budget)
  - prefix caching, in real workloads with system prompts, drops effective prefill cost 30–80%
  - quoted public API prices ($3–15 / M output tokens) bake in margin, fault tolerance, and the weight licensing.
```

### How each lever moves the number

| Lever | Effect on $/M-tok | Mechanism |
|---|---|---|
| FP8 weights vs FP16 | **−40 to −50%** | 2× decode throughput (bandwidth-bound) |
| Larger batch (B=64 → 256) | **−30%** | Amortize weight read across more tokens; bounded by attention KV growth |
| Prefix caching at 50% hit rate | **−25%** | Skip prefill for hit portion |
| Continuous + chunked prefill vs static | **−60%** | Eliminate idle gaps |
| Speculative decoding (γ=4, accept ~70%) | **−30 to −50%** ITL, **+0 to −20%** $/M-tok | Tradeoff; throughput depends on draft model |
| Disaggregated prefill/decode | **−20 to −30%** at scale | Independent scaling, no chunked-prefill ITL tax |
| H100 → H200 | **−30 to −40%** for long context | Capacity dominates when context is long |

**[STAFF+] economic case for disaggregation:** prefill and decode have orthogonal scaling curves. Prefill scales with FLOPs/s; decode scales with HBM bandwidth × replica count. Co-located, you provision for the worse of the two. Disaggregated, you provision each pool independently and route. At Anthropic-API scale, the savings are >20%. The cost is a KV-transfer plane (50 GB/s IB rails) and replica-affinity routing, both of which are operational, not algorithmic, complexity.

---

## 11. Senior says vs Staff says

| # | Senior says | Staff says |
|---|---|---|
| 1 | "Use vLLM" | "Continuous batching + paged KV is the load-bearing pair; vLLM is one implementation. We pick X over Y because of $\{number\}$." |
| 2 | "Tensor parallel for big models" | "TP=8 within node, never across; all-reduce is on the critical path twice per layer, NVL fits in the latency budget, IB doesn't." |
| 3 | "KV cache uses memory" | "320 KiB/token at this config, 40 KiB per GPU at TP=8. With 60 GB free per GPU, 1.7M token capacity per replica. 128K-context request = 3% of that on its own." |
| 4 | "Decode is slower than prefill" | "Decode is bandwidth-bound on weight reads (2.6 ms/tok floor on H100 FP8) and on KV reads when context grows; prefill is compute-bound above S=512." |
| 5 | "We'll batch requests together" | "Iteration-level: batch at every forward pass, decoders and prefill chunks coexist in the same step, attention via PagedAttention with block tables." |
| 6 | "Quantize the model" | "FP8 E4M3 weights with per-channel scaling, BF16 KV cache by default. NE-sensitivity audit per layer; outlier layers stay BF16. <1% eval delta, 2× throughput." |
| 7 | "Use CUDA graphs" | "Capture per batch-shape bucket, replay on hot path; graph cache LRU; prefill skips capture (one-shot). Block-table tensors are stable across replays — capture against pointers." |
| 8 | "Add more GPUs to scale" | "DP replication scales throughput linearly; TP=8 stays fixed. Replica routing on prefix hash for cache locality, fallback to least-loaded." |
| 9 | "Handle backpressure" | "Watermark-based admission: high-water triggers preemption to a target free pool, low-water resumes admits. Frontend load-sheds with 429 + jittered retry to break cascade." |
| 10 | "Long contexts are slower" | "Decode KV read scales linearly with context; at 128K, attention dominates the per-token cost. Page size still 16, but attention kernel cost is now 60% of step time." |
| 11 | "Prefill and decode interfere" | "Naive priority causes ITL spikes ≥ prefill latency. Sarathi-Serve chunked prefill (C≈1024) keeps iter latency bounded; disaggregation is the next move at scale." |
| 12 | "Use FlashAttention" | "FlashAttention-2/3 varlen + paged variant; tiles in SRAM, doesn't materialize the attention matrix. FA3 picks up FP8 + async on Hopper." |
| 13 | "Cache the prompts" | "Prefix tree (radix) over token IDs, blocks ref-counted, CoW on divergence. Tiered: HBM hot, host RAM warm, SSD cold. Eviction is per-block LRU, not per-request." |
| 14 | "Make it fault-tolerant" | "TP group is the failure domain; replica is the redundancy unit. NCCL watchdog drains within 30 s; in-flight requests fail and client retries via SDK with idempotency key." |
| 15 | "We can use speculative decoding" | "Draft model 7B, accept rate 65–75% on chat; γ=4. ITL improves 2–3× at low batch. At high batch the verification kernel saturates and benefit collapses — toggle per route." |
| 16 | "Streaming uses SSE" | "Detokenizer is stateful per-request (BPE merges, multi-byte UTF-8 handling); stream chunker flushes on token boundary; back-pressure from slow clients goes to scheduler as priority demotion, not block." |
| 17 | "Just rent more GPUs at peak" | "Auto-scale on rate-of-change, not absolute load; replica cold-start is 60+ s (weights + graph capture), so reactive scaling is too slow. Pre-warm pool at p95 of yesterday's curve." |

---

## 12. Anthropic-Specific Angles (public surface only)

Map design choices to publicly visible product/engineering surface. Don't speculate about internals.

- **Long context (200K)** → KV capacity is binding; H200 is the obvious choice; per-token KV math dominates fleet sizing; FP8 KV becomes attractive for the long tail; chunked prefill must work at S=200K (means C must be tuned per request, not global).
- **Prompt caching as a product feature** → prefix-tree cache is exposed as user-controlled cache breakpoints; cache locality is now a routing constraint, not just an optimization (pinning a tenant to a replica subset). KV must persist across requests within a TTL window.
- **Message streaming SSE** → streaming is first-class; ITL p99 SLO matters more than median; chunked-prefill tuning is the first lever.
- **Tool use rounds** → multi-turn tool-call loops increase total context per logical request; same KV math applies but reuse rate is high (good for prefix cache); each round has its own TTFT, so TTFT SLO compounds.
- **Multi-tier API SLOs** (latency, batch, prompt caching tiers) → admission control and routing must be tier-aware; batch tier can run on chunked-prefill-heavy schedules with higher batch sizes; latency tier may run with smaller batch and speculative decoding on.

---

## 13. Time Budget — 40-Minute Slot

| Min | Section | Tag |
|---|---|---|
| 0–2 | Draw the skeleton (§1) | **MUST** |
| 2–5 | Clarifying questions + state assumptions | **MUST** |
| 5–10 | Capacity math: KV/token, HBM budget, AI roofline | **MUST** |
| 10–18 | Core mechanisms (continuous batching + paged KV + chunked prefill + TP) | **MUST** |
| 18–22 | Memory layout single GPU, multi-node topology | **MUST** |
| 22–25 | CUDA graphs, admission control | should |
| 25–30 | Failure modes + cascade analysis | **MUST** (Staff signal) |
| 30–34 | Trade-offs + cost | should |
| 34–38 | Anthropic-specific surface mapping | should |
| 38–40 | Forward-reference: disaggregation, speculative decoding, MoE deltas | **MUST** (frame the next conversation) |

If short on time, **skip:** PP discussion, detailed CUDA graph mechanics, detailed cost table.
**Never skip:** capacity math, continuous batching + paged KV, failure cascade, TP=8 reasoning.

---

## 14. Likely Follow-Up Probes & Ideal Responses

**Q1: "What if a request needs 100K context?"**
> KV for one request: 100K × 320 KiB = 32 GiB across the TP group, 4 GiB per GPU. That's 6% of one replica's KV budget. Two of those concurrent and we're at 12%; ten of them and we've blown the budget. Mitigations: (1) admission control with per-request KV reservation, (2) dynamic batch sizing — fewer concurrent decoders when long contexts are present, (3) FP8 KV cache for the long tail to halve the cost, (4) at extreme scale, context parallelism (CP) shards the sequence dimension across replicas, but it adds a comm step per attention layer.

**Q2: "How do you handle a sudden 10× traffic spike?"**
> Three-stage response. (1) Frontend load-shed: 429 with retry-after to keep cascade from forming; cap retry storms with SDK-side jitter. (2) In-flight: scheduler downshifts batch composition — fewer admits, prioritize finishing in-flight requests over new prefills, preempt to host RAM. (3) Replica scale-out is reactive on rate-of-change, not absolute load; cold-start is ~60 s (weight load + graph capture) so it lags. Pre-warming a hot-spare pool at the diurnal p95 makes the cold-start invisible.

**Q3: "Walk me through what happens when the KV cache is full."**
> High-watermark crosses → admission stops for new prefills. If still pressured, the scheduler picks preemption victims — longest-running, lowest-priority decoders first — and either swaps their KV to host RAM (PCIe ~25 GB/s, ~40 ms per GiB) or drops it for re-prefill on resumption. Choice is "swap if KV size > swap_threshold else recompute," tuned around the prefill-cost vs swap-cost crossover. Once free pool reaches low-watermark, admits resume. Backpressure to the LB throughout. Key invariant: never preempt to *exactly* one block free — always to a target pool size.

**Q4: "Why not just use vLLM out of the box?"**
> vLLM is the right reference and the right baseline for v0. The reasons to fork or build are operational at scale: (a) tighter integration with your weight management/loading pipeline, (b) kernel work for specific quant formats your model uses (FP8 with custom scaling), (c) scheduler policies tuned to your tenancy and SLO model — vLLM's defaults are general-purpose, (d) instrumentation depth (per-token traces, per-layer profiling baked in), (e) custom features like global prefix cache, disaggregated prefill, that vLLM has but you may want differently shaped. **You don't fork to be different; you fork because the integration tax of upstreaming every change is higher than maintaining the divergence.**

**Q5: "How would you add speculative decoding?"**
> Draft model — typically 7B or a smaller sibling — proposes γ tokens (γ=4 is the sweet spot for chat). Target model verifies in one forward pass with `γ+1` token batch, accepts the longest prefix that matches its sampled tokens. Acceptance rate 65–75% on chat workloads → ~2–3× ITL improvement at low-to-medium batch. Integration with continuous batching is non-trivial: the verify step changes batch shape and KV write pattern. EAGLE-style (drafted from target's hidden states) is more accurate and avoids the second model in HBM, but requires training. Toggle per-route — at high batch (B>256), the verification cost stops amortizing and you give up throughput.

**Q6: "What changes if the model is MoE instead of dense?"**
> Three things. (1) Weight memory dominated by all experts (e.g., 671B for DeepSeek-V3-class) → no longer single-node TP=8; needs expert parallelism (EP) across nodes with rail-optimized IB. (2) Active params per token are smaller (e.g., 37B), so decode bandwidth read drops — decode is *faster* per token than dense 70B. (3) All-to-all in expert dispatch becomes a new latency hotspot, on the critical path of every layer. Token-routing imbalance becomes a failure mode (some experts hot, some cold). FP8 expert weights + careful expert placement matter more than any optimization on the dense path.

**Q7: "How do you keep TTFT predictable under load?"**
> TTFT = queue time + prefill time. Queue time is bounded by admission control + per-tenant fairness. Prefill time is bounded by chunked prefill: each iteration is ≤ chunk_latency, so TTFT ≤ ⌈prefill_tokens/C⌉ × iter_time. Under load, you trade TTFT for ITL by tuning C: smaller C = better ITL, worse TTFT. The controller keeps both within SLO by adjusting C and batch composition; if both can't fit, shed load.

**Q8: "What's your story for the multi-tenant noisy-neighbor problem?"**
> Fairness at admission, not at execution. Per-tenant token-bucket rate limit at the frontend; per-tenant queue with weighted-fair-queueing into the scheduler; per-tenant KV-cache quota (soft, with burst). At execution, all requests are equal once admitted — no priority lanes inside the scheduler, because dynamic priority inversion under continuous batching is a debugging nightmare. Isolation between tiers (latency vs batch vs free) is replica-level, not request-level.

**Q9: "How do you debug an ITL regression in production?"**
> Per-iteration timeline traces: capture iter_start, kernel_launch_done, kernel_complete, all-reduce_complete, sample_complete for a sampled fraction of iterations. Aggregate by batch composition (decoders count, prefill chunk size, total tokens). The regression usually decomposes into one of: (a) prefill chunks growing (someone bumped C), (b) batch size shifting (admission-control tuning), (c) attention KV cost growing (long contexts arrived), (d) host overhead growing (graph cache miss rate up). Each has a different fix.

**Q10: "When do you reach for disaggregation?"**
> Three signals together: (1) chunked-prefill ITL tax is consuming >15% of decode budget on hot routes, (2) prompt-to-output ratio is asymmetric (>5:1 or <1:5) so co-located provisioning wastes one resource, (3) you have spare IB bandwidth between potential prefill and decode pools (KV transfer is 320 KiB/token; for 1K tokens at 50 GB/s that's 6 µs, fine). At that point split the pools, route requests through a KV-transfer plane, and scale each independently.

---

## 15. Appendix: Numbers Worth Memorizing

### Hardware

| Metric | A100 80GB | H100 SXM | H200 SXM | B200 |
|---|---|---|---|---|
| HBM | 80 GB | 80 GB | 141 GB | 192 GB |
| HBM BW | 2.0 TB/s | 3.35 TB/s | 4.8 TB/s | 8 TB/s |
| BF16 TFLOPS | 312 | 989 | 989 | ~2250 |
| FP8 TFLOPS | — | 1979 | 1979 | ~4500 |
| NVLink/GPU | 600 GB/s | 900 GB/s | 900 GB/s | 1800 GB/s |

### Network

- NVLink (H100, per GPU, bidirectional, NVSwitch): **900 GB/s** aggregate
- InfiniBand NDR: **400 Gb/s = 50 GB/s** per NIC, typically 1 NIC per GPU (rail-optimized)
- PCIe Gen5 x16 (host↔GPU): **63 GB/s** theoretical, ~25–50 GB/s practical for swaps

### KV cache formula

```
KV_bytes_per_token = 2 × num_layers × num_kv_heads × head_dim × dtype_bytes

Worked examples (BF16):
  Llama-2 7B  (32 layers, 32 KV heads,  128 d): 2×32×32×128×2 = 524,288 = 512 KiB
  Llama-3 8B  (32 layers,  8 KV heads,  128 d): 2×32× 8×128×2 = 131,072 = 128 KiB
  Llama-3 70B (80 layers,  8 KV heads,  128 d): 2×80× 8×128×2 = 327,680 = 320 KiB
  Llama-3 405B(126 layers, 8 KV heads,  128 d): 2×126×8×128×2= 516,096 ≈ 504 KiB

GQA reduction is enormous: 7B w/o GQA at 32 KV heads = 4× the KV of 8B w/ GQA at 8.
```

### Arithmetic intensity formulas

```
Prefill (single layer, S tokens, batch B):
  Linear FLOPs: ~12 × B × S × d_model²   (Q,K,V,O,FFN)
  Linear bytes: 2 × d_model² + activations
  AI ≈ B × S    →   compute-bound for any practical B, S ≥ 256

  Attention FLOPs: 4 × B × S² × d_model
  Attention bytes (FA): O(B × S × d_model)   (no full matrix materialized)
  AI ≈ S        →   compute-bound for S ≥ ~256

Decode (1 token per request, B requests, K = avg KV length):
  Linear FLOPs: 12 × B × d_model²
  Linear bytes: 2 × d_model²   (one read of weights)
  AI ≈ B        →  bandwidth-bound on weights for B < ~300 (BF16) or ~600 (FP8)

  Attention FLOPs: 4 × B × K × d_model
  Attention bytes: 2 × B × K × d_model    (KV read, no reuse across requests)
  AI ≈ 2        →  always bandwidth-bound on KV
```

### Per-GPU decode throughput rule of thumb (mixed workload, H100 FP8)

| Model size | Single-stream tok/s | Saturated batch tok/s/node (TP=8) |
|---|---|---|
| 7B | ~1500 | ~80K |
| 13B | ~900 | ~50K |
| 70B | ~380 | ~25K–35K |
| 405B (across 2 nodes, PP) | ~120 | ~6K–10K |

These collapse hard with long context — at K=32K, expect 30–50% degradation; at K=128K, 60–80%.

### Key papers/systems to cite

- **vLLM / PagedAttention** (Kwon et al., SOSP '23) — paged KV, continuous batching reference impl.
- **FlashAttention 2/3** (Dao et al.) — IO-aware attention; FA3 = Hopper async + FP8.
- **Sarathi-Serve** (Agrawal et al., OSDI '24) — chunked prefill, prefill-decode mixing.
- **DistServe** (Zhong et al., OSDI '24) — disaggregated prefill/decode.
- **Splitwise** (Patel et al., ISCA '24) — phase-aware splitting on heterogeneous hardware.
- **Mooncake** (Qin et al., 2024) — KVCache-centric disaggregated arch, global prefix cache over RDMA.
- **SGLang / RadixAttention** (Zheng et al., NeurIPS '24) — prefix-tree KV reuse, structured generation.
- **EAGLE / Medusa** — speculative decoding from target hidden states / multi-head drafting.

---

## Closing — what to leave the interviewer with

1. The scheduler is the product; everything else is infrastructure to feed it.
2. KV memory, not FLOPs, is the binding constraint for decode at frontier scale.
3. TP=8 single-node, DP across nodes — every other topology has to defend itself.
4. Chunked prefill solves 90% of prefill-decode interference; disaggregation is the next architectural step, not the first.
5. Failure cascades are the Staff signal: load-shed at the frontend before the cascade forms; preempt to a target free pool, never to one block; pre-warm replicas on rate-of-change.