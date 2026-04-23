---
title: LLM Quantization
description: LLM Quantization
---

## 1. What a staff engineer actually needs to know

**In interviews, you are being tested on tradeoff reasoning, not algorithmic novelty.** The signal is whether you can distinguish *memory savings* from *bandwidth savings* from *actual speedup*, and whether you know *why* a specific quantization decision either succeeded or blew up. You are not expected to re-derive GPTQ on a whiteboard; you are expected to explain in 30 seconds why a W4A16 model can be faster than BF16 at batch=1 but slower at batch=256.

**In real systems work, the question is almost never "which algorithm."** The question is:

1. What format does the hardware support *natively*, and in which direction (element type, block size, scale type, accumulator)?
2. Which kernel am I actually hitting, and does it fuse dequant with GEMM?
3. Where does error compound — residual path, attention scores, long-context decode, MoE routing?
4. Does my calibration set match production traffic?
5. What stays in higher precision, and why?

**Depth expected:** You should be able to discuss GPTQ, AWQ, SmoothQuant, and NF4/QLoRA as four distinct ideas with four different goals; explain FP8 E4M3 vs E5M2 and when each is used; explain microscaling (MX) and NVFP4 at block/scale granularity; reason about prefill vs decode regimes; and describe *why* training quantization is harder than inference quantization without handwaving.

**What you can skip unless you are building a quantization algorithm from scratch:**

- The full literature of rounding schemes (AdaRound, BRECQ, OmniQuant details)
- Per-paper ablations on Wikitext perplexity
- Hardware details below the tensor-core instruction level
- Exact scaling-factor search heuristics inside GPTQ/AWQ

Keep the mental models. Drop the trivia.

---

## 2. Core mental model

### 2.1 Three distinct things lower precision buys you

```python
     ┌─────────────────────────────────────────────────────────────┐
     │  Lower precision delivers THREE separable wins              │
     ├─────────────────────────────────────────────────────────────┤
     │  1. Storage     (fits more in DRAM / on-disk)               │
     │  2. Bandwidth   (fewer bytes moved HBM→SM, NIC→NIC)         │
     │  3. Compute     (tensor core at lower precision = more FLOPs)│
     └─────────────────────────────────────────────────────────────┘
```

These are *not* the same. They are also not automatic.

- **Weight-only quant** (e.g., W4A16) buys **storage + weight-fetch bandwidth**. It does **not** buy compute; the matmul still runs in FP16/BF16 after dequant.
- **Weight + activation quant** (e.g., W8A8, W4A4) buys **compute** *if and only if* the hardware has native tensor-core support for that format.
- **KV-cache quant** buys **decode-phase bandwidth + capacity** (more concurrent sequences, longer context). It doesn't affect prefill much.

Internalize this triangle. Most quantization confusion comes from conflating the three.

### 2.2 Why LLMs are unusually sensitive

Three structural reasons LLMs are harder to quantize than CNNs:

1. **Outlier channels.** A handful of activation channels in every transformer layer carry huge magnitudes — orders of magnitude larger than the rest. Naive per-tensor activation quantization crushes the other channels to zero. This is the central observation behind LLM.int8(), SmoothQuant, and AWQ.
2. **Residual accumulation.** Transformer depth is 40–100 layers. Small per-layer quantization error compounds multiplicatively through residual paths, and long context amplifies it further (KV-cache errors accumulate over token position).
3. **Autoregressive decoding is memory-bound.** Latency at batch=1 is set by how fast you can pull weights and KV from HBM, *not* by how fast you can multiply. Quantization's bandwidth win dominates here; its compute win is nearly irrelevant at low batch.

### 2.3 Memory-bound vs compute-bound regimes

```
                     ARITHMETIC INTENSITY (FLOPs / byte)
                     ────────────────────────────────────►
   BANDWIDTH-BOUND                                      COMPUTE-BOUND
   ┌─────────────┐                                      ┌─────────────┐
   │ decode      │                                      │ prefill     │
   │ batch=1..8  │                                      │ batch≫1    │
   │ long ctx KV │                                      │ training    │
   └─────────────┘                                      └─────────────┘

   Quantization win = bandwidth reduction               Quantization win = FLOPs
   (weight-only + KV-cache helps)                       (need W8A8/FP8/FP4 + tensor core)
```

On an H100: ~3.35 TB/s HBM, ~1979 TFLOPs BF16 dense. Arithmetic intensity at which the roofline flips is ~590 FLOPs/byte. A decode step for a 70B model at batch=1 is ~2 FLOPs/byte per weight read — *deeply* memory-bound. No amount of compute quantization helps until you raise batch or fuse aggressively.

**Rule of thumb:** weight-only quant is a decode/low-batch optimization. Full low-precision compute (FP8/FP4) is a prefill/training/high-batch optimization.

### 2.4 The five distinct quantization problems inside an LLM

| Target | Dominant cost | Typical tolerance | Main failure mode |
|---|---|---|---|
| Weights | storage + bandwidth | tolerates 4-bit routinely | outlier rows/cols |
| Activations | compute bandwidth | 8-bit fine, 4-bit hard | outlier channels, per-token spikes |
| Gradients | comms + storage | needs FP8 E5M2-like range | underflow of small gradients |
| Optimizer states | storage | 8-bit works (bnb) | Adam second moment precision |
| KV cache | decode bandwidth + capacity | 8-bit trivial, 4-bit viable | long-context score drift |

Each of these is a *different* optimization problem with different constraints. Talking about "LLM quantization" as one thing obscures all the real engineering.

References: [NVIDIA H100 datasheet](https://resources.nvidia.com/en-us-tensor-core), [DeepSeek-V3 Technical Report §3.3](https://arxiv.org/html/2412.19437v1).

---

## 3. Basic numerics foundation

### 3.1 The affine quant/dequant mental model

For a real value `x` mapped to integer code `q`:

```
q   = round( x / s )  + z          (quantize)
x̂  = ( q - z ) * s                 (dequantize)
```

- `s` is the **scale** (FP32): the real-world step size between adjacent codes.
- `z` is the **zero point** (integer): the code that represents real zero.
- `round(·)` collapses infinite precision to discrete codes.
- **Symmetric**: `z = 0`, codes are [-2^(b-1), 2^(b-1)-1], real zero maps to integer zero, scales are simpler, GEMM cleaner. Standard for weights.
- **Asymmetric**: `z ≠ 0`, needed when the distribution is skewed (e.g., post-ReLU activations, which are ≥0). More accurate on skewed tensors but adds an extra term to matmul.

Two errors always exist:

- **Rounding error**: `|x - x̂| ≤ s/2` (at most half a step).
- **Clipping error**: when `|x|` exceeds the representable range. Catastrophic — the error is unbounded.

Choosing `s` is a tradeoff: larger `s` → less clipping, more rounding error; smaller `s` → opposite. This is why outliers dominate: one outlier forces `s` up and degrades everything else.

### 3.2 Symmetric INT8 worked example

Suppose a weight row has values in `[-0.8, 0.6]`. Pick `s = max(|x|) / 127 = 0.8/127 ≈ 0.0063`.

```
 x      =  0.60     → q = round(0.60 / 0.0063)   = 95     → x̂ =  95 * 0.0063 ≈  0.5985
 x      = -0.80     → q = round(-0.80 / 0.0063)  = -127   → x̂ = -127 * 0.0063 = -0.8001
 x      =  0.003    → q = round(0.003 / 0.0063)  = 0      → x̂ = 0             (lost)
 outlier:  8.0      → q = round(8.0  / 0.0063)   = 1270   → clamped to 127 → x̂ ≈ 0.80 (massive clip)
```

Notice: the tiny value (0.003) got quantized to zero (round-to-nearest underflow at this scale), and a single outlier destroys the range for everything else.

### 3.3 Scale granularity

```
          PER-TENSOR           PER-CHANNEL          PER-GROUP / BLOCK
          ┌──────────┐         ┌───────────┐         ┌─┬─┬─┬─┬─┬─┐
          │    s     │         │s₁ s₂ s₃ s₄│         │s│s│s│s│s│s│
          │          │         │           │         │ │ │ │ │ │ │
          │  tensor  │         │  row i    │         │blk blk blk│
          └──────────┘         └───────────┘         └─┴─┴─┴─┴─┴─┘
          1 scale              1 per row (or col)    1 per K-element block
          coarse, fast         middle ground         fine, best quality
          poor with outliers   standard for weights  essential for low-bit
```

- **Per-tensor**: one scalar `s` for the whole tensor. Dead simple, GEMM-friendly, but a single outlier anywhere torches quality. Common for activations in older schemes.
- **Per-channel** (a.k.a. per-row for weights along the output dimension): one scale per row. This is essentially free in a GEMM because it's a post-multiply scale on the accumulator. Standard for INT8 weights.
- **Per-group / per-block**: one scale per K contiguous elements (group size often 32, 64, 128). Required for 4-bit and below. Costs a tiny amount of overhead (group size 32 → 8-bit scale = 1 extra bit per weight on average for INT4). This is the foundation of all modern low-bit formats (GPTQ group-128, AWQ group-128, MXFP4 block-32, NVFP4 block-16).

### 3.4 Static vs dynamic quantization

- **Static**: compute scales *offline* from calibration data (typically 128–512 sequences). Applied at inference time as constants. Needed for hardware that wants scales known ahead of kernel launch. Sensitive to calibration distribution shift.
- **Dynamic**: compute scales *at runtime* from the actual activation tensor (e.g., running per-token amax). No calibration, more robust to distribution shift, but costs a small scan + reduction on every forward. Standard for activations in modern serving (vLLM FP8 dynamic).

Weights are always static — they don't change. Activations can be either. KV-cache is almost always dynamic per-token.

### 3.5 Integer vs low-bit floating-point quantization

```
  INT4 (uniform steps)            FP4 E2M1 (log-spaced steps)
  ─────|─────|─────|─────         ──|─|─|──|──|────|────|────
  -6  -4  -2  0  2  4  6          -6 -4 -3-2-1.5-1-.5 0 .5 1 1.5 2 3 4 6
  equal gap everywhere            dense near zero, sparse far out
```

- **INT-N**: uniform spacing. Optimal if your distribution is roughly uniform over the range. Tensor-core INT8 is mature on all GPUs; INT4 is weight-only (there is no A100/H100 INT4 tensor-core MMA for activations).
- **FP-N**: log-spaced (exponent bits) + linear interpolation (mantissa bits). Higher density near zero, lower density at the tails. Much better match for neural-network activations/weights/gradients, which are heavy-tailed and concentrated near zero.

For a given bit count, FP beats INT on Gaussian-ish distributions and is more robust to outliers because of its wider dynamic range. This is the primary reason FP8 displaced INT8 for high-end inference serving, and why FP4 (in microscaled form) displaced INT4 for Blackwell-generation inference.

FP8 variants matter:

- **E4M3**: sign + 4 exp + 3 mantissa. Range ±448. More mantissa, less range. **Used for weights and forward activations.**
- **E5M2**: sign + 5 exp + 2 mantissa. Range ±57344. More range, less mantissa. **Used for gradients** (huge dynamic range needed).

See [OCP 8-bit Floating Point Specification](https://www.opencompute.org/documents/ocp-8-bit-floating-point-specification-ofp8-revision-1-0-2023-12-01-pdf-1).

### 3.6 Why blockwise scaling matters

Dynamic range of a tensor is set by its largest element. If you use per-tensor scaling and one element is 100× the rest, the rest gets ~6.6 bits crushed out of usable precision (log2(100)). Blockwise scaling localizes the problem: within a 32-element block, the local amax rarely differs from the global by 100×.

Concretely, if a BF16 tensor has amax=10 and the 99th percentile of block-amax is 0.5, per-tensor FP4 has to encode [-10, 10] in 16 codes; per-block FP4 with block size 32 encodes each block's own ~[-0.5, 0.5] in 16 codes. The precision improvement is ~20× on typical blocks.

This is the entire point of microscaling formats (MX, NVFP4). You pay a small overhead for the scales (1 byte per 32 elements = 0.25 bits per weight) and recover most of the accuracy of FP16.

### 3.7 Signal-to-noise intuition

For uniform random rounding on a tensor with standard deviation σ at step size s:

```
 SNR ≈ σ² / (s² / 12)
 dB  ≈ 10 * log10(12 * (σ/s)²)
```

Roughly, each bit adds ~6 dB. INT8 on a Gaussian tensor gets ~48 dB SNR; INT4 gets ~24 dB. That's the *analog* SNR — the *effective* SNR in an LLM is worse because of outliers and heavy tails. This is why blockwise scaling is not optional for ≤4-bit schemes: you need to re-concentrate the distribution per-block to recover effective SNR.

---

## 4. What gets quantized in an LLM

### 4.1 The anatomy

```
     Token IDs
         │
         ▼
   ┌────────────┐
   │ Embeddings │  ← usually KEPT HIGH PRECISION (small, sensitive to SNR)
   └─────┬──────┘
         │
  ┌──────┴──────────────────────────────┐
  │ for L layers:                       │
  │   RMSNorm  ← typically FP32 or BF16 │
  │   Attention: Q,K,V,O proj (GEMMs)   │
  │     Softmax/score  ← BF16+ usually  │
  │     KV-cache  ← OFTEN QUANTIZED     │
  │   RMSNorm                           │
  │   MLP: gate/up/down proj (GEMMs)    │
  │     [+ MoE routing]                 │
  │   residual accumulations (FP32)     │
  └──────┬──────────────────────────────┘
         ▼
   ┌────────────┐
   │  lm_head   │  ← usually KEPT HIGH PRECISION
   └─────┬──────┘
         ▼
      logits → softmax
```

### 4.2 Weights

**Why it matters:** bulk of parameters (>99% in a dense LLM). Dominates storage, DRAM footprint, and decode-bandwidth.

**What works:** INT8 per-channel trivially. INT4/FP4 with group size 32–128 works with GPTQ/AWQ at <1% quality loss for most 7B+ models. 3-bit is viable with effort. 2-bit (HQQ, QuIP#) is research-grade.

**What breaks first:** certain rows correspond to "salient" weight channels that interact with outlier activations. AWQ specifically identifies these.

**Tradeoff:** weight-only quantization is almost always the highest-quality/lowest-risk quantization you can apply. The quality-vs-bits curve for weights alone is very forgiving.

### 4.3 Activations

**Why it matters:** without activation quantization, you cannot hit low-precision tensor cores for compute. You're stuck at FP16/BF16 matmul.

**What works:** INT8 dynamic per-token + INT8 per-channel weights (SmoothQuant-style). FP8 E4M3 per-tensor or per-token for Hopper-class FP8.

**What breaks first:** outlier channels. In GPT-style models, 1–2% of hidden dimensions carry activations ~100× larger than the median. Per-tensor quantization of these layers collapses. This discovery (Dettmers et al., LLM.int8()) is *the* foundational result in LLM activation quantization. [LLM.int8()](https://arxiv.org/abs/2208.07339).

**Tradeoff:** activation quant is where quality lives or dies. Weight-only first, activation second. Decision also depends on whether decode latency (bandwidth-bound, so activation quant adds little) or prefill throughput (compute-bound, activation quant crucial) is the target.

### 4.4 Gradients

**Why it matters:** training backward pass. Gradients have huge dynamic range (small magnitudes for deep network early layers, larger magnitudes near the loss). They also need to be all-reduced across workers — comms is often the bottleneck.

**What works:** FP8 E5M2 (wider exponent range) with loss scaling or per-tensor dynamic scaling. MXFP8 E5M2 for block-scaled.

**What breaks first:** underflow. Small gradient magnitudes quantized to zero → stuck neurons → instability → training divergence. Stochastic rounding (see §6) is the standard mitigation.

### 4.5 Optimizer states

**Why it matters:** Adam stores two extra states per parameter (m and v). At BF16/FP32 master, this is 4× the parameter memory. For a 70B model at FP32 Adam master, that's ~840 GB of optimizer state alone.

**What works:** [bitsandbytes 8-bit Adam](https://arxiv.org/abs/2110.02861) blockwise quantizes both moments. Paged 8-bit Adam from QLoRA is the standard for single-GPU finetuning of large models. [Shampoo/SOAP](https://arxiv.org/abs/2002.09018) also benefit from state compression.

**What breaks first:** the second moment `v` is sensitive near zero (it's a variance estimate; zero values break the `sqrt(v)+ε` denominator). Blockwise scaling is essential.

### 4.6 KV cache

**Why it matters:** at long context, KV cache dominates memory. For Llama-3-70B with 8K context and batch 32, KV cache is ~20 GB (in FP16) — more than the activations. Decode reads every cached K and V *every step*, so KV is the dominant bandwidth consumer during generation.

**What works:**
- **INT8 per-token per-head**: drop-in, ~0 quality loss, 2× capacity.
- **FP8 E4M3 per-token**: same size, slightly better range, native on Hopper+.
- **INT4 with group-scales**: more aggressive; viable with mild quality hit for decode, care needed for long context.

**What breaks first:** long-context degradation. Per-token quantization errors in K and V show up in attention scores via dot products that accumulate error over sequence length. Early tokens' K get revisited for every new decode step; quantization bias drifts the attention distribution. [KVQuant](https://arxiv.org/abs/2401.18079), [KIVI](https://arxiv.org/abs/2402.02750).

**Prefill vs decode:** prefill computes KV once per token; decode re-reads all of it every step. Quantizing KV therefore helps decode disproportionately.

### 4.7 Embeddings and lm_head

**Why it matters:** at the vocabulary edges. The lm_head projects to 32K–256K logits; the embedding is the inverse.

**What usually works:** keep them in BF16/FP16. They are small relative to transformer blocks (one layer each) and are logit-sensitive.

**What breaks if you quantize them:** subtle logit shifts that destabilize top-k sampling and break calibration for tasks like MMLU. Almost no serving stack quantizes lm_head by default.

### 4.8 Normalization-sensitive paths

- **RMSNorm / LayerNorm** itself: typically computed in BF16/FP32 regardless of surrounding precision. Cheap, critical for stability.
- **Residual adds**: usually accumulated in FP32. This is important — a BF16 residual accumulator can underflow small layer outputs over many layers.
- **Softmax / attention scores**: computed in BF16 with FP32 accumulation. Quantizing scores themselves is rarely done; the dynamic range post-exp is absurd.

### 4.9 Attention vs MLP sensitivity

In practice:
- **MLP** is 2/3 of parameters and FLOPs. Quantizing MLP gives you most of the memory/speed win.
- **Attention** (specifically Q/K/V/O projections) is more sensitive to outliers than MLP, especially in models with GQA (grouped-query attention) or MLA (multi-head latent attention) where the K/V projections are already compressed.
- The score computation and softmax should stay higher precision. DeepSeek-V3 explicitly keeps attention operators in BF16/FP32 even in their FP8 framework ([DeepSeek-V3 §3.3](https://arxiv.org/html/2412.19437v1)).

---

## 5. Inference quantization

### 5.1 The design space

```
                    activations
               BF16/FP16   FP8       INT8       FP4/INT4
     ┌────────┬──────────┬─────────┬──────────┬──────────┐
 W16 │baseline│  —       │ —       │ —        │ —        │
     ├────────┼──────────┼─────────┼──────────┼──────────┤
 W8  │ W8A16  │ W8A8 FP8 │W8A8 INT │ —        │ —        │
     │        │          │         │          │          │
     ├────────┼──────────┼─────────┼──────────┼──────────┤
 W4  │W4A16   │ W4A8 FP8 │W4A8 INT │ W4A4     │ —        │
     │ (AWQ,  │  (mixed  │         │ (aggres- │          │
     │ GPTQ)  │  bit)    │         │  sive)   │          │
     └────────┴──────────┴─────────┴──────────┴──────────┘
        ▲            ▲          ▲         ▲
   weight-only   mixed-bit   W8A8     fully quant
   memory only   (prod)      compute  NVFP4 etc.
```

### 5.2 Weight-only quantization (W4A16, W8A16)

**What:** weights in INT4/FP4/INT8, activations in FP16/BF16. At matmul time, dequantize weight tile → FP16/BF16 → standard GEMM.

**When to use:** decode-heavy workloads, consumer GPUs, small batch sizes. This is the right default for local inference of 70B+ models.

**Why it works:** weights are static and well-behaved; 4-bit with group-128 scales loses <1% on standard benchmarks. Activations are untouched, so outliers are no issue.

**Where the speed comes from:** you move 4× fewer bytes from HBM during matmul. At batch=1, the weight read is the bottleneck, so this is a near-linear win. The dequant is fused into the kernel so the actual math still runs in FP16/BF16 on standard tensor cores.

**Why it can be slow:** at large batch, the compute dominates, and you're still doing FP16 matmul. You've paid dequant overhead for no throughput win. Specialized kernels (Marlin, Machete) are designed to minimize this overhead.

**The Marlin trick:** highly-optimized W4A16 GEMM kernels that fuse dequant, handle weight repacking for memory coalescing, and reach near-peak memory bandwidth on Ampere+. [Marlin paper / vLLM Marlin kernels](https://github.com/vllm-project/vllm/blob/main/csrc/quantization/gptq_marlin/gptq_marlin.cu).

### 5.3 Weight + activation quantization (W8A8, W4A8, W4A4)

**What:** both weights and activations in low precision. The matmul runs on low-precision tensor cores (INT8 tensor core, FP8 tensor core, FP4 tensor core).

**When to use:** throughput-maximizing server inference, especially prefill-heavy or high-batch workloads.

**Formats that matter in 2025–2026:**
- **W8A8 FP8 (E4M3)**: the mainstream production choice on Hopper/Lovelace. ~2× throughput and memory vs BF16, with <0.5% quality loss using per-tensor or per-token scaling. See [vLLM FP8 docs](https://docs.vllm.ai/en/latest/features/quantization/fp8/).
- **W8A8 INT8 (SmoothQuant / Q-Channel)**: older path, still used on Ampere (no hardware FP8 on A100). Needs activation-smoothing to survive outliers.
- **W4A8 and W4A4 (NVFP4)**: Blackwell-era. W4A4 NVFP4 roughly doubles throughput vs FP8 on B200/B300 when the kernel is mature. See [NVFP4 on Blackwell (NVIDIA dev blog)](https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/).

**Where quality breaks:** activation outliers. SmoothQuant shifts the pain to weights; AWQ protects "salient" weights; microscaling (NVFP4) localizes via 16-element blocks. All three are responding to the same outlier problem.

### 5.4 KV-cache quantization

```
 PREFILL                           DECODE
 batch=B, seq_len=S                batch=B, seq_len=1 (new token)
 writes: B×S keys, B×S values      writes: B×1 key,  B×1 value
 reads:  same (compute attn)       reads:  B×S_total keys, values (every step!)

         ▲                                 ▲
         │ KV-cache quant negligible       │ KV-cache quant huge:
         │ (one-shot write)                │ S × decode_steps reads saved
```

At decode, every step reads the full KV cache. Quantizing KV to INT8 or FP8 halves that bandwidth. Quantizing to INT4 with per-head per-group scales quarters it but with long-context risk.

**Practical recipe (2025+):** FP8 E4M3 KV-cache on Hopper, per-token dynamic scales. On Blackwell, NVFP4 or FP8 depending on context length. Long-context (>32K) workloads should keep KV at ≥FP8 or use more sophisticated per-head per-group schemes.

[vLLM supports FP8 KV-cache](https://docs.vllm.ai/en/latest/features/quantization/fp8/), as does [TensorRT-LLM](https://nvidia.github.io/TensorRT-LLM/).

### 5.5 Prefill vs decode regime summary

| Regime | Bottleneck | What helps most |
|---|---|---|
| Prefill (long input) | Compute (tensor core throughput) | W8A8 FP8 / W4A4 NVFP4 — full low-precision compute |
| Decode batch=1 | Weight-fetch bandwidth | W4A16 weight-only + FP8 KV-cache |
| Decode large batch | Shared weight amortization + KV bandwidth | W8A8 FP8 + FP8 KV-cache |
| High concurrency / paging | Memory capacity + KV bandwidth | KV-cache quant + paged-attention |

A production serving stack (vLLM, SGLang, TensorRT-LLM) must pick a configuration that works across the full batch distribution and context-length mix.

### 5.6 Why kernel support dominates real speedup

A newly-invented 3-bit format with perfect accuracy is *useless* without a GEMM kernel that can run it faster than BF16. Kernel concerns:

- Dequant throughput: can you unpack 4-bit → 16-bit faster than tensor cores can consume it?
- Weight layout / packing: 4-bit weights often need bespoke shuffled layouts so that a warp can load 32 weights in a single instruction without bank conflicts.
- Accumulator precision: INT8 accumulates in INT32; FP8 accumulates in FP32 (with the caveat that Hopper tensor cores do partial FP22 accumulation, see §9).
- MoE / grouped GEMM: quantized MoE kernels are meaningfully harder than dense quantized GEMMs because expert assignment varies per-token.

This is why a paper showing 4-bit quality ≈ FP16 on benchmarks can take 6–12 months to reach production latency wins: the kernel work is where the time goes.

---

## 6. Training quantization

### 6.1 QAT vs low-precision training vs PTQ

Three distinct things that are often conflated:

- **PTQ (post-training quantization)**: take a pretrained FP16/BF16 model, calibrate and quantize, ship. No gradients involved. §7 covers this.
- **QAT (quantization-aware training)**: fine-tune the model with *fake quant* nodes in the forward pass — the model sees quantized values but gradients flow in FP32 via the straight-through estimator. Used to *recover quality* when PTQ drops too much.
- **Low-precision training / mixed-precision training**: run the *actual* forward and backward passes in low precision (FP8, MXFP8, NVFP4) to accelerate training throughput. This is the hard one.

### 6.2 Mixed precision training — the standard recipe

```
  PARAMETERS (master, FP32)
         │
         │ cast to BF16/FP8 for forward
         ▼
   FORWARD ─ in low precision (BF16, FP8, FP4)
         │
         ▼
   LOSS (FP32)
         │
         ▼
   BACKWARD ─ in low precision (BF16, FP8 E5M2)
         │
         │ upcast gradient to FP32
         ▼
   OPTIMIZER STEP (FP32 master weight update)
         │
         └─► updated master → repeated
```

The **master weights in FP32** is the canonical trick that makes mixed precision work. Each update step is tiny (learning rate × gradient), often below the resolution of BF16. Accumulating in FP32 preserves the small updates; the cast back down happens fresh each step. Without master weights, training diverges within tens of steps.

### 6.3 What DeepSeek-V3 actually did (the canonical FP8 pretraining recipe)

[DeepSeek-V3](https://arxiv.org/html/2412.19437v1) trained a 671B-parameter MoE in FP8 for all three major GEMMs: forward (Fprop), input-gradient backward (Dgrad), weight-gradient backward (Wgrad). Key details (section 3.3 of the paper, analyzed further in [Colfax's writeup](https://research.colfax-intl.com/deepseek-r1-and-fp8-mixed-precision-training/)):

- **Fine-grained scaling**: 1×128 tile for activations (per-token, 128-channel groups), 128×128 block for weights. Per-tensor scaling was insufficient.
- **FP32 accumulation on CUDA cores**: Hopper tensor cores accumulate FP8 matmuls in only ~14-bit precision internally. DeepSeek promoted partial sums to FP32 registers on CUDA cores at fixed intervals (N_c interval) to recover accuracy. This is a *major* and underappreciated detail.
- **Kept in higher precision**: embeddings, lm_head, MoE gating, normalization, attention operators, master weights, gradients, optimizer states. Only the three heavy GEMMs went to FP8.
- **Low-precision comms**: EP all-to-all dispatch in FP8, reducing MoE communication volume by 50%.
- **Loss error vs BF16 baseline**: <0.25% on DeepSeek-V2-scale validation.

This proved FP8 pretraining works at scale. The success is not the format — it's the co-design: blockwise scaling + FP32 accumulation + selective precision retention + custom CUTLASS kernels.

### 6.4 Gradient quantization challenges

Gradients are the hardest tensor to quantize because:

1. **Huge dynamic range**: log-magnitude distributions span 6+ orders. FP8 E5M2 covers it with margin; INT8 cannot without per-block scaling.
2. **Small values matter**: a gradient with magnitude 1e-4 updates the weight in Adam via `lr * m / (√v + ε)` — underflow here means a parameter stops learning.
3. **Backward propagates through long chains**: quantization bias in a gradient propagates to all upstream layers. Unbiased rounding is critical.

### 6.5 Stochastic rounding (SR)

```
  Round-to-nearest (RtN):    x=3.3  → 3 always. Biased on sub-unit values.
  Stochastic rounding (SR):  x=3.3  → 3 with prob 0.7
                                     4 with prob 0.3

                             E[SR(x)] = x   (unbiased!)
```

Stochastic rounding is *unbiased*: on average, the quantized value equals the true value. RtN systematically rounds small positive values toward zero, which is the exact failure mode for gradients.

**Why this matters for training, not inference:** training accumulates gradients across batches and steps. Biased rounding → biased weight drift → slow divergence from the "true" trajectory. Inference is one-shot — bias accumulates only across layers (much shorter). Hence SR for gradients and activations in low-precision training; RtN is fine for weights in inference.

SR is used explicitly in [NVFP4 pretraining recipes](https://developer.nvidia.com/blog/nvfp4-trains-with-precision-of-16-bit-and-speed-and-efficiency-of-4-bit/) and is in the OCP MX training guidance.

### 6.6 Loss scaling (pre-Hopper) and dynamic per-tensor scaling

For FP16 training (pre-BF16 era), gradients frequently underflowed FP16's narrow range. Solution: scale the loss by a large constant K before backward, then unscale gradients before the optimizer step. This shifts small gradients into representable range.

BF16 has FP32's exponent range (±~1e38) and needs no loss scaling. FP8 needs *per-tensor dynamic scaling* — each FP8 tensor gets its own FP32 scale updated each step based on observed amax. NVIDIA's Transformer Engine implements this with delayed amax tracking (history of past N steps) to avoid the sync cost of computing amax on the fly.

### 6.7 Optimizer state precision

Standard: Adam master in FP32. Memory = 8 bytes/param for m,v (vs 2 bytes for BF16 params → optimizer state is 4× parameter memory).

Reductions:
- **bitsandbytes 8-bit Adam**: blockwise quantize m,v. 4× memory reduction on optimizer state.
- **Paged optimizer** (QLoRA): page m,v to CPU/unified memory.
- **Adam → Adafactor / Lion / Muon**: reduce state count (Lion has no second moment; Adafactor factorizes second moment).

DeepSeek-V3 kept m,v in BF16 (compression 2×) but with FP32 master weights.

### 6.8 Why training is much harder than inference

| Factor | Inference | Training |
|---|---|---|
| Forward only | yes | no — also backward |
| Gradient numerics | N/A | critical; wide dynamic range |
| Weight updates | none | tiny, easy to underflow |
| Error compounding | within one forward | across steps for weeks |
| Calibration | can use held-out data | no fixed calibration set |
| Tolerated quality loss | ~0.5–1% often fine | divergence = restart from checkpoint |
| Recovery | re-quantize | expensive restart, lost GPU-hours |

The asymmetry is stark: an inference quant bug costs 1% accuracy; a training quant bug costs millions of dollars in re-training.

### 6.9 "Fully quantized training" — what it actually means

In practice, "FP8 training" or "FP4 training" almost always means:

- Heavy GEMMs in low precision (Fprop, Dgrad, Wgrad) ✓
- Master weights in FP32 ✗
- Optimizer states in FP32 or BF16 ✗
- Embeddings, lm_head, norms, attention ops in higher precision ✗
- Gradients in higher-range FP8 (E5M2) or MXFP8 ~
- Comms in low precision ✓ (partial)

Nothing currently trains a large model *entirely* in FP8/FP4 — the mastering, normalization, and attention paths stay higher. That's fine; the point is that the bandwidth and compute costs are dominated by the GEMMs and comms, which *are* in low precision.

[NVFP4 pretraining](https://developer.nvidia.com/blog/nvfp4-trains-with-precision-of-16-bit-and-speed-and-efficiency-of-4-bit/) demonstrated a 12B-parameter model trained with NVFP4 achieving accuracy comparable to FP8 baselines — a meaningful result but not yet standard practice at the 100B+ scale.

### 6.10 Where instability comes from

In descending order of frequency:

1. **Gradient underflow** in specific layers (usually embedding-adjacent or very deep layers).
2. **Outlier activations** saturating the FP8 forward and biasing the backward.
3. **Accumulator overflow** when accumulation precision is insufficient (the DeepSeek 14-bit TC issue).
4. **Amax history lag**: delayed scaling picks a scale that's wrong for the *current* step when distributions shift (warmup, LR changes).
5. **MoE routing imbalance**: if FP8 quantization biases the gating function, expert utilization collapses. This is why DeepSeek-V3 keeps gating in BF16.

---

## 7. PTQ methods that matter

### 7.1 Round-to-nearest (RTN) baseline

Take the FP16 tensor, pick `s = amax / q_max`, round. Done. No data needed.

- **Quality**: ~1% degradation on weights at INT8, 5–10%+ at INT4 on 7B models.
- **Use it**: as a sanity baseline and for FP8 weight quant where calibration is overkill.
- **Why others exist**: at ≤4 bits, RTN is not good enough.

### 7.2 Calibration-based PTQ (min-max, percentile, MSE)

Run a calibration set (128–512 sequences), collect activation statistics per tensor, pick scales that minimize a criterion (min-max, 99.9th percentile, or MSE between `x` and `x̂`). Mostly relevant for activations (weights don't need calibration data).

- Standard in TensorRT, ONNX Runtime INT8 paths.
- Sensitive to calibration distribution — garbage in, garbage out.

### 7.3 GPTQ (weights, 4-bit)

[GPTQ (Frantar et al., 2022)](https://arxiv.org/abs/2210.17323) is the canonical weight-only quantization algorithm.

**Core idea**: quantize weight columns one at a time, and after each column, *propagate the quantization error onto the remaining unquantized columns* so that the next column's values are adjusted to compensate. Uses second-order information (the Hessian of the layer output w.r.t. weights, approximated via the empirical covariance of the inputs) to weight the error appropriately.

```
   Naive RTN per-column:             GPTQ:
   col_0 → quantize (error ignored)  col_0 → quantize, spread err to cols 1..N-1
   col_1 → quantize                  col_1 → quantize (adjusted), spread err to 2..N-1
   ...                               ...
```

Pseudocode sketch:
```
 H = 2 X Xᵀ / N          # Hessian approximation, X = calibration activations
 H⁻¹ = Cholesky(H)
 for block of cols:
     for col in block:
         w_q = quantize(w_col)
         err = (w_col - w_q) / H⁻¹[col, col]
         w[:, col+1:] -= err * H⁻¹[col, col+1:]
```

- **What it solves**: much better 4-bit weight quality than RTN, with a single-pass algorithm (no retraining).
- **Assumes**: a small calibration set well-matched to deployment.
- **Works well**: general-purpose 4-bit weight PTQ.
- **Fails**: doesn't address activation outliers (it's weight-only). Calibration-mismatch sensitive.
- **Wins**: memory. Not directly latency (that's kernel-dependent).

### 7.4 AWQ (weights, 4-bit, activation-aware)

[AWQ (Lin et al., 2023)](https://arxiv.org/abs/2306.00978).

**Core idea**: not all weight channels are equally important. The "salient" 1% of channels — those multiplied by large activation magnitudes — matter disproportionately. Protect them by *scaling up* salient weight channels before quantization (and scaling down the corresponding activation channels to compensate). This is per-channel equivalent scaling: `y = (W ⊙ s) @ (x / s)`, which preserves the math but changes what gets quantized well.

```
 Original:  W x    →  quantize W coarsely  → large err on salient channels
 AWQ:       (Ws)(x/s) →  quantize (Ws)     → salient channels get more
                                              effective bits
```

- **What it solves**: 4-bit weight quality with better outlier behavior than GPTQ. Usually a bit better than GPTQ for LLMs.
- **Assumes**: activation-magnitude statistics identify importance.
- **Works well**: very general, popular in vLLM production.
- **Fails**: still weight-only; doesn't quantize activations.
- **Wins**: memory, slight quality edge, good kernel support (AWQ-Marlin).

### 7.5 SmoothQuant (W8A8 INT8)

[SmoothQuant (Xiao et al., 2022)](https://arxiv.org/abs/2211.10438).

**Core idea**: activation outliers make INT8 activation quant fail. Weights are easy to quantize. So *migrate the difficulty* from activations to weights via per-channel scaling: divide activation channel `c` by `s_c` and multiply weight row `c` by `s_c`. Choose `s_c` to balance the amaxes between activations and weights.

```
 Before:  activations [small small LARGE small] @ weights
          ← easy                                ← hard to quant (INT8 fails)

 After:  activations [small small small  small ] @ weights
                                                   [      BIG        ]
          ← easy to INT8 quant              ← still easy (static,
                                               per-channel fine)
```

- **What it solves**: enables W8A8 INT8 with minimal quality loss, unlocking INT8 tensor cores on Ampere.
- **Assumes**: you can observe per-channel activation scales from calibration.
- **Works well**: INT8 serving on A100-class hardware before FP8 existed.
- **Fails**: the scaling is a math-preserving transform, but kernels need to fuse it correctly; the scale factors have to be baked into the weights.
- **Wins**: compute (INT8 tensor core throughput), memory.

**In 2025+**: FP8 largely supplanted SmoothQuant-style W8A8 INT8 on H100/B200 because FP8's dynamic range absorbs the outliers natively. SmoothQuant remains relevant on Ampere and non-NVIDIA hardware without FP8.

### 7.6 bitsandbytes / NF4 / QLoRA

[QLoRA (Dettmers et al., 2023)](https://arxiv.org/abs/2305.14314) introduced NF4.

**NormalFloat 4 (NF4)**: a non-uniform 4-bit data type where the 16 codes are chosen to be the *quantiles of a standard normal distribution*. Since pretrained weights are approximately Gaussian-distributed, NF4 codes are information-theoretically near-optimal for this distribution — better than uniform INT4 or FP4 E2M1 for weights specifically.

```
 INT4 codes:  uniform spacing
 -8 -7 -6 -5 -4 -3 -2 -1 0 1 2 3 4 5 6 7

 NF4 codes:  quantiles of N(0,1), dense near mean
 -1.0  -0.69  -0.53  -0.39  -0.28  -0.18 -0.09  0  0.08  0.18  0.28  0.39  0.53  0.72  1.0
```

**Double quantization**: the per-block scales themselves are quantized (again, to 8-bit) to save the ~0.5 bits/param overhead.

**QLoRA**: combine NF4 frozen base weights + BF16 LoRA adapters. The base model doesn't need gradients (it's frozen), so the 4-bit representation is adequate. LoRA adapters train in BF16. This is the standard recipe for single-GPU finetuning of 70B models.

- **What it solves**: 4-bit weight storage with better-than-INT4 accuracy, trainable adapters on top.
- **Assumes**: weights are approximately Gaussian.
- **Works well**: finetuning, storage.
- **Fails**: inference speedup is limited (NF4 isn't a tensor-core format; kernels dequant to BF16). NF4 is a memory format, not a compute format.
- **Confusion to avoid**: QLoRA is a *finetuning* technique. Serving models in NF4 is possible but typically slower than AWQ/GPTQ INT4 on equivalent hardware.

### 7.7 Activation-aware and outlier-aware approaches (summary)

The pattern across GPTQ → AWQ → SmoothQuant → newer methods (QuaRot, SpinQuant):

1. **GPTQ**: second-order error compensation on weights.
2. **AWQ**: re-weight importance by activation magnitude.
3. **SmoothQuant**: migrate outliers from activations to weights.
4. **QuaRot / SpinQuant**: apply random Hadamard rotations to activations and weights to *smear outliers across channels*, making the distribution more amenable to low-bit quantization. [QuaRot](https://arxiv.org/abs/2404.00456), [SpinQuant](https://arxiv.org/abs/2405.16406).

Rotation-based methods have become important for W4A4 (4-bit weights + 4-bit activations) because naive W4A4 destroys quality without outlier mitigation. NVFP4 papers often combine blockwise FP4 + random Hadamard transform for this reason.

### 7.8 Summary table

| Method | Bits | W/A | Key idea | Calibration | Where it wins |
|---|---|---|---|---|---|
| RTN | any | W | just round | none | baseline only |
| GPTQ | INT4 | W | Hessian-based error compensation | yes | general-purpose 4-bit weights |
| AWQ | INT4 | W | protect salient weight channels | yes | general-purpose, slight edge |
| SmoothQuant | INT8 | W+A | migrate outliers a→w via scales | yes | INT8 on Ampere |
| NF4 / QLoRA | 4-bit | W | Gaussian-optimal codebook | none | finetuning, storage |
| QuaRot / SpinQuant | INT4 | W+A | rotations smear outliers | yes | W4A4, aggressive schemes |

---

## 8. Modern low-precision formats and hardware co-design

### 8.1 INT8 vs FP8

| Aspect | INT8 | FP8 E4M3 |
|---|---|---|
| Range | ±127 (uniform) | ±448 (exp-scaled) |
| Density near zero | flat | dense (good for Gaussian) |
| Outlier tolerance | poor | good (exponent range) |
| Tensor-core support | A100+ | H100+ (and Ada) |
| Calibration burden | high | low (dynamic works well) |
| Quality at W8A8 | needs SmoothQuant | works ~out of box |

FP8 won over INT8 for three practical reasons: outlier tolerance (less engineering work to make it "just work"), better match to LLM distributions (fewer calibration failures), and native support on modern NVIDIA hardware with identical throughput.

On hardware without FP8 (A100, MI250), INT8 + SmoothQuant remains the W8A8 choice.

### 8.2 FP8 E4M3 vs E5M2

```
 E4M3:  S EEEE MMM
        range: ±448 | fine-grained near zero | extra mantissa
        USE: weights, forward activations

 E5M2:  S EEEEE MM
        range: ±57344 | wider dynamic range | less precision
        USE: gradients (backward pass)
```

The NVIDIA Transformer Engine default is E4M3 forward / E5M2 backward. Hopper and Blackwell tensor cores accept both.

### 8.3 Microscaling (MX) — the block-scaled family

[OCP Microscaling Formats Specification v1.0 (2023)](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf) standardized the MX family. Key idea: each block of 32 elements shares one 8-bit exponent (E8M0, unsigned power-of-two scale):

```
   MX block (32 elements):
   ┌────────────────────────────────────────────┬───┐
   │  element₀  element₁ ... element₃₁          │ X │     X = E8M0 shared scale (8 bits)
   └────────────────────────────────────────────┴───┘
      each element = d bits (4, 6, 8)

   Concrete formats:
      MXFP8  = E4M3 or E5M2 | block 32 | E8M0 scale
      MXFP6  = E3M2 or E2M3 | block 32 | E8M0 scale
      MXFP4  = E2M1         | block 32 | E8M0 scale
      MXINT8 = INT8         | block 32 | E8M0 scale
```

Consortium: AMD, Arm, Intel, Meta, Microsoft, NVIDIA, Qualcomm. Standardized to encourage cross-vendor support.

Overhead: 8 bits / 32 elements = 0.25 bits per element. For MXFP4 that's 4.25 effective bits.

**Blackwell support**: Tensor cores natively execute MXFP8 matmul with block-scale application in hardware. MXFP4 also supported with 2× peak throughput over MXFP8. [Wikipedia: Block floating point](https://en.wikipedia.org/wiki/Block_floating_point), [NVIDIA dev blog on MX formats](https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/).

### 8.4 NVFP4 — NVIDIA's tightened FP4

NVIDIA argued MXFP4's E8M0 scale is too coarse (power-of-two only) and block size 32 is too large for heavy-tailed LLM distributions. [NVFP4](https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/) tightens both:

```
   NVFP4 block (16 elements):
   ┌────────────────────────┬────┐
   │  e₀ e₁ ... e₁₅ (E2M1)  │ S  │    S = E4M3 block scale (8 bits, fractional!)
   └────────────────────────┴────┘
                                    + per-tensor FP32 second-level scale

   Block size 16 (vs 32 for MXFP4) → 2× scale density
   Scale format E4M3 (vs E8M0)      → fractional scales, finer-grained
   Per-tensor FP32 outer scale      → absorbs overall tensor magnitude
```

Why this matters: the E4M3 fractional scale can encode `1.25 × 2^n` instead of only `1 × 2^n`, so scaling is more accurate. Smaller blocks mean local distributions stay bounded. Empirically, NVFP4 closes ~all of the gap vs FP8 for inference on large models, and NVIDIA demonstrated it for pretraining at 12B scale. See also [NVFP4 training blog](https://developer.nvidia.com/blog/nvfp4-trains-with-precision-of-16-bit-and-speed-and-efficiency-of-4-bit/).

### 8.5 Why hardware-native support matters

An emulated FP4 format (quantize/dequantize in software, compute in FP16) doesn't give you any compute win. You need:

1. **Tensor-core MMA instructions** for the format.
2. **Block-scale hardware** that applies the per-block scale during the matmul (not after) so you don't pay dequant cost.
3. **Sufficient accumulator precision** (FP32 or FP22+) so that summing thousands of low-precision products doesn't accumulate bias.
4. **Packing / load instructions** for the specific layout (e.g., loading two FP4 values per byte).

Blackwell's 5th-gen tensor cores (SM100) have native MXFP8/MXFP6/MXFP4/NVFP4 support with integrated block-scaling. This is what makes the format viable for real speedup, not just compression.

### 8.6 Why format alone is not enough

You can ship a model in NVFP4 on a Hopper GPU and it will still compile — but it'll run slower than BF16, because Hopper has no NVFP4 tensor core. The kernel will fall back to software dequantization + FP16 GEMM (vLLM does this with a warning: ["Your GPU does not have native support for FP4 computation..."](https://forums.developer.nvidia.com/t/your-gpu-does-not-have-native-support-for-fp4-computation-but-fp4-quantization-is-being-used/355494)).

Lesson: **format × hardware × kernel** is one triple. Never analyze the format in isolation.

### 8.7 Layout, packing, accumulator precision

- **Packing**: FP4 is 4 bits. Two values pack per byte. Weight tensors are stored in special shuffled layouts (Marlin, CUTLASS NVFP4) so that warps can load 8 packed weights in one instruction with coalesced access.
- **Accumulator**: on Hopper, FP8 MMA accumulates into FP32 *registers* but internally with only ~14–22 bit mantissa precision. DeepSeek showed that for long matmul reductions (K≥1024), this accumulates measurable error. Fix: promote partial sums to true FP32 on CUDA cores every N tiles.
- **Scale format**: the MX family stores scales as E8M0 (8 bits, power-of-two only). NVFP4 uses E4M3 scale (8 bits, fractional), gaining ~1 bit of scale precision at the cost of a slightly more expensive scale-apply.

---

## 9. Kernel and systems view

### 9.1 Where speedup actually comes from

For a matmul `Y = X @ Wᵀ`:

```
 Steps per output tile:
 1. Load X tile   (from HBM or L2 → shared memory)
 2. Load W tile   (from HBM → shared memory, quantized)
 3. Dequant W     (if weight-only quant: unpack + scale)
 4. Tensor core MMA
 5. Scale / bias
 6. Write Y tile
```

For **weight-only W4A16**:
- Bytes moved from HBM are 4× smaller (4-bit weights).
- Step 3 is new work: must dequant per-tile on the way to tensor cores.
- Steps 4 runs at FP16 tensor-core rate (no compute speedup).
- Win: bandwidth. Need: fused dequant kernel (Marlin, Machete) so step 3 is hidden behind step 2's HBM latency.

For **W8A8 FP8**:
- Both X and W are 8-bit.
- Step 4 runs at FP8 tensor-core rate (2× FP16 throughput on H100).
- Win: bandwidth + compute.
- Need: FP8 kernel path (CUTLASS FP8, cuBLAS FP8).

For **NVFP4 (W4A4)**:
- Both X and W are ~4-bit with block scales.
- Step 4 runs at FP4 tensor-core rate (2× FP8, 4× FP16 on B200+).
- Win: full triangle — bandwidth + compute + storage.
- Need: native NVFP4 kernel path — currently maturing in CUTLASS/vLLM/TRT-LLM.

### 9.2 Fused dequant + GEMM

Why fuse:
- Avoid a separate dequant pass that writes BF16 weights back to HBM (would undo the bandwidth saving).
- Overlap dequant with subsequent tile loads (hide latency).

Marlin-style kernels are the reference for INT4 weight-only fused GEMM on Ampere+. They achieve ~90% of FP16 peak bandwidth on decode-sized matmuls.

### 9.3 Accumulator precision and the 14-bit tensor-core issue

On Hopper (H100) FP8 MMA, the formal accumulator is FP32 but the internal precision of the Wgmma op is reduced. From [DeepSeek-V3 §3.3](https://arxiv.org/html/2412.19437v1): "accumulation inside Tensor Cores is limited to approximately 14-bit precision." For large K (inner dimension of the matmul), repeated accumulations lose bits.

The standard mitigation (and the one DeepSeek implemented): every N_c tiles (they used 128), promote the accumulator to a true FP32 value held in CUDA-core registers, reset the TC accumulator, and continue. This double-accumulation pattern is now visible in CUTLASS's `KernelTmaWarpSpecializedCooperativeFP8BlockScaledAccum` scheduler.

On Blackwell, tensor cores have stronger internal accumulation, but the same principle applies — very long reductions still benefit from explicit FP32 promotion.

### 9.4 Memory layout and batching

- **Quantized tensor layouts**: weights often stored in a tile-interleaved or swizzled layout (think bank-conflict-free shared memory addressing). Standard row-major will not perform.
- **Batching**: as batch grows, weight reuse per loaded byte grows. At batch=1, each weight is used once per layer. At batch=64, each weight is used 64× per layer. The compute-per-byte ratio grows linearly, shifting from bandwidth-bound to compute-bound. This is why W4A16 speedup flattens at high batch — you no longer need the bandwidth relief.
- **Sequence length**: at long prefill sequences, attention cost (O(S²) for dense, O(S) for linear) starts to matter against MLP cost. KV-cache quantization pays off on long decoded sequences.
- **Paged KV cache**: vLLM-style paging is orthogonal to quantization. FP8/INT8 pages still work the same; the page is just half-size.

### 9.5 When quantization helps throughput vs latency vs capacity

| Goal | What to quantize | Why |
|---|---|---|
| Single-stream latency | Weights (W4A16) + KV (FP8) | pure bandwidth problem |
| Server throughput | Weights + activations (W8A8 FP8 or NVFP4) | need compute |
| Fit-on-GPU capacity | Weights (W4) + KV (INT8/FP8) | it's a storage problem |
| Long-context decode | KV-cache aggressively (INT4 KV if tolerable) | KV dominates BW |
| Training throughput | GEMMs in FP8/MXFP8, gradients FP8 E5M2 | compute + comms |

---

## 10. Accuracy failure modes

### 10.1 Outlier channels (the central issue)

In every post-LayerNorm activation tensor of a modern LLM, ~0.1–1% of hidden dimensions carry 50–1000× the median magnitude. Per-tensor activation quantization collapses the remaining 99%.

Symptoms: sudden quality cliff on tasks that rely on specific attention heads; MMLU drops 5%+ while perplexity changes little.

Fixes: SmoothQuant (migrate to weights), AWQ (protect salient weight channels), rotation methods (spread outliers across dimensions), or block-scaling (localize the amax).

### 10.2 Activation spikes

Specific tokens (often BOS, punctuation, or reserved tokens) produce activations 10× larger than surrounding tokens. Per-token dynamic scaling handles this; per-tensor static scaling does not.

### 10.3 Layer sensitivity

Not all layers tolerate quantization equally. Early layers (close to embedding) and late layers (close to lm_head) tend to be more sensitive than the middle stack. Quantization-aware pipelines sometimes keep the first 2 and last 2 layers in higher precision as a cheap safety.

### 10.4 Residual path sensitivity

The residual stream accumulates small errors across all layers. BF16 accumulation in residuals is *usually* fine; FP16 can underflow for deep models. Quantizing the residual stream directly is almost never done.

### 10.5 Normalization interactions

LayerNorm/RMSNorm computes `(x - mean) / √var * γ`. Quantizing `x` before norm is fine; quantizing *inside* norm is risky. Standard practice: compute norms in BF16/FP32 regardless of surrounding precision. Norm weights (γ) stay in BF16.

### 10.6 Attention score sensitivity

Post-softmax attention weights have a concentrated distribution (most mass on 1–5 tokens). Quantizing them loses the small values and over-attends to the max. This is why softmax stays in BF16/FP32 in all mainstream recipes.

### 10.7 Long-context degradation

Quantized KV cache errors accumulate over sequence length. A 1% per-token K error → O(1%) degradation on attention scores for context length 8K, but noticeable degradation at 128K where a token looks back over tens of thousands of quantized K entries. Symptoms: passkey retrieval degrades faster than perplexity.

### 10.8 Compounding error across layers

A 0.5% per-layer RMS error over 80 layers compounds to ~49% if errors are uncorrelated — but they are partially correlated with the signal (biased rounding), so the actual failure mode is worse. This is why *bias*-free rounding (SR) matters, and why per-block scaling beats per-tensor.

### 10.9 Calibration mismatch

PTQ calibration sets (e.g., 128 Wikipedia samples) often don't match production traffic (chat, code, multilingual). An LLM quantized on English Wikipedia can lose 5–10% on code benchmarks.

Mitigations: (a) calibrate on a mixture matching deployment distribution, (b) use dynamic activation quantization (eliminates the issue for activations), (c) use AWQ over SmoothQuant (AWQ's activation-aware scaling is slightly less calibration-sensitive).

### 10.10 Distribution shift at serve time

Even a well-calibrated PTQ model can degrade over time as traffic shifts (new prompt styles, new languages). Dynamic activation scaling is the structural fix. Static-scale FP8/INT8 configs should be periodically re-calibrated against current traffic distributions.

---

## 11. How people actually use quantization in practice

### 11.1 Consumer GPU inference (single-user, local)

- **Scenario**: run a 70B model on a 24GB 4090, or a 7B on a 12GB card.
- **Typical choice**: W4A16 with AWQ or GPTQ, group size 128. bitsandbytes NF4 on older GPUs or for QLoRA-compatible setups.
- **Why**: bandwidth-bound single-stream decode, storage is the hard constraint, activation quant gives no meaningful win at batch=1.
- **Serving stacks**: llama.cpp (GGUF formats), exllamav2, vLLM's single-GPU mode.

### 11.2 Server GPU inference (production multi-tenant)

- **Hopper (H100/H200)**: FP8 W8A8 is the default. Per-tensor or per-token dynamic scales. KV-cache FP8 E4M3. Typically 1.5–1.8× throughput vs BF16 at iso-quality.
- **Ada Lovelace (L40S)**: similar to Hopper, FP8 supported.
- **Ampere (A100)**: no FP8; SmoothQuant W8A8 INT8, or W4A16 GPTQ/AWQ + Marlin. KV in FP16 or INT8.
- **Blackwell (B200/B300)**: FP8 still very common; NVFP4 for bleeding-edge throughput on workloads with mature kernel support. MXFP4 for models shipped in that format (e.g., gpt-oss 120B). [TensorRT-LLM blog on Blackwell](https://developer.nvidia.com/blog).

### 11.3 Finetuning / LoRA / QLoRA

- **QLoRA** (NF4 frozen base + BF16 LoRA) is the standard for finetuning 70B+ on a single A100/H100.
- **Low-rank adapters** add <1% parameters at BF16, learn quickly, can be merged back.
- **Do not confuse with serving**: the *finetuning* is done in NF4; the deployed model is often re-quantized to AWQ/FP8 for serving.

### 11.4 Production serving frameworks (2025–2026)

- **vLLM**: FP8 W8A8 (Hopper+), W4A16 GPTQ/AWQ via Marlin (Ampere+), FP8 KV-cache, NVFP4 weight-only and W4A4 on Blackwell. Full list: [vLLM quantization docs](https://docs.vllm.ai/en/latest/features/quantization/).
- **TensorRT-LLM**: deepest kernel optimization, FP8/FP4/INT8/INT4, hand-tuned for H100/B200. Shipping day-one FP4 support for Blackwell.
- **SGLang**: FP8 W8A8, strong on DeepSeek-V3-class MoE.
- **LMDeploy**: FP8/BF16 serving, strong on AMD MI300x.

### 11.5 Training at reduced precision

- **BF16 everywhere + FP32 master** is still the safe default for most training.
- **FP8 mixed-precision** (Transformer Engine, DeepSeek's recipe): ~1.3–1.7× training speedup for Hopper, increasingly common for 100B+ pretraining.
- **MXFP8/NVFP4** pretraining: emerging, Blackwell-era, not yet the industry default at trillion-token scale.
- **Optimizer state compression**: 8-bit Adam (bitsandbytes), paged Adam (QLoRA). Combines cleanly with mixed precision.

### 11.6 When "just use BF16" is the right answer

- Small models (<1B) where the absolute memory savings are trivial.
- Research workflows where debugging time dominates GPU cost.
- Novel architectures where no one has validated FP8/FP4 stability yet.
- Cases where you do not have the kernel support (e.g., running on hardware without FP8 like consumer 3090/4090 for training).

---

## 12. Recent developments (2025–2026)

### 12.1 FP8 pretraining has become mainstream

[DeepSeek-V3](https://arxiv.org/html/2412.19437v1) published in late 2024 was the first public demonstration of FP8 pretraining at 671B scale, with a clean <0.25% loss-error vs BF16. This shifted the industry consensus: FP8 training for 100B+ models is now a mainstream choice, not a research risk. Key ingredients (fine-grained scaling, FP32 promoted accumulation, selective precision retention) became canonical.

Since then, multiple frontier labs have adopted FP8 training. The relevant engineering problem shifted from "does it work" to "which flavor of FP8 and which kernel stack."

### 12.2 Blackwell and the microscaling era

NVIDIA Blackwell (B200/B300, released 2024–2025) introduced native tensor-core support for:
- FP8 E4M3/E5M2 (unchanged from Hopper)
- **MXFP8** (OCP spec, block 32, E8M0 scale)
- **MXFP6** (same block/scale, 6-bit element)
- **MXFP4** (same block/scale, 4-bit E2M1 element)
- **NVFP4** (NVIDIA's tighter 4-bit: block 16, E4M3 scale + per-tensor FP32)

Peak FP4 throughput is ~2× FP8, which is ~2× BF16. A B200 delivers ~20 PFLOPs of FP4 dense compute per GPU. See [NVIDIA Blackwell architecture whitepaper](https://resources.nvidia.com/en-us-blackwell-architecture).

### 12.3 NVFP4 inference — shipping now

[NVFP4 inference on Blackwell](https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/) demonstrates <1% accuracy loss on DeepSeek-R1-class models at W4A4, with ~2.3× throughput over FP8 when kernels are mature. Current reality (as of April 2026): NVFP4 is production-ready for weight-only W4A16 on Blackwell; full W4A4 kernels are still maturing in vLLM and CUTLASS but closing the gap ([vLLM NVFP4 Marlin util](https://docs.vllm.ai/en/stable/api/vllm/model_executor/layers/quantization/utils/marlin_utils_fp4/)).

Practical implication: if you are serving on B200+ and your workload is prefill-heavy, NVFP4 W4A4 is a significant throughput win. For decode-heavy, NVFP4-W4A16 using existing FP16 GEMM paths is often the safer choice until the W4A4 kernels mature.

### 12.4 NVFP4 pretraining — promising, not yet universal

NVIDIA published [NVFP4 pretraining recipes](https://developer.nvidia.com/blog/nvfp4-trains-with-precision-of-16-bit-and-speed-and-efficiency-of-4-bit/) and demonstrated it on a 12B model with accuracy matching FP8. Techniques: micro-block scaling, stochastic rounding (unbiased estimator), random Hadamard transforms (outlier smoothing), selective precision retention for non-GEMM ops.

Papers like [*FP4 All the Way*](https://openreview.net/pdf?id=kuzye4EPLR), TetraJet-v2, and Four Over Six (late 2025) explored scaling this to larger models with adaptive outlier mitigation. As of early 2026, NVFP4 pretraining is not yet the default for trillion-token runs — but the gap is narrowing. The consensus expectation: within the next model generation, NVFP4 or equivalent will be the dominant training precision on Blackwell-class hardware.

### 12.5 MXFP4 in production models

OpenAI's [gpt-oss 120B](https://openai.com/) ships natively in MXFP4 weights, explicitly trading a small accuracy difference for memory and bandwidth. This is the first widely-deployed model released in a sub-8-bit *format as the primary distribution*, rather than post-quantized. Inference stacks must load MXFP4 weights directly.

### 12.6 Serving stack FP8/FP4 maturity

- **vLLM**: FP8 W8A8 production-ready since 2024. FP8 KV cache, AWQ-Marlin. NVFP4 W4A16 via Marlin, NVFP4 W4A4 in progress with occasional perf gaps vs theoretical.
- **TensorRT-LLM**: strongest FP4 kernels in early 2026; fastest absolute throughput on Blackwell for most LLMs.
- **SGLang**: strong FP8 support; optimized for MoE.
- **CUTLASS**: the substrate for most of the above. FP8 blockwise scaling stable since 3.7; NVFP4 kernels added through 2025.

### 12.7 Genuine vs hype

**Genuinely shipping at scale:**
- FP8 inference and training — mature.
- W4A16 AWQ/GPTQ — mature.
- FP8 KV cache — mature.
- NVFP4 weight-only inference on Blackwell — mature.
- MXFP4 as a distribution format — emerging.

**Still research-heavy / risky:**
- W4A4 or lower fully-activated training.
- Sub-4-bit weights (2-bit, 3-bit) at production scale.
- Cross-vendor portability of NVFP4 (MXFP4 is OCP-standard; NVFP4 is NVIDIA-specific, though Marlin-style kernels exist for non-native hardware).

---

## 13. Interview reasoning patterns

### 13.1 "Why choose AWQ vs GPTQ vs FP8?"

- **FP8 W8A8** if Hopper/Blackwell hardware, throughput matters, batch is non-trivial. You get compute AND memory win.
- **AWQ W4A16** if you need maximum memory saving for decode (batch=1 or small), or if running on Ampere without FP8. Slight quality edge over GPTQ.
- **GPTQ W4A16** is the "baseline" 4-bit weight-only — generic, well-tooled, fine default.
- **NVFP4** on Blackwell when kernel is mature for your shape and throughput matters most.

Don't answer "use the best one." The interviewer wants to hear the *conditional*: hardware, batch regime, prefill/decode mix, accuracy tolerance.

### 13.2 "When is INT4 a bad idea?"

- When activations are not also quantizable (you get memory but no compute win at large batch).
- When the model is small (<3B) — quality loss is disproportionate.
- When you are training (INT4 training is not a solved problem).
- When kernel support for your format + hardware doesn't exist (you get slower, not faster).
- When your deployment has long-context KV-sensitivity that propagates across layers.

### 13.3 "Why does quantization save memory but not always latency?"

Because latency = f(bandwidth, compute), not f(memory). Weight-only quant reduces weight-fetch bandwidth *and* storage — but at large batch, matmul compute is the bottleneck, and W4A16 still computes in FP16. So you saved memory but not compute-time.

### 13.4 "Why is training quantization harder than inference quantization?"

1. Gradients have wider dynamic range than weights or activations.
2. Weight updates are tiny — easy to underflow in low precision.
3. Errors compound over *steps* (thousands to millions), not just layers.
4. No fixed calibration set — the activation distribution shifts as training progresses.
5. Cost of failure is catastrophic (restart training) vs annoying (re-quantize and redeploy).

### 13.5 "What stays in higher precision and why?"

- Master weights (FP32) — preserve tiny update accumulation.
- Optimizer states (FP32 or BF16) — second-moment precision.
- Embeddings and lm_head — small, logit-sensitive.
- Normalization operators — stability and cheapness.
- Attention softmax/scores — distribution sensitivity.
- MoE gating — routing stability.
- Residual accumulators — error accumulation across layers.

### 13.6 "When does KV-cache quantization help?"

Any time decode bandwidth or KV memory is the bottleneck: long-context generation, high concurrency, large batch. It's a near-free win (INT8 or FP8 KV is almost drop-in for modern LLMs). INT4 KV is viable but needs per-head per-group scales and care at long context.

It does *not* help prefill or training — KV is written once in those regimes.

### 13.7 "How do hardware support and kernels affect the decision?"

Three rules:

1. Never pick a format the hardware can't run natively unless the kernel explicitly benchmarks faster than BF16 on your shapes.
2. For a given format, kernel maturity matters more than the format's theoretical advantage.
3. Decode and prefill regimes may prefer different formats in the same deployment; some stacks support per-phase configurations.

### 13.8 "What are the first things that usually break?"

In order:

1. Activation outliers crush per-tensor quantization → fix with SmoothQuant/AWQ/rotation or FP8 exponent range.
2. Specific sensitive layers (first, last, attention softmax) → keep in higher precision.
3. KV-cache quantization degrades long-context tasks → use per-token per-head scales or keep KV at FP8+.
4. Calibration mismatch shows up as task-specific degradation → match calibration distribution to production traffic, or use dynamic scales.
5. Gradient underflow stalls specific layers during training → stochastic rounding + FP8 E5M2 + per-tensor dynamic scaling.

---

## 14. Common mistakes

1. **Treating all low-bit formats as interchangeable.** INT4 ≠ FP4 ≠ NF4 ≠ MXFP4 ≠ NVFP4. They have different step distributions, different scale granularity, different hardware support, different accuracy behavior.
2. **Assuming lower precision always speeds things up.** Only if a native kernel exists for that precision on your hardware. Otherwise you pay dequant overhead and run at the higher precision's throughput.
3. **Ignoring activation quantization difficulty.** Most of the engineering effort in LLM quantization is outlier mitigation for activations, not weights.
4. **Ignoring kernel availability.** A format that works perfectly in a paper but has no fused GEMM for your hardware = slower deployment.
5. **Ignoring accumulator precision.** The DeepSeek 14-bit TC issue surprised many practitioners. Always check what precision your matmul accumulates at, especially for long K dimensions.
6. **Handwaving calibration.** A 128-sample calibration on Wikipedia doesn't give you code/multilingual/chat robustness. Match calibration to serving distribution or use dynamic activation scales.
7. **Assuming PTQ success on one benchmark means production safety.** MMLU scoring identically doesn't mean all downstream tasks work — some (long-context retrieval, tool use, niche domains) degrade before perplexity does.
8. **Confusing QLoRA with general inference quantization.** QLoRA is a finetuning recipe. NF4 is primarily a storage format. Serving typically uses AWQ/GPTQ/FP8, not NF4, for speed reasons.
9. **Ignoring long-context and decode-path effects.** Quality at 4K does not imply quality at 128K for the same quant. KV quantization errors accumulate along context length.
10. **Quantizing lm_head without testing.** A subtle logit shift can destabilize top-k sampling for tasks with tight top-k distinctions (e.g., MMLU).
11. **Using per-tensor scaling when per-channel or per-block is cheap.** Per-channel weight scaling costs almost nothing and is almost always better. Per-block is the only viable option at ≤4 bits.
12. **Round-to-nearest in training.** Biased rounding in gradients leads to slow divergence. Use stochastic rounding for gradient and activation quantization in low-precision training.

---

## 15. Final cheat sheet

### 15.1 PTQ vs QAT vs low-precision training

| Dimension | PTQ | QAT | Low-precision training |
|---|---|---|---|
| Starts from | pretrained BF16 model | pretrained BF16 model | scratch or continued PT |
| Gradients involved | no | yes (fake-quant) | yes (real low-prec) |
| Compute cost | cheap (~hours) | medium (~fine-tune) | full training cost |
| Quality potential | good | better (recover PTQ loss) | matches BF16 with care |
| Primary goal | deploy smaller/faster | recover accuracy | train faster + cheaper |
| Typical tools | AWQ, GPTQ, SmoothQuant | LLM-QAT, OmniQuant | Transformer Engine, DeepSeek FP8 recipe |
| Common pitfall | calibration mismatch | hyperparameter tuning | gradient underflow |

### 15.2 INT8 vs INT4 vs FP8 vs modern FP4-family

| Format | Bits | Hardware (native) | Range | Best for | Gotcha |
|---|---|---|---|---|---|
| INT8 | 8 | A100+, all | ±127 uniform | mature W8A8 inference | activations need SmoothQuant |
| FP8 E4M3 | 8 | H100+, B200+ | ±448 | W8A8 inference, FP8 forward | narrower range than E5M2 |
| FP8 E5M2 | 8 | H100+, B200+ | ±57344 | gradients in training | less mantissa precision |
| INT4 | 4 | weight-only (Marlin) | ±7 uniform | weight-only W4A16 | no A100+ activation support |
| NF4 | 4 | weight-only (bnb) | Gaussian quantiles | QLoRA finetuning | not a compute format |
| MXFP4 | 4+0.25 | B200+ | block-scaled E2M1 | cross-vendor W4A4 future | coarse E8M0 scale |
| NVFP4 | 4+0.5 | B200+ | finer block (16) + E4M3 scale | Blackwell W4A4 | NVIDIA-specific |

### 15.3 Weight-only vs W8A8 vs deeper low-bit schemes

| Scheme | What's quantized | Wins | Losses | When to use |
|---|---|---|---|---|
| BF16/FP16 | nothing | baseline stability | no savings | debugging, small models |
| W8A16 | weights to 8-bit | 2× storage/bandwidth | no compute win | simplest drop-in |
| W4A16 (AWQ/GPTQ) | weights to 4-bit | 4× storage/bandwidth | still FP16 matmul | decode / small batch |
| W8A8 INT8 (SmoothQuant) | both to INT8 | 2× storage/compute | needs calibration | Ampere throughput |
| W8A8 FP8 | both to FP8 E4M3 | 2× storage/compute | needs FP8 HW | Hopper+ production |
| W4A4 NVFP4 | both to NVFP4 | 4× storage, ~2× vs FP8 compute | kernel maturity | Blackwell prefill |
| KV FP8 / INT8 | KV cache | 2× decode BW / capacity | slight long-ctx risk | add-on, almost always |

### 15.4 Decision framework

```
START
 │
 ├─ Training or inference?
 │   │
 │   ├─ TRAINING:
 │   │   ├─ <10B params, debugging: BF16 + FP32 master
 │   │   ├─ 10B–100B: BF16 + bnb 8-bit Adam for optimizer state
 │   │   └─ 100B+: FP8 mixed-precision (DeepSeek-style) + optimizer state compression
 │   │
 │   └─ INFERENCE:
 │       │
 │       ├─ Hardware?
 │       │   ├─ Consumer GPU (Ampere, 4090): W4A16 AWQ/GPTQ + FP16 KV
 │       │   ├─ A100 server: W8A8 INT8 SmoothQuant + INT8 KV
 │       │   ├─ H100/H200 server: W8A8 FP8 + FP8 KV
 │       │   └─ B200/B300 server: NVFP4 W4A16 (or W4A4 if kernels mature) + FP8 KV
 │       │
 │       └─ Workload?
 │           ├─ Decode-heavy, small batch: prioritize weight-only + KV quant
 │           ├─ Prefill-heavy, large batch: prioritize W8A8 / W4A4 (compute)
 │           └─ Mixed: per-phase precision config
```

### 15.5 Fifteen likely interview questions with short strong answers

**Q1: What's the difference between per-tensor, per-channel, and per-group scaling?**
One scale per tensor / one per row (or col) / one per fixed block (e.g., 32 or 128 contiguous elements). Per-channel is nearly free in GEMM. Per-block is essential for ≤4-bit. Per-tensor fails on outliers at low bits.

**Q2: Why FP8 over INT8 for modern inference?**
FP8 E4M3's exponent bits give it ±448 range vs INT8's ±127, absorbing outliers without calibration effort. FP8 distributions are log-spaced, matching LLM weight/activation distributions better. Hopper and newer have identical FP8 and INT8 throughput, so FP8 is strictly better engineering.

**Q3: Why is W4A16 good for decode but not prefill?**
Decode at small batch is bandwidth-bound on weight fetches — 4-bit weights → 4× smaller fetch → near-linear speedup. Prefill at large batch is compute-bound — W4A16 still runs the matmul in FP16, so no compute win.

**Q4: What does SmoothQuant do conceptually?**
Migrates the "difficulty" of quantization from activations (which have outliers) to weights (which don't). Applies a mathematically-equivalent per-channel scaling that divides activations and multiplies weights, balancing the amaxes so both become INT8-friendly.

**Q5: What's special about NF4 vs INT4?**
NF4 uses 16 non-uniform codes corresponding to the quantiles of a standard normal distribution. Since pretrained weights are approximately Gaussian, NF4 is near information-theoretically optimal for weight storage — better than uniform INT4 for this distribution.

**Q6: Explain microscaling (MX) formats in one sentence.**
Block of 32 elements (or 16 for NVFP4) shares one small-format scale (E8M0 for MX, E4M3 for NVFP4), giving most of the accuracy of FP16 at 4-bit storage with ~0.25–0.5 bits of scale overhead per element.

**Q7: Why does KV-cache quantization help decode but not prefill?**
Decode reads the entire KV cache every step (once per token generated). Prefill writes the KV once and reads it once for the same computation. Quantizing saves the O(S × decode_steps) decode reads; the one-shot prefill work is unaffected.

**Q8: Why is the accumulator precision of tensor cores relevant?**
Low-precision MMA accumulates thousands of products. If the internal accumulator has limited precision (Hopper FP8 ≈ 14-bit), error accumulates over long K dimensions. DeepSeek-V3 periodically promoted partial sums to FP32 CUDA-core registers to fix this.

**Q9: What is stochastic rounding and why is it used in training?**
Round x to ⌊x⌋ with probability 1-frac(x) and to ⌈x⌉ with probability frac(x). This makes the quantized value an unbiased estimator of x. Biased (RtN) rounding causes slow divergence when accumulated over millions of training steps; SR eliminates that bias. Used in gradient and activation quant during low-precision training.

**Q10: Why does "quantized model" not imply "fast model"?**
Speedup requires (a) a kernel that exploits the format, (b) native hardware support for that format (or extremely efficient dequant), and (c) the workload to be in the regime the quantization targets (bandwidth-bound for weight-only, compute-bound for weight+activation). Miss any and you get a smaller model that isn't faster.

**Q11: What breaks when you quantize the softmax or attention scores?**
Attention post-softmax weights are highly concentrated (one or a few tokens dominate). Low-bit quantization can't represent both the dominant values and the long tail of small values. The small values get rounded to zero, distorting the attention distribution. Standard practice: softmax stays in BF16/FP32.

**Q12: What's the difference between mixed precision training and quantized training?**
Mixed precision runs *math* in BF16/FP16 but keeps master weights and updates in FP32 — it's about speeding up computation, not reducing storage of weights. Quantized training goes further: the weights themselves are stored in low precision (FP8, MXFP8, NVFP4), optimizer states may also be compressed, and comms are in low precision. DeepSeek-V3's FP8 training is the latter.

**Q13: Why does DeepSeek-V3 keep embeddings, lm_head, norms, attention, and MoE gating in higher precision?**
Small in parameter count so little savings; high in sensitivity (logit stability, normalization stability, routing stability). The compute cost of leaving them in BF16 is negligible against the gain from making those components reliable. This is the canonical "selective precision retention" pattern.

**Q14: When would you use INT4 KV cache?**
When KV cache memory is a hard capacity constraint (fitting long contexts or high concurrency on limited-memory GPUs) and the workload is not heavily long-context sensitive. Need per-token per-head scales. Most production deployments with KV quant use INT8 or FP8 because INT4 KV's long-context degradation is hard to predict across task mixes.

**Q15: If I give you a new model and Blackwell hardware, how do you decide the quantization config?**
Start with a baseline: BF16 reference for quality. Measure the workload mix — prefill/decode ratio, context length distribution, batch distribution. For decode-heavy, try NVFP4 W4A16 + FP8 KV first (safest quality-wise). For prefill-heavy or mixed, try FP8 W8A8 + FP8 KV as a mature first step, then evaluate NVFP4 W4A4 as the kernel-support matures. Validate on a downstream task suite that matches production, not just MMLU. Keep lm_head, embeddings, norms, attention softmax, MoE gating at BF16 regardless. Expect ~1–2% accuracy loss as the budget; if exceeded, the first thing to relax is KV cache precision, then activation precision, then weight precision.

---

## References (selected)

**Foundational papers**
- LLM.int8(): Dettmers et al., [arXiv:2208.07339](https://arxiv.org/abs/2208.07339)
- GPTQ: Frantar et al., [arXiv:2210.17323](https://arxiv.org/abs/2210.17323)
- SmoothQuant: Xiao et al., [arXiv:2211.10438](https://arxiv.org/abs/2211.10438)
- AWQ: Lin et al., [arXiv:2306.00978](https://arxiv.org/abs/2306.00978)
- QLoRA / NF4: Dettmers et al., [arXiv:2305.14314](https://arxiv.org/abs/2305.14314)
- 8-bit Adam: Dettmers et al., [arXiv:2110.02861](https://arxiv.org/abs/2110.02861)
- KVQuant: Hooper et al., [arXiv:2401.18079](https://arxiv.org/abs/2401.18079)
- QuaRot: Ashkboos et al., [arXiv:2404.00456](https://arxiv.org/abs/2404.00456)

**Formats and specifications**
- OCP 8-bit FP spec: [OFP8 v1.0](https://www.opencompute.org/documents/ocp-8-bit-floating-point-specification-ofp8-revision-1-0-2023-12-01-pdf-1)
- OCP Microscaling spec: [MX v1.0](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf)
- Microscaling Data Formats for DL (Rouhani et al.): [arXiv:2310.10537](https://arxiv.org/abs/2310.10537)

**Modern training and hardware**
- DeepSeek-V3 Technical Report: [arXiv:2412.19437](https://arxiv.org/html/2412.19437v1)
- DeepSeek-V3 hardware co-design paper (ISCA'25): [arXiv:2505.09343](https://arxiv.org/html/2505.09343v2)
- NVFP4 inference (NVIDIA dev blog): [developer.nvidia.com](https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/)
- NVFP4 training (NVIDIA dev blog): [developer.nvidia.com](https://developer.nvidia.com/blog/nvfp4-trains-with-precision-of-16-bit-and-speed-and-efficiency-of-4-bit/)
- Colfax DeepSeek FP8 analysis: [colfax-intl.com](https://research.colfax-intl.com/deepseek-r1-and-fp8-mixed-precision-training/)
- FP4 All the Way (fully quantized training): [OpenReview](https://openreview.net/pdf?id=kuzye4EPLR)

**Framework and kernel docs**
- vLLM quantization: [docs.vllm.ai](https://docs.vllm.ai/en/latest/features/quantization/)
- vLLM FP8 W8A8: [docs.vllm.ai/fp8](https://docs.vllm.ai/en/latest/features/quantization/fp8/)
- NVIDIA TransformerEngine: [docs.nvidia.com](https://docs.nvidia.com/deeplearning/transformer-engine/)
- CUTLASS: [github.com/NVIDIA/cutlass](https://github.com/NVIDIA/cutlass)
- Microsoft microxcaling (MX emulation): [github.com/microsoft/microxcaling](https://github.com/microsoft/microxcaling)