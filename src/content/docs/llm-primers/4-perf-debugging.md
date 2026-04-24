---
title: Perf Debugging
description: Perf Debugging
---
> A dense working reference. No throat-clearing, no pep talk. Memorize the numbers in the tables. Redraw the diagrams on a whiteboard until they are muscle memory. Every section ends with what a staff candidate should be able to produce live.

---

## Table of Contents

- [Part 1 — Hardware foundations](#part-1--hardware-foundations)
  - [1.1 The modern accelerator zoo](#11-the-modern-accelerator-zoo)
  - [1.2 Memory hierarchy deep dive](#12-memory-hierarchy-deep-dive)
  - [1.3 The roofline model](#13-the-roofline-model)
  - [1.4 Numerics](#14-numerics)
- [Part 2 — Kernel-level performance and debugging](#part-2--kernel-level-performance-and-debugging)
  - [2.1 The GEMM hierarchy](#21-the-gemm-hierarchy)
  - [2.2 Profiler fluency](#22-profiler-fluency)
  - [2.3 Warp stall reasons — the full taxonomy](#23-warp-stall-reasons--the-full-taxonomy)
  - [2.4 Occupancy is a symptom](#24-occupancy-is-a-symptom-not-a-goal)
  - [2.5 Coalescing, bank conflicts, swizzling](#25-memory-coalescing-bank-conflicts-swizzling)
  - [2.6 Async copy and software pipelining](#26-async-copy-and-software-pipelining)
  - [2.7 Triton performance gotchas](#27-triton-performance-gotchas)
  - [2.8 CuTe DSL overview](#28-cute-dsl-overview)
  - [2.9 Fusion economics](#29-fusion-economics)
- [Part 3 — Distributed training performance](#part-3--distributed-training-performance)
  - [3.1 Parallelism taxonomy](#31-parallelism-taxonomy)
  - [3.2 Collective communication deep dive](#32-collective-communication-deep-dive)
  - [3.3 FSDP2 internals](#33-fsdp2-internals)
  - [3.4 Pipeline parallelism schedules](#34-pipeline-parallelism-schedules)
  - [3.5 Tensor parallelism](#35-tensor-parallelism)
  - [3.6 Expert parallelism for MoE](#36-expert-parallelism-for-moe)
  - [3.7 Overlap](#37-overlap)
  - [3.8 Debugging distributed failures](#38-debugging-distributed-failures)
  - [3.9 MFU vs HFU vs MBU](#39-mfu-vs-hfu-vs-mbu)
- [Part 4 — LLM inference performance](#part-4--llm-inference-performance)
  - [4.1 Prefill vs decode](#41-prefill-vs-decode-dichotomy)
  - [4.2 KV cache management](#42-kv-cache-management)
  - [4.3 Continuous batching](#43-continuous-batching)
  - [4.4 Chunked prefill and P/D disaggregation](#44-chunked-prefill-and-prefilldecode-disaggregation)
  - [4.5 Speculative decoding](#45-speculative-decoding-families)
  - [4.6 Tensor parallel inference](#46-tensor-parallel-inference)
  - [4.7 Quantization for inference](#47-quantization-for-inference)
  - [4.8 Long-context inference](#48-long-context-inference-specifically)
  - [4.9 Serving system architecture](#49-serving-system-architecture)
- [Part 5 — Debugging methodology](#part-5--debugging-methodology)
  - [5.1 The diagnosis ladder](#51-the-diagnosis-ladder)
  - [5.2 Bisection under distributed conditions](#52-bisection-under-distributed-conditions)
  - [5.3 Reproducing rare failures](#53-reproducing-rare-failures)
  - [5.4 Numerics debugging](#54-numerics-debugging)
  - [5.5 The hardest bug categories](#55-the-hardest-bug-categories)
- [Part 6 — Interview-specific preparation](#part-6--interview-specific-preparation)
  - [6.1 The Perf & Debugging round format](#61-the-perf--debugging-round-format)
  - [6.2 Back-of-envelope drills](#62-back-of-envelope-fluency-drills)
  - [6.3 Narration patterns](#63-narration-patterns)
  - [6.4 When you don't know](#64-when-you-dont-know)
  - [6.5 Pushback management](#65-pushback-management)
- [Part 7 — Frontier topics](#part-7--frontier-topics)
  - [7.1 Blackwell-specific](#71-blackwell-specific)
  - [7.2 MI300X/MI350 considerations](#72-mi300xmi350-considerations)
  - [7.3 Scaling laws for inference](#73-scaling-laws-for-inference)
  - [7.4 The economics layer](#74-the-economics-layer)
  - [7.5 Recent papers worth knowing](#75-recent-papers-worth-knowing)

---

# Part 1 — Hardware foundations

The ceiling of any perf work is set by silicon. You cannot optimize past the roofline, and the roofline is determined by four numbers per chip: peak FLOPs at the relevant numeric format, HBM bandwidth, interconnect bandwidth per tier, and memory capacity. Everything else is friction between you and those four numbers.

## 1.1 The modern accelerator zoo

### Comparison table (dense tensor-core FLOPs; no sparsity multiplier unless noted)

| Chip       | Process | HBM (GB) | HBM BW (TB/s) | FP32 (TF) | BF16/FP16 (TF) | FP8 (TF) | FP4 (TF) | Interconnect                           | Notable                                  |
|------------|---------|----------|---------------|-----------|-----------------|----------|-----------|----------------------------------------|------------------------------------------|
| H100 SXM   | 4N      | 80       | 3.35          | 67        | 989             | 1979     | —         | NVLink 4, 900 GB/s bidir per GPU       | Hopper, TMA, WGMMA, async-proxy          |
| H200 SXM   | 4N      | 141      | 4.8           | 67        | 989             | 1979     | —         | NVLink 4, 900 GB/s                     | Same compute as H100, more HBM3e         |
| B100       | 4NP     | 192      | ~8.0          | 60        | 1800            | 3500     | 7000      | NVLink 5, 1800 GB/s                    | Lower-TDP Blackwell variant              |
| B200       | 4NP     | 192      | 8.0           | 80        | 2250            | 4500     | 9000      | NVLink 5, 1800 GB/s                    | Dual-die via NV-HBI; TMEM; CTA pairs     |
| GB200 NVL72| 4NP     | 13,824   | —             | —         | 162,000         | 324,000  | 648,000   | 72 B200 in one NVLink domain, 130 TB/s | Rack-scale coherent interconnect         |
| MI300X     | N5/N6   | 192      | 5.3           | 163       | 1307            | 2615     | —         | Infinity Fabric 896 GB/s (8-GPU ring)  | 8 XCDs, 304 CUs                          |
| MI325X     | N5/N6   | 256      | ~6.0          | 163       | 1307            | 2615     | —         | Infinity Fabric                        | Capacity refresh of MI300X               |
| MI350X     | N3      | 288      | 8.0           | —         | 2300            | 4600     | 9200      | Infinity Fabric 1075 GB/s              | CDNA4, FP4/FP6 support                   |
| TPU v5p    | —       | 95       | 2.76          | —         | 459             | 918      | —         | ICI 4.8 Tb/s per chip (3D torus)       | 8960 chips / pod max                     |
| TPU v6e    | —       | 32       | 1.64          | —         | 918             | 1836     | —         | ICI 3.58 Tb/s (2D torus)               | Inference-tuned "Trillium"               |
| Trainium2  | —       | 96       | 2.9           | —         | 667             | 1299     | —         | NeuronLink                             | 2 Neuron cores × 8 tiles                 |

*Numbers are vendor-reported dense tensor-core throughput at typical clocks; sparsity-enabled peaks are ~2× higher but rarely achievable on real workloads. Treat everything to 2 sig figs.*

### B200 SM block diagram (what's new)

```
╔══════════════════════════════════════════════════════════════════════╗
║                       Blackwell SM (one of 148)                      ║
║                                                                      ║
║   ┌───────────────────────────────────────────────────────────────┐  ║
║   │  4 × Sub-partitions (warp schedulers)                         │  ║
║   │  ┌────────┐  ┌────────┐  ┌────────┐  ┌────────┐               │  ║
║   │  │ 16K reg│  │ 16K reg│  │ 16K reg│  │ 16K reg│               │  ║
║   │  │ + INT  │  │ + INT  │  │ + INT  │  │ + INT  │               │  ║
║   │  │ + FP32 │  │ + FP32 │  │ + FP32 │  │ + FP32 │               │  ║
║   │  │ + FP64 │  │ + FP64 │  │ + FP64 │  │ + FP64 │               │  ║
║   │  └───┬────┘  └───┬────┘  └───┬────┘  └───┬────┘               │  ║
║   └──────┼──────────┼───────────┼───────────┼───────────────────  │  ║
║          ▼          ▼           ▼           ▼                        ║
║   ╔════════════════════════════════════════════════════╗             ║
║   ║  5th-gen Tensor Core   (FP4 / FP6 / FP8 / BF16)    ║             ║
║   ║  TCGEN05  — operates on Tensor Memory               ║             ║
║   ╚════════════════════════════════════════════════════╝             ║
║                               │                                      ║
║         ┌─────────────────────┼───────────────────┐                  ║
║         ▼                     ▼                   ▼                  ║
║   ┌───────────┐       ┌───────────────┐    ┌──────────────┐          ║
║   │ SMEM 228K │       │ TMEM 256 KiB  │    │ TMA engine   │          ║
║   │ (shared)  │       │ (tensor mem)  │    │ (5D copies)  │          ║
║   └─────┬─────┘       └───────┬───────┘    └──────┬───────┘          ║
║         │                     │                   │                  ║
║         └──────────┬──────────┴───────────────────┘                  ║
║                    ▼                                                 ║
║               ┌─────────┐                                            ║
║               │ L1 / tex │                                           ║
║               └────┬────┘                                            ║
╚════════════════════│═════════════════════════════════════════════════╝
                     ▼
          L2 ~60 MB (shared across all SMs)
                     │
                     ▼
          HBM3e 192 GB @ 8.0 TB/s
```

The three things that matter versus Hopper:

1. **Tensor memory (TMEM)** — a 256 KiB per-SM scratchpad that lives between SMEM and the tensor core. Accumulators no longer sit in registers; they sit in TMEM. This frees register pressure and lets you keep larger tiles.
2. **CTA pairs** — two CTAs can be scheduled such that their SMs share a tensor-core issue. The `tcgen05.mma` instruction can span a pair. This lets one logical GEMM tile be twice as large without doubling register pressure.
3. **FP4/FP6 tensor cores with block-scaled microscaling (MX format)** support — the hardware natively handles per-block scale factors at group sizes of 32 (NVFP4) or 16 (MXFP4), which is what makes FP4 training tractable.

### Things to memorize per chip

- **H100**: 989 BF16 TF, 1979 FP8 TF, 3.35 TB/s HBM, 900 GB/s NVLink per GPU bidir. 80 GB.
- **B200**: 2250 BF16 TF, 4500 FP8 TF, 9000 FP4 TF, 8.0 TB/s HBM, 1800 GB/s NVLink per GPU. 192 GB.
- **MI300X**: 1307 BF16 TF, 2615 FP8 TF, 5.3 TB/s HBM, 896 GB/s Infinity Fabric. 192 GB.
- **TPU v5p**: 459 BF16 TF, 2.76 TB/s HBM, 4.8 Tb/s ICI. 95 GB.

Drill: someone says "405B dense model, BF16, how many H100s minimum for inference at 8k context, batch 1"? 405 × 2 = 810 GB weights alone → ≥ 11 H100 for weights, and you need headroom for KV + activations → round up to 16 (TP=8, PP=2, or TP=16 if feasible).

## 1.2 Memory hierarchy deep dive

The single most important internalization is the ratio between levels. Bandwidth drops ~2 orders of magnitude each step outward; latency climbs ~2 orders of magnitude each step outward. Arithmetic intensity requirements track this directly.

### H100 / B200 hierarchy with numbers

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     SM-local (per-SM, per-thread)                       │
│                                                                         │
│    Registers            ~250 KiB/SM    ~40 TB/s       ~1 cycle          │
│    ├── FP32 register file (256 KB on B200)                              │
│    │                                                                    │
│    SMEM / L1            228 KiB/SM     ~20 TB/s       ~20–30 cycles     │
│    ├── Configurable split with L1 cache                                 │
│    │                                                                    │
│    TMEM (B200 only)     256 KiB/SM     dedicated      TC-local          │
│                                                                         │
└──────────────────────────────┬──────────────────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────────────────┐
│                         Chip-global                                     │
│                                                                         │
│    L2 cache             60 MB (H100: 50 MB)  ~5 TB/s   ~200 cycles      │
│    │                                                                    │
└──────────────────────────────┬──────────────────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────────────────┐
│                              HBM                                        │
│                                                                         │
│    H100: 80 GB @ 3.35 TB/s                              ~450 ns         │
│    H200: 141 GB @ 4.8 TB/s                                              │
│    B200: 192 GB @ 8.0 TB/s                                              │
│                                                                         │
└──────────────────────────────┬──────────────────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────────────────┐
│                       NVLink (intra-node)                               │
│                                                                         │
│    NVLink 4 (H100): 900 GB/s per-GPU bidir       ~1 μs                  │
│    NVLink 5 (B200): 1800 GB/s per-GPU bidir                             │
│    NVLink Switch fabric within NVL72: 130 TB/s all-to-all               │
│                                                                         │
└──────────────────────────────┬──────────────────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────────────────┐
│                       InfiniBand / Ethernet (inter-node)                │
│                                                                         │
│    NDR IB: 400 Gb/s = 50 GB/s per link          ~2–10 μs                │
│    XDR IB: 800 Gb/s = 100 GB/s per link                                 │
│    Typical 8× per node → 400 GB/s aggregate                             │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### The ratios that matter

| Tier transition    | H100 BW ratio | B200 BW ratio | Implication                                          |
|--------------------|---------------|---------------|------------------------------------------------------|
| SMEM / HBM         | ~6×           | ~2.5×         | Tiling still helps; B200 closes the gap somewhat     |
| HBM / NVLink       | ~3.7×         | ~4.4×         | Intra-node comm must be hidden behind HBM traffic    |
| NVLink / IB (8×400)| ~1.8×         | ~3.6×         | Inter-node collectives need hierarchical algorithms  |

### Hopper → Blackwell hierarchy changes

- **Tensor memory (TMEM)** inserts a new tier *between* SMEM and the tensor core. On Hopper, the WGMMA accumulator lived in registers. On Blackwell, `tcgen05.mma` reads A from SMEM, B from SMEM (or TMEM), and writes C to TMEM. This decouples tile size from register pressure.
- **CTA pairs**: two neighboring SMs can cooperate on a single `tcgen05.mma` issue. The effective tile the tensor core operates on doubles along one dimension (typically M).
- **TMA async copies** existed on Hopper but Blackwell adds TMA multicast refinements and 5D descriptor support useful for attention KV blocks.

The simplest mental model: on Hopper, a well-written FP8 GEMM tile is something like 128×128×64 per CTA, and the accumulator is 64 registers × 32 threads × ... bounded by 256 KB register file. On Blackwell, you can push to 256×256×64 per CTA pair with the accumulator in TMEM.

## 1.3 The roofline model

### The canonical equation

$$
\text{Performance (FLOP/s)} = \min\left(\text{Peak FLOP/s},\; \text{Arithmetic Intensity} \times \text{Peak BW}\right)
$$

where arithmetic intensity *AI* = FLOPs / bytes moved from the operand source (typically HBM).

The roofline's "ridge point" — the AI at which you transition from memory-bound to compute-bound — is `Peak FLOP/s ÷ Peak BW`.

### Ridge points by format and chip

| Chip  | Format | Peak (TF/s) | BW (TB/s) | Ridge (FLOP/byte) |
|-------|--------|-------------|-----------|-------------------|
| H100  | BF16   | 989         | 3.35      | **295**           |
| H100  | FP8    | 1979        | 3.35      | **591**           |
| B200  | BF16   | 2250        | 8.0       | **281**           |
| B200  | FP8    | 4500        | 8.0       | **562**           |
| B200  | FP4    | 9000        | 8.0       | **1125**          |
| MI300X| BF16   | 1307        | 5.3       | **247**           |
| MI300X| FP8    | 2615        | 5.3       | **493**           |

### What this means operationally

- A **dense matmul** `C += A @ B` with M=N=K has AI = `2MNK / (2(MN+MK+NK) × bytes)`. For square matrices that's ~`K/3` in elements, or scaled by bytes. **AI grows linearly with K.** For FP8 on H100, you need K large enough that `K/3 ≈ 591 bytes`, i.e. K around ~1800 with FP8 operands for the AI to cross the ridge.
- **Decode attention** on a single query has AI close to 1 (you load the entire KV cache and do ~2× FLOPs per byte). It is *deeply* memory-bound and no amount of tensor core improvement helps.
- **Prefill attention** has AI = O(S) where S is the sequence length of the prefill chunk. Long prefill is compute-bound on H100 beyond ~1k context, easily.
- **FP4 pushed the ridge up** so aggressively that on B200, anything short of huge tiled GEMMs is bandwidth-bound. This is why FP4 training is only attractive for very large, very long training steps.

### Roofline plot, annotated

```
     FLOP/s
(log scale)
       ▲
  Peak │━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  FP8 peak (1979 TF)  H100
  FP8  │         ┌───────────────────────────────
       │       ╱
       │     ╱  <- compute-bound region
       │   ╱
  Peak │━╱━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━   BF16 peak (989 TF)
  BF16 │╱
       │                                           Workloads:
       │   [GEMV / decode attn]  AI≈1   memory-bound
       │   [LayerNorm / RMSNorm] AI≈2   memory-bound
       │   [FlashAttn decode]    AI≈4–8 memory-bound
       │   [Small GEMMs, K=256]  AI≈100 memory-bound at FP8
       │   [Prefill, S≥2k]       AI≈500 compute-bound at FP8
       │   [Large dense GEMM]    AI≈2000 firmly compute-bound
       │
       └──────────────────────────────────────────────────────▶
       1    10    100   1000  10000                  AI (FLOP/byte)
                        ▲
                        │
                   Ridge for FP8 H100 (~591 FLOP/B)
```

The practical taxonomy a staff candidate should recite:

- **Prefill**: compute-bound for any interesting context length. Optimize for tensor core utilization. MFU is the right metric.
- **Decode**: memory-bound until batch size is large. Optimize for HBM BW utilization. MBU is the right metric. Batching multiplies effective AI by batch size up until KV cache pressure or scheduler overhead kicks in.
- **Activation-only ops** (norms, elementwise, rotary, softmax in isolation): always memory-bound. Fuse them into adjacent kernels or live with it.

## 1.4 Numerics

### Format reference

| Format    | Exp | Mant | E-max | E-min | Smallest norm | Largest finite | Machine ε (ULP@1) |
|-----------|-----|------|-------|-------|---------------|----------------|-------------------|
| FP32      | 8   | 23   | 127   | −126  | 1.18e-38      | 3.40e38        | 1.19e-7           |
| TF32*     | 8   | 10   | 127   | −126  | 1.18e-38      | 3.40e38        | 9.77e-4           |
| BF16      | 8   | 7    | 127   | −126  | 1.18e-38      | 3.39e38        | 7.81e-3           |
| FP16      | 5   | 10   | 15    | −14   | 6.10e-5       | 65504          | 9.77e-4           |
| FP8 E4M3  | 4   | 3    | 8     | −6    | 1.56e-2       | 448            | 0.125             |
| FP8 E5M2  | 5   | 2    | 15    | −14   | 6.10e-5       | 57344          | 0.25              |
| MXFP8 E4M3| 4   | 3    | block | block | block-scaled  | block-scaled   | 0.125 relative    |
| MXFP4 E2M1| 2   | 1    | 2     | −1    | 0.5           | 6.0            | 0.5               |
| NVFP4 E2M1| 2   | 1    | block | block | block-scaled  | block-scaled   | 0.5 relative      |
| INT8      | —   | —    | —     | —     | —             | 127            | 1.0               |

*TF32 is FP32-exponent with FP16-mantissa, used only as an accumulator format on tensor cores.

### Where each format is appropriate

- **FP32**: master weights in mixed precision training, loss scaling pivots, LayerNorm accumulation, softmax denominator in attention.
- **BF16**: the default training compute format for activations and weights. Same range as FP32, lower precision — "free" for most LLM training workloads because the gradients are well-conditioned.
- **FP16**: legacy. Range issues (65504 ceiling) required loss scaling. Displaced by BF16 for training. Still relevant for some inference paths on older hardware.
- **FP8 E4M3**: forward activations and weights. More precision (7 levels in mantissa, range ±448 when unscaled; per-tensor scales make this work).
- **FP8 E5M2**: backward gradients. Wider range is necessary because gradients have long tails; less precision acceptable because grads are noisier anyway.
- **MXFP8 / MXFP4 / NVFP4**: block-scaled microscaling. A block of 32 (or 16) values shares an E8M0 scale. Dramatically better dynamic range utilization than per-tensor FP8, which is why FP4 training is becoming viable.
- **INT8 / INT4 (weight-only)**: inference quantization. INT4 AWQ/GPTQ for weights with BF16 activations is a very common W4A16 combo.

### Dynamic range and why it matters

The "range" of a format is what values it can represent without saturation or underflow. E4M3 saturates at 448; E5M2 at 57344. If your activations have outliers at ±1000 and you use E4M3 without scaling, you clip. Per-tensor scale factors solve this by rescaling values into range, but only up to the granularity of one scale per tensor.

Block scaling (MX formats) gives you one scale per 32-value block, which is why it handles outliers ~100× better than per-tensor. This is what enables FP4 to work.

### ULP and attribution

A ULP (unit in the last place) is the smallest representable difference between adjacent floating-point values at a given magnitude. Memorize:

```
FP32:      2⁻²³ ≈ 1.19e-7
FP16:      2⁻¹⁰ ≈ 9.77e-4
BF16:      2⁻⁷  ≈ 7.81e-3
FP8 E4M3:  2⁻³  = 0.125
FP8 E5M2:  2⁻²  = 0.25
MXFP4:     2⁻¹  = 0.5   (within a block, relative to scale)
```

If two training runs diverge by more than a few thousand ULPs early in training, suspect a correctness bug, not a numerics difference. If they diverge by ~100 ULPs in the last layers, that's consistent with non-deterministic reduction order.

### Stochastic rounding

Deterministic round-to-nearest is biased for accumulation: repeatedly rounding tiny updates to zero stalls learning. Stochastic rounding rounds up or down with probability proportional to the fractional distance, making the expectation unbiased. Empirically this is necessary for FP4 training and helpful (not required) for FP8 training with small optimizer updates.

A common staff-level trap: in Adam-style optimizers, the second moment `v` is always positive and small updates round to zero in FP8 — breaking effective learning rate. Master weights in FP32 (or BF16) and stochastic rounding when writing back to FP8 are the standard workarounds.

---

# Part 2 — Kernel-level performance and debugging

## 2.1 The GEMM hierarchy

Every fast GEMM in the last decade is built the same way: nest tiles until each level fits the resource at that tier, and use async copies to keep everything busy.

### Tiling levels

```
 ┌───────────────────────────────────────────────────────────────────────┐
 │ Global GEMM:  C[M,N] += A[M,K] × B[K,N]                               │
 │                                                                       │
 │    split M,N across thread blocks (CTAs)                              │
 │                                                                       │
 │  ┌─────────────────────────────────────────────────────────────────┐  │
 │  │ Thread block tile:  Cblk[Mb,Nb] += Ablk[Mb,K] × Bblk[K,Nb]      │  │
 │  │ Typical Hopper FP8:  Mb=128, Nb=128                             │  │
 │  │ Typical Blackwell:   Mb=256, Nb=256  (with CTA pair)            │  │
 │  │                                                                 │  │
 │  │   iterate K in Kb chunks (K / Kb stages async-pipelined)        │  │
 │  │                                                                 │  │
 │  │   ┌───────────────────────────────────────────────────────────┐ │  │
 │  │   │ Warp tile (4 warps × 32×32 typical on A100)               │ │  │
 │  │   │   ┌─────────────────────────────────────────────────────┐ │ │  │
 │  │   │   │ Register tile  (per-thread accum of e.g. 8×8 BF16)  │ │ │  │
 │  │   │   │   ↑                                                 │ │ │  │
 │  │   │   │   this is what the MMA instruction issues on        │ │ │  │
 │  │   │   │   (mma.m16n8k16 on Ampere, wgmma.m64n256k16 on H100)│ │ │  │
 │  │   │   └─────────────────────────────────────────────────────┘ │ │  │
 │  │   └───────────────────────────────────────────────────────────┘ │  │
 │  └─────────────────────────────────────────────────────────────────┘  │
 └───────────────────────────────────────────────────────────────────────┘
```

### Generation-by-generation evolution

**Ampere (A100)** — threads collectively issue `mma.m16n8k16`. A warp does a 16×8×16 MMA. Four warps per CTA assemble a 64×128 or 128×128 CTA tile. Async copies use `cp.async` to overlap global→shared loading with compute.

**Hopper (H100)** — `wgmma` (warpgroup MMA) lets 4 warps × 32 threads collectively issue one large MMA instruction, e.g. `wgmma.mma_async.sync.m64n256k16`. Crucially, one warpgroup is the producer (issues TMA copies), another is the consumer (issues wgmma). This is **warp specialization**.

```
 ┌────────────────────────── CTA on Hopper ──────────────────────────┐
 │                                                                   │
 │   Producer warpgroup (32 threads × 4 warps = 1 WG)                │
 │     │ Issues TMA loads of A, B blocks into SMEM buffers 0,1,2,3   │
 │     ▼                                                             │
 │   [ Ring of N SMEM buffers (mbarrier-gated) ]                     │
 │     ▲                                                             │
 │     │ Consumer warpgroups (2–3 WGs)                               │
 │     │ Issue wgmma from SMEM -> register accumulator               │
 │     │ mbarrier.arrive when buffer is consumed                     │
 │                                                                   │
 └───────────────────────────────────────────────────────────────────┘
```

**Blackwell (B200)** — `tcgen05.mma` with TMEM. Accumulator moves out of registers. CTA pair means two neighboring CTAs cooperate. Producer/consumer specialization is still the pattern but now the consumer is typically *one warp* issuing `tcgen05.mma`, with the rest of the block helping with TMA and TMEM staging.

### What CUTLASS does that you can't easily match by hand

- **Persistent kernels**: one CTA per SM, loops over output tiles internally rather than being re-launched. Eliminates launch overhead, improves cache reuse, and enables stream-K decomposition where work is split along K rather than output tiles (essential for small-M shapes like LLM decode).
- **Stream-K**: when M is small (say M=1 for decode), tiling output along MN underutilizes SMs. Stream-K tiles output but lets CTAs cooperate along K, each writing a partial, with a final reduction. CUTLASS implements this with atomic adds or a separate reduction kernel.
- **Epilogue fusion**: bias add, activation, quantization, scaling, residual adds — all fused into the tail of the GEMM kernel without an extra HBM roundtrip. Essential for performance on memory-bound tail ops.
- **Warp specialization**: producer/consumer as above, with mbarrier synchronization. CUTLASS handles the mbarrier topology so you don't have to.
- **Shape-specialized code paths**: different tile sizes selected per problem shape, with autotuning at compile time.

Hand-written CUDA catches up to CUTLASS only if you replicate all of the above. Triton catches up for most shapes but not for persistent warp-specialized kernels at the Hopper/Blackwell frontier.

## 2.2 Profiler fluency

You are expected to be able to read a Nsight Systems timeline and a Nsight Compute kernel report live in an interview. Know the top-priority metrics cold.

### Nsight Compute — metrics in priority order

| # | Metric                                           | What it tells you                                    | Acceptable target                 |
|---|--------------------------------------------------|------------------------------------------------------|-----------------------------------|
| 1 | `sm__cycles_active.avg.pct_of_peak_sustained...` | Fraction of time SMs are doing anything              | > 80% for compute-bound           |
| 2 | `smsp__average_warps_active...`                  | Warps resident per sub-partition                     | Depends; 8–12 typical             |
| 3 | `dram__throughput.avg.pct_of_peak_sustained_...` | HBM BW utilization                                   | > 80% for memory-bound kernels    |
| 4 | `sm__inst_executed_pipe_tensor.avg.pct...`       | Tensor core utilization                              | > 70% for matmul kernels          |
| 5 | Stall reasons (see 2.3)                          | Where warps are waiting                              | Diagnostic, not a target          |
| 6 | `l2__t_bytes...` / `l2_tex__...hit_rate`         | L2 hit rate                                          | Higher = more reuse (depends)     |
| 7 | `smsp__warps_issue_stalled_*` series             | Per-cause stall breakdown                            | Find the biggest; it's the bottleneck |
| 8 | Register count / spills                          | Register pressure                                    | No spills for hot kernels         |
| 9 | Occupancy (theoretical and achieved)             | How many warps could / do fit                        | See 2.4                           |

The lazy mental ordering: if SMs aren't active, either you're memory-bound (check dram%) or latency-bound (check stalls). If SMs are active but tensor pipe is idle, you're issuing non-tensor instructions (check for unnecessary conversions, elementwise epilogue taking too long).

### Nsight Systems — what a healthy timeline looks like

```
Healthy training step on H100 cluster:
                                                          time →
┌───────────────────────────────────────────────────────────────────────┐
│ CPU       │ cudaLaunchKernel ─────────────── many tiny gaps           │
│           │                                                           │
│ GPU kern  │ █████░██████████████░░░░██████████████████░░░░██          │
│           │  fwd  comm  fwd  comm  bwd    comm     bwd                │
│ NCCL      │      ████      ████    ████████      ████████             │
│           │                                                           │
│ PCIe/NVL  │      ═══       ═══     ═══════       ═══════              │
│                                                                       │
│ HBM use   │ ░░▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓░░         │
└───────────────────────────────────────────────────────────────────────┘

  Notes:
   - Kernel stream is tightly packed
   - NCCL overlaps with compute (comm bar runs during compute bar)
   - CPU only appears at kernel launches; no large gaps
```

### Dataloader-bound step

```
                                                          time →
┌───────────────────────────────────────────────────────────────────────┐
│ CPU       │ ████████████  <- dataloader working                       │
│           │             │                                             │
│ GPU kern  │             │██████████████        (GPU idle before here) │
│           │    IDLE     │  compute                                    │
│ NCCL      │             │     ████                                    │
│                                                                       │
│  Symptom: GPU idle at start of step, CPU busy                         │
│  Fix: larger prefetch, persistent workers, move preprocessing to GPU, │
│       pin_memory=True, non_blocking=True copies                       │
└───────────────────────────────────────────────────────────────────────┘
```

### Comm-bound step (exposed NCCL)

```
                                                          time →
┌───────────────────────────────────────────────────────────────────────┐
│ GPU kern  │ ██████                  ██████                            │
│           │  fwd    <- GPU IDLE ->   bwd                              │
│ NCCL      │       ████████████                                        │
│           │                                                           │
│  Symptom: compute gaps during NCCL; overlap not happening             │
│  Fix: async TP, FSDP prefetch, hierarchical collectives,              │
│       increase computation granularity so it covers collective        │
└───────────────────────────────────────────────────────────────────────┘
```

### Straggler step

```
Rank 0  │ ████████████      NCCL      ████████████
Rank 1  │ ████████████      NCCL      ████████████
Rank 2  │ ██████████████████ NCCL     ████████████    <- slow rank
Rank 3  │ ████████████      NCCL(wait) ████████████
                  ▲                  ▲
                  │                  │
             Rank 2 takes 20% longer; all ranks wait at collective
             
Symptom: perfectly clean timelines on 3 ranks, one rank's kernel bar is longer
Fix: Identify the slow rank. Likely causes: GPU throttling, bad node,
     uneven data (different seq lengths), ECC errors being retried.
```

### PyTorch Profiler and HTA

`torch.profiler.profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], with_stack=True, record_shapes=True)` emits a trace JSON loadable in chrome://tracing or Perfetto.

For distributed jobs, Meta's HTA (Holistic Trace Analysis) aggregates traces across ranks and surfaces:
- Communication/computation overlap percentage
- Kernel breakdown by category
- CUDA kernel launch statistics
- Idle time attribution

At staff level, know that HTA's comm-compute overlap metric is the single most useful summary for distributed training runs. Target: > 50% of comm time hidden, preferably 70%+.

## 2.3 Warp stall reasons — the full taxonomy

Nsight Compute's "warp state statistics" is where root-cause lives. Each stall reason names a specific hazard.

| Stall reason     | What's happening                                                               | Typical cause                                                  | Typical fix                                                    |
|------------------|---------------------------------------------------------------------------------|----------------------------------------------------------------|----------------------------------------------------------------|
| Long Scoreboard  | Warp waiting on global/shared memory load                                       | Memory-bound kernel, large read latency                        | Prefetch further, increase tile size, improve coalescing       |
| Short Scoreboard | Waiting on MIO (math I/O) — SFU, Tex, shared memory atomics                     | Heavy use of `__sinf`, `__expf`, shared atomics                | Reduce SFU pressure; use fused MMA epilogues; batch atomics    |
| Wait             | Waiting on fixed-latency instructions (MMA, others) to retire                   | Dependency chain on MMA                                        | More ILP; more warps in flight; larger tiles                   |
| IMC Miss         | Constant memory miss                                                            | Rare; large constant arrays                                    | Move data to `__constant__` or `__device__` + cached loads     |
| MIO Throttle     | MIO pipe saturated                                                              | Too many shared mem / SFU ops per cycle                        | Reduce shared mem pressure; rebalance                          |
| Tex Throttle     | Texture unit saturated                                                          | Heavy texture fetches (rare in LLM work)                       | Use LDG.CI or restructure                                      |
| Barrier          | Waiting at `__syncthreads()` or cluster barrier                                 | Imbalanced work across warps                                   | Rebalance; fewer syncs; warp specialization                    |
| Not Selected     | Eligible but scheduler picked another warp                                      | Benign — means you have enough warps                           | None — this is good                                            |
| Selected         | Currently issuing                                                               | Benign                                                         | None                                                           |
| No Instruction   | Instruction cache miss                                                          | Huge kernels, cold start                                       | Smaller kernel; warm up                                        |
| Drain            | At end of kernel, waiting for stores                                            | Usually benign                                                 | None                                                           |
| LG Throttle      | Local/Global memory pipeline throttled                                          | Bursty global ops                                              | Spread issue; reduce register spills (local memory)            |

### The usual suspects in LLM kernels

- **Long Scoreboard dominant** in a matmul → not enough pipeline depth. Either increase `num_stages` (Triton), add more `cp.async` buffers, or increase tile K.
- **Short Scoreboard dominant** in attention → softmax is thrashing SFU. Look at whether exp is using `__expf` or the MUFU path; consider replacing with faster approximations.
- **Wait dominant** with high tensor pipe utilization → you're compute-bound, which is the goal. Move on.
- **Barrier dominant** → something is imbalanced. In a warp-specialized kernel, check mbarrier setup.

## 2.4 Occupancy is a symptom, not a goal

The classic trap: candidate sees 25% occupancy, concludes "low occupancy, must increase it." Often wrong.

### When low occupancy is correct

- **Large tile kernels** with heavy register usage. A Hopper wgmma kernel with 128×128 tile in BF16 can use ~200 registers per thread. You get maybe 2–3 warpgroups per SM. That's ~12–24% occupancy but it's the performance-optimal config because tile size dominates.
- **ILP-heavy kernels** where each thread has enough independent work to hide latency without needing many warps. Volta-onward hardware schedules ILP aggressively.
- **Persistent kernels** — one CTA per SM, looping internally. Low occupancy by construction.

### When high occupancy is correct

- **Latency-bound kernels** where arithmetic intensity is low and the only way to hide memory latency is to have many warps in flight (elementwise, norms, simple reductions).
- **Kernels with variable-latency instructions** (lots of divides, transcendentals) where warp diversity hides the straggler.

### The Volta-onward ILP tradeoff

Pre-Volta, warps were the only way to hide latency (each warp had one instruction in flight at a time from the scheduler's POV, roughly). Volta added independent thread scheduling, and each thread can hide its own latency via ILP given enough registers.

Practical rule: if you have enough registers per thread to keep multiple pending memory/MMA operations in flight, more registers > more warps. This is why a 128-reg kernel often beats a 64-reg kernel at half the occupancy.

### Diagnosis flowchart

```
                 Kernel slow
                      │
                      ▼
             ┌──────────────────┐
             │  What's the bw / │
             │   compute util?  │
             └────────┬─────────┘
                      │
         ┌────────────┼────────────┐
         ▼            ▼            ▼
    BW > 80%    Compute > 80%   Neither
    (mem-bnd)    (good)         (latency-bound)
         │            │            │
         ▼            ▼            ▼
    Tile better,  Ship it.    Check stalls.
    prefetch.                  Long Scoreboard?
    Format↓?                   Wait with low tensor
                               pipe? Occupancy may
                               actually be the issue.
```

## 2.5 Memory coalescing, bank conflicts, swizzling

### Coalescing — global memory

A warp of 32 threads issues one memory transaction per aligned 128-byte cache line they collectively touch. Maximum throughput requires all 32 threads to access a single 128-byte-aligned contiguous region.

```
 COALESCED load of 32 × fp32:
   Thread:  0  1  2  3  ...  31
   Addr:    0  4  8  12 ...  124
           └────── 128 bytes, aligned ──────┘
   Result: 1 memory transaction

 UNCOALESCED (stride 2):
   Thread:  0  1  2  3  ...  31
   Addr:    0  8  16 24 ...  248
           └────────── 256 bytes ──────────┘
   Result: 2 transactions, 50% BW wasted

 UNCOALESCED (random):
   Thread:  0  1  2  3  ...  31
   Addr: 408 16 2044 ...
   Result: up to 32 transactions
```

LLM examples: rotary embedding implemented naively with per-head strided access kills coalescing. Solution: permute layout so consecutive threads handle consecutive elements along the innermost dim.

### Bank conflicts — shared memory

SMEM has 32 banks, each 4 bytes wide (configurable to 8 on newer GPUs). A conflict occurs when two threads in a warp address the same bank but different addresses in the same cycle.

```
 Layout: 32 banks, shared mem is laid out column-major per bank
 
 Addr offset (bytes): 0   4   8  ... 124 | 128 132 136 ... 252 | ...
 Bank:                0   1   2  ...  31 |  0   1   2 ...  31  | ...
 Row:                        0                     1
 
 SMEM[tid]        -> no conflict   (each thread one bank)
 SMEM[tid * 2]    -> 2-way conflict (threads 0,16 both hit bank 0)
 SMEM[tid * 4]    -> 4-way conflict
 SMEM[tid * 32]   -> 32-way conflict (all threads hit bank 0)

 Canonical broadcast (all read same addr) -> no conflict, broadcast unit
 Canonical 32-way -> *32x slowdown* on that load
```

### Swizzling

For matmul tiles, writing rows of A/B into SMEM then reading columns creates systematic bank conflicts. The fix is to **swizzle** the layout so that logical (row, col) maps to a physical address pattern that's conflict-free in both read and write patterns.

XOR swizzle (simple):

```
 physical_offset(row, col) = row * stride + (col XOR (row % N))
 
 For a 16×16 tile of fp16 (32 bytes per row), XOR with (row % 8) shifts:
   Row 0: cols 0,1,2,3,4,5,6,7
   Row 1: cols 1,0,3,2,5,4,7,6    (xor with 1)
   Row 2: cols 2,3,0,1,6,7,4,5    (xor with 2)
   ...
 
 A columnwise read now hits every bank once — no conflicts.
```

CUTLASS exposes these as layout primitives (`SwizzledSharedLayout`). In CuTe, you compose a `Swizzle<B,M,S>` with your atom layout.

## 2.6 Async copy and software pipelining

### Hardware evolution

- **Pre-Ampere**: global loads landed in registers then went to shared. Compute couldn't overlap with loads from the same warp easily.
- **Ampere (`cp.async`)**: load directly from HBM to SMEM, bypassing registers. Enables N-stage pipelines.
- **Hopper (TMA)**: a dedicated engine issues 1D–5D strided copies. The CTA just kicks off a TMA and waits on an mbarrier. Frees warp scheduler from address computation.
- **Hopper (TMA multicast)**: one TMA load delivers to up to 16 CTAs simultaneously (same data to a cluster).
- **Blackwell**: TMA persists; TMEM is the new staging buffer for tensor cores.

### Multi-stage pipeline anatomy

A 3-stage pipeline (`num_stages=3` in Triton):

```
                       time →
Stage buffer 0:   load─┤ compute─┤ (idle)
Stage buffer 1:        load─┤ compute─┤
Stage buffer 2:             load─┤ compute─┤
Stage buffer 0 (reuse):          load─┤ compute─┤
...

Each "load" runs ahead of the "compute" that consumes it by 2 iterations.
Compute and load overlap because they use different hardware units.
```

A 4-stage pipeline on Hopper with warp specialization:

```
 Producer WG:  ┤load 0├┤load 1├┤load 2├┤load 3├┤load 4├┤load 5├
 Consumer WG0: ─wait─┤ compute 0 ├┤ compute 2 ├┤ compute 4 ├
 Consumer WG1: ─wait─┤ compute 1 ├┤ compute 3 ├┤ compute 5 ├

 - Producer issues TMA loads 2 iterations ahead
 - Consumers alternate on even/odd K chunks
 - mbarrier signals producer when consumer done with buffer
```

### The Hopper producer/consumer pattern in detail

```
 ┌────────────────────────── CTA on Hopper ──────────────────────────┐
 │  Producer warpgroup (1 of 3 WGs)                                  │
 │  ┌───────────────────────────────────────────────────────────┐    │
 │  │  while (has work):                                         │    │
 │  │    arrive_and_wait(empty_mbarrier[stage])                  │    │
 │  │    tma.load(A[k,:,:], B[k,:,:]) → smem_buf[stage]          │    │
 │  │    arrive(full_mbarrier[stage], expect=bytes)              │    │
 │  │    stage = (stage + 1) % NUM_STAGES                        │    │
 │  └───────────────────────────────────────────────────────────┘    │
 │                                                                   │
 │  Consumer warpgroups (2–3 WGs)                                    │
 │  ┌───────────────────────────────────────────────────────────┐    │
 │  │  while (has work):                                         │    │
 │  │    wait(full_mbarrier[stage])                              │    │
 │  │    wgmma.mma_async(accum, smem_buf[stage])                 │    │
 │  │    wgmma.wait                                              │    │
 │  │    arrive(empty_mbarrier[stage])                           │    │
 │  │    stage = (stage + 1) % NUM_STAGES                        │    │
 │  └───────────────────────────────────────────────────────────┘    │
 └───────────────────────────────────────────────────────────────────┘
```

The mbarrier primitive is a 128-bit atomic counter in SMEM. `arrive` increments, `wait` spins until threshold met. The hardware fences loads into SMEM against the arrive count — when the count reaches expected bytes, the barrier flips.

## 2.7 Triton performance gotchas

Triton closes most of the gap to CUTLASS for "normal" shapes but has specific limitations staff candidates should know.

### Autotune space design

A well-configured Triton autotune looks like:

```python
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 32}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_stages=4, num_warps=8),
        # ... typically 6–12 configs
    ],
    key=['M', 'N', 'K'],
)
```

Traps:
- **Too many configs** → autotune takes minutes per unique shape key. Bound it.
- **`key=` too broad** → retunes for every shape. Round K down to power of 2 in the key if appropriate.
- **Missing the winning config** → common. If you know CUTLASS picks `BLOCK_M=256, num_stages=5` for a given shape, add that config explicitly.

### `num_stages` and `num_warps`

- `num_stages=2`: baseline pipelined load. Fine for compute-bound cases with simple kernels.
- `num_stages=3`: standard choice for Hopper matmul. Two stages loading while one computes.
- `num_stages=4–5`: useful when memory latency is high, e.g. attention with long sequence.
- `num_warps=4`: 128 threads per CTA, standard.
- `num_warps=8`: 256 threads, required for some Hopper wgmma shapes.

### When Triton can't match CUTLASS

- **Persistent warp-specialized kernels**: Triton 3.x is getting warp specialization but the producer/consumer pattern is still more fragile than CUTLASS. If you need absolute peak on Hopper, especially with TMA multicast and cluster-launched kernels, drop to CuTe or CUTLASS.
- **Stream-K**: Triton doesn't natively generate stream-K. You can emulate it with a split-K reduction but it's not the same as CUTLASS's cooperative CTAs.
- **Small-M / decode-shape GEMMs**: M=1, N=4096, K=4096. Triton's default scheduling underutilizes SMs because it tiles M,N. CUTLASS with stream-K wins. In Triton, you can manually split-K across CTAs, with atomic-add final reduction.
- **FP8 with block scales**: MXFP8/NVFP4 require per-block scale loads interleaved with operand loads. Triton supports this as of ~3.x but the performance isn't at CUTLASS level yet.

### Debugging with `TRITON_INTERPRET=1`

Runs the kernel as Python on CPU. Lets you `print(tl.load(...))` to see intermediate tensors. Slow but invaluable for correctness bugs.

Use `TRITON_CACHE_DIR=/tmp/triton_cache` to inspect generated PTX / cubin when you suspect the compiler is doing something unexpected.

```bash
# Dump Triton IR at various levels
TRITON_ALWAYS_COMPILE=1 TRITON_PRINT_TTIR=1 python my_kernel.py
# See the PTX
cat /tmp/triton_cache/*/my_kernel.ptx
```

### FP8 in Triton

As of Triton 3.x, `tl.float8e4nv` and `tl.float8e5` are supported on Hopper. Scale factors are handled as explicit tensors. Common patterns:

```python
# Per-tensor scale (simplest)
a_fp8 = tl.load(a_ptr + offsets).to(tl.float8e4nv)
b_fp8 = tl.load(b_ptr + offsets).to(tl.float8e4nv)
acc = tl.dot(a_fp8, b_fp8, out_dtype=tl.float32)
acc = acc * a_scale * b_scale  # rescale in the epilogue
```

For MXFP8 (block scales at 32), you load the scale tensor alongside and apply per-tile. This is where Triton's ergonomics get awkward compared to CuTe's layout-native composition.

## 2.8 CuTe DSL overview

CuTe (inside CUTLASS) is a layout algebra. You express tensors as a product of a pointer and a "layout" — a function from coordinates to offsets.

### Core abstractions

- **Layout**: a hierarchical tuple like `((4, 8), (2, 16))` meaning a 4-by-2 outer arrangement of 8-by-16 tiles. Maps logical (i, j) → memory offset.
- **Tensor**: pointer + layout. Algebraic operations (`local_tile`, `local_partition`, `composition`) produce sub-tensors.
- **Atom**: a hardware-specific primitive (e.g., an `SM90_64x64x16_F32F16F16F32_TN` is one wgmma instruction shape).
- **TiledMMA / TiledCopy**: compositions of atoms with layouts that describe how threads cooperate.

### When to drop from Triton to CuTe

- You need warp-specialized producer/consumer that Triton can't express cleanly.
- You need TMA multicast across a cluster.
- You need custom swizzle patterns tuned to a non-standard tile.
- You need cooperative thread arrays with precise mbarrier control.
- You need to compose with a non-standard accumulator format (FP4 with microscaling, for example).

### Common pitfalls

- **Numeric drift in recurrent state accumulation**: when you accumulate a running state across many K steps, small FP8/FP16 rounding errors compound. Keep the accumulator in FP32 even if operands are FP8. When porting from Triton (which often implicitly keeps FP32 accum) to CuTe, make this explicit.
- **TMA descriptor setup**: the TMA descriptor is a 128-byte blob that encodes base pointer, global dims, stride dims, box dims, swizzle, fill mode, and L2 promotion policy. Easy to mis-specify. Always print the descriptor and sanity-check each field.
- **Mbarrier token management**: `arrive_and_expect_tx` must match the actual bytes arriving. Off-by-one on expected byte counts causes deadlocks that look like kernel hangs.
- **Layout algebra mistakes**: `composition(A, B)` vs `product(A, B)` have different semantics. Read the CUTLASS examples; don't guess.

## 2.9 Fusion economics

Fusion is not free. Each fused op takes register space, shared memory, and instruction budget. When is it worth it?

### When fusion pays off

- **Memory-bound elementwise chains**: activation + bias + residual + dropout → fuse them all. Saves N−1 HBM roundtrips.
- **GEMM epilogue fusion**: GEMM + bias + activation + scale-quantize. Essential for FP8 inference where the dequant/requant otherwise doubles memory traffic.
- **LayerNorm + following projection**: LN output flows directly into the next GEMM's input loading. RMSNorm is the canonical fuse target because it's a simple variance calculation.
- **Attention**: FlashAttention fuses Q·Kᵀ, softmax, and ·V into one kernel, collapsing O(S²) activation memory to O(S). The canonical fusion win.

### When fusion doesn't pay off

- **Compute-bound GEMMs where the epilogue is trivial**: adding dropout to a BF16 dense GEMM saves a few percent BW at the cost of kernel complexity. Usually not worth it unless you're doing it in CUTLASS which handles it for free.
- **Fusing into a kernel that was already at peak**: if the GEMM is already at 75% of peak, adding an elementwise op that uses the same issue slots may hurt.
- **Very different operator shapes**: fusing attention into the Q/K/V projection means the projection kernel must handle the attention output shape, which complicates tiling.

### Decision framework

```
                    Should I fuse op A into kernel K?
                              │
                              ▼
              ┌──────────────────────────────┐
              │  Is A memory-bound on its    │
              │  own and producing HBM       │
              │  traffic already in K?       │
              └───────┬───────────────┬──────┘
                      │ yes           │ no
                      ▼               ▼
           ┌─────────────────┐   Skip fusion
           │  Does A add     │
           │  > 15% register │
           │  pressure to K? │
           └───┬────────┬────┘
               │ yes    │ no
               ▼        ▼
        Skip fusion    FUSE
        (or fuse into
         separate wave)
```

A staff candidate should be able to recite: the win from fusion ≈ (bytes saved × (1 / HBM BW)) − (overhead of added register pressure reducing occupancy or tile size). For a Hopper FP8 GEMM where the tile is already huge and register pressure is close to the limit, epilogue fusion is the only way; separating it doubles HBM traffic.


---

# Part 3 — Distributed training performance

## 3.1 Parallelism taxonomy

### The six dimensions

| Dimension | What's sharded                                           | Comm per step                                | Memory saving         | Typical size      |
|-----------|----------------------------------------------------------|----------------------------------------------|-----------------------|-------------------|
| DP (vanilla) | Gradients (all-reduce)                                | AllReduce(gradients) per step                | None on weights/acts  | As large as fits  |
| FSDP (ZeRO-3) | Params + grads + optimizer state (sharded)          | AllGather(params) fwd + bwd, ReduceScatter(grads) | Very large       | 8–64+             |
| TP        | Weight matrices along inner dim; activations along channel | AllReduce(activations) at partition points | Weights / attn heads  | 2–8 (intra-node)  |
| PP        | Layers across devices                                    | P2P send/recv activations, gradients        | Each rank holds N / P  | 4–32              |
| SP        | Activations along sequence dim within TP                 | AllGather / ReduceScatter at LN boundaries  | Activations           | = TP group size   |
| CP        | Attention along sequence dim across devices              | AllGather KV or Ring P2P                    | Activations for long S | 2–16              |
| EP        | MoE experts across devices                               | AlltoAll(tokens) before/after expert        | Expert params         | 4–128             |

### Interaction matrix

```
       ┌──────┬──────┬──────┬──────┬──────┬──────┐
       │  DP  │ FSDP │  TP  │  PP  │  CP  │  EP  │
  ┌────┼──────┼──────┼──────┼──────┼──────┼──────┤
  │ DP │  —   │ ≈    │  ✓   │  ✓   │  ✓   │  ✓   │
  │FSDP│  ≈   │  —   │  ✓*  │  ✓   │  ✓   │  ✓   │
  │ TP │  ✓   │  ✓*  │  —   │  ✓   │  ✓   │  ✓   │
  │ PP │  ✓   │  ✓   │  ✓   │  —   │  ✓   │  ✓   │
  │ CP │  ✓   │  ✓   │  ✓   │  ✓   │  —   │  ✓   │
  │ EP │  ✓   │  ✓   │  ✓   │  ✓   │  ✓   │  —   │
  └────┴──────┴──────┴──────┴──────┴──────┴──────┘

✓  = combinable
✓* = works but TP inside FSDP requires care (FSDP shards along DP dim
     which must be orthogonal to TP dim)
≈  = FSDP is a form of DP; they share the "outer" dimension
```

The canonical 3D (or 5D) parallelism for a modern MoE LLM training job on 1024 GPUs:

```
 World: 1024 GPUs
 Layout: TP=8 × PP=8 × EP=8 × DP(FSDP)=2

 Each axis is a process group. Collectives are scoped to the axis.
 TP and EP typically share the intra-node dimension (NVLink).
 PP crosses nodes along rails. DP/FSDP outermost.
```

## 3.2 Collective communication deep dive

### Ring all-reduce derivation

With N ranks, each holding a tensor of size M bytes, divide into N chunks. In N−1 steps, each rank sends a chunk to its neighbor and simultaneously receives from the other neighbor, reducing as it goes. After N−1 reduce-scatter steps, each rank has one final chunk. Then N−1 all-gather steps propagate the final chunks to all ranks.

Total bytes per rank per step: `M/N`. Steps: `2(N−1)`. Total communication time:

```
 T = 2(N−1)/N × M / BW
```

For large N, `(N−1)/N → 1`, so `T ≈ 2M / BW`. Crucially, **ring all-reduce is O(1) in N at the bandwidth level** — doubling GPUs doesn't increase the per-rank bytes moved (ignoring latency).

```
 Ring all-reduce, N=4 ranks:

   Rank 0         Rank 1         Rank 2         Rank 3
   ┌──┬──┬──┬──┐  ┌──┬──┬──┬──┐  ┌──┬──┬──┬──┐  ┌──┬──┬──┬──┐
   │A0│A1│A2│A3│  │B0│B1│B2│B3│  │C0│C1│C2│C3│  │D0│D1│D2│D3│
   └──┴──┴──┴──┘  └──┴──┴──┴──┘  └──┴──┴──┴──┘  └──┴──┴──┴──┘

   Step 1 (reduce-scatter): each rank sends chunk to right, reduces received from left
   0→1: A0         1→2: B1       2→3: C2       3→0: D3
   Rank 1: B0+A0   Rank 2: C1+B1 Rank 3: D2+C2 Rank 0: A3+D3

   ... 3 steps total ...

   After N-1 steps (reduce-scatter done), each rank has 1 full chunk reduced.
   Then N-1 all-gather steps propagate them.
```

### Tree all-reduce (bandwidth-optimal for small N, latency-optimal for large N)

```
 4-rank tree all-reduce:

   Reduce phase (up)           Broadcast phase (down)
        ┌──┐                          ┌──┐
        │R0│                          │R0│
        └┬─┘                          └┬─┘
       ┌─┴──┐                        ┌─┴──┐
      ┌┴┐  ┌┴┐                      ┌┴┐  ┌┴┐
      │R0│ │R2│                     │R0│ │R2│
      └┬┘  └┬┘                      └┬┘  └┬┘
       ▲    ▲                        ▼    ▼
       R1   R3                       R1   R3

   Latency: 2 log N hops instead of 2(N-1)
   Bandwidth: M bytes per step at the root (bottleneck)
```

**Double binary tree**: fixes the root-bottleneck by running two trees simultaneously where each rank is an interior node in one tree and a leaf in the other. Used extensively in NCCL for small-message all-reduce.

### Hierarchical all-reduce

```
 8-node cluster, 8 GPUs/node, hierarchical all-reduce:

  ┌─────────── Node 0 ───────────┐      ┌─────────── Node 1 ───────────┐
  │  G0  G1  G2  G3  G4  G5  G6  G7│    │  G0  G1  G2  G3  G4  G5  G6  G7│  ...
  │  └──intra-node ring AR─────┘  │     │  └──intra-node ring AR─────┘  │
  │       NVLink, 900 GB/s        │     │                               │
  └───────────────┬───────────────┘     └───────────────┬───────────────┘
                  │                                     │
                  └──── Step 2: inter-node AR ──────────┘
                       (only 1 GPU per node participates)
                       across IB at 400 Gb/s = 50 GB/s
                  │                                     │
  ┌───────────────┴───────────────┐     ┌───────────────┴───────────────┐
  │  Step 3: intra-node broadcast  │     │                               │
  └───────────────────────────────┘     └───────────────────────────────┘

 Effective BW for the bottleneck step = inter-node AR on M/NumNodes bytes
```

Hierarchical algorithms win whenever intra-node BW ≫ inter-node BW (almost always). NCCL's tuning picks this automatically.

### All-to-all (MoE's bread and butter)

```
 All-to-all, N=4 ranks, each sends chunk[j] to rank j:

 Before:           After:
 Rank 0: a b c d   Rank 0: a e i m   <- first chunk from each rank
 Rank 1: e f g h   Rank 1: b f j n
 Rank 2: i j k l   Rank 2: c g k o
 Rank 3: m n o p   Rank 3: d h l p

 Bytes moved per rank = (N-1)/N × M × total_size
 Often implemented as N-1 pairwise exchanges, or a hierarchical variant.
```

MoE expert parallelism sends each token to the expert that gates picked. If tokens are evenly distributed across experts, the all-to-all is balanced. If one expert is hot, its rank becomes a straggler. This is the single most common source of perf pain in MoE training.

### All-gather and reduce-scatter

- **AllGather**: each rank starts with `M/N` bytes, ends with `M` bytes (concatenated). Used by FSDP to reconstruct full params.
- **ReduceScatter**: each rank starts with `M` bytes, ends with `M/N` bytes (each rank gets a reduced slice). Used by FSDP for gradients.

Key identity: `AllReduce = ReduceScatter + AllGather`. A ring all-reduce is literally implemented as these two halves.

## 3.3 FSDP2 internals

FSDP2 (introduced in PyTorch 2.3+) is a from-scratch rewrite using DTensor. It replaces the FlatParameter approach with per-parameter sharding and explicit parameter groups.

### Core pattern

```
 Forward pass of one FSDP unit (one transformer block, typically):

  ┌─────────────────────────────────────────────────────────────┐
  │  AllGather params for block i  (async, overlapped with      │
  │                                  compute of block i-1)      │
  │  Compute block i forward                                    │
  │  Release params for block i-1  (free memory)                │
  └─────────────────────────────────────────────────────────────┘
```

### Timeline with prefetch

```
  Block:    ─0─│─1─│─2─│─3─│─4─│

  Compute:  [c0][c1][c2][c3][c4]     <- critical path

  AllGather: [ag1]|[ag2]|[ag3]|[ag4]  <- starts one block early
              ▲     ▲     ▲     ▲
              prefetched ahead of compute

  HBM:      resident params:
            0     01    12    23    34
                  (only 2 blocks of full params at once)
```

### DTensor sharding

DTensor describes a tensor as a pairing of a local tensor and a "placement" — a tuple of shardings per mesh dim. For FSDP on a 2D mesh `(dp, tp)`, a weight could be `[Shard(0), Shard(0)]` meaning sharded along dim 0 of both axes.

```python
 # FSDP2 canonical pattern
 from torch.distributed.fsdp import fully_shard
 from torch.distributed.device_mesh import init_device_mesh

 mesh = init_device_mesh("cuda", (8,), mesh_dim_names=("dp",))

 for block in model.transformer_blocks:
     fully_shard(block, mesh=mesh)
 fully_shard(model, mesh=mesh)  # root

 model(x)  # AllGather per block on-the-fly
```

### Mixed precision in FSDP2

Specify `MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)`. Parameters are gathered in BF16 (saving NVLink BW), compute is BF16, gradients are reduced in FP32 (preventing underflow in small gradients).

### Activation checkpointing interaction

With FSDP + AC, the forward pass gathers params, computes, releases (as above). The backward pass re-forward-computes checkpointed regions, which triggers **another** AllGather for those regions. Total AllGathers = 2× the number of FSDP units. This is one of the biggest sources of overhead in long-context training.

Mitigations:
- Use fewer, larger FSDP units (larger block grouping).
- Use **Selective Activation Checkpointing** (recompute only attention, not MLP, or vice versa).
- FSDP2 has "reshard_after_forward" control — if you keep params gathered between forward and backward, you save the second AllGather at the cost of memory.

## 3.4 Pipeline parallelism schedules

### GPipe (the original)

All microbatches go forward, then all go backward. Maximum memory pressure because all activations are stored until the backward sweep starts.

```
  GPipe, P=4 stages, M=8 microbatches
  (F = forward, B = backward)

  Stage 0: F0 F1 F2 F3 F4 F5 F6 F7                B7 B6 B5 B4 B3 B2 B1 B0
  Stage 1:    F0 F1 F2 F3 F4 F5 F6 F7          B7 B6 B5 B4 B3 B2 B1 B0
  Stage 2:       F0 F1 F2 F3 F4 F5 F6 F7    B7 B6 B5 B4 B3 B2 B1 B0
  Stage 3:          F0 F1 F2 F3 F4 F5 F6 F7 B7 B6 B5 B4 B3 B2 B1 B0
                    ──────── fwd ────────   ─────── bwd ───────
                    
  Bubble fraction: (P-1)/(M+P-1) = 3/11 ≈ 27% for P=4, M=8
  Activation memory per stage: O(M × activation_per_microbatch)
```

### 1F1B (one forward, one backward)

As soon as a stage finishes the first forward for a microbatch, it starts the backward as soon as the gradients return from downstream. This reduces peak activation memory to O(P) instead of O(M).

```
  1F1B, P=4 stages, M=8

  Stage 0: F0 F1 F2 F3 B0 F4 B1 F5 B2 F6 B3 F7 B4 B5 B6 B7
  Stage 1:    F0 F1 F2 F3 B0 F4 B1 F5 B2 F6 B3 F7 B4 B5 B6 B7
  Stage 2:       F0 F1 F2 F3 B0 F4 B1 F5 B2 F6 B3 F7 B4 B5 B6 B7
  Stage 3:          F0 F1 F2 F3 B0 F4 B1 F5 B2 F6 B3 F7 B4 B5 B6 B7
                    ──warmup──┤──steady state──┤──cooldown──

  Bubble fraction: same (P-1)/(M+P-1), but activation memory is O(P)
```

### Interleaved 1F1B (virtual pipelines)

Each device runs V "virtual stages" (chunks of layers). This divides each microbatch into more pieces, reducing bubble.

```
  Interleaved 1F1B, P=4 physical × V=2 virtual = 8 logical stages
  (each device runs layers [i, i+4])

  Bubble fraction: (P-1)/(V × M + P - 1)
  For P=4, V=2, M=8: 3/19 ≈ 16% (down from 27%)

  Cost: more P2P sends (V× more) and more scheduling complexity
```

### Zero-bubble pipeline (ZB-H1, ZB-H2)

Backward is split into **backward-for-input-grad (B)** and **backward-for-weight-grad (W)**. W only needs activations and output grads; I depends on upstream. By scheduling W ops into what would be bubbles, you can nearly eliminate them.

```
  ZB-H1 sketch:

  Stage 0: F0 F1 F2 F3 B0 F4 B1 W0 F5 B2 W1 F6 B3 W2 F7 B4 W3 B5 W4 B6 W5 B7 W6 W7
  
  The W ops fill what would otherwise be bubbles.
  ZB-H1: bubble = 0 at steady state (but warmup/cooldown still exist).
  ZB-H2: adds more reordering to eliminate warmup bubble too.
```

The catch with ZB: weight gradient accumulation order changes, which may affect numerics and optimizer behavior. Test carefully; it's not always a free lunch.

### Summary table

| Schedule         | Bubble fraction              | Activation mem | Complexity |
|------------------|------------------------------|----------------|-----------|
| GPipe            | (P−1)/(M+P−1)                | O(M)           | Low       |
| 1F1B             | (P−1)/(M+P−1)                | O(P)           | Low       |
| Interleaved 1F1B | (P−1)/(V·M+P−1)              | O(V·P)         | Medium    |
| ZB-H1            | ≈ 0 steady state; P−1 warmup | O(P)           | High      |
| ZB-H2            | ≈ 0 total                    | O(P)           | Very high |

## 3.5 Tensor parallelism

### Megatron column + row pattern for MLP

A transformer MLP is `Y = f(X @ W1) @ W2`. With TP=4:

```
  Column-parallel W1 (shard output dim):
    X:   [B, S, H]        (replicated)
    W1:  [H, 4F/4]        (each rank has H × F)
    Y1:  [B, S, 4F/4]     (each rank has B × S × F)

  Activation f (e.g., GELU/SwiGLU): pointwise, no comm needed

  Row-parallel W2 (shard input dim):
    W2:  [4F/4, H]        (each rank has F × H)
    Y2i: [B, S, H]        (each rank has partial sum)
    Y2:  AllReduce(Y2i)   <- one all-reduce per MLP
```

```
  ┌───────── Rank 0 ─────────┐  ┌───────── Rank 1 ─────────┐  ...
  │                          │  │                          │
  │  X:  [B, S, H]           │  │  X:  [B, S, H]           │
  │     ↓ × W1[:, 0:F]       │  │     ↓ × W1[:, F:2F]      │
  │  Y1: [B, S, F]           │  │  Y1: [B, S, F]           │
  │     ↓ f(·)               │  │     ↓ f(·)               │
  │  Y1': [B, S, F]          │  │  Y1': [B, S, F]          │
  │     ↓ × W2[0:F, :]       │  │     ↓ × W2[F:2F, :]      │
  │  Y2_partial: [B, S, H]   │  │  Y2_partial: [B, S, H]   │
  └────────────┬─────────────┘  └────────────┬─────────────┘
               │                             │
               └─────── AllReduce ───────────┘
                       ↓
                Y2: [B, S, H] (reduced)
```

### Attention under TP

Split attention along heads: `num_heads / TP` heads per rank. Each rank computes its heads' QKV and attention output independently. One AllReduce at the output projection (row-parallel).

### Where the AllReduces land

For a single transformer block under TP=k:
- 1 AllReduce after MLP W2
- 1 AllReduce after attention output projection
- Total: 2 AllReduces per block per forward pass (and 2 per backward)

For a 32-block model: 64 AllReduces per forward, 64 per backward = 128 per step.

### Sequence parallelism (SP) as memory optimization

The AllReduce can be decomposed as AllGather + ReduceScatter. If you keep the input of LayerNorm sharded along the sequence dimension (SP), you can:

1. ReduceScatter at the end of MLP (splits along seq dim)
2. Do LayerNorm on the sharded sequence (trivially parallel since LN is per-token)
3. AllGather before the next attention's Q/K/V projection

This saves the memory of holding full-sequence activations while doing the same bytes of comm.

```
 TP only:                 TP + SP:
 
 X: [B, S, H] (repl)      X: [B, S/k, H] (seq-sharded)
    ↓                        ↓
 LN: [B, S, H]            AllGather → [B, S, H]
    ↓                        ↓
 Attn → [B, S, H]         Attn → [B, S, H]
    ↓ AllReduce              ↓ ReduceScatter
 Y: [B, S, H] (repl)      Y: [B, S/k, H] (seq-sharded)
    ↓ LN                     ↓ LN (per-token OK on shard)
 ...                      ...

 SP saves activation memory of O(S/k) × H per rank.
```

### Async tensor parallelism (SymmetricMemory)

The AllReduce blocks compute. With async TP (enabled via `torch.ops._c10d_functional.all_reduce`, `torch.distributed.SymmetricMemory`, and Blackwell's NVLink SHARP support), you can overlap the AllReduce's traffic with the next operation's matmul.

## 3.6 Expert parallelism for MoE

### The pattern

```
  One MoE layer under EP=4, with 8 experts (2 per rank):

  X: [B, S, H]      (tokens, replicated within EP group)
     ↓
  Gating: top_k routing assigns each token to an expert
     ↓
  Permute: sort tokens by expert
     ↓
  AllToAll: each rank sends tokens assigned to expert on other ranks
     ↓
  Expert compute: each rank runs its 2 experts on its received tokens
     ↓
  AllToAll: each rank sends results back to original owner
     ↓
  Unpermute: restore original token order
     ↓
  Y: [B, S, H]
```

### Cost scaling

- Tokens per rank: `B × S`
- Average tokens routed to each expert's host: `B × S × top_k / num_ranks`
- AllToAll bytes per rank: `2 × B × S × H × bytes × top_k × (EP-1)/EP` (×2 for both directions)

For `B=4, S=8192, H=8192, BF16, top_k=2, EP=8`: that's `2 × 4 × 8192 × 8192 × 2 × 2 × 7/8 ≈ 1.8 GB per rank` per MoE layer per forward pass. At NVLink 900 GB/s, that's ~2 ms. With 32 MoE layers and 2 all-to-alls each (fwd + bwd = 4), you're looking at ~260 ms of AllToAll per step if unhidden.

### Load balancing

Load balancing loss (`LB = num_experts × sum(f_i × p_i)` where `f_i` is fraction of tokens routed to expert i, `p_i` is average gate weight for expert i) during training keeps expert utilization roughly uniform. Still, at inference, some experts get 2× the traffic of others — the hot-expert problem.

### Debugging imbalanced expert utilization

Symptoms:
- One rank's forward pass takes 1.5× as long as others.
- AllToAll hangs periodically (waiting on the slow rank).
- Tokens dropped (if drop-on-overflow is enabled).

Diagnosis:
- Log `num_tokens_per_expert` per step. Plot distribution.
- A healthy system has ~uniform distribution with std < ~1.3× mean.
- An unhealthy system has one or two experts getting 3–5× the mean. Check LB loss is actually on; check gating temperature.

Mitigations:
- **Expert parallelism with capacity factor > 1** (e.g., 1.25×) — each expert accepts up to capacity × expected_tokens, excess dropped or overflowed to next expert.
- **Expert placement**: interleave hot experts across ranks rather than concentrating them.
- **DeepEP / Megablocks**: variable-size expert buffers with no padding, handled via block-sparse GEMM.

## 3.7 Overlap

### FSDP AllGather prefetch

PyTorch FSDP pre-issues the AllGather for block `i+1` while compute is happening on block `i`. The overlap is perfect if:

```
  compute_time(block_i) >= allgather_time(block_{i+1})
```

Equivalently: the compute intensity per block must exceed the ratio of AllGather bytes to NVLink BW.

For a transformer block at 70B scale:
- Params per block: ~2 GB BF16
- AllGather across 8 ranks: `7/8 × 2 GB / (900 GB/s) ≈ 2 ms`
- Block forward compute (BF16, S=4k, B=4): roughly 2 × 4 × 4096 × 70B/32 × 2 ≈ 140 GFLOPs → at 989 TF ≈ 0.15 ms

Hence **compute is 10× shorter than AllGather at this scale**, and FSDP alone cannot hide the comm. This is why 70B+ training uses TP+FSDP hybrid, or moves to larger compute (higher S, higher B) to balance.

### TP async overlap

In the MLP pattern, the AllReduce after W2 blocks the next LayerNorm. With async TP:

```
  Regular TP:                    Async TP:
  [W2][AR       ][LN][Attn]      [W2][AR    ]
                                      [LN][Attn] <- starts as soon as
                                                    partial-reduced tensor
                                                    is available at current rank
```

Requires SymmetricMemory and a fine-grained protocol where each rank begins LN on its local partial result while the reduction completes across ranks.

### PP bubble filling

In ZB-H1, W ops are scheduled into bubbles. Effective overlap as long as W ops are available.

### What timelines look like when overlap fails

```
 Overlap success (healthy FSDP):
  
 Compute:  ████████████████████████████
 AllGather:   ░░██ ░░██ ░░██ ░░██ ░░██
              prefetched during compute
  
 Overlap failure:
  
 Compute:  ████  (idle)  ████  (idle)  ████
 AllGather:    ██████        ██████
           compute gap because AG too slow
  
 Indicator: torch.profiler shows cudaStreamWaitEvent stall
            between kernels; NCCL op dominates the gap.
```

## 3.8 Debugging distributed failures

### NCCL hang checklist

1. **Mismatched collectives** — rank 0 calls `all_reduce`, rank 1 calls `reduce`. Check `NCCL_DEBUG=INFO` for op mismatch messages.
2. **One rank stuck in CPU code** — dataloader blocked on disk I/O, or a Python exception caught silently. Look at py-spy dumps of each rank.
3. **Uneven iteration count** — rank 0 calls 1000 all-reduces, rank 1 calls 999. Usually caused by early exit on one rank (bad data sample dropped, OOM retry).
4. **Shape mismatch** — ranks have different tensor shapes for the same collective. NCCL will hang silently in older versions; newer versions surface an error.
5. **Mixed dtypes** — one rank is FP32, another BF16. Same failure mode.
6. **Deadlock via custom collective order** — rank 0 does `AllReduce(A); AllReduce(B)`, rank 1 does `AllReduce(B); AllReduce(A)`. NCCL is ordered; this deadlocks.

### Straggler detection

```
 Per-step timing per rank, plot as heatmap:
 
  Rank
    0 ████ ████ ████ ████ ████    <- consistent
    1 ████ ████ ████ ████ ████
    2 ████ ████ █████ ████ ████   <- occasional slow
    3 ████ ████ ████ ████ ████
    4 █████ █████ █████ █████     <- consistently slow (bad GPU or throttling)
    5 ████ ████ ████ ████ ████
    6 ████ ████ ████ ████ ████
    7 ████ ████ ████ ████ ████
        t=0    t=1    t=2    t=3
```

Tools:
- `torch.distributed.monitored_barrier(timeout=60)` — raises if any rank is slow.
- Meta's "flight recorder" (in PyTorch) — dumps pending NCCL ops per rank when a hang is detected, identifying which collective the slow rank was stuck on.
- `NCCL_ASYNC_ERROR_HANDLING=1` + `TORCH_NCCL_BLOCKING_WAIT=1`.

### NCCL_DEBUG=INFO — what to look for

Healthy: `NCCL INFO Ring 00 : ...` topology printed at init, then silence.

Pathological patterns:
- Repeated `NCCL INFO Channel ... : retrying after N µs` — network flakiness.
- `NCCL INFO Connect to ... returned X, retrying` — TCP/IB issues.
- `NCCL WARN Mismatched number of participants` — bug 1 above.
- Slow collective warnings with latency > some threshold.

### When in doubt

`TORCH_NCCL_DESYNC_DEBUG=1` + `TORCH_NCCL_DUMP_ON_TIMEOUT=1` — on timeout, writes per-rank dumps showing the last known collective each rank was in. This is the single highest-leverage flag for diagnosing distributed hangs.

## 3.9 MFU vs HFU vs MBU

### Definitions

- **MFU** (Model FLOPs Utilization) = measured FLOPs / peak FLOPs. Numerator is the theoretical FLOPs the model computes (e.g., `6 × N × D × T` for a dense LLM — 6 FLOPs per parameter per token for fwd+bwd).
- **HFU** (Hardware FLOPs Utilization) = (measured FLOPs + recompute FLOPs) / peak FLOPs. Counts the activation checkpointing cost.
- **MBU** (Model Bandwidth Utilization) = bytes moved / peak HBM BW. Primary metric for decode and other memory-bound workloads.

### MFU vs HFU

If you use activation checkpointing with full recompute of every layer, HFU can be ~1.33× MFU (because recompute adds ~33% extra work on top of fwd+bwd's 3× base). A job reporting 45% MFU and 60% HFU is using recompute effectively.

### Target numbers

| Workload                              | Target MFU  | Notes                                              |
|---------------------------------------|-------------|----------------------------------------------------|
| 70B dense LLM on H100 (FP8)           | 40–55%      | Depends on seq length; longer = higher             |
| 405B dense LLM on H100 (BF16)         | 35–50%      | Comm overhead larger at this scale                 |
| DeepSeek-V3-style MoE (FP8)           | 30–45%      | All-to-all overhead eats into it                   |
| Diffusion (UNet-based) training       | 30–45%      | Many elementwise ops, lower MFU typical            |
| Diffusion (DiT-based)                 | 40–55%      | More matmul-heavy                                  |

### MBU for decode

MBU target: ≥ 60% on H100/H200 with a well-tuned kernel, batch-1 decode. Batch > 1 can push MBU higher because attention's intensity grows.

### Computing MFU for a dense LLM

```
 FLOPs per token (fwd+bwd, dense) ≈ 6 × N_params
 
 Tokens per step = global_batch × sequence_length
 Step time = measured (e.g., 2.5 s)
 
 Measured FLOPs/s = 6 × N × tokens / step_time
 Peak FLOPs/s = num_gpus × per_gpu_peak (e.g., 1024 × 1979 TF for FP8 on H100)
 
 MFU = measured / peak
```

Worked example: 70B model, 1024 H100, GBS=1024, S=8192, step=2.5s, FP8.

```
  FLOPs = 6 × 70e9 × 1024 × 8192 = 3.52e18
  Rate = 3.52e18 / 2.5 = 1.41e18 FLOP/s
  Peak = 1024 × 1979e12 = 2.03e18 FLOP/s
  MFU = 69%  (excellent — probably hitting edge cases of model FLOP definition)
```

### Why MFU alone is misleading

- MFU counts only forward+backward FLOPs. It ignores optimizer step, loss computation, recompute.
- Can be gamed by selecting the most favorable FLOP formula (e.g., 6N vs 8N depending on what you include).
- A 50% MFU job might be worse in wall-clock than a 40% MFU job that runs bigger batches at higher AI.

Staff-level framing: "We report MFU for apples-to-apples across our fleet. In practice, what we care about is tokens/sec/dollar."

---

# Part 4 — LLM inference performance

## 4.1 Prefill vs decode dichotomy

### Arithmetic intensity

Prefill: takes a sequence of `S` prompt tokens, computes all QKV and attention in parallel. FLOPs scale as `O(S × N_params + S² × N_params_attn)` per layer. Bytes scale as `O(N_params + S × H)`. AI scales with `S`.

Decode: one token at a time. FLOPs = `O(N_params + S_context × H)`. Bytes = `O(N_params + S_context × H × bytes)`. AI is ~1 FLOP/byte (no kidding — memory-bound to the roof).

```
  Roofline: prefill vs decode

  Perf
  (TF/s)
       ▲
 Peak  │━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  
       │                  ┌─ prefill (long S)
       │                  │  AI = hundreds, compute-bound
       │                  ◇
       │                 ╱
       │               ╱
       │             ╱
       │           ╱
       │         ╱
       │       ╱
       │     ╱
       │   ╱
       │ ╱◇  decode AI≈1, memory-bound, far below peak
       │
       └──────────────────────────────▶  AI
       1    10   100   1000
```

### Batch size rescues decode

Decoding N requests in parallel with batch size B means every weight load from HBM is amortized over B × seq steps. AI grows linearly with B up until:

- KV cache memory becomes the constraint.
- Attention's compute starts to dominate (at very long contexts).

For a 70B model at seq 4k, batch 1 decode ≈ 40 tok/s on 4 H100s with TP=4. Batch 32 decode ≈ 800 tok/s (25× more with 32× batch). The amortization is nearly linear until you hit memory limits.

## 4.2 KV cache management

### Memory math

```
 KV cache size = 2 × L × H_kv × D_head × S × B × bytes
 
   L = num_layers
   H_kv = num_kv_heads   (= H_attn / GQA_ratio)
   D_head = head dim
   S = seq length
   B = batch size
```

Examples (BF16, 2 bytes):

- **Llama-3-8B** (L=32, H_kv=8, D=128): `2 × 32 × 8 × 128 × S × B × 2 = 131072 × S × B bytes`. At S=8192, B=1: 1.07 GB.
- **Llama-3-70B** (L=80, H_kv=8, D=128): `2 × 80 × 8 × 128 × S × B × 2 = 327680 × S × B bytes`. At S=8192, B=32: 86 GB. At 128k context, B=1: 42 GB.
- **DeepSeek-V3** with MLA reduces this drastically (~10× smaller KV): the compressed representation has D_c ≈ 128 across all heads instead of H_kv × D_head.

### Paged attention (vLLM)

Split the KV cache into fixed-size blocks (typically 16 tokens). A block table per sequence maps logical positions to physical blocks. Eliminates KV fragmentation and enables prefix sharing.

```
  Block Table for request R:
  Logical pos:   0-15   16-31   32-47   48-63
                   │      │       │       │
                   ▼      ▼       ▼       ▼
  Block ID:        7     23      42      91
  
  Physical KV cache (pool of blocks):
  ┌────┬────┬────┬────┬────┬────┬────┬─────┬──
  │ B0 │ B1 │ ...│ B7 │ ...│ B23│ ...│ B42 │...
  │free│free│    │R's │    │R's │    │R's  │
  │    │    │    │ 0- │    │16- │    │ 32- │
  │    │    │    │ 15 │    │31  │    │ 47  │
  └────┴────┴────┴────┴────┴────┴────┴─────┴──

  Free list tracks which blocks are available for allocation.
  Block granularity = 16 tokens (tuned per model / GPU).
```

Advantages:
- Zero fragmentation.
- Prefix sharing: multiple requests with the same prefix can share physical blocks (RadixAttention).
- Blocks can be swapped to CPU memory under pressure.
- Clean memory accounting.

Cost:
- Indirection: attention kernels need block tables, increasing register pressure.
- Small blocks reduce coalescing; vLLM tuned 16 as a balance.

### Contiguous vs paged

Contiguous: KV is one big tensor per sequence. Simple, but must pre-allocate `max_seq_len`. Inflates memory by a factor related to average/max sequence length — in practice 2–3×.

Paged: allocate blocks on demand. Memory scales with actual usage. This is what every serious inference server does now.

### RadixAttention / prefix caching

If two requests share the prefix "You are a helpful assistant. The user said:", you only need to compute KV for that prefix once. The block table points both requests to the same physical blocks for the shared prefix.

```
  Radix tree of prefixes:

       [root]
         │
         ▼
   "You are a helpful assistant."  <- blocks 0-2 shared by all requests
         │
         ├─→ "The user said: hi" <- request R1
         │
         └─→ "The user said: translate X" <- request R2
```

The savings scale with prefix depth and number of concurrent requests sharing that prefix. For a chatbot-style workload with system prompts, this is 30–60% KV memory savings.

### KV cache quantization

- **FP8**: straightforward, ~1% quality loss at typical scales. Batch compatible with FP8 attention kernels.
- **INT8 per-head, per-channel**: similar quality, requires a scale per KV dim. Common in production serving.
- **INT4 with group scaling** (KIVI, KVQuant): ~4× memory reduction, 1–2% quality loss. Worth it for very long contexts.

### KV cache offload

For very long contexts (>128k), KV cache can exceed even 80GB GPUs. Strategies:
- **CPU offload**: move inactive blocks (old context the current decoding step won't attend to) to CPU memory. Works for sparse-attention patterns; for full attention, you'll fetch it all anyway.
- **NVMe offload**: even slower tier for truly massive contexts (1M+ tokens). Mostly research territory.
- **Disaggregated KV**: a separate KV service holds caches, compute nodes pull blocks as needed. Used in some large-scale serving.

## 4.3 Continuous batching

Static batching: assemble a batch of N requests, run them together until all finish. The last-finishing request dictates latency for all.

Continuous batching: the scheduler runs a forward pass per step. After each step, finished requests drop out; new requests are admitted. The active batch changes every step.

```
  Static batching (N=4):
  
  Req 0 (50 tok): ████████████████████
  Req 1 (10 tok): ████  (wasted time - waits for longest)
  Req 2 (30 tok): ████████████
  Req 3 (80 tok): ████████████████████████████████
                                                  └─ step 80 finishes all
                                                  
  Continuous batching:
  
  Step:     1  2  3 ... 10 11 ... 30 31 ... 50 51 ... 80
  Req 0:    █  █  █  ... █  █  ... █  ░  ...  (finished at 50)
  Req 1:    █  █  █  ... █  ░  ...             (finished at 10)
  Req 2:    █  █  █  ... █  █  ... █  ░  ...  (finished at 30)
  Req 3:    █  █  █  ... █  █  ... █  █  ... █  █  ... █  ░
  Req 4:          █  ...                        (admitted at step 11)
  Req 5:                 █  ...                 (admitted at step 20)
  Req 6:                       █  ...           (admitted when Req 2 done)
```

Throughput increases roughly 2–3× vs static batching at realistic arrival distributions.

## 4.4 Chunked prefill and prefill/decode disaggregation

### Chunked prefill

For a 10k-token prefill, the attention FLOPs are O(S²) which can easily stall decoding for other concurrent requests.

Chunked prefill splits the prompt into chunks (e.g., 512 tokens each) and interleaves with decode steps:

```
  Request A arrives with 10k token prompt, while request B is decoding:

  Without chunking:
  ████████████████████████  (A's full prefill, 2s)  ░  (B's decode blocked)
  
  With chunked prefill (1k chunks):
  ███ █ ███ █ ███ █ ███ █ ███ █  ...  ███ ░   <- B's decode tokens interleaved
   A   B  A  B  A  B  A  B  A  B       A
```

### P/D disaggregation

A single instance can't simultaneously optimize for prefill (compute-bound) and decode (memory-bound). Solution: dedicate some GPUs to prefill and others to decode, transferring KV cache between them.

```
  ┌─────────────────┐              ┌─────────────────┐
  │  Prefill pool   │   KV xfer    │   Decode pool   │
  │  (large batch,  │ ───────────▶ │  (large batch,  │
  │  high MFU)      │  NVLink/IB   │  high MBU)      │
  │                 │              │                 │
  │  e.g., 2× H100  │              │  e.g., 4× H100  │
  └─────────────────┘              └─────────────────┘
```

KV transfer: for a 70B model, 8k prefill, ~1 GB of KV per request. At 50 GB/s IB, that's 20 ms per request — significant overhead. On NVLink within a rack, it's 1 ms — tolerable.

### When each wins

- **Uniform chunked prefill**: simpler, works well when prefill/decode ratio is stable, all GPUs are the same.
- **P/D disaggregation**: wins when prompts are very long and/or GPU fleet is heterogeneous (put prefill on H100, decode on H200 for more HBM). Also necessary when TTFT SLAs are tight and prefill interference with decode must be zero.

## 4.5 Speculative decoding families

### Vanilla draft model

A small model (e.g., 1B) generates K candidate tokens; the large model (e.g., 70B) verifies them in parallel in a single forward pass. Accepted prefixes are kept.

Math:
- If acceptance rate = α and draft cost is negligible, speedup ≈ `1 + α × K` up to the limit of parallel verification throughput.
- Empirically α ≈ 0.6–0.8 for well-matched drafts, K ≈ 4–8.
- Break-even: `K × cost(draft) / cost(target) < α × K` → draft must be at least `1/α` times cheaper per token than target.

### Medusa

Attach K extra prediction heads to the target model itself. Each head predicts positions (t+1, t+2, ..., t+K) from the same base hidden state. No separate draft model. Modest accept rate (α ≈ 0.5) but zero draft overhead.

### EAGLE / EAGLE-2 / EAGLE-3

Auto-regressive draft that takes the *hidden state* of the target's last layer as input (cheaper than running the full target). Produces a sequence of candidate tokens with a small autoregressive model. EAGLE-2 uses dynamic tree decoding (branching candidate sequences). EAGLE-3 expands the feature inputs to include multiple target layers and improves acceptance.

Typical acceptance rates:
- EAGLE:   α ≈ 0.7–0.8
- EAGLE-2: α ≈ 0.8–0.85  (tree decoding → effective K higher)
- EAGLE-3: α ≈ 0.85+

Break-even: for a 70B target, EAGLE draft is ~2% of target cost. The draft is essentially free, so speedup ≈ `1 + α × K` with K around 4–6 → **2–3× decode throughput**.

### Lookahead decoding

Uses n-gram completion guesses sampled from recent decode output. No draft model needed. Acceptance is lower (α ≈ 0.3–0.4) but zero training overhead. Useful when you cannot fine-tune a draft.

### When each is worth the complexity

| Method          | α typical | Eng complexity | Worth it when                            |
|-----------------|-----------|----------------|------------------------------------------|
| Draft model     | 0.6–0.8   | Medium         | Have a pretrained small family member    |
| Medusa          | 0.4–0.6   | Low            | Don't want to serve a separate draft     |
| EAGLE-2/3       | 0.8+      | High (training)| Production serving at scale; SOTA wanted |
| Lookahead       | 0.3–0.4   | Low            | Can't train anything; hot-swap addition  |

Note the quality caveat: speculative decoding is *exact* (verified by the target), so it doesn't change output distribution. However, integration with continuous batching, quantization, and TP requires careful engineering — the verification forward pass needs to process the K+1 candidate sequence efficiently.

## 4.6 Tensor parallel inference

### When to use

- **Large models that don't fit on one GPU**: 70B BF16 = 140 GB, needs 2× H100 minimum. 405B BF16 needs ≥ 6× H100.
- **Reducing per-token decode latency**: weights are sharded, so each GPU loads 1/TP of the weights per token. Faster per-token latency even when one GPU could fit the model.

### The AllReduce cost

Per-token cost of one AllReduce with `H` activation: `2 × H × (TP-1)/TP × bytes / NVLink_BW`.

For H=8192, BF16 (2 bytes), TP=8, NVLink 900 GB/s:
`2 × 8192 × 2 × 7/8 / 900e9 = 32 ns per all-reduce × 2 per block × 80 blocks = 5 μs per token.`

At 50 tok/s target → 20 ms per token. 5 μs of comm is 0.025% — trivial.

The overhead becomes significant only at very long sequences or very aggressive latency targets. For most serving setups, TP inference is close to linear scaling up to TP=8 intra-node.

### TP vs PP for serving

- **TP**: lower latency per token (all GPUs work on every token), but limited to intra-node scale (NVLink needed). Best for single-request latency.
- **PP**: higher throughput (pipelined microbatches), but per-token latency is P × per-stage-latency. Useless for single-stream decode.

Rule: for serving, use TP first (up to NVLink group size), then DP across groups for throughput. PP is rare in inference except for very large models or when GPUs don't have sufficient NVLink connectivity.

## 4.7 Quantization for inference

### Format matrix

| Format      | Weights | Activations | KV    | Typical quality loss | Speed vs BF16    |
|-------------|---------|-------------|-------|----------------------|-------------------|
| BF16        | BF16    | BF16        | BF16  | 0                    | 1.0× (baseline)   |
| W8A8        | INT8    | INT8        | BF16  | < 1%                 | 1.5–2×            |
| W8A16       | INT8    | BF16        | BF16  | ~0                   | 1.3–1.6× (mem)    |
| FP8 (E4M3)  | FP8     | FP8         | BF16  | < 1%                 | 1.7–2.0×          |
| W4A16       | INT4    | BF16        | BF16  | 1–3%                 | 1.8–2.5× (decode) |
| W4A8        | INT4    | INT8        | INT8  | 2–4%                 | 2.5–3.5×          |
| NVFP4       | FP4     | FP4         | FP8/4 | 1–2% (well-tuned)    | 3–4× (B200)       |

### Granularity axes

- **Per-tensor**: one scale for the whole tensor. Simplest, lowest quality.
- **Per-channel (per-row for weights)**: one scale per output channel. Standard for weight quantization.
- **Per-group**: one scale per 64 or 128 consecutive elements within a channel. Better outlier handling.
- **Per-token (for activations)**: one scale per sequence position. Expensive but high quality.
- **Per-block (MX formats)**: one scale per 32/16 elements. Hardware-supported.

### Methods

- **GPTQ**: calibration-based. Computes second-order updates to compensate for quantization error layer-by-layer. State of the art for INT4 weight-only.
- **AWQ**: observes that salient weight channels (those multiplied by large activations) should be quantized less aggressively. Uses per-channel scaling before quantization. Robust and fast.
- **SmoothQuant**: migrates difficulty from activations to weights via a diagonal rescaling. Enables W8A8 with minimal loss.
- **LLM-QAT / QLoRA**: quantization-aware training / fine-tuning. Highest quality for extreme compression (W4A4).

### Decision framework

```
  Need to quantize model for inference. Decision tree:

  1. Memory-bound at decode (common)?
     └─▶ Weight-only matters most. Use W4A16 (AWQ or GPTQ). 
         Aggressive: KV cache in FP8 too.

  2. Compute-bound at prefill (large batch or long prompt)?
     └─▶ Need W8A8 or FP8 to use tensor cores fully. 
         Use FP8 if hardware supports (Hopper+).

  3. Both matter (production serving)?
     └─▶ Per-tensor FP8 for weights+acts, INT8 KV cache.
         On B200, move to NVFP4 weights + FP8 acts.

  4. Quality budget extremely tight (research eval)?
     └─▶ Stay in BF16 for A and use W8A16 at worst.
```

## 4.8 Long-context inference specifically

### FlashAttention generations

- **FA-1**: tiled softmax, recomputes in backward. O(N) memory, O(N²) compute.
- **FA-2**: better parallelization (split along sequence within a warp, not just across warps). 2× faster than FA-1 on H100.
- **FA-3**: H100/H200 warp-specialized async TMA + wgmma. Another 1.5–2× faster than FA-2. Uses producer/consumer and writes results directly from TMEM.

### Ring Attention / Context Parallel

Shard K and V across the sequence dimension across GPUs. Each GPU computes attention for its local Q against the ring of KV blocks passed around.

```
  Ring Attention, CP=4:
  
  Rank 0 has Q[0:S/4], receives K,V rotating through ranks:
  
  Step 1: Q[0:S/4] @ K,V[0:S/4]       (local)
  Step 2: Q[0:S/4] @ K,V[S/4:S/2]     (from rank 1)
  Step 3: Q[0:S/4] @ K,V[S/2:3S/4]    (from rank 2)
  Step 4: Q[0:S/4] @ K,V[3S/4:S]      (from rank 3)
  
  Accumulate softmax statistics (m_i, l_i) across steps.
  Comm: each rank sends/receives O(S/CP × H) bytes per step × CP steps = O(S × H).
  Perfectly overlaps with attention compute when compute > comm per step.
```

### Star Attention, Striped Attention, StripedHyena

Variants that reduce inter-rank comm at the cost of restricted attention patterns (e.g., only causal masking, or sliding window). Useful for specific long-context workloads.

### KV pressure at 128k+ context

At 128k with a 70B MLA model, KV per request ≈ 10 GB. On an 8-H100 node with 640 GB total HBM, weights take ~140 GB, framework overhead ~50 GB, leaving ~450 GB. Max ~45 concurrent requests at 128k. At 1M context, 1 request per node.

### Attention approximations

- **Sliding window** (Longformer, Mistral): attend to only the last W tokens. O(S × W) instead of O(S²). Quality loss for global dependencies.
- **H2O, StreamingLLM**: keep "heavy hitter" tokens (sink + recent). Evict the rest. Works for chat-style workloads with limited long-range dependency.
- **Infinite-attention / Mamba / SSMs**: replace attention entirely with recurrent mechanisms. Linear memory, fundamentally different architecture.

## 4.9 Serving system architecture

```
  ┌────────────────────────────── Inference server ─────────────────────────────────┐
  │                                                                                 │
  │  ┌───────────────┐    ┌──────────────────┐    ┌───────────────────────────┐     │
  │  │   Frontend    │    │    Scheduler     │    │      Block Manager        │     │
  │  │               │    │                  │    │                           │     │
  │  │ - HTTP/gRPC   │───▶│ - admission      │◀──▶│ - KV block pool           │     │
  │  │ - token stream│    │ - continuous     │    │ - prefix radix tree       │     │
  │  │ - auth/limits │    │   batching       │    │ - allocate/free per req   │     │
  │  └───────────────┘    │ - P/D split      │    │ - CPU offload (optional)  │     │
  │                       │ - priority       │    └──────┬────────────────────┘     │
  │                       └─────────┬────────┘           │                          │
  │                                 │                    │                          │
  │                                 ▼                    ▼                          │
  │                       ┌────────────────────────────────────────────┐            │
  │                       │            Worker / Model Runner           │            │
  │                       │                                            │            │
  │                       │  - TP group (intra-node)                   │            │
  │                       │  - Attention kernel (FlashAttention/paged) │            │
  │                       │  - MLP / epilogue fusions                  │            │
  │                       │  - Speculative decoding (if enabled)       │            │
  │                       │  - Sampling (top-p, top-k, temp)           │            │
  │                       └────────────────────────────────────────────┘            │
  │                                                                                 │
  └─────────────────────────────────────────────────────────────────────────────────┘
```

### Core components

- **Scheduler**: decides which requests get admitted to the next forward pass. Handles priorities, SLA targets, preemption.
- **Block manager**: tracks free KV blocks, maintains the prefix radix tree, handles allocation/deallocation.
- **Worker**: actually runs the forward pass. One worker per TP group.

### vLLM vs SGLang vs TensorRT-LLM

- **vLLM**: pioneered PagedAttention. Good general-purpose baseline. Python-heavy in scheduler; newer CUDA graph support closes the Python overhead gap.
- **SGLang**: RadixAttention + programmatic frontend (structured generation, control flow). Strong for chat / tool-calling workloads. Lower overhead scheduler.
- **TensorRT-LLM**: NVIDIA-optimized, includes in-flight batching + paged KV. Fastest raw throughput on NVIDIA hardware but less flexible.

### Scheduler pseudocode

```python
def step(scheduler):
    while scheduler.can_admit():
        scheduler.admit_waiting_request()
    
    batch = scheduler.active_batch()  # include decode + any prefill chunks
    outputs = worker.forward(batch)
    
    for req, token in zip(batch, outputs):
        req.append_token(token)
        if req.is_done():
            scheduler.retire(req)
            block_manager.free(req.blocks)
        else:
            block_manager.maybe_allocate_block(req)
    
    scheduler.update_stats()
```

The `can_admit()` decision is the crux. It must weigh:
- Current batch compute time (do we have headroom?).
- KV memory (will new request fit?).
- Priority / SLA of waiting requests.
- Whether we're in a prefill-heavy or decode-heavy regime.

---

# Part 5 — Debugging methodology

## 5.1 The diagnosis ladder

The cheap mistake at staff level is to jump to algorithmic explanations before ruling out physical ones. A hardware-flaky GPU can look like a training instability for weeks before someone finally runs a health check. Always descend in this order:

```
                       ┌──────────────────────────────┐
                       │  Symptom observed            │
                       │  (slow, wrong, crashing)     │
                       └──────────────┬───────────────┘
                                      │
                                      ▼
                       ┌──────────────────────────────┐
                       │  1. Hardware health          │  <- check FIRST
                       │  - ECC errors, SDC           │
                       │  - GPU throttling            │
                       │  - Link speed (PCIe/NVLink)  │
                       │  - Thermal                   │
                       └──────────────┬───────────────┘
                                      │ clean
                                      ▼
                       ┌──────────────────────────────┐
                       │  2. System configuration     │
                       │  - Driver/firmware versions  │
                       │  - NCCL config, IB MTU       │
                       │  - NUMA pinning              │
                       │  - Container runtime         │
                       └──────────────┬───────────────┘
                                      │ clean
                                      ▼
                       ┌──────────────────────────────┐
                       │  3. Data                     │
                       │  - Dataset corruption        │
                       │  - Tokenizer drift           │
                       │  - Sharding imbalance        │
                       │  - Special tokens            │
                       └──────────────┬───────────────┘
                                      │ clean
                                      ▼
                       ┌──────────────────────────────┐
                       │  4. Algorithmic / code       │
                       │  - Model bug                 │
                       │  - Optimizer config          │
                       │  - Numerics (loss scale,     │
                       │    underflow, overflow)      │
                       └──────────────────────────────┘
```

### Why this order

- **Hardware is flaky and stochastic**. Diagnosing "intermittent NaN loss spike" as an optimizer bug when it's actually an SDC on rank 47 can burn a team for weeks. A 2-minute `nvidia-smi` sweep catches the obvious cases.
- **System configuration is silent**. A PCIe link downgraded to x4 doesn't announce itself. A driver mismatch between nodes causes NCCL to fall back to slower paths. These are the second-cheapest to check.
- **Data issues masquerade as model issues**. A bad shard can train fine for 1000 steps then crash with NaN when you hit the bad sample. Logging seen samples and hashing data shards is cheap.
- **Only after the above** do you look at your model code.

### Cheap hardware health checks

```bash
# ECC errors and GPU state
nvidia-smi -q | grep -E "ECC|Errors|Throttle|Temp|Power"

# Link speed (should be x16 Gen4 on H100)
nvidia-smi --query-gpu=pci.bus_id,pcie.link.gen.current,pcie.link.width.current --format=csv

# NVLink status
nvidia-smi nvlink --status

# Peak bandwidth test (intra-node)
./nccl-tests/build/all_reduce_perf -b 1M -e 1G -f 2 -g 8
# Expect > 250 GB/s busbw on H100 within a node
```

Red flags:
- Any ECC error count > 0 (especially uncorrected).
- PCIe gen/width below expected.
- NVLink reported down on any link.
- NCCL bandwidth < 60% of theoretical.

## 5.2 Bisection under distributed conditions

When one rank misbehaves in a 1024-rank job, the challenge is finding which rank without restarting the world. Strategies, in order of leverage:

### 1. Identify-by-elimination via per-rank logs

```python
# Cheap instrumentation in every training loop
import torch
import torch.distributed as dist

rank = dist.get_rank()
with open(f"/tmp/health_rank_{rank}.log", "a") as f:
    f.write(f"step={step} loss={loss.item():.4f} grad_norm={grad_norm:.4f} "
            f"elapsed_ms={step_elapsed_ms:.1f}\n")
```

After a failure, diff per-rank logs. Outlier rank shows up immediately.

### 2. Flight recorder

PyTorch's NCCL flight recorder captures the last N collective operations per rank with timestamps. On hang/timeout it dumps them. Reveals which rank is lagging and what collective.

```
  Enable: TORCH_NCCL_TRACE_BUFFER_SIZE=2000
          TORCH_NCCL_DUMP_ON_TIMEOUT=1
  
  On hang:
    Rank 0: last op AllReduce @ step 12345, completed ok
    Rank 47: last op AllReduce @ step 12345, STARTED but not completed
    ^-- rank 47 is the slow one
```

### 3. Binary search across dimensions

If all else fails, you can bisect:

- **Across ranks**: disable half the ranks, does problem go away? If yes, bisect the enabled half.
- **Across layers**: turn off gradient updates for half the layers, does divergence go away? Narrows to which layer is introducing the issue.
- **Across microbatches**: save a known-good checkpoint, run one microbatch at a time, find which introduces instability.
- **Across time**: bisect on commits — did this bug exist 2 weeks ago?

### 4. Check specific pathologies

Single-rank failures are usually:
- Bad GPU (SDC, ECC, throttling).
- Bad IB cable/switch port to that node.
- Unique dataset shard assigned to that rank has an issue.
- NUMA imbalance on that node's CPU.

Check all of these with targeted tools rather than guessing.

## 5.3 Reproducing rare failures

### Determinism flags

```python
import torch
torch.use_deterministic_algorithms(True, warn_only=True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Forcefully disable non-deterministic reductions in cuBLAS
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

# Seed everything
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
```

### NCCL seed control

NCCL doesn't directly seed, but the order of operations matters. For reproducibility:
- Pin ranks to specific GPUs deterministically.
- Same NCCL version across runs.
- `NCCL_ALGO` set explicitly (not `auto`) to prevent algorithm selection drift.
- `NCCL_PROTO=LL,LL128,Simple` enumerated explicitly.

### RNG state sharding

In data-parallel training, each rank needs a different RNG for dropout / data shuffling. Canonical pattern:

```python
import torch

rank = dist.get_rank()
gen = torch.Generator(device="cuda")
gen.manual_seed(base_seed + rank)

# Use this generator explicitly in ops that need randomness
dropout_mask = torch.bernoulli(probs, generator=gen)
```

For full reproducibility across cluster sizes, use a seed scheme tied to global position (e.g., layer index, microbatch index) rather than rank. This way, changing TP or PP doesn't change which bits of noise go where.

### Catching rare NaNs

```python
# Anomaly detection slows training significantly but catches the first bad op
torch.autograd.set_detect_anomaly(True)

# Cheaper: forward hook that checks output for NaN
def nan_check_hook(module, input, output):
    if torch.isnan(output).any() or torch.isinf(output).any():
        print(f"NaN/Inf at {module.__class__.__name__}")
        import pdb; pdb.set_trace()

for name, m in model.named_modules():
    m.register_forward_hook(nan_check_hook)
```

## 5.4 Numerics debugging

### The core observation

Two training runs with identical code, data, and seeds can diverge within hundreds of steps due to non-deterministic kernel scheduling. This is not a bug; it's the nature of floating-point accumulation order. The question is: when divergence happens between a "good" and "bad" run, how do you attribute cause?

### Per-layer norm comparison

Dump activation norms and gradient norms per layer per step for both runs. The layer where norms first diverge significantly is where to look.

```python
def dump_norms(model, step):
    out = {}
    for name, p in model.named_parameters():
        out[f"param_{name}"] = p.detach().norm().item()
        if p.grad is not None:
            out[f"grad_{name}"] = p.grad.detach().norm().item()
    with open(f"norms_{step}.json", "w") as f:
        json.dump(out, f)
```

Compare across runs:

```python
# Diff the norms
with open("norms_100_run_A.json") as f: a = json.load(f)
with open("norms_100_run_B.json") as f: b = json.load(f)
for k in a:
    diff = abs(a[k] - b[k]) / max(abs(a[k]), 1e-8)
    if diff > 0.01:
        print(f"{k}: A={a[k]:.3e} B={b[k]:.3e} relative_diff={diff:.3%}")
```

### Tensor hashing

For a known-determined checkpoint, compute hashes of every parameter tensor. Compare across runs to find the first divergence point.

```python
import hashlib
def tensor_hash(t):
    # Deterministic byte view; requires CPU
    return hashlib.sha256(t.detach().cpu().numpy().tobytes()).hexdigest()[:16]

for name, p in model.named_parameters():
    print(f"{name}: {tensor_hash(p)}")
```

### Loss curve bisection

If loss diverges between two runs:
- Plot both losses on the same axis.
- Identify first step where they differ by > threshold.
- Checkpoint both just before that step.
- Re-run from the checkpoint with `torch.autograd.set_detect_anomaly(True)` or enhanced logging.
- Narrow to which forward or backward op introduces divergence.

### When to suspect hardware

Red flags suggesting silent data corruption on a specific GPU:
- Divergence is rank-specific (one rank's grads are anomalous).
- Divergence is intermittent and correlates with specific GPU or node.
- Loss spikes recover, but perplexity retrospectively shows a slow degradation.
- Bit-exact reruns disagree on the same GPU.

Tools:
- `dcgmi diag -r 3` — NVIDIA's data center GPU diagnostic. Catches most hardware issues.
- Meta's "Project Helios" style SDC detection: run reference matmuls periodically, hash outputs, compare across ranks that should be identical.
- Periodic MatMul checks: compute the same matmul on adjacent GPUs, compare hashes.

## 5.5 The hardest bug categories

### Silent data corruption (SDC)

**Symptom**: training loss occasionally spikes, recovers, but model quality degrades slowly. Bit-exact runs don't match across attempts even with full determinism flags set.

**Cause**: a specific GPU (or set of GPUs) produces incorrect arithmetic on rare input patterns. Often thermal or voltage-related, sometimes manufacturing defect. ECC does not catch compute errors, only memory errors.

**Diagnosis**: run a known matmul on every GPU, hash outputs, look for outliers. This is literally how hyperscalers detect bad GPUs in their fleet.

**Vignette** (the kind you'll be asked about):
> "We trained a 405B model on 4096 H100s. The run was stable for 6000 steps, then loss spiked. We rolled back and restarted; it happened again at a different step. MFU was normal. No ECC errors. We ran grad-norm per rank: one rank was producing grad norms 50× larger than others, intermittently. That rank's GPU was on a node where CPU NUMA pinning was wrong, causing occasional HBM access thrashing through the wrong PCIe complex — which correlated with thermal pressure — which triggered an undocumented silent corruption in the tensor core path. Workaround: pin that node's CPU cores correctly. Real fix: decommission the GPU after replication across the fleet showed the same SDC signature under load."

### NCCL deadlock under specific topology

**Symptom**: training runs fine for hours, then all-reduce hangs indefinitely. No errors.

**Cause**: NCCL's algorithm selection picks a path that deadlocks under some topology or load condition. Seen with specific combinations of IB firmware + PXN + asymmetric rail configurations.

**Diagnosis**: `TORCH_NCCL_DUMP_ON_TIMEOUT=1`, check flight recorder. `NCCL_DEBUG=TRACE` to see exactly which channels are stuck.

**Workaround**: force `NCCL_ALGO=Ring`, disable `NCCL_PXN_DISABLE=1`, test with smaller world size to isolate.

### Driver/firmware mismatch across fleet

**Symptom**: job works on most nodes, fails on a specific subset with cryptic CUDA errors or hangs.

**Cause**: in a multi-tenant cluster, driver updates may not be applied atomically. Nodes in the bad set have driver version N, others N+1.

**Diagnosis**: collect `nvidia-smi -q | grep "Driver Version"` and GPU firmware versions across all nodes. Diff.

**Fix**: pin scheduler to nodes with matching versions; drain non-conforming nodes.

### Thermal throttling masked by scheduler

**Symptom**: MFU drops 20% over a long training run, slowly. Restarting restores performance temporarily.

**Cause**: some GPUs are thermally throttling (clock speeds drop under sustained load). The scheduler or framework doesn't report this cleanly.

**Diagnosis**: log `nvidia-smi --query-gpu=temperature.gpu,clocks.current.sm --format=csv,noheader` every 10s. Correlate throttled GPUs to slow-step ranks.

**Fix**: improve cooling or redistribute workload. In extreme cases, blacklist specific racks.

### PCIe downgrade (x16 → x8 → x4)

**Symptom**: host-to-GPU transfer latency is 2–4× higher on specific GPUs. MFU impact small but dataloader bottleneck visible on those ranks.

**Cause**: bent pins, partial seat, cold socket, firmware bug.

**Diagnosis**: `nvidia-smi --query-gpu=pcie.link.gen.current,pcie.link.width.current --format=csv` — anything below x16 Gen4 is a red flag.

**Fix**: physical inspection, reseat.

### NUMA misconfiguration

**Symptom**: CPU-side preprocessing is 2× slower than expected. Copies to specific GPUs are slower.

**Cause**: processes running on NUMA node 0 but reading from memory allocated on NUMA node 1. Cross-socket memory access is 2–3× slower.

**Diagnosis**: `numastat -p <pid>` shows where pages are allocated. `lstopo` shows the topology. If your DataLoader workers are not CPU-pinned to the NUMA node closest to their target GPU, you're paying cross-socket tax.

**Fix**:

```bash
# Pin dataloader process to NUMA node 0 (which hosts GPUs 0-3 typically)
numactl --cpunodebind=0 --membind=0 python train.py --local_rank=0
```

In Python: `psutil.Process().cpu_affinity([...])` or torchrun's `--localhost` CPU-affinity options.

### Container runtime interfering with MPS/MIG

**Symptom**: multi-instance GPU sharing doesn't split cleanly; one tenant's workload leaks into another's.

**Cause**: container runtime (Docker, containerd) may not pass through MIG device UUIDs correctly; MPS (Multi-Process Service) pipe configurations are host-level and can collide.

**Diagnosis**: inspect device cgroup, verify only intended MIG slice is exposed to the container. Check `nvidia-smi mig -lgi` from inside the container shows expected instances.

**Fix**: use NVIDIA Container Toolkit with explicit `NVIDIA_VISIBLE_DEVICES=MIG-xxxx-yyyy-zzzz`.

---

# Part 6 — Interview-specific preparation

## 6.1 The Perf & Debugging round format

### Typical structure (45–60 minutes)

1. **Intro / context** (5 min): interviewer describes a system or symptom.
2. **Open-ended diagnosis** (20–30 min): candidate walks through hypothesis space, asks clarifying questions, proposes experiments.
3. **Deep dive on one thread** (10–15 min): interviewer picks one of candidate's hypotheses and drives into specifics. Often here they inject curveballs ("what if PCIe is fine?").
4. **Back-of-envelope math** (optional, 5–10 min): "how much HBM does that cost?" / "what's the expected bubble fraction?" — candidate computes live.
5. **Wrap / candidate questions** (5 min).

### What interviewers look for at staff level

- **Calibrated confidence**: "I'm 80% sure X, let me verify." Not "it's definitely X" or "I have no idea."
- **Breadth + one deep**: you can enumerate 6 hypotheses and pick the one most worth investigating first, then go 3 levels deep on that one.
- **Numbers on demand**: can you produce an HBM calculation in 60 seconds without a calculator?
- **Systematic thinking over pattern matching**: you explain *why* a hypothesis is likely, not just that it is.
- **Debugging under pressure**: when the interviewer pushes back, you update beliefs rather than doubling down.
- **Ownership of tradeoffs**: "this will cost 2% MFU but buy us 30% MBU, acceptable because decode is our bottleneck."

### Common failure modes

- Jumping to "it's a model bug" before checking hardware.
- Reciting known optimizations without justifying why they apply *here*.
- Refusing to commit to a hypothesis ("could be anything").
- Committing too hard ("it's definitely X") when ambiguous.
- Getting lost in details and not zooming out when asked.
- Unable to produce numbers when asked (staff-level red flag).

## 6.2 Back-of-envelope fluency drills

Below are 15 drills. A staff candidate should produce the answer in ≤ 60 seconds with mental math, not a calculator.

### Drill 1: HBM for inference

> 70B model, BF16, 8k context, batch 32, GQA 8:1 (H_kv = H/8). How much HBM?

Weights: 70B × 2 = 140 GB.
KV cache: `2 × L × H_kv × D × S × B × 2 bytes`. For Llama-3-70B: L=80, H_kv=8, D=128. 
`2 × 80 × 8 × 128 × 8192 × 32 × 2 = 86 GB`.
Activations (working set): ~5–10 GB typical for attention/MLP temporaries.
**Total: ~240 GB**. Need ≥ 4× H100 (TP=4 at 80 GB each).

### Drill 2: Decode throughput

> 70B BF16 on 8× H100 with TP=8. What's single-stream decode throughput?

Per token, each GPU loads 140/8 = 17.5 GB. At 3.35 TB/s HBM, that's 5.2 ms per token.
Plus all-reduce: negligible (~0.1 ms).
**~190 tok/s per stream**.

For a higher batch (say B=16, fitting in the remaining KV budget), throughput scales roughly linearly until KV cache dominates HBM.

### Drill 3: Prefill compute

> 70B BF16, 8k context prefill on 8× H100 TP=8. Time?

FLOPs per token prefill ≈ 2 × N_params = 2 × 70e9 = 140 GFLOP/token. (Forward only; no bwd at inference.)
Plus attention: `2 × S × H` per token × num_layers = `2 × 8192 × 8192 × 80 ≈ 10 GFLOP/token` attention.
Total ≈ 150 GFLOP/token. For 8192 tokens: 1.23 PFLOP.
Peak BF16 = 989 TF/GPU × 8 = 7912 TF/s.
At 50% MFU: 3956 TF/s.
**Time ≈ 1.23e15 / 3.96e15 ≈ 310 ms**.

### Drill 4: Ring all-reduce time

> 8 GPUs, 1 GB BF16 gradient, ring all-reduce. Time?

`T = 2(N-1)/N × M / BW = 2 × 7/8 × 1 GB / 450 GB/s ≈ 3.9 ms`.
(NVLink busbw typically ~450 GB/s on H100 ring for all-reduce.)

### Drill 5: KV cache per request at long context

> MLA model with compressed KV dim 128, L=80, 128k context. KV per request in BF16?

`2 × L × D_c × S × 2 bytes = 2 × 80 × 128 × 131072 × 2 ≈ 5.4 GB`.

Compare to Llama-3-70B at 128k: `2 × 80 × 8 × 128 × 131072 × 2 = 43 GB`. MLA is ~8× smaller.

### Drill 6: FSDP overlap feasibility

> 70B model, FSDP across 8 H100. Forward pass of one block (B=1, S=4096, BF16) vs AllGather of one block's params.

Block params: 70e9 / 80 ≈ 875M × 2 bytes = 1.75 GB.
AllGather (7/8 × 1.75 GB / 900 GB/s) ≈ 1.7 ms.
Block compute (~2 × 875M × 4096 × 1 ≈ 7.2 GFLOP per sample × 1 sample / 989 TF @ 50% MFU) ≈ 0.015 ms.
**Compute is 100× shorter than AllGather.** Overlap impossible with B=1. Need B large or TP inside FSDP.

### Drill 7: All-to-all cost for MoE

> MoE with H=8192, B=4, S=8192, top_k=2, EP=8, BF16. Bytes per all-to-all?

`B × S × H × 2 × top_k × (EP-1)/EP = 4 × 8192 × 8192 × 2 × 2 × 7/8 = 0.94 GB`.
Two all-to-alls per MoE layer (dispatch + combine) = 1.88 GB per layer per forward.
At 900 GB/s: 2 ms per layer per forward. ×32 layers × 2 (fwd+bwd) ≈ 128 ms per step.

### Drill 8: FP8 crossover K

> FP8 GEMM on H100. Smallest K for which it's compute-bound, assuming M=N=4096?

AI = `2MNK / (2(MN + MK + NK))` in elements, × 1 byte for FP8 → AI(bytes) = `MNK / (MN + MK + NK)`.
Ridge is 591 FLOP/byte.
Square shapes: AI ≈ K/3. Need K/3 ≥ 591 → K ≥ 1773. Round to **K ≥ 1800**.

### Drill 9: NCCL busbw expected

> 8× H100 SXM in one node, all-reduce of 1 GB. What busbw should I see?

NCCL reports busbw = `2M/(N) / T × (N-1)`. Expected with NVLink4: ~450 GB/s busbw (85% of 900 GB/s peak × effective overhead) in practice.
Don't memorize formulas — memorize the 450 GB/s figure for H100 intra-node.

### Drill 10: MBU for decode

> 70B BF16, 8× H100 TP=8, measured 180 tok/s single stream. MBU?

Bytes per token: weights (17.5 GB/GPU) + KV read (negligible scaled to single GPU). 
Per-GPU BW: 180 × 17.5 GB/s ≈ 3.15 TB/s.
Peak: 3.35 TB/s.
**MBU ≈ 94%**. Excellent.

### Drill 11: PP bubble fraction

> 4-stage PP, 16 microbatches, 1F1B. Bubble?

`(P-1)/(M+P-1) = 3/19 ≈ 15.8%`.

### Drill 12: HBM cost for KV at scale

> 100 concurrent requests, Llama-3-70B, avg 2k context. How much KV HBM?

Per request at 2k: 86 GB × 2048 / 8192 / 32 = 0.67 GB.
Actually easier: `2 × 80 × 8 × 128 × 2048 × 2 = 670 MB` per request.
100 requests: 67 GB. Plus weights 140 GB. Plus slack → need 8× H100 minimum for 100-concurrent 2k avg at 70B.

### Drill 13: Quantization memory savings

> 70B BF16 → W4A16 (INT4 weights, BF16 acts). Memory saving?

Weights: 140 GB → 35 GB (4×).
KV cache unchanged (BF16).
Activations unchanged.
At decode, decode is memory-bound on weights → throughput improves ~3–3.5× (not 4× due to dequant overhead).

### Drill 14: Tensor core flops utilization

> My kernel reports 80% dram throughput and 40% tensor core utilization. What's the bottleneck?

Memory. DRAM is at 80%, tensor core at 40%: most cycles are spent waiting on memory.
If I increase tile size (reducing DRAM traffic per FLOP), tensor core util will rise and DRAM util will drop. 
If the kernel is already at max tile size, then **it's inherently memory-bound at this shape**.

### Drill 15: NVLink saturation for TP

> 8-way TP, H=16384, BF16. All-reduce time per token?

Per AR: `2H × 2 bytes = 65 KB`. Tiny.
Ring AR on 8 GPUs: `2 × 7/8 × 65 KB / 450 GB/s ≈ 250 ns`.
Compared to per-token decode compute of milliseconds, this is negligible. TP scales well for decode.

## 6.3 Narration patterns

The way you talk while debugging signals competence as much as the answer does. Use this scaffold:

> "The symptom is [X]. The hypothesis space is {A, B, C}. I rank them [ordering] because [reasons]. The cheapest to distinguish is [X] because [cost argument]. I'd do [experiment] first. If that returns [expected positive signal], I'd conclude [Y]. If it returns [expected negative signal], I'd move to [next hypothesis]."

### Example narration 1 — slow training step

**Prompt**: "Your 1024-GPU job ran fine yesterday, today steps are 30% slower. Diagnose."

**Narration**:
> "First, the symptom is a 30% step-time regression across a full job. Hypothesis space: (a) hardware — one or more GPUs throttling, bad links; (b) system — driver update, container change, scheduler placement change; (c) data — new shards, longer sequences; (d) code — someone pushed a regression; (e) contention — noisy neighbors on shared infra.
>
> I'd rank hardware and system highest because 'worked yesterday, broken today' strongly suggests environmental change. Data is third — did the sharding just roll? Code is fourth — what commits went in overnight?
>
> Cheapest first: check if any commits went into the training image or launch scripts in the last 24h. That's a 30-second git log. Assume that's clean for the sake of discussion.
>
> Next: per-rank step time from profiler or just wall-clock logs. Is every rank 30% slower, or is one rank 130% slower and blocking everyone? If uniform: driver or fleet-wide issue. If one rank: bad hardware on that rank.
>
> Assuming one rank: `nvidia-smi -q` on that host for throttle reasons, ECC errors, PCIe link state. Also check IB port counters on that node.
>
> If it's fleet-wide: check driver version distribution, NCCL version. Check if any cluster maintenance happened overnight. Check IB subnet manager for topology changes."

### Example narration 2 — numerical divergence

**Prompt**: "Training loss suddenly spikes at step 12000 on your new FP8 training run. Diagnose."

**Narration**:
> "FP8 training is numerically delicate, so my prior is that it's a numerics bug rather than hardware. Hypothesis space: (a) loss scale is wrong or hasn't adapted; (b) a specific layer's activations saturated FP8 range; (c) optimizer state in FP8 underflowed; (d) a bad gradient from data outlier; (e) hardware SDC, but deprioritize given recency of FP8 stack.
>
> First check: did gradient norm spike or blow up before loss? If yes: bad data or gradient explosion. If no: the loss itself is the first symptom, implying forward pass produced a bad prediction.
>
> Second: log FP8 scale factors per tensor. If any tensor hit max scale (saturated), that's our answer. This is essentially free to add.
>
> Third: dump activation norms per layer at step 11999 and step 12000. Find the first layer where norms differ meaningfully. That's where the bug manifests.
>
> Fourth: bisect on the specific microbatch. Save the dataloader state at step 11990, step through each batch — does a specific sample trigger the divergence? If yes, we have a problematic data sample and can investigate (long, repetitive, unusual tokens, etc.).
>
> If the layer turns out to be attention and scales are healthy, I'd also check the softmax temperature / attention score distribution at that layer."

### Example narration 3 — decode throughput low

**Prompt**: "Your Llama-70B inference server is hitting 40 tok/s per stream, you expect 200. Fix."

**Narration**:
> "For 70B on 8× H100 TP=8, 200 tok/s is about the right order — that's MBU ~95%. 40 tok/s is 20% of target. Something major is wrong.
>
> Hypothesis: (a) TP not configured — model is on 1 GPU with the others idle; (b) attention kernel is slow — not using FlashAttention; (c) KV cache layout is wrong — not paged, high stride; (d) overhead per token — CPU scheduler latency, Python overhead, cuda graph not used; (e) weights in wrong format — FP32 instead of BF16.
>
> First check: `nvidia-smi` on all 8 GPUs during inference. If only one shows high util, TP isn't really happening.
>
> Second: kernel trace with nsys for a single decode step. What ops dominate? If you see many small kernels per step (norms, rotary, sampling each their own launch), CUDA graphs are disabled — that's easily 3–5× overhead per token on decode.
>
> Third: are we using paged attention? FlashAttention? Check server config.
>
> Fourth: is it a single-stream test? At B=1 we expect to be near MBU limit, not far below. Confirm we're actually measuring steady-state decode and not prefill-included time."

### Example narration 4 — NCCL hang

**Prompt**: "Training hangs on step 500. No errors. Recover and find root cause."

**Narration**:
> "Hang after 500 clean steps suggests deterministic trigger at step 500, not random flakiness. Hypothesis: (a) mismatched collective — some rank diverged in call pattern, likely due to data-dependent branching; (b) one rank OOM'd silently and stuck in cuda-side recovery; (c) NCCL algorithm chose a deadlocking path under some load condition; (d) uneven seq length triggered a branch that some ranks took and others didn't.
>
> Immediate: `py-spy dump` on all ranks to see where each rank is stuck in Python. If they're all in AllReduce: normal NCCL hang. If some are past it: a rank skipped a collective.
>
> Enable `TORCH_NCCL_DUMP_ON_TIMEOUT=1` and wait for the timeout. The dump will show each rank's last pending op.
>
> Most likely root cause given 500-step lag: there's a data-dependent codepath — maybe 'if loss < threshold: extra logging' or 'if seq_len > X: skip grad norm' — where some ranks take it and others don't. Find the first step where ranks diverge in call sequence and look at the code there."

### Example narration 5 — sudden MFU drop

**Prompt**: "MFU was 50%, dropped to 35% after adding a new component. Diagnose."

**Narration**:
> "I assume 'new component' means a code change. Hypothesis: (a) the component introduced a synchronization point (CPU-GPU sync, host-triggered kernel launches) that breaks CUDA graph; (b) it added an unfused op sequence that adds HBM traffic; (c) it changed kernel launch pattern enough to lose persistent-kernel benefits; (d) it's on the critical path of a previously-overlapped comm; (e) it does blocking NCCL collectives that serialize what was previously parallel.
>
> First: profiler trace before/after. Overlay CPU and GPU streams. Is there a new GPU idle gap in the step? Where?
>
> Second: look at the component. Does it call `.item()`, `.cpu()`, `.cuda()` sync, print tensor values? Any of those are synchronization points.
>
> Third: is it running on every rank or some ranks? A component that runs only on rank 0 and others wait for it is a common cause of MFU drop.
>
> Fourth: has it been added inside the forward/backward path, or as a post-step? If it's outside the training step, it shouldn't affect MFU at all — so check we're measuring correctly (just step time ÷ target, and where step time is measured)."

## 6.4 When you don't know

Staff candidates are expected to hit the edge of their knowledge constantly — the interviewer is probing for it. What matters is what you do next.

**Pattern 1 — name the gap precisely**:
> "I haven't personally worked with NVL72 at rack scale, but my mental model is [X] based on [public sources / first principles]. Happy to be corrected."

**Pattern 2 — derive rather than recite**:
> "I don't remember the exact formula for zero-bubble steady-state, but I can derive it: W ops can fill any gap because they only depend on local activations and downstream grad. So if we have enough W ops to fill (P−1)×step_time of bubble, we reach zero bubble. That's O(M) ≥ O(P), so..."

**Pattern 3 — offer to investigate**:
> "I'm not sure of FP4's stability under stochastic rounding at long context — this is the area I'd want to run experiments on, and I'd start by [specific experiment]. Let me describe how I'd design the investigation."

**Anti-patterns** (never do):
- Making up numbers confidently ("I think H200 does 6.8 TB/s"). Wrong. Bluffing is a career-limiting move.
- Saying "I don't know" and stopping. Follow with a derivation or a plan.
- Long throat-clearing ("well it depends on a lot of factors...") without commitment.

## 6.5 Pushback management

The interviewer often pushes back mid-answer to test how you handle disagreement.

### Types of pushback

1. **Genuine correction**: interviewer knows something you got wrong. Update.
2. **Challenge to test conviction**: interviewer wants to see if you hold firm when correct, update when wrong.
3. **Red herring**: interviewer is testing if you chase every objection or prioritize.

### Decision rubric

> "When someone pushes back, I first try to understand: are they giving me new information, or testing my reasoning?"

If new information ("actually, we're on Hopper, not Blackwell"): **update and re-derive**. Don't be attached to your previous answer.

If they're challenging your reasoning ("are you sure all-reduce is that cheap?"): **restate your derivation** and identify where you might be wrong. If you're confident, say so and point to the step: "I'm confident because bytes per rank per AR at that shape is 65KB; even at reduced effective NVLink BW it's sub-microsecond."

If it's a red herring (contradicting a minor detail that doesn't change the conclusion): **acknowledge and move on**. "That might be right for X case, doesn't change the answer here because Y."

### When to hold your ground

- You have a derivation, they're asserting a fact you can check against the derivation.
- Your answer follows from established physics (roofline, memory hierarchy) and their challenge doesn't address the physics.
- You've already qualified with confidence ("80% sure") and they're not providing new evidence.

### When to update fast

- They give you a specific number or constraint you didn't know.
- Your reasoning had an assumption they contradicted and the contradicted assumption was load-bearing.
- They reveal domain knowledge (e.g., "we have custom IB firmware that behaves differently") that invalidates part of your model.

The worst outcome is stubborn wrongness. The second-worst is flip-flopping at any breeze. Staff candidates sit in the middle: structurally confident, promptly updating on new information, willing to say "I was wrong about Y, which changes my answer to Z."

---

# Part 7 — Frontier topics

## 7.1 Blackwell-specific

### Tensor Memory (TMEM)

A 256 KiB per-SM scratchpad, distinct from SMEM. The tensor core reads inputs from SMEM (or TMEM) and writes accumulators directly to TMEM. Key implications:

- Accumulator no longer in registers → larger tiles without register pressure explosion.
- `tcgen05.mma.async` can target TMEM without a CTA sync.
- TMEM has its own load/store instructions (`tcgen05.ld`, `tcgen05.st`) distinct from LDS/STS.

### CTA pairs

Two adjacent SMs can share a tensor core issue. The `.cta_group::2` modifier on `tcgen05.mma` instructs the hardware to split the tile between the pair. Doubles effective tile size in M (or N) without doubling per-SM register/TMEM cost. Requires that both CTAs in the pair be co-resident, which constrains scheduling.

### 5th-generation tensor cores

- Native FP4 (E2M1) support with block scales at group 16 (NVFP4) or group 32 (MXFP4).
- Block scales in E8M0 format — 8 exponent bits, no mantissa — exactly representable powers of 2.
- FP6 also supported at slightly lower throughput.
- FP4 peak is 2× FP8 peak at the same power envelope.

### FP4 training viability

The practical question: can you train at FP4 without losing quality? Current state (rapidly changing):

- Forward at FP4 with NVFP4 (block-16 scaled): works for well-conditioned MLPs and attention projections. Loss curves match BF16 within <1% perplexity delta.
- Backward at FP4: much harder. Gradients have heavier tails. Block scaling helps but some layers (especially embedding and final projection) need to stay at FP8 or BF16.
- Optimizer state at FP4: not viable currently. Needs FP32 or at minimum BF16.

**The realistic near-term recipe**: FP8 forward/backward with FP4 for selected layers' forward only. Full FP4 training is research frontier.

### NVL72 topology

72 Blackwell GPUs in one NVLink domain. 130 TB/s aggregate bisection bandwidth. Implications:

- Collectives that previously required inter-node IB now stay on NVLink.
- Effective "intra-node" size jumps from 8 to 72, changing parallelism strategies.
- TP=72 becomes feasible; EP=72 becomes feasible for large MoE.
- SHARP (Scalable Hierarchical Aggregation and Reduction Protocol) on NVL72 can do in-network reduction, cutting AR bandwidth cost.

### Tile-level hints for Blackwell kernels

- Use TMEM for accumulators universally in matmul kernels.
- Target CTA-pair tiles when the problem has M or N ≥ 256.
- Use `cp.async.bulk.tensor` (TMA 2.0) for 5D descriptors with swizzle.
- Verify with Nsight that TMEM utilization is high; low TMEM use on Blackwell is wasted silicon.

## 7.2 MI300X / MI350 considerations

### Where AMD wins

- **HBM capacity**: MI300X at 192 GB, MI325X at 256 GB, MI350X at 288 GB. A 405B model in BF16 fits on 4× MI325X vs 8× H100. Decode memory pressure is materially lower.
- **HBM bandwidth parity**: MI350X at 8 TB/s matches B200.
- **Infinity Fabric intra-node**: 896 GB/s on MI300X, competitive with NVLink 4.
- **Unified memory with CPUs** (on MI300A variants): removes host-GPU copies entirely for some workloads.

### Where AMD lags

- **Software ecosystem**: CUDA has 15+ years of tooling. ROCm is catching up but kernel libraries, profilers, frameworks all trail.
- **Triton-ROCm maturity**: works for most ops but hits edge cases. Some FP8 paths not yet at parity with NVIDIA.
- **RCCL vs NCCL**: RCCL is NCCL ported. Functionally equivalent but less mature in tuning for specific topologies.
- **Tensor core equivalent**: MI300X's matrix cores are comparable in theoretical throughput but the software to saturate them (CK — Composable Kernel — the ROCm equivalent of CUTLASS) is less ergonomic.

### FP8 story on AMD

MI300X supports FP8 in hardware. In practice, production FP8 training on AMD is ~6 months behind NVIDIA — you can do it, but expect more kernel tuning and more edge cases. FP8 inference is in better shape.

### Practical portability

- PyTorch runs on ROCm with same API surface.
- Triton runs on ROCm, code ports with minor tweaks.
- Custom CUDA → HIP is a real effort (hipify-perl works for most code, but performance tuning must be redone).

### When to use AMD

- You need HBM capacity more than ecosystem.
- You have software engineering budget to maintain parallel kernel paths.
- You want supply diversification.

### When to avoid

- Small team, tight timeline, need production quality today.
- Workload is bleeding-edge (FP4 training, latest attention variants).

## 7.3 Scaling laws for inference

### The decode-bound regime

For a model with `N` parameters, decode throughput per stream scales as `HBM_BW / (N × bytes_per_param)`. Increasing `N` linearly hurts latency.

But effective throughput per GPU, with batching, scales differently:
- Batch size B multiplies arithmetic intensity by B.
- At high enough B, AI crosses the ridge and decode becomes compute-bound.
- The "batch size break-even" is roughly `ridge_AI / baseline_AI ≈ ridge_AI × bytes_per_param`.

For H100 FP8: ridge ≈ 591 FLOP/byte. Baseline decode AI ≈ 1. So B ≈ 591 is where you saturate compute. In practice KV cache constraints cap B well before that.

### Optimal parallelism shifts with context

For short context (S < 4k), weights dominate KV. TP to fit model, DP for throughput.

For mid context (4k–32k), KV grows. May need larger TP group or offload.

For long context (32k+), KV is dominant. CP (ring attention) becomes useful. Parallelism should be chosen to balance KV memory per rank.

```
  Optimal parallelism as function of context:
  
  S < 4k:    TP=min_for_weights, DP for throughput
  4k–32k:    TP larger (to spread KV), DP still for throughput
  32k–128k:  TP + CP mix; each dimension cuts different memory
  128k+:     CP dominant (ring attention), TP moderate
  1M+:       PD disaggregation + multi-tier KV (HBM + CPU + NVMe)
```

### Sequence-dependent amortization

As S grows, each token's prefill amortizes over more compute. This argues for aggressive chunked prefill + prefix caching for workloads with repeated system prompts.

## 7.4 The economics layer

MAI's team name — Capacity & Efficiency — is the clue. Know how to frame any perf improvement in dollar terms.

### Cost per million tokens — serving

```
  $/M tokens = GPU_cost_per_hour × hours_per_M_tokens
             = (GPU_hourly) / (throughput_tok/s × 3600)
             × 1M

  Example: 8× H100 serving Llama-70B at 8000 tok/s aggregate.
  8× H100 on-demand ≈ $30/hr (rough).
  Throughput = 8000 tok/s = 28.8M tok/hr.
  $/M tok = $30 / 28.8 ≈ $1.04 per M tokens.
```

### Cost per training step

```
  $/step = GPU_cost_per_hour × num_gpus × step_time_hours
  
  Example: 1024-H100 training run. 
  Cluster cost ≈ $2000/hr (all-in, roughly).
  Step time 2.5s → 0.000694 hours.
  $/step = $2000 × 0.000694 ≈ $1.39.
  
  100k steps = $139k per run. Six runs per experiment = ~$1M.
```

### The tradeoff framing

Staff candidates frame all optimizations in this language:

> "This kernel improvement raises MBU from 75% to 85%. At current serving volume (50B tokens/day), that's 13% throughput → 13% fewer GPU-hours → at $30/hr per node × 100 nodes × 24 × 365 × 13% ≈ $3.4M/year."

> "FP8 training cuts HBM footprint 2× → per-GPU batch size 2× → MFU improves from 40% to 50% (better AI). Step time cuts 20%. Training run cost drops 20%. On a $30M run, that's $6M saved."

### Capacity planning framework

For a serving fleet:

1. **Demand model**: tokens/sec per region, diurnal curve, peak-to-avg ratio.
2. **Model mix**: what fraction of requests go to which model?
3. **Instance mix**: which hardware for which model? (70B on 8× H100; 8B on 1× H100 with MIG.)
4. **Headroom**: target 60–70% utilization to absorb bursts.
5. **Cost**: ∑(instance_cost × instance_hours × headroom_factor).

For a training fleet:

1. **Pipeline of runs**: how many large runs in flight at what size?
2. **Scheduler efficiency**: how much of the fleet is actually in use vs idle between runs?
3. **Checkpointing frequency**: checkpoint cost vs recovery cost (MTBF × step_cost / 2).
4. **Elasticity**: can you shrink in idle periods? Usually no for large runs — GPU reservation is expensive.

## 7.5 Recent papers worth knowing

### FlashAttention 3 (2024)

Warp-specialized async Hopper attention. TMA + wgmma producer/consumer. ~2× faster than FA-2 on H100. Published numbers: 740 TF/s for causal attention (~75% MFU).

*Why it matters*: it's the canonical demonstration that warp specialization + TMA is the new pattern; any serious attention kernel after 2024 must use it.

### ThunderKittens (Hazy Research, 2024)

A CUDA DSL sitting below Triton but above raw CUDA. Exposes "tiles" as first-class objects. Competitive with hand-written CUDA for attention kernels, much faster to write than CUTLASS.

*Why it matters*: interesting middle ground; good to know exists but ecosystem small.

### Triton 3.x changes

- Native block-scaled FP8/FP4 support.
- Warp specialization pragmas.
- `tl.async_copy` primitives.
- Better Hopper codegen.

### CuTe DSL evolution

CUTLASS 3.5+ exposes CuTe at Python level. Allows more ergonomic composition than CUTLASS 2.x template meta-programming. Expect this to be the preferred path for custom kernels on Blackwell.

### NVFP4 and MX-format training papers

MX-format (2023 OCP standard) defines block-scaled formats. NVIDIA's NVFP4 is a specific instantiation. Papers from 2024–2025 establish FP4 training viability for specific regimes. Key references: the MX paper itself, and the various "FP4 training" results from NVIDIA, Meta, DeepSeek.

### Quantization progression

```
  INT8-LLM / LLM.int8 (2022): W8A16 shown viable.
       ↓
  GPTQ (2022): INT4 weight quant via OBS.
       ↓
  SmoothQuant (2023): W8A8 viable via act-weight rescaling.
       ↓
  AWQ (2023): INT4 with salient-channel rescaling.
       ↓
  QServe (2024): W4A8 serving with improved kernels.
       ↓
  (ongoing): W4A4, W2 research, KV-cache INT4
```

### DeepSeek-V3 technical report

Practical demonstration of:
- FP8 mixed precision training at 671B scale.
- Multi-token prediction (MTP) for decode speedup.
- MLA (Multi-head Latent Attention) for KV compression.
- Auxiliary-loss-free load balancing for MoE.

*Why it matters*: a recipe book for what's actually working at frontier scale. Read it.

### Ring Attention (Liu et al., 2023)

Canonical CP paper. Establishes that attention comm overlaps with compute per-block when block size is chosen correctly.

### Star Attention / Striped Attention / Chunked Attention variants

Trade full attention for structured patterns that reduce CP communication. Useful for specific long-context deployments, not universal.

### StreamingLLM (2023)

Keep "attention sinks" (first few tokens) + sliding window. Unbounded context with bounded KV. Quality degradation for long-range dependencies.

---

## Coda

This primer isn't meant to be read linearly in one sitting; it's meant to be the thing you flip to before an interview. Three rehearsal passes:

1. **Numbers pass**: cover the tables and recite peak FLOPs, HBM bandwidths, ridge points, bubble fractions. Don't move on until you can do them cold.
2. **Diagrams pass**: redraw 5 of the diagrams on a whiteboard from memory. If you can't, re-read the section.
3. **Narration pass**: pick a scenario from 6.3, set a 5-minute timer, talk out loud to an empty room. Record yourself. Listen back for filler, bluffing, lack of commitment.

The goal is not to memorize everything. The goal is to have the *retrieval paths* warm — when an interviewer says "attention is slow," your brain should jump to FlashAttention 3, warp specialization, TMA, memory-bound vs compute-bound, and arithmetic intensity without thinking. The primer is scaffolding for that reflex.