---
title: "Optimization Track: Index and Study Guide"
description: "Map of the ML performance and optimization track — what each doc covers, the order to study them for staff-level TPU/GPU performance interviews, and a weekly plan."
sidebar:
  label: "Overview & Study Guide"
---

# Optimization Track: Index and Study Guide

Eighteen docs covering ML performance engineering at staff level: how models get faster, cheaper, and smaller — and how to reason about it out loud in an interview. This page is the map: what lives where, what order to read it in, and how to convert reading into interview-ready fluency.

## The index, by theme

**Foundations — measure first**

- [ML Performance Mental-Math Drills](/optimization/15-mental-math-drills/) — chip specs, roofline, and the eight identities you must do cold, with an [interactive trainer](/tools/perf-drills.html). Start here and never stop.
- [Perf Debugging](/optimization/17-perf-debugging/) — the deep reference: roofline, memory hierarchy, Nsight fluency, warp stalls, GEMM hierarchy, collectives, MFU/HFU/MBU.
- [Accuracy and Numerics](/optimization/4-accuracy-numerics/) — how optimizations silently regress quality, and the diffing discipline that catches it.

**Inference serving**

- [LLM Serving Optimization](/optimization/14-llm-serving-optimization/) — the canonical reference: continuous batching, paged KV, prefix caching, chunked prefill, disaggregation, goodput, benchmarking methodology.
- [TTFT Optimization Program](/optimization/1-ttft-optim/) — the applied worked answer: a staff-level optimization program from 1.8s to sub-500ms p99.
- [KV Cache Optimization](/optimization/8-kv-cache-long-context/) — KV memory math as capacity planning; paging, eviction, prefix reuse.
- [Speculative Decoding](/optimization/9-speculative-decoding/) — draft models, EAGLE/MTP, acceptance rate as the metric.

**Model compression**

- [LLM Quantization](/optimization/16-quantization/) — FP8/INT4/NVFP4, GPTQ/AWQ/SmoothQuant, prefill-vs-decode regimes. The highest-frequency inference interview topic.
- [Pruning](/optimization/2-pruning/) — canonical home for 2:4 and structured sparsity; why sparsity ≠ speed.
- [Sparse Models and Hardware Reality](/optimization/5-sparse-models-hardware/) — the Amdahl accounting that turns kernel speedups into (smaller) system speedups.
- [Knowledge Distillation](/optimization/3-knowledge-distillation/) — compressing capability; when to distill vs quantize vs route.

**Architecture-level levers**

- [Attention Optimization](/optimization/7-attention-optimization/) — canonical home for MQA/GQA/MLA/sparse-attention math.
- [Mixture of Experts](/optimization/6-mixture-of-experts/) — routing, load balancing, and the wide-EP inference-systems era (EPLB, DeepEP).

**Training at scale**

- [Tensor, Pipeline, Sequence, Context, and Expert Parallelism](/optimization/13-parallelism-stack/) — parallelism as a communication-design problem; the bottleneck→layout decision tree.
- [ZeRO and FSDP](/optimization/12-zero-fsdp-sharded-training/) — sharding model state; the 16-bytes/param math.
- [Activation Checkpointing](/optimization/11-activation-checkpointing/) — the memory↔compute dial sharding doesn't touch.

**Hardware and kernels**

- [FlashAttention and Kernel-Aware Optimization](/optimization/10-kernel-aware-optimization/) — IO-aware kernels, fusion, shapes, CUDA graphs.
- [TPU Performance: Architecture, XLA, and Pod-Scale Optimization](/optimization/18-tpu-xla-optimization/) — MXU/VMEM, XLA compilation, GSPMD sharding, Pallas, ICI topology, xprof.

## How to study

The failure mode to avoid is reading passively. Perf interviews score whether you can *derive* — do the arithmetic, name the bottleneck, propose the measurement — so every session should end with you talking through a problem, not highlighting a page.

**Daily, non-negotiable (15 min).** The [mental-math drills](/optimization/15-mental-math-drills/), cold, on paper, reasoning spoken aloud, under 2:00 per problem. This is the highest-leverage habit and it decays fastest.

**Week 1 — Foundations.** Drills + Perf Debugging (17) in chunks, with Accuracy & Numerics (4) as a palate cleanser. Goal: given any kernel or model op, you can place it on a roofline and say what you'd profile first.

**Week 2 — Inference serving.** Serving (14) → TTFT program (1) → KV cache (8) → speculative decoding (9). Then close the book and re-derive the TTFT program from scratch for a different scenario (e.g., a 8B model at 10× the traffic). Goal: fluency in the TTFT/TPOT/goodput vocabulary and the batching↔latency tradeoff.

**Week 3 — Compression and architecture.** Quantization (16) → pruning (2) → sparse-hardware reality (5) → distillation (3), then attention variants (7) and MoE (6). Goal: for any "make it cheaper" prompt, you can rank quantize/distill/prune/MoE-ify by expected win, risk, and engineering cost — and say which accuracy checks gate each.

**Week 4 — Scale and hardware.** Parallelism (13) → ZeRO/FSDP (12) → checkpointing (11) → kernels (10) → TPU/XLA (18). For the TPU role, read 18 twice and cross-check its numbers against the [JAX scaling book](https://jax-ml.github.io/scaling-book/). Goal: given a model size, chip, and pod, you can sketch the parallelism layout and say which axis saturates first.

**Final week — Integration.** Mock the two canonical prompts daily: "make inference cheaper" (serving + compression stack) and "make training faster" (parallelism + memory stack), 40 minutes each, out loud. Re-run every drill category. Skim each doc's closing interview-Q&A sections only.

**Three rules that hold throughout:**

1. **Numbers before names.** Never say "FlashAttention helps" without the bytes-moved argument for *this* shape. Techniques are conclusions, not answers.
2. **Measurement before optimization.** Every proposal starts with what you'd profile and what number would change your mind. That's the staff signal.
3. **State your regime.** Prefill or decode, compute- or memory-bound, latency- or throughput-bound — say which regime you're in before optimizing it; most wrong answers are right answers in the wrong regime.
