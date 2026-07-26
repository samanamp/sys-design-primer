---
title: "xprof: the TPU Profiler"
description: "Reading and diagnosing xprof/TensorBoard TPU profiles: trace viewer, op profile, memory viewer, the TPU fault catalog from exposed collectives to recompilation, and exercises."
---

# xprof: the TPU Profiler

xprof (the TensorBoard profiler for JAX/XLA) is the TPU counterpart of *all three* GPU tools at once: trace viewer ≈ nsys, op profile ≈ ncu's SOL at op granularity, memory viewer ≈ the allocator timeline. The XLA difference changes what "reading" means: the program is **one compiled module**, so faults are usually *compiler-visible decisions* (fusion breaks, sharding choices, remat) rather than per-kernel tuning — and the fix is usually an annotation, not a kernel.

## 1. Anatomy of the artifact

Capture: `jax.profiler.trace("gs://…/profile")` around 3–5 steps (or `--xprof` in MaxText), open in TensorBoard → Profile. Tabs in reading order:

```text
Overview page      — step time series, MXU utilization, input-bound %.
                     Read the verdict sentence first; it's usually right.
Trace viewer       — per-core timeline: TensorCore (MXU) row, sparsecore,
                     host threads, infeed/outfeed, ICI collectives.
Op profile         — time by HLO op, grouped by category (matmul / conv /
                     data formatting / all-reduce…), with FLOPS utilization
                     and memory-bandwidth utilization per op.
Memory viewer      — HBM usage over the program, per-buffer, peak & padding.
Pod viewer         — per-core step-time spread: the straggler instrument.
Input pipeline     — host-side analysis: infeed wait per step.
Graph viewer       — the HLO itself, for chasing a specific op back to JAX.
```

## 2. Reading order

1. **Overview verdict + step-time series.** Variance across steps → host/data/recompile; flat-but-slow → program-level.
2. **Op profile, category pie.** The staff summary in one view: matmul % (should dominate), *data formatting* % (transposes/reshapes — pure waste, usually a layout/sharding smell), all-reduce/all-to-all % (comm not hidden), vector/scalar % (VPU-bound ops).
3. **Trace viewer: are collectives under compute?** XLA emits async collective pairs (`all-gather-start/done`, `collective-permute-start/done`); exposed time is the gap where TensorCore sits idle between `start` and `done`. This is the [pod-training a2a-hides-iff inequality](/google-interview/6-answer-pod-training/), observed.
4. **Infeed check:** `infeed` ops or host-wait at step start → input pipeline tab → host analysis.
5. **Memory viewer** only when OOM-adjacent: peak buffer, padding waste, remat behavior.
6. **Pod viewer** for multi-core: step-time spread across cores; a hot core with longer *compute* (not comm) is load imbalance (MoE experts, uneven sequence sharding).

## 3. Fault catalog

### Major faults

| Fault | xprof signature | Confirming detail | Fix direction |
| --- | --- | --- | --- |
| **Exposed collective (FSDP AG / EP a2a)** | Gap between `*-start`/`*-done` with idle MXU; op profile: all-reduce % high | Per-layer: collective bytes ÷ ICI BW vs matmul time | Latency-hiding scheduler flags, bigger per-chip batch, re-map mesh axis ([roofline](/training/1-batch-size-primer/)) |
| **Wrong mesh→axis mapping** | One collective class suddenly multi-hop: a2a/AG takes ~integer× expected; ICI hops visible in op names (`collective-permute` chains) | Compare achieved vs single-hop bandwidth math | Fix `make_mesh` axis order / `PartitionSpec`; re-read the sharding ([TPU article](/optimization/18-tpu-xla-optimization/)) |
| **Unintended full rematerialization** | Backward contains full forward-shaped op sequence beyond policy; step FLOPs ≫ 6PT estimate | Op profile: fwd-op names duplicated in bwd with ~fwd total time | Fix remat policy (`jax.checkpoint` policy args), offload instead |
| **Data formatting tax** | Op profile: `transpose`/`reshape`/`copy` category ≥ ~5% | Graph viewer: which JAX op introduced it | Layout-friendly einsum order, avoid mid-graph dtype/layout flips, fix sharding that forces resharding |
| **Input starvation** | Overview "input-bound %" > few %; infeed wait at step starts | Input-pipeline tab: host read/decode vs step | Grain prefetch, more host workers, pre-tokenized data ([pipeline section](/google-interview/6-answer-pod-training/)) |
| **Recompilation** | Occasional steps ~seconds; host busy in XLA compile; trace shows no device work | Count distinct input shapes; JAX logs `Compiling…` | Static shapes, pad last batch, `jax.jit` donate/static args discipline |
| **VPU-bound stretch** | Op profile: high vector-ops %; MXU util low during those spans | Which ops (softmax/norm/router) and their layer share | Fuse via pallas kernel if hot; accept if small ([step anatomy](/google-interview/6-answer-pod-training/) prices this) |
| **Padding waste** | Memory viewer: logical vs padded shape gap; op profile shows small-dim matmuls at low FLOPS util | Shapes not multiples of 8/128 lanes | Pad dims to hardware tiles ([TPU-friendly model answer](/google-interview/4-answer-tpu-friendly-model/)) |
| **Straggler core / expert imbalance** | Pod viewer spread; slow cores' extra time is in expert matmuls, not comm | Per-expert token counts from router metrics | Load-balancing loss/bias fix — the [router health board](/google-interview/6-answer-pod-training/) |
| **DCN-exposed cross-slice reduce** | Multislice: gradient all-reduce tail after backward ends | Bytes ÷ DCN BW vs backward duration margin | More overlap (reduce earlier buckets), or check DCN health/degradation |

### Minor faults

- **Host callback stalls** — `io_callback`/debug prints forcing device→host syncs; visible as periodic step-gap growth (the ~100 ms "host gap" row in the step anatomy).
- **Donated-buffer misses** — missing `donate_argnums` doubles peak HBM for the params update; memory viewer shows two full param buffers alive at step boundary.
- **All-gather duplication from sloppy specs** — a `PartitionSpec(None)` activation that propagation replicates, costing a silent all-gather per layer; readable in graph viewer more than trace viewer.
- **Profile-during-warmup** — first steps include compilation and autotuning; always skip ≥2 steps before `trace()`.
- **Barrier-like `jax.block_until_ready` in loop** — benchmarking habit left in production code; serializes dispatch pipelining.
- **SparseCore/embedding overflow** (older TPUs/embedding-heavy models) — embedding lookups falling back to dense gathers; op profile category shift.

## 4. Getting artifacts to practice on

- **The [JAX scaling book's profiling chapter](https://jax-ml.github.io/scaling-book/profiling/)** walks real xprof screenshots end to end — the closest thing to a guided public TPU trace.
- **Free-tier practice:** Colab TPU or Kaggle TPU VMs run `jax.profiler.trace` fine; a 4-layer transformer in pure JAX gives you every row above except pod viewer.
- **MaxText** on a rented v4/v5e slice (even 8 chips) produces the full multi-core artifact set, including pod viewer and real async collectives.

## 5. Exercises

1. On any captured trace: from the op profile alone, write the step-anatomy decomposition (matmul / comm / VPU / data-format / idle) and check it sums to step time — the same exercise the [pod-training answer](/google-interview/6-answer-pod-training/) does with claimed numbers.
2. Take a working sharded matmul in JAX and *deliberately* transpose the mesh axis order; predict which collective appears, then confirm in the trace viewer and measure the cost.
3. Turn off the remat policy on one block; find the duplicated op sequence in backward and reconcile step-FLOPs before/after with the 6PT arithmetic.
4. Add `jax.debug.print` inside the step; measure what one host callback costs at your step time.
