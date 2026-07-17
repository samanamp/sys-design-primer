---
title: "TPU Performance: Architecture, XLA, and Pod-Scale Optimization"
description: "A staff-level guide to TPU performance: systolic arrays and the MXU, VMEM vs HBM, XLA compilation and fusion, GSPMD sharding, Pallas kernels, ICI torus topology, TPU vs GPU tradeoffs, and profiling with xprof."
---

# TPU Performance: Architecture, XLA, and Pod-Scale Optimization

GPU performance intuition transfers to TPUs only partially. TPUs are compiler-scheduled, statically shaped, torus-connected machines built around a systolic matrix unit. Most TPU performance wins and losses come from three places: whether your shapes fill the MXU, whether XLA compiled the program you think it did, and whether your parallelism layout matches the ICI topology.

The staff-level framing:

> A TPU program is a contract between your shapes, the XLA compiler, and a physical torus. Performance debugging is finding which party broke the contract.

---

## 1. The Interview Mental Model

When TPU performance comes up, answer in this order:

1. **Bottleneck:** MXU compute, HBM bandwidth, VMEM capacity, ICI collectives, DCN, or host?
2. **Shapes:** are dimensions multiples of the MXU tile? Is padding silently inflating FLOPs?
3. **Compiler:** what did XLA fuse, lay out, and recompile? Static shapes respected?
4. **Sharding:** what does GSPMD propagate, and does the mesh match the physical torus?
5. **Kernel:** is XLA's schedule good enough, or is this a Pallas case?
6. **Profile:** confirm with xprof, not intuition.

```text
Model code (JAX/PyTorch-XLA)
      |
      v
StableHLO / HLO
      |
      v
XLA: fusion, layout, sharding partitioning (GSPMD)
      |
      v
Per-chip program (MXU/VPU schedule, DMA, collectives)
      |
      v
ICI torus + DCN at pod scale
```

Unlike CUDA, there is no hand-written kernel between you and the hardware by default. The compiler is the kernel author. That is both the superpower and the failure mode.

---

## 2. TPU Chip Architecture

### The MXU: a systolic array

The core of a TensorCore (Google's term for the TPU compute core, not NVIDIA's) is the **MXU**, a systolic array of multiply-accumulators: $128 \times 128$ on v4/v5, $256 \times 256$ on v6e (Trillium) and TPU7x (Ironwood). Inputs are typically bf16 (or int8/fp8 on newer generations), accumulation is fp32.

A systolic array streams operands through a fixed grid of MACs; there is no instruction fetch per multiply, no warp scheduling, no register allocation pressure. That is why TPUs get very high FLOPs per watt and per mm² — and why they are unforgiving about shape:

- A matmul dimension of 129 is computed as 256 on a 128-wide MXU. Nearly half the FLOPs are padding.
- On Trillium/Ironwood, the natural tile is 256, so misalignment costs even more.
- Small matmuls (tiny batch, tiny hidden) cannot fill the pipeline; the array drains before it fills.

The staff-level point:

> On GPUs, bad shapes cost you some tensor-core utilization. On TPUs, bad shapes multiply your FLOP count, because the hardware physically computes the padded tile.

### The VPU

Elementwise work (activations, norms, softmax exponentials, masking) runs on the **VPU**, a 2D SIMD unit shaped $(8, 128)$ — 8 sublanes × 128 lanes, with multiple ALUs per lane. Vector-heavy ops that cannot be fused into a matmul-adjacent kernel become VPU- or bandwidth-bound. The (8, 128) shape is why TPU tiling constraints keep quoting "last dimension 128, second-to-last a multiple of 8."

### Memory hierarchy: HBM and VMEM

```text
HBM (tens–hundreds of GB, ~1–8 TB/s)
      |
      | DMA (asynchronous, compiler/kernel scheduled)
      v
VMEM (~64–128 MiB scratchpad, ~20x HBM bandwidth)
      |
      v
MXU / VPU
```

VMEM is not a cache. It is a software-managed scratchpad: XLA (or your Pallas kernel) explicitly schedules DMA copies between HBM and VMEM and double-buffers them against compute. Because VMEM bandwidth is roughly 20× HBM bandwidth, an operand that fits in VMEM can stay compute-bound at batch sizes where an HBM-streamed operand would be bandwidth-bound.

Consequences:

- The compiler's tiling decisions determine your effective arithmetic intensity.
- A working set slightly over VMEM capacity can fall off a performance cliff.
- Small models or small layers that fit weights in VMEM behave very differently from HBM-resident ones.

### SparseCore

Modern TPUs also carry **SparseCores** (four on v5p and TPU7x, two on v6e): dataflow processors specialized for embedding lookups and scatter/gather-heavy work. For recommendation models, embedding-dominated workloads, and increasingly for MoE-style irregular access, SparseCore offloads work that would waste the MXU and thrash HBM with random access.

---

## 3. The XLA Compilation Model

JAX (and PyTorch/XLA) trace your program into **HLO** (via StableHLO), and XLA compiles it whole-program:

```text
jit(f) traced with abstract shapes
      |
      v
HLO graph
      |
      v
Algebraic simplification, fusion, layout assignment,
rematerialization, memory planning, collective scheduling
      |
      v
One executable per (function, shape, sharding) signature
```

The performance-relevant passes:

- **Fusion.** XLA aggressively fuses elementwise chains into matmul producers/consumers, so norms, activations, and residual adds usually cost near-zero extra HBM traffic. This is fusion by default — the opposite of eager PyTorch, where fusion is the optimization.
- **Layout assignment.** XLA chooses physical layouts (which dimension is minor, tiling into (8,128) native tiles). Layout mismatches insert copies; a stray transpose in your model can materialize as real data movement.
- **Memory planning.** Static allocation of HBM and VMEM. No allocator jitter, but it means the compiler must know all shapes.

### Static shapes and recompilation

XLA specializes each executable to concrete shapes. Every new shape signature triggers a **full recompilation** — seconds to minutes for large models. This is the classic TPU production trap:

- Variable sequence lengths in serving → recompile per length → pad or bucket instead.
- Data-dependent shapes (boolean masking that changes size) → not expressible; use masks over fixed shapes.
- A training job that "hangs" every N steps is often recompiling, not computing.

### When XLA beats hand kernels, and when it doesn't

XLA wins when the program is dense, statically shaped, and fusible: it sees the whole graph, so it can fuse across operator boundaries, schedule DMA globally, and overlap collectives with compute — things per-kernel libraries cannot do.

XLA loses when the best algorithm is not expressible as a fusion of HLO ops:

- IO-aware attention with online softmax (Flash-style scheduling).
- Irregular memory access: MoE dispatch, paged KV caches, custom quantization formats.
- Kernels needing explicit semaphore/DMA choreography or persistent state across grid steps.

That boundary is exactly what Pallas exists for.

---

## 4. GSPMD and Sharding in JAX

TPU-scale parallelism is compiler-partitioned. You write single-device-looking code; **GSPMD** partitions it.

```python
mesh = jax.make_mesh((8, 16), ("data", "model"))
x_sharding = NamedSharding(mesh, P("data", None))
w_sharding = NamedSharding(mesh, P(None, "model"))

@jax.jit
def step(w, x):
    return x @ w   # GSPMD inserts the collectives
```

How propagation works:

1. You annotate shardings on inputs/outputs (and optionally interior values via `with_sharding_constraint`).
2. GSPMD propagates shardings through the HLO graph — forward and backward — choosing shardings for every intermediate.
3. The partitioner rewrites the global program into a per-device program, inserting all-gathers, reduce-scatters, all-reduces, and all-to-alls where shardings disagree.

This is how data, tensor, FSDP-style, and expert parallelism are all expressed on TPU: the same mechanism, different `PartitionSpec`s. The staff-level checks:

- **Inspect what propagation chose.** An unannotated intermediate can be resharded through an accidental all-gather. `jax.debug.visualize_array_sharding` and the compiled HLO are your friends.
- **Collective-matmul overlap.** For a sharded matmul, XLA can decompose it so each ICI transfer step overlaps with a partial matmul (the "collective matmul" / async collective optimization) — tensor-parallel all-gathers largely hidden behind compute. Whether this fires depends on shapes and flags; verify in the profile, don't assume.
- **`shard_map` when you need control.** GSPMD is automatic; `shard_map` gives you explicit per-device code with manual collectives (`psum`, `all_gather`, `ppermute`). Use it when you want a specific communication schedule (e.g., a hand-rolled ring for context parallelism) or to wrap Pallas kernels in a sharded program.

---

## 5. Pallas and Mosaic: Dropping Below XLA

**Pallas** is JAX's kernel language — Triton-shaped, but lowering through **Mosaic** to TPU. A Pallas kernel controls:

- **The grid and BlockSpecs:** how the iteration space is tiled and which HBM block is mapped into VMEM for each grid step.
- **VMEM residency:** operands are `Ref`s in VMEM/SMEM; you decide what lives on-chip.
- **Pipelining:** the pipeline that overlaps DMA with compute (automatic double-buffering across grid steps, or explicit `emit_pipeline`).
- **Explicit DMA and semaphores** for advanced kernels (async copies, cross-chip remote DMA for custom collectives).

Unlike GPU Triton, the TPU grid is (mostly) executed **sequentially** per core rather than as thousands of concurrent blocks — you can carry state across grid iterations (this is how online-softmax attention accumulators work naturally). Constraints follow the hardware: block last dimension a multiple of 128, second-to-last a multiple of 8 (larger for bf16/int8 due to packed tiles).

When to drop to Pallas:

- Flash-style attention variants, sparse/block-sparse attention.
- MoE routing and dispatch, quantized matmuls with custom formats.
- Fusing across a boundary XLA refuses to cross, or overlapping communication with compute in ways XLA's scheduler doesn't find.

When not to: dense matmul-and-elementwise programs. XLA's schedule is already near-roofline there, and a Pallas kernel is a permanent maintenance cost per TPU generation (Mosaic constraints shift with the hardware).

---

## 6. Interconnect: ICI, Torus Topologies, and OCS

TPUs scale with **ICI** (inter-chip interconnect), direct chip-to-chip links forming a torus — no switches inside a slice:

| Generation | Topology | Pod scale | Notes |
| --- | --- | --- | --- |
| v4 / v5p | 3D torus | v5p: 8,960 chips | OCS-reconfigurable, twistable |
| v5e / v6e (Trillium) | 2D torus | 256 chips | cost/inference oriented |
| TPU7x (Ironwood) | 3D torus | 9,216 chips | ~192 GB HBM3e, ~7.4 TB/s, ~4.6 PFLOPS FP8 per chip |

Key properties:

- **Per-hop, per-axis bandwidth.** A 3D torus gives each chip 6 links. Bisection bandwidth scales with the torus cross-section, and wraparound links halve the worst-case hop count.
- **OCS reconfigurability.** v4-and-later pods interconnect cubes through optical circuit switches. Slices of many shapes (e.g., 4×4×8 vs 8×8×16) are wired on demand, failed cubes are routed around, and "twisted torus" configurations improve all-to-all bandwidth. Scheduling flexibility and fault tolerance come from the OCS layer, not from packet switching.
- **ICI vs DCN.** Beyond a slice, you cross into the data-center network. DCN is order-of ~100× lower bandwidth per chip than ICI. Multi-slice training therefore puts only the most latency-tolerant parallelism (data parallelism, with gradient reduction overlapped) across DCN.

### Mapping parallelism onto the torus

The torus makes parallelism placement a physical layout problem:

```text
ICI axis X (fast, wraparound)  -> tensor/sequence parallelism (frequent, latency-sensitive)
ICI axes Y,Z                   -> FSDP / expert parallelism (bandwidth-heavy, overlappable)
DCN across slices              -> data parallelism only
```

Rules of thumb:

- Collectives on one torus axis are ring-shaped and cheap; collectives spanning multiple axes cost more hops.
- Keep the tensor-parallel group within a small torus dimension (often ≤ the axis length) so its all-gathers stay single-axis.
- All-to-all (MoE) wants the twisted/OCS-optimized topologies; it is the collective most sensitive to bisection bandwidth.
- Your `jax.make_mesh` axes are not abstract: mesh construction maps logical axes to physical torus axes, and a bad assignment silently multiplies hop counts.

---

## 7. TPU vs GPU

| Dimension | TPU | GPU (NVIDIA) |
| --- | --- | --- |
| Compute core | Systolic MXU, compiler-scheduled | SMs + tensor cores, warp-scheduled |
| On-chip memory | VMEM scratchpad (software-managed, large) | SMEM + L2 cache (smaller SMEM, HW cache) |
| Kernel model | XLA compiles whole program; kernels optional (Pallas) | Hand/library kernels are the norm |
| Shapes | Static; recompile on change | Dynamic shapes tolerable |
| Scale-up fabric | ICI torus, no switches, OCS between cubes | NVLink/NVSwitch (all-to-all inside domain) |
| Scale-up domain | Thousands of chips per slice | NVL72-class rack domains, then InfiniBand/Ethernet |
| Ecosystem | JAX/XLA-first, narrower | CUDA, broadest kernel/library ecosystem |
| Sweet spot | Dense, static, huge-batch training/serving at pod scale | Dynamic workloads, custom kernels, heterogeneous serving |

### Roofline: the ridge point moved

Ridge point = peak FLOPs ÷ HBM bandwidth, the arithmetic intensity where you stop being memory-bound:

$$
\text{ridge} = \frac{\text{FLOPs/s}}{\text{HBM bytes/s}}
$$

Approximate bf16 ridge points: v5p $\approx 4.6\mathrm{e}14 / 2.8\mathrm{e}12 \approx 165$ FLOPs/byte; Trillium $\approx 9.2\mathrm{e}14 / 1.6\mathrm{e}12 \approx 575$ FLOPs/byte.

For a weight-streaming matmul, arithmetic intensity is roughly the token batch dimension. So Trillium is memory-bound until per-chip batch is in the many-hundreds of tokens. That is enormous **batching pressure**: decode-style inference (small effective batch per weight read) leaves most of the MXU idle unless you batch aggressively, quantize (int8/fp8 doubles the effective ridge problem again), keep weights VMEM-resident, or restructure the workload. GPUs of the same era have lower FLOPs:bandwidth ratios and feel this less sharply. Ironwood pushes bandwidth (7.4 TB/s) precisely to pull the ridge back down for inference.

---

## 8. Worked Example: Debugging a Slow TPU Training Job

Symptom: an LLM pretraining job on a v5p-2048 slice gets 22% MFU; the team expected ~45%.

**Step 1 — capture a profile.** JAX profiler → xprof/TensorBoard ("Profile" plugin). Look at three views: the trace viewer (per-core timeline), the op profile (time by HLO op), and the memory viewer.

**Step 2 — read the top-level split.** The trace shows, per step: 41% matmul fusions, 31% all-gather/reduce-scatter (not overlapped), 15% a single fusion containing `pad`/`transpose`, 8% gaps, 5% other.

**Step 3 — the padding waste.** Op profile shows a matmul with one dimension of 4,104. On a 128-lane MXU that pads to 4,224 (worse on 256). Root cause: vocab or hidden size chosen without alignment. Fix: round dimensions to multiples of 128/256. The `pad`+`transpose` fusion also flags a layout mismatch — an einsum written in an order that forces a physical transpose; rewriting the contraction removes it.

**Step 4 — the exposed collectives.** 31% ICI time not overlapped. Check (a) mesh-to-torus mapping: the "model" axis was laid across two physical axes, doubling hops — reorder `make_mesh` axis assignment; (b) collective-matmul overlap didn't fire for the FSDP all-gathers — enable XLA's async/latency-hiding scheduler flags and confirm in the trace that transfers now sit under matmuls.

**Step 5 — the mystery gaps.** Host timeline shows periodic multi-second stalls: recompilation. The eval loop ran with a different batch size, and a data-pipeline edge case produced a short final batch each epoch. Fix: pad the last batch; keep eval shapes identical or accept one cached executable per shape.

**Step 6 — re-measure.** MFU 22% → 41%. Remaining gap is real communication and VPU-bound layernorm time — now a candidate for a Pallas fused kernel, considered only *after* the compiler-level fixes.

The order matters: shapes and layout first, sharding/overlap second, recompilation third, custom kernels last.

---

## 9. Failure Modes

### Padding tax

Unaligned dimensions silently inflate FLOPs; profiles show high "MXU utilization" against padded FLOPs, hiding the waste.

### Recompilation storms

Dynamic shapes in serving or eval paths trigger repeated multi-minute compiles; throughput graphs show a sawtooth.

### Accidental resharding

A missing sharding constraint lets GSPMD choose an all-gather of the full activation; one line of `with_sharding_constraint` fixes it.

### Mesh/torus mismatch

Logical mesh axes mapped across the wrong physical axes multiply hop counts; collectives are "correct" but 2–3× slower.

### VMEM cliff

A block size or fusion that slightly exceeds VMEM forces smaller tiles or HBM spills; small shape change, large regression.

### DCN in the hot path

A parallelism axis (TP, EP) accidentally spans slices; ICI-speed assumptions meet ~100×-slower DCN.

### Pallas kernel rot

A hand kernel tuned for 128-tile v5p underperforms or fails on 256-tile Trillium; kernels are per-generation liabilities.

---

## 10. Interview Q&A

**Q: Why does a TPU use a systolic array instead of many small cores?**

A: A systolic array amortizes control: operands flow through a fixed grid of MACs with no per-op instruction issue, register files, or scheduling hardware, so nearly all silicon and energy goes to multiply-accumulate. The price is rigidity — it only accelerates dense tiled matmuls at fixed tile sizes, which is why shape alignment and padding dominate TPU performance conversations.

**Q: Your model's step time doubled after a "minor" config change. First three checks on TPU?**

A: (1) Recompilation — did a shape change (batch, sequence, vocab) trigger recompiles or a new, worse executable? (2) Padding/layout — did a dimension fall off a 128/256 multiple, or an einsum order change introduce transposes? (3) Sharding propagation — did GSPMD start resharding an intermediate (look for new all-gathers in the HLO/profile)? All three are compiler-contract breaks, visible in xprof before touching the model.

**Q: When would you write a Pallas kernel instead of trusting XLA?**

A: When the winning algorithm isn't expressible as fused dense HLO: Flash-style attention scheduling, MoE dispatch, paged KV access, custom quantization, or explicit communication/compute overlap XLA won't find. For dense matmul-plus-elementwise programs, XLA is already near roofline and a custom kernel is negative-value maintenance. Decide from a profile showing XLA's schedule leaving specific time on the table.

**Q: How do you map 3D parallelism onto a v5p 3D torus?**

A: Match communication frequency to physical distance: tensor/sequence parallelism on a single fast torus axis (per-layer collectives, latency-sensitive), FSDP or expert parallelism on the remaining axes (bandwidth-heavy but overlappable), and data parallelism across slices over DCN, since gradient all-reduce tolerates latency and can hide behind the backward pass. Then verify the logical mesh axes actually map to those physical axes — the mesh constructor, not the math, decides hop counts.

**Q: Why does Trillium create "batching pressure" for inference?**

A: Its bf16 ridge point is roughly 575 FLOPs/byte (~9.2e14 FLOPs/s over ~1.6e12 B/s), versus ~165 on v5p. Weight-streaming matmul intensity scales with token batch, so you need several hundred tokens per weight read to leave the memory-bound region. Decode with small batches wastes most of the MXU; mitigations are aggressive continuous batching, int8/fp8 weights, VMEM-resident small layers, and speculative decoding to raise tokens per weight pass. Ironwood's 7.4 TB/s HBM is the hardware-side answer.

**Q: GSPMD vs shard_map — when do you use which?**

A: GSPMD for the 90% case: annotate inputs and key intermediates, let propagation and the partitioner insert collectives, and audit the result. shard_map when the communication *schedule* is the product — custom ring pipelines, wrapping Pallas kernels, or when propagation keeps making a resharding choice you can't constrain away. GSPMD is declarative and audited; shard_map is imperative and owned.

---

## 11. Important Papers and Docs

1. **[How to Scale Your Model](https://jax-ml.github.io/scaling-book/)** (Google DeepMind).  
   The best public treatment of TPU rooflines, ICI math, and sharding — read the TPU and "all about rooflines" chapters first.

2. **[TPU v4: An Optically Reconfigurable Supercomputer](https://arxiv.org/abs/2304.01433)** (ISCA 2023).  
   OCS, 3D torus, twisted topologies, and embeddings/SparseCore rationale.

3. **[Cloud TPU system architecture docs](https://cloud.google.com/tpu/docs/system-architecture-tpu-vm)** and the per-generation pages (v5p, v6e, TPU7x).  
   Authoritative specs: MXU sizes, HBM, ICI, pod topologies.

4. **[GSPMD](https://arxiv.org/abs/2105.04663)**.  
   The sharding-propagation partitioner underlying jit + sharding.

5. **[Pallas documentation](https://docs.jax.dev/en/latest/pallas/index.html)**.  
   TPU (Mosaic) kernel authoring: grids, BlockSpecs, pipelining, TPU-specific constraints.

6. **[Distributed arrays and automatic parallelization / shard_map docs](https://docs.jax.dev/en/latest/notebooks/Distributed_arrays_and_automatic_parallelization.html)**.  
   The practical JAX-side sharding workflow.

---

## 12. The Staff Engineer Summary

TPU performance is compiler-mediated performance. The hardware is simple and rigid; the leverage is in feeding it well.

The checklist:

- Align every hot dimension to the MXU tile (128, or 256 on Trillium/Ironwood).
- Keep shapes static; bucket or pad rather than recompile.
- Let XLA fuse; audit layouts and eliminate forced transposes.
- Annotate shardings, then read what GSPMD actually propagated.
- Map mesh axes deliberately onto physical torus axes; keep TP on one axis, DP on DCN.
- Verify collective/compute overlap in the trace, not in theory.
- Know your generation's ridge point; batch or quantize accordingly.
- Reach for Pallas only when a profile shows XLA's schedule is the bottleneck.
- Profile with xprof before and after every change.

The interview answer:

> A TPU is a systolic matmul machine wrapped in a whole-program compiler and a torus. Performance comes from shape alignment into the MXU, VMEM-aware scheduling, static shapes to avoid recompilation, GSPMD shardings that match the physical ICI topology, and dropping to Pallas only where XLA's schedule provably leaves time on the table. FLOPs are cheap; the ridge point keeps rising, so feeding the array — batching, bandwidth, and layout — is the job.
