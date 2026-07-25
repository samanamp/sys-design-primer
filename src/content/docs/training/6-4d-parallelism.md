---
title: 4D parallelism
description: 4D parallelism
---


Training a frontier model means splitting it across thousands of GPUs along **four orthogonal axes**: Data (DP), Tensor (TP), Pipeline (PP), Context (CP). Each axis trades a different bottleneck for a different communication pattern. The art is choosing axis sizes and mesh ordering so every comm fits in the bandwidth tier where it can hide behind compute.

(MoE adds a fifth axis — Expert Parallelism — covered briefly at the end.)

## The four axes

```
                     shards what?              comm pattern
DP   batch          ─ different samples       ─ all-reduce(grads) end-of-step
TP   within-layer   ─ rows/cols of weights    ─ all-reduce(act) per layer × 2
PP   across-layers  ─ contiguous layer stages ─ p2p send/recv at stage boundary
CP   sequence       ─ tokens along s          ─ ring exchange K/V per attn
```

### DP (and FSDP/ZeRO)

Pure DP replicates the model, shards the batch. Comm: one all-reduce of gradients per step, size = full param count.

**FSDP/ZeRO-3** shards params, grads, optimizer states across DP ranks. Reconstructs full layer params via all-gather just before forward, drops them after. Backward all-gathers again, then reduce-scatters grads. Total comm ≈ 1.5× pure DP, memory ÷N. The standard for non-MoE training.

### TP (Megatron-style)

Shard each linear layer's weight matrix, partition compute across `N_TP` ranks within a node:

```
y = x · W                       y = GeLU(x · W₁) · W₂
W ─split cols──► [W^(0) | W^(1) | ... | W^(N-1)]      column-parallel
                 each rank computes y^(i) = x · W^(i)
                 (no comm; output already sharded along feature dim)

W₂ ─split rows──► W₂ = [[W₂^(0)], [W₂^(1)], ...]      row-parallel
                  each rank computes partial sum
                  ALL-REDUCE to combine                ◄ comm here
```

Megatron pattern: column-parallel → row-parallel pair = **one all-reduce per attn block, one per FFN block**. Comm volume per layer per token: `2 · b · s · d` per rank. Massive — must be on NVLink.

### PP

Slice the model by layer into `P` stages, each stage on a different device. Microbatch the input to overlap stages.

```
Naïve (GPipe):
stage 0:  F₀F₁F₂F₃ . . . . . B₃B₂B₁B₀
stage 1:  . F₀F₁F₂F₃ . . . . B₃B₂B₁B₀ .
stage 2:  . . F₀F₁F₂F₃ . . B₃B₂B₁B₀ . .
stage 3:  . . . F₀F₁F₂F₃ B₃B₂B₁B₀ . . .
              └── bubble ──┘

1F1B (Megatron-LM, PipeDream):
stage 0:  F₀F₁F₂F₃B₀F₄B₁F₅B₂...                       ← steady state
stage 3:  . . . F₀B₀F₁B₁F₂B₂F₃B₃...                    overlapped
```

**Bubble fraction**: naïve = `(P−1)/(M+P−1)` for `M` microbatches. Interleaved 1F1B with `V` virtual stages: `(P−1)/(V·M+P−1)`. Llama 3 used `P=16, V=4, M=128` → ~3% bubble. Comm per stage boundary: just activations, `b · s · d` — small, fits on InfiniBand.

### CP

Already covered in detail. Shards `s`, ring-rotates K/V during attention. Comm `2·s·d` per layer per device, independent of `N_CP`.

## Mesh ordering — the actual design problem

Devices form a 4D mesh. The ordering determines which axis lives on which network tier:

```
World = 8192 GPUs
        ┌────────────────────────────────────────┐
        │  DP=64  (cross-rack, IB, 200 Gbps)     │
        │   └─ PP=16  (cross-node, IB)           │
        │        └─ CP=8  (intra/cross-node)     │
        │             └─ TP=8  (intra-node, NVLink 900 GB/s) │
        └────────────────────────────────────────┘

Rule: comm-heaviest axis innermost (highest BW link).
```

Why TP innermost: per-layer all-reduces of full activations. Cross-node TP is a non-starter — IB bandwidth is ~50× lower than NVLink and TP comm is on the critical path of every layer.

Why DP outermost: gradient all-reduce happens once per step, can be overlapped with backward pass entirely (FSDP reduce-scatter inline with backward), tolerates IB latency.

PP and CP go in the middle. PP comm is small (just activations at boundaries) but latency-sensitive due to the bubble; CP comm is medium volume but well-overlapped with the attention compute itself.

## Memory & comm summary (per device, per step)

| Axis | Memory factor | Comm volume                | Comm pattern         |
|---|---|---|---|
| DP=N | params: 1 (FSDP: 1/N) | `~1.5 · params`            | all-gather + reduce-scatter |
| TP=N | params: 1/N, act: 1/N | `2 · b·s·d · L`            | 2× all-reduce per layer |
| PP=N | params: 1/N, act: ~1  | `b·s·d · M`                | p2p, stage boundaries |
| CP=N | act (s-dim): 1/N       | `2·s·d · L`                | ring p2p, per attn |

`L`=layers, `M`=microbatches. Activations dominate memory at long context → CP is the only axis that touches that.

## Production layouts

| Model              | DP   | TP | PP | CP | EP  |
|---|---|---|---|---|---|
| Llama 3 405B (16K) | 128  | 8  | 16 | 1  | —   |
| Llama 3 405B (128K)| 64   | 8  | 16 | 16 | —   |
| DeepSeek-V3        | 240  | 1  | 16 | 1  | 64  |
| GPT-4 era (rumored)| ~64  | 8  | 16 | —  | ~16 |

DeepSeek-V3's `TP=1` is striking: they relied on FSDP for memory and EP for MoE, avoiding TP's NVLink pressure entirely. Workable because their model fits per-layer in HBM after FSDP sharding and EP handles the experts.

## EP (the fifth axis, MoE only)

Each MoE layer has `E` experts; shard them across `N_EP` ranks. Routing requires **all-to-all** dispatching tokens to expert-holders, then **all-to-all** returning outputs. Comm volume: `2 · b·s·d · top_k`. Highly imbalanced if expert load is skewed — auxiliary load-balancing loss is mandatory.

EP composes with the others as a 5D mesh `(DP, PP, EP, CP, TP)`. EP usually goes near TP (intra-node) because all-to-all is bandwidth-heavy.

## Trade-offs

You don't pick one axis; you pick a *budget* across all of them. The constraints:

- Activation memory at long `s` ⇒ need CP
- Param memory at large `d` ⇒ need TP or FSDP
- Layer count vs per-device HBM ⇒ need PP
- Throughput from many samples ⇒ need DP

Frontier training is constrained optimization: minimize `step_time` subject to `mem_per_device < HBM`, `bubble_fraction < ε`, `comm_overlap_fraction > δ`. The mesh shape is chosen by simulator (Megatron has one, so does Anthropic internally) before the run starts; getting it wrong costs millions in wasted compute.

# 4D Parallelism — Interview Q&A

Staff-level answers. Numbers from production configs (Llama 3, DeepSeek-V3, Megatron-LM defaults). Assume H100 80GB, NVLink 900 GB/s intra-node, IB 400 Gbps inter-node unless stated.

---

**Q1. Walk me through your mesh choice for training a 400B-param dense model at 128K context on 8192 H100s.**

`(DP=64, PP=16, CP=16, TP=8)`. TP=8 is forced by NVLink topology — TP all-reduces activations every layer, can't cross IB. PP=16 to fit params: 400B × 2 bytes weights + 2× grads + 8× optimizer states ≈ 4.8 TB total state, divided by `TP × PP × DP_FSDP_shard` = 8×16×64 = ~600 MB/rank for weights, leaves ~75 GB HBM for activations and KV. CP=16 is forced by activation memory at 128K — even with TP, per-rank activation is `b·s·d/TP ≈ 1·128K·16K·2/8 = 512 MB per layer`, ×80 layers ÷ activation-checkpointing ratio still OOMs without sequence sharding. DP=64 falls out: `8192/(8·16·16) = 4`, so I'd actually back off to CP=8 or PP=8 to get a meaningful DP dimension, since DP<8 makes FSDP sharding ineffective. Final answer depends on profiling; this is the starting point for the simulator.

---

**Q2. Training is at 42% MFU. How do you debug?**

First step: NSight Systems trace on rank 0 of each pipeline stage to classify the gap. Three usual suspects, ranked by frequency:

1. **PP bubble**: if stage 0 and stage `P-1` show idle gaps at start/end, increase `M` (microbatches) or switch to interleaved 1F1B with `V≥2`. Bubble fraction `(P-1)/(V·M+P-1)` — for `P=16`, going `V=1→4` at `M=64` drops bubble from 19% to 5.5%.
2. **TP all-reduce not overlapping**: check if NCCL kernels run serially after matmul. Fix is sequence parallelism (Megatron-SP) which converts the AR into reduce-scatter + all-gather and overlaps each half with adjacent compute.
3. **Straggler rank**: one slow GPU stalls the whole step. Look at NCCL timeline for ranks waiting on barriers. Common causes: thermal throttling, ECC retry, noisy neighbor. Fix is hardware-side.

Also check if you're DRAM-bound vs compute-bound on the matmul itself — at low arithmetic intensity (small batch, long seq) FlashAttention can be the culprit, not parallelism.

---

**Q3. Why can't you do TP across nodes over InfiniBand?**

TP all-reduces activations of size `b·s·d` on the critical path of every layer, twice (attn + FFN). For `b=1, s=8K, d=16K, bf16, L=80`: `2·8192·16384·2·80 = 42 GB` of comm volume per forward per rank. NVLink at 900 GB/s does this in ~50 ms; IB at 50 GB/s does it in ~840 ms — and that's the *unhideable* portion since it's between matmuls feeding each other directly. You can't overlap an AR with the matmul that produced its input. So cross-node TP turns a compute-bound forward pass into a comm-bound one with a 15× slowdown. The only escape is sequence parallelism + careful overlap, which buys back maybe 2× — still nowhere near worthwhile.

---

**Q4. Walk through the backward-pass communication for a `(DP=64-FSDP, PP=16, TP=8)` mesh.**

For a single layer's backward:

1. **TP backward all-reduce** on input gradient: row-parallel layer's input gradient needs AR across TP group (it was column-parallel sharded in forward, gradient flows back as scatter, must AR for next layer's input). Latency on critical path.
2. **Activation recompute** if checkpointing — extra forward of the layer.
3. **Weight gradient compute** — local matmul, no comm.
4. **FSDP reduce-scatter** of weight gradients across DP ranks once the layer's grad is ready. Overlaps with the next layer's backward compute.
5. **PP send** of input gradient to previous stage (only at stage boundary). Small, fits in IB latency.
6. At step end: optimizer step uses sharded grads + sharded optimizer states locally. No additional comm.

Critical path is TP-AR + matmul, repeated per layer. FSDP RS is fully hideable if `RS_time < layer_compute_time`, which holds for layers wider than ~8K dim.

---

**Q5. Why did DeepSeek-V3 train with TP=1?**

V3 is MoE: 671B total params, 37B activated. Three things break the usual TP argument:

1. **Activated params per token are small** — only 37B flows through any given matmul. With FSDP across 240 DP ranks, that's 154 MB per rank of weights, fits trivially in HBM. No need for TP to shrink params.
2. **EP=64 already burns the intra-node bandwidth** with all-to-all dispatch/combine — adding TP all-reduces on top would contend for NVLink and stall both.
3. **TP=1 simplifies the kernel surface** — no sequence-parallelism gymnastics, no fused TP+EP communication patterns, easier to debug at scale.

The trade is that activation memory per rank is full `b·s·d`, which they manage via aggressive activation checkpointing and modest per-rank batch sizes. The lesson: TP isn't a default; it's a tool for memory pressure. If FSDP+EP already gives you the memory math, skip it.

---

**Q6. How does CP compose with FSDP?**

CP ranks must hold *identical* weights — they're just dividing the sequence, not the model. So CP and FSDP are orthogonal axes. In practice you flatten the DP dimension into `(FSDP, CP)`: a 2D subgroup where FSDP all-gathers params along one axis and CP rotates K/V along the other. Gradient sync at step end reduce-scatters across the full DP×CP product (since CP ranks see different tokens, their gradients differ and must be averaged just like DP). The footgun: CP rank 0 and FSDP rank 0 of different CP groups must be on the same NCCL communicator hierarchy, otherwise the all-gather and ring-rotate kernels serialize. Get the process group construction order right and it composes cleanly; get it wrong and you lose 30% throughput to NCCL contention.

---

**Q7. Activation memory OOMs at 256K context. Fixes in priority order?**

1. **Activation checkpointing** (selective, not full) — recompute only attention forward, keep MLPs cached. Cuts activation memory ~3×, costs ~25% more compute. Always do this first.
2. **Increase CP**. Doubling CP halves activation memory along `s`. Linear scaling, comm is well-overlapped.
3. **Sequence parallelism within TP** (Megatron-SP). Shards LayerNorm and dropout activations along `s` across TP ranks — these were previously replicated. Free memory savings, slight comm restructuring.
4. **CPU offload** (ZeRO-Infinity style) for optimizer states. Don't do this for activations — PCIe BW kills throughput.
5. **Reduce microbatch size**. Last resort; hurts MFU because matmuls become smaller and less compute-bound.

In that order because (1) is free-ish, (2)-(3) are scaling tools without throughput cost, (4)-(5) are throughput-negative.

---

**Q8. How would you compose MLA with CP?**

The latent `c_KV` is what gets cached, so CP ring-rotates `c_KV` chunks (size `d_c ≈ 4·d_h`) instead of full K/V (size `2·h·d_h`). Comm volume per ring step drops by ~`2h·d_h / d_c ≈ 60×` for V3 dims. That's a major win — MLA's bandwidth advantage propagates into the distributed setting.

The kernel work: each rank receives a `c_KV` tile, must locally apply the absorbed `W_UQK` matmul to compute scores, then absorb `W_UV` into the value path. The decoupled RoPE branch (`k_R`) rotates as a separate small ring — adds a second p2p stream but it's tiny.

Catch: the absorbed-matmul tile sizes are awkward. `d_c` is 512-ish, not a friendly tensorcore shape, and you're doing it on small per-rank chunks. Need careful kernel fusion to avoid leaving tensorcores idle. Nobody has shipped this combo publicly yet — it's a real open kernel-engineering problem and a great thing to bring up unprompted in a design round.

