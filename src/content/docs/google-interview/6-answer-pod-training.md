---
title: "Worked Answer: Pod-Scale MoE Training on TPUs"
description: "A staff-level worked interview answer: end-to-end distributed training for a ~1T-parameter MoE across TPU pods — parallelism-to-torus mapping, data pipeline, goodput and failure handling, MoE-specific monitoring, and escalating follow-ups."
---

# Worked Answer: Pod-Scale MoE Training on TPUs

**The question:** "Design end-to-end distributed training for a roughly 1-trillion-parameter MoE model across TPU pods. Cover the parallelism layout, the data pipeline, failure handling, and what you monitor." (45–60 minutes, escalating follow-ups.)

This is the question where the ML-performance staff signal either shows up or doesn't. The taxonomy of parallelism strategies is table stakes — it's covered in [the parallelism stack](/optimization/13-parallelism-stack/) and [ZeRO/FSDP](/optimization/12-zero-fsdp-sharded-training/), and quoting it back is a senior answer. The staff answer maps the taxonomy onto a physical torus, does the arithmetic out loud, and spends real time on the part most candidates skip: **keeping 10,000 chips productive when they fail every few hours**.

---

## 1. Scope Moves (first 3 minutes)

Ask before designing. Each question changes the design:

- **"What's the token budget and timeline?"** — Compute total determines pod count. Say the interviewer offers: ~15T tokens, want it done in about a month. That's the driving constraint.
- **"Which chip generation do I get?"** — I'll design for **TPU v5p** (459 TFLOPs bf16, 95 GB HBM at 2.76 TB/s, 8,960-chip pods, 3D torus) and note where **Ironwood/TPU7x** (4,614 TFLOPs FP8; the ~2,307 bf16 figure is an assumption — FP8/2 — Google hasn't published a bf16 number; 192 GB at 7.37 TB/s, 9,216-chip superpods) changes answers. v5p is the well-understood workhorse; the layout logic transfers.
- **"One pod or multi-pod?"** — The arithmetic below says two pods. That immediately puts **DCN (multislice)** in scope, which shapes the parallelism split.
- **"MoE architecture fixed?"** — Assume ~1T total / ~100B active per token (DeepSeek/Switch-style fine-grained experts, top-k routing), sequence length 8K for the main run. Long-context extension is a follow-up, not the base design.
- **"Greenfield or existing stack?"** — Assume JAX/XLA on Cloud TPU, MaxText-shaped codebase. That's the realistic Google-adjacent answer and it lets me cite public MFU numbers as the bar.

State the success metric up front: **not peak MFU — trained tokens per wall-clock day**, which is MFU × goodput. That framing sets up the second half of the answer.

---

## 2. The Arithmetic First (minutes 3–12)

Do this on the whiteboard before naming a single parallelism strategy. It's what makes every later choice defensible.

### Compute

Training FLOPs ≈ 6 · P_active · T. The MoE discount is the whole point: you pay for **active** parameters, not total.

$$
6 \times 100\text{B} \times 15\text{T} = 9 \times 10^{24} \text{ FLOPs}
$$

One v5p pod at a **realistic MoE MFU of ~50%** (I'll defend that number below):

$$
8{,}960 \times 459 \times 10^{12} \times 0.5 \approx 2.06 \times 10^{18} \text{ FLOPs/s}
\Rightarrow \frac{9 \times 10^{24}}{2.06 \times 10^{18}} \approx 50 \text{ days}
$$

Too long for the timeline. **Two v5p pods (~18K chips) ≈ 25 days** at ideal goodput — call it ~30 calendar days at 85–90% goodput. That's the design point: two pods, multislice over DCN.

On MFU expectations: MaxText publishes ~65–70% MFU on v5p for **dense** models from 32B up to 1160B params. That's the public bar for dense. MoE gives some of it back to all-to-all dispatch, router overhead, and imperfect expert load balance — quoting 50–55% for a well-tuned 1T MoE is honest; promising 70%+ is a red flag the interviewer will poke.

### Per-step time — and picking the batch to fit the mesh

Don't pick a round batch number and hope it shards; pick the batch *from* the mesh. 8,960 chips per pod factors as $2^8 \times 5 \times 7$ — a power-of-two sequence count (say 2,048) can **never** divide evenly across it. The layout in section 3 uses a per-pod mesh `(fsdp=35, cp=4, ep=64)`, so choose **2,240 sequences per pod = 4,480 global = ~37M tokens** (4,480 × 8,192 = 36.7M). That lands exactly one sequence per (fsdp, expert) coordinate, split 4 ways along the context axis: **per-chip batch = one 2,048-token shard of one sequence**, no fractional sequences anywhere. A ~37M-token global batch is within the range frontier runs actually use late in training (DeepSeek-V3 ramped to ~60M); batch warmup early in the run is a schedule detail (start on one pod or a sub-mesh), not a layout change.

$$
\frac{6 \times 100\text{B} \times 36.7\text{M}}{2.06 \times 10^{18} \times 2 \text{ pods}} \approx 5.3 \text{ s/step}, \quad \sim 410\text{K steps total}
$$

The step count matters later: at ~5.3 s/step, a checkpoint-restore cycle that costs 20 minutes eats ~230 steps — that's the unit failure handling is priced in.

### Memory

Model state at 16 bytes/param (bf16 param + grad, fp32 master + Adam moments — the [standard breakdown](/optimization/12-zero-fsdp-sharded-training/)):

$$
1\text{T} \times 16\text{B} = 16 \text{ TB of model state}
$$

No chip holds that; sharded across one pod's 8,960 chips it's **~1.8 GB/chip** — nearly free against 95 GB HBM. The real memory pressure is **activations**: 8K sequences through a wide MoE. Answer: activation checkpointing (rematerialize per layer block) plus the sharding layout below. The headline: at pod scale, *model state sharding is a solved problem; activations and communication are the live constraints.*

Make the per-chip budget concrete, and *derive* the activation number from the per-chip batch (2,048 tokens; assume d_model ≈ 16K, ~60 layers — stated assumptions, see the confidence ledger):

```text
v5p HBM budget per chip (95 GB), 2,048 tokens/chip, per-layer remat
─────────────────────────────────────────────────────────────────────
Sharded model state    ▓ 1.8 GB    (16 TB / 8,960 — the "free" part)
FSDP gather buffers    ▓ ~1 GB     (gathered weights for ~2 layers in
                                    flight: 2 × ~0.5 GB — see §3 for
                                    why they can't all stay resident)
Checkpointed inputs    ▓▓ ~4 GB    (2,048 tok × 16K × 2 B × 60 layers
                                    ≈ 3.9 GB at every-layer remat)
Live-block remat peak  ▓▓▓▓ ~9 GB  (one block's full activations:
                                    expert FFN intermediates + ring-
                                    attention KV/workspace on the CP
                                    shard)
All-to-all buffers     ▓▓ ~4 GB    (EP dispatch/combine staging,
                                    double-buffered, both directions)
XLA workspace/frags    ▓▓▓ ~8 GB   (fusion temporaries, collective
                                    staging, static allocation slack)
─────────────────────────────────── derived floor ≈ 28 GB
Remat dial (spent)     ▓▓▓▓▓▓▓ ~35 GB  (relax to every-4th-layer
                                    checkpointing: ~4× checkpointed
                                    inputs + resident attention
                                    residuals → recompute FLOPs
                                    bought back → MFU)
Headroom (kept)        ░░░░░ ~32 GB (fragmentation slack + growth)
                                    ─── total: 95 GB
```

The bar is the argument: model state is 2% of HBM; the *derived* activation floor is ~15%, and the interesting part is the ~35 GB deliberately spent relaxing remat. That's a *tuning dial* — relax checkpointing until activations fill the budget, converting spare memory into fewer recomputed FLOPs.

This is the first staff signal: the memory math says you do **not** need aggressive tensor parallelism to fit. That frees the layout to be chosen for communication cost, not memory desperation.

---

## 3. Parallelism → Topology Mapping (minutes 12–22)

The [parallelism taxonomy](/optimization/13-parallelism-stack/) tells you the strategies; the v5p 3D torus and the ICI/DCN boundary tell you where each one is allowed to live. The rule from [the TPU article](/optimization/18-tpu-xla-optimization/): match communication frequency and volume to physical link speed.

```text
Physical hierarchy                 What runs on it
--------------------------------   ------------------------------------------
ICI, within a torus axis           expert parallel all-to-all (most
(fastest, ring collectives)        bandwidth/latency sensitive)

ICI, across torus axes             FSDP all-gather / reduce-scatter
(still fast, more hops)            (bandwidth-heavy, overlappable)

DCN, between pods (multislice,     data parallelism ONLY — gradient
~100x slower per chip than ICI)    all-reduce hidden behind backward
```

Before naming axes, do the check most candidates skip: **does the batch actually fit the mesh?** Per pod, ~18.4M tokens per step over 8,960 chips is 2,048 tokens/chip — a *quarter of one 8K sequence*. You cannot shard activations purely on a batch axis with fewer sequences than chips: attention needs the full sequence's keys and values, so once per-chip tokens drop below one sequence, **context parallelism stops being a long-context luxury and becomes structurally required at this batch/sequence config**. The honest choice is CP over more TP, because ring/all-gather attention costs one bandwidth-friendly collective per attention layer, while TP would put latency-sensitive collectives inside *every* matmul.

Concrete layout per pod — logical mesh `(dcn_data=2, fsdp=35, cp=4, expert=64)`, and verify the factorization out loud: 35 × 4 × 64 = 8,960 per pod, × 2 pods = 17,920. Mapped onto torus axes:

- **Expert parallelism (EP=64) on the fastest ICI extent — folded, not flat.** Every MoE layer is all-to-all dispatch → expert compute → all-to-all combine, the collective most sensitive to bisection bandwidth, so EP must stay inside ICI. But be honest about the folding: no physical axis is 64 long. On a 16×20×28 torus, EP=64 = X(16) × 4-of-Y — a *two-axis fold*, so the all-to-all is multi-hop with worse bisection than a hypothetical flat 64-axis. That makes "the a2a hides" a **condition, not an assertion**: per layer, dispatch+combine bytes/chip ÷ effective folded-group a2a bandwidth must be ≤ that layer's expert matmul time. Check it in the trace per layer; when expert balance degrades, this is the inequality that flips first.
- **Context parallelism (CP=4) on a 4-extent of Z.** Ring attention: each chip holds a 2,048-token shard, KV blocks rotate around the CP ring overlapped with attention compute. This is the axis the batch math forced; own it rather than hiding it.
- **FSDP-style parameter/optimizer sharding on the remaining 35-extent** (5-of-Y × 7-of-Z). ZeRO-3 semantics via GSPMD: all-gather weights around each layer's compute, reduce-scatter grads — prefetched and overlapped. Do the volume math honestly: each chip's forward touches the shared/attention weights plus its 1/64 slice of experts — **~1T/64 ≈ 15.6B params ≈ 31 GB in bf16 of *used* weights per chip**. Those 31 GB cannot stay resident between forward and backward (the memory bar above has ~30 GB of total slack, not 31 GB for weights alone), so the schedule is **per-layer gather → use → discard, in both passes**: ~31 GB of all-gather in forward, ~31 GB again in backward, plus ~31 GB of grad reduce-scatter ≈ **~90 GB/chip/step**. That sounds enormous until you divide by the step: 90 GB / 5.3 s ≈ **17 GB/s sustained — a small fraction of ICI's hundreds of GB/s**, which is why FSDP still hides. The design question this raises — *which* layers, if any, keep gathered weights resident fwd→bwd — is answered by the memory bar: almost none; you pay the double gather and it's still cheap.
- **No tensor-parallel matmul sharding.** The memory math didn't demand it, per-layer matmuls at a ~37M-token global batch fill the MXU without splitting, and TP adds per-layer latency-sensitive collectives that compete with EP and CP for the fast axes. Add TP ≤ 4 only if per-layer weight size or activation memory forces it. Precision matters here: the claim is *no TP* — not "no sequence-dimension sharding at all"; CP is doing that job, and saying so is the stronger signal.
- **No pipeline parallelism.** PP earns its bubbles when interconnect between stages is weak (GPU clusters crossing node boundaries). Inside a TPU slice, ICI + FSDP covers memory without bubble/imbalance tax. Mention DualPipe-style designs exist for GPU MoE ([DeepSeek-V3](https://arxiv.org/abs/2412.19437)) to show you know the other ecosystem's answer.
- **Data parallelism across the DCN boundary.** The two pods are pure DP replicas; the only cross-pod traffic is gradient reduction, overlapped with the backward pass. This is exactly the multislice pattern Google has demonstrated publicly at 50K+ chips on v5e, and Trillium's launch materials report ~96–99% scaling efficiency at 12 pods depending on configuration (Google blog) for DP-over-DCN workloads — evidence the boundary placement is right.
- **Long context deepens CP; it doesn't introduce it.** For a 128K–1M context phase, grow the CP axis (deeper ring) and shrink FSDP to pay for it — trade it explicitly rather than hand-waving "add CP."

Draw the mapping — the pod is a physical object, and each mesh axis has a bytes-per-step bill and a compute window it must hide under:

```text
        One v5p pod: 8,960 chips as a 3D torus (16 x 20 x 28, OCS-wired
        from 4x4x4 cubes; logical mesh (fsdp=35, cp=4, expert=64)
        folded on: EP = X(16) x 4-of-Y, CP = 4-of-Z,
        FSDP = 5-of-Y x 7-of-Z)

                    Z (28) ── CP(4) + FSDP part(7) ───────────┐
                   ╱                                          │
                  ╱   ┌────────────────────────────┐          │
                 ╱   ╱                            ╱│          │
                ╱   ╱     EP = 64 group          ╱ │   FSDP: per-layer
               ╱   ╱   (X=16 folded with        ╱  │   gather-use-discard,
              ╱   ╱    4-of-Y: multi-hop a2a,  ╱   │   BOTH passes:
             ╱   ╱    worse bisection than    ╱    │   ~31 GB AG fwd
            ╱   ╱     a flat axis)           ╱     │   + ~31 GB AG bwd
           ╱   ╱   ◄══ all-to-all ══►       ╱      │   + ~31 GB grad RS
          ╱   ╱   dispatch+combine         ╱       │   ≈ ~90 GB/chip/step
         ╱   ╱   ~27 GB/chip/step;        ╱        │   → ~17 GB/s over
        ╱   ╱   hides IF per-layer a2a   ╱         │   5.3 s — small vs
       ╱   ╱   time ≤ expert matmul     ╱          │   ICI's 100s of GB/s
      ╱   └────────────────────────────┘           │
     └───── X (16) ─────── Y (20) ─────────────────┘
                                │
                                │  DCN boundary (multislice,
                                │  ~100x slower per chip than ICI)
                                ▼
        ┌──────────────────────────────────────────────┐
        │  Pod B: pure DP replica (dcn_data = 2)       │
        │  grad all-reduce: ~0.22 GB/chip/step (bf16   │
        │  1T grads / 8,960) — must hide under the     │
        │  ~3.2 s backward pass → needs only           │
        │  ~70 MB/s/chip of DCN. Comfortable.          │
        └──────────────────────────────────────────────┘
```

The same argument as a table — dimension by dimension, with the alternative each choice beat:

| Dimension | Physical axis | Bytes moved / step / chip | Hidden under | Why not the alternative |
| --- | --- | --- | --- | --- |
| Expert (EP=64) | X(16) folded with 4-of-Y (multi-hop a2a) | ~27 GB (dispatch + combine, ~60 MoE layers × ~0.45 GB) | Expert matmuls, per layer — *iff* per-layer a2a ≤ matmul time on the folded group | On DCN or outer axes, all-to-all bisection cost sets step time |
| Context (CP=4) | 4-of-Z | Ring-attention KV rotation per attention layer | Attention compute on each shard | Without CP, per-chip batch (2,048 tok) < one sequence — batch-only sharding is impossible; TP instead would tax every matmul |
| FSDP (=35) | 5-of-Y × 7-of-Z | ~90 GB (31 AG fwd + 31 AG bwd + 31 grad RS; gather-use-discard both passes) | Adjacent layers' compute (prefetch); ~17 GB/s vs ICI's hundreds | Full replication wastes 16 TB nobody has; keeping gathered weights resident needs 31 GB nobody has either |
| Data (DP=2) | DCN, pod↔pod | ~0.22 GB (grad all-reduce) | ~3.2 s backward | EP or FSDP over DCN = 100× slower link in the per-layer path |
| Tensor (TP=1) | — (not used) | 0 | — | Memory math doesn't demand it; per-layer collectives would fight EP/CP for fast axes |
| Pipeline (PP=1) | — (not used) | 0 | — | Bubbles buy nothing when ICI already spans the slice |

**The JAX story** (say this concretely): single-controller Pathways runtime; the model is written with explicit shardings — `jax.jit` with `NamedSharding`/`PartitionSpec`, `shard_map` where the communication schedule matters, GSPMD/Shardy partitioning the rest. Note in passing that `pmap` has been deprecated and removed in recent JAX — `shard_map` and explicit sharding are the canonical path. Then the verification discipline from [the TPU article](/optimization/18-tpu-xla-optimization/): read what propagation actually chose, confirm in the xprof trace that FSDP all-gathers and the EP all-to-all sit **under** matmuls (collective-matmul/latency-hiding scheduler), and that the mesh's logical axes landed on the physical torus axes you intended — a wrong `make_mesh` ordering silently multiplies hop counts.

### Anatomy of one 5.25 s step

Decompose the step the way xprof will show it — and make the segments **sum to the step time**, because the gap between "MXU math" and "wall clock" is where MFU actually goes. Pure model-math time per step is $6 \times 100\text{B} \times 36.7\text{M} / (17{,}920 \times 459 \times 10^{12}) \approx 2.68$ s — and 2.68 / 5.25 is exactly the ~51% MFU claimed earlier. Everything else must be named:

```text
t=0 ms                  1,750                       4,900   5,150  5,250
├── forward ~1,750 ──────┼──── backward ~3,150 ──────┼─ opt ──┼─100─┤
│                        │                           │ +router upd │
│ MXU model math ~890    │ MXU model math ~1,790     │ ~250 ms     │
│ EP a2a / FSDP AG       │ EP a2a + FSDP AG + RS     │             │
│ (mostly overlapped)    │ (mostly overlapped)       │             │
│                        │ DCN grad all-reduce ══════╡ (fully      │
│                        │  ~0.22 GB under 3,150 ms  │  hidden)    │
└────────────────────────┴───────────────────────────┴─────────────┘

Where the other ~2,570 ms lives (every segment labeled):
  MXU model math (counts toward MFU)          ~2,680 ms
  Remat re-forward (recompute, NOT in MFU)      ~890 ms  (≈ one
                                                          forward)
  VPU-bound / non-matmul ops (router softmax,   ~950 ms
   norms, residuals, attention at low
   arithmetic intensity on 2K CP shards)
  Exposed collectives (a2a tails on imbalanced  ~380 ms
   expert layers, last RS with no compute
   left, CP ring bubbles)
  Optimizer + router bias update                ~250 ms
  Host callback + step gap                      ~100 ms
                                        ─────────────────
                                        sum:  ~5,250 ms ✓

MFU = 2,680 / 5,250 ≈ 51% — the chip looks "busy" for ~95% of the
step, but MFU only credits model math. Remat is real work that MFU
deliberately doesn't count; that gap is the teaching beat, not a bug.
```

Where monitoring attaches: **step time** is the outer bracket; the **per-collective exposed-vs-overlapped split** is the ~380 ms tail (it grows when expert balance degrades — the a2a tail is your router-health signal showing up in systems metrics); **host input-buffer occupancy** guards the 100 ms gap (if the gap grows, the pipeline, not the model, regressed); **DCN all-reduce completion margin** tells you how much backward-time budget remains before adding DP replicas would expose it.

---

## 4. Data Pipeline (minutes 22–27)

Underrated section; skipping it is a named failure mode below. At ~37M tokens per 5.25 s step across both pods, the input side must sustain **~7M tokens/s aggregate — ~3.5M tokens/s per pod, forever**, without ever making an accelerator wait.

- **Tokenize offline.** Pretokenized, pre-packed (sequence-packing to eliminate pad waste) fixed-length examples in sharded files on GCS. Never tokenize on the training hosts' hot path.
- **Host-side loading with Grain** (or tf.data): each TPU host reads only its shard subset, prefetches several batches ahead, overlaps H2D transfer with step compute. The metric is host-side prefetch buffer occupancy — if it ever drains, you're burning ~18K chips on filesystem latency.
- **Determinism as a feature, not a nicety.** Deterministic shuffle (fixed seed, checkpointable iterator state) means a restart resumes the exact token stream — no silently repeated or skipped data — and, critically, enables **deterministic replay for SDC hunting** (below). Grain's checkpointable iterators are exactly this.
- **Static shapes.** Fixed batch and sequence shapes end to end; a stray short final batch is a [recompilation storm](/optimization/18-tpu-xla-optimization/) at pod scale.
- **Mixture control:** data mixture weights as versioned config, mid-run mixture changes logged as first-class events — they look like loss anomalies later if untracked.

---

## 5. Reliability: The Goodput Deep Dive (minutes 27–40)

This is where staff shows. At ~18K chips, hardware interruptions are a **continuous process, not an event** — with per-chip MTBF measured in years, the fleet still throws multiple failures per day. Frame it with Google's own decomposition:

$$
\text{goodput} = \text{scheduling goodput} \times \text{runtime goodput} \times \text{program goodput}
$$

(Can we get the chips? Are they doing forward/backward vs. restarting? Is the program using them well — i.e., MFU?) Naive design loses double digits from the middle term. The mitigation ladder, cheapest first:

1. **Multi-tier asynchronous checkpointing — with the actual arithmetic.** The checkpoint is 16 TB (1T params × 16 B model state), but *sharded* it's 1.8 GB/chip, and that changes everything:
   - **Tier 0, HBM peer replica:** pod B's live DP copy *is* the checkpoint — zero write cost. Be honest about the restore path, though: the peer replica is in the *other pod*, so a cross-pod restore moves 1.8 GB/chip (16 TB total) over **DCN, not ICI**. At an effective DCN rate of O(0.1–1 GB/s/chip) that's **tens of seconds to a few minutes**, not "seconds." The seconds-scale fast path is *in-pod*: surviving peers within the same slice (FSDP shards, host RAM on healthy hosts) re-broadcast over ICI/PCIe. Both are still 10–100× better than object storage.
   - **Tier 1, host RAM:** each host DMAs its shard over PCIe (~1.8 GB × 4 chips/host at tens of GB/s → **~1–2 s stall**, then the training step resumes while the host drains).
   - **Tier 2, CNS/GCS:** async persist from host RAM. 16 TB across ~2,240 hosts is only ~7 GB/host — the write is easy; the *restore* is not: a cold restore fans 16 TB back out to 18K chips through object storage and re-establishes the job, realistically **10–20 minutes** (~115–230 lost steps at 5.25 s).
   - **Interval optimization (do the expected-value math out loud).** Assume per-chip MTBF ~5 years. Fleet failure rate: $17{,}920 / (5 \times 8{,}760\,\text{h}) \approx 0.41/\text{h}$ → **fleet MTBF ≈ 2.4 h ≈ 8,800 s**. Young's approximation for optimal interval with checkpoint cost $C$: $\tau^* \approx \sqrt{2 C \cdot \text{MTBF}}$. With the async tier-1 cost $C \approx 2$ s: $\tau^* \approx \sqrt{2 \times 2 \times 8{,}800} \approx 190$ s — **checkpoint every ~3 minutes**, total overhead ≈ $C/\tau^* + \tau^*/(2\,\text{MTBF}) \approx 1\% + 1\% = 2\%$. Now price the naive design: synchronous 30-min saves to CNS cost, per failure, ~15 min average lost work + ~15–20 min restore ≈ **1,800–2,100 s lost per 8,800 s MTBF ≈ 20–24% of the run** — before counting the save stalls. Same hardware, 10× difference, purely from checkpoint architecture. Treating checkpointing as "call `save()`" is a listed way to fail this question.
2. **Redundant in-memory model state instead of restore-from-disk.** The Gemini precedent ([arXiv 2312.11805](https://arxiv.org/abs/2312.11805)): keep redundant replicas of model state in memory across the data-parallel dimension; on failure, recover from a healthy replica's live copy rather than a checkpoint. Our two-pod DP layout gives this for free — pod B's state *is* pod A's hot backup. Gemini attributes goodput going from ~85% (their prior largest run) to ~97% partly to this.
3. **Topology reconfiguration around failures.** v5p pods interconnect 4×4×4 cubes through **optical circuit switches** ([TPU v4 paper, arXiv 2304.01433](https://arxiv.org/abs/2304.01433): OCS is <5% of system cost, <3% of power). When a cube fails, the OCS layer re-wires a spare cube into the slice — the ~10-second figure is not in the v4 paper; it comes from the Gemini report's description of reconfiguring around failures and should be quoted as approximate/reported — so the job resumes on an intact torus instead of draining to a smaller degraded one. Hot-spare cubes are a budget line item you argue for explicitly.
4. **Elastic training + node hot-swap** (per Google's elastic-training/goodput Cloud posts): the Pathways single-controller model lets the run **suspend-resume** and continue at reduced scale (e.g., drop one DP replica) while hardware is swapped, rather than sitting fully idle. Rescale events are logged as batch-size changes for later loss forensics.
5. **Silent data corruption (SDC).** The failure mode that doesn't page you. A marginal chip computes wrong numbers without crashing; you find out as an unexplained loss excursion 50K steps later — or never. The concrete detection recipe:
   - **Continuous, nearly free:** per-DP-replica grad-norm divergence. Both pods compute gradients over *different* data, so norms won't match exactly — but track the ratio's distribution; a replica whose grad-norm distribution drifts is the canary. Per-slice grad-norm splits inside a pod localize further.
   - **Periodic, cheap:** a **canary batch** — one fixed, versioned batch run through fwd+bwd every N thousand steps on every slice. Because shapes, program, and data are deterministic, the resulting loss/grad checksum must be bit-identical across slices and across time (for fixed weights). Any mismatch names the guilty hardware immediately.
   - **On suspicion:** **deterministic replay** — re-run the suspect step range (determinism gives you the exact batches) on known-good chips and diff gradients bit-for-bit. Replay convicts or acquits in minutes.
   - **The cost of NOT catching it:** an SDC event that corrupts optimizer state doesn't roll back with a 3-minute checkpoint — by the time the loss visibly excurses, every checkpoint tier may already contain the poison. Worst case you rewind 50K+ steps (at 5.25 s/step, ~73 hours ≈ **three days of the entire fleet's output, ~150 chip-years**) or, if it stays sub-visible, ship a subtly worse model with no line item explaining why. That asymmetry — detection costs ~0.1% of step time, non-detection costs days-to-unbounded — is the whole argument. Unprompted SDC discussion is one of the strongest staff signals available in this question.

### The goodput waterfall

Put numbers on the ladder — and derive both bars from the *same* stated failure rate (~0.41/h, fleet MTBF ≈ 8,800 s) and recovery times, rather than reverse-engineering toward a familiar anchor:

```text
Naive (sync 30-min CNS checkpoints, restore-from-disk, no spares)
100% ┤████████████████████████████████████████
     │  −2%  scheduling (pod acquisition, maintenance windows)
 98% ┤███████████████████████████████████████
     │ −23%  runtime failures: per fleet-MTBF window of 8,800 s,
     │        ~900 s avg lost work + ~1,100 s CNS restore
     │        = 2,000 / 8,800 ≈ 23%  (same math as §5.1's 20–24%)
     │  −2%  sync save stalls (~1–2 min blocking write / 30 min)
 ~75% ┤══ net ══  → tokens/day × ~0.75

Engineered (async multi-tier ckpt, replica restore, OCS re-wire)
100% ┤████████████████████████████████████████
     │  −1%  scheduling (hot-spare cubes absorb maintenance)
 99% ┤███████████████████████████████████████
     │  −2%  runtime: ~1% ckpt overhead + ~1% failures
     │        (avg loss ≈ 95 s work + ~10 s OCS re-wire
     │         + replica restore, per 8,800 s window)
 97% ┤═══ net ═══  → tokens/day × 0.97
```

Note the naive bar lands near **~75%, not 85%**. Gemini's oft-quoted 85% was the *observed* goodput of their prior largest run — an already partially engineered starting point — not a derivation target; don't bend your arithmetic to hit someone else's anchor. The honest gap here is **~22 points**, and 22% of an 18K-chip month is 0.22 × 17,920 × 30 ≈ **~118,000 chip-days (roughly 320 chip-years)** — the machinery in this section is worth more than any model-side optimization you could name.

### Failure modes, priced

| Failure | Detection signal | Blast radius | Recovery path | Goodput cost/event |
| --- | --- | --- | --- | --- |
| Single chip/host crash | Heartbeat loss, barrier timeout | Whole slice stalls (SPMD is synchronous) | Restart on spare host, restore from HBM replica / host RAM | ~1–3 min |
| 4×4×4 cube failure | ICI link errors, slice health | Slice torus broken | OCS re-wires spare cube (~10 s) + tier-0/1 restore | ~1–2 min |
| DCN partition | Cross-pod all-reduce timeout | DP sync lost; pods fine individually | Elastic: continue single-pod (half throughput), rejoin later | Minutes at 50% rate |
| Input pipeline stall | Prefetch buffer occupancy → 0 | Both pods idle at full power | Fix/reshard input; no state lost | 100% of stall duration |
| SDC (marginal chip) | Canary checksum mismatch, replica grad-norm drift, replay diff | Corrupted state propagates through *all* checkpoint tiers | Quarantine chip, rewind to pre-corruption ckpt, replay data window | Minutes if caught; days-to-unbounded if not |
| Recompilation storm | Sawtooth step time, host compile logs | Whole job, repeatedly | Fix shape leak (pad final batch, freeze eval shapes) | ~Minutes per compile × frequency |

---

## 6. MoE-Specific Training Failure Modes (minutes 40–45)

The MoE tax isn't just all-to-all bandwidth — it's a router that can destabilize the run:

- **Router collapse / load imbalance.** The router converges to favoring few experts; most of your 1T parameters go dead and EP groups sit idle behind hot ones (all-to-all is synchronous — the slowest expert group sets step time). Mitigate with auxiliary load-balancing losses, or **loss-free balancing** via per-expert bias adjustment (DeepSeek-V3's approach — worth naming because aux-loss weight is itself a quality/balance tradeoff you no longer have to tune).
- **Capacity-factor overflow.** With fixed expert capacity, overflowing tokens are dropped — a *silent quality* leak that shows up as a plateau, not a crash. Monitor dropped-token fraction per layer as a first-class metric; alert on drift.
- **Expert death.** Individual experts stop receiving tokens and their weights go stale. Track per-expert token counts over a trailing window; dead experts are a router-health regression even when aggregate loss looks fine.
- **Router numerics.** Keep router logits/softmax in fp32 even in a bf16 run; router z-loss to bound logit growth. Many "mystery loss spikes at scale" are router numerics.

The router health board, as concrete thresholds:

| Metric | Healthy range | Failure it catches |
| --- | --- | --- |
| Router entropy (per layer) | High and stable; slow drift down as experts specialize | Sudden drop → router collapse onto few experts |
| Per-expert token share (trailing window) | Within ~2–3× of uniform (1/64 per EP group) | Hot experts (step-time tail) and dead experts (stale weights) |
| Dropped-token fraction | < ~1% per layer, flat | Capacity-factor overflow — silent quality leak |
| Max router logit / z-loss magnitude | Bounded, non-growing | Router numerics divergence → the "mystery" loss spike |
| Aux-loss (or bias-update) magnitude | Small, stable fraction of total loss | Balance machinery fighting the task loss |
| EP all-to-all exposed time (from xprof) | ~flat ms tail per step | Load imbalance showing up as a *systems* regression |
| Expert-layer step-time variance across EP groups | < few % | The slowest expert group silently setting global step time |

The last two rows are the point of the table: router pathologies surface in the *systems* dashboard before the loss moves — the a2a tail in the step-time gantt above is a model-health signal wearing a systems costume.

---

## 7. Monitoring (minute 45+, or woven throughout)

Three dashboards, in causal order — systems health, model health, fleet health:

- **Systems:** step time (with per-collective breakdown from xprof: FSDP all-gather, EP all-to-all, DCN all-reduce — each with an "exposed vs. overlapped" split), MFU, host input-buffer occupancy, per-slice stragglers.
- **Model:** loss + smoothed derivative with spike alerts, grad norm (global and per-slice — the per-slice split is your first SDC signal), router entropy, per-expert utilization histograms, dropped-token fraction, aux-loss magnitude.
- **Fleet:** the **goodput dashboard** — the three-factor decomposition over time, checkpoint save/restore durations, interruptions and recovery time per event, SDC scan results. This is the dashboard the compute-budget conversation runs on.

---

## 8. Escalating Follow-Ups

**"You need to scale 4× overnight. What breaks?"**
Adding six more pods as DP replicas is mechanically easy under multislice — that's the point of putting DP on DCN. What breaks is upstream of topology: global batch grows 4× (~147M tokens), which is well past the useful batch-size regime — so I'd reject naive batch growth and instead shrink per-pod batch (fewer sequences per (fsdp, expert) coordinate, or deeper CP), trading MFU for optimizer sanity, with LR/warmup retuned either way. DCN gradient reduction volume per pod is unchanged (all-reduce cost per replica doesn't grow with replica count in the bandwidth term), but scheduling goodput drops — acquiring and holding 8 aligned pods is a real availability negotiation. And failure frequency scales linearly with chips, so the reliability machinery of section 5 goes from "important" to "the job."

**"Loss spike at step 800K. Walk me through triage."**
Ordered by cheapness: (1) Check event log — checkpoint restore, rescale, or data-mixture change near the spike? Restarts that skip/repeat data or reset optimizer state are the most common cause. (2) Grad-norm per slice at spike onset — one slice diverging says hardware, not math → quarantine and **deterministically replay** steps ~799K–800K on different chips; gradient diff convicts or clears SDC. (3) Inspect the offending data window (determinism gives me the exact batches) — a bad shard or corrupt file. (4) Router health at the spike — entropy collapse or overflow surge points at routing instability → z-loss/bias-update tuning. (5) Only if all of these clear do I treat it as an optimization event: rewind to pre-spike checkpoint, skip the offending data window, and resume — standard large-run practice, cheap because checkpoints are minutes apart in-memory.

**"What's your per-chip batch?"**
The question that collapses hand-wavy layouts, so answer it exactly: ~18.4M tokens per pod / 8,960 chips = **2,048 tokens per chip — one quarter of one 8K sequence**. That number is *why* the mesh has a CP axis: with fewer sequences than chips, you cannot shard activations on a batch axis alone — attention needs full sequences resident, so each sequence spans a CP=4 ring and each chip owns a 2,048-token shard. A candidate who claims a batch-sharded-only layout at these numbers hasn't divided; the interviewer will.

**"Why not GPUs for this?"**
Not a religious answer — a topology one. The TPU case: EP all-to-all lives happily inside an 8,960-chip switchless ICI domain, while a GPU cluster crosses from NVLink (rack-scale domain) to InfiniBand much earlier, which is why GPU MoE designs spend enormous engineering on comm scheduling (DeepSeek's DualPipe + custom kernels) to hide that boundary. Add OCS fault-reconfiguration and the whole-program XLA/GSPMD stack, and the pod-scale training story is more integrated. Honest other side: CUDA's kernel ecosystem is deeper, dynamic shapes are cheaper, and DeepSeek-V3 proves world-class MoE training on GPUs is achievable — with more bespoke systems work. Given the question says "TPU pods," the differentiated answer is knowing *what you'd have to rebuild* on the other stack, not asserting superiority.

**"Where does Ironwood change your answer?"**
~10× FP8 FLOPs (≈5× bf16, if bf16 is half FP8 — assumed, unpublished) and 2× HBM per chip: fewer chips for the same run (maybe one superpod), doubled HBM relaxes activation pressure, FP8 training becomes a first-class option (with loss-scale/numerics work), and 7.37 TB/s keeps the ridge point sane. The layout *logic* — EP inside ICI, DP over DCN, FSDP and CP in between — is unchanged; the constants move, and fewer faster chips can raise per-chip tokens back above a full sequence, shrinking or removing the CP axis.

---

## Staff+ Signals

- **Arithmetic before architecture:** 6·P_active·T, pod-count and step-time derivation, and the 16 TB → 1.8 GB/chip observation that *frees* the layout from memory panic.
- **Topology-aware layout reasoning:** each parallelism dimension justified by which physical link it lives on — including the negative choices (no tensor-parallel matmul sharding, no PP) *and* the forced positive one: per-chip batch of 2,048 tokens is a quarter-sequence, so CP is structurally required, and saying "CP because the batch math demands it, and it's cheaper than TP" beats bragging about an axis count.
- **Goodput and SDC raised unprompted**, with the Gemini-run mechanisms (in-memory replicas, OCS re-wire, deterministic replay) and the goodput = scheduling × runtime × program decomposition.
- **MoE training scar tissue:** router collapse, capacity overflow as silent quality loss, expert death, router-in-fp32 — failure modes you only list if you've watched an MoE run wobble.
- **Honest MFU expectations:** citing MaxText's ~65–70% dense v5p numbers as the public bar and explaining why a real MoE lands ~50–55%.
- **Determinism as an engineering tool** (replay, exact resume, data forensics), not a checkbox.

## What Falls Short

- Reciting the parallelism taxonomy without mapping dimensions to torus axes and the ICI/DCN boundary — the senior ceiling on this question.
- Designing as if hardware doesn't fail: no goodput discussion at 10K+-chip scale is disqualifying for this role.
- No data-pipeline section at all — input starvation is a real way to idle 18K chips.
- Treating checkpointing as solved ("we checkpoint every 30 minutes") without the async/multi-tier/in-memory-replica ladder or the cost math.
- Promising >75% MFU on a 1T MoE, or quoting dense-model MFU numbers as directly achievable.
- Ignoring MoE-specific instabilities — a router is a nonstationary load balancer inside your optimizer, and pretending it's a dense model forfeits the hardest part.
- Claiming a batch-sharded-only activation layout without dividing tokens by chips — the per-chip-batch question exposes it in one line.

---

## Confidence Ledger

Keep the epistemics honest — an interviewer probing any number should hear which bucket it's in:

- **Verified (public sources):** v5p specs (459 TFLOPs bf16, 95 GB HBM, 8,960-chip pods, 3D torus, OCS-wired 4×4×4 cubes); MaxText ~65–70% dense-model MFU on v5p; Gemini report's ~85% → ~97% goodput narrative and in-memory replica recovery; OCS <5% cost / <3% power (TPU v4 paper); DeepSeek-V3's DualPipe and loss-free balancing; Trillium ~96–99% multi-pod scaling (Google blog, config-dependent).
- **Derived (arithmetic from stated assumptions):** 9×10²⁴ FLOPs, two-pod/~25-day sizing; mesh factorization (2 × 35 × 4 × 64 = 17,920) and the 2,048 tokens/chip per-chip batch; ~5.25 s step and 51% MFU decomposition; ~90 GB/chip/step FSDP volume and the ~17 GB/s rate; Young's-approximation checkpoint interval; the goodput waterfall (both bars) and the ~118,000 chip-day gap; DCN grad-reduce hiding margin.
- **Priors / assumed (would verify before betting the run):** MoE MFU ~50–55%; d_model ≈ 16K and ~60 MoE layers for the activation and a2a estimates; per-chip MTBF ~5 years; effective DCN bandwidth O(0.1–1 GB/s/chip); OCS re-wire ~10 s (reported, Gemini description); Ironwood bf16 ≈ FP8/2 (unpublished); host-RAM checkpoint stall ~1–2 s.
