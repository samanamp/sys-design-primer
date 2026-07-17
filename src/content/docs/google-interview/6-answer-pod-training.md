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
- **"Which chip generation do I get?"** — I'll design for **TPU v5p** (459 TFLOPs bf16, 95 GB HBM at 2.76 TB/s, 8,960-chip pods, 3D torus) and note where **Ironwood/TPU7x** (2,307 TFLOPs bf16, 4,614 FP8, 192 GB at 7.37 TB/s, 9,216-chip superpods) changes answers. v5p is the well-understood workhorse; the layout logic transfers.
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

### Per-step time

Global batch ~16M tokens (2,048 sequences × 8K):

$$
\frac{6 \times 100\text{B} \times 16\text{M}}{2.06 \times 10^{18} \times 2 \text{ pods}} \approx 2.3 \text{ s/step}, \quad \sim 940\text{K steps total}
$$

The step count matters later: at ~2.3 s/step, a checkpoint-restore cycle that costs 20 minutes eats ~500 steps — that's the unit failure handling is priced in.

### Memory

Model state at 16 bytes/param (bf16 param + grad, fp32 master + Adam moments — the [standard breakdown](/optimization/12-zero-fsdp-sharded-training/)):

$$
1\text{T} \times 16\text{B} = 16 \text{ TB of model state}
$$

No chip holds that; sharded across one pod's 8,960 chips it's **~1.8 GB/chip** — nearly free against 95 GB HBM. The real memory pressure is **activations**: 8K sequences through a wide MoE. Answer: activation checkpointing (rematerialize per layer block) plus the sharding layout below. The headline: at pod scale, *model state sharding is a solved problem; activations and communication are the live constraints.*

Make the per-chip budget concrete — it shows *why* activations, not parameters, are the live constraint:

```text
v5p HBM budget per chip (95 GB), 8K seq, per-layer-block remat
─────────────────────────────────────────────────────────────────────
Sharded model state   ▓ 1.8 GB     (16 TB / 8,960 — the "free" part)
Activations + remat   ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓ ~38 GB
  (checkpointed layer inputs ~2 GB + live-block rematerialization
   peak + 8K-seq attention workspace — the actual constraint)
All-to-all buffers    ▓▓ ~5 GB     (EP dispatch/combine staging,
                                    double-buffered)
XLA workspace/frags   ▓▓▓ ~8 GB    (fusion temporaries, collective
                                    staging, static allocation slack)
Headroom              ░░░░░░░░ ~42 GB  (spent deliberately: less
                                    aggressive remat → recompute
                                    FLOPs bought back → MFU)
```

The bar is the argument: model state is 2% of HBM, activations are ~40%. The remaining headroom is a *tuning dial* — relax checkpointing until activations fill it, converting spare memory into fewer recomputed FLOPs.

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

Concrete layout per pod (one logical mesh, e.g. `(dcn_data=2, fsdp=~140, expert=64)` mapped deliberately onto physical torus axes):

- **Expert parallelism (EP) on the innermost ICI axes.** Every MoE layer is all-to-all dispatch → expert compute → all-to-all combine. All-to-all is the collective most sensitive to bisection bandwidth, so the EP group must stay inside ICI, sized so each chip holds a few experts (fine-grained experts make this flexible). This placement decision is the single most important line in the design.
- **FSDP-style parameter/optimizer sharding across the remaining ICI extent.** ZeRO-3 semantics via GSPMD sharding: all-gather weights around each layer's compute, reduce-scatter grads — prefetched and overlapped. With only ~1.8 GB/chip of state, the FSDP axis can be huge and shallow.
- **Little or no tensor parallelism.** The memory math didn't demand it, per-layer matmuls at 16M-token batches fill the MXU without splitting, and TP adds per-layer latency-sensitive collectives that compete with EP for the fast axes. Add TP ≤ 4 only if activation memory or per-layer weight size forces it. (Saying "no TP, and here's why" is a stronger signal than reciting 4D parallelism.)
- **No pipeline parallelism.** PP earns its bubbles when interconnect between stages is weak (GPU clusters crossing node boundaries). Inside a TPU slice, ICI + FSDP covers memory without bubble/imbalance tax. Mention DualPipe-style designs exist for GPU MoE ([DeepSeek-V3](https://arxiv.org/abs/2412.19437)) to show you know the other ecosystem's answer.
- **Data parallelism across the DCN boundary.** The two pods are pure DP replicas; the only cross-pod traffic is gradient reduction, overlapped with the backward pass. This is exactly the multislice pattern Google has demonstrated publicly at 50K+ chips on v5e, and Trillium's GA materials claim ~99% scaling efficiency at 12 pods for DP-over-DCN workloads — evidence the boundary placement is right.
- **Context parallelism: only if long-context is in scope.** For a 128K–1M context phase, add a CP axis (ring attention over an ICI axis) and shrink EP or FSDP to pay for it — trade it explicitly rather than hand-waving "add CP."

Draw the mapping — the pod is a physical object, and each mesh axis has a bytes-per-step bill and a compute window it must hide under:

```text
        One v5p pod: 8,960 chips as a 3D torus (16 x 20 x 28, OCS-wired
        from 4x4x4 cubes; logical mesh (fsdp=140, expert=64) folded on)

                    Z (28)  ── FSDP axis (part) ──────────────┐
                   ╱                                          │
                  ╱   ┌────────────────────────────┐          │
                 ╱   ╱                            ╱│          │
                ╱   ╱     EP = 64 group          ╱ │   all-gather bf16
               ╱   ╱   (innermost X + part Y)   ╱  │   weights fwd+bwd,
              ╱   ╱   ◄══ all-to-all ══►       ╱   │   reduce-scatter
             ╱   ╱   dispatch+combine         ╱    │   grads:
            ╱   ╱   ~12 GB/chip/step         ╱     │   ~6 GB/chip/step,
           ╱   ╱   hides under expert       ╱      │   prefetched under
          ╱   ╱   matmuls (~100 ms)        ╱       │   next layer's
         ╱   └────────────────────────────┘        │   compute
        └───── X (16) ─────── Y (20) ──────────────┘
                                │
                                │  DCN boundary (multislice,
                                │  ~100x slower per chip than ICI)
                                ▼
        ┌──────────────────────────────────────────────┐
        │  Pod B: pure DP replica (dcn_data = 2)       │
        │  grad all-reduce: ~0.22 GB/chip/step (bf16   │
        │  1T grads / 8,960) — must hide under the     │
        │  ~1.4 s backward pass → needs only           │
        │  ~160 MB/s/chip of DCN. Comfortable.         │
        └──────────────────────────────────────────────┘
```

The same argument as a table — dimension by dimension, with the alternative each choice beat:

| Dimension | Physical axis | Bytes moved / step / chip | Hidden under | Why not the alternative |
| --- | --- | --- | --- | --- |
| Expert (EP=64) | Innermost ICI axes (X + part of Y) | ~12 GB (dispatch + combine, ~60 MoE layers × ~0.2 GB) | Expert matmuls, per layer | On DCN or outer axes, all-to-all bisection cost sets step time |
| FSDP (~140) | Remaining ICI extent (Y/Z) | ~6 GB (bf16 weight all-gather fwd+bwd + grad reduce-scatter) | Adjacent layers' compute (prefetch) | Full replication wastes 16 TB nobody has; ZeRO-3 is ~free at 1.8 GB/chip |
| Data (DP=2) | DCN, pod↔pod | ~0.22 GB (grad all-reduce) | ~1.4 s backward | EP or FSDP over DCN = 100× slower link in the per-layer path |
| Tensor (TP=1) | — (not used) | 0 | — | Memory math doesn't demand it; per-layer collectives would fight EP for fast axes |
| Pipeline (PP=1) | — (not used) | 0 | — | Bubbles buy nothing when ICI already spans the slice |
| Context (CP) | Only for long-context phase | ring attention traffic | attention compute | Costs an axis EP/FSDP currently use — trade explicitly |

**The JAX story** (say this concretely): single-controller Pathways runtime; the model is written with explicit shardings — `jax.jit` with `NamedSharding`/`PartitionSpec`, `shard_map` where the communication schedule matters, GSPMD/Shardy partitioning the rest. Note in passing that `pmap` is gone (removed around JAX v0.10) — explicit sharding is the canonical path. Then the verification discipline from [the TPU article](/optimization/18-tpu-xla-optimization/): read what propagation actually chose, confirm in the xprof trace that FSDP all-gathers and the EP all-to-all sit **under** matmuls (collective-matmul/latency-hiding scheduler), and that the mesh's logical axes landed on the physical torus axes you intended — a wrong `make_mesh` ordering silently multiplies hop counts.

### Anatomy of one 2.3 s step

Decompose the step the way xprof will show it. Pure math time per step is $6 \times 100\text{B} \times 16\text{M} / (17{,}920 \times 459 \times 10^{12}) \approx 1.17$ s — everything above that line is overhead, and 1.17 / 2.3 is exactly the ~51% MFU claimed earlier:

```text
t=0 ms                750                        2,150   2,250  2,300
├── forward ~750 ─────┼──── backward ~1,400 ─────┼─ opt ─┼─ 50 ─┤
│                     │                          │ +router upd  │
│ MXU math ~390 ms    │ MXU math ~780 ms         │ ~100 ms      │
│ EP a2a (overlapped) │ EP a2a + FSDP RS         │              │
│ FSDP AG (prefetch)  │ (mostly overlapped)      │              │
│ remat replay        │ + recompute fwd blocks   │              │
│                     │ DCN grad all-reduce ═════╡ (fully       │
│                     │  ~0.22 GB under 1,400 ms │  hidden)     │
└─────────────────────┴──────────────────────────┴──────────────┘
Exposed (non-overlapped) collectives: ~100 ms — a2a tails on
imbalanced expert layers + the last reduce-scatter with no
compute left to hide under. Final 50 ms: host callback + step gap.

Sum of MXU math: ~1,170 ms  →  MFU = 1,170 / 2,300 ≈ 51%
```

Where monitoring attaches: **step time** is the outer bracket; the **per-collective exposed-vs-overlapped split** is the ~100 ms tail (it grows when expert balance degrades — the a2a tail is your router-health signal showing up in systems metrics); **host input-buffer occupancy** guards the 50 ms gap (if the gap grows, the pipeline, not the model, regressed); **DCN all-reduce completion margin** tells you how much backward-time budget remains before adding DP replicas would expose it.

---

## 4. Data Pipeline (minutes 22–27)

Underrated section; skipping it is a named failure mode below. At 2 pods × 16M tokens per 2.3 s step, the input side must sustain ~7M tokens/s **per pod, forever**, without ever making an accelerator wait.

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
   - **Tier 0, HBM peer replica:** pod B's live DP copy *is* the checkpoint — zero write cost, restore = re-broadcast 1.8 GB/chip over ICI, **seconds**.
   - **Tier 1, host RAM:** each host DMAs its shard over PCIe (~1.8 GB × 4 chips/host at tens of GB/s → **~1–2 s stall**, then the training step resumes while the host drains).
   - **Tier 2, CNS/GCS:** async persist from host RAM. 16 TB across ~2,240 hosts is only ~7 GB/host — the write is easy; the *restore* is not: a cold restore fans 16 TB back out to 18K chips through object storage and re-establishes the job, realistically **10–20 minutes** (~300–500 lost steps at 2.3 s).
   - **Interval optimization (do the expected-value math out loud).** Assume per-chip MTBF ~5 years. Fleet failure rate: $17{,}920 / (5 \times 8{,}760\,\text{h}) \approx 0.41/\text{h}$ → **fleet MTBF ≈ 2.4 h ≈ 8,800 s**. Young's approximation for optimal interval with checkpoint cost $C$: $\tau^* \approx \sqrt{2 C \cdot \text{MTBF}}$. With the async tier-1 cost $C \approx 2$ s: $\tau^* \approx \sqrt{2 \times 2 \times 8{,}800} \approx 190$ s — **checkpoint every ~3 minutes**, total overhead ≈ $C/\tau^* + \tau^*/(2\,\text{MTBF}) \approx 1\% + 1\% = 2\%$. Now price the naive design: synchronous 30-min saves to CNS cost, per failure, ~15 min average lost work + ~15–20 min restore ≈ **1,800–2,100 s lost per 8,800 s MTBF ≈ 20–24% of the run** — before counting the save stalls. Same hardware, 10× difference, purely from checkpoint architecture. Treating checkpointing as "call `save()`" is a listed way to fail this question.
2. **Redundant in-memory model state instead of restore-from-disk.** The Gemini precedent ([arXiv 2312.11805](https://arxiv.org/abs/2312.11805)): keep redundant replicas of model state in memory across the data-parallel dimension; on failure, recover from a healthy replica's live copy rather than a checkpoint. Our two-pod DP layout gives this for free — pod B's state *is* pod A's hot backup. Gemini attributes goodput going from ~85% (their prior largest run) to ~97% partly to this.
3. **Topology reconfiguration around failures.** v5p pods interconnect 4×4×4 cubes through **optical circuit switches** ([TPU v4 paper, arXiv 2304.01433](https://arxiv.org/abs/2304.01433): OCS is <5% of system cost, <3% of power). When a cube fails, the OCS layer re-wires a spare cube into the slice — reported on the order of ~10 seconds — so the job resumes on an intact torus instead of draining to a smaller degraded one. Hot-spare cubes are a budget line item you argue for explicitly.
4. **Elastic training + node hot-swap** (per Google's elastic-training/goodput Cloud posts): the Pathways single-controller model lets the run **suspend-resume** and continue at reduced scale (e.g., drop one DP replica) while hardware is swapped, rather than sitting fully idle. Rescale events are logged as batch-size changes for later loss forensics.
5. **Silent data corruption (SDC).** The failure mode that doesn't page you. A marginal chip computes wrong numbers without crashing; you find out as an unexplained loss excursion 50K steps later — or never. The concrete detection recipe:
   - **Continuous, nearly free:** per-DP-replica grad-norm divergence. Both pods compute gradients over *different* data, so norms won't match exactly — but track the ratio's distribution; a replica whose grad-norm distribution drifts is the canary. Per-slice grad-norm splits inside a pod localize further.
   - **Periodic, cheap:** a **canary batch** — one fixed, versioned batch run through fwd+bwd every N thousand steps on every slice. Because shapes, program, and data are deterministic, the resulting loss/grad checksum must be bit-identical across slices and across time (for fixed weights). Any mismatch names the guilty hardware immediately.
   - **On suspicion:** **deterministic replay** — re-run the suspect step range (determinism gives you the exact batches) on known-good chips and diff gradients bit-for-bit. Replay convicts or acquits in minutes.
   - **The cost of NOT catching it:** an SDC event that corrupts optimizer state doesn't roll back with a 3-minute checkpoint — by the time the loss visibly excurses, every checkpoint tier may already contain the poison. Worst case you rewind 50K+ steps (at 2.3 s/step, ~32 chip-hours × 18K chips ≈ **days of the fleet's output**) or, if it stays sub-visible, ship a subtly worse model with no line item explaining why. That asymmetry — detection costs ~0.1% of step time, non-detection costs days-to-unbounded — is the whole argument. Unprompted SDC discussion is one of the strongest staff signals available in this question.

### The goodput waterfall

Put numbers on the ladder — this is the difference the reliability machinery buys, and it's the ~85% vs ~97% story the Gemini report tells:

```text
Naive (sync 30-min CNS checkpoints, restore-from-disk, no spares)
100% ┤████████████████████████████████████████
     │  −2%  scheduling (pod acquisition, maintenance windows)
 98% ┤███████████████████████████████████████
     │ −13%  runtime: ~0.41 fail/h × (~900 s avg lost work
     │        + ~1,100 s CNS restore) + sync save stalls
 85% ┤██████████████████████████████████
     │  −?   program losses already counted in MFU
 85% ┤══ net runtime goodput ══  → tokens/day × 0.85

Gemini-style (async multi-tier ckpt, HBM-replica restore, OCS re-wire)
100% ┤████████████████████████████████████████
     │  −1%  scheduling (hot-spare cubes absorb maintenance)
 99% ┤███████████████████████████████████████
     │  −2%  runtime: ~1% ckpt overhead + ~1% failures
     │        (avg loss ≈ 95 s work + ~10 s OCS re-wire
     │         + seconds-scale replica restore, per 8,800 s)
 97% ┤═══ net runtime goodput ═══  → tokens/day × 0.97
```

12 points of goodput on an 18K-chip month is ~2,200 chip-days — the machinery in this section is worth more than most model-side optimizations you could name.

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
Adding six more pods as DP replicas is mechanically easy under multislice — that's the point of putting DP on DCN. What breaks is upstream of topology: global batch grows 4× (~64M tokens), which exceeds the useful batch-size regime unless I retune LR/warmup (or reject batch growth and shrink per-pod batch, trading MFU). DCN gradient reduction volume per pod is unchanged (all-reduce cost per replica doesn't grow with replica count in the bandwidth term), but scheduling goodput drops — acquiring and holding 8 aligned pods is a real availability negotiation. And failure frequency scales linearly with chips, so the reliability machinery of section 5 goes from "important" to "the job."

**"Loss spike at step 800K. Walk me through triage."**
Ordered by cheapness: (1) Check event log — checkpoint restore, rescale, or data-mixture change near the spike? Restarts that skip/repeat data or reset optimizer state are the most common cause. (2) Grad-norm per slice at spike onset — one slice diverging says hardware, not math → quarantine and **deterministically replay** steps ~799K–800K on different chips; gradient diff convicts or clears SDC. (3) Inspect the offending data window (determinism gives me the exact batches) — a bad shard or corrupt file. (4) Router health at the spike — entropy collapse or overflow surge points at routing instability → z-loss/bias-update tuning. (5) Only if all of these clear do I treat it as an optimization event: rewind to pre-spike checkpoint, skip the offending data window, and resume — standard large-run practice, cheap because checkpoints are minutes apart in-memory.

**"Why not GPUs for this?"**
Not a religious answer — a topology one. The TPU case: EP all-to-all lives happily inside an 8,960-chip switchless ICI domain, while a GPU cluster crosses from NVLink (rack-scale domain) to InfiniBand much earlier, which is why GPU MoE designs spend enormous engineering on comm scheduling (DeepSeek's DualPipe + custom kernels) to hide that boundary. Add OCS fault-reconfiguration and the whole-program XLA/GSPMD stack, and the pod-scale training story is more integrated. Honest other side: CUDA's kernel ecosystem is deeper, dynamic shapes are cheaper, and DeepSeek-V3 proves world-class MoE training on GPUs is achievable — with more bespoke systems work. Given the question says "TPU pods," the differentiated answer is knowing *what you'd have to rebuild* on the other stack, not asserting superiority.

**"Where does Ironwood change your answer?"**
~5× bf16 FLOPs and 2× HBM per chip: fewer chips for the same run (maybe one superpod), doubled HBM relaxes activation pressure, FP8 training becomes a first-class option (2× again, with loss-scale/numerics work), and 7.37 TB/s keeps the ridge point sane. The layout *logic* — EP inside ICI, DP over DCN, FSDP in between — is unchanged; the constants move.

---

## Staff+ Signals

- **Arithmetic before architecture:** 6·P_active·T, pod-count and step-time derivation, and the 16 TB → 1.8 GB/chip observation that *frees* the layout from memory panic.
- **Topology-aware layout reasoning:** each parallelism dimension justified by which physical link it lives on — including the negative choices (no TP, no PP) with reasons.
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
