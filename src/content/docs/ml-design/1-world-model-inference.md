---
title: Serving a Genie-scale World Model for Large-Scale Simulation
description: ML system design — make a frontier driving world model cheap enough to power closed-loop simulation at fleet scale
---

# Design the inference/runtime system for a Genie-scale driving world model

> **Interviewer prompt:** "We've post-trained a frontier world model for driving — Genie-3 class. It generates camera + lidar at 10 Hz, conditioned on ego actions, scene layout, and language. Quality is great. Problem: it's far too expensive to run at the scale our simulation program needs. Design the system that makes it cheap and fast enough to power large-scale closed-loop simulation."

*Interview-style answer. First-person, as the candidate. I talk through decisions out loud, flag tradeoffs explicitly, and mark **Staff-level signals** wherever they appear. Concrete numbers over hand-waving — all numbers are stated assumptions I'd sanity-check with the interviewer.*

---

## 1. Clarify scope and assumptions

Questions I'd ask, then assume answers for and move on:

1. **Who is the consumer?** Closed-loop simulation with the planner-under-test in the loop, or open-loop dataset generation? (I'll assume both, but closed-loop is the hard one.)
2. **Does it need to be real-time?** Is a human or real hardware ever in the loop, or is it purely machine-vs-machine?
3. **What's the scale target?** Simulated miles/day, and what's the budget?
4. **What's the quality bar?** Who decides the efficient variant is "good enough," and against what metric?
5. **Frozen model or can I touch training?** Am I allowed to distill/finetune, or strictly serve what exists?

**My assumptions:**

- Teacher: ~30B-param latent world model (DiT-style backbone, diffusion-based frame generation), 3 camera views + lidar BEV, 10 Hz world steps, ~30 denoising steps per frame as trained.

:::note[Diffusion in one paragraph, for transformer people]
An LLM produces one token per forward pass. A diffusion model produces a **whole frame per ~30 forward passes**: start from pure noise over all of the frame's tokens, run the transformer, get a slightly-less-noisy frame, feed it back in, repeat ~30 times. A **DiT** is literally a ViT-shaped transformer over image-latent tokens, with the noise level injected like a timestep embedding. So all tokens of a frame come out *in parallel* (no left-to-right decode), but you pay N full passes per frame instead of one pass per token. Almost everything in this design about "step counts" and "step distillation" is about shrinking that N.
:::
- Consumers: (a) bulk closed-loop sim for long-tail scenario testing, (b) counterfactual "what-if" exploration, (c) offline synthetic dataset generation.
- I'm allowed to distill. The teacher is the quality reference, not the serving target.
- Target: support ~100K world-model-simulated miles/day for the long-tail program, within a fixed GPU budget, without regressing realism on safety-critical slices.

---

## 2. Problem framing and metrics — what does "cheap enough" mean?

This is an ML system design, so before any GPU talk I need success metrics that aren't just "it's faster."

**North-star (cost):** **$ per simulated mile**, broken out by tier (more on tiers below). This is the number leadership actually trades off against real-world testing miles.

**Throughput:** simulated miles per GPU-hour. This, not latency, is the primary serving objective for the bulk fleet.

**Latency:** per-tick latency matters only for the *interactive* pool (engineers debugging scenarios, hardware-in-the-loop). Two SLOs, two pools.

**Quality (the gating metrics — this is where ML design lives):**

1. **Downstream-task agreement (primary):** run the *same* scenario suite through teacher-backed sim and efficient-variant sim. The planner-under-test's outcomes (collision rate, intervention rate, scenario pass/fail, comfort metrics) must agree within a tolerance band. The simulator exists to evaluate the Driver — so the real quality metric is "does it produce the same evaluations," not FVD.
2. **Perceptual/realism metrics (secondary, diagnostic):** FVD per slice, lidar occupancy IoU vs teacher, camera-lidar cross-modal consistency. *(FVD = Fréchet Video Distance — the video analog of FID: embed generated and real clips with a pretrained video network, compare the two feature distributions; lower = more realistic. It measures distribution-level realism, not per-frame correctness.)*
3. **Slice-based, not aggregate:** night, rain, dense urban, vulnerable road users, rare agents (the elephant). An efficient variant that's great on average but degrades night-rain-pedestrian slices is a **fail**, full stop.

**Staff-level signal:** the acceptance metric for an inference optimization on a *simulator* is **agreement of downstream evaluations**, not image quality. FVD can move and nobody cares; if the distilled model makes the planner pass a scenario the teacher fails, the simulation program is now lying to you. That's the safety-critical framing the rubric wants.

**Fallback behavior when the model is wrong:** rollout-health monitors (below) flag degenerate generations; flagged rollouts are discarded and re-queued against the teacher, never silently kept.

---

## 3. The napkin math that shapes the whole design

**Staff-level signal:** do this arithmetic *before* proposing architecture. It kills two naive designs immediately.

**Tokens per world step.** The model never touches raw pixels — frames become token sequences the same way a ViT does it, just on a compressed version of the image:

```
720p camera frame:            1280 × 720 pixels
→ VAE encoder, 8× per side:   1280/8 × 720/8  =  160 × 90  =  14,400 latent vectors
  ("f8" = compress the image 8× in each spatial dimension; a small learned
   autoencoder, so the transformer works on a 160×90 grid instead of pixels)
→ 2×2 patchify:               160/2 × 90/2    =   80 × 45  =  3,600 tokens
  (group each 2×2 block of latents into one token — identical in spirit
   to ViT's 16×16 pixel patches)
```

So one camera frame is a **3,600-token sequence** — to the transformer it looks exactly like a ViT input. Three cameras ≈ 10.8K tokens, plus ~2K tokens for the lidar bird's-eye-view grid (tokenized the same way) ≈ **12K tokens per 100 ms world step**.

**Naive design #1 killed — token-by-token autoregression:** 12K tokens per 100 ms = **120K tokens/sec sustained per rollout stream**. LLM-style sequential decode does tens of tokens/sec/stream. Off by four orders of magnitude. Whatever we serve must generate a frame's tokens **in parallel** — either diffusion over the frame (all tokens denoised at once, per the note above) or MaskGIT-style parallel decode (start with all tokens masked, predict *many* of them per forward pass instead of one, keep the confident ones, repeat a handful of times). This is *why* Genie-class models are designed the way they are.

**Naive design #2 killed — serving the teacher online:**

```
Teacher, per world step:
  FLOPs/pass  = 2 × 30e9 params × 12e3 tokens ≈ 0.72 PFLOP
  × 30 denoise steps                          ≈ 22 PFLOPs per step
  H100 sustained (~40% MFU, BF16)             ≈ 0.4 PFLOP/s
  → ~55 GPU-seconds per 100ms of sim time     (550× slower than real time)

Per simulated mile (30 mph → 120 s sim → 1,200 steps):
  ~18 GPU-hours ≈ $36/mile  (at $2/GPU-hr)
```

$36/mile makes 100K miles/day a **$3.6M/day** program. The teacher is a *reference and data generator*, never the bulk serving target. The whole design follows from this.

**Context/KV memory.** 10 s of visual memory at 10 Hz = 100 frames × 12K = **1.2M tokens** of context. At ~57 KB/token of KV (28 layers, GQA 8×128, FP8) that's ~68 GB — doesn't fit beside weights. So temporal context must be engineered: full-resolution attention over the last ~1 s (12 frames, 144K tokens ≈ 8 GB), plus a compressed long-horizon memory (strided frames or a learned state). This is a *model* decision driven by a *serving* constraint.

---

## 4. Architecture lineage — why this family, from an inference standpoint

One minute on why the field landed here, because the serving strategy differs per family:

| Family | Examples | Inference cost profile | Why (not) here |
|---|---|---|---|
| Latent RNN/RSSM | World Models '18, Dreamer | Tiny — one recurrent step/tick | Cheap but too low-fidelity for sensor-level sim; great mental baseline |
| GAN video | DriveGAN | Single pass, fast | Mode collapse[^1], poor controllability; long-tail[^2] is exactly the modes it drops |
| AR token transformer | GAIA-1, Genie family | Parallel-in-frame decode, **KV-cache reusable across ticks** | Streamable, branch-friendly; error accumulation over long horizons |
| Diffusion / DiT | GAIA-2, DriveDreamer | N denoise passes per frame — the N is the cost | Best fidelity + controllability; N is attackable via distillation |

**Staff-level signal:** the AR-vs-diffusion choice is an *inference economics* choice, not just a quality one. AR amortizes context via KV cache and makes counterfactual branching cheap (shared prefix). Diffusion pays per-frame but distills to few steps. Our assumed teacher is diffusion-based, so step-count is the dominant lever — but I'd flag to the interviewer that a hybrid (AR temporal backbone + few-step diffusion head per frame) is where these converge, and it inherits *both* serving benefits.

---

## 5. Baseline first, then the optimization ladder

**Baseline:** serve the teacher as-is, offline batch only, fully utilized. This is not a throwaway: it establishes the **quality ceiling**, the **cost floor of doing nothing**, and generates the distillation dataset. Some workloads should *stay* here (golden-scenario libraries: generate once at $36/mile, replay thousands of times for ~free).

**Staff-level signal:** generation cost can be **amortized** — a simulator output is a reusable artifact, unlike a chat response. "Generate once, replay many" is the single cheapest optimization in the whole design and it requires zero ML work. Say it before saying "distillation."

Then the ladder, ordered by risk-adjusted ROI (cheapest/safest first):

```
Risk ▲
     │ 4. Capacity distillation: 30B teacher → ~3B student      (~10× compute)
     │    (classic teacher→student: small model trained to match
     │     the big model's outputs)
     │ 3. Step distillation: 30 → 2-4 denoise steps             (~8-15×)
     │    (train a student to jump from noise to the teacher's
     │     30-step result in a few big hops — consistency /
     │     progressive distillation; same weights count, fewer passes)
     │ 2. Quantization FP8 weights+acts+KV; sparsity if it holds (~1.7-2×)
     │ 1. Serving engineering: continuous batching of rollouts,
     │    CUDA graphs, paged KV, temporal-window attention,
     │    prefix sharing across counterfactual branches          (~2-4× fleet-wide)
     └────────────────────────────────────────────────────────► Gain
```

Levels 1–2 don't change the model's function meaningfully — ship behind standard regression gates. Levels 3–4 *create a new model* — they go through the full ML evaluation gauntlet (section 7), and they compound:

```
Efficient variant: 3B student, 4 denoise steps, FP8
  FLOPs/step = 2 × 3e9 × 12e3 × 4 ≈ 0.29 PFLOP
  H100 FP8 sustained ≈ 0.8 PFLOP/s → ~0.36 GPU-s per world step
  → ~0.12 GPU-hr/mile ≈ $0.24/mile     (~150× cheaper than teacher)
  100K miles/day ≈ $24K/day            (program is now fundable)
```

Real-time check: 0.36 s/tick on one GPU is 3.6× too slow for the 100 ms interactive tick. Data parallelism can't fix this — a closed-loop rollout is sequential (the planner consumes frame *t* before action *t+1* exists), so replicas add concurrent *streams*, never faster *ticks*. The levers on single-stream latency are: fewer steps, a faster chip, or splitting the model. But splitting isn't free, and the LLM-serving intuition that "TP comms are negligible on NVLink" does **not** transfer:

```
TP4, per denoise pass (28 layers, hidden 3072, 12K tokens):
  compute:  7.4e13 FLOPs ÷ 4 GPUs ÷ 0.8 PFLOP/s (FP8)   ≈ 23 ms
  comms:    2 all-reduces/layer × 28 = 56 all-reduces
            message = 12K tok × 3072 × 2 B ≈ 75 MB each  (LLM decode: ~6 KB!)
            ring traffic ≈ 113 MB/GPU @ ~400 GB/s NVLink ≈ 0.3 ms each → ~16 ms
  → ~39 ms/pass unoverlapped → 4 passes ≈ 156 ms.  Naive TP4 MISSES the budget.
```

In LLM decode, TP all-reduces carry one token and are latency-bound; here every pass forwards all 12K frame tokens, so they're 75 MB and *bandwidth*-bound — ~40% overhead, and per-GPU ring traffic (2(N−1)/N × message) barely changes with N, so TP8 shrinks compute but not comms. Closing the gap to 100 ms takes some combination of: FP8 activations on the wire (halves comms), comm/compute overlap[^3], a 2-step interactive variant, or **sequence parallelism** instead of TP — split the 12K tokens across GPUs (Ulysses/ring-attention style), and the MLP (~⅔ of FLOPs) needs no comms at all, which is why DiT serving stacks tend to prefer SP over Megatron-style TP. Either way the split must stay inside one NVLink island; crossing nodes over IB is ~10× worse and instantly dominates.

**Staff-level signal:** parallelism strategy is *per pool*: the bulk fleet is pure data parallelism (comm-free replicas, near-linear scaling — exactly what a small model wants for throughput); model-splitting is conceded only where single-stream latency must shrink, and you do the comms arithmetic before claiming it fits. So: **a small SP/TP low-latency pool for interactive use; everything else runs throughput-mode on independent replicas.**

**Staff-level signal:** simulation does **not** need to run in real time unless a human or physical hardware is in the loop. Sim-time and wall-clock are decoupled; the bulk fleet should run at whatever rate maximizes miles/GPU-hour (large batches, full utilization), even if each rollout is slower than real time. Conflating "10 Hz world" with "10 Hz wall-clock" is the most common over-engineering trap in this problem.

---

## 6. System architecture

Top level — four components, one job each:

```
                  ┌─────────────────────┐
                  │  Scenario Scheduler │  decides WHAT to simulate,
                  └──────────┬──────────┘  at which tier, on which pool
                             │
              ┌──────────────┴───────────────┐
              ▼                              ▼
   ┌──────────────────────┐      ┌──────────────────────┐
   │      BULK POOL       │      │   INTERACTIVE POOL   │
   │  optimize: miles/    │      │  optimize: tick      │
   │  GPU-hour            │      │  latency             │
   │                      │      │                      │
   │  student · FP8       │      │  student · SP/TP     │
   │  big batches         │      │  10 Hz, small        │
   └──────────┬───────────┘      └──────────┬───────────┘
              │                             │
              └──────────────┬──────────────┘
                             ▼
                  ┌─────────────────────┐
                  │ Eval & Metrics Store│  outcomes, slices,
                  └─────────────────────┘  $/mile, realism scores

   off to the side, small and offline:
   ┌─────────────────────────────────────────────────┐
   │ TEACHER POOL — golden scenarios · distillation  │
   │ data · standing audit stream · second opinion   │
   │ on any rollout the health checks flag           │
   └─────────────────────────────────────────────────┘
```

Inside every rollout worker, the closed loop itself:

```
      ┌─────────────┐    action aₜ     ┌────────────────┐
      │   Planner   │ ───────────────► │  World Model   │
      │ under test  │ ◄─────────────── │  (KV / state   │
      └─────────────┘   frames tₜ₊₁    │   cache lives  │
             │                         │   here)        │
             │ health check fails?     └────────────────┘
             ▼
      discard rollout, requeue to teacher
```

Walking through it: the **scheduler** is a priority queue of simulation requests (long-tail mining, regression suites, counterfactual fanout) and routes each to a tier and pool. The **bulk pool** is where ~all the GPUs are — independent replicas, big batches, nothing fancy. The **interactive pool** is small and exists only because humans and hardware-in-the-loop need 10 Hz wall-clock. The **teacher pool** never serves production traffic; it generates golden scenarios and distillation data, and arbitrates anything the student's health checks flag. Everything writes outcomes to one **metrics store**, because the agreement dashboards (section 7) need student and teacher results side by side.

**The closed-loop tick** (bulk mode, batched across rollouts):

```
tick t:   [planner fwd]──►[cond. embed]──►[4× denoise, batched]──►[VAE decode]──►[planner obs t+1]
           ~5 ms            ~1 ms           ~bulk of compute         ~ms            next tick
                            (action/layout  (4 student fwd passes    (latents →
                             → tokens)       over 12K frame tokens)   pixels)
          (per rollout; batch dimension = active rollouts on this worker)
```

**Counterfactual branching is a first-class serving feature.** "Same world, what if ego brakes vs swerves?" forks a rollout. The branches share their entire past → share the cached temporal state/KV prefix. Marginal cost of branch #2..N is only the divergent suffix:

```
            shared prefix (cached once)
  ─────●────●────●────●─┬─► branch A: ego brakes
                        ├─► branch B: ego swerves
                        └─► branch C: ego maintains
```

**Staff-level signal:** counterfactual fanout is *the* workload where world-model sim beats log replay, and prefix sharing makes its marginal cost low. Designing the cache around branch-and-share (copy-on-write rollout state, like paged attention's block sharing) is what makes the "what-if" product economically different from "run N independent sims."

Versioning/ops: every rollout records `(model hash, sampler config, seeds)` → bit-reproducible re-runs; model registry with immutable versions; rollback = repoint the worker pool, no state migration.

---

## 7. Evaluation: how an efficient variant earns trust

The distilled student is a new model pretending to be the teacher. The gauntlet, in order:

1. **Offline regression suite:** fixed scenario set with frozen seeds. Per-slice FVD / lidar IoU / cross-modal consistency vs teacher outputs. Catches gross degradation cheaply.
2. **Downstream-agreement (the real gate):** run the full evaluation suite — same scenarios, same planner build — on teacher-sim and student-sim. Compare *outcome distributions*: scenario pass/fail agreement, collision/intervention deltas, per-slice. Pre-registered tolerance bands; safety-critical slices get the tightest bands.
3. **Long-horizon drift test:** distilled few-step models drift differently than their teacher over 60+ second rollouts. Measure perceptual drift and physics violations (object permanence, kinematic feasibility of other agents) as a function of horizon. Set a max-trusted-horizon; the scheduler enforces it.
4. **Shadow:** student runs alongside the teacher on a sampled slice of production sim traffic; outcomes compared continuously for a soak period. *Offline/online mismatch shows up here* — e.g., scenario distributions in production skew harder than the regression suite.
5. **Canary:** X% of the sim fleet, watch agreement + rollout-health dashboards, automated rollback on tripwire.

**Staff-level signal:** name the failure mode this protects against — **silent evaluation drift**. If the cheap simulator drifts, every downstream safety conclusion drawn from it is contaminated, and nothing crashes to tell you. That's why shadowing against the teacher never fully stops: a standing few-% audit stream is the price of trusting the student.

---

## 8. Monitoring and retraining triggers

- **Rollout health (online, per-rollout):** NaN/saturation detectors, frame-to-frame perceptual delta (frozen worlds / teleporting agents), physics sanity checks. Flagged → discard + requeue to teacher; flag *rate* is itself a drift signal.
- **Agreement drift (continuous):** the standing teacher-audit stream feeds an agreement dashboard, sliced by scenario type, weather, geography. Trend alarms, not just thresholds.
- **Input drift:** the scenario mix changes as the long-tail program evolves (new city, new scenario generators). Monitor scenario-feature distributions; a new mode in inputs invalidates the student's eval coverage before it degrades quality — that's a *re-validation* trigger, ideally before an incident, not after.
- **Cost/utilization:** $/mile and MFU per pool, regression-gated per release like a latency SLO.
- **Retraining triggers:** agreement below band on any safety slice → freeze rollouts on that slice to teacher, mine the gap, re-distill with targeted data. The teacher pool is sized to absorb this fallback load — fallback capacity is part of the design, not an incident response.

---

## 9. When simpler wins — the simulation portfolio

**Staff-level signal:** the world model is the *top* tier of a portfolio, not a replacement for it. Cost spans ~5 orders of magnitude; the scheduler's real job is routing each question to the cheapest tier that can answer it.

```
        $/mile   fidelity        use when
  ┌─────────────────────────────────────────────────────────────┐
  │ Log replay        ~$0.0001  recorded reality  regression on  │
  │                                               seen scenarios │
  │ Object-level sim  ~$0.001   abstract agents   behavior/      │
  │ (Waymax-style)              (boxes, no pixels) planning, RL  │
  │ Student world sim ~$0.25    sensor-level      long-tail,     │
  │                             generative        counterfactual,│
  │                                               perception-in- │
  │                                               loop testing   │
  │ Teacher world sim ~$36      reference-grade   golden sets,   │
  │                                               distill data,  │
  │                                               arbitration    │
  └─────────────────────────────────────────────────────────────┘
```

If the question is "does the planner merge correctly," object-level sim answers it 250× cheaper — burning world-model miles on it is malpractice. The expensive tiers are reserved for what *only* they can do: perception-in-the-loop, sensor-level long tail, generative counterfactuals.

---

## 10. Deep dives — where an inference interviewer will drill

Three places a runtime-optimization interviewer will push past the headline numbers. Each is also a chance to show the *second-order* reasoning.

### 10.1 The FLOPs the napkin math dropped: attention

Section 3 counted only weight matmuls (2 × params × tokens) — the standard LLM habit. A sharp interviewer asks: what about attention? At video context lengths it is **not** a rounding error:

```
Full spatiotemporal attention, per denoise pass
(12K current-frame tokens attending to a 144K-token window,
 d_model 3072, 28 layers):

  QKᵀ + AV ≈ 4 × 12e3 × 144e3 × 3072  ≈ 2.1e13 FLOPs/layer
  × 28 layers                          ≈ 5.9e14 FLOPs
  weight matmuls (from §3)             ≈ 0.7e14 FLOPs
  → attention = ~8× the matmuls. The §5 cost model would be off ~9×.
```

What rescues it is **factorized attention**, standard in video DiTs: *spatial* attention within each frame, *temporal* attention across the window at each spatial location — never full all-to-all over space×time:

```
  spatial:  per camera 3.6K × 3.6K, ×(3 cams + lidar), ×28  ≈ 1.5e13
  temporal: 12K queries × 12-frame depth, ×28               ≈ 5e10 (noise)
  → attention back to a ~20% tax on matmuls; §5 conclusions hold.
```

Trade-off: factorization weakens long-range space-time coupling (an agent moving fast across views in one tick); the common compromise is a few full-attention layers in an otherwise factorized stack — and *which* layers stay full-attention is a quality/latency knob you tune with the eval harness, not a fixed property of the model.

**Staff-level signal:** "transformer cost = 2·N·T" is an LLM-serving instinct. Know which regime you're in: at video sequence lengths, the attention pattern *is* a serving decision, and quoting per-mile costs without stating the attention structure is a ~9× error bar.

### 10.2 FP8 on a DiT — what actually breaks

FP8 gives ~2× tensor-core throughput and halves memory and wire bytes (it's load-bearing in both the §5 cost model and the TP comms math). The drill-down is *where it's unsafe*:

- **AdaLN modulation** — the conditioning pathway (timestep + action embeddings → per-layer scale/shift). Tiny tensors with huge dynamic range; quantizing them is all risk, no win. Keep BF16.
- **First and last blocks** — input patchify and the final projection back to latents. Output-layer quantization error shows up as visible banding/texture artifacts after VAE decode. Keep high precision; it's a few % of FLOPs.
- **Attention internals** — softmax and accumulators stay FP32 inside FlashAttention regardless; FP8 applies to the projections.
- **Everything else** — the QKVO and MLP GEMMs, ~95% of FLOPs — quantizes well with per-channel weight scales and delayed/dynamic activation scaling.

**Staff-level signal:** quantization error is **slice-biased**. Night and heavy-rain frames occupy a narrow band of the value distribution, so a global scale gives them *larger relative error* — the quantized model degrades most on exactly the safety-critical slices. Two consequences: the calibration set must be slice-stratified (not a random sample of sunny miles), and FP8 ships through the same per-slice agreement gates as a distilled model, even though it "didn't change the weights."

### 10.3 Branch fanout is memory-bound, not compute-bound

Counterfactual branching (§6) shares the past via copy-on-write state blocks — one block = one frame's KV ≈ 12K tokens × 57 KB ≈ **0.7 GB** (FP8). The natural question: how wide can you fan out per GPU?

```
Steady state, branches fully diverged (each holds its own 12-frame window):
  per-branch window: 12 × 0.7 GB ≈ 8.2 GB
  H100 80 GB − ~4 GB weights      → ~9 fully-diverged branches/GPU
  (compute would happily batch 9 branches — HBM runs out first)
```

So the levers are memory levers: frame-block CoW with refcounts (branches pay only for divergent *suffix* frames, so shallow fanouts are much wider than 9), harder temporal compression of older frames, spilling cold branches to CPU and prefetching at fork-resume. And one free gift this workload gives you: **every tick has an identical shape** (12K tokens, fixed window — no ragged sequence lengths like LLM serving), so CUDA graphs capture-once-replay-forever and static batch slots work perfectly; rollouts joining and leaving just swap into fixed slots between ticks.

**Staff-level signal:** state the binding constraint before optimizing. Fanout looks like a compute-scheduling problem; the arithmetic says it's an HBM-capacity problem, and the design answer (CoW blocks + eviction tiers) follows from that, not from FLOPs.

## 11. Research pass — new developments (as of June 2026)

What the frontier did while this doc was being written — and what it changes:

- **The field converged on exactly the hybrid §4 predicted.** The dominant 2025–26 recipe for real-time interactive video/world models is *asymmetric distillation*: a bidirectional diffusion teacher distilled into a **causal, few-step autoregressive student** — streamable, KV-cacheable, real-time. [CausVid](https://arxiv.org/abs/2412.07772) (CVPR 2025) established it; Self-Forcing (2025) trained the student on its own rollouts; [Causal Forcing](https://arxiv.org/abs/2602.02214) (Feb 2026) and Causal Forcing++ (May 2026) fixed its initialization pathologies; [Rolling Forcing](https://arxiv.org/abs/2509.25161) does real-time long-horizon streaming.
- **Real-time interactive world models are now shipping practice**, not research: HY-WorldPlay, Hunyuan-GameCraft-2, RELIC, Yume-1.5 (2025–26) all serve user-controllable world simulation live. The interactive-pool requirement in §5 has working existence proofs.
- **Waymo's own stack confirmed the shape** ([Waymo World Model](https://waymo.com/blog/2026/02/the-waymo-world-model-a-new-frontier-for-autonomous-driving-simulation/), Feb 2026): Genie-3-based, camera + lidar, action/layout/language control, and an explicit *efficient variant* for large-scale simulation — the teacher/efficient-variant split of this design, in production.
- **What this changes here:** the serving target is less "diffusion with fewer steps" and more "causal student with a KV cache plus a 1–4 step head." That promotes the prefix-sharing/branching machinery (§6, §10.3) from optimization to *central serving primitive* — a causal student makes counterfactual forking exactly as cheap as the CoW design assumed.

## 12. Summary — what I'd want the interviewer to remember

1. **Napkin math first**: 12K tokens / 100 ms kills naive AR; $36/mile kills serving the teacher. The design is forced by arithmetic, not preference.
2. **Two SLOs, two pools**: throughput (miles/GPU-hr) for the bulk fleet, latency only where a human/hardware is in the loop. Sim-time ≠ wall-clock.
3. **Optimization ladder by risk**: amortize (generate-once-replay-many) → serving engineering → quantization → step + capacity distillation. ~150× compounded.
4. **The gate is downstream agreement**, per-slice, not image quality — an efficient simulator that changes the planner's evaluations is worse than a slow one.
5. **Standing teacher audit** forever; silent evaluation drift is the failure mode that matters.
6. **Portfolio, not monolith**: route every sim question to the cheapest tier that answers it.

[^3]: **Comm/compute overlap:** GEMMs run on the SMs; an all-reduce mostly uses NVLink and the copy engines. Different hardware resources — so they *can* run concurrently. The catch is dependency: layer ℓ's all-reduce output feeds layer ℓ's residual-add/LayerNorm, so you can't just compute ahead. The standard trick is **chunking**: split the 12K frame tokens into k chunks, and the moment chunk 1's GEMM finishes, launch its all-reduce on a side CUDA stream while the SMs start chunk 2's GEMM:

    ```
    Unoverlapped (one 75 MB all-reduce after the full GEMM):
      SMs:     [████████ GEMM, 12K tokens ████████]
      NVLink:                                      [████ AR 75 MB ████]
      wall:    |—————— 23 ms ——————|—————— 16 ms ——————|   = 39 ms

    Overlapped (4 chunks × 3K tokens, all-reduces pipelined behind compute):
      SMs:     [ G1 ][ G2 ][ G3 ][ G4 ]
      NVLink:        [ A1 ][ A2 ][ A3 ][ A4 ]
      wall:    |—— ~max(compute, comms) + last chunk's AR ≈ 27 ms ——|
    ```

    Exposed comm time shrinks from the whole tensor to roughly the *last chunk's* all-reduce: ideal overlap turns `compute + comms` into `max(compute, comms)`, so our 39 ms pass becomes ~25–28 ms. Caveats that keep it from being free: NCCL kernels steal some SMs (the overlapped GEMMs run ~10–20% slower), more chunks hide more comms but make each GEMM smaller and less efficient, so k is tuned (4–8 typical), and the final chunk's all-reduce is always exposed. In practice you don't hand-roll this — it's a library feature: TransformerEngine's TP overlap, Megatron's sequence-parallel overlap, PyTorch async-TP.

[^2]: **Long tail:** picture a histogram of driving scenarios sorted by frequency. A handful of scenario types (cruising a lane, stopping at a light, routine merges) account for almost all miles — that's the head. Then comes an enormous number of scenario types that are each individually rare — a couch on the freeway, a tornado, an elephant on the road — stretching out as a long, thin tail. No single tail event matters statistically, but *collectively* the tail is where nearly all safety-critical risk lives, because the common cases were solved long ago. The brutal data property: you can 10× your fleet miles and still never observe a specific tail event — which is exactly why Waymo wants a generative world model that can *synthesize* tail scenarios instead of waiting to encounter them.

[^1]: **Mode collapse:** the classic GAN failure. The generator is only rewarded for fooling the discriminator, so it can "win" by producing a few safe, plausible outputs over and over instead of covering the full diversity of the data distribution — entire regions (modes) of reality are simply never generated, and no loss term complains. Each sample looks fine in isolation; the *distribution* is impoverished. For a driving simulator this is disqualifying: rare scenarios are by definition the low-density modes, i.e. exactly what gets dropped first. Diffusion and AR models train with likelihood-style objectives that pay a penalty for ignoring modes, which is a key reason the field moved to them for long-tail generation.
