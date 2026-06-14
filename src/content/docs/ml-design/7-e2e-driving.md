---
title: "Design: end-to-end driving with a world model"
description: "Staff-level end-to-end ML system design for a world-model-based / end-to-end driving system: framing, metrics, data, modeling, onboard serving, evaluation, and a safety-gated rollout."
---

# End-to-End Design: End-to-End Driving with a World Model

> **Problem.** Design an **end-to-end driving** system that maps sensor inputs (camera, lidar, radar) and a route to driving actions, using a **learned world model** to imagine and evaluate possible futures before committing to a plan. Cover training, the onboard inference path, and — above all — how you make a learned end-to-end stack *safe enough to ship*.

This is the **end-to-end driving** specialization, and it is the most ambitious of the three designs because it questions the modular stack itself. The classic AV stack is **perception → prediction → planning → control**, each a separately built, separately tested module. An end-to-end / world-model approach learns more of that pipeline jointly. The staff-level answer is *not* "replace everything with one network" — it is knowing **exactly which parts to learn jointly, which to keep modular, and how to bound a learned planner's failure modes**.

The world model is the through-line of this whole section; here it becomes a *planning* component. Walk the blueprint, with safety as the spine.

---

## 1. Clarify and scope

```text
classic:   sensors -> perception -> prediction -> planning -> control
e2e:       sensors -> [ learned model, possibly with a world model inside ] -> action
```

Resolve first:

- **How end-to-end?** A spectrum, and naming it is the staff move:
  - *Modular, learned components* — each stage a network, separately supervised (the conservative, debuggable default).
  - *End-to-end with intermediate representations* — one model, but with **interpretable middle outputs** (BEV occupancy, detected agents, predicted trajectories) you can supervise and inspect. **This is the sweet spot** (the UniAD / planning-oriented lineage).
  - *Fully black-box* sensors→steering — elegant in papers, **near-unshippable** as a primary stack: no interpretability, no intermediate checks, catastrophic failure modes. Useful as research / a secondary signal.
- **Where the world model sits.** Two roles: (a) a **driving policy** that outputs actions directly; (b) a **world model** that *predicts futures conditioned on candidate ego actions*, so the planner can roll out "what if I do X?" and pick the safest — model-based planning (this is exactly the [world-model design](/ml-design/) framing applied to driving).
- **Constraints.** Onboard, hard-real-time (~10 Hz), safety-critical, bounded compute/power. If the world-model rollout is in the planning loop, **rollout latency × candidate actions × horizon** dominates — the inference-optimization core of this section.
- **Failure cost.** Unbounded. A confident wrong action causes a crash. This single fact makes interpretability and fallback non-negotiable, and it is why "just train a bigger net" is the wrong answer.

Central tension:

> An end-to-end model can capture interactions and joint optimization a hand-built modular stack can't — but it trades away the **interpretability, testability, and bounded failure modes** that make the modular stack *safe to ship*. The design's job is to buy the joint-optimization upside **without** giving up safety — usually via interpretable intermediate representations and a hard safety layer.

---

## 2. Frame the ML problem

**Input:** multi-camera (+ lidar/radar) sensor stream over a short history, ego state, route/navigation goal.

**Output:** a planned **ego trajectory** (then handed to a low-level controller). Not raw steering — a trajectory keeps a clean interface to control and to the safety layer.

**Two architectural framings, and I'd combine them:**

1. **Direct policy (imitation / RL):** `sensors + route -> ego trajectory`, with **auxiliary supervised heads** for BEV occupancy, detection, and prediction so the middle is interpretable and gets dense gradient signal.
2. **Model-based planning with a world model:** learn `f(state, ego_action) -> future state distribution`; at inference, **roll out candidate ego actions** through the world model and pick the trajectory with the best cost (safety + comfort + progress). The [diffusion world model](/diffusion/3-diff-world-model/) — latent rollout of future occupancy/scene — is the engine.

**Training vs. real objective** — the gap is the whole problem:
- *Training:* imitate expert trajectories (log-likelihood / regression) and/or maximize a reward in sim.
- *Real:* drive **safely, comfortably, and make progress** across the full ODD including the tail.
- The gap is **causal confusion + compounding error + distribution shift**: imitation learns *correlations* ("the car ahead is braking → I brake"), not *cause*, and at test time the policy's own small errors take it off the expert manifold (the **DAgger** problem). This is *the* reason naive end-to-end imitation fails, and you must say so.

---

## 3. Metrics

Layered, with safety dominating.

**Open-loop (necessary, deeply insufficient):**
```text
trajectory L2 / ADE vs. expert
collision / off-road rate against logged future
```
**Open-loop L2 is famously misleading** — a model can score great by extrapolating the log yet be unable to *recover* from its own mistakes (it never sees off-manifold states in open-loop replay). The cautionary result to cite: on **nuScenes** open-loop planning, a tiny MLP fed only the **ego's own state** (no perception at all) rivals heavy sensor models — the benchmark is largely measuring "continue current kinematics," not driving. Never sell an end-to-end system on open-loop L2 alone.

**Closed-loop (the real metrics)** — in [simulation](/ml-design/6-closed-loop-simulation/):
```text
collision rate / safety violations    (the gating metric)
route completion / progress
comfort: jerk, hard-brake rate
rule compliance: red lights, stop signs, lane keeping
interventions / disengagements per mile
```

**Public benchmarks to name:** **CARLA** (and the CARLA Leaderboard / Bench2Drive) for *closed-loop* simulated driving, **nuPlan** for closed-loop planning on real logs, and **nuScenes** for open-loop (with the caveat above). The pattern to point out: the field is deliberately migrating from open-loop to closed-loop benchmarks for exactly the recovery/distribution-shift reason.

**Slice everything** by scenario type — unprotected lefts, dense peds, merges, occlusion, night/rain, construction. Aggregate "miles per intervention" hides that the system is fine on highways and dangerous at intersections.

**Calibration & uncertainty:** the model must *know when it doesn't know* — uncertainty quality is itself a metric, because it drives the fallback in section 4.

---

## 4. Fallback when uncertain or wrong — the safety layer

This section is the reason the whole design is shippable. A learned end-to-end model is **not trusted unconditionally.**

- **A separate, non-learned safety layer** sits *below* the policy: a verifiable checker (reachability / responsibility-sensitive-safety-style envelope, emergency braking) that can **override** any planned trajectory that violates a hard safety constraint. The neural planner *proposes*; the safety layer *disposes*. This bounds the worst case independently of the network.
- **OOD detection:** high predicted uncertainty, low world-model likelihood, or sensor degradation → **degrade gracefully**: hand to a simpler conservative policy, slow down, pull over, or (in supervised testing) request disengagement. Never let a low-confidence black-box action reach actuation.
- **Interpretable intermediates as runtime checks:** because we kept BEV occupancy / detections in the middle (section 1), we can sanity-check the plan against them onboard — if the planner steers into occupied occupancy, block it.
- **Redundancy:** keep a simpler independent fallback planner; the learned stack is the primary, not the sole, path to a safe action.

The principle, stated plainly: **the learned model is bounded by a verifiable safety envelope it cannot exit.**

---

## 5. Data and labeling

- **Expert demonstrations from logged drives** are the abundant imitation source (the ego's own trajectory is the label) — but imitation alone gives **causal confusion + no recovery data** (section 2). The whole data strategy is about fixing that.
- **The recovery / off-manifold problem:** logs only contain good driving, so the model never learns to recover from mistakes. Fixes: **DAgger-style** on-policy data aggregation (run the policy, have an expert/oracle label the states it actually visits), **perturbation augmentation** (synthesize off-nominal starts and the correction back), and **closed-loop training in the [simulator](/ml-design/6-closed-loop-simulation/)** where the policy *experiences and corrects* its own errors.
- **The long tail is safety-critical and rare:** mine interventions, near-misses, and disengagements (each is a gold-labeled "expert did something important" event); **generate** rare scenarios in sim; oversample the tail.
- **World-model training is self-supervised** — future sensor/occupancy frames are their own labels, abundant and cheap (the section's recurring point).
- **Leakage:** split by geography/time; hold out whole ODDs to measure generalization honestly.

---

## 6. Baseline first

- **The modular stack itself is the baseline** — production perception + prediction + a rule-based/optimization planner. It works, it's debuggable, and any end-to-end approach must beat it on closed-loop safety *and* keep its interpretability to justify the move. Stating this is the credible, non-hypey staff answer.
- **Simple imitation (CIL-style):** sensors → trajectory, no world model, no intermediates. Establishes the floor and immediately exposes causal confusion and the recovery problem — motivating everything in section 7.

Don't propose a moonshot end-to-end model without first respecting why the modular baseline is hard to beat on safety.

---

## 7. Modeling step-up

Layer in complexity only where it buys joint optimization the modular stack can't — and keep the middle interpretable.

**Backbone — BEV-centric, multi-modal, with interpretable intermediates:**
```text
multi-camera (+lidar) -> per-view encoders
        -> lift/fuse into a BEV / 3D feature space
        -> [aux heads: occupancy, detection, map, prediction]   <- supervised, interpretable
        -> planning head / world-model rollout -> ego trajectory
```
This is the **planning-oriented end-to-end** pattern (UniAD-style): one differentiable model, but each stage is supervised and inspectable, so gradients flow end-to-end *and* you keep testability.

**The world model for planning** — the section's core idea applied:
- Learn a **latent world model**: encode current scene → roll out **future occupancy/scene latents conditioned on candidate ego actions** → score each rollout for safety/comfort/progress → pick the best. This is **model-based planning**: the action condition (from [U-Net/DiT conditioning](/diffusion/6-unet-dit-backbones/)) is exactly the ego trajectory you're evaluating.
- *Why a world model over a direct policy:* it lets the planner **reason about consequences** ("if I nudge left, does that agent brake?") and is more interpretable than a black-box policy. *Cost:* you must roll out futures fast for many candidate actions — the inference problem of section 8.

**Multimodality & uncertainty:** futures are multimodal, so the world model must be **generative** (diffusion/flow over latents) to avoid mode-averaging, and must expose calibrated uncertainty for the fallback layer.

**Trade-off to narrate:** direct policy = cheap, fast, but black-box and brittle to distribution shift; world-model planning = interpretable, consequence-aware, handles multimodality, but expensive at inference and harder to train. The pragmatic answer is **interpretable-intermediate end-to-end + a world-model planning head + a hard safety layer** — not a single black box.

---

## 8. Serving and systems (go deep — onboard + rollout)

Onboard, hard-real-time, and if the world model is *in the planning loop*, **rollout cost dominates.**

**Cost structure (model-based planning):**
```text
total = perception/encode (once per frame, shared)
      + num_candidate_actions x horizon_steps x per-step world-model rollout cost
per-step cost = NFEs (if diffusion) x backbone cost
```
This is the exact NFE-accounting the section drills: cost = denoiser calls × guidance × horizon × candidate actions × samples. Levers:

- **Latent-space rollout.** Roll out in the compressed latent, **decode to pixels/occupancy only when needed** — the central latent-diffusion efficiency move. The denoiser runs many times; the decoder runs rarely.
- **Few-step sampling.** Distillation / consistency models / DPM-Solver++ to cut a 50-step diffusion world model to a handful of NFEs — non-negotiable for in-the-loop planning. Measure the fidelity/safety cost of each cut.
- **Bound the search.** Cap candidate actions (coarse-to-fine: evaluate few coarse trajectories, refine the best) and horizon. Fixed, predictable worst-case compute beats best-effort in a safety-critical real-time loop.
- **Amortize.** Encode sensors/scene once per frame; share across all candidate rollouts (the "encode once, denoise cheaply" pattern again).
- **Quantization (FP8/INT8), kernel fusion, KV-cache** for any autoregressive backbone; batch the candidate rollouts on the onboard accelerator.
- **Deterministic latency budget.** The worst-case (densest scene, full action set) must fit the cycle, every cycle.

**Versioning & train/serve consistency:** sensor calibration, BEV projection, and feature builders must be byte-identical train vs. onboard, or train-serve skew silently degrades a *safety-critical* model. Co-version everything with the model; full model registry and provenance.

---

## 9. Evaluation

- **Closed-loop in [sim](/ml-design/6-closed-loop-simulation/) is primary** — open-loop L2 is a sanity check only (section 3). Measure sliced safety/comfort/progress against *reactive* sim agents, because an end-to-end policy especially must be tested on the states *it* visits, not the log's.
- **The offline↔online gap is acute here** because of distribution shift — a model can ace open-loop logs and fail closed-loop. Weight closed-loop and on-road heavily.
- **Validate the world model itself** (the [validation hierarchy](/diffusion/5-diff-validation/)): does a candidate action *causally* change the predicted future? Are rollouts physically consistent and stable over the horizon? A pretty but non-reactive world model gives a confidently wrong plan.
- **Failure analysis** on closed-loop crashes → cluster → mine/generate more of that slice → retrain. This loop, plus the safety-layer override logs, is where real safety gains come from.

---

## 10. Rollout and monitoring

Safety-critical rollout is slow and gated by design:

- **Sim gates first:** millions of [simulated](/ml-design/6-closed-loop-simulation/) miles across sliced scenarios; must beat the modular baseline on closed-loop safety before any road exposure.
- **Shadow mode on the fleet:** the end-to-end model runs onboard and *proposes* trajectories, but the **current shipped stack drives**. Log every divergence and every case where the safety layer would have overridden — huge volumes of free, zero-risk real-world evaluation. This is the most important rollout stage.
- **Supervised canary:** safety-driver-monitored deployment in a tight ODD, expand ODD-by-ODD only as interventions/safety metrics clear bars.
- **The safety layer is always live**, even in canary — the learned model never has unbounded authority.
- **Monitor drift:** new ODDs, weather/season, construction, sensor degradation; track per-slice safety, intervention rate, uncertainty calibration, and safety-layer override frequency (a *rising* override rate means the learned planner is drifting OOD).
- **Retraining triggers:** slice regression, rising disengagements/overrides, new ODD, or accumulated mined long-tail volume. **Rollback** to the modular stack or prior model is instant and automatic on any safety-guardrail breach.

---

## What to say in one breath

> End-to-end driving questions the modular **perception→prediction→planning** stack by learning more of it jointly, but the staff answer is *not* a single black box — it's **interpretable-intermediate end-to-end** (BEV occupancy, detection, prediction supervised in the middle) with a **world-model planning head** and a **hard, verifiable safety layer the network cannot exit**. I'd use the world model for **model-based planning**: roll out candidate ego actions through a latent world model and pick the safest — which directly invokes this section's NFE accounting, latent rollout, and few-step sampling for onboard real-time. The core difficulty is **causal confusion + compounding error + distribution shift** from naive imitation, so the data strategy is **DAgger / closed-loop sim training / perturbation recovery**, not just log imitation, and the eval is **closed-loop, sliced** — open-loop L2 is a known liar. Rollout is **sim-gated → shadow → supervised canary**, the safety layer always live, with rollback to the modular stack on any breach. The spine: capture joint-optimization upside **without** surrendering interpretability, testability, and bounded failure.

---

## Further reading

- [Planning-oriented Autonomous Driving (UniAD)](https://arxiv.org/abs/2212.10156)
- [End-to-End Autonomous Driving: Challenges and Frontiers (survey)](https://arxiv.org/abs/2306.16927)
- [MILE: Model-Based Imitation Learning for Urban Driving](https://arxiv.org/abs/2210.07729)
- [Diffusion World Model](/diffusion/3-diff-world-model/) and [U-Net and DiT Backbones](/diffusion/6-unet-dit-backbones/) — the world-model engine and conditioning.
