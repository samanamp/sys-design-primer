---
title: "Design: onboard motion forecasting"
description: "Staff-level end-to-end ML system design for a real-time multi-agent motion forecasting system that feeds the AV planner: framing, metrics, data, modeling, serving, evaluation, and rollout."
---

# End-to-End Design: Onboard Motion Forecasting

> **Problem.** Design the **motion forecasting** (prediction) system on a self-driving car. Given the perception stack's current view of the world — ego state, tracked agents, the map, traffic lights — predict where every relevant agent will be over the next several seconds, as a *distribution* over futures, so the planner can choose a safe trajectory. Cover training and the onboard inference path you would own.

This is one of the three specialization designs in this section. It is the classic AV prediction problem, and the staff-level version is not "predict a trajectory" — it is "produce a calibrated, multimodal, multi-agent forecast within a hard onboard latency budget, and prove it is safe on the long tail."

Walk the standard blueprint: **clarify → frame → metrics → fallback → data → baseline → modeling → serving → evaluation → rollout**. Narrate the trade-offs as you go.

---

## 1. Clarify and scope

The first move is to nail down where this sits in the AV stack and what the consumer needs.

```text
sensors -> perception (detection + tracking) -> PREDICTION -> planner -> control
                                                   ^you are here
```

Questions to resolve before designing:

- **Consumer.** The planner. It does not want one trajectory per agent — it wants a *distribution* it can reason about for collision risk. That single fact drives the whole design toward multimodal, probabilistic output.
- **Horizon and rate.** Typical: predict **5–8 seconds** ahead at **10 Hz**, refreshed every cycle (so ~100 ms budget end to end, of which prediction gets a slice — call it a **10–30 ms** target).
- **Agents.** Vehicles, pedestrians, cyclists. Up to ~100 tracked agents in a dense urban scene; you predict for the ~10–30 that are *relevant* to the ego plan.
- **Scale of the problem, not QPS.** This is **onboard, single-stream, hard-real-time** inference, not a datacenter QPS problem. The constraint is worst-case latency on the densest scene, on the car's compute, every single cycle.
- **Failure cost.** A missed mode (you didn't predict the pedestrian *might* cross) is a safety failure. A spurious mode that makes the planner freeze is a comfort/availability failure. Both matter; they trade off.

State the central tension up front:

> Prediction must be **multimodal and well-calibrated** (never drop the dangerous future) yet **cheap and bounded** (fixed latency on the densest scene), and it must stay accurate on the **long tail** of rare interactions where the average metric tells you nothing.

---

## 2. Frame the ML problem

**Input** (the scene at time *t*):

```text
ego state:        pose, velocity, intended route
agent tracks:     for each agent, history of poses/velocities + type
map:              lanes, crosswalks, stop lines, connectivity (as a graph / polylines)
traffic lights:   current state per controlled lane
```

**Output**: for each relevant agent, a **distribution over future trajectories** — concretely, *K* modes, each a sequence of future positions, each with a probability and per-step uncertainty.

```text
agent -> { (trajectory_1, p_1), ..., (trajectory_K, p_K) }   K ~ 6
```

**Training objective vs. real-world objective** — these differ, and saying so is the staff signal:

- *Training objective:* maximize likelihood of the logged future (or a regression-to-the-closest-mode loss).
- *Real-world objective:* the **planner makes safe, comfortable decisions**. A forecast that is slightly off in displacement but captures the right *intent modes* (will it yield or go?) is far more useful than one with lower displacement error that collapsed to a single averaged mode.

Why a single deterministic trajectory is wrong: futures are genuinely multimodal. At an intersection an agent might turn or go straight. The MSE-optimal answer is the **average** of those — a trajectory that drives *through the median*, into oncoming traffic. Mode-averaging is the canonical failure; the architecture must avoid it.

---

## 3. Metrics

No single aggregate number. Use a layered set.

**Standard accuracy (necessary, not sufficient):**

```text
minADE_K   min over K modes of average displacement error
minFDE_K   min over K modes of final displacement error
miss rate  fraction where no mode lands within d meters of truth
```

`minADE_K` rewards *coverage* — did any of your modes get it right — which is exactly what a multimodal predictor should be graded on.

**What actually matters at staff level:**

- **Mode coverage / recall of the dangerous mode.** Did you produce the "pedestrian crosses" mode at all? Measured on curated interaction slices, not averaged.
- **Calibration.** When you assign 30% to a mode, does it happen ~30% of the time? A miscalibrated planner either freezes (over-cautious) or clips (over-confident). Reliability diagrams per agent type.
- **Closed-loop / planner-relevant metrics.** The real test: feed forecasts into the planner in sim and measure **collisions, hard-brakes, and progress**. Offline displacement error and closed-loop safety can disagree — optimize toward the latter.

**Know the public benchmarks** (so you can anchor numbers and metric definitions): the **Waymo Open Motion Dataset** and its **Sim Agents Challenge (WOSAC)**, **Argoverse 2 Motion Forecasting**, and **nuScenes prediction**. They standardize minADE/minFDE/miss-rate and, for WOSAC, *distributional* realism over joint multi-agent rollouts — a useful pointer that the field has moved past single-agent displacement toward scene-level realism.

**Always slice.** Aggregate ADE is dominated by cars going straight on empty roads. Slice by:

```text
agent type:        ped / cyclist / vehicle
interaction:       yielding, merging, unprotected left, jaywalking
map context:       intersection vs. straight, occlusion present
kinematics:        high accel / sudden stop / U-turn
```

The long tail is where prediction earns its keep. A model that is SOTA on average and bad on unprotected lefts is not shippable.

---

## 4. Fallback when uncertain or wrong

The system must degrade safely, not silently emit a confident wrong forecast.

- **Quantify uncertainty** and pass it to the planner so it widens margins rather than trusting a point estimate.
- **Out-of-distribution / novel agent behavior** (high predicted entropy, low likelihood under the model, or a track the model has never seen the kinematics of): fall back to a **conservative physics-based reachable-set** prediction — "this agent could be anywhere kinematically reachable" — which is over-cautious but safe.
- **Perception is upstream and noisy.** A flickering or just-appeared track has thin history. Default to a wide, conservative forecast until the track stabilizes; do not hallucinate a confident intent from two frames.
- **Never drop a relevant agent silently.** Better to over-predict and let the planner filter than to omit an agent the planner then ignores.

The principle: low confidence widens the safety envelope; it does not produce a confident guess.

---

## 5. Data and labeling

**The labels are nearly free — the future is the label.** Logged drives are self-supervised: roll the log forward and the actual trajectory each agent took *is* the ground truth. This is the cheap, abundant part.

The real work is curation and splitting:

- **Mine the long tail.** 95% of logged miles are cars going straight; they add almost nothing. Build mining pipelines to surface rare interactions (jaywalking, cut-ins, unprotected turns, emergency vehicles) using heuristics, anomaly scores, and **auto-labeling from future outcomes** (e.g., any scene that *led* to a hard brake is interesting).
- **Leakage prevention.** Split by **geography and time**, never by random scene. The same intersection or the same continuous drive in both train and test leaks. Hold out whole cities / date ranges.
- **Label quality = perception quality.** Your "ground truth" agent states come from an *offline, non-causal* perception pipeline (it can look ahead, use the full track) which is far better than the onboard perception you'll see at inference. That mismatch is a real train/serve skew — train with realistic onboard-quality noise injected, or you over-trust clean inputs.
- **Slice-balanced eval sets.** Hand-curate evaluation sets per interaction type so the long tail is *visible* in metrics, not drowned by highway cruising.

---

## 6. Baseline first

Start simple and credible. Two baselines worth stating:

1. **Constant-velocity (CV) / constant-turn-rate-and-acceleration (CTRA) kinematic model.** No learning. Surprisingly strong on the dominant "going straight" slice and a real production fallback. It immediately exposes where you *need* learning: interactions and intent, not straight-line extrapolation.
2. **Lane-following heuristic.** Project each agent along its most likely lane via the map graph. Captures structure the CV model misses.

These set the bar. If a heavy model can't beat lane-following on the unprotected-left slice, the complexity isn't paying for itself. Baselines also become the **safety fallback** from section 4.

---

## 7. Modeling step-up

Move to learning only where interactions and multimodality demand it.

**Representation — vectorized, not rasterized.** Early systems rasterized the scene into a bird's-eye image and ran a CNN. The modern default is **vectorized**: encode agents and map as polylines/tokens and use a transformer (the VectorNet → SceneTransformer → scene-centric transformer lineage). Cheaper, higher fidelity, and it scales with attention.

```text
each agent history  -> token
each map polyline   -> token
traffic light state -> token
        |
        v
transformer with attention over all tokens   (agent<->agent, agent<->map)
        |
        v
per-agent decoder -> K modes (trajectory + probability)
```

**Multimodality — pick a mechanism and defend it:**

- **Anchor / mode queries:** *K* learned query tokens decode *K* trajectories; a classification head scores them. Stable, low-latency, deterministic — the common production choice.
- **Diffusion / generative head:** sample futures from a conditional diffusion or flow model over trajectories (this is the section's through-line — trajectory diffusion conditioned on the scene encoding). *Pro:* rich, continuous, well-calibrated multimodality and natural diversity. *Con:* sampling cost (NFEs × agents) and you must guarantee you sample the rare dangerous mode, not just the common ones. A fast solver (DDIM / DPM-Solver++, low-double-digit NFE) or a distilled few-step student keeps it onboard-affordable.

**Marginal vs. joint vs. ego-conditioned — the interface question that separates senior from staff.** *How* you factor the prediction matters as much as the architecture, and an interviewer will push here because it defines the prediction↔planner contract:

```text
marginal:        p(future_i)              per agent, independently
joint:           p(future_1..N)           one consistent scene-level distribution
ego-conditioned: p(future_i | ego_plan)   reactive: others respond to MY candidate action
```

- **Marginal** (predict each agent independently) is cheap and the common default, but it produces *socially inconsistent* scenes — two agents both predicted to take the same gap, which never co-occurs. The planner then double-counts risk.
- **Joint / scene-centric** predicts a single distribution over all agents' futures so the modes are mutually coherent (if agent A yields, agent B goes). This is what interaction-heavy scenes actually need; the cost is a harder, combinatorial output space and scene-level mode representation.
- **Ego-conditioned (reactive) prediction** conditions other agents on the ego's *candidate* trajectory — "if I nudge into the gap, does this driver brake or accelerate?" This is what lets the planner evaluate its own options instead of treating agents as unresponsive obstacles (which causes the *frozen-robot problem*). The costs are real: a **circular dependency** with planning (you predict to plan, but predict *conditioned on* the plan), a **feedback-loop / self-fulfilling** risk, and you must re-predict per candidate ego plan, multiplying inference cost. Note this is exactly the [closed-loop simulator's](/ml-design/6-closed-loop-simulation/) reactive sim-agent — the same model run as a policy.

The pragmatic staff answer: ship **joint scene-consistent** prediction for the always-on forecast, and add **ego-conditioned reactive prediction for a small set of the planner's candidate maneuvers** where reactivity actually changes the decision — bounded so the per-candidate re-prediction stays in budget.

**The head trade-off to narrate:** anchor-based is fast and deterministic but its diversity is capped at *K* fixed modes; a diffusion head gives genuinely calibrated, diverse futures at the cost of sampling compute and a harder tail-coverage guarantee. Start with anchors; reach for the generative head when calibration and rare-mode coverage are the bottleneck.

---

## 8. Serving and systems (go deep)

This is onboard, hard-real-time. The design is dominated by **worst-case latency on the densest scene**.

**Cost structure:**

```text
total cost = scene encoding (shared, once)
           + sum over relevant agents of decode cost
           + (if diffusion head) NFEs x sampler cost per agent
```

Levers, in order of leverage:

- **Encode the scene once, decode many agents.** The transformer scene encoding is shared across all agents in the frame — amortize it. This is the single biggest win and mirrors the latent-diffusion "encode once, denoise/decode cheaply" principle from elsewhere in this section.
- **Bound the agent count.** Don't predict all 100 agents — gate to the ~10–30 *relevant* to the ego plan (in the route corridor / reachable). Hard-cap for worst-case latency; the planner doesn't care about a car two blocks behind.
- **Fixed compute budget, not best-effort.** Real-time means the *worst case* must fit, so prefer architectures with **deterministic cost** (fixed *K* anchors) or cap NFEs/agents for the diffusion head. Predictable beats occasionally-faster.
- **Few-step sampling** if using a diffusion head: distillation / consistency models / DPM-Solver++ to cut a 50-step sampler to a handful of NFEs. Measure the fidelity (mode-coverage) cost of every step you cut.
- **Quantization (INT8/FP8) and kernel fusion** for the onboard accelerator; batch the per-agent decode on the GPU.
- **Reuse across cycles.** Scenes change little frame to frame at 10 Hz — warm-start / cache encodings where the track set is stable.

**Versioning and train/serve consistency.** The map, perception output format, and feature builder must be *identical* in training and onboard, or you get train-serve skew that silently degrades the model. Pin and co-version them with the model.

---

## 9. Evaluation

- **Offline:** sliced minADE/minFDE/miss-rate + calibration (reliability diagrams) per interaction slice. Track the tail slices as first-class, not the aggregate.
- **Closed-loop in simulation:** plug forecasts into the real planner against logged or reactive sim agents and measure **collisions, hard-brakes, discomfort, progress**. This catches the offline↔online mismatch — a model with better ADE can make the planner *worse* if its modes are jittery or miscalibrated. (This is exactly where the [simulation system](/ml-design/6-closed-loop-simulation/) pays off.)
- **Failure analysis on slices.** Pull the worst closed-loop cases, cluster them, and feed clusters back into mining (section 5). Most real gains come from this loop, not from architecture changes.

---

## 10. Rollout and monitoring

- **Shadow mode first.** Run the new predictor onboard, log its forecasts, but let the *current* model drive. Compare against the shipped model and against the realized future on real miles, at zero risk.
- **Canary** on a small fleet / limited ODD, gated on closed-loop safety metrics, then expand.
- **Monitor drift:** input drift (new city, new agent behaviors, seasonal — e-scooters appear), **calibration drift**, and tail-slice regressions. A model can hold aggregate ADE while quietly degrading on cyclists.
- **Retraining triggers:** a slice regresses, a new ODD launches, or mined long-tail volume crosses a threshold. Auto-labeled logs make retraining cheap; the gate is eval, not training.
- **Rollback** is instant and automatic on any closed-loop safety guardrail breach.

---

## What to say in one breath

> Onboard motion forecasting is a **multimodal, multi-agent, hard-real-time** problem. I frame the output as a *calibrated distribution* over futures because the planner reasons about risk, not a point estimate — and because mode-averaging drives you through the median into oncoming traffic. I'd use a **vectorized scene transformer** encoding agents and map jointly, with either *K* anchor modes (fast, deterministic) or a **trajectory-diffusion head** (richer, calibrated multimodality at sampling cost), predicting **scene-consistent joint** futures plus **ego-conditioned reactive** prediction for the planner's candidate maneuvers (avoiding the frozen-robot problem). Labels are self-supervised from logs, so the work is **long-tail mining and leakage-safe geographic/temporal splits**. Serving amortizes one scene encoding across a *bounded* set of relevant agents under a fixed latency budget. I grade it **sliced and closed-loop** — collisions and hard-brakes through the real planner — not just aggregate ADE, and I ship **shadow → canary → full** with calibration and tail-slice drift monitoring. The spine of the answer is: never drop the dangerous mode, stay calibrated, and stay within a bounded onboard budget on the densest scene.

---

## Further reading

- [VectorNet: Encoding HD Maps and Agent Dynamics](https://arxiv.org/abs/2005.04259)
- [Scene Transformer: A unified architecture for predicting multiple agent trajectories](https://arxiv.org/abs/2106.08417)
- [MultiPath++ for behavior prediction](https://arxiv.org/abs/2111.14973)
- [U-Net and DiT Backbones](/diffusion/6-unet-dit-backbones/) — the denoiser substrate if you use a trajectory-diffusion head.
