---
title: "Design: closed-loop driving simulator"
description: "Staff-level end-to-end ML system design for a learned, closed-loop driving simulator with diffusion sim-agents and scenario generation for testing an AV planner."
---

# End-to-End Design: Closed-Loop Driving Simulator

> **Problem.** Design a **learned simulator** for testing a self-driving planner. Given a logged or generated scene, simulate how the world — other vehicles, pedestrians, cyclists — *reacts* to whatever the AV under test does, so the planner can be evaluated and trained in closed loop across millions of scenarios, including rare safety-critical ones we can't safely collect on-road. Cover scenario generation, reactive agents, training, and the serving/throughput path.

This is the **simulation** specialization design. It is the highest-leverage system in the AV stack: real-world miles are expensive, dangerous to collect for the tail, and you can never re-run the same risky moment twice. A good simulator turns a fixed log into an *interactive* test bench. The staff-level difficulty is the word **closed-loop**: the moment the AV deviates from the logged trajectory, the rest of the world must react plausibly, or your test is meaningless.

Walk the blueprint, but note the consumer here is **the AV development process**, not an end user.

---

## 1. Clarify and scope

```text
logs ----+                                +--> planner evaluation (metrics)
         |--> SCENARIO GEN --> SIM AGENTS  |
maps ----+         ^you are here           +--> planner training (RL / closed-loop)
```

Resolve before designing:

- **Two products, one system.** (a) **Scenario generation** — produce diverse, realistic, and adversarial *initial scenes + intents*. (b) **Reactive sim agents** — given the AV's action each tick, decide how every other agent responds. Both are needed for closed-loop.
- **Open-loop vs. closed-loop.** *Open-loop* replays logged agents on rails — cheap but useless once the AV deviates (log-divergence / "ghost" collisions where a logged car drives through where the AV now is). *Closed-loop* requires reactive agents. The whole design exists for closed-loop.
- **Fidelity target.** We do *not* need photorealistic pixels for planner testing — we need **behaviorally realistic agents** in a structured state (poses, velocities, intents) plus map. (Sensor-realistic *video* sim is a different, heavier problem — the [world-model / e2e](/ml-design/7-e2e-driving/) path — relevant only if testing perception.)
- **Throughput, not latency.** This is **offline, massively parallel, datacenter** simulation. Success = scenarios/GPU-hour, not per-step ms. We want to run *millions* of scenarios per night. This flips every serving trade-off relative to the [onboard predictor](/ml-design/5-motion-forecasting/).
- **Failure cost.** A simulator that is *too easy* ships an unsafe planner (false confidence). One that is *unrealistically adversarial* ("kamikaze" NPCs) makes every planner fail and is ignored. **Realism is the product.**

Central tension:

> Sim agents must be **realistic and reactive** (react to the AV the way real humans would) *and* **diverse / controllable** (cover the long tail, including safety-critical scenarios) — without being **unfairly adversarial**. And it must run at **massive throughput** so the planner team can test against millions of scenarios.

---

## 2. Frame the ML problem

This is, at its core, the [motion-forecasting](/ml-design/5-motion-forecasting/) model **run in reverse as a policy** — instead of *predicting* what agents will do for the planner to avoid, we *roll them out* as the environment.

**Reactive sim agent — framing:**

```text
input  (each tick): scene state including the AV's just-taken action, map, history
output (each tick): next state for every non-ego agent (a policy / next-state distribution)
        |
        v
step the world, feed back, repeat for the horizon   (autoregressive closed loop)
```

**Scenario generation — framing:** sample *initial conditions + agent intents* from a learned generative model, optionally **conditioned on a desired outcome** (e.g., "a cut-in at 30 mph," "an occluded pedestrian"). This is a conditional generation problem — a natural fit for **diffusion over agent trajectories / scene layouts** with controllable conditioning.

**Training vs. real objective:**
- *Training:* maximize likelihood of logged human behavior (imitation).
- *Real:* the simulator must (a) be **statistically indistinguishable** from real logs *in aggregate* and (b) produce **valid, useful tests** — reactive, non-cheating, covering the tail. Pure imitation gives realism but not coverage; you add controllability on top.

---

## 3. Metrics

The hardest part of this problem is that **"is the simulation good?" has no single label.** Use a hierarchy (this mirrors the validation hierarchy from [Diffusion Validation](/diffusion/5-diff-validation/)):

```text
1. distributional realism   do sim statistics match real logs?
2. reactivity / causality    do agents actually respond to the AV?
3. common-sense / physics    no collisions among NPCs, no off-road, kinematically valid
4. coverage / diversity      does it span the long tail, including rare scenarios?
5. downstream validity       does sim performance PREDICT on-road performance?
```

Concretely:

- **Distributional realism:** compare distributions of speed, acceleration, jerk, headway, time-to-collision, minimum gaps between sim rollouts and held-out real logs. ("Looks real" in aggregate.) Jensen–Shannon / Wasserstein distance per statistic.
- **Reactivity:** the key non-obvious metric. Perturb the AV's action and confirm the sim agents' responses *change causally and plausibly* (an agent brakes when the AV cuts in). A simulator whose agents ignore the AV is just open-loop replay in disguise.
- **Common-sense failures:** rate of NPC-NPC collisions, off-road, red-light running per mile — these are *simulator bugs*, not realism.
- **Coverage:** mode diversity and explicit long-tail scenario density.
- **The metric that matters most — downstream predictive validity (sim-to-real correlation):** rank a set of planner versions in sim, rank them on-road, and measure the correlation. **A simulator is only as valuable as its ability to predict reality.** If sim ranking doesn't match road ranking, nothing else matters.

**Public reference points** worth naming: the **Waymo Open Sim Agents Challenge (WOSAC)** operationalizes the distributional-realism + reactivity idea into a concrete metric for learned sim agents, and **nuPlan** is the large-scale **closed-loop** planning benchmark (with reactive vs. non-reactive background-agent modes) — exactly the open-loop-replay-vs-reactive distinction above, made measurable.

---

## 4. Fallback when uncertain or wrong

A sim agent in a state it has never seen (because the AV did something weird) must fail *safely toward realism*, not exploit the planner.

- **OOD AV behavior:** when the AV drives outside the data manifold, sim agents should fall back to a **conservative rule-based reactive model** (IDM car-following + lane-keeping) rather than a hallucinated learned response. A physics fallback is boring but never produces a nonsensical NPC.
- **Flag low-confidence rollouts** and exclude them from pass/fail verdicts — never fail a planner on a scenario the simulator itself couldn't render plausibly.
- **Guard against adversarial collapse:** if scenario generation is trained to *find* failures (useful!), it must stay on the realistic manifold — constrain it so generated scenarios remain something a real human *could* do. An "unavoidable" scenario is a bug in the test, not a planner failure.

---

## 5. Data and labeling

- **Self-supervised from logs**, like forecasting — the logged behavior *is* the imitation target. Cheap and abundant for the common case.
- **The tail is the whole game, and it's missing from logs by definition.** Safety-critical events are rare on-road, so you **can't** just imitate logs to get them. Strategies: (a) **mine** the rare interactions that *do* exist, (b) **generate** them with the controllable scenario model conditioned on outcomes, (c) **perturb** real scenes (shift timing/position) to synthesize near-misses, (d) **adversarial search** in scenario space for planner failures, then filter to realistic ones.
- **Leakage / splitting:** hold out cities and time ranges. Critically, evaluate the simulator on **held-out real logs** so distributional-realism metrics aren't measuring memorization.
- **Auto-labeling:** logged near-misses and hard-brakes are gold — auto-label them from kinematics and use them to seed scenario generation.

---

## 6. Baseline first

Strong, credible non-learned baselines that are *still used in production*:

1. **Log replay (open-loop).** Agents on rails. Zero learning, useful for regression testing as long as the AV tracks the log — and a clear demonstration of *why* you need reactivity (log-divergence collisions).
2. **Rule-based reactive agents:** **IDM** (Intelligent Driver Model) for car-following + a lane-change model + simple pedestrian rules. Genuinely reactive, fully controllable, interpretable, fast. This is the bar — and the safety fallback from section 4.

Rule-based agents are reactive but *robotic* and don't capture the diversity/realism of human behavior (nuanced gap acceptance, hesitation, aggressiveness). That gap — realism and diversity — is exactly what justifies the learned model.

---

## 7. Modeling step-up

**Reactive agents — a learned multi-agent policy / generative model.** The modern approach is a **conditional diffusion model over agent trajectories** (the scene-conditioned trajectory diffusion that recurs throughout this section), or an autoregressive transformer policy, that you roll out closed-loop:

```text
scene encoding (map + all agents + AV action)   <- shares the forecasting encoder
        |
        v
diffusion / AR head -> joint next-step states for all NPCs
        |
        v
step world, re-encode, repeat   (closed-loop autoregressive rollout)
```

Why generative / diffusion:
- **Multimodality & diversity** — sampling different rollouts gives a *distribution* of plausible human reactions, not one robotic response. This is the realism rule-based agents lack.
- **Joint / scene-consistent** modeling so NPCs don't collide with each other or all grab the same gap.
- **Controllability** — the conditioning stack (the lever from [U-Net/DiT backbones](/diffusion/6-unet-dit-backbones/)) lets you steer scenario generation: condition on a desired maneuver, agent aggressiveness, or target outcome. Classifier-free guidance dials how strongly a generated scenario obeys the control.

**Two failure modes to design against:**
- **Closed-loop drift / compounding error.** Each step conditions on the model's own previous output; small errors accumulate and rollouts diverge over a long horizon. Mitigate with **closed-loop training** (train on the model's own rollouts, not just teacher-forced logs — fixing the exposure bias) and history conditioning.
- **Mode collapse losing the tail.** If a fast sampler drops rare modes, the simulator silently stops testing dangerous futures — a *safety* failure, not just a quality one (the diversity/tail-coverage point from [Fast Diffusion Sampling](/diffusion/6-diff-optimization/)).

**Scenario generation** is the same generative machinery used to sample *initial conditions + intents*, conditioned on the scenario type you want to stress-test.

---

## 8. Serving and systems (go deep — throughput edition)

This is **offline, massively parallel**. Everything optimizes **scenarios per GPU-hour**, the opposite of the onboard predictor's worst-case-latency regime.

**Cost structure:**

```text
total = num_scenarios x horizon_steps x per-step rollout cost
per-step rollout cost = NFEs (if diffusion) x backbone cost x num_agents
```

Levers:

- **Batch massively.** Throughput, not latency — run thousands of independent scenarios in parallel on the GPU, large batches, maximize utilization. This is the single biggest lever and is *easy* here precisely because we don't care about per-scenario latency.
- **Cut NFEs hard.** Closed-loop means the denoiser is called *every step of every scenario*, so sampling cost multiplies through the entire rollout × scenario count. **Distillation / consistency models / few-step solvers** (DPM-Solver++, UniPC) are worth far more here than in a single-shot generator. A few-step student is the default for closed-loop sim.
- **Latent / structured state, not pixels.** Roll out in compact agent-state space; only render sensor data if you're testing perception. Keeps per-step cost tiny.
- **Quantization (FP8/INT8)** and a fixed agent budget per scene.
- **Reuse encodings** across the slowly-changing parts of the scene; amortize map encoding across all scenarios on the same map.

The trade-off to *measure*: every NFE you cut for throughput risks dropping a rare reactive mode (section 3, coverage). Throughput is worthless if it buys an unrealistic simulator — gate speedups on the realism/coverage metrics.

**Versioning** the simulator is itself a product: planner teams compare results across simulator versions, so the simulator must be versioned and its realism metrics tracked over time, or you can't tell whether a planner regressed or the sim changed under it.

---

## 9. Evaluation

- **Validate the simulator against real logs** (section 3 hierarchy) — distributional realism + reactivity on held-out data.
- **Sim-to-real correlation study (the crown metric):** take *N* historical planner versions whose on-road performance you know, run them in sim, and check that the sim ranking matches the road ranking. This calibrates *how much to trust* the simulator and is the thing to revisit whenever the simulator changes.
- **Failure analysis:** audit scenarios the planner fails — are they *real* failures or *simulator bugs* (unfair NPCs, impossible setups)? This audit loop is where most simulator credibility is won or lost.

---

## 10. Rollout and monitoring

- **Treat the simulator as infrastructure with gates.** A new simulator version ships only after passing realism + reactivity + **sim-to-real correlation** checks against the incumbent.
- **Shadow:** run new and old simulators on the same planner set; verdicts should only change where the new sim is *demonstrably* more realistic.
- **Monitor drift:** as the real fleet enters new ODDs, the log distribution shifts; the simulator's realism on fresh logs degrades and must be retrained. Track realism metrics on a rolling window of recent real miles.
- **Retraining triggers:** realism drift on new logs, a new ODD/city, or new agent types (e-scooters, new vehicle classes) appearing on-road.
- **Guard against the feedback loop:** if you *train the planner* against the simulator, the planner can learn to exploit simulator quirks ("sim-hacking"). Continuously hold out road validation and refresh the simulator so the planner can't overfit a stationary, gameable environment.

---

## What to say in one breath

> A learned closed-loop simulator is the highest-leverage AV system because real tail miles are scarce and unrepeatable. It's two products: **scenario generation** (controllable conditional generation — a natural diffusion use case) and **reactive sim agents** (the forecasting model run as a *policy*, rolled out closed-loop). The hard part is **closed-loop**: once the AV deviates, every other agent must react plausibly, so I need reactive generative agents — ideally a **scene-consistent trajectory-diffusion policy** — and I must fight **compounding rollout drift** (closed-loop training) and **mode collapse that drops dangerous futures** (a safety failure). Labels are self-supervised from logs, but the **tail is missing by definition**, so I *generate* and *mine* rare scenarios. It runs **offline at massive throughput**, so I batch thousands of scenarios and distill the sampler to few-step — the inverse of an onboard latency budget. I validate with a **realism → reactivity → coverage → sim-to-real-correlation** hierarchy; the simulator is only worth as much as sim ranking predicts road ranking. The spine: realistic, reactive, diverse, non-cheating agents at huge throughput, validated by sim-to-real correlation.

---

## Further reading

- [SimNet / data-driven simulation for AV planning](https://arxiv.org/abs/2105.12332)
- [TrafficSim: Learning to Simulate Realistic Multi-Agent Behaviors](https://arxiv.org/abs/2101.06557)
- [Diffusion Validation](/diffusion/5-diff-validation/) — the validation hierarchy this design leans on.
- [Fast Diffusion Sampling](/diffusion/6-diff-optimization/) — few-step sampling for closed-loop throughput.
