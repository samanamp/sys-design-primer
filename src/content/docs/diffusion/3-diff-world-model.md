---
title: "Diff world model"
description: "Diff world model"
---

"Design a diffusion-based world model for driving: a model that, given the current driving scene — and optionally a proposed AV action — generates plausible future evolutions of the scene. Walk me through the representation, architecture, training, how you'd use it, how you'd make it fast enough, and how you'd know it's realistic enough to trust."

---

# Designing a Diffusion-Based World Model for Driving

> A world model is **action-conditioned dynamics**, not video generation. The deliverable is *"intervene with an action, get the counterfactual future,"* and the hard part is knowing when to trust it.

---

## 1. Research pass — state of the art (2026)

**Driving-specific.** The lineage runs GAIA-1 → **GAIA-2** (Wayve). GAIA-2 is a *latent diffusion world model* with two parts: a learned continuous video tokenizer (≈8× spatial compression into a continuous latent, deliberately moving off GAIA-1's discrete tokens to reduce error propagation over horizon) and a space-time factorized transformer that predicts future latents conditioned on a rich structured set: ego dynamics (speed, curvature, **symlog-normalized**), per-agent 3D bounding boxes projected to the image plane, weather/time-of-day/country metadata, road semantics, *and* external latent embeddings (CLIP, a proprietary driving model). It generates up to **5 temporally and spatially consistent camera streams at 448×960**, trained with flow matching, and supports inpainting / scene editing / long-horizon rollout. **Vista** (OpenDriveLab, NeurIPS'24) is the open analog: SVD-based, 576×1024, with multi-modal action control (command, goal point, trajectory, angle, speed), a *latent-replacement* trick to inject history for coherent long rollouts, and — notably — use as a **reward function** scoring candidate actions. **OccWorld** takes the opposite abstraction bet: a VQ-VAE over 3D semantic occupancy + a GPT-style autoregressive transformer predicting next scene + ego tokens — cheaper, geometric, directly plannable, no appearance.

**Foundation world models.** **NVIDIA Cosmos** is a WFM *platform*: a Cosmos-Tokenizer (≈8× compression, continuous *and* discrete variants) feeding two families — latent **diffusion** (Text2World/Video2World) and **autoregressive** token models — trained on ~20M hours. Cosmos-3 (just released) folds reasoning + generation into one Mixture-of-Transformers (AR for understanding, diffusion for generation). **Genie / Genie-2/3** (DeepMind) learn *latent actions* unsupervised and roll out frame-by-frame, action-controllable. **GameNGen** showed a neural net can be a playable DOOM engine. **Sora** popularized the "video generation = world simulator" thesis — which the driving community treats skeptically: pretty ≠ physically reliable.

**Where the field is moving (2025–26).** (1) **Flow matching** displacing DDPM objectives (GAIA-2, Cosmos). (2) **Few-step distillation** for real-time use: consistency distillation, DMD, and **rCM** (continuous-time CM scaled to 10B+ video models, 2–4 steps). (3) **Streaming / autoregressive video diffusion** with sliding-window attention + **attention sinks** to fight error accumulation (self-forcing vs teacher-forcing), and real-time frame models (World Labs RTFM). (4) Honest reckoning that distilled few-step students **lag teachers on physical consistency** — the exact axis driving cares about.

---

## 2. Problem framing, use case, and the metric problem

**[SIGNAL: scope-and-framing]** A world model `p(s_{t+1:t+H} | s_{≤t}, a_{t:t+H})` learns environment dynamics: given scene history and a **proposed ego action sequence**, sample plausible futures. **[SIGNAL: action-conditioned-dynamics]** The spine of this design is the action argument `a`. Strip it out and you have a video generator that produces gorgeous driving clips no one can plan or test against. Keep it, and you can ask: *if the AV accelerates now, what does the cyclist do? if it yields?* That counterfactual query is the entire point.

**[SIGNAL: use-case-drives-design]** I am designing **primarily for (A) offline controllable scenario generation + closed-loop simulation**, and treating **(B) world-model-in-the-real-time-planning-loop** as a research aspiration I will argue against shipping today (§12). I reject the framing that one model serves both: A tolerates 50-step sampling and prioritizes controllability + diversity; B demands ≤4 steps × many candidate rollouts at 10 Hz. The speed and validation requirements are incompatible, so the design must commit.

**The metric problem [SIGNAL: metric-problem].** A world model has **no single accuracy number** — there is no ground-truth "the future" because (i) the future is multimodal and (ii) under a counterfactual action the real future *never happened*. So I build a metric **hierarchy**, weakest→strongest:

1. **Generation fidelity** — FVD / FID. Necessary, deeply insufficient: matches marginals, says nothing about dynamics or causality.
2. **Action-following / controllability** — does commanded ego trajectory match realized trajectory in the generated rollout (e.g., re-run perception on output, measure trajectory error vs command)? Does forcing "agent cuts in" actually produce a cut-in?
3. **Consistency** — temporal smoothness, **geometric/3D consistency** across cameras, **object permanence** (a car occluded for 1s must reappear, not vanish/teleport), kinematic feasibility.
4. **Distributional realism** — do generated scene *statistics* (speeds, gaps, TTC distributions) match real logs. Matching marginals is not enough; joint/conditional structure is what bites.
5. **Closed-loop realism** — does the model stay realistic *while reacting to novel AV behavior*? This is where it breaks and where it matters.
6. **Downstream-task utility** (the only one that pays rent) — does training/testing on world-model output improve or correctly predict *real* driving? Does a planner ranked good in sim rank good on road?

**Uncertainty/failure definition:** a world model that hallucinates an implausible-but-pretty future, then has a safety decision made on it, is *worse* than no model. So the system must (a) estimate when it is extrapolating off its action support and (b) refuse / flag rather than confidently confabulate (§14).

---

## 3. Scope and capacity / cost math

**[SIGNAL: capacity-and-cost-math]** Why pixel-space is dead on arrival, and what latent buys.

5 cameras × 448×960 × 3 channels × 25 frames (≈2.5 s @ 10 Hz) = **~1.6 billion pixel-values per clip**. Diffusing in pixel space at that token count is infeasible. A tokenizer compressing **8× spatially and ~4–8× temporally** cuts tokens by **~256–512×**:

```
ASCII — cost / compression / sampling budget
┌─────────────────────────┬───────────────┬──────────────┬───────────────────────────┐
│ Quantity                │ Pixel space   │ Latent (8×s, │ Effect                    │
│                         │               │  4×t)        │                           │
├─────────────────────────┼───────────────┼──────────────┼───────────────────────────┤
│ Tokens / 25-frame clip  │ ~1.6e9 vals   │ ~6.3e6       │ ~256× fewer (×5 cams)     │
│ Spatial grid / frame    │ 448×960       │ 56×120       │ 6720 latent tokens/cam/fr │
│ Self-attn cost ∝ N²     │ infeasible    │ tractable    │ enables DiT               │
├─────────────────────────┼───────────────┼──────────────┼───────────────────────────┤
│ Sampling: 50-step DiT   │ 50 fwd passes │ —            │ offline scenario-gen OK   │
│ Sampling: 1000-step DDPM│ 1000 fwd      │ —            │ never; ~20× too slow      │
│ Sampling: 4-step CM      │ 4 fwd         │ —            │ ~12–250× faster → "loop?" │
├─────────────────────────┴───────────────┴──────────────┴───────────────────────────┤
│ Drift: autoregressive H=2.5s OK; degrades sharply by H≈10–20s (error ∝ ~super-linear)│
│ Training scale: 10M–20M hr video class (Cosmos), 10s–100s GPU-months, DiT 1B–14B     │
└──────────────────────────────────────────────────────────────────────────────────────┘
```

Concretely: a 50-step teacher at, say, 30 ms/forward = **1.5 s per 2.5 s clip** — fine offline, fatal for a planner rolling out 20 candidate actions × multiple chunks per 100 ms decision (that would need ~10⁴× headroom). A 4-step distilled student at 30 ms = **120 ms** — borderline for closed-loop *sim agents*, still hard for in-the-loop planning. The arithmetic alone kills naïve "diffusion in the planner."

---

## 4. State representation and abstraction level — the cascade decision

**[SIGNAL: abstraction-level-cascade]** This single choice cascades through compute, validation tractability, and downstream usefulness. Three options:

```
ASCII — abstraction cascade
                generality   cost     validation     planning      appearance
                             (sampl.)  tractability   usefulness    fidelity
Pixel/sensor      HIGHEST     HIGHEST   HARDEST        indirect      full      (GAIA-2, Vista, Cosmos)
video             tests perception+planning together; can't easily check "is this physical?"
─────────────────────────────────────────────────────────────────────────────────────────
Occupancy / BEV   MEDIUM      MEDIUM    MEDIUM         DIRECT        none      (OccWorld)
                  geometric, kinematics checkable, feeds planner; loses texture/semantics-from-pixels
─────────────────────────────────────────────────────────────────────────────────────────
Agent-level /     LOWEST      LOWEST    EASIEST        DIRECT        none      (Waymax-style sim)
symbolic (traj+map)  trivially validate kinematics & permanence; assumes perception solved upstream
```

**My commitment: a two-tier design.** For **scenario generation that must exercise perception** (the full AV stack, sensor-realistic corner cases you can't collect), I use **pixel/multi-camera latent diffusion** (GAIA-2 class). For **closed-loop behavioral sim and any planning experiment**, I use an **occupancy/agent-level** world model. **[SIGNAL: rejected-alternative]** I reject "one pixel model for everything": pixel-space closed-loop sim is too slow for reactive agents, and its physics are nearly unverifiable, so you'd be making safety decisions on the least-checkable representation. I reject "agent-level only" for the perception-stressing use case because it *assumes* the perception problem away — exactly the failure mode (a sensor artifact in fog) you wanted to test.

**Latent choice (pixel tier).** **Continuous** latent (VAE/flow-matching tokenizer) over **discrete** (VQ): GAIA-2 explicitly moved continuous to cut quantization error compounding over rollout, and diffusion is native to continuous latents. Discrete tokens suit an *autoregressive-token* world model (OccWorld, Cosmos-AR, Genie) — a real alternative (§9), not my choice for the pixel tier. **Multi-camera 3D consistency** is enforced by cross-view attention in the tokenizer/DiT and shared geometric conditioning, because per-camera independent generation produces a scene that doesn't agree with itself across the rig.

---

## 5. High-level architecture

**[SIGNAL: diffusion-depth]** Latent video diffusion with a DiT denoiser, structured + history conditioning.

```
ASCII — world-model architecture (pixel tier)
                                          conditioning c
                          ┌───────────────────────────────────────────────┐
                          │ ego action a (speed,curvature; symlog-norm)    │
                          │ agent boxes (3D→2D proj)  weather/time/country  │
                          │ road semantics   external latent (CLIP/driving) │
                          └───────────────────────────────────────────────┘
                                          │ (embed → cross-attn / AdaLN)
   multi-cam video x ──► VIDEO TOKENIZER ─► z₀ (continuous latent, 56×120×T×C, 5 views)
   (history s_{≤t})        (VAE, 8×s/4×t)        │
                                                 ▼
              noise ε ──► z_τ ──►  ┌──────────── DiT denoiser ─────────────┐
                                  │ factorized attention:                  │
                                  │  • spatial (within frame, cross-view)  │
                                  │  • temporal (across frames)            │
                                  │  • cross-attn to c   • AdaLN(τ, a)      │
                                  └────────────────────────────────────────┘
                                                 │  predict velocity / ε (flow matching)
                                  iterate τ: 50 steps (teacher) / 4 (student)
                                                 ▼
                                   ẑ₀ (clean future latent)
                                                 ▼
                                  VIDEO DECODER ─► future multi-cam video x̂_{t+1:t+H}
```

History `s_{≤t}` enters as **conditioning context** (concatenated clean latents the denoiser attends to but does not noise) — Vista's "latent replacement," GAIA-2's context frames — which is the hinge that turns whole-clip diffusion into a *rollable* model (§8). The denoiser is **DiT not U-Net** (§9). Flow-matching velocity objective rather than ε-prediction matches current SOTA and gives straighter ODE paths (cheaper sampling).

---

## 6. Action-conditioning and controllability — the world-model core

**[SIGNAL: action-conditioned-dynamics]** This is what separates a world model from a generator, so it gets the most depth.

The ego action `a` (and other-agent controls) must enter such that **changing `a` changes the future causally**, not cosmetically. Mechanisms, with tradeoffs:

```
ASCII — action-conditioning mechanism
   ego action a_{t..t+H}        agent intents b_{t..t+H}      desired-event tokens
   (speed, curvature)            (per-agent 3D boxes/traj)     (e.g. "cut-in", "jaywalk")
        │ symlog norm                 │ project 3D→2D / token        │
        ▼                             ▼                              ▼
   ┌─────────────── conditioning encoder ───────────────┐
   │  MLP/embed → per-timestep tokens                    │
   └──────────────────────────────────────────────────┘
        │                │                       │
   AdaLN(scale,shift)  cross-attention      ControlNet branch
   on DiT blocks       (denoiser ← c)       (spatially-aligned controls)
        │                │                       │
        └──────► injected at EVERY denoising step & EVERY frame ◄────────┘
                              │
                   classifier-free guidance:
                   x̂ = x̂(∅) + w·(x̂(c) − x̂(∅))   ← w trades realism↔control
```

- **Ego action** via **AdaLN** (FiLM-style scale/shift on DiT blocks) + per-timestep tokens, because ego control is low-dimensional, dense in time, and should modulate the whole field. Symlog normalization (GAIA-2) handles the heavy-tailed speed/curvature range.
- **Other agents** via cross-attention over per-agent box/trajectory tokens — and crucially, in closed-loop sim, *the other agents are themselves driven by the same action-conditioning machinery* so they **react** to the AV rather than replay logs **[SIGNAL: closed-loop-reactivity]**.
- **Spatially-grounded controls** (lane layout, agent placement, weather mask) via a **ControlNet** branch, since these are pixel-aligned.
- **Rare/dangerous events** via dedicated event conditioning + **classifier-free guidance** toward the event.

**[SIGNAL: controllability-realism-tension]** Guidance scale `w` is a realism↔control dial. Push `w` high to *force* a rare jaywalk and the sample drifts off-manifold: physically impossible kinematics, ghosting, agents that pop into existence. So controllability and realism are in direct tension precisely where you most want control (the tails). Mitigation: bounded guidance, autoguidance (guide with a weaker model rather than the unconditional), and **rejecting** generated scenarios that fail the kinematic/permanence checks of §13 rather than shipping a forced-but-broken sample. **[SIGNAL: saying-no]** I push back on the implicit premise that "more controllable = better": a world model you can drive arbitrarily hard is a world model you can drive *off a cliff into nonsense*, and the validation stack must gate it.

---

## 7. The counterfactual-data problem

**[SIGNAL: counterfactual-data-honesty]** This is the deepest learning challenge and most candidates never name it. Training data is logged driving: the AV took **one** action and you observed **one** resulting future. You almost never see the counterfactual ("what if it had floored it?"). Worse than the classic single-observed-future forecasting problem, because here **the action itself is under-sampled**: a good AV drives reasonably, so the logged action distribution is **narrow and safe**. The regime you most need the model to get right — near-misses, hard braking, evasive swerves, the consequences of *reckless* actions — is exactly the regime with **near-zero training support**. Learning correct responses there is **extrapolation**, and a diffusion model extrapolating off its conditioning manifold produces confident, plausible-looking, *wrong* futures.

```
ASCII — action support vs. where you need fidelity
   density of logged ego actions
   ▲
   │           ████████              ← AV drives reasonably: tight, safe
   │         ████████████              cluster of actions
   │       ██████████████████
   │   ·  ░░               ░░  ·     ← evasive / aggressive / near-miss:
   └───────────────────────────────►   sparse data, HIGH safety need
        brake hard   normal   floor it / swerve
        └──── model is EXTRAPOLATING here, fidelity unknown ────┘
```

**Mitigations, honestly bounded:**
- **Human-fleet data** has a far wider action distribution (humans speed, swerve, brake late) — broadens support, but introduces distribution shift from AV behavior and you still rarely see true collisions.
- **Deliberate action perturbation / augmentation**: train on counterfactual action labels by perturbing logged actions, but you have **no ground-truth future** for the perturbed action — so you can only regularize, not supervise.
- **Adversarial / simulator data** (CARLA, hand-authored crashes) to inject the tail — at the cost of sim-to-real gap.
- **Reward-function framing** (Vista): rather than trusting absolute counterfactual futures, use the model to *rank* actions, where relative error is more forgiving.

**[SIGNAL: counterfactual-data-honesty]** The honest staff statement: **counterfactual fidelity degrades off the observed action manifold, and that manifold is narrow by construction.** Therefore the system must *detect* when it is extrapolating (action far from training support, e.g. via a density model / conditioning-distance score) and downweight or refuse decisions made there. This is not a bug to fix; it's a property to instrument.

---

## 8. Temporal modeling and long-horizon rollout / drift

**[SIGNAL: rollout-drift]** Two regimes:

**Full-sequence video diffusion** denoises a whole fixed-length clip jointly → strong temporal consistency *within* the clip, but fixed length and **not naturally closed-loop** (you can't react mid-clip to a new AV action). **Autoregressive / chunked rollout** generates the next chunk conditioned on past chunks → variable length, supports closed-loop reaction — but **drift**: the model conditions on its *own* generated output, errors compound, the trajectory leaves the real-data manifold, agents teleport, object permanence breaks, geometry warps.

```
ASCII — autoregressive rollout & where drift enters
 real history          chunk 1            chunk 2            chunk 3
 s_{≤t} (clean) ─► WM ─► ŝ_{t+1} ──► WM ─► ŝ_{t+2} ──► WM ─► ŝ_{t+3} ─►...
   │ a_t              │ a_{t+1}  ▲          │ a_{t+2} ▲          ▲
   │                  └──────────┘          └─────────┘          │
   condition on        condition on          condition on        DRIFT:
   REAL data           OWN output ε₁          OWN output ε₁+ε₂    ε accumulates
                       (small error)          (compounds)         off-manifold
 error(H): ~flat for H≲2–3s ──► super-linear blow-up by H≈10–20s; permanence fails first
```

**Mitigations (and how they bound drift):**
- **Train with rollout / scheduled sampling**: feed the model its own samples during training so it learns to recover, not just teacher-forced clean history. This is the *single most important* drift fix; teacher-forced-only training guarantees train/serve mismatch.
- **Longer / clean context + attention sinks** (2025 streaming work): retain initial real tokens in the KV cache as anchors so generation doesn't forget the grounded scene.
- **Latent-space rollout** to bound per-step error (continuous latent, GAIA-2's rationale) vs. discrete-token quantization error that compounds.
- **Periodic re-grounding**: in sim, re-inject real perception/occupancy when available; in pure generation, re-condition on the last high-confidence state.
- **Explicit consistency constraints**: geometric/permanence losses, 3D-aware tokenizer.
- **Self-forcing vs teacher-forcing** distillation (rCM direction): on-policy training of the few-step student so the *deployed* student is trained on the distribution it actually sees.

**[SIGNAL: capacity-and-cost-math]** Quantitatively: usable horizon is roughly the clip length the model was trained to be consistent over (≈2.5–5 s); beyond that, expect permanence failures *first*, then geometry, then full divergence by ~10–20 s without re-grounding. Any claim of "minute-long coherent rollout" should be met with: *show me object permanence and a closed-loop reaction at second 45.*

---

## 9. Diffusion mechanics and architecture choice

**[SIGNAL: diffusion-depth]** Defending the choices cold.

**Forward/reverse + objective.** Forward process adds Gaussian noise to the latent over τ; the network learns the reverse (denoise). Modern objective: **flow matching / velocity prediction** (predict the ODE velocity field) rather than ε-prediction — straighter probability-flow paths, fewer sampling steps, used by GAIA-2 and Cosmos. **Conditioning** injected via cross-attention (structured tokens), AdaLN (action/τ), ControlNet (spatial) — §6.

**DiT vs U-Net.** **DiT**: transformers scale predictably with data/compute (the whole point at 10M-hr scale), and **factorized spatial+temporal attention** is the natural way to handle video latents and cross-view consistency. U-Net's convolutional inductive bias is fine for images but doesn't scale or handle long-range temporal/cross-camera dependencies as cleanly. **Rejected: U-Net** for the scalability + multi-view reason.

**Diffusion vs alternatives [SIGNAL: rejected-alternative]:**
- **vs autoregressive discrete-token world model** (Genie, OccWorld, Cosmos-AR): AR-token models are naturally causal/streamable and cheaper per step, but quantization caps fidelity and compounds error, and **next-token-argmax tends to mode-collapse the future**. The future is genuinely **multimodal** (the pedestrian crosses *or* waits) and **diffusion samples that distribution natively** — its strongest single argument here. I *do* adopt AR-token for the cheap occupancy tier (OccWorld-style) where interpretability + speed win.
- **vs GANs** (DriveGAN): faster sampling, but unstable training and **mode collapse** — fatal for a model whose value *is* covering the multimodal future and the tails. Rejected.

So: **latent diffusion + DiT + flow matching for the pixel tier; AR-token for the occupancy tier.** The choice is use-case-driven, not dogmatic.

---

## 10. Sampling speed

**[SIGNAL: sampling-speed]** Vanilla diffusion = tens-to-1000 sequential denoising steps. The toolkit:

```
ASCII — speed / quality / diversity tradeoff
  steps   method            quality   DIVERSITY   latency   use case
  ─────   ───────────────   ───────   ─────────   ───────   ─────────────────────────
  1000    DDPM              ★★★★★     ★★★★★       worst     never (research only)
   50     DDIM              ★★★★★     ★★★★★       offline   scenario generation ✅
   8–20   DDIM + low CFG    ★★★★      ★★★★        medium    batch sim
   2–4    consistency/rCM   ★★★☆      ★★☆ ⚠       fast      closed-loop sim agents
   1      one-step CM/DMD   ★★☆       ★ ⚠⚠        fastest   real-time; risky for WM
   ────────────────────────────────────────────────────────────────────────────────
  ⚠ KEY DANGER: fewer steps + high CFG COLLAPSES DIVERSITY.
    A world model's value IS multimodal futures → over-distilling defeats the purpose.
```

Levers: **latent operation** (already 256×), **DDIM** (deterministic few-step), **consistency models / consistency distillation** and **DMD/rCM** (distill a 50-step teacher into a 2–4-step student). **The world-model-specific trap [SIGNAL: sampling-speed]:** image-gen distillation literature optimizes for fidelity, and aggressive distillation + high guidance *reduces sample diversity*. For a world model whose entire value is sampling the **multimodal** future and covering tails, a low-diversity 1-step student is a regression even if FVD looks fine. So **per use case**: offline scenario-gen keeps the 50-step teacher (quality + diversity, latency irrelevant); closed-loop sim agents use a **2–4-step student tuned to preserve diversity** (rCM's forward+reverse-divergence joint distillation exists precisely to keep diversity); the planner loop (§12) needs ≤4 steps × many rollouts and *still* doesn't close the budget today.

---

## 11. Use case — scenario generation + closed-loop sim

```
ASCII — scenario-gen / closed-loop-sim data flow
 OFFLINE SCENARIO GEN                          CLOSED-LOOP SIM
 ┌────────────────────────┐                    ┌──────────────────────────────────┐
 │ event spec / long-tail │                    │  AV stack under test (real planner)│
 │ "rainy cut-in @ merge" │                    │            │ ego action a_t          │
 └──────────┬─────────────┘                    │            ▼                        │
   conditioning c (CFG→event)                  │   WORLD MODEL (occupancy/agent tier)│
            ▼                                   │   reacts: other agents conditioned  │
   50-step teacher diffusion                    │   on AV action → NEW scene s_{t+1}  │
            ▼                                   │            │                        │
   diverse scenario clips ──► VALIDATE (§13) ──►│ feed s_{t+1} back to AV stack ◄─────┘
            │ pass                              │   (CLOSED LOOP, 2–4-step student)
            ▼                                   └──────────────────────────────────┘
   add to test/train suite                       agents must REACT, not replay logs
```

**Scenario generation** synthesizes the **long-tail you can't collect** — rare/dangerous corner cases — with controllability + diversity, runs slow/high-quality offline, and every output passes the validation gate before entering a suite. **Closed-loop sim** is where the action-conditioning machinery is **turned on the *other* agents** **[SIGNAL: closed-loop-reactivity]**: a log-replay sim is useless because replayed agents don't respond to the AV doing something new — the moment the AV deviates, the replay is counterfactually wrong. Reactive sim agents (same conditioned-dynamics model) close the loop. This ties directly to Waymax-style closed-loop evaluation; the world model is the reactive substrate.

---

## 12. Use case — world model for planning (judgment)

**[SIGNAL: planning-use-judgment]** The ambitious vision: model-based planning — imagine N candidate AV actions, roll the world model forward, evaluate outcomes, pick the safest. The appeal is real (data-driven, handles novel situations, no hand-coded cost model). My honest staff judgment: **not in the real-time on-vehicle loop today**, for three compounding reasons:

1. **Latency.** ≤10 Hz decisions × N candidate actions × multi-step horizon × even a 4-step diffusion student blows the budget by orders of magnitude (§3). On-vehicle diffusion rollout is infeasible now.
2. **Drift.** You'd be choosing actions by ranking *imagined* futures that diverge from reality within seconds (§8) — and the ranking is most fragile in the aggressive-maneuver tail (§7) where planning matters most.
3. **Validation.** Acting on imagined futures means a safety decision rests on a generative model whose counterfactual fidelity is, by construction, unverifiable in the regime that counts.

**Where it *does* belong now:** as a **training/testing tool** — a reward/critic for offline policy evaluation (Vista's reward-function framing), a scenario engine, a data augmenter, a sim substrate. **[SIGNAL: saying-no]** I push back on "world-model-in-the-loop" as a 2026 production claim. The defensible path is world-model-for-eval feeding a fast, verifiable on-vehicle planner — keep the unverifiable generative component *out* of the real-time safety path.

---

## 13. Realism validation — the safety-critical heart

**[SIGNAL: realism-validation-as-heart]** "Realistic-looking ≠ safe to decide on." Building the model is easier than knowing when to trust it. A multi-layered stack, each layer catching what the layer below misses:

```
ASCII — realism-validation stack (weakest → decision-grade)
 ┌───────────────────────────────────────────────────────────────┐
 │ L5  DOWNSTREAM UTILITY: does sim verdict predict ROAD outcome?  │  ← decision-grade
 │     sim-to-real correlation; train-on-sim → real perf delta    │
 ├───────────────────────────────────────────────────────────────┤
 │ L4  CLOSED-LOOP REALISM: stays realistic while reacting to      │  ← breaks here,
 │     NOVEL AV behavior (the regime it's most needed & most fails)│     needed most
 ├───────────────────────────────────────────────────────────────┤
 │ L3  GEOMETRIC/PHYSICAL: object permanence, kinematic feasibility│
 │     no teleporting agents, cross-view 3D consistency            │
 ├───────────────────────────────────────────────────────────────┤
 │ L2  DISTRIBUTIONAL: speeds/gaps/TTC match real (JOINT not just  │
 │     marginal — matching marginals is insufficient)              │
 ├───────────────────────────────────────────────────────────────┤
 │ L1  GENERATION FIDELITY: FVD/FID (necessary, weakest, gameable) │  ← weakest
 └───────────────────────────────────────────────────────────────┘
```

- **L1 FVD/FID**: necessary, deeply insufficient and **gameable** — a model can match clip statistics while being causally nonsense.
- **L2 Distributional**: match *joint/conditional* statistics (TTC given gap, agent response given ego brake), not just marginals; marginal-matching is the classic trap.
- **L3 Geometric/physical**: programmatic checks — object permanence (track IDs persist through occlusion), kinematic feasibility (no superhuman accel), 3D/cross-view agreement. These catch the "pretty but impossible" samples that L1 passes.
- **L4 Closed-loop realism**: the heart — drive the AV to do something *out-of-log* and check the model stays sane. This is where it's most likely to break (off action support, §7) and most consequential.
- **L5 Downstream/sim-to-real**: the only verdict that matters — do metrics computed in the world model **correlate with real-world outcomes**? Curate a held-out set of *real* events; check the model's predictions/rankings match what actually happened.

**[SIGNAL: metric-problem]** The **counterfactual-evaluation problem**: for a counterfactual rollout you *cannot* compare to ground truth because the counterfactual never happened. Partial answers: (a) validate on the *observed* action where ground truth exists and treat that as an upper bound on counterfactual trust; (b) detect extrapolation (action-support density) and *refuse to certify* off-support; (c) sim-to-real **rank correlation** on real events rather than per-sample accuracy; (d) human expert review for forced rare events. There is no clean number — the staff move is to be explicit that off-support counterfactuals are *uncertified by construction* and gate decisions accordingly.

---

## 14. Data, training, serving, monitoring, safety wrapper

**Data & training.** Logged driving is **self-supervised** — the observed future is the label; action labels come from logs (ego CAN bus, perceived agent boxes). Pipeline: (1) **pre-train the tokenizer/VAE first** (freeze the latent space), (2) train the latent diffusion world model with **flow matching + rollout/scheduled-sampling** (not teacher-forced-only), (3) **distill** few-step students for sim. Curate for **geographic/weather/scenario diversity** and **long-tail mining** (the tail is where value and risk concentrate); blend AV-fleet + human-fleet data to widen action support (§7). Maintain **train/serve abstraction consistency** — the occupancy the planner sees in sim must match what perception produces on road.

**Serving.** Offline **batch** for scenario gen (50-step teacher, GPU farm). A **sim service** for closed-loop (few-step students, reactive agents). **On-vehicle in-the-loop = infeasible today** — say so (§12).

**Monitoring & safety wrapper.** Monitor generation quality drift (rolling FVD, permanence-failure rate), and — most important — an **extrapolation detector**: score conditioning-distance / action-support density per request; when the model is asked to roll out off-support, **flag, downweight, or refuse** rather than confabulate. **Governance**: any safety/rollout decision made on generated evidence carries an explicit trust level tied to which validation layers (L1–L5) it cleared and whether it was on- or off-support. **Fallback** when the model is out of its depth: defer to real-data scenarios / hand-authored tests / conservative human review — never let an uncertified generated future silently drive a safety decision.

---

## 15. Tradeoffs taken and what would change them

I took **two abstraction tiers** (pixel for perception-stressing scenario gen; occupancy/agent for closed-loop + planning) over one unified model — accepting two systems to keep each verifiable and fast enough. I took **continuous-latent diffusion + DiT + flow matching** over AR-token/GAN for the pixel tier (multimodal sampling, fidelity), and **AR-token** for the cheap tier. I kept the **50-step teacher offline** and **distilled students** only for sim, refusing to over-distill at diversity's expense. **What would change this:** a distillation method that provably preserves multimodal diversity at 1–2 steps *and* an extrapolation detector good enough to certify off-support counterfactuals would reopen the in-the-loop-planning question. Cheaper verifiable physics (differentiable kinematic priors baked into the tokenizer) would push me toward a single higher-abstraction model.

## 16. What I would push back on

**[SIGNAL: saying-no]** Three premises I'd challenge before accepting the brief: (1) **"Generate realistic driving video = world model."** No — without causal action-conditioning it's a generator; I'd reframe the project around the counterfactual query. (2) **"Use it in the real-time planning loop."** Not in 2026 — latency + drift + unverifiable counterfactuals keep the generative component out of the safety path; it belongs in eval/training. (3) **"FVD/looks-realistic is the bar."** No — realistic-looking is L1 of five layers; the bar is closed-loop realism and sim-to-real downstream correlation, and off-action-support counterfactuals are **uncertified by construction**. If I caught myself writing something that would dazzle at a video-gen demo but couldn't answer "what does the cyclist do if the AV brakes, and how do you know that's right?" — I'd delete it.