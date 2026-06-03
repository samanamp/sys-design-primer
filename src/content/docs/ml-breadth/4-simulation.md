---
title: Simulation
description: Simulation
---

# Generative Scenario Generation & Closed-Loop Simulation for AV Validation

*ML System Design — Diffusion, World Models, and the Safety Case. Target: Waymo / Wayve / NVIDIA / Nuro-class AV org, 2026.*

---

## 1. Research pass — state of the art as of 2026 (~350 words)

The field has consolidated around a hard distinction the demo answer misses: **what abstraction you generate at depends on what you test.** Behavior/planning testing runs at the *agent level* — bounding boxes, maps, intents — in fast, differentiable simulators (**Waymax**, JAX, Waymo Open Motion Dataset; **nuPlan**; CARLA for sensor sim). Perception testing needs *sensor-realistic* generation: **GAIA-2** (Wayve, 2025) and **NVIDIA Cosmos** produce multi-camera video conditioned on ego dynamics, agent boxes, weather, and road semantics; **Vista**, **DriveDreamer**, and **OccWorld** (occupancy-level) sit between. Sensor sim is ~2–3 orders of magnitude more expensive and far harder to validate, so it is reserved for perception, not yielding decisions.

Controllable agent-level generation is dominated by **guided diffusion**: **CTG** uses Signal Temporal Logic as differentiable guidance; **CTG++** adds an LLM that compiles a language query into a guidance loss; **LCTGen** is language-conditioned; **RealGen** is retrieval-augmented; **SLEDGE** generates lane graphs + agents. **MotionDiffuser** supplies differentiable cost guidance.

The adversarial line is where the *realism-vs-criticality* tension lives. **AdvSim** perturbs trajectories physically-plausibly and re-renders LiDAR for full-stack testing; **STRIVE** (NVIDIA, graph-CVAE) finds plausible planner-breaking scenarios; **KING** uses kinematic gradients. The 2024–2026 work explicitly attacks the "unrealistic gotcha" failure mode: **FREA** (2024) bounds adversariality with a *feasibility* upper bound to avoid unavoidable collisions; **RCG** (2025) grounds criticality in real crash embeddings; **AuthSim** (2025) enforces collision *responsibility* on the NPC; **AdvDiffuser** uses a realism-preserving diffusion prior; **SCENGE** (2025) uses an LLM grounded in traffic law + accident records to keep adversaries plausible. **LD-Scene** (2025) is LLM-guided diffusion for *closed-loop reactive* adversarial scenarios.

Closed-loop realism is benchmarked by the **Waymo Open Sim Agents Challenge (WOSAC)**. 2025 top entries (TrajTok+SMART ≈0.785 realism meta; UniMM) use tokenized next-token behavior models; **SMART-R1** adds RL fine-tuning. Critically, 2025 work (Schofield et al.) shows realism scores alone don't predict closed-loop robustness, adding *causal-agent* and *delta/confusion* metrics. **nuPlan-R** (2025) replaces IDM with diffusion reactive agents.

On the safety side: **Kalra & Paddock (2016)** showed naturalistic testing needs *billions* of miles to demonstrate reliability statistically. **Importance sampling** (Zhao et al.; Deep IS; TrimFlow normalizing-flow IS) gives 2,000–20,000× acceleration and *unbiased reweighted* risk estimates. **ISO 21448 (SOTIF)** frames simulation as evidence in a scenario-based safety argument, not proof.

---

## 2. Problem framing, objective, and scenario-system metrics (~400 words)

**[SIGNAL: scope-and-framing] [SIGNAL: failure-finding-objective]**

The goal is **not generating realistic traffic.** The goal is **finding AV failures before the road and producing evidence for a safety case.** Scenario generation is in service of validation. This reframe drives every downstream decision.

The corollary kills the naïve design: **random sampling from the data distribution is near-useless for testing.** Highway cruising dominates the distribution and the AV already handles it; sampling it just re-confirms the easy 99%. The entire value is in the scenarios that *probe the AV's decision boundary* — the ones it fails or barely passes. A system optimized for generation fidelity (low FID, high WOSAC realism) rather than *test utility* has optimized the wrong objective.

So I separate two metric families and **never conflate them**:

- **AV metrics** (the thing under test): collision, near-miss / min-TTC, hard-brake count, rule violations, ride comfort, route progress.
- **Scenario-system metrics** (the thing I am *building*), in priority order:
  1. **Criticality / difficulty** — does the scenario actually stress the AV? Measured by induced min-TTC, required deceleration, achieved failure/near-failure rate.
  2. **Realism / plausibility** — could this happen on a real road? Kinematic feasibility, world-model likelihood, learned-discriminator score, responsibility assignment.
  3. **Coverage / diversity** — does the suite span the ODD's scenario space, not just one easy mode?
  4. **Downstream utility** — *the validator metric:* do sim failures correspond to real disengagements/incidents, and does fixing a sim failure improve road performance? This closes the loop and is the only metric that proves the system works.

"More scenarios" is explicitly **not** a goal. A million near-duplicate highway clips is worse than a thousand diverse, critical, plausible ones — it costs more compute and inflates false confidence.

**What am I testing?** I commit up front: **planning/behavior** (does the AV make safe *decisions*?) at the **agent level**, with a **secondary perception track** at sensor level. The bulk of long-tail behavior failures (yielding, cut-ins, occluded pedestrians, intersection negotiation) are decision failures and are cheaply, controllably testable at the agent level — so that is where I concentrate.

**Mode of use:** three distinct products from one pipeline — (a) a frozen **regression suite** (does a code change re-break a known failure?), (b) **active failure-search** (find *new* failures near the current boundary), and (c) **risk estimation** (turn found failures into a calibrated probability number). They have different generation, sampling, and statistics; I design for all three.

---

## 3. Scope, abstraction-by-test-target, and capacity/cost math (~350 words)

**[SIGNAL: abstraction-by-test-target] [SIGNAL: capacity-and-cost-math]**

The single biggest waste in this space is generating photoreal video to test a yielding decision. Don't.

```
TEST TARGET            ABSTRACTION              SIM            COST/SCENARIO   VALIDATABLE?
-------------------    ----------------------   ------------   -------------   ------------
Planning / behavior    Agent-level (boxes,      Waymax/nuPlan  ~0.01-0.1 GPU-s  YES (kinematics,
  (yield, cut-in,        map, intents,                                          WOSAC metrics,
   intersection)         trajectories)                                          STL checkable)
Perception (detect,    Sensor-realistic         GAIA-2 /       ~10-100 GPU-s   HARD (validate
   classify, track)      (multi-cam video,       Cosmos / NeRF                  the renderer
                         LiDAR)                                                 itself)
Occupancy / fusion     Occupancy grid           OccWorld       ~1-10 GPU-s     PARTIAL
Full stack (rare)      Sensor + closed-loop     AdvSim-style   ~100+ GPU-s     HARDEST
```

**Coverage scale.** A single metropolitan ODD decomposes into roughly: ~10 road-geometry classes × ~8 maneuver types × ~6 agent-density bands × ~5 weather/lighting × ~4 occlusion regimes × continuous timing/speed/intent parameters. The discrete cross-product alone is ~10⁴–10⁵ *cells*; with ~10²–10³ samples per cell to characterize each, you need **10⁶–10⁸ scenario-evaluations** for a meaningful ODD sweep. This is the combinatorial explosion that forces prioritization (§13).

**Compute arithmetic.** Closed-loop cost per scenario ≈ (AV-stack inference per step) × (steps) + (reactive-agent generation per step). Agent-level: AV planner ~5–20 ms/step × ~90 steps (9 s @10 Hz) ≈ 0.5–2 GPU-s, plus reactive agents ~0.5 GPU-s ⇒ ~1–3 GPU-s/scenario. At 10⁷ scenarios ⇒ ~10⁷ GPU-s ≈ **2,800 GPU-hours ≈ ~120 GPU-days** per full sweep — a few hours on a 1k-GPU cluster, runnable nightly. Sensor-realistic at ~50 GPU-s/scenario blows this to ~140k GPU-hours per 10⁷ — *prohibitive*, which is precisely why perception sim is targeted, not swept, and why generator **sampling speed** (diffusion distillation / consistency models) is on the critical path.

---

## 4. High-level pipeline (~400 words)

**[SIGNAL: pipeline-and-triage]**

```
                       ┌───────────────────────────────────────────────┐
   FLEET LOGS ───────► │  (A) MINE: cluster logs, surface critical/      │
   (WOMD + fleet)      │      interesting seeds (low TTC, hard brakes,    │
                       │      disengagements, rare maneuvers)             │
                       └───────────────┬─────────────────────────────────┘
                                       │ real seed scenarios
                       ┌───────────────▼─────────────────────────────────┐
   SCENARIO SPEC ────► │  (B) GENERATE: diffusion world model            │
   (intents, events,   │   • perturb seeds (timing, count, intensity)     │
    language, target    │   • from-scratch guided gen toward criticality  │
    criticality)        │   • guidance strength CAPPED (§5)                │
                       └───────────────┬─────────────────────────────────┘
                                       │ candidate scenarios
                       ┌───────────────▼─────────────────────────────────┐
                       │  (C) VALIDITY/REALISM FILTER  ◄── HARD GATE       │
                       │   kinematic feasibility | physics | map-legal     │
                       │   world-model likelihood | realism discriminator  │
                       │   responsibility check (is ego at fault avoidable?)│
                       └───────────────┬─────────────────────────────────┘
                          REJECT ◄─────┤ pass
                       ┌───────────────▼─────────────────────────────────┐
                       │  (D) CLOSED-LOOP SIM  (full AV stack under test)  │
                       │   AV action ⇄ REACTIVE sim agents (world model    │
                       │   applied to NPCs) — multi-agent consistent       │
                       └───────────────┬─────────────────────────────────┘
                                       │ rollouts
                       ┌───────────────▼─────────────────────────────────┐
                       │  (E) OUTCOME SCORING: collision / min-TTC /       │
                       │   hard-brake / rule-violation / comfort           │
                       └───────────────┬─────────────────────────────────┘
                       ┌───────────────▼─────────────────────────────────┐
                       │  (F) TRIAGE: cluster, dedupe, root-cause,         │
                       │   SEPARATE real AV bug from sim artifact (§12)     │
                       └───────┬───────────────────────────┬──────────────┘
                               │ real bugs → AV dev          │ active-search feedback:
                               ▼                             ▼ generate MORE near
                       ┌─────────────────┐          discovered failure boundaries
                       │ RISK ESTIMATE   │          (importance-sampling, §9)
                       │ (IS reweight,§9)│
                       └─────────────────┘
```

The pipeline is an **ML/validation system**, not a one-shot generator. Two feedback edges matter: failures route to AV development (fix the car) **and** back into generation as an active-learning signal (densely sample the neighborhood of each discovered failure to map the boundary). The validity filter is a **hard gate**, not a soft preference — an adversarial scenario that fails the realism check is *deleted*, never run. The triage stage's job of distinguishing **real bugs from sim artifacts** is what keeps engineer trust in the suite alive.

---

## 5. Realism-vs-criticality — the core tension (~500 words)

**[SIGNAL: realism-vs-criticality] [SIGNAL: saying-no]**

This is the spine of the whole system. State it plainly: **the most useful test scenarios are the ones the AV fails — so I want to guide generation toward failure. But it is trivial to generate failures the AV could never avoid, and those are worse than useless.** A pedestrian teleporting under the wheels, a car materializing at 200 km/h into the ego's lane, an NPC rear-ending the ego from behind while ego is stationary — every AV "fails" these, none represents real-world risk, and chasing them does three concrete harms:

1. **Wasted engineering** — analysts burn time triaging non-bugs.
2. **Pathological over-conservatism** — if you adversarially train or tune the AV against impossible attacks, it learns to freeze/brake for phantom threats, *degrading real road behavior* (the classic phantom-braking failure).
3. **Eroded trust** — once engineers see junk failures, they stop believing the suite, and the safety case collapses socially.

So I push back on the naïve adversarial framing directly: **adversarial generation without a hard realism/plausibility constraint is not a useful tool — it is a liability.** The objective is not "maximize AV failure"; it is **scenarios on the boundary of plausible-AND-hard.**

```
            CRITICALITY (how hard for AV) ──►
        low                                   high
  high ┌─────────────────┬─────────────────────────────┐
   │   │  EASY & REAL    │   ★ TARGET ZONE ★            │
 R │   │  (regenerates   │   plausible AND hard —        │
 E │   │  the easy 99%,  │   real failures that          │
 A │   │  low value)     │   could actually happen       │
 L │   ├─────────────────┼─────────────────────────────┤
 I │   │  irrelevant     │  ⚠ UNREALISTIC GOTCHAS ⚠      │
 S │   │                 │  teleporting peds, impossible │
 M │   │                 │  closing speeds — DELETE,     │
   ▼   │                 │  drives over-conservatism     │
  low  └─────────────────┴─────────────────────────────┘
                 PLAUSIBILITY CONSTRAINT (hard gate) sits HERE ──┘
        Guidance pushes RIGHT; the constraint blocks the bottom-right.
```

**Mechanisms to keep adversarial scenarios plausible** (defense in depth — no single one suffices):

- **Kinematic / physics feasibility:** every agent trajectory must respect a bicycle/kinematic model — bounded accel, jerk, steering, friction. KING-style and AdvSim-style perturbations operate *within* these bounds by construction.
- **On-manifold constraint:** restrict perturbations to the manifold of real behavior. Mine a real seed (§10) and perturb *locally*; or constrain generation to high-likelihood regions of the world model's own density. AdvDiffuser does this with a diffusion prior; RealGen via retrieval.
- **Learned realism discriminator:** a critic trained to distinguish real fleet logs from generated scenarios; score below threshold ⇒ reject. (This is itself a validator that must be validated — §11.)
- **Bounded adversariality (FREA):** cap adversariality at the point where collisions become *unavoidable for the ego*. An attack the ego cannot possibly avoid tests nothing about the ego's competence.
- **Responsibility / fault assignment (AuthSim, SCENGE):** require that a competent driver *could* have avoided the outcome, and that the NPC's behavior is itself traffic-legal or at least human-plausible. Ground the adversary in real accident taxonomies and traffic law.
- **Guidance-strength limits:** in diffusion, criticality guidance trades off against the data prior. Too-strong guidance drags samples off-manifold into garbage. I cap guidance weight and *monitor* realism-discriminator score as a function of guidance strength, backing off when realism degrades.

The target is the boundary, not the corner.

---

## 6. Controllable generation (~450 words)

**[SIGNAL: controllability-mechanics] [SIGNAL: rejected-alternative]**

I need to generate *specific* long-tail events on demand — jaywalk, occluded-pedestrian step-out, aggressive cut-in, run-red-light, lead-vehicle hard-brake, double-parked-car occlusion — not just sample and hope. Three control surfaces:

**1. Structured specification.** A scenario = initial conditions (map cell, ego start, agent placement) + per-agent intents + scheduled events ("agent_3 begins lane change toward ego lane at t=2.0s"). This is the most reliable and most validatable surface; it's how the regression suite is parameterized.

**2. Language conditioning (LCTGen / CTG++ style).** An LLM compiles "an occluded pedestrian steps out from behind a parked van as the ego approaches a crosswalk at 25 mph" into either a structured spec or a differentiable **guidance loss**. This scales authoring throughput enormously and lets domain experts write scenarios in English. Caveat from CTG++'s own ablations: LLM-generated guidance code fails to compile/has wrong semantics a non-trivial fraction of the time, so I wrap it with auto-validation (run it, check the scenario satisfies the intent, repair via feedback loop) — never trust raw LLM guidance code unverified.

**3. Diffusion guidance mechanics.** The world model is a conditional diffusion model over agent trajectories. Control comes via:
- **Classifier-free guidance (CFG):** train with conditioning dropout, sample with `ε = ε(∅) + w·(ε(c) − ε(∅))`. `w` controls how hard generation adheres to the condition.
- **Classifier / cost guidance (CTG, MotionDiffuser):** at each denoising step, nudge the sample by `−∇ₓ J(x)` where `J` encodes the target — STL rule satisfaction, a criticality objective (minimize ego TTC), or a target event. This is how I steer toward the *critical tail* specifically.

**The control-vs-realism degradation is the key mechanic to surface:** as guidance weight `w` (or cost-guidance scale) increases, condition-adherence rises but sample likelihood under the data prior falls — trajectories get jerky, off-manifold, eventually physically impossible. I treat `w` as a tuned, *capped* knob and monitor realism-discriminator score against it (ties directly to §5).

**Rejected alternatives:**
- *Pure distribution sampling* — rejected: regenerates the easy head, can't hit named long-tail events. (§2)
- *Hand-scripted scenarios only* — rejected: doesn't scale to 10⁶–10⁸, brittle, encodes only what engineers already imagined; misses the *unknown* tail.
- *"Generate broadly then filter for critical"* — rejected as the *primary* strategy: critical events are ~10⁻³–10⁻⁶ rare, so broad-then-filter wastes >99.9% of generation compute. **Guide generation toward criticality** instead, then reweight (§9). Broad-then-filter survives only as a cheap diversity supplement.

---

## 7. Closed-loop reactive sim agents (~450 words)

**[SIGNAL: closed-loop-reactivity] [SIGNAL: saying-no]**

**I will not use log-replay for behavior testing, and I'll push back hard if asked to.** Open-loop replay — fix the logged scenario, let the AV react, but keep other agents on their *recorded* trajectories — is fundamentally invalid as a behavior test. The instant the AV does anything different from the logged ego (and it will, that's the point of testing it), the replayed agents become nonsensical: they drive *through* where the AV now is, or fail to yield to a maneuver that demanded yielding. Waymax's own docs show log-playback agents driving straight through a stopped vehicle. A "pass" against non-reactive agents proves nothing.

**Valid behavior testing requires reactive sim agents** — and a reactive sim agent is exactly **the action-conditioned world model applied to the *other* agents.** They observe the AV's actual realized state each step and respond plausibly to whatever it does.

```
   ┌──────────────┐   ego state, scene    ┌─────────────────────┐
   │   AV STACK   │ ────────────────────► │  REACTIVE SIM AGENTS │
   │ (under test) │                       │  (world model on     │
   │ perception → │ ◄──────────────────── │   NPCs: vehicles,    │
   │ pred → plan  │   NPC states (react    │   peds, cyclists)    │
   └──────┬───────┘    to AV's action)     └─────────┬───────────┘
          │                                          │
          │   ego action a_t                         │ all NPCs react to
          ▼                                          │ ego AND to each other
   ┌───────────────────────── SIM STEP ──────────────▼────────────┐
   │  advance physics; t←t+1; multi-agent JOINT consistency        │
   └───────────────────────────────────────────────────────────────┘
        ▲                                                    │
        └──────────────── closed loop @ 10 Hz ───────────────┘
```

**Reactivity realism is itself a calibration problem with safety consequences:**
- **Too passive** (e.g., naive IDM that always yields) → the AV *looks safer than it is*; it never gets tested under contested right-of-way. False confidence.
- **Too aggressive** → the AV *looks worse than it is*; you over-conservatize fixing failures that no reasonable agent would create.

So I calibrate reactive-agent aggressiveness to *human distributions* and report AV metrics across a *spectrum* of agent assertiveness, not a single point. Learned agents (WOSAC-class: SMART/TrajTok tokenized models, nuPlan-R's diffusion reactive agents) beat IDM on behavioral diversity and human-likeness; I use learned agents for realism and keep IDM/rule agents as a cheap, interpretable, long-horizon-stable baseline (per the 2025 Waymo2SUMO finding that rule models are more stable at 60 s horizons).

**Multi-agent consistency:** all NPCs must react coherently to the AV *and to each other* — a scene-level (joint) model, not per-agent independent sampling. CTG++'s whole motivation was that independent per-agent diffusion produces colliding, incoherent scenes. And I track the 2025 WOSAC finding (Schofield et al.): aggregate realism scores don't predict closed-loop robustness, so I add **causal-agent metrics** (only score agents that can actually affect the ego) and monitor compounding autoregressive drift.

---

## 8. Coverage, ODD, and scenario taxonomy (~400 words)

**[SIGNAL: coverage-and-ODD]**

If the AV passes every scenario I generate, that's worthless unless I can say *what space those scenarios cover* and *bound what they don't.* Coverage is half the load-bearing assumption of the safety claim (§11).

I decompose the **ODD into a structured scenario taxonomy** — a parameter space:

```
ODD COVERAGE VIEW (each cell = a region of scenario parameter space)

  ROAD GEOMETRY × MANEUVER × AGENTS × ENVIRONMENT × OCCLUSION
  ┌──────────┬──────────┬──────────┬──────────┬──────────┐
  │ highway  │ merge    │ density  │ clear    │ none     │
  │ urban    │ unprot.  │ ped dens │ rain     │ parked   │
  │  intxn   │  L-turn  │ cyclist  │ fog      │  veh     │
  │ rural    │ cut-in   │ presence │ night    │ building │
  │ roundab. │ jaywalk  │  ...     │ glare    │ ...      │
  │  ...     │  ...     │          │  ...     │          │
  └──────────┴──────────┴──────────┴──────────┴──────────┘
   COVERAGE HEATMAP per cell:  ███ dense   ▓▓ sparse   ░░ UNCOVERED
   ┌───────────────────────────────────────────────────┐
   │ unprot-L-turn × high-ped × night × glare :  ░░ ◄── known blind spot
   │ highway × cut-in × clear × none          :  ███     (argue/bound it)
   └───────────────────────────────────────────────────┘
```

**Coverage metrics:** (1) *combinatorial cell coverage* — fraction of taxonomy cells with ≥N scenarios; (2) *continuous-parameter coverage* — discrepancy/space-filling within each cell (timing, speed, intent angle); (3) *behavioral coverage* — distribution of induced criticality, ensuring I'm not just covering easy cells.

**Deliberate tail-sampling:** uniform sampling over the taxonomy still under-weights the dangerous tail (it's rare *by construction*). So I *oversample* high-criticality cells and reweight for risk (§9). The mine-and-perturb path (§10) anchors coverage of the *observed* tail; from-scratch generation extends to the *unobserved* tail.

**The honest part — exhaustive coverage is impossible.** The parameter space is effectively infinite (continuous timing/intent/speed, open-set agent types, novel geometries). I cannot prove I've covered everything. What I *can* do:
- **Bound the uncovered remainder:** explicitly enumerate cells with zero/sparse coverage and argue each — either it's outside the declared ODD (so the AV must *detect and disengage*, a separately tested capability), or it's low-exposure and low-severity, or it's flagged as residual risk requiring road validation.
- **Monitor coverage drift:** as the fleet encounters new geometries/behaviors, new cells appear; coverage is a *living* quantity tracked over time, not a one-time checkbox.

The combinatorial explosion (§3, ~10⁴–10⁵ cells) means I can't sample every cell densely under budget — which makes prioritization (§13) and the bounded-remainder argument the real coverage deliverables, not a green checkmark.

---

## 9. Rare-event statistics / importance sampling (~350 words)

**[SIGNAL: rare-event-statistics]**

Generating a rare failure is not the same as *estimating its probability*, and the safety case needs the probability. Kalra & Paddock (2016): demonstrating a fatality rate better than humans (~1 per 10⁸ miles) by naturalistic driving needs *billions* of miles — infeasible. Naïve Monte Carlo for a 10⁻⁶ event needs ~10⁸ samples for a tight estimate, and worse, *dangerously underestimates* the rate when it sees zero events.

**Importance sampling (IS) is the answer.** Sample from a biased proposal `q(x)` that over-generates critical scenarios, then **reweight** to recover an unbiased estimate of risk under the true distribution `p(x)`:

```
   p_fail = E_p[1(fail)]  =  E_q[ 1(fail) · p(x)/q(x) ]
                              └──────────┬──────────┘
                              estimate with samples from q,
                              weight w(x) = p(x)/q(x)
```

The biased proposal is exactly my criticality-guided generator (§5–6): it boxes the ego in, raises closing speeds, times the cut-in adversarially (cf. the NeurIPS rare-event-sim work, which learned a `q` that shifts initial conditions and speeds toward accidents). The importance weight `p(x)/q(x)` then corrects the bias so the pass/fail rate becomes a **calibrated risk number**, not an anecdote.

**Acceleration:** lane-change/cut-in IS studies report 2,000–20,000× — 1,000 simulated miles ≈ 2–20M naturalistic miles of exposure. Deep IS and normalizing-flow IS (TrimFlow: 86% test reduction) extend this to high-dimensional, learned proposals.

**What a sim pass/fail rate actually estimates:** with a *known, validated* proposal `q` and a *correct* density ratio, it estimates `p_fail` over the **covered** ODD region, with a confidence interval set by effective sample size `n_eff = (Σw)²/Σw²`. IS can have *huge* variance if `q` is mismatched (a few enormous weights dominate), so I monitor `n_eff` and use **robust/conservative IS upper bounds** (Robust Deep IS) for the safety case rather than the point estimate — I'd rather over-state residual risk than under-state it.

**Caveat:** IS gives me risk *over the regions I model*. It says nothing about un-modeled failure modes — that's coverage (§8) and validation-of-the-validator (§11), not statistics.

---

## 10. Mine-and-perturb vs generate-from-scratch (~350 words)

**[SIGNAL: mine-and-perturb] [SIGNAL: rejected-alternative]**

Pure from-scratch generation drifts off-manifold into the unrealistic gotchas of §5. Pure mining only ever re-tests what the fleet already saw. The staff move is **combining both, explicit about each one's role:**

**Mine (realism-anchored coverage of the *observed* tail).** Cluster fleet logs + WOMD for already-interesting/critical events: low min-TTC, hard brakes, disengagements, near-misses, rare maneuvers, unusual agent configurations. These are *guaranteed real* — they happened. They seed the regression suite and anchor realism.

**Perturb (explore the neighborhood of known-real events).** Take a real seed and ask counterfactuals that stay *local* to the real manifold:
- *Timing:* "what if the pedestrian started crossing 0.5 s earlier?"
- *Count:* "what if there were two cyclists instead of one?"
- *Intensity:* "what if the lead vehicle braked at 0.7 g instead of 0.4 g?"
- *Geometry:* "same interaction, but at a tighter intersection."

Because perturbations are bounded around a real event, they inherit its plausibility while expanding coverage — this is the most *trustworthy* source of new critical scenarios. Constrain perturbation magnitude to keep on-manifold (and validate with the realism gate anyway).

**Generate-from-scratch (novel/unobserved tail).** The world model proposes scenarios the fleet has *never* seen — genuinely novel geometries, agent combinations, event sequences. This is the only way to probe the *unknown* unknowns, but it's also where realism is most at risk, so it gets the strictest realism gate and the heaviest discriminator scrutiny.

```
   OBSERVED tail            NEIGHBORHOOD            UNOBSERVED tail
   ───────────────         of observed             ────────────────
   MINE (real,     ──perturb──►  (realistic,    GENERATE (novel,
    guaranteed)               anchored coverage)   risky, gated hard)
        │                          │                     │
        └──── realism HIGH ────────┴──── realism LOWER ───┘
              criticality from real events    criticality from guidance
```

**Rejected:** *from-scratch only* (drifts unrealistic, no anchor); *mine only* (can't extend past observed history, misses the novel tail that's exactly what road-collection can't cheaply get). Use both; weight toward mine-and-perturb for trust, lean on generation for novelty and gate it hardest.

---

## 11. Validation-of-the-validator: does passing mean safe? (~500 words)

**[SIGNAL: passing-means-what] [SIGNAL: sim-to-real-validation] [SIGNAL: saying-no]**

This is the deepest safety question in the round, and the honest answer is: **passing my suite does not prove the AV is safe.** It produces *evidence* that *bounds residual risk*, and only if two load-bearing assumptions hold. I will not let anyone treat "passed the suite" as "is safe."

```
        SAFETY CLAIM: "sim pass ⇒ road safe"
        rests on TWO load-bearing assumptions:

  ┌─────────────────────────────┐   ┌─────────────────────────────┐
  │ (1) SIM REALISM              │   │ (2) COVERAGE                 │
  │  sim outcome predicts real   │   │  suite represents reality    │
  │  outcome                     │   │  / blind spots are bounded   │
  │  ├ behavior fidelity         │   │  ├ ODD taxonomy filled       │
  │  ├ perception fidelity       │   │  ├ tail sampled              │
  │  └ reactive-agent fidelity   │   │  └ uncovered remainder       │
  │                              │   │     enumerated & argued      │
  └──────────────┬──────────────┘   └──────────────┬──────────────┘
                 │                                  │
                 ▼                                  ▼
     ┌─────────────────────────────────────────────────────┐
     │  VALIDATE THE VALIDATOR: correlate sim vs real        │
     │  • replay real disengagements/incidents in sim —      │
     │    does sim reproduce the outcome?                    │
     │  • track sim-predicted vs road-observed failure rates │
     │  • measure false-pass (sim OK, road bad) & false-fail │
     └───────────────────────┬───────────────────────────────┘
                             ▼
            RESIDUAL RISK ARGUMENT (SOTIF / ISO 21448):
       sim evidence + road testing + bounded unknowns → rollout decision
```

**Assumption 1 — sim realism (does a sim outcome predict the real outcome?).** Two sub-layers. *Behavior fidelity:* do reactive agents behave like real humans (§7)? *Perception fidelity:* if I tested at the agent level, I've *assumed perfect perception* — so a behavior "pass" says nothing about whether the AV would have *seen* the pedestrian. This is why the abstraction choice (§3) is a safety statement: agent-level results are valid *only* for the decision, conditioned on perception being separately validated. I validate the sim itself by **replaying real disengagements and incidents** in sim and checking the sim reproduces the real outcome; if the sim says "fine" where the road said "bad," my realism is broken.

**Assumption 2 — coverage (§8).** A sim the AV passes perfectly with a giant blind spot is *more* dangerous than an honest partial suite, because it manufactures false confidence.

**The overfitting trap (also §12).** If the AV is tuned/trained against the sim, a passing rate measures *memorization of the suite*, not safety — like reporting train accuracy. The validator has been gamed. I hold out scenarios, rotate the suite, and **never let the suite that grades the rollout be the suite the AV trained on.**

**The honest limit.** You cannot prove safety by passing a self-generated suite — the generator and AV share blind spots (both miss the same unimagined failure mode). Simulation *bounds* residual risk and *finds* failures cheaply; it does not *certify*. That is why **real-world testing remains irreplaceable** for the un-modeled tail and for validating the sim itself, and why ISO 21448/SOTIF frames sim as one evidence stream in a layered argument, with governance deciding how much weight it carries (§13).

---

## 12. Failure triage, feedback, and overfitting-to-sim (~350 words)

**[SIGNAL: pipeline-and-triage] [SIGNAL: overfitting-to-sim]**

A nightly run over 10⁶–10⁷ scenarios surfaces thousands of failures. Raw failure count is noise; **actionable, deduped, root-caused signal** is the product.

**Triage pipeline:**
1. **Cluster** failures by signature — failure type (collision/near-miss/rule-violation), scenario taxonomy cell, AV stack stage implicated, geometry. Thousands of failures collapse to tens of *failure modes*.
2. **Dedupe** near-identical failures (same root cause, perturbed surface) so one bug isn't counted 500 times and prioritized 500×.
3. **Separate real AV bugs from sim artifacts** — *the trust-critical step.* A failure might be (a) a genuine AV bug, or (b) an "unrealistic gotcha" that slipped the realism gate (§5). I re-run the failing scenario through a *stricter* realism check, assign responsibility (could a competent driver avoid it? is the NPC plausible?), and route sim-artifact failures back to *fix the generator/gate*, not the car. Misclassifying artifacts as bugs is how you over-conservatize the AV.
4. **Root-cause & route** to the owning team (perception miss vs prediction error vs planner logic).
5. **LLM-assisted triage (2025–2026):** use an LLM to summarize failure clusters, draft root-cause hypotheses, and generate human-readable scenario descriptions, scaling analyst throughput — with human sign-off on anything safety-relevant.

**Feedback loops:**
- *To the car:* fix prioritized real bugs.
- *To generation (active failure-search):* densely sample around each confirmed failure to map the boundary — how robust/fragile is the fix, what's the basin of failure.

**Guarding overfitting-to-sim** (the §11 trap, operationalized):
- **Train/test separation:** scenarios used to *develop/tune* the AV are disjoint from scenarios used to *grade* it.
- **Suite rotation & freshness:** continuously regenerate held-out evaluation scenarios; a static suite gets memorized.
- **Diversity floors:** monitor that fixes generalize across the taxonomy cell, not just the exact failing instance.
- **Watch for "improving sim score, flat/declining road metrics"** — the canonical overfitting signature; if sim pass-rate climbs but road disengagements don't fall, the suite is being gamed, not the safety improved.

---

## 13. Compute, prioritization, monitoring, governance (~300 words)

**[SIGNAL: capacity-and-cost-math]**

**Compute & the diffusion-speed problem.** Closed-loop cost = AV-stack × generator-sampling × scenario-count (§3). Multi-step diffusion sampling (CTG++ reported ~1 min/scenario) is the bottleneck at scale. Mitigations: **consistency models / distillation** to 1–4 step sampling, caching reactive-agent rollouts, and agent-level (not sensor) abstraction wherever the test target allows. Sampling speed is genuinely on the critical path — a 10× generator speedup is a 10× coverage budget.

**Prioritization under fixed budget.** I can't densely sweep 10⁴–10⁵ cells × 10⁷ scenarios nightly. So:
- **Criticality/novelty-weighted sampling:** spend budget where failures are likely or coverage is sparse, not on the easy head.
- **Active learning toward failure boundaries:** allocate marginal compute to mapping discovered boundaries (highest information per scenario).
- **Two modes, two budgets:** a cheap **regression suite** (fixed, deterministic, runs every commit) vs expensive **exploratory search** (runs nightly/weekly, finds new failures).

**Monitoring (the validator drifts):**
- **Sim-realism drift:** track realism-discriminator scores and sim-vs-real correlation over time; the world changes (new vehicle types, e-scooters, construction patterns) and the generator decalibrates. Re-validate continuously.
- **Coverage drift:** new fleet-observed cells appear; coverage is a living metric (§8).
- **Overfitting signature:** sim-score-up / road-metric-flat (§12).

**Governance & the safety wrapper.** The hard question isn't technical — it's *how much weight simulation evidence carries in a go/no-go rollout decision.* My stance: sim evidence **bounds and prioritizes** but does not **certify**; the rollout argument (ISO 21448/SOTIF) layers sim evidence with structured road testing and an explicit, signed-off **residual-risk statement** enumerating what the suite does *not* cover. Real-world testing is irreplaceable for the un-modeled tail and for validating the sim itself. Simulation makes the safety case *affordable and fast*; it does not make it *complete*.

---

## 14. Tradeoffs taken and what would change them (~150 words)

**[SIGNAL: rejected-alternative]**

- **Agent-level primary, sensor-realistic secondary.** Most long-tail failures are decision failures, cheaply and validatably testable on boxes. *Changes if* the dominant failure mode shifts to perception (e.g., novel-object detection misses), then sensor-realistic generation (GAIA-2/Cosmos) gets the budget despite cost.
- **Guide-toward-criticality + reweight, over broad-then-filter.** Critical events are too rare to filter for efficiently. *Changes if* generation gets nearly free (strong distillation), making broad sampling viable for diversity.
- **Mine-and-perturb weighted over from-scratch.** Realism trust dominates early. *Changes as* the realism discriminator and world model mature enough to trust novel generation, shifting weight toward from-scratch for tail coverage.
- **Robust/conservative IS bound over point estimate.** Over-stating residual risk is the safe error for a safety case. *Changes* only with very high `n_eff` and validated proposals.
- **Learned reactive agents over IDM,** with IDM as interpretable baseline. *Changes* for very long horizons where rule models are more stable.

---

## 15. What I would push back on (~150 words)

**[SIGNAL: saying-no] [SIGNAL: passing-means-what]**

Three pushbacks, stated to the interviewer directly:

1. **"Passing the suite means the AV is safe" — no.** It bounds residual risk *given* sim realism and coverage hold, and given the AV didn't overfit the suite. Passing is evidence, not proof. I'd refuse to sign a rollout on sim alone.

2. **"Generate adversarial scenarios to break the AV" — not without a hard realism/plausibility constraint.** Unconstrained adversarial generation produces unrealistic gotchas that waste effort, erode trust, and drive pathological over-conservatism. The objective is plausible-AND-hard, on the boundary, never impossible-and-hard.

3. **"Use log replay, it's cheaper" — invalid for behavior testing.** The moment the AV diverges from the logged ego, replayed agents are nonsensical. Reactive agents are non-negotiable; replay is fine only for perception regression where the ego trajectory is fixed.

And one meta-pushback: **"more scenarios = better testing" — no.** Useful, realistic, covering scenarios are the goal; a million highway clips is negative value.