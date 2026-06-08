---
title: diffusion validation
description: diffusion validation
---
"You've built a generative world model that produces driving scenarios and simulates how scenes evolve, including how other agents react to the AV. The team wants to use it to validate the AV and inform rollout decisions. Before we trust it for that, how do you know the simulator is realistic enough? Design the validation strategy — what you measure, how you'd catch the ways it's wrong, and how much weight its evidence should carry in a safety decision. Walk me through it."
---
# Validating a Generative World Model for a Safety Decision

*ML System Design, Diffusion, and World Models — the validation question.*

The spine of this answer: **realistic-looking is not safe-to-decide-on.** A world model can produce gorgeous, physically plausible driving that is wrong in exactly the ways that corrupt a safety decision — wrong in the tails, wrong when reacting to a novel AV maneuver, right on marginals but wrong on the joint. So I will not hand you a realism number. I will tell you *which decision the sim informs*, what *predictive validity* that decision requires, build a *layered* validation stack where each layer catches what the others miss, confront the outputs that *have no ground truth*, and end at a *bounded-evidence safety case* that is honest about what simulation can and cannot prove.

---

## 1. State of the art as of 2026 (research pass)

Three threads define the current frontier.

**Distributional metrics are known-broken as trust certificates.** Ge et al. (CVPR 2024, "On the Content Bias in Fréchet Video Distance") showed FVD is dominated by *per-frame* appearance, not temporal dynamics: they applied severe temporal corruption that humans see instantly, yet FVD *dropped* (improved) because its I3D backbone, trained on content-biased Kinetics-400, is largely blind to motion. They further showed you can *halve* FVD by resampling motion-free clips. For a driving world model — where the entire point is dynamics — a low FVD is close to meaningless as evidence of behavioral realism.

**Sim-agent realism has a mature decomposition but a known closed-loop hole.** The Waymo Open Sim Agents Challenge (WOSAC; Montali et al. 2023) scores simulated agents against logged data via an approximate-NLL "Realism Meta" metric decomposed into *kinematic*, *interactive*, and *map-based* components, with collisions/off-road double-weighted for safety. Top 2025 entries (SMART-R1, TrajTok, UniMM) reach ~0.785. But Schofield et al. (ICRA 2025, "Beyond Simulation") demonstrated the critical failure: top-ranked models score well under *no perturbation*, then **fail when the ego is forced off its logged trajectory** — i.e., exactly the reactive, causal regime you test the AV in. Open-loop replay metrics (ADE/minADE vs the logged future) systematically *overstate* realism because they never exercise reactivity.

**Validation-by-simulation is being formalized into safety cases — and the ground truth itself is suspect.** ISO 21448 (SOTIF) frames the deliverable as a *residual-risk argument* over an ODD, and its Clause 12 requires that *development evidence and validation evidence be kept statistically independent* — the data used to build the system cannot validate it. Waymo's published philosophy (SimulationCity; the 2026 Waymo World Model; the reconstructed-fatal-crash counterfactual study) is *convergence of outcome distributions* between sim and real, calibrated against billions of real miles. Yet a 2025 validation study (Wang et al., arXiv 2509.03515) found that **the Waymo Open Motion Dataset itself falls outside the naturalistic behavioral envelope** of independently collected Phoenix L4 data — underrepresenting short headways and hard decelerations. The data your generator imitates may already be biased away from the safety-relevant tail.

**Still unsolved:** validating *counterfactual* fidelity (no ground truth exists for what didn't happen), establishing *predictive validity* for absolute rare-event rates, and preventing AV-overfits-the-sim coadaptation.

---

## 2. Reframing: valid for *what decision*?

**[SIGNAL: valid-for-what-decision]** A simulator does not have a realism score. It has *fitness for a specific decision*. "Is it realistic?" is the wrong question; the right one is "does a result computed in sim predict the corresponding result in reality, to the precision this decision requires?" That is **predictive validity**, and it is the operational definition of "realistic enough."

This matters because the required validity differs by orders of magnitude across decisions:

| Decision the sim informs | What predictive validity is required | Why |
|---|---|---|
| **A) Regression test** ("did this commit make behavior worse?") | Sensitivity to a *change*; sign-consistency on a fixed scenario suite | Only relative deltas matter; systematic bias cancels |
| **B) Policy ranking** ("is planner B better than A?") | High *rank* correlation (Spearman ρ) between sim and road metrics | Absolute calibration irrelevant; only ordering must transfer |
| **C) Failure discovery** ("find scenarios where the AV fails") | High *recall* of real failure modes; low false-negative rate | Misses are dangerous; false alarms are merely expensive |
| **D) Absolute risk estimate** ("collision rate < X per mile") | *Calibrated* absolute rates in the safety-critical tail | A 2× bias is a safety-case failure |
| **E) Rollout sign-off** | All of the above *plus* a residual-risk argument | The sim becomes load-bearing for a deployment decision |

**[SIGNAL: rejected-alternative]** I reject *"validate the simulator"* as a single global claim, and I reject *distributional-similarity-as-validation*. The first is incoherent (a sim great for B can be useless for D); the second measures whether samples *look* like data, not whether *decisions* transfer. **[SIGNAL: saying-no]** Concretely: if you ask me to certify "the sim is realistic," I will refuse and instead certify "the sim has predictive validity ρ≥0.9 for planner ranking within the urban-unprotected-left envelope, and is *not* validated for absolute collision-rate estimation." Those are different artifacts with different evidence.

The metric hierarchy follows from the decision. A regression test needs only a stable, sensitive sim. A ranking decision needs proven rank-correlation. An absolute-risk decision needs calibration in the tail — the hardest and the one a self-built generative sim will most likely fail to support.

---

## 3. Scope and the validation-data math

**[SIGNAL: scope-and-framing]** Commit before designing. I scope this sim as: **behavioral / agent-reaction simulation** (not the full perception stack), conditioned on the AV's actions, used primarily for **policy ranking and failure discovery (B, C)**, with a *bounded, explicitly-flagged* contribution to **rollout sign-off (E)** and an *explicit refusal* to be the sole basis for **absolute-rate claims (D)**. Different scope → different validation; I am choosing the scope where a learned world model is most defensible and being honest about where it is not.

**[SIGNAL: validation-data-statistics]** Everything rests on **held-out real data the generator never saw** — temporally and geographically split, so "held-out" means *future* and *other-city*, not a random shard of the training distribution. Random splits leak: the generator has seen the same intersections, the same agent population.

The rare-event arithmetic is the brutal constraint. Suppose a safety-relevant event occurs at rate p ≈ 10⁻⁶ per mile. To *validate* that the sim reproduces that rate within ±20% at 95% confidence, the binomial requirement is roughly n ≳ z²(1−p)/(p·ε²). With z=1.96, ε=0.2:

```
n ≈ (1.96² · 1) / (10⁻⁶ · 0.04) ≈ 3.84 / (4×10⁻⁸) ≈ 9.6 × 10⁷ events-worth of exposure
→ ~10⁸ real miles to see enough events to even check the rate to ±20%.
```

| Quantity | Order of magnitude | Implication |
|---|---|---|
| Real miles to validate a 10⁻⁶/mi rate to ±20% | ~10⁸ mi | Direct rare-event rate validation is **infeasible** for most events |
| Miles to validate a 10⁻⁴/mi rate | ~10⁶ mi | Feasible for a mature fleet |
| Cost anchor | real miles are the scarce, expensive ground truth | Spend them on *anchoring*, not brute force |

**The conclusion is structural, not a detail:** you cannot brute-force-validate absolute tail rates with a self-built sim. So the strategy must (a) push absolute-rate questions onto real-world exposure + importance-sampling arguments, and (b) use the sim where *relative* validity (ranking, regression, failure discovery) needs far less data — establishing a rank correlation needs hundreds of paired scenarios, not 10⁸ miles. This reallocation is the whole game.

---

## 4. The multi-layered validation stack

**[SIGNAL: multi-layered-stack]** No single test is sufficient; a single metric is a single point of failure for the safety case. Each layer is a *necessary* condition that catches failures invisible to the others. Lower layers are cheap, ground-truth-free, and weak; upper layers are expensive, real-data-anchored, and decision-relevant.

```
                         WEIGHT IN SAFETY DECISION
   ┌─────────────────────────────────────────────────────────┐ ▲ high
   │ L7  RESIDUAL-RISK / SAFETY CASE                           │ │
   │     "trusted for B,C within envelope; NOT for absolute D" │ │
   ├─────────────────────────────────────────────────────────┤ │
   │ L6  PREDICTIVE VALIDITY vs REAL OUTCOMES (gold standard)  │ │
   │     sim ranking == road ranking? sim failures reproduce? │ │
   ├─────────────────────────────────────────────────────────┤ │
   │ L5  DOWNSTREAM-DECISION VALIDITY                          │ │
   │     does the decision made-in-sim survive in reality?    │ │
   ├─────────────────────────────────────────────────────────┤ │
   │ L4  CLOSED-LOOP / REACTIVE BEHAVIORAL REALISM             │ │
   │     agents react to NOVEL ego behavior like real humans  │ │
   ├─────────────────────────────────────────────────────────┤ │
   │ L3  PHYSICAL / KINEMATIC / GEOMETRIC CONSISTENCY          │ │
   │     necessary conditions; checkable w/o ground truth     │ │
   ├─────────────────────────────────────────────────────────┤ │
   │ L2  COVERAGE (precision/recall — fidelity vs diversity)  │ │
   ├─────────────────────────────────────────────────────────┤ │
   │ L1  DISTRIBUTIONAL REALISM (FVD/FID/WOSAC) — FLOOR ONLY   │ │
   └─────────────────────────────────────────────────────────┘ ▼ low
        cheap, ground-truth-free          expensive, real-anchored
```

The discipline: **pass-low-layers is necessary, never sufficient.** A model can ace L1 and fail L4 catastrophically (precisely the "Beyond Simulation" result). Promotion up the stack requires passing every layer below; the *weight* a decision places on sim evidence is set by the highest layer you've actually validated for that decision.

---

## 5. Distributional realism — the necessary floor, and why it's insufficient

**[SIGNAL: looks-real-vs-behaves-real]** L1 metrics — FVD/FID for sensor-realistic sims, the WOSAC Realism Meta for agent sims — measure whether *samples look like the data distribution*. They are genuinely useful as a **regression guard and a gross-unrealism floor**: they catch mode collapse, obvious artifacts, agents teleporting, distributions that have drifted off the manifold. Cheap to compute, no real-world rollout needed. Keep them.

But as a *trust certificate* they fail on documented grounds:

- **They measure appearance, not dynamics.** Ge et al. showed FVD barely moves under severe temporal corruption and can be *halved* by resampling motion-free clips. A world model whose physics are subtly wrong but whose frames look crisp scores *well*.
- **Backbone-sensitive and gameable.** FVD depends on the I3D feature extractor; change the backbone and the ranking changes. Anything optimizable is Goodhart-able.
- **Blind to the tail.** A Fréchet distance between Gaussians fit to features is dominated by the bulk. The 10⁻⁶ safety-critical events contribute negligibly. You can match the marginal beautifully and miss every rare event that matters.
- **Marginal ≠ joint.** Matching the marginal distribution of, say, agent speeds says nothing about whether the *joint* — this agent's deceleration *given* the ego's lane change — is right. Safety lives in the conditional dynamics, which a marginal-matching metric does not see.

**[SIGNAL: rejected-alternative]** "Compute FVD against real driving video and check it's low" — rejected as the validation. It is the demo answer. Low FVD with wrong reactions is the *expected* failure, not an edge case.

**L2: precision/recall to separate fidelity from coverage.** A single Fréchet number conflates two failures. Improved-precision/recall for generative models (Kynkäänniemi et al. 2019) decomposes into **precision** (are generated samples on the real manifold — *fidelity*, "looks real") and **recall** (does the generator cover the real distribution — *diversity*). A safety sim with high precision but low recall is the silent killer: every scenario it produces looks real, but it *never produces* the weird construction-zone-plus-cyclist-plus-glare scenario that crashes the AV. For failure discovery (decision C), **recall of the tail is the metric that matters**, and it is exactly what a likelihood-trained generator under-delivers.

---

## 6. Physical, geometric, and kinematic consistency

**[SIGNAL: consistency-as-necessary-condition]** L3 is the most underrated layer because it is the one place we get *ground-truth-free necessary conditions* — checks that must hold for *any* valid future, including counterfactual ones we can never observe. This is the lever for the no-ground-truth problem (§8).

Hard checks, asserted on every rollout:

- **Kinematic feasibility** — no agent exceeds physically achievable acceleration/jerk/curvature for its class; no teleporting; tire-friction and turning-radius limits respected. A bicycle model or per-class dynamics envelope is the referee.
- **Object permanence & temporal coherence** — agents don't pop in/out, swap identities, or flicker; tracks are continuous.
- **Collision / interpenetration plausibility** — bodies don't overlap; the rate and geometry of contacts are physically sane.
- **Map compliance** — agents respect drivable area, lane topology, signal states (unless deliberately modeling a violation, which must be labeled as such, not a glitch).
- **Conservation / smoothness** — momentum continuity, no instantaneous velocity reversals.

The power of L3: a counterfactual rollout that *violates physics is invalid even though we cannot observe the true counterfactual.* We can't measure how right an off-distribution reaction is, but we can prove a teleporting pedestrian is wrong. Necessary conditions bound the space of "not-obviously-broken" futures from below.

**Failure modes of L3:** it is necessary, never sufficient — a perfectly physical agent can still behave socially absurdly (yields when no human would, never accepts a gap). Physics passes; behavior fails. And consistency checks can be *gamed* by a generator that learns to be conservative — physically immaculate but unrealistically timid, which corrupts a ranking that rewards assertiveness. So L3 gates, L4 judges.

---

## 7. Closed-loop / reactive behavioral realism — where it breaks

**[SIGNAL: closed-loop-realism]** This is the layer that breaks, and it breaks where it matters most. The core asymmetry:

```
 OPEN-LOOP (replay)                    CLOSED-LOOP (reactive)
 ─────────────────                     ──────────────────────
 sim agents follow / are scored        AV acts → world model must
 against their LOGGED future.          generate agents' REACTIONS →
                                        those reactions change the
 ego replays logged trajectory.        next AV action → compounding.

 ADE/minADE vs logged future.          No logged future exists for the
 EASY to measure.                      AV's novel maneuver. HARD.

 ┌───────────────────────────┐        ┌───────────────────────────┐
 │ scored against ground truth│        │ ego does something NOVEL   │
 │ that ACTUALLY HAPPENED      │        │ (off-distribution) →       │
 │ → metric looks GREAT        │        │ world model's reactions go │
 │                             │        │ off-distribution too →     │
 │ but reactivity NEVER tested │        │ realism DEGRADES exactly   │
 └───────────────────────────┘        │ when you need it most      │
                                        └───────────────────────────┘
        ⇒ OPEN-LOOP SYSTEMATICALLY OVERSTATES REALISM
```

**Why open-loop replay overstates realism:** ADE-against-logged-future rewards a model for reproducing what happened. But the AV under test is *not* the logged AV — that's the point of testing it. The instant the AV deviates, the logged future is counterfactual and the replay metric is scoring against a world that no longer exists. The "Beyond Simulation" benchmark (ICRA 2025) made this empirical: WOSAC-top models scored well under no perturbation and **failed when the ego was forced off its logged trajectory**, especially for agents *causal* to the ego. The metric that ranked them was blind to the failure that matters.

**Measuring closed-loop realism — what L4 actually does:**

1. **Sim-Agents-style decomposition under perturbation.** Take WOSAC's kinematic/interactive/map-based decomposition but evaluate it *with the ego perturbed* — replay the ego off its logged path and re-score. The interactive component (gap acceptance, yielding, following distances, lead-vehicle response) is where reactive realism lives. Compare the *distribution* of simulated reactions to the distribution of real human reactions in *matched* situations (same geometry, same approach speed) drawn from held-out logs.

2. **Inject novel AV behaviors and measure reaction drift.** Deliberately make the AV do things the logged AV didn't — late merges, assertive unprotected lefts, hard stops. For each, ask: do simulated agents yield/brake/accept-gaps with the *distribution* real humans show in comparable real situations? Quantify with the right tools: distributional distance (Kolmogorov–Smirnov, Wasserstein) on reaction variables, Dynamic Time Warping on reaction trajectories (as in the WOMD naturalistic-validation study), not point ADE.

3. **OOD-reaction probe.** Build a held-out set of *real* human reactions to genuinely unusual ego behavior (mined from the fleet: every time a human driver did something surprising and we logged how others reacted). Score the world model's reactions against *that* held-out distribution. This directly attacks the regime where realism degrades.

**Failure modes:** even "matched situations" are never exactly matched, so there's irreducible noise; the held-out OOD-reaction set is small (rare by construction); and a model can learn to react *plausibly* (smooth, physical) without reacting *correctly* (right magnitude, right timing). L4 measures reactive realism on the *observable* perturbations; it cannot directly reach the truly never-observed counterfactual — which is §8.

---

## 8. The counterfactual / no-ground-truth problem

**[SIGNAL: counterfactual-no-ground-truth]** This is the intellectual core and the part most answers skip. The world model's *entire value* is generating futures that didn't happen: counterfactual reactions to actions the logged AV never took, rare events, "what if the AV had merged here." But for *exactly those outputs there is no real-world ground truth* — the counterfactual never occurred, so there is nothing to compare against. You cannot directly measure the fidelity of your most important outputs. Any answer that doesn't say this out loud has skipped the question.

You cannot measure it. You can *bound* it, *cross-check* it, and *argue* it. The honest proxy stack:

```
            THE COUNTERFACTUAL HAS NO GROUND TRUTH
                          │
     ┌────────────────────┼────────────────────────────┐
     ▼                    ▼                              ▼
 [A] GENERALIZATION    [B] NECESSARY              [C] INDEPENDENT
     ARGUMENT              CONDITIONS                  CROSS-CHECK
 validate on the      physics/kinematics/         a SEPARATE model
 OBSERVABLE held-out  permanence MUST hold         (different arch,
 futures; argue the   even off-distribution.       data, or a
 model generalizes    A counterfactual that        rule-based sim)
 to nearby            violates them is INVALID     should agree on
 counterfactuals.     even if we can't see          the counterfactual.
 (bounded by how      the true one.                 Divergence = distrust.
  far we extrapolate)                              
     ▼                    ▼                              ▼
 [D] EXPERT REVIEW     [E] BOUND, DON'T MEASURE    [F] HONEST RESIDUAL
 trained AV operators  give a conservative          state the
 adjudicate            envelope (worst-case         irreducible
 counterfactual        plausible reaction)          uncertainty; do
 plausibility on       rather than a point          NOT claim a number
 sampled rollouts.     estimate.                    you cannot earn.
```

Concretely:

- **[A] Validate on observable futures, argue generalization.** We *can* measure fidelity on held-out futures that *did* happen (the AV's logged behavior, other agents' logged reactions). If the model is accurate on observed reactions and the counterfactual is a *small* perturbation from observed conditions, generalization is plausible — and the validity *degrades with extrapolation distance*, which we measure and use to bound trust (this is the trust envelope, §10). A counterfactual one standard-deviation off the data is far more trustworthy than one in never-seen territory.
- **[B] Necessary conditions for the unobservable** (from §6): physics/permanence/kinematics constrain counterfactuals from below. Can't confirm the true reaction; *can* reject impossible ones.
- **[C] Independent cross-check:** if a structurally different model (different architecture, different data, or a calibrated rule-based microsimulator like a SUMO-style model — which the benchmarking literature shows is weaker short-horizon but more stable long-horizon) *agrees* on the counterfactual, confidence rises; divergence is a flag to distrust.
- **[D] Expert adversarial review:** experienced AV operators/safety engineers adjudicate plausibility of sampled counterfactual rollouts. Not rigorous, but catches socially-absurd-but-physical failures L3 misses.
- **[E] Bound rather than measure:** for safety, a *conservative envelope* ("under this counterfactual, the worst plausible human reaction is X") is more useful than a point estimate you can't validate.
- **[F] Honest residual:** there remains irreducible uncertainty on counterfactual fidelity. The safety case must *carry* that uncertainty, not paper over it.

---

## 9. Predictive validity against real outcomes

**[SIGNAL: predictive-validity]** The gold standard (L6): does a result *in sim* predict the corresponding result *in reality*? Three concrete forms, in increasing difficulty:

```
        THE PREDICTIVE-VALIDITY LOOP
   ┌──────────────────────────────────────────────┐
   │  pick N scenarios / K policies                  │
   │            │                                    │
   │   ┌────────┴────────┐                           │
   │   ▼                 ▼                           │
   │ RUN IN SIM      RUN ON ROAD (or held-out logs)  │
   │   │                 │                           │
   │ sim_metric_i     real_metric_i                  │
   │   └────────┬────────┘                           │
   │            ▼                                     │
   │   correlate:  Spearman ρ (ranking)              │
   │              failure-reproduction rate (recall) │
   │              metric correlation (calibration)   │
   │            │                                     │
   │   ρ ≥ threshold-for-this-decision? ─── no ──┐   │
   │            │ yes                            ▼   │
   │   TRUST sim for this decision-class   DON'T TRUST│
   └──────────────────────────────────────────────┘
```

1. **Rank correlation (for decision B).** Run K candidate planners in sim *and* on the road (or against held-out real logs in pseudo-simulation). Does sim rank them the same way reality does? Spearman ρ is the statistic. For a ranking decision, *absolute calibration is irrelevant* — only ordering must transfer. This is the cheapest strong validity and the one a learned sim is most likely to support. With K=10 policies, a Spearman ρ≥0.9 is significant at p<0.001; you need a *modest* paired set, not 10⁸ miles. The catch: ρ must hold *in the tail*, not just on average — a sim that ranks correctly in nominal driving but inverts the ranking on safety-critical scenarios passes the aggregate test and fails the safety case. So compute ρ *stratified by scenario criticality*.

2. **Failure reproduction (for decision C).** Sim-discovered failures must reproduce on the road (or in held-out real logs); and real failures must be discoverable in sim. The metrics are **precision** (sim failures that are real / not artifacts) and, more important for safety, **recall** (real failures the sim catches). A sim that invents failures the AV would never hit wastes engineering; a sim that *misses* real failures is dangerous. Recall is the safety metric.

3. **Metric correlation (toward decision D).** Do sim proxies (collision proxy, disengagement proxy, near-miss rate) correlate with the real ones? This is the hardest and is where I am most cautious: correlation in the bulk does not establish calibration in the tail, and the tail is the safety case.

**Building the paired dataset is the cost center.** You need the *same* scenarios/policies run in both worlds — which means spending real miles (or carefully de-confounded held-out logs) deliberately on validation rather than on more sim. **[SIGNAL: rejected-alternative]** Predictive-validity validation vs distributional-similarity validation: I choose predictive validity because it measures *decision transfer*, the only thing the safety case can use; distributional similarity measures sample resemblance, which we've shown is gameable and tail-blind.

---

## 10. Coverage and the validated envelope

**[SIGNAL: trust-envelope]** A sim is trustworthy *only within the region where its predictive validity was established.* Outside it, the sim is extrapolating, and its evidence weight should fall toward zero.

```
            THE TRUST ENVELOPE
   ┌───────────────────────────────────────────┐
   │  ODD / scenario / ego-behavior space         │
   │                                              │
   │     ╔══════════════════════╗                 │
   │     ║ VALIDATED ENVELOPE     ║   ← here: ρ,    │
   │     ║ (urban unprotected     ║     recall,     │
   │     ║  left, dry, daytime,   ║     calibration  │
   │     ║  ego within tested     ║     ESTABLISHED  │
   │     ║  maneuver range)       ║                 │
   │     ╚══════════════════════╝                 │
   │   · snow  · novel ego maneuvers  · new city    │
   │   · e-scooter swarms  · construction patterns  │
   │        ↑ EXTRAPOLATION — flag, don't trust     │
   └───────────────────────────────────────────┘
```

Operationally: define the envelope along ODD axes (weather, lighting, geography), scenario types, and *AV-behavior range* (how far the ego deviates from logged distribution). Every sim run is tagged with where it sits; a run that leaves the envelope is *flagged as extrapolation* and its evidence is downweighted in any decision. **[SIGNAL: rejected-alternative]** "Trust the sim broadly" vs "trust it only within a validated envelope": broad trust is how a sim that's great for highway following gets used to sign off on a construction-zone rollout it was never validated for. The danger is silent: nothing errors, the numbers look fine, and the validity simply isn't there. The un-validated remainder must be carried explicitly into the residual-risk argument (§13), not assumed safe.

---

## 11. Gaming, Goodhart, and overfitting-to-the-sim

**[SIGNAL: gaming-and-goodhart]** Validation becomes circular in four ways, each with a safeguard:

- **AV implicitly tuned to pass the sim.** If the planner is developed against the sim, it co-adapts to the sim's quirks; passing the sim stops measuring reality. *Safeguard:* a strictly held-out real validation set the AV-tuning loop never touches; track the *divergence* between sim-performance trend and road-performance trend over releases — if sim improves while road stalls, the AV is overfitting the sim.
- **Realism metric optimized rather than reality matched.** WOSAC scores are directly optimizable (the leaderboard exists). A generator RL-tuned to maximize Realism Meta may game the metric — and "Beyond Simulation" shows high-meta models that fail under perturbation. *Safeguard:* treat metric-pass as necessary-not-sufficient; hold out *unoptimized* validity tests (predictive validity, expert review) the generator was never trained against.
- **Validation/real data leaking into the generator's training.** If the held-out set overlaps the generator's training distribution, "held-out" performance is memorization. *Safeguard:* enforce SOTIF Clause-12 independence — development evidence and validation evidence statistically independent; temporal+geographic splits; a validation set owned by a separate team and never released to model training.
- **Sim and AV co-adapting** into a private equilibrium that satisfies both and resembles neither reality. *Safeguard:* independent validation team; the reality anchor (real miles, real failures) as the non-negotiable arbiter.

The meta-principle: *when passing the sim becomes the target, it ceases to measure reality* (Goodhart). The structural defense is keeping a real-world ground-truth channel that no optimization loop is allowed to touch.

---

## 12. Calibration and uncertainty / OOD

**[SIGNAL: calibration-and-ood]** A world model should know when it doesn't know. Two uses:

- **Calibration of its predicted distributions:** if the model assigns probabilities to futures, are they honest? Reliability diagrams / proper scoring rules on held-out observable futures test whether its 80%-confidence reactions happen ~80% of the time. A miscalibrated-but-low-FVD model is a trap — confident and wrong.
- **Uncertainty as an OOD know-when-not-to-trust signal:** the model's own likelihood / predictive entropy on a rollout is a cheap detector of leaving the data manifold. Low likelihood / high entropy → the rollout is in extrapolation territory → downweight its evidence (links directly to the trust envelope, §10). The honest failure mode: deep generative models are notoriously *over*confident OOD (can assign high likelihood to garbage), so this signal is a *useful flag, not a proof* — calibrate it against known-OOD held-out data and treat it as one input, never the certificate.

---

## 13. The safety case and residual-risk argument

**[SIGNAL: residual-risk-honesty]** The deliverable is not a realism number. It is a *defensible statement of how much a rollout decision should rely on sim evidence, where it's trusted, where it's not, and what residual risk remains.* SOTIF/ISO 21448 framing: the sim contributes *one branch* of evidence toward an argument that residual risk from functional insufficiency is acceptably low across the ODD — alongside real-world testing, scenario-based testing, and field monitoring.

The honest staff position, stated plainly: **you cannot prove AV safety with a self-built simulator.** A simulator you built and your team validated, used to validate your own AV, is structurally vulnerable to shared blind spots and co-adaptation. What you *can* produce is **bounded evidence** with explicit scope:

- **Trusted (high weight):** relative decisions — regression detection, policy ranking, failure *discovery* — *within the validated envelope*, where predictive validity (rank ρ, failure recall) is established. The sim's superpower is scale (Waymo's ~20M sim miles/day) applied to *relative* questions.
- **Bounded (medium weight):** failure discovery extending slightly past the envelope, with uncertainty inflated by extrapolation distance; counterfactual exploration backed by necessary-condition checks and cross-model agreement.
- **Not trusted (low/zero weight):** absolute rare-event rate claims (the §3 arithmetic forbids it); anything outside the validated envelope; novel-geography rollout where the envelope hasn't been re-established.
- **Irreplaceable real-world:** absolute risk anchoring, the tail rates, and the final confidence in the residual-risk argument come from real exposure — not the sim. Real miles are spent *anchoring and predictive-validity pairing*, not brute-forcing.

The argument structure: *claim* ("planner B's safety-critical-scenario performance dominates A within envelope E") → *evidence* (rank ρ stratified by criticality, failure recall, closed-loop reactive realism, consistency checks) → *defeaters addressed* (gaming safeguards, leakage independence, OOD flags) → *residual risk* (un-validated envelope remainder, counterfactual uncertainty, tail mis-calibration) → *acceptance* (decision-maker accepts a stated, bounded residual, with real-world monitoring as the backstop). That honesty — naming what the sim *cannot* establish — is the strongest signal a safety-critical org can get from an engineer.

---

## 14. Monitoring, drift, and re-validation

**[SIGNAL: re-validation-lifecycle]** Trust is time-varying. The world changes — new vehicle types, e-scooters, construction patterns, new cities — and the generator drifts as it's retrained. Validation is not one-time; it's a lifecycle.

- **Continuous sim-vs-real divergence monitoring:** track whether the gap between sim-predicted and road-observed metrics widens over releases. A widening gap is the earliest warning that the validated envelope no longer covers reality (and a flag for AV-overfits-sim, §11).
- **Envelope-coverage monitoring:** watch the real fleet's operating distribution; when it moves into regions the envelope doesn't cover (new ODD, new agent populations), trigger re-validation *before* trusting sim evidence there. The 2025 finding that WOMD itself drifts from naturalistic Phoenix behavior is the cautionary tale — even the *data* the generator imitates can fall outside the live envelope.
- **Re-validation triggers:** generator retrain, ODD expansion, a real-world surprise (a failure the sim said wouldn't happen), or a divergence-monitor alarm. Each re-opens the predictive-validity loop on the affected envelope region.

---

## 15. Tradeoffs taken and what would change them

I scoped the sim to **behavioral/reactive simulation for relative decisions (ranking, regression, failure discovery)** and explicitly refused to make it load-bearing for **absolute rare-event rates** — because the §3 arithmetic makes direct tail-rate validation infeasible and a self-built generative sim is least defensible exactly there. I prioritized **predictive validity over distributional similarity** because only decision-transfer is usable in a safety case. I weighted **recall over precision** for failure discovery because missed real failures are dangerous while false alarms are merely costly.

*What would change this:* if the decision were pure *training-data augmentation* (not validation), distributional coverage would matter more and predictive validity less. If we had access to a vastly larger independent real-world ground-truth corpus, the absolute-rate refusal could soften toward a bounded calibration claim. If the sim were perception-stack rather than behavioral, sensor-domain-gap validation (FID-style on sensor data, plus real-sensor replay) would re-enter as a first-class layer I deprioritized here.

## 16. What I'd push back on

**[SIGNAL: saying-no]** I'd push back on the premise that we should ask "is the simulator realistic?" at all — it's the wrong question and invites a single global trust claim the sim can't support. I'd push back on any plan that treats a low FVD or a high WOSAC Realism Meta as a trust certificate — both are tail-blind and gameable, and the "Beyond Simulation" result shows high-meta models failing exactly under the perturbations safety depends on. And I'd push back, gently but firmly, on any framing where the self-built sim is the *primary* evidence for rollout sign-off: it is powerful, scalable, bounded evidence for *relative* decisions within a validated envelope — and it is not, and cannot be, a proof of safety. The real world remains the irreplaceable arbiter, and the safety case has to say so out loud.
