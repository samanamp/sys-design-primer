---
title: "Distillation as a Design Problem: Building the Efficient Variant"
description: ML system design — the program that turns a 30B teacher world model into the cheap student that powers the simulation fleet
---

# Design the teacher→student distillation program for a world model

> **Interviewer prompt:** "You own the 30B teacher world model from the earlier design. Leadership has approved the 'efficient variant' strategy: a distilled student powers the bulk simulation fleet. Design the distillation program end to end — data, training, evaluation, and the ongoing process — not just the loss function."

*Interview-style answer. First-person, as the candidate. I talk through decisions out loud, flag tradeoffs explicitly, and mark **Staff-level signals** wherever they appear. Numbers are stated assumptions, consistent with designs #1 and #2.*

---

## 1. Clarify scope and assumptions

Questions I'd ask:

1. **What's the compression target?** Design #1's cost model wants ~150×: capacity (30B → ~3B) *and* steps (30 → ~4 denoising passes). Is that fixed, or do I get to trade quality against it per use case?
2. **Is the teacher frozen, or does it keep improving?** (It keeps improving — which changes this from a project into a *process*.)
3. **One student or a family?** (Bulk-fleet student, maybe a tinier triage variant later. I'll design for one and keep the pipeline reusable.)
4. **What's the acceptance bar?** Inherited from designs #1/#2: per-slice downstream agreement with the teacher, plus the validation tiers. I'm not inventing new gates; I'm building the thing that has to pass them.

**Assumptions:** 30B diffusion-based teacher (camera + lidar, 10 Hz, 30 steps), target student ≈ 3B at 2–4 steps, FP8 serving. Teacher retrains roughly quarterly. I control a distillation compute budget and the teacher pool's idle capacity.

---

## 2. Framing — what distillation actually has to preserve

The lazy framing is "make a small model match the big model's outputs." The correct contract, inherited from design #2's task-relative realism:

> The student must preserve the **distribution of futures** the teacher generates, well enough that the planner-under-test's evaluations are unchanged — per slice, including the tail.

Two failure modes follow directly from that sentence, and they're the two things naive distillation does worst:

1. **Oversmoothing (mode averaging).** The future is genuinely multimodal: the pedestrian at the curb either steps out or doesn't. A student trained with a regression-style loss against teacher samples learns the *average* of the modes — a half-committed ghost pedestrian that neither crosses nor stays. The planner reacts to that blur with unrealistic caution, and every downstream metric shifts. Sample-matching losses break exactly where simulation matters most: decision points.
2. **Mode dropping.** The student covers the common futures and quietly loses rare ones — the tail slices, again. Note the irony: the sharpest few-step distillation methods are *adversarial* (distribution-matching with a discriminator-like critic), which reintroduces the GAN-family failure we rejected in design #1's architecture table[^1]. Sharpness and tail coverage trade against each other, and the gate has to measure both.

**Staff-level signal:** state what must be preserved as a *distributional* property and connect each loss-function choice to a named failure mode and a named gate. "I'd use consistency distillation" is a junior answer; "here's which failure each candidate loss invites, and which eval catches it" is the staff answer.

[^1]: Design #1, footnote on **mode collapse**: a GAN's generator can win by producing a few safe outputs, dropping low-density modes — and the long tail *is* the low-density modes. Adversarial distillation objectives (e.g., distribution-matching distillation) inherit a milder version of the same risk. The standard mitigation is keeping a likelihood-style anchor term in the loss alongside the adversarial one — a plain reconstruction/matching penalty that charges the student *everywhere it deviates*, not just where the critic happens to look — plus per-slice tail metrics in the acceptance gate so any dropping is caught rather than assumed away.

---

## 3. The distillation dataset — where the design is actually won

### What goes in it

- **Stratified scenario coverage:** every slice × scenario-type cell filled deliberately — uniform-ish over *conditions*, not over miles driven (fleet-frequency sampling would hand the student a sunny-highway curriculum and starve the tail).
- **Deliberate tail oversampling:** the slices where agreement gates are tightest get the most teacher examples, not the fewest. The sampling weights come straight from design #2's per-slice acceptance bands — tightest band, highest weight.
- **The divergence queue from design #2:** every paired-replay case where the *teacher itself* was wrong-but-correctable is gold; every case where the previous student diverged from the teacher is targeted curriculum for the next one.
- **Multiple teacher samples per condition:** to teach a distribution you must show more than one draw from it — K samples per scenario seed at decision points, so the student sees that the pedestrian sometimes crosses and sometimes doesn't.

### The napkin math that reorders priorities

```
Generating fresh teacher data (design #1 rates: 18 GPU-hr, $36 per mile):
  50K stratified miles  →  ~900K GPU-hours  ≈  $1.8M

Training the 3B student on it:
  50K miles ≈ 60M frames ≈ 7.2e11 token-instances
  ~6 × 3e9 × 7.2e11 × ~2 epochs  ≈  2.6e22 FLOPs
  on 512 H100s @ 0.4 PFLOP/s     ≈  ~36 hours  ≈  18K GPU-hours  ≈  $37K
```

**Data generation outweighs training compute ~50:1.** Three consequences:

1. **Free-ride aggressively.** The teacher pool already produces rollouts as a byproduct — golden-scenario libraries, the standing audit stream, arbitration re-runs (design #1 §6). Log all of it in distillation-ready form. A large fraction of the dataset costs zero marginal dollars.
2. **Iterate on training, not data.** At $37K a run, training experiments are cheap; fresh data campaigns are not. Sweep losses and schedules freely; commission new teacher miles only when a slice gate *fails* and the failure is traced to coverage.
3. **The program pays for itself absurdly fast.** Running the bulk fleet on the teacher would cost $3.6M/day; on the student, $24K/day (design #1). The entire ~$2M program is paid back in under a day of fleet operation. Ask for the budget without apologizing.

**Staff-level signal:** doing the cost asymmetry math *before* designing the loss. The interviewer learns more from "data is 50× the training cost, so here's my data strategy" than from any amount of loss-function erudition.

---

## 4. Technique choices — and the order of operations

### Step distillation (30 → 2–4 passes)

```
teacher:      noise ●→●→●→●→●→●→ ... →●  clean   (30 small denoise hops)

progressive:  noise ●→──→●→──→●→ ... →●  clean   (student learns 2-hops:
                                                   30→15→8→4, retrain each halving)

consistency:  noise ●─────────────────►●  clean   (from ANY noise level,
                                                   jump straight to the answer)
```

- **Progressive distillation:** stable, well-understood; multiple rounds of training; quality degrades gracefully as steps halve.
- **Consistency-style distillation:** reaches 1–4 steps directly; tends to soften fine detail — watch lidar-camera consistency metrics, since "softened" geometry breaks cross-modal coherence before it looks bad to a human.
- **Adversarial / distribution-matching:** sharpest few-step results; carries the mode-dropping risk of footnote 1 and the training instability of its GAN ancestry. If used, anchor it with a regression term and gate it on tail slices.

### Capacity distillation (30B → 3B)

Output matching on teacher samples is table stakes. The decisions that matter:

- **Feature matching** (align intermediate activations, not just outputs) transfers more per example but requires an architectural correspondence between teacher and student layers — constrain the student to be a narrower/shallower sibling of the teacher, not an exotic new design, and this comes nearly free.
- **Trajectory-level supervision:** match multi-frame rollouts, not single frames, so temporal coherence is learned rather than hoped for.

### Order of operations

Capacity first, then steps: train the 3B student as an ordinary 30-step diffusion model against teacher outputs (a stable, well-posed target), *then* step-distill the student. Doing both jointly stacks two unstable optimizations; doing steps first means re-doing them after every capacity change. The boring sequencing is the right one.

### The on-policy correction loop — the piece most designs miss

Everything above is **off-policy** — borrowing RL's vocabulary: the training data comes from contexts the *teacher* generated, not from situations the student gets itself into. (**On-policy** = trained on data produced by the student's own behavior.) The problem: in production (closed-loop, design #2's distinction) the student consumes *its own* slightly-degraded history — and small errors compound into drift the teacher's data never showed it how to escape.

```
  ┌──────────────────────────────────────────────────────────┐
  │ 1. STUDENT rolls out N frames closed-loop (its own       │
  │    imperfect history — including its drift)              │
  │ 2. TEACHER, conditioned on that SAME student-generated   │
  │    history, produces the next-frame target               │
  │ 3. student is trained toward the teacher's recovery      │
  │ 4. repeat, mixing on-policy batches with the offline set │
  └──────────────────────────────────────────────────────────┘
```

This is the DAgger recipe from imitation learning[^2], transplanted: the expert labels the *student's* states, so the student learns to recover from mistakes only it makes. In world-model terms it directly attacks the error-accumulation curve — which is the metric (max trusted horizon, design #2 §3) where distilled students fail first.

**Staff-level signal:** off-policy distillation alone produces students that ace open-loop evals and drift in closed-loop. Naming exposure bias as the gap, and on-policy correction as the fix, connects the training design to the *deployment* failure mode — the full-system thinking the rubric asks for.

[^2]: **DAgger** ("dataset aggregation"): an imitation-learning loop where the learner acts, an expert labels the states the learner actually visited, and the learner retrains on the aggregate. The point: a policy trained only on expert demonstrations never sees the states its own mistakes create, so one error walks it off the training distribution — same mechanism as exposure bias in sequence generation.

---

## 5. The evaluation gauntlet — inherited, plus distillation-specific traps

The student passes through design #1 §7's gauntlet unchanged (regression suite → downstream agreement → drift horizon → shadow → canary). What distillation adds is *which* failures to hunt:

- **Mode coverage per decision point:** generate K rollouts per scenario seed from teacher and student; compare the *spread* of outcomes (did the student preserve the cross/don't-cross split, or average it?). A diversity metric per slice, gated.
- **Sharpness vs tail tradeoff:** adversarial-distilled students get an extra tail-slice audit at elevated sample sizes, per footnote 1.
- **Cross-modal coherence:** few-step students soften geometry first; lidar-camera consistency (design #2's L0) catches it before FVD does.
- **Drift horizon regression:** the closed-loop divergence curve (design #2 §3) re-measured per candidate; the max-trusted-horizon is recomputed and *may shrink* — which the scheduler must then enforce. A student that's 150× cheaper but trusted for 4 s instead of 8 s may or may not be a win; that's a portfolio decision, made explicit.
- **Controllability preservation:** the teacher's control surfaces — driving-action conditioning, scene-layout control, language control — are product features, and distillation can silently weaken them (a student can match unconditional realism metrics while half-ignoring its conditioning). Gate it directly: same control input, K seeds, compare teacher-vs-student *response* to the control, per control axis. A student that renders beautiful rain but won't produce rain on command is a regression no realism metric sees.
- **Gate what you ship:** the gauntlet runs on the FP8-quantized student exactly as served — not the BF16 training checkpoint. Quantizing *after* gating quietly un-gates the model (and per design #1 §10.2, quantization error lands hardest on the tail slices, exactly where the bands are tightest).

---

## 6. Distillation as a process, not a project

The teacher retrains quarterly; cities launch; the divergence queue never empties. If producing a student takes a bespoke heroic effort, the fleet runs stale models. So the deliverable is a **distillation CI pipeline**:

```
 teacher release (quarterly-ish)        divergence queue (continuous)
          │                                      │
          ▼                                      ▼
  ┌───────────────────────────────────────────────────┐
  │  DATA REFRESH — stratified set + byproduct logs   │
  │  + targeted miles for any slice that failed last  │
  │  cycle (and only those)                           │
  └─────────────────────────┬─────────────────────────┘
                            ▼
  ┌───────────────────────────────────────────────────┐
  │  TRAIN — capacity distill → step distill →        │
  │  on-policy correction rounds   (~$40K, ~2 days)   │
  └─────────────────────────┬─────────────────────────┘
                            ▼
  ┌───────────────────────────────────────────────────┐
  │  GAUNTLET — design #1 §7 + §5 traps above;        │
  │  auto-promote on green, auto-page on red          │
  └─────────────────────────┬─────────────────────────┘
                            ▼
              fleet rollout: shadow → canary → bulk
```

Operating notes:

- **Distillation lag is a tracked metric:** days between teacher release and student promotion. While lagging, the fleet runs the *old* student against the *old* teacher's gates — never a new-teacher/old-student mismatch, which silently invalidates the agreement baseline.
- **Slice failures trigger targeted data, not full campaigns:** a failed night-rain gate buys night-rain teacher miles, ~$50K, not a $1.8M refresh.
- **Every student is versioned with its teacher** (design #2's versioning discipline): a student is only meaningful relative to the teacher it was distilled from and gated against.

**Staff-level signal:** the question asks for a distillation *design*; the senior answer is mostly about the second and tenth iterations, not the first. One-shot distillation is a demo. The pipeline — lag metric, targeted refresh, auto-gauntlet — is the product.

---

## 7. When simpler wins

- If a workload needs only ~4× (not 150×), quantization + serving engineering (design #1's ladder, levels 1–2) gets there with no new model and no gauntlet. Don't distill for fun.
- If a use case doesn't need sensor realism at all, it belongs in object-level sim (design #1 §9) — the cheapest distillation is the one you skip.
- A *worse but honest* student beats a better but uncharacterized one: a student with a known 4 s trusted horizon is usable inside the portfolio today; one with great averages and unmeasured tail behavior is not usable for anything safety-adjacent.

---

## 8. Research pass — new developments (as of June 2026)

- **The sharpness-vs-tail trade now has a published mitigation:** [rCM](https://arxiv.org/abs/2510.08431) (NVIDIA, ICLR 2026) regularizes continuous-time consistency distillation with a score term — matching DMD2's quality while *explicitly mitigating mode collapse* and improving diversity, at 1–4 steps (15–50× sampling speedup), scaled to 10B+ video models. This is §2's footnote-1 tension, solved-ish in the literature: the frontier answer is consistency/score *hybrids*, not a pick between the adversarial and consistency columns.
- **Distillation training itself got cheaper:** SGMD (May 2026) reports ~3× training speedup over DMD2 with better motion dynamics in 4-step students — relevant to §3's economics, though data generation still dominates by an order of magnitude.
- **On-policy correction is now the named SOTA recipe, not a nice-to-have:** [Self-Forcing](https://arxiv.org/abs/2506.08009) (2025) trains the student on *its own* rollouts — §4's loop, as the paper's title concept — and [Causal Forcing](https://arxiv.org/abs/2602.02214) (Feb 2026) fixed the initialization pathologies in that family. The DAgger framing went from analogy to literature.
- **A fourth distillation axis emerged: factorization.** [CausVid](https://arxiv.org/abs/2412.07772)-style *asymmetric* distillation changes the student's generation order, not just its size: bidirectional teacher → **causal autoregressive student**, making the student streamable and KV-cacheable. For this program, that means the §4 sequencing gains a step — capacity → steps → *factorization* → on-policy — and the student inherits the serving benefits design #1 §11 now treats as central.

## 9. Summary — what I'd want the interviewer to remember

1. **The contract is distributional:** preserve the spread of futures per slice — the two named enemies are mode averaging (ghost pedestrians at decision points) and mode dropping (the tail, again).
2. **Data strategy beats loss strategy:** generation outweighs training ~50:1, so free-ride on teacher-pool byproducts, oversample by gate tightness, and let the $37K training runs iterate freely.
3. **Boring sequencing:** capacity-distill at full steps, then step-distill, then on-policy correction — never stack unstable optimizations.
4. **On-policy correction is the difference** between a student that aces open-loop evals and one that survives closed-loop; exposure bias is the named gap, DAgger-style teacher labeling of student states is the fix.
5. **Gate the distillation-specific failures explicitly:** outcome diversity per decision point, tail audits for adversarial losses, cross-modal coherence, re-measured drift horizon, controllability response per control axis — and run the gauntlet on the quantized artifact you actually ship.
6. **Build the pipeline, not the artifact:** quarterly teachers make distillation a CI process with a lag metric — the tenth student should be push-button.
