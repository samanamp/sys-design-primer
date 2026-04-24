---
title: Model Registry & Deployment System (Simplified)
description: Model Registry & Deployment System
---
```
"Design the system that manages ML model versions from training to production. Supports A/B testing, shadow deployment, rollback, and works across PyTorch, TF, and ONNX. 500 models in production, 50 deployments per day. Walk me through it."
```
---
# Model Registry & Deployment System — Staff-Level Answer (v2)

## 1. Scope and Reframing

Before I draw any boxes, I want to make three things explicit, because each one changes the design downstream.

### 1a. A model is not a file. It's a contract with lineage.

**[STAFF SIGNAL: models-are-not-binaries]**

The most common mistake on this question is to treat a model as a weights file that lives in S3 and gets copied to servers. That framing leads to a design that's a glorified file-copy service, and it will fail in production the first time a team changes a feature and doesn't understand why the model broke.

A model version is actually a bundle of things that all have to agree with each other:

- The **weights** (the binary file itself).
- The **feature contract** — what features the model expects, their types, their definitions. If the feature store changes a feature's definition, the model may silently break even though the weights never changed.
- The **preprocessing and postprocessing code** — usually separate repos, separate commits, and often the real source of bugs.
- The **training data version** — which snapshot of which tables did we train on.
- The **evaluation metrics** at training time, on a specific eval set.
- The **runtime requirements** — GPU class, memory, batch size assumptions.
- The **rollback properties** — can this model actually be rolled back cleanly? (More on this in §6.1.)
- The **downstream dependents** — who else consumes this model's outputs as training data or as an index?

The registry's job is not to store the weights. It's to track this bundle and enforce, at deploy-time and at every single request-time, that all parts still agree with each other. This reframing — models as contracts with lineage, not blobs in a bucket — drives the schema, the state machine, and the rollback story.

### 1b. "500 models" covers very different things. Commit to one.

**[STAFF SIGNAL: scope negotiation]**

"500 models in production" is underspecified. Realistically, the fleet includes:

- **Small tabular models** (GBDTs, linear): kilobytes to megabytes, stateless, easy case.
- **Deep ranking / recommendation models**: hundreds of MB to 10 GB. Weights are stateless, but the model's *outputs* change user behavior (personalization). This is the hard case.
- **Embedding models** feeding retrieval indices: the embedding space itself is a contract — downstream indices are tied to a specific version.
- **Continuously-updating models**: weights change every few minutes from streaming data. "Version" barely makes sense.
- **LLMs**: 10 GB to 500 GB, with special serving requirements (KV cache, continuous batching, paged attention) that break generic assumptions.
- **Batch-scoring models**: no request-response; they run in offline jobs.

I'm going to primarily design for **deep ranking/recommendation models, trained offline, deployed frozen, serving request-response traffic at 10–1,000 QPS each**. That's the center of gravity of a Meta/Netflix-class 500-model fleet, and it's where the genuinely hard problems live (rollback with state, shadow's counterfactual problem, A/B testing at scale). I'll flag in each section what changes for batch, for LLMs, and for continuous learners. I'll argue in §6.4 that LLMs should live in a sibling serving plane behind a shared registry API — not crammed into the same runtime.

### 1c. "50 deploys per day" is probably the wrong goal.

**[STAFF SIGNAL: saying no]**

Fifty deploys per day across 500 models is one deploy per model every ten days. That's either a lot of weight-refreshes (fine, make them boring) or a lot of real contract changes (alarming). The platform's job isn't to make 50 heroic deploys per day work — it's to make 90% of them *boring non-events* that auto-promote with no human involvement, and to make the remaining 10% explicit, gated, and visible. I'll design for that target, not for 50 hand-approved deploys.

### 1d. Rollback is the hardest problem in this design. I'll say so upfront.

**[STAFF SIGNAL: rollback honesty]**

For stateless models, rollback is trivial: point traffic at the old weights. For models whose outputs have shaped user state, trained downstream systems, or updated features over time, "rollback to the pre-deployment world" is sometimes *not possible at all*. Any design that promises clean rollback for every model is lying. The architecture has to classify each model's rollback viability at deploy time and refuse to promise what it can't deliver. I cover this in §6.1.

---

## 2. Capacity Math, Shown Step by Step

**[STAFF SIGNAL: capacity math]**

I'll walk through the numbers that drive the design. Each one has an assumption, a formula, and a result.

### 2.1 Deploy rate and review budget

Assumption: 50 deploys/day, concentrated in ~10 working hours (not uniform over 24h).

```
  Average rate (uniform):       50 / 24h = 2.08/hour = one every ~29 minutes
  Peak rate (working hours):    50 / 10h = 5.0 /hour = one every ~12 minutes
```

If we target **90% automation**, we get **5 deploys/day needing human review**. At 30 minutes of reviewer attention per gated deploy, that's **2.5 reviewer-hours/day** — tractable, distributed across model owners.

If we targeted only **50% automation**, we'd need **25 reviews/day × 30 min = 12.5 reviewer-hours/day**. That's a full-time job for someone, and it's also the rate at which reviewers start rubber-stamping. So the 90% automation target isn't aspirational — it's forced by the review-capacity arithmetic.

### 2.2 Storage

Assumption: we retain 10 active versions per model (current prod, previous prod for rollback, 2–3 canaries/shadows, a few recent archived), plus cold-storage history.

Model size distribution — I'll assume a plausible mix:

```
  60% of models are small (linear, small DNN):   10 MB avg
  30% are medium (ranking DNNs):                  1 GB avg
  10% are large (LLMs, large embedding):         50 GB avg

  Weighted average per version:
  = 0.60 × 10 MB  + 0.30 × 1000 MB + 0.10 × 50000 MB
  = 6 MB           + 300 MB         + 5000 MB
  ≈ 5.3 GB per version

  Hot storage (10 versions × 500 models × 5.3 GB):
  = 26,500 GB = 26.5 TB
```

On S3-class storage at ~$0.023/GB-month, that's ~$600/month. Cheap. **Storage is not the constraint.** This is important, because it means we can afford to retain more versions than strictly necessary, which buys us rollback optionality.

Cold storage for full history (say 5× the hot storage, compressed): ~130 TB at ~$0.004/GB-month ≈ $520/month. Also cheap.

### 2.3 Shadow traffic cost

Assumption: a model runs at 100 QPS average. We shadow 10% of its traffic for one week to validate a new version.

```
  Shadow QPS per model:          100 × 0.10 = 10 QPS
  Shadow inferences/week:        10 × 604,800 s = 6.05 M inferences

  Cost per inference (GPU amortized, rough):
  One T4-class GPU serves ~100 QPS for ~$0.35/hr (spot)
  Cost per inference = $0.35 / (100 × 3600) ≈ $1e-6 per inference

  Shadow cost per model-week:    6.05M × $1e-6 ≈ $6

  If 10% of fleet is shadowing at any time (50 models):
  = 50 × $6 = $300/week ≈ $1,200/month
```

Cheap. Even at 10× these rates for large GPU inference on big models, we're at $12k/month fleet-wide. **Shadow is affordable at this scale** — the reason to be selective about shadow is not cost but *signal quality* (see §6.2 on the counterfactual problem).

### 2.4 A/B testing sample sizes

This is the most important piece of capacity math, because it sets the minimum time-to-decision.

Formula for a two-sample proportion test (α=0.05, power=0.80):

```
  n_per_arm ≈ 16 × p(1-p) / δ²
  where p = baseline rate, δ = absolute effect size you want to detect
```

Worked example — detecting a 1% *relative* lift on a 5% baseline CTR:

```
  p = 0.05
  δ = 0.05 × 0.01 = 0.0005    (1% relative of 5% absolute)

  n = 16 × 0.05 × 0.95 / (0.0005)²
    = 16 × 0.0475 / 0.00000025
    = 0.76 / 0.00000025
    ≈ 3,040,000 users per arm
```

Duration math, at 1M DAU on the model with 50/50 split:

```
  Users per arm per day:    500,000
  Days to reach n:          3,040,000 / 500,000 = 6.08 days (minimum)
  With weekly seasonality:  14 days recommended
```

**Implication:** Even a generous A/B for a modest effect takes two weeks. If we tried to A/B every single deploy, we'd need ~50 concurrent experiment slots × 2 weeks = ~700 experiment-weeks of running experiments at any given moment. That's not feasible. **So the system must commit to tiered evaluation:** most deploys pass offline + automated online gates (canary with guardrails, no full A/B); only contract-changes and high-tier models get the 14-day A/B ride.

### 2.5 Rollback time budgets

Stating these as SLOs, to be defended later:

```
  Stateless rollback:              ≤ 5 minutes  (weights redeploy + traffic cutover)
  State-snapshot rollback:         ≤ 30 minutes (snapshot restore + weights redeploy)
  Irreversible rollback:           undefined    (see §6.1 — we're honest about this)
```

### 2.6 Capacity summary table

| Metric | Value | Notes |
|---|---|---|
| Deploy rate | 50/day (peak 1/12 min) | |
| Human-review deploys | 5/day (10% of total) | 2.5 reviewer-hours/day |
| Hot storage | 26.5 TB | ~$600/month; not a constraint |
| Cold storage | ~130 TB | ~$520/month |
| Shadow cost | ~$1,200/month fleet-wide | signal quality is the real constraint |
| A/B minimum duration | 14 days | enforces tiering |
| A/B concurrent capacity | ~10 at a time | forces most deploys to skip full A/B |
| Stateless rollback TTR | 5 min | SLO |
| Stateful rollback TTR | 30 min | SLO |
| Irreversible rollback | no SLO | refuse to promise |

---

## 3. High-Level Architecture

Here's the full picture before we drill in. The key claim: the registry is thin and sits at the junction of four adjacent systems. It doesn't own them.

```
 ┌────────────────────────────────────────────────────────────────┐
 │                    TRAINING + OFFLINE EVAL PLANE               │
 │  Orchestrator (Airflow/Flyte) → Trainer → Offline Evaluator    │
 │            │                │                │                 │
 │            ▼                ▼                ▼                 │
 │   (training_data_ver)  (weights blob)  (eval_metrics/slices)   │
 └────────────┬────────────────┬────────────────┬─────────────────┘
              │                │                │
              │   ┌────────────┴────────────────┴──────┐
              │   │         MODEL REGISTRY             │
              │   │  metadata DB + artifact references │
              │   │  + lineage graph + contract store  │
              │   │  + state-machine authority         │
              │   └────┬──────────────────────┬────────┘
              │        │                      │
              │        │  (read: contract)    │  (state transitions,
              │        │                      │   gate results,
              │        ▼                      │   audit log)
              │   ┌──────────────────┐        │
              │   │ FEATURE STORE    │◄───────┤
              │   │ (Feast/Tecton)   │  contract validation:
              │   │ versioned schema │  "does feature v7 still
              │   └──────────────────┘  satisfy the model's needs?"
              │
              ▼
        ┌──────────────────────────────────────────────┐
        │           DEPLOYMENT CONTROL PLANE           │
        │  Deploy Controller (K8s operator pattern)    │
        │  Traffic Manager (service mesh splits)       │
        │  State-Machine Enforcer                      │
        └───┬─────────────┬─────────────┬──────────────┘
            │             │             │
            ▼             ▼             ▼
  ┌──────────────┐ ┌──────────────┐ ┌──────────────────┐
  │ DNN SERVING  │ │ LLM SERVING  │ │ BATCH SCORING    │
  │ (Triton)     │ │ (vLLM /      │ │ (Spark/Ray jobs) │
  │ TF,PT,ONNX   │ │  TRT-LLM)    │ │                  │
  └──────┬───────┘ └──────┬───────┘ └──────┬───────────┘
         │                │                │
         └────────┬───────┴────────────────┘
                  ▼
        ┌────────────────────────┐
        │  ONLINE EVAL / OBSERV. │
        │  prediction logging    │
        │  input/output drift    │
        │  per-slice metrics     │
        │  experiment framework  │
        └────────┬───────────────┘
                 │
                 │ (promotion signal,
                 │  regression alert,
                 │  rollback trigger)
                 ▼
           REGISTRY STATE MACHINE
```

What each piece does, in plain terms:

- **Registry** = the metadata source of truth. It knows what a model version is, what it references, what state it's in, who approved what. It does *not* store the weights or the features. It holds pointers to those and validates them.
- **Training plane** = where models get trained. Emits events that the registry ingests.
- **Deployment control plane** = an operator (in the Kubernetes sense) that watches the registry for "desired state" and reconciles the actual serving infrastructure to match.
- **Serving plane** = three separate runtimes, on purpose. Triton handles conventional DNNs (PyTorch, TF, ONNX) efficiently. vLLM or TensorRT-LLM handles LLMs. Spark/Ray handles batch. One registry API, three backends. **[STAFF SIGNAL: framework heterogeneity discipline]**
- **Online eval** = the feedback loop. Metrics flow back to the registry, which uses them to trigger promotions and rollbacks.

---

## 4. Registry Schema and Lineage

**[STAFF SIGNAL: lineage as first-class]**

Here's the minimum information a model version record carries:

```
ModelVersion {
  model_id:                 str     # the logical model, stable across versions
  version:                  int     # 1, 2, 3...
  weights_uri:              str     # pointer to blob store, content-addressed
  weights_hash:             str     # sha256 of the weights
  framework:                enum    # pytorch | tf | onnx | vllm_engine
  runtime_requirements:     {gpu_class, memory_gb, compute_capability}
  feature_contract_ref:     str     # pointer into feature store's schema
  preprocess_code_ref:      (repo, commit_sha, path)
  postprocess_code_ref:     (repo, commit_sha, path)
  training_run_id:          str     # pointer to orchestrator run
  training_data_version:    str     # pointer to dataset snapshot
  training_code_version:    (repo, commit_sha)
  parent_model_version:     str|null # if warm-started from another model
  eval_set_version:         str     # which eval set produced metrics
  eval_metrics:             {...}
  eval_slice_metrics:       {slice: {...}}   # per-country, per-language, etc.
  rollback_class:           enum    # STATELESS | STATE_SNAPSHOT | IRREVERSIBLE
  rollback_snapshot_uri:    str|null
  tier:                     enum    # L0..L3 (see §6.5)
  state:                    enum    # state machine (see §5)
  state_history:            [{state, timestamp, actor, gate_results}]
  audit_log:                append-only
}
```

Two invariants that the registry enforces and never relaxes. **[STAFF SIGNAL: invariant-based thinking]**

**Invariant 1: the contract must match at request time.** Before a request reaches model version V, the serving layer checks that the feature store still satisfies V's `feature_contract_ref`. If a feature got dropped or its type changed, the request fails closed or falls back — depending on the model's tier. The invariant catches the silent failure where "the model's fine, but feature_B changed shape yesterday."

**Invariant 2: lineage is append-only. References don't break.** A training dataset referenced by any non-archived model cannot be deleted. The registry holds reference counts against the dataset registry. If someone tries to delete a referenced dataset, it fails. This isn't just operational hygiene — under the EU AI Act (Article 11, technical documentation requirement for high-risk AI systems), training-data provenance is a regulated asset, and deleting a dataset referenced by a deployed model becomes a compliance event. **[STAFF SIGNAL: modern awareness]**

### Example lineage graph

Concrete example — what the lineage looks like for a real deployed ranking model:

```
                     ┌─────────────────────────────┐
                     │ model_v47 (state=PROD)      │
                     └──────┬──────┬──────┬────────┘
                            │      │      │
               ┌────────────┘      │      └──────────────┐
               ▼                   ▼                     ▼
    ┌──────────────────┐  ┌────────────────┐   ┌──────────────────┐
    │ weights_blob     │  │feature_contract│   │ parent_model_v44 │
    │ sha256:af3…      │  │ ref=fs://v7    │   │ (warm-started)   │
    └──────────────────┘  └───────┬────────┘   └─────┬────────────┘
                                  │                   │
                          ┌───────┴────────┐          │
                          ▼                ▼          ▼
                ┌──────────────┐ ┌──────────────┐  (recursive...)
                │ feature_A v3 │ │ feature_B v7 │
                │ def@repo:sha │ │ def@repo:sha │
                └──────┬───────┘ └──────┬───────┘
                       │                │
                       ▼                ▼
                ┌──────────────────────────┐
                │ training_data_ver=ds:9f1 │
                │ (lake snapshot)          │
                └──────────────────────────┘

  Also referenced (not shown): eval_set_version, training_code_commit,
  preprocess/postprocess code commits, rollback snapshot (if any),
  and — crucially — downstream_dependents (who consumes this model's outputs).
```

The `downstream_dependents` edge is the one most systems forget, and it's the one that matters for rollback. When you roll v47 back to v44, the graph tells you *what else is now inconsistent*: a retrieval index that was rebuilt against v47's embeddings, a downstream ranker that was retrained on v47's scores. Without this edge, rollback is blind.

---

## 5. Deployment State Machine

Every model moves through a defined set of states, and transitions are gated. The gates are automated for low-risk models and require human sign-off for high-risk ones.

```
                                     ┌──────────────┐
                                     │  REGISTERED  │
                                     └──────┬───────┘
                                            │ offline eval passes automated gates
                                            ▼
                                     ┌──────────────┐
                                     │OFFLINE_PASSED│
                                     └──┬──────┬────┘
                       fails fairness/  │      │
                       slice-regression │      │
                           gate         │      │ passes
                                        ▼      ▼
                                   ┌────────┐  ┌─────────────────┐
                                   │ FAILED │  │ APPROVED_SHADOW │
                                   └────────┘  └────────┬────────┘
                                                        │ infra health ok
                                                        ▼
                                                ┌───────────────┐
                                                │    SHADOW     │──┐
                                                │  (≤7 days)    │  │ divergence
                                                └───────┬───────┘  │ beyond
                                                        │          │ threshold
                                               passes   │          ▼
                                               shadow   │      ┌──────┐
                                               checks   │      │FAILED│
                                                        ▼      └──────┘
                                                ┌───────────────┐
                                                │APPROVED_CANARY│
                                                └───────┬───────┘
                                                        │
                                     (tier gate: L2/L3 → human approval)
                                                        ▼
                                           ┌──────────────────────┐
                                           │    CANARY 1%→10%     │──┐ auto-
                                           │   (online eval SLO)  │  │ regression
                                           └─────────┬────────────┘  │ → rollback
                                                     │               ▼
                                                     │           ┌────────┐
                                     (tier gate)     ▼           │ROLLBACK│
                                           ┌──────────────────┐  │(see §6)│
                                           │     AB_TEST      │  └────────┘
                                           │  50/50, ≥14 days │──┐
                                           └─────────┬────────┘  │ metric
                                                     │           │ regression
                                                     ▼           ▼
                                             ┌──────────┐    ┌────────┐
                                             │ PROMOTED │    │ROLLBACK│
                                             └─────┬────┘    └────────┘
                                                   │
                                                   ▼
                                             ┌──────────┐
                                             │DEPRECATED│
                                             └─────┬────┘
                                                   ▼
                                             ┌──────────┐
                                             │ ARCHIVED │
                                             └──────────┘

  A model can be in SHADOW for one traffic slice while a sibling version is
  in AB_TEST for another slice. State is per-(model, slice); the registry
  tracks the full Cartesian product.
```

A few things about this design worth calling out:

- **Every forward transition has a matching backward edge.** You can fail out of any state.
- **The state machine is not flat.** Low-tier (L0) models skip straight from `OFFLINE_PASSED` to `PROMOTED`. High-tier (L2/L3) walk every state. Flattening the state machine to "simplify" it, which I've seen proposed many times, just deletes the places where gates attach — and those gates are the whole point.
- **Shadow and A/B can run in parallel** for different slices of a model. The registry supports this; the experimentation platform has to be aware.

---

## 6. Deep Dives

### 6.1 Rollback for stateful models — the hardest problem

**[STAFF SIGNAL: rollback honesty]** **[STAFF SIGNAL: failure mode precision]**

Rollback has three classes, and the registry forces each model to declare which class it belongs to at deploy time. This isn't just metadata — it determines what the system promises.

#### Class A: STATELESS

The model has no persistent state. Its outputs aren't fed back as training data. Users have no memory of its specific behavior.

Rollback = repoint the traffic manager to the previous version's pods, drain the old pods. Done in ~5 minutes. Easy. This is the minority of the fleet at 500-model scale, unfortunately.

#### Class B: STATE_SNAPSHOT

The model reads or writes state that lives outside its weights — a per-user embedding cache, a counter table, a calibration table, a deduplication Bloom filter. Rollback needs to restore that state alongside the old weights.

The discipline: at every promotion to CANARY and AB_TEST, the deploy controller forces a snapshot of all externalized state, content-addressed, stored with `rollback_snapshot_uri`.

Rollback = redeploy old weights + restore snapshot. Target: 30 minutes, dominated by how long the snapshot takes to restore.

Failure mode: the snapshot was taken at time T, then state continued mutating during the canary/A-B period as live writes went through. Rolling back to the snapshot loses those writes. Fixes are either (a) quiesce writes during snapshot (unacceptable in production) or (b) keep a write-ahead log with a cursor and replay selectively (complex and brittle). Most teams pick (b) and accept the complexity.

#### Class C: IRREVERSIBLE

This is the case that mid-level answers skip. There are three sub-cases, and they all make "rollback" a misleading word.

**C1 — Personalization leakage.** Model v47 is promoted. For a week, users are shown v47's recommendations. They click on them, scroll past them, dwell on them. All of that behavior becomes part of each user's feature vector (recent-click history, dwell-time, category preferences). A week later we roll back from v47 to v44. v44's weights are restored, but the features v44 now receives have been shaped by v47's recommendations. **You cannot un-show the user what they saw.** v44's post-rollback performance will be different from v44's pre-v47 performance, probably worse for a couple of weeks while the user state decays.

**C2 — Online-learning drift.** A CTR model updates continuously from streaming impressions. Its weights at time T+7 days are, by design, different from its weights at T. "Rollback" is ambiguous. Do we roll back to the snapshot at T (discarding a week of legitimate learning from real distribution shift)? Or to a hypothetical "what the weights would have been without the bad change" (which doesn't exist anywhere)? The question is unanswerable without more discipline (see below).

**C3 — Downstream training contamination.** Model v47's scores were logged and used to train downstream ranker R_v3, which was itself promoted. Now we roll v47 back to v44. R_v3 is still in production, but it was trained on a score distribution that no longer exists. Either we cascade-rollback R_v3 (destructive, and probably the wrong call), or we accept that the system is in a hybrid state and mark R_v3 as needing retraining.

#### Architectural disciplines that make rollback *more* feasible

None of these make Class C easy. They just make it less bad.

1. **Externalize state.** Keep stateful components outside the weights. Snapshot them independently. The registry refuses to promote a model to L2/L3 tier with `rollback_class=IRREVERSIBLE` without an explicit signed override. **[STAFF SIGNAL: invariant-based thinking]**

2. **Bound online-learning windows.** Continuous learners get hard checkpoints every N hours. Production always serves from the most recent checkpoint, never from a continuously mutating blob. "Rollback" means "serve an older checkpoint." **[STAFF SIGNAL: continuous-learning edge case]**

3. **Track downstream dependents in the lineage graph.** On rollback, compute the transitive closure. Default: warn the operator about everything in the closure. Don't auto-cascade — cascading rollback is usually worse than the disease.

4. **Be honest in the UI.** Models with `rollback_class=IRREVERSIBLE` get a visible badge at promotion time that says "This model cannot be cleanly rolled back. Promotion is close to a one-way decision." The system never pretends otherwise. **[STAFF SIGNAL: saying no]**

#### Rollback scenario diagram

Concrete walkthrough: v47 is a Class C model (ranking, personalization, downstream-training), promoted a week ago, now declared bad.

```
  t = 0       v47 promoted to 100% traffic.
              v44 pods drained. v44 weights retained in registry.
              v47's outputs flow to:
                - end users (as recommendations)
                - request logs, joined with clicks → training data for R_v3
                - feature store: "recent recommendations" feature updated per user
                - nightly rebuild of embedding retrieval index

  t = 7 days  v47 declared bad. Rollback requested.

  Action cascade (computed from the downstream_dependents graph):
   ┌─────────────────────────────────────────────────────────────┐
   │ 1. Traffic manager: switch v47 → v44 (5 min, weights cutover)│
   │                                                              │
   │ 2. Feature store: the "recent recommendations" feature is    │
   │    now v47-contaminated for 7 days of users. NO ACTION —     │
   │    we can't un-write user history. Mark v44 as running with  │
   │    a shifted input distribution; adjust alert thresholds.    │
   │                                                              │
   │ 3. Downstream ranker R_v3 was trained on v47's scores. Mark  │
   │    R_v3 as NEEDS_RETRAIN, but don't auto-roll-back R_v3 —    │
   │    the cascading rollback risk is worse than the status quo. │
   │                                                              │
   │ 4. Embedding retrieval index was rebuilt on v47 embeddings.  │
   │    Roll the index back to its pre-v47 snapshot (this is      │
   │    state-snapshotted, so it's cheap).                        │
   │                                                              │
   │ 5. Post-rollback monitoring: expect v44's metrics to be      │
   │    worse than its pre-v47 baseline for ~14 days while user   │
   │    state decays. Tune alert thresholds to avoid a spurious   │
   │    second rollback of v44.                                   │
   └─────────────────────────────────────────────────────────────┘

  What "rollback" actually delivered:
    - User-facing serving path:       restored (step 1)
    - Input distribution:             permanently perturbed (step 2)
    - Downstream ranker consistency:  broken, flagged for retrain (step 3)
    - Embedding index:                restored (step 4)
    - Audit trail:                    complete
```

The mid-level answer is step 1. The staff answer is all five steps, plus the honest acknowledgment that step 2 is permanent.

### 6.2 Shadow deployment — mechanics and the counterfactual trap

**[STAFF SIGNAL: counterfactual awareness]**

Shadow deployment means: the new model receives real production traffic, produces predictions, but its predictions don't affect users. They're logged for comparison against the current production model.

#### Mechanics

```
     Incoming request R
           │
           ▼
    ┌──────────────┐   mirror (async, out-of-band)
    │ Traffic mgr  │──────────────────────────┐
    └──────┬───────┘                          │
           │ 100% → prod (v44)                │ 10% → shadow (v47)
           ▼                                  ▼
    ┌──────────────┐                  ┌──────────────┐
    │ v44 serving  │                  │ v47 serving  │
    └──────┬───────┘                  └──────┬───────┘
           │ response → user                 │ response → logged only
           │ prediction → log (key=req_id)   │ prediction → log (key=req_id)
           └────────────────┬────────────────┘
                            ▼
                  ┌───────────────────┐
                  │ Divergence engine │
                  │ - agreement rate  │
                  │ - KL divergence   │
                  │ - per-slice diff  │
                  │ - calibration     │
                  │ - latency, errors │
                  └───────────────────┘
```

Two implementation options:
- **Live mirror**: duplicate the real request in real time, send to both models. Fresh data, but doubles compute on the shadowed fraction.
- **Log replay**: periodically replay a sample of production requests through the shadow. Cheaper and repeatable, but you have to log the feature values the production model saw at request time — otherwise the shadow sees different features than production did. (This is why point-in-time feature logging is a big deal, and it's where the registry and feature store APIs can't stay independent.)

#### When shadow is genuinely useful

- **Infrastructure validation.** Does v47 crash? OOM? Exceed the latency SLO on real traffic shape? Shadow catches this cheaply.
- **Calibration and output-agreement.** For classifiers or scorers where the output itself is the product (fraud scores consumed by a downstream rule engine), shadow gives you a real measurement of how the two models' outputs diverge at the same input.
- **Per-slice divergence detection.** You can measure how the new model behaves on a specific country, language, or content category — without exposing any user to it.

#### When shadow is misleading — the counterfactual problem

Here's the trap. Consider a recommendation model:

- v44 recommends item A. v47 would have recommended item B.
- The user is shown A (because v44 is prod).
- The user clicks A, spends 3 minutes on A, buys A.
- Shadow logs: "v47 would have recommended B." And... nothing else. We have no idea what the user would have done if shown B. We can't measure it.

Shadow shows us **divergence at fixed context**. It does *not* show us what would have happened in the counterfactual world where v47 was actually serving. All the closed-loop metrics that actually matter — CTR, session length, retention, revenue — are invisible to shadow for any model whose outputs change what users do next.

This isn't a bug we can fix with more engineering. It's a property of the system: a recommendation model's outputs change the user's behavior, and shadow can only observe the world as it actually was. So:

**What I commit to in this design:** shadow is positioned as *infrastructure and calibration validation*, not as a predictor of online metric movement. For recommendation models, the only honest evaluator of online metrics is A/B testing. If a team wants a "safer, longer shadow" as a replacement for A/B, I push back — that's asking the system to lie. **[STAFF SIGNAL: saying no]**

For the top-tier models where A/B is too slow, we can invest in counterfactual estimators (inverse propensity scoring, doubly-robust estimators). They help but have high variance on the tail, and they require logging propensity scores at serving time, which is a non-trivial infrastructure commitment. I'd build those only for L3 critical models, not as a default.

### 6.3 A/B testing and online evaluation

**[STAFF SIGNAL: evaluation divergence]**

#### Traffic splitting

Per-user sticky assignment, by hash. Not per-request. Reasons:

- Personalization state accumulates per user; per-request splitting contaminates both arms.
- Session-level metrics (session length, retention) require within-user consistency.
- Concurrent experiments need coherent bucketing across the experimentation platform.

#### Sample size and duration

Math done in §2.4: 3M users per arm for a 1% relative lift on a 5% baseline, ~6 days minimum at 1M DAU with 50/50 split, 14 days recommended to capture weekly seasonality. I enforce a **14-day minimum for contract-changing L2/L3 deploys** unless the effect size is large enough that a shorter window is well-powered.

#### The multi-metric problem

A new model ships with: CTR +1.2%, session length −0.4%, revenue +0.3%, content diversity −2%. Is that a win?

The registry can't decide this. What it can do is store, per model, a **decision policy** declared by the owning team:
- One primary metric.
- A set of guardrails with fail-thresholds.

Promotion rule: primary metric is significantly positive AND no guardrail trips. If any guardrail trips, it goes to human review. The registry doesn't invent decisions — it enforces the policy the team declared in advance, which is how we keep the decision auditable rather than political.

#### Offline vs. online metric divergence

This is a systemic problem, not an incidental one. A model that wins offline (higher AUC on the eval set) often doesn't win online. Why:

- The eval set was built from production traffic, which was generated by the *previous* model. So it has the previous model's blind spots baked in.
- Offline metrics (AUC, NDCG) don't model user behavior — they model the labels, which are proxies.
- Long-term effects (user retention, diversity) can't be captured in a point-in-time eval set.

Two mitigations:
- Maintain a **held-out counterfactual eval set**, logged from a small exploration bucket that gets served a near-uniform policy. This is expensive (it's a small permanent regression on user experience) but it's the only way to get an eval set uncontaminated by the production feedback loop. This is a business-wide policy decision, not a registry one.
- Track the **offline-online correlation over time** per model family. If we empirically know that offline AUC gains correlate with online CTR gains 60% of the time for ranking models, we can calibrate how much to trust offline wins.

#### Auto-promotion rules

- **L0/L1 models:** auto-promote on passing automated gates. 48-hour post-promotion monitoring window with auto-rollback on guardrail breach.
- **L2/L3 models:** human approval required on promotion. No auto-promote, full stop.

### 6.4 Framework heterogeneity — PyTorch, TF, ONNX, and the LLM wedge

**[STAFF SIGNAL: rejected alternative]**

The prompt asks for support across PyTorch, TensorFlow, and ONNX. Four options:

**Option A: Everything on NVIDIA Triton Inference Server.**
Triton supports TF, PyTorch (via TorchScript or Python backend), ONNX, and custom backends. Multi-model-per-process gets GPU utilization to 40–60% (vs. 10–20% for container-per-model). Dynamic batching helps moderate-QPS models. **Rejected as sole backend** because: (a) Triton's LLM story via TensorRT-LLM works but lags dedicated LLM runtimes on continuous batching, paged attention, and speculative decoding; (b) framework-specific custom ops are awkward and may force the Python backend, which loses the multi-model efficiency advantage. **Kept as primary backend for traditional DNNs.**

**Option B: Per-framework runtimes (TorchServe, TF Serving, ONNX Runtime Server) behind a gateway.**
Full framework fidelity. **Rejected** because N runtimes means N-fold SRE load, and the gateway becomes a routing ball of string. Only worth it if framework-specific features are load-bearing, which for ranking models they usually aren't.

**Option C: Convert everything to ONNX, single runtime.**
**Rejected** because ONNX conversion is lossy for cutting-edge models — attention variants, custom CUDA kernels, FP8/MXFP4 quantization often don't cross over cleanly. Forcing ONNX on the training team is a tax that makes good teams route around the platform. That's the worst outcome.

**Option D: Per-model container (BentoML, Ray Serve-style).**
**Rejected as default** because GPU utilization is unacceptable at this scale; storage and cold-start costs dominate. **Kept as escape hatch** for ~10% of models whose custom-op requirements don't fit Triton.

**Final commitment:**

| Model class | Runtime | Reason |
|---|---|---|
| Traditional DNN (TF, PT, ONNX) | Triton | GPU efficiency, multi-framework |
| LLMs | vLLM or TensorRT-LLM | Paged attention, continuous batching |
| Batch scoring | Spark/Ray | No request-response loop |
| Custom-op escape hatch | Ray Serve | ~10% of models |

The **registry API is uniform**; the runtimes are not. Pretending LLMs are just another DNN framework under one abstraction is the generalization trap, and you pay for it in 2–3× worse latency or 2–3× worse cost. The registry has a `model_class` field, and the deployment controller dispatches to the right runtime. **[STAFF SIGNAL: framework heterogeneity discipline]**

**LLM-specific note:** the registry's "weights + contract" schema mostly survives, but `runtime_requirements` gets richer for LLMs (KV cache memory per sequence, max context length, tokenizer version, draft model reference for speculative decoding). And the state machine's canary stage has to consider context-length distribution in the canary vs. prod — a model that looks fine on 4K-token requests can fall over on 128K-token requests that weren't well-represented in the canary sample.

### 6.5 Governance at 500-model scale

**[STAFF SIGNAL: governance at scale]**

We established in §2.1 that 90% of deploys must be automated or we drown. The structure that makes that safe:

#### Tiering

- **L0 — experimental / internal only.** No user exposure. Batch scoring to internal dashboards. Approval: none, just automated offline gates. Deploys are non-events.

- **L1 — user-facing, non-revenue.** Content sorting on unmonetized surfaces. Approval: automated gates only (regression thresholds, fairness across declared protected slices, latency/memory SLO). No human in the loop for weight refreshes on unchanged contracts.

- **L2 — user-facing, revenue or retention impact.** Ranking, recommendations, search. Approval: automated gates + human review at promotion + required 14-day A/B for contract-changing deploys. Weight refreshes on unchanged contracts can still auto-promote if canary guardrails hold.

- **L3 — safety, fraud, compliance, medical, moderation.** Approval: human review at every state transition. No auto-promotion. Secondary reviewer required from a different team. Every deployment is a logged audit event with written justification.

#### Audit logging

Every state transition, every gate result, every approver identity, every rollback — written to an append-only audit log keyed by `(model_id, version, action, timestamp)`. This is the source of truth for EU AI Act Article 11 documentation (high-risk AI systems require maintained technical documentation of training methodology, data provenance, and post-deployment monitoring) and for internal postmortems.

#### Emergency path

The system supports a `break-glass` deployment mode for urgent hotfixes. It:

- Requires two senior approvers (not one).
- Bypasses the shadow-duration and canary-duration gates.
- Forces a 24-hour heightened-monitoring state post-deploy.
- Auto-generates a postmortem ticket.
- **Is rate-limited**: a team that uses break-glass more than twice in 30 days gets locked out pending a process review.

Without the rate limit, break-glass becomes the normal path and governance collapses. With it, break-glass is the escape valve it's supposed to be.

---

## 7. Observability: "Why is model v47 behaving differently today?"

This is the hardest operational question in ML, and the lineage graph is what makes it answerable.

Worked example. Alert: v47's predicted CTR (tracked continuously post-deploy) is down 2% day-over-day.

Diagnosis walks the lineage graph:

1. **Did the model change?** Check `weights_hash`. Unchanged. Not a weight issue.
2. **Did the inputs change?** Per-feature distribution monitoring. `feature_B`'s histogram shifted on the Brazil slice. Cross-reference feature_B in the feature store — did its upstream source change? Yes: feature_B's upstream logging added a new enum value yesterday, which shifted the distribution.
3. **Who else depends on feature_B?** Lineage reverse lookup. Fourteen other models depend on feature_B. Four are L2. Alert their owners.
4. **Did outputs shift?** Prediction histogram shows a long-tail shift on the Brazil slice.
5. **Did downstream consumers notice?** The downstream ranker is clipping v47's scores differently because v47's output range moved.
6. **Did the metric definition change?** Check metric-definition version. Unchanged.

Root cause: feature_B's upstream logging change. The contract check didn't catch it because the schema didn't change — only the distribution did.

This is a known and accepted weakness: **schema match is easy to enforce; distribution match is hard.** We monitor distributional drift and alert on it. We don't hard-enforce it, because a lot of distribution drift is legitimate (e.g., real user-behavior shifts). Hard-enforcing would cause constant false rollbacks.

Per-slice monitoring is non-negotiable at L2/L3. A model that looks fine globally but has regressed for one country is a real problem that aggregates hide — and it's also the typical shape of a fairness regression.

---

## 8. API Boundaries With Adjacent Systems

**[STAFF SIGNAL: API boundary clarity]**

The registry is the hub, but it's only the hub. Here's what it owns and what it just references.

| System | What the registry does | What the registry does NOT do |
|---|---|---|
| Feature store (Feast/Tecton) | References features by ID. Calls its schema-compat API to validate contracts. | Host feature definitions. Duplicate feature schemas. |
| Training orchestrator (Airflow/Flyte) | Subscribes to "training run complete" events. | Call into the orchestrator. Duplicate pipeline definitions. |
| Dataset registry (lake snapshots) | Reference-counts dataset snapshots. Blocks deletion of referenced data. | Store datasets. |
| Experimentation platform | Declares A/B test requests. Reads back significance and guardrail outcomes. | Implement bucketing. Implement statistical tests. |
| Deployment controller (K8s operator) | Writes `desired_state`. | Imperatively deploy. |
| Serving plane (Triton, vLLM, Ray) | Provides weights pointers and contract. | Serve requests. |
| Metrics store (time-series DB) | Queries metrics for gate evaluation. | Store metrics itself. |

#### Why operator pattern, not imperative deploy API

The registry writes "model v47 should be at state=canary, 10% traffic." An operator continuously reconciles the actual serving infrastructure toward that state. Benefits:

- **Drift detection**: if someone hand-edits a Kubernetes resource, the operator notices and corrects (or alerts).
- **GitOps compatibility**: `desired_state` is a YAML object that can live in a git repo with PR review.
- **Self-healing**: if pods fail, the operator brings them back without any deploy action.

An imperative deploy API ("call deploy(model, version)") loses all of these and becomes the thing you need to rebuild a year in.

#### The god-service failure mode

Every adjacent team will want to put their stuff in the registry, because the registry is where enforcement happens at deploy time. Feature teams will want feature definitions here. Training teams will want pipeline configs here. Experiment teams will want experiment configs here.

Every time, the answer is no. The registry references, it doesn't duplicate. Keeping it thin is what keeps it correct, fast, and surviving across years.

---

## 9. Recent Developments Worth Noting

**[STAFF SIGNAL: modern awareness]**

- **MLflow.** Fine for small teams; its registry is a thin metadata database. No first-class lineage-graph, no contract validation, no framework-aware runtime requirements. At 500-model scale, MLflow becomes a *prefix* of what we need, not a solution.
- **KServe and Seldon Core.** KServe's `InferenceService` CRD is a reasonable abstraction for the deployment-plane operator pattern in §8. It doesn't cover registry, lineage, or governance — it's one layer of this system, not the whole thing.
- **Triton Inference Server.** Committed to as the DNN-serving backend (§6.4). Its model-repository abstraction is a reasonable foundation for the serving contract; the registry sits above it.
- **vLLM, TensorRT-LLM, SGLang.** Required for LLM serving. Paged attention, continuous batching, speculative decoding aren't nice-to-haves — they're how you get acceptable cost on any LLM workload. The registry has to be wide enough to carry their contracts.
- **Feast / Tecton.** Feature platforms with versioned schemas. The contract-validation API in §8 is what I'd build against (or demand, if using Feast).
- **EU AI Act (Articles 9–15).** Materially changes the lineage schema for high-risk systems. Training-data provenance, post-deployment monitoring, human-oversight records — all become compliance artifacts. Most pre-2024 registries don't carry enough metadata, and retrofitting is painful.
- **Published platforms.** Uber's Michelangelo, Meta's FBLearner Flow, Netflix's Metaflow, LinkedIn's Pro-ML, Airbnb's Bighead — their published talks describe variations of this design. Common thread: thin registry, first-class lineage, serving plane is where the money is made or lost.

---

## 10. Explicit Tradeoffs, and What Would Force a Redesign

Tradeoffs I've taken:

- **Registry as metadata-only, not blob store.** Cost: two-hop lookup on deploy. Benefit: registry stays small and fast; blobs scale independently.
- **Split serving plane (Triton + vLLM) over unified.** Cost: two runtimes to operate. Benefit: LLM-appropriate serving without gutting DNN efficiency.
- **Tiered governance over uniform.** Cost: policy complexity. Benefit: 90% of deploys auto-promote; human capacity matches review load.
- **Per-user sticky A/B over per-request.** Cost: slightly noisier arm balance if users churn. Benefit: session metrics are actually measurable.
- **Operator pattern over imperative deploy.** Cost: learning curve. Benefit: drift detection, GitOps, self-healing.
- **Refuse to promise rollback for IRREVERSIBLE models.** Cost: some teams unhappy. Benefit: system doesn't lie.

What would force a redesign:

- **If >30% of the fleet became LLMs**: the split serving plane becomes the *primary* plane, not a sibling. Registry schema needs first-class prompt/context/tokenizer fields. The whole governance model would shift because LLM outputs are harder to quantitatively evaluate than scoring outputs.
- **If >30% did online learning**: the discrete-version concept collapses into checkpoint sequences. The state machine needs a continuous-mode branch, not a bolt-on.
- **If per-prediction explainability became a regulatory requirement**: prediction logging and feature attribution become mandatory at serving time. Serving-plane cost goes up 2–3×.
- **If teams converged on one framework**: the heterogeneity support is dead weight. Platforms that keep abstractions they don't need are the platforms teams route around.

---

## 11. What I Would Push Back On

**[STAFF SIGNAL: saying no]**

Four things in the prompt deserve pushback.

**"50 deploys/day" as a target.** A healthy platform at 500 models has 45 automated non-events (weight refreshes on unchanged contracts, passing gates, auto-promoted in under 30 minutes) and 5 contract-changing or high-tier deploys that get the full state machine. If the prompt means "50 full state-machine walks per day," that's a spec for something unhealthy. I'd reframe the requirement as "support 50 deploys/day with ≥90% automation rate and ~5 human-review deploys/day."

**"Supports PyTorch, TF, and ONNX" as a flat requirement.** If the actual distribution across 500 models is 90% PyTorch, investing equally in TF and ONNX is political balance, not engineering value. I'd measure the distribution first, then offer first-class support for the dominant framework, best-effort for exports, and document that the others work but don't get dedicated investment. The LLM pressure makes generic multi-framework runtimes more suspect, not less — LLM serving needs specialized runtimes, full stop.

**The implicit assumption that all 500 models share a lifecycle.** A fraud model, a tabular churn predictor, an LLM assistant, and a recommendation ranker don't share deployment mechanics or evaluation semantics. This design handles that with tiering and with a split serving plane. A design that forces one lifecycle on all of them will be right for the median model and wrong for the models that matter most.

**"Rollback" as a requirement without qualification.** The prompt lists rollback alongside A/B testing and shadow deployment as if they're equivalent features. They're not. For about 30% of a realistic fleet, rollback to the pre-deployment world is not achievable, and the platform's job is to refuse to promise it — and to make the rollback class a visible attribute that owners see at promotion time. Rollback isn't a toggle; it's a property of the model.

---

## Closing

The hard work here is:

1. **Refusing to treat models as binaries.** They're contracts with lineage.
2. **Admitting where rollback can't do what people want.** Classify, don't lie.
3. **Splitting the serving plane along the grain of real runtime requirements.** Not a generic abstraction.
4. **Designing governance that automates the boring path and gates only the consequential one.** This is the organizational design problem in technical clothing.

Everything else is a database and some routers. The parts I'd defend hardest in follow-up questions are §6.1 (rollback honesty) and §4 (lineage as a first-class enforced graph). Those are where the system earns its keep.