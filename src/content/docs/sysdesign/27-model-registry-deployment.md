---
title: Model Registry & Deployment System
description: Model Registry & Deployment System
---
```
"Design the system that manages ML model versions from training to production. Supports A/B testing, shadow deployment, rollback, and works across PyTorch, TF, and ONNX. 500 models in production, 50 deployments per day. Walk me through it."
```
---
# Model Registry & Deployment System — Staff-Level Answer

## 1. Scope and Reframing

Before architecture, three reframings, because otherwise we design the wrong system.

**[STAFF SIGNAL: models-are-not-binaries]** A "model version" is not a weights file. It is a tuple: `(weights, feature_contract, preprocessing_code, postprocessing_code, training_data_version, training_code_version, eval_set_version, eval_metrics, runtime_requirements, rollback_viability_class, downstream_dependents)`. The registry does not *own* most of these — it *references* them and validates that the tuple is internally consistent. A deployed model whose feature contract silently drifts from what feature-store serves today is broken even though its weights never changed. The registry's job is to enforce, at deploy-time and at every request-time, that the model's contract matches reality. This reframing — models as *contracts with lineage*, not artifacts — drives every downstream decision: the schema, the state machine, the rollback semantics, the governance.

**[STAFF SIGNAL: scope negotiation]** "500 models" is underspecified. Concretely, the range:

- Tabular classifiers / GBDTs, KB–10MB, stateless, request-response. Trivial case.
- Deep ranking / recommendation models, 100MB–10GB, stateless weights but **accumulate user-facing state** via their outputs (personalization, exposure bias).
- Embedding models feeding downstream retrieval, 1–50GB, where the embedding space is itself a contract — downstream retrieval indices are trained against a specific embedding version.
- Online-learning / continuously-updated models (CTR predictors with streaming updates). Weights change by the minute.
- LLMs, 10GB–500GB, with KV cache, paged attention, continuous batching, speculative decoding — a serving regime that generic model-registry abstractions handle poorly.
- Batch-scoring models run in offline jobs, not request-response.

I am **primarily solving for deep ranking / recommendation models — stateless-weight, request-response, 100MB–10GB, 10–1000 QPS, trained offline and deployed frozen**, because this is the center of gravity of a 500-model fleet at a Meta/Netflix-class company and it's where the hard problems (stateful-rollback, shadow counterfactual, A/B testing) actually bite. I will note in each section what changes for batch, online-learning, and LLMs, and I will argue later that LLMs should live in a sibling serving plane behind a shared registry API, not inside the same runtime.

**[STAFF SIGNAL: saying no]** "50 deploys/day" is a symptom, not a spec. If a team is deploying twice a day per model, either (a) most of those should be non-events — weight refreshes on the same contract, same feature schema, promoted automatically on passing gates — or (b) the platform has no governance and teams are using prod as staging. The system's job is to make safe deploys into boring, automated, unremarkable events (target: 90% fully automated, zero human-in-loop) and to make the *unsafe* ones (contract change, rollback-class change, tier-escalation) explicit, gated, and rare. I'll design for that target, not for 50 heroic deploys/day.

**[STAFF SIGNAL: rollback honesty]** Rollback is the hardest subsystem in this design. For stateless request-response models, rollback is ~5 minutes and trivial. For models with accumulated production state, personalization state, downstream-trained dependents, or online-learning drift, *rollback in the sense of "restore the world to its pre-deploy state" is not always possible*. Any design that claims otherwise is lying. The architectural move is to classify each model's `rollback_viability` at deploy time and refuse to promise what the system cannot deliver.

---

## 2. Capacity Math and Governance Budget

**[STAFF SIGNAL: capacity math]**

| Quantity | Value | Derivation |
|---|---|---|
| Deploy rate | 50 / day → 1 per ~30 min, bursty to 1 per ~5 min in working hours | given |
| Target automation | ≥90% deploys fully automated | governance design goal |
| Human-review budget | ≤5 deploys/day needing human gate | 50 × 10% |
| Active versions per model | ~10 (current prod, prev prod, 2–3 canaries, 1–2 shadows, recent archived) | policy |
| Total retained artifacts | 5,000 | 500 × 10 |
| Avg artifact size (mixed fleet) | 500 MB | weighted: many small, few LLMs |
| Retained artifact storage | 2.5 TB hot + ~50 TB cold (full history, incl. LLMs) | standard |
| Shadow traffic @ 10% for 1 wk, 100 QPS/model avg, 500 models | ~3 × 10¹¹ extra inferences/wk | 0.1 × 100 × 500 × 604800 |
| Shadow compute cost @ $1e-5/inference (rough GPU amortization) | ~$3M/wk if *every* model were in shadow simultaneously | capacity reality: shadow in < 5% of fleet at any time → ~$150k/wk, tractable |
| A/B sample size, 1% relative CTR lift on 5% baseline, α=0.05, power=0.8 | n ≈ 16·p(1−p)/δ² ≈ 3M per arm | standard formula |
| A/B duration, 1M DAU on model, 50/50 split | ~6 days minimum | 3M / 500K per day |
| Offline eval storage (eval sets, per-version metrics, per-slice metrics) | ~10 GB/model × 500 = 5 TB | dominant cost is held-out counterfactual sets, not metrics |
| Rollback TTR, stateless | ≤ 5 min | SLO |
| Rollback TTR, stateful-medium (state snapshot restore) | ≤ 30 min | SLO |
| Rollback TTR, stateful-hard (personalization, online learning) | **undefined** — see §6.1 | honest |

Key implication: the A/B duration math alone means we cannot evaluate most deploys with a rigorous A/B. The system must commit to a tiered strategy where most deploys pass offline + automated-online gates without full A/B, and only contract-changing or high-tier deploys get the 6-day ride.

---

## 3. High-Level Architecture

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
              │   │  metadata DB + artifact store ref  │
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
              │   │ versioned schema │  "does feature_v7 still
              │   └──────────────────┘  satisfy model's contract?"
              │
              ▼
        ┌──────────────────────────────────────────────┐
        │           DEPLOYMENT CONTROL PLANE           │
        │  Deploy Controller (operator pattern on K8s) │
        │  Traffic Manager (mesh-level splits)         │
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

Separation I am committing to:
- **Registry** is metadata + lineage + state-machine authority. It *references* artifacts (S3/GCS), *references* feature definitions (feature store), *references* dataset versions (lake). It does not own blobs beyond small artifacts like eval reports.
- **Training plane** produces artifacts and emits events; registry ingests and validates.
- **Deployment control plane** is the operator that translates "registry says model M version V should be at state=canary@10%" into actual serving infrastructure changes.
- **Serving plane is split**: Triton for traditional DNN, vLLM/TensorRT-LLM for LLMs, Ray/Spark for batch. The registry API is uniform; the backends are not. **[STAFF SIGNAL: framework heterogeneity discipline]**
- **Online eval plane** is the feedback loop that drives promotion and rollback decisions.

---

## 4. Registry Schema and Lineage

**[STAFF SIGNAL: lineage as first-class]**

A model version record, minimal:

```
ModelVersion {
  model_id:                 str     # logical model, stable across versions
  version:                  int     # monotonic
  weights_uri:              str     # blob store ref, content-addressed
  weights_hash:             str     # sha256
  framework:                enum    # pytorch | tf | onnx | vllm_engine
  runtime_requirements:     {gpu_class, mem_gb, compute_cap}
  feature_contract_ref:     str     # pointer into feature store's versioned schema
  preprocess_code_ref:      (repo, commit_sha, path)
  postprocess_code_ref:     (repo, commit_sha, path)
  training_run_id:          str     # → training orchestrator
  training_data_version:    str     # → dataset registry
  training_code_version:    (repo, commit_sha)
  parent_model_version:     str|null # warm-start lineage
  eval_set_version:         str     # which eval set produced metrics
  eval_metrics:             {...}
  eval_slice_metrics:       {slice: {...}}
  rollback_class:           enum    # STATELESS | STATE_SNAPSHOT | IRREVERSIBLE
  rollback_snapshot_uri:    str|null
  tier:                     enum    # L0..L3 (see §6.5)
  state:                    enum    # see §5
  state_history:            [{state, ts, actor, gate_results}]
  audit_log:                append-only
}
```

Two invariants I will enforce and never weaken:

**Invariant 1 (contract consistency at request time).** Before a request is routed to model version V, the serving layer validates that the feature-store's current schema for `feature_contract_ref` is still satisfiable. If a feature was dropped or its type changed, the request does not reach V — the request fails closed or falls back, depending on tier policy. **[STAFF SIGNAL: invariant-based thinking]**

**Invariant 2 (lineage is append-only, references never break).** A training dataset referenced by any non-archived model version cannot be deleted. This is a hard constraint on the dataset registry, enforced via reference counting. If the dataset store violates this, the registry cannot answer audit questions — this is the operational break where EU AI Act compliance falls apart. **[STAFF SIGNAL: modern awareness]** The EU AI Act's Article 11 technical documentation requirement effectively promotes training-data lineage from a nice-to-have to a regulated asset for high-risk systems; deleting a dataset referenced by a deployed model becomes a compliance event, not an operational convenience.

Concrete lineage graph for one deployed model:

```
                     ┌─────────────────────────────┐
                     │ model_v47 (state=PROD)      │
                     └──────┬──────┬──────┬────────┘
                            │      │      │
               ┌────────────┘      │      └──────────────┐
               ▼                   ▼                     ▼
    ┌──────────────────┐  ┌────────────────┐   ┌──────────────────┐
    │ weights_blob     │  │ feature_contract│   │ parent_model_v44 │
    │ sha256:af3…      │  │ ref=fs://v7     │   │ (warm-start)     │
    └──────────────────┘  └───────┬────────┘   └─────┬────────────┘
                                  │                   │
                          ┌───────┴────────┐          │
                          ▼                ▼          ▼
                ┌──────────────┐ ┌──────────────┐  (recursive…)
                │ feature_A v3 │ │ feature_B v7 │
                │ def@repo:sha │ │ def@repo:sha │
                └──────┬───────┘ └──────┬───────┘
                       │                │
                       ▼                ▼
                ┌──────────────────────────┐
                │ training_data_ver=ds:9f1 │
                │ (lake snapshot)          │
                └──────────────────────────┘

  Also referenced (not shown): eval_set_ver, training_code_commit,
  preprocess/postprocess commits, rollback snapshot (if any),
  downstream_dependents (embedding consumers, retrained downstreams).
```

The `downstream_dependents` edge is often forgotten and is the one that matters for rollback: it tells you, when rolling back v47 → v44, what *else* is now inconsistent (an embedding-based retrieval index built on v47's embedding space, a downstream ranker trained on v47's scores).

---

## 5. Deployment State Machine

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
                                     (tier gate: L2/L3 → human approve)
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
                                           │  50/50, ≥6 days  │──┐
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

  A model can be in SHADOW for a different traffic slice while a sibling
  version is in AB_TEST for another slice. The state is per-(model, slice);
  the registry tracks the Cartesian product.
```

The backward transitions are the interesting ones. Every forward transition has a corresponding rollback edge that invokes §6.1. The state machine is not flattened on purpose: L2/L3 models must walk every gate; L0 models can skip directly from `OFFLINE_PASSED` to `PROMOTED` with no shadow or canary. Flattening is the common mistake at this scale — it reduces the state-machine cost at the price of losing the surface area where gates attach.

---

## 6. Deep Dives

### 6.1 Rollback mechanics for stateful models — the hardest problem

**[STAFF SIGNAL: rollback honesty]** **[STAFF SIGNAL: failure mode precision]**

Three classes, explicitly declared per model at promotion time:

**Class A — STATELESS.** Request → features → inference → response. The model produces no persistent state, its outputs are not stored as training signal for downstream systems (or are, but on a well-bounded delay), and users have no memory of its specific behavior. Rollback = point traffic-manager at prior version's service, drain old pods. Target TTR ≤ 5 min. This is the easy case and it's the minority at 500-model scale.

**Class B — STATE_SNAPSHOT.** The model produces or consumes externalized state (per-user embedding cache, counter table, exposure-deduplication bloom filter, calibration table) that the model itself maintains but that lives outside the weights. Rollback requires restoring state to its pre-deploy snapshot. The discipline: at every state transition into CANARY and AB_TEST, the deploy controller forces a snapshot of all externalized state, content-addressed and stored with `rollback_snapshot_uri`. Rollback = redeploy old weights + restore snapshot. TTR ≤ 30 min dominated by snapshot restore time. Failure mode: snapshot taken *before* cutover includes state changes from the 30-min window of parallel writes during canary; resolution is either (a) quiesce writes during snapshot (unacceptable for production) or (b) use a write-ahead log with a snapshot cursor and replay backward on rollback (complex, brittle).

**Class C — IRREVERSIBLE.** This is where mid-level answers hand-wave. Three sub-cases:

*C1 — Personalization leakage.* A recommendation model v47 is promoted. Users are shown v47's recommendations for a week. Their click behavior on those recommendations trains their user-level feature vectors (recent-click history, dwell-time signals). A week later v47 is declared bad and rolled back to v44. v44 now receives feature vectors that have been shaped by v47's recommendations. The "rollback" serves v44's weights but the input distribution has permanently shifted — v44's performance post-rollback will not match v44's performance pre-v47. You cannot un-show users the recommendations they saw.

*C2 — Online-learning drift.* A CTR model updates its weights from streaming impressions. Deployed weights at time T and at T+1 week are different by design. "Rollback" is ambiguous: to the snapshot at T (discarding a week of learning, including any legitimate adaptation to genuine distribution shift), or to a hypothetical "weights at T+1 if we had not made the change" (which does not exist anywhere)?

*C3 — Downstream training.* Model v47's scores were logged and used to train downstream ranker R_v3, which was also promoted. Rolling back v47 → v44 leaves R_v3 in place, but R_v3 was trained on a score distribution that no longer exists. Either you cascade rollback (R_v3 → R_v2, plus anything trained on R_v3's outputs — a dependency graph traversal), or you accept that the system is now in a hybrid state.

Architectural disciplines that make rollback *more* feasible — none of which make it trivially so:

1. **State externalization.** Keep stateful components outside the weights and snapshot them independently. Rollback class is a property of the model's architecture, not its deployment — and the registry enforces that a model cannot be promoted to L2/L3 tier with rollback_class=IRREVERSIBLE without a signed override. **[STAFF SIGNAL: invariant-based thinking]**
2. **Bounded online-learning windows.** Continuous learners get periodic hard checkpoints (every N hours), and production serves from the most recent checkpoint, not from a continuously mutating weight blob. Rollback = serve an older checkpoint. The continuous update becomes "update the next checkpoint candidate," not "mutate prod." **[STAFF SIGNAL: continuous-learning edge case]**
3. **Downstream-dependents graph traversal.** The registry stores the reverse edges. On rollback, the deploy controller computes the transitive closure of dependents and either (a) warns the operator, (b) initiates cascade rollback, or (c) marks dependents as needing re-training. Default behavior is (a) — cascade is too destructive for automation.
4. **Honest labeling.** A model whose rollback is Class C gets a big ugly badge in the UI at promotion time. "This model cannot be cleanly rolled back. Promotion is an approximately one-way decision." The system never pretends it can do what it can't. **[STAFF SIGNAL: saying no]**

Rollback scenario diagram for a Class C model:

```
  t=0   v47 promoted to 100% traffic.
        v44 pods drained. v44 weights retained.
        v47's outputs flow to:
          - users (recommendations)
          - logging (join with clicks → training data for R_v3)
          - feature store (recent-rec feature, user-level)
          - embedding retrieval index rebuild (nightly)

  t=7d  v47 declared bad. Rollback requested.

  Action cascade (computed from downstream_dependents graph):
   ┌─────────────────────────────────────────────────────────┐
   │ 1. Traffic manager: v47 → v44 (5 min, weights cutover)  │
   │ 2. Feature store: recent-rec feature IS v47-contaminated│
   │    for 7 days of users.  NO ACTION — cannot un-write.   │
   │    Flag: v44 running with shifted input distribution.   │
   │ 3. R_v3 trained on v47 scores. Mark R_v3 as NEEDS_RETRAIN│
   │    but do not auto-rollback R_v3 (cascading rollback    │
   │    risk > status quo).                                  │
   │ 4. Embedding index rebuilt on v47 embeddings.  Roll back│
   │    index to pre-v47 snapshot (cheap, state-snapshotted).│
   │ 5. Post-rollback monitoring: expect v44 metrics to be   │
   │    WORSE than pre-v47 v44 metrics for ~14 days as input │
   │    distribution decays. Set alert thresholds accordingly│
   │    to avoid spurious second rollback.                   │
   └─────────────────────────────────────────────────────────┘

  What "rollback" delivered:
    - User-facing serving path: restored (5 min).
    - Input distribution: permanently perturbed.
    - Downstream R_v3: still contaminated.
    - Audit trail: complete.
```

This is what the honest rollback contract looks like. The mid-level answer of "redeploy v44" is step 1 of 5.

### 6.2 Shadow deployment and the counterfactual problem

**[STAFF SIGNAL: counterfactual awareness]**

Mechanics:

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
           │ response → user                 │ response → /dev/null
           │ prediction → log (key=req_id)   │ prediction → log (key=req_id)
           └────────────────┬────────────────┘
                            ▼
                  ┌───────────────────┐
                  │ Divergence engine │
                  │ - agreement rate  │
                  │ - KL(out_v44||v47)│
                  │ - per-slice diff  │
                  │ - calibration     │
                  │ - latency, errors │
                  └───────────────────┘
```

Two implementation choices: live-mirror (duplicate the request to both models in real time) or log-replay (periodically replay a sample of production requests through the shadow). Live-mirror has fresh-data fidelity but doubles compute on shadowed traffic; log-replay is cheaper and repeatable but misses time-varying state (e.g., the shadow misses the feature-store values that were live at request time unless features are also logged point-in-time — which, at scale, is a significant feature-store design requirement; this is one of the places the registry and feature-store APIs are not orthogonal).

Where shadow is meaningful:
- Infra validation: does v47 crash, OOM, exceed latency SLO on prod traffic shape? Yes, shadow catches this.
- Calibration and agreement: for classifiers and scorers where the *output* is the product (fraud scores, quality scores, ranking scores consumed by a downstream system), shadow gives a real measurement of how outputs diverge at fixed input.
- Per-slice divergence: the model's behavior on a specific country, language, or content type can be measured without user exposure.

Where shadow is **meaningless or actively misleading**:
- Recommendation/ranking systems with closed-loop feedback. v47 recommends item B; v44 recommends item A; user sees A (prod), clicks A, dwells on A. Shadow logs "v47 would have recommended B." You have no measurement of what the user would have done if shown B. Shadow shows you divergence *at fixed context*; it does not show you divergence in the counterfactual world where v47 was actually serving. The metric movements that actually matter (CTR, session length, retention) are not observable from shadow. This is the counterfactual trap.
- Any model whose output alters the future input distribution (personalization, exploration-exploitation, content-moderation removing content that then never appears in the next request).

What I promise and what I don't: shadow in this system is positioned as **infrastructure and calibration validation**, not as a predictor of closed-loop online metric movement. For recommendation models, the only honest evaluator of closed-loop movement is A/B testing. Teams that want a "safer A/B" via a longer shadow are asking the system to lie; I refuse to build that. **[STAFF SIGNAL: saying no]** For some model classes, counterfactual estimators (IPS, doubly-robust) can partially bridge, but their variance on tail events is large and they require logging propensity scores at serving time — a non-trivial infrastructure commitment that I'd make only for the top-tier models where A/B alone is too slow.

### 6.3 A/B testing and online evaluation

Traffic split: per-user sticky hash (user_id → bucket), not per-request. Stickiness matters because: personalization state accumulates per user and per-request splitting pollutes both arms; session-level metrics (session length, retention) require user-level consistency; interaction effects between experiments require coherent bucketing across the experimentation platform.

Sample size: for a 1% relative lift on a 5% baseline CTR, ~3M users per arm at α=0.05, power=0.8. At 1M DAU on a model with a 50/50 split, the minimum-detectable-effect horizon is ~6 days. Most teams run experiments for 14 days to capture weekly seasonality — this is correct and I'd enforce a minimum of 14 days or one full weekly cycle for tier L2/L3 unless the effect size is large enough that a shorter experiment is well-powered.

Multi-metric problem: CTR up 1.2%, session length down 0.4%, revenue up 0.3%, content-diversity down 2%. The registry cannot make this call. The registry stores a **decision policy per model class** declared by the owning team: primary metric + guardrails with fail-thresholds. Promotion requires primary-metric significant positive AND no guardrail trips. If any guardrail trips, the decision escalates to human review. **[STAFF SIGNAL: evaluation divergence]** The systemic issue — offline AUC improves while online CTR regresses — is handled by (a) maintaining an offline eval set that is explicitly held out of the production feedback loop (logged from a small exploration bucket that is always served a uniform-random-ish policy; this is expensive and a business-wide policy decision, not a registry one), and (b) tracking the offline-online metric correlation *over time per model family*, so we know empirically how much to trust offline improvements.

Concurrent experiments: at 50 deploys/day, many experiments run simultaneously. The experimentation platform (not the registry — a distinct system) must assign users to orthogonal layers so that model A's experiment and model B's experiment do not confound. The registry's obligation is to declare which layer a deploy consumes, and to refuse to promote if the layer is saturated.

Auto-promotion: I promote automatically for L0/L1 on passing gates, with a 48h post-promotion monitoring window where auto-rollback triggers on guardrail breach. L2/L3 require human approval on promotion, full stop.

### 6.4 Framework heterogeneity: Triton + specialized LLM plane

**[STAFF SIGNAL: rejected alternative]**

Three options evaluated:

*Option A — everything on NVIDIA Triton Inference Server.* Triton supports TF, PyTorch (TorchScript and via Python backend), ONNX, and custom backends. Multi-model per process gives 40–60% GPU utilization vs 10–20% for container-per-model. Dynamic batching is a real win for moderate-QPS models. Rejected as the *sole* backend because: (a) Triton's LLM story (via TensorRT-LLM backend) works but is behind dedicated LLM runtimes for continuous batching, paged attention, and speculative decoding; (b) framework-specific custom ops (PyTorch custom CUDA kernels, TF custom ops) are awkward via ONNX and may require a Python backend that loses the multi-model efficiency advantage. Kept as primary backend for traditional DNN ranking/recommendation models.

*Option B — per-framework runtimes (TorchServe, TF Serving, ONNX Runtime Server) behind a common gateway.* Full framework fidelity. Rejected because N runtimes to operate, multiply SRE load by N, and the gateway becomes a routing ball-of-string. Worth it only if framework-specific features are load-bearing, which for ranking models they typically are not.

*Option C — convert everything to ONNX, single runtime.* Rejected because ONNX conversion is lossy for cutting-edge models (attention variants, custom kernels, quantization-aware components — this is exactly the pain point when the training team uses FP8 or MXFP4 quantization and the ONNX exporter either drops the quantization or preserves it in a way the runtime doesn't execute efficiently). Forcing ONNX on the training team is a tax the platform imposes that will make good teams route around the platform.

*Option D — bento/ray-serve-style per-model container.* Rejected as primary: GPU utilization unacceptable at 500-model scale; storage and cold-start costs dominate. Kept as escape hatch for the ~10% of models whose framework or custom-op requirements don't fit Triton.

Commitment: Triton for traditional DNN models (TF, PyTorch, ONNX), vLLM or TensorRT-LLM for LLMs (behind the same registry API but with an LLM-aware contract — KV cache sizing, prompt-length distribution, streaming), Ray Serve as escape hatch. The registry has a `model_class` field and the deployment controller dispatches. The registry is uniform; the runtimes are not, and pretending otherwise is the generalization trap. **[STAFF SIGNAL: framework heterogeneity discipline]**

LLM-specific note: the registry's "weights + contract" abstraction mostly holds for LLMs, but `runtime_requirements` gets much richer (KV cache memory per sequence, max context length, tokenizer version, draft model reference for speculative decoding), and the state machine's canary stage needs to consider context-length distribution of canary traffic vs prod traffic — a model that's great on 4k contexts can fall over on 128k contexts that weren't sampled in canary.

### 6.5 Governance and approval at 500-model scale

**[STAFF SIGNAL: governance at scale]**

Tiering:

- **L0 (experimental/internal).** No user exposure. Batch scoring to internal dashboards. Approval: none beyond automated offline gates. Deploys are non-events.
- **L1 (user-facing, non-revenue).** Features like content sorting in non-monetized surfaces. Approval: automated gates (regression thresholds, fairness across declared protected slices, latency/memory SLO). No human in loop for weight refreshes on unchanged contract.
- **L2 (user-facing, revenue or retention impact).** Ranking, recommendation, search relevance. Approval: automated gates + human review on promotion + required 14-day A/B for contract-changing deploys. Weight refreshes on unchanged contract can still auto-promote if online guardrails hold in canary.
- **L3 (safety, fraud, compliance, medical, moderation).** Approval: human review at every state transition. No auto-promotion. Required secondary reviewer from a different team. Deployment is an explicit audit event with logged justification.

Every deployment action — transition, gate result, approver identity, rollback — is written to an append-only audit log keyed by `(model_id, version, action, ts)`. This log is the source of truth for EU AI Act Article 11 documentation and for internal postmortems. Under Article 11, high-risk AI systems require maintained technical documentation including training methodology, data provenance, and post-deployment monitoring — so L3 (and some L2) models are not just an internal-governance concern but a regulated asset.

Emergency deploy path: the system supports a `break-glass` deployment mode that bypasses some gates (e.g., shadow/canary duration) but (a) requires two senior approvers, (b) forces a 24-hour heightened-monitoring state post-deploy, (c) generates an automatic postmortem ticket, and (d) is rate-limited — a team that uses break-glass more than twice in 30 days is locked out of the mode pending a process review. Without the rate limit, break-glass becomes the normal path and governance collapses.

---

## 7. Observability: "why is model v47 performing differently today?"

Worked example. Alert fires: v47's offline-proxy CTR (predicted CTR, not click-through — we track this continuously post-deploy) is down 2% day-over-day.

Diagnosis tree driven by the lineage graph:

1. *Is the model different?* Check weights_hash — unchanged. Rule out weight change.
2. *Is the input distribution different?* Per-feature distribution monitoring: feature_B's histogram shifted on a particular country slice. Cross-reference feature_B's definition in feature store — did its upstream data source or transformation change? Check feature-store's change log. Yes, feature_B's upstream logging added a new enum value yesterday, shifted the distribution.
3. *Which models depend on feature_B?* Lineage reverse lookup: 14 models depend on feature_B. Four of them are L2. Alert those model owners.
4. *Is the output distribution different?* Prediction histogram shows a long-tail shift on the affected slice.
5. *Is the downstream consumer behaving differently?* The ranker downstream of v47 is clipping v47's scores differently because v47's output range shifted.
6. *Is the evaluation metric itself defined consistently?* Check metric-definition version — unchanged.

Resolution: feature_B's upstream change is the root cause. The registry's contract check did not catch this because the schema didn't change — only the distribution. This is a known weakness: *semantic contract* (schema match) is a weaker invariant than *distributional contract* (distribution match within tolerance). The system monitors the latter and alerts but does not enforce, because distributional drift is often legitimate and hard-enforcing would cause constant false rollbacks.

Per-slice monitoring is non-negotiable at L2/L3. A model that performs fine globally but has regressed for users in one locale has a real problem that global aggregates hide — and this is also the typical shape of a fairness regression.

---

## 8. API Boundaries with Adjacent Systems

**[STAFF SIGNAL: API boundary clarity]**

What the registry owns: model metadata, lineage graph, state machine, audit log, contract declaration, deployment authorization. Nothing else.

Boundaries:

- **Feature store.** The registry *references* feature definitions by ID. It does not host them. Contract validation calls out to the feature store's schema-compatibility API: `feature_store.validate_contract(contract_ref, requested_features) → ok | breaking_change | compatible_extension`. Anti-pattern to avoid: registry duplicating feature schemas, which guarantees drift.
- **Training orchestrator (Airflow/Flyte).** The registry is notified of a new candidate model via an event from the orchestrator, including `training_run_id`, `training_data_version`, artifact pointer. The registry does not call the orchestrator; it subscribes. This keeps the orchestrator's iteration speed independent of the registry's availability.
- **Dataset registry (lake snapshot system).** The registry references `training_data_version` and `eval_set_version`. It holds reference-count on dataset snapshots so the dataset registry cannot delete a dataset referenced by a non-archived model. Anti-pattern: registry storing the dataset. It doesn't — it pins it.
- **Experimentation platform.** The registry declares an A/B test request; the experimentation platform owns user bucketing, layer orthogonality, and statistical decision APIs. The registry reads back significance and guardrail outcomes. Anti-pattern: registry implementing its own stats.
- **Deployment controller / K8s operator.** The registry declares desired state; the operator reconciles actual state. Kubernetes operator pattern, not imperative deploy API. The difference: imperative deploy = "call deploy(model, version)"; operator = the registry writes `desired_state: v47@canary@10%` and the operator continuously reconciles. The operator wins because it gives us drift detection, self-healing, and GitOps compatibility for the `desired_state` object.
- **Serving plane.** Triton / vLLM / Ray Serve receive model-loading instructions from the deployment controller. They pull weights from blob storage (content-addressed) and validate the weights_hash. They report health and metrics back. Registry does not talk to serving plane directly.
- **Observability / metrics plane.** Per-model, per-slice metrics flow into a time-series store. The registry queries this store for gate evaluation. Anti-pattern: registry storing metrics itself.

The god-service failure mode for the registry is real: every adjacent team wants to put their stuff in the registry because the registry is where the deploy-time enforcement happens. Resisting this keeps the registry small, correct, and fast.

---

## 9. Recent Developments Worth Referencing

**[STAFF SIGNAL: modern awareness]**

- **MLflow's evolution.** Fine for small teams; its model-registry abstraction is a thin database and does not address lineage-as-first-class, framework-specific runtime requirements, or contract validation. At 500-model scale, MLflow becomes a prefix of the system I've described, and the gap is wider than it looks from its docs.
- **KServe and Seldon Core.** KServe's `InferenceService` CRD is close to the right abstraction for the deployment-plane, and I'd lean on it for the operator pattern described in §8. It does not address registry, lineage, or governance — it's one layer of this system.
- **Triton Inference Server.** Committed to as the DNN-serving backend per §6.4. Its model-repository abstraction is a reasonable bottom-half of the serving contract; registry sits above it.
- **vLLM / TensorRT-LLM / SGLang.** Required for LLM serving. Paged attention, continuous batching, speculative decoding are not afterthoughts — they're how you get acceptable cost on any LLM workload. The registry abstraction must be wide enough to carry their contract, not so narrow that it pretends an LLM is just a big DNN.
- **Feast / Tecton.** Feature platforms with versioned schemas. The contract-validation API described in §8 is the thing I'd build against (or demand, if using Feast).
- **EU AI Act, effective 2025–2026.** Articles 9–15 materially change the lineage schema for high-risk systems. Training-data provenance, post-deployment monitoring, and human-oversight records are compliance artifacts. Most registries built pre-2024 do not carry enough metadata; retrofitting is painful.
- **Uber's Michelangelo, Meta's FBLearner, Netflix's Metaflow, LinkedIn's Pro-ML** — the published talks from these teams describe variations of the system I've sketched. The common pattern: registry is thin, lineage is first-class, the serving plane is where the money is made or lost.

---

## 10. Explicit Tradeoffs Taken

- **Chose registry-as-metadata-only over registry-as-blob-store.** Cost: two-hop lookup (registry → blob) on deploy. Benefit: registry stays fast, small, correct; blob store scales independently.
- **Chose split serving plane (Triton + vLLM) over unified.** Cost: two runtimes to operate. Benefit: LLM-appropriate serving without gutting traditional DNN efficiency.
- **Chose tiered governance over uniform.** Cost: policy complexity. Benefit: 90% of deploys auto-promote without human bottleneck.
- **Chose per-user sticky A/B over per-request.** Cost: arm balance slightly noisier, crossover contamination if users churn between buckets. Benefit: session metrics are measurable at all.
- **Chose operator-pattern deploy over imperative API.** Cost: learning curve for teams used to "deploy" verbs. Benefit: drift detection, GitOps, auditable desired-state.
- **Chose to refuse to promise rollback for IRREVERSIBLE models.** Cost: some teams unhappy. Benefit: system does not lie.

What would force a redesign:
- **If >30% of fleet became LLMs:** split serving plane should become the primary plane, not a sibling. Registry schema needs first-class prompt/context/tokenizer as primary contract fields.
- **If >30% did online learning:** the version concept collapses into checkpoint sequences; the state machine needs a continuous-mode branch that's more than a bolt-on.
- **If regulatory regime tightened to per-prediction explainability:** prediction logging and feature attribution become mandatory at serving time, changing serving-plane cost structure by 2–3x.
- **If teams converged on a single framework:** the framework-heterogeneity layer is dead weight and should be removed. Platforms that keep abstractions they don't need are the platforms teams route around.

---

## 11. What I Would Push Back On

**[STAFF SIGNAL: saying no]**

Four things in the prompt deserve pushback.

*"50 deploys/day."* This is a symptom. A well-designed system at 500 models does not have 50 heroic daily deploys — it has 45 automated non-events (weight refreshes on unchanged contracts, passing gates, auto-promoted in <30 min with no human involvement) and 5 contract-changing or tier-sensitive deploys that get the full state machine. If the prompt means "50 full state-machine walks per day," that's a spec for something unhealthy. I'd reframe the requirement as "support 50 deploys/day with ≥90% automation rate and O(5) human-review deploys/day."

*"Supports PyTorch, TF, and ONNX" as a flat requirement.* At 500 models, if most are trained in one framework, the cost of supporting the long-tail frameworks in a first-class way may exceed the benefit. I'd measure the distribution first. If 90% are PyTorch, I'd offer first-class PyTorch serving, best-effort ONNX (exported), and document that TF gets Triton's TF backend but no additional platform investment. Supporting three equally is usually a symptom of political balance, not engineering value. The LLM pressure makes the "generic multi-framework runtime" assumption more suspect — LLM serving needs specialized runtimes, full stop, and pretending they're just "another framework" under the same abstraction is how you end up with bad LLM latency.

*The implicit assumption that all 500 models share a lifecycle.* A fraud model, a tabular churn predictor, an LLM assistant, and a recommendation ranker do not share deployment mechanics or evaluation semantics. The system I described handles this by tiering and by splitting the serving plane; a system that tries to force one lifecycle onto all of them will be right for the median model and wrong for the models that matter most (critical fraud, flagship recommendation, LLM).

*"Rollback" as a requirement without qualification.* The prompt lists rollback alongside A/B testing and shadow deployment as if they're equivalent features. They're not. For roughly 30% of a realistic fleet, rollback to "pre-deployment world state" is not achievable, and the right platform response is to refuse to promise it and to make the rollback class a first-class attribute that users see at promotion time. Rollback is not a toggle; it's a property of the model, and the platform's job is to enforce honesty about which models have it.

---

**Closing.** The hard work in this system is (1) refusing to treat models as binaries, (2) admitting where rollback cannot do what people want, (3) splitting the serving plane along the grain of real runtime requirements instead of forcing a generic abstraction, and (4) designing governance that scales by automating the boring path and gating only the consequential path. Everything else is a database and some routers. The parts I'd defend hardest in a follow-up are §6.1 (rollback honesty) and §4 (lineage as a first-class enforced graph, not a metadata field); those are where the system pays for itself.