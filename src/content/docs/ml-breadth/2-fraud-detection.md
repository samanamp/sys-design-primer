---
title: "fraud detection"
description: "fraud detection"
---

# Designing a Real-Time Fraud Detection System
*Staff / Senior-Staff ML System Design — Payments / Marketplace, 2026 frontier*

---

## 1. Research pass — state of the art as of 2026 (~350 words)

**Models.** Gradient-boosted trees (XGBoost/LightGBM) remain the production workhorse for tabular fraud. The Booking.com 2025 work explicitly set out to *beat* GBDTs with tabular transformers and self-supervised pretraining and found they do not reliably win — tabular data is an unordered, heterogeneous feature set without the spatial/sequential structure deep nets exploit (Shwartz-Ziv & Armon; Grinsztajn 2022). GBDTs also give cheap SHAP reason codes, which matters for adverse-action law. **GNNs** are the real frontier addition: 2025–26 surveys report 12–25% AUROC lift over XGBoost on *relational* fraud (rings, mule chains, synthetic-identity clusters) because the i.i.d. assumption underlying row-based models breaks for coordinated fraud (Thoughtworks, Springer survey 2026). But GNNs are message-passing systems that *amplify* class imbalance through neighborhood aggregation, are acutely sensitive to concept drift, and are themselves attackable via graph-injection (AAAI 2025, "fraud gangs" multi-target attacks). The consensus architecture is therefore **hybrid**: GBDT scorer + offline-computed graph features (or a GNN producing embeddings consumed by the GBDT) + an unsupervised anomaly layer.

**Feature platforms.** Feast / Tecton / SageMaker Feature Store patterns dominate: an offline store for point-in-time-correct training joins and an online store (Redis-class) for single-digit-ms reads. Flink/Kafka compute streaming velocity aggregates; Databricks' 2026 Real-Time Mode and Coinbase's 250-feature, sub-100ms-p99 pipeline are the reference points. The repeated theme is **train/serve skew elimination** via shared feature definitions ("no logic drift").

**Labels & adversary.** Reject-inference and selection-bias literature (credit scoring: Kozodoi self-learning, semi-supervised S4VM, deep generative RI) directly addresses that you only observe outcomes for *approved* transactions. PU learning and semi-supervised methods exploit unlabeled rejects.

**LLMs in the loop.** As of 2026 LLM agents are in production for the *investigation* tier, not the real-time decision — Unit21 reports ~a year of production agent use for alert triage and narrative drafting; FinCEN's 2026 proposed rule explicitly encourages AI provided it's explainable, validated, and keeps humans in the loop. SOC benchmarks (Simbian 2025) show frontier models hitting 61–72% autonomous triage accuracy — useful augmentation, not autonomy.

*Sources read: Booking.com tabular-transformer paper (arXiv 2405.13692); Springer GNN-fraud survey 2026; Thoughtworks GNN-prevention; AAAI 2025 graph-injection; Feast/Tecton/SageMaker PIT docs; Kozodoi reject-inference; Unit21 financial-crime agents; Stripe 3DS-trends; FraudGNN-RL.*

---

## 2. Problem framing, cost function, operating point (~360 words)

**[SIGNAL: scope-and-framing]** Before any model: I am designing for **card-not-present payment/transaction fraud** with an **inline authorization decision** (synchronous, pre-completion). I am explicitly *excluding* a single-system answer for account-takeover, synthetic-identity, and promo-abuse — they have different signal and label structure (ATO's label is a customer report in hours; synthetic identity's "label" may never arrive). I'll note where ATO needs a parallel pipeline. I commit to this scope because, per Rule 8, you cannot design one system for all fraud types.

**[SIGNAL: cost-function-is-the-design]** The model's job is **not classification** — it is to emit a *calibrated probability of fraud* `p(fraud | x)` that feeds a cost-sensitive decision. **[SIGNAL: saying-no]** I'll push back immediately on the reflex to optimize accuracy or F1: at <1% prevalence, "always approve" scores >99% accurate and catches zero fraud. Accuracy is disqualifying here.

The decision minimizes **expected dollar loss**, not error count. For a transaction of amount `A`, customer lifetime value `V`, and approval probability the model implies:

- **False negative (approve fraud):** cost ≈ `A + chargeback_fee` (the loss amount plus a $15–30 scheme fee, plus dispute-ops cost).
- **False positive (decline a good customer):** cost ≈ `margin(A) + friction + P(churn)·V + support_cost`. A wrongly declined $5 coffee from a 10-year customer can cost more in churn than the sale.

These costs are **asymmetric and per-transaction-varying** — they depend on `A` and `V`, so a single global threshold is wrong. The operating point is:

```
approve  if  E[cost_approve] < E[cost_decline] and < E[cost_challenge]
where
  E[cost_approve]   = p · (A + fee)
  E[cost_decline]   = (1−p) · (margin + churn_risk·V)
  E[cost_challenge] = challenge_friction + p·(1−pass_rate)·0 + (1−p)·abandon·margin
```

We pick the action with minimum expected cost **per transaction**, which yields *thresholds that move with amount and customer value*. Evaluation follows the cost function: **PR-AUC**, **recall at a fixed false-positive rate**, **calibration**, and **dollar-weighted recall** (fraction of fraud *dollars* stopped), never F1. The threshold is a business lever, not a model property.

---

## 3. Scope, capacity, and cost math (~300 words)

**[SIGNAL: capacity-and-cost-math]** Concrete numbers (Rule 10):

| Quantity | Value | Implication |
|---|---|---|
| Volume | 5M txns/day | ≈ **58 txns/s** average, **~300/s** peak burst |
| Fraud rate | 0.3% | **~15,000 fraud txns/day** — positives are not scarce *per day*, but… |
| Confirmed-label lag | 30–90 days | …recent positives are **unconfirmed**; trainable confirmed positives lag weeks |
| Avg fraud loss | $200 | FN cost baseline |
| Avg false-decline cost | ~$50 (margin + expected churn·V) | FP cost baseline |
| Inline latency budget (p99) | **100 ms total** | model + features get a slice |
| Analyst review capacity | ~2,000 cases/day | <0.04% of volume — **must prioritize by expected $** |

**Threshold arithmetic.** With FN=$200 and FP=$50, the naive break-even decline threshold (decline-vs-approve, ignoring the challenge tier) is where `p·200 = (1−p)·50` → `p* = 50/250 = 0.20`. So absent a challenge tier we'd decline above ~20% fraud probability — *far* below 50%, because missing fraud is 4× costlier than a false decline. This single number already refutes "decline if score > 0.5."

But the **challenge tier changes the math**: a step-up (3DS/OTP) costs only `challenge_friction` (~$1–3 of abandonment risk) and *recovers* legit customers while blocking fraud, so the challenge band sits *below* the decline band. We get two thresholds, `t_challenge ≈ 0.05` and `t_decline ≈ 0.40`, both tuned per-segment by minimizing the expected-cost integral on backtested data.

**Latency budget decomposition** (p99 = 100ms): network/orchestration ~20ms, online feature fetch ~15ms, rules ~2ms, GBDT inference ~5ms, graph-feature lookup (precomputed) ~10ms, anomaly score ~5ms, decision + logging ~5ms → ~62ms, leaving ~38ms headroom for tail. This budget is what *forbids* running a live GNN inline (more in §9).

---

## 4. High-level architecture (~340 words)

Two planes: a **real-time decision path** (hard latency budget) and an **offline training/feedback plane** (no latency budget, owns label generation and retraining).

```
                    ┌──────────────────── REAL-TIME PATH (p99 ≤ 100ms) ────────────────────┐
  txn ─▶ ORCHESTRATOR
          │
          ├─▶ [1] RULES ENGINE        (~2ms)  hard blocks, velocity limits, blocklists
          │        │ hard-fail? ─────────────────────────────▶ DECLINE
          │        ▼
          ├─▶ [2] ONLINE FEATURE FETCH (~15ms) Redis: velocity aggs, entity stats,
          │        │                            precomputed GRAPH features, device rep
          │        ▼
          ├─▶ [3] GBDT SCORER          (~5ms)  calibrated p(fraud)
          ├─▶ [4] ANOMALY SCORER       (~5ms)  isolation forest / AE  → novelty score
          │        ▼
          └─▶ [5] COST-SENSITIVE DECISION (~5ms)
                   │  expected-$ argmin over {approve, challenge, decline, review}
                   ▼
            APPROVE / STEP-UP CHALLENGE / DECLINE / QUEUE-FOR-REVIEW
          └────────────────────────────────────────────────────────────────────────────┘
                   │ every decision + features logged (point-in-time snapshot)
                   ▼
  ┌──────────────────────── OFFLINE PLANE (no latency budget) ────────────────────────┐
  │  Kafka ─▶ Flink streaming aggregates ─▶ ONLINE store (Redis) + OFFLINE store (PIT) │
  │  Chargebacks/disputes (30–90d) ─▶ LABEL JOIN ─▶ training table (censored!)         │
  │  Analyst reviews ─▶ fast labels                                                    │
  │  Graph builder (entities, edges) ─▶ GNN/community detect ─▶ graph feature tables   │
  │  Retraining (champion/challenger, shadow) ─▶ model registry ─▶ canary ─▶ serve     │
  │  Monitoring: score-dist drift, approval rate, CB rate, per-segment fairness        │
  └────────────────────────────────────────────────────────────────────────────────────┘
```

The logged point-in-time feature snapshot at decision time is the single most important artifact in the system: it is what makes training features identical to serving features (§8) and what lets us reconstruct exactly what the model knew when it decided (governance, §15).

---

## 5. The adversarial reality — the spine (~340 words)

**[SIGNAL: adversarial-reframe]** This is the defining property and I lead with it. In almost every ML problem the data-generating process is indifferent to your model. **In fraud it is an intelligent adversary who observes your decisions and adapts.** Drift here is not seasonal noise — it is *deliberate evasion*. The cruel consequence: **a model degrades precisely because it works.** When you start declining a fraud pattern, fraudsters stop using it and route around it; your best-performing rule today trains the adversary to make it useless tomorrow. A passive-drift mental model (retrain on a fixed cadence, assume stationarity between retrains) silently fails because the distribution shift is *caused by your own deployment*.

This reshapes the entire system:

- **Defense in depth, so no single model is a single point of evasion.** Rules + GBDT + graph + anomaly detector means an attacker who reverse-engineers the GBDT still trips the velocity rule or the anomaly layer. A monoculture (one classifier) is one thing to evade.
- **[SIGNAL: unsupervised-for-novelty] Anomaly detection for zero-day attacks.** The supervised model can only catch fraud *resembling labeled history*. A novel attack has no labels and no chargebacks yet, so the supervised score is blind to it. An unsupervised layer (isolation forest / autoencoder reconstruction error / density estimate) flags transactions that are simply *unlike anything normal*, catching the novel pattern in the weeks before labels exist. Tradeoff: high false-positive rate, so it routes to **review/challenge**, never to a hard decline.
- **Fast, asymmetric retraining and monitoring for attack onset.** The earliest signal of a new attack is a **sudden shift in the score distribution** or a spike in a feature's marginal — not a metric you can compute from labels (those are weeks away). Monitoring score-distribution drift is the early-warning radar.
- **Champion/challenger + shadow models** so a new model can be validated against live traffic before it owns decisions, and so we always have a fallback when the adversary breaks the champion.

If an answer treats this as ordinary concept drift, it has missed the problem. Everything downstream — anomaly layer, retraining cadence, drift monitoring — exists *because the data fights back*.

---

## 6. Class imbalance, calibration, thresholding (~330 words)

**[SIGNAL: imbalance-in-eval]** Imbalance is handled **mostly in evaluation and thresholding, not resampling.** At 0.3% positives, ROC-AUC is misleadingly flattering (the huge true-negative mass inflates it); **PR-AUC** is the honest summary because it focuses on the positive class. The operating metric is **recall at a fixed, business-acceptable false-positive rate** (e.g., "what fraction of fraud $ do we catch while declining ≤0.5% of legit traffic?").

**Resampling — rejected as the primary lever. [SIGNAL: rejected-alternative]** SMOTE fabricates synthetic positives by interpolating between existing fraud points. In an *adversarial* space this is actively harmful: it invents fraud that never existed and smears the decision boundary into regions the adversary doesn't occupy, while teaching nothing about the patterns the adversary *will* move to. It also corrupts calibration. I reject SMOTE-as-main-solution.

**What I use instead:**
- **Cost-sensitive learning at the algorithm level** — class weights / `scale_pos_weight` in LightGBM, or **focal loss** (down-weights easy negatives so gradient focus stays on hard, near-boundary cases). This handles imbalance without fabricating data.
- **Calibration is non-negotiable** because the decision (§2) consumes `p` as a *probability* in an expected-dollar comparison. Tree ensembles and especially focal-loss-trained models are *not* calibrated out of the box. I fit **isotonic regression** (more flexible, enough data) or Platt scaling on a held-out, temporally-later slice, and I monitor calibration in production (reliability curves per segment). An uncalibrated 0.2 that's really 0.5 silently approves fraud.

```
raw model margin ──▶ [isotonic calibration] ──▶ p(fraud) ∈ [0,1] ──▶ expected-$ decision
                          ▲ fit on temporally-later holdout, re-fit each retrain
```

Failure mode: calibrating on a random split leaks future information and produces a model that looks calibrated offline and isn't online. Calibration data must be *temporally after* training data, mirroring deployment.

The deepest point: imbalance is not a data problem to be "fixed," it's a property of the cost geometry. The right response is to optimize the cost-weighted objective and read results through imbalance-aware, dollar-weighted metrics.

---

## 7. Label latency, noise, and selection bias — the deepest area (~500 words)

**[SIGNAL: censored-labels]** This is what separates fraud fluency from generic classification, and it breaks the standard supervised loop in three compounding ways.

**(a) Latency.** You don't know a transaction was fraud until a chargeback/dispute arrives — typically 30–90 days. So your *recent* data (the data most relevant to the current adversary) has *no confirmed labels yet*. You're always training on a partially-blind, weeks-stale view while the adversary operates in the present.

**(b) Noise.** Labels are wrong in both directions. **Friendly fraud** (a real customer disputes a legitimate charge) injects false positives into your fraud labels. **Silent fraud** (the victim never notices/disputes) means true fraud is labeled "good." Chargeback ≠ fraud, and no-chargeback ≠ legitimate.

**(c) Selection bias / censoring — the deep one.** **You only observe outcomes for transactions you approved.** Declined transactions never complete, so you *never learn* whether they would have been fraud. Your training distribution is shaped by your own past decisions — the model is trained on the slice it previously chose to let through. Over time the model becomes confident about the region it approves and blind to the region it blocks; the adversary, who probes the blocked region, lives in your blind spot. **[SIGNAL: feedback-loop-awareness]** This is a feedback loop: today's model censors tomorrow's training data.

```
            ┌────────────── THE CENSORED LABEL LOOP ──────────────┐
            │                                                       │
  txn ─▶ MODEL ─▶ decision                                          │
            │        ├─ DECLINE ─▶ ✗ outcome NEVER observed ────────┘  (censored region grows)
            │        ├─ CHALLENGE ▶ pass/fail = a CHEAP fast label
            │        └─ APPROVE ──▶ wait 30–90d ──▶ chargeback? ──▶ label (noisy, delayed)
            │                                            │
            │                          friendly-fraud + silent-fraud noise
            ▼                                            ▼
   model trained ONLY on what it approved ◀──── training table (biased + delayed + noisy)
```

**Mitigations, concrete:**

1. **Hold-out random-approval slice.** Approve a tiny random fraction (e.g., 0.1–0.5%) of transactions the model *would have declined*, to get **unbiased** labels in the censored region. This is expensive — you're knowingly approving some fraud — so it's budgeted as a *cost of learning* and capped at a dollar limit. It is the only source of truly unbiased boundary labels.
2. **The challenge tier as a label generator.** A step-up converts an uncertain block into an observable outcome: pass → likely legit, fail/abandon → likely fraud, at *low* cost and *low* latency. This is the highest-leverage way to shrink the censored region without eating fraud loss (§10).
3. **Reject inference / PU learning.** Treat declined transactions as unlabeled (not negative) and use semi-supervised methods (self-learning à la Kozodoi, S4VM, or deep generative RI) to infer their likely status, correcting the train-distribution mismatch. Use with caution — RI can launder the model's own bias back in if naively applied.
4. **Provisional + analyst labels** to shorten latency: manual-review decisions are high-quality labels available in hours, not weeks (§12); early fraud signals (e.g., issuer fraud alerts) give provisional labels before the chargeback clears.

Failure mode to call out: training on chargebacks only, on a random split, with no reject inference — you'll ship a model that's excellent on the slice you already approve and blind exactly where the adversary attacks.

---

## 8. Feature platform: aggregates, parity, point-in-time (~440 words)

**[SIGNAL: feature-platform-parity]** A single transaction's raw fields (amount, MCC, BIN) are weak signal. The strong signal is in **behavioral aggregates and graph structure** (§9). The feature platform is most of the system.

**Feature families:**
- **Velocity/aggregate** — counts/sums over sliding windows per entity: `#txns_on_card_last_1h`, `sum_amount_last_24h`, `#distinct_merchants_last_10m`, `#declines_last_1h`. These are the backbone of fraud signal.
- **Behavioral / relational-to-history** — `is_new_device_for_account`, `geo_velocity` (distance from last txn ÷ time — flags impossible travel), `amount_vs_account_p95`, `time_since_last_txn`.
- **Entity reputation** — device-fingerprint risk, IP/ASN reputation, merchant fraud-rate history, BIN risk.
- **Cold-start** — for a brand-new card/device/account with no history, aggregates are null. Fall back to population priors and segment-level features, and route more aggressively to the *challenge* tier rather than decline (we lack evidence either way).

**The architecture — and why parity is the silent killer:**

```
        ┌──────────────── SHARED FEATURE DEFINITIONS (write once) ────────────────┐
        │                                                                          │
  events──▶ Flink/Kafka streaming agg ──┬──▶ ONLINE store (Redis)  ── serve: ~15ms │
        │                               │      "num_txn_last_10m" read at decision │
        │                               └──▶ OFFLINE store (PIT)  ── train joins    │
        │                                                                          │
        │  POINT-IN-TIME JOIN: training row at time T sees ONLY values known < T   │
        │  (event-time, not processing-time) — no future leakage                   │
        └──────────────────────────────────────────────────────────────────────────┘
```

Two distinct correctness problems, both fatal if missed:

1. **Point-in-time correctness.** A training row for a txn at time `T` must use feature values computed from data *strictly before* `T`. If `num_txn_last_24h` accidentally includes the txn itself or later txns, the model trains on the future, looks brilliant offline, and collapses in production. The offline store must store every feature value with its *event-time* and do as-of joins.

2. **Online/offline parity (train/serve skew) — the silent killer.** The exact same feature must be computed identically at training time (batch, over historical logs) and serving time (streaming, single record). If the batch job computes `geo_velocity` with a different timezone, rounding, null-handling, or window boundary than the online path, the model is fed a *different distribution* at serve time than it trained on, and accuracy quietly bleeds with no error thrown. The 2026 mitigation (Databricks RTM, Tecton, Feast) is **single feature definitions** that compile to both batch and streaming — "no logic drift." This is non-negotiable.

**Rejected alternative [SIGNAL: rejected-alternative]:** computing features ad hoc inside the serving service with separate offline ETL for training. It's faster to ship and *guarantees* skew. The feature store's whole reason to exist is to make this impossible.

---

## 9. Graph / linkage layer (~340 words)

**[SIGNAL: graph-linkage]** Modern fraud is coordinated: rings, mule-account chains, synthetic-identity clusters. The i.i.d. assumption underlying row-based GBDTs breaks — the signal lives *in the edges*, not the nodes. A card that looks clean in isolation is damning when it **shares a device with 50 flagged accounts**, reuses an IP/shipping-address with a known ring, or matches a behavioral fingerprint of a cluster.

```
                 ┌──────────── ENTITY GRAPH ────────────┐
   device_42 ───shares──▶ acct_A ──uses──▶ card_X        │
       │                    │                  │          │
     shares               same-IP           same-addr     │
       ▼                    ▼                  ▼          │
   acct_B ◀──flagged    acct_C ◀──flagged   acct_D        │
                                                          │
   Confirmed fraud on acct_B ──▶ suspicion PROPAGATES ──▶ │ risk↑ on device_42,
                                                          │ acct_A, card_X, acct_C…
                 └────────────────────────────────────────┘
```

**Design choice — features offline, not a live GNN inline. [SIGNAL: rejected-alternative]** A live multi-hop GNN forward pass cannot fit the ~10ms graph slice of a 100ms budget at 300 txns/s, and GNNs amplify imbalance through aggregation and are attackable via graph injection (AAAI 2025). So:

- **Precompute graph features offline / near-real-time** (community IDs, ring membership, shared-entity counts, neighbor fraud-rate, GraphSAGE-style *inductive* embeddings that handle new nodes without full retrain) and write them to the **online store** as plain features the GBDT reads in O(1). Update incrementally as edges arrive.
- **Propagation on confirmed fraud:** when a chargeback confirms fraud on a node, raise risk across its linked subgraph immediately (this is fast, rule-like, and doesn't need the latency budget — it updates the online feature for the *next* txn).
- **Community detection** surfaces dense suspicious clusters distinct from legitimate high-velocity customers (a power-seller hub vs. a mule fan-out).

**Rejected alternative:** GNN as the *primary inline scorer*. It buys 12–25% offline AUROC on relational fraud but loses on latency, calibration, adversarial robustness, and explainability. Better to *distill* the graph's relational signal into features and let the calibrated, interpretable GBDT make the call. The GNN earns its place offline.

The graph layer is also where ATO and ring-takedown investigations live — a single confirmed account compromise lets analysts walk the subgraph and proactively challenge linked accounts.

---

## 10. Model architecture and the graded system (~440 words)

**[SIGNAL: graded-response]** Real fraud defense is **defense-in-depth with a graded response**, not a single classifier with one threshold. The graded action set is the staff move because it converts a binary block into a spectrum that recovers legit customers while stopping fraud.

```
   txn
    │
 ┌──▼─────────────────────────────────────────────────────────────────────┐
 │ TIER 0 — RULES (fast, interpretable)                                     │
 │   known-bad BIN/device, velocity limit breach, blocklist  ──▶ DECLINE    │
 │   (also hard ALLOW for trusted recurring / allowlist)                    │
 └──┬───────────────────────────────────────────────────────────────────────┘
    │ pass
 ┌──▼─────────────────────────────────────────────────────────────────────┐
 │ TIER 1 — ML SCORE   p = calibrated GBDT  ; a = anomaly/novelty score     │
 │          features include precomputed graph signals                      │
 └──┬───────────────────────────────────────────────────────────────────────┘
    │  expected-$ decision over thresholds (per-segment, amount/value-aware)
    ▼
 p < t_chal ────────────────────────────────────────────────▶ APPROVE
 t_chal ≤ p < t_dec  OR  high anomaly a ───▶ STEP-UP CHALLENGE (3DS / OTP)
 p ≥ t_dec  (and low value)  ───────────────────────────────▶ DECLINE
 p ≥ t_dec  (high value / high LTV / uncertain) ────────────▶ MANUAL REVIEW QUEUE
```

**The components and why:**
- **Rules layer** — cheap, deterministic, instantly editable. When a new attack breaks through, a rule is the *fastest* mitigation (minutes) while the model retrains (hours/days). Also enforces hard regulatory/limit constraints.
- **GBDT scorer** — the calibrated workhorse (§1 rationale: tabular dominance + SHAP reason codes).
- **Anomaly scorer** — routes novelty to challenge/review, never hard-decline (§5).
- **Per-segment vs global. [SIGNAL: rejected-alternative]** I use a **global model with strong segment features** plus **per-segment thresholds**, rather than fully separate per-segment models. Separate models fragment the already-scarce positive signal and multiply maintenance/monitoring surface; segment-aware thresholds capture most of the benefit (a new-user $5,000 txn and a 10-year-customer $5 txn get different operating points) without splitting the data.

**[SIGNAL: graded-response] The challenge tier is the highest-leverage friction-vs-safety lever.** A step-up (3DS challenge, OTP, biometric push) costs only a small abandonment risk and *recovers* legitimate customers a hard decline would have lost, while stopping fraudsters who can't pass it. Stripe's 2026 data shows SCA-region optimization can lift conversion ~1.2% *while* cutting fraud ~7.7% — the challenge band is where that win lives. It also carries a **liability-shift** benefit under 3DS (fraud chargeback liability moves to the issuer on authenticated transactions), which feeds directly back into the cost function: an authenticated challenge *changes the FN cost*, so the decision is liability-aware, not just probability-aware.

**Rejected alternative:** single XGBoost + one threshold + hard decline. It's the mid-level baseline; it has no recovery path for false positives, no novelty defense, and no fast-mitigation lever.

---

## 11. Real-time serving and fallback (~300 words)

**[SIGNAL: real-time-budget]** The decision is inline and synchronous, before auth completes. Budget decomposition from §3 (p99 ≤ 100ms): orchestration 20 + feature fetch 15 + rules 2 + GBDT 5 + graph-feature read 10 + anomaly 5 + decision/log 5 ≈ 62ms, ~38ms tail headroom. This budget is *why* graph signal is precomputed and the GNN is offline (§9), and why the inline model is a bounded-depth GBDT, not a deep ensemble.

**Fallback — a fraud-specific policy decision, not a default.** When the feature store is slow/unavailable or the model times out, you must choose:

```
   feature/model unavailable?
        │
        ├─ FAIL-OPEN  → approve anyway   : protects conversion, but a degraded
        │                                  dependency becomes an OPEN FRAUD WINDOW
        │                                  the adversary will probe for and exploit
        └─ FAIL-CLOSED→ decline/challenge : stops fraud, but a feature-store outage
                                            becomes a mass false-decline / outage of
                                            your own revenue
```

The right answer is **neither globally** — it's **graded and segmented**: fail *open* for low-amount, high-trust segments (cost of a missed small fraud < cost of declining everyone); fail to *challenge* for the uncertain middle (cheap insurance); fail *closed* (decline) for high-amount/high-risk segments. Critically, an adversary *will* learn to trigger your degraded path (e.g., flood to cause feature-store latency) and ride the fail-open window — so the fallback policy itself must assume adversarial probing, and degraded-mode decisions get logged and rate-limited.

**Async alternative [SIGNAL: rejected-alternative]:** near-real-time with *provisional approval* and retroactive clawback (approve instantly, score in the background, reverse/hold if high-risk). Good for low-value, reversible flows; rejected as the *primary* design for card auth because most card fraud isn't cleanly reversible and the auth decision is genuinely inline. Used selectively for specific reversible flows.

---

## 12. Human review queue and label generation (~290 words)

**[SIGNAL: human-loop-labels]** The review queue is not a dumping ground — it is a **label-generation engine** that attacks the latency problem from §7. Analyst decisions are **high-quality labels available in hours**, not the 30–90 days a chargeback takes. Every reviewed case shrinks the censored region and feeds the next retrain.

**Capacity-aware prioritization.** With ~2,000 review slots/day against 5M txns, the queue can cover <0.04% of volume, so **the model must rank the queue by expected value of review**, not by raw score. Expected value ≈ `P(decision flips) × dollar_impact` — prioritize high-amount, genuinely-uncertain cases (near the threshold) where a human label both prevents a large loss *and* maximally informs the model. Routing a confidently-fraud $5 txn to a human wastes a scarce slot.

```
  uncertain/high-value txns ─▶ REVIEW QUEUE (priority = E[$ impact of a flip])
        │
        ▼
  ANALYST + LLM-ASSIST  ──▶ decision  ──▶ FAST LABEL ──▶ training table
   (case summary,                              │             (shortens latency,
    linked-entity pull,                        │              de-biases censored region)
    narrative draft)                           ▼
                                        action: approve / clawback / block ring
```

**LLM-assisted triage (2026).** Per the research pass, LLM agents are in production for *investigation*, not the real-time decision. They draft case summaries, pull the linked subgraph (§9), check watchlists, and surface the 3 features driving the score — cutting analyst time per case and raising consistency (Unit21 ~1yr production; FinCEN 2026 rule encourages this *with humans in the loop*). I deliberately keep the LLM **out of the inline decision**: it's too slow for the budget, not calibrated, and adverse-action law wants deterministic reason codes. The human + LLM accelerate label production and ring takedown; the GBDT makes the call.

---

## 13. Evaluation (~340 words)

**[SIGNAL: dollar-weighted-eval]** Metrics follow the cost function (§2), not classification convention.

- **PR-AUC** as the primary ranking metric (ROC-AUC is inflated by the negative mass at 0.3% prevalence).
- **Recall at fixed FPR** — "what % of fraud do we catch at ≤0.5% false-decline rate?" — the metric ops and finance actually negotiate.
- **Dollar-weighted recall** — **% of fraud *dollars* caught, not % of fraud transactions.** A model that catches 90% of fraud *count* but misses the 10% of high-value attacks can be a net loss. Fraud loss is heavy-tailed; optimize the tail.
- **Expected-cost curve** — sweep thresholds, plot total expected $ (FP cost + FN cost) vs operating point, pick the minimum. This *is* the evaluation; everything else is diagnostic.
- **Calibration** (reliability curves, per segment) — because the decision consumes probabilities.

**Backtesting — temporal splits only. [SIGNAL: saying-no]** Never random splits. Fraud has temporal structure (the adversary moves) and entity leakage (the same card/device in train and test). A random split leaks the future and inflates every metric. Train on `[t0, t1]`, validate on `[t1, t2]`, test on `[t2, t3]` — and crucially, **only evaluate on labels that are now confirmed**, accepting that the most recent weeks are still censored. Use a **rolling-origin** backtest to mimic the production retrain cadence.

```
  |--- train ---|-- calib --|-- test --|  ← time →   (rolling origin, no shuffling)
                                  ▲ labels confirmed (older);  recent weeks still censored
```

**The counterfactual problem.** Standard metrics can't tell you what the *declined* transactions would have done — you never observed them (§7). So offline metrics systematically overstate real performance on the censored region. Two honest tools: evaluate on the **random-approval hold-out slice** (the only unbiased view of the boundary), and run **shadow/champion-challenger online** to measure a new model's decisions against live outcomes before it owns traffic. Offline PR-AUC is necessary but never sufficient; the unbiased slice and shadow deployment are what you actually trust.

---

## 14. Feedback loops, retraining, monitoring (~290 words)

**[SIGNAL: feedback-loop-awareness]** The model biases its own future training data (§7) — so I close the loop *deliberately*, not accidentally.

**Retraining cadence under adversarial drift.** Cadence is driven by the adversary, not the calendar. Baseline: frequent scheduled retrains (e.g., daily/weekly incremental) *plus* **event-triggered retrains** when drift monitors fire. Because confirmed labels lag, retrains lean on fast labels (analyst, challenge-tier, provisional) for recency and confirmed labels for ground truth.

**Champion/challenger + shadow + canary.** New models run in **shadow** (score live traffic, make no decisions) to measure calibration and decision agreement, then **canary** on a small traffic slice, then promote. Always keep the prior champion as instant rollback when the adversary breaks the new model.

**Monitoring — the early-warning radar (labels are too slow):**
```
  ┌─ score-distribution drift ──▶ sudden shift = likely NEW ATTACK (acts before labels)
  ├─ approval-rate & decline-rate by segment ──▶ spikes = model or attack
  ├─ challenge pass-rate & abandonment ──▶ friction creeping up?
  ├─ chargeback rate (lagging truth) ──▶ confirms what drift hinted weeks earlier
  ├─ feature freshness / null-rate ──▶ pipeline break = silent skew
  └─ per-segment fairness (decline rate by protected group / geo) ──▶ §15
```

**Incident path** when a new pattern breaks through: (1) detect via score-dist/CB-rate anomaly, (2) **ship a rule** as immediate mitigation (minutes), (3) route the pattern to review for fast labels, (4) retrain with the new labels + reject inference, (5) promote via shadow→canary, (6) retire the temporary rule once the model absorbs the pattern. The rule layer buys the hours the model needs — this is *why* defense-in-depth exists.

The discipline: treat the censored-label feedback loop as a designed component (random-approval slice + challenge-tier labels + reject inference continuously de-bias the training set), not an accident you discover when the model quietly rots.

---

## 15. Explainability, fairness, governance (~250 words)

**[SIGNAL: fairness-and-explainability]** These are *design constraints*, not an afterthought.

**Adverse-action / explainability.** In regulated contexts (and increasingly everywhere — FinCEN's 2026 rule conditions AI use on explainability and validation), a decline may legally require a **reason**. This is a first-class reason the inline scorer is a **GBDT with SHAP reason codes**, not a black-box net or LLM: every decline carries the top contributing features (`geo_velocity`, `new_device`, `velocity_1h`) as auditable reason codes. The decision, the point-in-time feature snapshot, the model version, and the threshold are **all logged** for audit and dispute reconstruction.

**Fairness in declines.** A false-positive (wrong decline) is not just lost revenue — if false-positive *rates* skew by protected group, geography, or income proxy, it's discriminatory exclusion from payments, a legal and ethical failure. So fairness monitoring is built into §14's dashboard: **decline rate and FPR disaggregated by protected attributes and geography**, with alerting on disparities. Watch for proxy leakage — ZIP code, device type, and BIN can encode protected attributes; a feature that boosts AUROC by riding a proxy is unacceptable even if "accurate."

**Governance.** Model registry with versioning and lineage; documented validation (the temporal backtest + unbiased-slice evaluation of §13) before promotion; the random-approval slice is itself governed (capped dollar exposure, documented as a learning cost); audit logging of every automated decision. The challenge tier helps here too — a step-up is a *less harmful* error than a wrongful decline, so routing uncertainty to challenge reduces both fairness exposure and friction.

---

## 16. Tradeoffs taken and what would change them (~150 words)

- **GBDT inline + graph features offline, GNN offline only.** Chosen for latency, calibration, explainability, and adversarial robustness. *Would change* if inline graph hardware (sub-10ms multi-hop) matured or if relational fraud dominated loss enough to justify a calibrated, distilled inline GNN.
- **Global model + per-segment thresholds**, not per-segment models. *Would change* if a segment (e.g., a new geography/product) had genuinely distinct signal and enough volume to support its own model and monitoring.
- **Random-approval hold-out slice** despite knowingly approving some fraud. *Would change* if regulatory or loss limits forbade it — then lean harder on challenge-tier labels and reject inference, accepting more censoring bias.
- **Inline synchronous decision**, not async provisional. *Would change* per-flow for reversible, low-value rails where clawback is clean.
- **Fail policy graded by segment.** *Would change* with dependency reliability and the measured adversarial exploitation rate of degraded mode.

---

## 17. What I would push back on (~150 words)

**[SIGNAL: saying-no]** Three pushbacks on the framing as posed:

1. **"Approve / decline / challenge" understates the system.** The real product decision includes a **manual-review tier and a random-approval learning tier**. If the interviewer wants a pure approve/decline binary, I'd argue that throws away the cheapest false-positive recovery (challenge) and the only unbiased label source (random slice) — both load-bearing, not optional.

2. **One system for "fraud" is the wrong scope.** I committed to CNP payment fraud. ATO, synthetic identity, and promo abuse have different signal, label latency, and decision points; bundling them yields a mediocre everything-detector. I'd insist on naming the fraud type before architecting.

3. **The biggest risk isn't model quality — it's the censored-label feedback loop.** A team can ship a high-PR-AUC model that quietly rots because it only ever learns from what it approved. If forced to cut scope, I'd cut model sophistication before I'd cut the de-biasing machinery (random slice + challenge labels + reject inference). That's the part that's genuinely hard and the part generic classification answers omit.