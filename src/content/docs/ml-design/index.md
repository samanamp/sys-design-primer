---
title: "ML System Design: 10 End-to-End Problems"
description: "Ten general end-to-end ML system design problems for a modern ML role, weighted toward world models, diffusion, and inference/serving optimization. Each is a complete real-world problem designed in full — framing, metrics, data, modeling, serving, evaluation, monitoring, and rollout."
---

# ML System Design — End-to-End Problems

This section preps the **ML System Design** interview: a mostly **general** exercise that tests whether you can take an ambiguous, real-world problem and design a working ML system **end to end** — framing, metrics, data, modeling, serving, evaluation, monitoring, and rollout. It is *not* a trivia round and usually *not* hyper domain-specific. Expect general "design X" prompts; **one** question may brush a **world model, diffusion model, or inference/serving optimization** — the areas to steer toward if you can.

Each of the 10 problems below is a **complete real-world problem**. A strong answer walks the *entire* arc, not one slice. The set is intentionally general, with three problems (6, 7, 8) that let you show depth in world models, diffusion, and inference optimization.

## The end-to-end blueprint (apply to every problem)

This is the skeleton each answer fills in. Walk it every time.

1. **Clarify & scope** — Objective, users, constraints, scale (QPS, data size), latency/cost budget, failure modes. Resolve ambiguity *before* designing.
2. **Frame the ML problem** — Inputs, outputs, the ML task, and the **real-world objective** vs. the **training objective** (they differ — optimize a proxy, get judged on the product).
3. **Success metrics** — Offline *and* online, **slice-based** and robustness-aware, not one aggregate number. Name the metric you'd actually optimize.
4. **Fallback when uncertain/wrong** — What the system does on low confidence or novel input: abstain, default safely, escalate to a human, degrade gracefully.
5. **Data & labeling** — Sources, label quality, train/val/test splits, leakage prevention, and a concrete plan for the **long tail** / weak slices.
6. **Baseline first** — A simple, credible system (heuristic / linear / small model). Then justify complexity only where a constraint demands it.
7. **Modeling** — Architecture, why it beats the baseline, and the trade-offs you're buying.
8. **Serving & systems** — Batch vs. online, feature generation, **latency/throughput/cost**, caching, model versioning. (Your wheelhouse — go deep here.)
9. **Evaluation** — Offline ↔ online mismatch, calibration, A/B testing, failure analysis on slices.
10. **Rollout & monitoring** — Shadow → canary → full, rollback plan, **drift detection**, and retraining triggers.

Always narrate trade-offs (accuracy vs. latency, complexity vs. reliability) and adapt as the interviewer adds constraints. **Start simple, then layer in complexity** — that progression is itself what they're grading.

---

## Template (fully worked) — Problem 6

I worked the **world-model** problem as the template since it's closest to your role (inference & optimization of world models). The same blueprint produces every other page.

### 6. Design a world model that predicts how an environment evolves

> **Problem.** Design a learned **world model**: given the current state and a candidate action/context, predict how the environment evolves over the next horizon, so a downstream system (planner, simulator, RL agent) can roll out and evaluate possible futures. Cover training *and* the production inference path you'd own.

**1) Clarify & scope.** What state (raw sensor frames? a compact latent? structured agents?), what horizon, and who consumes it (closed-loop planner vs. offline scenario gen)? Constraints: it must roll out **many futures fast** — so per-step inference latency and the autoregressive step count dominate the design. Budget: e.g., predict 5 s at 10 Hz, many parallel rollouts, within a tight onboard/cluster latency target.

**2) Frame the ML problem.** Input: history of states (+ actions/conditioning). Output: a **distribution** over next states (futures are multimodal — never a single deterministic frame). Training objective: next-state prediction likelihood / denoising loss; real-world objective: rolled-out futures are **realistic, diverse, and stay stable** over the horizon. The gap between one-step accuracy and long-horizon rollout quality is the crux.

**3) Metrics.** One-step: reconstruction / NLL. What actually matters: **multi-step rollout** fidelity — distributional realism (do speeds/positions/dynamics match held-out logs?), **calibration** of uncertainty, diversity (mode coverage, not collapse), and **stability** (does it diverge or hallucinate after N steps?). Slice by scenario type and horizon length. One aggregate loss hides rollout drift.

**4) Fallback when uncertain/wrong.** Detect out-of-distribution states (high predicted uncertainty / low likelihood) and signal **low confidence** to the consumer so it widens safety margins or falls back to a simpler conservative model. Never emit a confident hallucinated future silently.

**5) Data & labeling.** Mostly **self-supervised** from logged trajectories (the future *is* the label) — cheap and abundant. Real work is curation: dedup near-identical highway miles, **mine the long tail** (rare interactions, edge dynamics), and split by **time/geography** to prevent leakage. Optionally augment with sim for rare regimes.

**6) Baseline.** A deterministic next-state regressor (e.g., a recurrent or short transformer predicting the next latent). Credible, fast, exposes where determinism fails (mode averaging / blur).

**7) Modeling step-up.** Move to a **generative** world model to capture multimodality — typically a **latent diffusion** or autoregressive transformer over a learned latent (encode state → model dynamics in latent space → decode). Latent space is the key choice: it makes rollouts cheap and is where most of your inference optimization lives.

**8) Serving & inference optimization (go deep — it's the job).** Rollouts are autoregressive, so cost = steps × per-step cost × number of futures. Levers:
  - **Fewer denoising steps** for the diffusion dynamics — distillation / consistency / few-step samplers to cut a 50-step sampler to a handful.
  - **Latent-space rollout** so you decode to pixels/state only when needed, not every step.
  - **KV-cache** for the autoregressive backbone; **batch parallel rollouts** across futures on the GPU.
  - **Quantization (FP8/INT8)** and kernel fusion for the dynamics net.
  - **Caching / amortization** of the encoder for shared history across rollouts.
  Trade off each against rollout fidelity — and *measure* the fidelity cost of every speedup.

**9) Evaluation.** Offline rollout metrics on held-out logs; **closed-loop** eval (does a planner using the world model behave better/safer?) to catch offline↔online mismatch; targeted failure analysis on long-horizon divergence and mode collapse.

**10) Rollout & monitoring.** Ship behind **shadow mode** (world model runs, predictions logged, not acted on) → **canary** → full. Monitor input drift (new environments), rollout-stability metrics, and uncertainty calibration in production; trigger retraining when realism on a slice degrades.

**Central tension:** rollouts must be **fast enough** to evaluate many futures in real time, yet **faithful and stable** over a long horizon. Your inference-optimization work (step reduction, latent rollouts, quantization, batching) is precisely the lever that buys speed without giving up fidelity — make that trade-off the spine of the answer.

---

## The other 9 problems

Each is a complete end-to-end problem; expand into its own page using the blueprint and the template above.

### 1. Design a recommendation / personalized ranking system
> Recommend the most relevant items (feed, products, videos) for each user at scale.

End-to-end scope: candidate generation → ranking → re-ranking, the **two-tower retrieval + cross-encoder rank** pattern, features (user/item/context) and a feature store, label definition (clicks vs. dwell vs. conversion) and **feedback-loop bias**, offline metrics (nDCG/AUC) vs. **online A/B** (engagement), cold-start fallback, low-latency serving, and freshness/retraining.

### 2. Design a fraud / abuse detection system
> Catch fraudulent or abusive activity in near-real-time on a heavily **imbalanced** stream.

End-to-end scope: framing as ranking/anomaly vs. classification, the rare-positive metric set (**PR-AUC, recall at fixed precision**, cost-weighted), labels with delay/feedback, the **fallback** (flag for human review vs. auto-block), real-time feature serving, threshold calibration, and adapting to an **adversary that shifts** (drift, retraining cadence).

### 3. Design a demand / time-series forecasting system
> Forecast demand / load / ETA to drive downstream decisions.

End-to-end scope: horizon and granularity, baseline (seasonal naïve) before ML, features (calendar, lags, exogenous), **backtesting without leakage**, point vs. **probabilistic** forecasts and which the consumer needs, metrics (MAPE/quantile loss) sliced by segment, and retraining on drift.

### 4. Design a search & retrieval ranking system
> Return the best results for a user query over a large corpus.

End-to-end scope: lexical (BM25) + **semantic (embeddings/ANN)** hybrid retrieval, learning-to-rank reranker, query understanding, label sources (clicks/human judgments), metrics (nDCG/MRR) and online A/B, index freshness/sharding, and latency budget across the retrieve→rank pipeline.

### 5. Design a content classification / moderation system
> Classify or moderate content (toxicity, spam, policy) at platform scale.

End-to-end scope: multi-label framing, **precision/recall trade-off** tied to enforcement cost, labeling and policy drift, the **human-in-the-loop** fallback for uncertain cases, multimodal inputs, inline-latency serving, and monitoring for new evasion patterns.

### 7. Design a diffusion-based generative system and optimize its inference
> Build a generative model (images / trajectories / scenarios) and make it cheap enough to serve.

End-to-end scope: latent diffusion architecture and conditioning, training/data curation, **the iterative-sampling cost** and how you cut it (fewer steps, distillation/consistency, **query-aware cascades** — light model for easy inputs), batching/quantization/caching, quality metrics (FID / task metrics) vs. latency, safety filtering, and a quality-vs-cost rollout. (Pairs directly with your inference-optimization focus.)

### 8. Design the inference & serving stack for a large model
> Serve a large model under heavy QPS with tight latency and cost targets.

End-to-end scope: **continuous batching, KV-cache / paged attention**, prefill vs. decode, **quantization (FP8/INT8)**, speculative decoding, tensor/pipeline parallelism, prefix/result caching, autoscaling, distinguishing **TTFT vs. throughput** and tying each to a product need, plus cost/SLO monitoring. (This *is* your job framed as a design problem.)

### 9. Design a personalization / online-learning system
> Adapt model behavior to each user/context, updating quickly from fresh interactions.

End-to-end scope: online vs. batch updates, exploration/exploitation (**bandits**), the feedback loop and its bias/runaway risks, guardrails against degenerate personalization, low-latency feature + model serving, and A/B measurement of long-term vs. short-term reward.

### 10. Design the ML platform: experimentation & safe rollout
> Design the meta-system that lets teams evaluate and ship model changes safely.

End-to-end scope: tiered gates (offline eval → **shadow → canary → full**), **A/B framework**, guardrail metrics, automated **rollback**, feature/serving consistency (train-serve skew), model registry/versioning, **drift monitoring**, and retraining triggers feeding back into the pipeline.

---

## AV specialization deep-dives (fully worked)

If the role leans toward autonomous driving, expect the "design X" prompt to land on a **perception/forecasting, simulation, or end-to-end-driving** problem. These three are worked end to end at staff level using the same blueprint, and they deliberately reuse one engine — a **scene-conditioned (often diffusion) generator** of trajectories/futures — across three serving regimes: **onboard with bounded latency**, **offline at huge throughput**, and **in the planning loop**. The contrast in serving strategy is itself a strong interview point. (Background substrate: the [Diffusion section](/diffusion/).)

| # | Problem | Specialization | The core thing it tests |
|---|---|---|---|
| [5](/ml-design/5-motion-forecasting/) | Onboard motion forecasting | Motion forecasting | Calibrated multimodal multi-agent prediction under a hard onboard latency budget; never drop the dangerous mode. |
| [6](/ml-design/6-closed-loop-simulation/) | Closed-loop driving simulator | Simulation | Realistic, reactive, diverse sim-agents + scenario gen at massive throughput, validated by sim-to-real correlation. |
| [7](/ml-design/7-e2e-driving/) | End-to-end driving with a world model | End-to-end driving | Joint optimization via interpretable-intermediate end-to-end + world-model planning, bounded by a hard safety layer. |

---

## One-line cheat sheet

| # | End-to-end problem | The core thing it tests |
|---|--------------------|-------------------------|
| 1 | Recommendation / ranking | Retrieval→rank, online vs. offline metrics |
| 2 | Fraud / abuse detection | Imbalance, drift, adversary |
| 3 | Demand forecasting | Backtesting, probabilistic forecasts |
| 4 | Search & retrieval | Hybrid retrieval + LTR rerank |
| 5 | Content moderation | Precision/recall vs. enforcement cost |
| **6** | **World model** | **Fast, stable, multimodal rollouts** |
| **7** | **Diffusion + inference opt** | **Sampling cost vs. quality** |
| **8** | **Large-model serving** | **Batching, KV-cache, latency vs. throughput** |
| 9 | Personalization / online | Bandits + feedback-loop safety |
| 10 | ML platform / rollout | Proving a change is safe to ship |

## Closing stance

Every answer follows the same arc — **clarify → frame (real-world vs. training objective) → metric you'd optimize → fallback when uncertain → data & long-tail plan → baseline → justified modeling → serving within latency/cost limits → offline+online eval → shadow/canary/rollback with drift monitoring.** Keep it general and start simple; the ten problems are that arc applied ten times. Problems **6–8** are where you can show world-model, diffusion, and inference-optimization depth — Problem 6 is the worked template.
