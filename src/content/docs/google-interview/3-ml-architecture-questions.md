---
title: "ML Architecture Rounds: Probable Questions"
description: "The likely question pool for the two 'AI/ML System Architecture in your domain' rounds — ranked by probability, mapped to the four stated focus areas, with format intel from reported Google staff loops."
---

# ML Architecture Rounds: Probable Questions

Two rounds of "AI/ML System Architecture within your domain specialization." Research finding stated plainly: **no verbatim questions from this exact team exist publicly** — it's too niche. This list is derived from the round's own four focus areas, the sister co-design role's JD, and reported questions from adjacent Google ML loops.

## Format intel (solid evidence)

This maps to Google's RRK/domain deep-dive slot: 45–60 min, one interviewer, **one main problem with escalating follow-ups** — discussion-driven, sometimes no diagram at all. It leans heavily on **your past projects**: expect "tell me about a system you built" threads probed for tradeoffs. Prepare two of your own projects as tradeoff narratives (what you chose, what you gave up, what you measured, what you'd do differently) — that prep is as important as any question below.

## Tier 1 — prepare full answers (high probability)

1. **The co-design question:** "Take a strong OSS model designed on GPUs (Llama/Qwen-class). Design a TPU-friendly model that beats it on quality-per-dollar. What changes and why?" → [worked answer](/google-interview/4-answer-tpu-friendly-model/)
2. **The serving question:** "Design the serving system for a Gemini-class model on TPU: TTFT/TPOT SLOs, minimal cost per token, production monitoring." → [worked answer](/google-interview/5-answer-tpu-serving/)
3. **The training question:** "Design end-to-end distributed training for a ~1T-param MoE across TPU pods — parallelism layout, data pipeline, failure handling, and what you monitor." → [worked answer](/google-interview/6-answer-pod-training/)

## Tier 2 — prepare skeletons (medium probability)

4. **The diagnosis variant:** "An OSS model ported to TPU runs 40% below roofline projection. Walk me through finding and fixing it." — the co-design question in reverse; your trace-reading + [TPU primer](/optimization/18-tpu-xla-optimization/) material, spoken over xprof.
5. **The cost variant:** "Cut serving cost 3× at fixed quality." — the decision tree: quantization (AQT int8; FP8 on Ironwood) → distillation (Gemini Flash precedent) → speculative decoding (EAGLE-3-class, MTP) → batching/disaggregation; each with its accuracy gate and measurement.
6. **The kernel question:** "When do you drop below XLA to Pallas, and walk me through a kernel you'd write." — have one concrete story ready (e.g., a fused ragged-attention or quantized-matmul kernel; splash attention and Ragged Paged Attention v3 are the public reference points).
7. **The classic MLE staple done to staff depth:** YouTube-recommendations / content-moderation / Smart-Compose-style autocomplete, end to end — data pipeline, features/embeddings (SparseCore is your differentiator here), training, serving under a latency budget, drift/retraining/monitoring. Reported repeatedly in adjacent Google loops.

## Tier 3 — be ready to improvise (the "Key Qualities" wildcard)

8. **Product invention:** "Propose a product feature where ML is key, then architect it within a latency/cost budget." First-principles → paradigm mapping → architecture; they explicitly list this ability.
9. **Data-pattern probe:** "Here's a system's telemetry/usage pattern — what ML opportunity do you see?" Practice narrating from data to hypothesis to system.

## The spoken skeleton for every answer

State the regime (prefill/decode; compute/memory/comms-bound; latency/throughput) → arithmetic before architecture (roofline, bytes/token, chips needed — out loud) → name the binding bottleneck → design around it → two deliberate deep dives → failure modes and monitoring → one pushback on the prompt's requirements with numbers. See the [battle plan](/google-interview/0-overview/) for the rubric this maps to.
