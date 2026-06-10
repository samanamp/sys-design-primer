---
title: "Crib Sheet: Numbers & One-Liners to Know Cold"
description: One page of memorizable numbers, staff one-liners, and the answer spine for the ML design set
---

# Crib sheet — rehearse from this, not the docs

*Every number comes with its one-line derivation: if you remember the derivation, you can rebuild the number live, which is worth more than reciting it.*

## The answer spine (walk every question through this)

**1. Clarify & frame** → **2. Metrics** (training vs real objective; per-slice; fallback when wrong) → **3. Data** (sources, leakage, tail) → **4. Baseline, then complexity** → **5. Serving & systems** → **6. Eval & monitoring** (shadow, canary, rollback, retrain triggers) → **7. Safety / long tail**.

## World-model numbers (designs #1–#3)

| Number | Derivation |
|---|---|
| **12K tokens / 100 ms step** | 720p → VAE f8 → 160×90 latents → 2×2 patch = 3.6K/cam × 3 cams + 2K lidar |
| **120K tok/s per stream** | 12K × 10 Hz — kills token-by-token AR decode (LLMs do tens/s) |
| **Teacher $36/mile** | 2·30B·12K ≈ 0.72 PF/pass × 30 steps = 22 PF/step ÷ 0.4 PF/s ≈ 55 GPU-s/step × 1,200 steps/mile ≈ 18 GPU-hr × $2 |
| **1,200 steps/mile** | 30 mph → 120 s/mile × 10 Hz |
| **Student $0.24/mile (~150×)** | 3B, 4 steps, FP8: 0.29 PF/step ÷ 0.8 PF/s ≈ 0.36 GPU-s/step → 0.12 GPU-hr/mile |
| **57 KB/token KV (FP8)** | 28 layers × 2(K,V) × 8 heads × 128 dim × 1 B → frame 0.7 GB, 12-frame window 8.2 GB |
| **~9 branches/H100** | (80 GB − 4 GB weights) ÷ 8.2 GB window — fanout is HBM-bound, not compute-bound |
| **Attention 8× matmuls (full)** | 4 × 12K × 144K × 3072 × 28 ≈ 5.9e14 vs 0.7e14; factorized spatial+temporal → ~20% tax |
| **TP4 misses 100 ms** | 23 ms compute + 16 ms comms (56 all-reduces × 75 MB @ ~400 GB/s) → 39 ms/pass × 4 = 156 ms |
| **Tier costs per mile** | replay $0.0001 → object-sim $0.001 → student $0.25 → teacher $36 |
| **Distill data : training = 50 : 1** | 50K teacher miles ≈ $1.8M vs 3B training ≈ 2.6e22 FLOPs ≈ $37K → payback < 1 day ($3.6M vs $24K/day fleet) |
| **Validation ≈ $400/night** | 10K segments × 20 s ≈ 1,700 mi × $0.24 (+5% teacher audit ≈ $3K) |
| **WOSAC realism ~0.78** | 2025 leaders on Realism Meta-metric — SOTA sim agents still measurably non-human |

## LLM-platform numbers (design #4)

| Number | Derivation |
|---|---|
| **48 tok/s unbatched** | 3.35 TB/s HBM ÷ 70 GB FP8 weights — decode = one full weight-read per token |
| **160 KB/token KV** | 80 layers × 2 × 8 KV-heads × 128 × 1 B → 32K stream = 5 GB; TP2 leaves ~90 GB → ~18 long streams |
| **Prefill 4K ≈ 1 s** | 2 × 70e9 × 4096 ≈ 0.57 PFLOP ÷ ~0.45 PF/s — compute-bound (decode is bandwidth-bound) |
| **$0.37 / M output tokens** | TP2, batch 32: 70 GB ÷ 6.7 TB/s ≈ 10.5 ms/step → ~3,000 tok/s/pair at $2/GPU-hr |
| **16 KB vs 75 MB all-reduce** | LLM decode moves 1 token (latency-bound, benign); world-model pass moves 12K tokens (bandwidth-bound, ~40% tax) — same formula, opposite regime |
| **EAGLE 3.1: 2.0× / 1.7× / 1.66×** | at concurrency 1 / 4 / 16 — spec decode survives moderate batch now; profile the crossover |
| **B200 ≈ 2.4× decode ceiling** | ~8 TB/s HBM — constants move, regime logic doesn't |

## Staff one-liners (the sentences that score)

1. **Napkin math first** — the arithmetic forces the design; do it before proposing architecture.
2. **Sim-time ≠ wall-clock** — only humans/hardware in the loop need real time; bulk fleet optimizes miles/GPU-hr.
3. **Amortize before you distill** — generate once with the teacher, replay many times for free.
4. **The gate is downstream agreement per slice** — an efficient simulator that changes the planner's evaluations is worse than a slow one.
5. **Silent evaluation drift** is the failure mode — the standing teacher audit never stops.
6. **Portfolio, not monolith** — route every question to the cheapest tier that answers it.
7. **DP adds streams, never faster ticks** — a closed loop is sequential; only model-splitting, fewer steps, or faster chips cut tick latency.
8. **Overlap turns compute + comms into max(compute, comms)** — and the last chunk's all-reduce is always exposed.
9. **"Realistic enough — for what?"** — validity is claim-scoped, per slice; unconditional realism is unfalsifiable.
10. **Open-loop flatters, closed-loop tells the truth** — the gap *is* the error-accumulation rate → max trusted horizon.
11. **Backtest, don't philosophize** — "has the sim predicted road outcomes across past releases?" is falsifiable and self-renewing.
12. **The tail is partially unverifiable — say so in the artifact** — certified / screened / exploratory tiers.
13. **The distillation contract is distributional** — the enemies are mode averaging (ghost pedestrian) and mode dropping (the tail, again).
14. **Data strategy beats loss strategy** — generation outweighs training 50:1, so free-ride on teacher byproducts.
15. **Off-policy students ace open-loop and drift closed-loop** — on-policy correction (Self-Forcing/DAgger) is the fix.
16. **Build the pipeline, not the artifact** — quarterly teachers make distillation CI with a lag metric; the tenth student is push-button.
17. **State the binding constraint before optimizing** — fanout looks compute-bound; the arithmetic says HBM.
18. **Goodput per traffic class, not tokens/s** — 100% utilization missing every SLO is a failed platform at great efficiency.

## Architecture lineage in one breath

RSSM/latent-RNN: tiny but blurry → GAN: one pass but drops modes (and the long tail *is* the low-density modes) → AR transformer: streamable + KV-cacheable but error-accumulating → diffusion/DiT: best fidelity, pays N denoise steps → **2025–26 convergence: bidirectional diffusion teacher distilled into a causal, few-step, KV-cacheable student** (CausVid → Self-Forcing → Causal Forcing) — which is also why prefix-sharing/branching became the central serving primitive.
