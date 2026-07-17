---
title: "Worked Answer: Gemini-Class Serving on TPU"
description: "A staff+ worked interview answer: multi-region serving for a Gemini-class MoE model on Ironwood TPUs — capacity math first, prefill/decode disaggregation, SLO-aware scheduling, goodput monitoring, and cost-per-token levers."
---

"Design the serving system for a Gemini-class model on TPU: p99 TTFT ≤ 800ms, p99 TPOT ≤ 40ms, multi-region, minimize cost per token, with production monitoring." (45–60 min, one interviewer, escalating follow-ups.)

This is a worked answer in interview order. The mechanics of each technique live in [LLM serving optimization](/optimization/14-llm-serving-optimization/); the program-management version of the TTFT problem lives in [TTFT optimization](/optimization/1-ttft-optim/). This document doesn't re-derive those — it shows how to *deploy* them on TPU under these SLOs, with the arithmetic out loud.

---

## 1. Opening Moves — Scope Before Architecture (3–4 minutes, no more)

Four questions, each of which changes the design; state working assumptions immediately so the interview doesn't stall:

1. **Traffic shape.** Requests/s at peak, peak-to-trough ratio, burstiness. *Assume: 20K req/s global peak, 3:1 peak-to-trough, diurnal per region.*
2. **Context distribution.** p50/p95/p99 input length, output length. *Assume: input p50 ~6K, p95 ~32K, p99 ~128K (uploads, RAG, code); output mean ~300 tokens.* The p99 TTFT SLO is decided almost entirely by that input tail.
3. **One model or a family?** Flash-class + Pro-class tiers, or one? *Assume: one flagship model here; I'll note where a family changes the fleet.* Also: revision cadence — weekly model pushes change the weight-distribution story.
4. **Batch/offline API on the same fleet?** *Assume: yes, as a scavenger class — it's the single best cost lever and I'll design it in rather than bolt it on.*

And one framing statement: **TTFT is a prefill/queueing problem, TPOT is a decode memory-bandwidth problem, and they compete for the same chips.** The whole design is about refusing to let them compete.

---

## 2. Capacity Math First — Out Loud

I do this before drawing boxes, because the box diagram is the same for every candidate; the arithmetic is not. All numbers are order-of-magnitude and I'd re-derive them from profiled measurements before believing them.

**Model.** Gemini-class: assume a ~400B-total-parameter MoE with ~40B active per token (the [MoE](/optimization/6-mixture-of-experts/) shape that makes flagship serving affordable). Serve weights in FP8/int8 — Ironwood has native FP8 (4,614 TFLOPS/chip), and int8 via AQT is the proven TPU path (see [quantization](/optimization/16-quantization/)).

**Bytes.**
- Weights at 1 byte/param: **~400 GB**. Doesn't fit one chip → the serving unit is a multi-chip slice.
- KV per token: assume ~64 layers, GQA with 8 KV heads × head_dim 128, FP8 KV → 2 × 8 × 128 × 1 B × 64 ≈ **128 KB/token**. An 8K-context request holds ~1 GB of KV; a 128K request holds ~16 GB.

**Chip.** Ironwood: 192 GB HBM at 7.37 TB/s, ICI ~1.2 TB/s, sold as 4-chip VMs, scaling to 256-chip and 9,216-chip pods.

**Serving slice.** Pick 8 chips (two 4-chip VMs, ICI-connected): 1,536 GB HBM, ~59 TB/s aggregate bandwidth. Weights take 400 GB, leaving ~1.1 TB for KV — room for ~130 concurrent 8K-context requests, fewer with long-context mix.

**Decode ceiling (roofline).** Decode is memory-bound: each step streams the weights once (MoE experts amortize across a large batch — at batch 128 essentially every expert is touched) plus each request's KV:
- Weights: 400 GB / 59 TB/s ≈ 6.8 ms/step.
- KV at batch 128 × 8K avg: 128 GB / 59 TB/s ≈ 2.2 ms/step.
- Ideal ≈ 9 ms; at a realistic ~60% achieved bandwidth → **~15 ms/step**, i.e. ~15 ms TPOT at batch 128. Comfortably inside 40 ms p99 with headroom for long-context mix and interference — headroom I will spend deliberately, not accidentally.
- Compute check: 2 × 40B = 80 GFLOP/token × 128 = 10 TFLOP/step vs. ~15 PFLOPS effective slice compute → sub-millisecond. Memory-bound confirmed; this is why FP8 KV matters as much as FP8 weights.
- **Throughput: ~128 tokens / 15 ms ≈ 8.5K output tokens/s per 8-chip decode slice**, ~1,070 tokens/s/chip.

**Prefill ceiling.** Prefill is compute-bound: 80 GFLOP/token. On an 8-chip slice at ~40% MFU FP8 (~14.8 PFLOPS effective): an 8K prompt ≈ 44 ms; a 128K prompt ≈ 700 ms — *right at* the 800 ms budget before queueing, network, or attention's quadratic term. That single number motivates chunked prefill, prefix caching, and a dedicated prefill pool. Sustained: ~185K prefill tokens/s per slice.

**Fleet.** Peak 20K req/s × 300 output tokens = 6M output tokens/s → 6M / 8.5K ≈ **~700 decode slices ≈ 5,600 chips**. Prefill: 20K × 6K input = 120M tokens/s raw; at a 60% prefix-cache hit rate → 48M tokens/s → 48M / 185K ≈ **~260 prefill slices ≈ 2,100 chips**. Add ~30% for regional peaks not coinciding, failover headroom, and canaries: **~10K chips ≈ 40 × 256-chip pods**, spread over 3+ regions. That's roughly one 9,216-chip Ironwood superpod's worth of silicon — a useful sanity check that the ask is "a pod-scale fleet," not a data-center-scale one.

Every number above has ±2× error bars; the *method* — bytes/token, roofline per slice, tokens/s → slices → chips — is the deliverable. In week one I'd replace all of it with measured MFU and achieved bandwidth.

---

## 3. Architecture

```
Client → Global anycast LB → Regional gateway (GKE Inference Gateway)
  → llm-d-style router: prefix-cache-aware + load-aware + SLO-class-aware
    → PREFILL POOL (8-chip slices, chunked prefill, prefix cache)
        └─ KV transfer (ICI within superblock; DCN across) ─┐
    → DECODE POOL (8-chip slices, continuous batching) ◄────┘
  KV tiers: HBM → host DRAM offload → (regional prefix store)
```

- **Gateway/router.** GKE Inference Gateway (GA) with an llm-d-style scheduler: routes on prefix-cache overlap, queue depth, and SLO class — not round-robin. Cache-aware routing is a multiplier on everything downstream; without it, prefix caching delivers a fraction of its value ([why](/optimization/14-llm-serving-optimization/#5-prefix-caching-and-cache-aware-routing)).
- **Engine.** The current TPU stack is **tpu-inference** (the unified vLLM TPU backend, Oct 2025 — ~3.6× E2E over the prior vLLM-TPU on v6e-1 for Llama-3.1-8B; chunked prefill, prefix caching, KV host-offload, Eagle3/ngram speculation). JetStream is archived (Feb 2026) — proposing it dates you.
- **Prefill/decode disaggregation.** Separate pools; deep dive in §4. Pathways-based disaggregated serving on TPU has reported ~60% throughput gains; the DistServe/Mooncake lineage established the pattern on GPUs.
- **Batching.** Continuous batching on decode; chunked prefill *within the prefill pool* to bound head-of-line blocking for the 128K tail. Note the tension I'll return to: chunked prefill's original purpose — interleaving with decode — is mooted by disaggregation, so on a dedicated prefill pool chunks should be large (maximize MFU) and exist only for scheduling granularity and memory bounds.
- **KV management.** Paged KV via Ragged Paged Attention v3; prefix caching keyed on token-block hashes; host-DRAM offload for turns likely to resume (multi-turn chat gaps). Long-context KV strategy per [KV cache & long context](/optimization/8-kv-cache-long-context/).
- **Speculative decoding.** Eagle3 or ngram (both in tpu-inference) as a *conditional* lever: at low-to-moderate load it cuts TPOT 1.5–2.5×; at saturation the extra verify compute and wasted rejected tokens *reduce* goodput, and acceptance rate drops on high-temperature/creative traffic. It ships behind a load-aware switch, off above a batch-occupancy threshold ([mechanics](/optimization/9-speculative-decoding/)).
- **Quantization path.** FP8 weights + FP8 KV on Ironwood, int8 AQT fallback; per-slice eval gates before any quantized revision ships. XLA-level tuning per [TPU/XLA optimization](/optimization/18-tpu-xla-optimization/).

---

## 4. Deep Dive 1 — Disaggregation: Pool Sizing and the KV Transfer Bill

**Why disaggregate here specifically.** The two SLOs are adversarial on shared chips: one 128K prefill (~700 ms of solid compute) stalls every colocated decode stream — that's 700 ms of TPOT jitter against a 40 ms p99. Chunked prefill alone softens this but caps prefill MFU. With a 128K p99 input tail, separation is the honest answer, and the ~60% Pathways throughput number says the industry agrees.

**Pool sizing is a ratio you retune continuously.** From §2: prefill demand ≈ 260 slices, decode ≈ 700 slices → roughly **1:2.7 prefill:decode** at this workload. But the ratio is a function of (input length × cache hit rate) / output length — a product launch that doubles average prompt length flips it. So the autoscaler scales the pools *independently* on their own signals: prefill pool on queue-wait p95 (drives TTFT), decode pool on batch occupancy / KV headroom (drives TPOT).

**The KV transfer bill.** After prefill, the full KV must move to a decode slice:
- 8K-context request: 8K × 128 KB = **1 GB**. 128K request: **16 GB**.
- Over ICI (~1.2 TB/s/chip, and transfers stripe across the slice): ~1 GB moves in single-digit ms — negligible against an 800 ms TTFT budget. This is why prefill and decode slices for a given request should live in the same ICI domain (same pod/superblock) whenever possible.
- Over DCN between pods: at an effective few-hundred GB/s per slice, 16 GB is tens to ~100+ ms — tolerable for the p99 tail, poisonous if it's the common path.
- **Placement rule:** the router picks a (prefill, decode) pair inside one ICI domain; DCN transfer is the fallback, and its rate is a monitored metric with an alert threshold. Transfer overlaps with the last prefill chunks (Mooncake-style layer-wise streaming), so it hides in the shadow of compute rather than adding to TTFT.
- **Interviewer trap to name unprompted:** disaggregation resurrects the chunked-prefill question. On the prefill pool there's no decode to protect, so run large chunks (~8–16K tokens) for MFU, chunked only to bound scheduler latency and enable early KV streaming. Anyone who copies GPU-colocated chunked-prefill configs onto a disaggregated TPU pool is paying attention-recompute overhead for a benefit they no longer receive.

**Failure isolation bonus:** a poison-pill prompt that OOMs or wedges a prefill slice takes out prefill capacity, not 128 in-flight decode streams.

---

## 5. Deep Dive 2 — SLO-Aware Scheduling, Autoscaling, Multi-Region

**Goodput, not utilization.** The scaling metric is the fraction of chip-time producing tokens *within SLO* — Google's own framing decomposes goodput into scheduling × runtime × program goodput. A 95%-utilized decode slice at 60 ms TPOT is a *worse* fleet member than an 80%-utilized one at 25 ms. Utilization-based autoscaling actively fights the SLO: it packs batches until TPOT breaches, then scales — after the damage. Scale-out signals: decode batch-occupancy vs. the occupancy→TPOT curve measured for *this* model on *this* chip, and prefill queue-wait p95 with slack against the TTFT budget.

**Scale on warm state.** A new decode slice is useless until 400 GB of weights are resident. From regional SSD-backed cache at ~10 GB/s effective, that's ~40 s just for weights, plus server start and compilation-cache warm — call it **2–5 minutes cold-start**. So: (a) predictive scaling on the diurnal curve, provisioning ahead of the ramp; (b) a warm pool of weight-loaded, traffic-less slices sized to the historical burst delta; (c) XLA compilation caches pre-populated — a recompile storm on a new revision is its own outage class.

**Admission control and priority classes.** Three classes: interactive (full SLO), standard, and batch/offline (no TTFT SLO, scavenger). When goodput headroom shrinks, shed in reverse order — batch API work drains *first*, which is precisely why it lives on this fleet: it's the compressible load that turns burst absorption into revenue instead of idle headroom. Interactive requests that would breach TTFT anyway (queue estimate > budget) are rejected fast at the gateway — a quick 429 with retry-after beats a 3 s TTFT for everyone behind it ([admission control](/optimization/14-llm-serving-optimization/#6-admission-control)).

**Multi-region.** Three-plus regions, anycast front door, per-region fleets sized so that N-1 regions absorb peak (that's part of the 30% headroom in §2). Two problems people skip:

1. **Weight distribution is a huge-file fan-out problem.** 400 GB × thousands of slices × weekly revisions. Naive pulls from a single blob store melt it. Design: region-local mirrored store, tree/peer-to-peer fan-out within a pod (chips pull shards from ICI neighbors, not from the network), content-addressed shards so a revision that touches 10% of weights ships 10% of bytes. Rollout is region-by-region canary with per-revision perf CI (below), and the previous revision stays resident on disk for instant rollback.
2. **Failover is a KV-loss event.** Cross-region failover drops prefix caches and any in-flight KV — the surviving regions see a cache-hit-rate crater and thus a *prefill demand spike* exactly when they've absorbed extra traffic. The failover capacity model must size the prefill pool for the post-failover (low-hit-rate) regime, not the steady-state one. This is the kind of arithmetic that only exists if you did §2 out loud.

Reliability culture reference: the Gemini training report (arXiv 2312.11805) describes goodput at scale going 85% → 97% via redundant in-memory state — the serving-side analog is the same instinct: measure goodput, keep warm state redundant, make recovery fast rather than failures rare.

---

## 6. Monitoring — What I'd Actually Page On

- **SLO surfaces:** TTFT and TPOT p50/p95/p99 *sliced by input-length bucket, region, and priority class* — a global p99 hides a single region or a single length bucket burning. Plus **goodput under SLO** as the headline fleet metric, decomposed scheduling/runtime/program so a regression is attributable.
- **Leading indicators:** prefill queue-wait p95 (leads TTFT breach), decode batch occupancy and KV headroom (leads TPOT breach and OOM-driven preemption), DCN KV-transfer rate (leads TTFT tail), prefix-cache hit rate per region (a slow drift here silently re-inflates the prefill fleet bill).
- **Speculation health:** acceptance-rate distribution per traffic slice. Acceptance drift is both a perf signal and a *model-change* signal — a new revision or a shifted traffic mix shows up here first.
- **Per-revision perf CI:** every model or server revision runs a fixed replay set (representative length mix, open-loop load) and must hold TTFT/TPOT/goodput within tolerance before regional rollout — benchmarking methodology per [LLM serving optimization §11](/optimization/14-llm-serving-optimization/#11-benchmarking-methodology). Quality evals gate quantization and speculation changes the same way.
- **SDC awareness:** at 10K chips, silent data corruption is a when, not an if. Lightweight online checks (logit-distribution canaries, periodic golden-prompt checksums per slice) plus a quarantine path for suspect chips. This is a fleet-scale habit, not paranoia.

---

## 7. Cost per Token — Derivation and the Three Levers

Cost/token = (chips × $/chip-hr) / (tokens/hr within SLO). Illustrative: at ~$2/chip-hr, a decode chip producing ~1,070 tok/s → ~**$0.52 per million output tokens** decode-side; prefill adds its share via (input tokens / hit rate). The absolute number is assumption-laden; the *ranking of levers* is robust:

1. **Prefix-cache hit rate.** Prefill is ~a quarter of the fleet at 60% hit rate; every 10 points of hit rate is ~65 slices of prefill capacity. Cheapest capacity you'll ever buy — it's a routing and cache-keying problem, not silicon.
2. **Batch-API infill.** The interactive fleet is sized for peak; trough + headroom is ~40–50% of chip-hours. Scavenger-class batch work converts that to sold tokens at near-zero marginal cost. This can halve effective cost/token by itself.
3. **Occupancy against the TPOT curve.** Ride decode batch size up to the occupancy where p99 TPOT still clears 40 ms — the gap between "safe default" and "measured frontier" is often 1.5–2× throughput. FP8/quantization and speculation are real but smaller multipliers after these three.

---

## 8. Escalating Follow-Ups

**"Would you run the batch API on the same fleet?"** Yes — designed in from the start (§5): scavenger priority class, preempted first, checkpointable at token granularity (KV snapshot to host DRAM so preemption wastes nothing irrecoverable). The alternative — a dedicated batch fleet — buys isolation at the cost of the single biggest cost lever. I'd only split fleets if batch demand grows to where preemption churn measurably degrades interactive goodput.

**"Context jumps to 1M — what breaks?"** Everything KV-shaped. 1M × 128 KB = **128 GB of KV per request** — one request eats a chip's worth of HBM under this KV layout. Responses: much more aggressive GQA/MLA-style KV compression and/or sliding-window layers (a model co-design conversation, not purely serving); KV offload to host DRAM with layer-wise streaming for the decode working set; prefill for 1M is minutes of compute → it becomes a *batch-like* product tier with its own SLO, chunked and checkpointed; DCN KV transfer at 128 GB stops being tolerable, so 1M requests pin prefill+decode into one ICI domain. And I'd push back on product: what p99 TTFT is a 1M-token request actually entitled to? 800 ms is not a physical answer here.

**"You get a Trillium (v6e) fleet instead — 32 GB @ 1.6 TB/s. What changes?"** The serving-unit math reruns: weights alone need ≥ 13 chips → a 16-chip slice has 512 GB (400 weights + only ~100 GB KV — KV-starved) so realistically 32-chip slices; aggregate bandwidth 32 × 1.6 = 51 TB/s, similar step time to the 8-chip Ironwood slice but with 4× the chips and 4× the ICI collective surface per step. Consequences: per-token interconnect overhead rises (more all-to-all hops for MoE dispatch), KV capacity per slice is the binding constraint → prefix caching and host offload go from "cost lever" to "load-bearing," long-context requests may simply not fit → tighter admission by length, and FP8 isn't native → int8 AQT path. Cost per token likely rises ~2× even at equal $/FLOP. Same method, different constants — which is why the method was the deliverable.

**"Your p99 TTFT is breaching but p50 is fine — first three graphs?"** Prefill queue-wait p95 by region (queueing vs. compute), TTFT sliced by input-length bucket (is it the 128K tail or 4K requests stuck behind it), DCN KV-transfer rate (did ICI-domain placement degrade). Then cache hit rate — a hit-rate drop masquerades as a prefill capacity problem.

---

## 9. Staff+ Signals

- **Arithmetic first, boxes second.** Bytes/token → roofline per slice → tokens/s → chips, stated before any architecture, with error bars owned out loud.
- **Goodput vs. utilization distinction**, and autoscaling signals derived from the SLO curve, not from chip busy-ness.
- **Naming the chunked-prefill × disaggregation conflict unprompted** — the config that's right colocated is wrong disaggregated.
- **Failover reasoned as a KV/cache event**, not just a traffic event; prefill pool sized for the post-failover hit-rate regime.
- **Speculative decoding as conditional**, with the load- and temperature-dependence stated, behind a switch, monitored by acceptance rate.
- **Current-stack awareness:** tpu-inference not JetStream (archived), GKE Inference Gateway GA, Pathways disagg numbers — and skepticism: every vendor number gets re-measured on our workload before it enters the capacity plan.
- **Honesty about measurement:** the week-one deliverable is replacing every constant in §2 with profiled values; the design survives them changing, the plan doesn't pretend to know them.

## What Falls Short

- **Feature-listing** vLLM/tpu-inference flags — continuous batching! paged attention! prefix caching! — without sizing anything. Every candidate says the same nouns; the arithmetic is the differentiation.
- **Utilization-based autoscaling** proposed as if it were neutral. It's not neutral; it's anti-SLO.
- **Ignoring multi-region weight distribution and cold start** — "just autoscale" with a 400 GB weight payload and minutes-long cold starts is a design that fails its first traffic burst.
- **Speculative decoding with no quality/latency/load caveat** — presenting it as a free 2× is the tell that the candidate has never watched acceptance rate collapse under a saturated, high-temperature workload.
- **Treating the 800 ms / 40 ms SLOs as unquestionable** for every length bucket. Staff behavior includes negotiating a stratified SLO for the 128K/1M tail rather than silently over-building the fleet for it.
