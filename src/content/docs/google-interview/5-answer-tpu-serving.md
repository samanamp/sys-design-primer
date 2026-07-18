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

**Chip.** Ironwood: 192 GB HBM at 7.37 TB/s (note: 192 GB is two chiplets × 96 GB — two memory spaces, not one flat pool, which the partitioner has to respect), ICI ~1.2 TB/s, sold as 4-chip VMs, scaling to 256-chip and 9,216-chip pods.

**Blended KV per request — compute it once, use it everywhere.** The mix is tail-heavy but the *mean* is what memory and bandwidth see: roughly 90% of requests near 6K, ~8% near 32K, ~2% near 128K → **E[len] ≈ 11K tokens** ≈ 1.4 GB of KV, call it **~1.5 GB/request with fragmentation and paging slack**. (Using the p50 here — or worse, mixing "8 GB blended" in one place and "1 GB" in another — makes every downstream number inconsistent; deriving one blended constant and propagating it is the whole game.)

**Serving slice.** Pick 8 chips (two 4-chip VMs, ICI-connected): 1,536 GB HBM, ~59 TB/s aggregate bandwidth. Weights take 400 GB, leaving ~1.1 TB for KV — at ~1.5 GB blended that's capacity for **~600 resident requests**. KV *capacity* is therefore not what limits the batch; the bandwidth knee below is.

**Decode-slice memory map** — where every gigabyte of a 1,536 GB decode slice goes, and what one 128K request does to it:

```
8-CHIP IRONWOOD DECODE SLICE — 1,536 GB HBM (8 × 192 GB)
──────────────────────────────────────────────────────────────────────
|■■■■■■■■■■■■ WEIGHTS 400 GB ■■■■■■■■■■■■|▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒
 FP8 MoE, all experts resident            KV PAGE POOL ~1,000 GB
                                          ▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒|░░░|▓▓|
                                                    workspace ~90 ─┘    └─ headroom ~46
                                                    (activations,       (paging slack,
                                                     collectives,        OOM guard)
                                                     compile bufs)

KV pool at the blended mix (~1.5 GB/request incl. paging slack):
  slot capacity ≈ 600 resident requests — NOT the limiter
  operating batch 128 (bandwidth knee, see roofline) → ~200 GB typical
  occupancy; the other ~800 GB is deliberate headroom for the 128K
  tail, failover re-warm, and the batch class — not waste
One p99 request:   128K ctx = 16 GB ≈ 11 blended slots    — priced in
One 1M request:    128 KB × 1M = 128 GB ≈ 13% of the pool
Eight 1M requests: > entire KV pool                        — see §8
```

**Decode ceiling (roofline).** Decode is memory-bound: each step streams the weights once (MoE experts amortize across a large batch — at batch 128 essentially every expert is touched) plus each request's KV:
- Weights: 400 GB / 59 TB/s ≈ 6.8 ms/step.
- KV at batch 128 × ~1.4 GB blended (the slack pages aren't streamed): ~180 GB / 59 TB/s ≈ 3 ms/step.
- Ideal ≈ 10 ms; at ~60% achieved bandwidth — an *assumption*, and a sensitive one: at 50% achieved the step is ~20 ms and every throughput number below shrinks ~20% — → **~16 ms/step**, i.e. ~16 ms TPOT at batch 128. Inside 40 ms p99 with headroom for long-context mix and interference — headroom I will spend deliberately, not accidentally.
- Compute check: 2 × 40B = 80 GFLOP/token × 128 = 10 TFLOP/step vs. ~15 PFLOPS effective slice compute → sub-millisecond. Memory-bound confirmed; this is why FP8 KV matters as much as FP8 weights.
- **Throughput: ~128 tokens / 16 ms ≈ 8K output tokens/s per 8-chip decode slice**, ~1,000 tokens/s/chip.

**The batching-vs-TPOT curve** — step time ≈ (400 GB weights + B × 1.4 GB blended KV) / 35.4 TB/s achieved. Weights amortize across the batch, so tokens/s/chip climbs steeply, then the KV term takes over and TPOT walks toward the ceiling:

```
tok/s/chip                                p99 TPOT (long-ctx mix ≈ +60% over p50 —
2000 ┤                                    ● B=512   ← 32 ms p50 / ~50 ms p99: BREACH
     │                                     (the +60% is an assumed inflation; widens
     │                              ●  B=384          if the 128K share grows)
     │                                   ↑ 27 ms p50 / ~42 ms p99: BREACH
1500 ┤                        ●  B=256   ← 21 ms p50 / ~34 ms p99: no headroom
     │                   ●                  ····40 ms p99 TPOT CEILING····
1000 ┤          ◉  B=128   ←── OPERATING POINT: 16 ms p50 / ~26 ms p99
     │       ●             ~1,000 tok/s/chip, headroom for tail + interference
 500 ┤    ●  B=64   ← 14 ms p50 — ~60% of the throughput for 2 ms of latency
     │  ● B=32     (this is why "run at low batch to be safe" is a ~2× cost error)
   0 └┴────┴────┴────┴────┴────┴────┴──── batch (resident requests)
     8   32   64  128  192  256  384  512
```

The picture that justifies goodput-aware scheduling: throughput rises monotonically but the *SLO-legal* region ends where the measured occupancy→p99-TPOT curve crosses 40 ms — around **B ≈ 250** for this mix. Note what limits the batch: the bandwidth knee, *not* KV capacity — the pool holds ~600 residents but the SLO stops admitting near 250. Utilization-based autoscaling drives you up this curve past the ceiling; goodput-based scaling parks you at the knee.

**Prefill ceiling — derived, both terms.** Prefill is compute-bound, and it has *two* FLOP terms that cross over with sequence length:

```
Per-token FFN/MoE:  2 × 40B active        = 80 GFLOP/token   (linear in s)
Causal attention:   ≈ 2 · L · d_model · s²                    (quadratic in s)
                      (L = 64, d_model ≈ 8,192)
Slice compute:      8 × 4,614 TF FP8 = 36.9 PF peak × 40% MFU target
                    ≈ 14.8 PF effective  (MFU is a TARGET with a wide range —
                     20–30% is common for MoE prefill; at 30% every fleet
                     number below grows ~1.3×)

time = work / rate: PFLOP of work ÷ 14.8 PFLOP/s effective
s = 8K:    FFN 8,192 × 80 G = 0.66 PFLOP ÷ 14.8 PFLOP/s → 44 ms;  attn 0.07 PFLOP → ~5 ms (noise)
s = 32K:   FFN ~2.6 PFLOP → ~175 ms;                              attn ~1.1 PFLOP → ~75 ms (visible)
s = 128K:  FFN 10.5 PFLOP → ~710 ms;                              attn ~18 PFLOP → ~1.2 s (DOMINANT)
Crossover: attn = FFN at s ≈ 80 G / (2·L·d) ≈ 76K tokens
```

So **a fully uncached 128K prefill on one slice is ~2 s — 2.5× the 800 ms budget**, not marginally over it; and attention runs below GEMM MFU, so 2 s is the friendly estimate. I say that plainly rather than hide it: what covers the p99 in practice is (a) prefix caching — most 128K traffic re-reads uploaded/RAG prefixes, (b) chunked prefill parallelized across slices for the residual (context-parallel prefill is what makes the ultra tier feasible at all), and (c) an explicit SLO carve-out negotiated for *uncached* ≥128K requests (§8). Sustained throughput: at the fleet's *mean* length (11K), attention adds ~15% over the FFN-only rate → **~160K prefill tokens/s per slice** at the target MFU (185K FFN-only).

**Fleet.** Peak 20K req/s × 300 output tokens = 6M output tokens/s → 6M / 8K ≈ **~750 decode slices ≈ 6,000 chips**. Prefill demand is driven by the **mean** input length, not the median — with this tail E[len] ≈ 11K vs. a p50 of 6K, and sizing on the p50 under-builds the pool by ~2×. (Median-vs-mean here is an explicit interview signal: queues and fleets bill by the mean.) Raw: 20K × 11K ≈ **220M tokens/s**; at a 60% prefix-cache hit rate — an assumption, and a load-bearing one: every 10 points is ~120 prefill slices (§7) — → 88M tokens/s → 88M / 160K ≈ **~550 prefill slices ≈ 4,400 chips**. Add ~30% for regional peaks not coinciding, failover headroom, and canaries: **~13.5K chips ≈ 53 × 256-chip pods**, spread over 3+ regions. That's under 1.5 Ironwood superpods' worth of silicon — a useful sanity check that the ask is "a pods-scale fleet," not a data-center-scale one.

**SLO budget** — where the 800 ms and 40 ms actually go, per component, with an owner (the table I'd put on the whiteboard before any box diagram):

| Component | p50 | p99 | Budget share of p99 | Owner / control surface |
|---|---|---|---|---|
| Client → anycast LB → regional gateway | 25 ms | 60 ms | 7.5% | Edge/network team, POP placement |
| Router decision + admission | 2 ms | 5 ms | <1% | Gateway scheduler |
| Prefill queue wait | 10 ms | 150 ms | 19% | Prefill autoscaler (queue-wait p95 signal) |
| Prefill compute (8K / cached-suffix) | 44 ms | — | — | Prefill pool, MFU, chunk size |
| Prefill compute (128K tail, 60% cached) | — | ~350–700 ms | up to 70% | Prefix cache hit rate + pool size |
| KV transfer, ICI path (1 GB) | 3 ms | 8 ms | 1% | Placement (same ICI domain) |
| KV transfer, DCN fallback (16 GB) | — | 60–120 ms | tail only | Router; alert if >5% of requests |
| First-token sample + egress | 8 ms | 20 ms | 2.5% | Engine/streaming path |
| **TTFT total** | **~90 ms** | **≤ 800 ms** | | must hold *with* the 128K tail |
| Decode step (TPOT) | 16 ms | ≤ 40 ms | | occupancy vs. measured TPOT curve |

Every number above has ±2× error bars; the *method* — bytes/token, roofline per slice, tokens/s → slices → chips — is the deliverable. In week one I'd replace all of it with measured MFU and achieved bandwidth.

---

## 3. Architecture

Fleet topology with the §2 numbers attached — every arrow carries its rate, every pool its capacity:

```
                        Client (20K req/s global peak, 3:1 diurnal)
                                        │
                              Global anycast LB
                     ┌──────────────────┼──────────────────┐
                REGION A            REGION B            REGION C     (each sized so
              ~6.7K req/s         ~6.7K req/s         ~6.7K req/s     N-1 absorb peak)
                     │
      GKE Inference Gateway + llm-d-style router
      (prefix-overlap × queue-depth × SLO-class scoring; ~30M prefill tok/s
       regional demand post-cache; admission control at TTFT-estimate > budget)
                     │
        ┌────────────┴──────────────────────────────┐
        ▼                                           ▼
  PREFILL POOL                                DECODE POOL
  ~183 × 8-chip slices/region                 ~250 × 8-chip slices/region
  (550 global ≈ 4,400 chips)                  (750 global ≈ 6,000 chips)
  160K tok/s/slice @ ~40% MFU target          8K out-tok/s/slice @ batch 128
  chunked 8–16K, prefix cache 60% hit         continuous batching, 16 ms TPOT p50
        │                                           ▲
        │   KV transfer (per request):              │
        ├── ICI, same pod/superblock: 1 GB (8K ctx) in ~3 ms,   ──┐
        │   16 GB (128K ctx) in ~40 ms — layer-wise, overlapped   ├──► 90%+ of
        └── DCN, cross-pod fallback: 16 GB in 60–120 ms ──────────┘    transfers
            (monitored; alert if >5% of requests take this path)       must be ICI

  KV tiers: HBM (1 TB/slice pool) → host DRAM offload (multi-turn resume)
            → regional prefix store        + warm pool: weight-loaded idle slices
  Weights: 400 GB/revision, region-local mirror → ICI peer fan-out (§5)
```

- **Gateway/router.** GKE Inference Gateway (GA) with an llm-d-style scheduler: routes on prefix-cache overlap, queue depth, and SLO class — not round-robin. Cache-aware routing is a multiplier on everything downstream; without it, prefix caching delivers a fraction of its value ([why](/optimization/14-llm-serving-optimization/#5-prefix-caching-and-cache-aware-routing)).
- **Engine.** The current TPU stack is **tpu-inference** (the unified vLLM TPU backend, Oct 2025; chunked prefill, prefix caching, KV host-offload, Eagle3/ngram speculation). The published ~3.6× E2E gain over the prior vLLM-TPU is a v6e-1 / Llama-3.1-8B number — an irrelevant scale for a 400B MoE, so I cite it only as evidence the stack is current and moving, not as a planning constant. JetStream is archived (Feb 2026) — proposing it dates you.
- **Prefill/decode disaggregation.** Separate pools; deep dive in §4. Google's AI Hypercomputer inference results report **~7× TTFT and ~3× TPOT improvement** for Pathways-based disaggregated serving vs. interleaved (Llama2-70B on Trillium); the DistServe/Mooncake lineage established the pattern on GPUs.
- **Batching.** Continuous batching on decode; chunked prefill *within the prefill pool* to bound head-of-line blocking for the 128K tail. Note the tension I'll return to: chunked prefill's original purpose — interleaving with decode — is mooted by disaggregation, so on a dedicated prefill pool chunks should be large (maximize MFU) and exist only for scheduling granularity and memory bounds.
- **KV management.** Paged KV via Ragged Paged Attention v3; prefix caching keyed on token-block hashes; host-DRAM offload for turns likely to resume (multi-turn chat gaps). Long-context KV strategy per [KV cache & long context](/optimization/8-kv-cache-long-context/).
- **Speculative decoding.** Eagle3 or ngram (both in tpu-inference) as a *conditional* lever: at low-to-moderate load it cuts TPOT 1.5–2.5×; at saturation the extra verify compute and wasted rejected tokens *reduce* goodput, and acceptance rate drops on high-temperature/creative traffic. It ships behind a load-aware switch, off above a batch-occupancy threshold — the crossover math is in §5 ([mechanics](/optimization/9-speculative-decoding/)).
- **Quantization path.** FP8 weights + FP8 KV on Ironwood, int8 AQT fallback; per-slice eval gates before any quantized revision ships. XLA-level tuning per [TPU/XLA optimization](/optimization/18-tpu-xla-optimization/).

---

## 4. Deep Dive 1 — Disaggregation: Pool Sizing and the KV Transfer Bill

**Why disaggregate here specifically.** The two SLOs are adversarial on shared chips: one 128K prefill (~700 ms of solid compute) stalls every colocated decode stream — that's 700 ms of TPOT jitter against a 40 ms p99. Chunked prefill alone softens this but caps prefill MFU. With a 128K p99 input tail, separation is the honest answer — and the published Pathways disaggregation results (~7× TTFT, ~3× TPOT vs. interleaved for Llama2-70B on Trillium) say exactly this: the wins land on the two tail latencies these SLOs are written against.

**Request lifecycle, disaggregated vs. colocated** — the timeline that makes the case. Top two lanes: the design working as intended. Bottom lane: the failure mode disaggregation removes.

```
ms:      0        100       200       300       400  ...  700       800
         │─────────│─────────│─────────│─────────│─  ...  ─│─────────│
                                                     p99 TTFT BUDGET ▲ 800 ms
LANE 1 — 8K prompt, disaggregated (the p50 path)
 gateway ▌6
 queue    ▌▌30 (prefill queue-wait, p50)
 prefill    ▐████▌ 44 ms (8K × 80 GFLOP/tok ÷ 14.8 PFLOPS)
 KV xfer        ▌3 (1 GB over ICI, overlapped with last chunk → ~free)
 decode          ▌▌▌▌▌▌▌... 300 tokens × 16 ms TPOT ≈ 4.8 s stream
 TTFT ≈ 90 ms ✓ (710 ms under budget — that slack is the 128K tail's budget)

LANE 2 — 128K prompt, disaggregated, 60% prefix hit
 queue    ▌▌▌90
 prefill     ▐██████████████▌ ~280 ms (51K uncached tok on a large-chunk slice)
 KV xfer                    ▐█▌ 16 GB: ~40 ms ICI, mostly hidden by layer-wise
 TTFT ≈ 430 ms ✓            streaming (Mooncake-style)   [DCN fallback: +60–120 → tight]

LANE 3 — same 128K prompt, COLOCATED (what we refused to build)
 prefill  ▐██████████████████████████████████████████████████▌ 700 ms monolithic
 ...and every decode stream on that slice:
 decode-neighbors  ▌15 ▌15 ▌───────── 700 ms STALL ─────────▌15 ▌15
                            ^ TPOT p99 = 700 ms vs. a 40 ms SLO — 17× breach,
                              inflicted on ~128 innocent in-flight requests
```

**Pool sizing is a ratio you retune continuously.** From §2: prefill demand ≈ 550 slices, decode ≈ 750 slices → roughly **1:1.4 prefill:decode** at this workload — prefill is a far bigger share than the p50 input length suggests, because prefill bills by the mean. But the ratio is a function of (input length × cache hit rate) / output length — a product launch that doubles average prompt length flips it. So the autoscaler scales the pools *independently* on their own signals: prefill pool on queue-wait p95 (drives TTFT), decode pool on batch occupancy / KV headroom (drives TPOT).

**The KV transfer bill.** After prefill, the full KV must move to a decode slice:
- 8K-context request: 8K × 128 KB = **1 GB**. 128K request: **16 GB**.
- Over ICI (~1.2 TB/s bidirectional per chip, striped across the slice): ~1 GB moves in single-digit ms — negligible against an 800 ms TTFT budget. Two honesty caveats: that link is **not a dedicated pipe** — the decode slice's ICI carries MoE all-to-all dispatch every step, so KV ingress contends with it and the single-digit-ms figure holds only when per-layer streaming keeps the residual transfer inside the gaps between all-to-all phases (that hiding condition is a thing to *measure*, not assume). And cross-job prefill→decode KV routing at this granularity is a Pathways-managed capability, not something the slices improvise. This is why prefill and decode slices for a given request should live in the same ICI domain (same pod/superblock) whenever possible.
- Over DCN between pods: at an effective few-hundred GB/s per slice, 16 GB is tens to ~100+ ms — tolerable for the p99 tail, poisonous if it's the common path.
- **Placement rule:** the router picks a (prefill, decode) pair inside one ICI domain; DCN transfer is the fallback, and its rate is a monitored metric with an alert threshold. Transfer overlaps with the last prefill chunks (Mooncake-style layer-wise streaming), so it hides in the shadow of compute rather than adding to TTFT.
- **Interviewer trap to name unprompted:** disaggregation resurrects the chunked-prefill question. On the prefill pool there's no decode to protect, so run large chunks (~8–16K tokens) for MFU, chunked only to bound scheduler latency and enable early KV streaming. Anyone who copies GPU-colocated chunked-prefill configs onto a disaggregated TPU pool is paying attention-recompute overhead for a benefit they no longer receive.

**Failure isolation bonus:** a poison-pill prompt that OOMs or wedges a prefill slice takes out prefill capacity, not 128 in-flight decode streams.

---

## 5. Deep Dive 2 — SLO-Aware Scheduling, Autoscaling, Multi-Region

**Goodput, not utilization.** The scaling metric is the fraction of chip-time producing tokens *within SLO* — Google's own framing decomposes goodput into scheduling × runtime × program goodput. A 95%-utilized decode slice at 60 ms TPOT is a *worse* fleet member than an 80%-utilized one at 25 ms. Utilization-based autoscaling actively fights the SLO: it packs batches until TPOT breaches, then scales — after the damage. The concrete signal set:

| Signal | Threshold (from §2 curves) | Action | Why not utilization |
|---|---|---|---|
| Prefill queue-wait p95 | > 150 ms (19% of TTFT budget) | Scale prefill pool +; pull from warm pool | Queue-wait *leads* TTFT breach by minutes; utilization confirms it after |
| Decode batch occupancy | > B≈250 on the measured occupancy→p99-TPOT curve (knee −10%) | Scale decode +; stop admitting to slice | 95% busy at B=300 is a breach in progress; busy-ness is the *symptom* |
| KV headroom per slice | < 15% of the ~1 TB page pool | Shed batch class; host-offload idle turns; scale | OOM-driven preemption wastes finished prefills — worse than queueing |
| DCN KV-transfer share | > 5% of requests | Rebalance (prefill, decode) pairing within ICI domains | Pure placement pathology; chips look *idle*, not hot |
| Prefix-cache hit rate | drift > 5 pts off 60% baseline | Re-examine routing keys/eviction; resize prefill (each 10 pts ≈ 120 slices) | A cache regression masquerades as organic prefill load |
| Speculation acceptance α | < ~0.45 per traffic slice, or occupancy > knee | Disable speculation on that slice (see below) | Utilization *rises* while goodput falls — the exact inversion |
| Warm-pool depth | < historical 10-min burst delta | Pre-warm more slices (2–5 min cold start, §below) | Utilization says warm spares are "waste"; the SLO says they're insurance |

**Speculative decoding under load — the actual crossover math.** With draft length k = 4 and per-token acceptance α, each verify step spends **k + 1 = 5 token-slots** of compute/KV-write per request but yields only **αk + 1** real tokens. Wasted work per step = **batch × (1 − α) × k** draft tokens. The regime decides the sign:

- **Latency-limited (occupancy below the knee):** token-slots are cheap — the step was going to stream 400 GB of weights anyway. Effective TPOT = **step_time_spec / (αk + 1)**, where step_time_spec is *longer* than the plain step: the verify step writes KV for all k + 1 positions and adds verify compute. At α = 0.8, αk + 1 = 4.2 and step_time_spec ≈ 17–18 ms vs. 16 plain → effective TPOT ≈ 4 ms. Speculation still wins big — just not by the naive 15/4.2 arithmetic.
- **Slot-limited (at saturation, queue non-empty):** every wasted slot displaces a *guaranteed* token for a queued request. Efficiency per slot = (αk + 1)/(k + 1): α = 0.8 → 0.84 (16% of the slice wasted); α = 0.5 → 0.60 (**40% wasted**); α = 0.3 (high-temperature creative traffic) → 0.44 — speculation is now *halving* saturated-fleet goodput while the utilization graph shows the chips maximally busy.
- **Bandwidth cost of drafts:** rejected tokens aren't only wasted compute — their KV was *written* to HBM and gets *re-read* on subsequent steps until reclaimed, so speculation inflates the per-step KV byte term. That shifts the bandwidth knee left: the spec-on occupancy→TPOT curve crosses 40 ms at a lower B than the spec-off curve.
- **Crossover:** roughly where demand starts queueing — in the neighborhood of the occupancy knee from the TPOT curve, though the measured spec-on knee (draft KV traffic included) sits somewhat below it, so the switch threshold comes from measurement, not the plain-decode curve. Below it, speculation converts idle bandwidth into latency; above it, it converts sold tokens into rejected drafts. There's also a latency edge: verify compute (80 GFLOP × B × k) adds ~3 ms at B = 128 but ~11 ms at B = 512, pushing the spec-on TPOT through the 40 ms ceiling *before* the spec-off curve gets there. Hence the design: speculation behind a per-slice switch keyed on (occupancy > measured spec-on knee) ∨ (α < 0.45), with α monitored per traffic slice (§6) — never a global on/off.

**Scale on warm state.** A new decode slice is useless until 400 GB of weights are resident. From regional SSD-backed cache at ~10 GB/s effective (an assumed number — at 3 GB/s the weight pull alone is >2 min), that's ~40 s just for weights, plus server start and compilation-cache warm — call it **2–5 minutes cold-start** (also an assumption to be measured; if it's 10 min, the warm pool roughly doubles). So: (a) predictive scaling on the diurnal curve, provisioning ahead of the ramp; (b) a warm pool of weight-loaded, traffic-less slices sized to the historical burst delta; (c) XLA compilation caches pre-populated — a recompile storm on a new revision is its own outage class.

**Admission control and priority classes.** Three classes: interactive (full SLO), standard, and batch/offline (no TTFT SLO, scavenger). When goodput headroom shrinks, shed in reverse order — batch API work drains *first*, which is precisely why it lives on this fleet: it's the compressible load that turns burst absorption into revenue instead of idle headroom. Interactive requests that would breach TTFT anyway (queue estimate > budget) are rejected fast at the gateway — a quick 429 with retry-after beats a 3 s TTFT for everyone behind it ([admission control](/optimization/14-llm-serving-optimization/#6-admission-control)).

**Multi-region.** Three-plus regions, anycast front door, per-region fleets sized so that N-1 regions absorb peak (that's part of the 30% headroom in §2). Two problems people skip:

1. **Weight distribution is a huge-file fan-out problem.** 400 GB × thousands of slices × weekly revisions. Naive pulls from a single blob store melt it. Design: region-local mirrored store, tree/peer-to-peer fan-out within a pod (chips pull shards from ICI neighbors, not from the network), content-addressed shards so a revision that touches 10% of weights ships 10% of bytes. Rollout is region-by-region canary with per-revision perf CI (below), and the previous revision stays resident on disk for instant rollback.
2. **Failover is a KV-loss event.** Cross-region failover drops prefix caches and any in-flight KV — the surviving regions see a cache-hit-rate crater and thus a *prefill demand spike* exactly when they've absorbed extra traffic. Walk the arithmetic, because it's where naive N-1 sizing fails:

   **Region-loss walkthrough (lose 1 of 3 at peak).** Steady state per region: ~6.7K req/s, ~183 prefill slices, ~250 decode slices, 60% prefix hit → ~30M prefill tok/s per region actually computed.
   - *Traffic:* each survivor now takes ~10K req/s — **×1.5**. Decode demand scales linearly: 250 → ~375 slices needed; the 30% headroom (250 → ~325) plus shedding the batch class's decode share and riding occupancy from B = 128 toward the ~250 knee (the KV pool has the slots — §2) closes it. Decode is the easy half.
   - *Cache crater, steady state:* a third of each survivor's traffic is diverted users whose prefixes exist in *no* local cache — 0% hit. Blended hit falls 60% → ~40%, so **miss rate goes 40% → 60% (×1.5)** on ×1.5 traffic → prefill demand **×2.25**: 183 slices of load becomes ~412 against ~238 provisioned with headroom.
   - *Transient is worse than ×2.25:* the ×2.25 is the *post-failover steady state*. During re-warm, the flood of cold prefills evicts warm entries, so blended hit dips **below** 40% — call it ~30% → miss ×1.75 on ×1.5 traffic → **~×2.6**, i.e. ~410 slices of demand at the trough. (The extreme case the model must cover: a surface at 80% hit cratering to 0% is miss 20% → 100% — **×5** from cache loss alone, before any traffic shift.)
   - *Closing the gap — with numbers, not verbs.* Transient gap ≈ 410 − 205 = **~205 slices (~1,640 chips)**. Sized mitigations: batch-class prefill shed frees only **~15 slices** at peak — batch is a *trough-filler*, so there's little of it to shed exactly when a peak-hour failover needs it; the regional warm pool contributes **~30 slices**; admission-clamping *uncached* ≥128K prefills (about 2% of requests but ~23% of raw prefill tokens, and almost all uncached post-failover) sheds **~80 slices** of demand under the stratified SLO (§8). Total ≈ 125 slices — leaving an honest **residual of ~80 slices**, which manifests as prefill queueing: degraded TTFT for the standard class for the duration of cache re-warm. That duration is bounded: at ~30M+ tok/s of cold traffic per survivor, the prefix caches repopulate in single-digit minutes, and demand decays from ×2.6 toward ×2.25 toward ×1.5 as hit rate recovers.
   - *The sizing rule this implies:* **prefill pool ≥ transient post-failover demand minus honestly-sized sheddable load, for the re-warm window** — and if the business requires the full TTFT SLO to hold through a peak-hour region loss, the answer is ~1.5× prefill headroom instead of 1.3×, stated as a priced decision, not a flat margin. This is the kind of arithmetic that only exists if you did §2 out loud.

Reliability culture reference: the Gemini training report (arXiv 2312.11805) describes *training* goodput going 85% → 97% via redundant in-memory state — a training figure, cited here only as an instinct-analogy, not a serving benchmark: measure goodput, keep warm state redundant, make recovery fast rather than failures rare.

---

## 5.5 Length-Stratified Subfleets — Why the Decode Pool Should Not Be One Pool

The mixed-decode design above has a structural flaw the admission clamp only papers over: **long-context requests are contagious**. Decode step time on a slice is shared — step ≈ (weights + ΣKV over all residents) / BW — so one admitted 1M-context request (~128 GB KV) adds ~128 GB / 36.9 TB/s ≈ **+3.5 ms to every step, ~+22% TPOT for all ~127 cohabitants**. Three such residents and the slice breaches p99 TPOT for everyone on it. KV streaming is a shared-bandwidth tax, not a private cost, so the tail doesn't just pay more — it makes the median pay.

The structural fix is to stratify the decode fleet by context tier:

| Tier | Context | Config | Operating point |
| --- | --- | --- | --- |
| Short | ≤16K | knee-batch slices (B≈200–250), minimal KV headroom | throughput-optimal; the bulk of the fleet |
| Long | 16K–256K | low batch (B≈16–32), KV pool dominant, CP-sharded prefill feeding it | latency-honest TPOT at high per-request KV |
| Ultra | >256K | dedicated slices, B≈2–8, product-tier pricing/SLO | capacity product, not a latency product |

What this buys, in order of value: (1) **isolation** — the contagion math above becomes impossible by construction; (2) **predictability** — each tier has a homogeneous KV-per-request, so its batch/TPOT curve is honest instead of a blended average (the exact averaging that makes mixed-pool capacity math treacherous); (3) **per-tier operating points** — the short tier rides the bandwidth knee while the long tier trades batch for residency, instead of one compromise batch for both; (4) **clean signals** — a long-context burst reads as "long tier needs slices," not fleet-wide TPOT drift.

What it costs, stated honestly: per-tier failover headroom (N−1 reserved in each pool beats pooled slack by less than it looks), a tier-migration path — input length is known at admission but output growth and multi-turn accumulation are not, so sessions cross tiers mid-conversation and need either inter-pool KV transfer (a Pathways-managed move, priced like the prefill→decode transfer in §4) or a re-prefill against the prefix cache — and split prefix-cache locality. The break-even is tier size: stratification pays when every tier is big enough to batch efficiently alone. At this design's ~13K-chip scale each tier is thousands of chips — stratify. At a 100-chip deployment, don't; run mixed with admission clamps and accept the tail coupling.

## 6. Monitoring — What I'd Actually Page On

- **SLO surfaces:** TTFT and TPOT p50/p95/p99 *sliced by input-length bucket, region, and priority class* — a global p99 hides a single region or a single length bucket burning. Plus **goodput under SLO** as the headline fleet metric, decomposed scheduling/runtime/program so a regression is attributable.
- **Leading indicators:** prefill queue-wait p95 (leads TTFT breach), decode batch occupancy and KV headroom (leads TPOT breach and OOM-driven preemption), DCN KV-transfer rate (leads TTFT tail), prefix-cache hit rate per region (a slow drift here silently re-inflates the prefill fleet bill). Thresholds and actions per the autoscaling table in §5 — the pager and the autoscaler read the same signals; the pager fires when the autoscaler's action didn't work.
- **Speculation health:** acceptance-rate distribution per traffic slice, against the α ≈ 0.45 switch threshold from §5. Acceptance drift is both a perf signal and a *model-change* signal — a new revision or a shifted traffic mix shows up here first.
- **Per-revision perf CI:** every model or server revision runs a fixed replay set (representative length mix, open-loop load) and must hold TTFT/TPOT/goodput within tolerance before regional rollout — benchmarking methodology per [LLM serving optimization §11](/optimization/14-llm-serving-optimization/#11-benchmarking-methodology). Quality evals gate quantization and speculation changes the same way.
- **SDC awareness:** at ~13K chips, silent data corruption is a when, not an if. Lightweight online checks (logit-distribution canaries, periodic golden-prompt checksums per slice) plus a quarantine path for suspect chips. This is a fleet-scale habit, not paranoia.

---

## 7. Cost per Token — Derivation and the Three Levers

Cost/token = (chips × $/chip-hr) / (tokens/hr within SLO). Illustrative: at ~$2/chip-hr, a decode chip producing ~1,000 tok/s → ~**$0.56 per million output tokens** decode-side; the full ~13.5K-chip fleet at ~$27K/hr over 6M output tok/s ≈ 21.6B tok/hr → ~**$1.18/M output tokens all-in** (prefill, headroom, and canaries carried by output tokens). The absolute number is assumption-laden; the *ranking of levers* is robust:

| # | Lever | Mechanism | Δ cost/M output tokens (illustrative) | Why this rank |
|---|---|---|---|---|
| 1 | Batch-API infill | Sell trough + headroom chip-hours (~40–50% of fleet-hours) as scavenger tokens | ~−$0.35 to −$0.45 (can halve effective cost) | Zero silicon, zero SLO risk — it *is* the headroom, monetized |
| 2 | Prefix-cache hit rate | Each +10 pts ≈ 120 prefill slices ≈ 950 chips ≈ $1,900/hr freed | ~−$0.09 per 10 pts (60% → 80% ≈ −$0.18) | Routing + cache-keying work, not hardware; compounds with lever 1 |
| 3 | Decode occupancy vs. TPOT curve | Ride B = 128 → ~250 along the measured bandwidth knee (§2 curve): ~1.5× tok/s/chip; the KV pool has the slots (~600), the knee is the limit | ~−$0.12 to −$0.18 decode-side | Needs the occupancy→p99 curve measured per model/chip; free after that |
| 4 | Speculation (below the knee only) | αk + 1 effective tokens/step off-peak | ~−$0.03 to −$0.08, load-dependent | Conditional (§5); negative at saturation — hence rank 4, not 1 |
| 5 | FP8 KV / quantization deltas | More resident requests per slice, faster steps | small after 1–3 | Already assumed in the baseline; marginal from here |

In prose: prefill is ~40% of the pre-headroom fleet at 60% hit rate — the mean-driven input tail, not the median, put it there — and every 10 points of hit rate is ~120 slices of capacity, the cheapest capacity you'll ever buy. The interactive fleet is sized for peak; scavenger-class batch work converts trough to sold tokens at near-zero marginal cost. And the gap between "safe default batch" and "measured frontier" on the §2 curve is ~1.5× throughput. FP8/quantization and speculation are real but smaller multipliers after these three.

---

## 8. Escalating Follow-Ups

**"Would you run the batch API on the same fleet?"** Yes — designed in from the start (§5): scavenger priority class, preempted first, checkpointable at token granularity (KV snapshot to host DRAM so preemption wastes nothing irrecoverable). The alternative — a dedicated batch fleet — buys isolation at the cost of the single biggest cost lever (lever 1 in §7). I'd only split fleets if batch demand grows to where preemption churn measurably degrades interactive goodput.

**"Context jumps to 1M — what breaks?"** Everything KV-shaped. 1M × 128 KB = **128 GB of KV per request** — ~13% of a slice's KV pool, or ~85 blended-mix slots, on the §2 memory map; eight such requests overflow the entire 1 TB pool. Responses: much more aggressive GQA/MLA-style KV compression and/or sliding-window layers (a model co-design conversation, not purely serving); KV offload to host DRAM with layer-wise streaming for the decode working set; prefill for 1M is minutes of compute → it becomes a *batch-like* product tier with its own SLO, chunked and checkpointed; DCN KV transfer at 128 GB stops being tolerable, so 1M requests pin prefill+decode into one ICI domain. And I'd push back on product: what p99 TTFT is a 1M-token request actually entitled to? 800 ms is not a physical answer here.

**"You get a Trillium (v6e) fleet instead — 32 GB @ 1.6 TB/s. What changes?"** The serving-unit math reruns: weights alone need ≥ 13 chips → a 16-chip slice has 512 GB (400 weights + only ~100 GB KV — KV-starved) so realistically 32-chip slices; aggregate bandwidth 32 × 1.6 = 51 TB/s, similar step time to the 8-chip Ironwood slice but with 4× the chips and 4× the ICI collective surface per step. Consequences: per-token interconnect overhead rises (more all-to-all hops for MoE dispatch), KV capacity per slice is the binding constraint → prefix caching and host offload go from "cost lever" to "load-bearing," long-context requests may simply not fit → tighter admission by length, and FP8 isn't native → int8 AQT path. Cost per token likely rises ~2× even at equal $/FLOP. Same method, different constants — which is why the method was the deliverable.

**"Your p99 TTFT is breaching but p50 is fine — first three graphs?"** Prefill queue-wait p95 by region (queueing vs. compute), TTFT sliced by input-length bucket (is it the 128K tail or 4K requests stuck behind it), DCN KV-transfer rate (did ICI-domain placement degrade). Then cache hit rate — a hit-rate drop masquerades as a prefill capacity problem (exactly the §5 failover signature, in slow motion).

---

## 9. Staff+ Signals

- **Arithmetic first, boxes second.** Bytes/token → *one* blended-KV constant → roofline per slice → tokens/s → chips, stated before any architecture, with error bars owned out loud.
- **Mean, not median.** Prefill demand and KV occupancy are driven by E[len] (~11K here), not the p50 (6K); sizing on the median under-builds ~2× and is a classic silent error.
- **Goodput vs. utilization distinction**, and autoscaling signals derived from the SLO curve, not from chip busy-ness.
- **Naming the chunked-prefill × disaggregation conflict unprompted** — the config that's right colocated is wrong disaggregated.
- **Failover reasoned as a KV/cache event**, not just a traffic event; prefill pool sized for the post-failover hit-rate regime — the ×2.25 steady-state / ~×2.6 transient walkthrough in §5, with each mitigation sized and the residual owned — not steady state plus a flat margin.
- **Speculative decoding as conditional**, with the slot-economics crossover derived (wasted slots = B(1−α)k; per-slot efficiency (αk+1)/(k+1)), behind a switch, monitored by acceptance rate.
- **Current-stack awareness:** tpu-inference not JetStream (archived), GKE Inference Gateway GA, the Pathways disaggregation results (~7× TTFT / ~3× TPOT vs. interleaved) — and skepticism: every vendor number gets re-measured on our workload before it enters the capacity plan.
- **Honesty about measurement:** the week-one deliverable is replacing every constant in §2 with profiled values; the design survives them changing, the plan doesn't pretend to know them.

## What Falls Short

- **Feature-listing** vLLM/tpu-inference flags — continuous batching! paged attention! prefix caching! — without sizing anything. Every candidate says the same nouns; the arithmetic is the differentiation.
- **Utilization-based autoscaling** proposed as if it were neutral. It's not neutral; it's anti-SLO.
- **Ignoring multi-region weight distribution and cold start** — "just autoscale" with a 400 GB weight payload and minutes-long cold starts is a design that fails its first traffic burst.
- **Speculative decoding with no quality/latency/load caveat** — presenting it as a free 2× is the tell that the candidate has never watched acceptance rate collapse under a saturated, high-temperature workload.
- **Treating the 800 ms / 40 ms SLOs as unquestionable** for every length bucket. Staff behavior includes negotiating a stratified SLO for the 128K/1M tail rather than silently over-building the fleet for it.

---

## Confidence Ledger

Where each class of number in this answer comes from — the same honesty I'd volunteer in the room:

**Verified (public, checkable):**
- Ironwood: 192 GB HBM (2 × 96 GB chiplets, separate memory spaces), 7.37 TB/s, ~1.2 TB/s ICI, FP8 at 4,614 TFLOPS/chip, 256-chip and 9,216-chip pod scales.
- Pathways disaggregated serving vs. interleaved: ~7× TTFT, ~3× TPOT (Llama2-70B on Trillium; Google AI Hypercomputer inference results).
- tpu-inference is the current vLLM TPU backend (Oct 2025); JetStream archived (Feb 2026). The 3.6× E2E figure is real but v6e-1/8B — stack-currency evidence only.
- Gemini training goodput 85% → 97% (arXiv 2312.11805) — a *training* figure, used as an instinct-analogy only.

**Derived (arithmetic from stated assumptions — right method, ±2× constants):**
- 128 KB KV/token; blended ~1.4–1.5 GB KV/request from the length mix; E[len] ≈ 11K.
- 16 ms decode step at B = 128; the batch curve and the B ≈ 250 bandwidth knee; ~600-slot KV capacity; 750 decode + 550 prefill slices → ~13.5K chips.
- The failover ×1.5 / ×2.25 / ~×2.6 cascade and the 205-slice transient gap with its ~80-slice residual.
- Speculation slot economics: (αk+1)/(k+1), effective TPOT = step_time_spec/(αk+1).

**Priors (assumed operating points — first things replaced by measurement in week one):**
- 60% achieved HBM bandwidth; 40% prefill MFU *target* (20–30% is common for MoE prefill); 60% prefix-cache hit rate; +60% long-context p99 TPOT inflation; α ≈ 0.45 speculation switch threshold; ~10 GB/s SSD restore; 2–5 min cold start; $2/chip-hr; the traffic shape and length mix themselves.

If a number above moves, the design survives; the constants in §2 do not, and were never meant to.
