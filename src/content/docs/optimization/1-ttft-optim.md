---
title: "TTFT Optimization Program: 70B Chat"
description: "A staff-level worked interview answer: driving p99 time-to-first-token from 1.8s to under 500ms on a 70B chat model — measurement, budget decomposition, sequenced roadmap, and pushback."
---

"We serve a chat product on a 70B-class model. TTFT (time-to-first-token) at p99 is currently 1.8 seconds. Product wants it under 500ms p99 within 2 quarters. You have 6 engineers. Walk me through your optimization program — what would you do, in what order, what would you measure, and what would you push back on?"

---

# TTFT Optimization Program: 70B Chat, 1.8s → <500ms p99 in 2 Quarters

**Owner:** Tech Lead, Inference. **Team:** 6 engineers. **Horizon:** Q1+Q2 (~48 engineer-months).

---

## 1. State of the Art (2026) — Research Pass

Before any planning, what do frontier teams actually do for TTFT in 2026? The landscape has moved hard in the last 18 months.

**Disaggregated prefill/decode is mainstream.** Mooncake (Kimi) showed the architecture in production, separating prefill and decode clusters with a KVCache-centric Conductor scheduler that balances cache reuse, prefill load, and TTFT SLO simultaneously. NVIDIA Dynamo 1.0 (GA March 2026) productized this above vLLM/SGLang/TensorRT-LLM, with a KV-aware router, SLA-based Planner that autoscales prefill and decode pools independently, and NIXL for fast KV transfer over EFA/InfiniBand. SemiAnalysis InferenceX shows up to 7× throughput on GB200 NVL72 with wide-EP. DeepSeek has run disaggregated since late 2024. The architectural decision is no longer "should we?" but "when does the operational cost pay off?"

**Chunked prefill is the default.** Sarathi-Serve's stall-free scheduling is now the default in vLLM V1 and SGLang. Token budgets: A100 saturates at ~2K, H100 at ~8K, B200 higher. This is plumbing, not differentiation.

**Prefix caching is productized at the API layer.** Anthropic exposes explicit `cache_control` breakpoints with 5-min or 1-hour TTL; cache reads cost 0.1× base input, writes cost 1.25× (5m) or 2.0× (1h). Anthropic reports up to 85% latency reduction on long prompts — a 100K-token document goes from 11.5s to 2.4s TTFT. OpenAI has automatic implicit caching. Gemini 2.5 has implicit caching. NVIDIA TensorRT-LLM reports 5× TTFT from KV early reuse. Cross-engine sharing via LMCache and tiered storage (HBM → DRAM → NVMe → object store) extends hit rates further.

**FlashAttention-3 + FP8 on Hopper is table stakes.** FA3 hits 740 TFLOPS FP16 (75% of H100 peak) and ~1.2 PFLOPS FP8, with 2.6× lower numerical error than naive FP8 attention. FlashInfer kernels are competitive and integrate cleanly with paged KV.

**NVFP4/MXFP4 on Blackwell.** NVFP4 uses 16-element blocks with E4M3 scale + FP32 second-level scaling — finer-grained than MXFP4's 32-element E8M0 blocks. MR-GPTQ achieves ~2.0–2.2× end-to-end speedup vs BF16 on 70B Llama, with academic benchmark deltas inside ±0.01. NVFP4 helps prefill (compute-bound) substantially.

**Speculative decoding has consolidated on EAGLE-3** — the EAGLE-3 paper reports roughly 3-6.5× decode speedup over vanilla autoregressive decoding, supported in vLLM/SGLang/TRT-LLM. **Decode-only.** Frontier teams treat it as a separate workstream from TTFT.

**Goodput is the metric that matters.** Fraction of GPU-seconds producing tokens within SLO. Frontier teams have moved off raw throughput.

---

## 2. Reframing the Question — and the First Pushback

"Chat product on a 70B-class model, p99 TTFT 1.8s → <500ms in 2 quarters" is the prompt. Before I touch a roadmap, two questions need answers from product, and one assumption needs naming.

**What I need from product before committing:** (1) prompt-length distribution at p50/p95/p99 — is p99 driven by 32K+ prompts or by 4K prompts queueing? (2) System prompt structure — single shared system prompt across users, or per-tenant? (3) Concurrent users per replica at peak, and the current peak-to-trough ratio. (4) Multi-turn? Average turn count? Average gap between turns?

**Why these matter:** the highest-leverage optimization is determined entirely by these answers. Big shared system prompt + multi-turn → prefix caching dominates. Long-tail document-grounded prompts → chunked prefill + disaggregation. High concurrency at a single hot replica → scheduling/admission, not kernels.

**Working assumptions for the rest of this doc**, to be confirmed in Week 1: ~4–8K-token system prompt, mostly shared per app surface; user-turn distribution heavy at 1–4K tokens, ~5% tail at 32–128K (uploads, code, RAG); peak concurrency 20–40 active prefills per H100/H200 replica at TP=2 or TP=4; multi-turn typical session of 3–8 turns with gaps under 5 minutes.

**[STAFF SIGNAL: workload-first]** — "Chat product" is not a workload spec. The program is shaped by these answers, not by my favorite optimization. **[STAFF SIGNAL: measurement-first]** — Week 1 deliverable is the trace and the budget decomposition, not a fix. **[STAFF SIGNAL: pushback-on-metric]** — and before I commit to 500ms p99, I need to question whether p99 is the right target (returned to in §8).

---

## 3. TTFT Budget Decomposition — Where Is the 1.8s Actually Going?

Until we have traces, the budget is a hypothesis. Below is a defensible *prior* for a 70B at TP=4 on H100, FP8 weights/BF16 activations, vLLM-class server at ~70% utilization, ~8K avg prompt, modest prefix caching. **The first deliverable of the program is replacing this prior with measured data per percentile.**

```
p99 TTFT BUDGET DECOMPOSITION (hypothesized, pre-measurement)
====================================================================

p50  (~700ms,  short prompts, warm replica):
[NW 80][Q 40][Tok 10][Prefill 480][KV 30][Samp 5][Egress 55]
 ▓▓▓▓ ▓▓     ▓        ▓▓▓▓▓▓▓▓▓▓▓▓ ▓      ▓     ▓▓▓

p95  (~1.3s,  longer prompts + some queueing):
[NW 80][Q 180][Tok 15][Prefill 880][KV 60][Samp 5][Egress 75]
 ▓▓▓▓ ▓▓▓▓▓▓ ▓        ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓ ▓▓     ▓     ▓▓▓

p99  (~1.8s,  long-prompt tail + queue contention):
[NW 80][Q 450][Tok 15][Prefill 1080][KV 80][Samp 5][Egress 90]
 ▓▓▓▓ ▓▓▓▓▓▓▓▓▓▓▓▓ ▓ ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓ ▓▓     ▓     ▓▓▓▓
       ^^^^                    ^^^^^^^^^^^^^^^
       biggest                 second biggest

Components:
  NW       = client→LB→frontend network ingress + TLS
  Q        = scheduler queue + admission wait
  Tok      = tokenization + request setup
  Prefill  = forward pass over input tokens (compute-bound)
  KV       = paged KV allocation + materialization to HBM
  Samp     = first-token sampling
  Egress   = first-byte → client (LB, region hop, TLS frame)
```

**Read this carefully.** At p99, queue time and prefill compute together account for ~85% of the budget. Network and KV materialization combined are ~10%. Sampling is negligible. **If our prior is roughly right, optimizations that don't move queue or prefill are not optimizations for *this* p99 problem.**

**Two non-obvious points the diagram surfaces:**

1. **p50 looks healthy already.** The product target of 500ms p99 is fundamentally a *tail* problem. Optimizations that improve average prefill throughput (e.g., a new kernel that's 15% faster across the board) move p50 down to 600ms but move p99 from 1800 to perhaps 1500. Tail-specific work — chunked prefill, prefix-cache routing, admission control — moves p99 disproportionately.

2. **Queue time at p99 (450ms) is not a kernel problem.** It is a scheduling/utilization problem. If we are running at 70% steady-state utilization with bursty arrivals, a long prefill from one request blocks everyone behind it. The fix is chunked prefill + decode-priority scheduling + admission control, not faster matmul. **[STAFF SIGNAL: scheduling-and-queue]**

**Week 1–2 trace plan:** request-level spans through frontend → router → scheduler → engine → kernel → response, with timestamps per stage. Tools: PyTorch Profiler for kernel breakdowns, Nsight Systems for the GPU pipeline, custom request-level tracing through the gateway/scheduler. Dashboards must show p50/p95/p99 per component, sliced by prompt length bucket and tenant. If we discover the prior is wrong — e.g., network egress is actually 300ms because the LB has a buggy proxy — the roadmap reshuffles. **[STAFF SIGNAL: measurement-first]**

---

## 4. Optimization Inventory

For each, what it does, expected TTFT impact, engineering cost, dependencies, risks. **[STAFF SIGNAL: prioritization-with-quantification]** This is the inventory; the sequencing is §5.

### 4.1 Prefix Caching + Cache-Aware Routing — First Move

**What:** Cache the KV state of shared prefixes (system prompt, tool schemas, repeated few-shot blocks, previous user turns). On a hit, skip prefill for the cached span; only the suffix needs computing. RadixAttention (SGLang), vLLM automatic prefix caching, or LMCache for cross-engine sharing. (Canonical treatment: [LLM serving optimization](/optimization/14-llm-serving-optimization/).)

**Expected TTFT impact:** highly workload-dependent.
- 4K shared system prompt, fresh prefill on 70B TP=4 H100: ~250–400ms.
- Same prompt, cache hit: KV in HBM, ready in ~5–20ms.
- For a 4K-system-prompt + 1K-user-turn workload with 80% cache hit rate, p50 TTFT drops by ~200–350ms; p99 drops less because long-tail prompts have less cache benefit.

**Engineering:** 2–4 engineer-weeks for vLLM-class single-engine. +4–6 weeks if we need cross-replica sharing (LMCache + KV transfer over RDMA). **Routing is the mandatory companion.** Round-robin destroys hit rate. Consistent hashing on the cache key or cache-overlap-aware routing (à la Mooncake Conductor, Dynamo KV router) is required.

**Risks:** subtle correctness bugs around tokenizer normalization, cache invalidation on model updates, multi-tenancy isolation. Low quality risk — cached KV is bitwise the same KV.

**Why first:** highest impact for typical chat workloads, lowest risk, well-understood. **[STAFF SIGNAL: 2026-frontier awareness]** Anthropic, OpenAI, Google all expose this at the API layer in 2026; the internal architecture follows.

### 4.2 Chunked Prefill — The Long-Prompt Tail

**What:** Sarathi-Serve. Split a long prefill into N chunks, interleave with decode iterations. A 32K prefill no longer blocks every concurrent decode.

**Expected TTFT impact:**
- For the long-prompt request itself: TTFT marginally *worse* (5–15%) due to attention recompute across chunks.
- For everyone else queued behind: massive improvement. Removes the "one user submitted a giant document and now every interactive user waits 4s" failure mode.
- **p99 TTFT for the *fleet* drops significantly** even if the slowest individual request gets slightly slower.

**Engineering:** 4–8 engineer-weeks. Scheduler change, not just a flag. Chunk size tuning per hardware (A100 saturates at ~2K, H100 at ~8K, B200 higher). In vLLM V1 it's default-on but requires tuning `max_num_batched_tokens` for our workload.

**Risks:** correctness regressions in chunked attention masking; throughput regressions if chunk size is too small. Moderate. **[STAFF SIGNAL: interaction awareness]** Chunked prefill changes the scheduler contract; interacts with prefix caching (cached prefix doesn't need chunking) and with disaggregation (chunked prefill on a colocated server vs. monolithic on a dedicated prefill node — different optimal config).

### 4.3 Scheduling, Admission Control, Cache-Aware Routing

**What:** SLO-aware scheduling (prioritize requests near SLO violation), admission control (shed at queue-depth threshold instead of letting tail explode), prefill/decode interleave policy, cache-aware routing.

**Expected TTFT impact:** This often moves p99 more than any kernel work. If 450ms of our p99 budget is queue time, even a 30% reduction is 135ms. Likely 200–300ms p99 win in our prior.

**Engineering:** 4–8 engineer-weeks. Mostly serving-layer code, no kernel work. Coordinates with autoscaling — admission control without autoscaling is just dropping users.

**Risks:** dropped-request rate becomes a new SLO. Need product alignment on shed policy (which requests are sheddable). **[STAFF SIGNAL: scheduling-and-queue]**

### 4.4 Prefill–Decode Disaggregation — Conditional

**What:** Mooncake/DistServe/Splitwise/Dynamo: separate prefill GPU pool and decode GPU pool. Transfer KV via NVLink (same rack) or RDMA/EFA (cross-node). (Canonical treatment: [LLM serving optimization](/optimization/14-llm-serving-optimization/).)

**Expected TTFT impact:** Pays off when prompt-length variance is the dominant p99 driver. Each pool can be sized and scheduled independently — prefill bursts don't degrade decode SLOs, and vice versa. Reported: NVIDIA Dynamo on GB200 NVL72 shows ~7× throughput at iso-latency for reasoning workloads.

**Cost:** KV transfer adds 20–80ms to TTFT depending on fabric (NVLink << RDMA << TCP). Operational complexity is real: two pools, capacity planning per pool, failure isolation, transfer-engine ops. Engineering: 8–16 engineer-weeks for a non-trivial production rollout.

**When to do it:** if after prefix caching + chunked prefill + scheduling we still see prefill bursts harming decode SLOs (decode ITL p99 spiking when prefill demand spikes), or if our prompt-length distribution is extremely bimodal. Otherwise, deferred. **[STAFF SIGNAL: engineering-reality]** Not first; possibly second-half-Q2 if the data justifies.

### 4.5 Quantization — FP8 Activations on Prefill Path

**What:** Prefill is compute-bound on long prompts. FP8 (E4M3) weights *and* activations on H100/H200 ~doubles compute throughput vs BF16. Weight-only INT4 (AWQ/GPTQ) is decode-relevant, not prefill-relevant — for TTFT we want compute-quantization.

**Expected TTFT impact:** ~1.5–1.8× prefill speedup if we are not already FP8 activations. On our 1080ms p99 prefill component, that's ~400–500ms. **Biggest single kernel-level win available.**

**Engineering:** Mostly Q-team / kernel-team work. Serving-team integration: 2–4 engineer-weeks plus eval discipline. Calibration sweep (per-tensor vs per-channel scaling, sensitivity per layer), held-out quality evals across capability slices.

**Risks:** quality regression. Llama 3 paper and subsequent work documents 0.5–1% NE-style regression with care, more without. Per-layer sensitivity analysis is mandatory. NVFP4 on B200 is another step (~2× more) but adds calibration risk; not Q1. **[STAFF SIGNAL: cross-team awareness]** This is owned by Q-team; TL coordinates serving integration and owns the eval gate.

### 4.6 Kernels — FlashAttention-3 / FlashInfer

**What:** If we are not already on FA3, the migration is high-leverage. FA3 is 1.5–2.0× faster than FA2 in FP16, ~75% of H100 peak; FP8 FA3 hits ~1.2 PFLOPS with 2.6× lower numerical error than naive FP8 attention.

**Expected TTFT impact:** ~10–20% prefill speedup on the attention component. End-to-end prefill speedup is smaller (attention is a fraction of prefill compute; MLP dominates at moderate seq lengths). Maybe ~80–150ms on p99 prefill.

**Engineering:** 1–2 engineer-weeks to integrate, much less to flip a flag if already supported in vLLM. Writing *custom* kernels is 8–16+ engineer-weeks for similar end-to-end gains; not justified unless profiling shows a specific bottleneck unaddressed by FA3/FlashInfer. **[STAFF SIGNAL: engineering-reality]** Use published kernels.

### 4.7 Hardware-Aware Routing — Conditional

**What:** If we have heterogeneous fleet (H100, H200, B200), route long prompts to the fastest prefill hardware and short prompts to cheaper hardware. B200's NVFP4 + bandwidth advantage makes it 2–3× faster on long prefills than H100.

**Expected TTFT impact:** Up to 200–400ms on long-prompt p99 *if* we have the hardware and the routing layer.

**Engineering:** Routing layer is straightforward; the operational complexity is fleet management. ~4–8 weeks if hardware exists. If not, this is gated on hardware acquisition (not a 6-engineer problem).

**Deferred to Q2 or beyond** unless hardware mix already exists.

### 4.8 Network and Client-Side Latency

**What:** Inspect LB/gateway overhead, TLS handshake costs, region routing, HTTP/2 connection reuse, response streaming (SSE chunk size, Nagle, TCP_NODELAY).

**Expected TTFT impact:** Highly site-specific. If we have multi-region users hitting a single-region cluster, network ingress + egress can be 200–400ms; if we are single-region with good edge POPs, 30–80ms. Often-neglected — easy to find 100ms here if no one has looked. **[STAFF SIGNAL: network-and-client]**

**Engineering:** 1–3 engineer-weeks to audit and fix. Cheap, high probability of finding something. Should run in parallel with Week 1–2 measurement work.

### 4.9 Speculative Decoding — Not a TTFT Optimization

**What:** EAGLE-3, Medusa, MTP. Speeds up *decode* — the EAGLE-3 paper reports roughly 3–6.5×.

**TTFT impact:** ~zero direct. Indirect: faster decode → engines free up faster → queue time drops marginally. At our utilization, maybe 30–60ms on p99.

**Engineering:** Significant. Train/distill a draft model, integrate into the engine, eval for losslessness.

**Verdict:** Run as a parallel decode-latency workstream, not as a TTFT lever. Counting it as a TTFT optimization is the conflation error the prompt was testing for. **[STAFF SIGNAL: TTFT-specific reasoning]**

---

## 5. The Roadmap — Sequenced With Reasoning

### Q1: Measurement and Tail-Specific Wins

```
WEEK:  1   2   3   4   5   6   7   8   9   10  11  12  13
       │   │   │   │   │   │   │   │   │   │   │   │   │
T1: Tracing/dashboards/budget       │   │   │   │   │   │
       ████                          │   │   │   │   │   │
T2: Network/client audit (parallel) │   │   │   │   │   │
       ██████                        │   │   │   │   │   │
T3: Prefix caching + cache-aware routing │   │   │   │   │
           ████████████████          │   │   │   │   │   │
T4: Chunked prefill rollout         │   │   │   │   │   │
                       ████████████████   │   │   │   │
T5: Scheduling, admission, prio     │   │   │   │   │   │
                               ████████████████   │   │
T6: Eval/regression harness (continuous)
       ████████████████████████████████████████████████

EXIT GATE Q1: p99 TTFT 1800ms → ~900ms (50% reduction)
Confidence: medium-high. Largest single win is prefix caching;
chunked prefill drops the queue-induced tail; scheduling cleans up.
```

**Q1 engineer-weeks:** roughly 6 engineers × 13 weeks = 78. Allocation: T1 (8), T2 (6), T3 (16), T4 (16), T5 (12), T6 (10), slack/oncall/eval (10). Tight, achievable. **[STAFF SIGNAL: engineering-reality]**

**Q1 fallbacks per workstream:**
- T3 (prefix caching): if cache hit rate is <40% after rollout (workload not as cache-friendly as assumed), invest the next 2 weeks in tenant-specific cache breakpoint placement and refactor system prompt structure with product. If hit rate is still poor, this is a workload reality and we re-plan around chunked prefill + disaggregation as the primary tail levers.
- T4 (chunked prefill): if chunked prefill regresses long-prompt TTFT more than 20% individually (chunk overhead high), tune chunk size up and accept some decode contention; we control this with the scheduler's prefill/decode token budget.
- T5 (scheduling): if admission control causes >0.5% drop rate at peak, add capacity (autoscale floor up). Latency-vs-availability tradeoff goes through product.

### Q2: Higher-Investment Work

```
WEEK:  14  15  16  17  18  19  20  21  22  23  24  25  26
       │   │   │   │   │   │   │   │   │   │   │   │   │
T7: FP8 weights+activations integration (with Q-team)
       ████████████████████              │   │   │   │
T8: FA3 / FlashInfer adoption (if not Q1)
       ████████                          │   │   │   │
T9: CONDITIONAL — pick ONE based on Q1 data:
   A. P-D disaggregation (if prefill bursts hurt decode SLO)
   B. Hardware-aware routing (if heterogeneous fleet exists)
   C. Deeper scheduler work + cross-replica KV sharing
                       ████████████████████████████████
T10: Eval and canary (continuous)
       ████████████████████████████████████████████████

EXIT GATE Q2: p99 TTFT 900ms → ~450ms (target hit with margin)
Confidence: medium. FP8 activations give large prefill win;
T9 path determined by Q1 exit data.
```

**Q2 engineer-weeks:** ~78. T7 (16) + T8 (4 if needed) + T9 (32) + T10 (10) + slack/oncall (16). **[STAFF SIGNAL: roadmap-with-fallbacks]**

**Q2 conditional logic — T9 decision tree:**
```
At end of Q1, what is the dominant remaining p99 contributor?
│
├── Prefill compute on long prompts dominates
│   AND prefill load causes decode ITL p99 spikes
│   → T9-A: Prefill-decode disaggregation (8–16 wks)
│
├── We have heterogeneous fleet (H100 + H200 + B200)
│   AND long-prompt routing is feasible
│   → T9-B: Hardware-aware routing (4–8 wks)
│       + remaining time for cross-replica KV sharing
│
└── Queue / scheduling still dominates
    → T9-C: Deeper scheduler work, LMCache cross-engine
            KV sharing, predictive autoscaling
```

**Per-milestone expected delta** (cumulative on p99):
- After T3 (prefix caching + cache-aware routing): –250ms → ~1550ms
- After T4 (chunked prefill): –350ms (tail-specific) → ~1200ms
- After T5 (scheduling): –250ms → ~950ms
- *Q1 exit: ~900ms* (matches gate above; small buffer absorbs measurement uncertainty)
- After T7 (FP8 prefill): –300ms → ~600ms
- After T8 (FA3): –80ms → ~520ms
- After T9-A/B/C: –50 to –150ms → 370–470ms
- *Q2 exit: ~400–500ms* (target ≤500ms hit)

**Each delta is an expected value with significant variance.** The roadmap is robust to any single optimization underperforming by ~30% because we have multiple sources of gain. It is *not* robust to two simultaneous underperformances. The Q2 mid-quarter checkpoint exists to catch that. **[STAFF SIGNAL: prioritization-with-quantification]**

---

## 6. Interaction Matrix

Optimizations don't compose additively. The matrix below captures the non-trivial interactions. Diagonal is single-op effect; off-diagonal is what happens when combined.

```
                  | PrfxC | ChnkP | Sched | DisAg |  FP8  |  FA3  | SpecD |
                  |-------|-------|-------|-------|-------|-------|-------|
PrfxCache         |   +   |   +   |   ++  |   +   |   ~   |   ~   |   ~   |
ChunkedPrefill    |   +   |   +   |   ++  |   −   |   ~   |   ~   |   ~   |
Sched/Admit       |   ++  |   ++  |   +   |   +   |   ~   |   ~   |   ~   |
Disaggregation    |   +   |   −   |   +   |   +   |   ~   |   ~   |   ~   |
FP8 act+weight    |   ~   |   ~   |   ~   |   ~   |   +   |   ++  |   ~   |
FA3 kernels       |   ~   |   ~   |   ~   |   ~   |   ++  |   +   |   ~   |
SpecDecoding      |   ~   |   ~   |   ~   |   ~   |   ~   |   ~   |   +   |

++ strongly synergistic    +  weakly synergistic
~  independent / orthogonal  −  conflict, needs tuning
```

**Key interactions worth spelling out:**

- **Prefix caching × Cache-aware routing × Scheduling (++).** All three together is the win. Prefix caching alone with round-robin routing gives ~20% of the achievable benefit; cache-aware routing without scheduling priority still queues hits behind misses. **You cannot evaluate the prefix-caching investment without committing to the routing and scheduling pieces.**

- **Chunked prefill × Disaggregation (−, needs care).** On a disaggregated prefill pool, chunked prefill provides less benefit — there's no decode work to interleave with on the prefill node, so chunking just adds attention recompute overhead. The right config on a dedicated prefill node is *larger* prefill chunks, possibly monolithic. **If we ship disaggregation in Q2, the chunked-prefill config from Q1 must be revisited.**

- **FP8 activations × FA3 (++).** FA3's FP8 path is the path where the FP8 throughput is realized at the attention layer. Doing one without the other leaves throughput on the table.

- **Disaggregation × Prefix caching (+).** KV transfer between prefill and decode is the right bus on which to put cross-replica KV sharing for prefix cache. Same fabric, same transfer engine (NIXL or equivalent). Plan the integration. **[STAFF SIGNAL: interaction awareness]**

---

## 7. Quality Regression Discipline

Every optimization landing in production must clear the same gate. **The gate is the deliverable, not an afterthought.** **[STAFF SIGNAL: quality-regression discipline]**

**Pre-deployment.** Held-out eval suite, not just average quality — capability slices: instruction-following, math (GSM8K/MATH), code (HumanEval+), multi-turn coherence, long-context retrieval (RULER), refusals/safety, multilingual. Quantization specifically requires per-layer sensitivity analysis (which layers tolerate FP8 vs need BF16). Tolerances are negotiated with the model team per slice; no optimization ships through a "no regression on average" filter — averages hide capability-specific regressions.

**Canary.** New optimization rolls to 1% → 5% → 25% → 100% traffic over days. At each step: latency monitoring (TTFT/ITL p50/p95/p99), quality monitoring (live evals on traffic copy + sampled human review), error-rate monitoring. Automatic rollback on any of: p99 latency regression >10%, eval slice regression beyond tolerance, error-rate increase >X bps.

**Long-tail evals.** Capability-specific holdouts run nightly. Specifically: an FP8 prefill change that doesn't budge MMLU may still tank a niche code-completion capability. Catch these before they hit users.

**The hard rule:** an optimization shipping faster but worse is a regression, not a win. The TL is accountable for both axes. The eval engineer on the team owns the harness; the TL signs off on every promotion.

---

## 8. What I Push Back On

**[STAFF SIGNAL: pushback-on-metric]** Three pieces of the prompt deserve scrutiny before we commit 48 engineer-months.

**Is p99 TTFT the right metric?** p99 is dominated by ~5% of users submitting long prompts (uploads, RAG, code dumps). Driving p99 from 1.8s to 500ms is much more expensive than driving p95 from 1.4s to 600ms — the marginal optimization cost is steep on the tail. Three reframings worth a conversation:

1. **Bound the prompt length.** If the product UX accepts a cap (e.g., 32K user input), a large chunk of p99 disappears for free. Worth checking what fraction of users hit the cap and whether the product team will accept it.
2. **Stratified SLOs.** Short-prompt requests get a tight TTFT SLO (e.g., p99 ≤ 400ms on <4K). Long-prompt requests get a relaxed SLO (e.g., p95 ≤ 2s on >16K) and visible-progress streaming. Different optimization budgets per stratum.
3. **TTFT is not perceived speed.** Users perceive *when streaming feels fluid*. 800ms TTFT with smooth 50ms ITL feels faster than 400ms TTFT with stuttering 100ms ITL. If we are over-budget on engineering, ITL consistency may be a better lever than chasing TTFT p99.

**Is the cost worth it?** Some optimizations (disaggregation, hardware-aware routing onto B200) have real infra-cost implications. Cutting p99 by 200ms via 2× the replica count is a different conversation than cutting it via prefix caching. Product needs to put a dollar value on the latency improvement so we can compare against compute cost. The TL brings the cost table to the conversation; product brings the willingness-to-pay. **[STAFF SIGNAL: cost-explicit]**

**Is two quarters realistic for everything in the plan?** Roughly yes for the Q1+Q2 above as scoped — but only because we are explicitly *not* doing: custom kernels, NVFP4 on B200, full cross-engine KV sharing for arbitrary tenant prefixes. If product wants those too, that's Q3+ or more engineers.

---

## 9. Out of Scope (Q1+Q2) — Deferred With Reasoning

**Custom kernels.** Writing FA3-equivalent kernels for a specific bottleneck is 8–16 engineer-weeks for ~10–30% local kernel speedup, often <5% end-to-end. Defer until profiling shows a specific bottleneck unaddressed by published kernels. **[STAFF SIGNAL: engineering-reality]**

**NVFP4 / MXFP4 on B200.** Real upside (~2× over FP8) but the calibration and eval cost is significant, and most teams are still finishing the FP8 transition. Q3 candidate once Blackwell capacity is broadly available and the eval methodology has a few production cycles behind it.

**Speculative decoding (EAGLE-3) as a TTFT lever.** It isn't. As a *decode latency* lever it's a separate workstream that should run in parallel under a different owner, on the same eval harness.

**Cross-engine KV sharing for arbitrary user prefixes.** LMCache / KV transfer over RDMA is plumbing we partially get with disaggregation in Q2. Full cross-tenant cross-engine sharing with HBM→DRAM→NVMe tiering is its own project.

**Multimodal routing (Dynamo's embedding cache).** If our chat product handles image inputs, this is real (~30% TTFT win on repeat-image workloads per NVIDIA's GB200 numbers). Conditional on workload. Not assumed in scope.

**B200 acquisition and fleet transition.** Procurement-gated. The TL surfaces it to leadership but doesn't own it.

---

## Closing — What This Document Is and Isn't

**This is a hypothesis-driven program**, not a system design. The hypothesis is: prefix caching + chunked prefill + scheduling discipline get us to ~900ms p99 by end of Q1; FP8 prefill + FA3 + one conditional Q2 investment get us under 500ms by end of Q2. Each milestone has a measurable exit gate. Each has a fallback. The whole thing collapses gracefully if Week 1 measurement reveals the budget is materially different from the prior — in which case we re-plan and don't pretend we ran the right program.

The signals I want the interviewer to take away: I led with measurement, not optimization. I separated TTFT levers from decode levers. I sequenced with engineer-weeks and expected impact, not a list of cool techniques. I named the conflict between chunked prefill and disaggregation that a junior engineer misses. I pushed back on the metric before committing to it. And I scoped honestly — six engineers in two quarters do not ship everything, and I named what we don't do and why.