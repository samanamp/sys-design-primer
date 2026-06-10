---
title: "Fundamentals: Design an LLM Inference Platform"
description: ML system design — the general transformer-serving question, as a 40-minute interview answer with the napkin math that drives every decision
---

# Design a production inference platform for a large transformer

> **Interviewer prompt:** "Design the serving platform for a ~70B-parameter LLM: mixed traffic — interactive chat, agentic tool-use loops, and offline batch jobs — at, say, 10K requests/minute peak. Walk me through the architecture and the key trade-offs."

*Interview-style answer. First-person, as the candidate. This is the safety-net question — if the interviewer goes generic instead of world-models, this is the 40-minute version. Deeper treatments live in my LLM Sys Design notes (context management, KV cache management, LLM inference); this doc is the interview-shaped spine.*

---

## 1. Clarify scope and assumptions

1. **Traffic mix and SLOs?** Chat wants low TTFT[^1]; agent loops want low per-token latency and cheap repeated context; batch wants throughput and doesn't care about latency.
2. **One model or a family?** (Assume one 70B dense model; note where a small model changes the answer.)
3. **Context lengths?** (Assume up to 128K, median ~4K — the gap between median and max drives several designs.)
4. **Quality constraints on optimization?** (Quantization allowed; engine swaps must pass quality regression.)

**Assumptions:** 70B dense, GQA (grouped-query attention — many query heads share a few KV heads, shrinking the KV cache ~8× vs. one-KV-per-query; it's why the numbers below are merely painful instead of fatal), FP8 weights ≈ **70 GB**; H100s ($2/GPU-hr, 80 GB HBM @ 3.35 TB/s, ~1 PFLOP/s dense FP8 of which ~40–50% achievable).

[^1]: **TTFT / TPOT:** time-to-first-token (how long before the user sees anything — dominated by prefill) and time-per-output-token (the streaming rate — dominated by decode). They're the two latency SLOs of LLM serving, and almost every architecture decision trades one against the other or against cost.

---

## 2. The napkin math that generates the whole design

**Staff-level signal:** every serving technique below is a consequence of three numbers. Derive them first and the rest of the interview is downhill.

**(a) Prefill is compute-bound.** Processing a 4K-token prompt = 2 × 70e9 × 4096 ≈ 0.57 PFLOP — about a second of one GPU's compute. All prompt tokens are processed in parallel; the GPU's arithmetic units are the bottleneck.

**(b) Decode is memory-bandwidth-bound.** Generating *one* token requires reading **all 70 GB of weights** from HBM to do ~2 FLOPs per weight (≈0.14 TFLOP) — a trivial amount of math gated by an enormous read. (An MoE model changes this arithmetic: per-token weight traffic scales with *active* parameters, which is most of why sparse models win on serving cost.)

```
single stream:  3.35 TB/s ÷ 70 GB  ≈  48 tokens/s  — and the GPU's
                compute sits ~99% idle while you get it.
```

The fix is **batching**: B concurrent streams share one weight-read per step, so decode throughput scales ~linearly with B until something else runs out. What runs out is…

**(c) KV cache[^2] is the currency of the system.**

```
per token (70B-class: 80 layers, GQA 8 KV-heads × 128, FP8):
  2 × 80 × 8 × 128 × 1 B  ≈  160 KB/token
one 32K-context stream    ≈  5 GB
one H100 after weights    ≈  ~8 GB free → barely 1–2 long streams!
```

So a single 80 GB GPU can't even hold the weights plus a respectable batch's KV. **The three numbers force the architecture:** split the model (TP2 → 160 GB, ~90 GB for KV ≈ batch of ~18 full-32K streams, far more at the 4K median), batch decode aggressively, and treat KV bytes — not FLOPs — as the resource you schedule, evict, and cache.

[^2]: **KV cache:** in attention, every generated token attends to all previous tokens' key/value vectors. Recomputing them every step would be quadratic-and-ruinous, so they're cached — which converts the problem from "compute" to "memory": every active conversation holds megabytes-to-gigabytes of state on the GPU for its whole lifetime. My KV-cache doc covers the full frontier (Mooncake, Dynamo/NIXL, LMCache); here it's enough that KV is the scarce resource.

---

## 3. The serving engine — six techniques, each earning its place

Each one exists because of a specific failure of the naive design:

1. **Continuous batching.** Naive batching waits for the whole batch to finish; one 2K-token answer holds 31 finished streams hostage. Instead, admit/retire streams *every step*. This alone is typically 2–4× throughput, and it's why every modern engine (vLLM, TRT-LLM, SGLang) is built around it.
2. **Paged KV cache.** Contiguous per-stream KV allocations fragment HBM exactly like un-paged RAM; paging KV into fixed-size blocks (vLLM's core idea) gets utilization from ~60% to >95%, and the block table gives you copy-on-write sharing for free.
3. **Prefix caching.** The system prompt, the few-shot header, the agent's tool definitions — identical across thousands of requests. Cache their KV once (paged blocks + hashing make this natural), and a 3K-token shared prefix turns into a lookup instead of 3K tokens of prefill. For agentic traffic — same context replayed every tool-call iteration — this is the single biggest cost lever there is.
4. **Chunked prefill.** A 128K-token prefill is ~35 s of compute; scheduled whole, it stalls every decode stream on the GPU (TPOT spikes — head-of-line blocking). Slice prefills into chunks interleaved with decode steps: TTFT for the big request degrades slightly, TPOT for everyone else stays flat.
5. **Speculative decoding.** A small draft model proposes k tokens; the 70B verifies them in one parallel pass — one weight-read for several tokens, 2–3× single-stream speedup. The nuance an interviewer wants: it spends *compute* to save *bandwidth*, so it shines at low batch (latency-critical, GPU half-idle) and is **worthless at high batch** (compute is already the constraint — turn it off in the batch tier).
6. **Quantization.** FP8 weights halve the decode weight-read (≈2× decode speed) and the footprint; FP8 KV doubles how many streams fit. Gated per-release by quality regression, same discipline as every other optimization.

**Parallelism note:** TP2–TP4 inside an NVLink island, replicas beyond that. Decode-phase TP all-reduces carry *one token's* activations (~16 KB) — latency-bound, benign. This is the exact opposite regime from the world-model design (#1), where every pass moved 12K tokens and 75 MB per all-reduce — same formula, the token count flips which term dominates. Knowing *why* the same technique is cheap here and expensive there is the transferable skill.

---

## 4. Platform architecture — tiers, routing, disaggregation

```
                      ┌───────────────────────┐
                      │  Router / Scheduler   │
                      │  prefix-aware, load-  │
                      │  aware, priority-aware│
                      └──┬────────┬────────┬──┘
        interactive      │        │        │        batch
        (TTFT SLO)       ▼        ▼        ▼        (no SLO, spot/
   ┌──────────────────────┐  ┌─────────────────┐     preemptible)
   │  PREFILL POOL        │  │  DECODE POOL    │  ┌─────────────────┐
   │  compute-heavy,      │─►│  bandwidth-     │  │  BATCH TIER     │
   │  chunked, KV         │KV│  heavy, big     │  │  max batch, no  │
   │  streamed out        │  │  cont. batches  │  │  spec-decode,   │
   └──────────────────────┘  └─────────────────┘  │  fills valleys  │
              ▲                                    └─────────────────┘
              │ shared prefix KV store (paged blocks, hash-addressed,
              ▼ HBM → CPU DRAM → SSD tiers)
   ┌──────────────────────────────────────────────┐
   │  KV / PREFIX CACHE  (the system's real state)│
   └──────────────────────────────────────────────┘
```

- **Prefill/decode disaggregation:** the two phases want opposite hardware profiles (compute vs bandwidth) and pollute each other's SLOs when colocated. Separating them — with KV streamed between pools — is the Mooncake/Dynamo architecture; worth it at scale, overkill for a single-node deployment (say which regime you're in).
- **Routing is cache-aware first:** sending a request to the replica that already holds its prefix KV beats any load-balancing heuristic. Session affinity for agents (their KV is here), hash-routing for shared prefixes.
- **The batch tier is the economic shock absorber:** it runs preemptible, fills diurnal valleys, and makes peak capacity for interactive traffic affordable. **Goodput** — requests completing *within their SLO* per GPU-hour — is the metric, not raw tokens/s; a platform at 100% utilization missing every TTFT target is a failed platform at great efficiency.

**Cost sanity check:** TP2 decode at batch ~32: step time ≈ 70 GB ÷ 6.7 TB/s ≈ 10.5 ms → ~95 steps/s × 32 streams ≈ 3,000 tok/s per pair ≈ 1,500 tok/s/GPU → at $2/hr ≈ **$0.37 per million output tokens** before prefill, cache hits, and margin. Knowing this number to within 2× lets you sanity-check every vendor claim and capacity plan in the room.

---

## 5. Evaluation, monitoring, rollout

The serving platform changes weekly (engine versions, kernels, quantization, schedulers) under a model that's supposed to stay *exactly the same* — so the discipline is "prove nothing changed":

- **Quality invariance gates:** golden-prompt suite with pinned seeds — logprob deltas (the per-token probabilities the model assigns; far more sensitive to a kernel or quantization change than eyeballing outputs) and output diffs per release. Quantization, kernel, and engine swaps all ship through it. (Bitwise-identical output is not achievable across kernels; *statistically indistinguishable* is the standard, and the gate encodes it.)
- **Latency:** TTFT/TPOT percentiles *per traffic class* — p99, not means; the means always look fine.
- **Cache health:** prefix hit rate, KV utilization, eviction rates. A hit-rate drop is an early-warning for both a cost regression and a routing bug.
- **Failure modes worth naming before they page you:** OOM cascades (one long-context burst evicts the cache, misses pile up, prefill load doubles — admission control by *predicted KV footprint*, not request count); hot prefixes (one viral system prompt — replicate its blocks); head-of-line blocking (chunked prefill plus a max-tokens-in-flight cap).
- **Rollout:** shadow traffic for engine upgrades, canary by traffic percentage, instant rollback (stateless workers + versioned engine images — the KV cache drains, nothing else to migrate).

---

## 6. When simpler wins

- **A smaller model with a fatter cache** often beats a bigger model: if 8B + good retrieval passes the task evals, it's ~9× cheaper per token *and* fits whole on one GPU (no TP, trivial ops). Run the task eval before the capacity plan.
- **Semantic/response caching** above the platform: the cheapest token is one you never generate.
- **Single-node first:** one TP2 box with vLLM serves a surprising amount of traffic; disaggregation, KV tiering, and multi-pool routing earn their complexity only past the point where you can measure the colocation interference they remove.

---

## 7. Summary — what I'd want the interviewer to remember

1. **Three numbers force everything:** prefill is compute-bound, decode is one-full-weight-read-per-token (48 tok/s unbatched), and KV at 160 KB/token is the scarce resource — the architecture is just these facts arranged in order.
2. **Batch to amortize the weight read; page and share the KV; cache every repeated prefix** — continuous batching, paged KV, prefix cache are the non-negotiable core.
3. **Know which regime each trick lives in:** speculative decoding at low batch only; chunked prefill to protect TPOT; disaggregation at scale only. Same math, opposite answer to the world-model case — and saying *why* is the differentiator.
4. **Goodput per GPU-hour, per traffic class** — not tokens/s. The batch tier absorbs the peaks and pays for the valleys.
5. **The model is constant, the platform isn't:** quality-invariance gates on every engine change, p99s per class, admission control in KV-bytes, boring rollbacks.
