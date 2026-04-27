---
title: LLM Inference Serving at Scale
description: LLM Inference Serving at Scale
---
```
"Design an inference serving system for a 70B-parameter LLM. Target 10K concurrent users, mix of interactive chat (low TTFT matters) and batch (throughput matters). Multiple GPU types available (H100, A100, B200). Design the serving stack."
```
---

### Designing an Inference Serving System for a 70B LLM at 10K Concurrent Users

## 1. Scope, Reframing, and What I'm Pushing Back On

Before architecture, I want to reframe the question and surface ambiguities that change the design materially.

**The question conflates two workloads that have nothing in common at the hardware level.** Prefill — the first forward pass over the input prompt — is compute-bound: matmuls dominate, GPUs run near peak FLOPs, and latency scales linearly with prompt length. Decode — generating one token at a time autoregressively — is memory-bandwidth-bound: every step reads the full weight matrix and the entire KV cache from HBM, FLOPs are wasted, and latency is set by HBM bandwidth divided by working-set size. Serving them as a single workload is the central design error. **[STAFF SIGNAL: prefill-decode reframing]**

**The KV cache, not the weights, is what determines concurrency, latency under load, and cost.** Weights are a fixed cost. KV grows with concurrency × context length, and at 70B even with GQA it dominates HBM beyond weights very quickly. Every major decision below — paging, sharing, eviction, offloading, disaggregation — is a KV-cache decision in disguise. **[STAFF SIGNAL: KV cache as primary resource]**

**"10K concurrent users" is under-specified to the point of being misleading.** Concurrent users at 50 tok/s of decode with 200-token average outputs is ~2,000 RPS sustained. Concurrent users meaning "logged in but idle 95% of the time" is closer to 100 RPS. These differ by 20× in capacity. I also need: **prompt length distribution** (a workload with p99 prompt = 32K tokens needs twice the prefill capacity of one with p99 = 4K), **output length distribution** (long outputs amortize TTFT cost; short outputs make TTFT dominate UX), and **the interactive/batch ratio** (60/40 vs 95/5 changes whether disaggregation is worth the operational cost). **[STAFF SIGNAL: saying no]**

**Committed assumptions for this answer:**
- **Model:** dense 70B with GQA, 80 layers, 64 attention heads, 8 KV heads, 128 head_dim (Llama-3.3-70B-class). MoE changes parallelism strategy substantially; I call this out where it matters.
- **Workload:** 70/30 interactive/batch by request count; interactive p99 prompt = 8K tokens, mean = 1.5K; batch p99 = 32K, mean = 6K. Average output 250 tokens interactive, 800 batch.
- **Tenancy:** multi-tenant. Multiple customers share GPUs; KV is sharable for identical prefixes within a tenant, never across tenants.
- **SLOs:** interactive p99 TTFT ≤ 500ms, p99 ITL ≤ 50ms; batch best-effort with throughput SLO of 95% of theoretical peak.
- **Streaming:** required. SSE or HTTP/2.
- **Multi-modal, RAG-with-injection, and tool-use orchestration:** out of scope for the serving layer; assume those happen upstream and arrive as text tokens.

**[STAFF SIGNAL: scope negotiation]** With this nailed down, the design follows.

---

## 2. Capacity Math — The Numbers That Drive Every Decision

**[STAFF SIGNAL: capacity math]**

### Weight memory
| Precision | Total | Per GPU at TP=2 |
|---|---|---|
| BF16 | 140 GB | 70 GB |
| FP8 (E4M3) | 70 GB | 35 GB |
| INT4 (AWQ) | 35 GB | 17.5 GB |

A single H100 SXM (80 GB) cannot hold BF16 weights. TP=2 is the floor for BF16; TP=1 becomes feasible with FP8 (35 GB leaves 45 GB for KV, activations, and overhead — tight but workable for short contexts). **I commit to TP=2 H100 with FP8 weights as the primary configuration.**

### KV cache per token (the number that matters)
With GQA at 8 KV heads, 128 head_dim, 80 layers:
```
KV per token, per layer = 2 (K+V) × 8 × 128 × 2B (BF16) = 4096 B = 4 KiB
KV per token, all layers = 4 KiB × 80 = 320 KiB (BF16)
KV per token, FP8 KV    = 160 KiB
```
**Without GQA** (full MHA at 64 heads): 2 × 64 × 128 × 2 × 80 = 2.56 MiB per token. 8× larger. GQA is non-negotiable for serving economics; I'd refuse to deploy a non-GQA 70B model at this scale.

### HBM bandwidth budget (the metric that actually predicts decode latency)
| GPU | HBM | BW | FP8 TFLOPs (peak) |
|---|---|---|---|
| A100 80GB | 80 GB | 2.04 TB/s | — (no FP8) |
| H100 SXM | 80 GB | 3.35 TB/s | 1,979 |
| B200 | 192 GB | 8.0 TB/s | 4,500 (FP4: 9,000) |

### Decode latency floor (single user, TP=2 H100, FP8 weights)
```
Time per token = (weights read + KV read) / aggregate_BW
              ≈ 70 GB / 6.7 TB/s = 10.4 ms
```
That's **~96 tok/s** as the bandwidth-imposed ceiling for single-user decode at TP=2 H100. At TP=4 across NVLink (4× H100), the same model gives ~5 ms/tok = ~190 tok/s, but at higher fixed cost per replica. **[STAFF SIGNAL: roofline reasoning]**

### Decode at batch (the metric that drives throughput economics)
At batch B with avg context L, with FP8 weights and FP8 KV:
```
per-step cost ≈ 70 GB (weights, amortized over B requests)
              + B × L × 160 KiB (KV reads per request)

At B=64, L=4K:
  KV traffic = 64 × 4096 × 160 KiB = 41 GB
  Total = 70 + 41 = 111 GB
  Time = 111 / 6700 = 16.6 ms/token
  Throughput = 64 / 16.6ms = 3,855 tok/s aggregate
```
**Per-token cost drops as batch grows until KV traffic eats the win.** At B=128, L=4K, KV is 82 GB and total is 152 GB → 22.7 ms/token. Aggregate goes to 5,640 tok/s, but per-user ITL goes to 22.7ms. The batch-vs-ITL Pareto curve is the core tension of the scheduler.

### Prefill latency
```
4K prompt FLOPs = 2 × 70B × 4096 = 573 TFLOPs
H100 FP8 at 60% MFU = 1,180 TFLOPs/s; TP=2 effective ≈ 2,000 TFLOPs/s
Prefill time (4K) ≈ 287 ms
Prefill time (8K) ≈ 575 ms — already over interactive TTFT SLO
Prefill time (32K) ≈ 2.3 s
```
**8K-token prompts already blow the 500ms TTFT SLO without help.** Chunked prefill, prefix caching, and disaggregation are not optional luxuries — they are SLO-mandatory.

### Concurrency budget per replica (TP=2 H100)
```
HBM available = 160 GB
Weights (FP8) = 70 GB
Activations + overhead = ~20 GB
KV available = ~70 GB

At 4K avg context, 160 KiB/token (FP8 KV):
  KV per request = 4096 × 160 KiB = 640 MiB
  Concurrent requests per replica ≈ 70 GB / 640 MiB ≈ 110
```
Roughly **100 concurrent in-flight requests per TP=2 H100 pair** before paging/eviction kicks in. To serve 10K concurrent at this density: ~100 replicas = ~200 H100s. With prefill/decode disaggregation, decode replicas are denser (smaller activation footprint), prefill replicas are larger but transient — total fleet is similar.

---

## 3. High-Level Architecture

```
                          ┌────────────────────────────┐
   Clients ──────►        │   LLM Gateway / Router     │
                          │  - auth, rate limit, RPM   │
                          │  - tenant priority class   │
                          │  - prefix-cache aware route│
                          └────────────┬───────────────┘
                                       │
                          ┌────────────┴───────────────┐
                          │  Global Scheduler (cell)   │
                          │  - SLO classes, queues     │
                          │  - chunked prefill policy  │
                          │  - hardware routing        │
                          └─┬──────────┬───────────┬───┘
                            │          │           │
              ┌─────────────┘          │           └────────────┐
              ▼                        ▼                        ▼
    ┌───────────────────┐    ┌──────────────────────┐    ┌──────────────┐
    │ Aggregated Pool   │    │ Disaggregated Pool   │    │ Batch Pool   │
    │ (short interactv) │    │                      │    │              │
    │ H100 TP=2 replicas│    │ Prefill: H100/B200   │    │ B200 TP=2,   │
    │ Prefill+Decode    │    │ Decode:  H100 TP=2   │    │ FP4 weights, │
    │ on same engine    │    │ KV transfer over NVL │    │ huge batches │
    └───────────────────┘    └──────────────────────┘    └──────────────┘
                            ▲                          ▲
                            │                          │
                  ┌─────────┴────────────┐    ┌────────┴──────┐
                  │ Distributed KV Store │    │ Object Storage│
                  │ (per-tenant prefix)  │    │ (model wts)   │
                  │ Hot:HBM Warm:DRAM    │    └───────────────┘
                  │ Cold:NVMe            │
                  └──────────────────────┘
```

Three serving paths, one scheduler:

1. **Aggregated path** — short prompts (<2K), prefill and decode co-located. Lowest TTFT for short prompts because no KV transfer hop. ~60% of interactive traffic.
2. **Disaggregated path** — long prompts or high-load periods. Prefill happens on dedicated prefill workers, KV ships to decode workers. Higher TTFT floor (transfer cost) but predictable, doesn't head-of-line-block decode for other users.
3. **Batch path** — distinct hardware preference (B200 with FP4 if quality allows, else H100 with very large batches). Best-effort SLO, scheduled in gaps.

The router is **prefix-cache-aware**: it hashes the prompt prefix and routes requests with shared prefixes to the same engine to maximize cache hits. This is the single most important routing decision in production serving.

---

## 4. Parallelism and Quantization — The Committed Choices

### Parallelism: TP=2 on H100 within NVLink

**[STAFF SIGNAL: parallelism precision]**

For a 70B dense model, the choice space is TP=2, TP=4, TP=8, or TP+PP. I commit to **TP=2 within an NVLink-connected pair**, with no pipeline parallelism.

- **TP=2** keeps weight memory at 35 GB/GPU (FP8), leaves ~70 GB for KV. Two all-reduces per layer at the matmul outputs (column-parallel → row-parallel). Each all-reduce on NVLink (900 GB/s SXM) at activation size ~32KB per token per layer is sub-100μs — negligible. 80 layers × 2 all-reduces × ~100μs ≈ 16 ms of comm overhead per token. This is the dominant non-bandwidth cost.
- **TP=4 rejected for default:** doubles all-reduce count, halves KV-per-GPU available so concurrency drops, doesn't actually help latency for 70B because we're already bandwidth-bound on weights, and burns 2× the GPUs per replica. Justifiable only if single-user latency must be minimized regardless of cost.
- **TP=8 rejected:** same reasons more so. Also crosses NVSwitch boundaries on some topologies, with non-uniform bandwidth.
- **PP rejected:** adds pipeline bubbles (20-30% efficiency loss in inference where micro-batch counts are small) and breaks continuous batching scheduling. PP is the right call when the model doesn't fit in TP within a single NVLink island — for 405B-class models across nodes, not for 70B.

**MoE caveat:** if the model were actually a 70B-active / 400B-total MoE (DeepSeek-V3-style), the answer changes radically: expert parallelism (EP) becomes the dominant axis, and the all-to-all from token routing becomes the bottleneck instead of all-reduce. I'd commit to TP=2 + EP=8 across nodes, and the scheduler would need expert load-balancing on top.

### Quantization: FP8 weights + BF16 attention compute + FP8 KV cache

**[STAFF SIGNAL: quantization discipline]**

- **Weights: FP8 E4M3.** H100 has hardware FP8 throughput at 2× BF16. Calibration with ~512 sequences from a representative distribution; per-channel scaling for linear layers; dynamic activation scaling for the matmul inputs. I'd accept FP8 only after a published-quality eval pass: lm-eval-harness on MMLU, GSM8K, HumanEval; LLM-as-judge on internal eval set with score delta < 0.5%. NE-style per-layer sensitivity analysis to identify layers that need to stay BF16 — typically the first attention block and the LM head. Adaptive replacement: ship one model artifact with per-layer precision encoded in metadata.
- **Attention compute: BF16** with FP8 inputs. FlashAttention-3 supports FP8 attention; in practice, FP8 attention has subtle quality regressions on long contexts because softmax is sensitive to dynamic range. I'd run BF16 softmax with FP8 GEMM inputs (the FA3 mixed mode) and keep this configurable per-layer.
- **KV cache: FP8 E4M3** with per-token scaling. Halves KV memory traffic and storage, doubles concurrency. The cost is ~0.3% on perplexity; recoverable with calibration.
- **FP4 (MXFP4) on B200:** reserved for batch path. The Blackwell hardware support is real, but the calibration story is harder — block-scaled FP4 needs careful per-block scale handling and the quality gap on instruction-following/code is non-trivial. I'd ship FP4 only after an A/B with explicit user opt-in.
- **Rejected: INT4 (AWQ/GPTQ).** Lossier than FP8 in our quality eval and the throughput win on H100 is illusory because there's no native INT4 tensor core on H100 — you dequantize-on-load. AWQ wins on memory-constrained setups (e.g., consumer hardware), not here.

### Heterogeneous routing across H100 / A100 / B200

**[STAFF SIGNAL: heterogeneity reasoning]**

The fleet has all three for cost reasons. Routing logic:
- **H100:** primary for both interactive prefill and decode. FP8 weights, FP8 KV. Best $/tok for interactive workloads.
- **B200:** batch-only by default. 192 GB HBM lets us hold the full 70B FP8 weights on a single GPU (TP=1 viable!) with 120 GB headroom for KV. At 8 TB/s, single-token decode ≈ 9ms — but the throughput win is at huge batches. B200 with TP=1, FP4 weights, batch=512 on a single GPU is the throughput sweet spot for batch.
- **A100:** overflow / cost-sensitive tier. No FP8, so weights run in BF16 → 140 GB → TP=2 minimum, lower throughput per dollar than H100 on FP8. A100 only earns its keep on long-context decode where bandwidth (2 TB/s) doesn't dominate the way it does on shorter sequences. Practically: A100s carry the lowest-priority tenants and absorb traffic spikes when H100 capacity is tapped.

The complexity cost is real: three model artifacts (BF16 for A100, FP8 for H100, FP4 for B200), three calibration runs, three quality evals per release, three sets of kernels. Worth it because the price-per-token spread across these tiers is 3-4×.

---

## 5. Deep Dives

### 5.1 Continuous Batching with Paged KV Cache

The pre-2023 default was static batching: collect N requests, pad to the longest, run them as a batch through both prefill and decode until all finish. This wastes massively. Suppose request lengths are 100, 200, 1000 tokens. The 100-token request finishes first but its slot is held until the 1000-token request finishes. GPU utilization on the wasted slots is zero. With realistic length variance (LogNormal output lengths typical of chat), static batching achieves ~30% of theoretical peak throughput.

**Continuous batching** (also called in-flight batching) makes the batch dimension dynamic per decode step. At each step, finished sequences exit, new sequences enter, and the batch is rebuilt. The decode kernel handles a varying batch size each step. This recovers the ~70% throughput loss from static batching and is the table-stakes baseline.

The blocker historically was KV cache fragmentation. A new sequence joining the batch needs contiguous KV memory the size of its expected output. You don't know that size in advance. So you either over-allocate (wasting memory and capping concurrency) or you allocate as you go and fragment — with old static allocators, fragmentation reached 60-80% of allocated memory.

**PagedAttention (vLLM)** treats the KV cache as paged virtual memory. KV is stored in fixed-size pages (typically 16 tokens per page). Each sequence has a **block table** mapping its logical token positions to physical page IDs. A page is allocated when the sequence advances past its current page boundary. The attention kernel takes the block table as an input and indirects through it to fetch K and V for each token.

```
Sequence A (logical positions 0..47):
  block table = [page_5, page_12, page_8]
  page 5  → tokens 0..15
  page 12 → tokens 16..31
  page 8  → tokens 32..47

Free list: [page_3, page_9, page_15, ...]
When seq A advances to position 48: pop a page from the free list,
append to block table. No copy, no fragmentation.
```

Costs:
- Extra indirection per attention op (load block table → load page → load K/V). Modern paged-attention kernels (FlashInfer, vLLM v1) hide this near-perfectly with prefetching.
- Page size choice: 16 is the typical default. Smaller pages reduce internal fragmentation but increase block-table size and indirection overhead. Larger pages waste memory at sequence boundaries.

```
Continuous batching timeline (single decode loop):

step:  t=0    t=1    t=2    t=3    t=4    t=5    t=6    t=7
slot0: [A0]   [A1]   [A2]   [A3]   [A4]   [A5]   [A6=END] [F0]
slot1: [B0]   [B1]   [B2=END] [D0] [D1]   [D2]   [D3]    [D4]
slot2: [C0]   [C1]   [C2]   [C3]   [C4]   [C5]   [C6]    [C7]
slot3: [...]  [...]  [E_pf]<-prefill chunk for new seq E
                              [E0]   [E1]    [E2]    [E3]

Slot reuse is immediate. Prefill of E is interleaved as a chunk
into the decode batch (chunked prefill).
```

**[STAFF SIGNAL: rejected alternative]** Static batching rejected: ~3× lower throughput at our length variance. Iteration-level priority inversion (where a long sequence head-of-lines a short one) rejected: continuous batching admits new requests every step.

### 5.2 Prefill/Decode Disaggregation

This is the most important architectural choice in modern serving and is the headline win of systems like Mooncake, DistServe, and Splitwise.

**The problem with co-located prefill/decode:**

When prefill and decode share a GPU, you have two failure modes. If prefill is run as part of the decode batch ("piggybacked"), a single 8K-token prompt arriving causes decode for all other users to stall for ~600ms — that's a 12× ITL violation for ~30 in-flight users. If prefill is run separately with strict priority over decode, decode is bursty and TTFT for the prefill request is great but ITL for everyone else suffers and effective decode throughput drops because the GPU keeps switching workloads.

**Chunked prefill** (the in-cluster fix) breaks the prefill into chunks (e.g., 512 tokens) and interleaves chunks with decode steps. Each iteration of the decode loop processes one chunk of prefill and one decode step for everyone. This bounds ITL inflation — at 512-token chunks the prefill cost per iteration is ~70ms vs ~17ms for decode-only, so ITL inflates ~4× when prefill is active, but capped. Chunked prefill is necessary regardless of disaggregation.

**Disaggregation goes further:** prefill happens on dedicated prefill workers, KV cache is shipped to decode workers, decode happens on dedicated decode workers. Now neither workload can interfere with the other.

```
Disaggregated topology:

Request ──► Prefill Pool (H100 TP=2 or B200 TP=1)
            - Optimized for compute throughput
            - Large prefill batches (compute-bound favors batching)
            - Produces KV cache + first token
                  │
                  │ KV transfer (NVLink island OR NVLink-Switch network)
                  ▼
            Decode Pool (H100 TP=2)
            - Optimized for HBM bandwidth
            - High concurrency, paged KV
            - Streams tokens to client
```

**KV transfer cost:**
For a 4K-token prefill at 70B with GQA, FP8 KV:
```
KV size = 4096 × 160 KiB = 640 MiB
Over NVLink (900 GB/s SXM)        ≈ 0.7 ms
Over NVSwitch fabric (450 GB/s)   ≈ 1.4 ms
Over 400 Gbps RDMA (50 GB/s)      ≈ 13 ms
Over 100 Gbps Ethernet (12 GB/s)  ≈ 53 ms
```
The transfer cost has to be small relative to prefill time. For 4K prompts, prefill is ~280ms; even Ethernet-class transfer (50ms) is acceptable but degraded. For RDMA fabrics, transfer is ~5% of prefill — clearly worth it.

The trick is that the transfer can be **pipelined per layer** — as soon as layer L's KV is computed, start shipping it while layers L+1..N continue computing. With pipelining, the perceived transfer cost is the cost of the *last layer's* KV, not the whole stack. That's ~8 MiB per layer for FP8 KV at 4K tokens, or ~10μs over NVLink. Effectively free.

**When disaggregation pays off:**
- High prefill heterogeneity (mix of short and long prompts). The piggybacked or chunked-prefill approaches degrade gracefully; disaggregation gives strict isolation.
- Different optimal hardware for prefill vs decode. B200 prefill + H100 decode is a real config — B200's compute is ~2× H100 but its bandwidth advantage is wasted on prefill, and H100's lower memory cost wins for decode.
- High QPS. The fixed cost of the transfer infrastructure amortizes.

**When disaggregation doesn't pay off:**
- Low QPS. The fixed cost of separate pools dominates.
- Workloads dominated by short prompts. Aggregated path is fine; disaggregation adds complexity for marginal gain.
- Networks worse than 100 Gbps Ethernet without RDMA. KV transfer becomes a TTFT killer.

**[STAFF SIGNAL: rejected alternative]** Single aggregated pool rejected for the interactive path because mixed prompt-length distribution causes head-of-line blocking that we can't get below SLO with chunked prefill alone above ~80% utilization.

### 5.3 KV Cache Management: Sharing, Eviction, Offloading

**[STAFF SIGNAL: KV sharing as product lever]**

Most production prompts share prefixes. System prompts (often 1-4K tokens) repeat across every request from a tenant. Conversation histories repeat across every turn of a chat. RAG chunks repeat across queries hitting similar documents. **The first 80-90% of any given prompt's KV is computable once and shared.**

**Prefix caching with radix tries (SGLang's RadixAttention):**

Maintain a radix tree keyed by token prefix. Each node holds a reference to KV pages. When a new request arrives, walk its tokens through the tree to find the longest matching prefix; reuse that KV; only run prefill for the suffix.

```
Radix tree state:

  ROOT
   ├── ["You are a helpful..."] (sys prompt A) ─┬─ ["What's the weather"] (KV)
   │                                            └─ ["Translate to French"] (KV)
   └── ["You are a code..."] (sys prompt B) ────── ["Refactor this..."] (KV)

New request: "You are a helpful assistant\n\nWhat's the weather like?"
→ Walk: matches sys prompt A + matches "What's the weather" mostly
→ Hit length: 2,012 tokens
→ Run prefill only for the remaining 8 tokens
→ TTFT drops from 287ms (4K full prefill) to ~5ms
```

**Eviction:** LRU on prefix tree nodes weighted by hit frequency. Hot system prompts pinned (refcount-based); rare branches evicted first. The tradeoff: aggressive eviction frees memory for new requests but loses cache locality; conservative eviction maintains hit rate but caps concurrency.

**Multi-tenant isolation:** tenants A and B never share KV pages even if their prompts are byte-identical. Side-channel risk through prefix-cache hit timing is real (you can probe whether another tenant has a specific system prompt by measuring TTFT). Per-tenant radix trees, or per-tenant salt prefixes that hash differently.

**KV offloading hierarchy:**
```
Hot:  HBM (3.35 TB/s, 80 GB)    — active sequences
Warm: DRAM (PCIe Gen5, 64 GB/s) — recently-finished, likely-resumed
Cold: NVMe (12 GB/s)            — long-tail conversation history

Cost to restore 4K-token KV (640 MiB) to HBM:
  From DRAM: 10 ms — cheaper than recomputing (287ms prefill)
  From NVMe: 53 ms — still cheaper than recomputing
```

**Recompute vs swap-in:** if memory pressure forces evicting a sequence's KV, the choice is swap-in-on-resume vs recompute. Swap-in costs ~bandwidth × KV size. Recompute costs prefill time. For sequences past their first token, swap is almost always cheaper. The exception is when the prefix is itself cached elsewhere — then recompute via the cache hit is free.

**Prompt caching as a product feature:** Anthropic, OpenAI, and Google all expose prompt caching at the API tier with explicit pricing. From the serving design's view, prompt caching is just prefix caching with TTL and explicit cache keys. The product surface — "your cached prompts persist for 5 minutes for free, you pay 10% of input cost on cache hits" — directly maps to the eviction policy.

### 5.4 Scheduling: Interactive + Batch on Shared Hardware

**[STAFF SIGNAL: scheduling discipline]**

The scheduler is the single most important piece of code in this system. Its job: maintain interactive p99 TTFT ≤ 500ms and p99 ITL ≤ 50ms while extracting maximum throughput from batch.

**Mechanisms:**

1. **Priority queues with class-based admission.** Three classes: `interactive_paid`, `interactive_free`, `batch`. Each has a deadline and a max-queue-time.
2. **Chunked prefill with adaptive chunk size.** Default chunk = 512 tokens. When interactive queue depth grows, shrink chunks to 256 to reduce ITL stalls. When system is idle, grow to 2048 to amortize launch overhead.
3. **Decode preemption.** A long-running batch decode can be preempted at iteration boundaries. KV is preserved (paged, easy). Interactive prefill runs. Batch resumes. Preemption cost is one decode iteration (~17ms) of latency for the preempted batch sequence — invisible to a batch SLO.
4. **SLO-aware batch sizing.** Per the Section 2 math, batch=64 gives 16.6ms ITL; batch=128 gives 22.7ms. Under load with interactive p99 ITL = 50ms, run batch=128. Under low load, run batch=32 to free capacity for surge response. The scheduler estimates current load and sets the batch ceiling each iteration.
5. **Admission control on backpressure.** When interactive queue depth exceeds 2× p99 expected, return 503 with retry-after. Better to fail fast than hold a request in queue past its TTFT SLO.

```
Scheduling decision when interactive request arrives during batch run:

state: 64 batch decode sequences mid-flight, ITL ~17ms
event: interactive prefill of 2K tokens arrives, deadline 500ms TTFT

option A (chunked prefill, no preempt):
  Insert 512-token prefill chunk in next iteration
  Iteration cost: 17ms decode + 70ms prefill chunk = 87ms
  4 chunks needed → 4 × 87ms = 348ms total prefill
  Batch sequences see 4 stalls of 70ms each
  TTFT met (348ms < 500ms), batch ITL inflated ~4× temporarily

option B (preempt batch, dedicated prefill):
  Drain current decode iteration (17ms)
  Run 2K prefill standalone: 144ms
  Resume batch decode
  TTFT = 17 + 144 = 161ms (much better)
  Batch sequences see 161ms gap (one or two missed decode iterations)

choice: option A under low load (preserve batch progress);
        option B under high load (preserve interactive SLO budget).
```

**Failure mode: thundering herd.** A traffic spike of 1000 interactive requests in 200ms. Queue blows past admission threshold. Two responses: (a) 503 + retry-after with jitter; (b) drop low-priority batch immediately to free capacity. Both happen. The autoscaler then spins up replicas, but cold-start is 30+ seconds (Section 8) so this only helps the next burst, not this one.

### 5.5 Quantization (Already Committed in §4 — Adding Operational Detail)

**Calibration pipeline:** weekly automated. Take 1024 sequences sampled across tenants and use cases, run BF16 forward, run FP8 forward with current scale factors, measure per-layer activation MSE. If drift exceeds threshold, recalibrate. Ship the new artifact through the staged rollout (canary at 1% traffic, watch eval scores, expand).

**Per-layer sensitivity:** in practice for 70B Llama-class:
- LM head: keep BF16. The output distribution is sensitive to small quantization noise; FP8 here costs measurable downstream task performance.
- First 2 attention layers: BF16. Early layers' activations have unusually wide dynamic range.
- All other linears: FP8.
- All KV projections: FP8 (matching FP8 KV cache).
- LayerNorms / RMSNorms: BF16 always (cheap, sensitive).

**Quality monitoring in production:** sample 0.1% of production responses, score them with an LLM-judge against a fixed rubric, alert on score drift > 0.5%. Side-by-side eval: 0.01% of requests get both a BF16 and FP8 response on the same prompt; comparison runs offline. Catches regressions that lm-eval-harness misses.

### 5.6 Speculative Decoding

Decode is bandwidth-bound. The GPU's compute units sit ~95% idle waiting for HBM. **Speculative decoding fills that idle compute by speculatively generating multiple candidate tokens and verifying them in parallel.**

A draft model (much smaller, ~1B params) generates K candidate tokens autoregressively. The target model runs one forward pass over all K candidates simultaneously — same memory traffic as one decode step, K× more compute (which was idle anyway). The target's logits at each position are compared to the draft's; matching tokens are accepted; the first mismatch falls back to the target's token at that position.

Variants:
- **Vanilla SpD** (Leviathan et al.): independent draft model. Draft model is a separate small LLM. Acceptance rate ~50-70% on similar-distribution traffic. Speedup ~1.5-2.5×.
- **Medusa:** multiple decoding heads on the target model itself. Each head predicts a different future position. No separate draft model. Acceptance ~60-70%. Speedup ~2-3×. Operational win: one model, no draft training.
- **EAGLE / EAGLE-2 / EAGLE-3:** draft model is a small autoregressive head conditioned on the target's hidden states (shares feature representation). Acceptance ~80%, speedup ~3-3.5×. State-of-the-art at time of writing. Cost: training a draft model that tracks each target model release.

**Choice:** EAGLE-2 for interactive decode. The acceptance rate is high enough that the speedup is real (~3×), and the additional engineering cost is amortized across the deployment. Draft model adds ~500MB to HBM (negligible at 80GB).

**When speculative loses:**
- Very large batch sizes. At batch=128, decode is already throughput-saturated — there's no idle compute to fill. The K× extra compute now competes with the existing batch and slows it down.
- Workloads where prefill dominates total latency (very short outputs).
- Tool-calling / structured-output workloads where every token has very high entropy — acceptance rate craters.

The scheduler turns speculative on/off per-replica based on current batch size: enable when batch < 64, disable above.

### 5.7 Multi-Tenancy and Long Context (Briefer)

**Multi-tenancy:** rate limits expressed in TPM (tokens-per-minute), not RPM. A tenant burning 10× more tokens than another is the one consuming 10× more capacity. RPM is a UX backstop. Per-tenant TPM budgets, per-tenant priority class, per-tenant prefix cache namespace. Billing event per token, not per request. Noisy neighbor protection via per-tenant prefill capacity reservation (no single tenant can occupy >X% of prefill capacity).

**Long context (32K+):** TTFT scales linearly with prompt length, by definition. At 100K tokens, prefill on H100 TP=2 is ~7 seconds. SLO must change. Architectural pushback: most "long context" use cases (document Q&A, codebase navigation) are better served by RAG with short retrieved chunks; long-context inference is the right answer for tasks where the context structure can't be chunked (whole-document summarization, cross-document synthesis). For genuine long-context, ring attention distributes the attention computation across multiple GPUs so prefill scales to many GPUs in compute. This is a different parallelism axis (sequence parallelism) layered on top of TP.

---

## 6. Failure Modes and Graceful Degradation

**[STAFF SIGNAL: failure mode precision] [STAFF SIGNAL: blast radius reasoning]**

| Failure | Detection | Response | User-visible impact |
|---|---|---|---|
| GPU OOM from long-prompt outlier | Allocator failure exception | Reject request with 413; tighter prompt-length validation upstream | One user 4xx |
| One GPU in TP=2 fails | NCCL hang timeout (10s) or driver event | Mark replica down; in-flight sequences fail; KV gone; clients retry against another replica | All sequences on that replica error; clients see SSE disconnect |
| KV cache exhaustion under load | Eviction queue depth metric | Evict the longest-running batch sequence's KV (recompute on resume); if interactive, swap to DRAM | Batch sequence sees ~50ms swap stall; interactive unaffected |
| NCCL hang mid-decode | Watchdog timeout per iteration (default 30s, set to 1s for inference) | Kill the engine process; replica cycles through health check; fail in-flight requests | All in-flight on that replica error |
| Scheduler crash | Process supervisor + heartbeat | Standby scheduler takes over from persisted queue state; in-flight requests served by engines independently until completion | <5s admission gap |
| Prefix cache corruption (one block bad) | Per-block checksum on eviction read | Invalidate the block, force recompute for affected sequences | TTFT spike for those sequences only |
| Disaggregated KV transfer fails | Transfer timeout (50ms) | Fall back to aggregated path on a single engine: prefill+decode locally | TTFT inflation for that request |
| Cold start during traffic spike | Autoscaler signal | Admission control returns 503 with `Retry-After`; existing replicas drain queue | Some users see 503; clients with retry succeed |

**Specific scenario: GPU loss in TP=2 group.** TP groups are atomic — a 2-GPU replica with one dead GPU is dead. There's no graceful degradation at the replica level (you can't shrink TP from 2 to 1 mid-flight; the weight sharding is wrong). The replica's in-flight requests' KV caches are gone with that GPU. Decode mid-flight responses cannot be replayed from where they stopped — that requires reconstructing KV from the prompt + tokens-emitted-so-far, which means a full prefill of (prompt + emitted tokens). Implementation: on replica failure, the gateway retries the request from scratch on a different replica, with the original prompt + an instruction not to repeat already-streamed tokens. Some duplicate tokens leak to the user; the alternative is the connection just dying. Tradeoff documented; product decides which is acceptable.

**Invariants the system preserves:** **[STAFF SIGNAL: invariant-based thinking]**
- TP rank consistency: all TP ranks of a replica process the same logical step at the same time. Loss of any rank invalidates the whole replica.
- KV referential integrity: a sequence's block table never points to a freed page (refcount-protected).
- Tenant KV isolation: tenant A's prefix tree cannot reference pages owned by tenant B's tree.
- Streaming monotonicity: once a token is sent to a client, it cannot be retracted (no rollback after speculative misverification — rollback happens before send).

---

## 7. Observability

The right set of metrics, broken down by stakeholder:

**Per-request trace:**
```
queue_time → admission_time → prefill_start → first_token (TTFT)
  → per_token_latency × N → finish_time
```

**Distributions to alarm on:**
- TTFT p50/p95/p99/p99.9 broken out by prompt-length bucket and tenant class
- ITL (inter-token latency) p50/p95/p99 — the metric most predictive of "feels laggy"
- Tokens/sec aggregate per replica, per pool
- KV occupancy % per replica (alarm at 90%; eviction storm imminent at 95%)
- Prefix cache hit rate per tenant (alarm on sudden drop = something invalidated all caches)
- Preemption rate (high = scheduling thrash)
- Per-iteration batch size distribution (mode tells you actual operating regime)

**Stakeholder-specific:**
- **Product:** TTFT p99, ITL p99, success rate. The numbers users feel.
- **SRE/Infra:** GPU HBM bandwidth utilization (NOT GPU compute utilization — compute can be 95% on a stalled decode; bandwidth is the truth), throughput per replica, queue depth, preemption rate.
- **Finance:** tokens generated per dollar per hour, by GPU type. Per-tenant token consumption vs billing.
- **ML/Quality:** judge-eval score on sampled production responses, by quantization tier. Per-layer activation drift.

**Anti-pattern to avoid:** "we monitor GPU utilization." `nvidia-smi` GPU utilization is a percentage of time the GPU is doing *anything*, including reading memory while compute units are idle. A bandwidth-bound decode kernel pegs `nvidia-smi` at 99% while running at <10% of peak FLOPs — the metric tells you nothing useful for inference. **DCGM HBM bandwidth and FP8 tensor pipe utilization** are the metrics that matter. If your dashboard is GPU-util%, you don't have an inference observability story.

---

## 8. Cold Start and Autoscaling

**Model load cost:**
- 70B FP8 = 70 GB
- Local NVMe @ 5 GB/s: 14 seconds raw
- With sharded parallel load across TP ranks: 7 seconds
- Plus CUDA context init + kernel autotune: another 15-30 seconds on a cold worker
- Total cold start: **30-45 seconds** before a replica accepts traffic

This is too slow for spike absorption. Solutions:

1. **Hot spare replicas** — keep 10-15% headroom of warm replicas always ready. Costs money; saves SLO.
2. **Predictive autoscaling** — scale on rate-of-change of queue depth, not absolute queue depth. Triggers ~30s before SLO breach so the new replica is warm in time.
3. **Preloaded model artifacts on local NVMe** — never fetch from object storage during scale-up. Pre-stage on every node at deploy time.
4. **Process-level warm start** — the model worker process can be kept warm with weights loaded but no traffic, just consuming HBM. Faster than full restart.
5. **Multi-LoRA with shared base** — for adapter-based fine-tunes, base model loaded once per node, adapters (50-200 MB each) swap per request in <100ms. Critical for any workload where many tenants have their own fine-tunes.

**Autoscaling policy:** scale on a composite signal: `queue_depth × p99_TTFT_violation_rate × prefill_capacity_utilization`. Single-metric autoscaling on QPS or GPU util will undershoot or overshoot.

---

## 9. Recent Developments Worth Referencing

**[STAFF SIGNAL: modern awareness]**

- **vLLM v1** rebuilt the scheduler around chunked prefill + paged KV as the unified path; deprecated the old static-batching code paths. The architectural simplification is the news.
- **SGLang's RadixAttention** generalized prefix caching from "system prompt cache" to a full prefix tree with sharing across requests, which became the template for production prompt caching products.
- **NVIDIA Dynamo** is the productized disaggregated-prefill-decode + KV-aware routing layer; effectively Mooncake for the H100/B200 ecosystem. Worth understanding as the reference implementation, even if you build your own.
- **TensorRT-LLM** still wins single-replica throughput on NVIDIA hardware via aggressive kernel fusion; the operational cost is rebuild-per-model. Use for stable long-lived deployments, not for fast-moving research models.
- **FlashAttention-3** added FP8 attention support and pipelined producer-consumer warps, recovering the 1.5-2× throughput gap between BF16 attention and what the H100 tensor cores can theoretically deliver.
- **EAGLE-2/3** raised speculative-decoding acceptance into the 0.8 range with relatively cheap draft-model training; this is the first speculative variant where the speedup is large enough to be product-relevant for chat.
- **DeepSeek-V3 inference techniques** (MLA — multi-head latent attention — for KV compression, FP8 mixed-precision) showed that sub-100KB-per-token KV is achievable with architectural changes; relevant when picking the model architecture, not when serving an existing one, but moving the field.
- **Mooncake / DistServe / Splitwise** independently published the disaggregation case with measured numbers: typical 1.5-2× throughput at fixed SLO vs aggregated.
- **B200 + MXFP4 numerics:** Blockwise-scaled FP4 with hardware support. The throughput is real (~2× FP8); the calibration story is harder than FP8 and quality eval is non-negotiable. Most production deployments are still FP8 on H100.

---

## 10. Tradeoffs Taken and What Would Force Redesign

**Tradeoffs taken:**
- TP=2 H100 + FP8 chosen over TP=4 H100 + BF16: 2× throughput per replica, ~0.3% quality regression accepted with monitoring.
- Disaggregation chosen over aggregated-only: ~2× peak throughput, ~10% TTFT inflation on short-prompt path, real operational complexity.
- EAGLE-2 speculative accepted: 3× decode latency win, draft-model maintenance burden across releases.
- Three-tier hardware fleet (H100/A100/B200): higher operational complexity, 3-4× cost spread captured.
- Strict tenant KV isolation: foregoing a 5-15% global cache hit rate for security/correctness.

**What would force redesign:**
- **Model becomes a 405B MoE.** Parallelism shifts to TP+EP across nodes; all-to-all becomes the bottleneck; expert load-balancing becomes a scheduler concern. Disaggregation may become more important (prefill all-to-all is heavier than decode's). Hardware ratio shifts toward B200 for decode (192GB lets you hold more experts active).
- **Workload becomes 95% batch.** Disaggregation overhead doesn't pay off; collapse to aggregated B200 batch path with very large batches. Prefix caching less valuable.
- **Multi-modal becomes in scope.** Vision encoder front-end (separate compute pool, similar prefill/decode logic for the encoder), KV cache must include image-patch embeddings, prompt cache key changes to include image hashes. Adds at least one new pool to the topology.
- **Tool calling / agentic workloads.** Output distribution changes (tool-call tokens are high-entropy), speculative decoding becomes a loss not a win, request structure becomes "many short turns" rather than "one long turn" — completely different SLO shape. May need per-step (not per-request) priority.
- **Long context (1M tokens) becomes primary.** Ring attention / sequence parallelism becomes mandatory, prefill becomes a multi-node job, KV cache size becomes the binding constraint over weights.
- **Latency budget tightens to p99 TTFT < 100ms.** Speculative decoding alone won't get there. Disaggregation transfer hops have to go. Either smaller model + larger TP, or fundamentally different (cached-everything, RAG-only) architecture.

---

## 11. What I'd Push Back on in the Original Prompt

**[STAFF SIGNAL: saying no]**

1. **"10K concurrent users" without QPS or prompt distribution.** I gave a defensible interpretation, but in a real design discussion I'd refuse to commit to a sizing number until I had p50/p95/p99 prompt length, p50/p95/p99 output length, and the diurnal QPS curve. The capacity answer varies 10× across plausible interpretations.
2. **"Mix of interactive and batch on the same hardware" framed as a deployment question.** It's a scheduling and pricing question. Conflating "shared hardware" with "shared SLO" is the error. Batch should be priced separately, SLO'd separately, and accounted separately even when running on the same GPU as interactive — otherwise the interactive cost-per-token is artificially inflated by batch overhead in finance reports.
3. **"Multiple GPU types available."** This is a constraint, but it should be a decision. If the company has a free choice, the operational cost of three GPU tiers is non-trivial. I'd push for a two-tier fleet (H100 + B200, A100 only as legacy) unless the workload actually has long-tail cost-sensitive traffic that justifies the third tier. Heterogeneity is a tax you pay for cost optimization; it should be earned, not assumed.
4. **No SLO definition.** I made one up. The biggest failure of this prompt is that the SLO isn't specified. p50? p99? p99.9? Per tenant or global? TTFT only or ITL too? "Reasonable latency" is not an engineering input.
5. **Implicit assumption that one model serves all requests.** In production you also have model variants per tenant (fine-tunes), guardrail models in the request path, retrieval embedding models, and reranker models. The serving layer needs to handle the broader ML graph, not just one 70B model. Multi-LoRA covers some of this; the rest is a routing/composition concern that this prompt elides entirely.

If I had only one of these to push on, it's #4: I'd refuse to commit to capacity or topology without a written SLO, because every decision in this design — disaggregation, batch ceiling, speculative on/off, cold-start headroom — falls out of the SLO numbers. Designing without it is fitting curves to imaginary data.

---

*Word count: ~5,800. Staff signals tagged: 14 of 16 listed.*