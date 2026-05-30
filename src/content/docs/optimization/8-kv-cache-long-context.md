---
title: "KV Cache Optimization for Long-Context LLM Inference"
description: "A staff-level guide to KV cache optimization: memory math, MQA, GQA, MLA, paging, prefix caching, eviction, compression, disaggregation, and long-context serving reality."
---

# KV Cache Optimization for Long-Context LLM Inference

The KV cache is the memory LLMs keep during autoregressive generation so they do not recompute keys and values for previous tokens. It is one of the main reasons modern LLM serving is a memory-systems problem, not just a matrix-multiplication problem.

For long context, multi-turn chat, RAG, agents, and high-concurrency serving, the KV cache often determines:

- How many users fit on one GPU.
- Whether long-context requests cause admission failures.
- Whether p99 latency explodes.
- Whether prefill/decode can be disaggregated.
- Whether a model is economically servable.

The staff-level framing:

> KV cache optimization is capacity planning for attention state. If you cannot budget the cache, you cannot serve long-context LLMs predictably.

---

## 1. The KV Cache Formula

For a decoder-only transformer, each generated token attends to previous tokens. During prefill, the model computes keys and values for the prompt. During decode, it appends one new key and value per generated token.

Approximate KV cache memory:

$$
\text{KV bytes} =
2 \cdot B \cdot L \cdot N_L \cdot H_{kv} \cdot d_h \cdot b
$$

where:

- $2$ is for keys and values.
- $B$ is batch/concurrency.
- $L$ is cached sequence length.
- $N_L$ is number of layers.
- $H_{kv}$ is number of KV heads.
- $d_h$ is head dimension.
- $b$ is bytes per cache element.

This formula explains most KV-cache design decisions.

```text
More users      -> B increases
Longer context  -> L increases
More layers     -> N_L increases
MHA -> GQA/MQA  -> H_kv decreases
FP16 -> FP8 KV  -> b decreases
MLA             -> cached representation changes
```

If you are interviewing for model optimization, write this formula. It shows you understand why long-context serving gets expensive.

---

## 2. Prefill vs Decode

KV cache matters differently in prefill and decode.

Prefill:

- Processes the full input prompt.
- Computes initial KV cache.
- Usually more compute-heavy.
- Long prompts can monopolize the GPU.

Decode:

- Generates one token at a time.
- Reads the KV cache for all previous tokens.
- Often memory-bandwidth-bound.
- Cache size directly affects concurrency.

```text
Request lifecycle:

Prompt tokens
     |
     v
Prefill: compute hidden states + build KV cache
     |
     v
Decode step 1: read KV cache + append new KV
Decode step 2: read KV cache + append new KV
Decode step 3: read KV cache + append new KV
```

Optimizing prefill without managing KV cache may improve TTFT but not concurrency. Optimizing KV cache can increase concurrency and reduce decode stalls.

---

## 3. KV Head Reduction: MQA and GQA

Standard multi-head attention stores keys and values per attention head:

$$
H_{kv} = H_q
$$

Multi-Query Attention stores one KV head:

$$
H_{kv} = 1
$$

Grouped-Query Attention stores several KV groups:

$$
1 < H_{kv} < H_q
$$

KV cache savings:

$$
\text{savings factor} = \frac{H_q}{H_{kv}}
$$

Example:

```text
Query heads: 64
MHA KV heads: 64  -> baseline
GQA KV heads: 8   -> 8x smaller KV
MQA KV heads: 1   -> 64x smaller KV
```

This is why GQA is now a mainstream serving architecture. It reduces cache pressure while preserving more quality than MQA.

---

## 4. MLA: Cache a Latent Instead of Full KV

Multi-head Latent Attention changes the cache representation. Instead of storing full expanded keys and values per token, MLA stores a compact latent vector and reconstructs attention quantities from it.

```text
MHA / GQA:
  token -> K cache + V cache

MLA:
  token -> compressed latent cache -> reconstruct K/V-like projections
```

Simplified:

$$
c_t = W_D h_t
$$

where $c_t$ is the latent cache for token $t$. Later:

$$
k_t, v_t = f(c_t)
$$

DeepSeek-V2 and DeepSeek-V3 made MLA prominent, and DeepSeek-V3.2 continued the efficiency-focused lineage while adding sparse attention for long context.

MLA is powerful because it attacks the KV cache at the architecture level. But it is not a serving-only patch. It requires:

- A model trained or converted for MLA.
- Runtime support for latent cache layout.
- Kernels that handle reconstruction efficiently.
- Evaluation for attention/recall behavior.

---

## 5. Paged KV Cache

PagedAttention, popularized by vLLM, treats the KV cache like virtual memory. Instead of allocating one large contiguous block per sequence, it divides KV cache into fixed-size blocks.

```text
Logical sequence:
  token 0 ... token 8191

Physical KV blocks:
  block 17 -> tokens 0-15
  block 42 -> tokens 16-31
  block 03 -> tokens 32-47
  ...
```

Why paging helps:

- Reduces fragmentation.
- Supports variable-length requests.
- Allows cache blocks to be shared.
- Improves memory utilization.
- Enables continuous batching with many sequence lengths.

Without paging, a serving engine may reserve too much contiguous memory for each request. Long-tail prompt lengths waste memory. With paging, memory can be allocated incrementally.

Production point:

> Paged KV does not make attention cheaper mathematically. It makes GPU memory usable under real traffic.

---

## 6. Prefix and Prompt Caching

Prefix caching reuses KV cache for repeated prompt prefixes:

- System prompts.
- Tool schemas.
- Few-shot examples.
- RAG templates.
- Multi-turn conversation history.
- Agent instructions.

If the prefix has length $L_p$, and the request has total input length $L$, a cache hit reduces prefill work to roughly:

$$
L - L_p
$$

instead of:

$$
L
$$

```text
Shared prefix:
  [system prompt][tools][policy][user query]

Cache hit:
  reuse KV for [system prompt][tools][policy]
  prefill only [user query]
```

Prefix caching improves TTFT, GPU utilization, and cost, but requires routing:

- Round-robin routing destroys cache locality.
- Cache-aware routing improves hit rate.
- Multi-tenant isolation must be respected.
- Model/version/template changes invalidate cache keys.

Modern production APIs increasingly expose or exploit prompt caching because it is one of the cleanest long-prompt optimizations.

---

## 7. KV Cache Eviction

For long-running sessions, KV cache grows. Eviction policies decide what to keep.

Options:

- Keep full cache until request ends.
- Sliding-window eviction.
- Keep attention sinks.
- Keep system prompt and recent turns.
- Summarize older context.
- Retrieve older context on demand.
- Tier KV across HBM, CPU DRAM, NVMe, or remote storage.

Eviction is dangerous because removing a token's KV means the model can no longer attend to it directly.

```text
Session memory:

Keep:
  system prompt
  tool schema
  recent turns
  attention sinks

Evict/summarize:
  stale conversation
  retrieved docs no longer needed
  low-attention spans
```

Evaluation must include:

- Long-context recall.
- Multi-turn consistency.
- Tool-use state.
- RAG groundedness.
- Safety instructions.

Evicting the wrong tokens can create correctness bugs that look like model hallucination.

---

## 8. KV Compression and Quantization

KV cache can be compressed by:

- Lower precision KV cache.
- Per-channel scaling.
- Low-rank compression.
- Token pruning.
- Latent cache, as in MLA.
- Sliding-window retention.
- Sparse attention selection.

If KV uses BF16:

$$
b = 2
$$

If KV uses FP8:

$$
b = 1
$$

In the simple formula, that halves KV memory:

$$
\text{KV bytes}_{FP8} \approx \frac{1}{2}\text{KV bytes}_{BF16}
$$

But KV quantization can affect attention scores and retrieval. Small errors in keys can change attention weights. Small errors in values can change the resulting hidden state.

KV compression must be evaluated differently from weight compression:

- Long-context recall.
- Attention distribution KL.
- Needle and multi-needle tests.
- RAG answer grounding.
- Codebase navigation.
- Output stability over long decode.

---

## 9. Disaggregated Prefill and Decode

Prefill and decode stress hardware differently. Some systems split them into separate GPU pools:

```text
Prefill pool:
  handles prompt processing
  computes initial KV cache

KV transfer:
  sends cache to decode pool

Decode pool:
  serves token generation
  manages long-lived KV
```

Benefits:

- Prefill bursts do not block decode.
- Pools can be scaled separately.
- Hardware can be specialized.
- Scheduling becomes more controllable.

Costs:

- KV transfer latency.
- RDMA/NVLink/EFA complexity.
- More routing and orchestration.
- Cache ownership and lifecycle complexity.

Disaggregation is not the first optimization for every system. It pays off when prompt length variance and decode concurrency fight each other.

---

## 10. Long-Context Production Reality

From 2025 into 2026, long-context optimization moved from research to production pressure:

- GQA is common because full MHA KV cache is too expensive.
- MLA became a serious architecture-level cache reduction approach through DeepSeek-V2/V3 and later DeepSeek releases.
- DeepSeek-V3.2-Exp and V3.2 made sparse attention a visible late-2025 open-model example for long-context cost control.
- Serving engines like vLLM and SGLang emphasize paged KV, prefix caching, continuous batching, and cache-aware scheduling.
- Provider APIs increasingly expose prompt caching or implicit cache reuse because repeated context dominates many agent/RAG workloads.

The production reality:

> Long context is not just a model feature. It is a memory allocation, routing, caching, and scheduling problem.

---

## 11. Failure Modes

### Cache math is ignored

The model supports 128K context, but concurrency collapses because KV memory was not budgeted.

### Prefix cache hit rate is low

The cache exists, but routing sends repeated prompts to different workers.

### Eviction breaks correctness

Important instructions or facts are evicted and the model appears to hallucinate.

### KV quantization hurts retrieval

Aggregate benchmarks pass, but long-context tasks regress.

### Disaggregation adds more latency than it saves

KV transfer dominates for small prompts or poor network topology.

### Cache invalidation is wrong

Changed system prompts, tokenizer versions, or model weights reuse stale cache entries.

---

## 12. The Staff Engineer Summary

KV cache optimization is the core memory discipline of LLM serving.

The checklist:

- Calculate KV cache memory explicitly.
- Separate prefill compute from decode memory bandwidth.
- Use GQA/MQA/MLA to reduce KV head or latent size.
- Use paged KV to avoid fragmentation.
- Use prefix caching with cache-aware routing.
- Evaluate eviction and compression on long-context tasks.
- Treat disaggregated prefill/decode as an architecture decision, not a flag.
- Benchmark under real concurrency and prompt-length distributions.

The interview answer:

> KV cache is often the serving bottleneck for long-context LLMs. MQA/GQA reduce KV heads, MLA changes the cached representation, paging improves memory utilization, prefix caching avoids repeated prefill, and eviction/compression control long sessions. The right answer depends on prompt length distribution, concurrency, and quality risk.

