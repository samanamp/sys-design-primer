---
title: "LLM Serving Optimization: Batching, Scheduling, Caching, and SLOs"
description: "A staff-level guide to LLM serving optimization: continuous batching, paged attention, prefix caching, disaggregated prefill/decode, admission control, autoscaling, and throughput-latency tradeoffs."
---

# LLM Serving Optimization: Batching, Scheduling, Caching, and SLOs

LLM serving optimization is where model architecture meets production traffic. A model can be efficient in isolation and still perform badly under real request distributions, long prompts, multi-turn sessions, tool calls, bursty traffic, and strict latency SLOs.

The central serving problem:

> Maximize useful tokens per GPU-second while meeting latency and quality constraints.

The word "useful" matters. Raw throughput is not enough. Tokens generated after the user has already timed out are not useful. A system that maximizes tokens/sec while violating p99 SLO is not optimized.

---

## 1. Serving Metrics

Track at least:

- TTFT: time to first token.
- ITL: inter-token latency.
- E2E latency.
- Tokens/sec/GPU.
- Requests/sec.
- Queue time.
- Prefill time.
- Decode time.
- KV cache memory.
- Prefix cache hit rate.
- Goodput: work completed within SLO.
- Cost per successful request.

Goodput:

$$
\text{goodput} =
\frac{\text{tokens produced within SLO}}
{\text{GPU seconds}}
$$

This is better than raw throughput for production optimization.

---

## 2. Continuous Batching

Static batching waits for a fixed batch, runs it, then returns results. LLM traffic is variable, so static batching wastes capacity.

Continuous batching lets requests enter and leave the batch dynamically at token boundaries.

```text
Decode step 1: A B C D
Decode step 2: A B C D E   (E joins)
Decode step 3: A C D E     (B finishes)
Decode step 4: A C E F     (F joins, D finishes)
```

Benefits:

- Higher GPU utilization.
- Better throughput.
- Less idle time.
- Natural fit for streaming generation.

Costs:

- More complex scheduler.
- KV cache management.
- Fairness issues.
- Shape variability.
- Harder CUDA graph capture.

Continuous batching is table stakes for serious LLM serving.

---

## 3. Chunked Prefill

Long prefills can block decode. Chunked prefill splits prompt processing into smaller chunks and interleaves them with decode work.

```text
Bad:
  long prefill monopolizes GPU
  decode requests wait

Better:
  prefill chunk
  decode step
  prefill chunk
  decode step
```

Chunked prefill can slightly worsen the long request's own TTFT but improve fleet p99 by preventing one giant prompt from blocking everyone else.

This is a staff-level tradeoff:

> Optimize the system SLO, not one request's isolated runtime.

---

## 4. Paged KV Cache

Paged KV cache divides sequence cache into blocks. This reduces fragmentation and supports variable-length requests.

```text
Request A: blocks 1, 8, 9
Request B: blocks 2, 3
Request C: blocks 4, 10, 11, 12
```

Benefits:

- Better memory utilization.
- More concurrent sequences.
- Efficient continuous batching.
- Easier prefix sharing.

PagedAttention made this idea mainstream in vLLM-style serving.

---

## 5. Prefix Caching and Cache-Aware Routing

Repeated prefixes are common:

- System prompts.
- Tool definitions.
- Policy text.
- Few-shot examples.
- Agent scaffolds.
- Multi-turn history.

Prefix caching reuses KV for those prefixes. The scheduler must route requests to workers that have the relevant cache.

```text
Round-robin routing:
  low cache hit rate

Cache-aware routing:
  route to worker with prefix KV
```

Failure modes:

- Tokenizer mismatch.
- Template changes.
- Model version mismatch.
- Tenant isolation bugs.
- Cache memory crowding out active requests.

Prompt caching is one of the highest-ROI optimizations for agent/RAG workloads.

---

## 6. Admission Control

If every request is admitted, overload turns into tail latency collapse.

Admission control decides:

- Admit now.
- Queue.
- Route elsewhere.
- Degrade.
- Reject.

Signals:

- Queue depth.
- KV cache free blocks.
- Estimated prefill cost.
- Current decode load.
- Tenant priority.
- SLO deadline.

```text
Incoming request
      |
      v
Estimate cost: prompt length, max tokens, cache hit
      |
      +-- enough capacity -> admit
      +-- near capacity   -> queue / lower priority
      +-- overloaded      -> reject or fallback
```

Admission control is not failure. It is how the system preserves SLOs under load.

---

## 7. Disaggregated Prefill and Decode

Prefill and decode stress hardware differently:

- Prefill: compute-heavy, large prompt matrix work.
- Decode: memory-bandwidth-heavy, KV reads, small steps.

Disaggregated serving uses separate pools:

```text
Router
  |
  +-- prefill workers compute prompt + KV
          |
          v
      KV transfer
          |
          v
      decode workers stream tokens
```

Benefits:

- Independent scaling.
- Better isolation.
- More predictable decode latency.
- Hardware specialization.

Costs:

- KV transfer overhead.
- More complex routing.
- Failure handling.
- Cache ownership.
- Network dependency.

Disaggregation is powerful for high-scale long-context workloads, but overkill for small deployments.

---

## 8. Speculative Decoding in Serving

Speculative decoding changes the scheduler because requests advance by variable numbers of tokens.

Serving engine must manage:

- Draft/proposer execution.
- Verification pass.
- Variable accepted tokens.
- KV updates for accepted tokens.
- Streaming cadence.
- Batch fairness.

Speculative decoding is production-useful when decode dominates and acceptance rate is high. It is not a primary long-prompt TTFT fix.

---

## 9. Autoscaling

LLM autoscaling is harder than stateless web autoscaling because GPU replicas have warm state:

- Loaded weights.
- KV cache.
- Prefix cache.
- CUDA graphs.
- Engine warmup.

Scale decisions should use:

- Queue time.
- Goodput.
- KV cache pressure.
- Prefix cache hit rate.
- Tokens/sec utilization.
- Prompt length mix.
- Decode length mix.

Scaling only on GPU utilization can be misleading. A server can show high utilization while producing bad p99 latency, or lower utilization while protecting SLOs.

---

## 10. Production Reality

As of 2025-2026, serious serving stacks are converging on:

- Continuous batching.
- Paged KV.
- Prefix/prompt caching.
- Chunked prefill.
- Speculative decoding options.
- Structured output support.
- Quantization integration.
- Tensor/expert parallel serving.
- Increasing interest in disaggregated prefill/decode.
- Model-specific kernels for MLA, sparse attention, and MoE.

Engines such as vLLM, SGLang, TensorRT-LLM, and related vendor stacks compete on these details.

The model architecture increasingly dictates serving architecture. MLA, DeepSeek Sparse Attention, MoE, MTP, and FP8 are not isolated model features; they require serving support.

---

## 11. Failure Modes

### Optimizing throughput while p99 gets worse

Large batches improve tokens/sec but increase queueing or TTFT.

### Prefix cache exists but routing ignores it

Cache hit rate remains low.

### KV cache OOM under long-context traffic

The model supports the context length, but concurrency does not.

### Chunked prefill is tuned badly

Chunks are too small, causing overhead, or too large, causing stalls.

### Admission control is absent

Overload turns into timeout storms.

### Speculation hurts under high temperature

Acceptance rate falls and draft overhead is wasted.

### Autoscaling is too slow

New replicas take too long to load weights and warm caches.

---

## 12. Staff Checklist

For a serving optimization program:

- Break latency into queue, tokenize, prefill, decode, network.
- Track p50/p95/p99 by prompt length and output length.
- Optimize goodput, not raw throughput.
- Use continuous batching and paged KV.
- Add prefix caching with cache-aware routing.
- Use chunked prefill for long-prompt tails.
- Add admission control before overload.
- Consider disaggregation when prefill and decode interfere.
- Use speculative decoding only when decode dominates.
- Benchmark under production-like traffic.

The interview answer:

> LLM serving optimization is scheduler and memory engineering around an expensive model. Continuous batching keeps GPUs busy, paged KV makes memory usable, prefix caching avoids repeated prefill, chunked prefill protects decode latency, admission control protects SLOs, and disaggregation helps when prefill and decode need different scaling.

