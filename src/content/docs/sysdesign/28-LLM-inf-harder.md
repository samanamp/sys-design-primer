---
title: LLM Inference Serving at Scale (MoE+1M prompt)
description: LLM Inference Serving at Scale
---

# Companion: 1M-Token Serving and Large MoE Variants

This doc treats two design pivots on the original 70B-dense, 8K-prompt-mean question. Each is large enough that "what changes" is the wrong frame — these are *different* serving systems that share some primitives. Where assumptions from the original carry, I name them; where they break, I rebuild.

---

# Part 1: Serving 1M-Token Requests

## 1.1 Reframing — This Is a Different Product, Not a Bigger Version of the Same One

A 1M-token request is not "the long-context tail of the same workload." It's a different product with a different SLO shape, different hardware needs, different concurrency profile, and different cost model. **[STAFF SIGNAL: prefill-decode reframing]**

Three things break compared to the 8K-mean baseline:

1. **Prefill compute is dominated by attention, not weights.** At 1M context the attention compute (O(n²)) is 20× larger than the weight matmuls (O(n)). FlashAttention helps memory IO, not FLOP count. Ring/sequence/context parallelism becomes mandatory, not optional.
2. **KV cache exceeds any single GPU.** 1M × 320 KiB (BF16, GQA) = 305 GiB per request. FP8 KV halves it to 152 GiB — still larger than an H100. Even B200 (192 GB) holds exactly one request's KV with a few headroom GB. Concurrency per replica goes from ~110 (in the original) to ~1.
3. **TTFT moves from milliseconds to tens of seconds.** Even with maximal parallelism, prefilling 1M tokens at 70B is a multi-second job. The interactive SLO (p99 TTFT ≤ 500ms) is physically impossible. The product needs a different SLO contract: streaming "thinking..." indicators, prefill progress events, or async submission.

**Pushback before I commit anything:** **[STAFF SIGNAL: saying no]** the right first question is whether this is genuinely long-context. ~80% of "1M token" use cases are RAG problems where the user has a corpus and is asking questions about it. RAG with 8K context windows + good retrieval beats 1M direct context on cost (100×) and quality (retrieval reduces noise). The cases where 1M direct is genuinely the right answer: cross-document synthesis where the chunking boundary destroys structure (whole-codebase refactoring, full-book translation with consistency requirements, multi-document legal analysis). Outside those, I'd push the architectural decision back to product before designing.

Committing for the rest of this section: this is a real long-context product. A second-tier offering at premium pricing, served by a dedicated long-context pool, distinct from the main interactive cluster.

## 1.2 Capacity Math — The Numbers That Drive Everything

**[STAFF SIGNAL: capacity math]**

Same model assumptions as the original (70B dense, GQA 8KV/64Q heads, 128 head_dim, 80 layers).

### KV cache at 1M tokens
```
BF16 KV:  1,000,000 × 320 KiB = 305 GiB ≈ 328 GB
FP8 KV:   1,000,000 × 160 KiB = 152 GiB ≈ 164 GB
```
Neither fits in an H100 (80 GB). Only B200 (192 GB) holds the FP8 case for one request. To put one request's KV in HBM with weights, you need:
- Either: TP across enough GPUs that aggregate HBM holds weights + KV
- Or: KV sharding via context parallelism (KV pages distributed across CP ranks)
- Or: KV offload to DRAM/NVMe (with bandwidth penalty)

For TP=8 H100 (640 GB aggregate): weights at FP8 = 70 GB, FP8 KV for 1M = 152 GB → 222 GB used, 418 GB free. One request fits comfortably. Concurrency: ~3 simultaneous 1M-token requests per TP=8 replica, before activations and overhead bite.

### Prefill compute decomposition (this is the surprise)
```
Weight matmuls (linear cost in n):
  2 × 70B × 1,000,000 = 140 × 10¹⁵ FLOPs = 140 PFLOPs

Attention compute (quadratic cost in n²):
  Per layer, both QK^T and AV:
    2 × 2 × n_heads × seq² × head_dim
    = 2 × 2 × 64 × 10¹² × 128
    = 32.8 PFLOPs per layer
  All 80 layers: 2,624 PFLOPs

Ratio: attention is ~19× the weight matmul cost
Total: ~2,764 PFLOPs
```

GQA helps the KV memory footprint but **does not reduce attention compute** — Q is still 64 heads attending to (replicated) KV. The math is brutal.

**Time to prefill on TP=8 H100 (FP8, ~7.5 PFLOPs/s aggregate at realistic MFU):**
```
2,764 / 7.5 = 369 seconds ≈ 6.1 minutes per 1M-token prefill
```
Six minutes. This is unworkable. Two ways out:

1. **Add context parallelism** (CP): split the sequence dimension across more GPUs. Each rank holds a chunk of tokens and uses ring-attention to pass KV around the ring. CP=8 stacked on TP=8 = 64 GPUs for one prefill, prefill time drops to ~46s. CP=16 on TP=8 = 128 GPUs, ~23s. Communication cost grows.
2. **Architectural change**: sliding window + attention sinks (StreamingLLM-style), MLA (DeepSeek), linear attention. Reduces attention compute from O(n²) to O(n × W) where W is window size. With W=8K the compute drop is 125× — back into the seconds.

I commit to CP=8 + TP=8 as the standard config (64 GPUs per long-context request). Sliding-window architectures are a model-side decision, not a serving decision; if the model has it, the serving picks the cheaper path.

### Decode at 1M context (this also breaks)
```
Per decode step, single user, TP=2 H100 with FP8 KV:
  Weights read: 70 GB / 6.7 TB/s = 10.4 ms
  KV read: 152 GB / 6.7 TB/s = 22.7 ms
  Total: 33 ms/tok (~30 tok/s)

At TP=8: weights+KV bandwidth = 26.8 TB/s
  Total: 8.3 ms/tok (~120 tok/s)
```
Decode at 1M context is **KV-bound, not weight-bound** — the inversion of the short-context regime. Doubling FLOPs gets you nothing; doubling bandwidth halves decode time.

**Decode at 1M context is essentially single-batch.** At batch=8, KV traffic is 8 × 152 GB = 1.2 TB per token. At 26.8 TB/s, that's 45 ms/token aggregate — useable for 8 concurrent users at ~22 tok/s each, but the per-replica throughput is 175 tok/s total. Compare to short-context decode at batch=64 hitting ~3,800 tok/s per replica. **Per-replica throughput drops 20× at long context.** This sets the cost-per-token.

## 1.3 Architecture for the Long-Context Pool

```
┌─────────────────────────────────────────────────────────┐
│   Long-Context Gateway (separate from interactive)      │
│   - Async submission API option (job ID + polling)      │
│   - Streaming with progress events for sync mode        │
│   - Aggressive prefix-cache lookup before admission     │
└────────────┬────────────────────────────────────────────┘
             │
             ▼
   ┌──────────────────────┐
   │ Long-Context         │
   │ Scheduler            │
   │ - Per-request resv:  │
   │   reserves CP+TP grp │
   │ - No mixing with     │
   │   short-context wkld │
   └─┬──────────────────┬─┘
     │                  │
     ▼                  ▼
┌────────────────┐  ┌─────────────────────┐
│ Prefill Pool   │  │ Decode Pool         │
│ TP=8 × CP=8    │  │ TP=8                │
│ = 64 GPUs/req  │  │ Single-batch decode │
│ B200 preferred │  │ H100 acceptable     │
│ Ring attention │  │ FP8 KV              │
└────┬───────────┘  └─────────────────────┘
     │  KV transfer (per-layer pipelined)
     │  At 1M FP8: 152 GB total — over NVLink-Switch
     │  fabric ≈ 340ms; pipelined per-layer ≈ ~5ms 
     │  perceived if last layer pipelined right
     └──────────────────────►
                  (decode replica)
                              ▲
                              │
                  ┌───────────┴────────────┐
                  │ Long-Context KV Store  │
                  │ DRAM tier: 2 TB/node   │
                  │ NVMe tier: 30+ TB/node │
                  │ Aggressive prefix cache│
                  └────────────────────────┘
```

Key topology decisions:

- **Prefill and decode are mandatorily disaggregated** at this scale. The hardware shape is so different (CP=8 prefill cluster vs single-replica decode) that co-location wastes everything.
- **CP groups must live within an NVLink-Switch fabric.** Ring attention's all-gather pattern across nodes over Ethernet/RDMA is the actual TTFT killer at this scale; the per-step communication is large. NVL72 (B200) or NVLink-Switch H100 nodes are ideal — 72 or 256 GPUs in one NVLink fabric.
- **Decode replicas reserve full TP groups per request.** No batch sharing across long-context requests (KV pressure won't allow it at meaningful concurrency). One TP=8 replica = 1-3 concurrent long-context users.
- **A separate gateway** because the SLO contract differs: TTFT in seconds, possible async pattern, different rate-limit math (long requests cost orders of magnitude more compute).

## 1.4 Deep Dives

### 1.4.1 Context Parallelism / Ring Attention

**[STAFF SIGNAL: parallelism precision]**

Plain TP doesn't reduce attention compute — TP shards the head dimension and the heads, so per-rank work scales the same as the unsharded compute (each rank does the same n² work over its head shard). To reduce attention compute, you need to shard the **sequence dimension**.

Context parallelism (CP) shards Q/K/V along the sequence axis across CP ranks. To compute full attention, each rank needs to see all KV. Three implementation styles:

1. **All-gather KV** (simplest): each rank computes its Q chunk's attention by gathering all KV at the start. Memory cost: each rank holds full KV (defeats the point at scale).
2. **Ring attention** (Liu et al.): KV travels around a ring. Rank R computes attention against rank R's KV, then receives KV from rank R-1 and computes against that, etc. Memory: O(seq/CP) per rank. Communication: KV blocks circulate. Pipelining hides much of the comm cost behind compute.
3. **All-to-all-based CP** (Megatron CP): all-to-all redistributes Q across ranks for each attention call. Tradeoffs differ; works better when CP=TP (combined sharding pattern).

At CP=8 with 1M tokens:
```
Per-rank seq: 125K tokens
Per-rank KV (FP8): 152 GB / 8 = 19 GB — fits comfortably
Ring step compute (one chunk): 32.8 / 64 (8×8 way sharded) = 0.5 PFLOPs/layer
Ring step communication: 19 GB over NVLink-Switch (~450 GB/s effective per rank)
                       ≈ 42 ms per ring step per layer
80 layers × 8 ring steps × 42 ms = if not pipelined: 27 seconds of pure comm
With pipelining (overlap with compute): ~3-5 seconds incremental
```

Ring attention at CP=8 over NVLink-Switch is feasible. CP=8 over RDMA-Ethernet is not — communication dominates and the prefill takes minutes anyway.

**Causal mask asymmetry:** with causal masking (autoregressive LMs), the work per ring step is unbalanced — early ranks (low sequence positions) do much less attention work than late ranks (because they only attend to earlier positions). Naive ring attention has the GPU at rank 0 idle for most of the prefill. Solutions: token zigzag distribution (each rank gets two non-contiguous chunks, one early one late, so total work balances), or DistFlash-style work redistribution. This is a real production concern; assume the implementation handles it.

### 1.4.2 KV Architecture Changes That Save You

Three architectural levers that move 1M context from "barely possible" to "tractable" — but they are *model-side* decisions:

1. **Multi-head Latent Attention (MLA)** — DeepSeek-V2/V3. Compresses the KV cache via low-rank projection: instead of storing K and V per head, store a small latent representation that all heads decompress from. Effective KV per token drops from ~320 KiB to ~70 KiB (or lower depending on latent dim). At 1M tokens that's 70 GB instead of 305 GB (BF16) — fits in a single H100 with weight headroom. Catch: MLA is intrusive in the attention kernel, requires retraining (or surgical conversion), and the kernel implementations are still maturing relative to FlashAttention.
2. **Sliding Window Attention + Attention Sinks** — Mistral, StreamingLLM. Each layer attends only to the last W tokens (W=4K typical) plus a few "sink" tokens at the start. KV traffic per decode step: only W tokens, regardless of sequence length. Attention compute per prefill step: O(n × W) instead of O(n²). At W=4K, 1M-token prefill drops from 2,764 PFLOPs of attention to ~22 PFLOPs — back to the same order of magnitude as the weight matmuls. Catch: information past W tokens must be carried in residual stream alone — quality on long-range reasoning suffers measurably. Some hybrid models alternate full-attention and sliding-window layers.
3. **Dynamic sparse attention** — many flavors (Native Sparse Attention, Quest, etc.). Heuristically pick a sparse subset of past tokens to attend to. Compute drops; quality varies by workload.

The serving system *exposes* these as different model variants; the picking happens upstream by product. From the serving design's view: **MLA changes the KV math by 4-5×, sliding window changes the compute math by 100×**. They compose.

### 1.4.3 Long-Context Decode: The Hostile Regime

**[STAFF SIGNAL: roofline reasoning]**

Decode at 1M context is dominated by KV reads. Bandwidth utilization on this workload is how you measure quality of implementation. Three ways to claw back throughput:

1. **FP8 (or lower) KV cache.** Already assumed. Halves traffic.
2. **KV deduplication / compression.** If multiple users share a 1M-token prefix (rare but happens — e.g., querying the same codebase), serve them off the same KV pages. Per-token cost amortizes. Section 1.4.4 covers this.
3. **Speculative decoding.** Even more valuable here than at short context, because the bandwidth-bound regime has even more idle compute. EAGLE-2 acceptance rates hold up at long context; the speedup is ~3× decode latency, same as short context. The cost (running the draft model + verification compute) doesn't change with context length, but the win is on the dominant cost. Net: speculative is *more* worth it at long context, not less.

Per-replica throughput at 1M context, single user, with speculative: ~360 tok/s on TP=8 H100. With 3-way concurrency: ~150 tok/s per user. Compare to ~96 tok/s single-user at 4K context — the user-perceived ITL is similar, but per-replica throughput is 20× lower than short-context, which directly sets the price.

### 1.4.4 Prefix Caching Becomes Survival

**[STAFF SIGNAL: KV sharing as product lever]**

In the original design, prefix caching was a 30-50% cost optimization. Here it's the difference between a workable product and an unworkable one.

Empirical observation: long-context users iterate. Someone querying their codebase asks 5-10 questions in a session, each with the same 800K-token prefix and a different 200-token suffix. The first request pays the 6-minute prefill; the next 9 cost ~30 seconds each (suffix prefill + decode). The amortized cost drops 10×.

Aggressive prefix-cache policy:
- **Tiered storage**: HBM (active sequence), DRAM (recent context, ~30 min), NVMe (longer retention, ~24 hours), cold blob storage (multi-day for paying users).
- **Cache key includes embedding**: tokens alone aren't enough — model version, quantization tier, KV scaling factors all enter the key.
- **Restore cost vs recompute**:
  ```
  1M FP8 KV = 152 GB
  Restore from DRAM (PCIe Gen5, ~64 GB/s): 2.4 s
  Restore from local NVMe (~12 GB/s):       12.7 s
  Recompute (full prefill at TP=8 CP=8):    23-46 s
  ```
  NVMe restore beats recompute by 2-4×. The NVMe tier is mandatory for any product where 1M-context iteration is a real use case.

- **Anthropic-style prompt caching as paid product feature**: explicit cache breakpoints in the prompt, TTL, separate pricing tier. The user is buying the right to amortize KV prefill cost across requests. Maps directly onto the storage hierarchy.

**Pushback worth making:** if a customer's traffic pattern is "submit 1M token prompt once, get one answer, never iterate," prefix caching offers nothing and the unit economics are terrible. Pricing must reflect this — flat per-token pricing makes one-shot 1M-context queries massively unprofitable. Tiered pricing or compute-time pricing for this product class.

## 1.5 Failure Modes Specific to Long Context

**[STAFF SIGNAL: failure mode precision]**

| Failure | What's different at 1M | Response |
|---|---|---|
| GPU loss in CP=8 ring | Can't continue ring without all 8 ranks; whole prefill fails | Restart prefill from scratch on healthy replica; all 6 minutes of work gone. Mitigation: per-layer KV checkpoints to NVMe so prefill can resume from layer L on replacement hardware. Worth it given duration. |
| KV transfer (prefill→decode) interrupted | 152 GB transfer, multi-second | Resumable transfer protocol. Stash partial KV at receiver; restart from layer L. NVLink-Switch failures are rare; RDMA transient errors more common. |
| Decode replica OOM mid-stream | KV pressure pushes out an active sequence | Swap the *other* sequence's KV to DRAM, not this one (penalize the lower-priority request). If only one sequence on the replica, 4xx the request — there's no other replica that has its KV. |
| Prefill > 30 min | Some pathological prompts (mostly attention-pattern collapse) | Hard timeout. Fail with a debug-friendly error. Don't waste 64 GPUs forever. |
| User cancels mid-prefill at minute 4 | 4 minutes of GPU work wasted | Cancellation must propagate through the CP ring. Stop wasting compute. Some KV may be cacheable for a future identical prompt. |

The blast radius of a single GPU failure is enormous (64 GPUs idle, multi-minute work lost). The right response is to design for it: per-layer checkpointing, replicated CP groups for paying tier, aggressive health monitoring.

## 1.6 What I'd Build Differently if Forced to Choose

If the product owner insisted on long context but I had budget for one architecture move, I'd push for **MLA + sliding-window-hybrid model** (DeepSeek-V3.2-Sparse-style architecturally) over the dense GQA Llama. Compute cost drops 100×, KV cost drops 5×, and the throughput economics finally make 1M context a viable product instead of a money-losing flagship. The serving-side wins from this dwarf any serving-side cleverness.

---

# Part 2: Large MoE Model

## 2.1 Reframing — Decode Latency Is No Longer Weight Bandwidth

**[STAFF SIGNAL: prefill-decode reframing]**

For dense models, the decode latency floor is `total_weights / aggregate_bandwidth`. For MoE, only the **active** weights are read per token — typically 5-10% of total. The bandwidth-bound floor drops 10×. Three things break in the original design's mental model:

1. **Decode latency is dominated by all-to-all communication, not HBM bandwidth.** Token routing dispatches each token to its top-K experts. The all-to-all between attention and MoE blocks is the new critical path.
2. **Total weight memory is huge — fitting the model is the design driver.** A 671B FP8 MoE = 671 GB. No 8×H100 node holds it. B200 nodes (1.5 TB) become the default, not a premium tier.
3. **Batch sizes need to be much larger for expert utilization.** With 256 experts and 8 active per token, a batch of 64 tokens distributes ~2 tokens per expert on average — terrible amortization. MoE wants batch sizes in the hundreds at least to amortize expert weight reads.

Anchoring on a DeepSeek-V3-class model: 671B total params, 37B active per token, 256 routed experts + 1 shared expert, 8 experts active per token, 61 layers, MLA attention with effective KV ~70 KiB/token. Numbers below use this; Llama-4-Maverick or Mixtral classes scale similarly with their own constants.

## 2.2 Capacity Math

**[STAFF SIGNAL: capacity math]**

### Weights
```
Total params:    671B
Active per tok:   37B (~5.5% activation rate)
FP8 weights:     671 GB total
FP4 weights:     336 GB total

Per-GPU at TP=8 H100 (640 GB total): 671 GB doesn't fit, even FP8. 
Per-GPU at TP=8 B200 (1536 GB):      671 GB fits with 865 GB free for KV
Per-GPU at EP=32 (4×8 H100, 2.5 TB): fits with EP, but cross-node all-to-all
```
**B200 single-node deployment is the cleanest topology.** Cross-node MoE runs into the all-to-all wall.

### KV (with MLA, since this matches the architecture)
```
KV per token (MLA): ~70 KiB BF16, ~35 KiB FP8
1M tokens: 70 GB BF16, 35 GB FP8 — 4-9× smaller than Llama-class

Concurrency budget on TP=8 B200 (after weights):
  865 GB available
  At 4K avg context, 35 KiB/token (FP8 MLA) = 140 MB per request
  865 GB / 140 MB ≈ 6,000 concurrent requests per replica
```
This is wildly different from the dense case. KV cache is no longer the binding constraint at typical context lengths; weights are. Concurrency is bounded by *throughput* (tokens-per-second the replica can produce), not memory.

### Decode latency (the surprising number)
**[STAFF SIGNAL: roofline reasoning]**

```
Single-user decode, TP=8 B200, FP8 active weights, all-to-all on NVLink-Switch:

Active weights read per token: 37 GB
Aggregate B200 BW (TP=8): 8 × 8 TB/s = 64 TB/s
Memory time: 37 / 64,000 = 0.58 ms

All-to-all per layer:
  Token dispatch: send routing info + activations, 1 MB per token at hidden_dim=7168
  All-to-all latency on NVL72: ~30-50 μs per call
  61 layers × 2 all-to-alls per layer (pre+post MoE) = 122 all-to-alls
  Total: 122 × 40 μs = 4.9 ms

Total per-token latency: 0.58 + 4.9 = ~5.5 ms (≈ 180 tok/s single-user)
```

**The all-to-all dominates by 10×.** Doubling memory bandwidth gets you ~5% on decode latency. Halving all-to-all latency gets you ~40%. The design priorities invert.

This is also why **single-node MoE deployment is vastly preferable to multi-node MoE.** Cross-node all-to-all over RDMA InfiniBand: latency floor ~15-25 μs/call, but bandwidth divides by per-node link count and NIC saturation. Practical: ~5-10× slower per all-to-all. A 61-layer model with ~120 all-to-alls per token, with 5× slower per call, adds 20+ ms/token. Multi-node MoE decode is a different SLO regime.

## 2.3 Architecture for the MoE Case

```
                    ┌───────────────────────────┐
                    │ Gateway / Router          │
                    │ - tenant routing          │
                    │ - prefix-cache lookup     │
                    └────────────┬──────────────┘
                                 │
                    ┌────────────┴──────────────┐
                    │ MoE-Aware Scheduler       │
                    │ - batch admission for     │
                    │   expert utilization      │
                    │ - capacity factor tuning  │
                    └─┬─────────────────────┬───┘
                      │                     │
                      ▼                     ▼
          ┌────────────────────┐   ┌──────────────────────┐
          │ Aggregated Pool    │   │ Disaggregated Pool   │
          │ Single-node B200   │   │ Prefill: B200 EP=8   │
          │ TP=8 + EP=8        │   │  (compute-heavy, OK  │
          │ All-to-all on NVL  │   │   to use less        │
          │ FP8 weights        │   │   memory headroom)   │
          │ MLA KV             │   │ Decode: B200 EP=8    │
          │ 6K+ concurrent req │   │   high-batch, FP8 KV │
          └────────────────────┘   └──────────────────────┘
                                              │
                                              ▼
                                   ┌─────────────────────┐
                                   │ Expert State Tracker│
                                   │ - load per expert   │
                                   │ - hot expert        │
                                   │   replication signal│
                                   └─────────────────────┘
```

Key choices:

- **B200 single-node (NVL72 or 8x NVLink-Switch) is the default.** TP=8 + EP=8 — every GPU holds 256/8 = 32 experts, all-to-all stays inside the NVLink fabric.
- **EP is the dominant new parallelism axis.** TP and EP compose; PP becomes interesting for very deep MoE models (DeepSeek-V3 at 61 layers is borderline) to overlap stages.
- **Prefix caching still matters.** Same as dense — system prompts, conversation history, RAG-injected context all cache the same way. MLA's compressed KV makes the caching cheaper per stored token.
- **Disaggregation is *more* attractive than dense.** The optimal EP for prefill (which is compute-bound and benefits from large batches per expert) differs from optimal EP for decode (where all-to-all latency is the cost). Prefill can run at lower EP with more replicated experts; decode runs at higher EP with sharded experts. Disaggregating lets each pool tune EP independently.

## 2.4 Deep Dives

### 2.4.1 Expert Parallelism

**[STAFF SIGNAL: parallelism precision]**

EP shards experts across GPUs. With 256 experts and EP=8, each GPU holds 32 experts. At inference time:

1. Attention output is computed per-GPU (TP-sharded as usual).
2. The **router** (a small linear layer) computes top-K expert IDs per token.
3. **Token dispatch** all-to-all: send each token to the GPU(s) that hold its top-K experts. Bandwidth is hidden_size × n_active × tokens_per_step.
4. Each GPU runs its experts on its received tokens.
5. **Token combine** all-to-all: send results back, weighted by router gates.
6. Continue to next layer.

Two all-to-alls per MoE layer × 61 MoE layers = 122 all-to-alls per token in the decode case. Each is a hard latency event — collective primitives don't pipeline across calls without explicit overlap.

**The single most important MoE optimization in serving:** overlap the all-to-all communication with the next layer's attention compute. Modern kernels (DeepEP, NVIDIA's TransformerEngine MoE kernels) do this. Without overlap, decode is comm-bound. With overlap, the all-to-all hides behind attention partially — saves 30-50% of decode latency.

**EP across nodes (when single-node doesn't fit):**
- Cross-node all-to-all over IB/RoCE: ~3-5× higher latency, ~3× lower bandwidth per-link
- **Hierarchical all-to-all**: do intra-node all-to-all first, then inter-node, then intra-node again. Reduces inter-node traffic to once per all-to-all.
- Practical limit: EP=16 across 2 nodes is sometimes worth it; EP=32+ across 4 nodes rarely is.

### 2.4.2 Expert Load Imbalance (the production problem)

In theory, the router learns to balance tokens across experts. In practice:
- **Skewed traffic**: some workloads (code-heavy, math-heavy) preferentially route to a small subset of experts. The "hot" expert's GPU saturates; other GPUs idle.
- **Routing drift**: as the model is fine-tuned or used differently, expert imbalance gets worse than the original training signal.
- **Single hot expert** can throttle the entire layer: all-to-all is bulk-synchronous, every GPU waits for the slowest.

Mitigations the serving system implements:

1. **Capacity factor**: at training and inference, allow each expert to handle up to `capacity_factor × tokens / n_experts` tokens. Excess tokens are dropped (or routed to fallback). Capacity factor 1.5 is common; tightens utilization at slight quality cost.
2. **Hot expert replication**: the most-loaded experts get replicated on multiple GPUs. Token routing splits across replicas. Costs memory; recovers utilization. Requires online monitoring of expert load distribution.
3. **Load-balanced batch admission**: the scheduler tries to admit batches whose token-to-expert distribution is balanced (expensive to compute; pragmatically uses recent history as a predictor).
4. **Dropless MoE alternatives**: keep all tokens, accept the worst-case latency. For latency-sensitive workloads, capacity-factor-based dropping is preferable; for quality-sensitive batch workloads, dropless is preferable.

This is a real production concern. The Llama-4 and DeepSeek-V3 serving codepaths have visible code for online expert balancing.

### 2.4.3 Batching Dynamics for Expert Utilization

**[STAFF SIGNAL: scheduling discipline]**

In dense, decode batch size is set by the latency-vs-throughput Pareto. In MoE, there's a third constraint: **batch size must be large enough that each expert receives a useful amount of work.**

Math: with 256 experts, top-8 routing, batch size B tokens-per-step:
```
Tokens routed per step = 8 × B
Expected tokens per expert = 8B / 256 = B / 32

For expert weight read to be meaningfully amortized, want
   (tokens_per_expert × hidden_size) >> expert_weight_size / bandwidth
   
If B = 64: ~2 tokens/expert. Expert weight (~600 MB FP8) read for 2 tokens.
Per-token cost: 300 MB. Same as reading weights for one token at a dense model.
  → MoE provides NO savings vs dense at low batch.

If B = 512: ~16 tokens/expert. Per-token cost: 37 MB.
  → MoE delivers its theoretical 10× advantage.

If B = 2048: ~64 tokens/expert. Per-token cost: 9 MB.
  → Diminishing returns; KV traffic now dominates.
```

**MoE wants batch sizes in the 256-1024 range to be efficient.** This pushes the scheduler in the opposite direction from interactive ITL goals. Two operational consequences:

1. **MoE rewards traffic.** At low QPS, an MoE replica is *less* efficient than a dense replica with comparable active-param count. The cost-per-token argument for MoE only holds at production-scale traffic.
2. **Batch-and-decode pools should be separated** for MoE more aggressively than for dense. Interactive ITL goals are at war with expert utilization. Solution: dedicate decode replicas to high batch size with relaxed ITL (call it ITL p99 = 80 ms instead of 50 ms), and run a small low-latency pool at smaller batch for premium-tier interactive.

### 2.4.4 Quantization for MoE — Differs from Dense

**[STAFF SIGNAL: quantization discipline]**

The dense playbook (FP8 weights, BF16 attention compute, FP8 KV) carries with one important addition: **expert weights tolerate quantization differently from each other**. Production observation from Llama-4 MoE FP8 deployment: a small fraction of experts (~2-5%) show noticeable quality degradation under FP8 and need BF16 retention. NE-style sensitivity analysis identifies them; the serving system ships with a per-expert precision map.

Per-expert quantization decisions are non-trivial because expert-level eval is harder — you need traffic that exercises each expert enough to evaluate it independently. Production workflow: run calibration on production-distributed data, score per-expert via activation MSE against BF16 reference, retain the top-N most sensitive experts in BF16. Net memory cost ~5-10% over uniform FP8; quality preserved.

**FP4 (MXFP4) on B200 for MoE:** even more attractive than dense because the larger total weight footprint benefits proportionally more. 671B at FP4 = 336 GB → fits in half a B200 node → opens up TP=4 EP=4 deployment options. Quality story is harder; per-expert sensitivity work plus block-scaled FP4 calibration is real engineering effort.

**KV: FP8 with MLA** is the standard. Compressing already-compressed KV further is tricky; MLA's latents are fragile under aggressive quantization.

### 2.4.5 Disaggregation Interaction

**[STAFF SIGNAL: rejected alternative]**

Disaggregation in dense (Section 5.2 of original) was about isolating workloads with different bottlenecks (compute-bound prefill vs bandwidth-bound decode). For MoE, the case is even stronger because:

- **Prefill in MoE** runs through the same expert structure. Token batches are big (the whole prompt is one batch). Expert utilization is excellent without scheduler tricks. Prefill compute throughput on MoE is excellent.
- **Decode** has the all-to-all problem and the small-batch expert-utilization problem.
- **Optimal EP for prefill**: lower (more memory headroom, batches are large). EP=4 with replicated experts.
- **Optimal EP for decode**: higher (each expert sharded across more GPUs to spread the all-to-all load). EP=8 or 16.

Different EP topologies for prefill and decode → mandatory disaggregation. The KV transfer between them is the same cost as dense (KV is per-attention not per-MoE) but the prefill/decode hardware ratio differs more dramatically.

**Rejected: aggregated-only MoE serving.** At meaningful traffic levels the disaggregation is worth it because the prefill/decode optimal configs diverge.

## 2.5 What Would Force a Different Design

**[STAFF SIGNAL: invariant-based thinking]**

- **Total params > 1 TB (FP8).** Single-node B200 (1.5 TB) becomes tight. Need cross-node EP. Hierarchical all-to-all becomes mandatory. Acceptable but worse cost-per-token.
- **Active params >> 50B.** The all-to-all dominance fades; bandwidth becomes the constraint again, similar to dense regime. Serving math approaches the dense-70B math.
- **No MLA-like KV compression.** Long-context MoE without KV compression is brutal because the per-replica concurrency benefit of MoE is lost — KV becomes the binding constraint again.
- **Workload skews to one expert subset.** Hot-expert replication is a band-aid; if the skew is structural, retrain the model with auxiliary load-balancing loss or rebalance the expert assignment.

## 2.6 Failure Modes Specific to MoE

**[STAFF SIGNAL: failure mode precision]**

| Failure | What's different in MoE | Response |
|---|---|---|
| One GPU in EP group dies | That GPU's experts are gone; all tokens that route there fail or fall back | Hot-spare replica with expert state synced; fallback router that drops to top-(K-1) routing temporarily for affected layers; rebuild EP group on replacement |
| Hot expert overload | One GPU's queue blows up; whole batch slows to that GPU's speed | Online detection via per-expert latency monitoring; trigger expert replication for the hot expert; if persistent, autoscale up |
| Router drift / mode collapse | Online fine-tuning or adversarial prompts cause expert imbalance | Periodic auxiliary load-balance loss check via shadow inference; if drift detected, alert; in worst case, temporarily disable affected experts |
| All-to-all hang | NCCL all-to-all has known liveness issues at scale | Per-call timeout (200 ms hard); kill-and-restart engine on timeout; lose in-flight requests on that replica |

The new failure mode that doesn't exist in dense is **routing drift**: production traffic shifts the empirical expert distribution away from training. This can develop slowly and silently. Monitoring per-expert load over time is essential.

---

# Part 3: 1M Context on a Large MoE — Brief

This is what frontier labs are actually building.

The wins compose well:
- **MLA KV math at 1M tokens**: 70 GB BF16, 35 GB FP8 — fits in a single B200's HBM. Concurrency on TP=8 B200 = several long-context requests per replica.
- **Active param math doesn't blow up** with context length; only attention compute does.
- **All-to-all per token is a fixed cost** independent of context.

The losses also compose:
- **Attention compute at 1M is still O(n²)** — needs CP/ring attention regardless of MoE. CP across the all-to-all-heavy MoE compute is more complex (you're now sharding sequence within an EP topology). Implementation cost is real.
- **Decode at 1M context, MoE**: KV bandwidth (35 GB FP8) and active weight bandwidth (37 GB) are now comparable. Decode is bound by `(KV + active_weights) / bandwidth + all-to-all`. At TP=8 B200: (35 + 37) / 64 TB/s + 5 ms all-to-all ≈ 6-7 ms/token. Workable.
- **Per-replica long-context concurrency**: ~3-5 simultaneous 1M-context requests, each at 100-150 tok/s, on a TP=8 B200 node.

Net: a 671B/37B MoE with MLA serving 1M context is roughly the same throughput-per-replica as dense 70B serving 1M context, but at 10× the parameter count and ~2× the per-replica capex. The cost-per-quality is much better because the model is much more capable.

This is roughly the design point of frontier-lab inference today.

---

*Total word count: ~5,200. Companion doc to the original; staff signals tagged at relevant decision points.*