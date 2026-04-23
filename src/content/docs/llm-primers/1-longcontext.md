---
title: Long-Context LLMs & Context Management
description: A deep-but-selective study guide for a long-context team interview. Assumes you already know standard Transformer mechanics, inference serving basics (vLLM-level), CUDA/GPU fundamentals, and quantization. Focus is on the **specific things that matter when sequences get long and agents get complicated**, not a survey.
---

A deep-but-selective study guide for a long-context team interview. Assumes you already know standard Transformer mechanics, inference serving basics (vLLM-level), CUDA/GPU fundamentals, and quantization. Focus is on the **specific things that matter when sequences get long and agents get complicated**, not a survey.

---

## 1. What a staff interviewee actually needs to know

### What gets tested in interviews

A long-context team interview typically probes five axes:

1. **Modeling intuition** — Why is long context hard? What does attention look like at 128K vs. 1M tokens? What breaks first?
2. **Systems reasoning** — Given a request arriving at 1M tokens, walk me through prefill, decode, KV management, batching, and where you bottleneck.
3. **Quality vs. nominal window** — Why does a "1M token model" fail a task at 200K? What's the difference between supporting a context length and being usable at it?
4. **Context engineering for agents** — How do you manage context across a long-running coding agent? What goes in the prompt, what gets retrieved, what gets summarized?
5. **Production tradeoffs** — Latency budgets, cost, observability, failure modes, eval.

Interviewers at long-context teams (OpenAI long-context inference, Anthropic inference/GPU perf, PI) want to hear **specific tradeoffs with numbers**, not generic survey talk. A strong answer reaches for mechanisms (KV cache bytes, HBM bandwidth, prefix hash invalidation, attention sink positions) rather than abstractions.

### What you can skip

- Memorizing every sparse attention variant from 2020–2022 (Longformer, BigBird, Performer, Linformer). Know the *taxonomy*, not the per-paper details.
- Deep state-space model internals (Mamba selective scan kernels) unless the role is specifically SSMs.
- Exhaustive benchmark catalog. Know 3–4 well (NIAH, RULER, HELMET, NoCha / LongCodeBench).
- Building a long-context model from scratch. You should know context extension *at a high level* (position interpolation, YaRN), not reproduce YaRN from memory.

### What depth is expected

Staff-level means you can reason about **the coupling between modeling decisions and systems behavior**. Example: "We use GQA with 8 KV heads, which cuts KV cache bytes by 8× vs. MHA. At 1M context for a 70B model that moves KV cache from ~320 GB to ~40 GB per request, which determines whether we need offload." You should be able to do this kind of back-of-envelope in the room.

---

## 2. Core mental model

### What "long context" actually means

A model has four different "context lengths" and they are all different:

```
+---------------------------------------------------------------+
|  Nominal context window (marketing number)                    |
|   e.g. "1M tokens"                                            |
|                                                               |
|   +--------------------------------------------+              |
|   |  Trained / effectively trained length      |              |
|   |   e.g. 256K with strong NIAH through 1M    |              |
|   |                                            |              |
|   |   +-----------------------------+          |              |
|   |   | Usable retrieval length     |          |              |
|   |   |  (NIAH / needle recall)      |         |              |
|   |   |                             |          |              |
|   |   |  +-----------------------+  |          |              |
|   |   |  | Usable reasoning len. |  |          |              |
|   |   |  |  (multi-hop, synth)   |  |          |              |
|   |   |  +-----------------------+  |          |              |
|   |   +-----------------------------+          |              |
|   +--------------------------------------------+              |
+---------------------------------------------------------------+
```

- **Nominal window**: What the API accepts. Gemini 2.5 Pro takes 1M, Claude Sonnet 4.x supports up to 1M, GPT-4.1/5 accept ~256K–1M depending on tier.
- **Effective retrieval length**: Where NIAH-style lookups stay at ≥95% accuracy. Historically degrades well before the nominal limit.
- **Effective reasoning length**: Where multi-hop reasoning, aggregation, and long-form generation stay reliable. Usually shorter still.
- **System-level effective context**: What the end-to-end product can actually use, after the cost of latency, TTFT, and quality degradation is priced in.

### Why longer is not automatically better

1. **Quality degrades with length** even when nominally supported. Models show position bias, distraction, and attention dilution.
2. **Cost scales super-linearly** (attention compute is O(n²); KV cache is O(n) memory but at scale dominates HBM).
3. **Latency explodes on prefill**. A 1M-token prefill on a 70B model can take tens of seconds.
4. **Freshness is often a bigger problem than capacity**. You can fit the codebase; you can't fit yesterday's codebase if it's different.
5. **More context = more noise**. Irrelevant content actively hurts accuracy, sometimes more than missing context does.

### Why it's both a modeling *and* a systems problem

Modeling sets the ceiling (what the architecture and training can represent); systems set the floor (what you can actually serve with acceptable cost and latency). A 1M-context model is useless if prefill takes a minute per request and your KV cache blows out HBM at batch size 1.

The "just fit more tokens" answer is shallow because it ignores:
- KV cache memory (usually the hard wall, not compute).
- Prefill TTFT (latency wall).
- Attention dilution and lost-in-the-middle (quality wall).
- Cost-per-query (economic wall).

Staff answers reason about all four.

---

## 3. The big picture: four ways systems handle more context

Strong systems rarely pick one. Interviews often phrase this as "RAG vs. long context"; the right staff answer is **it's always both, plus compression, plus memory, with the split chosen per workload.**

### 3.1 Bigger native windows

**Core idea**: Train/finetune the model to attend over more tokens.

- *Solves*: Tasks where you genuinely need cross-document reasoning with full fidelity (codebase-wide refactors, multi-document legal/scientific analysis, long transcripts). No retrieval miss.
- *Doesn't solve*: Freshness (needs re-prompting), cost, tail latency, attention dilution.
- *Tradeoffs*: Highest per-query cost and latency; best grounding when the task actually needs everything; worst for freshness.
- *Fails when*: Signal is sparse inside a sea of irrelevant tokens (distraction hurts), or when most of the context is stable across requests but you still pay full prefill every time.
- *Right when*: The task legitimately needs joint reasoning over all of it, and you have aggressive prefix caching.

### 3.2 Retrieval / RAG

**Core idea**: Index external corpus; at query time fetch a small relevant subset and put it in the prompt.

- *Solves*: Scale (corpora >> context window), freshness (re-index), cost per query.
- *Doesn't solve*: Tasks that need joint reasoning across many documents; retrieval failures (the #1 hidden cost of RAG); ranking quality.
- *Tradeoffs*: Low per-query cost; fast; quality depends entirely on retrieval recall/precision and how you pack.
- *Fails when*: The query is under-specified for retrieval ("fix all memory bugs in this service"); when evidence is distributed across hundreds of chunks; when chunks lose context at their boundaries.
- *Right when*: The corpus is large, the query is specific enough to retrieve well, and you have a good retriever.

A common interview trap: candidates say "use RAG" without engaging with *packing order*, *relevance ranking*, *recency vs. relevance weighting*, *chunk size vs. overlap*, *query rewriting*, or *eval of retrieval recall @ k*. Don't fall into it.

### 3.3 Compression / summarization / compaction

**Core idea**: Summarize older context into shorter forms. Working memory stays small even as conversation history grows.

- *Solves*: Long-horizon conversations and agent trajectories where raw history grows without bound.
- *Doesn't solve*: Specific fact retrieval (summaries lose detail); anything that needs exact quotes or line numbers.
- *Tradeoffs*: Low runtime cost; cumulative summarization error; irreversible information loss.
- *Fails when*: Facts that seemed irrelevant at compaction time become critical later; when summaries hallucinate or edit state.
- *Right when*: Conversation is mostly narrative/progress-tracking and rarely requires exact recall.

Anthropic's Claude Code and context-editing features are concrete examples: `clear_thinking_20251015` drops old thinking blocks; context editing trims stale `tool_result` blocks; compaction replaces history with a summary while keeping the cached prefix (system prompt + tools) so KV-cache prefix stays warm.

### 3.4 External memory / agent memory architectures

**Core idea**: Move state out of the live prompt into a structured store (key-value, vector, doc store, or structured blob), fetched on demand.

- *Solves*: Cross-session persistence; user profile / preferences; episodic memory; scalable tool result history.
- *Doesn't solve*: Retrieval quality from the memory (you're back to RAG problems); consistency across writes; access control and provenance.
- *Tradeoffs*: Fastest per-query for anything that can be retrieved cleanly; hardest to evaluate; largest blast radius when memory is poisoned.
- *Fails when*: Staleness (old preferences); poisoning (prompt injection that writes bad memory); missing provenance.
- *Right when*: Cross-session continuity matters; individual turns don't need everything in-prompt.

### Why strong systems combine these

A production coding agent looks like:

```
+-------------------------------------------------------------+
|  SYSTEM (stable, cached ~1h)                                |
|    - persona, tool schemas, safety rules                    |
+-------------------------------------------------------------+
|  PROJECT CONTEXT (semi-stable, cached ~5m)                  |
|    - repo structure summary, conventions, CLAUDE.md         |
+-------------------------------------------------------------+
|  RETRIEVED FILES (per-query, not cached)                    |
|    - top-k files by relevance                               |
+-------------------------------------------------------------+
|  WORKING STATE (growing)                                    |
|    - plan, scratchpad, recent tool outputs                  |
+-------------------------------------------------------------+
|  CURRENT TURN                                               |
|    - user message                                           |
+-------------------------------------------------------------+
```

Each layer uses a different strategy: big static blocks get prefix-cached, dynamic state gets retrieved, old history gets compacted, durable knowledge goes to external memory. No single technique works across all layers.

---

## 4. Transformer mechanics under long context

### Why attention becomes expensive

Self-attention at sequence length n computes an n×n attention matrix. Compute is O(n²·d) where d is head dim. Memory for the attention matrix is O(n²) per head (though FlashAttention avoids materializing it in HBM).

Two operating regimes:

| Phase | Compute intensity | Bound by |
|---|---|---|
| Prefill (long) | High; O(n²) attention dominates | **Compute / tensor cores** |
| Decode (step) | Low; 1 query attending to n KV | **HBM bandwidth (KV movement)** |

```
Prefill on long prompt:
 ┌────────────────────────────────┐
 │  Input: N tokens                │
 │                                │
 │  For each layer:                │
 │    QKV proj  :  matmul  (compute)│
 │    Attention :  N×N (FlashAttn) │   <-- O(N²d) FLOPs; compute-bound
 │    FFN       :  matmul  (compute)│
 └────────────────────────────────┘

Decode step:
 ┌────────────────────────────────┐
 │  Input: 1 new token             │
 │  KV cache: N past keys/values   │
 │                                │
 │  For each layer:                │
 │    QKV proj (1×d)   (tiny matmul)│
 │    Attention: 1 query × N keys  │   <-- O(Nd); bandwidth-bound
 │                                │     must stream N KV entries
 │    FFN: (tiny matmul)           │
 └────────────────────────────────┘
```

### Arithmetic intensity (roofline intuition)

Arithmetic intensity = FLOPs / bytes moved. Compute-bound when AI > (peak FLOPs / peak bandwidth). On H100 SXM: ~989 TFLOPs BF16 / ~3.35 TB/s HBM → ≈ 295 FLOPs/byte.

- **Prefill attention** at long n: AI scales with n (reusing loaded KV across many queries), so it sits well above the roofline → compute-bound.
- **Decode attention**: each KV byte is used by exactly 1 query → AI ≈ 1 FLOP/byte → deeply bandwidth-bound. Decode is not limited by FLOPs; it's limited by how fast you can stream KV out of HBM.

This has a concrete implication: at long context, **decoding a single token** requires reading the *entire* KV cache from HBM. That's why long-context decode is dominated by memory movement, not math.

### Memory formulas at interview depth

KV cache bytes per request (BF16/FP16):

```
KV_bytes = 2 × L × H_kv × d_head × n × 2 bytes
         (2 for K and V; final 2 bytes = FP16/BF16)
```

Where L = layers, H_kv = KV heads (smaller than query heads under GQA), d_head = head dim, n = sequence length.

**Llama-70B example (GQA, 80 layers, 8 KV heads, 128 head_dim):**

```
per-token KV = 2 × 80 × 8 × 128 × 2 = 327,680 bytes ≈ 320 KB/token
 
  1K tokens  →   320 MB
 32K tokens  →    10 GB
128K tokens  →    40 GB
  1M tokens  →   320 GB  (exceeds a single H100's 80GB HBM)
```

Two things jump out:
1. A **single** 1M-token request already needs tiered memory (offload) on an 80GB H100.
2. Batch size is brutally limited. At 128K/request, you can fit maybe 1 request per H100 for a 70B model just for KV.

### What changes as n grows

- **Prefill wall**: TTFT grows super-linearly. At 1M, you *need* chunked prefill + pipeline/tensor parallelism or ring attention.
- **Decode bandwidth wall**: Per-token latency rises with n because each step must stream more KV.
- **Batching collapses**: Bigger per-request KV = fewer concurrent requests = lower throughput.
- **Scheduling gets harder**: Heterogeneous lengths (1K and 1M in the same queue) break uniform batching; you need chunked prefill or disaggregated prefill/decode (see §7).
- **Kernels shift focus**: FlashAttention-3 (Shah et al., 2024; uses Hopper WGMMA/TMA) is table stakes for prefill; decode kernels are increasingly about KV layout, paging, and quantization.

---

## 5. Positional encoding and extrapolation

### Why position handling breaks

Transformers are permutation-equivariant without positional information. Long-context issues live in *how* positions are encoded and how that interacts with attention as n grows far beyond training length.

Three position-encoding families:

1. **Absolute sinusoidal / learned** (original Transformer): Don't extrapolate.
2. **Relative position biases** (T5, ALiBi): ALiBi (Press et al., 2022) extrapolates by construction but has weaker long-range retrieval.
3. **Rotary (RoPE)** (Su et al., 2021): Dominant in modern LLMs (Llama, Mistral, Qwen, GPT-family). Rotates Q and K by position-dependent angles so inner products depend on relative positions.

### RoPE intuition

Each pair of Q/K channels is rotated by angle θ_i·m at position m, where θ_i = base^(-2i/d) and base is commonly 10,000.

```
RoPE: q'_{m,i} = R(θ_i · m) q_{m,i}
      k'_{n,i} = R(θ_i · n) k_{n,i}
      
⟨q'_{m}, k'_{n}⟩  depends only on (n - m).
```

Low-frequency channels (small θ_i) capture long-range; high-frequency channels (large θ_i) capture local.

### Extrapolation vs. interpolation

A model trained to length L_train attends well at positions ≤ L_train but not beyond — the high-frequency components alias badly when you feed positions >> L_train. Two families of extension:

- **Position Interpolation** (Chen et al., 2023): Rescale positions so all m ∈ [0, L_new] map into [0, L_train]. Trivial to apply, needs some fine-tuning, uniformly compresses all frequencies — hurts high-frequency (local) channels.
- **NTK-aware / YaRN** (bloc97 / Peng et al., 2023): Scale the base differently per frequency band so high-freq is preserved and only low-freq is compressed. YaRN adds temperature scaling on the attention logits. This is the de facto recipe for extending Llama-class models from, say, 4K → 128K with short fine-tuning.

### Why "1M context" ≠ "reliable 1M-token reasoning"

A model may nominally accept 1M tokens because its positions are encoded and it doesn't crash, yet:

- Retrieval recall drops beyond some length (the "trained effective length").
- Attention scores flatten at very long distances — the model effectively ignores far-away tokens (**attention dilution**).
- Position bias: attention concentrates at prompt start (the **attention sink**, Xiao et al., 2023) and end, with a sag in the middle. This is the mechanism behind **lost-in-the-middle** (Liu et al., 2023) — information in the middle of a long prompt is used less reliably.

```
Retrieval accuracy vs. needle position for a hypothetical 1M-context model
at various prompt lengths:

  accuracy
   1.0 ┤████████████████████████████████████  (32K prompt)
   0.9 ┤████████████████████████████████████  (128K, good model)
   0.8 ┤████▓▓▓▓▓▓▓▓▓▓░░░░░░░░░░▓▓▓▓▓▓▓▓████  (512K: U-curve, LIM)
   0.7 ┤██░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░██  (1M: sparse, only ends)
        └────────────────────────────────────
          start                           end
          (position in prompt)
```

### What interviewers want to hear

- Understand RoPE at the level of "rotate each channel pair by θ·position, inner products depend on relative position, different channels capture different frequency bands."
- Know the failure mode: extrapolation is hard because high-frequency channels alias at positions they weren't trained on.
- Know the fix families: position interpolation (uniform) vs. NTK-aware/YaRN (frequency-dependent).
- Know the quality story: nominal support doesn't imply usable quality. Lost-in-the-middle and attention sinks are real.

---

## 6. Long-context modeling strategies

The rough taxonomy and what each attacks:

| Family | Bottleneck attacked | Accuracy hit | Systems hit |
|---|---|---|---|
| Full attention + FlashAttention | Compute constant / HBM traffic | None | Still O(n²) compute |
| Sliding window / local | O(n²) → O(n·w) | Loses global links | Simple kernels |
| Sparse / structured sparse | Attention compute | Often lossy at long range | Kernel complexity |
| Linear / kernelized attention | O(n²) → O(n) | Big quality hit historically | Easier to serve |
| SSMs (Mamba) | O(n²) → O(n) | Competitive at scale, weak at retrieval | Different kernels |
| Hybrid (Griffin, Jamba) | Mix local + global/SSM | Best of both | Moderate |
| Recurrent / memory-augmented | Fixed compute/step | Info loss in compression | Harder to batch |
| Hierarchical / chunked | Parallelism + compression | Depends on chunking | Pipeline-friendly |
| RAG | Scale beyond any window | Retrieval recall limits | Two systems |

### Full attention + better kernels

The dominant answer for mainstream production LLMs (Claude, Gemini, GPT-4-class, Llama 3.x). Combined with GQA/MQA to shrink KV, FlashAttention-2/3 for memory-efficient attention kernels, and tensor parallelism for prefill, you can serve long context without changing the architecture. The "just fit more tokens" path — but only viable with aggressive systems work.

### Sliding-window / local

Mistral 7B used a 4K sliding window. Cheap, parallel, but you lose long-range links unless you combine with a few global tokens (Longformer-style) or interleave with global attention layers. Good for very long low-resolution context (e.g., long audio).

### Sparse attention

Longformer (Beltagy et al., 2020), BigBird (Zaheer et al., 2020): combinations of local + strided + global. Mostly academic today; kernel complexity didn't pay for itself once FlashAttention made dense attention cheap.

### Linear / kernelized attention

Performer (Choromanski et al., 2020), Linformer (Wang et al., 2020): O(n) attention via kernel approximations or low-rank projection. Historically weaker on retrieval. Modern linear-ish variants (RetNet, RWKV, gated linear attention) have gotten closer but still trail full attention on exact retrieval tasks.

### State-space models

Mamba (Gu & Dao, 2023) and Mamba-2 are the practical SSM line. O(n) compute, constant per-step memory, strong on language modeling, but materially weaker on precise retrieval (NIAH) and in-context learning vs. full attention. This is why **hybrids** dominate production for models that claim serious long context:

- **Jamba** (AI21, 2024): Mamba + Transformer blocks + MoE.
- **Griffin / RecurrentGemma** (Google, 2024): mix of local attention and gated linear recurrences.

Hybrid intuition: attention layers handle retrieval and in-context learning; SSM layers carry bulk sequence compute cheaply. At 256K+ contexts, the cost savings matter.

### Chunked / segment-based (training-time)

Rather than running attention across an entire 1M sequence during training, split it into chunks. Ring Attention (Liu et al., 2023) and variants (DeepSpeed-Ulysses, context parallelism in Megatron) shard the sequence dimension across devices and pipeline KV across a ring, making full attention tractable at training time for sequences that don't fit on one device.

This is the training-time analogue of tensor parallelism on the sequence axis. Essential for pre-training long-context models.

### Hierarchical / blockwise (inference-time)

Summarize chunks, then attend over summaries; or retrieve at query time from a chunk store. Effectively a form of RAG applied internally to a single long input. Used in long-document QA systems.

### RAG as a modeling strategy

Don't extend the model; give it a retriever. Scales to corpora far beyond any context window. The dominant practical answer for enterprise search, Q&A over unbounded corpora, and freshness-sensitive applications. But "use RAG" is the interviewer-trap answer unless you engage with retrieval recall, ranking, packing, chunking, and eval.

---

## 7. Long-context inference and serving

This is where staff-level inference-team interviews spend most of their time.

### Prefill vs. decode at long context

```
Request lifecycle:

  Admission  ─►  Tokenize  ─►  Prefill  ─►  Decode (loop) ─►  Detokenize
                               ▲    ▲       ▲
                               │    │       └── per-token; bandwidth-bound
                               │    └── produces KV cache (all layers, all pos)
                               └── dominates TTFT at long ctx
```

- **Prefill** produces the full KV cache for the prompt. Compute-bound for long prompts; O(n²) attention.
- **Decode** produces tokens one at a time, reading the entire KV cache per step. Bandwidth-bound.

TTFT is roughly prefill latency. TPOT (time per output token) is decode latency. These have *different* bottlenecks, which is the entire motivation for disaggregation (§7.3).

### 7.1 Chunked prefill

Instead of running prefill for the full prompt in one giant attention op, split into chunks of, say, 4K–8K tokens and process them sequentially, appending to the KV cache each chunk. Benefits:

- Bounds the per-step work; a 1M-token prefill doesn't freeze the GPU for 30 seconds.
- Lets you **interleave decode steps from other requests between prefill chunks**, greatly improving TPOT stability for already-running requests.
- Gives the scheduler finer-grained admission control.

Introduced in Sarathi-Serve (Agrawal et al., 2024). Standard in vLLM, SGLang, TensorRT-LLM. The knob is chunk size — too small wastes launch overhead; too big re-creates the original problem.

### 7.2 Continuous batching

Iteration-level batching (Orca, Yu et al., 2022): at each step, the scheduler decides what's in the batch. Finished requests leave, new requests join, no per-request padding waste. The assumed-current baseline.

Under long context, continuous batching alone isn't enough: a single long prefill holds up the batch for all the short decodes stuck behind it. This is why chunked prefill exists, and why disaggregation is now standard.

### 7.3 Disaggregated prefill / decode (PD)

Run prefill and decode on separate GPU pools, transfer KV cache between them. Proposed in DistServe (Zhong et al., 2024), now production reality:

- **NVIDIA Dynamo** (GTC 2025): first-class prefill workers + decode workers + KV-aware router + NIXL (transfer library over NVLink/IB/PCIe/SSD).
- **vLLM**: disaggregated-prefill feature with P2pNccl / Mooncake connectors; LMCache-based P/D in llm-d 0.3 reaching ~2.2k tokens/s/H200 with 32-way EP (October 2025).
- **SGLang**: first-class PD disaggregation; DeepSeek-R1 on 96 H100s (24 prefill, 72 decode) achieved 52.3k input / 22.3k output TPS/node; GB200 NVL72 follow-up showed 3.8× prefill and 4.8× decode throughput gains over H100.

Why disaggregate:

- Prefill is compute-bound → tensor-core-rich GPUs, possibly lower HBM.
- Decode is memory-bandwidth-bound → HBM-rich configurations, possibly pipeline-parallel for KV capacity.
- You size each pool independently to the workload's prefill:decode ratio.
- Eliminates head-of-line blocking: a 1M-token prefill doesn't pause 500 in-flight decodes.

Costs: KV cache transfer. At 320 KB/token for Llama-70B-class, a 4K prompt needs ~1.3 GB transferred before decode starts. At the ~500ms TTFT budget, you need ≥ 4–5 GB/s of cross-node bandwidth, which is why NIXL and RDMA-based transports matter. Layer-pipelined transmission (start transferring earlier layers while later layers are still being computed) hides a large fraction of this.

### 7.4 Prefix caching / prompt caching

The single highest-leverage optimization for long-context serving.

Observation: many requests share a prefix (system prompt + tools + long document + conversation history, with only the last user message varying). Prefill is deterministic given prefix tokens, so the KV cache for the shared prefix can be reused.

Two variants:

- **Engine-side RadixAttention** (SGLang, Zheng et al., 2024): Maintains a radix tree of KV cache prefixes; on new request, finds the longest matching prefix and prefills only the suffix.
- **API-exposed prompt caching**: Claude's `cache_control` blocks with 5-min or 1-hour TTL, writes at 1.25× base price and reads at 0.1× base price; OpenAI and Gemini have analogous features.

Claude Code's case is the clean example: ~18K tokens of system prompt + tools + `CLAUDE.md` get written once per session and read on every turn. Even on compaction, Claude Code deliberately preserves the same cached prefix and only changes the conversation portion so the cache stays warm. Tool list is locked at session start precisely because adding one tool invalidates the entire prefix hash.

The operational consequence: cache hit rate becomes a first-class metric and you design the prompt layout around it (stable blocks first, varying content last).

### 7.5 Scheduling and admission

At long context, scheduling ≳ kernel work. What the scheduler decides:

- **Admission**: reject / queue when aggregate KV would blow HBM.
- **Priority**: short prompts first (SPF) improves mean TTFT; SLA-based priorities for paid tiers.
- **Chunked prefill sizing**: per-step prefill budget vs. decode latency.
- **KV paging / eviction**: which requests keep KV in HBM vs. offload.
- **PD routing**: KV-aware routing to prefill instances holding a warm prefix.

### 7.6 What bottlenecks shift with context length

```
 Context → short          medium         long            very long
                |              |              |              |
 Bottleneck: compute/FFN → prefill attn → KV bandwidth → KV capacity
 Also:        batch size        TTFT     decode latency  admission
                                         & HBM pressure  & offload
```

Short requests: throughput-bound by FFN compute and batch pack efficiency. Medium: attention in prefill dominates TTFT. Long: decode per-step latency is set by KV bandwidth. Very long: you run out of HBM and become scheduler-bound — offload, eviction, admission control dominate.

---

## 8. KV cache deep dive

### What it is, why it exists

After prefill, every layer has produced K and V tensors for every input position. Decode queries attend against these stored K/V rather than recomputing from scratch. Without the KV cache, each decode step would re-run the full sequence — quadratically expensive and intolerable.

```
For each layer l, each position t:
   K[l,t] = X_t · W_K[l]
   V[l,t] = X_t · W_V[l]

KV cache stores these across all t ∈ [0, n-1], all layers.
At decode step n:
   Q_n attends to K[l, 0..n-1], V[l, 0..n-1]
   KV cache grows by one K and one V per layer per step.
```

### Sizing (recap with more detail)

```
per-token KV bytes = 2 · L · H_kv · d_head · bytes_per_element
Total KV bytes     = per-token KV · n · batch

Llama-3-70B (L=80, H_kv=8 under GQA, d_head=128, BF16):
  per-token      = 2 · 80 · 8 · 128 · 2    = 327,680 B ≈ 320 KB

Llama-2-70B (MHA, H_kv=64):
  per-token      = 2 · 80 · 64 · 128 · 2   = 2,621,440 B ≈ 2.6 MB  (8× larger)

DeepSeek-V3 (MLA, latent KV, very compact):
  per-token      ≈ 2 · L · d_c · 2 (d_c is latent dim, ~512)
                 ≈ 120 KB / token at 61 layers, d_c=512  (further shrunk)
```

GQA moves us from MHA's 2.6 MB/token to 0.32 MB/token for Llama-70B. MLA (multi-head latent attention, DeepSeek) pushes another order further by caching a low-rank latent rather than full K/V.

### The dominance of KV at long context

At 1M tokens, Llama-70B KV = 320 GB. That's:

- 4× an H100's 80 GB HBM.
- Larger than the model weights themselves (~140 GB in BF16).

For any long-context deployment, KV management — not weight shuffling — is the primary memory problem. Batch size, admission, and paging policies are all downstream of KV.

### Paged / block-based KV cache

Originally, vLLM's PagedAttention (Kwon et al., 2023) solved KV fragmentation. Classic allocation — one contiguous KV buffer per request sized to max_len — wastes memory when requests end early and fragments when requests have heterogeneous lengths.

Paging chops KV into fixed-size blocks (e.g., 16 tokens) and allocates them on demand from a pool:

```
Request A:  block_ids = [7, 2, 11, 18]
Request B:  block_ids = [4, 9, 3, 21, 14]
Request C:  block_ids = [1, 6]

Physical KV pool: one large buffer of blocks.
Attention kernel: indirection via block table → gather K/V per block.

Benefits:
 - No fragmentation: blocks free independently
 - Copy-on-write for prefix sharing (beam search, parallel samples)
 - Clean prefix caching (RadixAttention): shared prefix → shared blocks
```

PagedAttention adds an indirection layer in the attention kernel (block-table lookups during K/V gather), but FlashAttention-style paged kernels now land close to dense. This is the foundation on which prefix caching, prefix sharing, and KV offload are built.

### Prefix sharing / caching

Two levels:

1. **In-memory prefix sharing** across concurrent requests (beam search, sampling with the same prefix): the block table points the same block IDs at shared physical blocks.
2. **Cross-request prefix caching**: keep popular prefixes' blocks alive across requests. RadixAttention organizes these as a radix tree keyed by token sequences. On a new request, find the longest matching prefix, skip prefill for its length, and prefill only the suffix.

Prefix-cache hit rate is a dominant factor in long-context serving cost. Production systems often see 70–95% cache hits on conversational / tool-heavy workloads, which can cut TTFT by an order of magnitude.

### KV transfer, offload, tiering

When aggregate KV exceeds HBM:

- **Offload to CPU DRAM** over PCIe (~60–120 GB/s effective). Requires prefetching/streaming because decode reads every KV every step.
- **Offload to SSD/NVMe** (~7 GB/s) for cold prefixes. Only viable if access is infrequent.
- **Cross-node transfer** for PD disaggregation or KV migration (NIXL, Mooncake, LMCache).

Tiering policy: LRU-ish at the prefix-block level, with hotness driven by prefix-cache hit rate. LMCache and Mooncake formalize this as a KV cache storage system with tiered HBM → DRAM → disk → remote.

### KV compression / quantization

Two families:

- **Quantization**: K and V in 8-bit (KIVI, Liu et al., 2024) or 4-bit; approximately lossless at int8, noticeable quality degradation under int4 for some tasks. Halves or quarters the capacity wall.
- **Selective retention / eviction**: keep only "important" KV entries. H2O (Heavy Hitter Oracle, Zhang et al., 2023), SnapKV, StreamingLLM (Xiao et al., 2023) — the last exploits the observation that the initial tokens ("attention sinks") disproportionately carry attention mass, so keeping those plus a sliding window of recent tokens is often sufficient.

The tradeoff: eviction methods are aggressive and can hurt exact retrieval (NIAH) even when perplexity stays fine. Safer as a degraded-mode fallback than as a default.

### Correctness and latency tradeoffs

- Prefix cache invalidation: any change in prompt before the breakpoint invalidates the cache. Tool additions, timestamps in the system prompt, non-deterministic template formatting — all silent cache killers.
- Quantization: can introduce subtle quality drops not caught by short benchmarks; need long-context eval.
- Paging: kernel overhead is small but real; watch tail latency at small batch sizes.
- Offload: always costs bandwidth; only worth it when the alternative is dropping the request.

---

## 9. Memory hierarchy for long-context systems

Staff-level systems view:

```
                     Capacity    Bandwidth       Latency     Use
  ┌──────────────┐
  │ HBM (GPU)    │   80-192 GB   3-8 TB/s        ~ns         Active KV, weights
  ├──────────────┤
  │ NVLink       │   intra-node  0.9 TB/s P2P    ~µs         TP, PD transfer
  ├──────────────┤
  │ CPU DRAM     │   0.5-2 TB    200-400 GB/s    ~100ns      Warm-but-not-hot KV
  ├──────────────┤
  │ Local NVMe   │   8-60 TB     7-14 GB/s       ~10-100µs   Cold KV, prefix store
  ├──────────────┤
  │ RDMA / IB    │   network     50-400 Gbps     ~µs-ms      Cross-node KV (PD)
  ├──────────────┤
  │ Remote store │   unlimited   varies          ~ms         Prefix registry, logs
  └──────────────┘
```

### What belongs in which tier

- **HBM**: active requests' KV, model weights, scratch.
- **NVLink**: TP-shard transfers, intra-node PD KV transfer.
- **DRAM**: prefix cache overflow, KV offload for long-lived requests, tokenizer/serialization state.
- **NVMe**: cold prefix registry (e.g., multi-day conversation KV that might be revived), checkpointed large-document KV for customers paying for persistence.
- **Network (RDMA/IB)**: cross-node PD KV transfer, multi-node KV cache pools.

### Offload vs. recompute

The trade is: is transferring the cached KV back into HBM cheaper than recomputing prefill for the prefix?

Rule of thumb:

```
Recompute prefill cost ≈ 2 · n · P FLOPs for an n-token prefix, where P = model params active.
                        ≈ (2 · 1M · 70B) / 1000 TFLOPs/s ≈ ~140s for a 1M prefix on H100.
                        
Transfer cost          ≈ KV_bytes / bandwidth
                        ≈ 320 GB / 200 GB/s ≈ 1.6s from DRAM, 
                          320 GB / 14 GB/s ≈ 23s from NVMe.
```

DRAM offload dominates for anything with cached KV. NVMe only wins over recompute for very long prompts (>200K or so, depending on model). Eviction to NVMe is reasonable for cold prefixes; eviction to "gone" with recompute on miss is fine for short-to-medium prompts.

### When remote/context transfer helps

- PD disaggregation: KV transfer is amortized over a long decode, and keeps prefill GPUs hot.
- Multi-region failover: ship KV rather than reprefill.
- Cross-session resume: persisted KV lets users resume a 500K-token coding conversation without paying prefill again.

When it doesn't: short prompts (just recompute), ultra-low-latency SLAs (transfer adds 100s of ms), or when bandwidth is not provisioned (consumer networks).

---

## 10. Long-context quality issues

Nominal support ≠ reliable reasoning. The failure modes:

- **Lost-in-the-middle** (Liu et al., 2023): Multi-doc QA accuracy traces a U-shape with answer position — high at start and end, sagging in the middle. Reflects position bias from training data distributions. Mitigations: put critical evidence first or last, use query-aware reranking before packing.

- **Retrieval failure inside the window**: Even if you fit everything, the model may miss facts buried in noise. NIAH tests measure this crudely; RULER/HELMET measure it better.

- **Attention dilution**: At long n, softmax attention spreads mass over many positions, so relevant tokens get diluted. Hurts precise retrieval more than summarization.

- **Distraction from irrelevant context**: Adding relevant *plus* irrelevant text often performs worse than relevant alone. Less context can be better (see Liu, 2023 and subsequent work on noise robustness).

- **Instruction drift / prompt contamination**: In long prompts, injected instructions (e.g., from user-provided documents) can override system instructions. This is a security issue and a quality issue.

- **Compounding summarization error**: Each compaction loses information and can introduce hallucinated detail. In long agent runs with 10+ compactions, the state diverges from ground truth.

- **Context poisoning via tools or memory**: Tool outputs written into the context (or into agent memory) can contain injected instructions. If they survive compaction, they persist.

- **Stale memory**: Old preferences, outdated state, resolved issues re-appearing as if they still matter.

- **Long-context hallucination modes**: At long n, models are more willing to fabricate when the needle isn't there (because the surrounding context creates plausible-sounding fillers). Makes refusal / "not found" harder to train.

- **Effective vs. nominal gap**: Gemini 2.5 Pro reports ~100% recall to ~530K, ~99.7% at 1M on NIAH — but NIAH is the easy case. Multi-hop, aggregation, or reasoning tasks degrade much faster.

Interview-ready framing: **"A 1M-token model is a 1M-token *NIAH* model. For real tasks, assume the usable length is 2–8× smaller, and design around that."**

---

## 11. Evaluation and benchmarks

### Benchmarks to know

| Benchmark | What it measures | Why it matters |
|---|---|---|
| **NIAH** (Needle in a Haystack, Kamradt 2023) | Retrieve a single fact from long filler | The easy case; saturates quickly |
| **RULER** (Hsieh et al., 2024) | 13 tasks × 4 categories: retrieval, multi-hop, aggregation, QA | Goes beyond single-needle; better than NIAH |
| **HELMET** (Yen et al., ICLR 2025) | 7 task categories incl. RAG, reranking, ICL, QA, summarization, cite-gen | Current best at frontier model discrimination at 128K |
| **∞Bench** (Zhang et al., 2024) | Realistic QA, math, code at ≥128K | Real-world lengths |
| **NoCha** (Karpinska et al., 2024) | Narrative true/false over novels, 2023+ books | Mitigates train-test leakage |
| **NoLiMa** (Adobe Research) | Long-context beyond literal matching | Tests reasoning-style retrieval |
| **LongCodeBench** | Coding at 1M context | Code-specific, realistic |
| **LongGenBench** | Long-form generation | Generation, not just retrieval |
| **Michelangelo, LOFT** | Latent reasoning / retrieval-with-generation | Newer, more challenging |

Key HELMET findings (Yen et al., 2025): NIAH is saturated across frontier models and no longer discriminates; RULER and ∞Bench sometimes give "weird" results (Gemini Flash > Pro on RULER, 70B Llama < 8B on ∞Bench). HELMET is better at producing consistent rankings at 128K.

### Why retrieval benchmarks aren't enough

- NIAH passes don't predict multi-hop reasoning, summarization, or generation.
- Benchmarks on 2023-era text are contaminated in modern models.
- Synthetic tasks don't map to agent failure modes (context poisoning, tool-result drift).
- Short benchmarks' results don't extrapolate to 512K/1M.

### Measuring production effective context

For staff-level answers on "how do you know your long-context design works":

1. **Task-anchored eval**: build evals from your actual tasks (e.g., for code agents: "given repo of X tokens, fix bug Y"). Measure accuracy vs. context length.
2. **Position-swept eval**: vary where the needle/evidence sits. Quantify the lost-in-the-middle curve for your workload.
3. **Noise-injection eval**: add irrelevant documents to measure distraction robustness.
4. **Context-compaction eval**: run the agent with/without compaction, measure task success. You want compaction to be ≈ neutral; if it's not, you're losing info.
5. **Tail-latency eval**: p50/p95/p99 TTFT and TPOT at long context — quality is nothing if p99 is 2 minutes.
6. **Cache hit rate** as a quality-adjacent metric: low cache hit = cold prefill = frequently different system prompts = often a bug.

What to say in interviews: name 2–3 benchmarks (e.g., HELMET, NoCha, RULER), explain why NIAH isn't enough, then describe a task-specific eval methodology.

---

## 12. Context engineering for agentic systems

Context engineering is the discipline of deciding **what tokens end up in each model call**, across a session, across sessions, across agents and subagents. It subsumes prompt engineering; it's to prompt engineering what distributed systems design is to threading.

### Context engineering vs. prompt engineering

- **Prompt engineering**: choose phrasing, format, examples for a single call.
- **Context engineering**: architect *pipelines* that assemble context — retrieval, ranking, packing, summarization, memory — across many calls.

Most agent failures at the staff level are context failures, not model failures. "The model got confused" usually means: the context had contradictory facts, stale state, duplicated tool outputs, instruction drift from a prior turn, or missing info that was summarized away.

### Assembly pipeline

```
User turn ──► Query rewrite ──► Memory fetch ──► Retrieval ──► Rerank ──► Pack ──► Call model
                                       │                                    │
                                       ▼                                    ▼
                                  Working state                       Compaction, if over budget
                                  (plan, scratch,                     
                                   tool results)                      
```

### What lives where

| Layer | Examples | Strategy |
|---|---|---|
| System | Persona, safety, core rules, tool schemas | Static, cached (1h TTL), never changes |
| Long-stable project | Codebase overview, CLAUDE.md, docs summary | Cached (5m TTL or 1h) |
| Task context | Current plan, goal, constraints | Lives in prompt; small |
| Retrieved content | Files, docs, search results | Refetched per turn, not cached |
| Tool state | Recent tool calls and results | Rolling window; old entries compacted or evicted |
| Working memory | Scratchpad, intermediate reasoning | Recent only; compact or drop old |
| Episodic memory | Past sessions, prior task outcomes | External store; fetched on demand |
| Semantic memory | User preferences, learned facts | External store; fetched on demand |

### What should stay in the prompt

- Content that is **used this turn**, *and* small, *and* sensitive to exact wording.
- Recent tool results and reasoning.
- Current plan and state object.

### What should move to retrieval

- Anything large where only a subset is relevant now.
- Anything refreshed out of band (documentation, knowledge base).
- Anything searchable by the current query.

### What should be summarized

- Old conversation turns whose details no longer matter.
- Tool output logs ("I ran the tests, 12 passed, 3 failed" rather than 500 lines of output).
- Multi-step subagent work whose final result is what matters.

### What should be dropped

- Stale plans superseded by current plan.
- Completed subgoals.
- Errored tool calls that got successfully retried.
- Empty/negative search results older than a turn.

### Why agent failures are context failures

Concrete examples of "the model got confused" translated into context terms:

- *Agent keeps retrying a failed tool call*: the error message got compacted away; the agent doesn't know it already tried.
- *Agent "forgets" the user's preference*: preference was in turn 3, compacted out of working memory, not in semantic memory.
- *Agent hallucinates an API*: retrieval didn't surface the real API docs; model filled from priors.
- *Agent follows injected instructions from a document*: no separation between trusted and untrusted content.
- *Agent flip-flops on plans*: plan isn't in a stable, canonical location — multiple partial plans in history.

All of these are fixable via context engineering, not via bigger models.

---

## 13. Agent memory architecture

### Three-layer memory model (working / episodic / semantic)

```
┌─────────────────────────────────────────────────────────┐
│ Semantic memory                                         │
│   - Stable facts (user preferences, learned knowledge)  │
│   - Slowly updated, long TTL                            │
│   - Typically vector store + structured KV              │
└─────────────────────────────────────────────────────────┘
           ▲                            ▲
           │ consolidation              │ retrieval
           │                            │
┌─────────────────────────────────────────────────────────┐
│ Episodic memory                                         │
│   - Past sessions, past tasks, past tool traces         │
│   - Medium TTL, append-mostly                           │
│   - Vector + metadata store                             │
└─────────────────────────────────────────────────────────┘
           ▲                            ▲
           │ session close              │ retrieval / recall
           │                            │
┌─────────────────────────────────────────────────────────┐
│ Working memory (live context)                           │
│   - Current task state, plan, recent tool results       │
│   - Short-lived, held in prompt or scratchpad           │
│   - Compacted under pressure                            │
└─────────────────────────────────────────────────────────┘
```

- **Working memory** is what's in-prompt or in the immediate scratchpad right now.
- **Episodic memory** is a record of specific prior events (sessions, tool calls with arguments, outcomes).
- **Semantic memory** is distilled, stable knowledge (preferences, learned invariants about the user or task).

Cognitive-science terms, but they map cleanly to engineering distinctions.

### Memory writes

Explicit write policies:

- **Write-through on event**: every tool result goes to episodic log.
- **Write-on-close**: summarize session on exit, write to semantic memory.
- **Write-on-salience**: "this looks important" — either user-flagged, agent-flagged, or gated by a lightweight classifier.
- **Write-on-compaction**: when working memory compacts, the dropped content lands in episodic store.

Bad write policies (frequently observed in production):

- Writing every turn unconditionally → unbounded store growth, retrieval noise.
- Agent self-writes anything it thinks matters → prompt injection becomes persistent memory poisoning.

### Memory reads

- **Recency-based**: last N items.
- **Similarity-based**: dense retrieval from vector store.
- **Hybrid**: BM25 + dense + rerank.
- **Structured query**: "all tool calls to service X in this session".
- **Triggered**: memory accessed only when some predicate fires (e.g., user mentions a named entity).

### Compaction / summarization / reflection

The mechanism for moving content up the hierarchy (working → episodic → semantic) without blowing up storage:

- **Rolling summaries**: summarize the last N turns periodically; replace them with the summary.
- **Milestone summaries**: at task boundaries, produce a structured summary (facts learned, decisions made, open questions).
- **Reflection**: offline process that reviews episodic logs, distills patterns into semantic memory.

Anthropic's Claude Code auto-compaction is a milestone summary run: when context fills, a compaction run produces a summary with the identical cached prefix (same system prompt, tools, CLAUDE.md) and appends a compaction instruction as a new user message; only the conversation portion changes, preserving the KV-cache prefix.

### Freshness, provenance, access

- **Freshness**: each memory entry needs a timestamp and TTL. Stale user preferences are worse than missing ones.
- **Provenance**: record where a memory came from (user said, agent inferred, tool returned). Needed for audit and for poisoning mitigation — never treat tool-output-derived memory as authoritative.
- **Access control**: user-scoped keys, no cross-user leakage; tool-output-derived memory should not escape a session without explicit review.

### Observability

Without observability, memory systems fail silently and catastrophically. Minimum:

- Every memory write logs: who wrote, why, content, TTL.
- Every retrieval logs: query, retrieved IDs, scores, which were used in prompt.
- Dashboards on memory store growth, retrieval hit rate, retrieval precision vs. a gold set.
- Traces that tie an end-to-end agent run to every memory read/write.

### Common patterns

- **Task-local context**: session-scoped; cleared at task close.
- **User profile / preferences**: account-scoped; semantic memory; read on session open.
- **Tool result store**: canonicalize results (especially from non-deterministic tools) so identical queries produce identical keys; dedupe.
- **Checkpoints**: periodic "everything I know right now" blobs so interrupts are recoverable.
- **Subagent summaries**: parent agent sees the summary, not the full trace.
- **Hierarchical orchestration**: orchestrator holds plan; subagents hold their local state; orchestrator receives subagent summaries only.
- **Context compaction**: automatic or manual; must preserve cache prefix.

---

## 14. Agent context management patterns

### Retrieval-before-generation

Default pattern: (query → retrieve → rerank → pack → generate). Details that matter:

- **Query rewriting**: the user's message is often not a good retrieval query. Rewrite using recent context and plan.
- **Multiple retrievals**: code + docs + tickets, each with different retrievers.
- **Rerank**: LLM-based reranker or a cross-encoder against the top-k.
- **Packing order**: put evidence at the boundaries (start/end) to mitigate lost-in-the-middle; or use structured markup so the model can navigate.

### Budgeted context packing

Given a token budget B, and candidate items each with (score, tokens, freshness):

```
maximize Σ score_i · x_i
subject to Σ tokens_i · x_i ≤ B
           x_i ∈ {0, 1}
           (plus coverage constraints, freshness penalties, diversity)
```

In practice: greedy by score/tokens ratio, with diversity and freshness bonuses. Key inputs:

- **Salience scoring**: relevance to the current query (retrieval score + recency bonus + explicit user marks).
- **Freshness**: newer items preferred for time-sensitive tasks.
- **Diversity**: don't fill the budget with five near-duplicates.
- **Must-haves**: plan, system, last-N turns are non-negotiable.

### Hierarchical / milestone summaries

Instead of summarizing every N turns (uniform), summarize at semantic boundaries: plan completed, tool invocation succeeded, user goal shifted. This preserves information density better than uniform summarization and is less lossy.

### Rolling state objects

A structured blob (JSON / YAML / markdown table) that is the canonical "what I know now":

```
{
  "goal": "...",
  "current_plan": [...],
  "completed_steps": [...],
  "open_questions": [...],
  "known_facts": [...],
  "recent_tool_results": [...],
  "blockers": [...]
}
```

Update the object at each step; include it in the prompt. This gives the model a single authoritative place to look rather than scanning 20 turns of history for current state. The pattern also survives compaction trivially — you compact the narrative, not the state.

### Checkpoint / resume

Periodically persist state object + relevant memory IDs. On resume:

1. Load state object.
2. Refetch referenced files/tools (don't rely on cached copies).
3. Short synthetic "here's where we are" message.
4. Continue.

This is what well-designed coding agents do on session restart.

### Subagent decomposition

Break work into subtasks, each run as a separate model call with its own scoped context:

```
Orchestrator: plan, assign subtasks, aggregate summaries
   │
   ├── Subagent A: "read service X, report interface summary"
   ├── Subagent B: "analyze failing tests"
   └── Subagent C: "draft fix"

Each subagent only sees what it needs. Orchestrator only sees summaries.
```

Advantages: parallelism, bounded per-agent context (thus better quality), clean interfaces. Disadvantages: summaries hide detail; coordination overhead; increased total tokens.

### Tool result canonicalization

Identical operations should produce identical tokens (enables dedup and cache reuse). Concretely:

- Strip timestamps, request IDs, non-deterministic formatting from tool output.
- Sort unordered outputs.
- Truncate predictably (deterministic head/tail, not random).

Without canonicalization, two semantically identical tool calls produce different tokens, bloating context.

### Context compaction

Reactive (when approaching limit) or proactive (at semantic milestones). Key design choices:

- **Preserve cache prefix**: compact only the volatile section, keep the cached prefix byte-identical.
- **Preserve state objects**: compact narrative, keep structured state.
- **Write dropped content to episodic memory** before removal.
- **Include a "compaction occurred at turn N" marker** so the agent knows memory gaps exist.

### Prompt / tool schema caching

Tool definitions are typically 5–20K tokens in big agent systems. Cache them aggressively (5m / 1h TTL). Claude's `cache_control` on tool definitions + system prompt + CLAUDE.md is the standard pattern: write once, read on every turn at ~10% of input price.

If the toolset exceeds 20 or so tools, consider **tool search**: ship a lightweight `search_tools` primitive that retrieves the relevant tool schemas on demand instead of including all of them up front. Claude has this as a first-class feature (`tool_search`).

### What can go wrong with each pattern

| Pattern | Failure mode |
|---|---|
| Retrieval before generation | Retrieval miss → silent failure; query rewriting loses intent |
| Budgeted packing | Scoring miscalibration drops critical facts; diversity penalty drops the only relevant chunk |
| Milestone summaries | Cumulative error; early milestones diverge from ground truth |
| Rolling state object | State object drifts from reality; model edits it carelessly |
| Checkpoint/resume | Referenced resources no longer exist; state is stale |
| Subagent decomposition | Summary loses detail parent needed; coordination bugs |
| Tool canonicalization | Over-aggressive normalization hides real differences |
| Context compaction | Drops info that becomes critical later; compacts over an injected instruction |
| Schema caching | Adding a tool invalidates the whole cache |

---

## 15. Retrieval vs. long context (explicit comparison)

| Axis | Huge window direct | Classic RAG | Hybrid RAG+LC | Memory-based agent | Summary compression |
|---|---|---|---|---|---|
| Recall | Perfect (in-window) | Retriever-limited | Good | Retriever+memory-limited | Lossy |
| Precision | Diluted by length | Focused | Focused | Focused | Depends |
| Freshness | Per-request only | Re-index → fresh | Fresh | Fresh | Per-compaction |
| Latency (TTFT) | Worst | Best | Middle | Middle | Middle |
| Cost/query | Worst | Best | Middle | Low | Low |
| Grounding quality | Strong (exact text) | Depends on chunks | Good | Depends | Weak |
| Controllability | Low (pack all) | High | High | High | Medium |
| Eval difficulty | Hard at length | Moderate (recall@k) | Hard (both layers) | Hard | Hardest |
| Failure mode | Lost-in-middle, cost | Retrieval miss | Both | Memory staleness | Cumulative error |
| Right when | Task needs everything | Corpus >> window, query is specific | Most enterprise agents | Cross-session persistence matters | Long narrative with low exact-recall need |

**Staff answer**: In serious production systems, you have all of them, composed:

- Static long content (big doc, codebase) → prefix-cached long context.
- External corpora → RAG with query rewriting, rerank, and eval.
- Conversation history → summarization / compaction with state object preservation.
- Cross-session durable state → semantic memory with provenance.

"RAG vs. long context" is usually a false binary; the only interesting version of the question is "in this specific workload, where does the boundary go?"

---

## 16. Operational and product concerns

### Latency budgets

For a user-facing coding agent:

```
Turn budget: ~3-8s end-to-end acceptable
  TTFT:       ~500-1500ms
  Tool calls: 500-3000ms each (often the dominant term)
  Decode:     varies; often 50-200 tokens at 30-80 tok/s

For a long-document analysis:
  TTFT:       3-20s acceptable (up to 60s for 1M prompts)
  Decode:     can be longer
```

Prefix caching cuts TTFT by 5–10× on warm caches. Chunked prefill prevents worst-case stalls.

### Cost budgets

- Input vs. output pricing asymmetry: input is often 5–10× cheaper than output; design for more input tokens, fewer output tokens where possible.
- Cache write markup: Claude prompt-cache writes at 1.25× base, reads at 0.1× base — break-even after 2–3 hits. Cache everything that hits ≥2× per TTL window.
- 1h vs. 5m TTL: Claude 1h writes cost 2× 5m writes, break-even on hit rate. 1h makes sense for tool schemas and system prompts in slow-churn deployments.

### Token accounting

Build observability into every call:
- Input tokens (cached / uncached)
- Output tokens (regular / reasoning)
- Cache hit ratio
- Tokens per context layer (system / tools / retrieval / history / user)
- Retrieval tokens returned vs. used

Staff-level operational maturity means knowing **where the tokens go** across a session, and debugging cost regressions by layer.

### Rate limits and throughput

Long contexts consume token-per-minute budgets fast. 1M-token prefills can exhaust per-user quotas in a handful of requests. Mitigation:

- Per-user quotas on input tokens, not requests.
- Asymmetric priorities: interactive vs. batch jobs.
- Retry with back-off specifically for rate limit errors (429s), not for all 5xx.

### Observability for long-context systems

- **Per-request traces**: tokenize time, prefill time, first-token time, decode rate, total.
- **Prefix cache hit/miss with keys**: which prefixes are hot, which churn.
- **KV cache occupancy**: HBM usage over time, paging events, offload events.
- **Queue depths**: prefill queue, decode queue, PD transfer queue.
- **Eval hooks**: sample a percentage of production requests into a replay-able eval set with gold answers.

### Safety / privacy in memory

- Memory writes derived from user input can include sensitive content (PII, secrets). Classify before writing; optionally redact.
- Memory writes derived from tool output can include injected instructions. Treat as untrusted.
- Cross-user / cross-tenant isolation: hard keys, not soft.

### Blast radius of bad memory

A poisoned memory entry persists across sessions, contaminates every future retrieval, and gets laundered through summarization into semantic memory. Design for:

- Audit logs on every memory write.
- Easy revocation (delete by provenance / time window).
- Replay capability: reproduce a session with memory state as of time T.

### Debugging long-running agents

- Log every tool call, every retrieval, every memory op with causal IDs.
- Snapshot context at each turn (or only on error).
- Reproduce from snapshot with deterministic sampling for bug isolation.

---

## 17. Recent developments through 2025–2026

Not a survey — the shifts that changed design practice.

### Million-token-class production windows are standard

Gemini 2.5 Pro ships 1M context with 2M in beta ([Google blog, Mar 2025](https://blog.google/innovation-and-ai/models-and-research/google-deepmind/gemini-model-thinking-updates-march-2025/)), near-100% NIAH recall out to ~530K and ~99.7% at 1M (reported by Google). Claude Sonnet 4 launched a 1M beta on Anthropic API and Bedrock (2025); Gemini 3.x Pro Preview lands with even stronger reasoning at long context. **Implication**: 1M is no longer exotic. The binding constraint has shifted from "does the model accept it" to "can I serve it economically with acceptable quality".

### Prompt caching as first-class product surface

Claude's `cache_control` with 5-min and 1-hour TTLs, up to 4 breakpoints, GA across Anthropic API, Bedrock, Vertex. Cache reads at 10% of input price; writes at 1.25×. Minimum cache tokens raised to 4096 for Claude Opus 4.5/4.6 and Haiku 4.5. ([Claude docs](https://platform.claude.com/docs/en/build-with-claude/prompt-caching)). **Implication**: agent designs are now prefix-layout–aware by default; the mantra is "stable content first, varying content last." Equivalent features exist across OpenAI and Gemini.

### Context editing and tool-context management

Claude API shipped explicit context management: `clear_thinking_20251015` beta header for automatic thinking-block clearing, context editing for trimming stale `tool_result` blocks, and `tool_search` for lazy tool loading past ~20 tools. ([Claude release notes](https://platform.claude.com/docs/en/release-notes/overview), [manage tool context](https://platform.claude.com/docs/en/agents-and-tools/tool-use/manage-tool-context)). **Implication**: context engineering became a first-class API surface, not a client-side hack.

### Disaggregated prefill/decode moved to production

DistServe (OSDI 2024) → NVIDIA Dynamo (GTC 2025) → Mooncake, LMCache, NIXL. Production deployments at Meta (vLLM with Meta's routing), Perplexity, Mistral, LinkedIn. SGLang's DeepSeek-R1 PD deployment on 96 H100s hit 52.3k input / 22.3k output TPS per node; a GB200 NVL72 follow-up reached 3.8× prefill / 4.8× decode throughput vs. H100. vLLM llm-d 0.3 reached ~2.2k tokens/s per H200 with 32-way EP ([Hao AI Lab retrospective, Nov 2025](https://haoailab.com/blogs/distserve-retro/)). **Implication**: staff-level inference interviews now expect fluency in TTFT/TPOT SLAs, KV transfer bandwidth budgets, and PD ratio sizing.

### KV-cache compression and tiering maturing

Quantized KV (int8 standard, int4 experimentally; KIVI, KVQuant), eviction-based compression (H2O, SnapKV, StreamingLLM with attention sinks), tiered KV storage (LMCache, Mooncake KV store). MLA (multi-head latent attention, DeepSeek-V3) demonstrated that low-rank KV compression can be learned rather than bolted on. **Implication**: KV bytes/token varies by an order of magnitude across 2025 architectures; naive capacity planning (based on Llama-2 MHA numbers) overprovisions wildly for MLA/GQA models.

### Sparse / structured / hybrid attention momentum

Jamba (Mamba + Transformer + MoE, AI21, 2024), Griffin / RecurrentGemma (Google DeepMind, 2024), DeepSeek V3/R1 (MLA), and a flurry of hybrid blocks in 2025. The practical picture: pure SSMs still trail attention on retrieval; hybrids get the sequence-length cost reduction without the retrieval regression. **Implication**: "long-context model" ≠ "full-attention Transformer" anymore; a serving system needs to handle heterogeneous architectures.

### Context engineering as a design discipline

2025 saw explicit framing (in Anthropic, Google, OpenAI literature, plus public agent system writeups) of context engineering as distinct from prompt engineering — with pipelines, memory architectures, and observability. Agent memory surveys proliferated. **Implication**: expect interview questions that go beyond "design prompt X" into "design the memory and context pipeline for a long-running agent."

### Benchmark evolution

- NIAH saturated for frontier models — no longer useful for discrimination.
- RULER / ∞Bench started showing anomalous orderings (Gemini Flash > Pro, 70B Llama < 8B), suggesting their own calibration issues ([HELMET paper](https://arxiv.org/pdf/2410.02694)).
- HELMET (ICLR 2025) became the best-supported comprehensive long-context eval; HELM Long Context (Stanford, Sept 2025) followed.
- Task-specific benchmarks (LongCodeBench for code at 1M, LongGenBench for long generation, NoCha for books, NoLiMa for reasoning beyond literal match) fill the gap.

**Implication**: "we score X on NIAH" is a weak answer. Strong answers reference HELMET or task-specific evals with explicit length sweeps.

### ProLong and effective-length training practice

Work like ProLong (Gao et al., 2025, ACL) demonstrated training recipes that produce models with genuinely stronger effective context at 128K+, using HELMET subsets for model development. Tricks: RoPE base tuning, cross-document attention masking, targeted long-context SFT. **Implication**: the gap between nominal and effective length is narrowing for labs that train seriously, but it still exists.

---

## 18. Interview reasoning patterns

The move-by-move for likely questions.

### "Why is long context hard?"

Four-layer answer:
1. **Modeling**: O(n²) attention, position encoding extrapolation, attention dilution, lost-in-middle — nominal ≠ effective.
2. **Systems**: KV cache dominates HBM (hundreds of GB at 1M for 70B-class); prefill is compute-bound; decode is bandwidth-bound.
3. **Quality**: failures under noise and position bias; eval is hard beyond NIAH.
4. **Cost**: $ per request scales steeply; cache hit rate becomes a first-order metric.

### "Walk me through the bottlenecks from request arrival to first token."

```
Arrival → queue → tokenize → cache lookup → prefill → first token
```

For each stage, say what dominates at long context:

- Queue: scheduler admission, KV budget check.
- Tokenize: trivial on host, may add 10-50ms at 1M.
- Cache lookup: prefix-hash match against RadixTree / prefix store.
- Prefill: compute-bound; chunked; may be disaggregated.
- First token: for PD disaggregated, KV transfer cost; for co-located, immediate.

Call out which stage is dominant for which length.

### "Why does long-context serving become memory-bound?"

KV cache grows linearly in n; decode reads all of it per step. At 1M tokens for Llama-70B, KV is 320 GB; even at H100's 3.35 TB/s HBM, reading that per decode step takes ~100ms — which sets the floor for decode latency regardless of FLOPs available. Compute is underutilized in decode because arithmetic intensity ≈ 1 FLOP/byte. This is also why GQA/MLA and KV quantization matter so much.

### "When is bigger context right vs. retrieval?"

- **Bigger context**: task needs joint reasoning over all tokens; corpus fits; freshness is per-request; prefix-cacheable; latency budget accommodates prefill.
- **Retrieval**: corpus >> window; queries are specific enough to retrieve well; freshness matters; cost per query is bounded; you can invest in retrieval eval.
- **Both**: everything interesting. Retrieval feeds into a long context window, cached aggressively, with compaction for conversation state and external memory for cross-session durability.

Don't pick one.

### "Design your KV-cache subsystem."

Walk through:
1. **Per-token size** formula and GQA/MLA selection.
2. **Paging**: block-based allocation with block table, prefix sharing via copy-on-write.
3. **Prefix cache**: RadixTree keyed by token sequences, LRU eviction, tiered (HBM → DRAM → NVMe).
4. **Scheduler**: admission check against KV budget; cache-aware routing to warm prefixes.
5. **Compression/quantization**: optional int8 KV default; int4 under pressure.
6. **Offload / transfer**: NIXL-style point-to-point; layer-pipelined to hide transfer latency in PD disaggregation.
7. **Observability**: per-prefix hit rate, paging events, offload events, fragmentation metrics.

### "Support mixed 1K to 1M contexts in one system."

- PD disaggregation with separately sized prefill and decode pools, so long prefills don't block short decodes.
- Chunked prefill to bound per-step work and allow interleaving.
- Admission by KV budget, not request count.
- Cache-aware routing so long prompts land on workers with warm prefixes.
- Priority tiers: short requests preempt prefill chunks of long requests, bounded by fairness policies.
- Monitoring: separate p50/p95/p99 latencies by length bucket.

### "Manage context for a long-running coding agent."

- Layered prompt: stable system/tools/CLAUDE.md (1h cache) → retrieved files per turn → rolling state object → recent tool results → user message.
- Retrieval: repo-aware, query-rewritten, reranked.
- Compaction: milestone-based (on plan completion), preserves cache prefix + state object, writes dropped turns to episodic store.
- Subagent decomposition for large subtasks; orchestrator holds plan.
- Memory: user preferences in semantic store; episodic log of past tool calls for resumable behavior.
- Tool schema caching; tool search if toolset large.
- Observability: token accounting by layer, cache hit rate, compaction rate, memory write/read rates.

### "What breaks first?"

Depends on operating point, but in order of likelihood:

1. Prefix cache hit rate drops (often a bug — timestamp drift, tool list churn, non-deterministic template).
2. KV HBM pressure → admission stalls → queue depth grows → tail TTFT explodes.
3. One big request (1M prefill) stalls behind short requests and inflates tail.
4. Retrieval quality regresses silently — task accuracy drops without any infra alert.
5. Compaction starts dropping critical facts — agent quality regresses.

### "How do you know the design works?"

- Task-specific eval, not NIAH alone.
- Position sweeps to measure lost-in-the-middle for your workload.
- Length sweeps at 32K / 128K / 512K / 1M.
- Noise-injection eval.
- Production replay with gold answers.
- Cache hit rate, TTFT/TPOT p99, KV occupancy, memory store growth — all dashboarded.

### "What recent developments changed the design space?"

- Million-token windows in production.
- Prompt caching and context editing as API-level primitives.
- PD disaggregation standard.
- KV quantization / MLA cutting KV bytes per token by 2–8×.
- HELMET-era eval replacing NIAH-dominant benchmarking.
- Context engineering as an explicit discipline.

---

## 19. Common mistakes

- **Treating nominal context = effective context**. Always ask "at what length does quality degrade, and on what task?"
- **"Just use RAG"** without engaging with retrieval recall, ranking, packing, freshness, eval.
- **Ignoring prefill cost**. "We support 1M context" is meaningless if TTFT is 90s.
- **Ignoring KV-cache memory growth**. Calculate it in the room. GQA, MLA, quantization options.
- **Ignoring fragmentation / scheduling**. Paged KV, chunked prefill, PD disaggregation are not optional at scale.
- **Assuming long context eliminates memory systems**. Long context ≠ cross-session memory ≠ scalable corpora. They solve different problems.
- **Hand-waving summarization quality loss**. Compaction is lossy and errors compound; evaluate it.
- **Ignoring stale or poisoned memory**. Provenance, TTL, audit — not optional for production.
- **Confusing conversation history with useful context**. History is not state. State is state. Keep state objects.
- **Ignoring observability and cost**. Token accounting by layer, cache hit rate, per-request traces.
- **Treating NIAH pass as long-context success**. NIAH is saturated; use HELMET or task-specific evals.
- **Static toolset assumption**. Cache-aware designs break if tool list isn't stable.
- **Not thinking about the breakpoint**. Prefix cache writes happen at the cache_control marker; placement is a design decision.

---

## 20. Final cheat sheet

### 20.1 Big context vs. RAG vs. compression vs. memory

| | Big context | RAG | Compression | Memory |
|---|---|---|---|---|
| Core | Fit in window | Retrieve subset | Summarize old | External store |
| Win when | Joint reasoning | Large corpus | Long conversation | Cross-session |
| Cost | High per call | Low per call | Low per call | Low per call |
| Freshness | Per-request | Re-indexed | Per-compaction | Per-write |
| Main failure | Lost in middle, $$ | Retrieval miss | Cumulative error | Poisoning, staleness |
| Eval | Task at length | Recall@k + task | Compaction diff | Memory replay |

### 20.2 Prefill bottlenecks vs. decode bottlenecks

| | Prefill | Decode |
|---|---|---|
| Phase cost | O(n²d) attention + O(nd²) FFN | O(Nd) attention per step |
| Bound by | Compute / tensor cores | HBM bandwidth |
| Arithmetic intensity | High (scales with n) | Low (~1 FLOP/byte) |
| Kernel focus | FlashAttention-3, TMA, WGMMA | KV layout, paging, quantization |
| Scaling lever | Tensor parallel, chunked prefill, ring attn | KV offload, quantization, speculative decoding |
| SLA metric | TTFT | TPOT / ITL |
| Optimal GPU profile | Compute-rich | HBM-rich |

### 20.3 Working vs. episodic vs. semantic memory

| | Working | Episodic | Semantic |
|---|---|---|---|
| Contents | Live state, recent turns | Past sessions, tool traces | Preferences, learned facts |
| TTL | Seconds to minutes | Days to months | Months to indefinite |
| Update | Every turn | On session events | On consolidation |
| Storage | In-prompt or scratchpad | Vector + metadata store | Vector + structured KV |
| Retrieval | Always (in prompt) | Query-triggered | Query-triggered or always |
| Compaction | Reactive under pressure | Periodic consolidation | Rare / curator-managed |
| Poisoning risk | Low (short-lived) | Medium | High (persistent) |

### 20.4 Long-context failure modes and mitigations

| Failure | Mitigation |
|---|---|
| Lost in the middle | Place critical evidence at boundaries; rerank; hierarchical packing |
| Retrieval miss | Query rewriting; hybrid BM25+dense+rerank; multiple retrievals |
| Attention dilution | Shorter effective context; pruning irrelevant chunks |
| Distraction | Relevance threshold; drop low-score chunks even if budget allows |
| Instruction drift | Trusted vs. untrusted content separation; output constraints |
| Compounding summarization | Milestone (not uniform) summaries; state object preservation |
| Context poisoning | Provenance tracking; review before semantic-memory promotion |
| Stale memory | TTL; re-verification on read for sensitive facts |
| Prefix cache miss | Audit timestamp/template drift; lock tool lists; freeze prefix |
| KV HBM pressure | Admission; paging; quantization; offload to DRAM/NVMe |
| Long-prefill head-of-line | Chunked prefill; PD disaggregation; SPF scheduling |
| Token budget blowout | Per-user quotas by token (not request); asymmetric priorities |

### 20.5 Decision framework for system design

Given a long-context system design question, walk this order:

1. **Workload shape**: lengths distribution (short? long? mixed?); prefill:decode ratio; prefix-sharing patterns; multi-turn vs. one-shot.
2. **SLA**: TTFT target, TPOT target, throughput target, cost target.
3. **Model choice**: architecture (full attention? hybrid?), GQA/MLA, trained effective length, KV bytes per token.
4. **Memory math**: per-request KV; aggregate KV at target batch; offload thresholds.
5. **Serving architecture**: co-located vs. PD disaggregated; chunked prefill; continuous batching; prefix caching.
6. **Context strategy**: what's in-prompt, what's retrieved, what's compacted, what's in external memory.
7. **Eval**: task-anchored; length sweeps; position sweeps; noise injection; cache hit rate.
8. **Observability & cost**: token accounting by layer; per-request traces; store growth; cost dashboards.
9. **Failure modes**: what breaks first; mitigation for top 3; blast radius for bad writes.
10. **Recent developments**: cite 2–3 production systems/papers to anchor choices.

### 20.6 Twenty likely interview questions with short strong answers

**1. Why does decode become memory-bound at long context?**
Each decode step reads the entire KV cache; AI ≈ 1 FLOP/byte; at 1M tokens for 70B-class, KV alone is 100s of GB — HBM bandwidth sets the latency floor, not FLOPs.

**2. Prefill vs. decode: where do you spend your engineering budget?**
Prefill: kernel optimization (FlashAttention-3), chunked prefill, tensor/sequence parallelism. Decode: KV layout, paging, quantization, speculative decoding. In PD-disaggregated systems, size the two pools separately.

**3. Explain chunked prefill and what it buys you.**
Split prefill into fixed-size chunks (e.g., 4K tokens) processed sequentially. Bounds per-step work; allows interleaving decode steps of other requests; reduces TPOT tail caused by long prefills.

**4. What does disaggregated PD give you that chunked prefill doesn't?**
Separate compute pools sized to prefill:decode ratio; different GPU profiles (compute-rich vs. HBM-rich); eliminates interference entirely. Costs: KV transfer bandwidth.

**5. How does prefix caching work and why is it so valuable?**
Cache KV per prefix, keyed by token sequence. On repeat requests, skip prefill for matched prefix and only prefill the suffix. Cuts TTFT by 5–10× on warm caches. Implemented as RadixAttention (SGLang) or API-level (`cache_control`).

**6. What invalidates prefix cache?**
Any byte change in prefix: timestamps, updated tool schemas, reordered blocks, non-deterministic templates. Keep prefix stable; varying content at the suffix; stable tool lists.

**7. What's GQA and why does it matter for long context?**
Grouped-query attention: many Q heads share each KV head. Cuts KV bytes per token by the sharing ratio (e.g., 8× for Llama-3-70B). Makes long context economically serveable without quality regression.

**8. What's MLA and how is it different from GQA?**
Multi-head latent attention (DeepSeek): cache a low-rank latent per token, reconstruct K/V on demand. Reduces KV bytes further (often another 2–3× beyond GQA). Requires model-side training change.

**9. Why extend context with YaRN instead of Position Interpolation?**
Position Interpolation uniformly compresses all RoPE frequency bands, hurting high-frequency (local) information. YaRN scales per-band so high frequencies stay intact and only low-frequency (long-range) bands are compressed.

**10. What's lost-in-the-middle and how do you mitigate it?**
U-shape accuracy curve: information at prompt start or end is used more reliably than middle. Mitigate with placement (evidence at boundaries), reranking, hierarchical packing, or training targeted at position robustness.

**11. What's an attention sink?**
Initial tokens attract disproportionate attention mass, often acting as a "default" bucket. StreamingLLM exploits this: keep attention sinks + sliding window to handle unbounded streams without full KV.

**12. When would you use RAG instead of a longer context window?**
Corpus >> window; freshness required; specific queries (retrievable well); cost sensitivity. Not when evidence is diffuse across many chunks, or queries don't retrieve well.

**13. When would you use a longer window instead of RAG?**
Tasks with joint reasoning over full input (codebase refactors, long doc analysis), prefix-cacheable workloads, no meaningful retrieval signal.

**14. How do you evaluate a long-context production system?**
Task-anchored eval at realistic lengths; position and noise sweeps; HELMET or task-specific benchmarks (not NIAH alone); dashboard p99 TTFT/TPOT by length; production replay with gold answers.

**15. Agent keeps forgetting user preferences. Where do you look?**
Memory architecture. Preferences belong in semantic memory with long TTL and read-on-session-open. If they're in working memory only, compaction will drop them.

**16. How do you design context for a long-running coding agent?**
Layered prompt: cached system/tools/CLAUDE.md; retrieved files per turn; rolling state object; compact on milestones; write dropped turns to episodic store; semantic memory for user/repo-level preferences; subagents for large subtasks with summaries propagated up.

**17. Walk through the KV cache math for Llama-3-70B at 128K.**
per-token KV = 2 × 80 layers × 8 KV heads × 128 head_dim × 2 bytes = 320 KB. 128K tokens = 40 GB. One request fits per H100 with ~40 GB headroom for weights; batch size > 1 requires PD disaggregation or offload.

**18. What's the role of speculative decoding at long context?**
Shifts some decode work off-GPU to a small draft model; accepts multiple tokens per step on agreement. Helps TPOT when decode is bandwidth-bound. Less leverage at very long context where KV read per step dominates regardless of how many tokens you accept; still helps, just less.

**19. What changed in 2025 for long-context serving?**
PD disaggregation became standard (Dynamo, vLLM, SGLang); prompt caching GA with 1h TTL; context editing and tool-search as APIs; million-token windows in production (Gemini 2.5 Pro, Claude Sonnet 4); HELMET replaced NIAH as primary eval; MLA demonstrated materially smaller KV.

**20. What's the first thing you check if effective context quality regresses?**
Prefix cache hit rate (silent bugs from template drift); compaction frequency and whether state object is preserved; retrieval recall on a gold set; length-stratified eval by position of evidence. Then dashboards on KV occupancy and tail latency — quality regressions often correlate with scheduler stress.

---

## Key references

**Foundations**
- Vaswani et al., 2017. *Attention Is All You Need.*
- Dao et al., 2022; Dao, 2023; Shah et al., 2024. FlashAttention-1/2/3.
- Su et al., 2021. *RoFormer: RoPE.*
- Ainslie et al., 2023. *GQA.*
- DeepSeek, 2024. *DeepSeek-V2/V3* (MLA).

**Context extension**
- Press et al., 2022. *ALiBi.*
- Chen et al., 2023. *Position Interpolation.*
- Peng et al., 2023. *YaRN.*
- Xiao et al., 2023. *StreamingLLM / Attention Sinks.*

**Serving systems**
- Yu et al., 2022. *Orca* (continuous batching).
- Kwon et al., 2023. *PagedAttention / vLLM.*
- Zheng et al., 2024. *SGLang / RadixAttention.*
- Agrawal et al., 2024. *Sarathi-Serve* (chunked prefill).
- Zhong et al., 2024. *DistServe* (PD disaggregation).
- Hao AI Lab, 2025. *Disaggregated Inference: 18 Months Later.*
- vLLM docs, 2025–2026. Disaggregated prefilling.
- SGLang repo and blog posts, 2024–2025.

**Long-context models**
- Gu & Dao, 2023. *Mamba.*
- AI21, 2024. *Jamba.*
- Google DeepMind, 2024. *Griffin / RecurrentGemma.*
- Liu et al., 2023. *Ring Attention.*
- Gao et al., 2025 (ACL). *ProLong: How to Train LCLMs Effectively.*

**KV optimization**
- Liu et al., 2024. *KIVI.*
- Zhang et al., 2023. *H2O.*

**Quality and eval**
- Liu et al., 2023. *Lost in the Middle.*
- Kamradt, 2023. *Needle in a Haystack.*
- Hsieh et al., 2024. *RULER.*
- Yen et al., 2025 (ICLR). *HELMET.*
- Karpinska et al., 2024. *NoCha.*
- Adobe Research. *NoLiMa.*
- Stanford CRFM, 2025. *HELM Long Context.*

**Product / platform**
- Anthropic Claude docs: [prompt caching](https://platform.claude.com/docs/en/build-with-claude/prompt-caching), [manage tool context](https://platform.claude.com/docs/en/agents-and-tools/tool-use/manage-tool-context), [release notes](https://platform.claude.com/docs/en/release-notes/overview).
- Google Gemini long context docs.
- OpenAI API prompt caching docs.

---

*End of primer.*
