---
title: "Attention Optimization: MQA, GQA, MLA, Sparse Attention, and Long Context"
description: "A staff-level guide to attention optimization for LLMs: MHA, MQA, GQA, MLA, sparse attention, sliding windows, KV cache cost, long-context inference, and production tradeoffs."
---

# Attention Optimization: MQA, GQA, MLA, Sparse Attention, and Long Context

Attention optimization is one of the highest-leverage areas in LLM systems because attention controls two expensive things at once:

1. **Compute:** how many query-key interactions are evaluated.
2. **Memory:** how much key-value state must be cached during decoding.

For short prompts, dense attention may not dominate. For long-context serving, multi-turn chat, agents, RAG, and code workloads, attention and KV cache design become first-order product constraints. The optimization question is not just "can we make attention faster?" It is:

> Which attention representation lets us preserve model quality while reducing KV cache size, memory bandwidth, and long-context compute?

This article covers Multi-Query Attention (MQA), Grouped-Query Attention (GQA), Multi-head Latent Attention (MLA), sparse attention, sliding windows, and modern long-context production reality.

---

## 1. The Interview Mental Model

When attention optimization comes up, answer in this order:

1. **Phase:** Is the bottleneck prefill, decode, or long-context retrieval?
2. **Resource:** Are we constrained by FLOPs, HBM capacity, memory bandwidth, interconnect, or latency tail?
3. **KV cache shape:** How many key-value heads are cached per layer?
4. **Attention pattern:** Dense, local, block sparse, learned sparse, or compressed latent?
5. **Quality risk:** Does the method change what tokens can attend to, or only how KV is represented?
6. **Runtime support:** Are there kernels for the exact pattern?

Useful decision tree:

```text
Attention bottleneck
    |
    +-- Decode memory bandwidth / KV cache size
    |       |
    |       +-- MQA / GQA / MLA
    |
    +-- Long-context prefill or retrieval cost
    |       |
    |       +-- sparse attention / sliding window / block sparse / DSA
    |
    +-- Kernel IO bottleneck in dense attention
            |
            +-- FlashAttention / FlashMLA / FlashInfer-style kernels
```

MQA, GQA, and MLA mostly attack KV cache and decode bandwidth. Sparse attention attacks the number of attended positions. FlashAttention-style kernels attack memory movement for dense attention.

---

## 2. Baseline: Multi-Head Attention

For an input sequence $X$, attention forms queries, keys, and values:

$$
Q = XW_Q,\quad K = XW_K,\quad V = XW_V
$$

Scaled dot-product attention is:

$$
\text{Attention}(Q,K,V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_h}}\right)V
$$

In multi-head attention (MHA), each attention head has its own query, key, and value projections:

```text
Head 1: Q1 K1 V1
Head 2: Q2 K2 V2
Head 3: Q3 K3 V3
...
Head H: QH KH VH
```

During autoregressive decoding, each new token attends to all previous tokens. The model caches previous keys and values so it does not recompute them.

Approximate KV cache size per layer:

$$
\text{KV bytes} = 2 \cdot B \cdot L \cdot H_{kv} \cdot d_h \cdot \text{bytes\_per\_element}
$$

where:

- $B$ is batch size.
- $L$ is sequence length.
- $H_{kv}$ is number of KV heads.
- $d_h$ is head dimension.
- The factor 2 is for keys and values.

In standard MHA:

$$
H_{kv} = H_q
$$

where $H_q$ is the number of query heads. This is expensive for long context and high concurrency.

---

## 3. Multi-Query Attention

Multi-Query Attention, proposed in **[Fast Transformer Decoding: One Write-Head is All You Need](https://arxiv.org/abs/1911.02150)**, shares one key head and one value head across all query heads.

```text
MHA:
  Q1 K1 V1
  Q2 K2 V2
  Q3 K3 V3
  Q4 K4 V4

MQA:
  Q1 \
  Q2  \
  Q3   -> shared K, shared V
  Q4  /
```

In MQA:

$$
H_{kv} = 1
$$

instead of:

$$
H_{kv} = H_q
$$

KV cache reduction:

$$
\text{reduction} \approx \frac{H_q}{1}
$$

If a model has 32 query heads, MQA can reduce KV cache size by roughly 32x for attention state. That is enormous for decode.

Why it helps:

- Less KV cache memory.
- Less HBM bandwidth during decode.
- Higher batch/concurrency before memory fills.
- Better long-context serving economics.

Tradeoff:

- Sharing one KV head can reduce representational capacity.
- Quality can degrade compared with full MHA.
- Retrofitting an MHA checkpoint into MQA usually needs uptraining.

MQA is a decode optimization first. It does not remove the need to compute attention scores over the context.

---

## 4. Grouped-Query Attention

Grouped-Query Attention (GQA), from **[GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints](https://aclanthology.org/2023.emnlp-main.298/)**, is the compromise between MHA and MQA.

Instead of one KV head for all query heads, GQA uses $G$ KV groups:

```text
GQA with 8 query heads and 2 KV groups:

  Q1 Q2 Q3 Q4 -> K1 V1
  Q5 Q6 Q7 Q8 -> K2 V2
```

In GQA:

$$
1 < H_{kv} < H_q
$$

KV cache reduction:

$$
\text{reduction} \approx \frac{H_q}{H_{kv}}
$$

If $H_q = 32$ and $H_{kv} = 8$, KV cache is reduced by about 4x relative to MHA.

GQA became common because it gives much of MQA's serving benefit with less quality risk. Many modern open LLMs use GQA rather than full MHA.

Interview phrase:

> MQA minimizes KV cache. GQA buys back quality by using several KV groups. It is the production compromise.

---

## 5. Multi-Head Latent Attention

Multi-head Latent Attention (MLA), introduced in DeepSeek-V2 and used in DeepSeek-V3, attacks the KV cache differently. Instead of caching full per-head keys and values, MLA compresses key-value information into a latent representation and reconstructs what is needed for attention.

The high-level idea:

```text
Standard KV cache:
  cache K per layer/head/token
  cache V per layer/head/token

MLA:
  cache compact latent vector per token
  reconstruct projected K/V-like quantities when needed
```

A simplified view:

$$
c_t = W_{DKV} h_t
$$

where $c_t$ is a compressed latent KV vector for token $t$. Later projections produce key/value components from this latent state:

$$
k_t = W_{UK} c_t,\quad v_t = W_{UV} c_t
$$

The actual DeepSeek implementation is more nuanced, especially around RoPE dimensions and query/key decomposition, but the optimization principle is simple:

> Cache a smaller latent representation instead of full expanded KV tensors.

Why MLA matters:

- It reduces KV cache footprint.
- It reduces long-context memory pressure.
- It is architecture-level, not just a serving trick.
- It pairs naturally with MoE, where active compute is controlled but KV cache can still be large.

DeepSeek-V3's technical report states that the model uses MLA for efficient inference and DeepSeekMoE for economical training. DeepSeek-V3.1 and later variants continued this design line. DeepSeek-V3.2, released in December 2025, added DeepSeek Sparse Attention for long-context efficiency while keeping the broader efficiency-focused architecture.

Tradeoffs:

- MLA is not a drop-in runtime flag for arbitrary checkpoints.
- Retrofitting MHA/GQA models into MLA is possible research-wise, but requires careful conversion and tuning.
- Kernels matter: efficient MLA serving needs attention kernels that understand the latent cache layout.
- Debugging is harder because KV cache no longer has the same simple per-head interpretation.

Staff-level answer:

> GQA reduces the number of KV heads. MLA changes what we cache. Sparse attention changes which tokens we attend to. These are different levers.

---

## 6. Sparse Attention

Dense attention lets every token attend to every previous token. For sequence length $L$:

$$
\text{attention scores} \sim O(L^2)
$$

Sparse attention restricts the attention pattern. If each token attends to only $w$ relevant positions:

$$
\text{attention scores} \sim O(Lw)
$$

where $w \ll L$.

Patterns:

- Sliding window.
- Global tokens.
- Block sparse.
- Retrieval-selected tokens.
- Learned top-k sparse attention.
- Hybrid dense local + sparse global.

```text
Dense causal attention:

  #...............
  ##..............
  ###.............
  ####............
  #####...........

Sliding window:

  #...............
  ##..............
  ###.............
  .###............
  ..###...........

Sparse global + local:

  #...............
  ##..............
  ###.............
  #.###...........
  #..###..........
```

Sparse attention is attractive for long context because the dense $L^2$ term becomes the enemy. Longformer and BigBird are classic examples. More recently, DeepSeek-V3.2 introduced **DeepSeek Sparse Attention (DSA)** for long-context efficiency, using a learned sparse attention mechanism designed to reduce complexity while preserving model performance. DeepSeek released V3.2-Exp in September 2025 as an experimental sparse-attention model, then DeepSeek-V3.2 in December 2025 with DSA as one of the key technical pieces.

Production reality:

- Sparse attention must be trained or adapted into the model.
- A dense attention kernel with a mask is not enough.
- The sparse pattern must map to efficient kernels.
- Sparse attention can hurt exact recall if important tokens are skipped.
- Evaluation must include long-context retrieval, code, agents, and RAG.

---

## 7. Sliding Windows and Attention Sinks

Sliding-window attention keeps only a local context window:

$$
\text{Attend}(t) = \{t-w, ..., t-1\}
$$

This makes decode and prefill more manageable for very long sequences, but pure sliding windows can lose global information.

Attention sinks preserve a small set of early tokens or special tokens that many later tokens can attend to:

```text
Token t attends to:
  - recent local window
  - sink tokens near beginning
  - optional global/retrieved tokens
```

This is useful when the model relies on early anchor tokens or global state. Long-context systems often combine:

- Local sliding window.
- Global summary tokens.
- Retrieval.
- Context compression.
- Sparse attention.
- KV cache eviction.

No single trick solves long context. It is a stack.

---

## 8. Prefill vs Decode

Attention optimizations affect prefill and decode differently.

Prefill:

- Processes all prompt tokens.
- Large dense matrix multiplications.
- Attention score matrix can be large.
- Compute-bound for long prompts.
- FlashAttention-style kernels matter.
- Sparse attention can reduce $L^2$ work.

Decode:

- Generates one or a few tokens at a time.
- Reads KV cache for all previous tokens.
- Often memory-bandwidth-bound.
- MQA/GQA/MLA reduce KV bandwidth.
- Speculative decoding can reduce number of serial steps.

```text
Optimization          Prefill impact        Decode impact
----------------------------------------------------------
MQA                  modest                high
GQA                  modest                high
MLA                  modest/high           high
FlashAttention       high                  medium
Sparse attention     high on long ctx      high if KV reads shrink
Sliding window       high on long ctx      high
```

This table is why "attention optimization" needs a workload. The right answer for 2K-token chat is not the same as 1M-token document analysis.

---

## 9. KV Cache Math

Suppose:

- Batch size $B = 32$
- Sequence length $L = 32{,}768$
- Layers $N_L = 80$
- Query heads $H_q = 64$
- KV heads $H_{kv} = 8$
- Head dimension $d_h = 128$
- BF16 cache, 2 bytes per element

KV cache:

$$
2 \cdot B \cdot L \cdot N_L \cdot H_{kv} \cdot d_h \cdot 2
$$

The first factor 2 is keys plus values. Plugging in:

$$
2 \cdot 32 \cdot 32768 \cdot 80 \cdot 8 \cdot 128 \cdot 2
$$

This is hundreds of GB of KV cache. That is why KV cache design is not a detail. It determines how many concurrent long-context users fit on the serving fleet.

Reducing $H_{kv}$ from 64 to 8 cuts KV cache by 8x. Compressing KV into MLA-style latents can reduce it further depending on latent dimension. Sparse attention or sliding windows can reduce which cached tokens must be read.

---

## 10. Production Reality in 2025-2026

The direction of travel is clear:

- **GQA is mainstream** for efficient dense LLM serving.
- **MLA is a serious architecture-level alternative**, validated publicly through DeepSeek-V2/V3/V3.1 and used in later DeepSeek models.
- **Sparse attention is becoming production-relevant for long context**, with DeepSeek-V3.2-Exp and V3.2 making learned sparse attention a prominent open-model example in late 2025.
- **Kernel support is decisive.** FlashAttention, FlashInfer, FlashMLA, TileLang/Triton kernels, and engine integration determine whether attention changes pay off.
- **Long-context economics drive architecture.** A model can be strong but too expensive to serve if its KV cache and attention pattern scale poorly.

Recent model examples:

- DeepSeek-V3 uses MLA and DeepSeekMoE.
- DeepSeek-V3.1 continued the V3 line and appeared in August 2025 model releases.
- DeepSeek-V3.2-Exp, released September 29, 2025, introduced DSA for long-context cost reduction.
- DeepSeek-V3.2, released December 1, 2025, made DSA part of the successor model line.
- Many dense open models use GQA because it is the practical compromise between MHA quality and MQA memory savings.

The lesson is not "copy DeepSeek." The lesson is that attention architecture is now a cost-control surface, not just a modeling detail.

---

## 11. Failure Modes

### Reducing KV heads hurts quality

MQA or aggressive GQA can lose attention diversity. Uptraining or distillation may be needed.

### Sparse attention misses important tokens

Long-context tasks often depend on rare but crucial tokens. Sparse selection must be evaluated on retrieval-heavy tests.

### Kernel mismatch erases gains

The model uses an efficient attention pattern, but the serving engine falls back to dense kernels or inefficient masking.

### Decode improves but prefill does not

MQA/GQA reduce KV cache bandwidth during decode but do not remove all prefill compute.

### Long-context benchmarks are too weak

Needle tests are useful but insufficient. Use RAG, multi-document QA, codebase navigation, agent traces, and production prompts.

### Output behavior changes

Attention changes can alter recall, verbosity, tool-use accuracy, and stop-token behavior even when aggregate benchmarks look fine.

---

## 12. Important Papers and Docs

1. **[Fast Transformer Decoding: One Write-Head is All You Need](https://arxiv.org/abs/1911.02150)** — Shazeer, 2019.  
   The MQA paper.

2. **[GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints](https://aclanthology.org/2023.emnlp-main.298/)** — Ainslie et al., 2023.  
   The key GQA paper and uptraining recipe.

3. **[DeepSeek-V2](https://arxiv.org/abs/2405.04434)** and **[DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437)**.  
   Read for MLA and MoE as a combined efficiency architecture.

4. **[Hardware-Centric Analysis of DeepSeek's Multi-Head Latent Attention](https://arxiv.org/abs/2506.02523)**.  
   Useful for understanding MLA from a hardware perspective.

5. **[DeepSeek-V3.2](https://arxiv.org/abs/2512.02556)**.  
   Read for DeepSeek Sparse Attention and late-2025 long-context production direction.

6. **[Longformer](https://arxiv.org/abs/2004.05150)** and **[BigBird](https://arxiv.org/abs/2007.14062)**.  
   Classic sparse-attention designs.

7. **FlashAttention papers and implementations.**  
   Essential for dense attention IO-aware optimization.

---

## 13. The Staff Engineer Summary

Attention optimization is about controlling memory and context cost without losing recall.

The checklist:

- Separate prefill from decode.
- Compute KV cache size explicitly.
- Understand whether the method changes KV representation, KV head count, or attention pattern.
- Use MQA/GQA for decode bandwidth and memory.
- Use MLA when architecture-level latent KV compression is available.
- Use sparse attention for long-context scaling, but only with matching kernels and evals.
- Validate long-context retrieval, tool use, code, and production prompts.
- Benchmark with real sequence lengths and concurrency.

The interview answer:

> MQA and GQA reduce how much KV we store. MLA changes what KV representation we cache. Sparse attention changes which tokens we attend to. FlashAttention changes how efficiently dense attention runs. The right choice depends on prefill vs decode, context length, kernel support, and quality risk.

