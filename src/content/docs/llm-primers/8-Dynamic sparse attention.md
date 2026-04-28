---
title: Dynamic Sparse Attention
description: Dynamic Sparse Attention
---

# Dynamic Sparse Attention

Where SWA fixes the sparsity pattern (band) and MLA shrinks per-token cache, **dynamic sparse attention** lets each query choose its own keys at runtime. The pattern is data-dependent. Modern instances — DeepSeek **NSA** (2025), Kimi **MoBA**, **Quest**, **H2O** — all share the same skeleton: cheaply estimate per-block importance, top-k select, attend densely within the selection.

## General structure

```
Step 1  SCORE       cheap proxy over all blocks   O(s/B) heavy
Step 2  SELECT      top-k blocks                  O((s/B) log k)
Step 3  ATTEND      dense softmax on selection    O(k·B)  ◄ the win
```

The win is Step 3: actual attention cost drops from `O(s)` to `O(k·B)` with `k·B ≪ s`. Steps 1–2 must be cheap enough not to eat the savings.

## Block-sparse top-k (NSA / MoBA flavor)

Partition the sequence into blocks of size `B`. For each query `q_t`, score every block, keep top-`k`, plus the current block for causal coverage:

```
seq:   [ B_0 ][ B_1 ][ B_2 ] ... [ B_{n-1} ][ q_t ]

        score(q_t, B_j)  ∀ j
                │
                ▼ top-k
        q_t attends to:
        ┌────────────────────────────────────────┐
        │ B_{i_1}  B_{i_2}  ...  B_{i_k}  B_cur │
        └────────────────────────────────────────┘
                (k+1) blocks × B tokens each
```

The design knob is *how* you score:

- **Quest** (inference-only): per-block elementwise `min`/`max` of K. Bound the max possible `q·k` in a block by `Σ_i max(q_i · k^max_i, q_i · k^min_i)`. Score = that upper bound.
- **MoBA** (Kimi): learned per-block score, gated against a sliding-window branch. Trained from scratch.
- **NSA** (DeepSeek): three parallel paths — *compressed* (learned block summaries), *selected* (top-k via compression-branch scores), *sliding* (recent `W`). Gate-mixed. Differentiable via straight-through on the gate.
- **H2O / SnapKV**: not block-based; track per-token accumulated attention as the score, evict permanently. Inference-only, lossy.

## Math vs full attention

```
Full:    o_t = Σ_{j ≤ t}     softmax(q_t · k_j / √d) · v_j
Sparse:  o_t = Σ_{j ∈ S(q_t)} softmax(q_t · k_j / √d) · v_j
```

**Critical: softmax must be renormalized over `S(q_t)` alone.** The denominator sums only over selected keys. Easy to get wrong in custom kernels — the FlashAttention online-softmax pattern still works, but the running max / running denominator must be initialized *after* the gather, not before. Mixing in unselected keys' contributions silently destroys scale.

## Memory & bandwidth (decode)

| Variant           | KV traffic / token / layer        | FLOPs / token / layer            |
|---|---|---|
| Full attn         | `2·h·d_h·s`                       | `2·h·d_h·s`                      |
| SWA, window W     | `2·h·d_h·W`                       | `2·h·d_h·W`                      |
| Dyn sparse, k·B   | `2·h·d_h·(k·B)` + score overhead  | `2·h·d_h·(k·B)` + score overhead |

NSA's reported config (`s=64K, k·B ≈ 4K`) drops the dense-attention path ~16× in both BW and FLOPs. Scoring overhead — block summaries or min/max bounds — adds back ~`s/B` work, typically 10–20% of the saved cost.

vs **SWA**: can route to *relevant* tokens anywhere in context — needle-in-haystack works, which pure SWA cannot do. vs **MLA**: orthogonal and composable; NSA layers GQA on top of its three-path structure, and you could in principle stack MLA-style latent compression underneath.

## Trade-offs

The kernel story is meaningfully harder. Block sparsity with data-dependent indices means **non-contiguous KV reads** — every query potentially touches a different set of blocks, breaking the contiguous-stream pattern FlashAttention is designed around. NSA and MoBA both ship custom kernels that gather selected blocks into SMEM before the dense step; that gather is real cost and dominates at small `k`. Warp-level coalescing falls apart unless you batch queries that happen to select overlapping blocks — which they often do, but the dispatcher logic is non-trivial.

Training is harder than SWA: selection is non-differentiable, requiring straight-through estimators or auxiliary losses to push gradients through the scorer. That's exactly why Quest / H2O / SnapKV exist as inference-only retrofits — they trade accuracy on long-context regimes the base model didn't train into for the ability to ship on top of any pretrained dense-attention model.

Current frontier: NSA-style trained block sparsity for >100K-context pretraining. Inference-only retrofits remain the practical path for taking older dense-attention models into long-context serving without retraining.