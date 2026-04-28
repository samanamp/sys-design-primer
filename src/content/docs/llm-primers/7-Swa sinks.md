---
title: Multi-head Latent Attention (MLA)
description: Multi-head Latent Attention (MLA)
---
# Sliding Window Attention + Attention Sinks

Sliding Window Attention (SWA) caps each query's receptive field at the last `W` tokens, turning attention's KV cost from `O(s)` to `O(W)`. On its own, SWA breaks badly in streaming generation — perplexity explodes the moment the oldest tokens get evicted. **Attention sinks** fix this by permanently retaining the first few tokens (typically `k=4`) in the cache, restoring near-full quality at constant memory.

## Standard causal attention — the baseline

```
q_t attends to:
┌─────────────────────────┐
│ k_0  k_1  k_2 ...  k_t │   ← grows without bound
└─────────────────────────┘
            ▲
   cache = O(s) per layer
```

Per-query compute: `h · s · d_h`. Cache per layer: `2 · h · d_h · s`. Both linear in context — fatal for long-running streaming workloads.

## Sliding Window Attention (Longformer / Mistral)

```
              ┌─── window W ───┐
q_t  ────►   k_{t-W+1}  ...  k_t      (causal AND band)
```

Mask: `M[i,j] = 1  iff  i-W < j ≤ i`. Per-token compute: `h · W · d_h`, constant in `s`. Cache: rolling ring buffer, `2 · h · d_h · W` per layer.

Stacked SWA layers compose: layer `ℓ`'s effective receptive field is roughly `ℓ · W` (each layer pulls info one window deeper). That's why Mistral-7B with `W=4096` and 32 layers works at all on long inputs — the global path exists, just diluted through depth.

## The streaming failure mode

Pure SWA collapses once `s ≫ W`. Xiao et al. (StreamingLLM, 2024) isolated why: softmax forces the row to sum to 1, so when a head has nothing relevant to look at, it dumps mass somewhere benign. During training that "somewhere" becomes the first few tokens — every query sees them, so they accumulate large key norms and absorb leftover mass. Evict them and every softmax distribution gets distorted in ways that compound through layers.

```
attention mass per query (illustrative):

   ●         ●●●                    ← sinks: absorb leftover mass
   ●         ●●●●●●●●●●●●●●●●●      ← actual content attention
  ─┬─────────┬───────────────────►
   0 1 2 3                    t
   └─sinks─┘     └─── window ───┘
```

## SWA + Sinks — the construction

Keep the first `k` tokens permanently, plus the rolling window of `W`:

```
q_t attends to:  { k_0 ... k_{k-1} }  ∪  { k_{t-W+1} ... k_t }
                       sinks                     window

cache layout:
┌──────────────────┬─────────────────────────────┐
│ s_0 s_1 s_2 s_3  │  ... rolling W slots ...    │
│   permanent      │   evict-oldest-on-write     │
└──────────────────┴─────────────────────────────┘
```

Mask: `M[i,j] = 1  iff  j < k  OR  i-W < j ≤ i`.

Critical detail: **position encodings (RoPE) are assigned by index in the cache, not absolute position in the source sequence**. Otherwise rotated keys drift out of training distribution as the absolute index grows past anything seen in pretraining.

## Memory & bandwidth

`h=32, d_h=128, bf16, MHA, s=128K`:

| Variant       | Cache / layer       | Decode FLOPs / token / layer |
|---|---|---|
| Full attn     | `2·h·d_h·s` ≈ 1 GB  | `2·h·d_h·s` ≈ 1 GFLOP |
| SWA W=4096    | `2·h·d_h·W` ≈ 33 MB | `2·h·d_h·W` ≈ 33 MFLOP |
| SWA + 4 sinks | `2·h·d_h·(W+4)` ≈ 33 MB | ≈ 33 MFLOP |

Both cache footprint and per-token attention compute become **independent of context length beyond W**. That's the whole point: constant-state streaming generation, ~30× cheaper here, and the gap widens linearly with `s`.

## Trade-offs

You are not getting long context; you are getting *bounded-state streaming*. Information older than `W` that isn't encoded into the sinks is gone — needle-in-haystack beyond `W` fails. The effective global reach comes only from layer-stacking the windows, which constrains architecture.

Modern hybrids — Gemma 2, GPT-OSS, Qwen3-Next — alternate SWA layers with occasional full-attention (or linear/SSM) layers to preserve a true global path while keeping per-token cost near SWA. Sinks are now a default ingredient because they're free: no retraining required, retrofit at inference by simply not evicting positions `0..k-1`. Variants like learned sink tokens, or the "softmax-off-by-one" trick (Miller, 2023) that gives softmax an implicit zero-token escape valve, attack the same root cause without burning cache slots.