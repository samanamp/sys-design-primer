---
title: Multi-head Latent Attention (MLA)
description: Multi-head Latent Attention (MLA)
---

# Multi-head Latent Attention (MLA)

MLA (DeepSeek-V2/V3) attacks the dominant decode-time bottleneck — **KV cache bandwidth** — by caching a single low-rank latent vector per token instead of full per-head K and V tensors. The up-projection matrices are then **absorbed** into Q and the output projection at inference, so cache size collapses while attention stays mathematically equivalent to standard MHA (modulo a decoupled RoPE branch, explained below).

## Standard MHA — the baseline

```
X : (s, d)
   │
   ├── W_Q ──► Q : (s, h, d_h)
   ├── W_K ──► K : (s, h, d_h)   ← cached
   └── W_V ──► V : (s, h, d_h)   ← cached

attn = softmax(Q Kᵀ / √d_h) V          out = attn · W_O
```

Per-token KV cache, per layer: `2 · h · d_h` elements. DeepSeek-V2 has `h=128, d_h=128`, so in bf16 that's **64 KB / token / layer**. Across 60 layers and a 32K context that's ~120 GB — infeasible to serve.

Decode arithmetic intensity is brutal: each generated token streams the whole KV cache once for `2 · h · s · d_h` FLOPs. Roofline plants you firmly in BW-bound land.

## MLA — the construction

Cache one shared latent `c_KV ∈ R^{d_c}` per token, plus a small RoPE'd key `k_R ∈ R^{d_h^R}` shared across heads (decoupled because RoPE doesn't commute with the absorption trick).

```
                        cached
                          ▼
X ──► W_DKV ──► c_KV : (s, d_c)            d_c ≈ 4·d_h
X ──► W_KR  ──► k_R  : (s, d_h^R)          d_h^R = d_h/2,  shared across heads

K_C = c_KV · W_UKᵀ     ← never materialized at inference
V   = c_KV · W_UV      ← never materialized at inference

c_Q = X · W_DQ
Q_C = c_Q · W_UQ                  ← per-head content query
Q_R = RoPE(c_Q · W_QR)            ← per-head RoPE query
```

Score for head `i`, query `t`, key `j`:

```
score = Q_C[t,i] · W_UK[i] · c_KV[j]ᵀ  +  Q_R[t,i] · k_R[j]ᵀ
        └──────── absorbed ──────────┘    └── decoupled RoPE ──┘
```

Precompute `W_UQK[i] = W_UQ[i] · W_UK[i]ᵀ` once at load time. At decode you only read `c_KV` and `k_R`. Likewise `W_UV` folds into `W_O`, so the value path also never materializes V.

## Memory & bandwidth — why this matters

| Variant | Cache / token / layer (elements) | bf16 bytes (DeepSeek-V2 dims) |
|---|---|---|
| MHA      | `2 · h · d_h` = 32,768 | 64 KB |
| GQA-8    | `2 · 8 · d_h` = 2,048  | 4 KB |
| MQA      | `2 · d_h` = 256        | 0.5 KB |
| **MLA**  | `d_c + d_h^R` ≈ 4.5·d_h = 576 | **1.1 KB** |

MLA lands between MQA and GQA on cache size while preserving expressivity close to full MHA — V2's ablations show MLA *beating* MHA at iso-params, presumably because the latent bottleneck acts as a regularizer.

## Compute trade-off

MLA does *more* FLOPs than MHA at equal context. Each score now goes through the `d_c`-dim latent instead of `d_h`. Per-token decode attention FLOPs scale roughly as:

```
MHA: h · s · d_h         (Q·Kᵀ and attn·V, both head-local)
MLA: h · s · (d_c + d_h^R) ≈ h · s · 4.5·d_h
```

So ~**4.5× more attention FLOPs**. But decode is BW-bound, so this is the right trade. Cutting cache traffic ~57× while paying ~4.5× more FLOPs raises arithmetic intensity by an order of magnitude, pushing the kernel toward compute-bound territory where H100 / MI300 actually have headroom. Net effect: large-batch and long-context throughput jump substantially — which is what V3 and R1 ride on at serving time.

The remaining engineering knobs are familiar ones from FlashAttention: fuse the absorbed `W_UQK` matmul into the score computation, keep `c_KV` tiles in SMEM, and treat the decoupled RoPE branch as a small additive correction so it doesn't bloat register pressure.