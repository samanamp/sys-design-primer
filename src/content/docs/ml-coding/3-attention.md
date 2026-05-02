---
title: Attention
description: Attention
---
# Reference Solution: Multi-Head Attention Forward + Backward in NumPy

## 1. Forward pass

```python
import numpy as np

def mha_forward(X, Wq, Wk, Wv, Wo, num_heads, mask=None):
    B, T, D = X.shape                                            # (B, T, D)
    H = num_heads
    assert D % H == 0, "num_heads must divide D"
    d_k = D // H
    scale = 1.0 / np.sqrt(d_k)

    # --- Linear projections ---
    Q = X @ Wq                                                   # (B, T, D)
    K = X @ Wk                                                   # (B, T, D)
    V = X @ Wv                                                   # (B, T, D)

    # --- Head split: reshape THEN transpose ---
    # Reshape splits the last dim into (H, d_k); transpose moves H next to batch.
    # Doing reshape(B, H, T, d_k) directly would interleave heads incorrectly.
    Qh = Q.reshape(B, T, H, d_k).transpose(1,2)           # (B, H, T, d_k)
    Kh = K.reshape(B, T, H, d_k).transpose(1,2)           # (B, H, T, d_k)
    Vh = V.reshape(B, T, H, d_k).transpose(1,2)           # (B, H, T, d_k)

    # --- Scaled dot-product scores ---
    # (B, H, T, d_k) @ (B, H, d_k, T) -> (B, H, T, T)
    S = Qh @ Kh.transpose(-1, -2) * scale               # (B, H, T, T)

    # --- Additive mask (broadcast across H) ---
    if mask is not None:
        if mask.ndim == 2:
            mask_b = mask[None, None, :, :]                      # (1, 1, T, T)
        elif mask.ndim == 4:
            mask_b = mask                                        # (B, 1, T, T)
        else:
            raise ValueError("mask must be 2D or 4D")
        S = S + mask_b                                           # (B, H, T, T)

    # --- Numerically stable softmax (row-wise max subtraction) ---
    S_max = S.max(axis=-1, keepdims=True)                        # (B, H, T, 1)
    expS = np.exp(S - S_max)                                     # (B, H, T, T)
    Z = expS.sum(axis=-1, keepdims=True)                         # (B, H, T, 1)
    P = expS / Z                                                 # (B, H, T, T)

    # --- Apply attention to V ---
    # (B, H, T, T) @ (B, H, T, d_k) -> (B, H, T, d_k)
    A = P @ Vh                                         # (B, H, T, d_k)

    # --- Head merge: transpose THEN reshape (inverse of head split) ---
    A_merged = A.transpose(1, 2).reshape(B, T, D)          # (B, T, D)

    # --- Output projection ---
    out = A_merged @ Wo                                          # (B, T, D)

    cache = dict(
        X=X, Wq=Wq, Wk=Wk, Wv=Wv, Wo=Wo,
        Qh=Qh, Kh=Kh, Vh=Vh, P=P, A_merged=A_merged,
        mask=mask, scale=scale, shape=(B, T, D, H, d_k),
    )
    return out, cache
```

**Cache contents and why:** `X` (for `dWq, dWk, dWv` and the three `dX` paths), `Wq/Wk/Wv/Wo` (for `dX_q, dX_k, dX_v` via `Wᵀ`), `Qh/Kh/Vh` (for `dP`, `dV`, `dQ`, `dK`), `P` (for softmax backward — we cache the *post-softmax* probabilities, not `S`, because the backward identity is in terms of `P`), `A_merged` (for `dWo`), `mask` (to re-zero gradients on masked positions), and `scale`.

## 2. Backward derivation

### 2.1 ∂L/∂Wo and ∂L/∂A_merged

`out = A_merged @ Wo`, both shapes `(B, T, D)` and `(D, D)`.

```
dWo        = sum_{b,t} A_merged[b,t,:]ᵀ · dout[b,t,:]   shape (D, D)
dA_merged  = dout @ Woᵀ                                  shape (B, T, D)
```

### 2.2 Through the head-merge reshape

The forward did `A.transpose(0,2,1,3).reshape(B, T, D)`. This is a *pure rearrangement* — no arithmetic, no broadcasting, no reduction. The backward of any pure rearrangement is its inverse rearrangement:

```python
dA = dA_merged.reshape(B, T, H, d_k).transpose(0, 2, 1, 3)       # (B, H, T, d_k)
```

Why this works: `out[b,t,h*d_k + j] = A[b,h,t,j]`. Differentiating, `∂out/∂A[b,h,t,j]` is a permutation matrix (1 in one place, 0 elsewhere). The JVP is just inverting the permutation.

### 2.3 ∂L/∂V and ∂L/∂P

`A = P @ V` where for each `(b, h)` pair `P` is `(T, T)` and `V` is `(T, d_k)`.

```
dP = dA @ Vhᵀ          shape (B, H, T, T)        = matmul(dA, Vh.swapaxes(-1,-2))
dVh = Pᵀ @ dA          shape (B, H, T, d_k)      = matmul(P.swapaxes(-1,-2), dA)
```

### 2.4 Softmax Jacobian — full proof

Let `p_i = exp(s_i) / Z` with `Z = Σ_k exp(s_k)`. Compute `∂p_i / ∂s_j`:

**Case i = j:**
```
∂p_i/∂s_i = (exp(s_i)·Z − exp(s_i)·exp(s_i)) / Z²
          = p_i − p_i²
          = p_i (1 − p_i)
```

**Case i ≠ j:**
```
∂p_i/∂s_j = (0 · Z − exp(s_i)·exp(s_j)) / Z²
          = −p_i · p_j
```

Combining: `∂p_i/∂s_j = p_i (δ_{ij} − p_j)`.

Now apply the chain rule to get `dS` from `dP`:
```
dS_j = Σ_i dP_i · ∂p_i/∂s_j
     = Σ_i dP_i · p_i (δ_{ij} − p_j)
     = p_j · dP_j  −  p_j · Σ_i p_i · dP_i
     = p_j ( dP_j  −  ⟨p, dP⟩ )
```

where `⟨p, dP⟩ = Σ_i p_i · dP_i` is a scalar per row. **Vectorized across all rows of all heads of all batches:**

```python
dS = P * (dP - (dP * P).sum(axis=-1, keepdims=True))             # (B, H, T, T)
```

Why this matters numerically: computing the full Jacobian `J ∈ ℝ^{T×T}` per row is `O(T²)` storage *per row* — `O(T³)` total. The identity above is `O(T²)` total because `⟨p, dP⟩` is a single scalar per row.

If a mask zeroed out positions in the forward (those `P[i,j] = 0`), then `dS[i,j]` is automatically 0 there because of the leading `P` factor. **This means you do not need to re-mask `dS`** — but you should re-mask out of paranoia in case any `P` value drifted to a non-zero ulp due to FP rounding; in practice with FP64 it's exactly zero from `exp(-inf) = 0`.

### 2.5 ∂L/∂Q, ∂L/∂K — including the scale

`S = (Qh @ Khᵀ) · scale`. Let `S' = Qh @ Khᵀ`, so `S = scale · S'` and `dS' = scale · dS`. Then:
```
dQh = dS' @ Kh   = scale · dS @ Kh         shape (B, H, T, d_k)
dKh = dS'ᵀ @ Qh  = scale · dSᵀ @ Qh        shape (B, H, T, d_k)
```

In code:
```python
dQh = scale * np.matmul(dS, Kh)                                  # (B, H, T, d_k)
dKh = scale * np.matmul(dS.swapaxes(-1, -2), Qh)                 # (B, H, T, d_k)
```

### 2.6 Through the head-split reshape

Inverse of the forward `reshape(B, T, H, d_k).transpose(0, 2, 1, 3)`:
```python
dQ = dQh.transpose(0, 2, 1, 3).reshape(B, T, D)                  # (B, T, D)
dK = dKh.transpose(0, 2, 1, 3).reshape(B, T, D)
dV = dVh.transpose(0, 2, 1, 3).reshape(B, T, D)
```

### 2.7 ∂L/∂Wq, ∂L/∂Wk, ∂L/∂Wv, ∂L/∂X

`Q = X @ Wq` with `X: (B,T,D)`, `Wq: (D,D)`, `Q: (B,T,D)`.

```
dWq    = Σ_{b,t} X[b,t,:]ᵀ · dQ[b,t,:]   shape (D, D)
dX_q   = dQ @ Wqᵀ                         shape (B, T, D)
```

Same for `Wk, Wv`. **`dX` accumulates from all three paths because `X` is consumed three times** (it feeds `Q`, `K`, and `V` projections independently). By the multivariate chain rule, when a node feeds multiple downstream nodes, gradients **sum**:

```
dX = dX_q + dX_k + dX_v
```

This is the single most-skipped step in interview-quality answers and is the staff bar.

### 2.8 Full backward implementation

```python
def mha_backward(dout, cache):
    X, Wq, Wk, Wv, Wo = cache['X'], cache['Wq'], cache['Wk'], cache['Wv'], cache['Wo']
    Qh, Kh, Vh, P, A_merged = cache['Qh'], cache['Kh'], cache['Vh'], cache['P'], cache['A_merged']
    scale = cache['scale']
    B, T, D, H, d_k = cache['shape']

    # 1. Output projection
    # dout: (B, T, D); A_merged: (B, T, D); Wo: (D, D)
    dWo = np.einsum('btd,bte->de', A_merged, dout)               # (D, D)
    dA_merged = dout @ Wo.T                                      # (B, T, D)

    # 2. Inverse head-merge
    dA = dA_merged.reshape(B, T, H, d_k).transpose(0, 2, 1, 3)   # (B, H, T, d_k)

    # 3. A = P @ V
    dP  = np.matmul(dA, Vh.swapaxes(-1, -2))                     # (B, H, T, T)
    dVh = np.matmul(P.swapaxes(-1, -2), dA)                      # (B, H, T, d_k)

    # 4. Softmax backward (row-wise Jacobian-vector product, closed form)
    dS = P * (dP - (dP * P).sum(axis=-1, keepdims=True))         # (B, H, T, T)

    # 5. S = (Q @ K^T) * scale
    dQh = scale * np.matmul(dS, Kh)                              # (B, H, T, d_k)
    dKh = scale * np.matmul(dS.swapaxes(-1, -2), Qh)             # (B, H, T, d_k)

    # 6. Inverse head-split
    dQ = dQh.transpose(0, 2, 1, 3).reshape(B, T, D)              # (B, T, D)
    dK = dKh.transpose(0, 2, 1, 3).reshape(B, T, D)
    dV = dVh.transpose(0, 2, 1, 3).reshape(B, T, D)

    # 7. Linear projection backward (three paths into X)
    dWq = np.einsum('btd,bte->de', X, dQ)                        # (D, D)
    dWk = np.einsum('btd,bte->de', X, dK)
    dWv = np.einsum('btd,bte->de', X, dV)

    dX_q = dQ @ Wq.T                                             # (B, T, D)
    dX_k = dK @ Wk.T
    dX_v = dV @ Wv.T
    dX   = dX_q + dX_k + dX_v                                    # CRITICAL: sum

    return dX, dWq, dWk, dWv, dWo
```

## 3. Common pitfalls

| # | Bug | Symptom | Fix |
|---|-----|---------|-----|
| 1 | Drop `1/√d_k` in forward | Softmax saturates at large `d_k`, gradients vanish, training stalls | Always scale before softmax |
| 2 | Drop `1/√d_k` in backward only | Forward looks fine, gradient check fails by exactly factor `√d_k` on `dQ`/`dK`; `dV` is correct | Scale `dQh`, `dKh` |
| 3 | Mask after softmax (mult by 0) | Softmax doesn't renormalize; rows don't sum to 1 over visible tokens | Use **additive** mask before softmax: add `-inf` (or large negative like `-1e9`) at masked positions |
| 4 | Mask broadcast wrong shape | `(B, T, T)` instead of `(B, 1, T, T)` — heads dimension collides with batch | Always insert head axis: `mask[:, None, :, :]` |
| 5 | Causal off-by-one | Position `i` can't see itself, or sees `i+1` | Use `np.triu(np.full((T,T), -np.inf), k=1)` — keeps diagonal, blocks above |
| 6 | No `S_max` subtraction | `exp(S)` overflows to `inf` for `T ≥ ~700` with FP32; NaN downstream | Subtract row max before exp |
| 7 | `reshape(B, H, T, d_k)` directly | Heads interleave wrong; silently wrong outputs that look "almost right" | `reshape(B, T, H, d_k).transpose(0, 2, 1, 3)` |
| 8 | Non-contiguous reshape after transpose | `ValueError` from NumPy; or in PyTorch `.view()` fails | NumPy `.reshape()` handles it; in PyTorch use `.contiguous().view()` or `.reshape()` |
| 9 | Forget to sum `dX` from three paths | `dX` is wrong by factor of ~3, sometimes worse depending on `Wq/k/v` magnitudes | `dX = dX_q + dX_k + dX_v` — always |
| 10 | Use `dP` directly as `dS` (skip Jacobian) | `dQ`/`dK`/`dV` all wrong; gradient check fails everywhere downstream of softmax | Apply `dS = P * (dP - (dP*P).sum(-1, keepdims))` |
| 11 | Don't re-mask gradients | In FP32 it's usually fine because `P=0` zeros it out; in FP16/BF16, denormals may leak | Optionally `dS = np.where(mask==-inf, 0, dS)` |
| 12 | Compose padding + causal incorrectly | Padding tokens attend to each other or are attended to | Combine masks **additively**: `mask = causal + padding` (both as 0 / `-inf`), not by AND/OR |
| 13 | Cache `S` instead of `P` | Either recompute softmax in backward (slow + numerically different) or write the wrong Jacobian | Cache `P` |
| 14 | Reference mismatch with `torch.nn.MultiheadAttention` | Numbers don't match; you assume bug in your code | PyTorch uses **fused** QKV `in_proj_weight` of shape `(3D, D)`; split it as `Wq, Wk, Wv = in_proj_weight.chunk(3, 0)`, then transpose for our convention |

## 4. Gradient check

```python
def mha_grad_check(seed=0):
    rng = np.random.default_rng(seed)
    B, T, D, H = 2, 4, 8, 2
    X  = rng.standard_normal((B, T, D)).astype(np.float64)
    Wq = rng.standard_normal((D, D)).astype(np.float64) * 0.1
    Wk = rng.standard_normal((D, D)).astype(np.float64) * 0.1
    Wv = rng.standard_normal((D, D)).astype(np.float64) * 0.1
    Wo = rng.standard_normal((D, D)).astype(np.float64) * 0.1

    out, cache = mha_forward(X, Wq, Wk, Wv, Wo, H)
    dout = rng.standard_normal(out.shape)
    dX, dWq, dWk, dWv, dWo = mha_backward(dout, cache)

    def loss(X_, Wq_, Wk_, Wv_, Wo_):
        o, _ = mha_forward(X_, Wq_, Wk_, Wv_, Wo_, H)
        return (o * dout).sum()

    eps = 1e-6
    def num_grad(arr, name):
        g = np.zeros_like(arr)
        it = np.nditer(arr, flags=['multi_index'])
        while not it.finished:
            idx = it.multi_index
            old = arr[idx]
            arr[idx] = old + eps; lp = loss(X, Wq, Wk, Wv, Wo)
            arr[idx] = old - eps; lm = loss(X, Wq, Wk, Wv, Wo)
            arr[idx] = old
            g[idx] = (lp - lm) / (2 * eps)
            it.iternext()
        return g

    for arr, ana, name in [(X,dX,'X'),(Wq,dWq,'Wq'),(Wk,dWk,'Wk'),
                            (Wv,dWv,'Wv'),(Wo,dWo,'Wo')]:
        num = num_grad(arr, name)
        rel = np.abs(num - ana).max() / (np.abs(num).max() + np.abs(ana).max() + 1e-12)
        print(f"{name}: max rel err = {rel:.2e}")
        assert rel < 1e-7, f"{name} grad check failed"
```

**Threshold:** in FP64 with central differences and `eps=1e-6`, max relative error below `1e-7` passes. Anything above `1e-5` is a real bug; between is suspicious and worth re-running with `eps=1e-5`.

**Diagnostic if you drop `scale` in backward only:** `dV` passes (it doesn't pass through the scaled scores), but `dQ` and `dK` are off by the scalar factor `√d_k`. With `d_k=4`, that's a relative error of ~0.5 (since you computed `1·X` instead of `(1/2)·X` after the chain rule routes through scale). The localization signal: **`dV` clean, `dQ`/`dK` both off by the same constant factor → look at the `S = QKᵀ · scale` step.** If only one of `dQ`/`dK` is broken, it's a transpose bug in step 5; if both, it's the scale.

## 5. Complexity analysis

For batch `B`, `H` heads, sequence `T`, head dim `d_k`, model dim `D = H · d_k`:

| Step | Time | Memory |
|---|---|---|
| 4 projections (Q, K, V, O) | `O(B·T·D²)` | `O(B·T·D)` |
| `Q @ Kᵀ` | `O(B·H·T²·d_k) = O(B·T²·D)` | `O(B·H·T²)` ← dominant |
| Softmax | `O(B·H·T²)` | `O(B·H·T²)` |
| `P @ V` | `O(B·T²·D)` | `O(B·T·D)` |

**Largest intermediate: `P` (and `S`), both `(B, H, T, T)`** — this is the `T²` term that prevents long context. At `B=1, H=32, T=100k` in FP16, `P` alone is `32 · 10¹⁰ · 2 bytes = 640 GB`. This is why naive attention is unrunnable past ~32k tokens on a single GPU and why Flash Attention exists.

The crossover where attention dominates projections: `4·T²·D > 8·T·D²`, i.e. `T > 2D`. For `D=4096`, that's `T > 8192`.

## 6. Follow-ups

### a. Causal masking

Build the mask **once** outside the call, share it across batch and heads:

```python
def causal_mask(T):
    m = np.zeros((T, T), dtype=np.float32)
    m[np.triu_indices(T, k=1)] = -np.inf      # block strictly upper triangle
    return m   # (T, T) — broadcasts to (1, 1, T, T)
```

Position `i` attends to `j ≤ i` (including itself: `k=1` keeps the diagonal). Memory: `O(T²)` for the mask, but only one copy regardless of batch and heads — broadcasting is free. Don't materialize `(B, 1, T, T)`; that's wasteful. In production, frameworks often skip the mask entirely and compute only the lower triangle (saves ~2× FLOPs); Flash Attention does this.

### b. KV cache for autoregressive decoding

```python
def mha_decode_step(x_t, Wq, Wk, Wv, Wo, num_heads, K_cache, V_cache, t):
    # x_t: (B, 1, D)  — single new token
    # K_cache, V_cache: (B, H, T_max, d_k) — preallocated, write-only up to t
    B, _, D = x_t.shape; H = num_heads; d_k = D // H; scale = 1/np.sqrt(d_k)

    Qh = (x_t @ Wq).reshape(B, 1, H, d_k).transpose(0, 2, 1, 3)  # (B, H, 1, d_k)
    K_new = (x_t @ Wk).reshape(B, 1, H, d_k).transpose(0, 2, 1, 3)
    V_new = (x_t @ Wv).reshape(B, 1, H, d_k).transpose(0, 2, 1, 3)

    K_cache[:, :, t:t+1, :] = K_new
    V_cache[:, :, t:t+1, :] = V_new
    K_so_far = K_cache[:, :, :t+1, :]                            # (B, H, t+1, d_k)
    V_so_far = V_cache[:, :, :t+1, :]

    S = np.matmul(Qh, K_so_far.swapaxes(-1, -2)) * scale          # (B, H, 1, t+1)
    P = softmax(S)                                                # no mask needed: only past tokens cached
    A = np.matmul(P, V_so_far)                                    # (B, H, 1, d_k)
    out = A.transpose(0, 2, 1, 3).reshape(B, 1, D) @ Wo
    return out
```

Per-step complexity is **`O(T·D)` for attention** (we contract over `t+1` positions, not `T²`), vs `O(T²·D)` for prefill. **Decode is memory-bandwidth-bound** — every step you must re-read `(B, H, t, d_k)` of KV cache from HBM. This is why decode hardware utilization is ~5–20% on H100s and why MQA/GQA exist (shrink the KV cache).

**Most common bug:** off-by-one when `t=0`. Candidates either (i) read `K_cache[:, :, :0, :]` which is empty and `matmul` returns shape `(B, H, 1, 0)` causing softmax to produce NaN, or (ii) compute `K_cache[:, :, :t, :]` (excluding the new token) so the model can never attend to itself. Always slice `:t+1` after writing.

### c. Why divide by √d_k

Assume `Q[i, k]` and `K[j, k]` are independent, zero-mean, unit-variance. Then
```
S[i, j] = Σ_{k=1..d_k} Q[i,k] · K[j,k]
```
is a sum of `d_k` independent zero-mean unit-variance products. Variance of a product of independent zero-mean unit-variance variables is 1, so `Var(S[i,j]) = d_k`, `Std(S[i,j]) = √d_k`.

Without scaling, scores grow as `√d_k`. For `d_k=128`, that's `≈ 11.3` standard deviations of dynamic range entering softmax. Softmax with one logit of ≈11 above the rest assigns ≈99.998% of the mass to the argmax — it's a hard one-hot. Gradients then look like `p_i (1 − p_i) ≈ 0` everywhere except possibly one position; **gradient flow collapses**.

Dividing by `√d_k` keeps `Var(S) ≈ 1` regardless of head dimension, which is the regime softmax was designed for.

### d. Grouped-query attention (GQA)

Let `H_q` = query heads, `H_kv` = KV heads, with `H_q % H_kv == 0` and `g = H_q / H_kv`. Shapes change as:

- `Wq: (D, D)` (unchanged) → `Qh: (B, H_q, T, d_k)`
- `Wk, Wv: (D, H_kv · d_k)` → `Kh, Vh: (B, H_kv, T, d_k)`

Then **broadcast / repeat KV across the group**:
```python
# Option A: explicit broadcast via repeat (memory-cheap if you're careful)
Kh_b = np.repeat(Kh, g, axis=1)        # (B, H_q, T, d_k)
Vh_b = np.repeat(Vh, g, axis=1)
# Option B: reshape Q to (B, H_kv, g, T, d_k), let matmul broadcast over g
Qh_g = Qh.reshape(B, H_kv, g, T, d_k)
S    = np.matmul(Qh_g, Kh[:, :, None, :, :].swapaxes(-1, -2)) * scale  # (B, H_kv, g, T, T)
```

Option B avoids actually replicating KV in memory — important for KV cache where reducing KV by `g×` is the entire point. Llama 3 70B uses `H_q=64, H_kv=8` (g=8); MQA is the special case `H_kv=1`.

### e. Flash Attention — tiling and online softmax

The core observation: softmax can be computed **online** as you stream tiles of `K, V`. Maintain three running statistics per query row:
- `m` — running max of seen scores
- `ℓ` — running denominator (sum of `exp(s - m)`)
- `o` — running output accumulator

When a new tile arrives with scores `s_new` and values `v_new`:
```
m_new   = max(m, max(s_new))
α       = exp(m - m_new)                 # rescale factor for old stats
β       = exp(s_new - m_new)             # weights for new tile
ℓ_new   = α · ℓ + sum(β)
o_new   = (α · ℓ · o + β @ v_new) / ℓ_new
m, ℓ, o = m_new, ℓ_new, o_new
```

This produces the exact same softmax output as the naive version, never materializing the full `(T, T)` matrix. Tile sizes `B_r, B_c` are chosen so `Q_tile`, `K_tile`, `V_tile`, `O_tile`, and the running stats fit in **on-chip SRAM** (~192KB per SM on H100). 

**Why it's IO-bound, not compute-bound:** naive attention reads `O(T²)` bytes of HBM for `S` and `P`. Flash Attention reads `O(T·d_k)` per pass — roughly `T/d_k` × less HBM traffic. The arithmetic intensity (FLOPs/byte) goes from `O(d_k)` to `O(d_k·T/B_c)`, crossing the H100 ridge point so the kernel becomes math-bound only above some `T`. For typical `d_k=128, B_c=64`, this is ~16× more arithmetic intensity, which is the actual speedup source.

### f. Flash Attention backward — recompute vs store

In the forward, Flash Attention stores **only** the per-row scalar statistics `(m, ℓ)` and the output `O`. It does **not** store `S` or `P`.

In the backward, you need `P` (for the softmax Jacobian) and `S` (well, `dS`). The trick: with `(m, ℓ)` and `O` saved, you can **recompute `P` tile-by-tile on the fly** in the backward, since `P_ij = exp(s_ij - m_i) / ℓ_i` and the saved `(m_i, ℓ_i)` lets you do this in one pass without needing the global softmax denominator.

**Why backward is more memory-intensive than naive backward despite forward being more memory-efficient:** the backward must compute `dQ, dK, dV` simultaneously, and `dQ` accumulates *across all key tiles* (each query row sees every key). This forces either (a) storing `dQ` in HBM and atomically accumulating across blocks, or (b) re-streaming over K twice. Flash Attention 2 chose (a) with split-K parallelism; Flash Attention 3 (Hopper) refines this further. The recomputation also adds ~30% FLOPs over naive backward.

### g. Diagnosis: dQ/dK/dV pass, dWq fails

Decision tree:

1. **First check shapes.** If `dWq.shape != Wq.shape`, you have a transpose bug in the einsum. Should be `np.einsum('btd,bte->de', X, dQ)` not `('btd,bte->ed', ...)`.

2. **Check the einsum contraction.** `dWq[d, e] = Σ_{b,t} X[b,t,d] · dQ[b,t,e]`. If you wrote `np.einsum('btd,bte->de', dQ, X)` (swapped operands), you get `Wqᵀ`'s gradient instead of `Wq`'s.

3. **Are you summing over batch?** A common bug: `dWq = X.swapaxes(-1,-2) @ dQ` per batch but forgetting to reduce: gives `(B, D, D)` instead of `(D, D)`. Either `.sum(0)` after, or use einsum.

4. **Numerical magnitude check.** If `dWq` is roughly 1/B times the true value, you accidentally averaged instead of summed. If it's roughly B× too large, you summed twice (e.g., `np.einsum` plus an extra `.sum(0)`).

5. **Test with B=1, T=1, D=2.** Walk it by hand. If `B=1, T=1`, then `dWq[d,e] = X[0,0,d] · dQ[0,0,e]`, an outer product. Verify byte-for-byte.

If `dQ/dK/dV` are correct, the entire upstream chain through softmax and matmul is sound — the bug is *strictly* in the projection backward, which is a 2-line operation. Don't waste time re-deriving softmax.

### h. NaN at T=8192 in bf16, fine at T=2048

Three compounding issues:

**1. Fully-masked rows produce NaN.** With causal masking, the *first* row attends only to position 0 — fine. But with padding masks combined with causal masks, you can get rows where every position is masked (e.g., a query position that's itself padding). Then `S = -inf` everywhere, `S_max = -inf`, `S - S_max = -inf - (-inf) = NaN`. Then `exp(NaN) = NaN` → propagates everywhere.

   *Fix:* detect rows where `S_max == -inf` and zero out the entire output for that row. Most production kernels do `S_max = max(S_max, -1e30)` to prevent the subtraction, then `P` will be zero for those rows naturally.

**2. bf16 mantissa precision at long T.** bf16 has 7 mantissa bits. At T=8192, after `1/√d_k` scaling, the softmax denominator `Σ exp(s_i - s_max)` accumulates 8192 terms, each in `[0, 1]`. Many will be tiny. In bf16, summing 8192 small numbers into a running total loses precision rapidly (catastrophic cancellation in the last bits). The fix in production kernels is to do the softmax accumulation in fp32 even when inputs are bf16 — Flash Attention does exactly this.

**3. `1/√d_k` interaction with `-inf` masks.** If you implement masking as multiplying by 0 and adding `-1e9`, then scaling by `1/√d_k` makes it `-1e9/√d_k`, which is still `-inf`-effective. But if your mask is `-65504` (bf16 max negative finite), scaling shrinks it toward 0 — and at T=8192, `-65504/√128 ≈ -5790`, which `exp(-5790)` underflows to 0 in bf16, fine. But at smaller `d_k`, `-65504/√16 ≈ -16376`, still fine. The actual issue is when a fp16 mask is added to bf16 scores — type promotion + range. Standardize: use `-inf` for masks, not large finite values, when working in bf16.

The cleanest answer: it's the fully-masked row, exposed at long T because more rows trigger the edge case as padding becomes proportionally more common.

### i. End-to-end test against PyTorch reference

```python
def test_against_pytorch(B=2, T=8, D=16, H=4, seed=0):
    import torch
    rng = np.random.default_rng(seed)
    X  = rng.standard_normal((B, T, D)).astype(np.float64)
    Wq = rng.standard_normal((D, D)).astype(np.float64) * 0.1
    Wk = rng.standard_normal((D, D)).astype(np.float64) * 0.1
    Wv = rng.standard_normal((D, D)).astype(np.float64) * 0.1
    Wo = rng.standard_normal((D, D)).astype(np.float64) * 0.1

    out_np, _ = mha_forward(X, Wq, Wk, Wv, Wo, H)

    # Map our convention to torch.nn.MultiheadAttention's fused QKV
    mha = torch.nn.MultiheadAttention(D, H, bias=False, batch_first=True).double()
    with torch.no_grad():
        # torch's in_proj_weight is (3D, D) — concat of [Wq; Wk; Wv]^T convention
        # We store Wq with X @ Wq, torch uses Wq @ x => store Wq.T into the row.
        mha.in_proj_weight.copy_(torch.from_numpy(np.concatenate([Wq.T, Wk.T, Wv.T], axis=0)))
        mha.out_proj.weight.copy_(torch.from_numpy(Wo.T))

    Xt = torch.from_numpy(X)
    out_torch, _ = mha(Xt, Xt, Xt, need_weights=False)
    err = np.abs(out_np - out_torch.numpy()).max()
    print(f"max abs err vs PyTorch: {err:.2e}")
    assert err < 1e-10
```

**Test harness checklist:**
- **FP64 everywhere** for the comparison test — eliminates noise. FP32 or bf16 tests are separate (with looser tolerance).
- **Deterministic seed.** Same RNG for both paths.
- **Convention adapter** that converts your projection weights to the reference's layout. PyTorch's `MultiheadAttention` uses fused `in_proj_weight` of shape `(3D, D)` with the `nn.Linear` convention `y = x @ Wᵀ + b`, while we use `y = x @ W`. Always convert via `.T` and concat in `[Wq; Wk; Wv]` order.
- **Tolerance:** FP64 → `1e-10`. FP32 → `1e-5`. bf16 → `1e-2` (accept much larger error).
- **Test forward and backward separately.** For backward, run autograd on the torch reference and compare to your hand-written gradients.
- **Edge cases:** `T=1`, fully-masked rows, `H=1`, `B=1`, asymmetric padding masks.

### j. Param and FLOP count: D=4096, H=32, T=2048

**Parameters (one MHA layer, no biases):**
- `Wq, Wk, Wv, Wo`: each `D² = 4096² = 16.78M`
- Total: `4 · D² = 67.1M params`

**Forward FLOPs (B=1):** count multiply-adds as 2 FLOPs.
- 4 projections: `4 · 2 · T · D² = 8 · 2048 · 4096² = 2.75 × 10¹¹` FLOPs
- `Q @ Kᵀ`: `2 · H · T · T · d_k = 2 · T² · D = 2 · 2048² · 4096 = 6.87 × 10¹⁰` FLOPs
- Softmax: `O(H · T²) = 0.13 × 10⁹` — negligible
- `P @ V`: `2 · T² · D = 6.87 × 10¹⁰` FLOPs
- **Total forward: ≈ 4.12 × 10¹¹ FLOPs ≈ 412 GFLOPs**

**Which term dominates:**
- Projections: `8 · T · D² = 8 · D² · T`
- Attention (`QKᵀ` + `PV`): `4 · T² · D`
- Crossover when `8 · D² · T = 4 · T² · D`, i.e. **`T = 2D = 8192`**.

At `T=2048`: projections are `2.75e11`, attention is `1.37e11` → projections dominate ~2:1.
At `T=8192`: equal.
At `T=32k`: attention dominates ~4:1, and the `T²` quadratic blowup is why production training caps `T` and uses Flash Attention to amortize the IO.

**Backward** is roughly 2× forward FLOPs, so ~824 GFLOPs total. On an H100 at ~750 TFLOP/s bf16, this layer's compute is ~1.1 ms — but real-world wall time is dominated by HBM traffic for the `(B, H, T, T)` attention matrix, which is the entire reason Flash Attention exists.

---

If anything in the softmax Jacobian step or the `dX = dX_q + dX_k + dX_v` accumulation feels rusty, drill those two specifically — they're the two places where I've seen staff candidates lose the round even when the rest is clean. The Jacobian identity is the kind of thing a strong interviewer will ask you to derive on the whiteboard cold; have it memorized at the level of "I can prove `∂p_i/∂s_j = p_i(δ_ij - p_j)` from the quotient rule in 30 seconds."