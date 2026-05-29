---
title: RoPE — Paper-to-Code Mock Interview
description: A combined mock (read paper, explain benefit, implement in Colab) using Rotary Position Embedding (RoPE) as the worked example.
sidebar:
  order: 11
  label: RoPE
---

> **Paper:** *RoFormer: Enhanced Transformer with Rotary Position Embedding* — Su et al., 2021. arXiv: [2104.09864](https://arxiv.org/abs/2104.09864)
>
> **Format:** Read (~15 min) → explain the *real* benefit → implement the core idea in Colab → sanity-check it.
>
> **Companion notebook:** [`rope_mock.ipynb`](/notebooks/rope_mock.ipynb) (download) — a relative-position-invariance demo + an `apply_rope` stub to fill in, plus verification cells. Open in [Google Colab](https://colab.research.google.com/) via *File → Upload notebook*. A reference solution is included at the bottom of this page.
>
> **Difficulty:** 🟡 Medium. You need to be comfortable with attention (see the [attention mock](/papers/attention-mock-interview/)) and a little trig.

---

## How to run this as a timed drill (~55 min)

| Time | Block | What you produce |
|------|-------|------------------|
| 0:00–0:15 | **Read** (use the [three-pass method](/papers/lora-mock-interview/#part-0--how-to-read-a-paper-in-15-minutes-the-three-pass-method)) | Why rotate q/k + how relative position falls out |
| 0:15–0:20 | **Explain the benefit** out loud (cover Part 2) | The "relative for free, no params" pitch + extrapolation |
| 0:20–0:50 | **Implement** from the stub (Part 3) | A working `apply_rope` + a score that depends only on `m−n` |
| last 5 min | **Sanity-check** (Part 4) | All checks passing, narrated out loud |

### Self-grading rubric — "what good looks like"
- ✅ Explained RoPE as **rotating** q/k by an angle ∝ absolute position, so the dot product depends only on the **relative** offset — not "it adds position vectors."
- ✅ Knew **why the dot product becomes relative**: composing rotations subtracts angles (`R_m^T R_n = R_{n−m}`).
- ✅ Implemented it as a **per-pair 2-D rotation**, with frequencies `θ_i = base^(−2i/d)`, applied to **q and k only** (not V).
- ✅ Demonstrated the benefit with the **shift-invariance** property (max diff ≈ 0), not just "it runs."
- ⚠️ Red flags: applying RoPE to V, treating it as a learned/added embedding, forgetting it has **zero parameters**, claiming it changes vector norms.

---

## Part 1 — Structured read of THIS paper

### The 30-second summary (the "benefit")
Transformers have no built-in notion of order, so you must inject position. Learned **absolute** embeddings add a position vector to each token; they cost parameters and extrapolate poorly past the trained context length. RoPE instead **rotates** each query and key vector by an angle proportional to its **absolute** position. Because of how rotations compose, the attention score `q_m · k_n` ends up depending only on the **relative** offset `(m − n)` and the content. The payoff:

- **Relative position "for free"** — you encode absolute position per token, but the dot product sees only the relative offset.
- **Zero extra parameters** — it's a fixed, deterministic rotation, not a learned table.
- **Better length extrapolation** and a clean way to integrate with standard scaled-dot-product attention (it's now the default in LLaMA, GPT-NeoX, PaLM, etc.).

### The core idea (Method — you implement this)
Split the `d`-dim vector into `d/2` consecutive **pairs** `(x_{2i}, x_{2i+1})`. For a token at position `m`, rotate pair `i` by angle `m·θ_i`, where the per-pair frequency is

$$\theta_i = \text{base}^{-2i/d}, \qquad i = 0, 1, \dots, \tfrac{d}{2}-1, \quad \text{base}=10000.$$

Each pair is rotated by the standard 2-D rotation matrix:

$$\begin{pmatrix} x'_{2i} \\ x'_{2i+1} \end{pmatrix} = \begin{pmatrix} \cos m\theta_i & -\sin m\theta_i \\ \sin m\theta_i & \cos m\theta_i \end{pmatrix} \begin{pmatrix} x_{2i} \\ x_{2i+1} \end{pmatrix}.$$

Write the whole rotation as a block-diagonal orthogonal matrix `R_m`. Apply it to query and key, then take the dot product. The magic is that **rotations compose by adding angles**, so

$$(R_m\,q)^\top (R_n\,k) = q^\top R_m^\top R_n\, k = q^\top R_{n-m}\, k,$$

which depends on positions **only through the offset** `n − m`. That is relative position, obtained by encoding *absolute* position on each side.

Key details (the things an interviewer probes):
- **Applied to q and k only**, *not* to V — RoPE shapes the *score*, not the value mix.
- **No parameters.** `R_m` is fixed; the only knob is `base` (10000), which sets the wavelength spectrum.
- **Different frequencies per pair.** Low `i` rotates fast (short wavelength, local), high `i` rotates slowly (long wavelength, global) — like the sinusoidal-embedding spectrum, but multiplicative.
- **Norm-preserving.** Rotation is orthogonal, so `‖R_m x‖ = ‖x‖`; RoPE never changes a vector's magnitude.
- **Position 0 is the identity** (angle 0 ⇒ no rotation).

### Where the evidence lives (tables that matter)
*(Hedge: figure/table numbers below are from memory of the RoFormer paper — verify against the PDF.)*
- **Machine-translation / GLUE-style language-modeling tables:** RoPE matches or beats sinusoidal and learned absolute embeddings → the quality claim.
- **Faster/lower training-loss curves vs the BERT-style baseline** → the convergence claim.
- **Long-sequence experiments:** stable behavior as context grows → the extrapolation claim that made RoPE ubiquitous in modern LLMs.

### The honest limitations (have an opinion)
- **Vanilla RoPE still degrades far beyond the trained context length.** The "for free" extrapolation is *better*, not unlimited — hence the whole follow-up family (NTK-aware scaling, Position Interpolation, YaRN) that rescales `base`/frequencies.
- **Relative, not arbitrary.** It encodes a smooth function of `(m − n)`; it can't represent arbitrary learned position-pair interactions the way a full relative-attention bias table could.
- **Pairing/interleaving convention matters.** The `(2i, 2i+1)` interleaved layout vs the "rotate-half" (split-in-two) layout differ; mixing conventions between training and inference silently breaks a model.

---

## Part 2 — The interview dialogue (interviewer ⇄ interviewee)

> **🧑‍💼 Interviewer:** One paragraph — what does RoPE actually buy me over absolute position embeddings?
>
> **🧑‍💻 Interviewee:** It gives you *relative* position essentially for free and with zero parameters. Instead of adding a learned position vector, I rotate each query and key by an angle proportional to its absolute position. Because rotations compose by adding angles, when I dot a rotated query at position `m` with a rotated key at position `n`, the position dependence collapses to a function of the offset `m − n`. So I encode absolute position on each side but the attention score only sees relative position — and it extrapolates to longer contexts better than a learned absolute table, which is why modern LLMs use it.

> **🧑‍💼 Interviewer:** Walk me through *why* the dot product becomes relative.
>
> **🧑‍💻 Interviewee:** RoPE multiplies the query by an orthogonal rotation `R_m` and the key by `R_n`. The score is `(R_m q)ᵀ(R_n k) = qᵀ R_mᵀ R_n k`. For rotations, `R_mᵀ = R_{−m}`, and they compose additively, so `R_mᵀ R_n = R_{n−m}`. The whole thing is `qᵀ R_{n−m} k` — positions enter only through `n − m`. Per 2-D pair it's just the angle-subtraction identity for `cos`/`sin`.

> **🧑‍💼 Interviewer:** Why apply it to q and k but not V?
>
> **🧑‍💻 Interviewee:** RoPE's job is to make the *attention score* position-aware. The score is the only place where q and k meet, and that's where the rotation cancels into a relative offset. V carries the content you actually aggregate; rotating it would inject position into the output values for no benefit and would break the clean relative property. So RoPE touches q and k, attention proceeds normally, V is untouched.

> **🧑‍💼 Interviewer:** It has no parameters — so what's the one knob, and what does it do?
>
> **🧑‍💻 Interviewee:** The `base` (default 10000). It sets the geometric spread of per-pair frequencies `θ_i = base^(−2i/d)`: low dimensions rotate fast (capture local, fine-grained offsets), high dimensions rotate slowly (capture long-range structure). Increasing `base` lengthens wavelengths, which is exactly the lever the extrapolation methods like NTK-aware scaling and YaRN tune to stretch a model to longer contexts without retraining from scratch.

> **🧑‍💼 Interviewer:** Implement it and show the score depends only on the relative offset.

---

## Part 3 — Implementation

The whole method is a per-pair 2-D rotation applied to q and k. No parameters, no learned state.

```python
import torch


def apply_rope(x, positions, base=10000.0):
    """Rotate consecutive dim pairs (2i, 2i+1) by angle = pos * base^(-2i/dim).

    x:          (..., seq, dim) with EVEN dim.
    positions:  (seq,) absolute positions for each token.
    returns:    same shape as x, rotated.
    """
    *_, seq, dim = x.shape
    assert dim % 2 == 0, "dim must be even: dims are rotated in pairs"
    half = dim // 2

    i = torch.arange(half, device=x.device, dtype=x.dtype)
    theta = base ** (-2.0 * i / dim)                         # (half,) per-pair frequencies
    angles = positions.to(x.dtype)[:, None] * theta[None, :] # (seq, half) angle per (pos, pair)
    cos, sin = torch.cos(angles), torch.sin(angles)

    x_even, x_odd = x[..., 0::2], x[..., 1::2]               # the two halves of each pair
    rot_even = x_even * cos - x_odd * sin                    # standard 2-D rotation
    rot_odd  = x_even * sin + x_odd * cos
    out = torch.empty_like(x)
    out[..., 0::2], out[..., 1::2] = rot_even, rot_odd       # re-interleave
    return out
```

### Why each line matters (talk through it)
- `assert dim % 2 == 0` — RoPE rotates **pairs** of dimensions; an odd `dim` has a leftover scalar with no partner to rotate against.
- `theta = base ** (-2.0 * i / dim)` — the frequency spectrum: pair 0 is the fastest, the last pair the slowest. This is the *only* design knob.
- `positions[:, None] * theta[None, :]` — broadcasts to one angle per (position, pair): the angle is **proportional to absolute position**, the heart of RoPE.
- `x_even * cos - x_odd * sin` / `x_even * sin + x_odd * cos` — the literal 2-D rotation matrix applied to each pair. Orthogonal, so it preserves norms.
- `0::2` / `1::2` — the interleaved `(2i, 2i+1)` convention. (LLaMA's reference uses a "rotate-half" split layout; same idea, different bookkeeping — pick one and be consistent.)

### Demonstrating the property (relative-position invariance)
This is the headline correctness demo — **not** a benchmark. Take *fixed content* q and k. The attention score between q at position `m` and k at position `n` must be unchanged when you shift **both** positions by any `s`, because it depends only on `m − n`.

```python
torch.manual_seed(0)
dim = 8
q = torch.randn(dim)      # fixed CONTENT for the query
k = torch.randn(dim)      # fixed CONTENT for the key
m, n = 5, 2               # absolute positions; relative offset m - n = 3

def score(content_q, content_k, pm, pn):
    qr = apply_rope(content_q[None, :], torch.tensor([pm]))[0]
    kr = apply_rope(content_k[None, :], torch.tensor([pn]))[0]
    return torch.dot(qr, kr)

base_score = score(q, k, m, n)
print(f"score(q@{m}, k@{n}) = {base_score.item():.6f}   (offset {m-n})")
diffs = []
for s in (1, 3, 7, 50, 123):
    sc = score(q, k, m + s, n + s)
    diffs.append((sc - base_score).abs().item())
    print(f"score(q@{m+s}, k@{n+s}) = {sc.item():.6f}   shift s={s}")
print(f"max abs difference across shifts = {max(diffs):.2e}  (~0 => depends only on m-n)")
```

Expected output (numbers are seed-dependent; the **invariance** is the point):

```
score(q@5, k@2) = 1.178293   (offset 3)
score(q@6, k@3) = 1.178293   shift s=1
score(q@8, k@5) = 1.178293   shift s=3
score(q@12, k@9) = 1.178293   shift s=7
score(q@55, k@52) = 1.178293   shift s=50
score(q@128, k@125) = 1.178293   shift s=123
max abs difference across shifts = 2.38e-07  (~0 => depends only on m-n)
```

The score is identical for every equal shift — absolute positions changed by up to 123, but because the *offset* stayed 3, the attention score never moved. That is relative position, encoded for free.

---

## Part 4 — Sanity checks (don't skip)

### Check 1 — RoPE preserves the L2 norm (rotation is orthogonal)
```python
dim = 8
x = torch.randn(4, dim)
xr = apply_rope(x, torch.arange(4))
assert torch.allclose(x.norm(dim=-1), xr.norm(dim=-1), atol=1e-5)
print("OK: per-vector norm unchanged")
```

### Check 2 — Position 0 is the identity
```python
x0 = torch.randn(1, dim)
assert torch.allclose(apply_rope(x0, torch.tensor([0])), x0, atol=1e-6)
print("OK: position 0 == identity")
```

### Check 3 — Relative-offset invariance of the score (the core property)
```python
m, n, s = 5, 2, 17
assert torch.allclose(score(q, k, m, n), score(q, k, m + s, n + s), atol=1e-4)
print("OK: q.k score invariant under equal shift (depends only on m-n)")
```

### Check 4 — Shape is preserved
```python
big = torch.randn(2, 6, 10)            # (batch, seq, dim)
assert apply_rope(big, torch.arange(6)).shape == big.shape
print("OK: output shape == input shape")
```

### Check 5 — Different relative offsets give DIFFERENT scores (it actually encodes position)
```python
s_off3 = score(q, k, 5, 2)             # offset 3
s_off5 = score(q, k, 5, 0)             # offset 5
assert not torch.allclose(s_off3, s_off5, atol=1e-3)
print(f"OK: offsets differ => scores differ ({s_off3.item():.4f} vs {s_off5.item():.4f})")
```

### Check 6 — Composition: rotate(a) then rotate(b) == rotate(a+b)
```python
a, b = 3.0, 4.0
twostep = apply_rope(apply_rope(x0, torch.tensor([a])), torch.tensor([b]))
onestep = apply_rope(x0, torch.tensor([a + b]))
assert torch.allclose(twostep, onestep, atol=1e-5)
print("OK: rotations compose additively")
```

All six should print `OK`. Check 3 is the one that matters most — it's the property the whole paper is built on; checks 1, 2, 6 confirm it's a genuine rotation; checks 4, 5 confirm it's non-trivial.

---

## Part 5 — Likely follow-up questions

- *"Interleaved `(2i,2i+1)` vs LLaMA's rotate-half layout?"* — Same rotation, different dimension pairing. Interleaved rotates adjacent dims; rotate-half pairs dim `i` with dim `i + d/2`. Both are valid as long as train and inference agree; weights are **not** portable across conventions.
- *"How does RoPE extrapolate to longer contexts, and where does it break?"* — Better than learned absolute embeddings because it's a smooth function of relative offset, but vanilla RoPE still degrades well past the trained length. Fixes rescale frequencies: **Position Interpolation** squashes positions into the trained range, **NTK-aware / YaRN** adjust `base` per-frequency to stretch the context with little or no retraining.
- *"Why not just add sinusoidal embeddings (Vaswani et al.)?"* — Sinusoidal/absolute embeddings are **added** to inputs and bias toward absolute position; RoPE is **multiplicative** and makes the *score* depend on relative offset, which generalizes across positions better and composes cleanly with attention.
- *"Does RoPE cost FLOPs or memory?"* — Negligible: two elementwise mul-adds per element on q and k, no parameters, no extra activations to store. The `cos`/`sin` tables can be precomputed and cached per position.
- *"Why apply it inside each head rather than once on the embedding?"* — Position must enter at the q·k interaction *per head*, after the q/k projections, so each head sees rotated queries/keys. Rotating the shared embedding once wouldn't survive the per-head linear projections.

---

## TL;DR cheat sheet
| Thing | Answer |
|---|---|
| Core idea | Rotate q/k by an angle ∝ absolute position; dot product then depends only on offset `m−n` |
| Formula | rotate pair `i` by `m·θ_i`, `θ_i = base^(−2i/d)`, `base=10000` |
| Why relative | `R_mᵀ R_n = R_{n−m}` — composing rotations subtracts angles |
| Applied to | **q and k only**, not V |
| Parameters | **None** (fixed rotation); only knob is `base` |
| Benefit | Relative position for free + better length extrapolation than learned absolute |
| Norm | Preserved (rotation is orthogonal) |
| #1 gotcha | Mixing interleaved vs rotate-half conventions across train/inference |
| Limitation | Still degrades far beyond trained length → PI / NTK / YaRN rescaling |
