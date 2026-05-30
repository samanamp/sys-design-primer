---
title: Attention — Paper-to-Code Mock Interview
description: A combined mock (read paper, explain benefit, implement in Colab) using scaled dot-product / multi-head attention as the worked example.
sidebar:
  order: 19
  label: Attention
---

> **Paper:** *Attention Is All You Need* — Vaswani et al., 2017. arXiv: [1706.03762](https://arxiv.org/abs/1706.03762)
>
> **Format:** Read (~15 min) → explain the *real* benefit → implement the core idea in Colab → sanity-check it.
>
> **Companion notebook:** [`attention_mock.ipynb`](/notebooks/attention_mock.ipynb) (download) — content-retrieval demo + a `MultiHeadAttention` stub to fill in, plus verification cells. Open in [Google Colab](https://colab.research.google.com/) via *File → Upload notebook*.
>
> **Difficulty:** 🔴 The hardest of the set. Scope to *scaled dot-product + multi-head self-attention* — not the whole Transformer.

---

## How to run this as a timed drill (~65 min)

> ⚠️ **Scoping move (do this out loud first):** "Attention Is All You Need" is a whole architecture. Tell the interviewer you'll implement the **core: scaled dot-product attention + multi-head**, on a toy input — not the full encoder-decoder, positional encodings, or training a translation model. Picking the core is itself a signal.

| Time | Block | What you produce |
|------|-------|------------------|
| 0:00–0:15 | **Read** (use the [three-pass method](/papers/lora-mock-interview/#part-0--how-to-read-a-paper-in-15-minutes-the-three-pass-method)) | The attention equation + why `√dₖ` + what multi-head buys |
| 0:15–0:20 | **Explain the benefit** out loud (cover Part 2) | Parallelism + constant path length vs RNNs |
| 0:20–0:55 | **Implement** from the stub (Part 3) | `scaled_dot_product_attention` + `MultiHeadAttention`, runs forward/backward |
| last 10 min | **Sanity-check** (Part 4) | Shapes, rows sum to 1, causal mask, scaling — narrated |

### Self-grading rubric — "what good looks like"
- ✅ **Scoped** to the core instead of trying to build a Transformer.
- ✅ Explained the benefit vs RNNs concretely: **parallel over positions** + **O(1) path length** between any two tokens.
- ✅ Could justify the **`1/√dₖ`** scaling (softmax saturation / vanishing gradients), not just recite it.
- ✅ Got the **multi-head reshape** right (`split → attend per head → concat → project`) and handled **masking**.
- ✅ Knew the cost: **O(n²·d)** — quadratic in sequence length.
- ⚠️ Red flags: forgetting the scale, wrong softmax axis, leaking future tokens (bad causal mask), reshape that mixes head/feature dims.

---

## Part 1 — Structured read of THIS paper

### The 30-second summary (the "benefit")
RNNs process a sequence **step by step**, so training can't parallelize across time and information between distant tokens must pass through many intermediate steps (long gradient paths → forgetting). Self-attention lets **every position directly attend to every other position in one operation**. The payoff:

- **Parallelism:** the whole sequence is processed at once (matmuls), not sequentially → far better hardware utilization, faster training.
- **Constant path length:** any two tokens interact directly (O(1) hops), so long-range dependencies are easy to learn.
- **Content-based addressing:** each token retrieves information from others by *similarity*, and **multiple heads** attend to different relationships in parallel.
- Cost: **O(n²·d)** compute/memory in sequence length `n` — the well-known quadratic bottleneck.

### The core idea (Method — you implement this)
Given queries $Q$, keys $K$, values $V$ (each a matrix of row-vectors):

$$\text{Attention}(Q,K,V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right) V$$

**Multi-head:** project $Q,K,V$ into `h` lower-dimensional subspaces, run attention in each independently, concatenate, and project back:

$$\text{MultiHead}(x) = \big[\text{head}_1, \dots, \text{head}_h\big]\,W^O, \quad \text{head}_i = \text{Attention}(xW_i^Q, xW_i^K, xW_i^V)$$

Key details (the things an interviewer probes):
- **The `√dₖ` scaling.** Dot products of two `dₖ`-dim vectors have variance ∝ `dₖ`; large values push softmax into saturated regions where gradients vanish. Dividing by `√dₖ` keeps the logits' variance ~1.
- **Softmax over the *key* axis** (last dim of the `Tq × Tk` score matrix) — each query forms a distribution over keys.
- **Masking.** Set disallowed scores to `−∞` **before** softmax so they get exactly 0 weight. Causal mask = upper-triangular block (position `t` can't see `>t`); padding mask hides pad tokens.
- **Multi-head reshape.** `d_model = h · dₖ`. Split the projected vectors into `h` heads, attend per head, concat back to `d_model`, then apply `W^O`. The reshape must keep head and feature dims separate.
- **Permutation equivariance.** Bare self-attention has no notion of order — permuting the input permutes the output identically. That's why Transformers add **positional encodings** (out of scope here, but name it).

### Where the evidence lives (tables that matter)
- **Main MT results (Table 2):** Transformer beats prior models at lower training cost → the parallelism/quality claim.
- **Complexity table (Table 1):** self-attention vs recurrent/conv on per-layer complexity, sequential ops, and max path length → the O(1) path-length and parallelism argument, made precise.
- **Ablations (Table 3):** number of heads, `dₖ`, etc. → multi-head and scaling justified empirically.

### The honest limitations (have an opinion)
- **O(n²) in sequence length** — the dominant scaling pain; spawned a whole literature (sparse/linear/flash attention).
- **No inductive bias for locality or order** on its own — needs positional information and lots of data.
- **Memory** for the `n×n` attention matrix is the practical bottleneck for long context (what FlashAttention addresses by not materializing it).

---

## Part 2 — The interview dialogue (interviewer ⇄ interviewee)

> **🧑‍💼 Interviewer:** One paragraph — why did this replace RNNs?
>
> **🧑‍💻 Interviewee:** Two reasons. First, parallelism: an RNN must process tokens in sequence, so it can't use the hardware fully and training is slow; self-attention computes all positions at once as matmuls. Second, path length: in an RNN, information between two distant tokens crosses many steps and tends to vanish, whereas attention connects any two positions directly in one hop, so long-range dependencies are easy. The cost is that attention is quadratic in sequence length.

> **🧑‍💼 Interviewer:** Why divide by `√dₖ`? What breaks if you don't?
>
> **🧑‍💻 Interviewee:** Each score is a dot product of two `dₖ`-dimensional vectors, so its variance grows with `dₖ`. Large logits push softmax into a near-one-hot, saturated regime where its gradient is almost zero — training stalls. Dividing by `√dₖ` normalizes the variance back to ~1 so softmax stays in a responsive range. You can show it: the std of `QKᵀ` scales like `√dₖ` without the correction.

> **🧑‍💼 Interviewer:** What do multiple heads give you that one big head doesn't?
>
> **🧑‍💻 Interviewee:** Each head attends in its own subspace, so different heads can capture different relationships — say, one tracking syntactic dependencies and another tracking coreference — and they run in parallel. One big head of the same total width has to average all those patterns into a single attention distribution, which is less expressive. Same parameter/compute budget, more representational diversity.

> **🧑‍💼 Interviewer:** How does causal masking work, and why `−∞`?
>
> **🧑‍💻 Interviewee:** For autoregressive decoding a position must not see the future. I add a mask to the score matrix setting all entries where the key index is greater than the query index to `−∞` **before** the softmax. After softmax those become exactly 0, so no weight leaks from future tokens. Using `−∞` rather than a big negative keeps it exact and numerically clean.

> **🧑‍💼 Interviewer:** What's the catch with attention at scale?
>
> **🧑‍💻 Interviewee:** It's O(n²) in time and memory — the `n×n` score matrix. For long context that's the bottleneck, which is why there's so much work on sparse, linear, and IO-aware (FlashAttention) variants that avoid materializing the full matrix.

> **🧑‍💼 Interviewer:** Implement scaled dot-product + multi-head and show it doing content-based retrieval.

---

## Part 3 — Implementation

```python
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def scaled_dot_product_attention(q, k, v, mask=None):
    """q,k,v: (..., T, d_k). Returns (output, attention_weights)."""
    d_k = q.size(-1)
    scores = (q @ k.transpose(-2, -1)) / math.sqrt(d_k)      # (..., Tq, Tk)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float("-inf"))  # block disallowed keys
    attn = scores.softmax(dim=-1)                            # distribution over keys
    return attn @ v, attn                                    # (..., Tq, d_k), (..., Tq, Tk)


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model, self.n_heads, self.d_k = d_model, n_heads, d_model // n_heads
        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)
        self.wo = nn.Linear(d_model, d_model)

    def _split(self, x):                          # (B,T,d_model) -> (B,heads,T,d_k)
        B, T, _ = x.shape
        return x.view(B, T, self.n_heads, self.d_k).transpose(1, 2)

    def forward(self, x, mask=None):
        q, k, v = self._split(self.wq(x)), self._split(self.wk(x)), self._split(self.wv(x))
        out, attn = scaled_dot_product_attention(q, k, v, mask)   # (B,heads,T,d_k)
        B, _, T, _ = out.shape
        out = out.transpose(1, 2).contiguous().view(B, T, self.d_model)  # concat heads
        return self.wo(out), attn
```

### Why each line matters (talk through it)
- `/ math.sqrt(d_k)` — the variance fix; mention it *before* the interviewer asks.
- `masked_fill(mask == 0, -inf)` **before** softmax — masked keys get exactly zero weight, not "small."
- `softmax(dim=-1)` — over keys, so each query row is a probability distribution that sums to 1.
- `view(B,T,heads,d_k).transpose(1,2)` — splits into heads **without mixing** the head and feature dimensions; the `transpose` puts heads on a batch-like axis so attention runs per head.
- `transpose(1,2).contiguous().view(B,T,d_model)` — the inverse: concatenate heads back, then `wo` mixes them.

### Demonstrating the benefit (content-based retrieval, no positions needed)
Each sequence has one "special" token flagged in feature 0; its payload lives in feature 1. The target is that payload. The layer must learn to **attend by content** to the flagged token and read its payload — something an RNN would have to carry across steps, and which needs *no positional encoding* because it's purely content-based.

```python
torch.manual_seed(0)
B, T, d = 256, 6, 16
mha = MultiHeadAttention(d_model=d, n_heads=1)        # 1 head so we can read the attention map
readout = nn.Linear(d, 1)
opt = torch.optim.Adam(list(mha.parameters()) + list(readout.parameters()), lr=3e-3)

def make_batch():
    x = torch.randn(B, T, d) * 0.5
    special = torch.randint(0, T, (B,))              # which token is flagged
    payload = torch.randn(B)
    x[torch.arange(B), special, 0] = 3.0            # flag in feature 0
    x[torch.arange(B), special, 1] = payload        # payload in feature 1
    return x, payload.unsqueeze(1), special

for step in range(800):
    x, y, _ = make_batch()
    out, _ = mha(x)                                 # (B,T,d)
    pred = readout(out.mean(dim=1))                 # pool over positions -> (B,1)
    loss = F.mse_loss(pred, y)
    opt.zero_grad(); loss.backward(); opt.step()
    if step % 200 == 0:
        print(f"step {step:3d}  loss {loss.item():.4f}")

# Did attention learn to point at the flagged token?
x, y, special = make_batch()
_, attn = mha(x)                                    # (B,1,Tq,Tk)
avg_over_queries = attn[:, 0].mean(dim=1)           # (B, Tk)
on_flag = avg_over_queries[torch.arange(B), special].mean().item()
print(f"avg attention mass on the flagged token: {on_flag:.2f} (chance = {1/T:.2f})")
```

The loss should fall and the attention mass on the flagged token should end up **well above chance (`1/T`)** — the layer learned content-based addressing.

---

## Part 4 — Sanity checks (don't skip)

### Check 1 — Output shape matches input
```python
mha = MultiHeadAttention(d_model=32, n_heads=4)
x = torch.randn(8, 10, 32)
out, attn = mha(x)
assert out.shape == x.shape and attn.shape == (8, 4, 10, 10)
print("OK: shapes", out.shape, attn.shape)
```

### Check 2 — Attention rows are probability distributions
```python
assert torch.allclose(attn.sum(dim=-1), torch.ones(8, 4, 10), atol=1e-5)
assert (attn >= 0).all()
print("OK: each query's attention sums to 1 and is non-negative")
```

### Check 3 — Causal mask leaks no future
```python
T = 10
causal = torch.tril(torch.ones(T, T)).view(1, 1, T, T)   # lower-triangular = allowed
_, attn = mha(x, mask=causal)
upper = torch.triu(torch.ones(T, T), diagonal=1).bool()
assert attn[..., upper].abs().max() < 1e-6, "future tokens received weight!"
print("OK: causal mask -> zero weight above the diagonal")
```

### Check 4 — The √dₖ scaling actually controls logit variance
```python
import math
for d_k in (8, 64, 512):
    q, k = torch.randn(2000, d_k), torch.randn(2000, d_k)
    raw    = (q * k).sum(-1)                      # unscaled dot products
    scaled = raw / math.sqrt(d_k)
    print(f"d_k={d_k:3d}  std(raw)={raw.std():6.2f}  std(scaled)={scaled.std():.2f}")
# std(raw) grows ~sqrt(d_k); std(scaled) stays ~1
```

### Check 5 — Permutation equivariance (no positional info)
Permute the tokens; the output permutes the same way (bare attention is order-agnostic).
```python
mha.eval()
x = torch.randn(1, 6, 32)
perm = torch.randperm(6)
out_a, _ = mha(x)
out_b, _ = mha(x[:, perm])
assert torch.allclose(out_a[:, perm], out_b, atol=1e-5)
print("OK: attention is permutation-equivariant -> why we need positional encodings")
```

### Check 6 — Gradients flow to all four projections
```python
mha.train()
out, _ = mha(torch.randn(4, 7, 32))
out.sum().backward()
for name in ("wq", "wk", "wv", "wo"):
    g = getattr(mha, name).weight.grad
    assert g is not None and g.abs().sum() > 0, f"{name} got no gradient"
print("OK: gradients reach wq, wk, wv, wo")
```

---

## Part 5 — Likely follow-up questions

- *"Self-attention vs cross-attention?"* — Same math; in self-attention Q, K, V all come from the same sequence, in cross-attention Q comes from one sequence (e.g. decoder) and K, V from another (e.g. encoder output).
- *"Why is it O(n²)? How is that mitigated?"* — The `n×n` score matrix. Mitigations: sparse/local attention, low-rank/linear attention (Performer, Linformer), and IO-aware exact attention (**FlashAttention**) that never materializes the full matrix.
- *"Where do positional encodings go and why?"* — Added to the inputs (sinusoidal or learned; modern LLMs use RoPE). Needed because attention itself is permutation-equivariant and has no notion of order.
- *"What's the role of `W^O`?"* — After concatenating heads, `W^O` mixes information across heads into the model dimension; without it heads stay in separate subspaces.
- *"KV cache?"* — At autoregressive inference you cache past K and V so each new token is O(n) instead of recomputing all of them — the practical reason decoding is feasible.

---

## TL;DR cheat sheet
| Thing | Answer |
|---|---|
| Core formula | `softmax(QKᵀ/√dₖ) V` |
| Why `√dₖ` | keeps logit variance ~1 → softmax not saturated |
| Softmax axis | over keys (last dim) |
| Multi-head | split → attend per head → concat → `W^O` |
| Masking | set disallowed scores to −∞ **before** softmax |
| Benefit vs RNN | parallel over positions + O(1) path length |
| Cost | O(n²·d) in sequence length |
| No order on its own | permutation-equivariant → need positional encodings |
