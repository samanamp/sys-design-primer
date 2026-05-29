---
title: GQA / MQA — Paper-to-Code Mock Interview
description: A combined mock (read paper, explain benefit, implement in Colab) using Grouped-Query and Multi-Query Attention as the worked example.
sidebar:
  order: 12
  label: GQA / MQA
---

> **Papers:** *GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints* — Ainslie et al., 2023. arXiv: [2305.13245](https://arxiv.org/abs/2305.13245) · and *Fast Transformer Decoding: One Write-Head Is All You Need* (Multi-Query Attention) — Shazeer, 2019. arXiv: [1911.02150](https://arxiv.org/abs/1911.02150)
>
> **Format:** Read (~15 min) → explain the *real* benefit → implement the core idea in Colab → sanity-check it.
>
> **Companion notebook:** [`gqa_mock.ipynb`](/notebooks/gqa_mock.ipynb) (download) — content-retrieval demo + a `GroupedQueryAttention` stub to fill in, plus verification cells (including the exact KV-cache ratio). Open in [Google Colab](https://colab.research.google.com/) via *File → Upload notebook*.
>
> **Difficulty:** 🟡 Medium. Builds directly on [the attention mock](/papers/attention-mock-interview/) — do that one first. GQA is a small, countable variant of multi-head attention.

---

## How to run this as a timed drill (~60 min)

> ⚠️ **Scoping move (do this out loud first):** GQA is a *variant* of multi-head attention, not a new architecture. Tell the interviewer you'll implement **scaled dot-product + grouped-query attention** on a toy input — reusing the standard MHA pattern — and you'll show the headline win is the **KV-cache size**, which is exactly countable, while the *speed* win needs real-scale serving.

| Time | Block | What you produce |
|------|-------|------------------|
| 0:00–0:15 | **Read** (use the [three-pass method](/papers/lora-mock-interview/#part-0--how-to-read-a-paper-in-15-minutes-the-three-pass-method)) | Why the KV cache dominates decode + what `G` groups buy |
| 0:15–0:20 | **Explain the benefit** out loud (cover Part 2) | KV-cache memory/bandwidth, MQA↔GQA↔MHA spectrum |
| 0:20–0:50 | **Implement** from the stub (Part 3) | `GroupedQueryAttention` that reduces to MHA and MQA |
| last 10 min | **Sanity-check** (Part 4) | Shapes, group mapping, rows sum to 1, KV-cache ratio — narrated |

### Self-grading rubric — "what good looks like"
- ✅ **Framed the problem as inference, not training:** at autoregressive decode the **KV cache** dominates memory and memory-bandwidth.
- ✅ Placed MQA, GQA, MHA on one spectrum: **`G=1` is MQA, `G=H` is full MHA**, in between is GQA.
- ✅ Got the **`repeat_interleave`** (not `repeat`) right so each KV head is shared by `H/G` *consecutive* query heads.
- ✅ Quoted the cache reduction as a clean ratio: **KV cache scales with `G/H`**.
- ✅ Was honest that the **speed** win shows up only at serving scale; the **memory** win is exactly countable.
- ⚠️ Red flags: claiming it saves **training FLOPs** (it doesn't, materially), using `repeat` instead of `repeat_interleave` (wrong group mapping), shrinking the *query* heads instead of the *KV* heads.

---

## Part 1 — Structured read of THIS paper

### The 30-second summary (the "benefit")
A standard multi-head Transformer keeps `H` separate key and value heads. At autoregressive inference you must **cache** every past token's K and V (the "KV cache"), and the decoder is **memory-bandwidth bound** — each generated token re-reads the whole cache. That cache scales with the **number of KV heads**, so it, not the matmuls, is what limits batch size and decode speed. The fix:

- **MQA (Shazeer 2019):** share a **single** key/value head across *all* query heads. The KV cache shrinks by a factor of `H`, decoding gets much faster — but quality can drop and training can get unstable.
- **GQA (Ainslie 2023):** generalize MQA. Use `G` KV-head **groups** (`1 ≤ G ≤ H`); each KV head is shared by `H/G` query heads. `G=1` recovers MQA, `G=H` recovers full MHA, and a middle `G` recovers **most of MHA's quality with most of MQA's savings**.
- **Uptraining:** you can convert an existing MHA checkpoint to GQA by mean-pooling the K/V heads into groups and fine-tuning for a small fraction (~5%) of original compute — no need to train from scratch.
- The win is an **inference memory/bandwidth** win, **not** a training-FLOPs win.

### The core idea (Method — you implement this)
Project queries to `H` heads as usual, but project keys and values to only `G` heads. Then **replicate** each KV head so each query head sees its group's shared K, V, and run ordinary scaled dot-product attention:

$$\text{head}_i = \text{Attention}\!\left(xW_i^Q,\; xW_{\lceil i / (H/G) \rceil}^K,\; xW_{\lceil i / (H/G) \rceil}^V\right), \qquad i = 1, \dots, H$$

$$\text{Attention}(Q,K,V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V$$

Concretely, after projecting `K, V` to `G` heads you `repeat_interleave` them `H/G` times along the head axis so head index `i` maps to KV group $\lfloor i/(H/G) \rfloor$, then concatenate the per-head outputs and apply $W^O$.

Key details (the things an interviewer probes):
- **Only the *KV* heads shrink.** Query heads stay at `H`; expressivity of the queries is unchanged. `W^K` and `W^V` are smaller (`G·d_k` wide instead of `H·d_k`).
- **`repeat_interleave`, not `repeat`.** Interleave gives `[kv0, kv0, kv1, kv1, ...]` so *consecutive* query heads share a group. Plain `repeat` gives `[kv0, kv1, kv0, kv1, ...]` — the wrong mapping.
- **The cache ratio is `G/H`.** KV-cache element count `= 2 · B · G · T · d_k`; MHA is `2 · B · H · T · d_k`. Ratio `= G/H` (MQA `= 1/H`). That's the headline number.
- **`H % G == 0`.** Groups must divide evenly so every KV head serves the same number of query heads.
- **Why it's bandwidth, not compute:** the attention *matmuls* are roughly unchanged after replication; the win is that you **store and stream** `G/H` as much K/V per token during decode.

### Where the evidence lives (tables/figures that matter)
- **Quality-vs-speed curve (the main figure):** GQA sits near MHA quality at close to MQA speed — the "most of both" claim. *(Figures here are approximate; treat them as directional.)*
- **Uptraining ablation:** converting MHA→GQA with a small fraction of pretraining compute recovers quality → the "from multi-head checkpoints" point.
- **`G` sweep:** quality rises and inference cost rises as `G` grows from 1 (MQA) toward `H` (MHA) → GQA is a tunable knob, and a modest `G` (e.g. 8) is the sweet spot.

### The honest limitations (have an opinion)
- **No training-FLOPs savings.** Forward/backward cost is essentially the same as MHA; the benefit is purely at **inference** (cache size + bandwidth). Don't oversell it.
- **The speed win is workload-dependent.** It shows up when you're memory-bandwidth bound (long context, large batch, big model). On a tiny toy it's invisible — only the *cache size* is countable.
- **Still some quality cost vs MHA.** GQA recovers *most* of it, not all; MQA more so. You're trading a little quality for a lot of cache.

---

## Part 2 — The interview dialogue (interviewer ⇄ interviewee)

> **🧑‍💼 Interviewer:** One paragraph — what does GQA actually buy me?
>
> **🧑‍💻 Interviewee:** It shrinks the KV cache at inference. During autoregressive decoding you cache every past token's keys and values, and decode is memory-bandwidth bound — each new token re-reads the whole cache — so cache size, which scales with the number of KV heads, is the real bottleneck. GQA keeps all `H` query heads but uses only `G` key/value heads, each shared by `H/G` query heads, so the cache scales by `G/H`. `G=1` is MQA (maximum savings, some quality loss), `G=H` is plain multi-head, and a middle `G` keeps most of the quality with most of the savings. It's an inference memory and bandwidth win, not a training-FLOPs win.

> **🧑‍💼 Interviewer:** Why is the KV cache, not the matmuls, the thing that hurts at decode?
>
> **🧑‍💻 Interviewee:** At generation you produce one token at a time, and for each one you read back all the cached K and V for the context. The arithmetic per step is small, but the memory traffic is large and grows with sequence length and the number of KV heads — so you're bandwidth bound, and the cache also caps how big a batch fits in memory. Cutting KV heads from `H` to `G` cuts both the bytes stored and the bytes streamed per step by `G/H`.

> **🧑‍💼 Interviewer:** Why `repeat_interleave` and not `repeat`?
>
> **🧑‍💻 Interviewee:** Because the grouping has to be contiguous. `repeat_interleave(H/G)` turns `[kv0, kv1]` into `[kv0, kv0, kv1, kv1]`, so query heads 0 and 1 share KV head 0 and heads 2 and 3 share KV head 1 — the intended group layout. Plain `repeat` tiles the whole block, `[kv0, kv1, kv0, kv1]`, which assigns query heads to the wrong groups. Same shapes, wrong semantics.

> **🧑‍💼 Interviewer:** Does GQA save training compute?
>
> **🧑‍💻 Interviewee:** Not meaningfully. After you replicate the KV heads the attention math is basically the same as MHA, and the projection savings on `W^K`/`W^V` are tiny relative to the rest of the model. The whole point is inference: smaller cache, less bandwidth, bigger batches, faster decode. And practically you don't even retrain — you uptrain an existing MHA checkpoint by mean-pooling its K/V heads into `G` groups and fine-tuning briefly.

> **🧑‍💼 Interviewer:** How would you pick `G`?
>
> **🧑‍💻 Interviewee:** It's a quality-vs-cache knob. `G=1` (MQA) gives the smallest cache but the biggest quality risk; `G=H` is full quality with no savings. In practice a modest `G` like 8 recovers nearly all of MHA's quality while still giving a large cache reduction, so I'd sweep `G` against my latency/memory budget and pick the smallest `G` that holds quality.

> **🧑‍💼 Interviewer:** Implement GQA so it reduces to both MHA and MQA, and show the exact cache reduction.

---

## Part 3 — Implementation

GQA is standard multi-head attention with one twist: project K, V to `G` heads, then `repeat_interleave` them up to `H` before attending.

```python
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def scaled_dot_product_attention(q, k, v, mask=None):
    """q,k,v: (..., T, d_k). Returns (output, attention_weights)."""
    d_k = q.size(-1)
    scores = (q @ k.transpose(-2, -1)) / math.sqrt(d_k)        # (..., Tq, Tk)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float("-inf"))  # block disallowed keys
    attn = scores.softmax(dim=-1)                             # distribution over keys
    return attn @ v, attn


class GroupedQueryAttention(nn.Module):
    """n_heads query heads, n_kv_heads (=G) shared key/value heads.
    G == n_heads -> standard MHA.  G == 1 -> MQA."""

    def __init__(self, d_model, n_heads, n_kv_heads):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        assert n_heads % n_kv_heads == 0, "n_heads must be divisible by n_kv_heads"
        self.d_model, self.n_heads, self.n_kv_heads = d_model, n_heads, n_kv_heads
        self.d_k = d_model // n_heads
        self.group_size = n_heads // n_kv_heads                 # H/G query heads per KV head
        self.wq = nn.Linear(d_model, n_heads * self.d_k)        # H heads
        self.wk = nn.Linear(d_model, n_kv_heads * self.d_k)     # G heads (smaller!)
        self.wv = nn.Linear(d_model, n_kv_heads * self.d_k)     # G heads (smaller!)
        self.wo = nn.Linear(n_heads * self.d_k, d_model)

    def _split(self, x, n):                       # (B,T,n*d_k) -> (B,n,T,d_k)
        B, T, _ = x.shape
        return x.view(B, T, n, self.d_k).transpose(1, 2)

    def forward(self, x, mask=None):
        q = self._split(self.wq(x), self.n_heads)       # (B,H,T,d_k)
        k = self._split(self.wk(x), self.n_kv_heads)    # (B,G,T,d_k)
        v = self._split(self.wv(x), self.n_kv_heads)    # (B,G,T,d_k)
        k = k.repeat_interleave(self.group_size, dim=1) # (B,H,T,d_k): each KV head shared H/G times
        v = v.repeat_interleave(self.group_size, dim=1) # (B,H,T,d_k)
        out, attn = scaled_dot_product_attention(q, k, v, mask)
        B, _, T, _ = out.shape
        out = out.transpose(1, 2).contiguous().view(B, T, self.n_heads * self.d_k)
        return self.wo(out), attn                       # concat heads, then mix with W^O
```

### Why each line matters (talk through it)
- `wk`/`wv` are `n_kv_heads * d_k` wide, not `d_model` — this is the *only* structural change vs MHA, and it's what shrinks the cache.
- `repeat_interleave(group_size, dim=1)` — replicates KV head `g` into `group_size` consecutive slots so query heads `[g·H/G … (g+1)·H/G)` all attend to it. **Interleave, not `repeat`.**
- `/ math.sqrt(d_k)` — the usual variance fix for the softmax (carried over from plain attention).
- `softmax(dim=-1)` — over keys, so each query row is a distribution that sums to 1.
- `group_size == 1` (when `n_kv_heads == n_heads`) makes `repeat_interleave` a no-op → the module **is** standard MHA. `n_kv_heads == 1` makes one KV head feed all query heads → **MQA**.

### Demonstrating the benefit (content-based retrieval + the cache ratio)
We reuse the retrieval task from the attention mock: each sequence has one flagged token (feature 0) whose payload (feature 1) is the target; the layer must **attend by content** to it. We train MHA (`G=H`), GQA (`G=2`), and MQA (`G=1`) and check they all learn it about equally well — then we compute the **exact** KV-cache ratio `G/H`.

```python
def make_batch(B, T, d):
    x = torch.randn(B, T, d) * 0.5
    special = torch.randint(0, T, (B,))
    payload = torch.randn(B)
    x[torch.arange(B), special, 0] = 3.0      # flag in feature 0
    x[torch.arange(B), special, 1] = payload  # payload in feature 1
    return x, payload.unsqueeze(1), special

def train_model(n_kv_heads, steps=800, seed=0):
    torch.manual_seed(seed)
    B, T, d, H = 256, 6, 16, 4
    gqa, readout = GroupedQueryAttention(d, H, n_kv_heads), nn.Linear(d, 1)
    opt = torch.optim.Adam(list(gqa.parameters()) + list(readout.parameters()), lr=3e-3)
    for step in range(steps):
        x, y, _ = make_batch(B, T, d)
        loss = F.mse_loss(readout(gqa(x)[0].mean(dim=1)), y)
        opt.zero_grad(); loss.backward(); opt.step()
    x, y, special = make_batch(B, T, d)
    attn = gqa(x)[1]                                    # (B,H,Tq,Tk)
    on_flag = attn.mean(dim=(1, 2))[torch.arange(B), special].mean().item()
    return loss.item(), on_flag

H, chance = 4, 1 / 6
for G in (4, 2, 1):
    loss, flag = train_model(n_kv_heads=G)
    name = {4: "MHA", 2: "GQA", 1: "MQA"}[G]
    print(f"{name} (G={G}): loss {loss:.4f}  attn-on-flag {flag:.2f} "
          f"(chance {chance:.2f})  KV-cache = {G}/{H} = {G/H:.2f} of MHA")
```

You should see all three drive the loss low and put attention mass on the flagged token **well above chance** — GQA and MQA reach roughly MHA quality on this toy — while the KV cache prints `1.00`, `0.50`, `0.25` of MHA. The *quality parity* is approximate and seed-dependent; the *cache ratio* is exact. (The **speed** benefit needs real-scale, bandwidth-bound serving to show up — but the **cache-size** reduction is countable right here.)

---

## Part 4 — Sanity checks (don't skip)

### Check 1 — Q has `n_heads`, K/V have `n_kv_heads`; output shape == input
```python
gqa = GroupedQueryAttention(d_model=32, n_heads=8, n_kv_heads=2)
x = torch.randn(4, 10, 32)
q = gqa._split(gqa.wq(x), gqa.n_heads)
k = gqa._split(gqa.wk(x), gqa.n_kv_heads)
out, _ = gqa(x)
assert q.shape == (4, 8, 10, 4) and k.shape == (4, 2, 10, 4)
assert out.shape == x.shape
print("OK: Q has 8 heads, K/V have 2; out", out.shape, "== in")
```

### Check 2 — `repeat_interleave` maps each query head to the correct KV group
```python
gqa = GroupedQueryAttention(d_model=16, n_heads=4, n_kv_heads=2)
xb = torch.randn(2, 5, 16)
k = gqa._split(gqa.wk(xb), gqa.n_kv_heads)            # (B,2,T,d_k)
k_rep = k.repeat_interleave(gqa.group_size, dim=1)   # (B,4,T,d_k)
assert torch.equal(k_rep[:, 0], k[:, 0]) and torch.equal(k_rep[:, 1], k[:, 0])  # heads 0,1 -> KV0
assert torch.equal(k_rep[:, 2], k[:, 1]) and torch.equal(k_rep[:, 3], k[:, 1])  # heads 2,3 -> KV1
print("OK: group_size", gqa.group_size, "-> query heads [0,1]->KV0, [2,3]->KV1")
```

### Check 3 — Attention rows are probability distributions
```python
gqa = GroupedQueryAttention(d_model=32, n_heads=8, n_kv_heads=4)
_, attn = gqa(torch.randn(3, 7, 32))
assert torch.allclose(attn.sum(dim=-1), torch.ones(3, 8, 7), atol=1e-5)
assert (attn >= 0).all()
print("OK: each query's attention sums to 1 and is non-negative")
```

### Check 4 — KV-cache element count = `G/H` of full MHA
```python
B, T, d_model, H, G = 1, 128, 512, 16, 4
d_k = d_model // H
mha_kv = 2 * B * H * T * d_k          # 2 = (K and V), all H heads
gqa_kv = 2 * B * G * T * d_k          # only G KV heads cached
ratio = gqa_kv / mha_kv
assert abs(ratio - G / H) < 1e-9
print(f"OK: KV-cache GQA/MHA = {gqa_kv}/{mha_kv} = {ratio:.3f} == G/H = {G/H:.3f}")
```

### Check 5 — `n_kv_heads == n_heads` reduces to standard MHA
With `G=H`, `group_size==1` so `repeat_interleave` is a no-op and the forward matches a plain MHA forward built from the same projections.
```python
gqa = GroupedQueryAttention(d_model=32, n_heads=4, n_kv_heads=4)
assert gqa.group_size == 1
xb = torch.randn(2, 6, 32)
out_gqa, _ = gqa(xb)
qm, km, vm = (gqa._split(p, 4) for p in (gqa.wq(xb), gqa.wk(xb), gqa.wv(xb)))
om, _ = scaled_dot_product_attention(qm, km, vm)         # no sharing
om = gqa.wo(om.transpose(1, 2).contiguous().view(2, 6, 32))
assert torch.allclose(out_gqa, om, atol=1e-6)
print("OK: n_kv_heads == n_heads is exactly standard MHA")
```

### Check 6 — Gradients flow to q, k, v, o projections
```python
gqa = GroupedQueryAttention(d_model=32, n_heads=8, n_kv_heads=2)
gqa(torch.randn(4, 7, 32))[0].sum().backward()
for name in ("wq", "wk", "wv", "wo"):
    g = getattr(gqa, name).weight.grad
    assert g is not None and g.abs().sum() > 0, f"{name} got no gradient"
print("OK: gradients reach wq, wk, wv, wo")
```

---

## Part 5 — Likely follow-up questions

- *"GQA vs MQA vs MHA in one sentence?"* — One spectrum on the number of KV heads `G`: `G=1` is MQA (one shared K/V head, smallest cache), `G=H` is full multi-head, and `1<G<H` is GQA (most of MHA's quality, most of MQA's savings).
- *"How do you convert an existing MHA model to GQA?"* — Uptraining: mean-pool the `H` K/V heads into `G` groups to initialize the smaller `W^K`/`W^V`, then fine-tune for a small fraction (~5%) of original pretraining compute. No training from scratch.
- *"Why doesn't this help training cost?"* — After replicating KV heads the attention matmuls are essentially MHA-sized; the only saving is the smaller `W^K`/`W^V` projections, which is negligible. The win is inference cache + bandwidth.
- *"How does this interact with the KV cache and the attention mock's `O(n²)`?"* — `O(n²)` per-step compute is unchanged; GQA attacks the *cache* (memory `O(n)` per KV head). It composes with FlashAttention (compute/IO) — they fix different bottlenecks.
- *"What about RoPE / positional encodings?"* — Apply rotary embeddings to Q and to the `G` K heads *before* `repeat_interleave`; values are untouched. Position handling is orthogonal to grouping.

---

## TL;DR cheat sheet
| Thing | Answer |
|---|---|
| Core idea | Keep `H` query heads, use only `G` shared KV heads; replicate K/V `H/G`× |
| Spectrum | `G=1` MQA · `G=H` MHA · `1<G<H` GQA |
| Key op | `K,V.repeat_interleave(H/G, dim=heads)` (interleave, not `repeat`) |
| Benefit | Smaller **KV cache** + less decode bandwidth at inference |
| Cache ratio | `G/H` of MHA (MQA = `1/H`) |
| NOT a benefit | Training-FLOPs savings (negligible) |
| Conversion | Uptrain from MHA checkpoint: mean-pool K/V heads, brief fine-tune |
| Constraint | `H % G == 0` |
| Cost | Some quality loss vs MHA (small for modest `G`) |
