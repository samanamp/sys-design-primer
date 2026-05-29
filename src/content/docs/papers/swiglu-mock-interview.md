---
title: SwiGLU — Paper-to-Code Mock Interview
description: A mock interview (read paper, explain the benefit, implement in Colab) using the GLU-variant SwiGLU feed-forward block as the worked example.
sidebar:
  order: 4
  label: SwiGLU
---

> **Paper:** *GLU Variants Improve Transformer* — Noam Shazeer, 2020. [arXiv:2002.05202](https://arxiv.org/abs/2002.05202)
>
> **Format:** Read (~12 min) → explain the *real* benefit → implement the core idea in Colab → sanity-check it.
>
> **Companion notebook:** [`swiglu_mock.ipynb`](/notebooks/swiglu_mock.ipynb) (download) — param-matched SwiGLU-vs-ReLU FFN demo + a `SwiGLUFFN` stub to fill in, plus verification cells. Open in [Google Colab](https://colab.research.google.com/) via *File → Upload notebook*.
>
> **Difficulty:** 🟢🟡 Easy-to-medium. The code is short; the subtlety is the parameter-matching arithmetic and being honest that the benefit is *empirical*, not theoretical.

---

## How to run this as a timed drill (~40 min)

| Time | Block | What you produce |
|------|-------|------------------|
| 0:00–0:12 | **Read** (use the [three-pass method](/papers/lora-mock-interview/#part-0--how-to-read-a-paper-in-15-minutes-the-three-pass-method)) | What a GLU variant is + why hidden size shrinks to `8d/3` |
| 0:12–0:17 | **Explain the benefit** out loud (cover Part 2) | Gated FFN intuition + "the win is empirical" |
| 0:17–0:33 | **Implement** from the stub (Part 3) | A working `SwiGLUFFN`, param-matched, + an honest read of the toy result |
| last 5 min | **Sanity-check** (Part 4) | All 6 checks passing, narrated out loud |

### Self-grading rubric — "what good looks like"
- ✅ Described SwiGLU as **replacing** `Linear→ReLU→Linear` with a **gated** block `Swish(xW_gate) ⊙ (xW_up)`, then `W_down`.
- ✅ Knew the **three-matrix → scale hidden by 2/3** trick, so params/compute stay matched to a `4d` ReLU-FFN.
- ✅ Was **honest** that the paper gives no theory — Shazeer attributes the gains "to divine benevolence." The benefit is empirical.
- ✅ Could name **where it shipped**: LLaMA, PaLM, Mistral, etc.
- ⚠️ Red flags: claiming SwiGLU "adds capacity for free" (it's param-matched), forgetting the `2/3` hidden scaling, inventing a theoretical justification the paper explicitly disclaims.

---

## Part 1 — Structured read of THIS paper

### The 30-second summary (the "benefit")
The standard Transformer FFN is `FFN(x) = W_down · ReLU(W_up · x)` — two matrices and a pointwise nonlinearity. This paper asks: what if we replace ReLU with a **Gated Linear Unit (GLU)** variant? A GLU multiplies one linear projection by a (nonlinearly activated) gating projection. **SwiGLU** uses the **Swish/SiLU** activation on the gate:

- It's a **drop-in replacement** for the FFN sublayer — same place in the Transformer, same input/output shape.
- At **equal parameter count and compute**, GLU variants (SwiGLU, GEGLU) reach **lower perplexity** and better downstream scores than ReLU/GELU FFNs in the paper's experiments.
- It is **cheap and boring to implement** — three matmuls and an elementwise multiply — which is exactly why it got adopted everywhere (LLaMA, PaLM, Mistral, …).

### The core idea (Method — you implement this)
A **GLU** combines two linear projections of the input, one of which is squashed by an activation `σ` and used as a multiplicative gate:

$$\text{GLU}(x) = (\sigma(x W_{\text{gate}})) \odot (x W_{\text{up}})$$

**SwiGLU** picks `σ = \text{Swish}` (a.k.a. SiLU), where Swish is:

$$\text{Swish}(z) = z \cdot \text{sigmoid}(z) = \frac{z}{1 + e^{-z}}$$

The full feed-forward block then projects the gated hidden state back to model dimension:

$$\text{FFN}_{\text{SwiGLU}}(x) = \big(\text{Swish}(x W_{\text{gate}}) \odot (x W_{\text{up}})\big)\, W_{\text{down}}$$

The crucial bookkeeping detail — **parameter matching**: a standard FFN has **two** weight matrices (`W_up`, `W_down`) with hidden size `h = 4d`. A gated FFN has **three** (`W_gate`, `W_up`, `W_down`). To keep params and FLOPs equal to the `4d` baseline, you shrink the hidden size by a factor of `2/3`:

$$h_{\text{gated}} = \tfrac{2}{3}\cdot 4d = \tfrac{8d}{3}$$

(LLaMA rounds this to a convenient multiple, e.g. of 256.) So a "bigger-looking" three-matrix block actually has the *same* budget as the two-matrix baseline.

Key details (the things an interviewer probes):
- **The gate is the whole point.** `Swish(xW_gate)` is a soft, learned, per-feature multiplier on `xW_up`. Because `Swish(0)=0`, a zero gate fully closes that channel.
- **Why Swish and not sigmoid?** GLU originally used sigmoid; Shazeer tries several activations (ReGLU, GEGLU, SwiGLU, Bilinear). SwiGLU/GEGLU win empirically. Swish is smooth and non-monotonic, with no hard zero region like ReLU.
- **No biases in practice.** The paper (and LLaMA) use **bias-free** linears in the FFN. That also makes "zero the gate ⇒ zero output" exactly true.
- **It's parameter-matched, not free capacity.** The `2/3` scaling is what makes the comparison fair — and the win still shows up.

### Where the evidence lives (tables that matter)
- **Pre-training perplexity table** (≈Table 1): GLU variants get **lower log-perplexity** than ReLU/GELU FFNs at matched params. *(Figures here are approximate from memory — re-check the PDF.)*
- **GLUE / SuperGLUE / SQuAD fine-tuning tables**: GEGLU/SwiGLU lead on most downstream tasks, confirming the perplexity gains transfer.
- **The "no explanation" line:** the paper closes by noting it offers **no theoretical reason** the variants help, attributing success "to divine benevolence." That candor is itself a talking point.

### The honest limitations (have an opinion)
- **No theory.** The paper is a careful empirical sweep, not a mechanism. If asked "why does it work," the honest answer is "it just does, at scale, repeatedly."
- **Three matmuls, not two.** Same params, but the kernel/IO pattern differs; well worth it in practice but not literally identical engineering.
- **Gains are modest per-token but compound at scale.** On a tiny toy you may see **no** advantage — even a regression, since the `2/3` hidden split narrows the layer; the headline result is an at-scale, multi-task pre-training improvement that toys don't reproduce.

---

## Part 2 — The interview dialogue (interviewer ⇄ interviewee)

> **🧑‍💼 Interviewer:** One paragraph — what is SwiGLU and what does it buy me?
>
> **🧑‍💻 Interviewee:** SwiGLU replaces the Transformer FFN's `Linear→ReLU→Linear` with a *gated* block: I compute two projections of the input, run Swish on one of them, multiply them elementwise, then project back down. So instead of a fixed pointwise nonlinearity, the network learns a data-dependent multiplicative gate. Empirically, at matched parameters and compute, it gives lower perplexity and better downstream metrics — which is why LLaMA, PaLM and friends all use it. The catch: the paper offers no theory for *why*; it's an empirical win.

> **🧑‍💼 Interviewer:** It has three weight matrices instead of two — isn't that just a bigger FFN?
>
> **🧑‍💻 Interviewee:** That's the trap. To keep it a fair fight, you shrink the hidden dimension by 2/3, so `h = (2/3)·4d = 8d/3`. Two matrices at `4d` and three matrices at `8d/3` have essentially the same parameter count and FLOPs. So no, it's not extra capacity — it's the *same* budget spent on a gated structure, and the gate is what helps.

> **🧑‍💼 Interviewer:** Why Swish specifically? Why not sigmoid like the original GLU?
>
> **🧑‍💻 Interviewee:** Shazeer sweeps several gating activations — sigmoid (vanilla GLU), ReLU (ReGLU), GELU (GEGLU), Swish (SwiGLU), and a no-activation bilinear variant. SwiGLU and GEGLU come out on top. Swish is `z·sigmoid(z)`: smooth, non-monotonic, and unlike ReLU it has no hard dead zone, which seems to help optimization. But the paper is explicit that there's no principled reason it's the winner.

> **🧑‍💼 Interviewer:** What's the single most likely bug when someone implements this?
>
> **🧑‍💻 Interviewee:** Forgetting the 2/3 hidden scaling, so you accidentally compare a bigger SwiGLU against a smaller baseline and "prove" it wins for the wrong reason. The second is mixing up which projection gets the activation — only the *gate* gets Swish; the *up* projection is linear. And in the LLaMA-style formulation you use bias-free linears.

> **🧑‍💼 Interviewer:** The paper says it can't explain the improvement. How do you feel shipping that?
>
> **🧑‍💻 Interviewee:** Totally fine, and honestly refreshing. It's a clean, reproducible, parameter-matched ablation across many tasks — that's strong empirical evidence. We ship things that work and keep looking for the theory. The famous "divine benevolence" line is the author being candid that the *mechanism* is open, not that the *result* is weak.

> **🧑‍💼 Interviewer:** Implement it, param-matched, and show it isn't just a bigger network.

---

## Part 3 — Implementation

The whole method is three linear projections, a Swish gate, and an elementwise multiply — plus the `2/3` hidden-size arithmetic.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


def swiglu_hidden(d_model, ffn_mult=4):
    """Param-matched hidden size for a 3-matrix gated FFN.

    A standard FFN (2 matrices) uses hidden = ffn_mult * d_model.
    A gated FFN has 3 matrices, so scale hidden by 2/3 to match params.
    Round up to a multiple of 8 (LLaMA rounds to a larger multiple).
    """
    h = int(2 / 3 * ffn_mult * d_model)
    return (h + 7) // 8 * 8


class SwiGLUFFN(nn.Module):
    """SwiGLU feed-forward: down( Swish(x W_gate) * (x W_up) )."""

    def __init__(self, d_model, hidden=None, ffn_mult=4):
        super().__init__()
        if hidden is None:
            hidden = swiglu_hidden(d_model, ffn_mult)
        self.gate = nn.Linear(d_model, hidden, bias=False)   # gated branch
        self.up   = nn.Linear(d_model, hidden, bias=False)   # value branch
        self.down = nn.Linear(hidden, d_model, bias=False)   # project back

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


class ReLUFFN(nn.Module):
    """Standard Transformer FFN baseline: down(ReLU(x W_up))."""

    def __init__(self, d_model, hidden=None, ffn_mult=4):
        super().__init__()
        if hidden is None:
            hidden = ffn_mult * d_model
        self.up   = nn.Linear(d_model, hidden, bias=False)
        self.down = nn.Linear(hidden, d_model, bias=False)

    def forward(self, x):
        return self.down(F.relu(self.up(x)))
```

### Why each line matters (talk through it)
- `swiglu_hidden` computes `8d/3` — this is the **parameter-matching** step. Without it you're comparing different-sized models.
- `self.gate`, `self.up`, `self.down` — **three** bias-free linears. The gate and up branches share the input but are independent projections.
- `F.silu(self.gate(x))` — Swish/SiLU **only** on the gate. Getting this on the wrong branch is the classic bug.
- `... * self.up(x)` — the **elementwise gate**: a learned, data-dependent multiplier on the value branch.
- `self.down(...)` — projects the gated hidden state back to `d_model`, exactly like a normal FFN's second matrix.
- `bias=False` everywhere — matches the paper/LLaMA and makes "zero gate ⇒ zero output" exact.

### Demonstrating the mechanism — and being honest about toy results
SwiGLU's benefit is an **at-scale, empirical** result that a 5-minute toy *cannot* reproduce. To stay honest we fit a **neutral** target — a fixed random MLP teacher that favors neither block — with two **param-matched** FFNs: SwiGLU at `hidden=8d/3` and ReLU at `hidden=4d`.

```python
d_model = 32
n_train, n_test = 2048, 4096

# NEUTRAL target: a fixed random MLP teacher (tanh). It has no special affinity
# for either FFN — the fair way to compare. (A SwiGLU-shaped target would rig it.)
torch.manual_seed(0)
teacher = nn.Sequential(
    nn.Linear(d_model, 64), nn.Tanh(),
    nn.Linear(64, 64), nn.Tanh(),
    nn.Linear(64, d_model),
)
for p in teacher.parameters():
    p.requires_grad_(False)

def target(x):
    with torch.no_grad():
        return teacher(x)

Xtr, Xte = torch.randn(n_train, d_model), torch.randn(n_test, d_model)
Ytr, Yte = target(Xtr), target(Xte)
mu, sd = Ytr.mean(), Ytr.std()                  # standardize so MSE is unit-scale
Ytr, Yte = (Ytr - mu) / sd, (Yte - mu) / sd

def n_params(m):
    return sum(p.numel() for p in m.parameters())

def train_eval(model_fn, seed=1):
    torch.manual_seed(seed)
    net = model_fn()
    opt = torch.optim.Adam(net.parameters(), lr=3e-3)
    for _ in range(2000):
        loss = F.mse_loss(net(Xtr), Ytr)
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        return F.mse_loss(net(Xte), Yte).item(), n_params(net)

relu_loss,   relu_p   = train_eval(lambda: ReLUFFN(d_model))
swiglu_loss, swiglu_p = train_eval(lambda: SwiGLUFFN(d_model))

print(f"ReLU-FFN   (hidden={4*d_model}): params={relu_p:,}  test_loss={relu_loss:.4f}")
print(f"SwiGLU-FFN (hidden={swiglu_hidden(d_model)}): params={swiglu_p:,}  test_loss={swiglu_loss:.4f}")
print(f"param ratio SwiGLU/ReLU = {swiglu_p / relu_p:.3f}  (≈1.0 = matched)")
```

Verified output (seed 1; essentially unchanged across seeds 1–3):

```
ReLU-FFN   (hidden=128): params=8,192  test_loss=0.0350
SwiGLU-FFN (hidden=88): params=8,448  test_loss=0.1619
param ratio SwiGLU/ReLU = 1.031  (≈1.0 = matched)
```

**Read this honestly — and *this is the lesson*.** On a neutral toy, SwiGLU does **not** win; here it's clearly *worse* than ReLU at a matched budget, because splitting the same parameters across **three** matrices shrinks the hidden width (88 vs 128). That's the truth about SwiGLU: its advantage is a **small, consistent perplexity improvement at LLM scale across many tasks** — exactly what a tiny toy can't surface (just like AdamW's real win doesn't show on a toy). So in the interview, don't try to "prove SwiGLU wins" — prove you understand the **mechanism**, can **param-match** it, and are candid that the benefit is empirical-at-scale. The sanity checks below verify the parts a toy genuinely *can* establish: shape, parameter parity, the gate, and nonlinearity.

> ⚠️ **Anti-pattern to avoid:** it's tempting to fit a *gated-structured* target like `(silu(x @ G) * (x @ A)) @ B` — then SwiGLU "wins" by ~1000×. That's **rigging the demo**: you built the target to match the model, so the result is circular and meaningless. A neutral target is the honest test, even when the answer is "no advantage at this scale."

---

## Part 4 — Sanity checks (don't skip)

### Check 1 — Output shape == input shape (it's a drop-in FFN)
```python
ffn = SwiGLUFFN(64)
x = torch.randn(8, 16, 64)
out = ffn(x)
assert out.shape == x.shape
print("OK: output shape", tuple(out.shape), "== input shape")
```

### Check 2 — Param count of SwiGLU(8d/3) ≈ ReLU-FFN(4d)
```python
d = 64
sw, re = SwiGLUFFN(d), ReLUFFN(d)
np_sw = sum(p.numel() for p in sw.parameters())
np_re = sum(p.numel() for p in re.parameters())
ratio = np_sw / np_re
print(f"SwiGLU params={np_sw:,}  ReLU params={np_re:,}  ratio={ratio:.3f}")
assert 0.9 <= ratio <= 1.1, "the 2/3 hidden scaling should match params"
```

### Check 3 — The gate actually gates (zero gate ⇒ zero output, since Swish(0)=0)
```python
ffn = SwiGLUFFN(64)
with torch.no_grad():
    ffn.gate.weight.zero_()          # bias-free, so gate(x) == 0 for all x
out = ffn(torch.randn(4, 64))
assert torch.allclose(out, torch.zeros_like(out), atol=1e-6)
print("OK: zeroed gate -> output max abs =", out.abs().max().item())
```

### Check 4 — Our Swish/SiLU matches the reference
```python
x = torch.randn(1000)
mine = x * torch.sigmoid(x)          # definition of Swish/SiLU
assert torch.allclose(mine, F.silu(x), atol=1e-6)
print("OK: x*sigmoid(x) == F.silu(x)")
```

### Check 5 — Gradient flows to gate, up, AND down projections
```python
ffn = SwiGLUFFN(64)
ffn(torch.randn(4, 64)).sum().backward()
for name in ("gate", "up", "down"):
    g = getattr(ffn, name).weight.grad
    assert g is not None and g.abs().sum() > 0, name
print("OK: nonzero grads in gate, up, down")
```

### Check 6 — The gate makes the FFN nonlinear: f(2x) ≠ 2·f(x)
```python
ffn = SwiGLUFFN(64)
x = torch.randn(4, 64)
assert not torch.allclose(ffn(2 * x), 2 * ffn(x), atol=1e-3)
print("OK: f(2x) != 2 f(x) -> the multiplicative gate is genuinely nonlinear")
```

---

## Part 5 — Likely follow-up questions

- *"Difference between SwiGLU, GEGLU, ReGLU, and Bilinear?"* — Same gated structure, different gate activation: Swish, GELU, ReLU, and identity (no activation) respectively. Shazeer finds SwiGLU/GEGLU best; Bilinear (no activation) is a useful "is the activation even needed?" baseline.
- *"Where exactly does the `2/3` come from?"* — Standard FFN: 2 matrices of size `d×4d`. Gated FFN: 3 matrices. To match `2·(d·4d)` parameters with three matrices `2·(d·h) + (h·d) = 3·d·h`, set `3·d·h = 2·d·4d ⇒ h = 8d/3 = (2/3)·4d`.
- *"Why does it help, really?"* — Honest answer: nobody has a clean theory; the paper explicitly declines to give one. Intuitions: multiplicative gating adds a cheap second-order interaction; the smooth non-monotonic Swish helps optimization. But it's empirical.
- *"Where is it used in production?"* — LLaMA / LLaMA-2 / LLaMA-3, PaLM, Mistral, and many other modern LLMs use SwiGLU FFNs (usually bias-free, hidden rounded to a hardware-friendly multiple).
- *"Does it interact with anything else?"* — It pairs naturally with RMSNorm and RoPE in the LLaMA-style block; nothing fancy, but it's part of the now-standard "modern Transformer" recipe.

---

## TL;DR cheat sheet
| Thing | Answer |
|---|---|
| Core idea | Replace FFN's `Linear→ReLU→Linear` with a gated block |
| Formula | `down( Swish(x·W_gate) ⊙ (x·W_up) )` |
| Swish | `Swish(z) = z·sigmoid(z)` (= SiLU); `Swish(0)=0` |
| Matrices | **3** (gate, up, down) vs 2 in a plain FFN |
| Hidden size | `8d/3 = (2/3)·4d` to **match params/compute** |
| Biases | None (bias-free, LLaMA-style) |
| Benefit | Lower perplexity / better downstream at **equal** budget |
| Why it works | **Empirical** — the paper offers no theory ("divine benevolence") |
| Used in | LLaMA, PaLM, Mistral, … |
| #1 bug | Forgetting the `2/3` scaling (unfair, bigger model) |
