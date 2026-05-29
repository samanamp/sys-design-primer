---
title: Focal Loss (RetinaNet) — Paper-to-Code Mock Interview
description: A timed mock (read paper, explain benefit, implement in Colab) using Focal Loss for dense object detection as the worked example.
sidebar:
  order: 10
  label: Focal Loss
---

> **Paper:** *Focal Loss for Dense Object Detection* — Lin et al., 2017. [arXiv:1708.02002](https://arxiv.org/abs/1708.02002)
>
> **Format:** Read (~12 min) → explain the *real* benefit → implement the core idea in Colab → sanity-check it.
>
> **Companion notebook:** [`focal_loss_mock.ipynb`](/notebooks/focal_loss_mock.ipynb) (download) — imbalanced-classification demo + a focal-loss stub to fill in, plus verification cells. Open in [Google Colab](https://colab.research.google.com/) via *File → Upload notebook*.
>
> **Difficulty:** 🟡 Intermediate. A loss-function paper — the math is short but the class-imbalance reasoning is where interviews go deep.

---

## How to run this as a timed drill (~40 min)

| Time | Block | What you produce |
|------|-------|------------------|
| 0:00–0:12 | **Read** (use the [three-pass method](/papers/lora-mock-interview/#part-0--how-to-read-a-paper-in-15-minutes-the-three-pass-method)) | Why class imbalance breaks CE + what the modulating factor does |
| 0:12–0:17 | **Explain the benefit** out loud (cover Part 2) | The "down-weight easy examples" intuition + role of γ and α |
| 0:17–0:33 | **Implement** from the stub (Part 3) | A working `focal_loss` + minority recall that beats plain CE |
| last 5 min | **Sanity-check** (Part 4) | All checks passing, narrated out loud |

### Self-grading rubric — "what good looks like"
- ✅ Explained focal loss as **down-weighting easy, well-classified examples** so the rare/hard class isn't drowned out — not just "a weighted cross-entropy."
- ✅ Knew the difference between **α (static class weighting)** and **γ (dynamic, per-example focusing)** and why you need both.
- ✅ Computed `p_t` (probability of the *true* class) correctly and worked **from logits** for numerical stability.
- ✅ Demonstrated the benefit with a **minority-class recall** improvement, not just "it runs."
- ⚠️ Red flags: confusing α with γ, computing the loss from probabilities instead of logits (NaNs), claiming γ alone fixes imbalance (it focuses on *hard* examples, which is correlated but not identical to *rare*).

---

## Part 1 — Structured read of THIS paper

### The 30-second summary (the "benefit")
In dense detection a model scores ~10⁴–10⁵ candidate boxes per image, and the overwhelming majority are **easy background**. Ordinary cross-entropy sums a tiny loss over each of those many easy negatives, and that flood **dominates the gradient** — the few foreground objects get drowned out and training stalls. **Focal loss** multiplies cross-entropy by a **modulating factor `(1 − p_t)^γ`** that shrinks the loss for well-classified examples, so training automatically focuses on the hard, rare ones. The payoff:

- **Trains a one-stage detector to two-stage accuracy** — no sampling heuristics (hard-negative mining, OHEM, fixed 1:3 ratios) needed.
- **One extra hyperparameter `γ`** (plus an optional class weight `α`) — a drop-in replacement for the classification loss.
- Enabled **RetinaNet**, which matched/beat the slower two-stage detectors of its time (the paper's headline result).

### The core idea (Method — you implement this)
Start from binary cross-entropy written in terms of the probability of the **true** class, `p_t`:

$$p_t = \begin{cases} p & \text{if } y = 1 \\ 1 - p & \text{otherwise} \end{cases}, \qquad \text{CE}(p_t) = -\log(p_t)$$

Focal loss adds a **modulating factor** `(1 − p_t)^γ` and an optional class-balancing weight `α_t`:

$$\text{FL}(p_t) = -\,\alpha_t\,(1 - p_t)^{\gamma}\,\log(p_t)$$

When an example is **easy** (`p_t → 1`) the factor `(1 − p_t)^γ → 0`, so its loss is crushed. When it's **hard** (`p_t` small) the factor `→ 1`, so the loss is essentially unchanged. The focusing parameter `γ ≥ 0` controls how aggressively easy examples are suppressed (`γ = 0` recovers ordinary cross-entropy; the paper uses `γ = 2`).

Key details (the things an interviewer probes):
- **`p_t` is the probability of the *true* class**, not the predicted class. Computing it wrong is the #1 implementation bug.
- **γ vs α are different knobs.** `γ` is *dynamic*: it reweights per example based on how well it's classified. `α` is *static*: a fixed class weight (the paper uses `α = 0.25` for the foreground in detection). They are complementary and the paper tunes them together.
- **Work from logits.** Compute the loss via a stable BCE-with-logits / log-sum-exp path, never `−log(sigmoid(x))` directly, or easy examples (huge `|logit|`) produce NaNs.
- **Special init matters.** RetinaNet initializes the final classification bias so the model starts predicting a low foreground probability (≈ π = 0.01); otherwise the huge initial loss from background destabilizes the first iterations.

### Where the evidence lives (tables that matter)
- **γ ablation (≈ Table 1b):** AP rises as `γ` goes from 0 → 2, then plateaus/declines → the focusing factor is what's doing the work. *(Figures approximate; check the paper for exact numbers.)*
- **FL vs OHEM / hard mining (≈ Table 1d):** focal loss beats the sampling-based imbalance fixes → you don't need heuristics.
- **RetinaNet vs two-stage detectors (≈ Table 2):** a one-stage detector reaches competitive/better COCO AP → the headline payoff.

### The honest limitations (have an opinion)
- **γ is dataset-dependent.** It controls *how much* to down-weight easy examples; the right value depends on how severe the imbalance is. `γ = 2` is a good default, not a law.
- **Focuses on *hard*, which only correlates with *rare*.** A hard but mislabeled/noisy example also gets up-weighted, so focal loss can amplify label noise.
- **It addresses the *classification* imbalance, not everything.** RetinaNet's gains also rely on the anchor design and the bias init — focal loss alone isn't a silver bullet.

---

## Part 2 — The interview dialogue (interviewer ⇄ interviewee)

> **🧑‍💼 Interviewer:** One paragraph — what does focal loss actually buy me over weighted cross-entropy?
>
> **🧑‍💻 Interviewee:** In dense detection you have a flood of easy background and only a handful of objects, so plain cross-entropy's gradient is dominated by the many easy negatives and the rare class is ignored. Static class weighting (α) rebalances positives vs negatives, but it can't tell an easy negative from a hard one — it up-weights all positives and down-weights all negatives equally. Focal loss adds a *dynamic* factor `(1 − p_t)^γ` that down-weights any example the model already gets right, regardless of class, so training automatically concentrates on the hard, informative examples. That's what let RetinaNet, a one-stage detector, hit two-stage accuracy without hard-negative mining.

> **🧑‍💼 Interviewer:** Walk me through `(1 − p_t)^γ`. Why that form?
>
> **🧑‍💻 Interviewee:** `p_t` is the model's probability for the *true* class, so it's near 1 when the example is classified well. `(1 − p_t)` is therefore the "wrongness," and raising it to `γ` makes the down-weighting steep: an easy example with `p_t = 0.99` and `γ = 2` gets its loss multiplied by `0.0001`, basically zeroing it, while a hard example with `p_t = 0.1` keeps `(0.9)² ≈ 0.81` of its loss. `γ = 0` is exactly cross-entropy. So `γ` is a smooth dial from "treat all examples equally" to "almost only learn from the ones I'm getting wrong."

> **🧑‍💼 Interviewer:** What's the difference between α and γ, and do you need both?
>
> **🧑‍💻 Interviewee:** α is a fixed per-class weight — it shifts the balance between positive and negative classes once, statically. γ is per-example and dynamic — it reweights based on how hard each example currently is. They're orthogonal: α handles the *class frequency* imbalance, γ handles the *easy-vs-hard* imbalance. The paper finds they're complementary and tunes them jointly, landing around `α = 0.25, γ = 2` for COCO.

> **🧑‍💼 Interviewer:** What's the most common bug implementing this?
>
> **🧑‍💻 Interviewee:** Two. First, computing `p_t` as the probability of the *predicted* class instead of the *true* class — then the modulating factor is meaningless. Second, computing the loss from probabilities (`−log(sigmoid(x))`) instead of from logits; for confident easy examples the logit is huge, `sigmoid` saturates to exactly 1, and you take `log(0) = −inf`. You compute it through a stable BCE-with-logits path so it never blows up.

> **🧑‍💼 Interviewer:** Implement it and show the minority-class recall beat plain cross-entropy.

---

## Part 3 — Implementation

The whole method is one modulating factor on top of a numerically-stable binary cross-entropy.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


def focal_loss_with_logits(logits, targets, gamma=2.0, alpha=0.75, reduction="mean"):
    """Binary focal loss computed from logits (numerically stable).

    logits, targets: same shape; targets in {0, 1} as floats.
    alpha weights the POSITIVE (rare/foreground) class; (1 - alpha) the negatives.
    """
    # per-element CE = -log(p_t), computed stably straight from logits
    ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p = torch.sigmoid(logits)
    p_t = p * targets + (1 - p) * (1 - targets)        # prob of the TRUE class
    modulating = (1.0 - p_t) ** gamma                  # down-weights easy (high p_t)
    loss = modulating * ce
    if alpha is not None:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss
    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    return loss                                         # reduction="none"
```

### Why each line matters (talk through it)
- `binary_cross_entropy_with_logits(..., reduction="none")` — the **stable** `−log(p_t)` per element, straight from logits. Never `−log(sigmoid(x))`; that NaNs on confident examples.
- `p_t = p * targets + (1 - p) * (1 - targets)` — picks `p` for positives and `1 − p` for negatives, i.e. the probability of the **true** class. This is the line everyone gets wrong.
- `(1.0 - p_t) ** gamma` — the **modulating factor**: ≈ 0 for easy examples (`p_t → 1`), ≈ 1 for hard ones. `γ = 0` recovers plain CE.
- `alpha_t = alpha * targets + (1 - alpha) * (1 - targets)` — the **static** class weight, applied on top of γ's dynamic weighting; `alpha=None` turns it off.
- `reduction` branch — match PyTorch loss conventions so it drops into a training loop.

### Demonstrating the benefit (imbalanced classification toy task)
A 2-D dataset with ~95% negatives and ~5% positives from **overlapping** Gaussians (so it's non-trivial). Train one model with plain cross-entropy and one with focal loss, then compare **minority-class recall**. Plain CE can hit high accuracy by mostly ignoring the rare class; focal loss should recover far more of it.

```python
def make_data(n, frac_pos, seed):
    g = torch.Generator().manual_seed(seed)
    n_pos = int(n * frac_pos); n_neg = n - n_pos
    neg = torch.randn(n_neg, 2, generator=g)
    pos = torch.randn(n_pos, 2, generator=g) + torch.tensor([1.4, 1.4])  # overlapping
    X = torch.cat([neg, pos]); y = torch.cat([torch.zeros(n_neg), torch.ones(n_pos)])
    perm = torch.randperm(n, generator=g)
    return X[perm], y[perm]

def train(loss_kind, Xtr, ytr, seed=0, gamma=2.0, alpha=0.75, epochs=400):
    torch.manual_seed(seed)
    net = nn.Sequential(nn.Linear(2, 32), nn.ReLU(), nn.Linear(32, 1))
    opt = torch.optim.Adam(net.parameters(), lr=1e-2)
    for _ in range(epochs):
        logits = net(Xtr).squeeze(1)
        if loss_kind == "ce":
            loss = F.binary_cross_entropy_with_logits(logits, ytr)
        else:
            loss = focal_loss_with_logits(logits, ytr, gamma=gamma, alpha=alpha)
        opt.zero_grad(); loss.backward(); opt.step()
    return net

def recalls(net, X, y):
    with torch.no_grad():
        pred = (torch.sigmoid(net(X).squeeze(1)) > 0.5).float()
    rec_pos = (pred[y == 1] == 1).float().mean().item()
    rec_neg = (pred[y == 0] == 0).float().mean().item()
    return rec_pos, rec_neg

Xtr, ytr = make_data(4000, frac_pos=0.05, seed=1)
Xte, yte = make_data(4000, frac_pos=0.05, seed=2)
net_ce = train("ce",    Xtr, ytr)
net_fl = train("focal", Xtr, ytr, gamma=2.0, alpha=0.75)
ce_pos, ce_neg = recalls(net_ce, Xte, yte)
fl_pos, fl_neg = recalls(net_fl, Xte, yte)
print(f"CE    : minority recall {ce_pos:.3f}   majority recall {ce_neg:.3f}")
print(f"Focal : minority recall {fl_pos:.3f}   majority recall {fl_neg:.3f}")
```

You should see plain CE recover only a small fraction of the minority class (high overall accuracy from the easy majority), while focal loss roughly **doubles minority recall** at a tiny cost to majority recall. With the seeds above the run prints CE minority recall ≈ 0.29 vs focal ≈ 0.51. (Exact numbers are seed-dependent; the *direction* — higher minority recall with focal loss — is the point.)

---

## Part 4 — Sanity checks (don't skip)

### Check 1 — γ=0 reduces to (un-weighted) cross-entropy
```python
torch.manual_seed(0)
logits = torch.randn(64)
targets = (torch.rand(64) > 0.5).float()
fl0 = focal_loss_with_logits(logits, targets, gamma=0.0, alpha=None)
bce = F.binary_cross_entropy_with_logits(logits, targets)
print("focal(γ=0):", fl0.item(), " BCE:", bce.item())
assert torch.allclose(fl0, bce, atol=1e-6)
```

### Check 2 — Easy example is crushed; hard example is untouched
```python
easy_l, easy_t = torch.tensor([6.0]),  torch.tensor([1.0])   # p_t ≈ 0.9975
hard_l, hard_t = torch.tensor([-6.0]), torch.tensor([1.0])   # p_t ≈ 0.0025
ce_easy = F.binary_cross_entropy_with_logits(easy_l, easy_t).item()
fl_easy = focal_loss_with_logits(easy_l, easy_t, gamma=2.0, alpha=None).item()
ce_hard = F.binary_cross_entropy_with_logits(hard_l, hard_t).item()
fl_hard = focal_loss_with_logits(hard_l, hard_t, gamma=2.0, alpha=None).item()
print(f"easy: CE {ce_easy:.4e}  focal {fl_easy:.4e}")
print(f"hard: CE {ce_hard:.4f}  focal {fl_hard:.4f}")
assert fl_easy < ce_easy * 1e-3      # easy example's loss is essentially zeroed
assert fl_hard > ce_hard * 0.99      # hard example's loss is barely changed
```

### Check 3 — Higher γ down-weights an easy example MORE (monotonic in γ)
```python
easy_l, easy_t = torch.tensor([6.0]), torch.tensor([1.0])
vals = [focal_loss_with_logits(easy_l, easy_t, gamma=g, alpha=None).item()
        for g in (0.0, 1.0, 2.0, 5.0)]
print("easy-example loss vs γ:", [f"{v:.2e}" for v in vals])
assert all(vals[i+1] < vals[i] for i in range(len(vals) - 1))  # strictly decreasing
```

### Check 4 — Reduction / shape is correct
```python
none_out = focal_loss_with_logits(logits, targets, reduction="none")
mean_out = focal_loss_with_logits(logits, targets, reduction="mean")
sum_out  = focal_loss_with_logits(logits, targets, reduction="sum")
print("none shape:", tuple(none_out.shape))
assert none_out.shape == logits.shape
assert torch.allclose(none_out.mean(), mean_out, atol=1e-6)
assert torch.allclose(none_out.sum(),  sum_out,  atol=1e-5)
```

### Check 5 — Focal beats CE on minority recall (the demonstration)
```python
# uses net_ce / net_fl trained in Part 3
print(f"minority recall — CE {ce_pos:.3f}  vs  focal {fl_pos:.3f}")
assert fl_pos > ce_pos
```

### Check 6 — Gradient flows
```python
lg = torch.randn(16, requires_grad=True)
tg = (torch.rand(16) > 0.5).float()
focal_loss_with_logits(lg, tg, gamma=2.0, alpha=0.75).backward()
print("|grad| sum:", lg.grad.abs().sum().item())
assert lg.grad is not None and torch.isfinite(lg.grad).all() and lg.grad.abs().sum() > 0
```

---

## Part 5 — Likely follow-up questions

- *"Why not just use class weights (α) alone?"* — α is static: it can't distinguish an easy negative from a hard one, so the flood of easy negatives still dominates within the negative class. γ adds the *dynamic*, per-example down-weighting that actually solves the easy/hard imbalance.
- *"What's the bias-initialization trick and why?"* — RetinaNet inits the final classification bias so the model starts predicting foreground probability ≈ 0.01. Without it, the first forward pass produces a giant loss from the ~100k background anchors and training diverges. It's a stability fix, separate from the loss form.
- *"Does focal loss replace hard-negative mining / OHEM?"* — Yes; that's the point. Sampling heuristics throw away examples; focal loss keeps all of them but reweights, and the paper shows it beats OHEM.
- *"When can focal loss hurt?"* — When 'hard' examples are actually mislabeled or noisy: it up-weights whatever the model gets wrong, so it can amplify label noise. Also if there's no real imbalance, γ just slows learning.
- *"How would you do the multiclass version?"* — Use softmax `p_t` for the true class and the same `(1 − p_t)^γ` factor, computed from `log_softmax` for stability; α becomes a per-class weight vector.

---

## TL;DR cheat sheet
| Thing | Answer |
|---|---|
| Core idea | Down-weight easy, well-classified examples so the rare/hard class isn't drowned out |
| Formula | `FL(p_t) = −α_t (1 − p_t)^γ log(p_t)`, `p_t` = prob of the true class |
| γ (focusing) | Dynamic, per-example; `γ=0` → plain CE; paper uses `γ=2` |
| α (balancing) | Static per-class weight; up-weights the rare class (`α=0.25` in the paper's detection setup) |
| Numerical stability | Compute from logits via BCE-with-logits, never `−log(sigmoid(x))` |
| Benefit | One-stage detector (RetinaNet) reaches two-stage accuracy, no hard-negative mining |
| #1 bug | `p_t` of the predicted class instead of the true class; or computing from probabilities |
| Limitation | γ is dataset-dependent; up-weights *hard* (can amplify label noise), not just *rare* |
