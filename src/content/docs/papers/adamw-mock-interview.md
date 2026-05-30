---
title: AdamW — Paper-to-Code Mock Interview
description: A paper-to-code mock (read paper, explain the real benefit, implement in Colab) using Decoupled Weight Decay Regularization (AdamW) as the worked example.
sidebar:
  order: 5
  label: AdamW
---

> **Paper:** *Decoupled Weight Decay Regularization (AdamW)* — Loshchilov & Hutter, 2017. [arXiv:1711.05101](https://arxiv.org/abs/1711.05101)
>
> **Format:** Read (~15 min) → explain the *real* benefit → implement the core idea in Colab → sanity-check it.
>
> **Companion notebook:** [`adamw_mock.ipynb`](/notebooks/adamw_mock.ipynb) (download) — a mechanism demo + an `Adam`/`AdamW` step stub to fill in, plus verification cells. Open in [Google Colab](https://colab.research.google.com/) via *File → Upload notebook*.
>
> **Difficulty:** 🟡 Medium. The benefit is *subtle* — it's a correction to a bug almost everyone had, not a flashy new mechanism. Get comfortable explaining why L2 ≠ weight decay for adaptive optimizers.

---

## How to run this as a timed drill (~40 min)

| Time | Block | What you produce |
|------|-------|------------------|
| 0:00–0:12 | **Read** (use the [three-pass method](/papers/lora-mock-interview/#part-0--how-to-read-a-paper-in-15-minutes-the-three-pass-method)) | Why L2 ≠ weight decay under Adam + the decoupling fix |
| 0:12–0:17 | **Explain the benefit** out loud (cover Part 2) | The coupling-to-`1/√v` argument + what AdamW changes |
| 0:17–0:33 | **Implement** from the stub (Part 3) | An `Adam` step in two modes; show per-param effective decay differs under L2 but not AdamW |
| last 5 min | **Sanity-check** (Part 4) | All checks passing, narrated out loud |

### Self-grading rubric — "what good looks like"
- ✅ Explained that under Adam, **L2 regularization is NOT weight decay** — the penalty gets divided by `√v̂`, so high-second-moment params decay *less*.
- ✅ Stated the fix precisely: **decouple** decay from the adaptive step — `param -= lr · wd · param`, applied directly, not through `m`/`v`.
- ✅ Knew this makes weight decay **uniform across params** and more **independent of `lr`** (cleaner hyperparameter search).
- ✅ Was honest that on a small toy task the **generalization win is often negligible** — the demo should show the *mechanism*, not chase an accuracy number.
- ⚠️ Red flags: claiming "Adam with `weight_decay=` already does weight decay" (that's L2 in legacy PyTorch `Adam`), or saying AdamW just "moves where you add `wd·w`" without explaining the `1/√v̂` coupling.

---

## Part 1 — Structured read of THIS paper

### The 30-second summary (the "benefit")
For plain SGD, two things are mathematically identical: (a) adding an L2 penalty `½λ‖w‖²` to the loss, and (b) "weight decay" — shrinking every weight by a constant factor each step. People used these interchangeably and wired "weight decay" into Adam as an L2 term added to the gradient.

The paper's observation: **for adaptive optimizers like Adam, these are NOT the same.** When the L2 term `λw` is added to the gradient, it flows through Adam's per-parameter adaptive learning rate and gets divided by `√v̂` (the second-moment estimate). So weights with large gradient history get decayed *less*, and weights with small gradient history get decayed *more* — the regularization strength is silently coupled to each parameter's update statistics. **AdamW decouples** weight decay from the gradient/adaptive step so every weight decays by the same multiplicative factor `(1 − lr·λ)`. The payoff:

- **Weight decay behaves as intended** — uniform shrinkage, not a per-parameter accident.
- **Better decoupling of hyperparameters** — the best `lr` and the best `wd` become much more independent, so grid search is easier and more transferable.
- **Improved generalization** for Adam-trained models in the paper's experiments — closing much of the historical generalization gap between Adam and SGD-with-momentum.

### The core idea (Method — you implement this)
Adam maintains a first moment `m` (mean of grads) and second moment `v` (mean of squared grads), with bias correction:

$$m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t, \qquad v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2$$

$$\hat m_t = \frac{m_t}{1-\beta_1^t}, \qquad \hat v_t = \frac{v_t}{1-\beta_2^t}, \qquad \theta_t = \theta_{t-1} - \frac{\eta\,\hat m_t}{\sqrt{\hat v_t}+\epsilon}$$

The **coupled L2** way (what legacy `Adam(weight_decay=λ)` does) folds the penalty into the gradient *before* everything else:

$$g_t \leftarrow g_t + \lambda\,\theta_{t-1} \quad\Rightarrow\quad \text{decay term ends up as } \frac{\eta\,\lambda\,\theta}{\sqrt{\hat v}+\epsilon}\ \ (\text{scaled by } 1/\sqrt{\hat v})$$

**AdamW** instead applies decay **directly to the parameter**, outside the adaptive machinery:

$$\theta_t = \underbrace{(1 - \eta\,\lambda)\,\theta_{t-1}}_{\text{decoupled decay, uniform}} - \frac{\eta\,\hat m_t}{\sqrt{\hat v_t}+\epsilon}$$

Now the decay factor `(1 − ηλ)` is the **same for every parameter**, independent of `v̂`.

Key details (the things an interviewer probes):
- **`m`/`v` capture the gradient's mean and variance.** Dividing by `√v̂` gives each parameter its own effective step size — that's what makes Adam "adaptive", and that's exactly what corrupts coupled L2.
- **Bias correction matters early.** `m` and `v` start at zero, so without `1/(1−βᵗ)` the first steps would be far too small. On step 1, `m̂/√v̂ ≈ sign(g)`, so the move is ≈ `η`.
- **Why decoupled decay is independent of the gradient.** `(1 − ηλ)θ` doesn't touch `m` or `v` at all. Even with a zero gradient, the weight still shrinks by exactly `(1 − ηλ)`.
- **`λ` couples to `lr` in PyTorch's AdamW.** Because the update uses `lr·wd·θ`, doubling `lr` doubles the per-step decay. (The original paper proposed normalizing `wd` by a schedule multiplier to fully decouple from `lr`; PyTorch's default does not.)

### Where the evidence lives (tables/figures that matter)
- **The `lr × wd` heatmaps:** for coupled Adam+L2 the good region is a diagonal "valley" (best `wd` depends on `lr`); for AdamW it's a more axis-aligned basin → the **decoupling/hyperparameter-independence** claim, visualized.
- **CIFAR-10 / ImageNet test-error tables:** AdamW matches or beats Adam+L2 and narrows the gap to SGD+momentum → the **generalization** claim.
- **Training/validation curves with cosine restarts:** AdamW + a good schedule competes with SGD → supports the practical recommendation.

### The honest limitations (have an opinion)
- **The win can be small or task-dependent.** On a tiny toy problem the generalization difference is often within noise — the *mechanism* (uniform vs coupled decay) is rock-solid, but don't oversell a headline accuracy number.
- **PyTorch's AdamW still couples `wd` to `lr`** (decay is `lr·wd·θ`). True scale-free decoupling needs the paper's schedule normalization, which most implementations skip.
- **It's a fix, not a new capability.** AdamW doesn't let you do anything Adam couldn't; it makes a knob you were already turning behave the way you assumed it did.

---

## Part 2 — The interview dialogue (interviewer ⇄ interviewee)

> **🧑‍💼 Interviewer:** One paragraph — what does AdamW actually fix?
>
> **🧑‍💻 Interviewee:** For SGD, L2 regularization and weight decay are identical, so everyone wired "weight decay" into Adam by adding `λw` to the gradient. But in Adam the gradient gets divided by `√v̂`, the per-parameter second-moment estimate — so that `λw` term gets divided too. The net effect is that parameters with large gradient history decay *less* and small-history ones decay *more*: your regularization strength is silently coupled to each parameter's update statistics. AdamW decouples it — it shrinks every weight by the same factor `(1 − lr·λ)` directly, outside the adaptive step. So weight decay finally does what you thought it did, and `lr` and `wd` become much more independent to tune.

> **🧑‍💼 Interviewer:** Walk me through *exactly where* the coupling happens.
>
> **🧑‍💻 Interviewee:** Coupled L2 sets `g ← g + λw` at the top of the step. That modified `g` feeds `m` and `v`, then the update is `lr · m̂ / (√v̂ + ε)`. The decay contribution that reaches the weight is roughly `lr · λw / √v̂` — explicitly divided by `√v̂`. AdamW never puts `λw` into `g`; it does `w ← (1 − lr·λ)·w` and *separately* subtracts the adaptive step `lr · m̂ / √v̂`. The `(1 − lr·λ)` factor has no `v̂` in it, so it's identical for every parameter.

> **🧑‍💼 Interviewer:** How would you *prove* the difference in code without training a whole model?
>
> **🧑‍💻 Interviewee:** Isolate the decay term. Take two params at the same value but with very different second moments `v`, set the gradient to zero, and run one step. Under AdamW both shrink by exactly `(1 − lr·λ)`. Under Adam+L2 the two shrink by *different* factors, because the residual decay still gets divided by `√v̂`. Printing the two effective decay factors side by side is the whole proof — no accuracy metric needed.

> **🧑‍💼 Interviewer:** Will AdamW always generalize better than Adam+L2?
>
> **🧑‍💻 Interviewee:** No — on a small toy task the difference is often inside the noise. The reliable claim is mechanistic: decay is uniform and `lr`/`wd` decouple, which makes tuning cleaner and transfers across settings. The paper shows real generalization gains on CIFAR/ImageNet, but I wouldn't promise a win on every problem. I'd demo the mechanism, not a cherry-picked accuracy number.

> **🧑‍💼 Interviewer:** Implement an Adam step in both modes and show the per-param decay diverge under L2.

---

## Part 3 — Implementation

The whole method is one Adam step with the weight-decay term moved *out* of the gradient path.

```python
import torch


class ManualAdam:
    """Minimal manual Adam over a list of parameter tensors, two WD modes.

    mode="L2"    -> classic Adam with L2 folded into the gradient (coupled).
    mode="AdamW" -> decoupled weight decay (the AdamW correction).
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0.0, mode="AdamW"):
        self.params = list(params)
        self.lr, self.eps, self.wd, self.mode = lr, eps, weight_decay, mode
        self.b1, self.b2 = betas
        self.t = 0
        self.m = [torch.zeros_like(p) for p in self.params]  # 1st moment
        self.v = [torch.zeros_like(p) for p in self.params]  # 2nd moment

    @torch.no_grad()
    def step(self):
        self.t += 1
        for i, p in enumerate(self.params):
            g = p.grad
            if self.mode == "L2" and self.wd != 0.0:
                g = g + self.wd * p                              # COUPLED: into the grad

            self.m[i].mul_(self.b1).add_(g, alpha=1 - self.b1)
            self.v[i].mul_(self.b2).addcmul_(g, g, value=1 - self.b2)

            m_hat = self.m[i] / (1 - self.b1 ** self.t)          # bias correction
            v_hat = self.v[i] / (1 - self.b2 ** self.t)
            step = self.lr * m_hat / (v_hat.sqrt() + self.eps)   # adaptive step

            if self.mode == "AdamW" and self.wd != 0.0:
                p.mul_(1 - self.lr * self.wd)                    # DECOUPLED: on the param

            p.sub_(step)

    def zero_grad(self):
        for p in self.params:
            if p.grad is not None:
                p.grad.zero_()
```

### Why each line matters (talk through it)
- `g = g + self.wd * p` (L2 branch) — the **coupled** way. This `wd·p` now contributes to `m` and `v` and gets divided by `√v̂` downstream — that's the bug.
- `m.mul_(b1).add_(g, …)` / `v.…addcmul_(g, g, …)` — the EMA of grads and squared grads. `v` is what makes Adam adaptive (and what corrupts coupled L2).
- `m_hat = m / (1 - b1**t)` — bias correction. Moments start at 0; without this the early steps are far too small.
- `p.mul_(1 - self.lr * self.wd)` (AdamW branch) — the **decoupled** decay. It multiplies the parameter directly, never touching `m`/`v`, so the factor `(1 − lr·wd)` is identical for every parameter.
- `p.sub_(step)` — the adaptive Adam update, applied after the decay so decay and the gradient step don't interfere.

### Demonstrating the MECHANISM (not a generalization win)
A toy generalization win for AdamW is usually negligible and would be a *misleading* demo. Instead we prove the mechanism directly: take two params at the same value but with very different second moments `v`, zero out the gradient (so only decay moves them), and run one step.

```python
lr, wd = 0.1, 0.1
for mode in ("L2", "AdamW"):
    print(f"\nmode = {mode}")
    for v_state in (1.0, 1e-4):            # param A: big grad history; param B: tiny
        p = torch.tensor([1.0], requires_grad=True)
        opt = ManualAdam([p], lr=lr, weight_decay=wd, mode=mode)
        opt.t = 1000                        # pretend we're deep into training
        opt.v[0].fill_(v_state)             # seed the second moment
        opt.m[0].zero_()                    # no first moment -> isolate decay
        p.grad = torch.zeros_like(p)        # ZERO gradient: only decay can move p
        before = p.item(); opt.step()
        print(f"  v={v_state:>8.0e}  ->  effective decay factor = {p.item()/before:.6f}")
print(f"\nAdamW target (1 - lr*wd) = {1 - lr*wd:.6f} for BOTH params.")
```

Observed output (mechanism is deterministic; the exact L2 numbers depend on the seeded `v`):

```
mode = L2
  v=   1e+00  ->  effective decay factor = 0.999204
  v=   1e-04  ->  effective decay factor = 0.924127

mode = AdamW
  v=   1e+00  ->  effective decay factor = 0.990000
  v=   1e-04  ->  effective decay factor = 0.990000

AdamW target (1 - lr*wd) = 0.990000 for BOTH params.
```

Under **Adam+L2** the two params decay by *different* factors (the residual `λw` term is divided by `√v̂` — the high-`v` param at left decays barely at all). Under **AdamW** both decay by exactly `(1 − lr·wd) = 0.99`. That divergence — not any accuracy number — *is* the paper's point.

---

## Part 4 — Sanity checks (don't skip)

### Check 1 — Manual Adam (wd=0) matches `torch.optim.Adam` [KEY reference check]
```python
def run(make_ref, make_mine, steps=20, shape=(8,), seed=0):
    torch.manual_seed(seed)
    p0 = torch.randn(shape); grads = [torch.randn(shape) for _ in range(steps)]
    pr = p0.clone().requires_grad_(True); o_r = make_ref([pr])
    pm = p0.clone().requires_grad_(True); o_m = make_mine([pm])
    for g in grads:
        pr.grad = g.clone(); o_r.step(); o_r.zero_grad()
        pm.grad = g.clone(); o_m.step(); o_m.zero_grad()
    return pr.detach(), pm.detach()

r, m = run(lambda ps: torch.optim.Adam(ps, lr=1e-2),
           lambda ps: ManualAdam(ps, lr=1e-2, weight_decay=0.0))
assert torch.allclose(r, m, atol=1e-6)
print("OK: manual Adam == torch.optim.Adam", (r - m).abs().max().item())
```

### Check 2 — Manual AdamW matches `torch.optim.AdamW` [KEY reference check]
```python
r, m = run(lambda ps: torch.optim.AdamW(ps, lr=1e-2, weight_decay=0.1),
           lambda ps: ManualAdam(ps, lr=1e-2, weight_decay=0.1, mode="AdamW"))
assert torch.allclose(r, m, atol=1e-6)
print("OK: manual AdamW == torch.optim.AdamW", (r - m).abs().max().item())
```

### Check 3 — Bias correction makes the first step ≈ `lr`
```python
p = torch.tensor([5.0], requires_grad=True)
opt = ManualAdam([p], lr=1e-2, weight_decay=0.0)
p.grad = torch.tensor([0.37]); before = p.item(); opt.step()
move = abs(before - p.item())
assert abs(move - 1e-2) < 1e-3     # m_hat/sqrt(v_hat) ~= 1 on step 1
print(f"OK: first-step move {move:.6f} ~= lr (bias correction)")
```

### Check 4 — AdamW decay is independent of the gradient AND of `v`
```python
lr, wd = 0.1, 0.05
p = torch.tensor([3.0], requires_grad=True)
opt = ManualAdam([p], lr=lr, weight_decay=wd, mode="AdamW")
opt.v[0].fill_(123.0)              # nonzero v would matter IF coupled
p.grad = torch.zeros_like(p); opt.step()
assert abs(p.item() - 3.0 * (1 - lr * wd)) < 1e-6
print(f"OK: AdamW zero-grad decay = {p.item():.6f} == 3*(1-lr*wd), any v")
```

### Check 5 — Adam+L2 zero-grad does NOT decay by `(1−lr·wd)` (the coupling)
```python
p = torch.tensor([3.0], requires_grad=True)
opt = ManualAdam([p], lr=lr, weight_decay=wd, mode="L2")
opt.t = 1000; opt.v[0].fill_(1.0); opt.m[0].zero_()
p.grad = torch.zeros_like(p); opt.step()
l2_factor, adamw_factor = p.item() / 3.0, 1 - lr * wd
assert abs(l2_factor - adamw_factor) > 1e-3
print(f"OK: L2 factor {l2_factor:.6f} != AdamW {adamw_factor:.6f} (coupled to 1/sqrt(v))")
```

### Check 6 — With `wd=0`, the two modes are identical
```python
torch.manual_seed(7)
p0 = torch.randn(8); gs = [torch.randn(8) for _ in range(10)]
pa = p0.clone().requires_grad_(True); pb = p0.clone().requires_grad_(True)
oa = ManualAdam([pa], lr=1e-2, weight_decay=0.0, mode="L2")
ob = ManualAdam([pb], lr=1e-2, weight_decay=0.0, mode="AdamW")
for g in gs:
    pa.grad = g.clone(); oa.step(); oa.zero_grad()
    pb.grad = g.clone(); ob.step(); ob.zero_grad()
assert torch.allclose(pa.detach(), pb.detach(), atol=1e-7)
print("OK: with wd=0, L2 and AdamW coincide (decay is the only difference)")
```

---

## Part 5 — Likely follow-up questions

- *"Why are L2 and weight decay identical for SGD but not Adam?"* — For SGD the update is `θ ← θ − η(g + λθ) = (1 − ηλ)θ − ηg`, so the L2 term *is* a constant multiplicative decay. Adam divides the whole gradient (including `λθ`) by `√v̂`, which is per-parameter and time-varying, so the decay is no longer constant.
- *"Which params decay less under coupled L2?"* — Those with large second-moment `v̂` (large/volatile gradients). Their `1/√v̂` is small, so the `λθ` penalty reaching them is small — they're *under*-regularized exactly where you might want more.
- *"Does PyTorch's `AdamW` fully decouple from `lr`?"* — No. Its decay is `lr·wd·θ`, so `wd` still scales with `lr`. The original paper proposed normalizing `wd` by the schedule multiplier for true scale-free decoupling.
- *"Is `Adam(weight_decay=…)` the same as `AdamW`?"* — No. Legacy `torch.optim.Adam(weight_decay=λ)` does **coupled L2**; `AdamW` does decoupled decay. This caused real-world reproducibility confusion.
- *"Should you apply weight decay to biases / norm params?"* — Usually no. Biases and LayerNorm/BatchNorm scale-shift params are typically excluded from decay (put them in a separate param group with `weight_decay=0`).
- *"How does this interact with learning-rate warmup/cosine schedules?"* — AdamW is commonly paired with warmup + cosine (the paper used warm restarts, SGDR). Because decay scales with `lr`, the schedule also modulates effective decay — another reason to be deliberate about it.

---

## TL;DR cheat sheet
| Thing | Answer |
|---|---|
| Core idea | Decouple weight decay from the adaptive step: `θ ← (1 − lr·λ)θ − lr·m̂/√v̂` |
| The bug it fixes | In Adam, L2 (`g += λθ`) gets divided by `√v̂` → non-uniform, coupled decay |
| Coupled L2 decay term | `lr·λθ / √v̂` (per-parameter, depends on grad history) |
| AdamW decay factor | `(1 − lr·λ)`, **identical for every parameter** |
| Main benefit | Uniform decay + `lr`/`wd` decoupling → easier tuning, better generalization |
| `Adam(weight_decay=)` vs `AdamW` | Coupled L2 vs decoupled decay — **not** the same |
| Caveat | Toy generalization win is often negligible; demo the *mechanism* |
| PyTorch caveat | `AdamW` still couples `wd` to `lr` (`lr·wd·θ`) |
