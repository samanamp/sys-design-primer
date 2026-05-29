---
title: Batch Normalization — Paper-to-Code Mock Interview
description: A full combined mock (read paper, explain the real benefit, implement BatchNorm1d in Colab, sanity-check it) using Batch Normalization as the worked example.
sidebar:
  order: 6
  label: BatchNorm
---

> **Paper:** *Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift* — Ioffe & Szegedy, 2015. arXiv: [1502.03167](https://arxiv.org/abs/1502.03167)
>
> **Format:** Read the paper (~15 min) → explain the *real* benefit → implement the core idea in Colab → sanity-check it.
>
> **Companion notebook:** [`batchnorm_mock.ipynb`](/notebooks/batchnorm_mock.ipynb) (download) — a WITH-vs-WITHOUT-BN training demo + a `BatchNorm1d` stub to fill in, plus verification cells. Or open it straight in [Google Colab](https://colab.research.google.com/) via *File → Upload notebook*. A reference solution is included at the bottom of this page.
>
> **Difficulty:** 🟡 Medium. The forward pass is short, but the train/eval split, running statistics, and the biased-vs-unbiased variance detail are exactly where interviewers probe.

---

## How to run this as a timed drill (~60 min)

Treat this like the real thing. Set a timer and don't look at the answers below until each block is done.

| Time | Block | What you produce |
|------|-------|------------------|
| 0:00–0:15 | **Read** (use the [three-pass method](/papers/lora-mock-interview/#part-0--how-to-read-a-paper-in-15-minutes-the-three-pass-method)) | The normalization equation + the one figure that proves faster training |
| 0:15–0:20 | **Explain the benefit** out loud (cover Part 2 without peeking) | 1-paragraph pitch + answers to "what's γ/β for", "train vs eval", "why running stats" |
| 0:20–0:50 | **Implement** in Colab from the stub (Part 3) | A working `BatchNorm1d` + a deep net that trains with BN but stalls without it |
| last 10 min | **Sanity-check** (Part 4) | All 6 checks passing, including the numerical match to `torch.nn.BatchNorm1d`, talked through out loud |

### Self-grading rubric — "what good looks like"
- ✅ Got the **train/eval split** cold: train uses *batch* stats and updates running buffers; eval uses *running* stats only.
- ✅ Knew **why γ and β exist** — to let the layer undo the normalization if that's what's optimal (preserves representational power).
- ✅ Used **`unbiased=False`** for the normalization variance (matches PyTorch) and could explain the unbiased correction in the *running* variance.
- ✅ Demonstrated the benefit with a **training-curve effect** (BN converges, plain deep net stalls at the same LR), not just "it runs."
- ✅ Had an **opinion on the mechanism**: the original "internal covariate shift" story vs. the later "it smooths the loss landscape" view.
- ⚠️ Red flags: using batch stats at eval, forgetting the running buffers, dropping γ/β, claiming BN's benefit is settled science.

---

## Part 1 — Structured read of THIS paper

Here's what each pass should surface in the Batch Normalization paper specifically: the summary and core idea come from Pass 2, the training-speed figure from the Pass 1 figure-skim, and the limitations/debate from Pass 3.

### The 30-second summary (the "benefit")
Deep nets are hard to train because the distribution of each layer's inputs keeps shifting as the layers below it update — the authors call this **internal covariate shift**. That forces tiny learning rates and careful initialization. Batch Normalization **normalizes each feature across the mini-batch** to zero mean and unit variance, then applies a learnable **scale γ and shift β** so the layer can recover any distribution it actually needs. The payoff:

- **Tolerates much higher learning rates** and is far less sensitive to initialization → **faster convergence** (the paper reports reaching the same accuracy in roughly an order of magnitude fewer steps — treat the exact figures as approximate).
- **Acts as a mild regularizer** (the per-batch noise), sometimes letting you reduce or drop dropout.
- **Smoother optimization** overall — the net just trains more easily, especially when it's deep.

### The core idea (Method — read this carefully, you implement it)
For a mini-batch $\mathcal{B} = \{x_1, \dots, x_m\}$ and each feature independently, compute the batch mean and (biased) variance, normalize, then scale and shift:

$$\mu_\mathcal{B} = \frac{1}{m}\sum_{i=1}^{m} x_i, \qquad \sigma_\mathcal{B}^2 = \frac{1}{m}\sum_{i=1}^{m} (x_i - \mu_\mathcal{B})^2$$

$$\hat{x}_i = \frac{x_i - \mu_\mathcal{B}}{\sqrt{\sigma_\mathcal{B}^2 + \epsilon}}, \qquad y_i = \gamma\,\hat{x}_i + \beta$$

The normalization variance uses $1/m$ (biased, i.e. `unbiased=False`). At **inference** you don't have a batch, so BN uses **population statistics** estimated during training as a running average of the batch means/variances:

$$\hat{x} = \frac{x - \mathrm{E}[x]}{\sqrt{\mathrm{Var}[x] + \epsilon}}, \qquad y = \gamma\,\hat{x} + \beta$$

Key details (the things an interviewer probes):
- **γ and β are learnable, per-feature.** Without them, forcing every layer's output to mean-0/var-1 would *limit* what the network can represent. With them, the layer can recover the identity ($\gamma=\sqrt{\sigma^2+\epsilon}$, $\beta=\mu$) if that's optimal — so BN never *removes* capacity.
- **Train vs eval is mandatory.** In training, normalize with the *current batch's* stats and *update* the running mean/var. In eval, use the stored running stats and update nothing. PyTorch toggles this via `model.train()` / `model.eval()`.
- **Biased vs unbiased variance.** The *normalization* uses the biased variance ($1/m$). PyTorch's running variance, however, is updated with the *unbiased* batch variance ($1/(m-1)$) — a subtle detail you need to match `torch.nn.BatchNorm1d` numerically.
- **`momentum`** controls how fast the running stats track the batches: `running = (1-momentum)·running + momentum·batch_stat` (PyTorch's convention; default 0.1).
- **`eps`** is added under the square root for numerical stability.

### Where the evidence lives (figures/tables that matter)
- **The MNIST training-curve figure:** BN reaches a given accuracy in far fewer steps than the baseline → the core "faster convergence" claim.
- **The ImageNet/Inception table:** BN-Inception matches the baseline's accuracy with roughly an order of magnitude fewer training steps, and at higher accuracy with more steps (report these speedups as approximate).
- **The activation-distribution figure:** BN keeps the distribution of a layer's inputs stable over training → the mechanism the authors propose (reduced "internal covariate shift").

### The honest limitations (have an opinion)
- **The *why* is debated.** The original "internal covariate shift" explanation was later challenged: *How Does Batch Normalization Help Optimization?* (Santurkar et al., 2018) argues BN's real effect is **smoothing the loss landscape** (smaller, more predictable gradients), not reducing covariate shift. Good answer: "It clearly helps optimization; the *mechanism* is still argued, with the smoothness view now more favored."
- **Batch-size dependence.** Stats are estimated per batch, so BN degrades with very small batches; LayerNorm/GroupNorm avoid this and dominate in Transformers/RNNs where batch stats are awkward.
- **Train/eval mismatch is a real-world footgun.** Forgetting `model.eval()` (so it normalizes with the tiny inference batch's stats) silently wrecks predictions.
- **Interacts awkwardly with dropout** (variance shift between train and test); many architectures use one or the other.

---

## Part 2 — The interview dialogue (interviewer ⇄ interviewee)

> **🧑‍💼 Interviewer:** One paragraph — what does Batch Normalization actually buy me?
>
> **🧑‍💻 Interviewee:** It makes deep nets much easier to optimize. For each feature I normalize across the mini-batch to zero mean and unit variance, then apply a learnable scale γ and shift β so the layer can still represent whatever distribution it needs. In practice that means I can use a higher learning rate, I'm less sensitive to initialization, and the network converges in far fewer steps — the paper reports roughly an order-of-magnitude speedup on Inception, though I'd treat the exact number as approximate. It also adds a little regularizing noise from the batch statistics.

> **🧑‍💼 Interviewer:** Why the learnable γ and β? Isn't normalizing the point?
>
> **🧑‍💻 Interviewee:** If I forced every layer's output to be exactly mean-0/var-1, I'd constrain what the network can represent — for instance a sigmoid would be pinned to its linear region. γ and β let the layer *undo* the normalization when that's optimal: with γ = √(σ²+ε) and β = μ it recovers the identity. So BN adds the normalization benefit during optimization without permanently removing capacity. They're per-feature and trained by backprop like any other parameter.

> **🧑‍💼 Interviewer:** What changes between training and inference?
>
> **🧑‍💻 Interviewee:** At training time I normalize using the *current batch's* mean and variance, and I update a running estimate of the population mean/variance. At inference I use those stored running stats — I don't have a batch and I don't want my output to depend on which other examples happen to be batched with it. In PyTorch this is the `training` flag that `model.train()` / `model.eval()` flip. Forgetting to call `eval()` is the classic bug: predictions become batch-dependent and metrics look randomly worse.

> **🧑‍💼 Interviewer:** Subtle one — what variance do you use, biased or unbiased?
>
> **🧑‍💻 Interviewee:** Two different ones. The *normalization* itself uses the biased variance, dividing by m — that's `unbiased=False`, and it's what makes the normalized output exactly unit-variance on the batch. But PyTorch updates the *running* variance with the *unbiased* estimate, dividing by m−1, because it's estimating the population variance from a sample. If you want to match `torch.nn.BatchNorm1d` numerically you have to get both right.

> **🧑‍💼 Interviewer:** Do you buy the "internal covariate shift" explanation?
>
> **🧑‍💻 Interviewee:** I buy that BN helps optimization — that's well established empirically. The *mechanism* is debated. The original paper attributes it to reducing internal covariate shift, i.e. stabilizing the input distribution each layer sees. But Santurkar et al. (2018) showed you can inject covariate shift after BN and still keep the benefit, and argued the real effect is making the loss landscape smoother — gradients become more Lipschitz and predictive, so larger steps are safe. So I'd present the smoothness view as the better-supported one while noting it's an open discussion.

> **🧑‍💼 Interviewer:** Implement it and show a deep net training with BN but stalling without it.

---

## Part 3 — Implementation

The whole method is: normalize per feature over the batch, apply learnable γ/β, and switch stats source between train and eval while maintaining running buffers.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class BatchNorm1d(nn.Module):
    """BatchNorm over a batch of feature vectors, shape (N, C)."""

    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super().__init__()
        self.eps = eps
        self.momentum = momentum
        self.gamma = nn.Parameter(torch.ones(num_features))   # learnable scale
        self.beta = nn.Parameter(torch.zeros(num_features))   # learnable shift
        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var", torch.ones(num_features))

    def forward(self, x):
        if self.training:
            mean = x.mean(dim=0)                       # per-feature batch mean
            var = x.var(dim=0, unbiased=False)         # BIASED var for normalization
            with torch.no_grad():
                n = x.shape[0]
                unbiased_var = var * n / (n - 1)       # UNBIASED var for running stat
                self.running_mean.mul_(1 - self.momentum).add_(self.momentum * mean)
                self.running_var.mul_(1 - self.momentum).add_(self.momentum * unbiased_var)
        else:
            mean = self.running_mean                   # eval: use stored population stats
            var = self.running_var
        xhat = (x - mean) / torch.sqrt(var + self.eps)
        return self.gamma * xhat + self.beta           # affine: never removes capacity
```

### Why each line matters (talk through it)
- `gamma` / `beta` as `nn.Parameter` — learnable, per-feature scale and shift; trained by backprop so the layer can recover any distribution (including the identity).
- `register_buffer(...)` — running stats are **state, not parameters**: they're saved/loaded and moved to device, but no gradient flows to them.
- `if self.training` — the train/eval switch `nn.Module` flips on `.train()` / `.eval()`. Get this wrong and inference depends on the batch.
- `x.var(dim=0, unbiased=False)` — the **biased** variance ($1/m$) used to *normalize*; this is what makes the output unit-variance on the batch and what matches PyTorch.
- `unbiased_var = var * n / (n - 1)` — the **unbiased** variance ($1/(m-1)$) used to *update the running buffer*; PyTorch does this, and it's needed to match `torch.nn.BatchNorm1d` exactly.
- `with torch.no_grad()` + in-place `mul_/add_` — buffer updates are bookkeeping, kept out of the autograd graph.
- `mean = self.running_mean` (eval branch) — at inference we use stored population stats and update nothing.

### Demonstrating the benefit (deep net that needs BN to train)
A 6-layer MLP at a learning rate that's "too high" for the plain net: without normalization the deep net's signal is poorly conditioned and it **stalls near the mean predictor**, while the same net **with BN converges almost to zero loss** at the *same* LR. This is the optimization/stability benefit in one plot.

```python
torch.manual_seed(0)
N, D = 512, 20
X = torch.randn(N, D)
w_true = torch.randn(D, 1)
y = torch.sin(X @ w_true) + 0.1 * torch.randn(N, 1)   # nonlinear target

def make_net(use_bn, depth=6, width=128, in_dim=20):
    torch.manual_seed(0)
    layers = [nn.Linear(in_dim, width)]
    if use_bn: layers.append(BatchNorm1d(width))
    layers.append(nn.ReLU())
    for _ in range(depth - 1):
        layers.append(nn.Linear(width, width))
        if use_bn: layers.append(BatchNorm1d(width))
        layers.append(nn.ReLU())
    layers.append(nn.Linear(width, 1))
    return nn.Sequential(*layers)

def train(use_bn, lr=0.05, steps=300):
    torch.manual_seed(0)
    net = make_net(use_bn)
    opt = torch.optim.SGD(net.parameters(), lr=lr)
    net.train()
    for _ in range(steps):
        opt.zero_grad()
        F.mse_loss(net(X), y).backward()
        opt.step()
    net.eval()
    with torch.no_grad():
        return F.mse_loss(net(X), y).item()

print(f"no-BN final loss = {train(False):.4f}")   # stalls ~0.50 (predicts the mean)
print(f"   BN final loss = {train(True):.6f}")     # converges ~0.00
```

You should see the plain net stuck around **0.50** (it basically learned to predict the mean) while the BN net drops to **~1e-5**. (Exact numbers are seed-dependent; the *direction* — BN trains, the plain deep net stalls at the same LR — is the point. Push the LR higher and even the BN net can eventually diverge under plain SGD, which is itself a good talking point about stability limits.)

---

## Part 4 — Sanity checks (don't skip)

### Check 1 — Train mode normalizes per feature (≈0 mean, ≈1 var, before γ/β)
```python
bn = BatchNorm1d(8).train()              # gamma=1, beta=0 by default
x = torch.randn(1024, 8) * 5 + 3
y = bn(x)
assert torch.allclose(y.mean(dim=0), torch.zeros(8), atol=1e-4)
assert torch.allclose(y.var(dim=0, unbiased=False), torch.ones(8), atol=1e-2)
print("OK: train-mode output ~0 mean, ~1 var per feature")
```

### Check 2 — Eval mode uses running stats, not batch stats
```python
bn = BatchNorm1d(8).train()
for _ in range(50):
    bn(torch.randn(256, 8) * 2 + 1)      # populate running stats
bn.eval()
xb = torch.randn(256, 8) * 10 + 50       # wildly different batch stats
y_eval = bn(xb)
assert not torch.allclose(y_eval.mean(dim=0), torch.zeros(8), atol=0.5)  # NOT batch-normalized
manual = (xb - bn.running_mean) / torch.sqrt(bn.running_var + bn.eps)
assert torch.allclose(y_eval, manual, atol=1e-5)                          # uses buffers
print("OK: eval uses running buffers, not the batch")
```

### Check 3 — Running mean/var actually update across batches
```python
bn = BatchNorm1d(4).train()
rm0, rv0 = bn.running_mean.clone(), bn.running_var.clone()
bn(torch.randn(128, 4) * 3 + 2)
assert not torch.allclose(bn.running_mean, rm0)
assert not torch.allclose(bn.running_var, rv0)
print("OK: running mean/var update")
```

### Check 4 — Numerical match to `torch.nn.BatchNorm1d` (the key reference check)
```python
mine = BatchNorm1d(16, eps=1e-5, momentum=0.1)
ref = nn.BatchNorm1d(16, eps=1e-5, momentum=0.1)
with torch.no_grad():
    ref.weight.copy_(mine.gamma); ref.bias.copy_(mine.beta)
mine.train(); ref.train()
x = torch.randn(64, 16) * 2.5 - 1.0
assert torch.allclose(mine(x), ref(x), atol=1e-5)                          # train output
assert torch.allclose(mine.running_mean, ref.running_mean, atol=1e-5)      # running buffers
assert torch.allclose(mine.running_var, ref.running_var, atol=1e-5)        # (needs unbiased!)
mine.eval(); ref.eval()
xe = torch.randn(64, 16)
assert torch.allclose(mine(xe), ref(xe), atol=1e-5)                        # eval output
print("OK: matches torch.nn.BatchNorm1d (train, eval, buffers)")
```

### Check 5 — Gradient flows to γ and β
```python
bn = BatchNorm1d(8).train()
bn(torch.randn(32, 8)).sum().backward()
assert bn.gamma.grad is not None and bn.gamma.grad.abs().sum() > 0
assert bn.beta.grad is not None and bn.beta.grad.abs().sum() > 0
print("OK: gradient flows to gamma and beta")
```

### Check 6 — γ/β scale and shift the normalized output
```python
bn = BatchNorm1d(4).train()
with torch.no_grad():
    bn.gamma.fill_(2.0); bn.beta.fill_(5.0)
y = bn(torch.randn(2048, 4))
assert torch.allclose(y.mean(dim=0), torch.full((4,), 5.0), atol=1e-3)             # beta -> mean
assert torch.allclose(y.var(dim=0, unbiased=False), torch.full((4,), 4.0), atol=5e-2)  # gamma^2 -> var
print("OK: affine gamma/beta scale+shift the output")
```

---

## Part 5 — Likely follow-up questions

- *"BatchNorm vs LayerNorm — when each?"* — BN normalizes **across the batch, per feature**; LN normalizes **across features, per example**. LN has no batch dependence, so it's the default in Transformers/RNNs and with tiny batches; BN shines in CNNs with reasonable batch sizes.
- *"Where do you put BN — before or after the activation?"* — The paper puts it **before** the nonlinearity (on the pre-activation). In practice both are used; "before activation" is the classic recipe. With BN you can also **drop the preceding layer's bias** since β subsumes it.
- *"What about very small batches?"* — Batch stats get noisy and BN degrades; use GroupNorm/LayerNorm, or sync stats across devices (SyncBN).
- *"Does BN regularize?"* — Yes, mildly — each example is normalized using batch-dependent stats, injecting noise. That's why BN sometimes lets you reduce dropout.
- *"Why does BN really work?"* — Open debate: original "internal covariate shift" vs. the loss-landscape **smoothing** view (Santurkar et al., 2018), which is now more favored. Either way it helps optimization and tolerates higher LRs.
- *"Can you fold BN into the previous layer at inference?"* — Yes: at eval BN is an affine map with fixed stats, so you can fuse it into the preceding `Linear`/`Conv` weights for zero overhead.

---

## TL;DR cheat sheet
| Thing | Answer |
|---|---|
| Core idea | Normalize each feature over the batch to 0-mean/1-var, then learnable γ scale + β shift |
| Formula | $\hat{x}=\dfrac{x-\mu_\mathcal{B}}{\sqrt{\sigma_\mathcal{B}^2+\epsilon}}$, $y=\gamma\hat{x}+\beta$ |
| Why γ/β | Recover any distribution (incl. identity) → never removes capacity |
| Train vs eval | Train: batch stats + update running buffers. Eval: running stats, no update |
| Variance gotcha | Normalize with **biased** (`unbiased=False`); update running var with **unbiased** |
| Benefit | Higher LR, less init-sensitivity, faster convergence, mild regularization |
| #1 bug | Forgetting `model.eval()` (inference becomes batch-dependent) |
| Mechanism | Debated: "internal covariate shift" vs. loss-landscape **smoothing** (now favored) |
| Limitation | Batch-size dependent; small batches hurt → LayerNorm/GroupNorm instead |
