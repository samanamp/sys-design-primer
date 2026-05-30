---
title: DDPM — Paper-to-Code Mock Interview
description: A mock interview (read paper, explain benefit, implement in Colab) using Denoising Diffusion Probabilistic Models as the worked example.
sidebar:
  order: 20
  label: DDPM
---

> **Paper:** *Denoising Diffusion Probabilistic Models* — Ho, Jain & Abbeel, 2020. [arXiv:2006.11239](https://arxiv.org/abs/2006.11239)
>
> **Format:** Read (~15 min) → explain the *real* benefit → implement the core idea in Colab → sanity-check it.
>
> **Companion notebook:** [`ddpm_mock.ipynb`](/notebooks/ddpm_mock.ipynb) (download) — a 2D generative demo + an `EpsTheta` / `q_sample` / `sample` stub to fill in, plus verification cells. Open in [Google Colab](https://colab.research.google.com/) via *File → Upload notebook*.
>
> **Difficulty:** 🔴 Hard. Several moving parts (forward schedule, noise-prediction loss, reverse sampler) have to line up before anything generates. Do Dropout and LoRA first.

---

## How to run this as a timed drill (~40 min)

| Time | Block | What you produce |
|------|-------|------------------|
| 0:00–0:12 | **Read** (use the [three-pass method](/papers/lora-mock-interview/#part-0--how-to-read-a-paper-in-15-minutes-the-three-pass-method)) | The forward/reverse split + why training is plain MSE |
| 0:12–0:17 | **Explain the benefit** out loud (cover Part 2) | Predict-the-noise reparam + the closed-form forward jump |
| 0:17–0:33 | **Implement** from the stub (Part 3) | A working `q_sample` + `EpsTheta` + `sample` that generates the target distribution |
| last 5 min | **Sanity-check** (Part 4) | All checks passing, narrated out loud |

### Self-grading rubric — "what good looks like"
- ✅ Explained the **fixed forward process** (add Gaussian noise on a schedule) vs the **learned reverse process** (a network that undoes one step).
- ✅ Knew the **closed-form forward jump** $x_t = \sqrt{\bar\alpha_t}\,x_0 + \sqrt{1-\bar\alpha_t}\,\varepsilon$ — you can sample any timestep directly, no loop.
- ✅ Stated the **training objective is a simple MSE** on predicted noise — stable, no adversary, unlike GANs.
- ✅ Demonstrated generation: **sample pure noise, run the reverse loop, recover the target distribution** — proven numerically, not just "it runs."
- ⚠️ Red flags: thinking the forward process is learned, predicting $x_0$ or $x_{t-1}$ directly without realizing it's reparametrized to predict $\varepsilon$, forgetting the extra noise term in the reverse step (except at $t=0$).

---

## Part 1 — Structured read of THIS paper

### The 30-second summary (the "benefit")
A diffusion model learns to generate data by **reversing a gradual noising process**. The forward process is *fixed* (no learning): it slowly corrupts a data point $x_0$ into pure Gaussian noise over $T$ steps. A network is then trained to **reverse** that — and thanks to a clever reparametrization, all it has to do is **predict the noise** that was added. The payoff:

- **Training is a simple, stable MSE regression** — `loss = ||ε − ε_θ(x_t, t)||²`. There's no discriminator, no adversarial game, no mode-collapse failure modes that plague GANs.
- **You can sample any timestep in closed form**, so each training step is one cheap forward jump — no need to simulate the whole chain during training.
- **Sampling from pure noise produces high-quality samples** by iterating the learned reverse step from $x_T \sim \mathcal{N}(0,I)$ down to $x_0$. This is what put diffusion on the map for image synthesis.

### The core idea (Method — you implement this)

**Forward process.** Pick a variance schedule $\beta_1,\dots,\beta_T$ (linear is fine). Define $\alpha_t = 1-\beta_t$ and the cumulative product $\bar\alpha_t = \prod_{s=1}^{t}\alpha_s$. The single-step forward is $q(x_t \mid x_{t-1}) = \mathcal{N}(\sqrt{\alpha_t}\,x_{t-1}, \beta_t I)$, but the key is that it **composes in closed form**:

$$q(x_t \mid x_0) = \mathcal{N}\big(\sqrt{\bar\alpha_t}\,x_0,\;(1-\bar\alpha_t) I\big) \quad\Longrightarrow\quad x_t = \sqrt{\bar\alpha_t}\,x_0 + \sqrt{1-\bar\alpha_t}\,\varepsilon,\;\; \varepsilon\sim\mathcal{N}(0,I)$$

Because $\bar\alpha_t \to 0$ as $t\to T$, the final $x_T$ is (approximately) standard normal — pure noise.

**Training objective.** The reverse step is also Gaussian, and Ho et al. show the optimal thing to learn is the noise. The network $\varepsilon_\theta(x_t, t)$ takes a noised point and the timestep, and predicts the noise:

$$\mathcal{L} = \mathbb{E}_{x_0,\,t,\,\varepsilon}\big[\,\lVert \varepsilon - \varepsilon_\theta(x_t, t)\rVert^2\,\big], \qquad x_t = \sqrt{\bar\alpha_t}\,x_0 + \sqrt{1-\bar\alpha_t}\,\varepsilon$$

**Reverse / sampling step.** Start from $x_T\sim\mathcal{N}(0,I)$ and walk back. Each step uses the predicted noise to form the posterior mean, then adds fresh noise (except at the last step):

$$x_{t-1} = \frac{1}{\sqrt{\alpha_t}}\Big(x_t - \frac{1-\alpha_t}{\sqrt{1-\bar\alpha_t}}\,\varepsilon_\theta(x_t, t)\Big) + \sqrt{\beta_t}\,z, \qquad z\sim\mathcal{N}(0,I)\;\text{ (}z=0\text{ at }t=0\text{)}$$

Key details (the things an interviewer probes):
- **The forward process has no parameters.** It's a fixed Markov chain; only the reverse network is trained.
- **Why predict $\varepsilon$ and not $x_0$?** It's a reparametrization of the same Gaussian; predicting noise gives a simpler, better-conditioned loss (the famous "$L_\text{simple}$") and empirically works best.
- **The timestep $t$ is an input.** The same network handles all noise levels, so it needs to know which level it's at — hence a time embedding (sinusoidal or learned).
- **Don't forget the $\sqrt{\beta_t}\,z$ term** in sampling — except at $t=0$, where you take just the mean. Dropping it everywhere collapses diversity.

### Where the evidence lives (tables that matter)
*(Figure/number references are from memory of the paper — verify against the PDF before quoting them in an interview.)*
- **CIFAR-10 FID/Inception table:** DDPM reaches competitive/state-of-the-art FID for the time → the headline quality claim.
- **Ablation: predicting $\varepsilon$ vs $\tilde\mu$ and fixed vs learned variances:** the simplified $\varepsilon$-prediction loss wins → justifies the objective you implement.
- **Sample grids (CelebA-HQ / LSUN):** qualitative high-resolution samples → the "produces high-quality samples" claim, visually.

### The honest limitations (have an opinion)
- **Sampling is slow:** generation requires $T$ (often hundreds to a thousand) sequential network evaluations. This is the big practical cost; later work (DDIM, distillation) attacks it.
- **No likelihood-optimal by default:** the simplified loss drops terms, so it's not directly optimizing a tight likelihood bound (a different trade-off than the full variational objective).
- **Compute-hungry to train** on real images relative to a single-pass model, because of the many noise levels it must cover.

---

## Part 2 — The interview dialogue (interviewer ⇄ interviewee)

> **🧑‍💼 Interviewer:** One paragraph — what does a diffusion model actually do?
>
> **🧑‍💻 Interviewee:** It learns to generate data by reversing a noising process. There's a *fixed* forward process that gradually adds Gaussian noise to a data point until it's pure noise, and a *learned* reverse process that undoes one noising step at a time. The trick is that the reverse step can be reparametrized so the network just predicts the noise that was added — so training is plain MSE regression, stable and adversary-free. To generate, I start from pure Gaussian noise and run the learned reverse step from $t=T$ down to $0$.

> **🧑‍💼 Interviewer:** During training, do you simulate the whole noising chain for each example?
>
> **🧑‍💻 Interviewee:** No — that's the elegant part. The forward process composes in closed form: $x_t = \sqrt{\bar\alpha_t}\,x_0 + \sqrt{1-\bar\alpha_t}\,\varepsilon$ where $\bar\alpha_t$ is the cumulative product of $\alpha_s = 1-\beta_s$. So I sample a random timestep $t$, sample one noise vector $\varepsilon$, jump straight to $x_t$, and train the network to predict that $\varepsilon$. One forward jump per step, no loop.

> **🧑‍💼 Interviewer:** Why predict the noise instead of the clean image or the previous state?
>
> **🧑‍💻 Interviewee:** They're algebraically interchangeable — given $x_t$ and a prediction of $\varepsilon$ you can recover $x_0$ and the posterior mean. But Ho et al. found the noise-prediction parametrization gives a much simpler, better-conditioned objective — their $L_\text{simple}$, just $\lVert\varepsilon-\varepsilon_\theta\rVert^2$ — and it trains better in practice. It also makes the target the same scale ($\mathcal{N}(0,I)$) regardless of timestep.

> **🧑‍💼 Interviewer:** Walk me through one sampling step. What's the easiest thing to get wrong?
>
> **🧑‍💻 Interviewee:** Given $x_t$, the network predicts $\varepsilon_\theta$. I form the posterior mean $\frac{1}{\sqrt{\alpha_t}}\big(x_t - \frac{1-\alpha_t}{\sqrt{1-\bar\alpha_t}}\varepsilon_\theta\big)$, then add $\sqrt{\beta_t}\,z$ with fresh $z\sim\mathcal{N}(0,I)$. The classic bug is forgetting that extra noise term — or, conversely, adding it at the final step $t=0$ when you should just take the mean. Drop the noise everywhere and samples collapse; add it at $t=0$ and your final output is needlessly jittery.

> **🧑‍💼 Interviewer:** What's the main downside versus a GAN or VAE?
>
> **🧑‍💻 Interviewee:** Sampling speed. Generating one sample needs $T$ sequential passes through the network — hundreds to a thousand — whereas a GAN is a single forward pass. That's the price for the stable training and sample quality. Follow-ups like DDIM and distillation cut the step count dramatically, but vanilla DDPM is slow at inference.

> **🧑‍💼 Interviewer:** Implement it and show it generate the target distribution.

---

## Part 3 — Implementation

Three pieces have to agree: the **schedule** (precomputed $\alpha$, $\bar\alpha$), the **forward jump** + **MSE loss**, and the **reverse sampler**.

```python
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

T = 400

def make_schedule(T, beta_start=1e-4, beta_end=0.04):
    betas = torch.linspace(beta_start, beta_end, T)   # variance added at each step
    alphas = 1.0 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)         # cumulative product -> closed-form jump
    return betas, alphas, alpha_bars

betas, alphas, alpha_bars = make_schedule(T)


def q_sample(x0, t, eps):
    """Forward jump: x_t = sqrt(abar_t) x0 + sqrt(1-abar_t) eps  (closed form)."""
    ab = alpha_bars[t].unsqueeze(-1)                  # (B,1) per-sample abar
    return torch.sqrt(ab) * x0 + torch.sqrt(1.0 - ab) * eps


def timestep_embedding(t, dim):
    """Sinusoidal embedding of the integer timestep so one net handles all noise levels."""
    half = dim // 2
    freqs = torch.exp(-math.log(10000.0) * torch.arange(half) / half)
    args = t.float().unsqueeze(-1) * freqs.unsqueeze(0)
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


class EpsTheta(nn.Module):
    """Noise predictor: (x_t, t) -> predicted epsilon, same shape as x_t."""
    def __init__(self, data_dim=2, t_dim=32, hidden=128):
        super().__init__()
        self.t_dim = t_dim
        self.net = nn.Sequential(
            nn.Linear(data_dim + t_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, data_dim),
        )

    def forward(self, x_t, t):
        temb = timestep_embedding(t, self.t_dim)      # tell the net which noise level
        return self.net(torch.cat([x_t, temb], dim=-1))


@torch.no_grad()
def sample(model, n, data_dim=2):
    """Reverse process: start from N(0,I), walk back to x_0."""
    x = torch.randn(n, data_dim)                       # x_T ~ pure noise
    for i in reversed(range(T)):
        t = torch.full((n,), i, dtype=torch.long)
        eps = model(x, t)
        a, ab = alphas[i], alpha_bars[i]
        coef = (1.0 - a) / torch.sqrt(1.0 - ab)
        mean = (1.0 / torch.sqrt(a)) * (x - coef * eps)   # posterior mean
        if i > 0:
            x = mean + torch.sqrt(betas[i]) * torch.randn_like(x)  # add noise...
        else:
            x = mean                                              # ...except at t=0
    return x
```

### Why each line matters (talk through it)
- `alpha_bars = torch.cumprod(alphas)` — this single cumulative product is what makes the closed-form forward jump possible; without it you'd have to iterate the chain.
- `q_sample` returns `sqrt(ab)*x0 + sqrt(1-ab)*eps` — the entire forward process in one line, for *any* `t` at once.
- `timestep_embedding` — the same network sees every noise level, so it must be conditioned on `t`; the embedding turns an integer into a smooth feature vector.
- In `sample`, `coef = (1-a)/sqrt(1-ab)` and the `1/sqrt(a)` prefactor are exactly the DDPM posterior-mean formula — get either wrong and samples drift off the manifold.
- `if i > 0: ... + sqrt(beta)*z else: mean` — the make-or-break detail: stochastic everywhere except the final denoising step.

### Demonstrating the benefit (2D generative toy task)
The claim is "train a denoiser on a distribution, then sample new points from noise that match it." We use a **ring** of radius 2 — structure that's obvious to check numerically (radius mean/std) and impossible to fake by collapsing to a single point (we check all angular sectors are populated).

```python
R_TRUE, SIGMA = 2.0, 0.05

def sample_ring(n):
    theta = torch.rand(n) * 2 * math.pi
    r = R_TRUE + SIGMA * torch.randn(n)
    return torch.stack([r * torch.cos(theta), r * torch.sin(theta)], dim=-1)

torch.manual_seed(0)
model = EpsTheta()
opt = torch.optim.Adam(model.parameters(), lr=2e-3)
losses = []
for step in range(3000):
    x0 = sample_ring(512)
    t = torch.randint(0, T, (512,))
    eps = torch.randn_like(x0)
    loss = F.mse_loss(model(q_sample(x0, t, eps), t), eps)   # L_simple
    opt.zero_grad(); loss.backward(); opt.step()
    losses.append(loss.item())

gen = sample(model, 4000)
gen_r = torch.sqrt((gen ** 2).sum(-1))
print(f"loss: {sum(losses[:50])/50:.3f} -> {sum(losses[-50:])/50:.3f}")
print(f"radius mean: gen {gen_r.mean():.3f}  (true {R_TRUE})")
print(f"radius std : gen {gen_r.std():.3f}")
print(f"fraction within 0.3 of ring: {((gen_r - R_TRUE).abs() < 0.3).float().mean():.3f}")
```

Expected output (seed 0): the loss drops from ~0.55 to ~0.28, generated radius mean ≈ 2.00, and ~1.00 of generated points land on the ring. Sampling from pure noise reproduced the target distribution — that's diffusion working.

---

## Part 4 — Sanity checks (don't skip)

### Check 1 — Forward process ends at ≈ standard normal
```python
x0 = sample_ring(20000)
tT = torch.full((20000,), T - 1, dtype=torch.long)
xT = q_sample(x0, tT, torch.randn_like(x0))
m, s = xT.mean().item(), xT.std().item()
print(f"x_T mean {m:.3f} std {s:.3f} (expected ~0, ~1)")
assert abs(m) < 0.1 and abs(s - 1.0) < 0.1
```

### Check 2 — ᾱ is decreasing and in (0, 1]
```python
assert (alpha_bars[1:] < alpha_bars[:-1]).all()
assert (alpha_bars > 0).all() and (alpha_bars <= 1.0).all()
print(f"abar_0={alpha_bars[0]:.4f}  abar_T={alpha_bars[-1]:.4f}  (decreasing toward ~0)")
```

### Check 3 — q_sample at t=0 returns ≈ x0 (ᾱ₀ ≈ 1)
```python
x0 = sample_ring(1000)
t0 = torch.zeros(1000, dtype=torch.long)
err = (q_sample(x0, t0, torch.randn_like(x0)) - x0).abs().max().item()
print(f"max|x_t - x0| at t=0: {err:.4f} (expected ~0)")
assert err < 0.05
```

### Check 4 — eps_theta output shape == input shape
```python
xin = torch.randn(7, 2)
tin = torch.randint(0, T, (7,))
out = model(xin, tin)
print(f"output shape {tuple(out.shape)} == input {tuple(xin.shape)}")
assert out.shape == xin.shape
```

### Check 5 — Generated samples match the target distribution
```python
gen = sample(model, 4000)
gen_r = torch.sqrt((gen ** 2).sum(-1))
ang = torch.atan2(gen[:, 1], gen[:, 0])
sectors = ((ang + math.pi) / (2 * math.pi) * 8).long().clamp(0, 7)
counts = torch.bincount(sectors, minlength=8)
assert abs(gen_r.mean().item() - R_TRUE) < 0.15           # right radius
assert ((gen_r - R_TRUE).abs() < 0.3).float().mean() > 0.85  # on the ring
assert (counts > 50).all()                                # all sectors covered (no collapse)
print("generated distribution matches ring target")
```

### Check 6 — Loss decreases and gradient flows
```python
assert sum(losses[-50:]) / 50 < sum(losses[:50]) / 50
m2 = EpsTheta()
x0 = sample_ring(64); t = torch.randint(0, T, (64,)); eps = torch.randn_like(x0)
F.mse_loss(m2(q_sample(x0, t, eps), t), eps).backward()
gnorm = sum(p.grad.abs().sum() for p in m2.parameters() if p.grad is not None).item()
print(f"loss decreased; grad norm {gnorm:.3f}")
assert gnorm > 0
```

---

## Part 5 — Likely follow-up questions

- *"How would you speed up sampling?"* — Use DDIM (deterministic, non-Markovian sampler that skips steps), fewer timesteps with a re-spaced schedule, or distillation (progressive / consistency models) to collapse many steps into one.
- *"Linear vs cosine schedule?"* — The cosine schedule (Nichol & Dhariwal) adds noise more gently at the start and end, keeping more signal in the middle timesteps; it improves likelihood and sample quality over the original linear schedule.
- *"How does this relate to score matching / SDEs?"* — Predicting $\varepsilon$ is equivalent (up to scaling) to estimating the score $\nabla_x \log p_t(x)$; Song et al.'s score-based SDE framework unifies DDPM as the discretization of a reverse-time SDE.
- *"How do you condition it (e.g. text-to-image)?"* — Feed the condition into $\varepsilon_\theta$ (cross-attention / concatenation) and use classifier or classifier-free guidance to trade diversity for fidelity at sampling time.
- *"Why is predicting noise better-conditioned than predicting $x_0$?"* — The target $\varepsilon$ is unit-variance at every timestep, so the loss scale is uniform across $t$; predicting $x_0$ or $\mu$ has wildly different difficulty/scale per noise level.

---

## TL;DR cheat sheet
| Thing | Answer |
|---|---|
| Core idea | Fixed forward noising + learned reverse denoising |
| Forward jump | $x_t = \sqrt{\bar\alpha_t}\,x_0 + \sqrt{1-\bar\alpha_t}\,\varepsilon$ (closed form) |
| Schedule | $\alpha_t=1-\beta_t$, $\bar\alpha_t=\prod\alpha_s$; $\bar\alpha_T\approx 0$ |
| Training loss | $\lVert\varepsilon - \varepsilon_\theta(x_t,t)\rVert^2$ — plain MSE, no adversary |
| Sampling | From $x_T\sim\mathcal{N}(0,I)$, iterate posterior mean $+\sqrt{\beta_t}z$ ($z=0$ at $t=0$) |
| Benefit | Stable training + high-quality samples from noise |
| #1 bug | Dropping the $\sqrt{\beta_t}z$ term (or adding it at $t=0$) |
| Limitation | Slow sampling: $T$ sequential network passes |
