---
title: VAE — Paper-to-Code Mock Interview
description: A combined mock (read paper, explain benefit, implement in Colab) using the Variational Autoencoder as the worked example — reparameterization + the ELBO.
sidebar:
  order: 15
  label: VAE
---

> **Paper:** *Auto-Encoding Variational Bayes* — Kingma & Welling, 2013. [arXiv:1312.6114](https://arxiv.org/abs/1312.6114)
>
> **Format:** Read (~15 min) → explain the *real* benefit → implement the core idea in Colab → sanity-check it.
>
> **Companion notebook:** [`vae_mock.ipynb`](/notebooks/vae_mock.ipynb) (download) — a 2D generation demo + a `VAE` stub to fill in, plus verification cells. Open in [Google Colab](https://colab.research.google.com/) via *File → Upload notebook*.
>
> **Difficulty:** 🟡🔴 Medium-hard. The reparameterization trick and the ELBO derivation are the parts interviewers love to probe.

---

## How to run this as a timed drill (~40 min)

| Time | Block | What you produce |
|------|-------|------------------|
| 0:00–0:12 | **Read** (use the [three-pass method](/papers/lora-mock-interview/#part-0--how-to-read-a-paper-in-15-minutes-the-three-pass-method)) | Why we can't backprop through sampling + the reparameterization fix |
| 0:12–0:17 | **Explain the benefit** out loud (cover Part 2) | The ELBO (reconstruction − KL) + why a smooth latent space lets you generate |
| 0:17–0:33 | **Implement** from the stub (Part 3) | A working `VAE` + generated 2D points that match the data distribution |
| last 5 min | **Sanity-check** (Part 4) | All checks passing, narrated out loud |

### Self-grading rubric — "what good looks like"
- ✅ Explained the **reparameterization trick** as the thing that makes sampling differentiable — `z = μ + σ⊙ε`, push the randomness into `ε` so gradients flow into `μ` and `logvar`.
- ✅ Wrote the **ELBO** as *reconstruction − KL* and knew the KL is a regularizer pulling `q(z|x)` toward the prior `N(0,I)`.
- ✅ Knew the **closed-form KL** for two Gaussians and why we predict **`logvar`** (not `σ`) for numerical stability.
- ✅ Demonstrated **generation**: sample `z ~ N(0,I)`, decode, and show the outputs look like the data — not just "the loss went down."
- ⚠️ Red flags: sampling `z` with `torch.randn` *without* the reparameterization (no gradient to `μ`/`logvar`), confusing the VAE with a plain autoencoder, forgetting the KL term, or predicting `σ` directly and getting NaNs.

---

## Part 1 — Structured read of THIS paper

### The 30-second summary (the "benefit")
A plain autoencoder learns to compress and reconstruct, but its latent space is full of holes — sample a random point and the decoder produces garbage. A **VAE** turns the encoder into a *probabilistic* one: it outputs a distribution `q(z|x) = N(μ, σ²)` per input, and a KL term forces those distributions to overlap with a fixed prior `N(0,I)`. The payoff:

- You get a **generative model** you can train end-to-end with plain gradient descent — no MCMC, no EM.
- The latent space becomes **smooth and sample-able**: draw `z ~ N(0,I)`, decode, and out comes a *new* data point that looks like your training distribution.
- It gives a principled objective — a **lower bound on the data log-likelihood** (the ELBO) — that you maximize directly.

### The core idea (Method — you implement this)
We want to maximize the (intractable) log-likelihood `log p(x)`. The VAE instead maximizes a tractable **lower bound**, the ELBO:

$$\log p(x) \ge \mathbb{E}_{q(z|x)}\big[\log p(x|z)\big] - \mathrm{KL}\big(q(z|x)\,\|\,p(z)\big)$$

The first term is **reconstruction** (decode `z`, compare to `x`); the second is a **KL regularizer** pulling the encoder's posterior toward the prior `p(z) = N(0,I)`. For continuous data we use Gaussian likelihood, so reconstruction becomes an MSE.

The crux is estimating that expectation with a gradient we can backprop. Sampling `z ~ q(z|x)` directly is **not differentiable** in `μ`/`σ`. The **reparameterization trick** rewrites the sample so the randomness lives in a parameter-free `ε`:

$$z = \mu + \sigma \odot \varepsilon, \qquad \varepsilon \sim \mathcal{N}(0, I)$$

Now `z` is a deterministic, differentiable function of `μ` and `σ`, and gradients flow through it. Both Gaussians, the KL has a **closed form**:

$$\mathrm{KL}\big(\mathcal{N}(\mu,\sigma^2)\,\|\,\mathcal{N}(0,1)\big) = -\tfrac{1}{2}\sum_j\big(1 + \log\sigma_j^2 - \mu_j^2 - \sigma_j^2\big)$$

Key details (the things an interviewer probes):
- **Predict `logvar`, not `σ`.** The encoder outputs `logvar = log σ²`; then `σ = exp(0.5·logvar)`. This keeps `σ > 0` automatically and is numerically stable. The KL formula above uses `logvar` directly.
- **Why reparameterize?** `z ~ N(μ,σ²)` is a random *node* — you can't differentiate through "draw a sample." Writing `z = μ + σ·ε` moves the stochasticity to `ε` (no parameters), leaving a smooth path from the loss back to `μ` and `logvar`.
- **The KL is a regularizer.** It prevents the encoder from cheating by assigning each `x` a tiny, far-apart `σ` (which would make it a plain autoencoder with a useless prior). Pulling `q` toward `N(0,I)` is exactly what makes the latent space sample-able.
- **β / KL warm-up.** In practice people scale the KL term (β-VAE) or anneal it from 0 to avoid early *posterior collapse*, where `q` snaps to the prior and ignores `x`.

### Where the evidence lives (figures/tables that matter)
- **Generated-sample grids (MNIST / Frey faces):** decode points from the prior → realistic new images → the generative claim.
- **Latent-space manifold figure:** a 2D latent traversal that smoothly morphs between data points → the "smooth, sample-able space" claim.
- **Marginal-likelihood / lower-bound curves vs. wake-sleep & Monte-Carlo EM:** the ELBO is competitive and scales to large datasets → the optimization claim.

### The honest limitations (have an opinion)
- **Blurry samples.** A Gaussian decoder + the KL pressure tends to produce blurry reconstructions; GANs and diffusion models sharpen this up at the cost of a clean likelihood objective.
- **Posterior collapse.** With a powerful decoder the model can ignore `z` entirely; needs KL annealing / β-tuning / weaker decoders.
- **The bound has a gap.** You optimize a *lower bound*, not the true likelihood; a too-simple `q(z|x)` (diagonal Gaussian) leaves the bound loose.

---

## Part 2 — The interview dialogue (interviewer ⇄ interviewee)

> **🧑‍💼 Interviewer:** One paragraph — what does a VAE actually buy me over a normal autoencoder?
>
> **🧑‍💻 Interviewee:** A *generative* model with a usable latent space. A plain autoencoder just memorizes a compression; its latent space has holes, so sampling a random code decodes to noise. A VAE makes the encoder output a distribution `N(μ,σ²)` per input and adds a KL term that pulls those posteriors toward a fixed `N(0,I)` prior. That regularization makes the latent space smooth and continuous, so I can draw `z ~ N(0,I)`, decode it, and get a brand-new sample that looks like my data. And I train the whole thing with ordinary backprop by maximizing the ELBO.

> **🧑‍💼 Interviewer:** Walk me through the reparameterization trick. Why is it necessary?
>
> **🧑‍💻 Interviewee:** The ELBO has an expectation over `z ~ q(z|x)`. To optimize it I need a gradient with respect to the encoder's `μ` and `σ`, but "sample from a distribution" is a stochastic node you can't differentiate through. Reparameterization rewrites the sample as `z = μ + σ⊙ε` with `ε ~ N(0,I)`. Now the only randomness is `ε`, which has no parameters, and `z` is a smooth deterministic function of `μ` and `σ`. So gradients flow from the reconstruction loss straight back into the encoder. Without it, the gradient estimator (score-function/REINFORCE) is much higher variance.

> **🧑‍💼 Interviewer:** Why does the encoder output `logvar` instead of `σ` directly?
>
> **🧑‍💻 Interviewee:** Two reasons. First, `σ` must be positive — predicting `logvar` and taking `σ = exp(0.5·logvar)` guarantees that without a clamp or softplus. Second, it's numerically stable: variances span many orders of magnitude, and the closed-form Gaussian KL is naturally written in terms of `logvar`, so `exp` and `log` cancel cleanly. Predicting `σ` directly tends to produce NaNs.

> **🧑‍💼 Interviewer:** What's the role of the KL term — what happens if I drop it?
>
> **🧑‍💻 Interviewee:** The KL pulls each `q(z|x)` toward the prior `N(0,I)`. Drop it and you have a plain autoencoder: the encoder is free to scatter codes anywhere with tiny variance, so the prior no longer matches where the data actually lives, and sampling `z ~ N(0,I)` decodes to nonsense. The KL is exactly what makes the prior a *valid* place to sample from. The flip side is over-weighting it causes posterior collapse, where `q` ignores `x` — that's why people anneal it or use β-VAE.

> **🧑‍💼 Interviewer:** Implement it and show that sampling the prior generates data-like points.

---

## Part 3 — Implementation

The whole method is: encoder → `(μ, logvar)` → reparameterize → decoder, trained on `reconstruction + KL`.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class VAE(nn.Module):
    def __init__(self, data_dim=2, hidden=64, latent_dim=2):
        super().__init__()
        self.latent_dim = latent_dim
        self.enc = nn.Sequential(
            nn.Linear(data_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.fc_mu = nn.Linear(hidden, latent_dim)
        self.fc_logvar = nn.Linear(hidden, latent_dim)
        self.dec = nn.Sequential(
            nn.Linear(latent_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, data_dim),
        )

    def encode(self, x):
        h = self.enc(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)        # logvar -> sigma, always positive
        eps = torch.randn_like(std)          # parameter-free noise
        return mu + std * eps                # differentiable in mu AND logvar

    def decode(self, z):
        return self.dec(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


def kl_divergence(mu, logvar):
    # KL( N(mu, sigma^2) || N(0,1) ) = -0.5 * sum(1 + logvar - mu^2 - exp(logvar))
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)


def vae_loss(recon, x, mu, logvar):
    recon_term = F.mse_loss(recon, x, reduction="none").sum(dim=1)  # Gaussian likelihood
    kl_term = kl_divergence(mu, logvar)
    return (recon_term + kl_term).mean(), recon_term.mean(), kl_term.mean()
```

### Why each line matters (talk through it)
- `self.fc_mu` / `self.fc_logvar` — two separate heads off the shared encoder body: the posterior mean and **log-variance**. Predicting `logvar` keeps `σ` positive for free.
- `std = torch.exp(0.5 * logvar)` — convert `logvar = log σ²` to `σ`. (`σ = (σ²)^{0.5} = exp(0.5·log σ²)`.)
- `eps = torch.randn_like(std); return mu + std * eps` — the **reparameterization trick**. The randomness is in `eps`; `mu` and `std` are deterministic, so autograd can backprop through `z`.
- `kl_divergence` — the **closed-form** KL between the diagonal-Gaussian posterior and the standard-normal prior. No sampling needed for this term.
- `recon_term … .sum(dim=1)` — Gaussian decoder ⇒ MSE reconstruction, summed over data dims (per-sample), then averaged. `recon + KL` is the negative ELBO we minimize.

### Demonstrating the benefit (2D generation toy task)
We use a **mixture of three 2D Gaussians** (clear, separated clusters). Train the VAE, then **sample `z ~ N(0,I)`, decode**, and check the generated points land near the true cluster centers — i.e. the model learned to *generate* the data distribution from the prior.

```python
CENTERS = torch.tensor([[2.0, 0.0], [-2.0, 0.0], [0.0, 2.0]])

def make_gmm(n, noise=0.15, seed=0):
    g = torch.Generator().manual_seed(seed)
    idx = torch.randint(0, CENTERS.shape[0], (n,), generator=g)
    return CENTERS[idx] + noise * torch.randn(n, 2, generator=g)

def nearest_center_dist(points):
    return torch.cdist(points, CENTERS).min(dim=1).values   # dist to closest true center

torch.manual_seed(0)
data = make_gmm(2000)
model = VAE(data_dim=2, hidden=64, latent_dim=2)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)

losses = []
for epoch in range(800):
    perm = torch.randperm(data.shape[0])
    epoch_loss = 0.0
    for i in range(0, data.shape[0], 256):
        xb = data[perm[i:i+256]]
        recon, mu, logvar = model(xb)
        loss, _, _ = vae_loss(recon, xb, mu, logvar)
        opt.zero_grad(); loss.backward(); opt.step()
        epoch_loss += loss.item() * xb.shape[0]
    losses.append(epoch_loss / data.shape[0])

# GENERATE: sample the prior, decode, measure how close to real clusters
g = torch.Generator().manual_seed(123)
z = torch.randn(2000, model.latent_dim, generator=g)
with torch.no_grad():
    gen = model.decode(z)
gen_d = nearest_center_dist(gen)
print(f"loss {losses[0]:.3f} -> {losses[-1]:.3f}")
print(f"generated mean dist-to-nearest-center: {gen_d.mean():.3f}")
print(f"fraction within 1.0 of a cluster: {(gen_d < 1.0).float().mean():.3f}")
```

Verified output (seeded; exact numbers are seed/hardware-dependent — the *direction* is the point, and the paper's headline MNIST/Frey-faces image grids are a far richer version of this same "decode the prior → realistic samples" demo):

```
loss 4.190 -> 1.577
generated mean dist-to-nearest-center: 0.331
fraction within 1.0 of a cluster: 0.888
```

So ~89% of points drawn from `N(0,I)` and decoded land within 1.0 of a real cluster — the VAE learned a latent space you can *sample from* to generate new data. (A scatter of real vs. generated points is the prettiest way to see this; the notebook includes an optional plot.)

---

## Part 4 — Sanity checks (don't skip)

### Check 1 — Reparameterization is differentiable (gradient flows to μ and logvar)
```python
mu = torch.zeros(4, 2, requires_grad=True)
logvar = torch.zeros(4, 2, requires_grad=True)
z = VAE().reparameterize(mu, logvar)
z.sum().backward()
assert mu.grad is not None and mu.grad.abs().sum() > 0       # gradient reaches mu
assert logvar.grad is not None and logvar.grad.abs().sum() > 0  # ...and logvar
print("OK: gradient flows through the sampled z")
```

### Check 2 — The closed-form KL is correct (q == prior ⇒ KL = 0; KL ≥ 0)
```python
kl_zero = kl_divergence(torch.zeros(5, 3), torch.zeros(5, 3))   # mu=0, logvar=0
assert torch.allclose(kl_zero, torch.zeros(5))                  # q == N(0,I) -> KL == 0
kl_rand = kl_divergence(torch.randn(100, 3), torch.randn(100, 3))
assert (kl_rand >= -1e-6).all()                                 # KL is non-negative
print("OK: KL(q==prior)=0 and KL>=0")
```

### Check 3 — Shapes (z is latent_dim, recon matches input dim)
```python
m = VAE(data_dim=2, latent_dim=2)
x = torch.randn(8, 2)
recon, mu, logvar = m(x)
z = m.reparameterize(mu, logvar)
assert z.shape == (8, m.latent_dim)
assert recon.shape == x.shape
assert mu.shape == logvar.shape == (8, m.latent_dim)
print("OK: shapes line up")
```

### Check 4 — Sampling the prior & decoding produces data-like outputs
```python
# (uses the trained `model` and `data` from Part 3)
gen_d = nearest_center_dist(gen)
assert gen_d.mean().item() < 0.6                       # generated points hug the clusters
assert (gen_d < 1.0).float().mean().item() > 0.85      # most land near a real cluster
print("OK: prior samples decode to data-like points")
```

### Check 5 — Loss / ELBO decreases over training
```python
assert losses[-1] < losses[0]
assert losses[-1] < 0.5 * losses[0]     # decreased substantially
print(f"OK: loss decreased {losses[0]:.3f} -> {losses[-1]:.3f}")
```

### Check 6 — With logvar very negative, z collapses to μ (σ → 0, deterministic)
```python
mu_fixed = torch.randn(50, 2)
z_det = VAE().reparameterize(mu_fixed, torch.full((50, 2), -30.0))
assert torch.allclose(z_det, mu_fixed, atol=1e-4)   # sigma ~ 0 => z == mu
print("OK: logvar -> -inf makes sampling deterministic")
```

---

## Part 5 — Likely follow-up questions

- *"VAE vs. a plain autoencoder?"* — The VAE encoder is probabilistic (`N(μ,σ²)`) and the KL term ties the aggregate posterior to a known prior, so you can *sample* the prior to generate. A plain AE has an arbitrary latent layout with holes — great for compression, useless for sampling.
- *"VAE vs. GAN vs. diffusion?"* — VAEs give a principled likelihood bound and a clean latent space but blurry samples. GANs sharpen samples but have no likelihood and can mode-collapse. Diffusion models give the sharpest samples today but are slower to sample. VAEs still show up as components (e.g. the latent space in latent diffusion).
- *"What is posterior collapse and how do you fix it?"* — When `q(z|x)` snaps to the prior and the decoder ignores `z`. Fixes: KL annealing / β-VAE (warm up the KL weight), weaker decoders, free-bits, or skip connections from `z`.
- *"Why is the objective called a lower *bound*?"* — `log p(x) = ELBO + KL(q(z|x) ‖ p(z|x))`, and KL ≥ 0, so the ELBO ≤ `log p(x)`. The gap is the KL between the approximate and true posterior; a richer `q` (e.g. normalizing flows) tightens it.
- *"How would you handle binary data (e.g. MNIST pixels)?"* — Use a Bernoulli decoder and **binary cross-entropy** for the reconstruction term instead of MSE; the KL term is unchanged.

---

## TL;DR cheat sheet
| Thing | Answer |
|---|---|
| Core idea | Probabilistic encoder + KL-to-prior ⇒ a smooth, sample-able latent space you can generate from |
| Objective | Maximize the ELBO = `E[log p(x\|z)] − KL(q(z\|x) ‖ p(z))` (reconstruction − KL) |
| Reparameterization | `z = μ + σ⊙ε`, `ε ~ N(0,I)` — makes sampling differentiable in `μ`, `logvar` |
| Why predict `logvar` | Keeps `σ>0` for free + numerically stable (`σ = exp(0.5·logvar)`) |
| Closed-form KL | `−0.5·Σ(1 + logvar − μ² − exp(logvar))` vs. `N(0,I)` |
| Reconstruction term | MSE (Gaussian decoder) or BCE (Bernoulli decoder for binary data) |
| Generate | Sample `z ~ N(0,I)`, decode |
| #1 bug | Sampling `z` without reparameterization (no gradient to `μ`/`σ`) |
| Limitation | Blurry samples; posterior collapse; the bound has a gap |
