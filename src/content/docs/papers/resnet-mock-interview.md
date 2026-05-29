---
title: ResNet — Paper-to-Code Mock Interview
description: A combined mock (read paper, explain benefit, implement in Colab) using ResNet / deep residual learning as the worked example.
sidebar:
  order: 3
  label: ResNet
---

> **Paper:** *Deep Residual Learning for Image Recognition* — He et al., 2015. [arXiv:1512.03385](https://arxiv.org/abs/1512.03385)
>
> **Format:** Read (~15 min) → explain the *real* benefit → implement the core idea in Colab → sanity-check it.
>
> **Companion notebook:** [`resnet_mock.ipynb`](/notebooks/resnet_mock.ipynb) (download) — a deep plain-vs-residual training demo + a `ResidualBlock` stub to fill in, plus verification cells. Open in [Google Colab](https://colab.research.google.com/) via *File → Upload notebook*.
>
> **Difficulty:** 🟢🟡 Warm-up to medium. The block is one line; the subtlety is *why* the skip fixes the degradation problem.

---

## How to run this as a timed drill (~40 min)

| Time | Block | What you produce |
|------|-------|------------------|
| 0:00–0:12 | **Read** (use the [three-pass method](/papers/lora-mock-interview/#part-0--how-to-read-a-paper-in-15-minutes-the-three-pass-method)) | The degradation problem + the residual reformulation |
| 0:12–0:17 | **Explain the benefit** out loud (cover Part 2) | Identity skip = gradient highway, easy-to-optimize residual |
| 0:17–0:33 | **Implement** from the stub (Part 3) | A working `ResidualBlock` + a deep net that trains where the plain one stalls |
| last 5 min | **Sanity-check** (Part 4) | All checks passing, narrated out loud |

### Self-grading rubric — "what good looks like"
- ✅ Framed the problem as **degradation**, not overfitting: deeper *plain* nets train to **worse training error**, so it's an optimization problem.
- ✅ Explained the skip as making the layer learn a **residual** `F(x)` on top of identity, which is easier to optimize (pushing `F→0` recovers identity).
- ✅ Knew the skip gives gradients a **highway**: `∂out/∂x = I + ∂F/∂x`, so the `I` term keeps gradients from vanishing through depth.
- ✅ Demonstrated the benefit with a **plain-vs-residual** gap at the same depth, not just "it runs."
- ⚠️ Red flags: calling it "regularization," saying it "adds capacity," or claiming it prevents overfitting (it fixes *trainability*); forgetting the skip needs matching dimensions.

---

## Part 1 — Structured read of THIS paper

### The 30-second summary (the "benefit")
Stacking more layers on a *plain* network eventually makes it **harder to train** — training error goes **up**, not down. That's the **degradation problem**, and it's not overfitting (the training set itself is fit worse). ResNet adds an **identity skip connection** around every couple of layers so each block computes `h + F(h)` instead of `F(h)`. The payoff:

- **Very deep nets become trainable** — the paper trains networks far deeper than previously practical (it reports roughly 152 layers on ImageNet, and even ~1000-layer variants on CIFAR).
- **The residual is easy to optimize:** if the best a block can do is "pass through," it just drives `F→0` instead of learning an identity map from scratch.
- **Gradients get a highway:** the `+h` term puts an identity in the backward path, so the signal reaching early layers doesn't vanish through depth.

### The core idea (Method — you implement this)
A plain block learns a target mapping `H(h)` directly. A residual block instead learns the **residual** `F(h) = H(h) − h`, and adds the input back:

$$y = h + F(h), \qquad F(h) = W_2\,\sigma(W_1 h)$$

The intuition is that if the optimal mapping is close to the identity, it is far easier to push the residual `F` toward zero than to fit an identity with a stack of nonlinear layers. The backward pass is where the magic is visible — differentiate the block:

$$\frac{\partial y}{\partial h} = I + \frac{\partial F}{\partial h}$$

The `I` term means gradient flows to `h` **unattenuated** regardless of what `F` does. Across a deep stack the product of Jacobians keeps a "+1 path," so early-layer gradients don't collapse to zero.

Key details (the things an interviewer probes):
- **Dimensions must match to add.** `h + F(h)` requires `F(h)` to have the same shape as `h`. When a block changes width/resolution, the paper uses a **projection shortcut** (a 1×1 conv / linear) on the skip; otherwise the skip is a plain identity (zero extra parameters).
- **Pre/post placement of the addition.** The original adds *before* the final activation (`σ(h + F(h))`); the later "pre-activation" variant moves norm/activation inside `F` for an even cleaner identity path. Either is fine to mention.
- **It's not regularization.** The benefit is **optimization**: same or lower *training* error at much greater depth. Generalization improves as a downstream consequence of being able to train deeper, better models.
- **Pairs with BatchNorm** in the real architecture, but the skip itself is what fixes degradation — the toy below shows the effect *without* BN.

### Where the evidence lives (tables/figures that matter)
- **Plain-vs-ResNet training-error curves (their Fig. ~1 / ~4):** the 34-layer *plain* net has **higher training error** than the 18-layer plain net — that's degradation. The ResNet versions reverse it: deeper is better.
- **ImageNet results table:** the deep ResNet (~152 layers) reports roughly **3.57% top-5 error** (ensemble), winning ILSVRC 2015 — phrase these as approximate.
- **CIFAR depth sweep:** they push to ~110 and even ~1000+ layers, showing the skip keeps very deep nets optimizable.

### The honest limitations (have an opinion)
- **Doesn't add representational capacity for free** — it makes existing capacity *trainable*. A shallow net that already trains fine won't magically improve.
- **Very deep ResNets show diminishing / unstable returns** (the ~1000-layer CIFAR net overfits and is finicky) — depth alone isn't a free lunch.
- **Skip plumbing has edge cases:** dimension changes need projection shortcuts; placement relative to BN/activation matters and got refined in follow-up work.

---

## Part 2 — The interview dialogue (interviewer ⇄ interviewee)

> **🧑‍💼 Interviewer:** One paragraph — what does a residual connection actually buy me?
>
> **🧑‍💻 Interviewee:** It fixes the *degradation* problem: past a certain depth, plain networks get *harder to optimize* and their training error goes up — not because they overfit, but because the optimizer struggles to fit even the training set. By making each block compute `h + F(h)`, the block only has to learn a residual on top of identity. If "pass through" is best, it drives `F→0`, which is trivial. And in the backward pass the `+h` adds an identity term to the Jacobian, so gradients reach early layers without vanishing. The net effect is that very deep nets become trainable.

> **🧑‍💼 Interviewer:** Be precise — *why* does the skip help gradients?
>
> **🧑‍💻 Interviewee:** Differentiate the block: `∂(h + F(h))/∂h = I + ∂F/∂h`. The identity `I` is a path with derivative exactly 1, so even if `∂F/∂h` is tiny, gradient still flows through the `I` term. Stack many blocks and the backward product always retains a "+1" highway, instead of multiplying many small Jacobians together and vanishing. That's the mechanism, and it's why a 24-block plain net can have a near-zero first-layer gradient while the residual version doesn't.

> **🧑‍💼 Interviewer:** Isn't this just regularization, like dropout?
>
> **🧑‍💻 Interviewee:** No — opposite category. Dropout *raises* training error to reduce overfitting; residual connections *lower* training error by making optimization easier. The headline evidence is on the **training set**: deeper plain nets train worse, ResNets don't. Better generalization is a downstream bonus of being able to train deeper models, not the direct mechanism.

> **🧑‍💼 Interviewer:** What breaks when a block changes the feature dimension?
>
> **🧑‍💻 Interviewee:** You can't add `h + F(h)` if shapes differ. The fix is a **projection shortcut**: put a 1×1 conv (or a linear layer) on the skip to map `h` to the new shape, optionally with a stride to match resolution. When dims already match, the identity skip is parameter-free, which is the common case and what keeps ResNets cheap.

> **🧑‍💼 Interviewer:** Implement it and show a deep plain net stall where the residual one trains.

---

## Part 3 — Implementation

The whole method is one line in the forward pass: add the input back. We build a `ResidualBlock` (`h + F(h)`), a `PlainBlock` (same params, no skip), and a deep stack of either — so we can race them at identical depth.

> **Scope note:** this toy demonstrates the **optimization / degradation** benefit — the *correctness of the mechanism* on a tiny deep MLP. It does **not** reproduce ImageNet numbers. For real vision use, reach for `torchvision.models.resnet18/50/...`; there's no single clean built-in "residual layer" because the skip wraps a sub-block.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """y = h + F(h), with F = Linear -> activation -> Linear (same width)."""

    def __init__(self, dim, act=nn.Tanh):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.act = act()
        self.fc2 = nn.Linear(dim, dim)

    def forward(self, h):
        return h + self.fc2(self.act(self.fc1(h)))   # identity skip + residual


class PlainBlock(nn.Module):
    """Same params as ResidualBlock but NO skip: y = F(h)."""

    def __init__(self, dim, act=nn.Tanh):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.act = act()
        self.fc2 = nn.Linear(dim, dim)

    def forward(self, h):
        return self.fc2(self.act(self.fc1(h)))


class DeepNet(nn.Module):
    """Stem -> n_blocks of (Residual|Plain) -> head. NO BatchNorm (degradation is the point)."""

    def __init__(self, in_dim, dim, n_blocks, out_dim, residual=True, act=nn.Tanh):
        super().__init__()
        self.stem = nn.Linear(in_dim, dim)
        Block = ResidualBlock if residual else PlainBlock
        self.blocks = nn.ModuleList([Block(dim, act) for _ in range(n_blocks)])
        self.head = nn.Linear(dim, out_dim)

    def forward(self, x):
        h = torch.tanh(self.stem(x))
        for b in self.blocks:
            h = b(h)
        return self.head(h)
```

### Why each line matters (talk through it)
- `return h + self.fc2(...)` — **this is the whole paper.** The `h +` is the identity skip; everything to its right is the residual `F(h)`. Drop the `h +` and you have `PlainBlock`.
- `F = Linear → act → Linear` keeping width constant — so `F(h)` has the **same shape as `h`** and the add is legal with a parameter-free skip (no projection needed here).
- `PlainBlock` has **identical parameter count** to `ResidualBlock` — the only difference is the skip, so any gap in the demo is attributable to the skip, not to capacity.
- **No BatchNorm anywhere** — BN would also help trainability and muddy the lesson. We isolate the skip's effect, using saturating `tanh` to make the plain net's gradients vanish.

### Demonstrating the benefit (degradation toy task)
A **deep (24 blocks), narrow** net with saturating `tanh` and no normalization is exactly the regime where plain nets degrade. We train plain vs residual on the same smooth regression target and also read the **first-layer gradient norm** to expose the vanishing-gradient mechanism.

```python
def make_data(seed=0, n=512, in_dim=8):
    g = torch.Generator().manual_seed(seed)
    X = torch.randn(n, in_dim, generator=g)
    w = torch.randn(in_dim, 1, generator=g)
    y = torch.sin(X @ w) + 0.1 * (X[:, :1] ** 2)   # smooth nonlinear target
    return X, y

def train(residual, depth=24, dim=16, steps=400, seed=0):
    torch.manual_seed(seed)
    X, y = make_data(seed=0)
    net = DeepNet(8, dim, depth, 1, residual=residual, act=nn.Tanh)
    opt = torch.optim.Adam(net.parameters(), lr=3e-3)
    first_grad = None
    for step in range(steps):
        loss = F.mse_loss(net(X), y)
        opt.zero_grad(); loss.backward()
        if step == 0:
            first_grad = net.stem.weight.grad.norm().item()  # grad reaching layer 1
        opt.step()
    return F.mse_loss(net(X), y).item(), first_grad

plain_loss, plain_g = train(residual=False)
res_loss,   res_g   = train(residual=True)
print(f"PLAIN    24-deep:  final loss {plain_loss:.4f}   first-layer grad {plain_g:.2e}")
print(f"RESIDUAL 24-deep:  final loss {res_loss:.4f}   first-layer grad {res_g:.2e}")
```

Verified output (seed-fixed):

```
PLAIN    24-deep:  final loss 0.5126   first-layer grad 2.35e-13
RESIDUAL 24-deep:  final loss 0.0013   first-layer grad 2.05e+00
```

The plain net **stalls** at ~0.51 loss with a first-layer gradient of ~`2e-13` (vanished — the optimizer can't update early layers). The residual net trains to ~0.0013 (≈390× lower) with a **healthy ~2.0** first-layer gradient. Same depth, same parameter count — the only difference is the skip. (Exact numbers are seed-dependent; the *direction* — plain stalls, residual trains, residual gradient larger — is the point, and it's asserted in Part 4.)

---

## Part 4 — Sanity checks (don't skip)

### Check 1 — Block output shape == input shape
```python
blk = ResidualBlock(16)
h = torch.randn(4, 16)
out = blk(h)
assert out.shape == h.shape
print("OK: shape preserved", tuple(out.shape))
```

### Check 2 — The residual path is additive (zero F ⇒ identity)
```python
blk = ResidualBlock(16)
with torch.no_grad():
    blk.fc2.weight.zero_(); blk.fc2.bias.zero_()   # force F(h) = 0
h = torch.randn(4, 16)
assert torch.allclose(blk(h), h, atol=1e-6)
print("OK: zeroing F's last layer -> block == identity")
```

### Check 3 — Gradient flows to the first layer
```python
net = DeepNet(8, 16, 24, 1, residual=True)
F.mse_loss(net(torch.randn(8, 8)), torch.randn(8, 1)).backward()
g = net.stem.weight.grad
assert g is not None and g.norm().item() > 0
print("OK: stem grad norm =", f"{g.norm().item():.3e}")
```

### Check 4 — First-layer grad norm is LARGER with residual than plain (the mechanism)
```python
torch.manual_seed(0); netp = DeepNet(8, 16, 24, 1, residual=False)
torch.manual_seed(0); netr = DeepNet(8, 16, 24, 1, residual=True)
X, y = make_data(seed=0)
F.mse_loss(netp(X), y).backward()
F.mse_loss(netr(X), y).backward()
gp, gr = netp.stem.weight.grad.norm().item(), netr.stem.weight.grad.norm().item()
assert gr > gp
print(f"OK: residual grad {gr:.2e} > plain grad {gp:.2e}")
```

### Check 5 — Plain deep net final loss > residual final loss (the demonstration)
```python
plain_loss, _ = train(residual=False)
res_loss,   _ = train(residual=True)
assert plain_loss > res_loss
print(f"OK: plain {plain_loss:.4f} > residual {res_loss:.4f}")
```

### Check 6 — Deterministic & finite in eval (no stochastic layers, no NaNs)
```python
net = DeepNet(8, 16, 24, 1, residual=True).eval()
x = torch.randn(4, 8)
with torch.no_grad():
    a, b = net(x), net(x)
assert torch.equal(a, b) and torch.isfinite(a).all()
print("OK: deterministic & finite outputs")
```

---

## Part 5 — Likely follow-up questions

- *"Degradation vs overfitting — what's the difference?"* — Overfitting = low train error, high test error. Degradation = **higher train error** as you add depth. The paper's whole point is the second one, which is an *optimization* failure, and the skip is the fix.
- *"What if a block changes width or downsamples?"* — Use a **projection shortcut**: a 1×1 conv (or linear) on the skip to match shape, optionally strided. When shapes match, the skip is identity and free.
- *"Original vs pre-activation ResNet?"* — Original: `σ(h + F(h))` (add before the activation). Pre-activation moves BN+activation *inside* `F`, leaving a clean identity path `h + F(h)`; it trains very deep nets slightly better.
- *"How does this relate to LSTMs / highway networks?"* — Same family: additive/gated skip paths to preserve signal through depth/time. Highway networks gate the skip; ResNet's skip is an ungated identity, which turned out to be simpler and stronger.
- *"Do Transformers use this?"* — Yes — every sub-layer is `x + Sublayer(x)` (residual + LayerNorm). The residual stream is the same idea applied to depth in attention stacks.

---

## TL;DR cheat sheet
| Thing | Answer |
|---|---|
| Problem solved | **Degradation** — deeper *plain* nets train *worse* (an optimization, not overfitting, problem) |
| Core idea | Learn a residual: block computes `y = h + F(h)` instead of `F(h)` |
| Formula | `y = h + F(h)`, `F(h) = W₂·σ(W₁h)`; backward: `∂y/∂h = I + ∂F/∂h` |
| Why it works | `F→0` recovers identity (easy to optimize); `+I` term is a gradient highway |
| Dimension change | Use a **projection shortcut** (1×1 conv / linear) on the skip |
| Benefit category | **Optimization / trainability**, NOT regularization |
| Paper scale (approx) | ~152 layers on ImageNet, ~3.57% top-5; even ~1000-layer CIFAR variants |
| This toy shows | The mechanism (plain stalls, residual trains; grad highway) — *not* ImageNet numbers |
| Real-world use | `torchvision.models.resnet18/50/...` |
| Limitation | Adds trainability, not free capacity; very deep variants overfit/unstable |
