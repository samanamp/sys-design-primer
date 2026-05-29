---
title: SimCLR (NT-Xent) — Paper-to-Code Mock Interview
description: A combined mock (read paper, explain benefit, implement in Colab) on SimCLR's contrastive NT-Xent / InfoNCE loss — learning representations with no labels.
sidebar:
  order: 14
  label: SimCLR
---

> **Paper:** *A Simple Framework for Contrastive Learning of Visual Representations (SimCLR)* — Chen et al., 2020. arXiv: [2002.05709](https://arxiv.org/abs/2002.05709)
>
> **Format:** Read the paper (~15 min) → explain the *real* benefit → implement the core idea in Colab → sanity-check it.
>
> **Companion notebook:** [`simclr_mock.ipynb`](/notebooks/simclr_mock.ipynb) (download) — a no-labels contrastive toy task + an `nt_xent_loss` stub to fill in, plus verification cells. Open it straight in [Google Colab](https://colab.research.google.com/) via *File → Upload notebook*. A reference solution is included at the bottom of this page.
>
> **Difficulty:** 🟡🔴 Medium-hard. The loss has several easy-to-botch details (normalization, self-masking, where the positive is).

---

## How to run this as a timed drill (~60 min)

Treat this like the real thing. Set a timer and don't look at the answers below until each block is done.

| Time | Block | What you produce |
|------|-------|------------------|
| 0:00–0:15 | **Read** (use the [three-pass method](/papers/lora-mock-interview/#part-0--how-to-read-a-paper-in-15-minutes-the-three-pass-method)) | The NT-Xent equation + the "no labels" claim + the augmentation/batch-size findings |
| 0:15–0:20 | **Explain the benefit** out loud (cover Part 2 without peeking) | 1-paragraph pitch + answers to "what's a positive", "why normalize", "why temperature" |
| 0:20–0:50 | **Implement** in Colab from the stub (Part 3) | A working `nt_xent_loss` + same-id similarity rising above different-id on the toy task |
| last 10 min | **Sanity-check** (Part 4) | All 6 checks passing, talked through out loud |

### Self-grading rubric — "what good looks like"
- ✅ Stated the core claim precisely: **representations learned with NO labels** by pulling positives together and pushing negatives apart.
- ✅ Got the **three loss details** right without prompting: **L2-normalize**, **mask the diagonal** (self-similarity), and the positive is the *other augmentation of the same image*.
- ✅ Knew **temperature** sharpens/softens the softmax and that it matters a lot empirically.
- ✅ Demonstrated the benefit with a **same-id vs different-id similarity gap that widens after training**, not just "it runs."
- ✅ Mentioned at least one real finding: **large batch / many negatives**, **strong augmentation**, or **the projection head helps but is thrown away**.
- ⚠️ Red flags: forgetting to normalize, not masking self-similarity, computing the loss on the backbone instead of the projection head, claiming it needs labels.

---

## Part 1 — Structured read of THIS paper

### The 30-second summary (the "benefit")
You usually need labels to learn good features. SimCLR shows you can learn **strong visual representations with no labels at all** using a contrastive objective. Take an image, make **two random augmentations** of it — that's a **positive pair** (same underlying content). Every *other* image in the batch is a **negative**. Train an encoder so that the two views of the same image land **close** in embedding space while all the other views are pushed **apart**. The payoff:

- **No labels required.** A simple framework — just augmentation + a contrastive loss — rivals supervised pretraining when you later fine-tune on a small labeled set.
- **It's "simple"** in the sense that it drops the memory banks / specialized architectures of earlier contrastive methods; the negatives are just *the rest of the batch*.
- The learned features **transfer**: a linear classifier on top of the frozen encoder gets competitive ImageNet accuracy (the headline result).

### The core idea (Method — you implement this)
For a minibatch of `N` images, create `2N` views (two augmentations each). Pass each through encoder `f` then a small projection head `g` to get embeddings, and **L2-normalize** them so dot products are **cosine similarities**. Define a temperature-scaled similarity and apply cross-entropy where each view's *positive* is its paired augmentation. For a positive pair `(i, j)`:

$$\ell_{i,j} = -\log \frac{\exp\!\big(\operatorname{sim}(z_i, z_j)/\tau\big)}{\sum_{k=1}^{2N}\mathbb{1}_{[k \neq i]}\,\exp\!\big(\operatorname{sim}(z_i, z_k)/\tau\big)}$$

where $\operatorname{sim}(u, v) = \dfrac{u^\top v}{\lVert u\rVert\,\lVert v\rVert}$ is cosine similarity and $\tau$ is the temperature. The NT-Xent loss is this averaged over all positive pairs. In code it's exactly a **cross-entropy over the `2N×2N` similarity matrix** (diagonal masked out) where the label for row `i` is the index of its paired view.

Key details (the things an interviewer probes):
- **What is a "positive"?** The two augmentations of the *same* image. Everything else in the batch is a negative — no labels involved.
- **Why L2-normalize?** So the dot product is **cosine similarity**, comparing *direction* not magnitude. Without it the loss can cheat by inflating norms.
- **Why mask the diagonal?** Row `i` includes `z_i` itself; `sim(z_i, z_i)=1/\tau` would dominate and trivially "win." You must exclude the self-pair (set it to `-inf`) so the model is forced to match its *partner* view.
- **Temperature `τ`.** Lower `τ` → sharper softmax → harder penalty on the closest negatives ("hard negatives" matter more). Too low destabilizes; the paper tunes it (≈0.1–0.5).
- **Projection head `g`.** The loss is on `g(f(x))`, but you **keep `f` and throw away `g`** for downstream tasks — the representation before the head transfers better.

### Where the evidence lives (tables/figures that matter)
- **Linear-eval ImageNet table:** top-1 accuracy of a linear classifier on frozen features → the core "good representations, no labels" claim.
- **Augmentation ablation (the composition grid):** *random crop + color distortion* together are what make it work → augmentation strength is central, not incidental.
- **Batch size / training length figure:** bigger batches (more negatives) and longer training help → contrastive learning is negative-hungry.
- **Projection-head ablation:** features *before* the nonlinear head outperform features after it → keep `f`, drop `g`.

### The honest limitations (have an opinion)
- **Hungry for negatives / compute:** results lean on **very large batches** (lots of in-batch negatives) and long training — expensive. (MoCo's queue and BYOL/SimSiam's no-negatives designs were direct responses.)
- **Augmentation-dependent:** the "right" augmentations are domain-specific; the crop+color recipe is tuned for natural images and may not transfer to other modalities.
- **Temperature is sensitive:** performance depends nontrivially on `τ` and the augmentation pipeline — more knobs than the "simple" name suggests.

---

## Part 2 — The interview dialogue (interviewer ⇄ interviewee)

> **🧑‍💼 Interviewer:** One paragraph — what does SimCLR actually buy me?
>
> **🧑‍💻 Interviewee:** Representations without labels. I take each image, make two random augmentations — that's a positive pair with the same content — and treat every other image in the batch as a negative. I train the encoder so the two views of one image are close in embedding space and far from everything else, using the NT-Xent contrastive loss. After this self-supervised phase, a simple linear classifier on the frozen features is competitive with supervised pretraining. The cost is it's negative-hungry: it wants big batches and strong augmentation.

> **🧑‍💼 Interviewer:** Walk me through the NT-Xent loss as code.
>
> **🧑‍💻 Interviewee:** Stack the `2N` views into one matrix, **L2-normalize** so dot products are cosine similarities, and form the `2N×2N` similarity matrix divided by temperature. **Mask the diagonal to `-inf`** so a view can't match itself. Then it's just cross-entropy: for row `i`, the target column is its paired augmentation — index `i+N` for the first half, `i-N` for the second. That single cross-entropy *is* the loss.

> **🧑‍💼 Interviewer:** Why normalize and why the temperature?
>
> **🧑‍💻 Interviewee:** Normalizing makes the comparison about **direction**, i.e. cosine similarity — otherwise the model can lower the loss by just scaling embedding norms, which isn't learning structure. Temperature scales the logits before the softmax: a **lower `τ` sharpens** the distribution so the loss focuses hard on the most similar negatives, while a higher `τ` is softer. It's a real knob — too low and training gets unstable.

> **🧑‍💼 Interviewer:** What's the role of the projection head, and why throw it away?
>
> **🧑‍💻 Interviewee:** The loss is computed on `g(f(x))` where `g` is a small MLP head. Empirically the features **before** the head — the output of `f` — transfer better to downstream tasks, so you keep `f` and discard `g`. The intuition is the head can afford to discard information useful for downstream tasks in order to be invariant to the augmentations the loss rewards.

> **🧑‍💼 Interviewer:** When would this struggle, and what fixed it?
>
> **🧑‍💻 Interviewee:** It leans on lots of in-batch negatives, so small-batch setups underperform — MoCo addressed that with a momentum-updated queue of negatives. It's also sensitive to the augmentation recipe and temperature. Later methods like BYOL and SimSiam showed you can even drop negatives entirely with the right architecture, so the negatives aren't strictly required.

> **🧑‍💼 Interviewer:** Implement the loss and show same-image views getting closer than different images.

---

## Part 3 — Implementation

The whole method is: build `2N` views, normalize, similarity matrix over temperature, mask the diagonal, cross-entropy to the paired view.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


def nt_xent_loss(z1, z2, temperature=0.5):
    """NT-Xent / InfoNCE over a batch of N pairs.

    z1, z2: (N, D) projection-head outputs for the two augmented views.
    Positives are (row i of z1) <-> (row i of z2). Returns a scalar loss.
    """
    N = z1.shape[0]
    z = torch.cat([z1, z2], dim=0)                 # (2N, D): stack both views
    z = F.normalize(z, dim=1)                       # L2-normalize -> dot = cosine sim

    sim = z @ z.t() / temperature                  # (2N, 2N) cosine-sim matrix / T

    # mask out self-similarity (the diagonal) so a view can't "match" itself
    self_mask = torch.eye(2 * N, dtype=torch.bool, device=z.device)
    sim = sim.masked_fill(self_mask, float("-inf"))

    # for row i, the positive is its paired augmentation:
    #   view i (0..N-1) pairs with i+N ; view i+N pairs with i-N
    targets = torch.cat([torch.arange(N, 2 * N), torch.arange(0, N)]).to(z.device)

    return F.cross_entropy(sim, targets)


class Encoder(nn.Module):
    """Small MLP encoder f(.) + projection head g(.). Loss is on g(f(x))."""

    def __init__(self, in_dim, hid=64, rep_dim=32, proj_dim=16):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(in_dim, hid), nn.ReLU(),
            nn.Linear(hid, rep_dim),
        )
        self.proj = nn.Sequential(
            nn.Linear(rep_dim, proj_dim), nn.ReLU(),
            nn.Linear(proj_dim, proj_dim),
        )

    def representation(self, x):
        return self.backbone(x)                    # h = f(x): kept for downstream

    def forward(self, x):
        return self.proj(self.backbone(x))         # z = g(f(x)): used for the loss
```

### Why each line matters (talk through it)
- `torch.cat([z1, z2])` — the loss operates on **all `2N` views at once**; every view is a negative for every other (except its positive).
- `F.normalize(z, dim=1)` — turns the dot product into **cosine similarity**; skip it and the model can cheat with magnitudes.
- `z @ z.t() / temperature` — the full pairwise similarity matrix, scaled by `τ` *before* the softmax inside cross-entropy.
- `masked_fill(self_mask, -inf)` — kills the diagonal so `sim(z_i, z_i)` (always the largest) can't be selected; the model must find its *partner*.
- `targets = cat([arange(N,2N), arange(0,N)])` — encodes "view `i`'s positive is view `i±N`"; `F.cross_entropy` then does the InfoNCE log-softmax for free.
- `representation` vs `forward` — downstream you use `f(x)` (`representation`) and discard the projection head `g`.

### Demonstrating the benefit (no-labels contrastive toy task)
We make `K` latent "identities" (random anchor vectors). An **augmentation** = anchor + small noise + random scale + random feature mask, giving two correlated views — a positive pair. We train the encoder with NT-Xent and **never use the identity labels**. Then we check whether embeddings of the *same* identity ended up closer than *different* identities.

```python
def augment(anchors, idx, noise=0.15):
    """One augmented view of the identities given by idx."""
    a = anchors[idx]
    a = a + noise * torch.randn_like(a)            # additive noise
    scale = 0.8 + 0.4 * torch.rand(a.shape[0], 1)  # random per-sample scale
    mask = (torch.rand_like(a) > 0.1).float()      # random feature dropout (10%)
    return a * scale * mask


def mean_same_diff_sim(model, anchors, n_per=20):
    """Mean cosine sim of same-identity vs different-identity representations."""
    model.eval()
    K = anchors.shape[0]
    with torch.no_grad():
        ids = torch.arange(K).repeat_interleave(n_per)
        h = F.normalize(model.representation(augment(anchors, ids)), dim=1)
        S = h @ h.t()
        same = ids.unsqueeze(0) == ids.unsqueeze(1)
        off = ~torch.eye(len(ids), dtype=torch.bool)
        return S[same & off].mean().item(), S[~same].mean().item()


torch.manual_seed(0)
in_dim, K = 24, 8
common = torch.randn(in_dim)
anchors = 0.6 * torch.randn(K, in_dim) + common    # identities overlap -> learning is visible

model = Encoder(in_dim)
s0, d0 = mean_same_diff_sim(model, anchors)          # BEFORE training

opt = torch.optim.Adam(model.parameters(), lr=2e-3)
model.train()
for step in range(800):
    idx = torch.randint(0, K, (64,))               # random identities, NO labels in loss
    z1, z2 = model(augment(anchors, idx)), model(augment(anchors, idx))
    loss = nt_xent_loss(z1, z2, temperature=0.5)
    opt.zero_grad(); loss.backward(); opt.step()

s1, d1 = mean_same_diff_sim(model, anchors)          # AFTER training
print(f"BEFORE  same {s0:+.3f}  diff {d0:+.3f}  gap {s0-d0:+.3f}")
print(f"AFTER   same {s1:+.3f}  diff {d1:+.3f}  gap {s1-d1:+.3f}")
```

Verified output (seed 0):

```
BEFORE  same +0.855  diff +0.672  gap +0.183
AFTER   same +0.962  diff +0.108  gap +0.854
```

Before training everything looks alike (different identities sit at +0.67 cosine sim). After training with **no labels**, same-identity views are nearly aligned (+0.96) while different identities are pushed apart (+0.11) — the gap widens from 0.18 to 0.85. The contrastive objective discovered the identity structure on its own. This is the core idea at toy scale; the paper's ImageNet linear-eval numbers are a different universe of scale and tuning, but the mechanism is the same.

---

## Part 4 — Sanity checks (don't skip)

### Check 1 — Embeddings are L2-normalized (unit norm)
```python
z = torch.randn(10, 16)
zn = F.normalize(z, dim=1)
norms = zn.norm(dim=1)
print("norms:", norms[:3].tolist())
assert torch.allclose(norms, torch.ones(10), atol=1e-5)
```

### Check 2 — The loss correctly identifies positives (perfect pairs → loss ≈ 0)
```python
N, D = 8, 16
base = torch.eye(N, D)                  # orthonormal => negatives are dissimilar
aligned = nt_xent_loss(base.clone(), base.clone(), temperature=0.1).item()
rand_l  = nt_xent_loss(torch.randn(N, D), torch.randn(N, D), temperature=0.1).item()
print(f"perfect positives {aligned:.4f}  <<  random {rand_l:.4f}")
assert aligned < 0.1 and aligned < rand_l
```

### Check 3 — Similarity matrix excludes self-pairs (diagonal masked)
```python
N, D = 8, 16
z = F.normalize(torch.randn(2 * N, D), dim=1)
sim = z @ z.t() / 0.5
sim = sim.masked_fill(torch.eye(2 * N, dtype=torch.bool), float("-inf"))
print("diagonal:", torch.diagonal(sim)[:3].tolist())
assert torch.isinf(torch.diagonal(sim)).all()
```

### Check 4 — Temperature has the expected effect (lower T → sharper)
```python
logits = torch.tensor([1.0, 0.5, 0.2, -0.3])
p_low  = F.softmax(logits / 0.1, dim=0).max().item()
p_high = F.softmax(logits / 1.0, dim=0).max().item()
print(f"max prob: T=0.1 -> {p_low:.3f}   T=1.0 -> {p_high:.3f}")
assert p_low > p_high
```

### Check 5 — After training, same-identity sim > different-identity (the demonstration)
```python
# uses the trained `model` and `anchors` from Part 3
s1, d1 = mean_same_diff_sim(model, anchors)
print(f"same {s1:+.3f}  >  diff {d1:+.3f}")
assert s1 > d1 + 0.2
```

### Check 6 — Gradient flows to the encoder backbone
```python
m = Encoder(24)
idx = torch.arange(4)
anc = torch.randn(8, 24)
loss = nt_xent_loss(m(augment(anc, idx)), m(augment(anc, idx)), 0.5)
loss.backward()
g = m.backbone[0].weight.grad
print("backbone grad-norm:", g.norm().item())
assert g is not None and g.abs().sum().item() > 0
```

---

## Part 5 — Likely follow-up questions

- *"Where do the negatives come from?"* — Just the **other examples in the batch** (all `2N−2` non-partner views). No memory bank in vanilla SimCLR; that's the "simple." MoCo adds a queue so you don't need a giant batch.
- *"Why is a bigger batch better?"* — More in-batch negatives = a harder, more informative contrastive task. The paper shows accuracy climbing with batch size and training length.
- *"NT-Xent vs InfoNCE vs triplet loss?"* — NT-Xent *is* InfoNCE with cosine similarity + temperature, using all in-batch negatives at once. Triplet uses one negative per anchor and a margin; InfoNCE-style losses use many negatives via a softmax, which is generally stronger.
- *"Why keep `f` and drop the projection head `g`?"* — Features before the nonlinear head transfer better; the head learns to be invariant to augmentations, discarding info useful downstream (e.g. color), so you don't want it at eval time.
- *"What about collapse — can't everything map to one point?"* — The **negatives** prevent it: pushing different images apart stops the trivial all-vectors-equal solution. Negative-free methods (BYOL/SimSiam) need other tricks (stop-gradient, predictor, momentum encoder) to avoid collapse.
- *"How do you evaluate self-supervised features?"* — **Linear probe**: freeze the encoder, train only a linear classifier on labels, report accuracy. Also kNN on embeddings and fine-tuning with limited labels.

---

## TL;DR cheat sheet
| Thing | Answer |
|---|---|
| Core idea | Pull two augmentations of one image together, push all other images apart — no labels |
| Loss | NT-Xent / InfoNCE: cross-entropy over the `2N×2N` cosine-sim matrix / temperature |
| Positive | The other augmentation of the same image (index `i±N`) |
| Negatives | Every other view in the batch (`2N−2` of them) |
| Must-do details | L2-normalize, mask the diagonal (self-sim), loss on `g(f(x))` |
| Temperature | Lower `τ` → sharper softmax → focuses on hard negatives |
| Projection head | Train on `g(f(x))`, **keep `f`, discard `g`** downstream |
| Benefit | Strong transferable representations from unlabeled data |
| Limitation | Negative-hungry (big batches), augmentation- and temperature-sensitive |
