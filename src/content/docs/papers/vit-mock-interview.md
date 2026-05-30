---
title: Vision Transformer (ViT) — Paper-to-Code Mock Interview
description: A combined mock (read paper, explain benefit, implement in Colab) using the Vision Transformer as the worked example — images as sequences of patches, no convolutions.
sidebar:
  order: 13
  label: ViT
---

> **Paper:** *An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale* (ViT) — Dosovitskiy et al., 2020. arXiv: [2010.11929](https://arxiv.org/abs/2010.11929)
>
> **Format:** Read (~15 min) → explain the *real* benefit → implement the core idea in Colab → sanity-check it.
>
> **Companion notebook:** [`vit_mock.ipynb`](/notebooks/vit_mock.ipynb) (download) — a quadrant-classification demo + a `ViT` stub to fill in, plus verification cells. Open in [Google Colab](https://colab.research.google.com/) via *File → Upload notebook*.
>
> **Difficulty:** 🟡 Medium. Builds directly on the [Attention mock](/papers/attention-mock-interview/) — ViT *reuses* multi-head self-attention. Scope to the patch-embedding + encoder pipeline, not a full pretraining run.

---

## How to run this as a timed drill (~65 min)

> ⚠️ **Scoping move (do this out loud first):** ViT is "apply a standard Transformer encoder to image patches." Tell the interviewer you'll implement the **patch embedding → CLS token → positional embeddings → a small encoder stack → classification head**, reusing multi-head attention. You will *not* reproduce the data-scaling result (that needs JFT-300M); you'll show the pipeline learns a visual rule on a toy problem.

| Time | Block | What you produce |
|------|-------|------------------|
| 0:00–0:15 | **Read** (use the [three-pass method](/papers/lora-mock-interview/#part-0--how-to-read-a-paper-in-15-minutes-the-three-pass-method)) | Image-as-patch-sequence idea + why it needs lots of data |
| 0:15–0:20 | **Explain the benefit** out loud (cover Part 2) | Global receptive field from layer 1, no conv inductive bias |
| 0:20–0:55 | **Implement** from the stub (Part 3) | A tiny `ViT` that classifies a visual rule above chance |
| last 10 min | **Sanity-check** (Part 4) | Patch count, sequence length, shapes, accuracy — narrated |

### Self-grading rubric — "what good looks like"
- ✅ **Scoped** to the patch-transformer pipeline, not a full pretraining reproduction.
- ✅ Explained the core trick: **flatten patches → linear embed → add CLS + positions → plain Transformer encoder**, no convolutions in the model body.
- ✅ Got the **patch math** right: `n_patches = (H/p)·(W/p)`, sequence length `n_patches + 1` after the CLS token.
- ✅ Knew **why positional embeddings are needed** (attention is permutation-equivariant) and **why a CLS token** (a learned slot to pool a global representation).
- ✅ Had an honest take on the **inductive-bias trade-off**: ViT lacks locality/translation-equivariance, so it **needs lots of data / pretraining** to beat CNNs.
- ⚠️ Red flags: forgetting positional embeddings, classifying from a patch instead of the CLS token, miscounting patches, claiming ViT beats CNNs at small data scale.

---

## Part 1 — Structured read of THIS paper

### The 30-second summary (the "benefit")
CNNs bake in **locality** and **translation equivariance** with convolutions — strong, hand-designed priors. ViT throws those away: it cuts the image into a grid of fixed-size patches, **flattens and linearly embeds each patch into a token**, and feeds the resulting sequence to a **plain Transformer encoder** — the same one used for text. The payoff:

- **No convolutions in the model body** — an image becomes a *sequence* and is handled by standard self-attention.
- **Global receptive field from the very first layer** — every patch can attend to every other patch immediately, unlike a CNN where the receptive field grows layer by layer.
- **Scales beautifully with data and compute** — with enough pretraining data (ImageNet-21k, JFT-300M) ViT matches or beats the best CNNs at lower pretraining cost.
- **The caveat (its known weakness):** with little data ViT *underperforms* CNNs, precisely because it lacks the conv inductive bias and must learn locality from scratch.

### The core idea (Method — you implement this)
Split an `H×W` image into non-overlapping `p×p` patches. The number of patches (the sequence length) is:

$$N = \frac{H}{p}\cdot\frac{W}{p}$$

Flatten each patch to a vector of length `p·p·C` and project it linearly to the model dimension `D`. Prepend a **learnable class token** `x_cls`, add **learnable positional embeddings** `E_pos`, and run the sequence through a Transformer encoder:

$$z_0 = \big[\,x_{\text{cls}};\ x^1_pE;\ x^2_pE;\ \dots;\ x^N_pE\,\big] + E_{\text{pos}}, \qquad E_{\text{pos}}\in\mathbb{R}^{(N+1)\times D}$$

Each encoder layer is **pre-norm** multi-head self-attention and an MLP, each wrapped in a residual:

$$z'_\ell = \text{MSA}(\text{LN}(z_{\ell-1})) + z_{\ell-1}, \qquad z_\ell = \text{MLP}(\text{LN}(z'_\ell)) + z'_\ell$$

The classification head reads the **final state of the CLS token** only:

$$y = \text{Head}\big(\text{LN}(z_L^{0})\big)$$

Key details (the things an interviewer probes):
- **Patch embedding = Conv2d with `kernel=stride=patch`.** A conv whose stride equals its kernel size is exactly "cut into non-overlapping patches and apply one shared linear map." It's a convenience, not a convolutional prior — there's no overlap and no spatial mixing inside it.
- **The CLS token** is a learned vector with no input content; through attention it aggregates a global summary that the head classifies. (Alternatively you can average-pool the patch tokens — the paper notes both work.)
- **Positional embeddings are mandatory.** Self-attention is permutation-equivariant; without positions the model can't tell a top-left patch from a bottom-right one. ViT uses *learned* 1-D position embeddings.
- **Pre-norm vs post-norm.** ViT puts LayerNorm *before* the sub-layer (pre-norm), which trains more stably for deep Transformers.
- **Inductive bias is the whole story.** ViT only has locality at the patch boundary and the (learned) positional structure — far less than a CNN — so its data hunger is the direct consequence of the design.

### Where the evidence lives (tables that matter)
*(From memory; verify exact numbers against the paper.)*
- **Data-scale figure / table:** ViT trails CNNs (BiT) when pretrained on small data but **overtakes them as pretraining data grows** (ImageNet-1k → 21k → JFT-300M) → the central "scales with data" claim.
- **Main results table:** ViT-L/H matches or beats SOTA CNNs on ImageNet/CIFAR/VTAB at **lower pretraining compute** → the efficiency argument.
- **Attention-distance / attention-map figures:** lower layers already attend globally (some heads), and attention focuses on semantically relevant regions → the "global from layer 1" mechanism, visualized.

### The honest limitations (have an opinion)
- **Data hungry.** Without large-scale pretraining it loses to CNNs at the same size — the inductive-bias deficit. Later work (DeiT) closes the gap with distillation/augmentation.
- **Quadratic attention cost** in the number of patches → high-resolution images are expensive (motivates hierarchical/windowed variants like Swin).
- **Fixed patch grid / position embeddings** make changing input resolution awkward (needs interpolation of position embeddings).

---

## Part 2 — The interview dialogue (interviewer ⇄ interviewee)

> **🧑‍💼 Interviewer:** One paragraph — what is ViT actually doing differently from a CNN?
>
> **🧑‍💻 Interviewee:** It treats an image as a *sequence*. It cuts the image into fixed `p×p` patches, flattens and linearly embeds each one into a token, adds a learnable class token plus positional embeddings, and runs the sequence through a plain Transformer encoder — no convolutions in the body. So instead of a receptive field that grows layer by layer, every patch can attend to every other patch from the first layer. The trade-off is that ViT drops the locality and translation-equivariance priors a CNN has, so it needs a lot of pretraining data to match or beat CNNs.

> **🧑‍💼 Interviewer:** Why do you need a CLS token and positional embeddings?
>
> **🧑‍💻 Interviewee:** The positional embeddings are because self-attention is permutation-equivariant — without them the model literally can't distinguish a patch in the top-left from one in the bottom-right; shuffling patches would give the same output. The CLS token is a learned slot with no image content; through attention it accumulates a global summary of all patches, and the classification head reads only that token. You can instead mean-pool the patch tokens — the paper says both work — but the CLS token mirrors the BERT design.

> **🧑‍💼 Interviewer:** The patch embedding is "just a Conv2d." Doesn't that sneak a convolutional prior back in?
>
> **🧑‍💻 Interviewee:** Not really. It's a Conv2d whose stride equals its kernel size, so the patches don't overlap and there's no spatial mixing across patch boundaries — it's exactly "chop into non-overlapping patches and apply one shared linear projection." It's an implementation convenience for the patchify-and-embed step. The actual convolutional priors — overlapping local windows, weight sharing across a sliding window, a slowly growing receptive field — are absent.

> **🧑‍💼 Interviewer:** When does ViT beat a CNN, and when does it lose?
>
> **🧑‍💻 Interviewee:** It loses at small data scale: with only ImageNet-1k, a comparable CNN generalizes better because its inductive biases substitute for data. ViT wins once you pretrain on large datasets — ImageNet-21k or JFT-300M — where it matches or beats the best CNNs at lower pretraining compute. The crossover is the headline figure of the paper: accuracy vs pretraining data size, ViT overtaking CNNs as data grows.

> **🧑‍💼 Interviewer:** What's the main scaling cost, and how is it mitigated?
>
> **🧑‍💻 Interviewee:** Attention is quadratic in the number of patches, so doubling resolution (4× patches) is ~16× the attention cost. Mitigations are hierarchical/windowed attention like Swin, which restricts attention to local windows and merges patches across stages, bringing it closer to linear in image area.

> **🧑‍💼 Interviewer:** Implement the patch-transformer pipeline and show it learns a visual rule.

---

## Part 3 — Implementation

We reuse the **exact multi-head attention** from the [Attention mock](/papers/attention-mock-interview/), then wrap it in pre-norm encoder blocks and the ViT patch pipeline.

```python
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def scaled_dot_product_attention(q, k, v):
    """q,k,v: (..., T, d_k). Returns (output, attention_weights)."""
    d_k = q.size(-1)
    scores = (q @ k.transpose(-2, -1)) / math.sqrt(d_k)
    attn = scores.softmax(dim=-1)                 # distribution over keys
    return attn @ v, attn


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model, self.n_heads, self.d_k = d_model, n_heads, d_model // n_heads
        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)
        self.wo = nn.Linear(d_model, d_model)

    def _split(self, x):                           # (B,T,d_model) -> (B,heads,T,d_k)
        B, T, _ = x.shape
        return x.view(B, T, self.n_heads, self.d_k).transpose(1, 2)

    def forward(self, x):
        q, k, v = self._split(self.wq(x)), self._split(self.wk(x)), self._split(self.wv(x))
        out, attn = scaled_dot_product_attention(q, k, v)
        B, _, T, _ = out.shape
        out = out.transpose(1, 2).contiguous().view(B, T, self.d_model)  # concat heads
        return self.wo(out), attn


class EncoderBlock(nn.Module):
    """Pre-norm Transformer encoder block: LN -> MHA -> +res, LN -> MLP -> +res."""

    def __init__(self, dim, n_heads, mlp_ratio=2):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiHeadAttention(dim, n_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio), nn.GELU(),
            nn.Linear(dim * mlp_ratio, dim),
        )

    def forward(self, x):
        a, attn = self.attn(self.norm1(x))
        x = x + a                        # residual around attention
        x = x + self.mlp(self.norm2(x))  # residual around MLP
        return x, attn


class ViT(nn.Module):
    def __init__(self, img_size=12, patch=4, in_ch=1, dim=32, depth=2,
                 n_heads=4, n_classes=4):
        super().__init__()
        assert img_size % patch == 0
        self.n_patches = (img_size // patch) ** 2
        # Patch embed = Conv2d with kernel=stride=patch == non-overlapping patchify + shared Linear
        self.proj = nn.Conv2d(in_ch, dim, kernel_size=patch, stride=patch)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))            # learnable CLS slot
        self.pos_embed = nn.Parameter(torch.zeros(1, self.n_patches + 1, dim))  # +1 for CLS
        self.blocks = nn.ModuleList([EncoderBlock(dim, n_heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, n_classes)
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.pos_embed, std=0.02)

    def patchify(self, imgs):                       # (B,C,H,W) -> (B, n_patches, dim)
        x = self.proj(imgs)                         # (B, dim, H/p, W/p)
        return x.flatten(2).transpose(1, 2)         # (B, n_patches, dim)

    def forward(self, imgs, return_attn=False):
        B = imgs.size(0)
        x = self.patchify(imgs)                     # (B, n_patches, dim)
        cls = self.cls_token.expand(B, -1, -1)      # (B, 1, dim)
        x = torch.cat([cls, x], dim=1)              # (B, n_patches+1, dim)
        x = x + self.pos_embed                      # add positions (mandatory!)
        last_attn = None
        for blk in self.blocks:
            x, last_attn = blk(x)
        x = self.norm(x)
        logits = self.head(x[:, 0])                 # classify from the CLS token only
        return (logits, last_attn) if return_attn else logits
```

### Why each line matters (talk through it)
- `nn.Conv2d(..., kernel_size=patch, stride=patch)` — stride == kernel means non-overlapping patches; this *is* "flatten each patch and apply a shared linear map," not a convolutional prior.
- `x.flatten(2).transpose(1, 2)` — turns the `(B, dim, H/p, W/p)` feature grid into a `(B, n_patches, dim)` **sequence** of tokens.
- `cls_token` + `torch.cat([cls, x], dim=1)` — prepends one learned token whose final state is the pooled global representation; that's why the head reads `x[:, 0]`.
- `x + self.pos_embed` — without this, attention is permutation-equivariant and the model can't use spatial layout. Shape `(1, N+1, dim)` covers CLS + every patch.
- pre-norm `LN` *inside* each residual branch — stabilizes training of the stacked encoder.
- `self.head(x[:, 0])` — classify from the CLS token, not from a patch.

### Demonstrating the benefit (a global visual rule the patch-transformer can learn)
We make tiny `12×12` grayscale images (patch `4` → `9` patches). Each image has a `3×3` bright square dropped into one of the **four quadrants**; the label is *which quadrant*. Solving it requires reasoning about **where** a patch is — exactly what positional embeddings + global attention provide.

```python
def make_data(n, img_size=12, seed=0):
    g = torch.Generator().manual_seed(seed)
    imgs = 0.1 * torch.randn(n, 1, img_size, img_size, generator=g)
    labels = torch.randint(0, 4, (n,), generator=g)
    half = img_size // 2
    for i in range(n):
        q = labels[i].item()
        r0 = 0 if q in (0, 1) else half      # top for 0,1 / bottom for 2,3
        c0 = 0 if q in (0, 2) else half      # left for 0,2 / right for 1,3
        rr = r0 + torch.randint(0, half - 2, (1,), generator=g).item()
        cc = c0 + torch.randint(0, half - 2, (1,), generator=g).item()
        imgs[i, 0, rr:rr + 3, cc:cc + 3] += 3.0   # bright square in that quadrant
    return imgs, labels


torch.manual_seed(0)
Xtr, ytr = make_data(600, seed=1)
Xte, yte = make_data(300, seed=2)
model = ViT(img_size=12, patch=4, dim=32, depth=2, n_heads=4, n_classes=4)
opt = torch.optim.Adam(model.parameters(), lr=3e-3)

model.train()
for epoch in range(40):
    perm = torch.randperm(Xtr.size(0))
    for i in range(0, Xtr.size(0), 64):
        idx = perm[i:i + 64]
        loss = F.cross_entropy(model(Xtr[idx]), ytr[idx])
        opt.zero_grad(); loss.backward(); opt.step()

model.eval()
with torch.no_grad():
    tr_acc = (model(Xtr).argmax(1) == ytr).float().mean().item()
    te_acc = (model(Xte).argmax(1) == yte).float().mean().item()
print(f"train acc {tr_acc:.3f}   test acc {te_acc:.3f}   (chance = 0.25)")
```

Test accuracy climbs far above the `0.25` chance level (in our run it reaches `1.000`), and it's reproducible under the fixed seed. That shows the **patch → embed → CLS → positions → encoder → head** pipeline genuinely learns a global spatial rule.

> **Important honesty note:** this only demonstrates the pipeline *works* (correctness / learning) on a small problem. The *real* ViT benefit — **matching/beating CNNs as you scale pretraining data** — cannot be shown at toy scale, and ViT's known weakness is exactly that it **needs lots of data**: at small scale a CNN's inductive biases would likely win. Treat the paper's data-scaling figures as the actual evidence; this toy is a correctness check.

---

## Part 4 — Sanity checks (don't skip)

### Check 1 — Patchify produces the right number of patches and shape
```python
model = ViT(img_size=12, patch=4, dim=32, depth=2, n_heads=4, n_classes=4)
imgs = torch.randn(8, 1, 12, 12)
p = model.patchify(imgs)
assert model.n_patches == (12 // 4) * (12 // 4) == 9
assert p.shape == (8, 9, 32)
print("OK: n_patches =", model.n_patches, "embedded shape", tuple(p.shape))
```

### Check 2 — After adding CLS, sequence length is n_patches + 1
```python
cls = model.cls_token.expand(8, -1, -1)
seq = torch.cat([cls, p], dim=1)
assert seq.shape[1] == model.n_patches + 1 == 10
print("OK: sequence length with CLS =", seq.shape[1])
```

### Check 3 — Positional embedding shape matches the CLS+patch sequence
```python
assert model.pos_embed.shape == (1, model.n_patches + 1, 32)
print("OK: pos_embed shape", tuple(model.pos_embed.shape))
```

### Check 4 — Output logits shape == (B, n_classes)
```python
logits = model(imgs)
assert logits.shape == (8, 4)
print("OK: logits shape", tuple(logits.shape))
```

### Check 5 — Trained test accuracy beats chance (the demonstration)
```python
Xte, yte = make_data(300, seed=2)
model.eval()
with torch.no_grad():
    te_acc = (model(Xte).argmax(1) == yte).float().mean().item()
assert te_acc > 0.6, te_acc          # chance = 0.25, so this is a clear margin
print(f"OK: test acc {te_acc:.3f} > chance 0.25")
```

### Check 6 — Attention rows sum to 1 and grads reach patch-embed + CLS + pos-embed
```python
model.train()
logits, attn = model(imgs, return_attn=True)
assert torch.allclose(attn.sum(-1), torch.ones_like(attn.sum(-1)), atol=1e-5)
assert (attn >= 0).all()
logits.sum().backward()
assert model.proj.weight.grad.abs().sum() > 0      # patch embedding learns
assert model.cls_token.grad.abs().sum() > 0        # CLS token learns
assert model.pos_embed.grad.abs().sum() > 0        # positions learn
print("OK: attn rows sum to 1; grads reach patch-embed, CLS, pos-embed")
```

---

## Part 5 — Likely follow-up questions

- *"CLS token vs global average pooling?"* — Both work for ViT. The CLS token is a learned aggregation slot (BERT-style); GAP mean-pools the patch tokens. The paper reports comparable accuracy with appropriate learning-rate tuning.
- *"Why learned position embeddings instead of sinusoidal (as in the original Transformer)?"* — ViT found learned 1-D embeddings work as well as 2-D-aware ones; the model learns the grid structure. Sinusoidal would also work — positions just have to be injected somehow.
- *"How do you change input resolution at fine-tuning time?"* — More patches means a longer sequence; you **interpolate** the pretrained position embeddings to the new grid size, keeping the CLS position embedding separate.
- *"DeiT — how did it make ViT work without JFT?"* — Heavy augmentation, regularization, and a distillation token learning from a CNN teacher, making ViT competitive on ImageNet-1k alone — directly addressing the data-hunger weakness.
- *"What's the relationship to Swin / hierarchical ViTs?"* — They restrict attention to local windows and merge patches across stages, reintroducing locality and a pyramid, which cuts the quadratic cost and helps at high resolution.
- *"Why does ViT need more data than a CNN?"* — It lacks the locality and translation-equivariance inductive biases that a CNN gets for free, so it must *learn* those from data; with enough data that flexibility becomes an advantage.

---

## TL;DR cheat sheet
| Thing | Answer |
|---|---|
| Core idea | Image → sequence of flattened patch tokens → plain Transformer encoder |
| Patch count | `N = (H/p)·(W/p)`; sequence length `N + 1` with the CLS token |
| Patch embed | Conv2d with `kernel=stride=patch` (non-overlapping patchify + shared Linear) |
| CLS token | Learned slot; its final state is classified by the head |
| Positions | Learned position embeddings — **mandatory** (attention is permutation-equivariant) |
| Encoder block | Pre-norm: `LN→MHA→+res`, `LN→MLP→+res` (reuses multi-head attention) |
| Benefit | Global receptive field from layer 1; scales with data/compute, no conv priors |
| Weakness | Data hungry — loses to CNNs without large-scale pretraining; O(n²) in patches |
| Can't show at toy scale | The data-scaling-vs-CNN crossover (needs ImageNet-21k / JFT-300M) |
