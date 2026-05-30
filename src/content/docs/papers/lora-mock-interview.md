---
title: LoRA — Paper-to-Code Mock Interview
description: A full combined mock for the "read a paper, explain the benefit, implement it in Colab" ML interview, using LoRA as the worked example.
sidebar:
  order: 7
  label: LoRA
---

> **Paper:** *LoRA: Low-Rank Adaptation of Large Language Models* — Hu et al., 2021. arXiv: [2106.09685](https://arxiv.org/abs/2106.09685)
>
> **Format:** Read the paper (~15 min) → explain the *real* benefit → implement the core idea in Colab → sanity-check it.
>
> **Companion notebook:** [`lora_mock.ipynb`](/notebooks/lora_mock.ipynb) (download) — toy task + a `LoRALinear` stub to fill in, plus verification cells. Or open it straight in [Google Colab](https://colab.research.google.com/) via *File → Upload notebook*. A reference solution is included at the bottom of this page.
>
> **Difficulty:** 🟡 Medium. The layer is ~10 lines; the subtlety is *why* `B` is zero-initialized and what the `α/r` scaling buys you.

---

## How to run this as a timed drill (~60 min)

Treat this like the real thing. Set a timer and don't look at the answers below until each block is done.

| Time | Block | What you produce |
|------|-------|------------------|
| 0:00–0:15 | **Read** (Part 0 method on the [real PDF](https://arxiv.org/abs/2106.09685)) | The core equation + the one table that proves the benefit |
| 0:15–0:20 | **Explain the benefit** out loud (cover Part 2 without peeking) | 1-paragraph pitch + answers to "why B=0", "what's α/r", "when not to" |
| 0:20–0:50 | **Implement** in Colab from the stub (Part 3) | A working `LoRALinear` + loss-goes-down on the toy task |
| last 10 min | **Sanity-check** (Part 4) | All 6 checks passing, talked through out loud |

### Self-grading rubric — "what good looks like"
- ✅ Anchored at least one benefit claim to a **specific table** ("the rank ablation shows…"), not just the abstract.
- ✅ Named the **tradeoff/limitation** unprompted, not only the upside.
- ✅ Got `LoRALinear` running **without** copy-pasting — froze the base, zero-init'd B, projected down-then-up.
- ✅ Wrote **at least 2 sanity checks** before being asked (shapes + "only A,B train").
- ✅ Narrated decisions while coding instead of going silent.
- ⚠️ Red flags: silent coding, summarizing the abstract instead of the contribution, forgetting to freeze the base, claiming a benefit with no number behind it.

---

## Part 0 — How to read a paper in 15 minutes (the three-pass method)

You will *not* read top-to-bottom. You read in **passes**, each adding detail, and you stop when you have enough to talk and implement. This is a compressed version of Keshav's well-known "three-pass" approach.

### Pass 1 — Map it (3 min)
Read **only**: Title → Abstract → Section headings → all **figures and tables** (and their captions) → Conclusion.
Goal: answer *"what problem, what's the claim, what's the shape of the solution?"* Do not read body paragraphs yet.

### Pass 2 — The method + the evidence (8 min)
Read the **Method/Approach** section carefully (this is the part you'll implement) and the **main results table**. Skim Related Work. Ignore proofs and most of the experimental setup details.
Goal: be able to write the core equation and know *which number proves the benefit*.

### Pass 3 — Pressure-test (4 min)
Find the **ablations** and **limitations**. Ask: what does the paper compare against (baselines)? Is the comparison fair? Where does the gain actually come from? When would this NOT work?
Goal: have an opinion, not just a summary.

> 💡 **Interview tip:** an interviewer can tell within 60 seconds whether you read the *figures* or just the abstract. Always anchor claims to a specific table/figure: *"Table 2 shows…"*.

---

## Part 1 — Structured read of THIS paper

Here's what each pass should surface in the LoRA paper specifically: the summary and core idea come from Pass 2, the tables from the Pass 1 figure-skim confirmed in Pass 2, and the limitations from Pass 3.

### The 30-second summary (the "benefit")
Fine-tuning a large model normally updates **all** weights — expensive to train and store (a full copy of the model per task). LoRA **freezes the pretrained weights** and injects a small pair of trainable low-rank matrices into each adapted layer. You train only those. Result:

- **~10,000× fewer trainable parameters** and ~3× less GPU memory for the optimizer state (paper's headline on GPT-3 175B).
- **Zero added inference latency** — because at deploy time you can *merge* the low-rank update back into the original weight matrix.
- **Cheap task-switching** — swap a few MB of LoRA weights instead of shipping a whole fine-tuned model per task.

### The core idea (Method — read this carefully, you implement it)
For a pretrained weight matrix $W_0 \in \mathbb{R}^{d \times k}$, instead of learning a full update $\Delta W$, constrain it to be **low-rank**:

$$\Delta W = B A, \qquad B \in \mathbb{R}^{d \times r}, \; A \in \mathbb{R}^{r \times k}, \; r \ll \min(d, k)$$

The adapted forward pass becomes:

$$h = W_0\,x + \frac{\alpha}{r}\, B A\, x$$

> **Convention note:** the math above uses column vectors ($W x$). The code below uses the PyTorch batch-first convention ($x W^\top$, with `x` of shape `(batch, in)`) — same operation, transposed. Keep the two straight when you talk through it.

Key details (these are the things an interviewer probes):
- **`r`** is the rank — the single most important hyperparameter. The paper shows surprisingly small `r` (even 1–4) works well.
- **Initialization:** `A` is random Gaussian, **`B` is initialized to zero** → so `ΔW = 0` at the start and training begins exactly from the pretrained model. (Probe: *"why init B to zero?"*)
- **`α` (alpha) / scaling `α/r`:** scales the update so you don't have to re-tune the learning rate when you change `r`.
- **W₀ is frozen** — it receives no gradient. Only `A` and `B` train.
- **Where to apply it:** the paper applies LoRA to the **attention projection matrices** (they ablate Wq, Wk, Wv, Wo) and finds adapting Wq and Wv gives the best bang for the buck.

### Where the evidence lives (the tables that matter)
- **Table 2 / 5:** LoRA matches or beats full fine-tuning on GLUE/E2E etc. with a tiny fraction of params → this is the core benefit claim.
- **Rank ablation (Table 6/18):** performance is roughly flat from r=1 to r=64 → the surprising result that the "intrinsic rank" of the update is tiny. This is the most interesting scientific finding.
- **Which weights to adapt (Table 5):** adapting more matrices at small rank beats one matrix at high rank.

### The honest limitations (Pass 3 — have an opinion)
- It's an **approximation** — low rank can't express every update; on tasks very far from pretraining, full fine-tuning can still win.
- You **choose which layers** to adapt and pick `r` — extra hyperparameters.
- Merging weights to remove latency means you **can't batch inputs from different tasks** in a single forward pass (each needs its own merged W). The paper notes this tradeoff.

---

## Part 2 — The interview dialogue (interviewer ⇄ interviewee)

This is roughly how the "explain the benefit" half should sound. Use it as a target for your own answers.

> **🧑‍💼 Interviewer:** Give me the one-paragraph version. What does this paper actually buy me?
>
> **🧑‍💻 Interviewee:** Full fine-tuning of a large model means updating and storing every weight — so each downstream task costs you a full model copy and a big optimizer state. LoRA's claim is that the *change* in weights during adaptation has very low intrinsic rank, so you can freeze the original weights and learn only a low-rank update `B·A`. The practical payoff is three things: ~10,000× fewer trainable parameters, no extra inference latency because you can merge the update back into the weights, and you can hot-swap tasks by loading a few MB of adapter weights. The cost is that it's a low-rank *approximation*, so on tasks far from pretraining it can underperform full fine-tuning.

> **🧑‍💼 Interviewer:** You said "low intrinsic rank" — what's the evidence, not just the claim?
>
> **🧑‍💻 Interviewee:** Their rank ablation — performance is essentially flat from rank 1 up to 64. If you needed high rank to match full fine-tuning, the method wouldn't be interesting. The flatness is what makes the whole thing work, and it's the most surprising result in the paper.

> **🧑‍💼 Interviewer:** Why initialize `B` to zero?
>
> **🧑‍💻 Interviewee:** So the product `B·A` is zero at step 0. That means the adapted model starts *identical* to the pretrained model — you're not injecting random noise into a model that's already good. You only depart from it as gradients flow. If both A and B were random, you'd start by corrupting the pretrained features.

> **🧑‍💼 Interviewer:** What's the `α/r` scaling for?
>
> **🧑‍💻 Interviewee:** It decouples the magnitude of the update from the rank. If you bump `r` to give the model more capacity, the raw `B·A` output grows, which would otherwise force you to re-tune the learning rate. Scaling by `α/r` keeps the effective update magnitude roughly stable across ranks, so `r` and the LR are more independent knobs.

> **🧑‍💼 Interviewer:** When would you NOT reach for LoRA?
>
> **🧑‍💻 Interviewee:** When the target task is far from pretraining and you have the compute for full fine-tuning — the low-rank constraint becomes a real ceiling. Also, if I need to serve many *different* tasks in a single batched forward pass, the merge-for-zero-latency trick breaks down, since each task needs its own merged weight matrix.

> **🧑‍💼 Interviewer:** Great — now implement the core layer and show me it works on a toy problem.

---

## Part 3 — Implementation

The whole method is one layer. Here is a clean, runnable reference implementation in PyTorch.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    """A Linear layer with a frozen base weight + a trainable low-rank update.

    Forward:  h = x W0^T  +  (alpha / r) * x A^T B^T
    where the base weight W0 is frozen and only A, B are trained.
    """

    def __init__(self, in_features, out_features, r=4, alpha=8, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.r = r
        self.scaling = alpha / r

        # --- frozen base layer (the "pretrained" weights) ---
        self.base = nn.Linear(in_features, out_features, bias=bias)
        self.base.weight.requires_grad_(False)
        if bias:
            self.base.bias.requires_grad_(False)

        # --- trainable low-rank update: ΔW = B @ A ---
        # A: (r, in)  initialized random ;  B: (out, r) initialized to ZERO
        self.A = nn.Parameter(torch.randn(r, in_features) * 0.01)
        self.B = nn.Parameter(torch.zeros(out_features, r))

    def forward(self, x):
        base_out = self.base(x)                      # frozen path: x @ W0^T
        lora_out = (x @ self.A.T) @ self.B.T         # low-rank path: x @ A^T @ B^T
        return base_out + self.scaling * lora_out

    @torch.no_grad()
    def merged_weight(self):
        """The effective weight if we fold the LoRA update into the base.
        Used at inference to get zero added latency."""
        return self.base.weight + self.scaling * (self.B @ self.A)
```

### Why each line matters (talk through this as you write it)
- `self.base.weight.requires_grad_(False)` — this is the "freeze pretrained weights" claim, in code. If you forget it, you're just doing full fine-tuning with extra steps.
- `self.B = nn.Parameter(torch.zeros(...))` — the zero-init that makes `ΔW = 0` at step 0.
- `(x @ self.A.T) @ self.B.T` — note the **order**: project down to rank `r` *first*, then back up. Doing `(B @ A)` as a full `out×in` matrix first would defeat the entire memory benefit. Mentioning this unprompted is a strong signal.
- `self.scaling = alpha / r` — the decoupling knob from the paper.

### Minimal training loop (toy task)
The target is deliberately **the frozen base plus a rank-`r` delta** — i.e. a function that a rank-`r` LoRA update *can* represent. That's the honest test: it isolates whether the low-rank path can adapt the frozen base to a reachable target, so the loss should drive down to ~0. (If you instead chase a full-rank random target, a rank-4 update can't express it and the loss plateaus high — a misleading demo.)

```python
torch.manual_seed(0)

in_dim, out_dim, r = 64, 32, 4
layer = LoRALinear(in_dim, out_dim, r=r, alpha=8)

# Target = frozen base + a rank-r delta -> reachable by a rank-r LoRA update.
with torch.no_grad():
    delta = (torch.randn(out_dim, r) @ torch.randn(r, in_dim)) * 0.1
    teacher_W = layer.base.weight + delta

base_snapshot = layer.base.weight.clone()   # for sanity check 3 (taken BEFORE training)
opt = torch.optim.Adam([p for p in layer.parameters() if p.requires_grad], lr=1e-2)

for step in range(500):
    x = torch.randn(128, in_dim)
    y = x @ teacher_W.T + layer.base.bias
    loss = F.mse_loss(layer(x), y)

    opt.zero_grad()
    loss.backward()
    opt.step()

    if step % 100 == 0:
        print(f"step {step:3d}  loss {loss.item():.5f}")
```

The loss should fall toward ~0 — proof the low-rank path adapted the frozen base to the target without touching `W₀`.

---

## Part 4 — Sanity checks (do NOT skip — interviewers love these)

Writing code that runs is table stakes. Writing code you can *prove* is correct is what separates candidates. Talk through each check out loud.

### Check 1 — Only A and B are trainable
```python
trainable = [n for n, p in layer.named_parameters() if p.requires_grad]
frozen    = [n for n, p in layer.named_parameters() if not p.requires_grad]
print("trainable:", trainable)   # -> ['A', 'B']
print("frozen   :", frozen)      # -> ['base.weight', 'base.bias']

n_train = sum(p.numel() for p in layer.parameters() if p.requires_grad)
n_total = sum(p.numel() for p in layer.parameters())
print(f"trainable params: {n_train} / {n_total} "
      f"({100*n_train/n_total:.1f}%)")
```
**Expected:** only `A`, `B` train; trainable params are a small fraction. This *is* the paper's headline benefit, measured on your own layer.

### Check 2 — At init, the LoRA layer == the base layer
Because `B = 0`, the model must start identical to the pretrained weights.
```python
x = torch.randn(10, in_dim)
fresh = LoRALinear(in_dim, out_dim, r=4, alpha=8)
assert torch.allclose(fresh(x), fresh.base(x), atol=1e-6), "B should be zero at init!"
print("OK: output == base output at initialization")
```

### Check 3 — The frozen base weight never changed during training
This relies on `base_snapshot`, which we captured **before** the training loop above (capturing it after would compare the tensor to itself and pass trivially — proving nothing).
```python
assert torch.equal(layer.base.weight, base_snapshot), "base weight must not move!"
print("OK: base weight unchanged after training")
```

### Check 4 — Shapes are right
```python
assert layer.A.shape == (4, in_dim)     # (r, in)
assert layer.B.shape == (out_dim, 4)    # (out, r)
assert layer(x).shape == (10, out_dim)  # (batch, out)
print("OK: shapes correct")
```

### Check 5 — Merged weight gives the same output (zero-latency claim)
The deploy-time trick: folding `B·A` into `W0` must produce identical outputs.
```python
x = torch.randn(16, in_dim)
out_lora   = layer(x)
W_merged   = layer.merged_weight()
out_merged = x @ W_merged.T + layer.base.bias
assert torch.allclose(out_lora, out_merged, atol=1e-5), "merge mismatch!"
print("OK: merged weight reproduces LoRA output -> zero inference latency")
```

### Check 6 — Gradients flow to A and B, not to the base
```python
loss = F.mse_loss(layer(x), torch.randn(16, out_dim))
loss.backward()
print("A.grad is None? ", layer.A.grad is None)              # False
print("B.grad is None? ", layer.B.grad is None)              # False
print("base.grad is None?", layer.base.weight.grad is None)  # True
```

---

## Part 5 — Likely follow-up questions (be ready)

- *"How does LoRA differ from adapter layers?"* — Adapters add **extra sequential modules** → inference latency. LoRA's update is **parallel** and **mergeable** → no latency.
- *"What if you set r = full rank?"* — You recover the expressiveness of full fine-tuning (no longer low-rank), losing the parameter savings. The whole bet is that you don't need to.
- *"Where does the memory saving actually come from?"* — Mostly the **optimizer state**: Adam stores momentum + variance per *trainable* param. Far fewer trainable params → far smaller optimizer state. The frozen weights still sit in memory but need no optimizer state and no gradients.
- *"Could you apply this to conv layers?"* — Yes, factorize the conv weight similarly; the paper focuses on attention matrices but the idea generalizes.
- *"QLoRA — what changes?"* — Quantize the frozen base to 4-bit and keep LoRA in higher precision on top; pushes memory down further. Good to mention as the natural follow-on work.

---

## TL;DR cheat sheet
| Thing | Answer |
|---|---|
| Core idea | Freeze `W₀`, learn low-rank `ΔW = B·A` |
| Key hyperparam | rank `r` (small — 1 to 8 often enough) |
| B init | **zero** (so ΔW=0 at start) |
| Scaling | `α/r` (decouples r from LR) |
| Benefit #1 | ~10,000× fewer trainable params |
| Benefit #2 | zero inference latency (merge weights) |
| Benefit #3 | cheap task-swapping (tiny adapters) |
| Main evidence | rank ablation is flat → low intrinsic rank |
| Limitation | low-rank approximation; can't batch tasks after merge |
