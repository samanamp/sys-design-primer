---
title: Knowledge Distillation — Paper-to-Code Mock Interview
description: A combined mock (read paper, explain benefit, implement in Colab) using Knowledge Distillation as the worked example — a small student learns from a big teacher's softened probabilities.
sidebar:
  order: 8
  label: Knowledge Distillation
---

> **Paper:** *Distilling the Knowledge in a Neural Network* — Hinton, Vinyals & Dean, 2015. [arXiv:1503.02531](https://arxiv.org/abs/1503.02531)
>
> **Format:** Read (~15 min) → explain the *real* benefit → implement the core idea in Colab → sanity-check it.
>
> **Companion notebook:** [`knowledge_distillation_mock.ipynb`](/notebooks/knowledge_distillation_mock.ipynb) (download) — a teacher/student demo + a `kd_loss` stub to fill in, plus verification cells. Open in [Google Colab](https://colab.research.google.com/) via *File → Upload notebook*.
>
> **Difficulty:** 🟡 Medium. A clean idea with two subtle knobs (temperature `T`, the `α` mix) and one classic gotcha (freezing the teacher).

---

## How to run this as a timed drill (~40 min)

| Time | Block | What you produce |
|------|-------|------------------|
| 0:00–0:12 | **Read** (use the [three-pass method](/papers/lora-mock-interview/#part-0--how-to-read-a-paper-in-15-minutes-the-three-pass-method)) | Why soft targets beat hard labels + the temperature trick |
| 0:12–0:17 | **Explain the benefit** out loud (cover Part 2) | "Dark knowledge" intuition + the `T²` factor |
| 0:17–0:33 | **Implement** from the stub (Part 3) | A working `kd_loss` + a student that beats hard-label training |
| last 5 min | **Sanity-check** (Part 4) | All checks passing, narrated out loud |

### Self-grading rubric — "what good looks like"
- ✅ Explained the benefit as **soft targets carry inter-class similarity** ("dark knowledge"), not just "copy the teacher."
- ✅ Knew **why the `T²` factor** is there (keeps gradient magnitudes comparable across temperatures).
- ✅ **Froze the teacher** — no gradient flows into it; it only produces fixed targets.
- ✅ Knew the loss is a **mix**: soft KL to the teacher + optional hard cross-entropy, weighted by `α`.
- ⚠️ Red flags: training the teacher and student jointly, forgetting `T²`, claiming distillation needs the teacher's training data (it needs a *transfer set*, often unlabelled), confusing temperature with the optimizer's learning rate.

---

## Part 1 — Structured read of THIS paper

### The 30-second summary (the "benefit")
A trained network's output isn't just "the answer" — its full probability vector encodes **how the model relates the classes**: a "2" gets a tiny but non-zero probability of being a "7", almost none of being a "cat". Those relative probabilities over the *wrong* answers are the **"dark knowledge."** A small **student** trained to match a big **teacher's** softened probabilities learns this structure and generalizes better than a student trained on hard one-hot labels alone. The payoff:

- **Compress a big model (or ensemble) into a small, cheap-to-serve one** with little accuracy loss.
- **Soft targets are a richer signal** than one-hot labels — each example teaches the student about *all* classes at once, so you need fewer examples / fewer epochs.
- Works on a **transfer set** that can be unlabelled, because the teacher supplies the targets.

### The core idea (Method — you implement this)
Soften both networks' outputs with a **temperature** `T` before comparing them. With logits `z`, the softened probability of class `i` is:

$$p_i(T) = \frac{\exp(z_i / T)}{\sum_j \exp(z_j / T)}$$

Higher `T` flattens the distribution, amplifying the small probabilities (the dark knowledge). The student is trained to match the teacher's softened distribution. The **distillation loss** is a temperature-scaled KL divergence, optionally blended with the ordinary hard-label cross-entropy:

$$\mathcal{L} = (1-\alpha)\, T^2 \cdot \mathrm{KL}\!\big(p^{\text{teacher}}(T)\,\|\,p^{\text{student}}(T)\big) \;+\; \alpha \cdot \mathrm{CE}\big(z^{\text{student}}, y\big)$$

The **`T²` factor** matters: the gradients of the soft term scale like `1/T²`, so multiplying by `T²` keeps the soft-loss gradient magnitude comparable to the hard-loss gradient as you change `T`. Without it, the soft term silently shrinks at high `T`.

Key details (the things an interviewer probes):
- **The teacher is frozen.** It is pre-trained and runs under `no_grad`; you only ever update the student. The teacher's logits are *fixed targets*.
- **Two temperatures, one rule.** Use the same `T > 1` to soften teacher *and* student during training. At **inference** the student runs at `T = 1` (ordinary softmax / argmax).
- **`α` mixes soft and hard.** `α = 0` is pure distillation; a small `α` adds the true labels as a light anchor. The paper found a low weight on the hard term works well.
- **It's not just copying.** Matching a soft distribution transfers the teacher's *relative* confidence over wrong classes — information a one-hot label literally cannot contain.

### Where the evidence lives (tables that matter)
- **MNIST distillation:** a distilled small net approaches the big net's test error; even omitting a digit class from the transfer set, the student still classifies it surprisingly well → soft targets carry transferable structure.
- **Speech (acoustic model) results:** a single distilled model recovers most of the gain of a 10-model ensemble → compression of an ensemble into one cheap model.
- **Ablation on temperature:** intermediate `T` works best → too low loses dark knowledge, too high washes out the signal.

### The honest limitations (have an opinion)
- **You need a good teacher first** — distillation is a *transfer* step, not a free lunch; the student is bounded by what the teacher knows.
- **Extra knobs to tune** (`T`, `α`) and an extra forward pass through the teacher per batch.
- **Gains shrink when the student is already large enough** or when you have abundant labelled data — the soft-target advantage is biggest when supervision is scarce or noisy.
- **Reported figures are dataset-specific.** Treat the paper's exact numbers as illustrative; the toy below demonstrates the *mechanism*, not those magnitudes.

---

## Part 2 — The interview dialogue (interviewer ⇄ interviewee)

> **🧑‍💼 Interviewer:** One paragraph — what does knowledge distillation actually buy me?
>
> **🧑‍💻 Interviewee:** A way to shrink a big, expensive model into a small one that's cheap to serve, with little accuracy loss. Instead of training the small "student" on hard one-hot labels, I train it to match the big "teacher's" full probability vector. That vector encodes how the teacher relates the classes — the relative probabilities it assigns to the *wrong* answers — which Hinton calls "dark knowledge." That's a much richer signal than a one-hot label, so the student generalizes better, especially when labels are scarce or noisy.

> **🧑‍💼 Interviewer:** Why the temperature `T`, and where does the `T²` come from?
>
> **🧑‍💻 Interviewee:** Temperature softens the softmax: dividing logits by `T > 1` flattens the distribution so the small probabilities — the dark knowledge — become large enough to actually train on. I soften both teacher and student by the same `T` and take the KL between them. The catch is that the gradient of that soft term scales like `1/T²`, so as I raise `T` the soft loss quietly shrinks relative to the hard cross-entropy. Multiplying the soft term by `T²` cancels that, keeping the two terms' gradient magnitudes comparable regardless of `T`.

> **🧑‍💼 Interviewer:** What's the most common bug when implementing this?
>
> **🧑‍💻 Interviewee:** Letting gradients flow into the teacher. The teacher must be frozen — `detach()` its logits or run it under `torch.no_grad()` — otherwise you're training both nets and the "targets" drift. The other classic slip is forgetting the `T²` factor, or softening only one of the two networks. And at inference you drop back to `T = 1`; the temperature is a training-time device.

> **🧑‍💼 Interviewer:** Does the student need the original labelled training data?
>
> **🧑‍💻 Interviewee:** Not necessarily. The teacher supplies the targets, so you can distill on any **transfer set** — often unlabelled, even out-of-distribution data. If you *do* have labels you can add a small hard-label cross-entropy term weighted by `α` as a light anchor, but pure soft-target distillation (`α = 0`) already works. That's part of the appeal: you can compress a model using cheap unlabelled data.

> **🧑‍💼 Interviewer:** Implement the loss and show the student beats hard-label training.

---

## Part 3 — Implementation

The whole method is a temperature-scaled KL between teacher and student, optionally blended with hard-label cross-entropy.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


def kd_loss(student_logits, teacher_logits, targets=None, T=4.0, alpha=0.1):
    """Hinton et al. 2015 distillation loss.

    soft term: T^2 * KL( softmax(teacher/T) || softmax(student/T) )
    hard term: cross-entropy with ground-truth labels (optional)
    total = (1 - alpha) * soft + alpha * hard
    """
    teacher_logits = teacher_logits.detach()              # teacher is a FIXED target
    log_p_student = F.log_softmax(student_logits / T, dim=-1)
    p_teacher = F.softmax(teacher_logits / T, dim=-1)
    soft = F.kl_div(log_p_student, p_teacher, reduction="batchmean") * (T * T)
    if targets is None or alpha == 0.0:
        return soft
    hard = F.cross_entropy(student_logits, targets)
    return (1.0 - alpha) * soft + alpha * hard
```

### Why each line matters (talk through it)
- `teacher_logits.detach()` — **freezes the teacher.** Its logits become constant targets; no gradient flows back into it. (Equivalently, generate them under `torch.no_grad()`.)
- `F.log_softmax(student / T)` — PyTorch's `kl_div` expects the *first* argument in **log-space**; this is the softened student log-probabilities.
- `F.softmax(teacher / T)` — the softened teacher probabilities (the dark knowledge). Same `T` for both, by design.
- `reduction="batchmean"` — averages the KL over the batch, the mathematically correct normalization (the default `"mean"` divides by the wrong count).
- `* (T * T)` — the **`T²` factor** that keeps the soft-loss gradient magnitude comparable across temperatures.
- `targets is None or alpha == 0.0` — pure distillation path; you can distill with no labels at all.

### Demonstrating the benefit (dark knowledge survives label noise)
A big teacher is trained on **clean** labels and learns the true class structure. A tiny student is then trained on **noisy** labels (40% flipped). We compare a student trained on those noisy labels alone vs. a student that *also* matches the frozen teacher's soft targets. The teacher's dark knowledge pulls the student back toward the truth.

```python
N_CLASSES = 4

def make_blobs(n_per_class, seed, spread=0.8):
    g = torch.Generator().manual_seed(seed)
    angles = torch.arange(N_CLASSES) * (2 * 3.14159265 / N_CLASSES)
    centers = torch.stack([torch.cos(angles), torch.sin(angles)], dim=1) * 2.5
    X = [centers[c] + spread * torch.randn(n_per_class, 2, generator=g) for c in range(N_CLASSES)]
    y = [torch.full((n_per_class,), c, dtype=torch.long) for c in range(N_CLASSES)]
    return torch.cat(X), torch.cat(y)

def corrupt_labels(y, frac, seed):
    g = torch.Generator().manual_seed(seed); yc = y.clone()
    flip = torch.rand(len(y), generator=g) < frac
    yc[flip] = torch.randint(0, N_CLASSES, (int(flip.sum()),), generator=g)
    return yc

def make_net(hidden):
    return nn.Sequential(nn.Linear(2, hidden), nn.ReLU(),
                         nn.Linear(hidden, hidden), nn.ReLU(),
                         nn.Linear(hidden, N_CLASSES))

def accuracy(net, X, y):
    net.eval()
    with torch.no_grad():
        return (net(X).argmax(-1) == y).float().mean().item()

def train_teacher(Xtr, ytr, seed):
    torch.manual_seed(seed); net = make_net(64)
    opt = torch.optim.Adam(net.parameters(), lr=1e-2); net.train()
    for _ in range(500):
        loss = F.cross_entropy(net(Xtr), ytr)
        opt.zero_grad(); loss.backward(); opt.step()
    return net

def train_student(Xtr, ytr_noisy, teacher, seed, use_kd, T=4.0, alpha=0.1):
    torch.manual_seed(seed); net = make_net(8)
    opt = torch.optim.Adam(net.parameters(), lr=1e-2)
    tlog = None
    if teacher is not None:
        teacher.eval()
        with torch.no_grad():            # teacher = fixed soft targets
            tlog = teacher(Xtr)
    net.train()
    for _ in range(400):
        slog = net(Xtr)
        loss = kd_loss(slog, tlog, ytr_noisy, T=T, alpha=alpha) if use_kd \
               else F.cross_entropy(slog, ytr_noisy)
        opt.zero_grad(); loss.backward(); opt.step()
    return net

# Average over a few seeds so the gap is reproducible, not luck.
seeds = [0, 1, 2, 3, 4]
a_accs, k_accs = [], []
for s in seeds:
    Xtr, ytr = make_blobs(60, seed=s)            # clean labels for the teacher
    Xte, yte = make_blobs(400, seed=s + 999)     # clean held-out test set
    teacher = train_teacher(Xtr, ytr, seed=s)
    ytr_noisy = corrupt_labels(ytr, frac=0.4, seed=s + 7)   # student sees noisy labels
    alone = train_student(Xtr, ytr_noisy, None, seed=s + 1, use_kd=False)
    kd    = train_student(Xtr, ytr_noisy, teacher, seed=s + 1, use_kd=True)
    a_accs.append(accuracy(alone, Xte, yte))
    k_accs.append(accuracy(kd, Xte, yte))

mean = lambda xs: sum(xs) / len(xs)
print(f"student-alone test acc : {mean(a_accs):.3f}")
print(f"student+KD   test acc : {mean(k_accs):.3f}")
```

Verified output (5-seed mean; numbers are seed-dependent, the *direction* is the point):

```
student-alone test acc : 0.940
student+KD   test acc : 0.964
```

The student trained on noisy labels alone partially memorizes the wrong labels; the distilled student recovers the teacher's clean structure and generalizes better — sometimes even **beating the teacher**, because soft targets act as a regularizer. (This is a small toy that demonstrates the dark-knowledge mechanism, not the paper's exact figures.)

---

## Part 4 — Sanity checks (don't skip)

### Check 1 — At `T=1`, `α=1` the KD loss reduces to ordinary cross-entropy
```python
torch.manual_seed(7)
B = 16
student_logits = torch.randn(B, N_CLASSES, requires_grad=True)
teacher_logits = torch.randn(B, N_CLASSES)
targets = torch.randint(0, N_CLASSES, (B,))

l_kd = kd_loss(student_logits, teacher_logits, targets, T=1.0, alpha=1.0)
l_ce = F.cross_entropy(student_logits, targets)
assert torch.allclose(l_kd, l_ce)
print("OK: alpha=1, T=1 == cross-entropy")
# and alpha=0, T=1 equals the plain KL to the teacher:
l_soft = kd_loss(student_logits, teacher_logits, T=1.0, alpha=0.0)
l_kl = F.kl_div(F.log_softmax(student_logits, -1), F.softmax(teacher_logits, -1), reduction="batchmean")
assert torch.allclose(l_soft, l_kl)
print("OK: alpha=0, T=1 == plain KL")
```

### Check 2 — Softened teacher targets sum to 1 and are softer (higher entropy) at higher `T`
```python
def soft_stats(T):
    p = F.softmax(teacher_logits / T, dim=-1)
    ent = -(p * p.clamp_min(1e-12).log()).sum(-1).mean()
    return p.sum(-1), ent

sums_lo, ent_lo = soft_stats(1.0)
sums_hi, ent_hi = soft_stats(8.0)
assert torch.allclose(sums_lo, torch.ones(B)) and torch.allclose(sums_hi, torch.ones(B))
assert ent_hi > ent_lo
print(f"OK: probs sum to 1; entropy {ent_lo:.3f} (T=1) < {ent_hi:.3f} (T=8)")
```

### Check 3 — The teacher receives NO gradient (frozen)
```python
teacher = train_teacher(*make_blobs(20, seed=0), seed=0)
for p in teacher.parameters():
    p.grad = None
student = make_net(8)
Xs, _ = make_blobs(8, seed=5)
teacher.eval()
with torch.no_grad():
    tlog = teacher(Xs)
kd_loss(student(Xs), tlog, T=4.0, alpha=0.0).backward()
assert all(p.grad is None for p in teacher.parameters())
print("OK: teacher params have no gradient")
```

### Check 4 — Gradient DOES flow to the student
```python
assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in student.parameters())
print("OK: student receives non-zero gradient")
```

### Check 5 — student+KD ≥ student-alone (the demonstration)
```python
assert mean(k_accs) >= mean(a_accs)
print(f"OK: KD {mean(k_accs):.3f} >= alone {mean(a_accs):.3f}")
```

### Check 6 — KD soft loss is ~0 when student logits == teacher logits
```python
sl = torch.randn(B, N_CLASSES)
zero = kd_loss(sl.clone().requires_grad_(True), sl.clone(), T=4.0, alpha=0.0)
assert zero.item() < 1e-6
print(f"OK: identical logits -> KD soft loss {zero.item():.2e} ~ 0")
```

---

## Part 5 — Likely follow-up questions

- *"Distillation vs. just training a small model on the same data?"* — The small model on hard labels only sees the right answer; the distilled model also sees the teacher's confidence over wrong answers (dark knowledge), which is extra free supervision per example.
- *"What is a transfer set and why can it be unlabelled?"* — It's whatever data you push through the teacher to generate soft targets. Since the teacher provides the targets, the transfer set needs no labels — handy for compressing on cheap unlabelled data.
- *"How does this relate to label smoothing?"* — Label smoothing replaces one-hot targets with a *uniform* soft target; distillation uses the teacher's *input-dependent* soft target, which carries real per-example structure rather than a constant.
- *"How do you distill an ensemble?"* — Average the ensemble members' soft predictions and distill into one model; the paper compresses a 10-model speech ensemble into a single net that keeps most of the gain.
- *"What's 'matching logits' as a special case?"* — At high `T`, minimizing the soft KL is approximately equivalent to matching the teacher's logits directly (the softmax linearizes), which is the original "dark knowledge" intuition.

---

## TL;DR cheat sheet
| Thing | Answer |
|---|---|
| Core idea | Train a small student to match a big teacher's *softened* probabilities |
| The benefit | Soft targets carry inter-class similarity ("dark knowledge") → better generalization + compression |
| Formula | `L = (1−α)·T²·KL(teacher(T) ‖ student(T)) + α·CE(student, y)` |
| Temperature `T` | Softens softmax (`logits/T`); amplifies the dark knowledge; `T=1` at inference |
| Why `T²` | Soft gradients scale like `1/T²`; the factor keeps magnitudes comparable across `T` |
| Teacher | **Frozen** — `detach()` / `no_grad`, only the student is trained |
| `α` | Mix of soft (teacher) vs. hard (true labels); `α=0` is pure distillation |
| Transfer set | Can be unlabelled — the teacher supplies targets |
| #1 bug | Gradients leaking into the teacher; forgetting the `T²` factor |
| Limitation | Bounded by the teacher; extra knobs; gains shrink with abundant clean labels |
