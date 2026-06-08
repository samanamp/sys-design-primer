---
title: "Optimizers"
description: "ML coding interview primer on optimizers: SGD, Momentum, Adam, learning rate schedules, and gradient clipping — all implemented in NumPy so you see exactly what's happening."
---

# Optimizers: ML Coding Interview Primer

## The one-sentence version

An optimizer uses the gradient to update weights. All optimizers do this — they differ in *how* they use the gradient: just the current one, a smoothed history of it, or a per-parameter scaled version.

---

## Quick reference

| Optimizer | Core idea | Default hyperparams | Reach for it when… |
|---|---|---|---|
| **SGD** | move opposite the gradient | lr=0.01 | baselines, simple problems |
| **SGD + Momentum** | accumulate velocity across steps | lr=0.01, β=0.9 | oscillating loss, CNN training |
| **Adam** | per-parameter adaptive LR | lr=0.001, β1=0.9, β2=0.999 | most deep learning; sparse features |
| **AdamW** | Adam + decoupled weight decay | same + λ=0.01 | transformer fine-tuning |

| Schedule | Core idea | Reach for it when… |
|---|---|---|
| **Constant** | no decay | quick experiments, small models |
| **Step decay** | halve LR every N epochs | CNNs, supervised learning |
| **Cosine decay** | smooth curve from max to min LR | most modern training |
| **Warmup + cosine** | ramp up, then decay | large models, fine-tuning |

---

## How gradient descent works (plain English)

Imagine the loss as a hilly landscape. Your model's parameters define a position in that landscape. Training is about finding the lowest valley.

The **gradient** tells you the slope at your current position: which direction is uphill and how steep it is. Moving *opposite* the gradient moves downhill. Repeat this enough times and you reach a valley.

The **learning rate** controls step size. Too large and you overshoot the valley. Too small and you barely move.

```
new_weight = old_weight − learning_rate × gradient
```

That one line is the core of all gradient descent. Every optimizer in this article is a variation on it.

---

## The three gradient descent paradigms

Before optimizers: what gradient are you computing?

| Paradigm | Gradient computed over | Updates per epoch | Noise |
|---|---|---|---|
| **Batch GD** | all N examples | 1 | none (exact) |
| **Mini-batch GD** | random subset of 32–512 | N / batch_size | moderate |
| **Stochastic GD** | 1 example | N | high |

In practice, "SGD" almost always means **mini-batch GD**. True single-example GD is rarely used.

```python
import numpy as np

# All three on the same data — just the sampling differs
n = 1000
X, y = np.random.randn(n, 2), np.random.randn(n)
W = np.zeros(2)
lr = 0.01

def grad(Xb, yb, W):
    err = Xb @ W - yb
    return 2 * Xb.T @ err / len(yb)

# Batch GD: one gradient from all 1000 examples, one update per epoch
W -= lr * grad(X, y, W)

# Mini-batch GD: gradient from 64 examples, ~16 updates per epoch
for i in range(0, n, 64):
    W -= lr * grad(X[i:i+64], y[i:i+64], W)

# Stochastic GD: gradient from 1 example, 1000 updates per epoch (very noisy)
for xi, yi in zip(X, y):
    W -= lr * grad(xi.reshape(1,-1), np.array([yi]), W)
```

**Why mini-batch wins:** batch GD on 10M examples means 10M gradient computations before one weight update. Single-example GD is so noisy the loss oscillates rather than converging. Mini-batch balances both — enough examples for a stable gradient, small enough to update frequently and fit in GPU memory.

**Interview answer:** "SGD in practice means mini-batch gradient descent. Batch size is a hyperparameter — larger batches give lower-variance gradients but need more memory and often generalize slightly worse because they see fewer update steps per epoch."

---

## Problem 1 — Train a model from scratch → Vanilla SGD

Problem: predict delivery time from distance (km).

```
x = [2,  5,  10, 3,  8 ]   # distance (km)
y = [15, 30, 55, 18, 48]   # delivery time (min)
```

Model: `ŷ = w * x + b`. Two parameters: `w` (slope) and `b` (intercept).

### Computing gradients — once, by hand

Loss = MSE = mean((ŷ − y)²) = mean((wx + b − y)²)

```
∂Loss/∂w = mean(2 × (ŷ − y) × x)
∂Loss/∂b = mean(2 × (ŷ − y))
```

Plain English: the gradient is "prediction error, weighted by how much each parameter contributed." If `x` is large, `w` contributed a lot to the error, so the gradient for `w` is large.

### SGD in NumPy

```python
import numpy as np

X = np.array([2.0, 5.0, 10.0, 3.0, 8.0])
y = np.array([15.0, 30.0, 55.0, 18.0, 48.0])

w, b = 0.0, 0.0
lr   = 0.001

for step in range(500):
    y_pred = w * X + b
    error  = y_pred - y             # prediction error per example

    d_w = (2 * error * X).mean()   # gradient for w
    d_b = (2 * error).mean()       # gradient for b

    w -= lr * d_w                   # move opposite the gradient
    b -= lr * d_b

    if step % 100 == 0:
        loss = (error**2).mean()
        print(f"step {step:3d}  loss={loss:.2f}  w={w:.3f}  b={b:.3f}")
```

```
step   0  loss=1025.20  w=0.268  b=0.048
step 100  loss=  18.43  w=4.891  b=3.017
step 200  loss=   5.54  w=5.231  b=1.862
step 400  loss=   4.72  w=5.333  b=1.384   ← converged near y ≈ 5.3x + 1.4
```

### Learning rate sensitivity

| Learning rate | Result |
|---|---|
| 10.0 | loss explodes to NaN in 10 steps |
| 0.1 | converges in ~50 steps |
| 0.001 | converges in ~400 steps |
| 0.00001 | barely moves after 500 steps |

**Interview answer:** "SGD is the baseline. It moves each parameter opposite its gradient. The learning rate controls step size — the most important hyperparameter. I'd start with a value from literature for the task type (0.1 for SGD, 0.001 for Adam), run a short sweep if convergence is poor, and treat divergence (rising loss) as the signal that LR is too high."

**Regularization here — L2:** after training, if the model overfits to five routes and fails on new ones, add weight decay. SGD with L2:

```python
d_w = (2 * error * X).mean() + 2 * lambda_ * w   # L2 gradient appended
```

All features (distance, traffic, restaurant load) should stay active, so L2 (shrink all, zero none) is the right choice over L1.

---

## Problem 2 — Loss oscillates and won't converge → Momentum

### Why SGD oscillates in narrow valleys

Suppose the loss landscape is steep in one direction and flat in another — a narrow ravine. The gradient in the steep direction is large; in the flat direction, small.

With a fixed learning rate:
- In the steep direction: SGD takes large steps, overshoots, and oscillates back and forth.
- In the flat direction: SGD takes tiny steps, moving along the valley slowly.

The same LR can't fix both at once.

```
Two parameters at different scales:
  w1 gradient ≈ 0.01  (flat direction — needs big steps)
  w2 gradient ≈ 5.0   (steep direction — needs small steps)

With lr=0.1:
  w1 update: 0.1 × 0.01 = 0.001  → barely moves
  w2 update: 0.1 × 5.0  = 0.5    → oscillates wildly
```

### Momentum: build up speed in consistent directions

Instead of using only the current gradient, keep a running weighted average — a **velocity** vector:

```
velocity = β × velocity + (1 − β) × gradient
weight   = weight − lr × velocity
```

- In the **steep direction**: consecutive gradients alternate sign (oscillating). They partially cancel in the velocity. The effective step shrinks.
- In the **flat direction**: consecutive gradients point the same way. They accumulate in the velocity. The effective step grows.

Momentum damps oscillations and accelerates progress simultaneously. β=0.9 means "90% last step's velocity, 10% current gradient."

### Momentum in NumPy

```python
class Momentum:
    def __init__(self, lr=0.01, beta=0.9):
        self.lr   = lr
        self.beta = beta
        self.v    = {}      # velocity per parameter index

    def step(self, params, grads):
        for i, (p, g) in enumerate(zip(params, grads)):
            if i not in self.v:
                self.v[i] = np.zeros_like(p)
            # 90% of last step + 10% of current gradient
            self.v[i] = self.beta * self.v[i] + (1 - self.beta) * g
            p -= self.lr * self.v[i]
```

### Oscillation comparison

```
Step  SGD (w2)         Momentum (w2)
   0  5.0 → update −0.50    5.0 → update −0.050   ← gentler start
   1 −4.8 → update +0.48   −0.45 → update +0.004   ← velocity partially cancels
   2  4.6 → update −0.46    0.41 → update −0.042   ← oscillation dampens
  10  still oscillating      near converged
```

**Interview answer:** "Momentum helps when the loss landscape has directions of very different curvature. It accumulates velocity in directions the gradient consistently points, and cancels velocity where gradients oscillate. β=0.9 is almost always the right setting — the main lever is the learning rate, not β."

---

## Problem 3 — Sparse features or fine-tuning a large model → Adam

### Why a single global learning rate fails for sparse features

Suppose you have a feature "user_has_coupon" that's 1 for 2% of examples. The gradient for its weight is nonzero only on those 2% of batches. When it is nonzero, the update must be large. But if you set lr high enough for sparse features, dense features (nonzero gradient every batch) overshoot.

Adam fixes this with **per-parameter adaptive learning rates**.

### How Adam works

Adam tracks two statistics per parameter across all training steps:

- **m** (first moment): running average of the gradient — "what direction has this parameter been moving?"
- **v** (second moment): running average of the *squared* gradient — "how large have the gradients been?"

The update divides by `sqrt(v)`:

```
m = β1 × m + (1 − β1) × g        # smoothed direction
v = β2 × v + (1 − β2) × g²       # smoothed squared magnitude

weight = weight − lr × m / (sqrt(v) + ε)
```

- **Sparse parameter** ("has_coupon"): v is small (rare gradients) → `1/sqrt(v)` is large → bigger effective step.
- **Dense parameter** (e.g., a bias updated every batch): v is large → smaller effective step.

Each parameter automatically gets the learning rate it needs.

### Bias correction: why Adam is slow to start without it

At step 1, m and v are initialized to zero. After one gradient:

```
m = 0.9 × 0 + 0.1 × g = 0.1 × g
```

m is 10× smaller than the actual gradient — the model barely moves early on. Adam corrects this:

```
m_hat = m / (1 − β1^t)    # at t=1: divides by 0.1, giving 10× amplification
v_hat = v / (1 − β2^t)    # restores true scale in early steps
```

The correction vanishes as t grows (1 − 0.9^100 ≈ 1). It only matters in the first ~10 steps, but without it, Adam is sluggish at initialization.

### Adam in NumPy

```python
class Adam:
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr    = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps   = eps
        self.m     = {}    # first moment per parameter
        self.v     = {}    # second moment per parameter
        self.t     = 0     # step counter for bias correction

    def step(self, params, grads):
        self.t += 1
        for i, (p, g) in enumerate(zip(params, grads)):
            if i not in self.m:
                self.m[i] = np.zeros_like(p)
                self.v[i] = np.zeros_like(p)

            # Update running moments
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * g
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * g**2

            # Bias-corrected estimates
            m_hat = self.m[i] / (1 - self.beta1**self.t)
            v_hat = self.v[i] / (1 - self.beta2**self.t)

            # Adaptive step: scale by 1/sqrt(v) per parameter
            p -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
```

`eps=1e-8` prevents division by zero when v is near zero (sparse parameters early in training).

### Adam vs SGD: when each wins

| Scenario | Better choice | Reason |
|---|---|---|
| Deep networks, NLP, sparse inputs | Adam | Adapts LR per parameter automatically |
| CNNs, image classification | SGD + Momentum | Often generalizes better with tuned LR schedule |
| Fine-tuning a pretrained transformer | AdamW | Stable convergence, correct weight decay |
| Quick experiment, unknown problem | Adam | Works without careful tuning |

### AdamW: Adam with regularization done correctly

Standard Adam applies L2 regularization by adding it to the gradient:

```python
g_with_l2 = g + lambda_ * w   # L2 term added to gradient
```

But Adam then scales this by `1/sqrt(v)`. Weights with large gradients get *less* regularization than those with small gradients — the opposite of what L2 intends.

AdamW decouples weight decay from the adaptive scaling:

```python
# Adam step (adaptive gradient)
p -= lr * m_hat / (np.sqrt(v_hat) + eps)

# Weight decay applied directly, NOT scaled by 1/sqrt(v)
p -= lr * lambda_ * p
```

**Regularization here:** for transformer fine-tuning, always use AdamW. The correct weight decay keeps all weights small (L2 behavior) without distorting the per-parameter adaptation. Use L1 only if you need sparse weights — and L1 via AdamW requires manual implementation since frameworks don't include it.

**Interview answer:** "Adam adapts the learning rate per parameter based on gradient history. Sparse parameters get larger effective LR; dense ones get smaller. β1=0.9 and β2=0.999 are nearly always left at defaults. ε prevents division by zero for sparse parameters. I'd use AdamW over Adam whenever there's L2 regularization — standard Adam's weight decay interacts badly with the adaptive scaling."

---

## Problem 4 — Training diverges early or plateaus late → Learning rate schedules

### Why a fixed learning rate fails

- **Too high at the start:** random initialization means large gradients. A high LR overshoots every update and loss explodes.
- **Too high at the end:** once the model is close to a solution, a large LR keeps jumping over the minimum. The model oscillates around it instead of settling in.

The fix: start with a different LR than you finish with.

### Schedule 1: Warmup + cosine decay (modern standard)

Ramp LR up linearly for the first few percent of training, then decay smoothly:

```python
import numpy as np

def lr_schedule(step, warmup_steps, total_steps, max_lr, min_lr=0.0):
    if step < warmup_steps:
        # Linear warmup: 0 → max_lr
        return max_lr * step / warmup_steps
    else:
        # Cosine decay: max_lr → min_lr
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        return min_lr + 0.5 * (max_lr - min_lr) * (1 + np.cos(np.pi * progress))

# 10k warmup, 100k total steps, peak lr = 3e-4
for step in [0, 5_000, 10_000, 55_000, 100_000]:
    lr = lr_schedule(step, 10_000, 100_000, max_lr=3e-4)
    print(f"step {step:6d}  lr={lr:.6f}")
```

```
step      0  lr=0.000000   ← just started
step   5000  lr=0.000150   ← halfway through warmup
step  10000  lr=0.000300   ← peak
step  55000  lr=0.000150   ← halfway through cosine decay
step 100000  lr=0.000000   ← end of training
```

### Why large models need warmup

At initialization, a pretrained model's weights encode useful structure. A high LR immediately disrupts that — the model "forgets" pretraining in the first few batches.

Warmup also gives Adam's m and v time to build up reliable estimates before taking large steps. In the first few steps, Adam's bias-corrected moments are based on almost no history — unreliable. Large steps on unreliable statistics → divergence.

**Rule of thumb:** warmup_steps ≈ 5–10% of total_steps for transformer fine-tuning.

### Schedule 2: Step decay

Halve the LR every N epochs. Simple, still standard for CNNs:

```python
def step_decay(initial_lr, epoch, drop_factor=0.5, drop_every=10):
    return initial_lr * (drop_factor ** (epoch // drop_every))

# epoch 0:  lr = 0.1
# epoch 10: lr = 0.05
# epoch 20: lr = 0.025
```

### Applying a schedule in a training loop

```python
optimizer = Adam(lr=0.001)
total_steps = 1000

for step in range(total_steps):
    # Update LR before each step
    optimizer.lr = lr_schedule(step, warmup_steps=50,
                               total_steps=total_steps, max_lr=0.001)
    # ... rest of training loop
```

**Interview answer:** "I'd use warmup + cosine decay for any large model. Warmup prevents early instability and gives Adam time to calibrate its moment estimates. Cosine decay makes the final approach to the minimum smoother than a step drop. For smaller CNNs, step decay is practical and easier to reason about."

---

## Problem 5 — Loss spikes suddenly mid-training → Gradient clipping

### When gradients explode

Transformers and RNNs processing long sequences can produce enormous gradients from a single unusual batch. One bad step → catastrophic weight update → loss jumps from 0.3 to 1000.

### Clip the total norm, not individual gradients

Clipping each gradient independently distorts the update *direction*. The right approach clips the *total norm* of all gradients together — preserving direction, capping only magnitude:

```python
def clip_grad_norm(grads, max_norm=1.0):
    # Global norm across all parameters
    total_norm = np.sqrt(sum(np.sum(g**2) for g in grads))

    if total_norm > max_norm:
        scale = max_norm / (total_norm + 1e-6)
        grads = [g * scale for g in grads]

    return grads, total_norm   # return norm for monitoring
```

In PyTorch (call this after `.backward()`, before `.step()`):

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

### What max_norm to use

Start with 1.0. Log `total_norm` during training. If clipping activates on nearly every batch, the gradient norm distribution is pathologically wide — clipping is hiding a real problem (bad loss function, bad architecture, or LR too high). Clipping should handle *occasional* spikes, not be permanently active.

**Interview answer:** "Gradient clipping prevents one bad batch from destroying training. I always clip by global norm, not per-parameter, so the update direction is preserved. max_norm=1.0 is the standard starting point. If clipping fires every step, I investigate the root cause rather than just tightening the clip threshold."

---

## Putting it together: a complete training loop

```python
import numpy as np

def sigmoid(x):
    return np.where(x >= 0, 1/(1+np.exp(-x)), np.exp(x)/(1+np.exp(x)))

# Binary classification: 2 features, 500 examples
np.random.seed(0)
n = 500
X = np.random.randn(n, 2)
y = (X[:, 0] - X[:, 1] > 0).astype(float)

W = np.zeros(2)
b = np.array([0.0])

opt          = Adam(lr=0.01)
lambda_      = 0.01     # L2 weight decay (AdamW-style)
total_steps  = 1000

for step in range(total_steps):
    # ── LR schedule ──────────────────────────────────
    opt.lr = lr_schedule(step, warmup_steps=50,
                         total_steps=total_steps, max_lr=0.01)

    # ── Mini-batch sample ─────────────────────────────
    idx = np.random.choice(n, size=64, replace=False)
    Xb, yb = X[idx], y[idx]

    # ── Forward pass ──────────────────────────────────
    logits = Xb @ W + b[0]
    p      = np.clip(sigmoid(logits), 1e-7, 1 - 1e-7)

    # ── BCE loss ──────────────────────────────────────
    loss = -(yb * np.log(p) + (1 - yb) * np.log(1 - p)).mean()

    # ── Gradients ─────────────────────────────────────
    d_logit = (p - yb) / len(yb)
    d_W     = Xb.T @ d_logit               # shape (2,)
    d_b     = np.array([d_logit.sum()])    # shape (1,)

    # ── Gradient clipping ─────────────────────────────
    grads, gnorm = clip_grad_norm([d_W, d_b], max_norm=1.0)

    # ── Optimizer step ────────────────────────────────
    opt.step([W, b], grads)

    # ── Weight decay (AdamW-style, after the Adam step) ─
    W -= opt.lr * lambda_ * W

    if step % 200 == 0:
        acc = ((sigmoid(X @ W + b[0]) > 0.5) == y).mean()
        print(f"step {step:4d}  lr={opt.lr:.5f}  loss={loss:.4f}"
              f"  acc={acc:.3f}  gnorm={gnorm:.3f}")
```

```
step    0  lr=0.00000  loss=0.6931  acc=0.510  gnorm=0.051
step  200  lr=0.00917  loss=0.1823  acc=0.932  gnorm=0.312
step  400  lr=0.00823  loss=0.1241  acc=0.960  gnorm=0.198
step  600  lr=0.00588  loss=0.0998  acc=0.968  gnorm=0.143
step  800  lr=0.00293  loss=0.0912  acc=0.972  gnorm=0.098
```

Every concept from the article appears in this loop: mini-batch sampling, Adam update, LR schedule, gradient clipping, and decoupled weight decay.

---

## Debugging: when optimization fails

### Loss won't decrease at all

1. **Overfit one tiny batch** (8 examples). If the model can't memorize 8 examples, gradient computation is wrong.
2. **Print gradient norms.** Zero means gradients aren't flowing (dead ReLU, wrong computation graph). Enormous means LR is too high.
3. **Try Adam if using SGD.** Adam is more forgiving of LR choice and often works out of the box.
4. **Check the loss formula.** Manually compute the loss for one example by hand and verify it matches code.

### Loss decreases then plateaus early

1. LR is probably too low after warmup. Try 3× increase.
2. Momentum/Adam may have accumulated stale state. Reset optimizer state and restart.
3. Model capacity is too low. Add parameters.

### Loss oscillates without settling

1. LR too high — halve it.
2. Batch size too small — gradient estimates are too noisy. Increase batch size or add momentum.
3. Adam β2 too low — v doesn't smooth enough. Try β2=0.9999.

### Loss spikes and partially recovers

1. Add gradient clipping (max_norm=1.0).
2. Inspect the highest-loss examples — mislabeled or corrupted data can cause large gradient spikes.
3. Check whether the spike correlates with a LR schedule restart.

### Trains well but validation loss diverges (overfitting)

1. Add L2 weight decay. Use AdamW for transformer models.
2. For models with many irrelevant features: switch to L1 (zero out weak features).
3. Reduce model capacity or add dropout.

---

## Interview framework: "why this optimizer / schedule?"

When asked to justify optimization choices, answer in order:

**1. Optimizer:** "I'd default to Adam (lr=0.001). For CNNs where generalization matters, I'd consider SGD + Momentum with a step decay schedule — it often generalizes better with more tuning. For any transformer, AdamW."

**2. Learning rate:** "I'd look at prior work for the architecture and task. If starting blind, I'd do a brief LR range test: train for one epoch while increasing LR from 1e-6 to 1e-1 and find where loss stops falling."

**3. Schedule:** "Warmup + cosine decay for large models or fine-tuning. Step decay for smaller models where I want explicit control. I'd set warmup to ~5% of total steps."

**4. Gradient clipping:** "I always add clipping (max_norm=1.0) for sequence models. For feedforward nets I'd add it reactively if I see loss spikes."

**5. Regularization:** "L2 via AdamW weight decay for general overfitting. L1 if I want automatic feature selection or need to explain which inputs drive the model. Elastic net (L1 + L2) if I need both stability and sparsity."
