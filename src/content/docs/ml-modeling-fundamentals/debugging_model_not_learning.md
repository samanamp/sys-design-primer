---
title: "Debugging a Model That Is Not Learning: A 1-Hour Interview Session"
description: "A practical one-hour debugging playbook for ML modeling interviews: data skew, labels, LR, normalization, loss bugs, leakage, imbalance, overfitting, and gradients."
---

# Debugging a Model That Is Not Learning: A 1-Hour Interview Session

Companion notebook: [debugging_model_not_learning_colab.ipynb](/notebooks/debugging_model_not_learning_colab.ipynb)

When a model is not learning, the worst response is random architecture tweaking.

The right response:

> Reduce the problem until it should work, then find which assumption is false.

For autonomous driving simulation, the bug is often not the neural network. It is data skew, label noise, coordinate frames, timestamp alignment, leakage, normalization, or a loss that rewards the wrong behavior.

## 0. One-hour plan

```text
0-10 min   The debugging mindset
10-20 min  Overfit-one-batch test
20-35 min  Data, labels, normalization, leakage
35-45 min  LR, gradients, loss bugs
45-55 min  Overfitting vs underfitting diagnosis
55-60 min  Interview drills
```

---

## 1. The debugging mindset

Training minimizes:

$$
\min_\theta
\frac{1}{N}
\sum_{i=1}^N
\mathcal{L}(f_\theta(x_i),y_i)
$$

If learning fails, inspect each piece:

```text
x_i      inputs
y_i      labels
f_theta  model
L        loss
optimizer update
eval     metric
```

Do not start with "make the model bigger." Start with:

> Can this model overfit one small batch?

If not, there is a bug or mismatch.

---

## 2. Overfit-one-batch test

Take 8-32 examples. Train only on them. The model should drive training loss very low.

If it cannot:

- labels may be wrong,
- target shape may be wrong,
- loss may be wrong,
- learning rate may be bad,
- gradients may be missing,
- model may be in eval mode,
- inputs may be normalized incorrectly.

This is the most useful interview debugging move.

```text
Full training fails
       |
       v
Can overfit one batch?
       |
       +-- no  -> bug in local training setup
       |
       +-- yes -> data scale, generalization, imbalance, eval mismatch
```

---

## 3. Data and label checks

For autonomous driving:

- Are positions in meters or centimeters?
- Are headings radians or degrees?
- Are trajectories in ego frame or world frame?
- Are labels aligned with timestamps?
- Are map features from the correct time/version?
- Are traffic lights current or future?
- Are rare labels reliable?

Bad label examples:

- future trajectory shifted by one timestep,
- actor IDs swapped,
- traffic light state from the future,
- map lane ID mismatched,
- cut-in label based on a heuristic with many false positives.

Data skew:

```text
99% lane following
1% rare interaction
```

The model may learn normal driving well and fail the interview-relevant rare cases.

---

## 4. Normalization problems

Normalization bugs are common in robotics and driving.

Check:

- feature means and standard deviations,
- train vs eval normalization,
- units,
- clipping,
- missing-value encoding,
- coordinate frame transforms.

Example:

```text
training uses ego-frame positions
inference uses world-frame positions
```

The model may appear not to learn because inputs are inconsistent.

Print:

```python
x.mean(dim=0), x.std(dim=0), x.min(), x.max()
```

For trajectories, plot them. Visual inspection catches bugs metrics hide.

---

## 5. Learning rate and gradients

Optimizer update:

$$
\theta_{t+1} =
\theta_t - \eta\nabla_\theta\mathcal{L}
$$

where $\eta$ is learning rate.

Symptoms:

```text
LR too high:
  loss explodes or oscillates

LR too low:
  loss barely changes

zero gradients:
  detach, no requires_grad, wrong branch, saturated activation

exploding gradients:
  huge norms, NaN, unstable loss
```

Track gradient norm:

$$
\|\nabla_\theta \mathcal{L}\|_2
$$

If gradient norm is zero, inspect graph wiring. If huge, lower LR, clip gradients, or inspect loss scale.

---

## 6. Loss function problems

The model may be learning exactly what the loss asks for, but the loss asks for the wrong thing.

Examples:

- MSE on multi-modal futures creates average trajectories.
- Unweighted CE ignores rare classes.
- Loss ignores collision/offroad.
- Regression loss scale dominates classification loss.
- Mode probability loss too weak.
- Labels are continuous but treated as class IDs.

For simulation, compare training loss to actual metrics:

```text
loss down, ADE down, collision up -> loss missing safety validity
loss down, rare recall flat       -> imbalance issue
loss down, diversity down         -> mode collapse
```

---

## 7. Data leakage

Data leakage means the model sees information during training that will not be available at inference.

Driving examples:

- future positions as input,
- future traffic light state,
- labels encoded in scenario metadata,
- same scene in train and eval split,
- route derived from future behavior,
- map annotations unavailable at runtime.

Leakage often causes suspiciously good validation performance and bad real-world performance.

Ask:

> At inference time, would this feature be known?

---

## 8. Overfitting vs underfitting

```text
Train loss low, eval bad:
  overfitting, leakage, distribution shift, weak regularization

Train loss high, eval bad:
  underfitting, optimization bug, bad data, weak model, wrong loss

Train loss low, eval metric bad:
  loss-metric mismatch
```

For driving, also slice eval:

- geography,
- weather,
- agent type,
- maneuver,
- speed,
- rare scenario,
- long-tail interactions.

Aggregate metrics hide failures.

---

## 9. Minimal PyTorch debugging snippet

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

def grad_norm(model):
    total = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total += p.grad.detach().pow(2).sum()
    return total.sqrt()

class TinyMLP(nn.Module):
    def __init__(self, d, c):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, 64),
            nn.ReLU(),
            nn.Linear(64, c),
        )
    def forward(self, x):
        return self.net(x)

B, D, C = 16, 8, 3
x = torch.randn(B, D)
y = torch.randint(0, C, (B,))

model = TinyMLP(D, C)
opt = torch.optim.Adam(model.parameters(), lr=1e-2)

for step in range(200):
    opt.zero_grad()
    logits = model(x)
    loss = F.cross_entropy(logits, y)
    loss.backward()
    g = grad_norm(model)
    opt.step()
```

Expected: one batch should overfit. If not, debug locally before training at scale.

---

## 10. Common interview questions and strong answers

**Q: First thing you do when a model is not learning?**  
A: Overfit one small batch. If that fails, the issue is local: data, labels, loss, optimizer, gradients, or model wiring.

**Q: How do you detect learning-rate issues?**  
A: Too high causes divergence or oscillation. Too low causes very slow loss movement. I would run a small LR sweep and inspect gradient norms.

**Q: What AD-specific bugs do you check?**  
A: Coordinate frames, timestamp alignment, actor IDs, traffic light timing, map validity, future leakage, rare-label quality.

**Q: How do you distinguish overfitting and underfitting?**  
A: Compare train and eval loss. Low train/high eval suggests overfitting or shift. High train/high eval suggests underfitting or optimization/data problems.

---

## 11. A 60-second explanation you can say out loud

If a model is not learning, I start by trying to overfit one small batch. If that fails, I look for local bugs: bad labels, wrong target shape, wrong loss, learning rate, missing gradients, eval mode, or normalization. If it can overfit one batch but fails broadly, I inspect data skew, class imbalance, train/eval shift, leakage, and whether the loss matches the metric. For autonomous driving, I specifically check coordinate frames, timestamp alignment, traffic light timing, map features, actor IDs, and rare-event label quality.

---

## 12. Practice exercises with answers

**Exercise 1:** Model cannot overfit one batch. Name four likely causes.  
**Answer:** Bad labels, wrong loss, target shape bug, LR issue, missing gradients, eval mode, normalization bug.

**Exercise 2:** Train loss low, eval collision high. What might be wrong?  
**Answer:** Loss does not penalize collisions, eval distribution differs, or model overfits common behavior.

**Exercise 3:** Give one driving-specific data leakage example.  
**Answer:** Future traffic light state or future agent trajectory included as input.

**Exercise 4:** Loss decreases but rare cut-in recall stays bad. What do you check?  
**Answer:** Class imbalance, label quality, per-class loss, sampling strategy, weighted/focal loss, and thresholding.

