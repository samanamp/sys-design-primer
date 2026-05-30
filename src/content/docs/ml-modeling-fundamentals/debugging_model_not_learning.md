---
title: "Debugging a Model That Is Not Learning"
description: "Interview-focused debugging playbook for ML models that fail to learn: data skew, labels, learning rate, normalization, loss, leakage, imbalance, overfitting, and gradient inspection."
---

# Debugging a Model That Is Not Learning

## 1. Interview-level intuition

When a model is not learning, do not randomly change architecture. Debug systematically.

The failure is usually in one of five places:

1. Data.
2. Labels.
3. Loss.
4. Optimization.
5. Evaluation.

For autonomous driving simulation, data issues are especially common: rare events, skewed logs, bad labels, map mismatch, inconsistent coordinate frames, and distribution shifts between train and eval.

The first rule:

> Before making the model bigger, prove the current model can overfit one small batch.

## 2. Mathematical formulation

Training minimizes empirical risk:

$$
\min_\theta \frac{1}{N}\sum_{i=1}^{N}\mathcal{L}(f_\theta(x_i), y_i)
$$

If loss does not decrease, one of these may be wrong:

- $x_i$: input features are bad.
- $y_i$: labels are bad.
- $f_\theta$: model is broken or too weak.
- $\mathcal{L}$: loss does not match the task.
- optimizer update:

$$
\theta_{t+1} = \theta_t - \eta \nabla_\theta \mathcal{L}
$$

where $\eta$ is learning rate.

If $\eta$ is too small, learning is slow. If too large, loss diverges.

Gradient norm:

$$
\|\nabla_\theta \mathcal{L}\|_2
$$

helps detect vanishing, exploding, or missing gradients.

## 3. Why this matters for autonomous driving simulation

Simulation models can fail silently:

- Predict all agents go straight because rare turns are underrepresented.
- Ignore traffic lights because labels are noisy.
- Generate offroad futures because map coordinates are misaligned.
- Overfit logs but fail on rare interactions.
- Look good on ADE but fail collision metrics.

A modeling interview may test whether you can debug from first principles rather than recite architectures.

## 4. Common interview questions and strong answers

**Q: What is the first thing you try if a model is not learning?**  
A: Overfit one small batch. If it cannot drive training loss near zero, there is likely a bug in data, labels, loss, model wiring, or optimizer.

**Q: How do you detect learning rate problems?**  
A: If loss diverges or oscillates wildly, LR may be too high. If loss barely changes and gradients are nonzero, LR may be too low. I would run an LR sweep.

**Q: How do you check class imbalance?**  
A: Print class frequencies, per-class loss, confusion matrix, and precision/recall. For driving, slice by rare maneuvers and agent types.

**Q: What is data leakage?**  
A: Train features contain information unavailable at inference, like future trajectory, future traffic light state, or labels encoded in metadata.

## 5. Minimal NumPy or PyTorch implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class TinyMLP(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(),
            nn.Linear(64, out_dim),
        )

    def forward(self, x):
        return self.net(x)

def grad_norm(model):
    total = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total += p.grad.detach().pow(2).sum()
    return total.sqrt()

# Overfit-one-batch test
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
    gnorm = grad_norm(model)
    opt.step()

    if step % 50 == 0:
        acc = (logits.argmax(dim=-1) == y).float().mean()
        print(step, float(loss), float(acc), float(gnorm))
```

Expected: on one batch, loss should drop and accuracy should approach 1.0. If not, debug.

## 6. Failure modes and debugging checklist

### Data skew

- Print distributions.
- Slice by scene type, agent type, time, geography.
- Check train/eval mismatch.

### Bad labels

- Visualize random examples.
- Inspect high-loss examples.
- Check label timestamp alignment.
- Verify coordinate frames.

### Learning rate problems

- Run LR sweep.
- Check loss curves.
- Inspect gradient norms.

### Normalization problems

- Print feature mean/std.
- Check units: meters vs centimeters, radians vs degrees.
- Verify train/inference normalization match.

### Bad loss function

- MSE for multi-modal futures.
- Unweighted CE under severe imbalance.
- Loss ignores safety-critical outputs.

### Data leakage

- Remove future-only features.
- Check split by scene/time/geography.
- Ensure map and traffic-light states are timestamp-correct.

### Overfitting / underfitting

- Train loss low, eval bad: overfitting or distribution shift.
- Both train and eval bad: underfitting, bad optimization, or bad data.

### Gradient inspection

- Check zero gradients.
- Check exploding gradients.
- Verify all intended parameters require gradients.
- Confirm model is in train mode.

## 7. A 60-second explanation I can say out loud

If a model is not learning, I debug systematically. First I try to overfit one small batch. If that fails, I suspect a bug in data, labels, loss, optimizer, or model wiring. Then I inspect data distributions, label correctness, normalization, class imbalance, and gradient norms. I check learning rate by looking for divergence or no movement. For autonomous driving, I pay special attention to coordinate frames, timestamp alignment, future-information leakage, rare event imbalance, and whether the loss matches multi-modal driving behavior.

## 8. 3 practice exercises with answers

**Exercise 1:** The model cannot overfit one batch. Name three likely causes.  
**Answer:** Bad labels, wrong loss/target shape, learning rate issue, missing gradients, model not in train mode, input normalization bug.

**Exercise 2:** Train loss is low but eval collision rate is high. What might be wrong?  
**Answer:** Overfitting, train/eval distribution shift, loss does not penalize collisions, or eval contains rare interactions underrepresented in training.

**Exercise 3:** What is an autonomous-driving-specific leakage example?  
**Answer:** Using future agent positions, future traffic light states, or log-derived labels unavailable at inference as input features.

