---
title: "Custom Losses for Autonomous Driving Simulation"
description: "Interview-focused guide to imbalanced loss, weighted cross entropy, focal loss, multi-modal trajectory losses, and why MSE fails for multi-modal future prediction."
---

# Custom Losses for Autonomous Driving Simulation

## 1. Interview-level intuition

A loss function tells the model what mistakes are expensive. In autonomous driving simulation, not all mistakes are equal. Missing a rare cut-in, pedestrian, red-light runner, or near-collision matters more than misclassifying another normal lane-following frame.

Custom losses usually solve one of two problems:

1. **Class imbalance:** rare but important events are underrepresented.
2. **Multi-modality:** there are many plausible futures, not one average future.

For trajectory prediction, the second point is critical. If a vehicle at an intersection might turn left or go straight, minimizing plain MSE against one logged future can teach the model to predict an impossible average path between the two.

## 2. Mathematical formulation

### Weighted cross entropy

For classification with classes $c \in \{1,\dots,C\}$:

$$
\mathcal{L}_{WCE} = -w_y \log p_y
$$

where:

- $y$ is the true class.
- $p_y$ is the predicted probability of the true class.
- $w_y$ is a class weight.

A common choice:

$$
w_c \propto \frac{1}{\text{frequency}(c)}
$$

But raw inverse frequency can overweight noisy rare classes, so weights are often clipped or smoothed.

### Focal loss

Focal loss downweights easy examples:

$$
\mathcal{L}_{focal} = -\alpha_y (1 - p_y)^\gamma \log p_y
$$

where:

- $\alpha_y$ handles class weighting.
- $\gamma \ge 0$ controls focus on hard examples.
- If $p_y$ is high, $(1-p_y)^\gamma$ is small.

When $\gamma = 0$, focal loss becomes weighted cross entropy.

### Multi-modal trajectory loss

Suppose the model predicts $K$ possible future trajectories:

$$
\hat{Y}^{(1)}, \dots, \hat{Y}^{(K)}
$$

and mode probabilities:

$$
\pi_1, \dots, \pi_K
$$

Each trajectory is:

$$
\hat{Y}^{(k)} \in \mathbb{R}^{T \times 2}
$$

where $T$ is future timesteps and each point is $(x,y)$.

A common winner-takes-all loss:

$$
k^* = \arg\min_k \|\hat{Y}^{(k)} - Y\|_2^2
$$

$$
\mathcal{L} = \|\hat{Y}^{(k^*)} - Y\|_2^2 - \lambda \log \pi_{k^*}
$$

This says: one mode should match the logged future, and the model should assign that mode high probability.

### Why MSE is bad for multi-modal futures

If two futures are equally likely:

$$
Y_1 = \text{turn left}, \quad Y_2 = \text{go straight}
$$

MSE learns the conditional mean:

$$
\hat{Y} = \mathbb{E}[Y|X]
$$

The mean of "turn left" and "go straight" may be a path into the curb or lane divider. Low MSE can still be physically unrealistic.

## 3. Why this matters for autonomous driving simulation

Simulation models often generate future agent behavior. The world is inherently multi-modal:

- A pedestrian may wait or cross.
- A car may yield or merge.
- A cyclist may continue or turn.
- A parked car may stay parked or pull out.

For simulation, realism and diversity matter. A model that predicts the average behavior can produce safe-looking but useless scenarios. It may erase rare high-risk behaviors that are exactly what a simulation team wants.

Loss design affects:

- Rare event recall.
- Scenario diversity.
- Realism.
- Safety-critical behavior coverage.
- Controllability.
- Evaluation stability.

## 4. Common interview questions and strong answers

**Q: When would you use weighted cross entropy?**  
A: When classes are imbalanced and false negatives on rare classes matter. I would start with smoothed inverse-frequency weights, then validate per-class precision/recall because overweighting noisy rare labels can hurt calibration.

**Q: How is focal loss different from class weighting?**  
A: Class weighting changes importance by class. Focal loss changes importance by difficulty. Easy correctly classified examples get downweighted, so training focuses more on hard or misclassified cases.

**Q: Why is MSE bad for trajectory prediction?**  
A: MSE estimates the conditional mean. In multi-modal futures, the mean of valid futures can be invalid. For autonomous driving, averaging left-turn and straight futures may produce a trajectory no vehicle would actually take.

**Q: How do you model multi-modal futures?**  
A: Predict multiple trajectory modes plus probabilities, then use a min-over-modes regression loss and a classification loss on the winning mode. More probabilistic versions use mixture density losses.

## 5. Minimal NumPy or PyTorch implementation

```python
import torch
import torch.nn.functional as F

def weighted_cross_entropy(logits, targets, class_weights):
    # logits: [B, C], targets: [B], class_weights: [C]
    return F.cross_entropy(logits, targets, weight=class_weights)

def focal_loss(logits, targets, alpha=None, gamma=2.0):
    # logits: [B, C], targets: [B]
    log_probs = F.log_softmax(logits, dim=-1)
    probs = log_probs.exp()

    pt = probs[torch.arange(targets.numel()), targets]
    log_pt = log_probs[torch.arange(targets.numel()), targets]

    if alpha is None:
        at = 1.0
    else:
        at = alpha[targets]

    loss = -at * (1.0 - pt).pow(gamma) * log_pt
    return loss.mean()

def multimodal_trajectory_loss(pred_trajs, mode_logits, target, cls_weight=1.0):
    """
    pred_trajs: [B, K, T, 2]
    mode_logits: [B, K]
    target: [B, T, 2]
    """
    diff = pred_trajs - target[:, None, :, :]
    sq_error = (diff ** 2).sum(dim=(-1, -2))  # [B, K]
    best_mode = sq_error.argmin(dim=1)        # [B]

    reg_loss = sq_error[torch.arange(target.size(0)), best_mode].mean()
    cls_loss = F.cross_entropy(mode_logits, best_mode)
    return reg_loss + cls_weight * cls_loss
```

## 6. Failure modes and debugging checklist

- Class weights too large: rare noisy labels dominate training.
- Focal loss gamma too high: model ignores many useful examples.
- Multi-modal modes collapse: all predicted trajectories look the same.
- Mode probabilities are uncalibrated: good trajectory exists but low probability.
- MSE produces averaged invalid trajectories.
- Training loss improves but scenario diversity drops.
- Evaluation only uses minADE and ignores probability ranking.

Checklist:

- Inspect per-class precision/recall.
- Plot predicted trajectories for ambiguous scenes.
- Track mode entropy.
- Track best-mode index distribution.
- Compare minADE and probability-weighted metrics.
- Overfit a tiny balanced batch.
- Check whether rare labels are noisy.

## 7. A 60-second explanation I can say out loud

Custom losses encode what mistakes matter. In autonomous driving, ordinary cross entropy or MSE can be wrong because data is imbalanced and futures are multi-modal. Weighted cross entropy increases the cost of rare classes. Focal loss focuses training on hard examples by downweighting easy ones. For trajectory prediction, MSE predicts the average future, which can be physically invalid when multiple futures are plausible. A better setup predicts multiple trajectory modes and probabilities, then trains the closest mode to match the logged future while also teaching the model to assign that mode high probability.

## 8. 3 practice exercises with answers

**Exercise 1:** If $p_y=0.9$, $\gamma=2$, what is focal loss's modulating factor?  
**Answer:** $(1 - 0.9)^2 = 0.01$. The easy example is downweighted heavily.

**Exercise 2:** Why might inverse-frequency weighting hurt?  
**Answer:** Very rare classes may have noisy labels. Huge weights can make the model chase noise and hurt calibration.

**Exercise 3:** A model predicts one future for a car that may turn left or go straight. Why can MSE fail?  
**Answer:** MSE learns the mean of possible futures. The average of left and straight may cut across lanes and be unrealistic.

