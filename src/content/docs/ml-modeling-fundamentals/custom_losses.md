---
title: "Custom Losses: A 1-Hour Interview Learning Session"
description: "A practical one-hour lesson on weighted cross entropy, focal loss, multi-modal trajectory losses, and custom loss debugging for autonomous driving simulation and robotics interviews."
---

# Custom Losses: A 1-Hour Interview Learning Session

Companion notebook: [custom_losses_colab.ipynb](/notebooks/custom_losses_colab.ipynb)

This lesson is for a Waymo-style ML Modeling & Fundamentals interview where you may need to explain a loss from first principles and implement a small PyTorch snippet in CoderPad.

You already know cross entropy and KL. The missing piece is not the formula. The missing piece is judgment:

> What behavior do I want the model to learn, and what does my loss accidentally reward?

For autonomous driving simulation, that question matters because logs are imbalanced, futures are multi-modal, and the rare cases are often the cases we care about most.

## 0. One-hour plan

```text
0-10 min   What a loss really does: incentives, not just math
10-20 min  Imbalance: weighted cross entropy
20-30 min  Hard examples: focal loss
30-45 min  Multi-modal futures: why MSE fails and what replaces it
45-55 min  Implementation patterns and debugging
55-60 min  Interview answers and practice drills
```

By the end, you should be able to:

- Explain why ordinary losses fail under imbalance and multi-modality.
- Derive weighted cross entropy and focal loss from cross entropy.
- Explain why MSE predicts the conditional mean.
- Implement focal loss and a multi-modal trajectory loss in PyTorch.
- Discuss tradeoffs: recall, calibration, realism, diversity, controllability, and latency.

---

## 1. The core idea: loss functions create incentives

Suppose a model predicts whether a scene contains a dangerous cut-in.

Dataset:

```text
normal lane-following: 99,000 examples
dangerous cut-in:      1,000 examples
```

A model that predicts "normal" for everything gets 99% accuracy. That is useless for safety.

The problem is not that cross entropy is mathematically wrong. The problem is that the empirical training distribution says normal examples dominate. If every example has equal weight, the optimizer spends most of its effort improving common cases.

In autonomous driving and robotics, the important mistakes are often:

- Rare pedestrian crossing.
- Aggressive merge.
- Cut-in.
- Red-light runner.
- Cyclist swerving.
- Vehicle reversing.
- Unprotected left turn.
- Construction-zone behavior.

Custom losses are how we say:

> This type of mistake should count more, or this kind of output structure should be rewarded.

---

## 2. Weighted cross entropy

### Interview-level intuition

Weighted cross entropy is ordinary cross entropy where some classes matter more.

For one example with true class $y$:

$$
\mathcal{L}_{CE} = -\log p_y
$$

Weighted cross entropy:

$$
\mathcal{L}_{WCE} = -w_y \log p_y
$$

where:

- $p_y$ is predicted probability for the true class.
- $w_y$ is the weight for the true class.

If cut-in examples are rare, assign them a larger weight. The gradient from a cut-in example becomes larger, so the optimizer pays more attention.

### Why you need to care

In driving simulation, if your model underpredicts rare events, your simulator becomes too easy. The autonomy stack passes simulated tests but fails on long-tail real-world situations.

Weighted CE is useful when:

- The label space is discrete.
- Rare classes are important.
- You can tolerate some calibration distortion.
- You care about recall for minority classes.

### How to choose weights

Naive inverse frequency:

$$
w_c = \frac{1}{f_c}
$$

where $f_c$ is class frequency.

But this can explode for rare noisy classes. Safer variants:

```text
1. normalized inverse frequency
2. sqrt inverse frequency
3. clipped weights
4. manually chosen business/safety weights
```

Example:

```text
normal:  99%
cut-in:   1%

raw inverse ratio: cut-in gets 99x weight
maybe too high

safer: cut-in gets 5x or 10x, then validate recall/calibration
```

### Mathematical effect

For logits $z$ and softmax probabilities $p$, cross entropy gradient is:

$$
\frac{\partial \mathcal{L}}{\partial z_j} = p_j - \mathbf{1}[j=y]
$$

Weighted CE scales it:

$$
\frac{\partial \mathcal{L}_{WCE}}{\partial z_j}
=
w_y(p_j - \mathbf{1}[j=y])
$$

So class weighting directly scales the update for examples of class $y$.

### Tradeoffs

Pros:

- Simple.
- Easy to implement.
- Strong baseline for imbalance.
- Good interview answer.

Cons:

- Can hurt probability calibration.
- Can overfit rare noisy labels.
- Does not distinguish easy vs hard examples.
- Requires tuning weights.

---

## 3. Focal loss

### Interview-level intuition

Weighted CE says: "some classes matter more."

Focal loss says: "hard examples matter more."

In a huge driving dataset, many examples are easy:

- Empty lane.
- Stationary parked vehicle.
- Normal car following.

Once the model already predicts those correctly, continuing to spend lots of gradient budget on them is wasteful. Focal loss downweights easy examples and focuses training on examples the model currently struggles with.

### Formula

Cross entropy:

$$
\mathcal{L}_{CE} = -\log p_y
$$

Focal loss:

$$
\mathcal{L}_{focal}
=
-\alpha_y(1-p_y)^\gamma \log p_y
$$

where:

- $p_y$ is probability assigned to the true class.
- $\alpha_y$ is optional class weight.
- $\gamma$ controls how aggressively easy examples are downweighted.

If $p_y=0.95$ and $\gamma=2$:

$$
(1-p_y)^\gamma = 0.05^2 = 0.0025
$$

The example contributes almost nothing.

If $p_y=0.2$:

$$
(1-p_y)^2 = 0.64
$$

The hard example still matters.

### Why you need to care

In perception, prediction, and simulation classifiers, easy background examples can dominate. Focal loss was popularized for dense object detection, where most anchors are background. The same idea appears in autonomous driving whenever easy negatives overwhelm rare positives.

Examples:

- "Is this actor likely to cut in?"
- "Is this pedestrian about to cross?"
- "Is this generated scenario invalid?"
- "Is this object a rare class?"
- "Is this trajectory mode safety-critical?"

### Focal loss vs weighted CE

```text
Weighted CE:
  class-level importance
  rare class gets larger gradient

Focal loss:
  example-level difficulty
  easy examples get smaller gradient

Weighted focal:
  both class importance and difficulty
```

### Tradeoffs

Pros:

- Useful when easy examples dominate.
- Improves attention to hard positives/negatives.
- Often improves minority recall.

Cons:

- Adds hyperparameter $\gamma$.
- Can under-train easy-but-important examples if too aggressive.
- Can hurt calibration.
- Hard examples may include mislabeled data.

Interview warning:

> Focal loss focuses on hard examples, but hard examples are not always valuable. Some are just bad labels.

---

## 4. MSE and the multi-modal future problem

### Interview-level intuition

MSE is fine when the target is roughly unimodal. It is bad when many futures are plausible.

At an intersection, a car may:

1. go straight,
2. turn left,
3. slow down,
4. yield.

If the model predicts one trajectory and trains with MSE, the optimal prediction is the average future.

For driving, the average of valid futures can be invalid.

```text
Future A: turn left
Future B: go straight
MSE average: cuts diagonally through the intersection
```

### Mathematical reason

MSE loss:

$$
\mathcal{L}(\hat{y}) = \mathbb{E}[(Y-\hat{y})^2|X=x]
$$

The minimizer is:

$$
\hat{y}^* = \mathbb{E}[Y|X=x]
$$

So MSE learns the conditional mean.

That is perfect if the conditional distribution is one blob. It is bad if the distribution has multiple modes.

### Toy example

Suppose:

$$
Y =
\begin{cases}
-1 & \text{with probability } 0.5 \\
1 & \text{with probability } 0.5
\end{cases}
$$

The MSE-optimal prediction is:

$$
\mathbb{E}[Y] = 0
$$

But $0$ is never observed. In trajectory terms, this is the impossible average trajectory.

### Why you need to care

Simulation needs plausible futures, not average futures. A simulator that averages away rare maneuvers will:

- under-test planners,
- reduce long-tail coverage,
- generate boring scenes,
- miss interaction diversity,
- produce physically invalid trajectories.

---

## 5. Multi-modal trajectory losses

### Setup

Instead of predicting one trajectory, predict $K$ modes:

$$
\hat{Y}^{(1)}, \dots, \hat{Y}^{(K)}
$$

and probabilities:

$$
\pi_1, \dots, \pi_K
$$

Each trajectory:

$$
\hat{Y}^{(k)} \in \mathbb{R}^{T \times 2}
$$

where $T$ is future timesteps.

### Winner-takes-all loss

Find the mode closest to the logged future:

$$
k^* = \arg\min_k \sum_{t=1}^T
\|\hat{Y}^{(k)}_t - Y_t\|_2^2
$$

Regression loss:

$$
\mathcal{L}_{reg}
=
\sum_{t=1}^T
\|\hat{Y}^{(k^*)}_t - Y_t\|_2^2
$$

Mode classification loss:

$$
\mathcal{L}_{mode} = -\log \pi_{k^*}
$$

Combined:

$$
\mathcal{L}
=
\mathcal{L}_{reg}
+
\lambda \mathcal{L}_{mode}
$$

### What this teaches

This loss says:

1. At least one mode should match the logged future.
2. The model should assign high probability to that matching mode.
3. Other modes are free to cover other plausible futures.

### Why this is not perfect

The logged future is only one sample from the real future distribution. If the car went straight in the log, a left turn might still have been plausible. Winner-takes-all loss may not reward that left-turn mode unless the dataset contains similar scenes where the car turned left.

This is why trajectory prediction evaluation often uses:

- minADE: did any mode match?
- minFDE: did any final point match?
- miss rate: did all modes miss?
- probability-aware metrics: did the model rank the right mode high?
- realism metrics: are other modes valid?

---

## 6. Mixture density loss

A more probabilistic approach is to predict a mixture distribution:

$$
p(Y|X) =
\sum_{k=1}^K
\pi_k \mathcal{N}(Y; \mu_k, \Sigma_k)
$$

where:

- $\pi_k$ is probability of mode $k$.
- $\mu_k$ is mean trajectory for mode $k$.
- $\Sigma_k$ is uncertainty.

Negative log likelihood:

$$
\mathcal{L}_{NLL}
=
-\log
\sum_{k=1}^K
\pi_k \mathcal{N}(Y; \mu_k, \Sigma_k)
$$

Why use it:

- More probabilistic.
- Can model uncertainty.
- Encourages calibrated probabilities.

Why it is harder:

- Numerical stability.
- Covariance parameterization.
- Mode collapse.
- Overconfident tiny variances.

In interviews, winner-takes-all multi-modal loss is usually easier to explain and implement. Mixture density loss is a good extension if asked.

---

## 7. Custom losses beyond driving

The same principles apply broadly.

Robotics:

- Grasp success is imbalanced.
- Contact-rich behavior is multi-modal.
- Imitation learning with MSE can average actions and fail.

Medical ML:

- Rare disease detection needs class weighting or focal loss.
- False negatives may be much worse than false positives.

Fraud:

- Rare positives dominate business value.
- Focal loss can help with many easy negatives.

Recommendation:

- Weighted losses encode business value.
- Pairwise/listwise losses may better match ranking objectives.

General lesson:

> A good custom loss aligns the training signal with the real decision cost and output structure.

---

## 8. Implementation patterns

### Weighted cross entropy

```python
import torch
import torch.nn.functional as F

def weighted_cross_entropy(logits, targets, class_weights):
    return F.cross_entropy(logits, targets, weight=class_weights)
```

### Focal loss

```python
def focal_loss(logits, targets, alpha=None, gamma=2.0):
    log_probs = F.log_softmax(logits, dim=-1)
    probs = log_probs.exp()

    row = torch.arange(targets.numel(), device=targets.device)
    pt = probs[row, targets]
    log_pt = log_probs[row, targets]

    if alpha is None:
        alpha_t = 1.0
    else:
        alpha_t = alpha[targets]

    loss = -alpha_t * (1.0 - pt).pow(gamma) * log_pt
    return loss.mean()
```

### Multi-modal trajectory loss

```python
def multimodal_trajectory_loss(pred_trajs, mode_logits, target, cls_weight=1.0):
    """
    pred_trajs: [B, K, T, 2]
    mode_logits: [B, K]
    target: [B, T, 2]
    """
    diff = pred_trajs - target[:, None, :, :]
    sq_error = (diff ** 2).sum(dim=(-1, -2))  # [B, K]
    best_mode = sq_error.argmin(dim=1)        # [B]

    reg_loss = sq_error[torch.arange(target.size(0), device=target.device), best_mode].mean()
    cls_loss = F.cross_entropy(mode_logits, best_mode)
    return reg_loss + cls_weight * cls_loss
```

### CoderPad tip

In an interview, first print shapes:

```python
print(pred_trajs.shape)  # [B, K, T, 2]
print(target.shape)      # [B, T, 2]
```

Most loss bugs are shape bugs.

---

## 9. Debugging checklist

### Weighted CE

- Are class weights on the same device as logits?
- Are weights too extreme?
- Did minority recall improve?
- Did precision collapse?
- Did calibration get worse?

### Focal loss

- Is $\gamma=0$ equivalent to weighted CE?
- Are hard examples actually valid labels?
- Is the loss ignoring too many easy examples?
- Did rare recall improve?
- Did probability calibration degrade?

### Multi-modal trajectory loss

- Are all modes collapsing to the same trajectory?
- Is one mode always winning?
- Are mode probabilities calibrated?
- Does minADE improve while probability-aware metrics get worse?
- Are trajectories physically valid?
- Do they stay on map?
- Do they collide unrealistically?

### General custom loss

- Can the model overfit one small batch?
- Does each term have comparable scale?
- Does the loss decrease while the real metric does not?
- Are units consistent: meters, seconds, radians?
- Are labels noisy or multi-modal?

---

## 10. Common interview questions and strong answers

**Q: Why use weighted cross entropy?**  
A: Because the empirical distribution may not match the cost distribution. In driving, rare cut-ins or pedestrian events may be more important than common normal driving. Weighting scales gradients for those examples.

**Q: Why not always use huge rare-class weights?**  
A: Rare labels may be noisy, and large weights can overfit noise or destroy calibration. I would tune weights and inspect precision/recall by class.

**Q: What does focal loss add beyond class weighting?**  
A: It downweights easy examples based on the model's current confidence. Class weighting is class-level; focal loss is example-difficulty-level.

**Q: Why is MSE bad for future prediction?**  
A: MSE learns the conditional mean. If futures are multi-modal, the mean can be an invalid trajectory that no agent would actually take.

**Q: How would you train a model to predict multiple futures?**  
A: Predict $K$ trajectories and mode probabilities. Use a min-over-modes regression loss to train the closest mode, plus cross entropy to assign probability to that mode.

**Q: What metrics would you check?**  
A: For classification, per-class precision/recall and calibration. For trajectories, minADE/minFDE, miss rate, mode entropy, probability calibration, collision, offroad, and realism.

---

## 11. A 60-second explanation you can say out loud

Custom losses are about aligning optimization with the real cost and structure of the problem. In autonomous driving, data is imbalanced and futures are multi-modal. Weighted cross entropy increases the gradient contribution of rare important classes like cut-ins. Focal loss goes further by downweighting easy examples, so training focuses on examples the model currently gets wrong. For trajectory prediction, plain MSE is dangerous because it predicts the conditional mean; the average of left-turn and straight futures may be physically invalid. A better approach predicts multiple trajectory modes and probabilities, trains the closest mode to the logged future, and also trains the model to assign that mode high probability. Then I would debug with per-class metrics, mode collapse checks, calibration, and physical validity metrics.

---

## 12. Practice exercises with answers

### Exercise 1

You have 99% normal driving and 1% cut-in. Why can unweighted CE fail?

**Answer:** The optimizer sees far more normal examples, so a model can get low loss and high accuracy while missing cut-ins. The training distribution underweights the safety-critical class.

### Exercise 2

For focal loss with $p_y=0.9$ and $\gamma=2$, what is the modulating factor?

**Answer:**

$$
(1-0.9)^2 = 0.01
$$

The easy example is downweighted by 100x.

### Exercise 3

Why does MSE predict an impossible average trajectory?

**Answer:** MSE minimizes squared error, whose optimum is the conditional mean. If valid futures are left and straight, their mean may cut through a lane boundary or curb.

### Exercise 4

In a 6-mode trajectory model, mode 0 wins for 95% of examples. What do you inspect?

**Answer:** Mode collapse, initialization, diversity regularization, mode classification weight, whether other modes are identical, and whether the data actually contains diverse futures.

### Exercise 5

Weighted CE improves rare-class recall but precision collapses. What happened?

**Answer:** The model may be overpredicting the rare class. The weight may be too high, labels may be noisy, or the decision threshold needs tuning.

