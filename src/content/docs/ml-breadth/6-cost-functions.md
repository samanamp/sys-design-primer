---
title: "Cost Functions"
description: "Interview primer on cost functions: plain-English explanations, when to use each one, and what to say out loud in an interview."
---

# Cost Functions: Interview Primer

## The one-sentence version

A cost function is the score that training tries to minimize. It answers: *how bad was that prediction?*

If you pick the wrong cost function, your model can score perfectly on paper and still behave badly in production. That's why interviewers care about it.

---

## Quick reference (read this first, return to it after)

| Loss function | Use when… | Output type | Key behavior |
|---|---|---|---|
| **MSE** | predicting a number, no big outliers | continuous | punishes large errors much harder than small ones |
| **MAE** | predicting a number, outliers in the data | continuous | treats all errors equally regardless of size |
| **Huber** | predicting a number, some outliers | continuous | MSE for small errors, MAE for big ones |
| **Binary cross-entropy** | yes/no outcome | probability 0–1 | punishes confident wrong predictions hardest |
| **Weighted BCE** | yes/no, class imbalance matters | probability 0–1 | multiplies the rare class's penalty |
| **Focal loss** | yes/no, millions of easy negatives | probability 0–1 | ignores the examples the model already handles well |
| **Softmax cross-entropy** | one of N categories | probabilities over N classes | same idea as BCE, extended to multiple classes |
| **Reconstruction loss** | generative models (VAE, autoencoder) | pixel / token values | measures similarity to original directly |
| **Adversarial loss** | generative models (GAN) | fooling a discriminator | rewards realistic-looking output, not pixel similarity |

---

## How training actually works (plain English)

Training repeats a loop:

1. Make a prediction.
2. Compute the cost (how wrong was it?).
3. Nudge the model's parameters in whichever direction reduces the cost.
4. Repeat millions of times.

The cost function is step 2. It turns "the model predicted 26 minutes, the real answer was 30 minutes" into a single number the optimizer can work with.

Two properties matter:
- **Sign doesn't cancel.** Raw error does. A prediction 4 minutes early and one 4 minutes late average to zero — the optimizer thinks everything is fine. The cost function must remove that cancellation.
- **Size of the penalty shapes behavior.** A loss that punishes a 10-minute error much harder than a 1-minute error teaches the model to avoid large misses. A loss that treats them equally teaches something different.

---

## Problem 1 — Predict delivery time → MSE

```
actual:     30 min
predicted:  26 min
error:      −4 min
```

Raw error cancels when you average positives and negatives, so we square it:

```
squared error = (26 − 30)² = 16
```

Averaged over many orders, this is **Mean Squared Error (MSE)**:

```
MSE = average of (predicted − actual)² over all examples
```

### Why squaring matters

| Error | Squared error |
|---:|---:|
| 2 min | 4 |
| 10 min | 100 |

A 5× larger error gets a 25× larger penalty. MSE tells the model: *big mistakes are much worse than small ones.*

This is the right call when one catastrophically wrong prediction (e.g., 3 hours late) is genuinely worse than many small ones, not just inconveniently wrong.

### When MSE breaks down

MSE learns to predict the **average** outcome. That's correct when there's one right answer. It becomes a problem when:

- There are extreme outliers in your labels (one corrupted record can dominate the whole loss).
- There are multiple valid answers (the average of "turn left" and "turn right" is "drive straight into the median").

### Interview answer

> "I'd start with MSE for regression. It penalizes large errors disproportionately, which is usually what you want. I'd reconsider if the training data has significant outliers or if the target has multiple valid values — both break MSE's assumptions."

**Regularization here — L2:** delivery time is influenced by many features (distance, time of day, traffic, restaurant load). You want all of them to contribute — just with controlled magnitude, not zeroed out. L2 is the right fit: it shrinks all weights proportionally but leaves every feature active. L1 would be wrong here — it might zero out "restaurant load" because it's correlated with "time of day," discarding a genuinely useful signal.

```python
import numpy as np

def mse_l2(y_pred, y_true, W, lambda_=0.01):
    mse    = ((y_pred - y_true) ** 2).mean()
    l2_reg = lambda_ * np.sum(W ** 2)
    return mse + l2_reg

def mse_l2_gradient(X, y_pred, y_true, W, lambda_=0.01):
    n   = len(y_true)
    d_W = 2 * (y_pred - y_true) @ X / n  +  2 * lambda_ * W   # ← L2 appended
    d_b = 2 * (y_pred - y_true).mean()
    return d_W, d_b
```

The only difference from plain MSE is the `+ 2 * lambda_ * W` on the gradient line. Large weights get a stronger push toward zero; small weights get almost none.

---

## Problem 2 — One corrupted record → MAE and Huber

Suppose nearly all deliveries take 10–60 minutes, but one record says 900 minutes because the order was never marked complete.

With MSE, that one record contributes:

```
(30 − 900)² = 756,900
```

That can outweigh thousands of normal examples. The model distorts its predictions for real orders just to partially reduce this one bad record's cost.

### MAE: treat every extra minute equally

**Mean Absolute Error (MAE)** uses the absolute value instead of squaring:

```
MAE = average of |predicted − actual| over all examples
```

The corrupted record now costs 870, not 756,900. Critically, once a prediction is wrong, the cost increases at a *constant* rate no matter how wrong it gets. An extreme outlier can't dominate the way it does with MSE.

**Key conceptual difference:**

| | MSE | MAE |
|---|---|---|
| What it learns | Conditional **mean** | Conditional **median** |
| Outlier influence | High | Low |
| Error sensitivity | Grows with error size | Constant |

*Concrete example:* Five deliveries take 20, 21, 22, 23, 100 minutes. MSE's optimal prediction is the mean (37.2 min — pulled by the outlier). MAE's optimal prediction is the median (22 min — not pulled at all).

### Huber: best of both

MAE's downside: near the correct answer, a constant push is less precise than the shrinking push from MSE. If you're 1 minute off versus 2 minutes off, MAE treats them identically; MSE applies a gentler correction for the 1-minute miss.

**Huber loss** combines them with a threshold δ:

```
error ≤ δ  →  use squared error  (smooth, like MSE)
error > δ  →  use absolute error  (robust, like MAE)
```

You pick δ based on what counts as a "large" error for your specific problem. For delivery time in minutes, δ might be 15 or 20.

```python
import numpy as np

# Six deliveries: five normal, one corrupted record
y_true = np.array([20., 21., 22., 23., 24., 900.])
y_pred = np.array([21., 22., 21., 24., 23.,  30.])  # model guesses ~30 for the outlier

def mse(y_pred, y_true):
    return ((y_pred - y_true) ** 2).mean()

def mae(y_pred, y_true):
    return np.abs(y_pred - y_true).mean()

def huber(y_pred, y_true, delta=15.0):
    error    = y_pred - y_true
    is_small = np.abs(error) <= delta
    return np.where(is_small,
                    0.5 * error**2,                        # MSE branch
                    delta * (np.abs(error) - 0.5*delta)   # MAE branch
                   ).mean()

print(f"MSE:   {mse(y_pred, y_true):>10.1f}")   # 756,905 — destroyed by the 900 record
print(f"MAE:   {mae(y_pred, y_true):>10.1f}")   #    146.5 — 900 costs 870 not 756k
print(f"Huber: {huber(y_pred, y_true):>10.1f}") #     13.0 — capped at δ past the threshold

# Per-example view: see exactly where MSE explodes
print("\ntrue   pred   MSE_cost   MAE_cost")
for t, p in zip(y_true, y_pred):
    print(f"{t:5.0f}  {p:5.0f}  {(p-t)**2:>9.0f}  {abs(p-t):>8.0f}")
```

```
MSE:     126151.2
MAE:        146.5
Huber:       13.0

true   pred   MSE_cost   MAE_cost
   20     21          1         1
   21     22          1         1
   22     21          1         1
   23     24          1         1
   24     23          1         1
  900     30     756900       870   ← one record, 756k vs 870
```

The mean vs median difference in one snippet:

```python
times = np.array([20., 21., 22., 23., 100.])

print(f"Mean   (what MSE learns): {times.mean():.1f}")    # 37.2 — pulled by outlier
print(f"Median (what MAE learns): {np.median(times):.1f}") # 22.0 — ignores outlier
```

Huber's switching behaviour at the threshold:

```python
delta  = 15.0
errors = np.array([2., 8., 15., 20., 50., 100.])

print("error   cost     mode")
for e in errors:
    if abs(e) <= delta:
        cost, mode = 0.5 * e**2, "MSE (quadratic)"
    else:
        cost, mode = delta * (abs(e) - 0.5*delta), "MAE (linear)"
    print(f"  {e:5.0f}  {cost:7.1f}  {mode}")
```

```
error   cost     mode
    2      2.0  MSE (quadratic)
    8     32.0  MSE (quadratic)
   15    112.5  MSE (quadratic)   ← last MSE step
   20    187.5  MAE (linear)      ← first MAE step
   50    637.5  MAE (linear)
  100   1387.5  MAE (linear)
```

### Interview answer

> "If training data has occasional extreme label errors or genuine outliers, I'd switch from MSE to Huber. I'd pick the δ threshold by plotting the distribution of prediction errors and finding where the tail starts. If the outliers are actually real and meaningful, MAE is the safer choice. If they're label noise, Huber is more practical."

---

## Problem 3 — Predict if it will rain → Binary cross-entropy

For a yes/no outcome, we want the model to output a **probability** (a number from 0 to 1). MSE on a probability is a bad fit — it treats "predicted 0.6, actual 0 (no rain)" the same as "predicted 0.6, actual 1 (rain)" with equal-but-opposite errors. It doesn't account for what the correct answer actually was.

Consider:

```
Model A predicted 60% chance of rain → no rain
Model B predicted 99% chance of rain → no rain
```

Both are wrong. But B was *confidently* wrong. Any good loss should punish B much more.

**Binary cross-entropy (BCE)** does this by measuring "how surprised the model should be by what actually happened":

```
BCE = −log(probability assigned to the correct outcome)
```

For no-rain day:

| Predicted rain probability | BCE penalty |
|---:|---:|
| 10% | small (~0.1) |
| 60% | moderate (~0.9) |
| 99% | large (~4.6) |

The penalty grows sharply as confidence grows. A model that said "99% rain" when it didn't rain gets hammered. A model that said "10% rain" gets a small nudge.

```python
import numpy as np

def sigmoid(x):
    return np.where(x >= 0, 1/(1+np.exp(-x)), np.exp(x)/(1+np.exp(x)))

def bce(logits, targets):
    p = np.clip(sigmoid(logits), 1e-7, 1 - 1e-7)
    return -(targets * np.log(p) + (1 - targets) * np.log(1 - p)).mean()

# Reproduce the table: no-rain day (target=0), three different predictions
target = np.array([0.])
for logit, label in [(-2.2, "10% rain"), (0.4, "60% rain"), (4.6, "99% rain")]:
    loss = bce(np.array([logit]), target)
    p    = sigmoid(np.array([logit]))[0]
    print(f"  Predicted {label:8s} (p={p:.2f})  BCE={loss:.3f}")
```

```
  Predicted 10% rain  (p=0.10)  BCE=0.105
  Predicted 60% rain  (p=0.60)  BCE=0.916
  Predicted 99% rain  (p=0.99)  BCE=4.605   ← confident wrong = expensive
```

### The implementation detail that matters

Models internally produce an unconstrained number (called a **logit**) and a sigmoid function converts it to a probability. If you compute `sigmoid(logit)` first and then `log(probability)`, you can get numerical errors (log of zero breaks everything).

Always use the library's built-in version that takes the raw logit:

```python
# Correct — stable, no NaN risk
loss = F.binary_cross_entropy_with_logits(logits, targets)

# Risky — manual sigmoid + log can produce NaN
p = torch.sigmoid(logits)
loss = -torch.log(p)  # breaks when p rounds to 0 or 1
```

### Softmax cross-entropy (multi-class)

The same idea scales to multiple classes (e.g., classifying an image as cat, dog, or bird). The model outputs one score per class, softmax converts them to probabilities summing to 1, and cross-entropy penalizes the probability assigned to the wrong class.

```python
def softmax(x):
    e = np.exp(x - x.max())   # subtract max for numerical stability
    return e / e.sum()

def cross_entropy(logits, true_class):
    probs = softmax(logits)
    return -np.log(probs[true_class] + 1e-7)

logits = np.array([2.0, 1.0, 0.1])   # model scores: cat=2.0, dog=1.0, bird=0.1
probs  = softmax(logits)
print(f"Probs — cat:{probs[0]:.2f}  dog:{probs[1]:.2f}  bird:{probs[2]:.2f}")

print(f"Loss if true=cat:  {cross_entropy(logits, 0):.3f}")  # small — model was right
print(f"Loss if true=bird: {cross_entropy(logits, 2):.3f}")  # large — model confidently wrong
```

```
Probs — cat:0.66  dog:0.24  bird:0.10
Loss if true=cat:   0.416   ← right answer, moderate penalty (not 100% confident)
Loss if true=bird:  2.303   ← wrong answer, high penalty
```

### Interview answer

> "For binary classification, I'd use BCE because the output is a probability and I want to penalize confident wrong predictions heavily. I'd always compute it from logits, not from the probability directly, for numerical stability. The decision threshold (0.5 by default) is a business lever I'd tune separately from training."

**Regularization here — L1:** a weather model might have hundreds of input signals — humidity, pressure, dew point, wind speed, cloud cover, and their lagged versions. Most of them are correlated. L1 regularization zeros out the redundant ones automatically, leaving only the features that independently predict rain. L2 would keep all of them small and active, which makes the model harder to interpret and may hurt performance when features are highly correlated. If an interviewer asks "how would you do feature selection?" — L1 regularization is a clean answer.

```python
def sigmoid(x):
    return np.where(x >= 0, 1/(1+np.exp(-x)), np.exp(x)/(1+np.exp(x)))

def bce_l1(logits, targets, W, lambda_=0.01):
    p      = np.clip(sigmoid(logits), 1e-7, 1 - 1e-7)
    bce    = -(targets * np.log(p) + (1 - targets) * np.log(1 - p)).mean()
    l1_reg = lambda_ * np.sum(np.abs(W))
    return bce + l1_reg

def bce_l1_gradient(X, p, targets, W, lambda_=0.01):
    n   = len(targets)
    d_W = X.T @ (p - targets) / n  +  lambda_ * np.sign(W)   # ← L1 appended
    d_b = (p - targets).mean()
    return d_W, d_b
```

Compare the gradient term to the L2 version from Problem 1:

```
L2:  + 2 * lambda_ * W        # push proportional to weight size — weakens as weight shrinks
L1:  + lambda_ * np.sign(W)   # push is always ±lambda_, regardless of weight size → can reach zero
```

`np.sign(W)` returns +1, −1, or 0. The gradient is a fixed-size shove toward zero on every step, which is why L1 can drive weights all the way to exactly zero while L2 only approaches it.

---

## Problem 4 — 1-in-1000 transactions is fraud → Weighted BCE and focal loss

**The trap:** at 0.1% fraud rate, a model that predicts "not fraud" for everything gets 99.9% accuracy. The cost function never told it that missing fraud is expensive.

**First move — don't reach for a custom loss immediately.** Check:
- How many flagged cases are actually fraud? (**precision**)
- How much fraud is caught overall? (**recall**)
- What happens as you move the decision threshold?

If you've confirmed that missed fraud is genuinely much more costly than a false alarm, then encode that cost.

### Weighted BCE

Add weights to the BCE so that each fraud example contributes more to the cost:

```
weighted BCE = −(w_fraud × log(p) for fraud) − (w_legit × log(1−p) for legit)
```

Set `w_fraud` higher than `w_legit`. The optimizer spends more effort on fraud examples.

```python
def weighted_bce(logits, targets, fraud_weight=19.0):
    p   = np.clip(sigmoid(logits), 1e-7, 1 - 1e-7)
    wts = np.where(targets == 1, fraud_weight, 1.0)
    bce = -(targets * np.log(p) + (1 - targets) * np.log(1 - p))
    return (wts * bce).mean()

# Compare: same model confidence, different label → very different cost
logit = np.array([-2.0])   # model says ~12% fraud probability

loss_fn = weighted_bce(logit, np.array([1.]), fraud_weight=19.)  # it WAS fraud
loss_fp = weighted_bce(logit, np.array([0.]), fraud_weight=19.)  # it was NOT fraud

print(f"Missed fraud   (FN): {loss_fn:.3f}")   # 19× amplified
print(f"Flagged legit  (FP): {loss_fp:.3f}")   # standard penalty
print(f"Ratio: {loss_fn/loss_fp:.1f}x")
```

```
Missed fraud   (FN): 39.178
Flagged legit  (FP):  0.131
Ratio: 299.1x   ← missing fraud is now vastly more expensive
```

**Tradeoff:** higher weight for fraud → more recalls → more false alarms. It's a dial, not a solution.

### Focal loss

A different problem: your dataset has millions of obvious legitimate transactions. The model learned to classify those perfectly in the first week of training — but they still dominate the loss because there are so many of them.

Focal loss multiplies each example's BCE by a factor that shrinks as the model gets more confident:

```
focal multiplier = (1 − p_correct)^γ
```

- If the model is 99% correct on an example: `(1 - 0.99)^2 = 0.0001` → the easy example barely counts.
- If the model is only 20% correct: `(1 - 0.20)^2 = 0.64` → the hard example still counts.

The γ (gamma) parameter controls how aggressively easy examples are suppressed. Higher γ = more focus on hard cases.

```python
def focal_bce(logits, targets, gamma=2.0, alpha=0.25):
    p         = np.clip(sigmoid(logits), 1e-7, 1 - 1e-7)
    p_correct = np.where(targets == 1, p, 1 - p)   # confidence on the right answer
    focus     = (1 - p_correct) ** gamma            # shrinks toward 0 when confident
    bce       = -(targets * np.log(p) + (1 - targets) * np.log(1 - p))
    return (alpha * focus * bce).mean()

# Show how the focusing factor suppresses easy examples
print("p_correct  focus (γ=2)  interpretation")
for pc in [0.99, 0.90, 0.70, 0.50, 0.20]:
    f = (1 - pc)**2
    print(f"  {pc:.2f}      {f:.4f}      {'easy — nearly ignored' if pc > 0.8 else 'hard — still trains'}")
```

```
p_correct  focus (γ=2)  interpretation
  0.99      0.0001      easy — nearly ignored
  0.90      0.0100      easy — nearly ignored
  0.70      0.0900      hard — still trains
  0.50      0.2500      hard — still trains
  0.20      0.6400      hard — still trains
```

**When NOT to use focal loss:**
- When "hard" examples are actually mislabeled (focal loss will overfit to noise).
- When you need the model's output to be a trustworthy probability (focal loss distorts calibration).

### Interview distinction

> "Weighted BCE expresses importance **by class** — fraud matters more than legit. Focal loss expresses importance **by difficulty** — examples the model struggles with now matter more than ones it already handles well. I'd use weighted BCE when I know the business cost ratio. I'd add focal loss if I also have a long-tail of genuinely hard examples drowning in easy ones."

**Regularization here — L2 first, then L1 if you need interpretability:**

Upweighting fraud examples makes the optimizer treat each fraud example as if it were 19 legitimate ones. That's intentional — but it also amplifies noise. The model might memorize specific patterns from 50 fraudulent transactions and fail on the 51st that looks slightly different. L2 regularization limits how strongly any individual feature can drive the output, which directly counteracts this overfitting risk.

L1 serves a different purpose here: fraud models often need to be explainable to compliance teams or regulators. A model with L1 regularization produces a short list of nonzero feature weights ("transaction velocity in last 1 hour, country mismatch, device age") that you can hand to a compliance officer. L2 produces a model where every feature has a small nonzero weight — explainable in principle, but not as a short list.

In practice: start with L2 for stability, add L1 (or an elastic net that combines both) if explainability is a hard requirement.

```python
def weighted_bce_l2(logits, targets, W, fraud_weight=19.0, lambda_=0.01):
    p      = np.clip(sigmoid(logits), 1e-7, 1 - 1e-7)
    wts    = np.where(targets == 1, fraud_weight, 1.0)
    bce    = -(targets * np.log(p) + (1 - targets) * np.log(1 - p))
    l2_reg = lambda_ * np.sum(W ** 2)
    return (wts * bce).mean() + l2_reg

# Elastic net = L2 stability + L1 sparsity
def weighted_bce_elastic(logits, targets, W,
                          fraud_weight=19.0, l1=0.001, l2=0.01):
    p      = np.clip(sigmoid(logits), 1e-7, 1 - 1e-7)
    wts    = np.where(targets == 1, fraud_weight, 1.0)
    bce    = -(targets * np.log(p) + (1 - targets) * np.log(1 - p))
    reg    = l2 * np.sum(W ** 2) + l1 * np.sum(np.abs(W))
    return (wts * bce).mean() + reg

def elastic_gradient(X, p, targets, W, fraud_weight=19.0, l1=0.001, l2=0.01):
    n       = len(targets)
    wts     = np.where(targets == 1, fraud_weight, 1.0)
    d_logit = wts * (p - targets) / n
    d_W     = X.T @ d_logit  +  2 * l2 * W  +  l1 * np.sign(W)   # three forces
    d_b     = d_logit.sum()
    return d_W, d_b
```

The gradient line shows all three forces at once:

```
X.T @ d_logit        — data: reduce prediction error
+ 2 * l2 * W         — L2:   shrink large weights proportionally
+ l1 * np.sign(W)    — L1:   constant push; zeros out weak features
```

Tune `l2` first (controls stability), then `l1` (controls how sparse the final weights are).

---

## Problem 5 — Multiple valid answers → Mixture loss (brief)

Suppose you're predicting a car's position 5 seconds from now. At an intersection, it might validly turn left (−10 m) or right (+10 m).

MSE will predict 0 m — the mathematical average of the two valid outcomes, but physically impossible.

This is called **multi-modality**: the answer distribution has multiple peaks.

**The fix:** instead of predicting one answer, predict several possibilities with probabilities:

```
Option A: turn left (−10 m), probability 50%
Option B: turn right (+10 m), probability 50%
```

The loss rewards the model for assigning high probability to whatever actually happened.

```python
import numpy as np

# Two equally likely futures: turn left (-10m) or turn right (+10m)
outcomes = np.array([-10., 10.])
probs    = np.array([0.5,  0.5])

# What MSE predicts: the weighted mean
mse_prediction = (probs * outcomes).sum()
print(f"MSE predicts: {mse_prediction:.1f} m")   # 0.0 — drives into the barrier

# MSE at the "optimal" prediction vs either valid answer
def expected_mse(prediction):
    return (probs * (prediction - outcomes)**2).sum()

print(f"MSE cost at m=0  (MSE optimum): {expected_mse(0):.1f}")   # 100
print(f"MSE cost at m=-10 (left turn):  {expected_mse(-10):.1f}") # also 100
print(f"MSE cost at m=+10 (right turn): {expected_mse(10):.1f}")  # also 100
# Every prediction is equally bad — MSE cannot represent two valid futures

# What a mixture model does: assign probability to each mode
import math

def log_gaussian(y, mu, sigma=2.0):
    return -0.5*((y-mu)/sigma)**2 - math.log(sigma) - 0.5*math.log(2*math.pi)

def mixture_nll(y_observed, mus, mix_probs, sigma=2.0):
    """Negative log likelihood for a 2-component mixture."""
    log_components = [math.log(pi) + log_gaussian(y_observed, mu, sigma)
                      for pi, mu in zip(mix_probs, mus)]
    # logsumexp for numerical stability
    max_log = max(log_components)
    log_sum = max_log + math.log(sum(math.exp(l - max_log) for l in log_components))
    return -log_sum

y_left  = -10.0   # car turned left
y_right =  10.0   # car turned right

loss_left  = mixture_nll(y_left,  mus=[-10., 10.], mix_probs=[0.5, 0.5])
loss_right = mixture_nll(y_right, mus=[-10., 10.], mix_probs=[0.5, 0.5])

print(f"\nMixture NLL for left  turn: {loss_left:.3f}")   # low — left mode covers it
print(f"Mixture NLL for right turn: {loss_right:.3f}")   # also low — right mode covers it
```

```
MSE predicts: 0.0 m
MSE cost at m=0  (MSE optimum): 100.0
MSE cost at m=-10 (left turn):  100.0   ← no prediction is better than another
MSE cost at m=+10 (right turn): 100.0

Mixture NLL for left  turn: 2.114   ← mixture handles both correctly
Mixture NLL for right turn: 2.114
```

This is more advanced and mainly comes up in autonomous vehicles, robotics, and trajectory prediction. For most product ML interviews, knowing *why* MSE fails here is enough.

**Interview answer:** "If the target has multiple valid outcomes — like a car that could turn left or right — MSE would predict the invalid average. I'd model the output as a distribution with multiple modes rather than a single point, and train with a likelihood loss."

---

## Problem 6 — Regularization and the L1 vs L2 visual guide

Regularization isn't a cost function — it's an addition to the cost function that penalizes the model for being unnecessarily complicated.

Imagine two models that fit the training data equally:
- Model A uses moderate weights.
- Model B uses enormous weights that happen to cancel each other out.

Model B is fragile: change the input slightly and the output changes wildly.

### The mathematical difference (visual)

Both regularizers add a penalty based on the model's weights. They differ in *how* the penalty grows:

```
Weight value:         0.0   0.1   0.5   1.0   2.0   5.0
L2 penalty (w²):      0    0.01  0.25  1.0   4.0   25.0
L1 penalty (|w|):     0    0.1   0.5   1.0   2.0    5.0
```

**What this means in plain English:**

- L2 penalty grows as the square of the weight. A weight of 5 gets penalized 25× harder than a weight of 1. The optimizer attacks big weights aggressively, but once a weight is small (say 0.1), the penalty is tiny (0.01) — the push nearly disappears. This is why L2 *shrinks* weights without fully zeroing them.

- L1 penalty grows linearly. A weight of 5 is penalized 5× harder than 1, but a weight of 0.1 still has a meaningful penalty (0.1). The push toward zero doesn't disappear even when the weight is nearly zero — so the optimizer keeps pushing until the weight *is* zero. This is why L1 creates **sparsity** (some weights become exactly zero).

```python
import numpy as np

weights = np.array([0.0, 0.1, 0.5, 1.0, 2.0, 5.0])

print(f"{'weight':>8}  {'L2 penalty':>12}  {'L1 penalty':>12}  {'L2 gradient':>13}  {'L1 gradient':>13}")
for w in weights:
    l2_pen  = w**2
    l1_pen  = abs(w)
    l2_grad = 2 * w        # ∝ w — weakens as w shrinks
    l1_grad = np.sign(w)   # always ±1 — constant push
    print(f"{w:>8.1f}  {l2_pen:>12.3f}  {l1_pen:>12.3f}  {l2_grad:>13.3f}  {l1_grad:>13.0f}")
```

```
  weight    L2 penalty    L1 penalty    L2 gradient    L1 gradient
     0.0         0.000         0.000          0.000              0
     0.1         0.010         0.100          0.200              1   ← L1 still pushes hard
     0.5         0.250         0.500          1.000              1
     1.0         1.000         1.000          2.000              1
     2.0         4.000         2.000          4.000              1
     5.0        25.000         5.000         10.000              1
```

The L1 gradient column is what drives zeroing: it's always 1 regardless of weight size. The L2 gradient column shrinks to 0.2 at w=0.1 — almost no push left. Give the optimizer 1000 steps at w=0.1 with L2 and it barely moves; with L1 it zeros out completely.

### Why sparsity matters

If you have 1,000 input features and only 20 are actually useful, L1 can zero out the irrelevant 980. L2 just makes them very small. For feature selection or interpretability, L1 is the stronger tool.

### Visual intuition: the diamond vs circle

Think of the regularizer as drawing a boundary around "allowed" weight values. L2 draws a **circle** — solutions land anywhere on the smooth curve, usually not at zero. L1 draws a **diamond** — it has sharp corners on the axes. The optimal solution tends to "snap" to a corner, where one or more weights are exactly zero.

```
    L2 region (circle)      L1 region (diamond)
         w2                       w2
          |                        |
      .---+---.                   /|\
     /    |    \                 / | \
    |     |     |               /  |  \
----+-----0-----+---- w1   ----+---0---+---- w1
    |     |     |               \  |  /
     \    |    /                 \ | /
      '---+---'                   \|/
                                   |
  solutions land anywhere     solutions snap to corners
  on the circle               (axis intersections = sparse)
```

### Don't confuse these

| Concept | What it scores |
|---|---|
| MSE / MAE | Prediction errors |
| L1 / L2 regularization | Model weight sizes |

### Interview answer

> "L1 and L2 both penalize large weights, but differently. L2 penalty grows quadratically so it attacks big weights hard and leaves small weights nearly alone — all weights shrink but rarely reach zero. L1 penalty grows linearly so the push toward zero stays constant even for tiny weights — some weights go all the way to zero, giving you sparsity and implicit feature selection. I'd use L1 when I want to identify which features actually matter. I'd use L2 for general regularization. I'd tune the strength λ on a validation set."

---

## Problem 7 — Implementing a custom loss function

Interviews sometimes ask you to write a loss for a non-standard problem. The approach is always the same regardless of the specific loss.

### The checklist before writing any custom loss

1. **Can you reuse a standard loss?** Weighted BCE, focal loss, and Huber cover most real cases. A custom loss adds maintenance cost.
2. **What does the model output?** A number? A probability? Multiple numbers? The loss must match.
3. **What's the expensive mistake?** Write down the mistake in English before writing code.
4. **Is the loss differentiable?** Training needs gradients. If you write `if error > threshold: ...`, make sure both branches have smooth gradients.

### Example 1: weighted BCE for imbalanced classes

Problem: 99% legitimate, 1% fraud. Standard BCE ignores fraud.

```python
def weighted_bce(logits, targets, fraud_weight=10.0):
    # pos_weight multiplies the loss for positive (fraud) examples
    # fraud_weight=10 means each fraud example counts 10× as much
    loss = F.binary_cross_entropy_with_logits(
        logits,
        targets,
        pos_weight=torch.tensor(fraud_weight)
    )
    return loss
```

Line by line:
- `logits` — raw model outputs before sigmoid, not probabilities (more stable)
- `pos_weight=10` — each fraud label's gradient is 10× stronger
- The library handles the numerical stability; don't reimplement it

**How to pick the weight:** start with `(# negatives) / (# positives)`. A 99:1 ratio → weight ≈ 99. Then tune based on the precision/recall tradeoff you need.

**The same thing in NumPy — no magic:**

```python
import numpy as np

def sigmoid(x):
    # Two-branch form avoids overflow for very large or very small x
    return np.where(x >= 0,
                    1 / (1 + np.exp(-x)),
                    np.exp(x) / (1 + np.exp(x)))

def weighted_bce_numpy(logits, targets, fraud_weight=10.0):
    p = np.clip(sigmoid(logits), 1e-7, 1 - 1e-7)  # clip so log(0) never happens

    # fraud examples use fraud_weight; legit examples use 1.0
    weights = np.where(targets == 1, fraud_weight, 1.0)

    # BCE formula written out: -[y*log(p) + (1-y)*log(1-p)]
    # When target=1: only the first term survives  → -log(p)
    # When target=0: only the second term survives → -log(1-p)
    bce_per_example = -(targets * np.log(p) + (1 - targets) * np.log(1 - p))

    return (weights * bce_per_example).mean()
```

Run it side by side with the PyTorch version and you'll get the same numbers (up to floating-point rounding).

### Example 2: focal loss for hard examples

Problem: millions of easy negatives drown out the few hard examples.

```python
def focal_bce(logits, targets, gamma=2.0, alpha=0.25):
    bce = F.binary_cross_entropy_with_logits(
        logits, targets, reduction="none"  # keep per-example losses
    )
    p_correct = torch.where(
        targets == 1,
        torch.sigmoid(logits),         # if target is 1, p_correct = p
        1 - torch.sigmoid(logits)      # if target is 0, p_correct = 1-p
    )
    focus = (1 - p_correct) ** gamma   # easy examples (high p_correct) → near 0
    return (alpha * focus * bce).mean()
```

Line by line:
- `reduction="none"` — keep per-example losses so we can multiply each one
- `p_correct` — "how confident was the model about the right answer?"
- `focus` — when p_correct is 0.99, focus ≈ 0.0001 (nearly zero). When 0.5, focus = 0.25 (still matters).
- `alpha` — balances how much fraud vs legit examples contribute on top of the focusing

**The same thing in NumPy:**

```python
def focal_bce_numpy(logits, targets, gamma=2.0, alpha=0.25):
    p = np.clip(sigmoid(logits), 1e-7, 1 - 1e-7)

    # How confident was the model about whichever label was correct?
    # If target=1 (fraud):  p_correct = p       (prob it said "fraud")
    # If target=0 (legit):  p_correct = 1 - p   (prob it said "not fraud")
    p_correct = np.where(targets == 1, p, 1 - p)

    # The focusing multiplier: shrinks toward 0 for confident correct predictions
    # Example: p_correct=0.99, gamma=2 → (1-0.99)^2 = 0.0001 (easy example, ignore it)
    #          p_correct=0.30, gamma=2 → (1-0.30)^2 = 0.49   (hard example, keep it)
    focus = (1 - p_correct) ** gamma

    bce_per_example = -(targets * np.log(p) + (1 - targets) * np.log(1 - p))

    return (alpha * focus * bce_per_example).mean()
```

The only difference from weighted BCE is the `focus` multiplier. Everything else is identical.

### Example 3: adding regularization to a custom loss

Regularization is not a separate concept — it's literally an extra term you add to the loss before returning it.

```
total_loss = data_loss + λ × regularization_penalty
```

Here's the complete weighted BCE + L2 in NumPy, with a real feature so there's an actual weight to regularize:

```python
import numpy as np

# A simple linear model: logit = w * x + b
# x = transaction amount (one feature), w = its weight, b = bias
# We regularize w (the learned coefficient), not b.

np.random.seed(0)
n = 200
X       = np.random.randn(n)                              # transaction amounts
targets = ((X + np.random.randn(n) * 0.5) > 0.5).astype(float)  # fraud label

w, b    = 0.0, 0.0   # model parameters
lr      = 0.1
lambda_ = 0.5        # regularization strength — larger = simpler model

for step in range(300):
    logits = w * X + b                                    # forward pass

    # ── data loss ──────────────────────────────────────
    p      = np.clip(sigmoid(logits), 1e-7, 1 - 1e-7)
    wts    = np.where(targets == 1, 9.0, 1.0)            # 9:1 class weight
    bce    = -(targets * np.log(p) + (1 - targets) * np.log(1 - p))
    loss_data = (wts * bce).mean()

    # ── L2 regularization penalty ───────────────────────
    loss_reg = lambda_ * w**2                             # only on w, not bias

    total_loss = loss_data + loss_reg                     # ← the one number training minimizes

    # ── gradients ──────────────────────────────────────
    d_logit = wts * (p - targets) / n                    # data loss gradient w.r.t. logit

    d_w = np.sum(d_logit * X) + 2 * lambda_ * w          # chain rule + L2 gradient
    d_b = np.sum(d_logit)                                 # bias has no regularization

    w -= lr * d_w
    b -= lr * d_b

    if step % 100 == 0:
        print(f"step {step:3d}  total={total_loss:.3f}  data={loss_data:.3f}  reg={loss_reg:.3f}  w={w:.3f}")
```

Expected output:
```
step   0  total=0.720  data=0.693  reg=0.000  w=0.000
step 100  total=0.374  data=0.316  reg=0.058  w=0.340
step 200  total=0.368  data=0.307  reg=0.061  w=0.349
step 300  total=0.366  data=0.305  reg=0.061  w=0.349
```

The `reg` column starts at zero (weight starts at zero) and grows as the model learns a nonzero `w`. The optimizer stops here because pushing `w` higher would reduce `data` loss less than it would increase `reg` loss. That's the regularizer working as intended.

**Switching to L1** is one line:

```python
loss_reg = lambda_ * abs(w)          # L1 penalty
d_w      = np.sum(d_logit * X) + lambda_ * np.sign(w)   # L1 gradient: constant ±λ
```

The sign of `w` determines direction. If `w` is positive, `sign(w) = +1` → constant push downward. The push doesn't weaken as `w` shrinks toward zero, which is why L1 can drive weights all the way to exactly zero.

**The two key things to remember:**
1. Regularize the **weights** (learned parameters), not the logits or the loss terms.
2. The gradient of the total loss = gradient of data loss + gradient of regularization term. They add together naturally.

### Verifying your custom loss works

Before training for hours, run this sanity check:

```python
# 1. Does it decrease when you make a better prediction?
bad_logit  = torch.tensor([-5.0])  # model says "definitely not fraud"
good_logit = torch.tensor([5.0])   # model says "definitely fraud"
target = torch.tensor([1.0])       # actual: fraud

print(focal_bce(bad_logit, target))   # should be HIGH
print(focal_bce(good_logit, target))  # should be LOW

# 2. Does it give a reasonable value for a random model?
# At initialization, loss should be near log(2) ≈ 0.693 for BCE
random_logits = torch.zeros(100)
random_targets = torch.randint(0, 2, (100,)).float()
print(focal_bce(random_logits, random_targets))  # should be ~0.35 with alpha=0.25
```

### What the gradient actually is (the signal that drives learning)

The optimizer never sees the loss value directly — it sees the **gradient**: how much the loss changes when you nudge each logit up or down.

For weighted BCE, this works out to a clean formula:

```
gradient for one example = weight × (predicted_probability − true_label)
```

In NumPy:

```python
def weighted_bce_gradient(logits, targets, fraud_weight=10.0):
    p = sigmoid(logits)
    weights = np.where(targets == 1, fraud_weight, 1.0)
    # Divide by n to match the .mean() in the loss
    return weights * (p - targets) / len(targets)
```

Trace through two examples to see the weighting in action:

```
Fraud example (target=1), model predicted p=0.30:
  gradient = 19 × (0.30 − 1.0) = −13.3   ← large negative push, logit goes UP → p rises

Legit example (target=0), model predicted p=0.30:
  gradient = 1  × (0.30 − 0.0) = +0.3    ← tiny positive push, logit goes DOWN → p falls
```

The fraud example has 44× more influence on the update. That's exactly what `fraud_weight=19` does — it's not magic, it's a multiplier on (prediction − label).

**A full gradient-descent loop in NumPy** (no PyTorch, no autograd):

```python
np.random.seed(0)

# 5 fraud, 95 legit — a single bias parameter (no features, just to see the loop)
targets = np.array([1.0]*5 + [0.0]*95)
logit   = np.array([0.0])   # one shared logit for all examples (simplification)
logits  = np.ones(100) * logit[0]

for step in range(200):
    loss = weighted_bce_numpy(logits, targets, fraud_weight=19.0)
    grad = weighted_bce_gradient(logits, targets, fraud_weight=19.0)
    logits -= 0.5 * grad          # gradient descent: move opposite the gradient

    if step % 50 == 0:
        print(f"step {step:3d}  loss={loss:.4f}  p={sigmoid(logits[0]):.3f}")

# step   0  loss=0.6931  p=0.500   ← starts at 50% for everything
# step  50  loss=0.3001  p=0.680
# step 100  loss=0.2544  p=0.751
# step 150  loss=0.2279  p=0.793
# step 200  loss=0.2106  p=0.822   ← pushed toward fraud (too many frauds were missed)
```

The model has no features here — it can only learn one number. But you can watch the gradient nudge the probability upward every step. In a real model this same loop runs for each weight in the network.

### Interview answer

> "I'd start by checking whether a standard loss with weights covers the case — it usually does. If I need something custom, I'd write it to operate on logits or log-probabilities (never raw probabilities) to avoid numerical issues, verify that a manually better prediction produces lower loss, and monitor gradient norms to confirm training signal is flowing."

---

## Problem 8 — Reward functions as cost functions (RL)

In reinforcement learning, a model (called an **agent**) takes actions and receives a **reward** signal. Maximizing reward is equivalent to minimizing negative reward — so the reward function *is* the cost function, negated.

This sounds simple. It causes most of the problems in RL.

### The core pitfall: reward hacking

You write a reward. The model optimizes *exactly what you wrote*, not what you meant.

```python
import numpy as np

# Intended reward: maximize delivery speed
def reward_speed(delivery_time_minutes):
    return 1.0 / delivery_time_minutes

print("Intended usage:")
for t in [10, 30, 60]:
    print(f"  {t} min delivery → reward {reward_speed(t):.3f}")

print("\nWhat the model discovers (drop package immediately):")
print(f"  0.1 min 'delivery' → reward {reward_speed(0.1):.1f}")  # 10× better!
print(f"  0.01 min           → reward {reward_speed(0.01):.1f}") # 100× better!
```

```
Intended usage:
  10 min delivery → reward 0.100
  30 min delivery → reward 0.033
  60 min delivery → reward 0.017

What the model discovers (drop package immediately):
  0.1 min 'delivery' → reward 10.0
  0.01 min           → reward 100.0
```

The reward function is technically correct — fast *is* better. But it never specified that the package must arrive. The model found the loophole.

**Interview signal:** always ask "what behavior does this reward incentivize that I *don't* want?" before finalizing a reward function.

### Pitfall: sparse rewards

Imagine learning to play chess where the only feedback is "you won" or "you lost" at the very end. For the first thousands of training games, every move gets the same signal — nothing. The model can't tell which of its 40 moves caused it to lose.

```python
import numpy as np

def compute_returns(rewards, gamma=0.99):
    """Convert per-step rewards into discounted future returns.
    This is how RL assigns credit backward through time."""
    returns = np.zeros_like(rewards, dtype=float)
    running = 0.0
    for t in reversed(range(len(rewards))):
        running     = rewards[t] + gamma * running
        returns[t]  = running
    return returns

# Chess game: 40 moves, reward only at the very end
sparse = np.zeros(40)
sparse[-1] = 1.0   # won the game

returns = compute_returns(sparse, gamma=0.99)

print("Step  Raw reward  Discounted return  (credit received)")
for t in [0, 10, 20, 30, 38, 39]:
    print(f"  {t:2d}       {sparse[t]:.1f}          {returns[t]:.4f}")
```

```
Step  Raw reward  Discounted return
   0       0.0          0.6676   ← heavily discounted — 40 steps ago
  10       0.0          0.7397
  20       0.0          0.8179
  30       0.0          0.9044
  38       0.0          0.9900
  39       1.0          1.0000   ← only this step has raw reward
```

Without discounting, every step before step 39 gets zero credit. With γ=0.99, step 0 gets 67 cents on the dollar — enough to train on, but very weak signal 40 steps back.

**Common fixes:**
- **Reward shaping**: add intermediate signals (e.g., +0.1 for capturing a piece). Risk: introduces new loopholes.
- **Potential-based shaping**: structure intermediate rewards as differences in a progress function — guaranteed not to change which final behavior is optimal.
- **Curriculum learning**: start with simpler problems that have denser rewards.

```python
# Potential-based shaping: safe intermediate rewards
# F(s, s') = γ * Φ(s') − Φ(s)   where Φ measures "progress"

def potential(position, goal=np.array([0., 0.])):
    return -np.linalg.norm(position - goal)   # negative distance = higher potential near goal

gamma = 0.99
# Robot moving from (5,5) toward goal (0,0)
trajectory = np.array([[5.,5.], [4.,4.], [3.,3.], [2.,2.], [1.,1.], [0.,0.]])

print("Step  Position   Shaped reward   (positive = moving toward goal)")
for t in range(len(trajectory)-1):
    s, s_next = trajectory[t], trajectory[t+1]
    shaped = gamma * potential(s_next) - potential(s)
    print(f"  {t}    {s}    {shaped:+.3f}")
```

```
Step  Position   Shaped reward
  0    [5. 5.]    +0.992   ← moving closer, positive reward
  1    [4. 4.]    +0.992
  2    [3. 3.]    +0.990
  3    [2. 2.]    +0.990
  4    [1. 1.]    +0.990
```

### Pitfall: numerical instability

Reward signals can have wildly different scales. If "win the game" gives +1,000 and each step costs −0.001, the win signal dominates gradient updates catastrophically when it finally arrives.

```python
import numpy as np

# Mixed-scale rewards: small step penalties + huge win bonus
raw_rewards = np.array([-0.001] * 39 + [1000.0])

print(f"Scale mismatch: {raw_rewards.max() / abs(raw_rewards.min()):.0f}x")
print(f"  Step penalty: {raw_rewards[0]}")
print(f"  Win bonus:    {raw_rewards[-1]}")

# Fix 1: clip rewards to [-1, 1]
clipped = np.clip(raw_rewards, -1.0, 1.0)
print(f"\nClipped — max: {clipped.max()}, min: {clipped.min()}")

# Fix 2: normalize per episode
def normalize(rewards):
    return (rewards - rewards.mean()) / (rewards.std() + 1e-8)

normed = normalize(raw_rewards)
print(f"Normalized — max: {normed.max():.2f}, min: {normed.min():.2f}")
# Now win bonus and step penalties are comparable in magnitude
```

```
Scale mismatch: 1000000x
  Step penalty: -0.001
  Win bonus:    1000.0

Clipped    — max: 1.0,   min: -1.0
Normalized — max: 6.08,  min: -0.16   ← win bonus no longer 1M× bigger
```

### Pitfall: delayed credit assignment

Which action caused the reward? If the reward arrives 500 steps after the action, the discounting in `compute_returns` above handles it mechanically — but with γ=0.99 over 500 steps, the credit is 0.99^500 ≈ 0.007. The signal is nearly zero. This is the **credit assignment problem**, and it's why RL problems with long horizons are fundamentally harder than supervised learning.

### Interview answer

> "Reward functions become cost functions by negating them, but they introduce unique failure modes. I'd watch for three things: reward hacking — verify the written rule can't be gamed without achieving the actual goal; sparse signals — add potential-based shaping so intermediate steps get credit; and scale mismatch — normalize or clip rewards so a late large bonus doesn't explode the gradient when it arrives."

---

## Problem 9 — Reconstruction loss vs adversarial loss (generative models)

If the task is to *generate* something new — an image, a sentence, a molecule — you need a loss that says "how realistic is this?" That's fundamentally harder than "how wrong is this number?"

### Reconstruction loss: pixel-by-pixel accuracy

**How it works:** compare the generated output to the original example directly. For images, this is typically MSE per pixel. For text, it's cross-entropy per token.

```python
import numpy as np

# 4-pixel "image" (simplified)
original  = np.array([0.9, 0.1, 0.8, 0.2])   # sharp: bright, dark, bright, dark
blurry    = np.array([0.5, 0.5, 0.5, 0.5])   # generator averaged everything
sharp_gen = np.array([0.85, 0.15, 0.75, 0.25])  # generator learned the pattern

def reconstruction_loss(generated, original):
    return ((generated - original)**2).mean()

print(f"Blurry output MSE:    {reconstruction_loss(blurry, original):.4f}")
print(f"Sharp output MSE:     {reconstruction_loss(sharp_gen, original):.4f}")
# Blurry output still gets low MSE — reconstruction loss rewards "safe" averages
```

```
Blurry output MSE:    0.1700   ← low! averaging is technically accurate
Sharp output MSE:     0.0038   ← lower, but blurry was already rewarded
```

**What it produces:** outputs that are "safe" — averaging over all the possible correct things. For faces, this often means blurry images. The model is unsure whether hair should be lighter or darker, so it outputs a grey average of both, which is technically low-MSE but looks fake.

**Why it's stable:** there's always a signal. Every generated example can be compared to a real one.

**Used in:** autoencoders, VAEs (variational autoencoders), early image generation.

### Adversarial loss: fool the judge

**How it works:** train a second model called the **discriminator** whose job is to distinguish real examples from generated ones. The **generator**'s loss is: "how often does the discriminator get fooled?"

```python
def bce_loss(p, target):
    p = np.clip(p, 1e-7, 1 - 1e-7)
    return -(target * np.log(p) + (1 - target) * np.log(1 - p)).mean()

# Discriminator outputs: probability that each image is real
d_on_real  = np.array([0.92, 0.88, 0.95])   # real images  → D should say ~1
d_on_fake  = np.array([0.35, 0.28, 0.41])   # generated    → D should say ~0

# Discriminator wants to be right on both:
disc_loss = (bce_loss(d_on_real, np.ones(3)) +
             bce_loss(d_on_fake, np.zeros(3)))

# Generator wants D to say its fake images are real:
gen_loss = bce_loss(d_on_fake, np.ones(3))

print(f"Discriminator loss: {disc_loss:.3f}   (lower = D better at spotting fakes)")
print(f"Generator loss:     {gen_loss:.3f}   (lower = G better at fooling D)")
```

```
Discriminator loss: 0.218   (lower = D better at spotting fakes)
Generator loss:     1.093   (lower = G better at fooling D)
```

Both models train simultaneously — a constant competition.

**What it produces:** sharp, realistic outputs. The discriminator can detect blurriness as "fake," so the generator learns to produce sharply detailed images.

**Why it's unstable:** the generator and discriminator must stay balanced. If the discriminator gets too good, it rejects everything and the generator gets no useful gradient. If the generator gets too good, the discriminator collapses and loses all discriminating ability.

**Mode collapse:** the generator discovers one or two outputs that always fool the discriminator and starts producing only those, ignoring the full variety of real data.

### Side-by-side comparison

| | Reconstruction loss | Adversarial loss |
|---|---|---|
| What it measures | Pixel/token similarity to original | Whether a judge can tell real from fake |
| Output quality | Blurry, averaged | Sharp, realistic |
| Training stability | Stable (always has signal) | Unstable (requires balance) |
| Main failure mode | Blurriness (averaging valid answers) | Mode collapse, training instability |
| Common in | VAE, autoencoder | GAN |

### In practice: combine them

Most modern generative systems use **both**:

```python
lambda_adv = 0.1   # how much adversarial loss contributes

rec_loss = reconstruction_loss(blurry, original)
total    = rec_loss + lambda_adv * gen_loss

print(f"Reconstruction loss: {rec_loss:.4f}")
print(f"Adversarial loss:    {gen_loss:.4f}  (× {lambda_adv})")
print(f"Total loss:          {total:.4f}")
# Tune lambda_adv: higher → sharper but less stable; lower → blurrier but stable
```

- Reconstruction provides a stable base signal and ensures the output is at least in the right ballpark.
- Adversarial adds the sharpness and realism the reconstruction loss can't provide.

**Diffusion models** (the architecture behind modern image generation) sidestep this entirely by using a simple noise-prediction objective — neither reconstruction nor adversarial — which is why they're much more stable to train than GANs.

### Interview answer

> "Reconstruction loss measures pixel-level accuracy, which is easy to train but produces blurry outputs because it averages over all valid answers. Adversarial loss trains a discriminator to catch fakes, which produces sharp outputs but is notoriously unstable and prone to mode collapse. In practice I'd combine them — reconstruction loss for stability, adversarial loss for realism — and tune the balance weight. For modern work I'd consider a diffusion-based objective, which avoids both failure modes."

---

### Loss isn't decreasing at all

1. Try to overfit one tiny batch (5–10 examples). If the model can't memorize 5 examples, something is broken.
2. Verify the target values — wrong dtype, wrong shape, or off-by-one in labels.
3. Manually make a better prediction and check that the loss goes down. If it doesn't, the loss function has a bug.
4. Check gradients — if they're zero, the model isn't learning.

### Loss explodes or goes NaN

1. Check for `log(0)` — use the library's numerically stable versions (BCE with logits, log-sum-exp).
2. Look at the loss scale — using `sum` over a batch instead of `mean` makes the loss grow with batch size.
3. Check for division by a very small variance in probabilistic losses.
4. Gradient clipping can contain occasional spikes, but it shouldn't hide a fundamentally broken loss.

---

## The interview framework

When asked "why this loss?", answer in this order:

**1. Output:** "The model outputs a probability / a continuous value / one of N classes."

**2. Data shape:** "The data has / doesn't have class imbalance / outliers / multiple valid outcomes."

**3. Mistake cost:** "The expensive mistake here is false negatives / large errors / confident wrong predictions."

**4. Loss choice:** "So I'd use X because it penalizes Y specifically."

**5. Tradeoff:** "The risk with X is Z, so I'd watch for it."

**6. Implementation:** "I'd implement it from logits / with log-sum-exp to avoid numerical issues."

---

### Example answer (fraud detection)

> "The model outputs a fraud probability. At 0.1% fraud rate, standard accuracy is useless — a model that says 'not fraud' every time scores 99.9%. I'd evaluate with precision-recall instead. If missing fraud costs much more than a false alarm, I'd encode that with weighted BCE, setting the fraud weight proportional to the cost ratio. If easy legitimate examples dominate training, I'd add focal loss on top. I'd implement both from logits, not from probabilities, and monitor calibration — if the model says 20% fraud, roughly 20% of those should actually be fraud."

---

## The core mental model

> **Start from the mistake you need the model to care about. Then choose the loss that makes that mistake expensive.**

| Mistake you want to avoid | Loss that punishes it |
|---|---|
| Big prediction errors (regression) | MSE |
| Outlier-driven distortion | MAE or Huber |
| Confident wrong probability | Binary cross-entropy |
| Missing the rare class | Weighted BCE |
| Easy examples drowning hard ones | Focal loss |
| Multiple valid answers averaged together | Mixture / multi-modal loss |
| Model too complex, generalizes poorly | + L2 regularization (shrinks all weights) |
| Many correlated features, want selection | + L1 regularization (zeros out weak features) |
| Need both stability and interpretability | + Elastic net (L1 + L2 combined) |
| Overfitting to a rare upweighted class | + L2 to limit noise amplification |
| Blurry generative output | + Adversarial loss on top of reconstruction |
| Unstable GAN training | Switch to reconstruction loss; consider diffusion objective |
| Reward hacking in RL | Redesign reward; use potential-based shaping |
| Sparse RL reward | Add intermediate shaped rewards; curriculum learning |
