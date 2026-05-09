---
title: "KL Divergence"
description: "KL Divergence"
---

Imagine we have a classifier like this:

```text
Z1 = X @ W1 + b1
A1 = max(0, Z1)
Z2 = A1 @ W2 + b2
P  = softmax(Z2)
```

Not hard. For this code, there are **two possible meanings**:

## 1. KL from true labels to prediction

Your current cross-entropy loss is already basically:

$$
KL(y_{\text{one-hot}} | P)
$$

Because for one-hot label:

$$
KL(q | P) = \sum_i q_i \log \frac{q_i}{P_i}
$$

Only the true class has $q_i = 1$, so:

$$
KL = -\log P_{\text{true}}
$$

So this line:

```python
loss = -np.mean(np.log(P[np.arange(N), y] + 1e-12))
```

is already the KL loss for one-hot labels.

Your backward is also correct:

```python
dZ2 = P.copy()
dZ2[np.arange(N), y] -= 1
dZ2 /= N
```

So for hard labels, **you do not need to change anything**.

---

## 2. KL between two probability distributions

Example: teacher model distribution `Q` and your model prediction `P`.

$$
KL(Q | P) = \sum_i Q_i(\log Q_i - \log P_i)
$$

Code:

```python
def kl_loss_from_probs(P, Q):
    eps = 1e-12
    return np.mean(np.sum(Q * (np.log(Q + eps) - np.log(P + eps)), axis=1))
```

Where:

```python
P.shape == (N, C)  # student prediction
Q.shape == (N, C)  # target/teacher distribution
```

Backward w.r.t. logits `Z2` is simple:

```python
dZ2 = (P - Q) / N
```

So your `backward` changes only this part:

```python
dZ2 = (P - Q) / N
```

instead of:

```python
dZ2 = P.copy()
dZ2[np.arange(N), y] -= 1
dZ2 /= N
```

---

## Clean version for soft-label KL

```python
def forward_kl_loss(X, Q, params):
    W1, b1 = params["W1"], params["b1"]
    W2, b2 = params["W2"], params["b2"]

    Z1 = X @ W1 + b1
    A1 = np.maximum(0, Z1)
    Z2 = A1 @ W2 + b2

    shifted = Z2 - np.max(Z2, axis=1, keepdims=True)
    exp_scores = np.exp(shifted)
    P = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

    eps = 1e-12
    loss = np.mean(np.sum(Q * (np.log(Q + eps) - np.log(P + eps)), axis=1))

    cache = {
        "X": X,
        "Z1": Z1,
        "A1": A1,
        "P": P,
        "Q": Q,
    }

    return loss, cache
```

Backward:

```python
def backward_kl(cache, params):
    X = cache["X"]
    Z1 = cache["Z1"]
    A1 = cache["A1"]
    P = cache["P"]
    Q = cache["Q"]

    W2 = params["W2"]
    N = X.shape[0]

    dZ2 = (P - Q) / N

    dW2 = A1.T @ dZ2
    db2 = np.sum(dZ2, axis=0)

    dA1 = dZ2 @ W2.T
    dZ1 = dA1 * (Z1 > 0)

    dW1 = X.T @ dZ1
    db1 = np.sum(dZ1, axis=0)

    return {
        "W1": dW1,
        "b1": db1,
        "W2": dW2,
        "b2": db2,
    }
```

Main idea:

```python
# hard labels
dZ2 = P - one_hot(y)

# soft labels / KL
dZ2 = P - Q
```

So difficulty is low. The math is almost identical to softmax cross-entropy.

## interview Q1

Compute KL Divergence (Example) – Given two simple distributions (p) and (q) over a discrete set, how do you compute the KL divergence $D_{KL}(p|q)$? For example, if (p = [0.5,0.5]) and (q=[0.8,0.2]), calculate $D_{KL}(p|q)$

I would answer:

KL divergence measures how different distribution `q` is from distribution `p` when `p` is treated as the reference.

The formula is:

$$
D_{KL}(p | q) = \sum_i p_i \log \frac{p_i}{q_i}
$$

For:

```
p = [0.5, 0.5]
q = [0.8, 0.2]
```

we compute:

$$
D_{KL}(p | q)=
0.5 \log \frac{0.5}{0.8}
+
0.5 \log \frac{0.5}{0.2}
$$

Using natural log:

$$
0.5(-0.4700) + 0.5(0.9163)
= -0.2350 + 0.4581
= 0.2231
$$

So:

$$
D_{KL}(p | q) \approx 0.223
$$

That is in **nats** because we used natural log. If we used log base 2, the answer would be in **bits**:

$$
0.2231 / \log(2) \approx 0.322
$$

Important interview note: KL is **not symmetric**:

$$
D_{KL}(p | q) \neq D_{KL}(q | p)
$$

And if `q_i = 0` where `p_i > 0`, KL becomes infinite.

## Interview Q2

```
Mode-Seeking vs. Covering (KL-Divergence) – Explain how the choice of KL divergence direction affects learned solutions. For example, minimizing KL(q‖p) (model q vs data p) tends to focus on the modes of p, while minimizing KL(p‖q) encourages covering the support of p
. What are the intuitive differences in the resulting model behavior?
```
I would answer like this:

KL is asymmetric, so the direction matters.

Assume:

```text
p = true data distribution
q = model distribution
```

The two objectives are:

$$
D_{KL}(p | q) = \sum_x p(x)\log \frac{p(x)}{q(x)}
$$

$$
D_{KL}(q | p) = \sum_x q(x)\log \frac{q(x)}{p(x)}
$$

### 1. Minimizing $D_{KL}(p | q)$, covering behavior

This is often called **forward KL**.

It heavily penalizes cases where:

$$$
p(x) > 0 \quad \text{but} \quad q(x) \approx 0
$$$

Meaning: if the data says something is possible, but the model assigns almost no probability to it, the loss becomes very large.

So the model is encouraged to **cover all regions where data exists**.

Behavior:

```text
less likely to miss modes
more likely to spread probability mass
can assign probability to low-density areas between modes
```

Example: if `p` has two modes and `q` is a single Gaussian, minimizing (KL(p | q)) may put `q` between the two modes and make it wide enough to cover both.

So it is **mode-covering**.

### 2. Minimizing (D_{KL}(q | p)), mode-seeking behavior

This is often called **reverse KL**.

It heavily penalizes cases where:

$$$
q(x) > 0 \quad \text{but} \quad p(x) \approx 0
$$$

Meaning: if the model puts probability somewhere the data distribution says is unlikely, it gets punished hard.

So the model prefers to place mass only where `p` is very high.

Behavior:

```text
sharp samples
avoids low-probability regions
may ignore some modes entirely
```

Example: if `p` has two modes and `q` is a single Gaussian, minimizing (KL(q | p)) may choose one mode and ignore the other, because placing mass between modes is punished.

So it is **mode-seeking**.

### Interview summary

I would say:

> The direction of KL controls what mistakes are expensive. $KL(p | q)$ punishes the model for missing data support, so it encourages broad coverage. $KL(q | p)$ punishes the model for putting mass where the data has little mass, so it prefers high-density regions and can collapse onto one mode. Forward KL is coverage-seeking, reverse KL is mode-seeking.


