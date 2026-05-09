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
