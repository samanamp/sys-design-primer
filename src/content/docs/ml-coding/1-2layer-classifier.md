---
title: 2 layer classifier
description: 2 layer classifier
---
## 1. Interview structure

I would say:

“We have a 2-layer classifier:

$$
X \rightarrow Z_1 = XW_1 + b_1 \rightarrow A_1 = ReLU(Z_1)
\rightarrow Z_2 = A_1W_2 + b_2
\rightarrow softmax \rightarrow CE
$$

Let:

```text
X:  (N, D)
W1: (D, H)
b1: (H,)
W2: (H, C)
b2: (C,)
y:  (N,) integer class labels
```

The key simplification is:

$$
\frac{\partial L}{\partial Z_2} = \frac{softmax(Z_2) - onehot(y)}{N}
$$

Then everything else is normal chain rule.”

---

## 2. Gradients

Forward:

```text
Z1 = X @ W1 + b1
A1 = max(0, Z1)
Z2 = A1 @ W2 + b2
P  = softmax(Z2)
L  = -mean(log(P[range(N), y]))
```

Backward:

```text
dZ2 = P
dZ2[range(N), y] -= 1
dZ2 /= N

dW2 = A1.T @ dZ2
db2 = sum(dZ2, axis=0)

dA1 = dZ2 @ W2.T
dZ1 = dA1 * (Z1 > 0)

dW1 = X.T @ dZ1
db1 = sum(dZ1, axis=0)
```

Update:

```text
W1 -= lr * dW1
b1 -= lr * db1
W2 -= lr * dW2
b2 -= lr * db2
```

---

## 3. Clean NumPy implementation

```python
import numpy as np


def init_params(D, H, C, seed=0):
    rng = np.random.default_rng(seed)
    return {
        "W1": 0.01 * rng.standard_normal((D, H)),
        "b1": np.zeros(H),
        "W2": 0.01 * rng.standard_normal((H, C)),
        "b2": np.zeros(C),
    }


def forward_loss(X, y, params):
    W1, b1 = params["W1"], params["b1"]
    W2, b2 = params["W2"], params["b2"]

    Z1 = X @ W1 + b1
    A1 = np.maximum(0, Z1)
    Z2 = A1 @ W2 + b2

    # stable softmax
    shifted = Z2 - np.max(Z2, axis=1, keepdims=True)
    exp_scores = np.exp(shifted)
    P = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

    N = X.shape[0]
    loss = -np.mean(np.log(P[np.arange(N), y] + 1e-12))

    cache = {
        "X": X,
        "y": y,
        "Z1": Z1,
        "A1": A1,
        "P": P,
    }

    return loss, cache


def backward(cache, params):
    X = cache["X"]
    y = cache["y"]
    Z1 = cache["Z1"]
    A1 = cache["A1"]
    P = cache["P"]

    W2 = params["W2"]
    N = X.shape[0]

    dZ2 = P.copy()
    dZ2[np.arange(N), y] -= 1
    dZ2 /= N

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


def train_step(X, y, params, lr=1e-1):
    loss, cache = forward_loss(X, y, params)
    grads = backward(cache, params)

    for k in params:
        params[k] -= lr * grads[k]

    return loss, grads
```

---

## 4. Gradient checking

Use central difference:

[
\frac{\partial L}{\partial \theta_i}
\approx
\frac{L(\theta_i + h) - L(\theta_i - h)}{2h}
]

```python
def rel_error(a, b):
    return np.max(
        np.abs(a - b) / np.maximum(1e-8, np.abs(a) + np.abs(b))
    )


def grad_check(X, y, params, grads, h=1e-5):
    errors = {}

    for name in params:
        p = params[name]
        grad_num = np.zeros_like(p)

        it = np.nditer(p, flags=["multi_index"], op_flags=["readwrite"])

        while not it.finished:
            idx = it.multi_index
            old = p[idx]

            p[idx] = old + h
            loss_plus, _ = forward_loss(X, y, params)

            p[idx] = old - h
            loss_minus, _ = forward_loss(X, y, params)

            p[idx] = old

            grad_num[idx] = (loss_plus - loss_minus) / (2 * h)

            it.iternext()

        errors[name] = rel_error(grad_num, grads[name])

    return errors
```

Toy test:

```python
np.random.seed(1)

N, D, H, C = 5, 4, 10, 3

X = np.random.randn(N, D).astype(np.float64)
y = np.array([0, 1, 2, 2, 1])

params = init_params(D, H, C, seed=42)

loss, cache = forward_loss(X, y, params)
grads = backward(cache, params)
errors = grad_check(X, y, params, grads)

print("loss:", loss)
for k, v in errors.items():
    print(k, v)
```

Expected output should be around:

```text
loss: 1.09866
W1 1e-7
b1 1e-8
W2 1e-8
b2 1e-10
```

Anything below around `1e-5` is usually fine for this tiny network. `1e-7` or better is very good.

---

## 5. Debugging checklist interviewers care about

Most common bugs:

```text
1. Forgot dZ2 /= N
   Symptom: gradients are exactly N times too large.

2. Modified P in-place without copy
   Wrong:
       dZ2 = P
   Correct:
       dZ2 = P.copy()

3. Wrong label shape
   y should be shape (N,), not (N, 1).
   Correct:
       P[np.arange(N), y]

4. Bad ReLU mask
   Correct:
       dZ1 = dA1 * (Z1 > 0)

5. Softmax not numerically stable
   Correct:
       shifted = logits - max(logits)

6. Using float32 for gradient check
   Prefer float64 for finite differences.

7. Updating params before gradient check
   Check gradients before applying SGD update.
```

A strong interview line:

“If gradient check fails, I first compare shapes, then check whether the error is a constant factor like `N`, then isolate layer-by-layer: check `dZ2`, then `dW2/db2`, then ReLU mask, then `dW1/db1`.”
