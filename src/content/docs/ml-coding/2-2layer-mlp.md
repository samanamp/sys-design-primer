---
title: 2 layer regression
description: 2 layer regression
---
## 1. Forward pass shapes

Assume:

```text
X:  [B, D]
W1: [D, H]
b1: [H]
W2: [H, C]
b2: [C]
y:  [B] integer class labels
```

Forward:

```text
Z1 = X @ W1 + b1        # [B, H]
A1 = ReLU(Z1)           # [B, H]
Z2 = A1 @ W2 + b2       # [B, C], logits
loss = softmax_cross_entropy(Z2, y)
```

Stable softmax cross entropy:

```text
shifted = Z2 - max(Z2, axis=1)
probs = exp(shifted) / sum(exp(shifted), axis=1)
loss = mean(-log(probs[range(B), y]))
```

---

## 2. Backward pass derivation

Key shortcut:

```text
dZ2 = probs
dZ2[range(B), y] -= 1
dZ2 /= B
```

Then:

```text
dW2 = A1.T @ dZ2        # [H, C]
db2 = sum(dZ2, axis=0)  # [C]

dA1 = dZ2 @ W2.T        # [B, H]
dZ1 = dA1 * (Z1 > 0)    # [B, H]

dW1 = X.T @ dZ1         # [D, H]
db1 = sum(dZ1, axis=0)  # [H]

dX = dZ1 @ W1.T         # [B, D]
```

Main interview invariant:

```text
Gradient of a tensor has the same shape as that tensor.
```

So:

```text
dW1.shape == W1.shape
db1.shape == b1.shape
dW2.shape == W2.shape
db2.shape == b2.shape
dX.shape  == X.shape
```

---

## 3. NumPy implementation

```python
import numpy as np


def forward_backward(X, y, W1, b1, W2, b2):
    """
    X:  [B, D]
    y:  [B]
    W1: [D, H]
    b1: [H]
    W2: [H, C]
    b2: [C]
    """

    B = X.shape[0]

    # ----- forward -----
    Z1 = X @ W1 + b1              # [B, H]
    A1 = np.maximum(0, Z1)        # [B, H]
    logits = A1 @ W2 + b2         # [B, C]

    # stable softmax
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp_scores = np.exp(shifted)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

    loss = -np.log(probs[np.arange(B), y]).mean()

    # ----- backward -----
    dlogits = probs.copy()
    dlogits[np.arange(B), y] -= 1
    dlogits /= B                 # because loss is mean over batch

    dW2 = A1.T @ dlogits          # [H, C]
    db2 = np.sum(dlogits, axis=0) # [C]

    dA1 = dlogits @ W2.T          # [B, H]
    dZ1 = dA1 * (Z1 > 0)          # [B, H]

    dW1 = X.T @ dZ1               # [D, H]
    db1 = np.sum(dZ1, axis=0)     # [H]

    dX = dZ1 @ W1.T               # [B, D]

    grads = {
        "dX": dX,
        "dW1": dW1,
        "db1": db1,
        "dW2": dW2,
        "db2": db2,
    }

    cache = {
        "Z1": Z1,
        "A1": A1,
        "logits": logits,
        "probs": probs,
    }

    return loss, grads, cache
```

Example usage:

```python
np.random.seed(0)

B, D, H, C = 4, 5, 10, 3

X = np.random.randn(B, D)
y = np.array([0, 2, 1, 2])

W1 = 0.01 * np.random.randn(D, H)
b1 = np.zeros(H)

W2 = 0.01 * np.random.randn(H, C)
b2 = np.zeros(C)

loss, grads, cache = forward_backward(X, y, W1, b1, W2, b2)

print(loss)
for k, v in grads.items():
    print(k, v.shape)
```

Expected shapes:

```text
dX  [B, D]
dW1 [D, H]
db1 [H]
dW2 [H, C]
db2 [C]
```

---

## 4. Parameter update

```python
lr = 1e-1

W1 -= lr * grads["dW1"]
b1 -= lr * grads["db1"]
W2 -= lr * grads["dW2"]
b2 -= lr * grads["db2"]
```

Do not update `X` unless you explicitly want gradients with respect to the input.

---

## 5. PyTorch autograd version

Using same layout as NumPy:

```python
import torch
import torch.nn.functional as F

B, D, H, C = 4, 5, 10, 3

X = torch.randn(B, D, requires_grad=True)
y = torch.tensor([0, 2, 1, 2])

W1 = torch.randn(D, H, requires_grad=True) * 0.01
b1 = torch.zeros(H, requires_grad=True)

W2 = torch.randn(H, C, requires_grad=True) * 0.01
b2 = torch.zeros(C, requires_grad=True)

# Need leaf tensors if using optimizer manually
W1 = W1.detach().requires_grad_()
W2 = W2.detach().requires_grad_()

# forward
Z1 = X @ W1 + b1
A1 = F.relu(Z1)
logits = A1 @ W2 + b2

loss = F.cross_entropy(logits, y)

# backward
loss.backward()

print(loss.item())
print(X.grad.shape)   # [B, D]
print(W1.grad.shape)  # [D, H]
print(b1.grad.shape)  # [H]
print(W2.grad.shape)  # [H, C]
print(b2.grad.shape)  # [C]
```

PyTorch’s `F.cross_entropy(logits, y)` internally does:

```text
log_softmax(logits) + negative log likelihood loss
```

So you should pass raw logits, not softmax probabilities.

Wrong:

```python
loss = F.cross_entropy(torch.softmax(logits, dim=1), y)
```

Correct:

```python
loss = F.cross_entropy(logits, y)
```

---

## 6. How PyTorch autograd works

When you do:

```python
Z1 = X @ W1 + b1
A1 = F.relu(Z1)
logits = A1 @ W2 + b2
loss = F.cross_entropy(logits, y)
```

PyTorch builds a dynamic computation graph.

Each tensor remembers:

```text
how it was created
which tensors created it
how to backprop through that operation
```

When you call:

```python
loss.backward()
```

PyTorch walks the graph backward and applies chain rule.

So internally it computes the same gradients:

```text
dlogits
dW2, db2
dA1
dZ1 through ReLU mask
dW1, db1
dX
```

Gradients are stored in:

```python
X.grad
W1.grad
b1.grad
W2.grad
b2.grad
```

Important: gradients accumulate.

So this:

```python
loss.backward()
loss.backward()
```

would add gradients twice unless you clear them.

With an optimizer:

```python
optimizer.zero_grad()
loss.backward()
optimizer.step()
```

---

## 7. Common pitfalls

### 1. Forgetting `/ B`

If loss is averaged over batch, then:

```python
dlogits /= B
```

Without this, gradients are too large by factor `B`.

---

### 2. Wrong label shape

Correct:

```python
y.shape == [B]
```

Wrong:

```python
y.shape == [B, 1]
```

For PyTorch `cross_entropy`, labels should be integer class IDs, not one-hot vectors.

---

### 3. Applying softmax before cross entropy

Wrong:

```python
probs = torch.softmax(logits, dim=1)
loss = F.cross_entropy(probs, y)
```

Correct:

```python
loss = F.cross_entropy(logits, y)
```

---

### 4. Bias broadcasting confusion

Forward:

```python
Z1 = X @ W1 + b1
```

`b1` broadcasts from `[H]` to `[B, H]`.

Backward:

```python
db1 = dZ1.sum(axis=0)
```

Because every row used the same shared bias.

---

### 5. In-place ops

This can break autograd:

```python
A1.relu_()
```

Safer:

```python
A1 = F.relu(Z1)
```

In-place ops are sometimes okay, but in interviews avoid them unless you are sure.

---

### 6. Forgetting `.zero_grad()`

PyTorch accumulates gradients:

```python
optimizer.zero_grad()
loss.backward()
optimizer.step()
```

Without `zero_grad()`, gradients from previous batches leak into the current step.

---

## 8. Interview explanation in one pass

I would say:

> I start with batched input `X` of shape `[B, D]`. The first affine layer gives `Z1 = XW1 + b1`, shape `[B, H]`. ReLU keeps the same shape. The second affine gives logits `[B, C]`. For classification, I use stable softmax cross entropy directly from logits. In backward, the key simplification is that gradient of softmax plus cross entropy is `probs - one_hot(y)`, divided by batch size if the loss is averaged. From there, gradients follow by matrix calculus: `dW2 = A1.T @ dlogits`, `db2 = sum(dlogits)`, then propagate through ReLU with `(Z1 > 0)`, then `dW1 = X.T @ dZ1`, `db1 = sum(dZ1)`, and `dX = dZ1 @ W1.T`. In PyTorch, the same operations build a dynamic computation graph, and `.backward()` applies the same chain rule automatically, accumulating gradients into `.grad`. Common bugs are wrong label shape, applying softmax before cross entropy, forgetting the batch normalization factor, bad bias broadcasting, in-place ops, and forgetting to zero gradients.
