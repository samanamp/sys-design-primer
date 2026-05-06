---
title: silu FFN/MLP
description: silu FFN/MLP
---

Full FFN is usually:

$$
\text{FFN}(x) =
\left(\text{SiLU}(xW_{\text{gate}}) \odot xW_{\text{up}}\right) W_{\text{down}}
$$

In code shape terms:
```python
gate = x @ W_gate   # [B, hidden]
up   = x @ W_up     # [B, hidden]

h = silu(gate) * up
out = h @ W_down
```
For LLaMA/Gemma-style MLPs, this version is the common one.

# Silu implementation

SiLU is:

$$
\text{silu}(x) = x \cdot \sigma(x)
$$

where:

$$
\sigma(x) =

\begin{cases}
\frac{1}{1 + e^{-x}} & x\ge0 \\
\frac{e^{x}}{1 + e^{x}} & x<0
\end{cases}
$$




Backward:

$$
\frac{d}{dx}\text{silu}(x)
=
\sigma(x) + x \cdot \sigma(x)(1 - \sigma(x))
$$

### NumPy implementation

```python
import numpy as np

def sigmoid(x):
    # Stable sigmoid
    return np.where(
        x >= 0,
        1 / (1 + np.exp(-x)),
        np.exp(x) / (1 + np.exp(x))
    )

def silu_forward(x):
    s = sigmoid(x)
    out = x * s

    cache = (x, s)
    return out, cache

def silu_backward(dout, cache):
    x, s = cache

    # d/dx [x * sigmoid(x)]
    dsilu_dx = s + x * s * (1 - s)

    dx = dout * dsilu_dx
    return dx
```


Conceptually:

```python
out = x * sigmoid(x)
dx = dout * (sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x)))
```

For interview code, this is enough.
## Swiglu FFN forward and backward

Yes. Here is a clean NumPy implementation for a **SwiGLU FFN**:

$$
\text{out} = \left(\text{SiLU}(xW_g + b_g) \odot (xW_u + b_u)\right)W_d + b_d
$$

```python
import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def silu(x):
    return x * sigmoid(x)

def silu_grad(x):
    s = sigmoid(x)
    return s * (1 + x * (1 - s))

def swiglu_ffn_forward(x, W_gate, b_gate, W_up, b_up, W_down, b_down):
    """
    x:      [B, D]
    W_gate: [D, H]
    W_up:   [D, H]
    W_down: [H, O]
    returns out: [B, O]
    """

    gate = x @ W_gate + b_gate      # [B, H]
    up   = x @ W_up + b_up          # [B, H]

    hidden = silu(gate) * up        # [B, H]
    out = hidden @ W_down + b_down  # [B, O]

    cache = (x, gate, up, hidden, W_gate, W_up, W_down)
    return out, cache


def swiglu_ffn_backward(dout, cache):
    """
    dout: [B, O]
    returns gradients for x and all parameters
    """

    x, gate, up, hidden, W_gate, W_up, W_down = cache

    # out = hidden @ W_down + b_down
    dW_down = hidden.T @ dout              # [H, O]
    db_down = np.sum(dout, axis=0)         # [O]
    dhidden = dout @ W_down.T              # [B, H]

    # hidden = silu(gate) * up
    silu_gate = silu(gate)

    dup = dhidden * silu_gate              # [B, H]
    dgate = dhidden * up * silu_grad(gate) # [B, H]

    # gate = x @ W_gate + b_gate
    dW_gate = x.T @ dgate                  # [D, H]
    db_gate = np.sum(dgate, axis=0)        # [H]

    # up = x @ W_up + b_up
    dW_up = x.T @ dup                      # [D, H]
    db_up = np.sum(dup, axis=0)            # [H]

    # x contributes to both branches
    dx = dgate @ W_gate.T + dup @ W_up.T   # [B, D]

    grads = {
        "dx": dx,
        "dW_gate": dW_gate,
        "db_gate": db_gate,
        "dW_up": dW_up,
        "db_up": db_up,
        "dW_down": dW_down,
        "db_down": db_down,
    }

    return grads
```

Key backward idea:

```python
hidden = silu(gate) * up

dup = dhidden * silu(gate)
dgate = dhidden * up * silu_grad(gate)
```

That is the core of SwiGLU backprop.
