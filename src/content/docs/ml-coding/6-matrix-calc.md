---
title: Matrix Calculus
description: Matrix Calculus
---

Matrix Calculus (Linear Layer) – For a linear operation (Y = XW), what is the gradient $\partial Y/\partial W$? Use the identity $(AB)^T = B^T A^T$ to show that $\partial (XW)/\partial W = X^T$ when backpropagating through a batch

---

I would answer:

For a linear layer:

$$
Y = XW
$$

Assume shapes:

$$
X \in \mathbb{R}^{B \times D}

W \in \mathbb{R}^{D \times C}


Y \in \mathbb{R}^{B \times C}
$$

Strictly speaking, $\partial Y / \partial W$ is a higher-rank tensor. But in backprop, we usually care about the gradient of a scalar loss (L) with respect to (W).

Let:

$$
G = \frac{\partial L}{\partial Y}
$$

where:

$$
G \in \mathbb{R}^{B \times C}
$$

Then:

$$
\frac{\partial L}{\partial W} = X^T G
$$

Shape check:

$$
X^T \in \mathbb{R}^{D \times B}

G \in \mathbb{R}^{B \times C}
$$

so:

$$
X^T G \in \mathbb{R}^{D \times C}
$$

same shape as (W).

For a single element:

$$
Y_{b,c} = \sum_d X_{b,d}W_{d,c}
$$

So:

$$
\frac{\partial Y_{b,c}}{\partial W_{d,c}} = X_{b,d}
$$

Accumulating over the batch and over output dimensions gives:

$$
dW_{d,c} = \sum_b X_{b,d} , dY_{b,c}
$$

which is exactly:

$$
dW = X^T dY
$$

So the interview-safe answer is:

> For (Y=XW), the local Jacobian with respect to (W) contributes $X^T$ during backprop. Given upstream gradient (dY), the parameter gradient is $dW = X^T dY$. Similarly, the input gradient is $dX = dY W^T$.
