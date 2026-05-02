---
title: LLM Coding Q Bank
description: LLM Coding Q Bank
---

# Mock Interview: Basic Linear Algebra & Back-Propagation

- **Implement and Debug Backpropagation (NumPy)** – Given a small 2-layer neural network (Affine→ReLU→Affine→Softmax), derive the gradient update equations for all parameters (weights/biases) using the chain rule, then implement the forward pass, loss (softmax cross-entropy), and backward pass in NumPy【54†L132-L140】. Perform gradient checking with finite differences on a toy dataset, report relative errors, and debug any mismatches (e.g. missing factors or incorrect ReLU mask)【54†L139-L144】.  

- **Forward/Backward Pass (NumPy & PyTorch)** – Starting from a batched input \(X\) of shape \([B\times D]\), implement a feedforward and backpropagation for a simple classification net (linear layer \(W_1,b_1\) → ReLU → linear \(W_2,b_2\)). Compute the loss, derive gradients for inputs and all parameters, and ensure tensor shapes match batched operations【49†L74-L82】. Then express the same computation using PyTorch’s autograd: explain how the computation graph is built, how `.backward()` accumulates gradients, and common pitfalls (e.g. broadcasting, in-place ops)【49†L118-L127】.

- **NumPy Neural Network Layers and Debugging** – Implement basic layers in NumPy: for input \(X\) (shape \([B\times d_{\text{in}}]\)), weight \(W\) (\(d_{\text{in}}\times d_{\text{out}}\)), bias \(b\) (\(d_{\text{out}}\)), compute \(Y = XW + b\) and follow-up layers like ReLU and softmax【51†L75-L83】. Explain NumPy broadcasting (e.g. adding a \((d_{\text{out}})\) bias to each row of \(XW\))【51†L75-L83】. Describe how to debug common implementation bugs: transposed matrices, incorrect broadcasting, unstable softmax, and mismatched batch dimensions【51†L129-L133】.

- **Backpropagation Theory** – Explain the mathematical foundation of backpropagation. How does the chain rule allow efficient gradient computation in a neural net, and how are gradients propagated layer by layer? (E.g. clarifying how an error at the output layer is backpropagated through each layer to compute ∂Loss/∂W,∂Loss/∂b)【36†L178-L186】.

- **Gradient Checking** – What is gradient checking and how do you use it? Describe the process of verifying analytical gradients by finite differences, and how a small relative error signals correctness. For example, if your backprop gradient is off by a constant factor, how would gradient checking reveal it【54†L139-L144】.

- **Softmax & Cross-Entropy** – Derive the softmax activation and cross-entropy loss for a multiclass output. Explain why numerical stability tricks are needed (e.g. subtracting the maximum logit before exponentiating)【54†L148-L152】. How do these choices affect the backprop equations?

- **PyTorch Autograd Mechanics** – In PyTorch, what role do `requires_grad=True`, `.backward()`, and `.zero_grad()` play in computing gradients? Describe how the computational graph is built and traversed to compute gradients automatically【49†L118-L127】.

- **Broadcasting in NumPy** – Describe how NumPy broadcasting works for array operations of different shapes. For instance, given `X.shape = (B, d_in)` and `W.shape = (d_in, d_out)`, if you compute `X @ W + b` with `b.shape = (d_out,)`, how is `b` broadcast across the batch? What are the resulting shapes?

- **Transformer Attention (Implementation)** – Implement the scaled dot-product attention mechanism from scratch (using NumPy or PyTorch tensors). Given query, key, and value tensors, compute their dot-product attention: \( \text{Attention}(Q,K,V) = \text{softmax}(QK^T/\sqrt{d_k})V \). Extend to multi-head attention by splitting into heads and concatenating. Analyze time/space complexity【36†L152-L160】.

- **Debug a Broken Transformer** – You have a Transformer code that isn’t training correctly. Describe a systematic debugging approach from data input to optimization. What common issues would you check (e.g. tokenization/labels, tensor shapes, attention masks, positional encodings, train vs eval mode, loss setup, optimizer configs)? As a follow-up, explain how to adapt a Transformer to a classification task and verify it by doing one forward/backward pass【50†L78-L87】【50†L119-L127】.

- **Efficient Model Inference** – Given a large pre-trained model, how would you optimize inference for low memory and high speed? Discuss techniques like quantization (e.g. 8-bit weights), pruning redundant neurons, batching inputs, and leveraging hardware. How do these affect accuracy and latency【36†L202-L210】?

- **Noisy Label Aggregation** – You have a binary text classification dataset with multiple noisy annotators per example (labels conflict). How would you analyze and handle this noise? Outline data analysis (measuring annotator agreement, label consistency), and propose a strategy for training and evaluation: how to split data to avoid leakage, how to aggregate multiple labels into a target (e.g. majority vote, soft targets), and which loss/calibration methods to use. Discuss potential failure modes【52†L124-L132】【52†L144-L150】.

- **Filter Bad Annotations** – Some training labels are low-quality or adversarial. Describe practical methods to identify and filter them before training【48†L73-L81】. What signals (example-level or annotator-level) would you use? How do you distinguish truly hard examples from mislabeled ones? Would you remove, relabel, or down-weight suspicious data? How would you evaluate this filtering process and what fairness issues might arise【48†L73-L81】.

- **Attention KV Cache (Transformer)** – Given a decoder-only Transformer, fix bugs related to label shifting, positional embeddings, and masking so that training works【68†L74-L83】. Then extend it with a key-value (KV) cache for autoregressive decoding: ensure the attention reads/appends to the cache properly, apply causal masks only when needed, and use correct positional offsets. Verify that incremental generation with the cache matches the original output【68†L139-L147】.

- **Mode-Seeking vs. Covering (KL-Divergence)** – Explain how the choice of KL divergence direction affects learned solutions. For example, minimizing KL(q‖p) (model q vs data p) tends to focus on the modes of p, while minimizing KL(p‖q) encourages covering the support of p【57†L171-L174】. What are the intuitive differences in the resulting model behavior?

- **Accuracy vs. Metrics** – You have two classifiers with accuracies 85% and 82%. Which would you choose and why? Explain how other evaluation metrics (precision, recall, F1, ROC AUC) or dataset considerations (class imbalance, error costs) influence your decision【67†L1188-L1190】.

- **Perfect Accuracy ⇒ Loss Bounds** – If a classifier achieves 100% accuracy on its training set, what are the minimum and maximum possible values of the cross-entropy loss on a single example【63†L1154-L1157】? (Hint: with perfect accuracy, the true class probability is 1, so loss can be arbitrarily low or high depending on confidence).

- **Compute KL Divergence (Example)** – Given two simple distributions \(p\) and \(q\) over a discrete set, how do you compute the KL divergence \(D_{KL}(p\|q)\)? For example, if \(p = [0.5,0.5]\) and \(q=[0.8,0.2]\), calculate \(D_{KL}(p\|q)\)【63†L1154-L1157】.

- **Matrix Calculus (Linear Layer)** – For a linear operation \(Y = XW\), what is the gradient \(\partial Y/\partial W\)? Use the identity \((AB)^T = B^T A^T\) to show that \(\partial (XW)/\partial W = X^T\) when backpropagating through a batch【54†L132-L135】.

- **Bayes’ Theorem** – State Bayes’ theorem and its use in probabilistic modeling: for events \(A\) and \(B\), \(P(A|B) = \frac{P(B|A)P(A)}{P(B)}\). How does this update our beliefs with new evidence? (General ML interview topics often include this theorem【74†L128-L133】).

**Sources:** We gathered these questions from interview reports and Q&A for OpenAI ML roles【54†L132-L140】【49†L74-L83】【51†L75-L83】【50†L78-L87】【48†L73-L81】【52†L124-L132】【36†L152-L160】【57†L171-L174】【67†L1188-L1190】【63†L1154-L1161】【68†L74-L83】【54†L139-L144】【49†L118-L127】. Other included questions are extrapolated from common ML interview topics. Each question above is supported by the cited sources.