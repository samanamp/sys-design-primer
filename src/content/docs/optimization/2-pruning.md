---
title: "Pruning: Why Fewer Weights Do Not Always Mean Faster Models"
description: "A staff-level guide to neural network pruning for model optimization: sparsity, saliency, structured pruning, LLM pruning, hardware reality, and interview-ready tradeoffs."
---

# Pruning: Why Fewer Weights Do Not Always Mean Faster Models

Pruning is the family of techniques that removes parts of a trained model while trying to preserve quality. The removed parts can be individual weights, blocks of weights, attention heads, MLP channels, layers, tokens, experts, or even whole branches of computation. At first glance, pruning sounds almost too simple: large neural networks are overparameterized, so delete the parts that matter least.

The engineering reality is more subtle. **A pruned model is not automatically faster.** It may be smaller on disk, easier to fit in memory, and cheaper to transmit, yet no faster at inference if the sparsity pattern does not map to efficient kernels. This is the central interview point. Pruning is an optimization technique only when the removed computation is visible to the runtime, compiler, and hardware.

For a staff engineer, the important question is not "can we set 50% of the weights to zero?" The important question is:

> Which computational structure can we remove without unacceptable quality loss, and can our serving stack actually turn that removal into lower latency, higher throughput, lower memory use, or lower cost?

This article builds that answer from first principles.

---

## 1. The Interview Mental Model

When pruning comes up in an interview, answer in this order:

1. **Goal:** Are we optimizing latency, throughput, memory footprint, model size, or cost?
2. **Unit:** Are we pruning weights, blocks, channels, heads, layers, or tokens?
3. **Criterion:** How do we decide what is unimportant?
4. **Recovery:** Do we fine-tune, distill, use LoRA, or accept the degradation?
5. **Runtime:** Can the target hardware and serving stack exploit the pruned structure?

This ordering matters because the wrong unit can make the whole project irrelevant. If the user-facing problem is H100 inference latency and we produce a checkpoint with 60% random zeros, we have probably optimized a file format, not the product.

The useful taxonomy is:

| Pruning type | Removes | Quality profile | Runtime profile |
| --- | --- | --- | --- |
| Unstructured | Individual weights | Best quality at high sparsity | Needs sparse kernels; often no dense-GPU speedup |
| Semi-structured | Patterns such as 2:4 or blocks | More constrained than unstructured | Can map to hardware acceleration |
| Structured | Channels, heads, layers, rows/columns | More quality risk | Produces smaller dense computation |
| Dynamic | Tokens, branches, early exits per input | Input-dependent | Needs scheduler/runtime support |

That table is the core. Most pruning debates are arguments about which row applies.

One way to keep the decision straight:

```text
Do we need real latency / throughput improvement?
|
+-- No, mostly storage or transfer size
|   |
|   +-- Unstructured pruning may be acceptable.
|
+-- Yes, on dense GPU/TPU inference
    |
    +-- Can our kernels exploit sparse metadata?
    |   |
    |   +-- Yes: consider semi-structured or block sparsity.
    |   +-- No: prefer structured pruning or a smaller dense model.
    |
    +-- Is quality loss too high after structural removal?
        |
        +-- Yes: use distillation / LoRA recovery, or switch to distillation.
        +-- No: benchmark the exported model in the real serving path.
```

---

## 2. What Pruning Optimizes

Pruning can optimize several different things:

1. **Model size:** fewer stored parameters.
2. **Memory bandwidth:** fewer weights loaded from memory.
3. **Activation memory:** smaller intermediate tensors.
4. **FLOPs:** fewer arithmetic operations.
5. **Latency:** lower wall-clock time per request.
6. **Throughput:** more requests or tokens per second per accelerator.
7. **Energy and cost:** fewer joules and fewer accelerator-hours.

These are related but not equivalent. A model with 50% unstructured sparsity has half of its scalar weights zeroed, but a dense GPU kernel still reads the full dense matrix and performs dense matrix multiplication unless the runtime uses a sparse representation and sparse kernels. In that case, model size improves but latency may not.

This distinction is why pruning conversations often split into two camps:

- **Compression pruning:** make the model smaller, perhaps for storage, deployment, or edge devices.
- **Acceleration pruning:** make the model faster or cheaper to run.

Hiring interviews for optimization roles usually care more about the second. Compression matters, but acceleration is where the systems judgment shows up.

---

## 3. The Basic Formulation

Let a neural network be parameterized by weights $W$. Pruning introduces a mask $M$ with the same shape as some subset of parameters:

$$
W' = M \odot W
$$

where $\odot$ is elementwise multiplication and $M_i \in \{0, 1\}$. If $M_i = 0$, weight $W_i$ is removed.

A generic pruning objective is:

$$
\min_{W, M} \mathcal{L}(M \odot W; \mathcal{D}) \quad \text{subject to} \quad \|M\|_0 \leq k
$$

Here:

- $\mathcal{L}$ is the training or validation loss.
- $\mathcal{D}$ is the data distribution or calibration set.
- $\|M\|_0$ counts the number of nonzero entries in the mask.
- $k$ is the parameter budget after pruning.

Equivalently, if target sparsity is $s$, then:

$$
\frac{\#\{i : M_i = 0\}}{\#\{i : M_i\}} = s
$$

For example, $s = 0.5$ means half the prunable units are removed.

This is a combinatorial optimization problem. Searching over all possible masks is impossible for modern networks, so pruning algorithms use approximations. Most methods answer two questions:

1. **Saliency:** how important is a weight, channel, head, or layer?
2. **Recovery:** after removing it, how do we restore quality?

---

## 4. A Short Historical Context

Pruning is not new. It is almost as old as practical neural networks.

In 1989, LeCun, Denker, and Solla introduced **Optimal Brain Damage**. The idea was to estimate the loss increase caused by removing a weight using second-order information. If removing a weight barely changes the loss, remove it. This paper matters because it framed pruning as a sensitivity problem, not just a magnitude trick.

In 1993, Hassibi, Stork, and Wolff introduced **Optimal Brain Surgeon**, a stronger second-order method that used the inverse Hessian to account for interactions between weights. Optimal Brain Surgeon was more principled but more expensive.

In 2015, Han, Mao, and Dally published **Deep Compression**, which popularized the modern compression pipeline: pruning, trained quantization, and Huffman coding. It showed that neural networks could be drastically compressed with small accuracy loss, especially on CNN-era models such as AlexNet and VGG. This work helped turn pruning from a research curiosity into a deployment technique.

In 2018, Frankle and Carbin proposed the **Lottery Ticket Hypothesis**: dense randomly initialized networks contain sparse subnetworks that, when trained from their original initialization, can reach comparable accuracy. This shifted part of the pruning discussion from "how do we compress a trained network?" to "why are overparameterized networks trainable, and where are the useful subnetworks?"

In the LLM era, pruning came back under a harder constraint: models are huge, but dense matrix multiplication is extremely optimized. Methods such as **SparseGPT**, **Wanda**, **Movement Pruning**, and **LLM-Pruner** explored ways to prune large pretrained transformers with limited or no retraining. The strongest lesson from this period is that a pruning result must be judged with both model quality and hardware speedup in mind.

---

## 5. Magnitude Pruning

The simplest pruning method is magnitude pruning: remove weights with the smallest absolute value.

For each weight $w_i$, define saliency:

$$
S_i = |w_i|
$$

Then prune the weights with the lowest $S_i$ until the target sparsity is reached.

Magnitude pruning works surprisingly well because small weights often contribute less to activations. It is cheap, easy to implement, and can be applied globally or layer by layer.

**Global magnitude pruning** ranks all candidate weights together:

$$
M_i =
\begin{cases}
0 & \text{if } |w_i| \leq \tau \\
1 & \text{otherwise}
\end{cases}
$$

where $\tau$ is chosen to reach the desired sparsity.

**Layerwise magnitude pruning** chooses a different threshold $\tau_l$ per layer. This avoids over-pruning sensitive layers, but it requires deciding the sparsity budget per layer.

The biggest limitation is that magnitude does not directly estimate loss impact. A small weight can matter if it sits in a sensitive direction, and a large weight can be redundant if other weights compensate for it.

Magnitude pruning is still worth knowing because it is the baseline. If a proposed pruning algorithm cannot beat magnitude pruning under the same evaluation and runtime conditions, it is probably not useful.

---

## 6. Second-Order Pruning: Loss Sensitivity

The classic theoretical foundation starts with a Taylor expansion of the loss around trained weights $W$:

$$
\mathcal{L}(W + \Delta W) \approx \mathcal{L}(W) + g^T \Delta W + \frac{1}{2}\Delta W^T H \Delta W
$$

where:

- $g = \nabla_W \mathcal{L}$ is the gradient.
- $H = \nabla_W^2 \mathcal{L}$ is the Hessian.

Near a trained local optimum, $g \approx 0$, so the loss increase is approximated by:

$$
\Delta \mathcal{L} \approx \frac{1}{2}\Delta W^T H \Delta W
$$

If pruning a single weight $w_i$ means setting it to zero, then $\Delta w_i = -w_i$. A diagonal Hessian approximation gives:

$$
\Delta \mathcal{L}_i \approx \frac{1}{2} H_{ii} w_i^2
$$

This is the core idea behind Optimal Brain Damage: prune weights whose removal has the smallest estimated effect on loss.

Optimal Brain Surgeon goes further. Instead of assuming the Hessian is diagonal, it allows other weights to compensate after one weight is removed. The resulting saliency for weight $i$ is often written as:

$$
S_i = \frac{w_i^2}{2 [H^{-1}]_{ii}}
$$

The appeal is clear: saliency is not just "is this weight small?" but "how much does loss rise if this weight is removed, accounting for curvature?"

The problem is cost. Exact Hessians are too expensive for modern models. For a model with $n$ parameters, the full Hessian has $n^2$ entries. With billions of parameters, this is impossible. Modern second-order pruning methods use approximations: blockwise Hessians, diagonal estimates, low-rank approximations, Fisher approximations, or layerwise calibration.

The staff-level takeaway:

> Second-order pruning gives the right conceptual model, but production pruning usually uses approximations that fit the memory, data, and runtime budget.

---

## 7. Unstructured vs Structured Pruning

This is the most important practical distinction.

### Unstructured pruning

Unstructured pruning removes individual scalar weights. A matrix remains the same shape, but many entries are zero:

$$
Y = X(W \odot M)
$$

Advantages:

- Usually preserves quality better at the same parameter sparsity.
- Easy to apply with magnitude, SparseGPT, Wanda, or movement methods.
- Can reach high sparsity numerically.

Disadvantages:

- Hard to accelerate on dense hardware.
- Sparse metadata adds overhead.
- Random sparsity causes irregular memory access.
- Dense kernels ignore the zeros unless replaced.

Unstructured pruning is often useful for compression, but it is not automatically useful for inference latency.

### Structured pruning

Structured pruning removes entire units of computation: rows, columns, channels, heads, layers, experts, or blocks. The resulting tensors become smaller and can use normal dense kernels.

Examples:

- Remove MLP hidden channels.
- Remove attention heads.
- Remove transformer layers.
- Remove rows or columns from projection matrices.
- Remove convolution filters.
- Remove blocks in block-sparse matrices.

Suppose a transformer MLP is:

$$
\text{MLP}(x) = W_2 \sigma(W_1 x)
$$

where $W_1 \in \mathbb{R}^{d_{ff} \times d_{model}}$ and $W_2 \in \mathbb{R}^{d_{model} \times d_{ff}}$. If we prune MLP channel $j$, we remove row $j$ from $W_1$ and column $j$ from $W_2$. The hidden width $d_{ff}$ decreases, so dense matrix multiplications become smaller.

This is acceleration-friendly. The runtime does not need sparse kernels. It just sees smaller dense matrices.

The tradeoff is quality. Structured pruning removes entire computational features, so it is more destructive than zeroing scattered weights.

Interview phrase:

> Unstructured pruning usually wins on compression-per-quality. Structured pruning usually wins on deployable speedup.

Here is the shape-level difference:

```text
Dense matrix:

  W: d x h
  +-----------------------+
  | x x x x x x x x x x x |
  | x x x x x x x x x x x |
  | x x x x x x x x x x x |
  | x x x x x x x x x x x |
  +-----------------------+

Unstructured pruning:

  W: still d x h
  +-----------------------+
  | x 0 x x 0 x 0 x x 0 x |
  | 0 x x 0 x x x 0 x x 0 |
  | x x 0 x 0 x x x 0 x x |
  | x 0 x x x 0 x x x 0 x |
  +-----------------------+
  Same dense shape unless sparse kernels are used.

Structured channel pruning:

  W: d x h' where h' < h
  +---------------+
  | x x x x x x x |
  | x x x x x x x |
  | x x x x x x x |
  | x x x x x x x |
  +---------------+
  Smaller dense GEMM; easier for normal kernels to accelerate.
```

---

## 8. Semi-Structured Sparsity: The Hardware Compromise

Modern accelerators sometimes support constrained sparsity patterns. NVIDIA Ampere and later GPUs support **2:4 sparsity** for certain matrix operations: in every group of four values, exactly two are nonzero.

A 2:4 mask satisfies:

$$
\|M_{g}\|_0 = 2 \quad \text{for each group } g \text{ of 4 weights}
$$

This is less flexible than arbitrary unstructured sparsity but more hardware-friendly. The hardware can skip known zero positions with compact metadata and predictable access patterns.

There are also patterns such as 1:4, 4:8, and block sparsity. The broader principle is:

> Sparsity becomes useful when it is regular enough for the hardware and compiler to exploit.

Semi-structured sparsity is attractive because it sits between two extremes:

- Fully unstructured: better quality at high sparsity, poor hardware utilization.
- Fully structured: strong speedup, larger quality hit.

However, actual speedups depend on dtype, tensor shapes, kernel maturity, batch size, and whether the workload is compute-bound or memory-bound. A theoretical 2x reduction in multiplications rarely becomes a 2x end-to-end latency win.

---

## 9. Pruning Transformers

For transformers, pruning can target several structures.

### Attention heads

Multi-head attention has several heads:

$$
\text{MHA}(X) = \text{Concat}(h_1, ..., h_H) W_O
$$

where:

$$
h_i = \text{Attention}(XW_i^Q, XW_i^K, XW_i^V)
$$

Head pruning removes some $h_i$. This reduces projection sizes and attention computation if implemented structurally. But not all heads are equal: some are redundant, some specialize in positional or syntactic behavior, and some matter only for certain tasks.

Common saliency signals:

- Magnitude of head output projection weights.
- Average activation norm.
- Gradient-based loss sensitivity.
- Change in validation loss when ablating a head.

### MLP channel pruning

Transformer MLPs often contain a large fraction of the model's parameters and FLOPs. In many decoder-only LLMs, the MLP block dominates per-token compute. Pruning MLP intermediate channels can therefore be more impactful than pruning attention heads.

For gated MLPs such as SwiGLU:

$$
\text{MLP}(x) = W_{down}(\text{SiLU}(W_{gate}x) \odot W_{up}x)
$$

Pruning an intermediate channel must consistently remove corresponding rows or columns from $W_{gate}$, $W_{up}$, and $W_{down}$. Missing one of these creates shape errors or silently changes the computation incorrectly.

### Layer pruning

Layer pruning removes entire transformer blocks. If a model has $L$ layers and we remove $r$ layers, depth becomes $L-r$.

This has obvious latency benefits because each token passes through fewer blocks. It is also risky because depth carries abstraction, composition, and iterative refinement. Layer pruning may work better when followed by distillation or supervised fine-tuning.

### Token pruning

Token pruning removes tokens from the sequence during computation. This is common in vision transformers and sometimes useful in long-context language workloads. For autoregressive LLMs, token pruning is difficult because generated tokens depend on previous context and positional structure. Long-context systems more often use context compression, retrieval, sliding windows, or attention sparsity rather than naive token deletion.

---

## 10. One-Shot, Iterative, and Training-Time Pruning

Pruning can happen at different points in the model lifecycle.

### One-shot pruning

One-shot pruning takes a trained model, computes saliency once, prunes to a target sparsity, and optionally does a small amount of recovery tuning.

This is operationally attractive for LLMs because full retraining is expensive. SparseGPT and Wanda are examples of LLM-era one-shot approaches. They use calibration data to decide which weights can be removed with minimal quality loss.

The risk is that a single pruning step can damage the model in ways that are hard to recover, especially at high sparsity.

### Iterative pruning

Iterative pruning removes a smaller fraction, fine-tunes, then repeats:

1. Train or load dense model.
2. Prune 10%.
3. Fine-tune.
4. Prune more.
5. Fine-tune again.

This often preserves quality better than one-shot pruning because the model adapts gradually. The cost is engineering and compute time.

### Training-time pruning

Training-time pruning learns sparsity during training or fine-tuning. A method may add sparsity-inducing regularization:

$$
\mathcal{L}_{total} = \mathcal{L}_{task} + \lambda \|W\|_1
$$

The $L_1$ term encourages weights to move toward zero. Other methods learn binary gates or pruning scores jointly with task loss.

Movement pruning is important here. Instead of pruning weights simply because they are small, it asks whether weights are moving toward or away from zero during fine-tuning. This is especially relevant for transfer learning, where pretrained weights may be small but still important, or large but no longer useful for the downstream task.

---

## 11. LLM Pruning: What Changes at Scale

LLM pruning has different constraints from pruning older CNNs or small transformers.

### Full retraining is usually unavailable

For a frontier-scale or even 70B-class model, full retraining after pruning is usually too expensive. Practical methods rely on:

- Calibration data.
- Lightweight fine-tuning.
- LoRA recovery.
- Distillation from the original dense model.
- Post-training pruning.

### Perplexity is not enough

A pruned LLM can have acceptable perplexity but degraded instruction following, tool use, coding ability, multilingual behavior, safety behavior, or reasoning depth. Evaluation must include capability slices.

A staff engineer should propose an evaluation matrix:

- Base language modeling perplexity on held-out text.
- Instruction-following benchmark.
- Coding benchmark if the product uses code.
- Math/reasoning benchmark if relevant.
- RAG and long-context tasks if served in production.
- Safety and refusal behavior.
- Latency, throughput, memory, and cost.
- Regression tests for high-value product prompts.

### Width, depth, and attention are not equally pruneable

Different structures have different quality profiles. Removing a few MLP channels may be less visible than removing late transformer layers. Removing attention heads might look safe on broad metrics but hurt long-context retrieval or tool-use behavior. Pruning embeddings or output heads can be especially risky because vocabulary distribution and rare token behavior matter.

### Sparse speedup is hard

LLM inference is dominated by highly optimized dense linear algebra. During decode, batch sizes may be small and memory bandwidth can dominate. During prefill, large matrix multiplications can be compute-bound. Sparse kernels must beat dense tensor-core kernels, which is a high bar.

This is why many production teams prefer techniques that preserve dense computation:

- Smaller dense models.
- Distillation.
- Structured pruning.
- Low-rank factorization.
- Quantization.
- Speculative decoding.
- Architectural changes such as GQA or MoE.

Pruning is still relevant, but the deployment story must be explicit.

---

## 12. Recent Real-World Examples

Public LLM pruning examples exist, but pruning is not as common in mainstream LLM serving as quantization, batching, KV-cache work, or distillation. The reason is not that pruning is fake. The reason is that pruning only pays when the exported model and runtime actually exploit the sparsity or smaller dense shape.

Recent examples worth knowing:

- **SparseGPT on OPT and BLOOM.** SparseGPT showed one-shot pruning on OPT and BLOOM models, including OPT-175B and BLOOM-176B, reaching around 50-60% unstructured sparsity with limited perplexity degradation and no full retraining. This is an important research milestone, but the result is mostly a compression/sparsity result unless paired with sparse kernels. Paper: [SparseGPT](https://arxiv.org/abs/2301.00774).

- **Wanda on LLaMA and LLaMA-2.** Wanda made LLM pruning simpler by using weight magnitude plus activation magnitude, evaluated across LLaMA-family models. It is useful because it is a strong, cheap baseline: if a more complicated pruning method barely beats Wanda, the complexity may not be justified. Paper: [Wanda](https://arxiv.org/abs/2306.11695).

- **LLM-Pruner on LLaMA, Vicuna, and ChatGLM.** LLM-Pruner focuses on structural pruning, which is closer to deployable acceleration because it can remove channels or coupled structures rather than merely inserting zeros. Paper: [LLM-Pruner](https://arxiv.org/abs/2305.11627).

- **Sheared-LLaMA.** Sheared-LLaMA pruned LLaMA2-7B down to 1.3B and 2.7B models, then continued training. This is a good example of pruning as a way to derive a smaller dense model, not just a sparse checkpoint. Project: [Sheared-LLaMA](https://princeton-nlp.github.io/sheared-llama/).

- **NVIDIA Minitron.** NVIDIA's Minitron work compresses models such as Llama 3.1 8B and Mistral NeMo 12B using structured pruning plus knowledge distillation. This is one of the more production-shaped examples because it combines pruning with recovery and focuses on useful smaller models. References: [NVIDIA research page](https://research.nvidia.com/publication/2024-08_llm-pruning-and-distillation-practice-minitron-approach), [NVIDIA pruning/distillation blog](https://developer.nvidia.com/blog/how-to-prune-and-distill-llama-3-1-8b-to-an-nvidia-llama-3-1-minitron-4b-model/).

- **2:4 Sparse Llama.** Neural Magic / Red Hat released Sparse-Llama-3.1-8B-2of4, a 50% pruned Llama 3.1 8B variant using a 2:4 semi-structured sparsity pattern. This matters because 2:4 sparsity maps to NVIDIA Sparse Tensor Cores on supported hardware and runtimes. References: [Red Hat Sparse Llama article](https://developers.redhat.com/articles/2025/02/28/24-sparse-llama-smaller-models-efficient-gpu-inference), [NVIDIA Ampere sparsity overview](https://developer.nvidia.com/blog/accelerating-inference-with-sparsity-using-ampere-and-tensorrt/).

So, is pruning common? **In research and model-compression pipelines, yes. In general-purpose production LLM serving, less so.** Production teams more often reach first for quantization, distillation, better batching, prompt/prefix caching, speculative decoding, and kernel upgrades because those techniques have more predictable serving wins. Pruning becomes common only in narrower settings: edge deployment, CPU inference, hardware-supported 2:4 sparsity, deriving smaller dense models, or organizations with enough kernel/runtime control to make sparse models fast.

The interview answer should be:

> Pruning is real and has strong public examples, but it is not the default first lever for LLM serving. I would use it when the model format, runtime, and hardware can convert removed parameters into real speed, memory, or cost savings.

---

## 13. SparseGPT and Wanda: Post-Training LLM Pruning

Two LLM-era methods are worth knowing.

### SparseGPT

SparseGPT is a one-shot pruning method designed for very large GPT-style models. It prunes layer by layer and uses approximate second-order information from calibration data. The method is related in spirit to Optimal Brain Surgeon but made practical for huge models through blockwise approximations.

At a high level, for a linear layer:

$$
Y = XW
$$

the goal is to find a sparse $W'$ such that:

$$
\|XW - XW'\|_2^2
$$

is small on calibration activations $X$. This matters because preserving the layer's output on realistic inputs is more useful than preserving weights in isolation.

SparseGPT showed that large open GPT-family models could be pruned to substantial unstructured sparsity with limited perplexity degradation and without full retraining. That was a major result because earlier pruning methods often did not scale cleanly to 100B+ parameter models.

### Wanda

Wanda, short for pruning by weights and activations, uses a simpler saliency score based on both weight magnitude and input activation magnitude. A representative score for weight connecting input dimension $j$ to output dimension $i$ is:

$$
S_{ij} = |W_{ij}| \cdot \|X_j\|_2
$$

where $X_j$ is the calibration activation for input feature $j$.

The intuition: a weight matters more if it is large and it multiplies an input feature that is frequently active. This is more data-aware than pure magnitude pruning but much cheaper than heavy second-order methods.

SparseGPT and Wanda are useful interview examples because they show the modern trend: post-training, calibration-aware, layerwise pruning designed around LLM constraints.

---

## 14. Recovery: Fine-Tuning, Distillation, and LoRA

Pruning damages the model. Recovery tries to repair it.

### Fine-tuning

The simplest recovery step is supervised fine-tuning after pruning. The model adapts remaining weights to compensate for removed capacity.

The risk is overfitting to the recovery dataset or damaging general capabilities. Recovery data should resemble the deployment distribution but include broad coverage.

### Distillation

Distillation trains the pruned model to match the dense teacher. For logits $z_T$ from the teacher and $z_S$ from the student:

$$
p_T = \text{softmax}(z_T / T), \quad p_S = \text{softmax}(z_S / T)
$$

The distillation loss is often:

$$
\mathcal{L}_{KD} = T^2 \cdot KL(p_T \| p_S)
$$

where $T$ is the temperature.

Distillation is useful because the dense model provides richer supervision than one-hot labels. For LLMs, sequence-level distillation can also use teacher-generated responses.

### LoRA recovery

For LLMs, LoRA is an attractive recovery tool because it adapts the pruned model cheaply. A frozen weight matrix $W$ is augmented with a low-rank update:

$$
W' = W + BA
$$

where $B \in \mathbb{R}^{d \times r}$, $A \in \mathbb{R}^{r \times k}$, and $r \ll \min(d,k)$.

After structured pruning, LoRA can recover some behavior without updating all remaining parameters. Some pruning pipelines prune first, then use LoRA to regain instruction quality.

---

## 15. The Runtime Reality

The most common pruning failure mode is reporting parameter sparsity as if it were latency speedup.

Suppose a dense layer computes:

$$
Y = XW
$$

with $X \in \mathbb{R}^{B \times d}$ and $W \in \mathbb{R}^{d \times h}$. Dense compute is roughly:

$$
2Bdh
$$

floating-point operations.

If we prune 50% of scalar weights but still call the same dense GEMM, the FLOPs executed by the kernel are still approximately:

$$
2Bdh
$$

The zeros are multiplied anyway.

If instead we structurally prune hidden dimension $h$ down to $h' = (1-s)h$, dense compute becomes:

$$
2Bd h' = 2Bd(1-s)h
$$

Now the kernel does less work because the matrix is smaller.

This is why acceleration-friendly pruning usually needs one of the following:

- Smaller dense tensors after structural pruning.
- Hardware-supported semi-structured sparsity.
- Mature sparse kernels for the target shape and dtype.
- Compiler support that lowers sparse patterns efficiently.
- A serving engine that can load and schedule the pruned representation.

Without one of these, pruning may only save disk space.

The practical runtime path looks like this:

```text
Pruning result
|
+-- Masked checkpoint only
|   |
|   +-- Dense loader materializes W with zeros
|       |
|       +-- Dense GEMM runs same shapes -> little/no latency win
|
+-- Exported smaller dense checkpoint
|   |
|   +-- Tensor shapes shrink
|       |
|       +-- Existing dense kernels run less work -> credible speedup
|
+-- Sparse checkpoint + sparse runtime
    |
    +-- Sparse metadata, sparse kernels, supported dtype/shape
        |
        +-- Possible speedup, but must beat dense tensor cores
```

---

## 16. How to Run a Pruning Program

A serious pruning program should be staged like an optimization project, not a notebook experiment.

### Step 1: Define the target

Be precise:

- Reduce p50/p95/p99 latency?
- Increase tokens/sec/GPU?
- Fit a model on a smaller GPU?
- Reduce cold-start time?
- Reduce memory so KV cache capacity increases?
- Reduce cost at a fixed quality bar?

The target changes the pruning method. If the goal is to fit on an edge device, unstructured compression may help. If the goal is H100 serving latency, structured pruning or semi-structured sparsity is more likely to matter.

### Step 2: Establish dense baselines

Measure:

- Quality metrics.
- Latency by prompt length and generation length.
- Prefill throughput.
- Decode throughput.
- Memory footprint.
- GPU utilization.
- Kernel breakdown.

Without this, there is no way to know whether pruning helped.

### Step 3: Choose pruning granularity

Pick the unit that maps to the goal:

- Scalar weights for compression.
- 2:4 or block sparsity for hardware sparse acceleration.
- MLP channels for dense speedup.
- Attention heads for architectural simplification.
- Layers for aggressive latency reduction.
- Tokens for long-context compute reduction.

### Step 4: Rank and prune

Use a saliency method appropriate for the budget:

- Magnitude for baseline.
- Activation-aware magnitude for cheap data-aware pruning.
- Gradient or movement-based scores during fine-tuning.
- Approximate second-order methods for higher-quality post-training pruning.
- Ablation-based methods for heads or layers.

### Step 5: Recover

Use fine-tuning, distillation, LoRA, or a combination. Track both broad metrics and product-specific regressions.

### Step 6: Prove speedup in the real runtime

Export the model, load it in the actual serving stack, and benchmark under realistic traffic. Include batching, sequence lengths, cache behavior, and concurrency. A pruning result that only speeds up a custom microbenchmark is not enough.

---

## 17. When I Would Not Prune

Pruning is not always the right answer. In an interview, it is valuable to say when you would avoid it.

I would be skeptical of pruning when:

- The bottleneck is queueing, routing, tokenization, network latency, or KV cache pressure rather than model compute.
- The serving stack cannot exploit the sparsity pattern.
- A smaller dense model already exists and can be distilled more cleanly.
- The model is safety-sensitive and the evaluation suite is too weak to catch rare regressions.
- Tensor parallel partitioning, fused kernels, or quantization formats would need major rework.
- The requested speedup is small enough that kernel upgrades, batching, or prompt caching are lower-risk.

For many LLM products, distillation or serving optimization beats pruning on engineering ROI. Pruning becomes attractive when there is a clear deployment path: fit a model into a target memory budget, unlock hardware-supported sparsity, or remove dense structures that dominate latency.

---

## 18. Interview-Ready Tradeoffs

Here are concise statements that signal real understanding.

**Pruning is not one technique.** It is a design space over what to remove, how to score importance, when to prune, how to recover, and how the runtime exploits the result.

**Unstructured sparsity is quality-friendly but hardware-hostile.** It can reduce parameter count dramatically, but dense accelerators do not automatically get faster.

**Structured sparsity is hardware-friendly but quality-expensive.** Removing channels, heads, or layers produces smaller dense computation, but it takes away whole features.

**Calibration data matters.** Activation-aware pruning beats weight-only pruning when weights interact strongly with the input distribution.

**Layer sensitivity is uneven.** Early, middle, and late transformer blocks do not tolerate pruning equally. A single global sparsity target can be crude.

**Perplexity is insufficient for LLMs.** A pruned model can look fine on perplexity and fail at instruction following, tool use, long-context recall, or safety behavior.

**Speedup must be measured end to end.** The only speedup that counts is in the target serving stack with real shapes, batching, dtypes, and hardware.

---

## 19. Failure Modes

Pruning fails in predictable ways.

### Reporting sparsity instead of speed

"We pruned 60% of weights" is not a business result. The result is latency, throughput, memory, quality, or cost.

### Ignoring kernel support

Sparse math requires sparse kernels. Sparse kernels require supported shapes and dtypes. Otherwise dense kernels win.

### Pruning the wrong bottleneck

If inference is memory-bound, reducing FLOPs may do little. If latency is dominated by queueing, pruning the model may help less than scheduling. If TTFT is dominated by prefill, decode-only optimizations will not solve it.

### Using weak evaluation

Small benchmark sets hide regressions. Pruning can remove rare capabilities first because rare behaviors may have less representation in calibration data.

### Overfitting recovery

A pruned model fine-tuned on narrow data may recover benchmark scores while losing generality.

### Breaking model compatibility

Structured pruning changes tensor shapes. That can break checkpoint loading, tensor parallel partitioning, fused kernels, LoRA adapters, speculative decoding heads, or quantization calibration.

---

## 20. Important Papers to Read

Read these in roughly this order.

1. **[Optimal Brain Damage](https://papers.nips.cc/paper/1989/hash/6c9882bbac1c7093bd25041881277658-Abstract.html)** — Yann LeCun, John Denker, Sara Solla, 1989.  
   The classic second-order pruning paper. Read this for the Taylor expansion framing and the idea of saliency.

2. **[Optimal Brain Surgeon and General Network Pruning](https://authors.library.caltech.edu/54981/)** — Babak Hassibi, David Stork, Gregory Wolff, 1993.  
   More principled than Optimal Brain Damage because it accounts for interactions through the inverse Hessian.

3. **[Deep Compression: Compressing Deep Neural Networks with Pruning, Trained Quantization and Huffman Coding](https://arxiv.org/abs/1510.00149)** — Song Han, Huizi Mao, William Dally, 2015 / ICLR 2016.  
   Historically important because it made compression feel practical and deployment-oriented.

4. **[The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks](https://arxiv.org/abs/1803.03635)** — Jonathan Frankle, Michael Carbin, 2018 / ICLR 2019.  
   Important for understanding sparse subnetworks and why overparameterization interacts with trainability.

5. **[Movement Pruning: Adaptive Sparsity by Fine-Tuning](https://arxiv.org/abs/2005.07683)** — Victor Sanh, Thomas Wolf, Alexander Rush, 2020.  
   Useful for transfer learning and pretrained language models, where magnitude alone can be misleading.

6. **[SparseGPT: Massive Language Models Can Be Accurately Pruned in One-Shot](https://arxiv.org/abs/2301.00774)** — Elias Frantar, Dan Alistarh, 2023.  
   Important LLM-era post-training pruning method using approximate second-order structure.

7. **[A Simple and Effective Pruning Approach for Large Language Models](https://arxiv.org/abs/2306.11695)** — Wanda, 2023.  
   A practical activation-aware baseline for LLM pruning.

8. **[LLM-Pruner: On the Structural Pruning of Large Language Models](https://arxiv.org/abs/2305.11627)** — 2023.  
   Useful for structured pruning of LLMs and recovery with lightweight tuning.

---

## 21. The Staff Engineer Summary

The naive version of pruning is "delete small weights." The staff-level version is a systems and modeling tradeoff:

- What structure are we removing?
- What saliency signal justifies removing it?
- What quality does the model lose?
- What recovery method restores that quality?
- What hardware and kernels exploit the new structure?
- What production metric improves?

For LLM optimization, pruning is most credible when it is tied to a deployment path. Unstructured pruning may be a good research or compression result. Structured pruning, semi-structured sparsity, and hardware-aware pruning are more likely to produce real serving wins.

The strongest interview answer is not "pruning makes models faster." It is:

> Pruning can make models faster when it removes computation in a form the runtime can exploit. Otherwise it mostly creates zeros.
