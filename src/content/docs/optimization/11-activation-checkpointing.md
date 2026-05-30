---
title: "Activation Checkpointing: Trading Compute for Training Memory"
description: "A staff-level guide to activation checkpointing and recomputation for LLM training: memory math, checkpoint granularity, selective recompute, sequence length, throughput tradeoffs, and debugging."
---

# Activation Checkpointing: Trading Compute for Training Memory

Activation checkpointing, also called gradient checkpointing or recomputation, is a training optimization that saves memory by not storing every intermediate activation from the forward pass. During backpropagation, the missing activations are recomputed.

The tradeoff is direct:

> Save activation memory, spend extra compute.

This is one of the most important techniques for training large transformers because activation memory grows with batch size, sequence length, hidden size, and layer count. When training long-context models, activation memory can dominate.

---

## 1. The Training Memory Budget

Training memory includes:

```text
parameters
gradients
optimizer states
activations
temporary buffers
communication buffers
fragmentation / framework overhead
```

For Adam-style training, parameter-related memory can be large, but activations are the part that scales strongly with:

$$
B \cdot L \cdot d_{model} \cdot N_L
$$

where:

- $B$ is microbatch size.
- $L$ is sequence length.
- $d_{model}$ is hidden dimension.
- $N_L$ is number of layers.

Long-context training makes $L$ large, so activation checkpointing becomes mandatory.

---

## 2. Basic Idea

Without checkpointing:

```text
Forward:
  layer 1 -> save activations
  layer 2 -> save activations
  layer 3 -> save activations
  layer 4 -> save activations

Backward:
  use saved activations for gradients
```

With checkpointing:

```text
Forward:
  layer 1 -> discard many activations
  layer 2 -> save checkpoint
  layer 3 -> discard many activations
  layer 4 -> save checkpoint

Backward:
  recompute missing forward activations
  compute gradients
```

The model computes some forward operations twice, but peak memory drops.

---

## 3. The Compute-Memory Tradeoff

Let:

- $M_A$ be activation memory without checkpointing.
- $M_C$ be activation memory with checkpointing.
- $F$ be forward compute.
- $B$ be backward compute.

Without checkpointing:

$$
\text{compute} \approx F + B
$$

With checkpointing:

$$
\text{compute} \approx F + B + R
$$

where $R$ is recomputation compute.

The goal is to reduce memory enough to increase batch size, sequence length, or model size, while keeping throughput acceptable.

Checkpointing is worth it when:

$$
\text{value of larger training configuration}
>
\text{cost of recompute}
$$

For LLM training, that is often true.

---

## 4. Granularity

Checkpointing granularity controls how much is saved.

Options:

- Whole transformer block.
- Attention sub-block.
- MLP sub-block.
- Every $k$ layers.
- Selective checkpointing for memory-heavy operations.
- Full recomputation.

```text
Coarse checkpointing:
  save at block boundaries
  less bookkeeping
  more recompute

Fine checkpointing:
  save inside block
  more control
  less recompute
  more complexity
```

Common transformer checkpoint boundary:

```text
RMSNorm -> Attention -> Residual -> RMSNorm -> MLP -> Residual
^                                                   ^
checkpoint boundary                                checkpoint boundary
```

Granularity should be chosen from profiling, not guesswork.

---

## 5. Selective Recompute

Not all activations cost the same. Some are cheap to recompute; others are expensive or numerically sensitive.

Good candidates for recompute:

- LayerNorm/RMSNorm outputs.
- MLP intermediate activations.
- Attention projections.
- Dropout masks if deterministic handling exists.

More delicate:

- Attention softmax intermediates.
- Random operations.
- Custom kernels.
- Operations with non-deterministic reductions.

FlashAttention already avoids storing the full attention matrix and recomputes pieces in backward. This is effectively an IO-aware recomputation strategy.

Staff-level point:

> Activation checkpointing is not only a PyTorch flag. Modern attention kernels and training stacks already make selective recomputation decisions.

---

## 6. Interaction With Parallelism

Checkpointing interacts with:

- Tensor parallelism.
- Pipeline parallelism.
- Sequence/context parallelism.
- ZeRO/FSDP.
- Expert parallelism.

Pipeline parallelism stores activations for in-flight microbatches. More microbatches can mean more activation memory. Checkpointing reduces that pressure but increases recompute inside pipeline stages.

Sequence/context parallelism splits sequence activations across devices. Checkpointing and sequence partitioning often combine for long-context training.

FSDP/ZeRO reduce parameter/optimizer memory, not activation memory. You still need checkpointing for long sequences.

---

## 7. Failure Modes

### Throughput collapses

Checkpointing is too aggressive and recompute dominates.

### Non-determinism breaks gradients

Random operations are not replayed consistently.

### Pipeline schedule gets worse

Recompute increases stage time and pipeline bubbles.

### Memory savings are smaller than expected

Temporary buffers, communication buffers, or fragmentation dominate.

### Debugging becomes harder

Intermediate activations are not saved, making numeric diffing more difficult.

---

## 8. Staff Checklist

Before enabling checkpointing broadly:

- Measure activation memory by layer.
- Separate parameter memory from activation memory.
- Profile recompute overhead.
- Choose checkpoint granularity intentionally.
- Test determinism.
- Check interaction with FlashAttention backward.
- Benchmark tokens/sec, not only max batch size.
- Validate loss curves after enabling.

The interview answer:

> Activation checkpointing trades extra forward recomputation for lower activation memory. It is essential for large and long-context training, but the right granularity depends on profiling and on interactions with attention kernels, pipeline parallelism, and sequence/context parallelism.

