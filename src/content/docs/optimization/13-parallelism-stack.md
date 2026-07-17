---
title: "Tensor, Pipeline, Sequence, Context, and Expert Parallelism"
description: "A staff-level guide to large-model parallelism: data, tensor, pipeline, sequence, context, expert parallelism, 3D/4D parallelism, communication patterns, and training/inference tradeoffs."
---

# Tensor, Pipeline, Sequence, Context, and Expert Parallelism

Large models exceed the memory and compute capacity of a single accelerator. Parallelism splits the work across devices.

The hard part is not naming the strategies. The hard part is choosing a parallelism layout that matches:

- Model architecture.
- Sequence length.
- Batch size.
- Interconnect topology.
- Memory pressure.
- Training vs inference.
- Dense vs MoE layers.

The staff-level framing:

> Parallelism is a communication design problem disguised as a model-splitting problem.

---

## 1. The Parallelism Map

Main forms:

| Type | Splits | Main communication |
| --- | --- | --- |
| Data parallelism | batch | gradient all-reduce / reduce-scatter |
| Tensor parallelism | tensors inside layers | all-reduce / all-gather |
| Pipeline parallelism | layers | activation send/recv |
| Sequence parallelism | sequence activations | gather/reduce-scatter |
| Context parallelism | long sequence across devices | attention communication |
| Expert parallelism | MoE experts | all-to-all |

These are usually composed:

```text
Training cluster
    |
    +-- data parallel groups
    +-- tensor parallel groups
    +-- pipeline stages
    +-- context/sequence parallel groups
    +-- expert parallel groups
```

"3D parallelism" often means data + tensor + pipeline. For long-context MoE models, teams may add context and expert parallelism, making the layout effectively 4D or 5D. For a deeper walkthrough of how these dimensions compose, see [the 4D parallelism primer](/llm-primers/10-4d-parallelism/).

---

## 2. Data Parallelism

Each GPU has a model replica and processes different data. Gradients are synchronized.

```text
GPU 0: model replica, batch shard 0
GPU 1: model replica, batch shard 1
GPU 2: model replica, batch shard 2
GPU 3: model replica, batch shard 3

Backward -> synchronize gradients
```

Simple and efficient when the model fits. But it replicates parameters, gradients, and optimizer state unless combined with ZeRO/FSDP.

Communication:

$$
\text{all-reduce gradients}
$$

Use data parallelism when batch scaling is available and memory fits or is sharded.

---

## 3. Tensor Parallelism

Tensor parallelism splits matrix operations across devices.

For a linear layer:

$$
Y = XW
$$

Column parallelism splits $W$ by output columns:

```text
W = [W1 W2 W3 W4]

GPU 0 computes XW1
GPU 1 computes XW2
GPU 2 computes XW3
GPU 3 computes XW4
```

Row parallelism splits by input rows and requires reduction.

Tensor parallelism is useful when individual layers are too large or when inference needs multiple GPUs for one model. It adds communication inside every transformer block.

Tradeoff:

- More TP reduces per-GPU memory.
- More TP increases collectives.
- TP wants fast interconnect such as NVLink/NVSwitch.

Inference note:

> Tensor parallelism can make a model fit, but it may hurt latency if communication dominates.

---

## 4. Pipeline Parallelism

Pipeline parallelism splits layers across devices.

```text
GPU 0: layers 0-7
GPU 1: layers 8-15
GPU 2: layers 16-23
GPU 3: layers 24-31
```

Microbatches flow through stages:

```text
mb0 -> stage0 -> stage1 -> stage2 -> stage3
mb1 -> stage0 -> stage1 -> stage2 -> stage3
mb2 -> stage0 -> stage1 -> stage2 -> stage3
```

The pipeline bubble is idle time while stages wait.

Approximate bubble fraction for $P$ pipeline stages and $M$ microbatches:

$$
\text{bubble} \approx \frac{P-1}{M+P-1}
$$

More microbatches reduce bubbles but increase activation memory.

Schedules:

- GPipe-style fill/drain.
- 1F1B: one forward, one backward.
- Interleaved pipeline.
- Zero-bubble variants.
- DualPipe-style bidirectional overlap, as described in DeepSeek-V3.

Pipeline parallelism is strongest when layers can be balanced across stages and inter-stage communication is manageable.

---

## 5. Sequence Parallelism

Sequence parallelism splits sequence-dimension activations within a tensor-parallel group. It reduces activation memory for operations such as LayerNorm and dropout while keeping tensor parallelism efficient.

```text
Sequence length L split across GPUs:

GPU 0: tokens 0..L/4
GPU 1: tokens L/4..L/2
GPU 2: tokens L/2..3L/4
GPU 3: tokens 3L/4..L
```

It is often used with tensor parallelism in Megatron-style training.

Sequence parallelism is not the same as context parallelism. Sequence parallelism historically split some activations; context parallelism more broadly partitions long-context computation, including attention-related state.

---

## 6. Context Parallelism

Context parallelism partitions long sequences across devices so very long context training can fit.

NVIDIA Megatron-Core describes context parallelism as partitioning network inputs and activations along the sequence dimension, unlike earlier sequence parallelism that only splits certain activations.

Why it matters:

- Long-context training makes activation memory huge.
- Attention needs access across sequence partitions.
- Communication must exchange key/value or attention information.

Context parallelism is useful when sequence length, not just model size, is the bottleneck. For a deeper treatment of ring/all-gather attention mechanics, see [the context parallelism primer](/llm-primers/9-context-parallelism/).

---

## 7. Expert Parallelism

Expert parallelism distributes MoE experts across devices. Tokens are routed to the devices that hold their selected experts, which turns every MoE layer into an all-to-all dispatch, dense expert compute, then an all-to-all combine. The full mechanics — routing, capacity factors, load balancing, and the dispatch/combine flow — are covered in [the MoE article](/optimization/6-mixture-of-experts/).

Expert parallelism is mandatory for large MoE models, but all-to-all can dominate if routing is imbalanced or interconnect is weak.

DeepSeek-V3 is a useful public case study: it used large-scale expert parallelism and pipeline parallelism while avoiding expensive tensor parallelism in parts of the training design through careful architecture and scheduling choices.

---

## 8. Choosing a Layout

Start with the bottleneck:

```text
Model does not fit:
  -> tensor parallelism, pipeline parallelism, ZeRO/FSDP

Optimizer state does not fit:
  -> ZeRO/FSDP

Activation memory too high:
  -> activation checkpointing, sequence/context parallelism

Sequence length too long:
  -> context parallelism, sparse attention, checkpointing

MoE experts too large:
  -> expert parallelism

Pipeline idle time high:
  -> more microbatches, better stage balance, different schedule

Communication too high:
  -> reduce TP/EP, improve placement, overlap communication
```

Parallelism is constrained by topology:

- TP wants fastest links.
- EP all-to-all wants high bisection bandwidth.
- PP can tolerate stage-to-stage links but suffers from imbalance.
- DP can span slower links if gradients are overlapped.

---

## 9. Training vs Inference

Training:

- Needs forward and backward.
- Stores activations.
- Synchronizes gradients.
- Uses optimizer states.
- Can use large global batches.

Inference:

- Forward only.
- KV cache dominates long-context serving.
- Latency matters more.
- Batch sizes are dynamic.
- Tensor parallelism and expert parallelism can hurt p99 if communication is high.

The best training layout is not always the best serving layout.

---

## 10. Failure Modes

### Too much tensor parallelism

The model fits, but collectives dominate.

### Pipeline imbalance

One stage is slower, so every other stage waits.

### Too few microbatches

Pipeline bubble is large.

### Too many microbatches

Activation memory grows and scheduling overhead increases.

### Expert all-to-all dominates

MoE compute savings are erased by communication.

### Context parallelism hurts attention efficiency

Long-sequence communication is not overlapped well.

### Checkpointing and parallelism interact badly

Recompute changes pipeline stage time or communication overlap.

---

## 11. Important Papers and Docs

1. **[Megatron-LM](https://arxiv.org/abs/1909.08053)**.  
   Foundational tensor/pipeline parallel transformer training.

2. **[GPipe](https://arxiv.org/abs/1811.06965)** and **[PipeDream](https://arxiv.org/abs/1806.03377)**.  
   Pipeline parallelism foundations.

3. **[Sequence Parallelism: Long Sequence Training from System Perspective](https://arxiv.org/abs/2105.13120)**.  
   Sequence-dimension training parallelism.

4. **[Megatron-Core context parallelism docs](https://docs.nvidia.com/megatron-core/developer-guide/0.15.0/api-guide/context_parallel.html)**.  
   Practical context-parallel implementation reference.

5. **[DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437)**.  
   Recent public example of large-scale parallelism co-design, including DualPipe and expert parallelism.

---

## 12. The Staff Engineer Summary

Parallelism is how large models fit and scale, but every split introduces communication.

The checklist:

- Identify whether parameters, activations, sequence length, or experts are the bottleneck.
- Use data parallelism when simple replication works.
- Use FSDP/ZeRO for training-state memory.
- Use tensor parallelism when layers are too large.
- Use pipeline parallelism when depth must be split.
- Use sequence/context parallelism for long-context activation pressure.
- Use expert parallelism for MoE.
- Place high-communication groups on fast interconnect.
- Measure tokens/sec/GPU and p95 latency, not just scale factor.

The interview answer:

> Large-model parallelism is about choosing where to pay communication. Tensor parallelism communicates inside layers, pipeline parallelism communicates activations across stages, FSDP/ZeRO communicate parameters and gradients, context parallelism communicates sequence state, and expert parallelism communicates routed tokens. The right layout is determined by memory bottleneck, topology, and workload.

