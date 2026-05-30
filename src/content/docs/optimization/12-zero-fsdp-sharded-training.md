---
title: "ZeRO and FSDP: Sharding Training State at Scale"
description: "A staff-level guide to ZeRO, FSDP, and sharded training: optimizer state, gradients, parameters, memory math, communication tradeoffs, FSDP2, and production debugging."
---

# ZeRO and FSDP: Sharding Training State at Scale

Large-model training is limited by memory before it is limited by arithmetic. A single GPU cannot hold parameters, gradients, optimizer states, activations, temporary buffers, and communication buffers for very large models.

ZeRO and FSDP solve this by sharding training state across data-parallel workers.

The key idea:

> Do not replicate all training state on every GPU if only a shard is needed at a time.

ZeRO comes from DeepSpeed. FSDP is PyTorch's fully sharded data parallel implementation. The conceptual model is the same: shard parameters, gradients, and optimizer states to reduce per-GPU memory.

---

## 1. Training State Memory

For each parameter, Adam-style mixed-precision training may store:

- Model parameter.
- Gradient.
- FP32 master weight.
- First moment.
- Second moment.

Very rough memory per parameter can be:

```text
BF16 param:        2 bytes
BF16 grad:         2 bytes
FP32 master:       4 bytes
Adam m:            4 bytes
Adam v:            4 bytes
--------------------------
Total:            ~16 bytes / parameter
```

For a 70B model:

$$
70B \cdot 16 \approx 1.12TB
$$

before activations and buffers. Replicating that on every GPU is impossible.

---

## 2. ZeRO Stages

ZeRO reduces memory in stages.

```text
ZeRO-1:
  shard optimizer states

ZeRO-2:
  shard optimizer states + gradients

ZeRO-3:
  shard optimizer states + gradients + parameters
```

If data parallel size is $N$, ideal sharding can reduce the sharded state roughly by:

$$
\frac{1}{N}
$$

Real memory is higher because of all-gather buffers, activations, fragmentation, and temporary full parameters.

ZeRO-3 has the largest memory savings but more communication because parameters must be gathered before computation.

---

## 3. FSDP

FSDP shards parameters across workers and all-gathers them when needed for forward/backward. After use, full parameters are freed and returned to sharded form.

Flow:

```text
Before layer:
  each rank owns parameter shard

Forward layer:
  all-gather full parameter
  compute
  free full parameter

Backward layer:
  all-gather full parameter
  compute gradient
  reduce-scatter gradient shard
```

PyTorch FSDP has evolved. FSDP2 (`fully_shard`) uses per-parameter sharding with DTensor-oriented APIs and is the direction PyTorch documentation now emphasizes for newer workflows.

The important concept remains:

> FSDP saves memory by materializing full parameters only around the computation that needs them.

---

## 4. Communication Tradeoff

Sharding saves memory but adds communication.

Main collectives:

- All-gather parameters.
- Reduce-scatter gradients.
- Optional all-reduce depending on strategy.

The performance question:

$$
\text{step time} =
\text{compute} + \text{communication} + \text{overlap gaps}
$$

Good FSDP/ZeRO setups overlap communication with compute:

```text
Compute layer L
    |
    +-- prefetch all-gather for layer L+1

Backward layer L
    |
    +-- reduce-scatter gradients while other work continues
```

Bad setups serialize communication and compute, causing GPUs to wait.

---

## 5. Sharding vs Activation Memory

ZeRO/FSDP solve parameter-state memory. They do not automatically solve activation memory.

```text
ZeRO/FSDP:
  parameters, gradients, optimizer states

Activation checkpointing:
  activations

Sequence/context parallelism:
  sequence-dimension activations
```

For long-context LLM training, you usually need all of them.

---

## 6. When to Use Which

Use DDP when:

- Model fits comfortably.
- Simplicity and throughput matter.
- Memory is not the bottleneck.

Use ZeRO-1/2 when:

- Optimizer/gradient memory is the issue.
- You want less communication than full parameter sharding.

Use ZeRO-3/FSDP when:

- Parameters do not fit replicated.
- You need maximum memory savings.
- Network bandwidth is strong enough.
- You can tune wrapping/prefetch/bucket sizes.

Use tensor/pipeline parallelism too when:

- A single layer is too large.
- Communication patterns need model-parallel partitioning.
- Training at frontier scale.

---

## 7. Production Debugging

Common metrics:

- Peak allocated memory.
- Peak reserved memory.
- All-gather time.
- Reduce-scatter time.
- Overlap percentage.
- Step time.
- Tokens/sec/GPU.
- GPU idle time.
- OOM location.
- Checkpoint save/load time.

Failure patterns:

- Too-small buckets increase overhead.
- Too-large buckets spike memory.
- Bad auto-wrap causes excessive all-gathers.
- CPU offload saves GPU memory but kills throughput.
- Activation memory still OOMs.
- Checkpointing becomes slow or fragile.
- Network topology limits scaling.

---

## 8. Recent Reality

As of 2025-2026:

- PyTorch documentation emphasizes FSDP2/`fully_shard` for newer sharded workflows.
- DeepSpeed ZeRO remains common in large-scale training stacks.
- Frontier-scale systems compose sharding with tensor, pipeline, expert, and context parallelism.
- DeepSeek-V3 is a public example of extreme training co-design: FP8 training, MoE, pipeline scheduling, and careful parallelism choices.

Sharding is no longer exotic. The hard part is composing it with the rest of the training system.

---

## 9. Important Papers and Docs

1. **[ZeRO: Memory Optimizations Toward Training Trillion Parameter Models](https://arxiv.org/abs/1910.02054)**.  
   Original ZeRO paper.

2. **[DeepSpeed ZeRO docs](https://deepspeed.readthedocs.io/en/latest/zero3.html)**.  
   Practical ZeRO stages and offload.

3. **[PyTorch FSDP paper](https://arxiv.org/abs/2304.11277)**.  
   Industry-grade FSDP experience.

4. **[PyTorch FSDP2 fully_shard docs](https://docs.pytorch.org/docs/stable/distributed.fsdp.fully_shard.html)**.  
   Current direction for PyTorch sharded training APIs.

5. **[DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437)**.  
   Useful recent case study in large-scale training co-design.

---

## 10. The Staff Engineer Summary

ZeRO and FSDP make large-model training possible by sharding training state. They trade memory for communication.

The checklist:

- Estimate parameter, gradient, and optimizer memory.
- Choose ZeRO/FSDP stage based on the actual memory bottleneck.
- Remember activation memory is separate.
- Tune wrapping, buckets, prefetch, and overlap.
- Profile communication, not just GPU memory.
- Validate checkpointing and restart behavior.
- Compose with tensor/pipeline/context/expert parallelism when needed.

The interview answer:

> ZeRO and FSDP reduce replicated training state. ZeRO-3/FSDP shard parameters, gradients, and optimizer states, but they add all-gather and reduce-scatter communication. The right setup depends on whether memory or communication is the bottleneck, and it usually needs activation checkpointing and model parallelism at large scale.

