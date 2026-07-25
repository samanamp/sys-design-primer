---
title: "Training: Overview — From FLOPs to a Finished Run"
description: "The continuous story of large-model training: sizing arithmetic, batch size and optimizer dynamics, memory, parallelism, MoE, data pipelines, and reliability — with each chapter's role in the arc."
sidebar:
  label: "Overview: the Arc"
---

# Training: From FLOPs to a Finished Run

Every article in this folder is a chapter of one story: you are given a model target, a token budget, and a fleet — and you must turn chip-seconds into a trained model without wasting them. The chapters, in the order the reasoning actually happens:

1. **Sizing arithmetic** — FLOPs ≈ 6·P·T, chips × MFU × time, 16 bytes/param. *Currently lives inside [the pod-training worked answer §2](/google-interview/6-answer-pod-training/) and the [mental-math drills](/optimization/15-mental-math-drills/); extraction into a standalone chapter is planned.*
2. **[Critical batch size](/training/1-batch-size-primer/)** — the optimizer ceiling: gradient noise scale, why B_crit grows during training, and how ceiling ÷ hardware floor caps useful fleet size.
3. **Memory** — [activation checkpointing](/training/2-activation-checkpointing/) (the compute↔memory dial) and [ZeRO/FSDP](/training/3-zero-fsdp-sharded-training/) (sharding model state). *Gap: a mixed-precision-training chapter (bf16/fp8, master weights, loss scaling) is planned.*
4. **Parallelism** — [the full stack](/training/4-parallelism-stack/) (data/tensor/pipeline/sequence/context/expert) as the spine, with deeper cuts in [context parallelism](/training/5-context-parallelism/) and [4D parallelism](/training/6-4d-parallelism/).
5. **[Mixture of Experts](/training/7-mixture-of-experts/)** — the parallelism specialization that dominates frontier training: routing, load balance, expert parallelism, and the failure modes.
6. **Data pipelines** — [a data platform for text-to-video training](/training/10-tiv-training/) as the worked design. *Gap: a general pretraining-data chapter (offline tokenization, packing, determinism, mixtures) is planned; the material exists in [pod-training §4](/google-interview/6-answer-pod-training/).*
7. **Infrastructure and reliability** — [distributed training infrastructure](/training/8-distributed-training/) ([simplified version](/training/9-distributed-training-simplified/)), plus [rapid-fire Q&A](/training/11-distributed-training-qa/). *Gap: the goodput/checkpointing/SDC deep material lives in [pod-training §5](/google-interview/6-answer-pod-training/) and deserves its own chapter.*

**Capstone:** the [pod-scale MoE training worked answer](/google-interview/6-answer-pod-training/) applies every chapter above to one design question, and stays in the Google-interview track because its framing is interview-specific. The hardware-side counterpart of this whole folder is the [JAX scaling book](https://jax-ml.github.io/scaling-book/) and the [TPU/XLA article](/optimization/18-tpu-xla-optimization/).

The through-line to keep while reading: **arithmetic sets the shape (chapters 1–2), memory and communication set the layout (3–5), and data + reliability decide whether the layout's throughput actually accumulates into a model (6–7).**
