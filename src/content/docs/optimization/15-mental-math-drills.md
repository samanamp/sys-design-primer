---
title: "ML Performance Mental-Math Drills"
description: "Daily 15-minute drills for TPU/GPU performance interviews: arithmetic intensity, decode bytes-per-token, roofline ridge points, and SRAM/register tile budgets — with an interactive randomized trainer."
---

# ML Performance Mental-Math Drills

Interviewers for ML performance roles use fumbled arithmetic as a shallowness signal. These drills build the reflexes: each answer in under two minutes, reasoning spoken aloud, no calculator.

**▶ [Open the interactive drill trainer](/tools/perf-drills.html)** — randomized problems across all four categories, a 2:00 countdown, and step-by-step reveal.

## Specs to memorize (dense BF16, no sparsity)

| Chip | TFLOP/s (BF16) | HBM BW (TB/s) | Ridge (FLOP/B) | HBM (GB) |
|---|---:|---:|---:|---:|
| NVIDIA H100 SXM | 989 | 3.35 | ~295 | 80 |
| NVIDIA B200 | 2,250 | 7.7 | ~292 | 180 |
| AMD MI300X | 1,307 | 5.3 | ~247 | 192 |
| TPU v5p | 459 | 2.76 | ~166 | 95 |
| TPU v6e (Trillium) | 918 | 1.64 | ~560 | 32 |
| NVIDIA A100 (anchor) | 312 | 2.0 | ~156 | 80 |

Ridge point = peak FLOP/s ÷ bytes/s. Arithmetic intensity above the ridge → compute-bound; below → memory-bound. FP8 doubles the FLOP number, so the FP8 ridge is ~2×. Quote dense (non-sparsity) numbers — marketing figures are often 2× with sparsity.

## The four identities

**GEMM arithmetic intensity.** For $M \times N \times K$ in bf16:

$$\text{AI} = \frac{2MNK}{2(MK + KN + MN)} = \frac{MNK}{MK + KN + MN} \;\; \text{FLOP/byte}$$

Square $N{\times}N$: $\text{AI} \approx N/3$. Shortcut: AI is bounded by the smallest of $M, N, K$ — a skinny GEMM (small batch) is memory-bound no matter how big the other dims are.

**Decode bytes/token (batch 1).**

$$\text{bytes/token} \approx P \cdot b_{\text{weights}} + \underbrace{2 \cdot L \cdot h_{kv} \cdot d \cdot s \cdot b_{kv}}_{\text{KV read}}, \qquad \text{tok/s ceiling} = \frac{\text{BW}}{\text{bytes/token}}$$

Batch-1 decode has AI ≈ 1–2 FLOP/byte — hopelessly memory-bound; batching raises effective AI roughly by the batch size. Prefill attention's AI grows with sequence length → compute-bound.

**Roofline attainable throughput.**

$$\text{attainable} = \min(\text{peak FLOP/s},\; \text{AI} \times \text{BW})$$

**Tile / SRAM budget (H100: 228 KB smem/SM, 255 regs/thread, 64K regs/SM).**

$$\text{smem} = \text{stages} \cdot (B_m B_k + B_k B_n) \cdot b_{\text{dtype}}, \qquad \text{accum regs/thread} = \frac{B_m B_n \cdot 4}{\text{threads}}$$

## Four more identities (distributed & capacity rounds)

**Ring all-reduce.** Bandwidth-optimal time for $n$ chips: $t = \frac{2(n-1)}{n} \cdot \frac{\text{bytes}}{\text{link BW}} \approx \frac{2 \cdot \text{bytes}}{\text{BW}}$ for large $n$ — the $n$-dependence nearly cancels. Compare against backprop time to see whether comms hide behind compute.

**Training FLOPs and MFU.** $\text{FLOPs} = 6 P T$ (forward 2, backward 4). MFU = achieved ÷ peak; step time $= \frac{6 P \cdot \text{batch tokens}}{\text{MFU} \cdot \text{peak} \cdot \text{chips}}$. Run it backwards to get MFU from measured tokens/s.

**KV-cache capacity.** Concurrent sequences ≈ (HBM − weights) ÷ KV-per-sequence. This bounds batch size, which bounds decode arithmetic intensity, which bounds throughput — the whole serving-economics chain in one division.

**Training memory.** Adam mixed precision ≈ **16 bytes/param** of model state (2 bf16 weight + 2 grad + 12 fp32 optimizer/master), sharded by ZeRO-3/FSDP degree, plus activations (shrinkable via checkpointing — see [Activation Checkpointing](/optimization/11-activation-checkpointing/)).

## Drill protocol

1. 15 minutes daily, cold, on paper. Rotate through the four categories.
2. Speak the reasoning aloud exactly as you would in the interview — the units narration ("989 teraflops over 3.35 terabytes per second is about 295 flops per byte") is part of what's being scored.
3. Under 2:00 per problem or it counts as a miss. The [interactive trainer](/tools/perf-drills.html) enforces the clock and shows the worked steps.

Useful talking point that falls out of the table: Trillium's ridge (~560 FLOP/B) is roughly double NVIDIA's (~280–300), which is why TPU inference leans harder on batching and on keeping working sets in its large on-chip memory.
