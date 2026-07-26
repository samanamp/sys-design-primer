---
title: "Nsight Compute (ncu): the Kernel Microscope"
description: "Reading and diagnosing ncu reports: Speed-of-Light, roofline position, occupancy, memory workload, the kernel fault catalog, capture recipes, and exercises."
---

# Nsight Compute: the Kernel Microscope

`ncu` profiles **one kernel at a time** by replaying it with hardware counters. It answers "*why is this kernel slow*" — and nothing else. Escalate here only after [nsys](/trace-reading/tools/1-nsys/) has shown the step is kernel-time-dominated *and* named the top kernel; ncu on a launch-bound or starved workload is a microscope pointed at the wrong problem.

## 1. Anatomy of the report

Sections in the order you read them:

```text
GPU Speed Of Light (SOL)     — Compute % and Memory % of peak. THE triage row.
Roofline chart               — achieved FLOP/s vs arithmetic intensity dot.
Occupancy                    — achieved vs theoretical; the limiter
                               (registers / shared mem / block size).
Memory Workload Analysis     — DRAM & L2 throughput, sectors/request
                               (coalescing), bank conflicts.
Compute Workload Analysis    — pipe utilization: tensor vs FMA vs ALU/SFU.
Scheduler / Warp State       — stall reasons: long scoreboard (memory),
                               barrier, not-selected, wait.
Launch Statistics            — grid size, waves per SM (tail effect).
Source Counters (with -lineinfo) — per-line memory/stall attribution.
```

**The SOL two-number triage:**

| Compute SOL | Memory SOL | Verdict |
| --- | --- | --- |
| High (>70%) | low | Compute-bound and healthy — you're done unless wrong-pipe (check tensor vs FMA) |
| Low | high (>70%) | Memory-bandwidth-bound — expected for elementwise/norm/decode-GEMV; a *fault* only if this kernel should be compute-bound |
| Low | low | The interesting case: latency-bound — occupancy, stalls, divergence, tail. Dig. |
| High | high | Streaming kernel at its roofline crossover — usually healthy |

## 2. Reading order

1. **SOL** → which regime (table above).
2. If memory-bound: **Memory Workload** → is achieved DRAM near peak? If yes, the kernel is *efficiently* memory-bound — the fix is algorithmic (fusion, quantization, recompute less), not tuning. If no (low DRAM *and* memory-SOL high in L2/texture): coalescing or working-set problem — check **sectors/request** (≈4 is coalesced fp32; ≈32 is scattered).
3. If latency-bound: **Occupancy** → limiter (registers? shared memory? block size?), then **Warp State** → dominant stall (long scoreboard = memory latency with too little parallelism to hide it; barrier = sync-heavy; not-selected = enough warps, scheduler saturated — fine).
4. If compute-bound: **pipe mix** — tensor-core kernels showing FMA-pipe dominance means you're not on tensor cores at all (dtype/alignment/layout broke the match).
5. **Launch stats** always: waves/SM < 1 means the grid can't fill the GPU (tail effect / small batch) — no amount of kernel tuning fixes an undersized grid.

## 3. Fault catalog

### Major faults

| Fault | ncu signature | Confirming detail | Fix direction |
| --- | --- | --- | --- |
| **Not using tensor cores** | GEMM-shaped kernel, tensor-pipe ~0%, FMA pipe high; SOL compute mediocre | Kernel name lacks `*_tn`/implicit-gemm/cutlass tags; dtype fp32 in trace | bf16/fp16 inputs, dims %8/%16 alignment, channels-last, let cuBLAS pick TC algo |
| **Uncoalesced global access** | Memory SOL high but DRAM throughput far below peak; sectors/request ≫ 4; L2 hit rate low | Source counters point at the offending load | Transpose/layout change, vectorized loads, shared-memory staging |
| **Occupancy starvation (registers)** | Theoretical occupancy capped (e.g. 25%) by registers/thread; long-scoreboard stalls dominate | Achieved ≈ theoretical (the cap is real) | `__launch_bounds__`/maxrregcount, smaller tile, or accept if compute-SOL high anyway |
| **Shared-memory bank conflicts** | Bank-conflict counter high; shared-mem-heavy kernel below expected SOL | L1/shared wavefronts ≫ requests | Pad the tile (+1 column), swizzle |
| **Tail effect / undersized grid** | Waves per SM ≪ 1 or e.g. 1.1 (one full wave + a sliver); duration ≫ theoretical | Grid size vs SM count × blocks/SM | Batch more work per launch, persistent kernels, merge small calls |
| **Warp divergence** | Avg active threads/warp ≪ 32 in Warp State; branch efficiency low | Source counters at the branch | Data layout so warps agree, predication |
| **Register spilling** | Local-memory traffic nonzero in Memory Workload; STL/LDL instructions | `-lineinfo` shows spilled temporaries | Reduce live range, smaller tile, `-maxrregcount` tradeoff |
| **Genuinely memory-bound op treated as a bug** | Elementwise/LayerNorm/softmax at 85% DRAM SOL | Roofline dot sits ON the bandwidth slope | Not a kernel fault: fuse it away or reduce bytes ([kernel-aware article](/optimization/10-kernel-aware-optimization/)) |

### Minor faults

- **L2-resident working set illusion** — a kernel benchmarked alone hits L2 (fast); in the real step, an intervening kernel evicts it. ncu-in-isolation numbers can flatter; trust the nsys duration, use ncu for *ratios*.
- **Replay distortion** — ncu replays kernels; clocks are locked (`--clock-control base`) by default. A kernel that looks 20% slower in ncu than nsys may just be boost-clock delta. Compare regimes, not absolute ms, across tools.
- **Async-copy misuse (Ampere+)** — `cp.async` without enough stages: memory pipe idle bubbles visible as long-scoreboard stalls despite "pipelined" code.
- **Instruction-mix drag** — heavy SFU (exp/rsqrt) in a "GEMM-adjacent" fused epilogue capping compute SOL; usually acceptable, occasionally worth a table lookup.
- **ECC / non-uniform DRAM** — a few % DRAM-peak shortfall that isn't your fault; know the achievable peak (~85–90% of spec) before calling a kernel inefficient.

## 4. Capture recipe

```bash
# top-3 kernels from nsys kern_sum, full sections, 3 launches each:
ncu --set full -k regex:"gemm|flash|ln" -c 3 \
    --clock-control base -o kern_report python train.py --steps 3
# per-line attribution needs -lineinfo at compile time (torch: TORCH_CUDA_ARCH_LIST + custom build, or Triton's default)
```

Traps: `--set full` replays each kernel dozens of times — scope with `-k`/`-c` or a 30 s capture becomes 30 min; profiling under DDP hangs collectives (profile single-rank or use `--target-processes application-only`); forgetting that ncu serializes streams — overlap behavior is invisible here *by design*.

## 5. Exercises

1. From [the MFU-gap answer](/answers/mfu-gap-investigation.html)'s ncu SOL section: classify its headline kernel into one of the four SOL quadrants and state, with the roofline arithmetic, whether tuning or algorithm-change is the fix — then compare with the answer.
2. Write a deliberately strided copy kernel (`out[i] = in[i*stride]`) in Triton/CUDA; capture ncu at stride 1 vs 32, and predict *before looking* the sectors/request and DRAM-SOL deltas.
3. Run a small GEMM in fp32 vs bf16 and diff the pipe-utilization sections; find the exact counter that proves tensor-core engagement.
