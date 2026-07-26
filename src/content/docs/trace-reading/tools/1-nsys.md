---
title: "Nsight Systems (nsys): the System Timeline"
description: "Reading and diagnosing nsys profiles: row anatomy, the reading order, the fault catalog from starvation to DVFS throttling, capture recipes, and exercises."
---

# Nsight Systems: the System Timeline

`nsys` is the **widest** instrument: whole-machine, whole-step, every stream, every thread, NCCL, OS calls. It answers "*where does the step time go*" and localizes the suspect; it cannot tell you *why a kernel is slow inside* — that's [ncu's job](/trace-reading/tools/2-ncu/), one altitude down.

## 1. Anatomy of the artifact

The rows that matter, top to bottom, in a typical training capture:

```text
CPU threads
  ├─ python main thread     — step loop, cudaLaunchKernel calls, syncs
  ├─ DataLoader workers     — read/decode; look for activity *during* compute
  └─ pt_autograd / NCCL     — backward thread, comm progress threads
CUDA HW (per GPU)
  ├─ Compute stream(s)      — the kernels; density ≈ utilization
  ├─ NCCL stream            — AllReduce/AllGather/ReduceScatter/AllToAll
  └─ Memcpy (H2D/D2H)       — input transfer, checkpoint D2H
OS runtime (osrt)           — read/poll/mmap; where input starvation shows
CUDA API row                — launches, cudaMemcpyAsync, cudaStreamSynchronize
```

The two summaries to run before opening the GUI (they answer 60% of questions):

```bash
nsys stats -r cuda_gpu_kern_sum report.nsys-rep   # top kernels by total time
nsys stats -r cuda_api_sum      report.nsys-rep   # API: launches, syncs, mallocs
nsys stats -r osrt_sum          report.nsys-rep   # OS: read/poll dominance
nsys stats -r cuda_gpu_mem_time_sum report.nsys-rep  # memcpy volume/direction
```

## 2. Reading order

Same discipline as the [vocabulary article](/trace-reading/1-trace-reading-vocabulary/), instantiated for nsys:

1. **Find the step boundary** (NVTX range if you annotated — always annotate — else the optimizer's kernel cluster / loss-scale pattern). Measure 3–5 steps; note variance. High variance is itself a finding (throttling, input jitter, checkpoint stalls).
2. **Compute-stream duty cycle** inside one step: eyeball the fraction of the step with kernels resident. The *gaps* are the work list.
3. **For each gap: what is running instead?** NCCL stream busy → exposed communication. CPU thread busy → launch/host-bound. osrt read/poll → input. Nothing anywhere → sync stall or throttle.
4. **Overlap check:** is NCCL concurrent with compute kernels, or serialized after them?
5. Only then look at kernel durations (`kern_sum`) — and take the top entry to ncu only if steps 2–4 cleared.

## 3. Fault catalog

### Major faults

| Fault | nsys signature | Confirming measurement | Fix direction |
| --- | --- | --- | --- |
| **Input starvation** | Gap at *start* of each step; DataLoader workers or osrt `read` busy during the gap; GPU fully idle | Gap length vs `num_workers` scaling; worker CPU% | More/persistent workers, prefetch, pre-decoded data, move transform off hot path |
| **Unoverlapped gradient all-reduce** | NCCL AllReduce serialized *after* backward's last kernel, compute stream idle under it | Exposed NCCL ms vs bucket size math (bytes/bus BW) | DDP bucketing/`no_sync` misuse fix, overlap hooks, FSDP prefetch |
| **Exposed all-to-all (MoE/EP)** | AllToAll blocks between expert matmuls each layer | Per-layer a2a time vs expert matmul time | Capacity/expert-balance fix, wider EP axis, comm-fusion |
| **Launch-bound / host-bound** | Thousands of <20 µs kernels with inter-kernel gaps; `cudaLaunchKernel` dominates api_sum; python thread pegged | Kernels/step count; gap sum vs step | Fusion (compile), CUDA graphs, bigger batch per kernel |
| **Straggler rank (multi-GPU)** | On *fast* ranks: long NCCL kernel (it's waiting inside the collective); on the straggler: late arrival, longer compute | Per-rank pre-collective timestamp spread | Balance data/work; find the slow GPU (thermal? shared host?) |
| **Blocking H2D copy** | `cudaMemcpy` (sync variant) on API row stalls launches; Memcpy row at step start not overlapped | api_sum: sync vs async memcpy count | Pinned memory + `non_blocking=True`, prefetch to device |
| **Hidden sync points** | Periodic `cudaStreamSynchronize`/`cudaDeviceSynchronize` mid-step; compute stream drains before each | Grep api_sum; correlate with python frames (`.item()`, `.cpu()`, logging, `max_norm` clip on CPU) | Defer metrics, async logging, keep reductions on device |
| **Allocator churn / fragmentation** | `cudaMalloc`/`cudaFree` mid-step in api_sum (caching allocator should make these rare); occasional 100 ms+ malloc stalls | Allocator stats; peak vs reserved | Expandable segments, preallocate, fix shape variance |
| **Recompilation storm (compile/XLA)** | Sawtooth: some steps 10–100× longer, CPU busy in compiler frames, GPU idle | Count distinct shapes fed | Pad/bucket shapes, mark dynamic dims |
| **DVFS / thermal throttle** | *Uniform* kernel slowdown in later steps; same kernels, same sizes, longer duration; no new gaps | `nvidia-smi dmon -s puct` alongside; SM clock trace | Power/thermal budget, spread load, lower clocks expectation |
| **Checkpoint stall** | Every N steps: long D2H memcpy burst + osrt write, compute idle | Periodicity × duration ÷ interval = goodput tax | Async/sharded checkpointing ([the ladder](/google-interview/6-answer-pod-training/)) |

### Minor faults (cheap points, often stacked on a major one)

- **`persistent_workers=False`** — worker respawn gap at every epoch boundary (a one-line fix that reads as a mystery periodic stall).
- **GC pauses** — irregular ~10–100 ms python-thread stalls; visible as launch droughts. `gc.disable()` + manual collection at step boundary.
- **Stream-priority inversion** — comm stream preempting compute on the same SMs; NCCL kernels dilate compute kernels running concurrently (kernel duration grows only when overlapped — the "overlap made it slower" trap).
- **PCIe path surprises** — H2D at ~6 GB/s instead of ~24 GB/s: wrong NUMA node, missing ACS/p2p, or unpinned staging. Check `cuda_gpu_mem_time_sum` throughput.
- **Profiler self-distortion** — capture with `--sample=none --cpuctxsw=none` for timing runs; CPU sampling can manufacture host-boundedness.
- **NVTX absence** — not a fault in the program, but in the capture: without step/phase ranges every other diagnosis costs 5× longer. Annotate before profiling.

## 4. Capture recipe

```bash
# training, 5 steps, delayed start to skip warmup/compile:
nsys profile -o run --capture-range=nvtx --nvtx-capture=step \
  -t cuda,nvtx,osrt,cudnn,cublas --sample=none \
  torchrun --nproc_per_node=8 train.py --profile-steps 100:105
# multi-rank: one report per rank (nsys profiles the launcher's children);
# diagnose stragglers by diffing per-rank reports, not one report alone.
```

Traps: capturing step 0 (warmup + autotune + compile pollutes everything); capturing too long (>30 s reports get unwieldy — 3–5 steps suffice); forgetting `osrt` (you lose the input-starvation signal); profiling with a different batch size than production.

## 5. Exercises

1. From [the MFU-gap worked answer](/answers/mfu-gap-investigation.html): given its `nsys stats` kern_sum and the exposed-NCCL screenshot, compute the exposed-comm fraction of step time and state which two catalog rows above are in play — before reading the answer's diagnosis.
2. Capture Lab C's torchrun script ([hands-on labs](/trace-reading/3-hands-on-profiling/)) twice: once stock, once with `num_workers=0`. Quantify the starvation gap and verify it disappears; write the five-sentence interview narration.
3. Inject a `torch.cuda.synchronize()` + `.item()` metric into the loop; find it in api_sum *without* looking at the timeline, then confirm visually.
