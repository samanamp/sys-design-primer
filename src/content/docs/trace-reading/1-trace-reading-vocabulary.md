---
title: "Trace Reading: The Vocabulary"
description: "A staff-level guide to reading GPU training and inference timelines: Nsight Systems and torch.profiler/Kineto trace anatomy row by row, a fixed reading discipline, the nine canonical timeline signatures with ASCII sketches and fixes, and the traps that make traces lie."
---

# Trace Reading: The Vocabulary

A performance interview at this level is often literally this: a timeline appears on screen, and you narrate what you see. Not "let me run some experiments" — *read the picture, out loud, in a fixed order, landing on a diagnosis*. That is a language skill. This article is the vocabulary and grammar. The kernel-level deep dive (warp stalls, occupancy, Nsight Compute metrics) lives in [ML Performance Debugging](/optimization/17-perf-debugging/); this article is about everything you decide *before* you ever open a single kernel.

The core claim:

> Almost every training performance bug is visible in the timeline before you look at any counter. The timeline tells you *which* of the five bottlenecks you have — compute, memory bandwidth, launch overhead, communication, or input pipeline. Counters only tell you *why*.

---

## 1. What a Trace Is

A trace is a set of **correlated timelines**: one row per actor (a CPU thread, a CUDA stream, a comm channel), events as horizontal bars, all on one shared clock. The power is entirely in the correlation — a single row tells you almost nothing; the *gaps in one row explained by activity in another row* is the whole game.

Three instruments, three altitudes:

| Tool | Altitude | Right question |
|---|---|---|
| **Nsight Systems (nsys)** | Whole system: CPU threads, CUDA API, GPU streams, NCCL, NVLink/PCIe, OS runtime | "Where does the step time go? Is the GPU even busy?" |
| **torch.profiler (Kineto)** → Perfetto / chrome://tracing | Framework: PyTorch operators correlated to kernels, with Python stacks and tensor shapes | "*Which line of my model code* produced that kernel / that gap?" |
| **Nsight Compute (ncu)** | One kernel: warp stalls, memory throughput, occupancy | "Why is this specific kernel slow?" |

Rules of engagement:

- Start with a timeline (nsys or torch.profiler). Never start with ncu — profiling a kernel that only accounts for 3% of the step is the classic junior mistake.
- torch.profiler when you need the *semantic* mapping ("this kernel is layer 17's QKV projection"); nsys when you need the *system* view (dataloader threads, NCCL, multi-process, CPU-side jitter).
- ncu only after the timeline has convicted one specific kernel. See [profiler fluency in 17](/optimization/17-perf-debugging/#22-profiler-fluency) for what to do once you're there.

Both nsys and Kineto traces are event logs; Kineto emits Chrome trace JSON that Perfetto renders. Same reading skills transfer.

---

## 2. Anatomy of a Training Trace, Row by Row

A single-GPU training step in Nsight Systems (torch.profiler in Perfetto is structurally the same, minus the OS rows, plus Python operator rows):

```text
time ──────────────────────────────────────────────────────────────────▶

CPU threads
  python (main)     │▒▒▒│launch│launch│launch│ ... │▒ optimizer ▒│
  dataloader wkr 0  │██ decode/augment ██│        │██ next batch ██│
  dataloader wkr 1  │  ██ decode/augment ██│    │██ ...
CUDA API            │cudaLaunchKernel│cudaLaunchKernel│cudaMemcpyAsync│...
NVTX                │========= step 42 =========│== fwd ==│== bwd ==│
CUDA HW (GPU)
  Stream 7 compute  │ █gemm█ █norm█ █attn████ █gemm█ ... │
  Stream 20 NCCL    │              ▓▓ all_reduce ▓▓      │
  Stream 14 memcpy  │ ─H2D─                              │
GPU metrics (nsys --gpu-metrics-devices)
  SM Active %       │ ▁▆▇█████▆▇█████▇▁                  │
  Tensor Active %   │ ▁▂▆██▂▁▂▆██▂▁                      │
  DRAM Bandwidth    │ ▂▃▇▇▃▂▇▇▃▂                         │
```

What each row means — and the reading you should do on each:

**CPU threads.** The Python main thread is where kernels get *launched*. Dataloader workers are separate processes/threads doing decode and augmentation. If the main thread shows long bars that aren't launches (Python overhead, logging, `.item()` waits), that time is stolen from feeding the GPU.

**CUDA API row.** Host-side duration of each CUDA runtime call: `cudaLaunchKernel`, `cudaMemcpyAsync`, `cudaStreamSynchronize`, `cudaMalloc`. Critical semantic: **this row is CPU time, not GPU time.** A `cudaLaunchKernel` bar is the ~5–10 µs the CPU spent enqueueing; the kernel runs later, on the GPU rows. A *long* bar here means the host thread is blocked — a launch blocking means the launch queue is full (you're launch-bound and the GPU got ahead... actually behind — see signature (b)); a long `cudaStreamSynchronize` means the CPU is waiting for the GPU.

**CUDA HW / per-stream GPU rows.** When work *actually executes* on the device, one row per stream. Typical layout: one compute stream (dense with kernels), one or more NCCL streams (collectives), a memcpy row (H2D/D2H). Kernels on different streams can overlap in wall time — that overlap is the thing you check for comm hiding.

**Correlation lines / flow arrows.** Click a kernel and the UI draws an arrow back to the `cudaLaunchKernel` (and, in Kineto traces, up to the `aten::` operator and Python frame) that produced it. This is how you answer "which module launched this?" In Perfetto: select the kernel, follow the *incoming flow*; select a launch, follow the *outgoing flow*. The horizontal distance between launch and execution is queueing delay — large distance is *healthy* (the CPU is running ahead); launch and kernel vertically aligned means the GPU is consuming work as fast as the CPU can produce it — a launch-bound smell.

**NVTX ranges.** Programmer-inserted named spans (`torch.cuda.nvtx.range_push("fwd")` or `record_function`). Without them a trace is 40,000 anonymous `sm90_xmma_...` kernels; with them it has chapters. PyTorch's profiler injects `ProfilerStep#N` ranges automatically when you call `prof.step()` — that's your step boundary. Instrument before you profile: step, forward, backward, optimizer, dataloader-wait, at minimum.

**GPU metrics rows.** With `nsys profile --gpu-metrics-devices=...`, nsys samples hardware counters (~10 kHz) into timeline rows: SM Active, Tensor Active, DRAM bandwidth, NVLink/PCIe throughput. This is where "the kernel row looks busy but SM Active is 20%" becomes visible — kernels *resident* but not *working*. It's the timeline-level preview of what ncu tells you per-kernel.

---

## 3. The Reading Order

Fixed discipline. Same order every time, narrated out loud in an interview. Skipping steps is how you end up optimizing a kernel in a step that's dataloader-bound.

```text
1. Find the step boundary        (NVTX ProfilerStep / repeating pattern)
2. Step time vs sum-of-kernels   (gap analysis: where is the non-GPU time?)
3. Compute/comm overlap          (do NCCL bars run under compute bars?)
4. Top kernels by total time     (only now: which kernels dominate?)
5. Zoom into ONE kernel          (hand off to Nsight Compute / article 17)
```

**Step 1 — find the step boundary.** Locate one full iteration: NVTX range, or just the repeating visual motif (steps look identical; anything aperiodic is suspicious by itself). Ignore the first several steps — warmup, compilation, autotuning (see traps).

**Step 2 — gap analysis.** The single most valuable measurement in trace reading:

```text
step_time  = 100 ms
Σ kernel_time (compute stream) = 62 ms
exposed comm (NCCL not under compute) = 13 ms
GPU idle (gaps) = 25 ms                      ← explain every millisecond
```

If kernels sum to ≈ step time, you're GPU-bound → go to step 4. If there are big gaps, the GPU is not the problem yet — find what the CPU/dataloader/comm was doing during each gap by reading the rows above. Every gap has an owner.

**Step 3 — overlap check.** For distributed jobs: does the NCCL row run *concurrently* with the compute row, or do compute bars stop while NCCL runs? Exposed communication time = NCCL time not covered by compute. Target >70% of comm hidden (HTA reports this directly for multi-rank Kineto traces).

**Step 4 — top kernels.** Sort by total time (`nsys stats`, or the profiler's `key_averages()` table). Usually a power law: top 5 kernels ≈ 70–80% of GPU time. Check for surprises — elementwise kernels near the top means missing fusion; `cast`/`copy` kernels near the top means dtype or layout churn; unrecognized fallback kernels means a fast path was missed.

**Step 5 — zoom into one kernel.** Only now do you reach for ncu, roofline reasoning, and warp-stall taxonomy — that's [article 17, Part 2](/optimization/17-perf-debugging/#part-2--kernel-level-performance-and-debugging).

---

## 4. The Nine Canonical Signatures

Nearly every trace you'll see in an interview is one of these. For each: the visual sketch, how you *confirm* it (never diagnose from the picture alone), and the fix.

### (a) Unoverlapped gradient all-reduce

```text
Compute │ ████ fwd ████ ████ bwd ████                    ████ opt ██
NCCL    │                             ▓▓▓▓ all_reduce ▓▓▓▓
                                      ▲ GPU compute idle here
```

**How you know:** NCCL bars sit *after* backward instead of *under* it; compute stream empty during the collective. Confirm: exposed comm time ≈ (step time − Σ compute kernels); comm time roughly matches `2·(N−1)/N · grad_bytes / bus_BW`.
**Fix:** DDP gradient bucketing so all-reduce fires per-bucket during backward (check bucket size — one giant bucket cannot overlap); FSDP prefetch/backward-prefetch; check for a graph break or a hook that forces backward to complete first.

### (b) Host/launch-bound: many tiny kernels

```text
CPU      │launch│launch│launch│launch│launch│launch│   ← saturated, no gaps
GPU      │▌│  │▌│  │▌│  │▌│  │▌│  │▌│                  ← slivers with gaps
             ▲ gap: GPU finished before CPU could launch the next one
```

**How you know:** thousands of kernels under ~20 µs each; the GPU row is confetti with inter-kernel gaps; the CUDA API row is wall-to-wall launches; launches and their kernels are vertically aligned (no queue depth — the CPU can't run ahead). Confirm: mean kernel duration ≲ launch overhead (~5–10 µs); GPU idle time ≈ number-of-kernels × per-launch gap.
**Fix:** CUDA graphs (capture and replay the step), `torch.compile` (fusion cuts the kernel count), bigger batch (amortize), remove per-op Python overhead. Common in decode loops and small-model training.

### (c) Input-pipeline starvation

```text
Dataloader │████ decode/augment ████│
GPU        │      IDLE             │███ fwd+bwd ███│      IDLE      │███
                  ▲ gap at the START of each step, dataloader busy during it
```

**How you know:** the gap is at the step boundary, and the dataloader worker rows are busy exactly during the GPU's idle window. Confirm: time the `next(dataloader)` call on the main thread (NVTX it); GPU-busy % rises when you swap in synthetic/cached data — the definitive experiment.
**Fix:** more workers, `pin_memory=True` + `non_blocking=True` copies, `prefetch_factor`, persistent workers, move decode/augment to GPU (DALI/torchvision-GPU), pre-tokenize/pre-shard offline.

### (d) Memory-bound dominant kernel

```text
GPU          │ ██████████ one long kernel ██████████ │
DRAM BW      │ ▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇ ~90% of peak ▇▇▇▇▇▇▇▇ │
Tensor Active│ ▂▂▂▂▂▂▂▂▂▂ low ▂▂▂▂▂▂▂▂▂▂             │
```

**How you know:** the timeline itself only tells you the kernel is long and the GPU is "busy". The nsys GPU-metrics rows give the tell: DRAM bandwidth pinned near peak while tensor pipe is low. Confirm in ncu: `dram__throughput` > 80% of peak, Long Scoreboard stalls dominant; or back-of-envelope — bytes moved / kernel time ≈ HBM peak means it's going as fast as memory allows.
**Fix:** this is a *kernel/algorithm* problem, not a scheduling problem — reduce bytes (fusion, FlashAttention-style IO-aware rewrites, lower-precision KV/activations, better layouts). Hand off to [article 17](/optimization/17-perf-debugging/) and [kernel-aware optimization](/optimization/10-kernel-aware-optimization/).

### (e) Straggler rank in a collective

```text
Rank 0 │ ████ compute ████ ▓▓▓▓▓▓ NCCL (waiting) ▓▓▓▓▓▓ ████
Rank 1 │ ████ compute ████ ▓▓▓▓▓▓ NCCL (waiting) ▓▓▓▓▓▓ ████
Rank 2 │ ████████ compute (slow) ████████ ▓ NCCL ▓ ████        ← straggler
Rank 3 │ ████ compute ████ ▓▓▓▓▓▓ NCCL (waiting) ▓▓▓▓▓▓ ████
```

**How you know:** a "slow NCCL op" on N−1 ranks that is really *wait time* — collectives are barriers, so one slow rank shows up as long NCCL bars everywhere else. The key move: **a long all-reduce bar does not mean bytes are moving slowly.** Confirm: collect traces from multiple ranks (HTA aggregates this); the rank whose *compute* is longest and *NCCL* is shortest is the culprit. Sanity-check against algorithm-bandwidth math — if implied bus bandwidth is absurdly low, it's a straggler, not the fabric.
**Fix:** find why that rank is slow — thermal/power throttling, a bad GPU, uneven sequence lengths (data skew), CPU contention on that node, ECC retirement. Fix the rank, not the collective.

### (f) Pipeline bubbles

```text
Stage 0 │ █F█F█F█F           ░░ bubble ░░        B█B█B█B█ │
Stage 1 │   █F█F█F█F       ░ bubble ░          B█B█B█B█   │
Stage 2 │     █F█F█F█F   ░ bubble ░          B█B█B█B█     │
Stage 3 │       █F█F█F█F█B█B█B█B█B█B ...                  │
```

**How you know:** periodic idle wedges on all but the last stage, shaped like the classic 1F1B diagram; idle fraction per stage ≈ `(p−1)/(m+p−1)` for p stages, m microbatches. Confirm: measure bubble fraction from the trace and check it against that formula — if it's *bigger* than predicted, you also have imbalanced stages (one stage's F/B bars visibly longer).
**Fix:** more microbatches, better schedule (interleaved 1F1B, zero-bubble variants), rebalance layers across stages. Details in [17 §3.4](/optimization/17-perf-debugging/#34-pipeline-parallelism-schedules).

### (g) Synchronization stalls

```text
CPU  │launch launch│── cudaStreamSynchronize ──│launch launch│── sync ──│
GPU  │ ████████████│           IDLE             │████████████│  IDLE    │
        ▲ CPU stops launching every time it syncs; pipeline drains
```

**How you know:** repeated `cudaStreamSynchronize` / `cudaDeviceSynchronize` bars on the CUDA API row, each followed by a GPU gap (the launch queue drained; the GPU then waits for the CPU to refill it). Confirm with correlation: what Python code sits above each sync? Usual culprits — `.item()`, `tensor.cpu()`, `print(loss)`, `float(loss)` for logging every step, gradient-norm checks, data-dependent control flow, non-`non_blocking` D2H copies.
**Fix:** log every N steps, keep metrics on-GPU and read asynchronously, `non_blocking=True`, remove data-dependent branches from the hot loop. One `.item()` per step can serialize the whole pipeline.

### (h) Memory-allocation stalls

```text
CPU  │launch│─── cudaMalloc ───│launch│─── cudaFree/cudaMalloc ───│
GPU  │ ████ │      IDLE        │ ███  │          IDLE             │
```

**How you know:** `cudaMalloc`/`cudaFree` bars on the CUDA API row *in steady state* — after warmup, PyTorch's caching allocator should serve everything from its pool, so any steady-state cudaMalloc means the cache missed. Worse: `cuda_free` retries after fragmentation, or `empty_cache()` calls in the loop. cudaMalloc is device-synchronizing, so each one also drains the pipeline (compounding with (g)). Confirm: `torch.cuda.memory_stats()` — rising `num_alloc_retries`, gap between reserved and allocated (fragmentation); memory-snapshot tooling for the sawtooth.
**Fix:** stabilize shapes (bucketing/padding — varying sequence lengths are the classic fragmenter), expandable segments (`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`), remove `empty_cache()` from the loop, pre-allocate buffers.

### (i) The healthy trace — what good looks like

```text
CPU     │launch launch launch (running ~ms AHEAD of the GPU) ...        │
GPU     │██████████████████████████████████████████████████████████████│
NCCL    │        ▓▓▓▓▓▓ under compute ▓▓▓▓▓▓        ▓▓▓▓ under ▓▓▓▓    │
SM Act  │ ▇▇█████████▇▇█████████▇▇                                     │
```

You must be able to say what *good* looks like, or you can't say what's wrong: compute stream >90–95% busy across the step; NCCL bars almost entirely under compute bars; launches running well ahead of execution (deep queue); no steady-state cudaMalloc; no per-step syncs; steps visually identical, period equal on every rank. At that point the timeline has nothing left to give — remaining wins are inside kernels (article 17) or algorithmic. The staff move is also knowing when to *stop*: a trace like this plus MFU near the hardware-realistic ceiling means you're done.

---

## 5. Traps: When the Trace Lies

**Profiling overhead distorts the trace.** `with_stack=True` and `record_shapes=True` add per-op CPU cost — they can *create* a launch-bound signature that isn't there in production. nsys is lighter but not free; GPU-metrics sampling adds its own load. Discipline: check step time with profiler on vs off; profile a few steps, not hundreds; diagnose CPU-boundness with the lightest config first.

**First steps are garbage.** Step 1 contains kernel JIT/autotuning, cuDNN heuristics, allocator pool growth (a wall of cudaMalloc that never recurs), `torch.compile` compilation (seconds to minutes), and NCCL communicator setup. Use the profiler schedule (`wait/warmup/active`) and read steady-state steps only. If someone shows you a trace with a huge cudaMalloc storm, first question: *which step is this?*

**CUDA is asynchronous: CPU time ≠ GPU time.** The most common misreading. A fat bar on the Python row named `aten::mm` is the *dispatch* cost, not the matmul; the matmul is on the stream row, possibly milliseconds later. Naive `time.time()` around GPU code measures launch time unless you synchronize — which is why "this op takes 2 µs" (async, unmeasured) and "this op takes 40 ms" (it absorbed a sync and got billed for every prior kernel) are both classic lies. In a trace, the same effect shows up as one innocent op appearing enormous because it was the first to *wait*. Always attribute wait time to the producer, not the op that happened to block.

**Averaged utilization lies.** `nvidia-smi` "GPU-Util 100%" means *at least one kernel was resident* during the sample window — a single warp spinning counts. It says nothing about SM occupancy, tensor-core activity, or bandwidth. Similarly, "average SM Active 80%" can be 100%-then-60% alternation hiding a periodic stall. Averages hide the gaps, and the gaps are the diagnosis — that's *why* traces beat metrics. When someone quotes a utilization number, ask for the timeline; when the timeline looks busy, check the GPU-metrics rows underneath it.

---

## 6. Interview Q&A

**Q: You get one profiling run on a slow training job. What do you collect?**
A: nsys with NVTX ranges, CUDA and NCCL tracing, and GPU-metrics sampling, over ~10 steady-state steps after warmup — plus traces from at least two ranks if distributed. That one artifact answers the first-order question (GPU-bound vs launch-bound vs input-bound vs comm-bound) and tells me whether I ever need Nsight Compute at all.

**Q: The GPU kernel row looks fully packed but the job is slow. Walk me through it.**
A: Packed ≠ productive. First check step time against expectation (MFU math). Then the GPU-metrics rows: if DRAM bandwidth is pinned and tensor pipe is low, I'm memory-bound inside kernels (signature d). Then top-kernels: elementwise/copy/cast kernels near the top means fusion or dtype churn problems. The timeline convicts a kernel; ncu explains it.

**Q: One all-reduce in the trace takes 10× longer than the math says it should. Fabric problem?**
A: Almost certainly not — a long collective bar usually means *waiting*, not moving bytes. Collectives are barriers, so a straggler rank appears as slow NCCL on every other rank. I'd pull traces from multiple ranks, find the one whose compute runs long and NCCL runs short, and debug that rank: throttling, data skew, bad node.

**Q: Where do the launch bar and the kernel bar for the same op appear, and what does the distance between them tell you?**
A: Launch on the CPU's CUDA API row, execution on the stream row, joined by a correlation arrow. Large horizontal distance = deep launch queue = healthy, CPU running ahead. Launch and kernel adjacent, with the API row saturated and tiny kernels = launch-bound; CUDA graphs or torch.compile.

**Q: When is torch.profiler the wrong tool?**
A: When the problem is outside the framework's view — dataloader worker processes, CPU contention from other jobs, NIC behavior, multi-process node-level effects — or when its own overhead perturbs a launch-bound workload. That's nsys territory. And it's the wrong tool the moment the question becomes "why is this one kernel slow": that's ncu.

---

## Where to Go Next

- **Practice the signatures interactively:** the [trace-reading trainer](/tools/trace-drills.html) drills you on identifying these nine patterns against the clock.
- **Guided walkthroughs of real traces:** [Guided Walkthroughs](/trace-reading/2-guided-walkthroughs/).
- **Collect your own:** [Hands-On Profiling](/trace-reading/3-hands-on-profiling/) — generating nsys and Kineto traces on a real training loop.
- **Below the timeline:** [ML Performance Debugging](/optimization/17-perf-debugging/) for warp stalls, rooflines, and Nsight Compute; [Kernel-Aware Optimization](/optimization/10-kernel-aware-optimization/) for what to do about the kernels you convict.

The staff summary:

> A timeline is a language. Fixed reading order: step boundary → gap analysis → overlap → top kernels → one kernel. Nine signatures cover almost everything, and each has a confirming measurement — never diagnose from the picture alone, and never trust an averaged utilization number when you could look at the gaps.
