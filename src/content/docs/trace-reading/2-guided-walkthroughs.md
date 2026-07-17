---
title: "Trace Reading: Guided Walkthroughs"
description: "Six interview-style trace walkthroughs — DDP comm exposure, input starvation, launch-bound inference, straggler ranks, prefill-blocking-decode, and the healthy-looking trace — each with a timeline, a narrated read, the confirming measurement, and the fix with arithmetic."
---
# Trace Reading: Guided Walkthroughs

This is the tier-2 doc: case studies, run the way the round actually runs. The interviewer shares a screen — an Nsight Systems or Perfetto view, sometimes just a `torch.profiler` export — and says: *"Here's the trace. Walk me through it."* Your job is not to spot the answer instantly. It is to demonstrate a repeatable read.

The discipline order, every time:

1. **Step boundary** — find one full iteration; get the period. Everything is a fraction of this number.
2. **Gaps** — where is the GPU compute stream idle, and what row is busy while it's idle?
3. **Overlap** — is communication running concurrently with compute, or serialized after it?
4. **Longest kernel** — only after the gaps are explained do you look inside the busy time.

Each case below: scenario, timeline, the narrated read (including at least one hypothesis you consider and reject out loud), the confirming measurement, the fix with arithmetic, and the 60-second spoken version.

---

## Case 1 — Unoverlapped all-reduce in DDP

**Scenario:** 8×H100 (NVLink), 8B-parameter dense model, DDP fine-tune, BF16. MFU stuck at 31%; expected ~45% for this stack.

```
      0ms       100       200       300       400       500       600  620ms
      |---------|---------|---------|---------|---------|---------|----|
CPU   [step_begin][ launch fwd  ][ launch bwd            ][opt][step_begin
CUDA  ▐▌▐▌▐▌▐▌▐▌▐▌▐▌▐▌▐▌▐▌▐▌▐▌▐▌▐▌▐▌▐▌▐▌▐▌▐▌▐▌▐▌       ▐▌▐▌  (API row: dense)
GPU   [====== forward 140 ======][========= backward 280 =========][opt 10]
COMP  ampere_bf16_s16816gemm / flash_fwd_kernel / vectorized_elementwise ...
NCCL  ..............(idle)..............................[== AllReduce 175 ==]
                                                        ncclDevKernel_AllReduce_Sum_bf16_RING_LL
      |<-------------- compute 430ms ------------------>|<-- comm 175 -->|<15>|
      |<----------------------- step = 620ms ---------------------------->|
```

### The read

**Step boundary.** `ProfilerStep#` annotations (or the repeating `step_begin` NVTX range) give a clean 620ms period. Sanity check against the log: 620ms/step matches reported throughput, so the trace is representative.

**Gaps.** The GPU *compute* stream is idle from 430ms to 605ms — 175ms, 28% of the step. But the GPU as a device is not idle: the NCCL stream is running one long `ncclDevKernel_AllReduce_Sum_bf16` for exactly that window. So this is not a starvation gap; it's serialized communication.

**Overlap.** This is the finding. Backward runs 150→430ms; the all-reduce starts *at* 430ms, not during backward. DDP's whole design is bucketed gradient all-reduce overlapped with backward — the first bucket should fire when the last layers' gradients are ready, seconds into backward, not after it. One monolithic 175ms NCCL kernel after backward ends means bucketing is not happening.

**Wrong hypothesis, considered and rejected:** "NCCL is slow — maybe NVLink is misconfigured, we should check `nccl-tests`." Do the math first. 8B params × 2 bytes = 16 GB of gradients. Ring all-reduce moves 2(N−1)/N × 16 GB = 28 GB per GPU; 28 GB / 175ms ≈ 160 GB/s busbw. That's below the ~350–400 GB/s NVLink ring achieves at large sizes but not pathological — and even a perfect 80ms all-reduce, still serialized, only takes MFU from 31% to ~38%. The bandwidth is not the bug; the *serialization* is. Don't fix the wrong 95ms while ignoring the free 175ms.

**Why is overlap broken?** Usual suspects: (a) `no_sync()` accidentally left on / gradient accumulation wrapper syncing only on the last micro-batch — but this trace shows one all-reduce per step, so it *is* syncing; (b) a `torch.cuda.synchronize()` or `.item()` between backward and the comm; (c) most common — the model was wrapped in a way that defeats DDP's autograd hooks (e.g., `find_unused_parameters` recompute path, or gradients produced by a custom autograd function that bypasses the reducer, so DDP falls back to an end-of-backward flat all-reduce).

### Confirming measurement

`nsys stats --report cuda_gpu_kern_sum`: one `ncclDevKernel_AllReduce` instance per step at ~175ms — not the ~25–30 bucket-sized instances you'd expect from `bucket_cap_mb=25`. One-line experiment: log `model.reducer._rebuild_buckets` / set `TORCH_DISTRIBUTED_DEBUG=DETAIL` and confirm bucket count; or simply print the number of NCCL kernels per step from the profiler.

### Fix and expected win

Restore bucketed overlap (fix the hook-defeating wrapper; verify with `bucket_cap_mb=25` → ~26 buckets). Backward is 280ms; total comm ~175ms < 280ms, so in principle all of it hides. Overlapped comm contends for SMs and HBM, so assume ~85% hiding: exposed comm ≈ 25ms.

- Before: 620ms/step, MFU 31%.
- After: 430 + 25 + 10 ≈ **465ms/step** → 1.33× throughput → MFU ≈ 31% × 620/465 ≈ **41%**.

### In the interview (60 seconds)

"Step is 620ms. The compute stream has a 175ms hole at the end of every step, and the NCCL stream is busy for exactly that hole — one monolithic all-reduce, fully serialized after backward. That's 28% of the step. 16 GB of BF16 gradients at ~160 GB/s busbw checks out for the duration, so NCCL isn't broken — overlap is. DDP should be firing ~26 bucket-sized all-reduces *during* backward; one big kernel means the reducer hooks aren't triggering, probably a wrapper bypassing autograd. Backward is 280ms, comm is 175ms, so it hides almost entirely: I expect ~465ms steps, MFU 31→41%. I'd confirm by counting NCCL kernel instances per step in `nsys stats` before touching anything."

---

## Case 2 — Input starvation

**Scenario:** 4×A100-80GB, ViT-L image classification pre-training, batch 256/GPU. GPU utilization graph in `nvidia-smi` shows 100%, but throughput is 35% below the reference run.

```
      0ms       100       200       300       400       500  510ms
      |---------|---------|---------|---------|---------|----|
DATA  [ worker decode batch N+2 ..............................] (2 workers, pegged)
CPU   [wait: next(dataloader) ~185ms][launch fwd][launch bwd  ]
CUDA                                 ▐▌▐▌▐▌▐▌▐▌▐▌▐▌▐▌▐▌▐▌▐▌▐▌
H2D   ....................[memcpy HtoD 12ms]
GPU   ......(idle 190ms)......[=fwd 105=][====== bwd 200 =====][opt 15]
NCCL                                          [AllReduce, overlapped, hidden]
      |<--- gap 190ms --->|<---------- busy 320ms ------------->|
      |<--------------------- step = 510ms -------------------->|
```

### The read

**Step boundary.** 510ms period. Reference hardware does this workload in ~330ms — so ~180ms is unexplained. Good: the trace should show us a ~180ms anomaly, and anything smaller is a distraction.

**Gaps.** There it is: the GPU compute stream is idle for the first 190ms of every step. During the gap, the CPU row is blocked inside `next(dataloader)` (in torch.profiler this shows as `enumerate(DataLoader)#...__next__`), and the dataloader worker processes are pegged doing JPEG decode + augmentation. The GPU is not waiting on itself; it's waiting on Python.

**Wrong hypothesis, considered and rejected:** "`nvidia-smi` said 100% util, so the GPU is busy — the problem must be slow kernels." `nvidia-smi` utilization means "at least one kernel was resident during the sample window," and it samples coarsely; a GPU that's idle 37% of every step can still read 100%. Also rejected: "it's the HtoD copy" — the memcpy is 12ms and could be fully hidden with `pin_memory` + a prefetch stream; it's real but it's 12ms, not 190.

**Overlap / longest kernel.** Once fed, the 320ms of busy time is healthy: the all-reduce hides under backward, and the kernel mix matches the reference run. Nothing to do inside the busy region. The discipline order saves you here — someone who jumps straight to "longest kernel" optimizes a GEMM in a step that's 37% idle.

**Quantify the supply/demand mismatch.** 2 workers × ~190 images/s/worker decode+augment ≈ 380 images/s supplied. GPU demand: 256 images / 320ms ≈ 800 images/s. Supply is ~half of demand — which is exactly why the gap ≈ the busy time × (deficit ratio): the loader needs ~670ms to produce a batch the GPU consumes in 320ms → ~190ms exposed after pipelining. The arithmetic is consistent; say it out loud.

### Confirming measurement

Two one-liners. (1) In the profiler, sum `DataLoader.__next__` wait time per step: 185ms, matching the gap. (2) The decisive experiment: swap in a synthetic in-memory dataset (`torch.randn` tensors) — step time drops to ~325ms. That isolates input pipeline vs everything else in one run.

### Fix and expected win

`num_workers=2 → 16` (supply 380 → ~3,000 img/s, now 3.8× demand), `pin_memory=True`, `prefetch_factor=4`; if CPU decode is still marginal at scale, move JPEG decode to GPU (NVJPEG/DALI) — but don't reach for DALI before trying the free flags.

- Before: 510ms/step.
- After: gap 190 → ~5ms; step ≈ **325ms** → **1.57×** throughput, matching reference.

### In the interview (60 seconds)

"Step is 510ms but the GPU compute stream is idle for the first 190ms of every one — the CPU row is stuck in `DataLoader.__next__` and two worker processes are pegged decoding JPEGs. `nvidia-smi` said 100% util, but that counter is 'any kernel resident,' not 'busy' — the trace overrules it. Supply math: 2 workers give ~380 images/s; the GPU consumes 800. The busy 320ms is healthy — comm hides under backward — so there's nothing to win inside it. Fix is boring: 16 workers, pinned memory, prefetch; GPU decode only if CPU becomes the wall again. Confirm with a synthetic-data run — if step time drops to ~325ms, case closed. Expected win: 1.57×."

---

## Case 3 — Launch-bound small-batch inference

**Scenario:** 1×L40S, 7B model, batch-1 decode (on-prem latency-sensitive product). Inter-token latency 28ms; roofline says a memory-bound decode step should take ~11ms.

```
   One decode step, zoomed to 3ms of it:
      0µs        500       1000      1500      2000      2500  3000µs
      |----------|---------|---------|---------|---------|-----|
CPU   [lnch][lnch][lnch][lnch][lnch][lnch][lnch][lnch][lnch][ln   ← wall of
CUDA  cudaLaunchKernel ×~1,100 per token, back-to-back              launches
GPU   ▐▌...▐▌....▐▌...▐▌....▐▌...▐▌....▐▌...▐▌....▐▌...▐▌...
       ^8µs ^17µs idle between kernels
      kernels: vectorized_elementwise_kernel, ampere_bf16_s16816gemm_64x64,
               flash_fwd_splitkv_kernel, cast / rotary / rmsnorm elementwise...

   Full token, to scale:
GPU   [▐▌▐▌▐▌ ... sparse picket fence ... ▐▌▐▌▐▌]        28ms/token
      GPU busy: Σ kernel time ≈ 9ms   idle-between-kernels ≈ 19ms
```

### The read

**Step boundary.** One decode step = one token = 28ms. Zooming in, there is no single gap and no single long kernel — the signature is different: a *picket fence*. Roughly **1,100 kernels per token**, average ~8µs each, with ~17µs of idle between consecutive kernels. Sum of kernel time ≈ 1,100 × 8µs ≈ 9ms; idle ≈ 1,100 × 17µs ≈ 19ms; 9 + 19 = 28ms. The arithmetic closes — the step is two-thirds launch overhead.

The CPU row confirms it: a solid wall of `cudaLaunchKernel` calls with the CPU never getting ahead of the GPU. The GPU finishes each 8µs kernel before the CPU has launched the next one. This is the classic launch-bound regime: kernel duration < launch+dispatch cost (~5–10µs CUDA API + Python/framework overhead on top).

**Wrong hypothesis, considered and rejected:** "The GEMMs are small and inefficient — let's tune tile sizes or quantize." At batch 1, decode GEMMs are GEMVs; they're memory-bound and already near their floor — 7B in FP16 is 14 GB of weights, and 14 GB / 864 GB/s (L40S) ≈ 16ms... wait, that exceeds our 9ms of busy time, so weights must already be INT8/FP8 (7 GB / 864 GB/s ≈ 8ms ✓ — consistent with the observed 9ms busy). Making kernels faster attacks the 9ms; the bug is the 19ms of *nothing*.

**Gaps / overlap.** No comm stream (single GPU). The "gap" is distributed: it's between every pair of kernels, which is why it hides from people who only look for big holes.

### Confirming measurement

`nsys stats --report cuda_api_sum`: `cudaLaunchKernel` count ≈ 1,100/token and total API time ≈ wall time. Or the counter that settles it in one number: **kernels per token**. A well-fused, graphed 7B decode is 200–400 kernels; 1,100 means eager mode with unfused elementwise soup (rmsnorm, rotary, residual adds, casts each as separate `vectorized_elementwise_kernel` launches). One-line experiment: run 2 tokens under `CUDA_LAUNCH_BLOCKING=0` vs a `torch.cuda.CUDAGraph` capture of the decode step and diff the timelines.

### Fix and expected win

1. **CUDA Graphs** on the decode step (static shapes at batch 1 make this easy; every serving framework does this — vLLM/TRT-LLM graph the decode path). Replaces 1,100 launches with one graph launch: idle-between-kernels ≈ 19ms → ~0.5ms.
2. **Fusion** (torch.compile / fused rmsnorm+rotary+residual) cuts kernel *count* and also shrinks the 9ms busy time by removing redundant activation traffic — call it 9 → 7.5ms.

- Before: 28ms/token.
- After: ~**8ms/token**, ≈ **3.5×**, now sitting on the weight-bandwidth roofline where batch-1 decode belongs.

### In the interview (60 seconds)

"28ms per token but the sum of GPU kernel time is only 9ms. The timeline is a picket fence: ~1,100 kernels per token averaging 8µs, with 17µs of idle between each — kernel duration below launch cost, CPU launch wall keeping the GPU starved. The tempting fix is 'optimize the small GEMMs,' but they're already at the weight-bandwidth floor — 7 GB of INT8 weights over 864 GB/s is ~8ms, which matches the busy time. The 19ms of overhead is the target: CUDA-graph the decode step to collapse the launches, then fuse the elementwise soup to cut kernel count. Confirm with `nsys stats cuda_api_sum` — launch count and API time tell the whole story. Expected: 28 → ~8ms/token, back on the roofline."

---

## Case 4 — Straggler rank in an 8-GPU all-reduce

**Scenario:** 8×H100, 13B model, FSDP-style training. Throughput dropped 30% after a node reprovision. Trace below is **rank 0**; the team is convinced "NCCL got slow."

```
   Rank 0 (what everyone is staring at):
      0ms       100       200       300       400       500       600ms
      |---------|---------|---------|---------|---------|---------|
GPU   [== fwd 130 ==][====== bwd 250 ======][opt]
NCCL  .....................................[==== AllReduce 215ms ====]
                                             ncclDevKernel_AllReduce_Sum_bf16
      |<-- compute 395 -->|                 |<---- "comm" 215 ----->|
      |<--------------------- step = 610ms ----------------------->|

   Per-rank compute time (the view that solves it):
   rank 0  [==== 395ms ====][wait 195][AR 20]
   rank 1  [==== 400ms ====][wait 190][AR 20]
   rank 2  [==== 395ms ====][wait 195][AR 20]
   rank 3  [=========== 590ms ===========][AR 20]   ← straggler
   rank 4  [==== 390ms ====][wait 200][AR 20]
   ...ranks 5–7 similar to 0–2...
   (collective can't start until the last rank arrives)
```

### The read

**Step boundary.** 610ms, up from ~430ms before the reprovision — a 180ms regression to explain.

**Gaps/overlap.** On rank 0's trace, the villain *appears* to be a 215ms all-reduce. But a collective's kernel duration on any one rank includes **wait time**: the NCCL kernel launches, then spins until every rank arrives. A long `ncclDevKernel_AllReduce` is a *symptom* that means either (a) the wire is slow or (b) somebody showed up late. You cannot distinguish these from one rank's trace.

**Wrong hypothesis, considered and rejected:** "NCCL/NVLink degraded — retune `NCCL_ALGO`, check the switch." First do the size math: this bucket is ~1.6 GB; ring all-reduce moves 2×7/8×1.6 ≈ 2.8 GB; even at a modest 150 GB/s that's ~19ms. Wire slowness can't produce 215ms on a 1.6 GB message inside an NVLink domain — but a rank arriving 195ms late produces exactly this.

**The straggler hunt.** Collect per-rank step decomposition (torch.profiler on all ranks, or just log per-rank `fwd+bwd` wall time — one line). Ranks 0–2, 4–7: ~395ms compute. **Rank 3: 590ms.** Same kernels, same shapes, ~1.5× slower across the board — uniformly slow compute on one GPU. And 590 − 395 = 195ms of wait on the other ranks: the arithmetic closes; rank 3's excess *is* everyone's "long NCCL kernel," and the true comm is ~20ms.

**Cause taxonomy for a uniformly-slow rank:** (a) **thermal/power throttle** — SM clocks pinned low (check first; cheapest); (b) **bad NIC/PCIe link** — but that shows up in comm, not uniform compute slowness, so it doesn't fit this signature; (c) **imbalanced shard / data skew** — different work per rank; doesn't fit either, since kernel *count* matches and each kernel is individually slower; (d) row-remapped/ECC-degraded HBM. Signature (a) or (d).

### Confirming measurement

`nvidia-smi -q -d PERFORMANCE,CLOCK -i 3` or DCGM (`DCGM_FI_DEV_SM_CLOCK`, `DCGM_FI_DEV_CLOCK_THROTTLE_REASONS`): rank 3's GPU reports SM clock **1,410 MHz vs 1,980 MHz** on its peers, throttle reason `SW Thermal Slowdown`. 1410/1980 ≈ 0.71 — a compute-bound-ish step at 0.71 clock lands within a few percent of the observed 395→590ms. Numbers agree; done. (After the reprovision: a fan profile / airflow issue on that tray.)

### Fix and expected win

Fix the cooling (or swap the GPU if it's ECC row-remapping). No code change.

- Before: 610ms/step.
- After: rank 3 rejoins at ~395ms compute; step ≈ 395 + 20 (real comm) ≈ **415ms** → **1.47×**, matching pre-incident throughput.

Staff add-on: this class of failure recurs, so leave behind a monitor — per-rank step time (max/median across ranks) and DCGM clock/throttle alerts. One slow GPU taxes all eight.

### In the interview (60 seconds)

"Rank 0 shows a 215ms all-reduce, but NCCL kernel time includes waiting for the slowest rank — a long collective means slow wire *or* late arrival, and a 1.6 GB message over NVLink can't take 215ms on bandwidth alone, so I go hunting for a straggler. Per-rank compute times: seven ranks at ~395ms, rank 3 at 590ms — and 590 minus 395 is exactly the 195ms everyone else waits. Uniformly slow kernels on one GPU means clocks, not workload: DCGM shows rank 3 throttled to 1,410 MHz, thermal slowdown after the reprovision. Fix the airflow, no code change, step goes 610 → ~415ms. And I'd add a per-rank step-time spread alert, because one throttled GPU taxes the whole ring."

---

## Case 5 — Serving: prefill blocking decode

**Scenario:** 2×H100 TP=2, 70B chat model, continuous batching, ~30 concurrent streams. p50 ITL 22ms, but users report "the stream freezes mid-answer": ITL p99 is 490ms.

```
   Decode-stream view over 1.5s, with request arrivals annotated:
      0ms       250       500       750       1000      1250    1500ms
      |---------|---------|---------|---------|---------|--------|
SCHED d d d d d d d [P: req#8812 arrives, 8K-token prompt] d d d d
GPU   [d][d][d][d][d][d][d][##### prefill 8K = 460ms #####][d][d][d]
       22ms each            flash_fwd_kernel + s16816gemm       22ms
                            (batch of decodes: NOT running)
ITL   22 22 22 22 22 22 |........ 482ms stall ...........| 22 22
      for every in-flight stream, simultaneously
```

### The read

**Step boundary.** In serving there's no training step; the unit is one scheduler iteration. Healthy iterations are ~22ms decode batches. The pathology is periodic: every time a long prompt arrives, one iteration balloons to ~460ms.

**Gaps.** None — the GPU is *fully busy* during the stall. This is the case that teaches "gap analysis" isn't "idle analysis": the decode streams stall because the engine scheduled an entire 8K-token prefill as one iteration, and every in-flight request's next token waits behind it. ITL spike = prefill duration + one decode: 460 + 22 ≈ 482ms ✓, matching the p99. Cross-check the rate: if ~1.5% of iterations are prefill-blocked, p99 ITL lands right on the spike value — consistent with the percentile data.

**Wrong hypothesis, considered and rejected:** "p99 spikes = GC pauses / CPU scheduler jitter — look at the Python row." The spikes correlate perfectly with `num_prompt_tokens > 4K` arrivals in the request log, and the GPU timeline shows dense `flash_fwd_kernel`/GEMM work during each spike, not idleness. Jitter produces gaps; this produces *someone else's compute*. Also rejected: "add a second replica" — that dilutes frequency but any replica that takes a long prompt still freezes its streams; it's an architecture problem, not capacity.

**Overlap.** The real question is batch *composition*: prefill (compute-bound, loves big token batches) and decode (bandwidth-bound, latency-critical) are being scheduled in mutually exclusive iterations. They should share iterations.

### Confirming measurement

One line from the serving metrics: scatter ITL vs "prefill tokens scheduled in the same iteration." Every ITL > 100ms sits on an iteration with ≥ 4K prefill tokens. That's the confirmation — no profiler needed, which is itself worth saying: in serving, the request log is often the fastest trace.

### Fix and expected win

**Chunked prefill** (Sarathi-style, default in vLLM V1): split the 8K prefill into 512-token chunks and co-schedule each chunk with the ongoing decode batch. Each mixed iteration: 512 prefill tokens ≈ 29ms + decode batch ≈ 8ms marginal → ~35ms iterations while a prefill drains over 16 iterations.

- ITL p99: 490ms → ~**40ms** (bounded by chunk iteration time).
- Cost, stated honestly: TTFT for the long prompt rises ~25% (460ms → 16 × 35 ≈ 560ms) and per-chunk KV re-reads add overhead — you're trading a little TTFT tail for the ITL tail, which is the right trade for chat. Tune chunk size against the H100 saturation point (~8K tokens fills the GPU; 512 keeps iterations short — sweep it).

Deeper coverage of chunked prefill, token budgets, and prefill/decode disaggregation (the next step if long prompts dominate): [/optimization/14-llm-serving-optimization/](/optimization/14-llm-serving-optimization/).

### In the interview (60 seconds)

"p50 ITL is 22ms, p99 is 490 — and the timeline shows why: healthy 22ms decode iterations, then a single 460ms iteration of pure prefill kernels whenever an 8K prompt arrives. The GPU isn't idle during the stall — it's doing someone else's prefill — so this isn't jitter and it isn't capacity; it's scheduling. 460 plus one decode is 482ms, exactly the p99. Fix is chunked prefill: 512-token chunks co-scheduled with the decode batch, iterations become ~35ms, ITL p99 drops to ~40ms. I'd say the trade out loud: the long prompt's TTFT goes up ~25%, which is the right trade for chat. Confirm before shipping with one scatter plot — ITL against prefill tokens per iteration."

---

## Case 6 — The healthy trace (when the timeline can't help you)

**Scenario:** 8×H100, 13B model training, grad accumulation with micro-batch 1 (seq 256 per micro-step after packing changes). Trace looks clean. But MFU is 33% and the team wants to know where the rest went.

```
      0ms        50        100       150       200  210ms (one micro-step)
      |----------|---------|---------|---------|----|
CPU   [launches, comfortably ahead of GPU]
GPU   [gemm][fa][gemm][gemm][ew][gemm][fa][gemm]...  ← wall-to-wall, 97% busy
NCCL  [ bucketed all-reduce, hidden under backward ]  ← overlapped ✓
      gaps: ~6ms total (3%)          longest kernel:
                                     ampere_bf16_s16816gemm_128x64_ldg8_stages
                                     M=256, N=4096, K=4096 — 31% of GPU time
```

### The read

Run the discipline anyway. **Step boundary:** clean 210ms micro-steps. **Gaps:** ~3%, nothing to collect. **Overlap:** comm fully hidden. **Longest kernel:** a BF16 GEMM family at 31% of GPU time. The timeline is *healthy* — and that is a finding, not a dead end. It means the remaining loss is **inside** kernels, and Nsight Systems / torch.profiler timelines cannot see inside a kernel. You have hit the resolution limit of the tool. Say that explicitly; knowing your instrument's limits is the staff signal in this case.

**Wrong hypothesis, considered and rejected:** "33% MFU with a busy timeline means the profiler's overhead is distorting things / MFU math is wrong." Check: MFU is computed from model FLOPs and wall time, both verified; profiler-off step time matches. The GPU genuinely spends 97% of the time in kernels that collectively deliver 33% of peak. So individual kernels are running far below roofline — the question moves from *when* to *how fast*.

**Hand off to the kernel-level tool.** Take the dominant GEMM into Nsight Compute:

```
ncu --set full --kernel-name regex:s16816gemm -c 3 ...
  SM Throughput (pipe_tensor):        24%
  DRAM Throughput:                    82%   ← memory-bound
  Achieved occupancy:                 fine; irrelevant (see 17-perf-debugging §2.4)
```

Now the shape math explains everything. The dominant GEMM is **M=256, N=4096, K=4096** — M collapsed to 256 because micro-batch 1 × seq 256.

- FLOPs = 2·M·N·K = 2 × 256 × 4096 × 4096 ≈ 8.6 GFLOP.
- Bytes ≈ 2(MK + KN + MN) = 2 × (1.05M + 16.8M + 1.05M) × 2 B ≈ 37.7 MB — dominated by the weight matrix, streamed for a skinny activation.
- Arithmetic intensity ≈ 8.6e9 / 37.7e6 ≈ **228 FLOP/B**. H100 ridge point: 989 TFLOPs / 3.35 TB/s ≈ **295 FLOP/B**. The GEMM sits *below the ridge* — memory-bound by shape, exactly matching ncu's 82% DRAM / 24% tensor-pipe readout. No kernel tuning fixes a shape problem.

### Confirming measurement

The ncu section above *is* the confirmation (DRAM 82%, tensor pipe 24%). One-line experiment: run the same GEMM at M=1024 in isolation (`torch.matmul` microbench) — TFLOPs roughly 2.5× higher at identical N, K.

### Fix and expected win

Raise M: micro-batch 1 → 4 (or repack to seq 1024) so the dominant GEMMs run at M=1024. AI ≈ 2·1024·4096·4096 / (2·(4.2M+16.8M+4.2M)·2B... ) ≈ 683 FLOP/B — comfortably compute-bound. Memory permitting via the grad-accum trade (activation memory ×4, offset by fewer accumulation steps).

- The GEMM family (31% of time) speeds up ~2.3×; kindred skinny GEMMs (another ~30%) similarly. Expected MFU: 33% → **~48%**, step time ~210 → ~145ms per (now 4×-larger) micro-batch equivalent — verify end-to-end, since attention scales differently with sequence length than the GEMMs do.

Warp-stall taxonomy, roofline discipline, and the rest of the ncu workflow: [/optimization/17-perf-debugging/](/optimization/17-perf-debugging/).

### In the interview (60 seconds)

"I run the same read and the trace passes: 3% gaps, comm hidden, launches ahead of the GPU. That's a finding — at 33% MFU with a wall-to-wall timeline, the loss is *inside* the kernels, and a timeline can't see inside a kernel, so I switch instruments. Nsight Compute on the dominant GEMM: 82% DRAM throughput, 24% tensor pipe — memory-bound. The shape explains it: M=256 from micro-batch 1, arithmetic intensity ~228 FLOP/byte against H100's ridge of ~295 — below the ridge by construction. No amount of kernel tuning fixes a shape. Raise the micro-batch to 4, M becomes 1024, intensity ~680, the GEMM goes compute-bound, and I'd expect MFU in the high 40s. The meta-point: know when your profiler has run out of resolution and hand off to the next tool."

---

## The pattern across all six

| Case | Timeline signature | Trap hypothesis | Decisive measurement |
|---|---|---|---|
| 1. Unoverlapped all-reduce | One long NCCL kernel *after* backward | "NCCL is slow" | NCCL kernel count per step (1 vs ~26 buckets) |
| 2. Input starvation | Idle GPU + busy CPU at step start | "nvidia-smi says 100%" | Synthetic-data run |
| 3. Launch-bound decode | Picket fence: ~1,100 × 8µs kernels, 17µs gaps | "Optimize the small GEMMs" | `cuda_api_sum`: launch count ≈ wall time |
| 4. Straggler rank | Long collective on *every* rank | "The interconnect degraded" | Per-rank compute times + DCGM clocks |
| 5. Prefill blocks decode | ITL spikes = busy GPU, not idle | "GC / jitter / add replicas" | ITL vs prefill-tokens-per-iteration scatter |
| 6. Healthy trace | Nothing wrong in the timeline | "The MFU math is wrong" | ncu: DRAM 82% / tensor 24% on the top GEMM |

Three habits worth making explicit in the room: **do the bandwidth/FLOP arithmetic before accusing a component** (cases 1, 3, 4, 6 are all solved by a two-line calculation); **a long collective is a symptom, not a cause** (case 4); and **know each tool's resolution limit** — timeline tools end at kernel boundaries, and the confident move is saying so and switching to Nsight Compute (case 6).
