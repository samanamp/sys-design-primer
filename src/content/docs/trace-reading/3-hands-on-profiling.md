---
title: "Hands-On: Capturing and Reading Your Own Traces"
description: "Three labs for building trace-reading muscle: torch.profiler + Perfetto on your Mac today, the full nsys/ncu workflow on a rented GPU box, and multi-GPU profiling with one report per rank."
---

# Hands-On: Capturing and Reading Your Own Traces

Reading traces other people captured is like reviewing code you never ran. To build real fluency you need to capture your own traces, predict what they will show, and then be wrong in instructive ways.

This page is three labs in increasing order of hardware requirements:

- **Lab A** runs on your Mac today, no GPU needed. It builds the mechanics: `torch.profiler`, Chrome trace export, Perfetto navigation.
- **Lab B** is the workflow you run the first hour you rent a GPU box: `nsys` end to end, NVTX annotation, delayed capture, then one `ncu` drill-down.
- **Lab C** extends Lab B to multi-GPU with `torchrun`: one report per rank, straggler hunting.

> **Companion notebook:** [`trace_reading_lab.ipynb`](/notebooks/trace_reading_lab.ipynb) (download) — a runnable version of Lab A plus a deliberately pathological training loop to diagnose in Perfetto. CPU-safe by default; GPU cells activate automatically when CUDA is available. Open in [Google Colab](https://colab.research.google.com/) via *File → Upload notebook*.

The vocabulary and reading order for interpreting what you capture is in [Trace Reading Vocabulary](/trace-reading/1-trace-reading-vocabulary/). This page assumes you have it open in another tab.

---

## 1. Lab A: torch.profiler on CPU (runs on your Mac)

You do not need a GPU to learn 80% of trace reading. `torch.profiler` on CPU produces the same trace format, the same tooling, and the same navigation skills. What changes on a GPU is *which rows exist*, not how you read them.

### 1.1 Profile a small transformer step

```python
import torch
import torch.nn as nn
from torch.profiler import profile, schedule, ProfilerActivity, tensorboard_trace_handler

model = nn.TransformerEncoder(
    nn.TransformerEncoderLayer(d_model=256, nhead=8, dim_feedforward=1024, batch_first=True),
    num_layers=4,
)
opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
x = torch.randn(8, 128, 256)

def train_step():
    opt.zero_grad(set_to_none=True)
    out = model(x)
    loss = out.pow(2).mean()
    loss.backward()
    opt.step()

with profile(
    activities=[ProfilerActivity.CPU],   # add ProfilerActivity.CUDA on a GPU box
    schedule=schedule(wait=1, warmup=2, active=3, repeat=1),
    record_shapes=True,
    with_stack=True,
    profile_memory=True,
    on_trace_ready=lambda p: p.export_chrome_trace("trace_step.json"),
) as prof:
    for _ in range(8):
        train_step()
        prof.step()          # advances the schedule; without this the schedule never fires

print(prof.key_averages(group_by_input_shape=True).table(
    sort_by="cpu_time_total", row_limit=15))
```

The pieces worth understanding rather than cargo-culting:

- **`schedule(wait=1, warmup=2, active=3)`** — the profiler skips `wait` steps (disabled), traces-and-discards for `warmup` steps (the first profiled steps carry instrumentation overhead and allocator noise), then records `active` steps. Always profile with a schedule; a trace of step 0 is a trace of memory allocation and autotuning, not of your training loop.
- **`prof.step()`** — the schedule is driven by you. Forgetting this call is the single most common "my trace is empty" bug.
- **`record_shapes=True`** — attaches input shapes to each op. This is how you notice a matmul running at `[8, 128, 256] x [256, 1024]` when you believed batch was 64. Costs a little overhead and temporarily holds tensor references, so turn it off for overhead-sensitive measurement runs.
- **`with_stack=True`** — attaches Python file:line to ops. This is what turns "some `aten::copy_` is slow" into "line 84 of `data.py` is slow". Expensive; use it for diagnosis captures, not timing captures.
- **`on_trace_ready`** — two idioms. `tensorboard_trace_handler("./log")` writes one file per capture cycle into a directory that TensorBoard (or Perfetto) can consume, and is the right choice when `repeat > 1` or in long jobs. A lambda calling `export_chrome_trace(path)` writes a single JSON you fully control — simpler for a lab, and the file opens directly in Perfetto. They produce the same trace data; the difference is file management.

### 1.2 Open it in Perfetto

Go to [ui.perfetto.dev](https://ui.perfetto.dev), *Open trace file*, select `trace_step.json`. (The file is parsed locally in your browser; nothing is uploaded.) Then practice the motions:

- **W/S to zoom, A/D to pan.** Trace reading is a keyboard activity. Mouse-only navigation is why people find traces exhausting.
- Find one `train_step`'s worth of activity. Click an `aten::addmm` slice; read its duration, shape (from `record_shapes`), and stack (from `with_stack`).
- Use *M* to mark a slice and see its duration against the whole capture window.
- Select a time range across one full step and read the aggregate slice table at the bottom — this is `key_averages()` scoped to a region, which is often more useful than the global table.

### 1.3 What transfers to GPU reading — and what does not

Transfers directly:

- All of the navigation, the JSON trace format, the schedule/step mechanics.
- The skill of matching op names (`aten::addmm`, `aten::scaled_dot_product_attention`) to your model code via stacks and shapes.
- The instinct of "select a region, read the aggregate, find the dominant op".

Does **not** transfer:

- On CPU, the op's duration *is* the work. On GPU, the CPU row shows *launches* and the real work is on separate GPU stream rows, connected by flow arrows. A CPU-side op that looks cheap can launch a kernel that dominates the GPU timeline — and vice versa.
- CPU traces cannot show you gaps between kernels, launch overhead, or copy/compute overlap — the core objects of GPU trace reading.
- There is no async on CPU: the timeline reads top-to-bottom like a call stack. Do not let that habit fool you into reading a GPU trace the same way.

---

## 2. Lab B: the nsys workflow on a rented GPU box

`torch.profiler` shows you PyTorch's view. Nsight Systems (`nsys`) shows you the machine's view: every kernel, every memcpy, every CUDA API call, OS scheduling, and (on newer GPUs) hardware SM/memory utilization sampled over time. This is the tool the interview question "how would you find out why this training job is slow?" is really asking about.

### 2.1 Annotate first: NVTX ranges

Raw kernel names are hostile (`ampere_sgemm_128x64_tn...`). NVTX ranges give the timeline your program's structure:

```python
import torch

torch.cuda.nvtx.range_push("forward")
out = model(x)
torch.cuda.nvtx.range_pop()

# or, scoped:
with torch.cuda.nvtx.range("backward"):
    loss.backward()
```

For a fully automatic version, `torch.autograd.profiler.emit_nvtx()` wraps *every* ATen op in an NVTX range (with shapes if `record_shapes=True`):

```python
with torch.autograd.profiler.emit_nvtx():
    train_step()
```

Use `emit_nvtx` for diagnosis (it is verbose and adds overhead); use a handful of manual `nvtx.range` calls (`dataload`, `forward`, `backward`, `opt`) for routine captures. Either way, annotate before you capture — an unannotated nsys trace of a transformer is a wall of anonymous cutlass kernels.

### 2.2 Delayed capture with cudaProfilerApi

Never trace from process start: you would capture CUDA context creation, allocator warmup, and cuDNN autotuning. Gate the capture from inside the program:

```python
for step in range(200):
    if step == 100:
        torch.cuda.profiler.start()   # cudaProfilerStart()
    train_step()
    if step == 110:
        torch.cuda.profiler.stop()    # cudaProfilerStop()
```

### 2.3 The capture command

```bash
nsys profile \
  --trace=cuda,nvtx,osrt \
  --gpu-metrics-devices=all \
  --capture-range=cudaProfilerApi \
  --capture-range-end=stop \
  -o train_trace \
  python train.py
```

Flag by flag:

- `--trace=cuda,nvtx,osrt` — CUDA API + kernels + memcpys, your NVTX ranges, and OS runtime calls (the `osrt` part is how you catch a dataloader blocked in `read()` or a thread stuck in a futex).
- `--gpu-metrics-devices=all` — periodic hardware counters (SM active, tensor core active, DRAM bandwidth) as timeline rows. This is what lets you distinguish "GPU busy" from "GPU busy doing useful math". Turing or newer; needs sufficient permissions on the box.
- `--capture-range=cudaProfilerApi` with `--capture-range-end=stop` — arms the profiler but records only between your `torch.cuda.profiler.start()/stop()` calls, then ends the collection.
- `-o train_trace` — writes `train_trace.nsys-rep`.

Short form you will see in the wild: `nsys profile -t cuda,nvtx,osrt -c cudaProfilerApi -o train_trace python train.py`.

### 2.4 Opening and summarizing

Copy the `.nsys-rep` to your Mac and open it in the Nsight Systems GUI — the host app runs fine on macOS/Apple Silicon for *viewing*; only the capture needs an NVIDIA machine. Or stay on the box and get text summaries:

```bash
nsys stats train_trace.nsys-rep                       # default report set
nsys stats --report cuda_gpu_kern_sum train_trace.nsys-rep   # kernel time ranking
nsys stats --report cuda_gpu_mem_time_sum train_trace.nsys-rep  # memcpy/memset time
nsys stats --report nvtx_sum train_trace.nsys-rep     # time per NVTX range
```

`cuda_gpu_kern_sum` is the GPU-side answer to `key_averages()`; `nvtx_sum` tells you forward vs backward vs optimizer at a glance before you ever open the GUI.

### 2.5 Checklist: the first six things to look at

In the reading order from [Trace Reading Vocabulary](/trace-reading/1-trace-reading-vocabulary/):

1. **Step boundary and step time.** Find one full iteration via your NVTX ranges. Is step time stable across steps, or noisy?
2. **GPU idle gaps.** Zoom the CUDA stream rows within one step. White space between kernels is the number-one finding: dataloader stalls, CPU-side launch overhead, or synchronization.
3. **CPU/GPU relationship.** Is the CPU thread running far ahead of the GPU (healthy, launch-bound-free) or lockstepped with it (something is synchronizing — look for `cudaStreamSynchronize`/`cudaMemcpy` in the CUDA API row)?
4. **Top kernels.** From `cuda_gpu_kern_sum`: do the top 5 kernels look like your model's math (gemm, flash attention), or like overhead (elementwise, copies, `cat`)?
5. **Memcpy traffic.** HtoD/DtoH rows during steady state. DtoH inside the step usually means an `.item()`/`.cpu()` sync in the loop.
6. **GPU metrics rows.** SM active and tensor-core active during the big kernels. High occupancy of the timeline with low SM/TC activity means the GPU is "busy" being memory-bound or launch-bound.

### 2.6 The single-kernel follow-up: one ncu command

nsys tells you *which* kernel dominates; Nsight Compute (`ncu`) tells you *why* that kernel is slow. Once the checklist has produced a suspect:

```bash
ncu --set full \
    --kernel-name "regex:.*sgemm.*" \
    --launch-count 3 \
    -o kernel_report \
    python train.py
```

- `--set full` collects every section (SOL, memory workload, occupancy, scheduler stats) — slow, because ncu replays the kernel many times, which is why you filter.
- `--kernel-name "regex:..."` matches the mangled kernel name you found in nsys; `--launch-count 3` stops after a few instances.

The reading order inside an ncu report is its own topic; the point here is the two-tool rhythm: **nsys for where the time goes, ncu for why one kernel is slow.** Running ncu without an nsys-identified target is the classic beginner inversion.

---

## 3. Lab C: multi-GPU with torchrun

The single-GPU workflow scales to distributed with one idea: **one nsys report per rank**, then compare timelines side by side.

### 3.1 One report per rank

Put `nsys` *inside* the per-rank launch so each process gets its own profiler, and use nsys's `%q{ENV}` substitution to name outputs by rank:

```bash
torchrun --nproc_per_node=4 --no-python \
  nsys profile \
    --trace=cuda,nvtx,osrt \
    --capture-range=cudaProfilerApi --capture-range-end=stop \
    -o trace_rank%q{RANK} \
    python train_ddp.py
```

`%q{RANK}` expands to the value of the `RANK` environment variable that `torchrun` sets for each worker, producing `trace_rank0.nsys-rep` … `trace_rank3.nsys-rep`. (If your nsys version predates `--no-python` support in this arrangement, the equivalent is a small wrapper script per rank that execs `nsys profile ... python train_ddp.py`.) Keep the delayed-capture gating from Lab B — all ranks call `torch.cuda.profiler.start()` at the same step, so the captures cover the same iterations.

### 3.2 Straggler hunting

Open two or more rank reports and align them on the same step (NVTX step ranges again). What you are looking for:

- **NCCL kernel duration skew.** All-reduce kernels (`ncclDevKernel...`) act as barriers: the *fastest* ranks show long NCCL kernels because they are waiting inside the collective for the slowest rank. Long NCCL time on rank 2 usually means the problem is on some *other* rank.
- **Compute skew.** Compare forward/backward NVTX durations per rank. A rank with a slower GPU (thermals, a shared box), an unbalanced data shard, or CPU contention shows longer compute and short NCCL waits — it is the straggler.
- **Dataloader skew.** `osrt` rows show one rank's workers stuck in disk reads while others idle in the collective.

The one-line diagnostic: *the straggler is the rank with the shortest communication time.*

### 3.3 NCCL sanity

Before profiling a "slow" distributed job at all, run one step with:

```bash
NCCL_DEBUG=INFO torchrun --nproc_per_node=4 train_ddp.py
```

and read the ring/tree topology and transport lines NCCL prints at init. A surprising number of "mysterious stragglers" are NCCL falling back to a slow transport (no NVLink/P2P where you expected it, sockets instead of IB). Two minutes of log reading can save an afternoon of trace reading.

---

## 4. Common capture failure modes

Every one of these has cost someone an hour on a rented box with the meter running:

- **Empty torch.profiler trace.** You used a `schedule` but never called `prof.step()`, or the loop ran fewer iterations than `wait + warmup + active`. The profiler exits without ever entering an active window.
- **Empty nsys capture with `-c cudaProfilerApi`.** Your program never called `torch.cuda.profiler.start()`, or crashed before reaching it. nsys arms, waits forever, and writes nothing useful. Test the gating logic with a 20-step run before a long one.
- **Trace of the wrong steps.** No warmup discipline: the capture window covers step 0-3 and is dominated by cuDNN autotuning, `cudaMalloc`, and JIT compilation. Steady-state claims need steady-state captures — gate the window to start at step 100, not step 0.
- **Timing with diagnosis flags on.** `with_stack=True`, `emit_nvtx`, and `--gpu-metrics-devices` all add overhead. Capture twice: once heavy for *where/why*, once minimal for *how much*. Never quote step times from the heavy capture.
- **`.nsys-rep` too large to open.** You traced 10 minutes of training. You almost never need more than 10-20 steps of steady state; use the delayed capture to keep files in the tens of MB.
- **Permissions on GPU metrics.** `--gpu-metrics-devices` and `ncu` counter collection can fail with `ERR_NVGPUCTRPERM` on boxes where the driver restricts performance counters. On a rented box you may need the provider to enable it (or a root container). Check this in the first five minutes, not after writing your harness.
- **Profiling a broken run.** If loss is NaN or the dataloader is erroring and retrying, the timeline is fiction. Verify the job is *correct* before asking why it is slow.

---

## 5. Getting GPU hours to practice on

You do not need to own an H100 to do Labs B and C. Marketplace-style providers (Lambda, RunPod, Vast.ai, Prime Intellect, and the spot/preemptible tiers of the major clouds) rent single consumer or datacenter GPUs by the hour, and the whole nsys workflow above is a one-to-two-hour session on a single cheap GPU — an older card is fine, since you are practicing reading, not chasing peak FLOPs. Lab C needs a multi-GPU instance for one short session. Colab-style notebooks are less suitable here because you want a real shell, `nsys` installed, and the ability to pull `.nsys-rep` files back to your Mac.

---

## 6. Where to go next

- Work through [`trace_reading_lab.ipynb`](/notebooks/trace_reading_lab.ipynb): it contains the Lab A code plus a deliberately slow training loop (three planted pathologies) and an exercise to find them in Perfetto before reading the solutions cell.
- Re-read [Trace Reading Vocabulary](/trace-reading/1-trace-reading-vocabulary/) after your first real capture — the vocabulary lands differently once you have seen your own gaps.
- In interviews, narrate this workflow in order: annotate, delayed capture, nsys, six-item checklist, then ncu on one kernel. The ordering itself is what signals hands-on experience.
