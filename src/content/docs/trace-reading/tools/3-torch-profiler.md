---
title: "torch.profiler & HTA: the Framework View"
description: "Reading and diagnosing PyTorch profiler traces: python↔ATen↔kernel correlation, Perfetto navigation, HTA multi-rank analyses, the framework-level fault catalog, and exercises on real public traces."
---

# torch.profiler & HTA: the Framework View

The PyTorch profiler sits **between** nsys and the model code: it sees python frames, ATen ops, autograd nodes, *and* the CUDA kernels they launched, correlated. It answers "*which line of my model is this GPU time*" — the question nsys can't answer (it sees kernels, not modules) and ncu doesn't ask. For multi-rank work, Meta's **HTA (HolisticTraceAnalysis)** computes fleet-style summaries over per-rank Kineto traces — and its [repo ships real demo traces](https://github.com/facebookresearch/HolisticTraceAnalysis), the best public artifacts to practice on.

## 1. Anatomy of the artifact

Kineto JSON, viewed in Perfetto (or `chrome://tracing`). Rows:

```text
python thread    — call stack (with with_stack=True), ProfilerStep# ranges
ATen ops         — aten::linear → aten::addmm …  (CPU-side op events)
Autograd         — autograd::engine ... backward node names
cuda_runtime     — cudaLaunchKernel etc., correlated to the op above them
GPU streams      — kernels + memcpy/memset, correlated to the launch
Memory timeline  — (profile_memory=True) allocator events, peak tracking
```

The **correlation arrows** (Perfetto: click kernel → "ArgSet/flow") are the whole point: kernel → launch → ATen op → python line. Learn the click path until it's reflex.

```python
with torch.profiler.profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    schedule=torch.profiler.schedule(wait=2, warmup=2, active=3),
    with_stack=True, profile_memory=True, record_shapes=True,
    on_trace_ready=torch.profiler.tensorboard_trace_handler("./tb"),
) as prof:
    for step, batch in enumerate(loader):
        train_step(batch); prof.step()
```

## 2. Reading order

1. **`ProfilerStep#` width** = step time. Compare CPU-row width vs GPU-row occupancy inside it: GPU sparse ⇒ the fault is *above* the kernels (host, input, sync) — the framework view will name it.
2. **Self-CPU vs CUDA table first** (`prof.key_averages().table(sort_by="self_cuda_time_total")`): the top-10 ops answer "where does GPU time go *by op*" before any timeline scrolling.
3. **Gap attribution:** for each GPU gap, what's on the python row? DataLoader `next` → input. `cudaStreamSynchronize` under an aten op → a sync-forcing op (find which — catalog below). Nothing → look for GC/GIL.
4. **Backward pass sanity:** autograd row should mirror forward ~2× wider; a *second forward-shaped block* inside backward is activation checkpointing's recompute — expected if you enabled it, a finding if you didn't.
5. **Multi-rank (HTA):** load all ranks, run the four analyses — temporal breakdown, comm-comp overlap %, idle-time attribution, kernel breakdown — then diff ranks: straggler hunting is a *distribution* question, not a timeline question.

## 3. Fault catalog

### Major faults

| Fault | Signature in this view | Confirming detail | Fix direction |
| --- | --- | --- | --- |
| **Sync-forcing op in hot loop** | `cudaStreamSynchronize` beneath `aten::item`/`aten::_local_scalar_dense`/`aten::nonzero`/`.cpu()` | key_averages: count per step ≥ 1 | Defer host reads, vectorize the condition, async metrics |
| **Input-bound step** | `ProfilerStep` opens with wide `enumerate(DataLoader)` frame, GPU empty | Self-CPU time of dataloader vs step | workers/prefetch/pre-decode (same row as [nsys catalog](/trace-reading/tools/1-nsys/)) |
| **Eager small-op storm** | Hundreds of sub-50µs ATen ops (elementwise chains, norm internals); CPU op row denser than GPU row; gaps between kernels | ops/step count; CPU launch time ≈ GPU busy time | `torch.compile` (fusion), fused optimizers, larger per-op work |
| **Unoverlapped DDP all-reduce** | `nccl:all_reduce` kernels serialized after last backward kernel | HTA overlap % ≈ 0 for the tail buckets | Check gradient-as-bucket-view, bucket sizes, static graph, no stray `no_sync` |
| **Straggler rank** | HTA: one rank's compute longer, others' NCCL wait longer, per step | HTA temporal breakdown per rank | Data imbalance (variable seq len!) → packing/sorting; or hardware |
| **Implicit dtype casts** | `aten::to`/`aten::copy_` pairs bracketing matmuls; fp32 GEMM kernels in a "bf16" run | record_shapes shows dtype; kernel names lack tensor-core tags | Fix autocast scope; keep master-weight casts out of hot loop |
| **Optimizer step dominance** | Post-backward: thousands of tiny `aten::add_`/`mul_` (per-param ops) | foreach/fused flag off in key_averages | `fused=True`/`foreach=True` Adam, or FSDP sharded step |
| **Activation-checkpointing surprise** | Recompute block present (unexpected) or absent (expected but missing) in backward | Compare fwd kernel sequence vs in-backward sequence | Fix `checkpoint()` coverage; see [activation checkpointing](/training/2-activation-checkpointing/) |
| **Memory churn → malloc stalls** | Memory timeline sawtooth; `cudaMalloc` events mid-step; peak near capacity | allocator stats: reserved vs allocated divergence (fragmentation) | Static shapes, expandable segments, preallocation |
| **GC / GIL stall** | Periodic all-thread python gap, no CUDA API activity | `gc` frames in stack view | Disable auto-GC in loop; move logging off-thread |

### Minor faults

- **Profiler schedule distortion** — `active` window catching a checkpoint step or the first post-compile step; always `wait` past warmup and capture ≥3 steps.
- **`record_function` absence** — unannotated custom blocks make attribution guesswork; wrap phases (`with record_function("router")`) before capturing, like NVTX for nsys.
- **`with_stack` overhead** — 2–3× CPU-side slowdown; fine for attribution runs, wrong for timing runs. Two captures, two purposes.
- **CUDA-graph regions** — captured graphs appear as one opaque kernel blob; per-op attribution inside is gone (expected, not a bug — attribute before graphing).
- **`torch.compile` regions** — Triton kernel names (`triton_per_fused_*`) replace ATen attribution; use `TORCH_LOGS=graph_breaks` alongside, and treat graph *breaks* (eager islands between compiled regions) as the finding.
- **DataLoader worker traces missing** — the profiler sees the main process only; worker-side slowness shows up merely as `next()` latency. Profile a worker separately (py-spy) when input-bound.

## 4. HTA on real public traces

```python
pip install HolisticTraceAnalysis
from hta.trace_analysis import TraceAnalysis
a = TraceAnalysis(trace_dir="tests/data/vision_transformer/")  # 8-rank demo traces in the repo
a.get_temporal_breakdown()             # compute/comm/memory/idle per rank
a.get_comm_comp_overlap()              # overlap % per rank
a.get_idle_time_breakdown()            # host-wait vs kernel-wait attribution
a.get_gpu_kernel_breakdown()           # top kernels per rank
```

These four outputs are the same four quantities the [pod-training step-anatomy](/google-interview/6-answer-pod-training/) decomposes by hand — computed from real 8-GPU traces.

## 5. Exercises

The full guided version of exercises 1–2 is **[Lab R1](/trace-reading/5-real-trace-labs/)** with its [companion notebook](/notebooks/hta_trace_lab.ipynb).

1. **HTA lab:** on the repo's demo traces, compute overlap % and idle-time attribution per rank *by hand from the timeline* for one rank (pick three gaps, attribute each), then run HTA and reconcile. Target: your manual numbers within ~10%.
2. **Straggler drill:** feed HTA the demo traces, find the worst rank by temporal breakdown, and write the five-sentence narration naming the catalog row.
3. **Sync hunt:** add `if loss.item() > 100: print(...)` to Lab A's loop ([hands-on](/trace-reading/3-hands-on-profiling/)), capture, and locate the sync from the key_averages table alone — count how many step-serializations one line costs.
4. **Cast hunt:** wrap the model in autocast but exclude one submodule; find the resulting `aten::to` traffic and the fp32 GEMM it feeds.
