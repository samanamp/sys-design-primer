---
title: "Real-Trace Labs & Datasets"
description: "Hands-on labs on real public artifacts — HTA's 8-GPU training traces in Perfetto + notebook, AcmeTrace fleet reliability analysis — and the reference list of public trace datasets with what each can and cannot teach."
---

# Real-Trace Labs and Public Datasets

The [drills](/trace-reading/4-trace-drills/) are synthetic and the [hands-on labs](/trace-reading/3-hands-on-profiling/) are self-captured; this page is the third leg — **real artifacts from other people's training runs**. The format split matters: timeline reading is a *viewer* skill (Perfetto/Nsight, eyes and scroll wheel), aggregate analysis is a *notebook* skill (pandas). Each lab below uses both, in that order: commit numbers from the viewer first, then compute the truth and reconcile.

## Lab R1 — Real 8-GPU training traces (HTA demo set)

**[▶ hta_trace_lab.ipynb](/notebooks/hta_trace_lab.ipynb)** · Artifacts: 8 per-rank Kineto JSONs from a real vision-transformer DDP job (HTA's demo data, ~4.5 MB/rank, auto-downloaded).

The flow: load `rank-0.json.gz` in [ui.perfetto.dev](https://ui.perfetto.dev) and commit five reads by eye (step time, idle fraction, NCCL-vs-compute ratio, overlap call, one-sentence diagnosis) — then run HTA's analyses and reconcile against the **ground-truth answer key computed from these exact files** (in the notebook, spoiler-folded). This artifact has a real, diagnosable pathology: it is a **communication-dominated job** (~61% of kernel time is NCCL, ~20% compute, overlap only ~20–22%, rank-0 idle 71% host-wait) — a *ratio* failure no scheduler can hide, i.e. the [scaling-book per-device-batch floor](/training/1-batch-size-primer/) violated in the wild. The final exercise is the five-sentence narration ending in a fix that changes the ratio, not just the schedule.

## Lab R2 — Fleet reliability from real LLM-cluster data (AcmeTrace)

**[▶ acmetrace_goodput_lab.ipynb](/notebooks/acmetrace_goodput_lab.ipynb)** · Artifacts: 880K job records from Shanghai AI Lab's two A100 clusters over six months (NSDI '24), including an explicit `NODE_FAIL` state.

The flow: job-mix shape (large jobs = sliver of count, bulk of GPU-hours) → terminal states by scale → **empirical MTBF from NODE_FAIL events in large long-running jobs** → the goodput waterfall from data. The punchline, which we validated while building the lab: real per-GPU MTBF comes out ≈ **15–21 years**, meaning the pod-training answer's 5-year prior is conservative by ~3–4× — and Young's √(2C·MTBF) means even a 4× better MTBF only *doubles* the checkpoint interval, so the "checkpoint every few minutes" conclusion survives contact with data. The lab ends with the confidence-ledger exercise: measured vs proxied vs assumed, applied to your own analysis. Bonus finding worth understanding before you run it: CANCELLED dominates GPU-hours (~66%) — big pretraining jobs run open-ended and get cancelled at convergence, the dataset's best lesson in *terminal state ≠ outcome quality*.

## Coverage: fault catalog × where you practice it

The [tools pages](/trace-reading/tools/1-nsys/) teach ~40 faults; not all are practicable on public artifacts. Where each major fault family gets hands-on reps:

| Fault family | Real artifact (R1/R2) | Synthetic [drills](/tools/trace-drills.html) | Self-capture ([Labs A–C](/trace-reading/3-hands-on-profiling/) + planned injection) |
| --- | --- | --- | --- |
| Exposed/unoverlapped communication | **R1 — the artifact's actual pathology** | ✓ | ✓ (injectable) |
| Host-bound / launch-bound | R1 idle-attribution (71% host_wait) + launch-stats stretch | ✓ | ✓ |
| Input starvation | — (not exhibited) | ✓ | ✓ (num_workers=0) |
| Straggler rank | R1 cross-rank diffs (mild spread) | ✓ | ✓ (throttle one rank) |
| Sync-forcing ops | — | ✓ | ✓ (`.item()` injection) |
| Memory churn / H2D | R1 memory-bw stretch | — | ✓ |
| Kernel-level (SOL/occupancy/coalescing) | — (needs [ncu](/trace-reading/tools/2-ncu/), no public reports) | — | ✓ (ncu exercises on own kernels) |
| Recompilation, DVFS, checkpoint stalls | — | ✓ (some) | ✓ |
| MoE all-to-all, multi-node NCCL | — (no public artifact exists) | ✓ (a2a scenario) | requires multi-node rental |
| Fleet: failures, MTBF, goodput, queueing | **R2 — measured from 880K real jobs** | — | — |

Read the gaps honestly: sync-hunting and kernel-level work only exist as self-capture; MoE-scale comm patterns have no public artifact at all — the synthetic drills and the worked answers carry those.

## The dataset shelf

| Dataset | What it is | Teaches | Can't teach |
| --- | --- | --- | --- |
| [HTA demo traces](https://github.com/facebookresearch/HolisticTraceAnalysis) | 8-rank Kineto JSONs, real DDP job | Step anatomy, overlap %, idle attribution, per-rank diffs — in Perfetto, zero setup | LLM-scale patterns (it's a ViT); why a kernel is slow |
| [AcmeTrace](https://github.com/InternLM/AcmeTrace) (NSDI '24) | 880K job records + 80 GB node telemetry, 2 LLM clusters, 6 months | Failure rates, MTBF, queue/goodput accounting, job-mix reality | Step-level anything; per-failure root cause |
| [Microsoft Philly](https://github.com/msr-fiddle/philly-traces) (ATC '19) | 2017-era DNN cluster job traces | Queueing, gang-scheduling effects, locality tradeoffs | Modern LLM workloads (pre-transformer era) |
| [Alibaba PAI GPU trace](https://github.com/alibaba/clusterdata) (2020) | Production ML cluster, task-level | Heterogeneous sharing, utilization patterns | Training-run internals |
| [MIT SuperCloud](https://dcc.mit.edu/) | Labeled GPU utilization time series | Utilization-signature classification | Causal diagnosis |
| [MLCommons Chakra](https://github.com/mlcommons/chakra) | Standardized execution traces (+ ASTRA-sim) | Comm-schedule replay/simulation, what-if topology experiments | Real-system noise, host effects |
| Your own captures ([Lab A–C](/trace-reading/3-hands-on-profiling/)) | nsys/torch.profiler with faults *you* injected | Ground-truth diagnosis practice — the only artifacts where you know the answer | Someone else's surprises |

The honest gap this shelf can't fill: **no lab publishes kernel-level traces of a frontier run.** The closest substitutes are self-captured traces of open training stacks (torchtitan, MaxText) and the worked artifacts in the [MFU-gap answer](/answers/mfu-gap-investigation.html).

## Iteration queue

Planned extensions, roughly in order: fault-injection flags for Lab C (capture-with-known-answer at multi-GPU scale), a Perfetto-permalink question set for each HTA rank (straggler diffing in the viewer), and an xprof lab once a cheap reproducible TPU capture recipe settles (Colab TPU + pure-JAX transformer).
