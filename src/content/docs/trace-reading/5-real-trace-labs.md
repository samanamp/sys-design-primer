---
title: "Real-Trace Labs & Datasets"
description: "Hands-on labs on real public artifacts — HTA's 8-GPU training traces in Perfetto + notebook, AcmeTrace fleet reliability analysis — and the reference list of public trace datasets with what each can and cannot teach."
---

# Real-Trace Labs and Public Datasets

The [drills](/trace-reading/4-trace-drills/) are synthetic and the [hands-on labs](/trace-reading/3-hands-on-profiling/) are self-captured; this page is the third leg — **real artifacts from other people's training runs**. The format split matters: timeline reading is a *viewer* skill (Perfetto/Nsight, eyes and scroll wheel), aggregate analysis is a *notebook* skill (pandas). Each lab below uses both, in that order: commit numbers from the viewer first, then compute the truth and reconcile.

## Lab R1 — Real 8-GPU training traces (HTA demo set)

**[▶ hta_trace_lab.ipynb](/notebooks/hta_trace_lab.ipynb)** · Artifacts: 8 per-rank Kineto JSONs from a real vision-transformer DDP job (HTA's demo data, ~4.5 MB/rank, auto-downloaded).

The flow: load `rank-0.json.gz` in [ui.perfetto.dev](https://ui.perfetto.dev) and commit four numbers by eye (step time, GPU-idle fraction, overlapped-vs-serialized NCCL, top kernel family) — then run HTA's four analyses (temporal breakdown, comm-comp overlap %, idle-time attribution, kernel breakdown) and reconcile. Target: manual reads within ~10% of computed. These four quantities are exactly the [step-anatomy decomposition](/google-interview/6-answer-pod-training/) the pod-training answer does with claimed numbers — here you compute them from a real trace.

## Lab R2 — Fleet reliability from real LLM-cluster data (AcmeTrace)

**[▶ acmetrace_goodput_lab.ipynb](/notebooks/acmetrace_goodput_lab.ipynb)** · Artifacts: 880K job records from Shanghai AI Lab's two A100 clusters over six months (NSDI '24), including an explicit `NODE_FAIL` state.

The flow: job-mix shape (large jobs = sliver of count, bulk of GPU-hours) → terminal states by scale → **empirical MTBF from NODE_FAIL events in large long-running jobs** → the goodput waterfall from data. The punchline, which we validated while building the lab: real per-GPU MTBF comes out ≈ **15–21 years**, meaning the pod-training answer's 5-year prior is conservative by ~3–4× — and Young's √(2C·MTBF) means even a 4× better MTBF only *doubles* the checkpoint interval, so the "checkpoint every few minutes" conclusion survives contact with data. The lab ends with the confidence-ledger exercise: measured vs proxied vs assumed, applied to your own analysis. Bonus finding worth understanding before you run it: CANCELLED dominates GPU-hours (~66%) — big pretraining jobs run open-ended and get cancelled at convergence, the dataset's best lesson in *terminal state ≠ outcome quality*.

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
