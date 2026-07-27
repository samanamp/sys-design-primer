---
title: "Real-Trace Labs & Datasets"
description: "Hands-on labs on real public artifacts — HTA's 8-GPU training traces in Perfetto + notebook, AcmeTrace fleet reliability analysis — and the reference list of public trace datasets with what each can and cannot teach."
---

# Real-Trace Labs and Public Datasets (R1–R10)

The [drills](/trace-reading/4-trace-drills/) are synthetic and the [hands-on labs](/trace-reading/3-hands-on-profiling/) are self-captured; this page is the third leg — **real artifacts from other people's training runs**. The format split matters: timeline reading is a *viewer* skill (Perfetto/Nsight, eyes and scroll wheel), aggregate analysis is a *notebook* skill (pandas). Each lab below uses both, in that order: commit numbers from the viewer first, then compute the truth and reconcile.

Sixteen labs, four sources: **real kernel-level traces** (R1, R3–R6, R14 — other people's systems), **fault-injected traces generated for this track** (R7–R9 — real profiler output, exact ground truth because the fault was injected on purpose; regenerate or extend with [generate.py](/traces/labs/generate.py)), **fleet data** (R2, R10), and **production LLM inference request traces** (R11–R13, R16 — the datasets the serving papers were actually built on; R15 — a frontier lab's own kernel traces). Protocol everywhere: commit numbers from the artifact before computing.

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

## Lab R3 — A real single-rank inference trace

**Artifact:** [`inference_rank_0.json.gz`](https://raw.githubusercontent.com/facebookresearch/HolisticTraceAnalysis/main/tests/data/inference_single_rank/inference_rank_0.json.gz) (HTA repo, 6 MB). Open in Perfetto; then HTA with `trace_dir` pointing at its folder.

**Task:** this is the opposite regime from R1. Commit from the timeline: GPU-idle fraction, comm share, and your diagnosis — then verify.

<details><summary>Answer key (computed from this file)</summary>

Idle **82.6%**; kernel time is **98.5% computation, 0% communication**; compute only 17% of wall-clock. Diagnosis: **small-batch inference that is host/launch-bound** — the GPU is mostly dark not because kernels are slow but because nothing keeps it fed. The catalog rows in play: launch-bound + host-side gaps ([torch-profiler page](/trace-reading/tools/3-torch-profiler/)). The instructive contrast with R1: same tooling, *inverted* diagnosis — comm-dominated vs starvation-dominated.
</details>

## Lab R4 — An H100 training trace, and cross-generation comparison

**Artifact:** [`h100_trace.json.gz`](https://raw.githubusercontent.com/facebookresearch/HolisticTraceAnalysis/main/tests/data/h100/h100_trace.json.gz) (HTA repo, 6 MB).

**Task:** run the same four analyses as R1, then *compare against R1's numbers*: idle %, comm share, overlap. Also: inspect kernel names — identify tensor-core generation markers (kernel families differ from the A100-era ViT trace).

<details><summary>Answer key</summary>

Idle **48.1%**, compute **42.3%**, comm ≈ **18% of kernel time with only ~8.5 points overlapped** — a *partially* exposed-comm profile, milder than R1's 61%-comm pathology but the same fault family. Two-trace takeaway: the same signature at different severities — practice saying "how bad" with numbers, not just "what."
</details>

## Lab R5 — Regression hunting with trace_diff (control vs test)

**Artifacts:** the HTA repo's `tests/data/trace_diff/control/` and `test/` directories (clone `HolisticTraceAnalysis` with `--depth 1`).

**Task:** the golden-trace discipline from the [debug cards](/trace-reading/tools/1-nsys/): given a known-good trace and a suspect one, produce the diff — `TraceDiff.compare_traces` gives added/removed/changed kernels and duration deltas. Questions: which ops changed, by how much, and what single sentence would you post in the incident channel? This lab is the muscle behind "keep golden traces per release; diffing beats profiling from scratch."

## Lab R6 — Critical path analysis (alexnet)

**Artifacts:** `tests/data/critical_path/alexnet/` in the HTA repo; API: `analyzer.critical_path_analysis(...)`.

**Task:** compute the critical path, then answer the question that matters: **what fraction of the step is on the critical path because of the GPU vs the CPU vs dependencies?** Optimizing anything off the critical path is wasted work — this is the formal version of the altitude-ladder's "localize first." Bonus: manually trace one critical-path segment in Perfetto and confirm the tool's claim.

## Labs R7–R9 — Fault-injected traces (known ground truth)

Real `torch.profiler` traces generated with deliberate faults ([generate.py](/traces/labs/generate.py) — CPU-only, runs anywhere, regenerate or add your own faults). Open each in [ui.perfetto.dev](https://ui.perfetto.dev); the `record_function` spans (`data_loading` / `forward` / `metrics`) are your row labels.

**R7 — the pair:** [`lab-healthy.json.gz`](/traces/labs/lab-healthy.json.gz) vs [`lab-starved.json.gz`](/traces/labs/lab-starved.json.gz). One is 100× slower per step. From the timeline: which span grew, by what factor, and what's the fix?

<details><summary>Ground truth</summary>
Healthy: p50 step **0.3 ms**. Starved: **36.9 ms** — `data_loading` totals 283 ms of 8 steps (~97% of wall time); `forward` is unchanged. Injected fault: a 30 ms sleep in the loader — the input-starvation signature: step opens with a dead span, compute unchanged. Fix family: workers/prefetch/pre-decode.
</details>

**R8 — op storm:** [`lab-opstorm.json.gz`](/traces/labs/lab-opstorm.json.gz). Same math as healthy, ~7× slower. The timeline looks *busy* — no big gaps. Count ops, then diagnose.

<details><summary>Ground truth</summary>
**7,504 aten ops vs healthy's 304** (~25×) — the same matmuls shredded into hundreds of trivial elementwise ops (`add`/`mul` chains). p50 step 2.2 ms vs 0.3 ms: pure per-op overhead, the eager small-op storm from the [fault catalog](/trace-reading/tools/3-torch-profiler/). Fix family: fusion/`torch.compile`. The lesson R8 exists to teach: *busy ≠ efficient* — this trace has no idle gaps at all.
</details>

**R9 — the mystery stall:** [`lab-hoststall.json.gz`](/traces/labs/lab-hoststall.json.gz). Step times are bimodal. Find the period, attribute both stall mechanisms (there are two!), and name the production analogue of each.

<details><summary>Ground truth</summary>
p50 **45 ms**, max **104 ms**, healthy math underneath (forward ≈ 1 ms). Fault 1: a 60 ms sleep in `metrics` every 3rd step (production analogue: synchronous logging/lock/nvml scrape). Fault 2: forced `gc.collect()` every 2nd step inside a span labeled `mystery` — 151 ms total (production analogue: GC pauses from allocation churn). The two periods (3-step and 2-step) interfere to produce the bimodal pattern — exactly the "periodic stalls, no load correlation" scenario from the debug catalog: measure the period first.
</details>

## Lab R10 — Fleet archaeology: Philly (2017) vs Acme (2023)

**Artifact:** [Microsoft Philly traces](https://github.com/msr-fiddle/philly-traces) (ATC '19) alongside R2's AcmeTrace numbers.

**Task:** repeat R2's Part-1/Part-2 analysis on Philly (job mix by GPU count, failure rates, queueing delay), then answer the comparative questions: how did the job-size distribution shift between the DNN era and the LLM era? Did large-job failure rates improve? Queue-wait? Write five sentences on what changed *because the workload changed* vs what stayed (the constants of GPU-cluster life). This is the systems-evolution perspective interviewers probe with "how has serving/training infrastructure changed since 2018?"

## Labs R11–R13 — Production LLM inference request traces

Kernel traces show *how* a step runs; these show *what production asks for* — request-level traces from real LLM services, released for serving research. They are the workload side of every `[batching]`/`[capacity]` card in the inference-optimization deck.

### R11 — Azure LLM inference traces (the Splitwise dataset)

**Artifact:** [Azure/AzurePublicDataset](https://github.com/Azure/AzurePublicDataset) — `AzureLLMInferenceDataset2023` (per-request arrival time + input/output token counts; the ISCA'24 Splitwise paper's data) and the 2024/2025 successors (DynamoLLM; multimodal ModServe).

**Task:** reproduce the argument for prefill/decode disaggregation from the raw data: (1) plot input-vs-output token distributions (code vs conversation subsets differ sharply); (2) compute each request's prefill FLOPs and decode bandwidth-seconds using the [roofline formulas](/training/1-batch-size-primer/); (3) show the fleet-level prefill:decode resource ratio and its variance over hours — that variance IS the case for independent pool scaling. Stretch: feed the arrival process into a simple goodput simulator at a fixed TTFT/TPOT SLO and find the utilization knee.

### R12 — Mooncake trace: prefix caching against real block hashes

**Artifact:** [`mooncake_trace.jsonl`](https://github.com/kvcache-ai/Mooncake/blob/main/FAST25-release/arxiv-trace/mooncake_trace.jsonl) (FAST'25 artifact) — arrivals, token counts, and **remapped KV block hashes** per request; the repo ships a cache-size calculator and hit-rate simulator.

**Task:** the only public dataset where prefix reuse is *measurable*, not assumed: (1) compute the theoretical prefix-cache hit rate vs cache size (LRU over blocks, leaf-first — the [PagedAttention card's](/trace-reading/tools/3-torch-profiler/) eviction story on real data); (2) convert hit rate into saved prefill tokens/s and TTFT delta; (3) evaluate KV-aware routing: how much does hit rate drop if requests scatter across 4 replicas vs route-by-prefix? That number is the entire KV-aware-routing debate, computed.

### R13 — BurstGPT: 121 days of real request arrivals

**Artifact:** [HPMLL/BurstGPT](https://github.com/HPMLL/BurstGPT) — ~5.3M ChatGPT/GPT-4 requests (Azure-powered) with timestamps, model, request/response tokens, conversation-vs-API flag, including failed requests.

**Task:** the burstiness lab: (1) test the Poisson assumption — compute the arrival process's coefficient of variation in 1-s/10-s/1-min windows (the paper's finding: bursty at short scales, Gamma-distributed); (2) quantify the capacity consequence: peak-to-mean ratio → the headroom a fixed SLO demands vs Poisson planning; (3) compare conversation vs API traffic shapes (session structure vs machine-gun calls); (4) design the autoscaling signal from the [serving cards](/trace-reading/tools/3-torch-profiler/): token backlog vs request rate, on real data. Stretch: the failed-request subsets — do failures correlate with bursts?

## Lab R14 — Cross-vendor kernel traces (MTIA, AMD)

**Artifacts:** HTA repo: `tests/data/mtia_inference_trace/` (Meta's MTIA accelerator running inference) and `tests/data/amd_trace/` (ROCm).

**Task:** run the R1/R3 reading order on non-NVIDIA timelines. The point is transfer: kernel names, stream semantics, and copy engines differ; the *method* (step boundary → gaps → overlap → longest kernel) does not. Write down which of the nine [vocabulary signatures](/trace-reading/1-trace-reading-vocabulary/) you can still identify without NVIDIA-specific cues — that's the tool-agnostic skill the whole track claims to build, tested.

## Lab R15 — DeepSeek's own traces: a frontier MoE stack, profiled

**Artifact:** [deepseek-ai/profile-data](https://github.com/deepseek-ai/profile-data) — real PyTorch-profiler traces from DeepSeek's production framework, released during open-infra week: `prefill.json` (EP32, 4K prompts, 16K tok/GPU, two micro-batches), `decode.json` (EP128, 128 req/GPU, RDMA all-to-all), `train.json` (DualPipe, EP64). Load directly in Perfetto or `chrome://tracing`.

**Task:** this is the closest public artifact to "what a frontier lab's timeline actually looks like." (1) In `decode.json`, find the **dual micro-batch overlap**: one micro-batch's attention/MLP compute hiding the other's all-to-all — the comm-hiding condition from the [pod-training answer](/google-interview/6-answer-pod-training/) (*a2a hides iff per-layer a2a ≤ paired compute*), observable on a real timeline. Measure the exposed remainder. (2) Compare prefill vs decode: same model, both phases — verify the regime split (dense compute blocks vs KV/comm-dominated steps) you know from the [roofline cards](/training/1-batch-size-primer/). (3) In `train.json`, identify DualPipe's interleaved forward/backward chunks and where the EP all-to-all sits relative to them. (4) Write the five-sentence narration for decode as if it were your fleet — then ask: what would you check next that this trace can't show you (the answer involves per-expert load balance)?

## Lab R16 — WildChat-1M: real multi-turn session structure

**Artifact:** [allenai/WildChat-1M](https://huggingface.co/datasets/allenai/WildChat-1M) (838K real conversations, UTC timestamps, turn counts to 249, model/country/language fields; ODC-BY, parquet on HF).

**Task:** the session-dynamics lab — the piece R11–R13's flat request streams can't teach: (1) compute **inter-turn gap distributions** — this is the real input to the [KV offload decision inequality](/trace-reading/tools/3-torch-profiler/) (keep KV resident vs page out between turns: what fraction of turn-gaps exceed the reload-beats-recompute threshold?); (2) session length × turn count → cumulative re-prefill cost naive vs prefix-cached (the multi-turn card's quadratic-vs-linear claim, on real sessions); (3) diurnal + geographic load curves → replica-count schedule; (4) turn-count long tail (max 249!) → what session-affinity routing must tolerate. Sibling dataset: LMSYS-Chat-1M for cross-checking distributions.

## The dataset shelf

| Dataset | What it is | Teaches | Can't teach |
| --- | --- | --- | --- |
| [HTA demo traces](https://github.com/facebookresearch/HolisticTraceAnalysis) | 8-rank Kineto JSONs, real DDP job | Step anatomy, overlap %, idle attribution, per-rank diffs — in Perfetto, zero setup | LLM-scale patterns (it's a ViT); why a kernel is slow |
| [AcmeTrace](https://github.com/InternLM/AcmeTrace) (NSDI '24) | 880K job records + 80 GB node telemetry, 2 LLM clusters, 6 months | Failure rates, MTBF, queue/goodput accounting, job-mix reality | Step-level anything; per-failure root cause |
| [Microsoft Philly](https://github.com/msr-fiddle/philly-traces) (ATC '19) | 2017-era DNN cluster job traces | Queueing, gang-scheduling effects, locality tradeoffs | Modern LLM workloads (pre-transformer era) |
| [Alibaba PAI GPU trace](https://github.com/alibaba/clusterdata) (2020) | Production ML cluster, task-level | Heterogeneous sharing, utilization patterns | Training-run internals |
| [MIT SuperCloud](https://dcc.mit.edu/) | Labeled GPU utilization time series | Utilization-signature classification | Causal diagnosis |
| [MLCommons Chakra](https://github.com/mlcommons/chakra) | Standardized execution traces (+ ASTRA-sim) | Comm-schedule replay/simulation, what-if topology experiments | Real-system noise, host effects |
| [Azure LLM inference traces](https://github.com/Azure/AzurePublicDataset) | Production request traces (arrivals, token counts; 2023–2025, incl. multimodal) | Workload characterization, disaggregation math, SLO capacity planning | Anything below the request level |
| [Mooncake trace](https://github.com/kvcache-ai/Mooncake/tree/main/FAST25-release) (FAST'25) | Requests + real KV block hashes | Prefix-cache hit rates, KV-aware routing value — measured, not assumed | Kernel/step internals |
| [BurstGPT](https://github.com/HPMLL/BurstGPT) | 5.3M requests, 121 days, incl. failures | Arrival burstiness, autoscaling signals, peak-to-mean headroom | Per-request latency internals |
| [DeepSeek profile-data](https://github.com/deepseek-ai/profile-data) | Real prefill/decode/DualPipe traces from a frontier MoE stack | Dual-microbatch comm hiding, EP all-to-all anatomy, phase contrast | Fleet behavior; per-expert balance over time |
| [WildChat-1M](https://huggingface.co/datasets/allenai/WildChat-1M) / LMSYS-Chat-1M | 838K real conversations with timestamps | Session structure: inter-turn gaps, KV offload economics, diurnal load | Token-level serving internals |
| [MLPerf inference results](https://github.com/mlcommons/inference_results_v5.0) | Vendor-submitted loadgen latency logs | SLO/percentile reading across hardware, scenario semantics (server vs offline) | Why a submission is fast (configs are tuned black boxes) |
| Your own captures ([Lab A–C](/trace-reading/3-hands-on-profiling/)) | nsys/torch.profiler with faults *you* injected | Ground-truth diagnosis practice — the only artifacts where you know the answer | Someone else's surprises |

The honest gap, updated: for years **no lab published kernel-level traces of a frontier run** — DeepSeek's open-infra-week release (R15) broke that in early 2025 with real prefill/decode/DualPipe traces, though they remain curated single-node excerpts, not a full-fleet capture. The remaining substitutes: self-captured traces of open stacks (torchtitan, MaxText) and the worked artifacts in the [MFU-gap answer](/answers/mfu-gap-investigation.html).

## Iteration queue

Planned extensions, roughly in order: fault-injection flags for Lab C (capture-with-known-answer at multi-GPU scale), a Perfetto-permalink question set for each HTA rank (straggler diffing in the viewer), and an xprof lab once a cheap reproducible TPU capture recipe settles (Colab TPU + pure-JAX transformer).
