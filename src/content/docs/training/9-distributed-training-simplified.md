---
title: Distributed Training Infrastructure (simplified)
description: Distributed Training Infrastructure
---
```
"Design the infrastructure for training a 500B-parameter model on a 4,000-GPU cluster. Cover the job submission, scheduling, checkpointing, failure handling, and the developer ergonomics. A single node failure should not kill a 3-week training run. Walk me through it."
```
---
# 500B Training Infra — 60-Minute Delivery

> Speaking pace target: ~140 words/min. Section timings noted. Total ~55 min of content + 5 min for interruptions/pushback.

---

## 1. Scope and Reframe (3 min)

Before I draw anything, let me push back on the framing of the question — because the way it's set up, you can write a wrong answer and not realize it.

**[STAFF SIGNAL: saying no]** "A 3-week training run" is calendar framing. Training runs end when the token budget is hit, not when the calendar runs out. For a 500B dense model on a frontier token budget — say 5 trillion tokens — at 50% MFU on 4,000 H100s, the run is around **70 days, not 21**. If "3 weeks" is a hard requirement, then either the model is smaller, the cluster is bigger, or the token budget is far below frontier. I'm going to design for the realistic case — a roughly 70-day run — because the failure budget is the architecture, and 70 days of failures is a fundamentally harder problem than 3 weeks. If the design survives 70 days, 3 weeks is trivial.

**[STAFF SIGNAL: saying no]** "A single node failure should not kill the run" is the floor, not the goal. Llama 3's published reliability data — the best public source we have — is 419 unexpected interruptions in 54 days on a 16K-H100 cluster. Scaled to 4,000 GPUs and 70 days, I expect around **140 interruptions over the run**. The real bar isn't "survive one node failure." It's: survive a rack failure, survive correlated network incidents, and survive silent data corruption in any of those 4,000 GPUs without invisibly poisoning the model.

**[STAFF SIGNAL: scope negotiation]** Here are the assumptions I'm committing to, so the rest of the answer is concrete:

| Dimension | Commitment |
|---|---|
| Model | Dense decoder transformer, 500B params, ~120 layers, hidden 12288 |
| Sequence length | 8K |
| Optimizer | AdamW with FP32 master weights and FP32 (m, v) |
| Precision | BF16 weights and activations, FP8 on linear-layer matmuls via Transformer Engine |
| Hardware | H100 SXM5 80GB, 8 GPUs per node, NVLink within node, 8×400Gbps InfiniBand between nodes |
| Token budget | 5T tokens |
| Cluster slice | **3,968 GPUs** in the training mesh (TP=8 × PP=8 × DP=62), **32 GPUs** held back as hot spares |

That last row matters. Most candidates put all 4,000 GPUs in the training mesh. I'm reserving 4 nodes — 32 GPUs — as **hot spares**, kept alive in a separate NCCL world, ready to be promoted into the training mesh when something dies. That's how you bound recovery time, and recovery time is the budget.

If the model were MoE, expert parallelism enters the picture and the math changes. If sequences were 1M tokens, context parallelism becomes mandatory. If the hardware were GB200 NVL72, the NVLink domain expands from 8 GPUs to 72 and the parallelism shape inverts — TP could grow to 64, PP could disappear. I'll come back to these at the end.

---

## 2. The Math (6 min)

Three numbers drive everything else: model state, FLOPs, failure budget. Let me put them on the board.

### Model state

```
500B params × 2 bytes (BF16 weights):                1.0 TB
500B params × 2 bytes (BF16 gradients):              1.0 TB
500B params × 8 bytes (Adam m + v in FP32):          4.0 TB
500B params × 4 bytes (FP32 master copy):            2.0 TB
─────────────────────────────────────────────────────────
Total persistent state:                              8.0 TB
```

**[STAFF SIGNAL: capacity math]** Eight terabytes. Not one. This is the number that breaks people who think "500B params is 1TB." Optimizer state and the master weights triple your memory bill. Activations are on top of that — for an 8K sequence with selective recomputation across 120 layers, peak activation memory in the busiest pipeline stage is around 800 GB across the TP group, or roughly 100 GB per GPU after sharding across TP=8.

Per H100 (80 GB) under TP=8 × PP=8 with ZeRO-1 sharding optimizer state across DP=62:

```
Weights, sharded by TP×PP=64:       1 TB / 64       =  15.6 GB
Adam state, sharded by TP×PP×DP:    4 TB / 3968     =   1.0 GB
FP32 master, sharded the same way:  2 TB / 3968     =   0.5 GB
Gradients (peak before reduce):     1 TB / 64       =  15.6 GB
Activations (selective recompute):                    ~30-50 GB
NCCL buffers, framework overhead:                      ~5-8 GB
────────────────────────────────────────────────────────────
Per-GPU peak:                                          ~65-75 GB
H100 capacity:                                            80 GB
```

We're at 80–95% of capacity. That's the right operating point. Any lower and we're leaving compute on the table; any higher and we're one OOM away from a crash on a long-tail step.

### FLOPs and wall-clock

The standard rule for transformer training is **6 × N × D FLOPs**, where N is parameters and D is tokens. The 6 comes from one forward pass (2× operations per parameter per token: a multiply and an add) plus a backward pass (4× — twice as expensive as forward).

```
6 × 500B × 5T tokens  =  1.5 × 10²⁵ FLOPs total
```

H100 BF16 peak is 989 TFLOPS. At 50% MFU that's roughly 495 TFLOPS effective per GPU. With FP8 on the dominant matmuls, we get roughly 1.4× on top of that, call it 650 TFLOPS effective.

```
3,968 GPUs × 650 TFLOPS  =  2.58 EFLOPS effective
1.5 × 10²⁵ / 2.58 × 10¹⁸  =  5.8 × 10⁶ seconds  =  ~67 days active compute
```

Add ~5–8% downtime → **~72-day end-to-end run**.

### Failure budget

```
Llama 3 rate:    419 events / (54 days × 16,384 GPUs)
                 = 4.7 × 10⁻⁴ events / GPU / day

Our cluster:     4,000 GPUs × 72 days × 4.7 × 10⁻⁴
                 ≈ 135 expected unexpected interruptions
```

**[STAFF SIGNAL: failure-budget reframing]** This is the central architectural pressure. ~140 failures over 72 days. If recovery time is 10 minutes per event with hot spares, total downtime is 23 hours = 1.3% of the run. If recovery is 30 minutes (no hot spares, scheduler reissue), it's 69 hours = 4.0%. The difference between those two designs — about 3 days of compute on a 4,000-GPU run at ~$2/H100-hour — is roughly **$575,000 of waste per run**. That's the engineering budget for getting recovery right.

Everything from here is in service of this number.

---

## 3. Architecture Overview (3 min)

Let me draw the whole thing first, then we'll go deep on each piece.

```
┌──────────────────────────────────────────────────────────────────────┐
│                       CONTROL PLANE (out of band)                    │
│   ┌────────────┐  ┌────────────┐  ┌──────────────┐  ┌─────────────┐  │
│   │ Job Submit │  │Orchestrator│  │ Hot-Spare    │  │Observability│  │
│   │ (config-   │  │(supervises │  │ Pool Manager │  │(MFU, loss,  │  │
│   │  as-code,  │  │ ranks,     │  │              │  │ traces, W&B)│  │
│   │  git SHA)  │  │ recovers)  │  │              │  │             │  │
│   └────────────┘  └────────────┘  └──────────────┘  └─────────────┘  │
└──────────────────────────────────────────────────────────────────────┘
                              │
                              v
┌──────────────────────────────────────────────────────────────────────┐
│              DATA PLANE: 4,000 H100s across 500 nodes                │
│                                                                      │
│   Training mesh: 3,968 GPUs        Hot spares: 32 GPUs (4 nodes)     │
│   ┌─────────────────────────────┐  ┌────────────────────────────┐    │
│   │  TP=8  (within a node,      │  │  Pre-warmed, NCCL already  │    │
│   │         over NVLink)        │  │  initialized in a "shadow" │    │
│   │  ×                          │  │  world group, can replace  │    │
│   │  PP=8  (across nodes,       │  │  a failed node in seconds  │    │
│   │         rail-aligned)       │  │                            │    │
│   │  ×                          │  │                            │    │
│   │  DP=62 (with ZeRO-1)        │  │                            │    │
│   └─────────────────────────────┘  └────────────────────────────┘    │
│                                                                      │
│   ┌─────────────────────┐    ┌──────────────────────────────┐        │
│   │ Streaming Dataloader│    │ Tiered Checkpoint Storage    │        │
│   │ (deterministic,     │    │  Tier 1: peer node DRAM      │        │
│   │  shard per DP rank, │    │  Tier 2: local NVMe          │        │
│   │  resumable)         │    │  Tier 3: object store (S3)   │        │
│   └─────────────────────┘    └──────────────────────────────┘        │
└──────────────────────────────────────────────────────────────────────┘
```

**[STAFF SIGNAL: blast radius reasoning]** Note the failure-domain boundaries built into the picture. A single GPU dying gets isolated to its node. A node dying gets swapped from the hot-spare pool. A whole rack dying — say a top-of-rack switch failure — pauses the run; we restart from the checkpoint when capacity comes back. We don't try to keep training with fewer GPUs because changing world size mid-run changes the global batch size, which changes the optimizer dynamics. I'll come back to that decision.

---

## 4. Parallelism Choice (10 min)

This is the most loaded section. Let me explain *why* TP=8 × PP=8 × DP=62, not just commit to it.

### Why TP=8 — the NVLink wall

Tensor parallelism splits each matrix multiply across multiple GPUs. After the matmul, the partial results have to be combined with an AllReduce. This AllReduce happens **twice per transformer layer** (once for attention, once for the MLP). For 120 layers, that's 240 AllReduces per forward pass, and another 240 in the backward.

The bandwidth difference between NVLink and InfiniBand is roughly:

```
NVLink (within a node):    900 GB/s per GPU pair
InfiniBand (between nodes): ~50 GB/s effective per GPU
                          ──────
                          ~18× slower
```

If TP crossed the node boundary, every one of those 480 AllReduces per step would slow down by ~18×. The model would be communication-bound to the point of uselessness. So **TP is bounded by the NVLink domain**. On H100 nodes that's 8 GPUs. So TP=8.

(If the hardware were GB200 NVL72, the NVLink domain expands to 72 GPUs and TP=64 becomes feasible. The whole architecture would be different — you might fit weights and optimizer state inside one NVL72 with TP-only, eliminating PP entirely.)

### Why not pure FSDP — the IB ceiling

**[STAFF SIGNAL: rejected alternative]** A natural alternative: forget 3D parallelism, just use FSDP across all 4,000 GPUs. That shards weights 4,000 ways — 256 MB per GPU — and reconstructs each layer on demand by all-gathering across the cluster.

The math kills it. FSDP all-gathers each layer's weights before its forward pass and again before its backward. Across 4,000 GPUs, those all-gathers go over IB, not NVLink. At 1 TB of weights touched twice per step and ~100 steps per minute, you need ~3 TB/sec of sustained inter-node bandwidth, sustained, overlapping with compute. The fabric supports the peak but not the sustained pattern. In practice, FSDP-only at this scale is dominated by collective wait time.

3D parallelism keeps the weight sharing inside the TP=8 NVLink island, which is essentially free, and only pays IB for two things: pipeline send/recv between PP stages (small, latency-tolerant) and gradient reduce-scatter across DP (large but happens once per step, not per layer).

### Why PP=8

Pipeline parallelism splits the model's layers across stages. Each stage holds ~15 layers at PP=8. After TP-sharding each layer's weights by 8, the per-stage weight memory is around 15.6 GB per GPU. That fits in 80 GB with room for activations.

PP communication is point-to-point send/recv between adjacent stages — small tensors, latency-tolerant, easy to overlap with compute. The cost of PP is the **pipeline bubble**, which I'll cover in a second.

### Why DP=62 with ZeRO-1

That leaves 62 data-parallel replicas to fill out the 3,968 training GPUs. ZeRO-1 shards the optimizer state across DP, which is what gives us the memory headroom for activations. We don't go to ZeRO-2 (sharding gradients) or ZeRO-3 (sharding weights too) because the gradient reduce-scatter is already paid in the DP collective, and weight all-gather in ZeRO-3 would explode communication.

### Pipeline schedule and the bubble

The naive pipeline schedule has a "fill" phase at the start where stages downstream are idle waiting for data, and a "drain" at the end. That idle time is the **bubble**.

Naive 1F1B (one forward, one backward) bubble fraction:
```
bubble = (p - 1) / m
```
where p = pipeline depth and m = number of microbatches per global batch. At p=8 and m=64, that's 7/64 = **10.9% wasted compute**.

**[STAFF SIGNAL: pipeline-bubble discipline]** Megatron's interleaved 1F1B (the V-schedule) splits each stage's layers into v chunks, interleaving forward and backward across them. The bubble formula becomes:
```
bubble = (p - 1) / (v × m)
```
At v=2: 7/128 = **5.5%**. At v=4: **2.7%**. Each step from v to v+1 doubles the number of pipeline send/recv operations and increases activation memory.

```
Time →
PP stage 0:  F0₀ F0₁ F0₂ F0₃ ... B0₀ F1₀ B0₁ F1₁ ... [ drain ]
PP stage 1:      F0₀ F0₁ F0₂ ... B0₀ F1₀ B0₁ ...    [ drain ]
PP stage 2:           F0₀ F0₁ ...                            
   ...                                                       
                          ↑                       ↑
                  steady state:               drain bubble
                  interleaved F+B            ~5.5% of step
```

I'm going to commit to **v=2**. Not v=4, because v=4 pushes us past the 80 GB ceiling at PP stage 0 due to the extra activation memory. At 5.5% bubble on a 72-day run, that's about 4 days of bubble. Going to v=4 would save ~2 days of compute, worth roughly $1M, but the memory math doesn't allow it under our chosen TP/PP layout.

**[STAFF SIGNAL: rejected alternative]** Zero-Bubble PP (Sun et al. 2023) splits backward into two passes — one for input gradients, one for weight gradients — and schedules them independently. The bubble drops below 1%. The gain is real, but the framework complexity on top of Megatron-Core is substantial and it's not yet proven at 4K-GPU production scale. I'd run interleaved 1F1B v=2 in version one and earmark Zero-Bubble for version two.

### The full layout

```
                  4 hot-spare nodes (32 GPUs, NCCL shadow)
                                |
                                v
   ┌─────────────────────────────────────────────────────────────┐
   │                    Training mesh: 3,968 GPUs                │
   │                                                             │
   │       PP stage:  0     1     2     3     4     5     6   7  │
   │                  |     |     |     |     |     |     |   |  │
   │   DP=0   TP=8  [N0]–[N1]–[N2]–[N3]–[N4]–[N5]–[N6]–[N7]      │
   │   DP=1   TP=8  [N8]–[N9]–...                                │
   │   ...                                                       │
   │   DP=61  TP=8  [N488]–...                  –[N495]          │
   │                                                             │
   │   Each [N#] = one node = 8 H100s = one TP=8 group           │
   │   Within node:    NVLink AllReduce for TP                   │
   │   Within row:     IB pipeline send/recv for PP              │
   │   Within column:  IB ReduceScatter / AllGather for DP       │
   └─────────────────────────────────────────────────────────────┘
```

**[STAFF SIGNAL: communication topology awareness]** One critical detail. The 62 nodes in each PP-stage column have to all-reduce gradients together every step. If those 62 nodes are randomly scattered across the cluster, every reduce-scatter hits the spine switch — that's the bottleneck, the part of the network everyone shares. The scheduler has to place each DP group on the same IB rail, or at least minimize spine hops. And the PP-axis send/recv has to use a different rail to avoid contention. Topology-aware placement isn't optional; without it, you're trying to drink the ocean through a coffee straw.

---

## 5. Checkpointing (6 min)

Eight terabytes of state. Question is: how often, what format, where does it go.

### Cadence

The math is a tradeoff between two losses:

```
Wasted work per failure   =  (checkpoint interval) / 2  ← average rewind on resume
Stall time per checkpoint =  ~10 sec  ← device → pinned host RAM, blocking
```

Total cost over the run:

```
total_loss = (interval/2) × N_failures + 10s × (run_duration / interval)
```

Optimize:
```
interval* ≈ √(2 × stall × run_duration / N_failures)
         = √(2 × 10 × 6.2×10⁶ / 138)
         ≈ 950 sec ≈ 16 min
```

**[STAFF SIGNAL: checkpoint cadence math]** Optimum is around 16 minutes. I'll commit to **30 minutes** for operational simplicity — adds less than 1% over the optimum, and round numbers are easier to reason about. At 30-min cadence:

```
Wasted on rewind:  138 × 15 min  =  34.5 hr  =  2.0% of run
Sync stalls:       3,456 × 10 sec =  9.6 hr  =  0.55% of run
                                  ─────────────
Total:                                ~2.5%
```

### What "10-second stall" actually means

The checkpoint isn't blocking for the full upload. It's blocking only for a **device-to-pinned-host-RAM copy**, which is a parallel operation across all 3,968 GPUs. Each GPU copies its ~2 GB shard at NVLink bandwidth. That takes about 10 seconds. After the copy, training resumes, and the upload to durable storage happens in the background.

```
            Async tiered checkpoint flow

Train step N completes
        │
        ├──► All 3,968 ranks: GPU → pinned host RAM
        │    (~10 sec stall, blocking, parallel)
        │
        ├──► Training resumes step N+1  ←──── critical path released
        │
        └──► Background uploader (per node):
                │
                ├──► Tier 1 (immediate): peer DRAM replica
                │    Each node's shard mirrored to a "buddy" node.
                │    Recovery from here: ~30 sec.
                │
                ├──► Tier 2 (~30 sec): local NVMe on each node.
                │    Survives DRAM loss. Recovery: ~2 min.
                │
                └──► Tier 3 (~5 min): object store / Lustre.
                     Survives rack failure. Recovery: ~5-10 min.

Atomic commit: manifest pointer updated only after all shards written.
A partial checkpoint never becomes "current."
```

**[STAFF SIGNAL: rejected alternative]** Naive `torch.save` on rank 0 would block all 3,968 GPUs for hours. Naively sharded `torch.save` per rank gives you 3,968 files with no consistent format and no recovery story. We use **PyTorch Distributed Checkpoint (DCP)** or a thin layer on top, where each rank writes its `(TP_rank, PP_rank, DP_rank)`-keyed shard, and a manifest ties them together. The manifest enables resharding too — useful if we need to resume on a slightly different cluster shape after a fabric incident.

**[STAFF SIGNAL: invariant-based thinking]** The invariant is: at any wall-clock time, there exists a globally consistent checkpoint at some training step S, durable to at least Tier 2, such that loading it on any 3,968-GPU subset reproduces the exact training state at step S. Atomic manifest commits enforce this — the manifest is updated only after all shards have written, so a partial write never becomes the new "current" checkpoint.

---

## 6. Failure Recovery (7 min)

A node fails. Walk through what happens.

```
                  Failure-recovery timeline (target: < 10 min)

  T+0:00    Node N123, GPU 4 throws an XID 79 (uncorrectable ECC).
            NCCL on that GPU raises an async error.
              ├──► Other ranks in the same TP group: NCCL collective hangs.
              │    Their watchdog fires after the timeout.
              └──► Other DP ranks: gradient reduce-scatter hangs.
                   Their watchdogs fire.

  T+0:30    Watchdogs time out (tuned to 30s, default is 5min).
            Each surviving rank logs the error, calls
            torch.distributed.destroy_process_group(), exits.

  T+0:45    Orchestrator (a privileged sidecar that supervises the job)
            sees ranks exiting, looks at the XID logs:
              ├──► Diagnoses: GPU on N123 is the actual cause.
              ├──► Quarantines N123 for repair.
              └──► Picks a hot-spare node from the pool.

  T+1:30    Hot-spare promotion:
              ├──► Spare node was running NCCL in a "shadow" world,
              │    so its IB endpoints were already in everyone's ARP cache.
              ├──► Orchestrator updates the topology config.
              ├──► Spare adopts N123's TP/PP/DP coordinates.

  T+2:00    All ranks re-init NCCL.
            (Cold init at this scale would be 60-90 sec.
             Pre-warming brings it to ~30 sec.)

  T+3:00    Tier 1 / Tier 2 checkpoint reads begin in parallel.
            Each rank reads its local shard from peer DRAM or local NVMe.
            ~2 min for 8 TB across 3,968 ranks.

  T+5:30    All ranks at consistent step S. Optimizer state, RNG state,
            dataloader state restored.

  T+5:45    Resume training from step S.

  T+10:00   Steady state confirmed: loss curve continues from pre-failure
            trend, MFU within 2% of baseline.
  ─────────────────────────────────────────────────────────────────
  Total: ~6 min to resume, ~10 min to verify steady state.
  Across 138 events × 10 min = 23 hr = 1.3% of run.
```

### Specific failure modes

**[STAFF SIGNAL: failure mode precision]**

1. **Hard GPU failure (XID 79)** — the timeline above. Most common case.

2. **NCCL hang without explicit error** — a deadlock without an underlying CUDA error. The watchdog catches it via timeout. Pathological case: a flaky link causes intermittent slowdowns that don't trip the watchdog but slow the run by 30%. Detection: per-step time anomaly monitoring. Response: flag the node, drain it during the next checkpoint window.

3. **CUDA OOM at hour 200** — usually activation memory fragmentation, not a real memory leak. Defenses: explicit memory limits, `expandable_segments` allocator, periodic `torch.cuda.empty_cache()` between epoch boundaries. If it happens, restart from checkpoint with memory profiling on.

4. **Rack failure (32 GPUs lost simultaneously)** — exhausts the hot-spare pool exactly. Response: pause the run, request capacity from the scheduler, restart when ≥3,968 GPUs are available. The state is durable in Tier 3, so the run survives the rack loss; it just pauses.

5. **Object-store outage during upload** — Tier 1 (peer DRAM) and Tier 2 (NVMe) protect against this. The run continues; the uploader retries until durable. Risk window is the time between Tier 2 commit and Tier 3 commit, about 5 minutes.

**[STAFF SIGNAL: rejected alternative]** Why not "elastic training" — continuing with fewer GPUs while we wait for replacement? Because changing the world size mid-run changes the global batch size, which changes the optimizer dynamics. The loss curve shifts. For short SFT runs that's fine. For a 72-day pretrain, the cost of a non-bit-reproducible perturbation is too high — you can't tell after the fact whether a downstream eval regression came from elastic training or from a real bug. Pause-and-resume is the right answer.

---

## 7. Silent Data Corruption (6 min)

**[STAFF SIGNAL: SDC as first-class]** This is the failure mode that ends careers, and the one most candidates wave away.

The setup: a GPU returns subtly wrong values for a matmul. No crash. No CUDA error. Nothing in the logs. The model continues training, gradient norms look fine on aggregate, loss looks fine. But the affected DP replica is contributing poisoned gradients into the all-reduce. Two days later loss starts diverging — or worse, doesn't, and you ship a degraded model.

Meta's "Detecting silent data corruptions in the wild" paper, the Llama 3 paper, and Google's published SDC studies all confirm this: it's not theoretical. Rates are typically 1 SDC per 10⁹–10¹⁰ FLOPs at fleet scale. On a 4,000-GPU 72-day run, **we should expect at least one SDC event**. Possibly many.

### Detection — three layers

```
       SDC detection topology

  Layer 1 — per step (essentially free):
      DP replica gradients → reduce-scatter → gradient norm tracker
                                                     │
                                                     v
                                       z-score against sliding window
                                                     │
                                                     v
                                       alert if > 3σ for 3+ steps
                                                     │
                                                     v
                                  flag suspect DP replica's stages

  Layer 2 — every ~1000 steps (~0.1% overhead):
      For each PP stage, pick two TP groups in the same physical node:
          run identical forward pass on both → bit-compare outputs
                                                     │
                                                     v
                                            mismatch = SDC

  Layer 3 — per checkpoint (free):
      Each shard → CRC → stored in manifest
      On reload: CRC mismatch = corruption
```

**Layer 1** uses the fact that all DP replicas should produce gradients with very similar norms (they differ only by data, which averages out). A replica with consistently outlier norm is suspect. False positive rate is high without smoothing — z-score over a sliding window damps it.

**Layer 2** is the gold-standard check. Two GPUs in the same node, same TP group, run the same forward pass on the same micro-batch. The results should be bit-exact. If they're not, one of them is corrupting. The cost is about 0.1% of compute if you do it once every 1000 steps.

**Layer 3** is free — CRC the shards on write, verify on read. Catches corruption in storage, not in compute, but you might as well.

### Response when detected

1. Quarantine the suspect node.
2. Use the gradient-norm history to estimate **when** the corruption started — the SDC window.
3. Restore from a checkpoint **before** the SDC window, not the most recent one.
4. Replace the node from the hot-spare pool.
5. Resume.

The reason the cadence math has to bound the rewind cost is partly so that "rewind to before the SDC window" doesn't lose a week of training.

---

## 8. Dataloader (4 min)

**[STAFF SIGNAL: dataloader as first-class]** Sounds like a footnote, isn't. If the dataloader stalls for 30 seconds, that's 3,968 H100s sitting idle at roughly $2/sec. No excuse.

### Architecture

Petabytes of tokenized data live in object store as ~100 MB shards. Each DP rank streams its assigned shards. The shard assignment is deterministic from `(epoch, dp_rank, global_seed)`. Within a shard, samples stream in order. Cross-shard shuffle is achieved by interleaving multiple shards per rank.

```
   Object store (S3/Ceph) — petabytes of tokenized shards
         │
         v
   ┌──────────────────────────────────────────────────────┐
   │ Streaming layer — per-DP-rank shard assignment       │
   │ shard_id = hash(epoch, dp_rank, global_seed) mod N   │
   └──────────────────────────────────────────────────────┘
         │
         v
   ┌──────────────────────────────────────────────────────┐
   │ Per-rank prefetch: 4-8 shards in flight,             │
   │ decompressed in background threads,                  │
   │ into pinned host memory                              │
   └──────────────────────────────────────────────────────┘
         │
         v
   ┌──────────────────────────────────────────────────────┐
   │ Sequence packing: pack samples to fill 8K context    │
   │ (cross-document attention masked)                    │
   └──────────────────────────────────────────────────────┘
         │
         v
       Train step

   Checkpointed state per rank:
     - current_shard_id
     - byte_offset_within_shard  
     - sample_counter (for global_step alignment)
```

### The determinism contract

Given `(global_seed, current_step, dp_rank)`, the dataloader can reproduce **exactly which sample** is being served. This is a hard invariant. Without it: SDC detection's "compare to a clean baseline" loses statistical power; resume becomes non-deterministic; debugging a numerical issue becomes impossible because you can't replay the exact data. The dataloader checkpoints alongside the model — current shard, byte offset, sample counter.

### Failure modes (all of these I've seen in production)

- **Shard corruption** — bad bytes crash the tokenizer. Defense: per-shard checksum on read, fail-fast with rank-id and shard-id logged so we know exactly which shard to regenerate.
- **Slow storage** — object store throttling. Defense: aggressive prefetching, multi-tier cache (warm shards on local NVMe), SLA-monitored backend.
- **Resume off-by-N** — the dataloader resumes but is now serving samples already trained on, or skipping samples. *The* most common subtle bug. Defense: index by `(global_step, rank)`, not "count of samples yielded so far." Test it explicitly with a "checkpoint, kill, resume, verify next batch matches" integration test, every CI run.
- **Memory pressure** — prefetch buffer competes with PyTorch for host memory. Defense: hard cap on dataloader, pinned-memory allocation only.

---

## 9. Communication and Topology (3 min)

**[STAFF SIGNAL: communication topology awareness]** I touched on this already. Quick deeper pass.

The IB fabric is a fat-tree or Dragonfly+ Clos with rail-optimized topology. Each H100 node has 8 NICs, one per GPU, each on a separate rail. Same-rail traffic stays local; cross-rail traffic hits the spine.

**Collective patterns:**
- **TP collectives** — intra-node, NVLink. Ring all-reduce, ~5 GB/s effective per GPU pair. Free.
- **DP gradient reduce-scatter** — 62 nodes, inter-rack. Hierarchical: within rail first, then across rail. NCCL 2.18+ does this with `NCCL_ALGO=Tree,Ring`.
- **PP send/recv** — point-to-point, latency-tolerant, easy to overlap with compute.

**SHARP** (in-network reduction on Quantum-2 InfiniBand) does the AllReduce aggregation in switch hardware. Gain is roughly 30–50% on AllReduce at this scale. Worth enabling.

**Gradient bucketing for overlap** — group gradients into ~25 MB buckets, start the reduce-scatter as soon as a bucket is full during backward. Tune bucket size: smaller = better overlap with compute but more collective overhead. ~25-50 MB is the sweet spot.

**Failure mode: one slow link.** A single degraded IB link with CRC errors silently slows every collective it participates in. Detection: per-step time anomaly + per-link counters. NCCL has hooks for per-link timing. Response: route around (NCCL ring re-formation), or drain the node on that rail.

**[STAFF SIGNAL: invariant-based thinking]** The operational invariant is **MFU ≥ 50%**. Any sustained drop below 45% means communication is leaking. That's the alarm metric.

---

## 10. Loss Spikes and Stability (3 min)

Loss diverges at hour 200. Was it the data, optimizer, hardware, numerics, or architecture?

**Detection:**
- Real-time loss z-score over a 1000-step window
- Pre-clip gradient norm (a spike here precedes loss divergence by tens of steps)
- Post-optimizer update norm (catches optimizer pathologies)
- Per-layer activation norm (catches numerical issues like attention-logit blowup)

**Response — human-in-the-loop:**
1. **Pause** the run. Don't kill. The state is intact.
2. Investigate. Is it correlated with a specific data window? A specific replica (suggests SDC)? A specific layer (architecture issue)?
3. **If data:** skip the window, resume from pre-spike checkpoint, log the offending shard.
4. **If SDC:** see SDC response above.
5. **If architecture/numerics:** rewind to pre-spike checkpoint, restart with reduced LR or tighter gradient clip.
6. **If unclear:** rewind further, more conservative restart.

**[STAFF SIGNAL: modern awareness]** Architectural choices that mitigate spikes — gradient clipping (global norm ~1.0), LR warmup (~2000 steps), careful initialization (μP or scaled init), loss scaling for FP8 (delayed scaling on Transformer Engine), z-loss for output logits. References: the BLOOM paper's exhaustive instability documentation, the OPT-175B logbook, Llama 3's spikes section.

The cost of a missed spike is days of training. The cost of being too aggressive in pausing is also days of training. The judgment is what makes this **a human in the loop**, not full automation. The system flags; humans decide.

---

## 11. Developer Ergonomics (3 min)

**[STAFF SIGNAL: developer ergonomics]** The infrastructure exists to serve research engineers. If only the on-call SRE can launch a job, the system is broken regardless of how good the failure recovery is. This is the section interviewers who've actually run these systems care about most.

**Submission interface** — config-as-code (YAML or Python dataclasses) with reproducibility metadata baked in:
- Git SHA of the training repo
- Container image digest
- Dataset version and shard manifest hash
- Random seed
- All hyperparameters

`train submit config.yaml`. The submission system validates the schema, reserves the GPU slice (gang-scheduled — all-or-nothing), pre-warms the hot spares, writes a run record to experiment tracking.

**Live observability:**
- **MFU as a first-class metric.** Not GPU utilization. Utilization can be 100% while doing useless work; MFU measures actual training throughput as a fraction of peak.
- Per-replica loss curves (catches SDC).
- Per-rank step time (catches stragglers).
- Gradient and activation norms.
- Per-link IB counters.
- GPU memory headroom (alarm at < 5 GB free).
- Distributed traces for slow steps.

**Debugging surface:**
- Attach a debugger to one rank without disturbing others.
- Shadow-forward: run the same forward on rank 0 and rank N, bit-compare. Used for SDC investigation and numerical debugging.
- Step-level loss reproduction from any checkpoint.

**Experiment tracking** — W&B or internal equivalent, integrated end-to-end. Every run, every metric, every config. Searchable. This is how research engineers find prior runs to compare against.

---

## 12. FP8 and Mixed Precision (2 min)

**[STAFF SIGNAL: modern awareness]** FP8 on H100 with Transformer Engine is real, deployed at frontier scale, and worth the effort:

- All linear-layer matmuls in FP8: E4M3 forward, E5M2 backward.
- LayerNorm, attention softmax, optimizer math, loss in BF16/FP32.
- Per-tensor scaling with delayed scaling — maintain a window of recent amax values rather than computing amax on every tensor.
- Calibration phase at the start of training to establish scaling factors.

Quantification: ~1.4–1.6× wall-clock speedup at 500B scale because matmuls dominate. Cost: numerical care in calibration, monitoring tensor-statistic drift, fallback to BF16 on layers that don't tolerate FP8.

The format choice matters for numerics. FP8 E4M3 has a ULP — the smallest representable difference at unity — of 0.125, versus BF16's 0.0078. That's a 16× resolution drop. Acceptable for matmul intermediates because rounding noise is dominated by the FP32 accumulator. Not acceptable for accumulation itself or for sensitive ops like LayerNorm statistics or the optimizer's second moment.

MXFP4/MXFP8 on Blackwell extends this with per-block scaling. Different framework integration; same architectural shape.

---

## 13. Tradeoffs and What Would Change the Design (2 min)

- **MoE instead of dense** — expert parallelism enters; communication shifts from AllReduce to all-to-all (more punishing on IB); routing imbalance becomes a top-2 design pressure.
- **Long context (1M tokens)** — context parallelism becomes mandatory; FlashAttention with sequence parallelism (Ring Attention) is the implementation.
- **GB200 NVL72** — TP=64 inside one NVLink domain; PP can be eliminated; weights and optimizer state fit; whole architecture rethought.
- **Smaller cluster (1,000 H100s)** — pipeline depth grows; ZeRO-3 might re-enter as the only way to fit the model.
- **RLHF instead of pretrain** — rollout serving + reward model + actor + critic + reference model = a 4-or-5-model coordination problem; a fundamentally different system, not a parallelism re-config.

---

## 14. What I'd Push Back On (2 min)

**[STAFF SIGNAL: saying no]**

1. **"Single node failure should not kill the run"** — that's the floor. The bar is rack-level survivability with bounded recovery time.
2. **"3-week run"** — wrong frame. The run ends when the token budget is hit. At 4,000 H100s with a frontier token budget, that's ~70 days. Designing for 3 weeks misses the failure-rate scaling.
3. **"4,000 GPUs is the cluster size"** — derived from what target? The right cluster size falls out of (token budget × MFU × wall-clock target), not the other way around.
4. **Developer ergonomics framed as a separate concern** — it's central. Bad ergonomics → research velocity collapses → cluster TCO is wasted on a system nobody can iterate on.
5. **The implicit assumption that this is one job** — in reality the cluster runs production training, debug runs, ablations, and evals concurrently. The infra has to multiplex; this design is for the dominant tenant.

---

## Summary (1 min)

To recap: 3D parallelism layout with TP=8 × PP=8 × DP=62 plus ZeRO-1, interleaved 1F1B pipelining at v=2, async tiered checkpointing at 30-min cadence with peer-DRAM / NVMe / object-store layers, hot-spare nodes pre-warmed in a NCCL shadow world for sub-10-minute recovery, layered SDC detection through gradient-norm anomaly plus periodic deterministic recompute plus checkpoint CRCs, a deterministic streaming dataloader with checkpointed state, hierarchical NCCL collectives with SHARP, FP8 GEMMs through Transformer Engine, observability centered on MFU as the operational invariant.

Total checkpoint and failure overhead is budgeted at ~3.3% of the 72-day run, the architecture survives any single GPU, node, or rack failure with the run resuming automatically, and the most likely thing to actually break the run — silent data corruption — is handled as a first-class concern rather than an afterthought.

That's the design.