---
title: Performance & Debugging
description: Performance & Debugging
---

Target: 15 min/scenario, 20 min hard stop. 8 scenarios, ~2.5 hours total.
Run in two sessions of 4 if doing in one day.
 
---
 
## Scenario 1 — Slow SGEMM kernel (warmup)
 
**Setup:** You wrote a custom FP16 GEMM kernel for M=N=K=4096 on an H100. Achieving 180 TFLOPS. cuBLAS gets 650 TFLOPS on the same shape. Here's the Nsight Compute summary:
 
```
SM Busy:                      94%
Tensor Core Utilization:      12%
DRAM Throughput:              45% of peak
L1/TEX Hit Rate:              62%
Warp Stall (Long Scoreboard): 38%
Warp Stall (MIO Throttle):    21%
Achieved Occupancy:           38%
Register usage:               168 per thread
Shared mem per block:         48 KB
Block size:                   256 threads
```
 
**Target diagnosis:** Tensor cores barely used → kernel is likely doing FMA on CUDA cores, not `mma.sync` / wgmma. High register pressure capping occupancy. Long scoreboard stalls = waiting on global loads, suggests no software pipelining / insufficient async copy overlap.
 
**Target fix ordering:**
1. Move to tensor core instructions (wgmma for Hopper, or CUTLASS/CuTe path)
2. Add async copy (`cp.async.bulk` / TMA on Hopper) with multi-stage pipeline
3. Reduce register pressure — smaller per-thread tile or fewer live accumulators
4. Swizzled shared memory layout to kill bank conflicts feeding tensor cores
**Failure mode to avoid:** jumping to "increase occupancy" as primary fix. Occupancy is a symptom here, not the bottleneck. Tensor core underuse is the 10x factor.
 
---
 
## Scenario 2 — MFU collapse mid-training
 
**Setup:** 70B dense model, 512 H100s, TP=8, PP=4, DP=16. MFU held at 48% for 3 hours, then dropped to 22% and stayed there. No OOM, no crash. Loss curve looks normal. What do you check?
 
**Target diagnosis path (in this order):**
1. **Straggler / node health first.** NCCL all-reduce times per rank — one slow node drags the whole collective. Check `nvidia-smi` for throttling (thermal, power), ECC errors, PCIe downgrade.
2. **Dataloader stall.** If step time increased but GPU busy time didn't, it's host-side. Check dataloader worker count, prefetch queue depth, disk/network read latency if streaming.
3. **Checkpoint activity.** Async checkpoint hanging on slow storage can serialize ranks.
4. **Network fabric.** InfiniBand link flap, congestion, one rail down forcing fallback.
**Target fix:** identify stragglers via per-rank timing histograms, drain and reschedule. The boring answer (one node degraded) is almost always right for "sudden MFU drop with no code change."
 
**Failure mode to avoid:** proposing code-level fixes (different sharding, recompute policy) before checking hardware. Nothing in the code changed.
 
---
 
## Scenario 3 — Inference throughput collapse under load
 
**Setup:** vLLM serving Llama-70B FP8 on 8×H100. At QPS=4 you get 180 tok/s/user, p50 TTFT 120ms. At QPS=16, per-user throughput drops to 22 tok/s and TTFT jumps to 4.2s. GPU util shows 99%. What's happening and how do you fix it?
 
**Target diagnosis:**
- At high QPS, continuous batching is admitting too many requests into the decode batch. KV cache pressure → either evictions or huge batch sizes where decode becomes memory-bandwidth bound on KV reads.
- TTFT spike = prefill queue backed up behind decode batches (if not using chunked prefill) or prefill-decode contention for SMs.
- 99% GPU util is misleading — it measures SM busy, not useful work. Decode at batch=64 is HBM-bound, SMs spin waiting.
**Target fixes (rank by impact):**
1. Chunked prefill + prefill-decode disaggregation if not already on
2. Cap max running batch based on KV budget arithmetic, not just request count
3. Speculative decoding to raise arithmetic intensity of decode
4. Tune max_num_seqs / max_num_batched_tokens explicitly
5. Consider MQA/GQA KV sharing if model supports it (Llama-70B has GQA already, so this is bounded)
**Failure mode to avoid:** suggesting "just add more GPUs." Interviewer wants to see you understand the decode-is-memory-bound dynamic.
 
---
 
## Scenario 4 — Numerics regression after FP8 conversion
 
**Setup:** You quantized a model to FP8 E4M3 for weights and activations. Eval loss on validation set is 2.31 vs BF16 baseline 2.28 — acceptable. But on a specific downstream task (long-context retrieval, 32k tokens), accuracy drops from 71% to 54%. Short-context tasks are fine. Diagnose.
 
**Target diagnosis:**
- FP8 E4M3 has max representable ≈ 448. Long context → attention scores over long sequences can have larger dynamic range in softmax inputs (pre-softmax logits) and in accumulated attention outputs.
- Per-tensor scaling calibrated on short sequences underestimates activation range at long context → saturation/overflow in specific layers.
- Check per-layer activation max across sequence length. Likely the attention output projection or late-layer residual stream is saturating.
**Target fixes:**
1. Recalibrate scales with long-context calibration data
2. Move to per-row / per-block scaling (MXFP) for the offending layers
3. Keep attention accumulation in higher precision, only quantize QKV projections and output proj
4. Per-layer sensitivity analysis — your own FP8 adoption playbook from Meta applies here
**Failure mode to avoid:** blaming calibration dataset size without decomposing where the drift actually lives. Show you can localize to specific layers/ops.
 
---
 
## Scenario 5 — OOM at specific sharding configuration
 
**Setup:** Training 405B model. FSDP2 full shard, activation checkpointing on every layer, ZeRO-3 optimizer. Works at TP=8, DP=64. Fails OOM at TP=4, DP=128 even though per-GPU model memory should be *larger* at TP=8 not TP=4. Why?
 
**Target diagnosis:**
- At TP=4, each TP rank holds 2x the weights → 2x the activation tensors per layer before TP-reduce.
- Activation checkpointing saves layer inputs but the *peak* during recompute requires holding full activations for one layer — which are now 2x larger.
- Also: optimizer state sharding is across DP group. DP=128 means each rank holds 1/128th of optimizer state, which is actually *less* than 1/64th. So optimizer isn't the cause.
- Real culprit: activation memory during attention (the QK^T matrix is O(seq_len² × heads_per_rank)). At TP=4 you have 2x heads per rank → 2x attention activation peak.
**Target fixes:**
1. Use sequence parallelism to shard the LayerNorm/dropout activations along seq dim
2. Use FlashAttention (eliminates the seq² materialization entirely — if they're not already using it, that's the headline fix)
3. Reduce microbatch size
4. Increase TP back to 8 if fabric supports it
**Failure mode to avoid:** reaching for "just turn on more activation checkpointing." The interviewer is testing whether you can reason about where memory actually goes under different parallelism configs.
 
---
 
## Scenario 6 — All-reduce latency cliff
 
**Setup:** Gradient all-reduce time per step: 45ms at 64 GPUs, 52ms at 128 GPUs, 61ms at 256 GPUs, 340ms at 512 GPUs. Ring all-reduce. Same fabric, same message size per rank. Why the cliff at 512?
 
**Target diagnosis:**
- Ring all-reduce time = 2(N-1)/N × message_size / bandwidth. Should scale mildly, not cliff.
- Cliff suggests crossing a topology boundary. 512 GPUs = crossing pod / rack / spine switch boundary where inter-pod bandwidth is lower than intra-pod.
- Or: NCCL fell back from NVLink/NVSwitch to IB for part of the ring → heterogeneous bandwidth, slowest segment dominates.
- Or: at 512, ring is long enough that latency × hops dominates bandwidth term.
**Target fixes:**
1. Switch to tree all-reduce or hierarchical (intra-node NVLink all-reduce, then inter-node IB all-reduce)
2. Use NCCL's double binary tree for large scale
3. Overlap all-reduce with backward pass (if not already — check bucket size)
4. Reduce frequency via gradient accumulation
**Failure mode to avoid:** proposing ZeRO/FSDP as a fix — that's a different problem class. This is about collective algorithm choice at scale.
 
---
 
## Scenario 7 — Triton kernel slower than PyTorch baseline
 
**Setup:** You wrote a fused RMSNorm + linear kernel in Triton. Expected ~1.5x over the unfused PyTorch version (eager). Measuring 0.7x — it's slower. Kernel compiles, produces correct output. Here's the autotune config you're using:
 
```python
configs = [
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=2, num_warps=4),
    triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 32}, num_stages=2, num_warps=4),
]
```
 
**Target diagnosis:**
- Autotune space is way too small. Only 2 configs, no variation on num_stages / num_warps.
- Likely missing tensor core path — on H100, small BLOCK_K (32) prevents efficient wgmma use. Want BLOCK_K ≥ 64 for FP16/BF16 tensor cores.
- num_stages=2 is conservative; Hopper benefits from 3-4 stages for async copy overlap.
- num_warps=4 = 128 threads/block, limits available parallelism.
- Also possible: PyTorch eager is calling cuBLAS for the linear, which is already extremely good. "Fused" kernel needs to beat cuBLAS on the GEMM portion, which requires real CUTLASS-level tuning, not a casual Triton fusion.
**Target fixes:**
1. Expand autotune grid: BLOCK_M ∈ {64,128,256}, BLOCK_N ∈ {64,128,256}, BLOCK_K ∈ {64,128}, num_stages ∈ {3,4,5}, num_warps ∈ {4,8}
2. Check SASS — verify wgmma instructions are emitted
3. Consider: is fusion actually beneficial here? If GEMM dominates, fusing a cheap RMSNorm gains little. The right fusion might be RMSNorm + Q/K/V projections, not RMSNorm + single linear.
4. Fall back to CUTLASS epilogue fusion if Triton can't hit the ceiling.
**Failure mode to avoid:** blaming Triton generally. Show you know the specific autotune levers and that fusion ROI depends on arithmetic intensity of the fused ops.
 
---
 
## Scenario 8 — The weird one (pipeline bubble mystery)
 
**Setup:** Pipeline parallel training, PP=8, interleaved 1F1B schedule, 32 microbatches. Expected bubble fraction ≈ (PP-1)/num_microbatches = 7/32 = 22%. Measured bubble: 41%. No stragglers, fabric healthy, no checkpoint activity. Loss is fine. Diagnose.
 
**Target diagnosis (this one rewards methodical elimination):**
- Per-stage timing: are stages balanced? Embedding stage and output/loss stage are notoriously imbalanced (embedding has vocab-size GEMM, loss has cross-entropy over vocab). Imbalance → bubble inflates because faster stages wait for slower ones.
- Recomputation imbalance: if activation checkpointing is applied uniformly but some stages have heavier activations (attention-heavy vs MLP-heavy split), recompute cost varies per stage.
- Send/recv overhead: P2P comm between pipeline stages has fixed latency. If microbatches are small (short seq len), comm cost per microbatch is amortized over less compute → bubble grows.
- Interleaving virtual stages: if `num_virtual_stages` is set suboptimally, interleaved 1F1B can actually have *more* bubble than naive 1F1B for some configs.
**Target fixes:**
1. Rebalance pipeline partitioning — put fewer layers on embedding and output stages, more on middle stages
2. Tune virtual pipeline stages count
3. Increase microbatch count (more microbatches = smaller bubble fraction, classical tradeoff with activation memory)
4. Consider zero-bubble pipeline schedule (ZB-H1/H2) if the framework supports it
**Failure mode to avoid:** taking the theoretical bubble formula at face value. The formula assumes balanced stages and zero comm overhead. Real systems deviate, and staff-level debugging is about knowing where the model breaks.
 
---
 
## Scoring rubric (self or interviewer)
 
| Dimension | Pass bar | Strong pass |
|-----------|----------|-------------|
| Time to initial hypothesis | < 90s | < 45s |
| Asked for the right metric/artifact before guessing | Yes | Yes, and named 2-3 specific ones |
| Root cause correct | Identified correct class | Identified specific mechanism |
| Fix ordered by impact | Mostly | Explicitly ranked with reasoning |
| Anchored on boring explanation first | Yes | Yes, and explained why the exotic ones are lower prior |
| Quantitative reasoning | Back-of-envelope present | Specific numbers (bandwidth, FLOPs, memory) |
 
## Meta-advice for the live round
 
1. **Narrate the decision tree, not the answer.** "Given X symptom, the hypothesis space is A, B, C. A is most likely because Y. To distinguish I'd look at Z." This shows reasoning, not memorization.
2. **Ask for the artifact you need.** If they describe a slow kernel, ask for the Nsight Compute output. If they describe training slowness, ask for per-rank timing. Senior candidates drive the diagnosis; juniors accept whatever was given.
3. **Quantify early.** "70B model, BF16, 140GB weights, 80GB HBM per H100, so we need at least TP=2 before activations..." — doing arithmetic out loud signals the level.
4. **When stuck, state what you'd rule out next and why.** Don't go silent. Interviewers can't calibrate silence.
5. **If you don't know, say so and say what you'd do to find out.** "I haven't debugged this specific failure mode but my approach would be..." — staff candidates are allowed gaps; they're not allowed to bluff.