---
title: Distributed Training Q&A
description: Distributed Training Q&A
---


Topics that separate senior from staff: zero-bubble PP / DualPipe, grad accumulation × FSDP, ZeRO variants, MoE failure modes. Answers assume H100, Llama-class dense or DeepSeek-class MoE.

---

**Q1. Explain zero-bubble pipeline parallelism. What's the key insight?**

The insight is that **backward decomposes into two independent computations**: `B` = input gradient (`dL/dx_in = dL/dx_out · Wᵀ`) and `W` = weight gradient (`dL/dW = x_inᵀ · dL/dx_out`). Standard 1F1B treats backward as monolithic, so stage `i` can't start `B_{k}` until stage `i+1` finishes its full backward and sends the gradient. But `B` only needs `dL/dx_out` and `W` from the next stage — the `W` computation on stage `i+1` can be **deferred to fill bubbles later**.

ZB-H1: split B and W, schedule W into the warmup/cooldown bubbles. Cuts bubble ~50%. ZB-H2: also splits the optimizer step's dependency, achieves *true* zero bubble at the cost of one extra microbatch's worth of activation memory. The non-trivial part is correctness — `W` must execute before the optimizer step but can otherwise float, so the scheduler does dependency tracking on a DAG, not a static schedule.

The catch nobody mentions: ZB-H2 needs ~2× more in-flight activation memory than 1F1B because more microbatches are live simultaneously. At long context this can flip you from "PP-bubble-bound" to "activation-OOM" — the win isn't free.

---

**Q2. Walk through DualPipe specifically. Why did DeepSeek-V3 build this instead of using ZB-H2?**

DualPipe runs **two pipelines in opposite directions simultaneously** through the same set of devices. Stream A flows forward stage 0→15 while Stream B flows 15→0; their bubbles interleave. Each device sees both streams' microbatches and the idle slots of one fill with compute of the other. End result: ~0% bubble at `M = P` (vs ZB-H2 needing `M ≥ 2P`).

Why V3 didn't just use ZB-H2: V3 is MoE, and the dominant comm cost isn't PP send/recv — it's **EP all-to-all** for token dispatch/combine. DualPipe was designed to overlap *that* comm with compute. The schedule explicitly places stream A's all-to-all-dispatch inside stream B's MLP compute window, and vice versa. ZB-H2 doesn't address this — it eliminates PP bubbles but leaves all-to-all on the critical path.

The trade is brutal scheduling complexity (DeepSeek released custom CUDA kernels for the comm-compute overlap) and 2× pipeline state in memory. For a 671B MoE at 16K context they had the HBM headroom because each rank holds only sharded params; for a dense 405B at 128K you wouldn't.

---

**Q3. How does gradient accumulation interact with FSDP/ZeRO-3? What's the trap?**

The trap is that naive grad accumulation **defeats ZeRO-3's memory savings**. FSDP's normal step: all-gather params → forward → drop → all-gather → backward → reduce-scatter grads → drop. The RS sharding is what keeps grad memory at `2ψ/N`. If you accumulate across micro-steps without RS-ing, you have to hold the *full unsharded* gradient on each rank for `N_accum` steps — back to `2ψ` memory, ZeRO-1 territory.

Two correct approaches:

1. **RS every micro-step, accumulate sharded grads locally.** Comm cost: `N_accum ×` reduce-scatter. Memory stays at `2ψ/N`. Default in PyTorch FSDP when not using `no_sync()`.
2. **`no_sync()` context for the first `N-1` micro-steps**, full sync on the last. Saves comm but materializes full grad — only viable if you have memory headroom (i.e., model wasn't really at the ZeRO-3 limit).

The non-obvious gotcha: option 1 with small per-microstep batches makes the RS comm small and *latency-bound*, not bandwidth-bound. NCCL's all-reduce/RS has fixed overhead per call (~10-50μs); doing 8 RSs of 100MB each is slower than 1 RS of 800MB. At very high accumulation counts you can hit a comm-latency wall that doesn't show up in any bandwidth-based simulator. Fix is **HSDP** — shard within node, replicate across nodes — which lets you use the larger node-level RS and skip per-microstep cross-node comm.

---

**Q4. Compare ZeRO-1 vs 2 vs 3 with concrete memory math for Llama 70B, mixed precision Adam.**

State per param: 2 bytes bf16 weight + 2 bytes bf16 grad + 4 bytes fp32 master + 4+4 bytes Adam m,v = **16 bytes/param** (`Kψ` with K=16, though many sources quote K=12 by excluding the master copy).

70B params → 1.12 TB total state. With `N=64` DP ranks:

| Variant | Sharded                     | Per-rank memory                  | Comm per step       |
|---|---|---|---|
| DP      | nothing                     | 1.12 TB ❌                       | AR(grad) ≈ 140 GB    |
| ZeRO-1  | optimizer states (12 of 16) | 280 GB (params+grad) + 13 GB ≈ 293 GB ❌ | AR(grad) ≈ 140 GB |
| ZeRO-2  | optimizer + grads (14 of 16)| 140 GB (params) + 16 GB ≈ 156 GB ❌ | RS(grad) ≈ 140 GB |
| ZeRO-3  | everything                  | **17.5 GB ✓**                    | AG+RS+AG ≈ 210 GB |

Only ZeRO-3 fits H100's 80 GB at this scale. ZeRO-1/2 are useless for 70B+ at typical DP sizes — they exist for smaller models where the comm savings (1× vs 1.5×) matter and memory isn't the binding constraint.

Practical rule: **ZeRO-1 if `model_state / N_DP` already fits, ZeRO-3 otherwise. ZeRO-2 is dead** — the memory savings over ZeRO-1 (`14/16 → 16/16` of state sharded) don't justify the engineering complexity, and if you need more savings you go to ZeRO-3 anyway.

---

**Q5. Your MoE training is at 35% MFU and you suspect EP imbalance. How do you diagnose and fix?**

Diagnostic chain:

1. Log **per-expert token count** per step. If max/min ratio > 2× consistently, you have routing imbalance, not just stochastic noise. Healthy looks like 1.1-1.3×.
2. Check **drop rate** (`tokens_dropped / total_tokens` per layer). Above 1% is a real signal loss.
3. NSight trace the all-to-all: if dispatch finishes 5ms after combine starts on some ranks but immediately on others, you have **rank-level imbalance** — the slow ranks hold popular experts and stall the all-to-all.

Fixes ordered by impact:

1. **Increase auxiliary load-balancing loss weight** (Switch Transformer style: `α · N_experts · Σ f_i · P_i` where `f_i` is fraction routed to expert `i`, `P_i` is mean gate prob). Typical α=0.01; bump to 0.1 if imbalance is severe.
2. **Increase capacity factor** from 1.25 to 1.5 or 2.0. Pads expert buffers so popular experts don't overflow. Wastes compute on padding but eliminates drops.
3. **Expert placement reshuffle**: spread popular experts across ranks. Requires knowing which experts are popular, which is data-dependent — usually done via offline analysis after a few thousand steps.
4. **Switch to aux-loss-free balancing** (DeepSeek-V3 style, see Q7).

The thing senior engineers miss: drops are only one failure mode. Even with zero drops, **rank-level imbalance directly costs MFU** because all-to-all is bulk-synchronous — the slowest rank determines the step time. Per-expert metrics tell you about drops; per-rank metrics tell you about MFU.

---

**Q6. Token dropping vs expert-choice routing — trade-offs?**

**Token-choice with drop** (Switch, Mixtral, DeepSeek): each token picks top-k experts. If an expert exceeds capacity, excess tokens are dropped (residual passes through). Pros: causal-friendly, simple to implement, well-understood. Cons: drops are signal loss, balance depends on aux loss working.

**Expert-choice** (Zhou et al. 2022): each *expert* picks top-k tokens. By construction, every expert gets exactly `capacity` tokens — perfect balance, zero drops. Cons: **breaks causality** because expert needs to see all tokens to rank them, which is fine in encoder-only or training (you have the full sequence) but fails for autoregressive *inference* — the expert can't rank token `t` against token `t+1` that doesn't exist yet.

Production reality: training-only models (BERT-style) can use expert-choice. Decoder-only LLMs use token-choice. There's no shipped autoregressive model using expert-choice, despite the perfect-balance pitch — inference incompatibility kills it.

Hybrid attempts exist (BASE layers, hash routing) but none have stuck. The frontier is making token-choice's balance better via aux-loss-free methods rather than abandoning it.

---

**Q7. How does DeepSeek-V3's auxiliary-loss-free load balancing work, and why is it better?**

V3 adds a **per-expert bias** `b_i` to the gating logits before top-k selection: `s'_i = s_i + b_i`. The bias is *not* learned via backprop — it's updated heuristically each step: `b_i ← b_i - γ · sign(load_i - target)`. Overloaded experts get their bias pushed down, underloaded ones up. Top-k uses `s'`, but the gate value used for weighting outputs uses raw `s_i` — so the bias only affects routing, not the model's gradient flow.

Why better than aux loss:

1. **Aux loss perturbs the main loss gradient** — there's a documented quality regression at strong aux-loss weights. Bias update is gradient-free, no perturbation.
2. **Faster convergence to balance** — aux loss has to fight the model's natural routing preferences; bias directly counters them by construction.
3. **Tunable separately** — γ is a routing-specific knob, doesn't trade against task loss.

The catch: bias updates are non-differentiable, so stability depends on γ being well-tuned. V3 uses a tiny γ (0.001 in their paper) and the bias drifts slowly. Too large and you get oscillation; too small and balance never converges. It's a hyperparameter that requires tuning per-model — but it's one knob, vs aux-loss which has both weight and form to tune.

---

**Q8. What's the comm cost of MoE all-to-all and how do you hide it?**

Two all-to-alls per MoE layer: dispatch (tokens → expert holders) and combine (expert outputs → token sources). Volume per rank per all-to-all: `b · s · top_k · d / N_EP` — for V3 (`top_k=8, d=7168, s=4K, b=1, N_EP=64`): `1 · 4096 · 8 · 7168 / 64 = 3.7 MB` per rank per direction. Sounds small, but × 58 MoE layers × 2 directions = 430 MB per rank per step on the critical path.

Hiding strategies:

1. **Compute-comm overlap within the layer**: dispatch happens after gate computation, combine happens before residual add. Dispatch can overlap with the gate softmax/top-k of the *next* token chunk if you tile the sequence. V3's custom kernel does this.
2. **DualPipe-style cross-stream overlap**: schedule stream A's dispatch during stream B's expert matmul (Q2). This is the dominant V3 win — it makes all-to-all effectively free.
3. **Topology-aware all-to-all**: NCCL's default all-to-all is naive. For EP within a node, use NVLink-aware kernels (NVSHMEM-based). Cross-node, use hierarchical: node-internal all-to-all first, then single cross-node send per node-pair.
4. **Reduce all-to-all volume**: top-k=8 is V3's choice, top-k=2 is Mixtral's. Linear in `top_k`. But top-k affects model quality, so this is a model-design knob, not a systems one.

The thing to flag in an interview: all-to-all is *latency-bound* at small messages and *bandwidth-bound* at large ones, and the crossover is ~1MB on H100. V3 sits right at the crossover, which is why kernel choice matters so much. Below 1MB you're paying NCCL launch overhead per call; above 1MB you're paying for actual bytes. Optimization strategy depends on which side you're on.