---
title: "Mixture of Experts: Scaling Parameters Without Scaling Every Token"
description: "A staff-level guide to Mixture of Experts model optimization: sparse routing, active parameters, top-k gating, capacity factor, load balancing, expert parallelism, serving, training, and failure modes."
---

# Mixture of Experts: Scaling Parameters Without Scaling Every Token

Mixture of Experts, or MoE, is a sparse model architecture that increases total model capacity without increasing the computation used by every token proportionally. Instead of sending every token through one huge dense feed-forward network, an MoE layer contains many expert networks and a router chooses a small subset for each token.

The intuition:

> A model can have many parameters available, but each token should only pay for the experts it needs.

This is why MoE is important for optimization. It separates **total parameters** from **active parameters**. A dense 70B model uses roughly all 70B parameters per token. An MoE model might have 140B total parameters but activate only 20B or 30B worth of parameters per token.

That sounds like a free lunch. It is not. MoE trades dense compute for routing, load balancing, communication, expert placement, memory pressure, and training instability. The staff-level understanding is:

> MoE is a way to scale capacity at roughly fixed per-token compute, but only if the router, expert layout, all-to-all communication, and batching system are engineered well.

---

## 1. The Interview Mental Model

When MoE comes up, answer in this order:

1. **What is sparse?** Usually the feed-forward block, not the whole transformer.
2. **What is active per token?** Number of selected experts times expert size.
3. **How does routing work?** Top-1, top-2, top-k, shared experts, fine-grained experts.
4. **How is load balanced?** Auxiliary loss, capacity factor, token dropping, expert bias, routing constraints.
5. **Where are experts placed?** One GPU, tensor parallel, expert parallel, or distributed across nodes.
6. **What is the bottleneck?** Dense expert compute, all-to-all dispatch, memory bandwidth, router imbalance, or small expert batches.
7. **How do we evaluate it?** Quality per active parameter, tokens/sec/GPU, p95 latency, routing stability, and expert utilization.

The shape of an MoE layer:

```text
Token hidden states
        |
        v
Router / gate
        |
        +---- token A -> expert 2, expert 7
        +---- token B -> expert 1, expert 2
        +---- token C -> expert 7, expert 9
        |
        v
Experts run dense FFNs on assigned tokens
        |
        v
Weighted combine back into original token order
```

The phrase to remember:

> MoE makes parameters sparse, but the selected experts are usually dense.

---

## 2. Dense FFN vs MoE FFN

Most transformer compute sits in the attention and feed-forward blocks. A simplified dense transformer block is:

$$
h' = h + \text{Attention}(\text{Norm}(h))
$$

$$
h_{out} = h' + \text{FFN}(\text{Norm}(h'))
$$

The dense FFN is often:

$$
\text{FFN}(x) = W_2 \sigma(W_1 x)
$$

MoE usually replaces the FFN with several experts:

$$
\text{MoE}(x) = \sum_{i \in \text{TopK}(r(x))} g_i(x) E_i(x)
$$

where:

- $r(x)$ is the router score for token representation $x$.
- $E_i$ is expert $i$.
- $g_i(x)$ is the router probability or weight for expert $i$.
- $\text{TopK}$ selects only a few experts.

If there are $N$ experts and each token uses $k$ experts, then the active expert fraction is:

$$
\frac{k}{N}
$$

For $N = 8$ and $k = 2$, each token uses 25% of the experts in that layer.

Important distinction:

```text
Total parameters:  all experts + shared model weights
Active parameters: shared weights + selected experts for one token
```

In MoE discussions, active parameters matter more for inference cost. Total parameters matter more for memory, storage, loading, and capacity.

---

## 3. A Short Historical Context

Mixture-of-experts ideas are old, but the modern sparse transformer lineage is clearer:

- **Sparsely-Gated MoE, 2017.** Shazeer et al.'s **[Outrageously Large Neural Networks](https://arxiv.org/abs/1701.06538)** introduced a sparsely-gated MoE layer that activated only a small number of experts per example.
- **GShard, 2020.** **[GShard](https://arxiv.org/abs/2006.16668)** scaled sparse MoE transformers with automatic sharding and multilingual translation, demonstrating models beyond 600B parameters.
- **Switch Transformer, 2021/2022.** **[Switch Transformers](https://arxiv.org/abs/2101.03961)** simplified routing to top-1 expert selection and scaled to trillion-parameter models, reporting strong compute efficiency.
- **Open and production-era MoEs, 2024 onward.** **[Mixtral 8x7B](https://arxiv.org/abs/2401.04088)** made sparse MoE widely visible in open-weight LLMs. DeepSeekMoE and DeepSeek-V2 explored fine-grained and shared experts. OLMoE released a fully open MoE training stack. DBRX demonstrated another large open MoE design.

The historical trend is from "can sparse conditional computation work?" to "can we make routing, training, and serving predictable enough for real systems?"

---

## 4. Routing and Gating

The router is a small network that maps each token hidden state to expert scores:

$$
s = W_r x
$$

Then a softmax gives routing probabilities:

$$
p_i = \frac{\exp(s_i)}{\sum_j \exp(s_j)}
$$

Top-k routing selects the $k$ highest-probability experts:

$$
\mathcal{E}(x) = \text{TopK}(p, k)
$$

The final MoE output is:

$$
y = \sum_{i \in \mathcal{E}(x)} p_i E_i(x)
$$

Top-1 routing, used by Switch Transformer, chooses one expert:

$$
y = E_{\arg\max_i p_i}(x)
$$

Top-2 routing uses two experts and combines them with router weights. Top-2 often improves quality and routing flexibility, but doubles expert compute relative to top-1.

```text
Top-1:
  token -> router -> expert 5

Top-2:
  token -> router -> expert 5
                  -> expert 12
  output = weighted sum
```

Routing choices are architectural choices:

- **Top-1:** cheaper, simpler, more brittle.
- **Top-2:** better capacity and smoother gradients, more compute.
- **Top-k:** more flexible, expensive if k grows.
- **Shared experts:** always-active experts handle common knowledge.
- **Routed experts:** selected experts specialize.
- **Fine-grained experts:** smaller experts with more routing combinations.

DeepSeekMoE is a useful example of the last two ideas: shared experts for common knowledge plus many fine-grained routed experts for specialization.

---

## 5. Capacity Factor and Token Dropping

Routing creates a load-balancing problem. If every token chooses the same expert, that expert becomes overloaded while others are idle.

For a batch with $T$ tokens, $N$ experts, and top-$k$ routing, the ideal number of token assignments per expert is:

$$
\frac{T k}{N}
$$

MoE systems often define expert capacity:

$$
C = \left\lceil \text{capacity factor} \cdot \frac{T k}{N} \right\rceil
$$

If an expert receives more than $C$ tokens, the overflow tokens may be dropped, rerouted, padded, or handled by a fallback path.

```text
Expert capacity example:

Tokens: 1024
Experts: 16
Top-k: 2
Ideal assignments per expert = 1024 * 2 / 16 = 128

Capacity factor 1.25:
Capacity per expert = ceil(1.25 * 128) = 160
```

Higher capacity factor reduces token dropping but increases padding and wasted compute. Lower capacity factor improves efficiency but risks quality loss from dropped or poorly routed tokens.

This is the MoE version of a systems tradeoff:

```text
Higher capacity factor:
  + fewer dropped tokens
  + better quality
  - more padding
  - more memory / compute waste

Lower capacity factor:
  + better utilization
  + less padding
  - more overflow risk
  - worse quality under imbalance
```

Capacity factor is not just a training hyperparameter. It affects serving behavior, tail latency, and quality.

---

## 6. Load Balancing Loss

Routers can collapse. Without pressure, a few experts may receive most tokens while other experts remain undertrained.

A common solution is an auxiliary load-balancing loss. The exact formula varies, but the goal is to encourage:

- Equal token assignment across experts.
- Router probability mass spread across experts.
- Avoidance of expert collapse.

In Switch-style routing, a simplified load-balancing term uses:

$$
f_i = \frac{\text{tokens assigned to expert } i}{T}
$$

and:

$$
P_i = \frac{1}{T}\sum_{x} p_i(x)
$$

where $f_i$ is actual assignment fraction and $P_i$ is average router probability. A balancing loss can be proportional to:

$$
N \sum_i f_i P_i
$$

The exact implementation details differ, but the concept is stable: penalize routing patterns that overload a small set of experts.

Other balancing techniques:

- Router z-loss for numeric stability.
- Expert-choice routing.
- Token-choice routing with capacity limits.
- Noisy routing.
- Expert bias adjustment.
- Dropless routing with more careful dispatch.
- Shared experts to reduce pressure on routed experts.

Interview phrase:

> The router is part of the model and part of the scheduler. If it collapses, quality and hardware utilization both suffer.

---

## 7. Expert Specialization

MoE is useful because experts can specialize. Specialization may happen by:

- Language.
- Domain.
- Syntax pattern.
- Task type.
- Token frequency.
- Position.
- Code vs natural language.
- Common vs rare knowledge.

But specialization is not guaranteed. Bad routing can create redundant experts or overloaded experts.

Useful diagnostics:

- Expert token counts.
- Expert utilization entropy.
- Top domains per expert.
- Language distribution per expert.
- Average router confidence.
- Expert co-activation matrix.
- Expert dropout sensitivity.
- Per-expert gradient norm.

Example expert utilization table:

```text
Expert  Tokens  Main traffic        Risk
------  ------  ------------------  ----------------
0       7.1%    English chat        healthy
1       6.8%    code                healthy
2       28.4%   everything          overloaded
3       0.3%    rare tokens         undertrained
4       5.9%    math                healthy
...
```

If one expert receives 28% of traffic in a 16-expert layer, the router is likely not balanced enough. If an expert receives 0.3%, it may not learn useful behavior.

Specialization is valuable only when it improves quality or efficiency. Pretty expert visualizations are not enough.

---

## 8. Training MoE Models

MoE training is harder than dense training.

Key issues:

- Router instability.
- Expert collapse.
- Token dropping.
- Load-balancing loss tuning.
- Expert undertraining.
- Communication overhead.
- Gradient variance.
- Mixed precision instability.
- Checkpoint and optimizer-state size.

Training flow:

```text
Token batch
    |
    v
Router computes expert assignments
    |
    v
Tokens are dispatched to experts
    |
    v
Experts compute dense FFNs
    |
    v
Outputs are combined
    |
    v
Auxiliary load-balancing losses added
```

MoE can be more sample-efficient because the model has more capacity at similar active compute. But it can also waste capacity if experts do not specialize or if the router learns poor assignments early.

Important training metrics:

- Main loss.
- Load-balancing loss.
- Router entropy.
- Token drop rate.
- Expert utilization.
- Expert gradient norms.
- All-to-all communication time.
- Tokens/sec/GPU.
- Quality per active parameter.

The training goal is not maximum total parameters. It is better loss and downstream quality at a fixed compute and serving budget.

---

## 9. Expert Parallelism and All-to-All

MoE layers often require expert parallelism: experts are distributed across devices. Tokens must be sent to the GPUs that own the selected experts.

```text
Before routing:

GPU 0: tokens A B C D
GPU 1: tokens E F G H

Experts:

GPU 0: expert 0, expert 1
GPU 1: expert 2, expert 3

After routing:

token A -> expert 3 -> send to GPU 1
token B -> expert 0 -> stay on GPU 0
token E -> expert 1 -> send to GPU 0
token F -> expert 2 -> stay on GPU 1
```

This creates all-to-all communication:

```text
Local tokens
    |
    v
All-to-all dispatch
    |
    v
Expert compute
    |
    v
All-to-all combine
    |
    v
Original token order
```

The communication cost can erase the compute benefit if:

- Experts are spread across slow interconnects.
- Expert batches are too small.
- Routing is imbalanced.
- All-to-all is not overlapped with compute.
- Tokens are frequently routed across nodes.

MoE serving and training are therefore topology-aware. NVLink, InfiniBand, TPU interconnects, and placement strategy matter.

Staff-level point:

> MoE is not only a model architecture. It is a distributed systems architecture.

---

## 10. MoE Inference

MoE inference has different problems from MoE training.

Training cares about throughput and stable learning. Inference cares about:

- Per-request latency.
- Batch formation.
- Expert cache locality.
- Expert placement.
- Memory footprint.
- Active expert batching.
- Tail latency from overloaded experts.

For autoregressive decode, each token may route differently. That makes batching harder:

```text
Decode step t:
  request 1 -> experts 2, 4
  request 2 -> experts 4, 9
  request 3 -> experts 1, 2

Decode step t+1:
  request 1 -> experts 7, 9
  request 2 -> experts 4, 5
  request 3 -> experts 1, 8
```

The serving engine must group token-expert assignments into efficient expert batches. If each expert receives only a few tokens, the dense expert GEMM is small and inefficient.

Inference bottlenecks:

- Routing overhead.
- Small expert batch sizes.
- All-to-all dispatch.
- Expert imbalance.
- Memory pressure from storing all experts.
- KV cache still exists and is not made sparse by MoE.
- Tensor/expert parallel coordination.

MoE can improve quality at fixed active compute, but it does not automatically improve latency. Some MoE models are better understood as **quality-per-FLOP** optimizations rather than low-latency optimizations.

---

## 11. Memory Reality

MoE reduces active compute, not total memory. All experts must exist somewhere.

If a model has:

- shared parameters $P_s$
- $N$ experts
- expert size $P_e$

then total parameters are:

$$
P_{total} = P_s + N P_e
$$

If each token activates $k$ experts, active parameters per token are roughly:

$$
P_{active} = P_s + k P_e
$$

The gap between $P_{total}$ and $P_{active}$ is the MoE advantage. But memory must hold or load $P_{total}$.

That affects:

- GPU memory capacity.
- Checkpoint size.
- Load time.
- Optimizer state during training.
- Expert placement.
- Multi-node serving.
- Fault recovery.

MoE is not magic for small-memory devices unless experts can be offloaded, paged, or distributed without killing latency.

---

## 12. MoE vs Dense Models

MoE is usually attractive when:

- Training compute is limited but parameter capacity helps.
- Multiple domains or languages benefit from specialization.
- Quality per active parameter matters.
- The infrastructure can handle expert parallelism.
- Large batch training or serving can amortize routing overhead.

Dense models are often better when:

- Latency simplicity matters more than peak quality.
- Batch sizes are small.
- Infrastructure cannot handle all-to-all efficiently.
- Memory capacity is tight.
- Predictable behavior is more important than routing specialization.
- The model must run on commodity hardware.

Tradeoff table:

| Dimension | Dense | MoE |
| --- | --- | --- |
| Per-token compute | All parameters active | Only selected experts active |
| Total memory | Lower for same active size | Higher due to inactive experts |
| Training stability | Simpler | Router/load balancing issues |
| Serving | Easier batching | Expert batching and all-to-all |
| Quality per active FLOP | Often lower | Often higher if trained well |
| Tail latency | More predictable | Expert imbalance can hurt |

Interview phrase:

> MoE is not "faster dense." It is sparse conditional capacity with distributed-systems overhead.

---

## 13. Recent Real-World Examples

MoE is common in frontier and open-weight LLMs, but the exact architecture is often partially disclosed.

Examples:

- **Switch Transformer.** Top-1 routing and very large sparse models. Important for simplicity and scale.
- **GShard.** Early large-scale sparse transformer with automatic sharding.
- **Mixtral 8x7B.** Open-weight sparse MoE with 8 experts and 2 active experts per token. It made MoE concrete for many practitioners.
- **DeepSeekMoE / DeepSeek-V2.** Uses shared experts and fine-grained routed experts to improve specialization and efficiency.
- **OLMoE.** A fully open MoE effort with released artifacts, useful for studying routing and training behavior.
- **DBRX.** Large open MoE model from Databricks, another example of MoE as a practical open LLM architecture.

The pattern is clear: MoE is no longer just a research curiosity. It is a standard way to push quality per active compute. But production MoE requires stronger infrastructure than dense model serving.

---

## 14. Failure Modes

### Expert collapse

The router sends too many tokens to a small set of experts. Quality suffers and hardware utilization becomes uneven.

### Undertrained experts

Some experts receive too little traffic and never learn useful behavior.

### Token dropping

Capacity limits overflow. Dropped tokens lose expert computation and quality regresses.

### Communication dominates

All-to-all dispatch takes more time than the saved expert compute.

### Small expert batches

Each expert receives too few tokens, so dense GEMMs are inefficient.

### Memory pressure

Total parameters are large even though active parameters are smaller.

### Router brittleness

Small changes in hidden states or numerics can flip expert assignment, causing behavior drift after quantization or kernel changes.

### Bad evaluation

Benchmarks report total parameter count instead of active parameters, or report quality without serving cost.

---

## 15. Benchmarking MoE

MoE benchmarks should report both model quality and system cost.

Minimum table:

| Metric | Value |
| --- | ---: |
| Total parameters | |
| Active parameters/token | |
| Number of experts | |
| Active experts/token | |
| Expert capacity factor | |
| Token drop rate | |
| Expert utilization entropy | |
| All-to-all time | |
| p50/p95 latency | |
| tokens/sec/GPU | |
| HBM used | |
| Quality score by slice | |

Useful derived metrics:

$$
\text{active fraction} = \frac{P_{active}}{P_{total}}
$$

$$
\text{quality per active parameter} =
\frac{\text{quality score}}{P_{active}}
$$

$$
\text{quality per dollar} =
\frac{\text{quality score}}{\text{serving cost}}
$$

The last one is often the only metric product leaders care about.

---

## 16. MoE Inference Systems in the Wide-EP Era

Sections 9-11 covered the mechanics: expert parallelism, all-to-all, and memory. Since roughly 2025, serving large MoEs (DeepSeek-V3/R1-class models with hundreds of routed experts) has produced a distinct systems playbook. This is what a staff-level answer about *serving* MoE should contain.

### Expert-parallel load balancing (EPLB)

Static expert placement assumes uniform routing. Real traffic does not route uniformly: some experts are hot for a given workload mix, and the GPU that owns a hot expert becomes the straggler for the whole layer. DeepSeek's open-source **[EPLB](https://github.com/deepseek-ai/EPLB)** attacks this with **redundant experts**: heavy-loaded experts are *replicated*, and the replicas are packed across GPUs so that per-GPU expected load is even. The output is a mapping from logical experts to physical expert slots, recomputed periodically from observed routing statistics.

EPLB ships two policies:

- **Hierarchical load balancing.** Used when node count evenly divides expert groups: pack expert groups onto nodes first (exploiting group-limited routing so a token's experts stay on few nodes), then replicate and place within each node. This minimizes inter-node all-to-all traffic. It fits **prefill**, where expert-parallel degree is smaller.
- **Global load balancing.** Replicate experts globally, ignoring groups. It fits **decode**, where EP degree is large and balance matters more than locality.

The systems tradeoff: rebalancing costs weight movement and a brief disruption, and it chases a moving target. Rebalance too often and you pay copy overhead; too rarely and workload shifts (a burst of code traffic, a new tenant) re-create hot experts. vLLM and SGLang both implement EPLB-style rebalancing with a configurable statistics window and rebalance interval — those two knobs are the thing to tune, and per-expert load skew over time is the thing to watch.

### DeepEP-style all-to-all kernels

Generic all-to-all collectives are a poor fit for MoE dispatch/combine: payloads are small, irregular, and latency-critical during decode. **[DeepEP](https://github.com/deepseek-ai/DeepEP)** is the reference open-source library for EP communication and shapes how vLLM/SGLang do wide-EP:

- **Normal (high-throughput) mode** targets training and **prefill**: kernels forward via NVLink within a node and RDMA across nodes, sustaining near-line-rate bandwidth (hundreds of GB/s intranode over NVLink, tens of GB/s per GPU over CX7-class RDMA internode).
- **Low-latency mode** targets **decode**: pure-RDMA dispatch/combine kernels tuned for tiny per-step batches, with a hook-based mechanism that overlaps communication with compute without occupying SMs.
- **FP8 dispatch** with higher-precision combine cuts wire bytes roughly in half where the model tolerates it.

The staff point: EP communication is a specialized kernel problem, not a "call NCCL" problem. If your all-to-all occupies SMs, it steals compute from the experts it is feeding; measure SM occupancy of communication, not just link bandwidth.

### Prefill vs decode want different EP layouts

Disaggregated prefill/decode (see the serving guide) interacts directly with MoE:

- **Prefill** has many tokens per step, so each expert sees a healthy GEMM even at modest EP degree. Smaller EP groups, hierarchical placement, throughput-mode kernels.
- **Decode** has roughly one token per request per step. To keep per-expert batches efficient you aggregate tokens across many requests, which pushes toward **large EP degrees** (EP=64 and beyond), global balancing, and low-latency kernels.

This is why production DeepSeek-class deployments run different parallelism configurations on the prefill pool and the decode pool, connected by KV transfer.

### Wide-EP on NVL72-class systems

GB200 NVL72 puts 72 GPUs on one NVLink domain, which changes the placement calculus: "cross-node" all-to-all penalties mostly disappear inside the rack, so very wide EP becomes practical. Published deployments ([LMSYS/SGLang](https://lmsys.org/blog/2025-06-16-gb200-part-1/), [vLLM](https://blog.vllm.ai/2025/12/17/large-scale-serving.html), [NVIDIA](https://developer.nvidia.com/blog/scaling-large-moe-models-with-wide-expert-parallelism-on-nvl72-rack-scale-systems/)) report that wide-EP with disaggregated prefill/decode delivers multiples of per-GPU decode throughput versus small-EP baselines, because fewer experts per GPU means less weight memory per GPU, bigger per-expert batches, and more HBM left for KV cache. The failure mode of wide EP is the flip side: the blast radius of one slow or failed GPU is now the entire EP group, so stragglers, elastic EP (shrinking the group on failure), and rebalancing become availability features, not just performance features.

### Expert offload and paging

When total parameters exceed GPU memory, experts can live in host memory (or NVMe) and be paged in on demand. The tradeoffs:

- Routing is only known per layer at runtime, so prefetch is speculative; mispredicted experts stall the layer on PCIe latency.
- Skewed routing helps here (hot experts stay resident, cold ones page), which is the mirror image of serving-cluster EPLB, where skew hurts.
- Offload is viable for low-QPS, single-node, or edge deployments; at high QPS the PCIe traffic serializes and destroys decode ITL.

Treat offload as a capacity tool, not a performance tool. Measure page-in rate per decode step; if it is not near zero at steady state, latency will not be stable.

### How routing skew shows up as tail latency

An MoE layer completes when its *slowest* expert rank completes. So imbalance does not show up as lower average throughput first — it shows up as **p99 ITL**. The mechanism: a routing burst overloads one GPU's experts, that rank's GEMM and combine finish late, every co-batched request in the EP group waits, and the stall repeats each layer. What to measure in production:

- Per-GPU expert load per window (tokens dispatched), and its max/mean ratio — the *balancedness* metric; 1.0 is perfect.
- Time gap between fastest and slowest rank per MoE layer (straggler time).
- p99 ITL correlated against balancedness — if they move together, you have a routing-skew problem, and EPLB/redundant experts are the fix; if not, look at all-to-all or scheduler first.

Staff phrase:

> In wide-EP serving, the router's imbalance becomes everyone's tail latency. Load balancing is no longer a training loss — it is a runtime placement service with its own control loop.

---

## 17. Important Papers to Read

Read these in roughly this order.

1. **[Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer](https://arxiv.org/abs/1701.06538)** — Shazeer et al., 2017.  
   The foundational modern sparse MoE paper.

2. **[GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding](https://arxiv.org/abs/2006.16668)** — Lepikhin et al., 2020.  
   Important for large-scale MoE training and sharding.

3. **[Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity](https://arxiv.org/abs/2101.03961)** — Fedus, Zoph, Shazeer, 2021 / JMLR 2022.  
   Read for top-1 routing, load balancing, and practical scaling.

4. **[Mixtral of Experts](https://arxiv.org/abs/2401.04088)** — Mistral AI, 2024.  
   Useful open-weight LLM-era MoE reference.

5. **[DeepSeekMoE: Towards Ultimate Expert Specialization](https://arxiv.org/abs/2401.06066)** — DeepSeek-AI, 2024.  
   Read for shared experts and fine-grained expert segmentation.

6. **[DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model](https://arxiv.org/abs/2405.04434)** — DeepSeek-AI, 2024.  
   Good example of MoE combined with other efficiency architecture choices.

7. **[OLMoE: Open Mixture-of-Experts Language Models](https://arxiv.org/abs/2409.02060)** — Ai2 / Contextual AI, 2024.  
   Valuable because it is open and useful for studying MoE behavior.

---

## 18. The Staff Engineer Summary

Mixture of Experts is one of the most important sparse architectures because it scales total model capacity without scaling every token's compute linearly.

The checklist:

- Compare active parameters, not just total parameters.
- Understand top-k routing and expert capacity.
- Monitor load balance and token drop rate.
- Treat the router as both model component and scheduling component.
- Plan expert placement around interconnect topology.
- Benchmark all-to-all, expert batch size, and tail latency.
- Remember that MoE reduces active compute but increases total memory.
- Evaluate quality per active parameter and quality per dollar.

The interview answer:

> MoE buys quality per active FLOP by routing each token to a few dense experts. The hard parts are router stability, load balancing, expert specialization, memory footprint, and all-to-all communication. It is a model optimization only when the distributed serving system can keep the experts efficiently utilized.

