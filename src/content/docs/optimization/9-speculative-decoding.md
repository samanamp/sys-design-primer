---
title: "Speculative Decoding: Faster Generation Without Changing the Target Model"
description: "A staff-level guide to speculative decoding for LLM inference: draft models, verification, acceptance rate, EAGLE, Medusa, MTP, production serving, and failure modes."
---

# Speculative Decoding: Faster Generation Without Changing the Target Model

Autoregressive decoding is serial. A model normally generates one token, appends it to the context, then generates the next token:

```text
x -> token 1 -> token 2 -> token 3 -> token 4
```

That serial dependency makes decode latency hard to reduce. Speculative decoding attacks this by proposing multiple future tokens cheaply, then verifying them with the full target model in fewer forward passes.

The key idea:

> Let a cheap proposer guess several tokens. Let the expensive target model verify those guesses in parallel.

If the guesses are accepted, the system advances several tokens for the cost of one target-model verification pass. If guesses are rejected, it falls back safely. In exact speculative decoding, the final distribution can match the target model distribution.

Speculative decoding is one of the most important decode-time optimizations because it can reduce inter-token latency without changing the served target model's weights.

---

## 1. The Interview Mental Model

When speculative decoding comes up, answer in this order:

1. **Target bottleneck:** Decode latency, not prefill TTFT.
2. **Proposer:** Small draft model, n-gram proposer, Medusa heads, EAGLE head, MTP module, or retrieval-based proposer.
3. **Verifier:** The original target model.
4. **Acceptance rate:** How many proposed tokens survive verification?
5. **Overhead:** Draft compute, target verification, memory, batching complexity.
6. **Serving fit:** Does it work with tensor parallelism, batching, quantization, KV cache, and model architecture?

Core flow:

```text
Current context
      |
      v
Draft/proposer generates k candidate tokens
      |
      v
Target model verifies candidates in one pass
      |
      +-- accept prefix of candidates
      +-- reject at first mismatch/sample failure
      |
      v
Advance by accepted tokens
```

Speculative decoding helps only when:

$$
\text{tokens accepted per target pass} > 1
$$

and proposer overhead does not erase the win.

For a deeper walkthrough of the acceptance sampling mechanics, see [the speculative decoding primer](/llm-primers/5-speculative-decod/).

---

## 2. Basic Algorithm

Let the target model distribution be:

$$
p(x_t | x_{<t})
$$

Let the draft model distribution be:

$$
q(x_t | x_{<t})
$$

The draft model proposes $\gamma$ tokens:

$$
\tilde{x}_{t:t+\gamma-1} \sim q
$$

The target model evaluates these positions in parallel. In exact speculative sampling, each proposed token is accepted with probability:

$$
\min\left(1, \frac{p(\tilde{x}_i | x_{<i})}{q(\tilde{x}_i | x_{<i})}\right)
$$

If accepted, the token is emitted. If rejected, a correction distribution is used so the final samples match the target distribution.

For greedy decoding, the idea is simpler: accept proposed tokens while they match the target model's greedy choice.

The speedup depends on accepted tokens:

$$
\text{effective speedup}
\approx
\frac{\text{baseline target passes}}
{\text{verification passes} + \text{draft overhead}}
$$

If a draft proposes 4 tokens but only 1.2 are accepted on average, the speedup will be weak. If 3.5 are accepted and the draft is cheap, speedup can be large.

---

## 3. Acceptance Rate Is the Main Metric

Acceptance rate determines whether speculative decoding works.

Track:

- Mean accepted tokens per step.
- Acceptance by prompt type.
- Acceptance by generation phase.
- Acceptance by temperature/top-p.
- Acceptance by domain.
- Draft model latency.
- Verification latency.
- End-to-end tokens/sec.
- p95/p99 inter-token latency.

```text
Good speculative setup:

Draft proposes:  [A B C D]
Target accepts:  [A B C]
Advance:         3 tokens

Bad speculative setup:

Draft proposes:  [A B C D]
Target accepts:  []
Advance:         1 corrected token
Draft work wasted
```

Acceptance rate tends to be higher when:

- Draft and target are from the same family.
- Temperature is low.
- Output is predictable.
- Task is formulaic or structured.
- The draft model is trained specifically for the target.

Acceptance rate tends to be lower when:

- Sampling temperature is high.
- The task is creative.
- The model is reasoning through uncertain states.
- Draft and target token distributions differ.
- The prompt domain is out-of-distribution for the draft.

---

## 4. Draft-Model Speculation

The classic setup uses a smaller draft model:

```text
Target: 70B model
Draft:   7B model
```

The draft generates candidate tokens quickly. The target verifies them.

Advantages:

- Conceptually simple.
- Can be exact.
- Works without modifying target model weights.
- Draft model can be independently trained or distilled.

Disadvantages:

- Need to host another model.
- Draft KV cache consumes memory.
- Draft and target tokenization/templates must match.
- Acceptance rate can be poor if draft is too weak.
- Batch scheduling becomes more complex.

Draft-model speculation is most useful when decode is the bottleneck and the draft model is much cheaper than the target while still close enough behaviorally.

---

## 5. Medusa and Multi-Head Proposers

Medusa-style methods avoid a separate draft model by adding extra decoding heads to the target model. These heads predict multiple future tokens from the current hidden state.

```text
Target trunk
    |
    +-- normal LM head: token t+1
    +-- Medusa head 1: token t+2
    +-- Medusa head 2: token t+3
    +-- Medusa head 3: token t+4
```

Advantages:

- No separate draft model serving stack.
- Shares target model trunk.
- Can be easier to deploy for one target model.

Disadvantages:

- Requires training extra heads.
- Proposed future tokens may be conditionally weak because they are predicted from the same hidden state.
- Verification is still needed.

Hydra-style methods improve on independent heads by modeling sequential dependencies between draft heads.

The production question:

> Is it easier for us to maintain a draft model or to modify/train extra heads for each target model?

There is no universal answer.

---

## 6. EAGLE-Style Speculation

EAGLE methods generate draft tokens using feature-level prediction rather than only token-level prediction. EAGLE-3 became a widely used open speculative decoding approach in 2025, with vLLM and SGLang integrations. In May 2026, the vLLM, EAGLE, and TorchSpec teams announced EAGLE 3.1, focusing on robustness, efficiency, and deployability.

Why EAGLE matters:

- Stronger draft quality than naive small-model speculation in many setups.
- Integrated into production-oriented engines.
- Good fit for large target models where decode dominates.
- Better practical acceptance rates when tuned for the model family.

Production reality:

- SGLang documents EAGLE-2/EAGLE-3 speculative decoding support.
- vLLM added EAGLE-3 integration and later EAGLE 3.1 collaboration work.
- Red Hat reported vLLM Eagle 3 serving examples with speedups up to about 2.5x across tested scenarios.

The exact speedup is workload-specific. Treat reported speedups as a ceiling, not a guarantee.

---

## 7. Multi-Token Prediction and DeepSeek

Multi-Token Prediction (MTP) trains a model to predict multiple future tokens. DeepSeek-V3 included MTP modules as an auxiliary training objective, and serving stacks such as vLLM-Ascend document MTP-based speculative decoding for DeepSeek-V3 models.

High-level:

```text
Hidden state h_t
    |
    +-- predict token t+1
    +-- predict token t+2
    +-- predict token t+3
```

MTP can serve as an internal proposer. This is attractive because the model is trained from the start to support multi-token prediction, rather than bolting on a separate draft later.

Tradeoffs:

- Requires architecture/training support.
- May add parameters or heads.
- Still needs verification for correctness.
- Serving engine must know how to use MTP proposals.

The broader trend:

> Speculation is moving from an external serving trick toward a training-time architectural feature.

---

## 8. Interaction With Batching

Speculative decoding complicates continuous batching.

Without speculation, each request usually advances one token per decode iteration. With speculation, requests advance variable numbers of tokens:

```text
Request A accepts 4 tokens
Request B accepts 1 token
Request C accepts 0 tokens
Request D accepts 3 tokens
```

This affects:

- KV cache updates.
- Scheduler fairness.
- Stop condition handling.
- Streaming cadence.
- Batch shape.
- Tail latency.

A serving engine must handle variable progress without wasting GPU time or delaying unlucky requests.

Speculative decoding is therefore not just an algorithm. It is a scheduler feature.

---

## 9. When Speculative Decoding Works

It works best when:

- Decode dominates latency.
- Target model is large.
- Draft/proposer is cheap.
- Acceptance rate is high.
- Output distribution is predictable.
- Temperature is low or moderate.
- Serving engine supports speculation natively.
- Extra memory for draft/proposer is available.

Examples:

- Coding autocomplete.
- Structured output.
- Low-temperature assistant responses.
- Summarization with predictable style.
- Large-model chat serving where decode ITL is the bottleneck.

It is weaker when:

- Prefill dominates TTFT.
- Acceptance rate is low.
- Draft model is too expensive.
- Sampling is high-temperature.
- Model outputs require deep reasoning at each step.
- Serving already has low decode latency but high queueing.

---

## 10. Speculation Is Not a TTFT Fix

Speculative decoding mostly improves decode throughput and inter-token latency. It does not remove the initial prefill cost.

```text
TTFT = queue + tokenize + prefill + first decode

Speculative decoding helps:
  - later decode steps
  - sometimes first decode marginally

Speculative decoding does not solve:
  - long prompt prefill
  - bad routing
  - KV cache fragmentation
  - network latency
```

It can indirectly reduce TTFT by freeing engines faster and reducing queueing, but it should not be sold as the primary fix for long-prompt TTFT.

---

## 11. Failure Modes

### Low acceptance rate

Draft work is wasted and latency can get worse.

### Draft model too large

The proposer consumes enough compute/memory to erase speedup.

### Mismatched tokenizer or template

Small prompt formatting differences destroy acceptance.

### Bad scheduling

Variable accepted-token counts create batching inefficiency.

### Quality drift from non-exact methods

Some multi-head or approximate variants may change the output distribution.

### Memory pressure

Draft KV cache, extra heads, or MTP modules consume memory that could have served more users.

### Overclaiming speedups

Microbenchmark speedups do not always survive production traffic, streaming, tool calls, and high variance prompts.

---

## 12. Important Papers and Docs

1. **[Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192)** and **[Accelerating Large Language Model Decoding with Speculative Sampling](https://arxiv.org/abs/2302.01318)**.  
   Core speculative decoding/sampling papers.

2. **[Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads](https://arxiv.org/abs/2401.10774)**.  
   Multi-head proposer approach.

3. **[Hydra: Sequentially-Dependent Draft Heads for Medusa Decoding](https://arxiv.org/abs/2402.05109)**.  
   Improves draft-head dependency modeling.

4. **[EAGLE-3](https://huggingface.co/papers/2503.01840)** and **[vLLM EAGLE 3.1 blog](https://vllm.ai/blog/2026-05-26-eagle-3-1)**.  
   Recent production-oriented speculative decoding direction.

5. **[SGLang speculative decoding docs](https://docs.sglang.ai/advanced_features/speculative_decoding.html)**.  
   Practical serving-engine support.

6. **[DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437)** and **[NVIDIA Megatron Bridge MTP docs](https://docs.nvidia.com/nemo/megatron-bridge/0.4.0/training/multi-token-prediction.html)**.  
   MTP as training-time support for future-token prediction.

---

## 13. The Staff Engineer Summary

Speculative decoding accelerates generation by turning several serial decode steps into one verification step.

The checklist:

- Confirm decode, not prefill, is the bottleneck.
- Measure accepted tokens per target pass.
- Include draft overhead and memory in the speedup calculation.
- Use model-family-compatible draft/proposer.
- Validate exactness or measure quality drift.
- Test under continuous batching and streaming.
- Watch p95/p99, not only average tokens/sec.

The interview answer:

> Speculative decoding is a decode optimization. It works when a cheap proposer has high acceptance under the target model and the serving engine can batch variable progress efficiently. EAGLE, Medusa/Hydra, and MTP are different proposer designs; acceptance rate and scheduler integration decide whether they matter in production.

