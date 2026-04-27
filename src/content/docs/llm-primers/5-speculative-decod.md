---
title: Speculative decoding
description: Speculative decoding
---
# Speculative Decoding: Staff-Level Interview Primer

## 1. The mechanic

Run a small **draft** model `q` for `K` steps, producing tokens `d_1...d_K` autoregressively. Then in **one** target forward pass, evaluate `p(x | prefix, d_1...d_{i-1})` for `i = 1...K+1` simultaneously — the target sees all `K` draft tokens as a flat sequence and produces logits at every position. Walk left-to-right and accept each `d_i` with probability `min(1, p_i / q_i)`. On the first rejection at position `j`, resample from the residual distribution `(p_j - q_j)+ / Σ(p_j - q_j)+` and discard `d_{j+1}...d_K`. If all `K` are accepted, you also get a free bonus token from the target's logits at position `K+1`.

This rejection-sampling formulation (Leviathan; Chen et al., 2023) is **provably lossless**: the marginal distribution of accepted tokens equals `p` exactly, regardless of how bad `q` is. For greedy decoding the rule degenerates to argmax-match: accept while `argmax q == argmax p`. The losslessness depends on (a) using `q` and `p` from the *same* sampling configuration (temperature, top-p, top-k applied identically to both), and (b) the target seeing the *same* prefix the draft conditioned on. Subtle bugs — mismatched temperature between draft and target, or applying top-p only to the target — silently bias the output distribution.

**Expected speedup math.** Let `α` be per-token acceptance rate, `c = T_draft / T_target` the relative draft-to-target latency. Expected accepted tokens per cycle is `E[n] = (1 - α^(K+1)) / (1 - α)` (geometric truncation). Cycle cost is `T_target + K·T_draft = T_target·(1 + Kc)`. Speedup is:

```
S = (1 - α^(K+1)) / [(1 - α)(1 + Kc)]
```

Plug in α=0.7, K=4, c=0.1: E[n] = 2.77, cost factor = 1.4, S ≈ 1.98×. At α=0.8, K=5, c=0.05: E[n] = 3.36, cost factor = 1.25, S ≈ 2.69×. The geometric truncation is why `K` doesn't help past ~4–6 for vanilla speculation — `α^K` decays fast.

## 2. Why it works: bandwidth, not compute

Decode at batch=1 is **memory-bandwidth-bound**, not compute-bound. A Llama-3-70B forward at bs=1 reads ~140 GB of weights from HBM and does ~140 GFLOPs of math against an 8 TB/s × 989 TFLOPS H100 — arithmetic intensity is laughably low (~1 FLOP/byte vs. roofline ~120). The matmuls are skinny GEMV, the tensor cores idle. Verifying `K` draft tokens runs the *same* weights through `K`-token GEMM. The weight read is amortized; FLOPs are nearly free until you hit roofline.

Empirically `T_target(K)` is flat up to K ≈ 8–16 on H100 for 70B-class dense models — meaning verification is essentially free for those K. The break-even shifts as you scale up: bigger batches saturate compute and the free-FLOPs argument collapses. **At high batch, speculation hurts**: GLM-4.7-Flash at B=1 sees 1.30× per-request latency win at 40% acceptance; at B=32 the same model loses throughput because the verification pass is no longer free and rejected tokens are pure waste (Thoughtworks/HuggingFace, Dec 2025).

This is the dominant failure mode of naive deployments: people enable speculation and benchmark at high QPS, see no speedup, and disable it. The right framing is **goodput at fixed interactivity SLO**, not raw throughput. NVIDIA's B200 + EAGLE-3 result on Llama-4-Maverick — 1,000+ TPS/user, 4× over the prior Blackwell baseline (May 2025) — only materializes at the latency-sensitive operating point.

```
Timeline: 4 generated tokens, T_target = 10 units, T_draft = 1 unit

NON-SPECULATIVE (baseline):
[fwd t1     ][fwd t2     ][fwd t3     ][fwd t4     ]   = 40 units
 0----10----20----30----40

SPECULATIVE (K=4, all accepted, +bonus = 5 tokens for ~free):
[d1][d2][d3][d4][verify d1..d4 + bonus    ]            = 14 units
 0--1--2--3--4----------14                              → 5 tokens

SPECULATIVE (K=4, accept first 2 then reject):
[d1][d2][d3][d4][verify, accept 2, resample at pos 3]  = 14 units
 0--1--2--3--4----------14                              → 3 tokens (incl. resample)
```

## 3. Variants — architecturally, not by name

| Variant | Mechanism | Acceptance | Engineering cost | When to use |
|---|---|---|---|---|
| **Vanilla (Leviathan/Chen)** | Independent small draft model (e.g. Llama-3-1B drafting Llama-3-70B) | 50–70% | Low — drop-in, but draft model must be hosted | Baseline; when no time to train a custom drafter |
| **Medusa** | Parallel decoding heads on target's last hidden state predict t+1, t+2, ... independently. Tree verification. No draft model. | ~60% (chained tokens conditionally independent given same hidden state — weak) | Medium — heads need training | Single-model deployment, mild gains |
| **EAGLE-1** | Lightweight autoregressive draft head conditioned on target's *penultimate hidden state* + previous token. | ~70%, 2.7–3.5× speedup | Medium — train one transformer-block-sized head | Solid default before EAGLE-3 |
| **EAGLE-2** | EAGLE-1 + dynamic draft tree. Confidence-driven branch expansion; prune low-prob branches before verification. | Higher effective acceptance via tree | Same as EAGLE-1 + tree-attention infra | Latency-sensitive, low concurrency |
| **EAGLE-3** | Removes the next-feature regression objective. Fuses **low + mid + high**-layer features. **Training-time test**: trains the head with its own outputs fed back, matching inference distribution. | ~70–80% **flat across draft positions** | Higher — needs ~500K-sample training run | Current SOTA for chain/tree spec; 3.0–6.5× over autoregressive on 70B-class |
| **DeepSeek-V3 MTP** | Speculation built into pretraining. Extra transformer layer (14B params over 671B base, model.layers.61) trained jointly to predict the +2 token. Discardable for plain inference, repurposed as a 1-step drafter at serving time. | ~85% on first MTP token; 1.8× speedup | Massive — must own pretraining | Frontier labs only; production design choice when training a new model |
| **Lookahead (Jacobi)** | No draft model. Maintains an n-gram cache from prior generations, proposes n-gram continuations, verifies via Jacobi-style fixed-point iteration. | Workload-dependent; high on repetitive output | Zero training | Code, structured output, repetitive domains |
| **PLD (Prompt Lookup)** | Trivial: search for the recently-decoded suffix as a substring of the prompt, propose the next k tokens from there. | 70%+ on RAG/code-edit/summarization (verbatim copy from prompt is common) | ~5 lines of code | RAG, code-edit, summarization. Underrated. |
| **Self-speculative (Draft & Verify, SWIFT, LayerSkip)** | Use early layers of the target as the drafter; full forward verifies. | 50–60% | Medium — needs adaptive layer-skip policy | Memory-constrained deployments where a draft model won't fit |
| **Speculative Streaming (Apple, 2024)** | Multi-stream attention with future-n-gram prediction objective in fine-tuning. Single model, no draft. | 1.9–3.1× speedup, ~10⁴× fewer extra params than Medusa | Requires model fine-tuning | On-device / parameter-constrained |
| **Mirror-SD (Apple, Dec 2025)** | Branch-complete rollouts from early-exit signals run *in parallel* with target's suffix; explicitly maps draft to NPU and target to GPU. | Breaks the draft-cost / acceptance tradeoff | Heterogeneous-accelerator scheduling | Apple silicon; future heterogeneous SoCs |

## 4. Tree vs linear speculation

Linear speculation bets on one continuation: if the draft says "the cat sat on the" and the target wants "the dog ...", you're done at position 2. Tree speculation drafts a *tree* of candidates and verifies all paths in one target pass via a custom block-causal mask:

```
Draft tree (depth 3, fanout 2 at root, then 1):

                        [the]
                       /     \
                    [cat]    [dog]
                      |        |
                    [sat]    [ran]
                      |        |
                    [on]     [away]

Flatten to sequence: [the, cat, dog, sat, ran, on, away]
                       0    1    2    3    4    5    6

Tree-aware attention mask (1 = visible):

           the cat dog sat ran on  away
   the     [1   0   0   0   0   0   0 ]
   cat     [1   1   0   0   0   0   0 ]
   dog     [1   0   1   0   0   0   0 ]    ← dog can see the, NOT cat
   sat     [1   1   0   1   0   0   0 ]    ← sat ∈ cat-branch
   ran     [1   0   1   0   1   0   0 ]    ← ran ∈ dog-branch
   on      [1   1   0   1   0   1   0 ]
   away    [1   0   1   0   1   0   1 ]
```

Each token attends only to its ancestors. The target evaluates all leaves in one pass and the verification path picks the longest accepted prefix in the **highest-scoring branch**. EAGLE-2's contribution was making the tree dynamic: expand only the branches whose draft-confidence exceeds a threshold.

The tradeoff that interviewers want you to articulate: tree decoding wins at low concurrency where verification is free, **and loses fast at high concurrency** because verifying 60 tree tokens vs. 5 chain tokens is no longer free once compute is saturated. vLLM dropped tree decoding for exactly this reason; SGLang keeps it gated behind a config flag.

## 5. Where speculation breaks

- **High batch.** Free-FLOPs evaporate. Production heuristic: speculation off above some `batch * K` saturation threshold — TurboSpec (Berkeley, Dec 2025) formalizes this as goodput-driven adaptive control.
- **Long prefill, short decode.** Speculation is decode-only. A 32K-input/64-output workload sees almost no speedup — prefill dominates wall time.
- **High temperature, top-p ≈ 1.** Stochasticity collapses acceptance because `p_i / q_i` rarely clears the rejection threshold.
- **Small target.** Draft latency `c` is no longer negligible. For a 7B target, even a 1B drafter has `c ≈ 0.15`; the `(1 + Kc)` overhead eats most of the win.
- **Long context.** Draft model's KV cache scales with context too, doubling per-request memory. EAGLE-3 heads weren't trained on long context — acceptance drops noticeably past ~4K (BentoML, Aug 2025).
- **MoE targets.** Each speculative token may activate a different expert subset. Routing overhead grows superlinearly with verified tokens; break-even acceptance is higher than for dense.
- **Domain mismatch.** Generic draft on translation: Eagle-3 reports optimal draft length **K=1, sometimes K=0** (i.e., disable). On RAG and code, K=5 still pays off (Red Hat, Jul 2025).

## 6. EAGLE's hidden-state conditioning — the key idea

Vanilla speculation gives the draft only the token sequence — the draft has to reconstruct what the target "thinks" from scratch. EAGLE feeds the *target's hidden states* into the draft, so the draft is essentially completing a thought the target has already started.

```
VANILLA SPEC                            EAGLE
─────────────                           ─────
                                        target_hidden_state[t-1]  ──┐
                                                                    │  (fused: low+mid+high
target tokens [..., t-1]                target tokens [..., t-1] ──┼   layers in EAGLE-3)
        │                                       │                   │
        ▼                                       ▼                   │
   [draft model]                         [draft head] ◄─────────────┘
        │                                       │
        ▼                                       ▼
   d_t, d_{t+1}, ...                      d_t, d_{t+1}, ...
        │                                       │
        ▼                                       ▼
   [target verifies]                      [target verifies]

α ≈ 0.5–0.7 (Leviathan)                 α ≈ 0.7–0.8 (EAGLE-3, flat across positions)
```

EAGLE-3's two contributions over EAGLE-1: (a) **multi-layer feature fusion** — top-layer features are over-specialized to next-token logits, so the head sees concatenated low/mid/high features instead; (b) **training-time test** — during training, randomly feed the head its own previous outputs (not ground-truth features), eliminating the train/inference distribution shift that caused EAGLE-1's acceptance to *decay* with draft position. EAGLE-3 holds ~70–80% acceptance flat from position 1 to position 5; EAGLE-1 dropped to ~50% by position 4.

## 7. Production realities interviewers care about

- **Memory.** Draft model + its KV cache. For a 70B+1B setup, the 1B + its KV is small but non-zero. EAGLE heads are typically 200–500 MB.
- **KV-cache duplication.** Draft has its own KV. Some systems (DeepSeek MTP) share embeddings with the target to reduce this.
- **Continuous-batching scheduling.** With variable per-request acceptance length, the batch's effective work is jagged. Schedulers must handle the case where request A accepts 5 tokens this step while request B accepts 1 — naive batched verification stalls A while B finishes. Padded vs. ragged speculative batches is a real implementation choice (`disable_padded_drafter_batch` in vLLM-Ascend).
- **Prefix caching.** Compatible but the draft model's prefix-cache must mirror the target's. PD-disaggregated systems (Perplexity, vLLM) skip drafting the last prompt token on the prefiller and treat it as a decode token on the decoder to keep semantics clean.
- **Tuning K.** Too short → wasted parallelism. Too long → wasted verification on rejections. EAGLE-3 + B200 + Llama-4-Maverick: K=3 is optimal (NVIDIA, May 2025). Adaptive K (TurboSpec/OSD) tunes per-request based on rolling acceptance.
- **Quality drift detection.** Speculation is *theoretically* lossless but bugs in the residual-distribution computation, mismatched sampler config, or float-precision drift between draft and target softmax can silently bias output. Detect via: (a) periodic shadow runs comparing speculative vs. non-speculative output distribution on a fixed eval set, (b) KL divergence monitoring on next-token distributions, (c) acceptance-rate tracking — sudden drops signal config drift, sudden *rises* near 100% signal a sampler-mismatch bug.

## 8. Recent developments (2025–2026)

- **EAGLE-3 (NeurIPS 2025).** Sets current SOTA at 3.0–6.5× over autoregressive, 20–40% over EAGLE-2. Training-time test + multi-layer fusion. *Why it matters:* the first speculative method to show a clean scaling law with training data — more drafter training data = more speedup.
- **DeepSeek-V3 MTP.** Speculation built into pretraining. ~85% acceptance on first MTP, 1.8× decode speedup. *Why it matters:* shifts speculation from a serving-time addon to a pretraining design decision. SGLang's production MTP integration on H200 hit 81.5 tok/s/rank — 60% throughput uplift on 16-GPU disaggregated deployment.
- **NVIDIA + EAGLE-3 on Blackwell.** Llama-4-Maverick at 1,000+ TPS/user on a single DGX B200 node, 4× the prior Blackwell baseline (May 2025). *Why it matters:* concrete production number, draft-length-3 sweet spot.
- **Mirror-SD (Apple, Dec 2025).** GPU+NPU parallel speculation breaks the draft-cost/acceptance tradeoff. *Why it matters:* future of heterogeneous-accelerator inference.
- **Speculators (Red Hat, Nov 2025).** Production-ready unified format (HuggingFace + vLLM) for speculative methods. *Why it matters:* standardization signal — speculation is now infrastructure, not research.
- **TurboSpec / OSD (Berkeley, Dec 2025).** Closed-loop adaptive control treating speculation as a goodput optimization problem. *Why it matters:* the field is past "does it work" and into "how do you tune it under load."
- **Blackwell impact.** B200's 8 TB/s HBM (vs H100's 3.4) widens the bandwidth-bound regime — the K at which `T_target(K)` stops being flat increases, so larger trees pay off. FP4 weights further shrink the bandwidth footprint, but speculation's *relative* gain shrinks with weight bandwidth shrinkage. The end state may be: at FP4 + B200, decode is less starved, and speculation contributes 1.5–2× rather than 3×.
- **Disaggregated PD + speculation.** Perplexity's production design: prefiller does not sample the last token; hidden states transfer to decoder, which performs one decode step before drafting begins. Adds one extra TTFT step but eliminates cross-node sampler-state sync.

## 9. Staff interview talking points

1. **It works because decode is bandwidth-bound, not compute-bound.** Verifying K tokens uses FLOPs the GPU was wasting at batch=1. This is a hardware-utilization argument, not a clever algorithm trick.
2. **Lossless under rejection sampling** with matched samplers and matched prefixes. Greedy is the trivial case (argmax-match); stochastic uses `min(1, p/q)` accept then residual-distribution resample.
3. **Speedup ceiling is geometric**: `(1−α^(K+1))/(1−α)/(1+Kc)`. Past K≈4–6 you stop gaining for vanilla; tree drafting raises the ceiling by sampling multiple branches.
4. **EAGLE's insight: condition the draft on target hidden states**, not just tokens. Took acceptance from ~60% → ~75%. EAGLE-3's training-time-test eliminated the position-dependent acceptance decay.
5. **Speculation pays off where the GPU has slack — low batch, latency-sensitive operating points.** It hurts at high batch where compute is saturated. Production deployments need adaptive enable/disable.
6. **DeepSeek-V3 MTP is the architectural endpoint**: speculation as a pretraining design choice, not a serving addon. Frontier labs training new models will likely include MTP-style heads.
7. **Tree speculation = block-causal attention mask with branch-aware visibility.** Wins at low concurrency, loses at high — vLLM dropped it for that reason.
8. **The right metric is goodput at SLO**, not raw tokens/sec. A speculative system can post worse aggregate throughput while delivering better per-user latency, which is the actual product requirement.
9. **Quality verification matters in production**: speculation is theoretically lossless but practically fragile. Sampler-config mismatch, fp16/bf16 softmax drift, top-p applied asymmetrically — all silently bias output. Monitor with shadow runs and KL divergence on next-token distributions.
10. **PLD is shockingly competitive on RAG/code-edit/summarization.** Five lines of code, no training, 70%+ acceptance. Always benchmark it as a baseline before training a draft model — sometimes you don't need one.