---
title: "Worked Answer: Design a TPU-Friendly Model That Beats GPU-Designed OSS"
description: "A staff-level worked interview answer: co-designing a model architecture for TPU serving — MoE over dense, MLA vs GQA arithmetic, MXU-aligned shapes, AQT/FP8 quantization-aware training, co-trained speculation, and a quality-per-dollar cost model against a dense 70B GPU baseline."
---

"Take a strong open-source model that was designed on GPUs — a Llama/Qwen-class dense 70B. Design a TPU-friendly model that beats it on quality-per-dollar for serving. What do you change and why?"

45–60 minutes, one interviewer, escalating follow-ups. This is a co-design question: they want to see whether you understand *why* GPU-native architectures leave money on the table on TPUs, not whether you can recite MoE papers.

---

# Beating a Dense 70B on Quality-per-Dollar: A TPU-Native Design

## 1. How to Open — Drive the Scope

Three clarifications before any architecture, because they change the answer:

1. **What does "beats on quality-per-dollar" mean operationally?** I'll define it as: iso-quality on an agreed eval suite (MMLU-Pro, GPQA, LiveCodeBench, IFEval, long-context RULER, plus the product's own evals), then compare $/1M tokens served at the product's latency SLO. Quality is a *gate*, dollars are the *objective*. If the interviewer wants "better quality at iso-cost" instead, the design barely changes but the training budget does.
2. **Train from scratch, or adapt?** Adapting (upcycling the dense 70B into MoE, distilling) is cheaper but inherits GPU-native shape decisions. I'll design from scratch with a distillation-from-teacher option, and defend "why not just port Llama" when it comes up — it will.
3. **Serving-dominant, and on what?** I'll assume lifetime inference FLOPs dominate training FLOPs by 10–100× (true for any successful product), a mixed prefill/decode chat+agent workload with growing long-context share, and that we get to choose the chip. I'll design primarily for **Ironwood (TPU7x)** with a Trillium (v6e) variant, and I'll say explicitly where the chip choice changes the architecture (§8).

Working target: a sparse MoE, ~250–400B total / ~30–40B active parameters, matching or beating the dense 70B on the eval gate — the precedent that this quality-per-active-param ratio is achievable is the entire GShard → GLaM → Gemini lineage (Gemini 1.5 and 2.5 are explicitly sparse MoE; arXiv 2403.05530, 2507.06261) plus DeepSeek-V3 (671B total / 37B active).

**The opening 90 seconds, roughly verbatim:** "Before I design anything: I'm going to treat quality as a gate and dollars as the objective, assume serving FLOPs dominate training FLOPs over the model's life, and design from scratch for Ironwood with a distilled Trillium tier — I'll defend each of those if you want to move one. My thesis is that a dense 70B is mis-designed for a chip whose ridge point is 300–600 FLOPs per byte: it maximizes bytes streamed per decoded token, and bytes are the scarce resource. So the design is six levers that reduce bytes-per-token at iso-quality, each with arithmetic, and a cost model at the end that says what they compound to. First lever: sparsity."

That opening does four things: pins the assumptions, states a falsifiable thesis, previews the structure so the interviewer can steer, and gets to content inside two minutes.

**Pacing the hour** — this question dies when candidates spend 40 minutes on MoE trivia:

```text
min  0–5    Scope + thesis (this section)
min  5–12   The frame: what GPU-design bakes in, ridge-point arithmetic (§2)
min 12–35   The six levers, arithmetic-first, config sheet on the board (§3)
min 35–42   Cost model + sensitivity sliders (§4)
min 42–48   Validation gates and measurement plan (§5)
min 48–60   Follow-ups — reserve real time; the escalations carry the level signal
```

If the interviewer dives deep early (they often grab MLA or the cost model), compress §3 to the config sheet plus the two levers they care about, and protect the cost model and measurement plan — an architecture without a dollar number and a gate is the exact failure mode this question screens for.

---

## 2. The First-Principles Frame — What "GPU-Designed" Bakes In

A Llama-class dense 70B encodes a set of GPU assumptions. Name them, then invert them:

| GPU assumption baked into Llama-70B | TPU reality |
| --- | --- |
| Hand-written CUDA kernels tolerate arbitrary shapes | MXU is a 128×128 (v4/v5) or 256×256 (Trillium/Ironwood) systolic array; a dim of 129 computes as 256 — padding multiplies FLOPs |
| Dynamic shapes are cheap (CUDA graphs optional) | XLA compiles per shape signature; dynamic shapes → recompilation storms → pad/bucket everything |
| NVSwitch all-to-all inside the node; dense TP is the default | ICI torus: cheap ring collectives per axis, all-to-all is the expensive one but scales to thousands of chips per slice — this *rewards* expert parallelism laid out on torus axes |
| SMEM is tiny; caches hide you | VMEM is a ~64–128 MiB software-managed scratchpad at ~20× HBM bandwidth; what fits in VMEM changes regime |
| Ridge point ~300 FLOPs/byte (H100 bf16) | Trillium bf16 ridge ≈ 918e12 / 1.64e12 ≈ **560 FLOPs/byte**; Ironwood bf16 ≈ 2,307/7.37 ≈ **313** (2,307 TF assumes bf16 ≈ FP8/2 — Google publishes only the 4,614 TF FP8 figure), FP8 ≈ 4,614/7.37 ≈ **626**. High ridge = brutal batching pressure on decode |
| Dense means every token pays full FLOPs — fine when FLOPs are the scarce resource | On TPU pods, FLOPs are abundant and *bandwidth per active byte* is the scarce resource for decode; sparsity buys quality per byte moved |

The one-sentence thesis I'd state to the interviewer:

> A dense 70B is a bandwidth-maximalist design: every decoded token streams 70B parameters through HBM. On a chip whose ridge point sits at 300–600 FLOPs/byte, the winning design minimizes *bytes touched per token at iso-quality* — sparse activation, compressed KV, low-precision weights — and keeps every shape a multiple of the MXU tile so the abundant FLOPs are actually realized.

Everything in §3 is that sentence applied six times.

**Where the decode dollar goes.** Before touching architecture, decompose the cost of one decoded token on the dense baseline, because each design choice attacks one component:

```text
COST OF ONE DECODED TOKEN — dense 70B, int8, GQA-8, 8K context, batch B
========================================================================

  Weight streaming:   70 GB / B per token        <- attacked by MoE (§3.1)
  KV cache read:      ~320 KB × context, per seq <- attacked by MLA + int8 KV (§3.2)
                      (80 layers × 8 KV heads × 128 × 2 (K,V) × 2 B)
  Padding waste:      0–40% of FLOPs             <- attacked by shape alignment (§3.3)
  Precision:          2 B/param if bf16          <- attacked by AQT int8 / FP8 (§3.4)
  Passes per token:   1.0                        <- attacked by MTP speculation (§3.5)
  Batch B ceiling:    KV residency in HBM        <- attacked by §3.2 + §3.6 chip choice

  tokens/s ≈ (HBM BW × MBU / bytes streamed) × spec factor × B
```

The design is six multiplicative levers on that one equation. None of them is exotic; the discipline is applying all six *at design time*, because four of the six are unrecoverable after training.

---

## 3. The Design — Each Choice With Arithmetic

### 3.0 The config sheet

Committing to concrete numbers up front — every one defensible against §2's constraints:

| Parameter | Value | Why this value |
| --- | --- | --- |
| Total / active params | ~350B / ~35B | ~10× sparsity ratio (GLaM/DeepSeek-V3 territory); active count sized to gate against dense-70B quality |
| Layers | 60 | Divisible pipeline stages; keeps per-layer KV budget sane |
| d_model | 7,168 | 56 × 128; MXU-aligned, SPMD-sharding-friendly |
| Experts | 256 routed + 1 shared, top-8 | Fine-grained specialization; dispatch fan-out matches torus all-to-all |
| Per-expert d_ff | 2,048 | 16 × 128; small enough for flexible placement, VMEM-friendly tiles |
| Capacity factor | 1.25 train / ~1.05 serve | Fixed shapes for XLA; overflow absorbed by shared expert |
| Attention | MLA: latent 512 + RoPE dim 64 | §3.2 arithmetic; GQA-8 fallback config maintained |
| head_dim / heads | 128 / 56 | Tile-aligned; heads divisible by 8 for sharding |
| Vocab | 200,192 | Multiple of 256; largest matmul in the model never pads |
| Precision | AQT int8 train→serve; FP8 activations on Ironwood | §3.4 |
| MTP heads | 1 (predict t+2) | §3.5; second head is proxy-run-conditional |
| Max context | 128K (Trillium SKU) / 1M-capable (Ironwood) | §3.6; sized to HBM, not to marketing |

### 3.1 MoE instead of dense

**The economics.** Scaling-law results since GLaM: an MoE with N_active ≈ 35B and 8–10× total/active ratio matches a dense model of roughly 1.5–2× its active count at equal training tokens. So ~35B active can gate against dense 70B quality — a claim strongly suggested by every frontier lab's revealed preference (Gemini 1.5/2.5, DeepSeek-V3, GLaM all shipped sparse), but never published as a controlled iso-token MoE-vs-dense comparison; that's why the proxy ladder in §5 gates on it, and the "what if it's only 1.3×" follow-up in §6 prices the downside.

The serving-bytes story, done honestly. With top-8-of-256 routing per layer, the *union* of experts a batch touches grows fast: expected fraction touched ≈ 1 − (1 − 8/256)^B. Per pass, with ~50 GB of always-touched weights (attention, shared expert, embeddings) and ~300 GB of routed experts, all int8:

```text
BYTES PER DECODE PASS vs BATCH  (per layer routing is independent → same fraction each layer)
  B     expert-union   bytes/pass      bytes/token (MoE)   bytes/token (dense 70B)
  8        22%           ~117 GB           ~15 GB               8.75 GB
  32       64%           ~240 GB           ~7.5 GB              2.2  GB
  128      98%           ~345 GB           ~2.7 GB              0.55 GB
                         (saturates at all ~350 GB by B ≈ 100)
```

So the "MoE streams only active params" story is true only at B ≲ 8. At any realistic serving batch the MoE streams *more* weight bytes per token than the dense 70B, because the whole 350 GB is touched every pass. The honest statement of the MoE win, and it's still decisive:

1. **Iso-quality at half the active FLOPs.** Every compute-bound regime — prefill, large-batch decode, and the entire training run — pays per *active* param. That's a ~2× FLOPs-per-token win at the quality gate, and it's the win that survives at exactly the large batches where the bytes story dies.
2. **It shifts the binding HBM constraint from weights to KV**, which MLA then attacks (§3.2): weight traffic amortizes over B either way; what limits B is KV residency, and that's where the design spends its effort.
3. **At genuinely small B** (latency-critical single-stream), the routing union is small and the active-param streaming win reappears — the regime crossover is at roughly B ≈ 8–32, and §4's cost table states which side of it each column sits on.

**Why MoE is TPU-native rather than a TPU liability:**

- **Experts map onto ICI all-to-all.** Expert parallelism is a dispatch/combine all-to-all — exactly what OCS-twisted torus topologies are optimized for, and GSPMD expresses it as one more mesh axis. This is why the GShard → Switch → GLaM line came out of Google: the interconnect made MoE cheap before NVLink domains did.
- **Static capacity factors give XLA fixed shapes.** Route with a fixed capacity factor (C ≈ 1.25 train, tighter at serve) so every expert matmul is a fixed `[capacity, d_model] × [d_model, d_ff]` — compiled once, MXU-aligned, no recompilation. Dropped/overflowed tokens fall to the shared expert. GPU stacks fight MoE's raggedness with grouped-GEMM kernels; XLA wants you to *design the raggedness out*, and capacity factors do that.
- **Fine-grained experts (d_ff per expert ~2,048, many experts)** keep each expert's matmul small enough to place many per chip, balance via bias-based load balancing (DeepSeek-V3's auxiliary-loss-free scheme), and — critically — every expert d_ff is a multiple of 128.

The physical layout, because the mesh assignment is where MoE designs die on TPU:

```text
SERVING MESH — Ironwood, 4-chip replica          TRAINING MESH — 256-chip slice
--------------------------------------           ------------------------------
axis 0 (fast ICI): TP=4 for shared               torus axis X: TP (per-layer,
  layers + attention (ring all-gather)             latency-sensitive)
experts: sharded across the same 4               torus axis Y: EP (all-to-all,
  chips, EP all-to-all stays on-replica            bisection-hungry, OCS-twisted)
DCN: data parallel replicas only                 DCN across slices: DP only,
                                                   gradient reduce overlapped
Rule: dispatch/combine never crosses the ICI/DCN boundary. Ever.
```

**What I would NOT do:** giant experts (8×top-2, Mixtral-style). Coarse experts create per-expert matmuls too big to place flexibly on a 2D Trillium torus and waste the fine-grained specialization quality win.

### 3.2 Attention: MLA over GQA — the decode arithmetic-intensity story

Decode attention reads the entire KV cache per token; its arithmetic intensity is ~1 FLOP per byte read times batch-of-1-per-sequence — hopelessly memory-bound on any chip, catastrophic at a 560 FLOPs/byte ridge. So KV bytes per token is the design variable.

Per token per layer, d_model = 7,168-class model:

- **GQA, 8 KV heads × head_dim 128, bf16:** 8 × 128 × 2 (K,V) × 2 B = **4 KB**.
- **MLA (arXiv 2405.04434, 2412.19437): compressed latent 512 + decoupled RoPE key 64, bf16:** (512 + 64) × 2 B ≈ **1.15 KB** — DeepSeek-V2 reports a 93.3% KV cache reduction vs their MHA baseline (~1/15), and vs this GQA config it's ~3.5×.

Full-model KV budget, the number that sets serving batch:

```text
KV BYTES PER TOKEN, 60 LAYERS
                              bf16        int8 KV
  MHA-56 (what MLA's ~15x      ~1.7 MB     —        (nobody ships this)
    claim is measured against)
  GQA-8 (Llama-style)           240 KB      120 KB
  MLA (512 + 64 latent)          70 KB       35 KB   <- ~7x vs bf16 GQA
```

Why that matters concretely on **Trillium's 32 GB**: 350 GB of int8 weights forces a **16-chip replica** (~22 GB of weights per chip after sharding — §4 uses this replica), leaving ~10 GB of HBM per chip, ~160 GB per replica, for KV. At GQA-8 bf16 (240 KB/token, 60 layers), 10 GB holds ~41K tokens per chip — about five 8K-context sequences per chip, batch too small to climb the ridge. MLA at ~70 KB/token holds ~143K tokens (~17 seqs @ 8K per chip): 3.5× more concurrent sequences, i.e., 3.5× more decode batch — the difference between a small and a large fraction of the memory-bound tokens/s ceiling being spent on *useful* batch (the specific 15%-vs-45% split I'd quote is a prior; §5's serving prototype replaces it with a measurement). Add int8 KV (§3.4) and double it again, to ~35 seqs @ 8K per chip.

**The tradeoff, stated honestly:** MLA is more complex — the absorbed-matmul decode trick, decoupled RoPE, and it needs a real kernel. On the TPU serving stack that kernel exists (Ragged Paged Attention v3 in the `tpu-inference` vLLM backend, which as of late 2025 is the supported path; JetStream was archived Feb 2026), but it is younger and less battle-tested than GQA paths. My call: MLA on the primary design because KV bytes dominate long-context decode economics; **GQA-8 as the de-risked fallback** if MLA kernel maturity or training instability shows up in the first scaling run. This is a reversible decision if made before the main run; irreversible after.

### 3.3 Shapes: design the checkpoint for the compiler

Free quality-per-dollar — costs nothing at design time, unrecoverable afterward:

- **Every hot dimension a multiple of 128, ideally 256** (Trillium/Ironwood MXU is 256×256): d_model 7,168 (= 56×128), per-expert d_ff 2,048, head_dim 128, num_heads a multiple of 8 for clean sharding.
- **Vocab a multiple of 256:** e.g., 200,192 rather than 200,019. Llama-3's 128,256 gets this right by accident; many OSS models don't. An unaligned vocab pads the largest single matmul in the model every step.
- **Serving shapes bucketed:** batch in multiples of 64 per Cloud TPU performance guidance, sequence buckets at powers of two; one compiled executable per bucket, zero recompiles in steady state.
- **Embedding tables → SparseCore** (2 per Trillium chip, 4 per Ironwood): gather/scatter off the MXU path, which also serves MoE dispatch traffic.

### 3.4 Quantization-aware from day one — not post-training

A GPU-designed model gets quantized after the fact (GPTQ/AWQ) and eats an unbudgeted quality tax. Design it in:

- **AQT int8 (github.com/google/aqt) for train→serve:** quantized matmuls *inside* training, so the served int8 model is bit-exact with what training optimized. Cost: ~1.2–1.4× step time during training. Benefit: no post-training calibration lottery, and the quality gate is evaluated on the exact serving numerics.
- **FP8 where it's native:** Ironwood is the first TPU with native FP8 (4,614 TF; E4M3 weights/activations, E5M2 where gradient-like ranges appear). Pre-Ironwood emulates FP8 without the throughput win, so the precision plan is chip-conditional: int8 AQT as the floor everywhere, FP8 activations on Ironwood to double the effective ridge-climbing rate on prefill.
- **int8 KV cache:** halves §3.2's KV bytes again → MLA + int8 KV ≈ 35 KB/token vs 240 KB bf16-GQA — call it **7×** more tokens resident, compounding directly into decode batch.

The precision plan as a table, because "quantize it" is not a plan:

| Component | Train | Serve (Trillium) | Serve (Ironwood) | Gate |
| --- | --- | --- | --- | --- |
| Expert + shared weights | AQT int8 fwd | int8 (bit-exact w/ train) | int8, FP8 evaluated | sliced evals vs bf16 ref |
| Attention weights | AQT int8 | int8 | int8 | same |
| Activations | bf16 | bf16 | FP8 E4M3 (native) | prefill-heavy slices |
| KV cache | bf16 | int8 | int8 | RULER / long-context |
| Router logits, norms | fp32 | fp32 | fp32 | never quantized — cheap, sensitive |
| Accumulation | fp32 (MXU native) | fp32 | fp32 | — |

### 3.5 Co-trained MTP head for speculative decoding

Decode is memory-bound; speculation converts one weight-streaming pass into k verified tokens, multiplying arithmetic intensity by the acceptance-weighted k. A bolt-on draft model has mediocre acceptance on a model it never met. Co-train a **multi-token-prediction head** (DeepSeek-V3-style MTP): reported ~85–90% acceptance on the second token, ~1.8× decode throughput in production settings. EAGLE-3-class methods claim up to ~6.5× in favorable regimes — I treat that as the *ceiling*, plan capacity at 1.8×, and let anything better be upside. TPU-specific note: speculation's variable accepted-length output must be masked into fixed verify shapes (verify k tokens always, discard rejects) to stay recompile-free — one more reason to co-design rather than bolt on.

### 3.6 Long-context strategy sized to HBM — and the chip decision

State which chip and let it drive the context architecture. The specs that matter:

| | Trillium (v6e) | Ironwood (TPU7x) | v5p (for contrast) |
| --- | --- | --- | --- |
| BF16 / FP8 compute | ~918 TF / emulated | ~2,307 TF (assumed bf16 ≈ FP8/2; unpublished) / 4,614 TF native | 459 TF / — |
| HBM | 32 GB @ 1.64 TB/s | 192 GB @ 7.37 TB/s | 95 GB @ 2.76 TB/s |
| ICI | 2D torus, 256-chip pods | 1.2 TB/s bidir, pods 256 / 9,216 | 3D torus, 8,960 chips |
| Cores | 256×256 MXU, 2 SparseCores | 2 TensorCores + 4 SparseCores | 128×128 MXU |
| Role in this design | Distilled serving SKU | Flagship train + serve | Legacy training capacity |


- **Trillium (v6e, 32 GB @ 1.64 TB/s, 256-chip pods):** cheap per chip, but 32 GB forces context discipline. Design: 128K max context, MLA + int8 KV mandatory (the §3.2 arithmetic *is* the feasibility argument), interleaved local/global attention (5:1, local window 4K) so most layers' KV is bounded, and prefill/decode disaggregation so long prefills don't occupy decode HBM.
- **Ironwood (192 GB @ 7.37 TB/s, ICI 1.2 TB/s bidir, pods 256/9,216):** 6× the HBM and ~4.6× the bandwidth *changes the answer* — 1M-token context becomes a product option rather than a stunt, per-chip expert residency goes up (fewer EP hops), and native FP8 doubles prefill throughput. 

My primary target is Ironwood for the flagship SKU and Trillium for a distilled small SKU (precedent: Gemini 1.5 Flash is officially distilled from Pro — the two-tier distillation play is the proven pattern for covering the cost-quality frontier with one training program).

---

## 4. The Cost Model — Show the Method

Numbers are constructed but the method is the deliverable. The method is a two-ceiling min, and every column must state which ceiling binds:

```text
tokens/s per replica = min( memory-bound:  B × (aggregate HBM BW × MBU) / bytes(B),
                            compute-bound: MFU × aggregate FLOP/s / (2 × P_active) )

bytes(B) = weights touched per pass (§3.1 bytes(B) table) + B × context × KV bytes/token
Speculation (×1.8) applies only in the memory-bound regime: it converts one
weight-streaming pass into ~1.8 verified tokens, but it does not create FLOPs.
```

Assumptions on the table: int8 weights, 8K-context decode, MBU 40% (achievable; >50% is good), MFU 40%, speculation 1.8×. Weights must be resident: 350 GB of int8 MoE weights means the Trillium replica is **16 chips** (512 GB HBM — the distilled Trillium SKU from §3.6 is the other way to serve Trillium, priced separately below). Hourly prices are assumptions, not quotes: H100 ≈ $2.50/chip-hr effective, Trillium ≈ $1.40, Ironwood ≈ $3.50 — sensitivity below.

| | Dense 70B, 4×H100, B=64 | MoE 350B/35B, 16×Trillium, B=256 | MoE 350B/35B, 4×Ironwood, B=512 |
| --- | --- | --- | --- |
| HBM: weights + KV + free | 320 GB: 70 + 168 + ~80 | 512 GB: 350 + 73 + ~89 | 768 GB: 350 + 147 + ~270 |
| KV bytes/pass (B × 8K × KV/token) | 64×8K×320 KB = 168 GB (GQA bf16) | 256×8K×35 KB = 73 GB (MLA int8) | 512×8K×35 KB = 147 GB |
| Weight bytes/pass | 70 GB | ~350 GB (all experts touched at B=256, per §3.1) | ~350 GB |
| bytes(B) total | 238 GB | 423 GB | 497 GB |
| Aggregate BW × 40% MBU | 13.4 → 5.36 TB/s | 26.2 → 10.5 TB/s | 29.5 → 11.8 TB/s |
| Memory-bound: B × BW/bytes × 1.8 | 64 × 22.5 × 1.8 ≈ **2,600 tok/s** | 256 × 24.8 × 1.8 ≈ **11,400 tok/s** | 512 × 23.7 × 1.8 ≈ **21,900 tok/s** |
| Compute ceiling: 0.4 × FLOPs/2P_act | 0.4×8 PF int8 / 140 GF ≈ 22,900 | 0.4×29.4 PF int8 / 70 GF ≈ 168,000 | 0.4×18.5 PF FP8 / 70 GF ≈ 105,000 |
| **Binding regime** | memory-bound | memory-bound | memory-bound |
| Replica $/hr (assumed) | $10.00 | $22.40 | $14.00 |
| $/1M decode tokens | $10 / 9.3M tok/hr ≈ **$1.07** | $22.40 / 41M ≈ **$0.55** | $14 / 79M ≈ **$0.18** |
| **Relative $ at iso-quality** | **1.0×** | **~0.5×** | **~0.17× paper, call it ~0.2–0.3× after sliders** |

Why the batches differ — that *is* the design working: batch is KV-residency-limited, and MLA+int8 KV is what buys it. Dense GQA bf16 at 320 KB/token caps the H100 replica near B≈95 at 8K (B=64 leaves activation headroom); MLA int8 at 35 KB/token supports B≈580 on the Trillium replica and B≈1,490 on Ironwood — the table runs both well below their caps. Note what the table does *not* claim: at these batches the MoE streams **more** weight bytes per pass than the dense model (350 vs 70 GB — §3.1's union arithmetic), and all three columns sit in the memory-bound regime, well under their compute ceilings. The win is composed of KV-enabled batch (~4–8×), bandwidth-per-dollar (Trillium 26.2 TB/s for $22.40 vs H100 13.4 for $10), and the iso-quality gate at half the active FLOPs — not "fewer bytes streamed." Prefill flips to compute-bound and favors the MoE harder still: 18.5 PF FP8 per 4-chip Ironwood replica vs ~8 PF dense-FP8 on 4×H100, at half the FLOPs per token. And the distilled Trillium SKU (§3.6) is the second Trillium answer: a model sized to fit a 4-chip replica outright, for the traffic slice that doesn't need the flagship.

**Sensitivity, named explicitly:** the model is most fragile to (1) the assumed Ironwood price — at 2× my assumption the Ironwood column degrades to ~0.35×, and at 2× plus a 25% MBU stack it converges toward Trillium; (2) achieved MBU — if the young MoE serving stack lands at 25% while the mature GPU stack holds 45%, the Trillium column moves from ~0.5× to ~0.85× and the margin thins to the quality-gate term, which is why MBU is the top validation metric in §5; (3) context mix — at 32K+ contexts KV bytes/pass dominate and the MLA term grows, at 2K they shrink. I'd present the table with these three sliders, not as a point estimate.

**Training amortization (the "but MoE costs more to train" objection, retired with arithmetic):** suppose the MoE program costs 2× the dense-70B training compute — ~8e24 vs ~4e24 FLOPs including the AQT tax and proxy sweeps. Price the delta with the same rates as the table: 4e24 FLOPs at 40% MFU on $2.50/hr H100s (989 TF bf16 → ~400 TF sustained) is 4e24/4e14 ≈ 1e10 chip-seconds ≈ 2.8M chip-hours ≈ **$7M** (double it for overheads and it's still ~$15M, not $50M). On the serving side: 100B tokens/day is 36.5T tokens/year — at a $1/1M-token baseline that's ~$36.5M/year *total*, i.e., ~$3.65M/year per 0.1× of cost ratio. The ~0.5× advantage saves ~$18M/year, repaying the ~$7M delta in **~5 months** at that volume. Repayment time scales inversely with volume: at Google scale (order 1T+ tokens/day) it's weeks; at a boutique 1B tokens/day it's ~$180K/year of savings and the payback is measured in decades — at which point Path B in §6 is the answer, not this training run. If the workload is *not* serving-dominant, this whole design brief changes — which is exactly why §1 pinned that assumption first.

---

## 4.5 How the Choices Compose — Interaction Notes

The six levers are close to multiplicative, but three interactions need explicit management:

- **MoE × MLA (+).** Both shrink bytes-per-token, but on different components (weights vs KV), so they compound rather than overlap. This is the DeepSeek-V3 recipe and the reason its serving economics shocked people.
- **MoE × speculation (− at small batch, needs care).** Verifying k speculative tokens routes k× the tokens through experts, widening the expert union touched per pass — speculation partially *erodes* the MoE bandwidth win at small batch. Net still positive (verified tokens per weight-byte rises), but the capacity model must use the joint number, not the two factors multiplied naively.
- **int8 KV × MLA (+, with an eval caveat).** Quantizing an already-compressed latent is numerically riskier than quantizing redundant GQA heads; the long-context RULER slice is the canary and int8-KV ships behind its own gate.
- **Capacity factor × quality (the knob with a body count).** Tightening serve-time capacity factor from 1.25 → 1.05 buys throughput and drops tokens under load. Token-drop rate is a *quality* metric and belongs in the live eval loop, not the infra dashboard.

---

## 5. What I'd Measure to Validate

- **Iso-quality gate first, sliced, not averaged:** the full eval suite from §1 with per-capability tolerances negotiated with the model team. An MoE that matches MMLU but drops 4 points of code is not iso-quality. No cost claim exists until this gate passes.
- **MFU (prefill) and MBU (decode)** per chip from xprof, against the roofline of the *actual* precision in use. Target: prefill MFU ≥ 40%, decode MBU ≥ 40% of the weight-streaming ceiling.
- **Speculation acceptance rate** by domain (code vs chat vs long-context) — the 1.8× is an average that hides domain variance; capacity planning uses the worst live slice.
- **Expert load balance and drop rate** at the serving capacity factor; recompile count in production (target: zero in steady state); ICI all-to-all time as % of step, verified overlapped in the trace.
- **The actual objective:** $/1M tokens at the latency SLO, measured on live traffic mix, versus the dense baseline serving the same traffic. Everything above is instrumentation for explaining this number.

**Phase-gated, with numbers attached:**

```text
PHASE            GATE (must pass to proceed)                        FAILURE ACTION
------------------------------------------------------------------------------------
Proxy ladder     MoE > dense quality/active-FLOP by >=1.5x;         Rescale active params;
(1B, 10B)        MLA within noise of GQA; AQT within budget         GQA fallback config
Serving proto    Decode MBU >= 35% on proxy model on real chips;    Fix stack before
(pre-flagship)   MTP acceptance >= 80% t+2; zero steady recompiles  locking shapes
Flagship train   Loss tracks proxy-extrapolated curve within band;  Pre-committed rollback
                 expert drop rate < 1% at CF 1.25                   points, not heroics
Serve canary     Sliced evals within per-slice tolerance vs bf16    Precision/CF rollback
                 reference; token-drop < 0.1% at serve CF           per §4.5 knob
Production       $/1M tokens <= 0.6x dense baseline at SLO on       Re-run cost model with
                 live mix, sustained 30 days                        measured sliders (§4)
```

The order encodes the risk philosophy: every unrecoverable decision (shapes, MLA, sparsity ratio) is validated by a phase that costs <2% of the flagship run.

---

## 5.5 Out of Scope — Deferred With Reasoning

Named exclusions are half the design. Things this program deliberately does not do:

- **Upcycling the dense 70B into the MoE as the flagship path.** Sparse upcycling + MLA conversion is real (see the scratch-vs-adapt follow-up in §6) and is used here as the de-risking stage — but as the *final* model it locks in donor shapes (vocab, head_dim, bf16-native numerics) and correlated experts. Adopted as stage one, rejected as the destination; the crossover arithmetic in §6 says why.
- **Sub-8-bit weights (int4/NVFP4-class) at launch.** The int8→int4 step roughly doubles the bandwidth win but the calibration/eval cost on an MoE with a compressed KV latent is a research project, not a launch dependency. Q3 candidate behind its own gate.
- **Custom Pallas kernels as a launch dependency.** Adopt Ragged Paged Attention v3; write nothing until a post-launch profile shows XLA leaving specific, named time on the table.
- **A third model tier.** Two SKUs (Ironwood flagship, Trillium distilled) cover the frontier; a third multiplies eval, serving, and distillation surface for marginal coverage.
- **Cross-slice expert parallelism.** EP stays inside a slice by construction; if the model outgrows a slice, the answer is more sparsity or bigger chips, not DCN in the dispatch path.

---

## 6. Escalating Follow-Ups

**"Why not just port Llama-70B to TPU and skip the training run?" — the scratch-vs-adapt decision, done properly**

This deserves more than a dismissal, because there are really three paths and a staff answer prices all of them:

*Path A — port and serve (the control arm).* Decode on a dense 70B at int8 is weight-streaming-bound, so the GPU-vs-TPU cost ratio is bandwidth-per-dollar times overhead: H100 gives 3.35 TB/s ÷ $2.50/hr = 1.34 TB/s per $·hr vs Trillium's 1.64 ÷ $1.40 = 1.17 (**×1.14**); 70 GB of weights doesn't fit in 32 GB HBM so you tensor-shard across ≥4 chips, paying a per-layer ICI all-reduce (~10–15% MBU haircut — a prior, to be replaced by §5's serving-prototype measurement, **×1.1** — partially offset by the 4-chip slice's 128 GB giving more KV/batch headroom than one 80 GB H100); donor shapes and hyper-tuned GPU kernels for exactly-Llama cost another ~5–10% (also a prior pending the same prototype, **×1.07**). Compound ≈ **1.3–1.4× of GPU cost**, load-bearing on the price assumption (§4's sliders move it 1.1–1.6×). Zero training spend, but every architectural lever stays where the donor left it. This is the baseline every other path must beat.

*Path B — adapt the checkpoint (the path that's gotten real since 2025).* The claim "you can't change the architecture without retraining" is no longer true, and pretending otherwise loses credibility:
- **Sparse upcycling** (Komatsuzaki et al., arXiv 2212.05055): initialize MoE experts from copies of the dense FFN — the 70B's weights seed a ~8×70B-FFN MoE that recovers quality with a small fraction of from-scratch tokens. This directly reuses the initial weights.
- **MHA/GQA → MLA conversion**: TransMLA (arXiv 2502.07864) and MHA2MLA (arXiv 2502.14837) retrofit latent-KV attention onto existing checkpoints with low-single-digit-percent fine-tuning budgets — the KV bytes-per-token lever, post hoc.
- **QAT fine-tune with AQT** recovers most of the post-hoc-quantization quality tax.
- **Bolt-on speculation** (EAGLE-class heads) trains against the frozen model — the acceptance lever without touching the base.

So Path B recovers roughly three of my four levers at maybe 5–10% of the flagship training cost. What it *cannot* recover: the dense-activation bytes term only partially (upcycled experts start correlated — they differentiate slowly and rarely reach trained-from-scratch MoE quality-per-active-param), MXU-hostile donor shapes (head_dim, vocab), and the tokenizer.

*Path C — from scratch (the doc's mainline).* Full lever access, full cost.

*On distillation specifically* — the objection "the models aren't in the same family" is half right. **Logit-level** KD across families is genuinely awkward (vocabulary/tokenizer mismatch means the teacher's distribution doesn't align token-for-token; ULD-style tricks exist but are lossy). **Sequence-level** distillation — generating data with the teacher and fine-tuning the student on it — has no family requirement at all, and it's how cross-family distillation actually ships. The two-SKU design in §3.6 is same-family (flagship→Trillium SKU), where logit KD works cleanly; that's deliberate.

*The decision rule*, which is the actual staff signal: scratch-vs-adapt is a tokens-served amortization question. Using §4's own rates, the from-scratch training delta is order ~$7–15M; if the from-scratch design saves an additional ~0.2×–0.3× on serving cost versus the best adapted model — ~$0.20–0.30 per 1M tokens at the $1/1M baseline — the crossover sits at roughly 3–7×10¹³ served tokens: weeks-to-months of Google-scale traffic, decades of a small deployment. **At Google volume, scratch wins and Path B is the de-risking stage** (upcycle + convert first, learn the serving numbers on real chips, then commit the flagship run). At startup volume, Path B *is* the answer and claiming otherwise is resume-driven engineering. Saying which side of the crossover you're on — and why — is the answer; picking a path without the crossover is the miss.

**"Ironwood matches B200 spec-for-spec. Doesn't that collapse your whole 'TPU-friendly design' thesis?"**
Partially — and conceding the true part first is the answer. On an Ironwood-only fleet, the decode roofline argument converges with the GPU's (bf16 ridge ~313 vs H100's ~295), and the honest proof is that DeepSeek-V3 — a GPU-designed model — already uses MoE + MLA + MTP + FP8: the six levers are frontier economics, not TPU physics. What survives spec parity is the *constraint surface*, and it has four TPU-specific components: (1) shape/compile discipline — 256×256 MXU tiles and XLA static shapes are stricter than the GPU toolchain, so GPU-native head dims and dynamic shapes still burn FLOPs and recompiles at parity; (2) the fabric — EP layout, disaggregation topology, and KV movement designed for a 9,216-chip ICI/OCS domain rather than 72-GPU NVLink islands, which is where co-design lives at pod scale; (3) fleet heterogeneity — nobody serves on Ironwood alone; the Trillium tier (32 GB, ridge ~560) still binds the fleet-wide design, which is what the two-SKU strategy answers; (4) the dtype roadmap cuts against us — B200's FP4 (9,000 TF) is a discount a TPU model can't take, and a serious cost model prices that threat instead of ignoring it. So the pitch shifts: on Ironwood, "TPU-friendly" means the same frontier levers plus stricter shape discipline plus a pod-scale serving topology GPUs can't express, minus FP4 — not salvation from a hostile roofline. Note also "Ironwood is cheaper" is a presumption (TCO/perf-per-watt), not a published price — say so. Sharper: distinguish the *price basis*. NVIDIA GPUs rent in a competitive multi-vendor market that bids price toward cost; TPUs rent from one capacity-constrained vendor, so rental price carries scarcity premium (estimates for Ironwood range ~$4.40 external to $8–10 on-demand speculation vs B200 at ~$3–5 neocloud). At external list prices the TPU $/token advantage is unproven and possibly negative. The TCO case is *internal*: Google's cost basis has no NVIDIA gross margin in it (~1/3 the capital cost per equivalent compute by street estimates) plus the perf/W edge — and this role optimizes at that basis. Say which basis you're pricing in before quoting any ratio; the doc's $/chip-hr priors are internal-basis assumptions.

**"What breaks at 9,216 chips?"**
That's the Ironwood full pod — a training-scale question. Failure modes in order: (1) anything crossing the ICI/DCN boundary by accident — EP or TP spanning slices meets ~100×-lower bandwidth; parallelism layout must be torus-aware by construction. (2) All-to-all bisection: EP dispatch is the collective most sensitive to it; keep EP groups within OCS-optimized sub-tori and hierarchical (in-slice dispatch, cross-slice only for data parallelism). (3) MTBF: at ~9K chips, preemptions and chip failures are continuous — OCS routing around failed cubes plus checkpoint/resume cadence become first-class design inputs. (4) Load imbalance amplification: a 2% expert hot-spot is invisible at 64 chips and a synchronous-step straggler at 9K.

**"When do you write Pallas kernels?"**
Only where a profile shows XLA's schedule leaving specific time on the table, and the algorithm isn't expressible as fused dense HLO. On this design, three candidates qualify a priori: MLA decode attention (ragged paged access — already covered by Ragged Paged Attention v3 in tpu-inference; adopt, don't write), MoE dispatch/combine if capacity-factor shapes still leave the MXU idle, and the fused int8/FP8 quantized matmul path if AQT's XLA lowering underperforms. Everything dense stays on XLA — a hand kernel is a per-generation liability (128-tile kernels rotted on 256-tile Trillium).

**"Your MoE trains unstably / router collapses — now what?"**
Pre-committed mitigations, in order: bias-based load balancing (aux-loss-free), router z-loss, fine-grained experts with a shared expert absorbing overflow, and the GQA-dense fallback config that shares the training infrastructure. The scaling run plan has a 1/10-scale proxy sweep specifically to buy this de-risking before the big run — architecture bets get retired at proxy scale, not flagship scale.

**"Quality-per-dollar for whom — what if traffic is 90% short-context chat?"**
Then long-context machinery is over-designed and the distilled Trillium SKU carries most traffic; the flagship serves the 10% that monetizes. This is a routing/product question and the two-SKU distillation design (§3.6) is the hedge — say so rather than defending 1M context for a workload that doesn't want it.

**"How would you validate the architecture before committing the flagship run?"**
A proxy-scale ladder: ~1B and ~10B-active versions of the exact architecture (same sparsity ratio, same MLA config, same AQT numerics) trained on the same data mix, fitted against the dense scaling curve at matched tokens. The decision rule is pre-registered: the MoE ladder must sit above the dense ladder's quality-per-active-FLOP by the margin the cost model needs, MLA must match GQA within noise on the proxy evals, and AQT-int8 must track bf16 within its budget. Any architecture bet that can't be tested at 1/50 scale doesn't go in the flagship. The proxy runs also produce the serving prototype — MBU and acceptance-rate measurements come from real chips, not the spreadsheet, before the big run locks the shapes.

**"You claimed a 2× quality-per-active-param win for MoE. What if it's 1.3×?"**
Then rerun §4 with active params scaled up to hit the quality gate — say 50B active instead of 35B, which at the same 10× sparsity ratio means ~500B total. The decode hit is through residency, not the active count: 500 GB of int8 weights grows the Trillium replica to ~20 chips and adds ~150 GB to every pass's weight streaming; redoing §4's arithmetic lands the Trillium column around ~0.7× instead of ~0.5× and Ironwood around ~0.22× paper instead of ~0.17×. Prefill and training costs rise the full 1.4× with active FLOPs. The program is still positive but no longer decisive on Trillium; at that point the Ironwood bandwidth-per-dollar term and MLA's KV win carry the case. This is the answer's load-bearing property: no single factor carries the whole margin, and I can tell you the break-even point of each one.

**"The serving stack: what do you actually run?"**
`tpu-inference` — the unified JAX/vLLM TPU backend (Oct 2025), which is the currently supported path; JetStream was archived Feb 2026. It brings Ragged Paged Attention v3, continuous batching, and prefix caching. What we own on top: the MoE dispatch config, MTP verify scheduling, shape-bucket policy, and the disaggregated prefill/decode topology on Trillium. What we explicitly do not build: our own serving engine — that's a two-year detour to parity.

---

## 7. Staff+ Signals

What earns the level in this interview:

- **Driving scope before designing:** defining "quality-per-dollar" as gated-quality-then-cost, fixing the serving-dominant assumption, and choosing the chip *explicitly* — the interviewer should never have to supply the frame.
- **Arithmetic before architecture:** ridge points (560/313/626 FLOPs/byte), KV bytes per token (4 KB vs 1.15 KB), bytes(B) per decode pass with its regime crossover (70 GB flat for dense vs a routing union that saturates at 350 GB by B≈100), and a cost table that states which ceiling — memory or compute — binds each column, with MBU and price assumptions named. Every design choice traced to a number, every number's derivation shown.
- **Naming what you would NOT do:** no coarse Mixtral-style experts, no Pallas by default, no post-training quantization, no 1M-context on Trillium, and "port Llama first as the control arm."
- **Quality-gate discipline:** sliced evals as a shipping gate, AQT so training numerics equal serving numerics, speculation losslessness by construction. Faster-but-worse is a regression.
- **Citing production precedent, not vibes:** GShard→GLaM→Gemini for MoE-on-TPU, DeepSeek-V3 for MLA/MTP/fine-grained experts, Gemini Flash-from-Pro for the distillation SKU, tpu-inference as the current serving stack.
- **Reversibility awareness:** MLA-vs-GQA flagged as a decision with a proxy-scale checkpoint and a fallback; shape alignment flagged as unrecoverable after training.

## What Falls Short

- **Technique listing without numbers.** "I'd use MoE, MLA, quantization, and speculative decoding" is a senior answer. The staff answer says *how much each buys, on which chip, and why* — the delta is the arithmetic.
- **Ignoring the quality gate.** A cost win at unverified quality is not quality-per-dollar; candidates who never define the eval suite have optimized an undefined ratio.
- **GPU-reflex answers:** "use CUDA graphs," "write a custom kernel first," "grouped-GEMM the MoE," "dynamic batching handles shapes." Each one signals the candidate is treating a TPU as a slow GPU instead of a compiler-scheduled machine that wants static shapes, capacity factors, and torus-aware layouts.
- **No measurement plan.** A design whose success is asserted rather than instrumented (MBU, acceptance rate, drop rate, $/1M tokens on live traffic) fails the "would I let this person own the program" test.
- **Ignoring the boring free wins.** Candidates who redesign attention but ship a vocab of 200,019 have missed that shape alignment is the highest ROI-per-effort item on the sheet.
- **No fallbacks.** MLA instability, router collapse, kernel immaturity — a plan with no pre-committed retreat positions is a bet, not a design.

---

## References Worth Citing in the Room

1. **[How to Scale Your Model](https://jax-ml.github.io/scaling-book/)** (Google DeepMind) — rooflines, ICI math, and the sharding arithmetic this answer leans on throughout.
2. **[Gemini 1.5](https://arxiv.org/abs/2403.05530)** and **[Gemini 2.5](https://arxiv.org/abs/2507.06261)** technical reports — the production precedent that frontier TPU-native models are sparse MoE; 1.5 Flash officially distilled from Pro.
3. **[DeepSeek-V2](https://arxiv.org/abs/2405.04434)** / **[DeepSeek-V3](https://arxiv.org/abs/2412.19437)** — MLA, fine-grained experts with shared expert, aux-loss-free balancing, co-trained MTP; the strongest public existence proof for §3.1–3.5.
4. **[AQT](https://github.com/google/aqt)** — quantized training with bit-exact train/serve numerics on TPU.
5. **Cloud TPU docs** — per-generation specs (v5p, v6e/Trillium, TPU7x/Ironwood) and the performance guide's batch/shape guidance.
6. **tpu-inference** — the vLLM TPU backend; Ragged Paged Attention v3 and the current supported serving path.

---

## Closing — The One-Minute Version

If the interviewer asks for the whole answer compressed: a dense 70B is the wrong shape for a machine whose ridge point is 300–600 FLOPs/byte. The TPU-native design attacks the scarce resources at iso-quality along six axes — sparse activation (MoE, ~35B active: half the active FLOPs everywhere, and the streaming win at small batch), compressed KV (MLA, ~7× with int8 — the lever that actually buys decode batch), tile-aligned shapes (zero padding tax), quantization-aware training (int8/FP8 with no post-hoc quality lottery), co-trained speculation (~1.8× passes-to-tokens in the memory-bound regime), and a chip-sized context strategy (Trillium SKU distilled from an Ironwood flagship). The compounded result is roughly 2× quality-per-dollar on a Trillium replica and 3–5× on Ironwood on paper — presented with its price/MBU/context sliders, gated on sliced iso-quality evals, validated by MBU/acceptance/drop-rate instrumentation on a proxy-scale ladder before the flagship run, with every architecture bet carrying a numbered break-even point and a pre-committed fallback.

---

## Confidence Ledger

Three buckets, because a staff answer knows which of its numbers are load-bearing facts, which are arithmetic on stated assumptions, and which are guesses awaiting a profiler.

**Verified facts** (checkable against public specs and papers):
- Chip specs: Trillium (v6e) 32 GB HBM @ 1.64 TB/s, ~918 TF bf16, 256-chip pods, 2 SparseCores; Ironwood (TPU7x) 192 GB @ 7.37 TB/s, 4,614 TF FP8, 1.2 TB/s bidir ICI, pods 256/9,216; v5p 95 GB @ 2.76 TB/s; H100 SXM 80 GB @ 3.35 TB/s.
- Papers and results: DeepSeek-V2 MLA with 93.3% KV reduction (arXiv 2405.04434); DeepSeek-V3 671B/37B, aux-loss-free balancing, MTP (arXiv 2412.19437); sparse upcycling (arXiv 2212.05055); TransMLA (arXiv 2502.07864); MHA2MLA (arXiv 2502.14837); Gemini 1.5/2.5 sparse MoE, Flash distilled from Pro (arXiv 2403.05530, 2507.06261); AQT (github.com/google/aqt).
- Stack claims: tpu-inference as the unified JAX/vLLM TPU backend with Ragged Paged Attention v3; JetStream archived Feb 2026; Llama-70B geometry (80 layers, GQA-8, head_dim 128 → 320 KB/token bf16 KV).

**Derived numbers** (method shown in the text; assumptions named):
- Ridge points 560/313/626 FLOPs per byte — peak FLOPs ÷ HBM BW; the Ironwood bf16 313 rests on the *assumed* 2,307 TF (bf16 ≈ FP8/2, unpublished).
- KV per token: 240 KB GQA-8 bf16 / 70 KB MLA bf16 / 35 KB MLA int8 at 60 layers — bytes-per-head arithmetic.
- bytes(B): 1−(31/32)^B expert-union fraction on 50 GB shared + 300 GB routed; saturation at ~350 GB by B≈100.
- Cost table tokens/s and $/1M: the two-ceiling min with MBU 40%, MFU 40%, spec 1.8×, assumed prices — the ~1.0×/~0.5×/~0.17× ratios move with every one of those sliders.
- Training delta ~$7M and the ~5-month repayment at 100B tok/day; the 3–7×10¹³-token scratch-vs-adapt crossover.

**Priors** (would replace with measurement, per §5's serving prototype and proxy ladder):
- 40% MBU / 40% MFU achieved; 15%-vs-45% useful-batch split; Path A's 10–15% ICI haircut and 5–10% donor-shape tax.
- MoE ≈ dense-2×-active at iso-tokens (revealed preference of frontier labs, never a published controlled comparison — gated at proxy scale).
- MTP acceptance ~85–90% / 1.8× decode; AQT training tax 1.2–1.4×; all hourly prices; the 100B tok/day volume assumption itself.
