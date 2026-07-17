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
| Ridge point ~300 FLOPs/byte (H100 bf16) | Trillium bf16 ridge ≈ 918e12 / 1.64e12 ≈ **560 FLOPs/byte**; Ironwood bf16 ≈ 2,307/7.37 ≈ **313**, FP8 ≈ 4,614/7.37 ≈ **626**. High ridge = brutal batching pressure on decode |
| Dense means every token pays full FLOPs — fine when FLOPs are the scarce resource | On TPU pods, FLOPs are abundant and *bandwidth per active byte* is the scarce resource for decode; sparsity buys quality per byte moved |

The one-sentence thesis I'd state to the interviewer:

> A dense 70B is a bandwidth-maximalist design: every decoded token streams 70B parameters through HBM. On a chip whose ridge point sits at 300–600 FLOPs/byte, the winning design minimizes *bytes touched per token at iso-quality* — sparse activation, compressed KV, low-precision weights — and keeps every shape a multiple of the MXU tile so the abundant FLOPs are actually realized.

Everything in §3 is that sentence applied six times.

**Where the decode dollar goes.** Before touching architecture, decompose the cost of one decoded token on the dense baseline, because each design choice attacks one component:

```text
COST OF ONE DECODED TOKEN — dense 70B, int8, GQA-8, 8K context, batch B
========================================================================

  Weight streaming:   70 GB / B per token        <- attacked by MoE (§3.1)
  KV cache read:      ~244 KB × context, per seq <- attacked by MLA + int8 KV (§3.2)
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

**The economics.** Scaling-law results since GLaM: an MoE with N_active ≈ 35B and 8–10× total/active ratio matches a dense model of roughly 1.5–2× its active count at equal training tokens. So ~35B active can gate against dense 70B quality — this is not speculative; it is the published Gemini lineage and DeepSeek-V3's demonstrated result. The serving win, per decoded token at batch B with E-of-N routing (top-k=8 of 256 fine-grained experts, DeepSeek-style, plus 1 shared expert):

- Dense 70B, int8: **70 GB** of weights streamed per forward pass regardless of batch.
- MoE 350B total / 35B active, int8: bytes streamed per pass = shared layers + *union of experts touched by the batch*. At small B that union is small (~35–60 GB); at large B it saturates toward all resident experts — but by then you're compute-bound and the MXU is earning its keep. The honest statement: MoE moves the memory-bound region's cost from "total params" to "active params + routing spread," a 1.5–2× decode-bandwidth win at realistic serving batches, *and* a ~2× quality win at iso-active-FLOPs. Multiplied, that's the quality-per-dollar gap.

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
- **MLA (arXiv 2405.04434, 2412.19437): compressed latent 512 + decoupled RoPE key 64, bf16:** (512 + 64) × 2 B ≈ **1.15 KB** — DeepSeek reports the equivalent of reducing KV to roughly 1/12 of their MHA baseline, and vs this GQA config it's ~3.5×.

Full-model KV budget, the number that sets serving batch:

```text
KV BYTES PER TOKEN, 60 LAYERS
                              bf16        int8 KV
  MHA-56 (what MLA's 12x       ~1.7 MB     —        (nobody ships this)
    claim is measured against)
  GQA-8 (Llama-style)           244 KB      122 KB
  MLA (512 + 64 latent)          70 KB       35 KB   <- 7x vs bf16 GQA

  Trillium chip, ~12 GB HBM left after sharded weights:
    GQA bf16:   ~49K resident tokens  (~6 seqs @ 8K)
    MLA int8:  ~340K resident tokens (~42 seqs @ 8K)
```

Why that matters concretely on **Trillium's 32 GB**: a 61-layer model at 4 KB/token/layer stores ~244 KB/token → after ~20 GB of int8 weights per chip (sharded), ~12 GB of HBM holds only ~49K tokens of KV — a handful of 8K-context sequences per chip, i.e., batch too small to climb the ridge. MLA at ~70 KB/token holds ~170K tokens: 3.5× more concurrent sequences, which is 3.5× more decode batch, which is the difference between 15% and 45% of the memory-bound tokens/s ceiling being spent on *useful* batch. Add int8 KV (§3.4) and double it again.

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
- **int8 KV cache:** halves §3.2's KV bytes again → MLA + int8 KV ≈ 35 KB/token vs 244 KB bf16-GQA — call it **7×** more tokens resident, compounding directly into decode batch.

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
| BF16 / FP8 compute | ~918 TF / emulated | 2,307 TF / 4,614 TF native | 459 TF / — |
| HBM | 32 GB @ 1.64 TB/s | 192 GB @ 7.37 TB/s | 95 GB @ 2.76 TB/s |
| ICI | 2D torus, 256-chip pods | 1.2 TB/s bidir, pods 256 / 9,216 | 3D torus, 8,960 chips |
| Cores | 256×256 MXU, 2 SparseCores | 2 TensorCores + 4 SparseCores | 128×128 MXU |
| Role in this design | Distilled serving SKU | Flagship train + serve | Legacy training capacity |


- **Trillium (v6e, 32 GB @ 1.64 TB/s, 256-chip pods):** cheap per chip, but 32 GB forces context discipline. Design: 128K max context, MLA + int8 KV mandatory (the §3.2 arithmetic *is* the feasibility argument), interleaved local/global attention (5:1, local window 4K) so most layers' KV is bounded, and prefill/decode disaggregation so long prefills don't occupy decode HBM.
- **Ironwood (192 GB @ 7.37 TB/s, ICI 1.2 TB/s bidir, pods 256/9,216):** 6× the HBM and ~4.6× the bandwidth *changes the answer* — 1M-token context becomes a product option rather than a stunt, per-chip expert residency goes up (fewer EP hops), and native FP8 doubles prefill throughput. 

My primary target is Ironwood for the flagship SKU and Trillium for a distilled small SKU (precedent: Gemini 1.5 Flash is officially distilled from Pro — the two-tier distillation play is the proven pattern for covering the cost-quality frontier with one training program).

---

## 4. The Cost Model — Show the Method

Numbers are constructed but the method is the deliverable. Assumptions on the table: int8 weights, decode-dominant workload, memory-bound decode (true at these ridge points until batch is very large), 40% MBU achieved on the weight-streaming ceiling (achievable; >50% is good), speculation 1.8×. Hourly prices are assumptions, not quotes: H100 ≈ $2.50/chip-hr effective, Trillium ≈ $1.40, Ironwood ≈ $3.50 — sensitivity in the last row.

**Decode ceiling method:** tokens/s per replica ≈ (aggregate HBM BW × MBU / bytes streamed per token) × speculation factor × batch (sequences served concurrently, bounded by KV residency).

| | Dense 70B on 4×H100 | MoE 350B/35B on 8×Trillium | MoE on 4×Ironwood |
| --- | --- | --- | --- |
| Weights streamed/token (int8) | 70 GB | ~45 GB (active + routing spread) | ~45 GB |
| Aggregate HBM BW | 4×3.35 = 13.4 TB/s | 8×1.6 = 12.8 TB/s | 4×7.37 = 29.5 TB/s |
| Weight-pass ceiling (40% MBU) | 76 passes/s | 114 passes/s | 262 passes/s |
| × speculation 1.8× | ~137 | ~205 | ~470 |
| KV-limited concurrent 8K seqs | ~40 (GQA bf16, 60 GB free) | ~90 (MLA int8, ~90 GB free) | ~400+ (700+ GB free) |
| Replica $/hr (assumed) | $10.00 | $11.20 | $14.00 |
| **Relative $/1M decode tokens at iso-quality** | **1.0×** | **~0.55×** | **~0.3–0.35×** |

Read the direction, not the third digit: the MoE wins ~1.5× on bytes-per-token, ~1.5–2× on KV-enabled batch, and Ironwood adds a raw bandwidth-per-dollar term. Even if any single factor underdelivers by 30%, the compounded margin holds. Prefill flips to compute-bound and favors TPU harder still (Ironwood FP8: 4×4,614 = 18.5 PF FP8 per 4-chip replica vs ~8 PF dense-FP8 on 4×H100).

**Sensitivity, named explicitly:** the model is most fragile to (1) the assumed Ironwood price — if it's 2× my assumption, the Ironwood column degrades to ~0.6–0.7× and Trillium becomes the value play; (2) achieved MBU — if the MoE serving stack lands at 25% MBU while the mature GPU stack holds 45%, half the paper margin evaporates, which is why MBU is the top validation metric in §5; (3) routing spread — adversarially uniform token routing at small batch pushes bytes-touched toward total params. I'd present the table with these three sliders, not as a point estimate.

**Training amortization (the "but MoE costs more to train" objection, retired with arithmetic):** suppose the MoE program costs 2× the dense-70B training compute — say ~8e24 vs ~4e24 FLOPs including the AQT tax and proxy sweeps, order-of $50M extra at cloud-ish rates. A successful product serving 100B tokens/day at even $1/1M tokens is ~$36M/year of serving cost per 0.1× of the cost ratio; the ~0.5× serving advantage repays the training delta in months and then compounds for the model's lifetime. If the workload is *not* serving-dominant, this whole design brief changes — which is exactly why §1 pinned that assumption first.

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

*Path A — port and serve (the control arm).* A well-served dense 70B on Trillium (int8, good sharding, bucketed shapes) probably gets within ~1.3× of GPU cost. Zero training spend, but every architectural lever stays where the donor left it. This is the baseline every other path must beat.

*Path B — adapt the checkpoint (the path that's gotten real since 2025).* The claim "you can't change the architecture without retraining" is no longer true, and pretending otherwise loses credibility:
- **Sparse upcycling** (Komatsuzaki et al., arXiv 2212.05055): initialize MoE experts from copies of the dense FFN — the 70B's weights seed a ~8×70B-FFN MoE that recovers quality with a small fraction of from-scratch tokens. This directly reuses the initial weights.
- **MHA/GQA → MLA conversion**: TransMLA (arXiv 2502.07864) and MHA2MLA (arXiv 2502.14837) retrofit latent-KV attention onto existing checkpoints with low-single-digit-percent fine-tuning budgets — the KV bytes-per-token lever, post hoc.
- **QAT fine-tune with AQT** recovers most of the post-hoc-quantization quality tax.
- **Bolt-on speculation** (EAGLE-class heads) trains against the frozen model — the acceptance lever without touching the base.

So Path B recovers roughly three of my four levers at maybe 5–10% of the flagship training cost. What it *cannot* recover: the dense-activation bytes term only partially (upcycled experts start correlated — they differentiate slowly and rarely reach trained-from-scratch MoE quality-per-active-param), MXU-hostile donor shapes (head_dim, vocab), and the tokenizer.

*Path C — from scratch (the doc's mainline).* Full lever access, full cost.

*On distillation specifically* — the objection "the models aren't in the same family" is half right. **Logit-level** KD across families is genuinely awkward (vocabulary/tokenizer mismatch means the teacher's distribution doesn't align token-for-token; ULD-style tricks exist but are lossy). **Sequence-level** distillation — generating data with the teacher and fine-tuning the student on it — has no family requirement at all, and it's how cross-family distillation actually ships. The two-SKU design in §3.6 is same-family (flagship→Trillium SKU), where logit KD works cleanly; that's deliberate.

*The decision rule*, which is the actual staff signal: scratch-vs-adapt is a tokens-served amortization question. If the training delta is ~$40M and the from-scratch design saves an additional ~0.2×–0.3× on serving cost versus the best adapted model, the crossover sits at roughly 10¹⁴–10¹⁵ served tokens — months of Google-scale traffic, decades of a small deployment. **At Google volume, scratch wins and Path B is the de-risking stage** (upcycle + convert first, learn the serving numbers on real chips, then commit the flagship run). At startup volume, Path B *is* the answer and claiming otherwise is resume-driven engineering. Saying which side of the crossover you're on — and why — is the answer; picking a path without the crossover is the miss.

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
Then rerun §4 with active params scaled up to hit the quality gate — say 50B active instead of 35B. Bytes streamed rises ~1.4×, the relative cost lands around 0.7–0.8× instead of 0.55× on Trillium, and the program is still positive but no longer decisive; at that point the Ironwood bandwidth-per-dollar term and MLA's KV win carry the case. This is the answer's load-bearing property: no single factor carries the whole margin, and I can tell you the break-even point of each one.

**"The serving stack: what do you actually run?"**
`tpu-inference` — the unified JAX/vLLM TPU backend (Oct 2025), which is the currently supported path; JetStream was archived Feb 2026. It brings Ragged Paged Attention v3, continuous batching, and prefix caching. What we own on top: the MoE dispatch config, MTP verify scheduling, shape-bucket policy, and the disaggregated prefill/decode topology on Trillium. What we explicitly do not build: our own serving engine — that's a two-year detour to parity.

---

## 7. Staff+ Signals

What earns the level in this interview:

- **Driving scope before designing:** defining "quality-per-dollar" as gated-quality-then-cost, fixing the serving-dominant assumption, and choosing the chip *explicitly* — the interviewer should never have to supply the frame.
- **Arithmetic before architecture:** ridge points (560/313/626 FLOPs/byte), KV bytes per token (4 KB vs 1.15 KB), bytes streamed per decode pass (70 vs 45 GB), and a cost table with stated MBU and price assumptions — every design choice traced to a number, every number's derivation shown.
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

If the interviewer asks for the whole answer compressed: a dense 70B is the wrong shape for a machine whose ridge point is 300–600 FLOPs/byte, because it maximizes bytes streamed per token. The TPU-native design minimizes bytes at iso-quality along six axes — sparse activation (MoE, ~35B active), compressed KV (MLA, ~7× with int8), tile-aligned shapes (zero padding tax), quantization-aware training (int8/FP8 with no post-hoc quality lottery), co-trained speculation (~1.8× passes-to-tokens), and a chip-sized context strategy (Trillium SKU distilled from an Ironwood flagship). The compounded result is roughly 2–3× quality-per-dollar on serving, gated on sliced iso-quality evals, validated by MBU/acceptance/drop-rate instrumentation on a proxy-scale ladder before the flagship run, with every architecture bet carrying a numbered break-even point and a pre-committed fallback.
