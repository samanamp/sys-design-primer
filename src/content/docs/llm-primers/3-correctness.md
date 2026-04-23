---
title: "LLM Inference Systems: Correctness & Operability"
description: "LLM Inference Systems: Correctness & Operability"
---

### A staff-level interview primer

Companion to the distsys correctness/operability and multi-tenancy primers. This one is domain-specific: what changes when the system you're serving is a large language model.

---

## 1. What a staff engineer actually needs to know

For a staff-level inference / GPU-perf interview, you need fluent answers on:

- **KV cache** as a first-class resource (memory, coherence, eviction, routing)
- **Prefix caching** correctness and invalidation
- **Continuous batching** and its scheduling consequences
- **Speculative decoding** safety (rejection sampling correctness, determinism)
- **Streaming** client lifecycle, cancellation, mid-response failure
- **Rollout safety** for models, kernels, and quantization
- **Inference-specific SLIs/SLOs** (TTFT, ITL, goodput)
- **Load shedding** with unknown request cost
- **Multi-replica routing** under prefix affinity vs load balance

Skip at most panels: model architecture trivia, training-only parallelism concerns, convergence proofs. These don't show up in inference-serving interviews.

---

## 2. What makes LLM inference different

```
  Classical request-reply               LLM inference
  ─────────────────────                 ─────────────
  fixed cost / request                  cost varies 100× by tokens
  known latency budget                  TTFT + N × ITL, N unknown
  stateless / thin state                KV cache is heavy stateful memory
  cache is an optimization              cache is capacity planning
  retry = repeat same work              retry produces different output
  single-shot response                  streaming, mid-response failure
  model binary = code                   model binary affects quality
  rollout = code change                 rollout = quality + latency + memory
```

Implications for every design:
- Capacity planning is in **tokens/sec** and **KV-bytes-seconds**, not QPS
- SLOs have ≥ 2 dimensions: **TTFT** (prefill-bound) and **ITL** (decode-bound)
- Observability needs per-phase metrics, not just per-request
- Rollout must detect **quality regressions**, not only error rates
- Fairness must be in **token-work units**, not request counts

---

## 3. KV cache & memory management

### The role
Each token's attention K and V are cached per-layer. Recomputing is O(n²); caching makes decode O(n) per step. KV cache *is* the inference session's state. It is not an optional optimization; it is the capacity constraint.

### Memory math (interview-ready)
```
  KV bytes per token = 2 × n_layers × n_kv_heads × head_dim × dtype_bytes
```

Example — Llama-3-70B (`n_layers=80, n_kv_heads=8, head_dim=128`) at FP16:
`2 × 80 × 8 × 128 × 2 ≈ 320 KB/token`. One 1M-token request = ~320 GB of KV. That's why 1M-context is an architecture problem, not a config flag.

GQA and MQA shrink this by factor `n_q_heads / n_kv_heads`. MLA (DeepSeek) compresses further via latent projection. Always ask about the KV variant before doing math.

### Paged KV
Allocate KV in fixed-size blocks (pages), mapped via a per-request block table. Kills external fragmentation, enables sharing for prefix cache.

```
  Logical view (per request):   [t0 t1 t2 ... tN]
                                      │
                                 block table
                                      │
  Physical pool (shared):       [P0][P1][P2]...[Pk]
  each page holds B tokens of KV
```

### Correctness invariants for KV reuse
A cached KV entry is valid only for an exact match of:
- **model identity + weights version**
- **dtype / quantization**
- **position** in the sequence
- **preceding token sequence** (the prefix that produced it)

Any mismatch → must not reuse. Cross-tenant use is safe in principle (KV is a deterministic function of tokens + params) but creates timing side-channels and makes quota accounting messy.

### Eviction under pressure
Three choices when memory is tight:
1. **Reject** new admissions (backpressure — preferred)
2. **Preempt** a live request: stash KV to host RAM / disk, or drop entirely (work redone)
3. **Swap** cold pages of a running request

Interviewer wants the policy named, with the cost. *"LRU at the session level. Preempt the lowest-priority paused request first, swap KV to pinned host memory, swap back on resume with latency penalty of ~X ms per GB. Drop-and-redo as last resort."*

### Cache is local to a replica
A replica crash loses its KV. Failover means clients redo the request from scratch. Designs claiming "HA inference" without redundant KV are papering over this.

---

## 4. Prefix caching

Same KV pages can serve any request whose prefix is **identical in tokens, position, and model/dtype**.

```
  Req A: "You are a helpful assistant. User: what is 2+2?"
  Req B: "You are a helpful assistant. User: who wrote Hamlet?"
                                             │
              common prefix (system prompt) → same KV pages reused
                                             │
  Impl: radix / trie keyed on token IDs → block list
```

### Key schema (must include)
- token IDs (not text — tokenizers differ)
- model version (weights hash)
- dtype / quantization scheme
- positional-encoding params (RoPE base, scaling)
- any runtime flag that alters KV math

Miss any parameter → silent correctness bug. Users get garbled output.

### Invalidation
- Model weight reload → flush
- Quantization scheme change → flush
- Tokenizer version change → flush
- Usually treated as *full* flush; partial is rarely worth the complexity

### Failure modes
- **Silent corruption** if key ignores a parameter that affects KV (common shipping bug)
- **Cross-tenant leakage risk** if cache is tenant-scoped but metadata isn't
- **Timing side-channel**: cache hit vs miss latency can leak presence of a prefix. Relevant for adversarial tenants on shared replicas; mitigate via padding or tenant isolation.

### Routing interacts with caching
Cache is per-replica. Random load balancing destroys hit rate.

```
  Random LB                          Prefix-aware routing
  ─────────                          ────────────────────
  req → any replica                  req → hash(prefix) % N → replica
  cache hit rate ≈ 1/N               hit rate ↑, hotspot risk ↑

  Hybrid: consistent hashing on prefix + load-based override
    → primary replica by prefix
    → if primary over threshold, fall back to secondary (with cold prefix)
```

Staff answer: consistent hashing on prefix, bounded-load variant (Vimeo / Google "Consistent hashing with bounded loads" paper — name-drop is fine). Fallback routes pay cache miss cost; tolerable as long as it's a small fraction.

---

## 5. Continuous batching & scheduling

Static batching pads all requests to the same length and ends when the longest does. Continuous batching adds and removes requests **at each decode step**.

```
  Step t:    [R1 dec][R2 dec][R3 prefill ]
  Step t+1:  [R1 dec][R2 done→out][R3 dec][R4 prefill]
  Step t+2:  [R1 dec][R3 dec][R4 dec][R5 prefill]
```

### Per-step scheduling decisions
- **Prefill vs decode mix.** Prefill is compute-bound; decode is memory-bandwidth-bound. Mixing helps utilization but hurts TTFT for new arrivals if decode dominates the step.
- **Chunked prefill.** Split long prefill across steps so decodes aren't starved. Production default for anything >~2K input.
- **Admission.** Admit new request when KV pages available and compute budget allows.
- **Preemption.** Under memory pressure, pause (swap or drop) a request.

### Prefill/decode scheduling tradeoffs
```
  Decode-only batch:                Prefill-heavy batch:
  - low TTFT for existing           - high throughput
  - no new arrivals served          - new requests enter fast
  - GPU under-utilized               - existing decodes stall (high ITL)

  Chunked prefill:
  - small prefill chunk per step alongside decodes
  - balances TTFT and ITL
  - more steps per prefill but each step has full batch
```

### Fairness concerns (see fairness primer)
- FCFS starves short requests behind long generations
- Preemption cost is real (loss of GPU slot + swap-in latency)
- Per-tenant budgets must be in **token-work units**

### Correctness invariants per step
- Token order strictly preserved per request
- Position indices monotonic, no gaps
- Sampling RNG state per-request, never shared
- Stop conditions (EOS, max tokens, stop strings) checked every step
- Cancellation checked every step (or every N steps) to free resources promptly

---

## 6. Speculative decoding safety

Draft model proposes K tokens; target model verifies in one forward pass; accepted tokens commit, rejected tokens discarded with a resample.

### The correctness property

**Modified rejection sampling**:
```
  For each draft token x with draft prob q(x), target prob p(x):
    accept with probability min(1, p(x)/q(x))
    on rejection: sample from normalized (p - q)+ distribution
```

When implemented correctly, the **output token distribution equals the target model's distribution exactly**. This is what makes speculative decoding "lossless."

### Where implementations go wrong
- `accept if p(x) > q(x)` (no stochasticity) → biased distribution
- Forgetting the residual resample on rejection → biased
- Computing `p(x)/q(x)` in linear space with fp16 → numerical bias on small probabilities. **Do it in log space.**
- Draft and target with different tokenizers → undefined; must share tokenizer
- Draft with different sampling temperature than target → broken; match configs
- Top-k / top-p applied inconsistently between draft and target → subtly biased

### Determinism
Common interview question: *"Is speculative decoding deterministic?"*

Answer framework:
- **In distribution**: yes, output matches target exactly.
- **Bit-exact vs non-spec**: generally no — rejection consumes extra RNG entropy, kernel paths differ, reduction orders differ.
- **Deterministic across runs with same seed and same spec config**: achievable but requires careful RNG management.

Don't promise seed-level equivalence across spec-on and spec-off.

### Operational signals
- **Acceptance rate** per (draft, target) pair. Falls → draft/target divergence, tokenizer drift, distribution shift. Auto-disable below a threshold.
- **Wasted compute**: target ran; rejected tokens are cost without throughput. Cost metric worth tracking.
- **Tail latency**: spec helps mean latency, can hurt p99 if draft is slow or rejection rate spikes. Monitor both.

### Multi-tenancy interaction
Spec decoding consumes target compute even on rejection. Under contention, spec can *hurt* aggregate throughput. Dynamically disable when utilization > threshold. Run drafts on a separate pool from targets so they don't compete.

---

## 7. Streaming & client lifecycle

Streaming introduces correctness concerns that classical request/response sidesteps.

### Client disconnect mid-stream
Must detect promptly and cancel generation. Otherwise:
- KV pages stay pinned
- Batch slot occupied
- Billing/metering for work the user never received

Detection: TCP keepalive + HTTP/2 RST_STREAM handling + periodic liveness check in the generation loop.

### Cancellation path
```
  client disconnect
        │
        ▼
  edge proxy signals cancel
        │
        ▼
  scheduler marks request cancelled
        │
        ▼
  generation loop checks at next step
        │
        ▼
  free KV pages, remove from batch, close billing window
```

Race: tokens generated between detection and cancel are charged/not per your policy. Document it.

### Retries and idempotency
A "regenerate" is **not idempotent** unless seed, temperature, and all sampling params are fixed — and even then, kernel nondeterminism can break bit-exact reproduction. For billing, meter per-token-delivered, not per-request-accepted.

### Partial structured output
If output is JSON / function call / tool invocation, partial output is invalid at intermediate states. The protocol must signal "in progress" vs "complete"; clients should not parse partial as final. Designs that stream partial JSON to clients need this invariant documented.

### Mid-response failures
Replica crash mid-stream: client sees truncated response. Options:
- **Resume**: rare, requires KV replication (expensive)
- **Restart**: easy, produces different output with stochastic sampling
- **Error**: simplest, pushes recovery to client

Most production systems restart or error; KV replication for resume is almost never worth it.

---

## 8. Rollout: models, kernels, quantization

A traditional rollout monitors latency and errors. LLM rollouts must also monitor **output quality** and **cost per token**. Missing either = regressions that ship silently.

### Model rollouts

```
  Phase 1: OFFLINE EVAL
    eval harness (MMLU, HumanEval, internal task sets)
    held-out sets, adversarial sets
    regression thresholds per metric

  Phase 2: SHADOW TRAFFIC
    serve old model, run new in parallel, log outputs
    compare: latency, cost, quality (auto-grader)
    no user impact

  Phase 3: CANARY
    route 1% to new model
    monitor: TTFT, ITL, goodput, refusal rate,
             user proxy metrics (regenerate rate, thumbs),
             auto-graded quality delta

  Phase 4: RAMP with guardrails
    5% → 25% → 50% → 100%
    auto-rollback on any of:
      latency SLO burn
      quality SLI regression (> ε)
      cost/token regression

  Phase 5: POST-RAMP
    feature flag stays hot for N days
    old model weights retained for fast rollback
```

Quality SLI options:
- Pairwise preference via LLM-as-judge (cheap, noisy, directionally reliable)
- Task-specific auto-graders (higher signal, narrower coverage)
- User proxies: regenerate rate, conversation length, explicit feedback
- Sampled human eval for high-stakes launches

### Kernel rollouts

A new attention / MoE / GEMM kernel must match reference within tolerance. "Within tolerance" is where staff judgment shows.

Validation gates:
1. **Unit numerical parity.** Compare vs reference on a curated input suite. Thresholds: max absolute error, relative error, cosine similarity. Typical BF16 tolerance: `max_abs_err < 1e-2`, `cos_sim > 0.999`. For FP8, widen by dtype ULPs (FP8 E4M3 ULP at 1.0 is 0.125; absolute tolerance must reflect this).
2. **Model numerical parity.** End-to-end eval with new kernel. Perplexity, task metrics within noise band vs reference.
3. **Performance regression gate.** p50/p99 latency and throughput within threshold.
4. **Adversarial inputs.** Denormals, overflow-prone sequences, empty/very-long tokens.
5. **Shadow traffic + canary** with quality and latency SLIs.

Feature flag per kernel. Old kernel resident for rollback. Per-layer flags if the kernel is partial.

### Quantization rollouts (FP8, INT8, MXFP4)

Everything from kernel rollout, plus:
- **Calibration dataset**: scales fitted to data may mismatch production distribution. Monitor outlier rates per layer in prod; re-calibrate if drift.
- **Per-layer sensitivity analysis**: not all layers tolerate the same precision. Adaptive per-layer precision (the practical production pattern for FP8) keeps quality while capturing most of the speedup.
- **Corner cases**: denormal handling, overflow on attention scores, softmax stability.
- **E4M3 vs E5M2 choice**: E4M3 for activations/weights (more precision, less range), E5M2 for gradients (more range). Getting this wrong silently degrades quality.

### Rollback properties
- Model binaries hot-swappable; weights reload without process restart ideally
- Prefix cache flushed on model swap
- Feature flag per model version; traffic splittable at 1% granularity
- Old version kept resident for ≥ N days
- Rollback is a **button**, not a redeploy

---

## 9. Observability for inference

### SLIs (what to measure)

| Signal | What it tells you |
|---|---|
| **TTFT** (time to first token) | Prefill capacity + queue time |
| **ITL / TPOT** (inter-token latency) | Decode capacity, batch pressure, KV pressure |
| **End-to-end latency** | TTFT + N × ITL; depends on output length |
| **Tokens/sec** (per request, aggregate) | Throughput |
| **Goodput** (tokens/sec meeting SLO) | True user-facing throughput |
| **Queue depth + admission wait** | Congestion |
| **KV cache utilization** | Memory pressure |
| **Prefix cache hit rate** | Routing + cache effectiveness |
| **Batch size distribution** | Utilization balance |
| **Spec decoding acceptance rate** | Quality + speed indicator |
| **Preemption rate** | Scheduler stress |
| **Refusal / safety filter rate** | Behavior regression |

**Goodput** is the metric most candidates miss. Raw throughput can hide SLO violations; goodput is tokens/sec delivered *within SLO*.

### SLOs (what to promise)
```
  TTFT p99 < 500ms
  ITL p99 < 50ms  (at nominal context length / batch size)
  Availability (2xx, non-cancelled) > 99.9%
  Goodput ≥ X tokens/sec at load Y
  Quality: | eval metric − reference | ≤ ε
  Cost/token: within Z% of baseline
```

Context-length tiers get separate SLOs (8K, 32K, 128K, 1M) because latency scales with context.

### Dashboards
Top row: TTFT, ITL, goodput, availability — the four vital signs.
Middle: batch composition, queue depth, KV utilization, prefix cache hit.
Bottom: per-replica tail latency, preemption, spec acceptance, per-tenant SLOs.

### Alerting (symptom-based)
- TTFT SLO burn → prefill capacity / admission issue
- ITL SLO burn → decode capacity / batch too large / KV pressure
- Goodput drop at stable QPS → performance or quality regression
- Spec acceptance drop → draft drift or tokenizer mismatch
- Prefix cache hit drop → routing degraded or cache flushed
- Quality SLI regression → model or kernel regression; page

---

## 10. Failure modes catalog

| Failure | Signal | Mitigation |
|---|---|---|
| **OOM on long context** | Admission rejection + KV util spike | Context tiers; chunked prefill; upstream reject |
| **Runaway generation** | Output-length p99 drifting | Max-tokens + stop-string enforcement; wall-clock timeout |
| **Poison prompt (infinite loop)** | Single request dominates batch | Per-request token + wall-clock budget; forced stop |
| **Prefix cache corruption** | Garbled output, user reports | Strict key schema; flush on any param change; add param to audit |
| **Kernel numerical drift** | Eval delta on canary | Numerical parity gate; auto-rollback; per-layer flag |
| **Draft model drift** | Spec acceptance drop | Alert + auto-disable spec; retrain / swap draft |
| **Hot replica (prefix affinity)** | Load imbalance, hot replica p99 | Bounded-load consistent hashing; cache warming; split hot prefix |
| **Stream leak** | Active-stream count growing | Idle timeout; liveness probe; cancellation path audit |
| **Quantization outlier** | Eval spikes on narrow input distribution | Per-layer sensitivity; representative calibration; fallback precision |
| **Tokenizer mismatch** | All decodes wrong | Version tokenizer with weights; reject load on mismatch |
| **Batch-size collapse** | ITL spikes, utilization drops | Chunked prefill; admission tuning; hot standby capacity |
| **Routing flap** | Cache hit rate oscillates | Dampening / hysteresis on load-based overflow |

---

## 11. Interview reasoning patterns

**"Design an LLM inference serving system for X QPS with N-token context."**
Open with capacity math. KV bytes/token → concurrent sessions at memory bound. Prefill FLOPs and decode FLOPs at compute bound. Replicas per model. Then: continuous batching, paged KV, prefix cache with consistent-hash routing, speculative decoding if latency-critical, chunked prefill. Then: per-tenant token-work admission, DRR scheduling, preemption, load shed at admit. Then: observability (TTFT, ITL, goodput per tier), rollout (model + kernel canary with quality SLI), blast radius (cells per context-length tier + per-tenant tier).

**"How do you safely roll out a new attention kernel?"**
Numerical parity (unit + model level) vs reference with explicit tolerances. Perf regression gate. Feature flag per kernel. Shadow traffic comparing outputs bit-by-bit. Canary 1% with TTFT / ITL / goodput / quality SLIs. Auto-rollback on any SLI breach. Old kernel resident for N days. Name the tolerance numbers (BF16: `cos_sim > 0.999, max_abs < 1e-2`).

**"How do you scale KV cache to 1M context?"**
Paged KV. Offload cold pages to host memory / NVMe. KV compression: token eviction (H2O, StreamingLLM), low-rank / MLA. Tiered cache: HBM → host → disk. Ring / sequence parallelism across GPUs for the monster contexts. Per-context-length admission tier with separate pools. Cost-per-token rises steeply; pricing should reflect. Accept that 1M is a different product, not a bigger version of 8K.

**"Speculative decoding in multi-tenant — concerns?"**
Spec consumes target compute on rejection; under contention it can reduce aggregate throughput. Per-tenant toggle. Auto-disable above utilization threshold. Draft on separate pool from target. Monitor acceptance rate; alert on drift. Correctness: modified rejection sampling, log-space probabilities.

**"One tenant sends a 1M-context request and hurts everyone."**
Per-tenant token-work budget at admit. Context-length tiers with separate pools. Chunked prefill so the giant request doesn't starve decodes. Preemption with swap-to-CPU if a latency-critical request arrives. Long-term: price differently, isolate to a dedicated pool/cell.

**"How do you detect a quality regression from a model change?"**
Offline eval gates before deploy. Shadow traffic with auto-graded comparison. Canary with quality SLI + user proxy metrics (regenerate rate, thumbs). Alert on SLI delta > ε over rolling window. Auto-rollback wired to the same feature flag as the deploy. For high-stakes launches, sampled human eval.

**"A request hangs. Everything else slows. Diagnose."**
Traces first: find request ID with open span. Check if stuck in prefill chunk, waiting on tool call, pinned to a KV allocation, or spinning in a generation loop that never hits stop. Mitigation: wall-clock and max-token guardrails, per-step cancel checks, kill path. Long-term: budgets enforced before batch entry.

---

## 12. Cheat sheet

### Inference correctness checklist
- [ ] KV cache key includes model / weights / dtype / position / tokens
- [ ] Prefix cache flushed on any param that affects KV math
- [ ] Tokenizer version bound to weights version at load time
- [ ] Spec decoding uses modified rejection sampling in log space
- [ ] Sampling RNG per-request, never shared across batch
- [ ] Stop conditions checked every step (EOS, max tokens, stop strings, cancel)
- [ ] Cancellation path frees KV + updates billing window
- [ ] Eviction policy named, preemption cost accounted
- [ ] Numerical parity validated for any kernel/quant change

### Inference operability checklist
- [ ] TTFT + ITL + goodput SLOs **per context-length tier**
- [ ] Per-phase metrics (prefill vs decode)
- [ ] Quality SLI in rollout pipeline (not just error rate)
- [ ] Numerical parity gate for kernel changes (tolerances stated)
- [ ] Prefix-aware routing (consistent hash + bounded load)
- [ ] Per-tenant token-work admission control
- [ ] Chunked prefill for long contexts
- [ ] Feature flag per model / per kernel / per layer
- [ ] Auto-rollback on any of: latency, quality, cost SLI burn
- [ ] Blast radius: cells per context tier × tenant tier
- [ ] Old model / kernel resident for fast rollback

### Decision framework: prefix routing vs load balance

```
  Prefix-aware routing (cache hit ↑)     Random / pure LB (balance ↑)
  ─────────────────────────────────       ─────────────────────────────
  ✓ high hit rate                         ✓ perfect load balance
  ✓ low prefill cost                      ✗ low hit rate ≈ 1/N
  ✗ hotspot risk                          ✗ high prefill cost
  ✗ imbalanced utilization                ✓ no hotspot

  Production choice: consistent hashing on prefix + bounded-load override.
  Primary = hash(prefix) % N.
  If primary utilization > threshold → fall back to next replica (pay cache miss).
```

### 10 likely interview questions + strong short answers

**1. "What's in the KV cache and why does it matter?"**
Per-layer K and V projections for every past token. Matters because it turns decode from O(n²) recompute to O(n) per step and because it is heavy GPU memory (100s of KB/token at scale). KV memory is the first capacity constraint — often before FLOPs. Architecture choice (MHA / GQA / MQA / MLA) directly affects how much inference a GPU can serve.

**2. "Design prefix caching across replicas."**
Per-replica radix tree keyed on (token IDs, model version, dtype, positional params). Edge router hashes the prefix; consistent-hash routes to primary replica. Bounded-load override: if primary saturated, fall back to next replica (cache miss). Page refcounts; evict LRU on pressure. Flush on model swap. Metric: cluster-wide hit rate + per-replica balance.

**3. "How does continuous batching affect tail latency?"**
Great for throughput, can hurt p99 TTFT when a long prefill blocks decodes of other requests in the same step. Mitigation: chunked prefill so prefill cost per step is bounded; prefill/decode ratio controller; admission gate on decode-budget availability. Without chunked prefill, a single 32K-input request can freeze decodes for hundreds of ms.

**4. "Is speculative decoding lossless?"**
In distribution yes — output token distribution matches target exactly when rejection sampling is modified correctly (accept with `min(1, p/q)`, resample from `(p - q)+` on reject). Bit-exact vs non-spec, no — RNG consumption and numerical paths differ. Never promise seed-level determinism across spec-on/spec-off. Monitor acceptance rate; auto-disable if it drops.

**5. "How do you do admission control when request cost is unknown?"**
Estimator (heuristic or small model) predicts total tokens from input. Admit using `estimated_cost + margin`. Meter actual cost during execution; enforce hard cap if estimate busts by factor k. Reconcile estimator calibration continuously (per-tenant if distributions differ). DRR deficit updated with *actual* cost.

**6. "Canary for a new model version — what do you watch?"**
TTFT / ITL / goodput vs baseline. Cost per token. Refusal / safety-filter rate. Quality SLI via auto-grader (pairwise preference) and/or task metrics. User-proxy signals: regenerate rate, conversation length, explicit feedback. Auto-rollback on any of latency / quality / cost SLI breach.

**7. "A request hangs. Everything else slows. Diagnose."**
Traces: find request ID with open span. Localize: stuck in prefill chunk, waiting on tool/RAG call, KV pinned, stop condition never hit (poison prompt). Mitigate: wall-clock + max-token guardrails, per-step cancel checks, kill path. Fix root cause: budget enforcement before batch entry.

**8. "How would you serve 1M-context on current GPUs?"**
Paged KV + offload (host RAM, NVMe). KV compression (H2O, StreamingLLM, MLA). Ring or sequence-parallel attention. Tiered cache HBM → host → disk. Context-length tier in its own pool. Per-context-length admission. Cost-per-token explodes; price accordingly. Treat 1M context as a distinct product.

**9. "How do you roll out an FP8 quantized model safely?"**
Per-layer sensitivity analysis first — identify layers that need BF16. Numerical parity vs BF16 on diverse inputs (max abs error + cos sim per layer, eval metric delta end-to-end). Shadow traffic comparing outputs. Canary 1% with latency + quality + cost SLIs. Feature flag per-layer precision. Old path resident for rollback. Watch for calibration drift in prod; re-calibrate if outlier rate spikes.

**10. "One tenant's prefix cache is 80% of memory. Problem?"**
Maybe not — if it's a shared system prompt across their users, high hit rate is desired behavior. Add per-tenant quota (max pages or bytes-seconds) so it's bounded. Evict that tenant's pages under pressure before global eviction. Count shared pages proportionally for billing. If genuinely harmful, separate cache namespace per tenant with weighted quotas.

---

One-line mental summary:

> *KV is state, not cache. Continuous batching couples tenants. Fairness is in token-work. Correctness is identical output distribution. Rollouts change quality — watch for it.*