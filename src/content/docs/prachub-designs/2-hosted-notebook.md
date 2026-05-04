---
title: Hosted Notebook Platform
description: Hosted Notebook Platform
---

# Designing a Hosted Notebook Platform — Staff Answer

## 1. Research pass — state of the art (2026)

Hosted notebooks have bifurcated into two architectural philosophies. The classic **persistent-kernel notebook** (Hex, Deepnote, Databricks, Colab, Kaggle, Jupyter Hub) keeps a long-lived kernel process holding cell state in RAM; the kernel is the system's center of gravity, and the design problems are sandboxing, idle compute, state preservation, and collaboration. The newer **ephemeral-sandbox notebook / agent execution surface** (Modal Sandboxes, OpenAI Code Interpreter, E2B, Daytona, AWS Bedrock AgentCore) treats compute as per-task disposable: an LLM produces code, a fresh sandbox runs it, results return, sandbox dies. State, if needed, lives in mounted volumes.

The execution substrate has standardized around three options. **Container/runc** is cheapest and ~1s cold; runc CVEs in 2024–2025 (CVE-2024-21626 and follow-ons) and the container-escape pattern at Ona — where a Claude Code agent traversed `/proc/self/root` to disable its own sandbox — have moved the industry away from raw containers for untrusted code. **gVisor** (user-space kernel, syscall interception) is what OpenAI Code Interpreter uses behind FastAPI-on-Kubernetes; it's also Modal's current default for Sandboxes. **Firecracker microVMs** boot in ~125ms cold, restore from snapshot in 28–200ms, and provide hardware-virtualization isolation; AWS Lambda, Fly.io, and AWS Bedrock AgentCore use them. Modal added GPU memory snapshotting (alpha, 2025) capturing VRAM + CUDA context for ~10× faster GPU cold starts.

**Collaboration** has standardized on Yjs (CRDT) over WebSockets for *editing*; almost no platform does collaborative *execution* against a shared kernel, because the UX of "your variables changed because someone else ran a cell" is bad. Hex and Deepnote both use single-executor semantics over a CRDT-synced document.

**Idle-compute economics** are the dominant operational pressure. A user opens a notebook, runs three cells, walks away. The kernel holds 1–4 GiB. At 100K idle kernels that's 100–400 TiB of RAM. Modal's `enable_memory_snapshot` (CRIU-derived) and Firecracker's MAP_PRIVATE-backed snapshot restore are both responses to this; aggressive hibernation has become table stakes.

**AI integration** has reshaped the requirements: cold-start matters more (every agent step), isolation matters more (LLM-generated code can be malicious by accident), tighter resource caps matter (LLMs hallucinate `while True` loops). E2B scaled from 40K → 15M sandboxes/month in one year; the agent pattern is now the dominant new workload.

---

## 2. Scoping — the most important section [STAFF SIGNAL: scoping-as-staff-signal]

"Hosted notebook platform" maps to at least four distinct products with very different architectural pressures:

- **A. Free-tier individual notebook (Colab/Kaggle).** Massive scale of free anonymous users, GPU is the hero, isolation against actively malicious code is critical, idle eviction is aggressive. Compute cost vs ad/marketing-funnel ROI is the central business constraint.
- **B. Prosumer/SMB collaborative data workspace (Hex, Deepnote).** Logged-in paying teams, Python+SQL, real-time collaborative editing, dashboards/data apps as a deliverable, AI assistance integrated. Sandboxing is moderate trust (paying users), collaboration and state are the hard problems, GPU is a nice-to-have.
- **C. Enterprise-integrated notebook (Databricks Notebooks).** Coupled to a Spark/Delta/warehouse engine, the notebook is a thin shell over a much heavier compute platform. The hard problems live in the engine, not the notebook.
- **D. Ephemeral agent sandbox / Code Interpreter as a primitive.** Per-request fresh execution environment, no persistent kernel, optimized for LLM agent loops. Cold-start and isolation are everything; collaboration doesn't exist.

I am committing to **scope B: a prosumer/SMB collaborative data workspace, Python + SQL, with an AI-agent overlay along the Code Interpreter pattern.** Reasoning:

1. **It maximizes architectural surface area.** B forces every classical problem — sandboxing, idle compute, kernel state, collaboration, GPU, multi-tenancy — *and* it requires reasoning about the AI-integration overlay that will dominate the next 3 years.
2. **The customer is well-defined.** Data teams in 10–500-person companies. SSO, audit, RBAC matter. Workloads are 80% pandas/duckdb/SQL, 15% sklearn/XGBoost/inference, 5% small-GPU work.
3. **Trust gradient is interesting.** Paid logged-in users (warmer trust) plus AI-generated code in their sandboxes (colder trust) plus enterprise BYOC tenants (must be hard-isolated) — the *same codebase* must serve all three.

What I am **not** building: a Colab-class anonymous free tier; a Spark engine; a general-purpose serverless compute platform; a desktop IDE. I am also not building **collaborative execution** (multiple users running cells against one shared kernel state) — I will support collaborative *editing* with single-executor semantics, and explain why.

**Target scale:** 500K registered users, 50K daily-active, 20K concurrent notebooks at peak, 3K kernels actively executing at any instant (vs idle), 200 concurrent GPU notebooks. Workload: P50 cell = 50ms, P99 = 30s, long-tail to hours. Median kernel: 2 GiB RAM, 2 vCPU. P95 kernel: 8 GiB, 4 vCPU.

---

## 3. Capacity and cost math [STAFF SIGNAL: capacity math]

| Quantity | Value | Source / reasoning |
|---|---|---|
| Concurrent notebooks (peak) | 20K | scoping target |
| Concurrent *executing* kernels | 3K (15%) | empirical: notebooks idle most of the time |
| Median kernel RAM | 2 GiB | data-science workload, pandas + a model |
| Naive RAM if all kept hot | 40 TiB | 20K × 2 GiB |
| RAM with hibernation (idle → snapshot) | ~6 TiB hot + 34 TiB on NVMe | hot = executing; cold compresses ~3×, costs ~50× less |
| Hot-RAM cost @ $5/GiB-month | $30K / month | 6,000 × $5 |
| Snapshot storage @ $0.10/GiB-month (NVMe-tier) | ~$1.1K / month | 34,000 × ~30% post-compress × $0.10 |
| Naive monthly cost | $200K | 40,000 × $5 |
| **Hibernation savings** | **~85%** | $200K → ~$31K |
| GPU notebooks @ 200 concurrent × $2/GPU-hr × 730 hr | $292K / month | if always-on |
| GPU with 5-min idle eviction | ~$100K / month | empirical: 30–40% effective duty cycle |
| Cold-start budget — CPU notebook | <2s warm-pool, <8s cold image | warm pool of pre-booted gVisor sandboxes |
| Cold-start budget — GPU notebook | <15s with GPU snapshot, ~45s cold | mirrors Modal's 2025 GPU snapshot path |
| Execute-request platform overhead | <100ms p99 | excludes user-code time |

The dominant cost is hot RAM, and hibernation is the only thing standing between the platform and a 6× cost overrun. **[STAFF SIGNAL: idle-compute-cost-as-central]** Idle compute is not a footnote; it is the central business pressure that forces hibernation, which forces snapshot/restore, which forces every downstream design decision about kernel state.

---

## 4. High-level architecture

```
                        ┌────────────────────┐
                        │  Browser (React)   │
                        │  Yjs CRDT client   │
                        └──────────┬─────────┘
                                   │ WebSocket (TLS)
                                   ▼
                ┌──────────────────────────────────┐
                │  Edge Gateway / Auth (envoy)     │
                │  AuthN, AuthZ, rate-limit, SSO   │
                └──────┬─────────────────┬─────────┘
                       │                 │
        Doc / collab   │                 │   Execute / kernel I/O
                       ▼                 ▼
            ┌──────────────────┐  ┌──────────────────────┐
            │  Doc Sync Svc    │  │  Kernel Gateway      │
            │  Yjs server,     │  │  routes execute_req  │
            │  presence, RBAC  │  │  to kernel by id     │
            └──────┬───────────┘  └──────────┬───────────┘
                   │                         │
                   ▼                         ▼
           ┌──────────────┐         ┌─────────────────────┐
           │  Doc Store   │         │  Kernel Scheduler   │
           │  Postgres +  │         │  placement, idle    │
           │  S3 (ops log)│         │  eviction, GPU      │
           └──────────────┘         └─────────┬───────────┘
                                              │
              ┌───────────────────────────────┼────────────────────────────┐
              ▼                               ▼                            ▼
    ┌──────────────────┐         ┌──────────────────────┐      ┌──────────────────┐
    │  Kernel Pool A   │         │   Kernel Pool B      │      │   GPU Pool       │
    │  gVisor on Kata  │         │   Firecracker μVMs   │      │   Firecracker +  │
    │  paid trusted    │         │   AI-agent / free /  │      │   GPU passthrough│
    │  CPU notebooks   │         │   BYOC enterprise    │      │   MIG-sliced     │
    └────────┬─────────┘         └──────────┬───────────┘      └────────┬─────────┘
             │                              │                           │
             ▼                              ▼                           ▼
   ┌────────────────────────────────────────────────────────────────────────────┐
   │  Snapshot store (NVMe-cached S3)   │   User Volumes (per-user EBS / EFS)   │
   └────────────────────────────────────────────────────────────────────────────┘

   ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
   │  Billing svc    │    │  Telemetry      │    │  AI-agent svc   │
   │  per-second     │    │  per-kernel     │    │  LLM proxy +    │
   │  metering       │    │  metrics → TSDB │    │  tool dispatch  │
   └─────────────────┘    └─────────────────┘    └─────────────────┘
```

Key invariants. **[STAFF SIGNAL: invariant-based thinking]**

- **The notebook document is durable; the kernel is ephemeral.** The document (cells, outputs, metadata) lives in Postgres + S3 ops log. The kernel state may be lost at any moment; we mitigate but do not promise.
- **One kernel == one notebook == one tenant.** Never multiplex tenants on a kernel. Resource limits enforce a hard ceiling per kernel.
- **The execute path is single-writer.** A kernel processes one execute_request at a time. Parallel cells from collaborators serialize into a queue with FIFO + per-user fairness.
- **Cost is bounded by the scheduler, not by user behavior.** A kernel cannot exceed its tier's CPU / RAM / runtime budget; a user cannot exceed monthly compute quota.

The split between Pool A (gVisor) and Pool B (Firecracker) is the most consequential placement decision; section 5 explains.

---

## 5. Sandboxing and execution substrate [STAFF SIGNAL: sandboxing-as-central]

This is the architectural decision the rest of the system bends around. The candidates:

| Substrate | Cold start | Per-kernel mem overhead | Isolation strength | Notes |
|---|---|---|---|---|
| Plain Docker / runc | ~0.5–1s | ~50 MiB | weak (shared host kernel) | container-escape CVEs in 2024–25; not acceptable for AI-generated or anonymous code |
| gVisor | ~1s | ~80 MiB | medium (user-space kernel) | OpenAI Code Interpreter, Modal Sandboxes use this; ~5–15% perf overhead |
| Firecracker μVM | 125ms boot, ~28–200ms snapshot-restore | ~5–200 MiB depending on guest | strong (KVM + hardware virt) | AWS Lambda; supports memory snapshots with MAP_PRIVATE COW |
| Kata Containers | ~1–2s | ~200 MiB | strong (lightweight VM around container) | OCI-compatible; useful when you want k8s ergonomics with VM isolation |
| Pyodide / WebAssembly | client-side, ~1–3s | 0 server-side | strong (browser sandbox) | no real GPU, no full Python ecosystem at native speed; right for educational/embedded uses, wrong here |

**Decision:** dual-substrate, placed by the scheduler.

- **Pool A — gVisor on Kata, for paid, logged-in, human-driven CPU notebooks.** Trust gradient is favorable (logged-in, paying, MFA). gVisor handles the bulk of runtime defense; Kata adds a VM boundary for the runc-CVE blast-radius case. Cold start ~1s, memory overhead ~150 MiB, integrates cleanly with Kubernetes. **Rejected alternative: plain runc** — the per-kernel overhead saving (~100 MiB × 20K = 2 TiB) is real but does not justify the escape risk after the Ona-style incidents. **[STAFF SIGNAL: rejected alternative]**
- **Pool B — Firecracker μVMs, for AI-agent execution, free-tier (if we add one), and enterprise BYOC.** Strong isolation, snapshot/restore is first-class, hardware-virt boundary lets us co-tenant aggressively. Firecracker's snapshot-restore is the only realistic path to <50ms warm-pool restore for the AI-agent loop. **Rejected alternative: Firecracker for everything** — μVM overhead and operational complexity (kernel images, virtio device plumbing, harder GPU passthrough) make it the wrong default for the high-volume case of a single user editing a CPU notebook. **[STAFF SIGNAL: rejected alternative]**
- **Pyodide is not on the path.** We considered it for the free-tier marketing surface (run a tutorial notebook with zero server compute). Rejected: our scoped users are doing real data work; pandas/duckdb/sklearn at WASM speeds is a strictly worse product, and the ecosystem fragmentation is not worth the cost saving.

```
                Sandboxing topology

  ┌────────────────────────────────────────────────────┐
  │  Bare-metal host (e.g. m7i.metal-48xl)             │
  │                                                    │
  │   ┌──────────────┐     ┌──────────────┐            │
  │   │ Kata VM      │     │ Kata VM      │            │
  │   │  ┌────────┐  │     │  ┌────────┐  │            │
  │   │  │gVisor  │  │     │  │gVisor  │  │  …         │
  │   │  │ Kernel │  │     │  │ Kernel │  │            │
  │   │  └────────┘  │     │  └────────┘  │            │
  │   └──────────────┘     └──────────────┘            │
  │                                                    │
  │   ┌────────────────────────────────────────────┐   │
  │   │  Firecracker control plane (jailer + VMM)  │   │
  │   │  ┌───────┐  ┌───────┐  ┌───────┐           │   │
  │   │  │ μVM 1 │  │ μVM 2 │  │ μVM N │  …        │   │
  │   │  │ guest │  │ guest │  │ guest │           │   │
  │   │  │kernel │  │kernel │  │kernel │           │   │
  │   │  └───────┘  └───────┘  └───────┘           │   │
  │   └────────────────────────────────────────────┘   │
  │                                                    │
  │   Network: per-VM tap + per-tenant SG; egress      │
  │   gated by L7 proxy; DNS through tenant-scoped     │
  │   resolver. No tenant ↔ tenant L3 reachability.    │
  └────────────────────────────────────────────────────┘
```

Resource limits are belt-and-suspenders: cgroup v2 caps inside the guest, plus a μVM-level hard ceiling. A misbehaving kernel hits its OOMkiller before the host scheduler ever sees it.

---

## 6. Kernel lifecycle and idle compute [STAFF SIGNAL: state-management discipline] [STAFF SIGNAL: cold-start awareness]

```
                                      Kernel lifecycle state machine

   ┌──────────┐  user opens nb     ┌────────────┐   exec_req      ┌──────────┐
   │ ABSENT   │ ─────────────────▶ │ PROVISION  │ ──────────────▶ │ RUNNING  │ ◀┐
   └──────────┘                    │  (warm     │                 └────┬─────┘  │
                                   │  pool hit) │                      │        │
                                   └────────────┘                      │        │
                                          │                            │ no reqs│
                                  cold (miss)                          │ for 10m│
                                          ▼                            ▼        │
                                   ┌────────────┐               ┌──────────┐    │
                                   │ COLD-BOOT  │               │  IDLE    │    │
                                   │ (image     │               │  (still  │    │
                                   │  pull,     │               │   hot)   │    │
                                   │  μVM init) │               └────┬─────┘    │
                                   └─────┬──────┘                    │          │
                                         │                       30m no reqs    │
                                         └──▶  RUNNING ◀─────────    │          │
                                                                     ▼          │
                                                              ┌────────────┐    │
                                            user returns      │ HIBERNATE  │    │
                                          ◀───────────────────┤ (snapshot  │    │
                                          (RESTORE <2s)       │  to NVMe)  │    │
                                                              └─────┬──────┘    │
                                                                    │           │
                                                              7d no activity    │
                                                                    ▼           │
                                                              ┌──────────┐      │
                                                              │ EVICTED  │      │
                                                              │ (snapshot│      │
                                                              │ deleted) │      │
                                                              └────┬─────┘      │
                                                                   │            │
                                                            user returns        │
                                                                   ▼            │
                                                              ┌──────────┐      │
                                                              │ FRESH    │──────┘
                                                              │ KERNEL   │
                                                              └──────────┘
```

The flow:

1. **Warm pool.** We keep N pre-booted Firecracker μVMs and gVisor sandboxes per region with the standard data-science image already loaded (pandas, numpy, duckdb, sklearn, pyarrow, requests). N is autoscaled from a 30-minute trailing P95 of new-kernel rate. Hit ratio target: 95% on paid CPU; 85% on GPU. **[STAFF SIGNAL: blast radius reasoning]** — the warm pool also absorbs the "popular template thundering herd" case where 1,000 users open the same template at once: admission control queues with backpressure once the pool is below 20% headroom.
2. **Idle detection.** No execute_request for 10 minutes → **IDLE**. The kernel is still resident. We do this rather than instant-snapshot because most users come back within ~5 minutes and a snapshot/restore round trip is latency the user can feel.
3. **Hibernate.** 30 min total idle → snapshot. Firecracker snapshot writes RAM + register state + virtio device state to NVMe (a few seconds for a 2 GiB kernel). The host instance reclaims the memory. We use **diff snapshots** against a base image where possible to keep snapshot sizes down.
4. **Restore.** User returns → MAP_PRIVATE the memory file → resume. Page faults pull in pages on demand; the kernel sees no discontinuity except a wall-clock jump. Target: <2s end-to-end.
5. **Snapshot-too-big bailout.** If the kernel's resident set is >10 GiB at hibernate time, snapshotting is uneconomical: the snapshot write takes longer than a re-execute would, and the storage/RAM cost of holding a 10 GiB snapshot dominates. In that case we **persist the user's volume and force re-execution on resume** — the user sees "Your kernel was hibernated; click Run All to restore your variables." This is honest about the cost/UX tradeoff. **[STAFF SIGNAL: persistent-vs-ephemeral framing]**
6. **Evict.** 7 days of no activity → delete snapshot → next return is a fresh kernel. We tell the user this in the UI.

GPU kernels run a tighter version of the same machine: 5 min idle → instant hibernate, no IDLE state. GPU memory snapshotting (Modal-style) is the only thing that makes GPU notebooks economically viable; without it the duty cycle is 20% and the cost line eats the business.

The math, again: hibernation reduces always-on hot RAM from 40 TiB to ~6 TiB. **At $5/GiB-month, that's $170K/month avoided.** The hibernation infrastructure (snapshot storage, NVMe-tier S3, the metadata DB tracking who's where) costs <$5K/month at this scale. This is the most important math in the design.

---

## 7. State management and the "kernel died" experience [STAFF SIGNAL: state-management discipline]

A notebook's value is in its in-memory state — the dataframe loaded from S3, the model fit on it. The notebook *document* is durable; the *kernel* is not. Concrete mechanisms:

- **Three layers of state, with explicit durability.** **[STAFF SIGNAL: invariant-based thinking]**
  - **Document state** (cell text, outputs, metadata): Yjs CRDT in memory, ops log → S3 every 5s, Postgres snapshot every 60s. Survives everything.
  - **User-volume state** (uploaded files, written intermediate files): per-user EBS-class volume, mounted on every kernel for that user. Survives kernel death.
  - **Kernel in-memory state** (Python variables): in the kernel process. Survives hibernate/restore (snapshot). Does **not** survive OOM, crash, host failure, or eviction-after-7-days.
- **Kernel snapshots are the primary mitigation.** Every 30 min of active use we silently take a snapshot. On crash → restore from last. Worst case, user loses 30 min of state.
- **The "kernel died" UX is honest.** When state is genuinely gone (OOM, host failure, no recent snapshot): "Kernel died (out of memory). Your notebook is intact. Click Run All to recompute, or upgrade to the 16 GiB tier." We expose memory-usage history on the cell that crashed it.
- **`%checkpoint` magic.** Power users mark a checkpoint mid-cell; we snapshot synchronously. The escape hatch for "I just spent 20 min loading a dataset."
- **OOM is a first-class signal.** Guests run with `vm.oom_kill_allocating_task=1` so the offending process dies, not init. Trapped OOM emits a structured event; the user sees a precise message rather than a mysterious WebSocket disconnect.

We do **not** promise variable persistence as a guarantee. A staff engineer is honest about what the system does and doesn't preserve. Promising "your variables will always be there" leads to systems that do impossible things and fail mysteriously.

---

## 8. Execution path

```
       Browser                Gateway          KernelGW         Kernel (μVM)
          │                      │                │                  │
          │  exec_req(cell_id)   │                │                  │
          ├─────────────────────▶│                │                  │
          │  WebSocket           │                │                  │
          │                      │  authZ + lookup│                  │
          │                      │  kernel addr   │                  │
          │                      ├───────────────▶│                  │
          │                      │                │  ZMQ exec_request│
          │                      │                ├─────────────────▶│
          │                      │                │                  │ runs cell
          │                      │                │   stream(stdout) │
          │                      │                │◀─────────────────│
          │  stream(stdout)      │  fan-out to    │                  │
          │◀─────────────────────│  collaborators │                  │
          │                      │◀───────────────│                  │
          │                      │                │   exec_reply     │
          │                      │                │◀─────────────────│
          │  exec_reply          │                │                  │
          │◀─────────────────────│                │                  │
          │                      │                │                  │
```

Latency budget for a no-op cell (`pass`):

| Hop | Target p99 | Notes |
|---|---|---|
| Browser → Gateway WebSocket | 30 ms | depends on user geo |
| Gateway authZ + lookup | 5 ms | kernel registry in Redis |
| Gateway → KernelGW | 2 ms | same region |
| KernelGW → Kernel (ZMQ) | 5 ms | over per-tenant tap |
| Kernel: parse, dispatch, return | 10 ms | Jupyter ipykernel overhead |
| Return path | 50 ms | mirror of inbound |
| **Total platform overhead** | **<100 ms** | excludes user-code time |

User-code time is unbounded and tracked separately; we SLO platform overhead, not user code. **[STAFF SIGNAL: failure mode precision]**

Streaming is end-to-end: the kernel emits IOPub messages on every `print`, the gateway forwards them on the WebSocket without buffering, the browser appends to the cell's output stream. A long-running `for` loop with progress prints feels live.

Interrupt and cancel: `interrupt_request` over the same WebSocket → KernelGW sends `SIGINT` to the kernel process → Python raises `KeyboardInterrupt`. If the cell is in a C extension that ignores SIGINT (numpy in some cases), the user can escalate to "Force terminate cell," which `SIGKILL`s the kernel and triggers a restore from the last snapshot. We log force-terminates as an SLI to track UX pain.

Background execution. A user can navigate away and the kernel keeps running. We mark the session as "executing in background"; on return, the cell's accumulated output is streamed in. We bound this at 4 hours per cell on paid tier; longer needs a "Job" — a different product surface.

---

## 9. Collaboration: editing vs execution [STAFF SIGNAL: collaboration-as-two-problems]

Most notebook platforms conflate these. They are different problems with different consistency models.

**Collaborative editing** (Yjs over WebSockets). The notebook document is a Yjs `Doc` with shared types: an array of cells, each cell a map of `{type, source, output_ref, metadata}`. Operations on cells (text edits, reorder, insert, delete, output update) are CRDT mutations; concurrent edits merge deterministically. Presence (who's where, cursor positions) is a lightweight awareness channel. Persistence: ops log → S3 (5s), Postgres snapshot (60s). The Doc Sync Service is horizontally scaled with sticky routing per document; cross-shard sync is rare because all collaborators on one notebook hit the same shard.

**Collaborative execution** is something we do **not** support. The argument:

- A kernel processes execute_requests serially. Two users hitting Run on different cells means one waits.
- Worse: user A defines `x = 5`, user B redefines `x = 'hello'`, user A's next cell that does `x + 1` now crashes mysteriously. The user model "my variables" breaks down.
- The set of users who genuinely want concurrent shared-kernel execution is small (some pair-programming workflows). The cost is large (debugging "why is this broken" gets vastly worse).

What we ship instead: **single-executor semantics with a serialized run queue.** Multiple users can edit cells freely, but only one user is the "active executor" at a time. Run requests from other users queue up; the UI shows "Alice is running cell 7; your run is queued." Anyone can take the executor token; we don't lock it.

**Rejected alternative:** ephemeral fork-on-edit (each collaborator gets their own kernel forked from the active one's snapshot). Tempting because it solves the "your variables changed" problem. Rejected because (a) snapshot/restore-per-edit is too expensive, (b) the merging-back-of-state problem is unsolvable in the general case, (c) the UX of "everyone's notebook is now slightly different" is worse than the UX we're trying to fix.

**Rejected alternative:** OT-based document sync. CRDT (Yjs) is strictly better for the offline / partition cases that matter for a SaaS data tool. **[STAFF SIGNAL: rejected alternative]**

---

## 10. GPU notebooks and resource scheduling

GPU is the most expensive idle resource in the system. At $2/GPU-hour, a single A100 sitting idle for an 8-hour workday is $16. At 200 concurrent GPU notebooks, the math is unforgiving.

- **Separate pool, separate scheduler, separate idle policy.** GPU notebooks live in dedicated hosts (GPU passthrough into Firecracker μVMs). Idle timeout is 5 minutes (vs 30 for CPU); hibernate is immediate; eviction is 24 hours (vs 7 days).
- **GPU memory snapshotting.** Modal shipped this in alpha 2025; we follow. The kernel's CUDA context, loaded model weights in VRAM, and CUDA kernels are captured to NVMe; restore is ~5–15s for a 10 GiB model vs ~45s cold. Without this, the GPU notebook product is not economically viable. **[STAFF SIGNAL: 2026 cutting-edge awareness]**
- **MIG slicing on H100/A100.** A single H100 splits into 7 MIG slices; many notebook workloads (inference, finetuning, light experimentation) fit in 1g.10gb or 2g.20gb. We bill per-slice; this triples effective GPU density for the right workload.
- **Queueing.** When the GPU pool is exhausted, the user gets a queue position and an ETA, with optional CPU-fallback. We do not silently downgrade.
- **Pricing reflects scarcity.** GPU is metered per-second of allocation, not per-second of *use*; the user pays from `start_kernel` to `kernel_evicted`. This aligns user incentive with our cost — they hibernate when they walk away because they're paying for it.

---

## 11. Multi-tenancy: free / paid / enterprise [STAFF SIGNAL: tier-aware design]

Same codebase, three deployment shapes, parameterized by tier:

| Dimension | Free (deferred) | Paid Team | Enterprise (BYOC) |
|---|---|---|---|
| Sandbox substrate | Firecracker (untrusted) | gVisor on Kata | gVisor on Kata in customer VPC |
| Idle timeout | 5 min | 30 min | configurable |
| Eviction | 24 h | 7 d | configurable, default 30 d |
| RAM cap | 1 GiB | 8 GiB default, up to 64 GiB | per-contract |
| GPU access | none | yes, metered | yes, customer-owned GPUs |
| Network egress | DNS-allowlisted | broad with audit | tenant-VPC routing |
| Storage | 1 GiB volume, auto-deleted on inactivity | 100 GiB / user | per-contract |
| Auth | Email | SSO (SAML/OIDC) | SCIM + SSO + RBAC + SCIM provisioning |
| Data residency | single region | region-pinned per workspace | per-customer region |
| Audit | none | basic action log | full audit log, exportable |

The **enterprise BYOC** case inverts the architecture: the customer owns the compute infra; we ship a control plane that runs in their VPC and a thin SaaS for collaboration that touches only document metadata. This is non-trivial — operationally it means we maintain a "shippable" version of the kernel runtime, the scheduler, and the gateway, and we cannot assume our usual telemetry/observability. We deliberately do not ship the AI-agent service to BYOC tenants by default; if they want it, it runs in their VPC against their LLM endpoints.

The **free tier is deferred** in v1. Cost-of-acquisition math doesn't pencil for our scoped product (data teams, not individual learners). If we add it, it gets the strictest sandbox (Firecracker), aggressive eviction, and a hard monthly compute cap.

---

## 12. Data and storage

- **Notebook document.** Postgres (rows per cell + doc metadata), with the Yjs ops log in S3 for time-travel. The notebook is the durable artifact.
- **Cell outputs.** Inline ≤256 KiB lives in the document. Larger outputs (rendered tables, images, dataframe HTML) go to S3 by reference. Streaming uses S3 multi-part upload.
- **User volumes.** Per-user EBS-class block volume mounted at `/home/jovyan` on every kernel that user starts. 100 GiB default on paid tier. Survives kernel death. Metered.
- **Kernel scratch.** `/tmp` is in-VM, ephemeral, evaporates on kernel termination. UI labels this "Will not persist."
- **Data integrations.** Connectors to Snowflake, BigQuery, Postgres, S3, GCS — credentials brokered, encrypted at rest, decrypted only in-memory inside the kernel μVM, never logged.
- **Disk caps.** Kernel disk capped at 50 GiB; writes past that fail `ENOSPC`. User volumes have soft + hard quotas with email warnings.
- **Notebook size.** Document ≤50 MB; outputs auto-truncated past 5 MB per cell with a "Show full output" link to S3.

---

## 13. Failure modes [STAFF SIGNAL: failure mode precision]

- **OOM.** Cgroup OOM kills the user's Python process inside the μVM; supervisor restarts the kernel. The user sees "Kernel died (out of memory at 7.8 GiB / 8 GiB)" with the offending cell highlighted and a "Run All from snapshot" button. Recent snapshot restores variables minus the offending allocation; otherwise full re-run.
- **WebSocket disconnect / partition.** Browser shows "Reconnecting…" and buffers user edits in IndexedDB (Yjs persistence). On reconnect, Yjs merges; in-flight execute_requests are ack'd or replayed via the kernel's last-seen sequence number.
- **Compute node failure.** Host dies. Scheduler detects via missed heartbeats (5s); kernels marked DEAD. With a snapshot ≤30 min old, scheduler restores on a new host transparently — user sees a 2–5s blip. Otherwise: "Kernel died on host failure; please rerun cells."
- **Storage failure.** Document reads fall back to a cross-region replica with a "Read-only mode (storage degraded)" banner; writes queue locally in Yjs and replay when storage recovers. We do not stop the world.
- **Adversarial code.** Cgroup PID limit (1024/kernel) blocks fork bombs. Egress is L7-proxied; outbound to known mining pools blocked and logged. gVisor handles syscall-level escapes; Kata adds a VM boundary. Repeated abuse → admission-control degradation + manual review.
- **Thundering herd.** Blog post links a template; 5K users click within 30s. **[STAFF SIGNAL: blast radius reasoning]** Pool below 20% headroom triggers per-tenant rate limiting on new-kernel creation; users see "Spinning up your environment, ETA 12s" rather than a failure. Warm pool autoscales on rate-of-cold-start; we provision 3× the trailing-1h P99.
- **AI agent runaway.** LLM produces `while True: pass`. Cell-runtime cap (60s for AI invocations vs 4h for human-driven) kills it. Cost-attribution charges agent budget, not user quota.

---

## 14. AI integration as architectural overlay [STAFF SIGNAL: AI-integration without dominance]

The Code Interpreter pattern is becoming a primary workload. Within our scoped product, AI integration appears in two shapes:

1. **Inline assistant** — user types prompt, LLM proposes a cell of code, user accepts and runs. The execution path here is the *same* execute path as a human-typed cell. The architectural change is upstream: a prompt-context service that gathers schema info, recent cell history, and dataframe heads as LLM context. No new sandbox primitives.
2. **Agent mode** — user gives a goal ("explore this dataset and find anomalies"), LLM plans, generates code, runs it in the kernel, observes output, generates the next step. This is the loop that needs architectural attention.

```
       AI agent loop (in our scoped product)

      ┌──────────┐                 ┌──────────────────┐
      │  User    │                 │  Agent Service   │
      │  goal    │ ──────────────▶ │  (LLM proxy +    │
      └──────────┘                 │   tool router)   │
                                   └────────┬─────────┘
                                            │
                                  generates code, calls
                                  the python tool
                                            │
                                            ▼
                                   ┌────────────────────┐
                                   │   Sandbox tool     │
                                   │   == kernel exec   │
                                   │   (same path!)     │
                                   └────────┬───────────┘
                                            │
                                            ▼
                                   ┌────────────────────┐
                                   │  Notebook kernel   │
                                   │  (gVisor / FCK)    │
                                   │  exec, stdout,     │
                                   │  output → context  │
                                   └────────┬───────────┘
                                            │
                                            ▼
                                   ┌────────────────────┐
                                   │  Output back       │
                                   │  into LLM context  │
                                   │  (truncated &      │
                                   │  summarized)       │
                                   └────────┬───────────┘
                                            │
                              loop until goal met or
                              budget (steps / cost) hit
```

Architectural implications:

- **Tighter resource limits per agent step.** Human-driven cell: 4h cap. Agent-driven cell: 60s default, extendable with explicit budget. LLMs hallucinate `for i in range(10**12)` more than humans do.
- **Cost attribution.** Each agent invocation charges the agent's session budget, not the user's notebook compute quota. Otherwise an agent that loops 200× silently drains the user's plan.
- **Output truncation into LLM context.** A 1M-row dataframe cannot all return to the LLM; we summarize (head/tail/describe) for the model and store the full output in the notebook for the human.
- **Persistent vs ephemeral, again.** [STAFF SIGNAL: persistent-vs-ephemeral framing] Our agent runs against the user's *persistent* notebook kernel — the right call when the agent benefits from accumulated state. Pure-ephemeral (Modal/Code-Interpreter, fresh sandbox per call) is right for a different product (headless tool-calls). Different optima for different masters.

Meaningful, but not the whole system. The classical problems (sandboxing, idle compute, kernel state, collaboration, GPU) are still 80% of the work.

---

## 15. Operational reality

- **Deployment with live kernels.** We cannot drain-and-restart kernel hosts on every release. The kernel runtime is a stable contract (Jupyter protocol + extensions); kernel hosts upgrade by **rolling drain + snapshot-migrate** — kernels on a draining host are snapshotted, restored on a new host, the gateway updates the kernel registry, the user sees a 2–5s reconnect. Control plane services deploy via standard rolling restarts.
- **Version skew.** A user opens a 2-year-old notebook with `python==3.9, pandas==1.3`. Environments pin per-notebook; the kernel image is built from the pinned manifest. We retain images for 3 years; past that, a one-click "migrate to latest stable" path. We do not silently rev environments.
- **Observability.** Per-kernel metrics → Prometheus → long-term Mimir. Per-execute traces (OTel) sampled at 1%, 100% on errors. Sandbox-escape signals → SIEM. Paged SLOs: kernel-start P99 < 8s, execute-overhead P99 < 100ms, hibernation success > 99.5%, restore success > 99.9%.

---

## 16. Tradeoffs and what would change them

- **If users were untrusted (free tier, anonymous):** drop gVisor-on-Kata in favor of Firecracker for *everything*; tighten egress; switch idle to instant-hibernate; add per-user lifetime compute quota.
- **If workload were heavy-GPU (LLM training):** the scheduler is dominated by GPU placement, multi-node MPI/NCCL networking, checkpoint-resume becomes the central problem, and the notebook UX layer is a thin shell on top of a job-runner.
- **If we wanted true real-time co-execution:** we'd have to commit to fork-on-edit (snapshot-based session forking) and accept the merge-back-of-state UX cost; this is a different product.
- **If enterprise BYOC dominates revenue:** the ephemeral SaaS layer shrinks, the per-tenant control plane grows, the scheduler must accept tenant-supplied node pools as a first-class concept.

---

## 17. What I would push back on [STAFF SIGNAL: saying no]

Three of the prompt's implicit assumptions deserve pushback.

1. **"Hosted notebook platform" is not one product.** I've designed for one (collaborative SMB data workspace + AI overlay). The same headline question, scoped to "Colab for free GPU notebooks" or "Code Interpreter as a primitive," yields a *meaningfully different* design at every layer. The interviewer should expect candidates to scope first, not paper over the ambiguity with a generic answer.
2. **"Always-on persistent kernel" is not the only architecture.** For agent-heavy workloads, ephemeral-sandbox-per-call is the right model and is what OpenAI ships. **[STAFF SIGNAL: persistent-vs-ephemeral framing]** Our scoped product picks persistent-kernel because the human-collaborator workflow benefits from accumulated state; an alternate-universe version of this product is a Modal-style ephemeral primitive with explicit state mounts, and that universe is not strictly worse — it's a different optimum.
3. **"Real-time collaboration" is over-prescribed.** Most users of paid data notebooks are solo. Collaborative editing earns its 30% complexity tax for the workflows where it matters (analyst hand-offs, pair work, review). Collaborative *execution* is a feature people ask for that they would dislike if they got. Saying no to it is the design call.