---
title: Priority Job Scheduler
description: Priority Job Scheduler
---


## 1. Clarifying questions + assumed answers

A few things to nail down before I start drawing boxes. I'll state assumptions; correct me where they're wrong:

- **Volume / shape?** ~10k jobs/day, jobs run on the order of minutes (seconds → hours), bursty arrivals.
- **Workload classes?** Multiple swimlanes — training, batch reporting, ad-hoc, A/B pipelines, encoder. Different SLOs, different resource shapes (GPU vs CPU vs memory-heavy).
- **Multi-tenant?** Yes, multiple internal teams. Fairness and isolation matter, not just priority within one user.
- **Preemption?** Non-preemptible v1 (jobs run to completion or fail). Preemption is v2.
- **Priorities?** Discrete integer tiers (P0..P9). Not floats. Ten tiers is plenty.
- **Cancel/reprio rate?** ~1% of submits. Not the hot path.
- **Where do executors live?** Pool of worker nodes, scaled horizontally, heterogeneous (some GPU, some CPU). Scheduler does not own the runtime — it decides what runs next.
- **Exactly-once?** Required for side-effecting jobs. Achieved via *at-least-once dispatch + idempotent execution + fencing tokens*, not via two-phase commit.

**Push back:** "priority-based" is underspecified at staff level. Strict global priority across tenants is wrong — one team submitting 1k P0s starves everyone else. I'm going to design this as **strict priority within a swimlane, weighted fair share (DRF) across swimlanes**.

---

## 2. Capacity math + reframe

```
10,000 jobs/day = 0.116 jobs/sec sustained
peak burst (10×)  ≈ 1.2 RPS submit
peak burst (100×) ≈ 12 RPS submit
```

A 10-line Python heap on a single box handles 10k enqueues/sec. **Throughput is not the problem.**

The hard problems here are semantic:

```
┌─────────────────────────────────────────────────────────────┐
│  not the hard part           │  the actual hard part         │
│ ─────────────────────────────┼───────────────────────────── │
│  heap micro-perf             │  cross-tenant fairness        │
│  log(n) vs O(1) on dispatch  │  starvation of low priority   │
│  "millions of jobs/sec"      │  priority inversion           │
│  serialization format        │  exactly-once execution       │
│                              │  isolation / noisy neighbor   │
│                              │  failure semantics             │
│                              │  cancel/reprio races          │
└─────────────────────────────────────────────────────────────┘
```

If I micro-optimize the heap I've spent my budget on the wrong axis. Most of the design effort goes to the right column.

---

## 3. SLOs and explicit non-goals

**SLOs**

| Metric | Definition | Target |
|---|---|---|
| Schedule latency P0 | submit-ack → executor `start()` (when capacity exists) | p50 < 500ms, p99 < 5s |
| Schedule latency P5 | same | p50 < 5s, p99 < 60s |
| Schedule latency P9 | same | best-effort, eventual |
| Submit ack | client RTT | p99 < 100ms |
| Durability | acked job survives any single-node failure | 100% |
| Effect semantics | side effects under retries | exactly-once *effect* via worker idempotency |

**Non-goals (v1)**
- Preemption of running jobs
- Cross-region failover (single region, multi-AZ)
- Automatic resource estimation (caller declares)
- DAG / job dependencies (separate orchestrator concern)
- Sub-second scheduling latency

---

## 4. API surface

```
submit(job_spec, idempotency_key, priority, swimlane, tenant_id) -> job_id
  - (tenant_id, idempotency_key) is unique within 24h
  - duplicate returns the original job_id, no side effect

cancel(job_id, idempotency_key) -> { cancelled | already_running | already_done }
  - cancel-after-dispatch race resolved via lease revocation (§8.4)

reprioritize(job_id, new_priority) -> { ok | already_running | already_done }
  - logical cancel + re-insert at new priority

status(job_id) -> { pending | dispatched | running | succeeded | failed | cancelled }

list(tenant_id, swimlane, filters) -> [jobs]
```

The idempotency key is non-negotiable. Without it, a client retrying on a network blip creates duplicate work.

---

## 5. Data model

```
Job {
  job_id              uuid           -- server-assigned
  tenant_id           string
  swimlane            enum           -- training | batch | adhoc | encoder | ...
  priority            int            -- 0..9, lower = higher priority
  state               enum           -- see state machine below
  spec                bytes          -- opaque payload for executor
  idempotency_key     string         -- (tenant_id, key) UNIQUE within 24h
  submitted_at        ts
  enqueued_at         ts             -- last (re)insert into a queue
  effective_priority  int            -- after aging; what the queue sorts by
  resource_req        {cpu,mem,gpu}
  retry_count         int
  attempt_id          uuid           -- bumped on each dispatch attempt (fencing)
}

Lease {
  job_id
  attempt_id          uuid           -- fencing token
  worker_id
  expires_at          ts
}
```

State machine:

```
                     submit
                       │
                       ▼
                  ┌─────────┐  cancel(pending)
                  │ pending │ ─────────────────► cancelled
                  └────┬────┘
                       │ dispatch (issue lease)
                       ▼
                 ┌────────────┐  cancel(disp)/lease revoked
                 │ dispatched │ ─────────────────► cancelled
                 └─────┬──────┘
                       │ worker ACKs running
                       ▼                        ┌────► succeeded
                  ┌─────────┐  complete         │
                  │ running │ ──────────────────┤
                  └────┬────┘                   └────► failed
                       │
                       │ lease expiry / worker death
                       ▼
                  re-enqueue (retry_count++, new attempt_id)
```

State transitions are linearizable through the scheduler leader. A worker presents its `attempt_id` on every state-change RPC; stale tokens are rejected.

---

## 6. Architecture

```
                        ┌──────────────────────────────┐
   clients ──submit──► │      API gateway (pool)      │   stateless,
   (gRPC)              │   auth, idempotency cache,   │   horizontal scale,
                       │   per-tenant rate limiting   │   admission control
                        └──────────────┬───────────────┘
                                       │ gRPC (write → leader)
                                       ▼
                        ┌──────────────────────────────────┐
                        │  Scheduler — Raft group (5 nodes) │
                        │  ┌────────────────────────────┐  │
                        │  │ Leader                     │  │
                        │  │   in-mem queues            │  │
                        │  │   dispatcher loop (DRF +    │  │
                        │  │     strict-prio-within)    │  │
                        │  │   index: job_id → ptr      │  │
                        │  └────────────────────────────┘  │
                        │  followers: hot standby           │
                        │  state replicated via Raft log    │
                        └─────────────┬────────────────────┘
                                      │ state-machine apply
                                      ▼
                        ┌──────────────────────────────┐
                        │  Postgres (materialization)  │
                        │   jobs, leases, dedup cache  │
                        │   for queryability/dashboards │
                        └──────────────────────────────┘
                                      ▲
                                      │ long-poll pull (lease-based)
                        ┌─────────────┴────────────────┐
                        │       Executor pool          │
                        │  ┌────┐ ┌────┐ ┌────┐ ...    │
                        │  │ W1 │ │ W2 │ │ W3 │        │
                        │  └────┘ └────┘ └────┘        │
                        │  heterogeneous (GPU/CPU)     │
                        │  pull, heartbeat, fence      │
                        └──────────────────────────────┘
```

**Edge semantics:**

| edge | protocol | semantics |
|---|---|---|
| client → API | gRPC | idempotent submit/cancel/reprio |
| API → leader | gRPC | follower redirects to leader |
| leader ↔ followers | Raft AppendEntries | log replication, quorum commit |
| leader ↔ Postgres | SQL | state-machine writes after Raft commit |
| worker → leader | gRPC long-poll | `fetch(caps) → job + lease`, `heartbeat`, `complete`, `fail` |
| worker liveness | TCP keepalive + heartbeat every 10s | lease expires at 30s |

**Push vs pull = pull, with leases.** Justification:
- Push requires the scheduler to track per-worker liveness, capacity, and capability — turns into reimplementing a service registry. Hot spots when a worker is briefly slow.
- Pull lets workers self-select based on local capacity and capability tags. Backpressure is automatic — saturated workers stop pulling, which propagates upstream as queue-depth growth, then admission control kicks in at the API.
- Borglet, kubelet, Slurm slurmd — all pull-shaped. Prevailing design for good reason.

Workers are explicitly **not** a single oval. They are a horizontally-scaled heterogeneous pool, and the "dispatch queue" is the lease-managed pull endpoint, not a separate broker.

---

## 7. Core algorithms

### 7.1 Queue structure

| option | push/pop | reprio | cross-swimlane policy | verdict |
|---|---|---|---|---|
| single global min-heap on (priority, enq_at) | O(log n) | decrease-key, awkward | encoded as synthetic key | rejected — couples policy to data structure |
| **N FIFO queues, one per (swimlane, priority_tier)** | **O(1)** | **unlink + push, O(1)** | **separate scheduling decision on top** | **picked** |
| skip-list keyed on priority | O(log n) | O(log n) | not natural | rejected — pays log n for nothing |

Pick FIFO-per-(swimlane, priority_tier). Priorities are discrete; cross-swimlane policy benefits from being explicit and decoupled from the data structure.

```
swimlane: training              swimlane: batch              swimlane: adhoc
┌─────────────────────┐        ┌─────────────────────────┐  ┌────────────────┐
│ P0: [j7]→[j12]      │        │ P0: [j3]                │  │ P0: []         │
│ P1: [j4]            │        │ P1: [j9]→[j15]→[j22]    │  │ P1: [j8]       │
│ P2: []               │       │ P2: [j1]                │  │ ...            │
│ ...                  │       │ ...                      │ │                │
│ P9: [j33]→[j40]     │        │ P9: [j2]                │  │ P9: []         │
└──────────┬──────────┘        └────────────┬────────────┘  └───────┬────────┘
           │                                │                       │
           └──────────────────┬─────────────┴───────────────────────┘
                              ▼
                ┌─────────────────────────────────────┐
                │  Cross-swimlane scheduler (DRF)     │
                │  picks next swimlane, then strict   │
                │  prio + FIFO-within-tier inside     │
                └─────────────────────────────────────┘

  index: hashmap[job_id] → (swimlane, tier, list_node_ptr)   ← O(1) cancel
```

### 7.2 Cancel + reprio (lazy deletion + tombstones)

```
cancel(job_id):
  entry = index[job_id]
  switch entry.state:
    pending     → mark entry.tombstoned = true        (O(1), no list surgery)
                  WAL: state = cancelled
    dispatched  → bump attempt_id (revokes lease)
                  WAL: state = cancelled
                  worker, on next heartbeat or complete, gets LEASE_REVOKED
    running     → send cancel signal via heartbeat response
                  worker honors best-effort; for non-idempotent jobs this
                  is the inherent limit
    terminal    → no-op
```

On `pop()`, dispatcher skips tombstones and unlinks them. Pattern: **lazy deletion**. Standard in priority queues with cancellation; avoids O(n) scans.

`reprioritize(j, P_new)` = tombstone old entry + insert fresh entry at new (swimlane, P_new). Ghost is skipped on pop. Reprio is O(1) and avoids decrease-key complexity.

### 7.3 Cross-swimlane fairness

| option | what it gives | where it breaks |
|---|---|---|
| strict priority across swimlanes | trivial | training P0 starves batch P0 forever |
| Weighted Fair Queuing | proportional shares per swimlane | misallocates with heterogeneous resources (one GPU job ≠ one CPU job) |
| **Dominant Resource Fairness** (Ghodsi et al., NSDI '11) | fair across heterogeneous resources; tenant's "share" = its dominant resource fraction; pick smallest dominant share next | more bookkeeping; worth it |

Pick **DRF.** Workloads are heterogeneous (GPU training vs CPU encoding vs memory-heavy reporting). Plain WFQ on job count or CPU share misallocates because a GPU-bound tenant looks cheap on CPU and gets unfairly favored. Used by YARN and Mesos.

Dispatch decision is two-level:

```
┌──────────────────────────────────────────────────────────┐
│ 1. DRF: among swimlanes with pending work,               │
│    pick the one whose dominant-resource share is smallest │
│                                                           │
│ 2. Within that swimlane, take head of highest non-empty   │
│    priority tier (FIFO inside tier)                       │
└──────────────────────────────────────────────────────────┘
```

### 7.4 Starvation prevention (priority aging)

A P9 job submitted at t=0 in a busy system can wait forever. Standard fix: **priority aging**. After `T_age = f(priority)`, promote the job by one tier. Aging is computed lazily on pop, not via wall-clock mutation:

```
effective_priority(j) = max(0, j.priority - floor((now - j.enqueued_at) / T_age))

Tuned so:
   P9 reaches P0 in ~24h
   P5 reaches P0 in ~1h
   P1 reaches P0 in ~5min
```

Aging coefficient is configurable per swimlane. Combined with DRF across swimlanes, this gives a strong guarantee that no job waits indefinitely while the system has capacity.

---

## 8. Failure modes (one by one)

### 8.1 Leader crash mid-WAL-write

The Raft log *is* the WAL. There is no separate "WAL file" + "Postgres" + "etcd" stack — that's the layering anti-pattern.

```
   client            leader              followers
     │ submit          │                     │
     │────────────────►│                     │
     │                 │ append entry        │
     │                 │ ──────────────────► │
     │                 │ ◄──── ack ────────  │  (quorum ack)
     │                 │                     │
     │                 │ apply state machine │
     │                 │ (mutate queues,      │
     │                 │  upsert Postgres)    │
     │                 │                     │
     │ ◄──── ack ──────│                     │
```

If the leader crashes *before* quorum, the entry is uncommitted; new leader's log wins. Client got a timeout, retries with same idempotency key, dedupe. If the leader crashes *after* quorum but before client ack, entry is committed; client retry dedupes. Either way: **no two sources of truth, no torn write.**

### 8.2 Executor crash mid-job (lease expiry → re-dispatch)

```
  T+0    W1 fetches job j, gets lease(j, attempt=A1, expires=T+30)
  T+5    W1 starts running, heartbeats at T+10, T+20
  T+22   W1 crashes (host failure)
  T+30   lease expires
  T+30   scheduler re-enqueues j with attempt=A2, retry_count++
  T+31   W2 fetches j with lease(j, attempt=A2, expires=T+61)
  T+90   zombie W1 boots, tries to send complete(j, A1)
         → scheduler rejects: current attempt is A2, A1 is stale
         → no double-write of result
```

This is the classic **fencing token** pattern. `attempt_id` plays the role of Raft term / Kleppmann's fencing number. The dead worker's fence is now stale; even if it resurrects mid-RPC, it cannot corrupt state.

### 8.3 Network partition during reprio

Three sub-cases:

```
  case A: leader received & committed before partition
          → client retry hits new leader, idempotent reprio is no-op. fine.
  case B: leader received but did not commit before partition
          → change is lost; client retry succeeds. fine.
  case C: client doesn't know which case
          → client retries with same idempotency key for the reprio op;
            backend dedupes via (tenant, idem_key) cache.
```

Non-trivial case is reprio racing with dispatch — same shape as 8.4.

### 8.4 Cancel-after-dispatch race

```
  T+0    cancel(j) arrives at leader
  T+1    dispatcher already popped j, issued lease(j, A1) to W1
  T+2    W1 has not yet ACKed running

  resolution:
    cancel handler reads job state.
      if dispatched: bump attempt_id → A2 (revokes lease A1)
                     WAL: state=cancelled
      W1's next heartbeat or running-transition:
        presents A1 → scheduler rejects with LEASE_REVOKED
        W1 aborts cleanly

    if W1 already started side effects:
      best-effort cancel — cancel signal arrives in heartbeat response,
      worker honors it. For non-idempotent side effects this is the
      inherent limit (you cannot cancel an HTTP POST already in flight).
      Document this.
```

### 8.5 Duplicate dispatch / thundering herd

Two scenarios:
- **Same job dispatched to two workers** (split-brain): prevented by Raft single-leader. If it leaks anyway, fencing token resolves it — only one `attempt_id` is current, the other gets `LEASE_REVOKED` on first state transition.
- **Worker thundering herd on leader after a wake-up**: bound by long-poll with jittered backoff. Workers don't tight-loop; `fetch(timeout=30s)`, leader notifies on new work. Jittered reconnect after disconnect prevents synchronized stampede.

### 8.6 Submit retry → duplicate work

`(tenant_id, idempotency_key) → job_id` cache, persisted in Postgres with TTL 24h. Duplicate submit returns the original `job_id` without re-enqueueing. Dedupe is checked at the API layer before going to the leader.

### 8.7 Multi-tenant abuse / noisy neighbor

A misbehaving tenant submits 1M P0 jobs. Mitigations:
- Per-tenant submit-rate quota at the API layer (token bucket).
- Per-tenant max-in-flight quota (admission control rejects at submit).
- DRF across swimlanes ensures other tenants make progress regardless of queue depth.
- Per-tenant queue-depth alarm; operator can quench.

Not perfect, but it prevents starvation and gives operators a knob.

### 8.8 Backpressure propagation

```
   executor pool saturated (no workers idle)
            │
            ▼
   pull rate drops; jobs accumulate in scheduler queues
            │
            ▼
   scheduler queue depth crosses threshold (per-tenant + global)
            │
            ▼
   API admission control: 503 with Retry-After,
   or accept and tag as "degraded SLO" for low-priority tiers
            │
            ▼
   client backs off (exponential + jitter), retries with same idem key
```

Crucial point: backpressure is **end-to-end**, not just at one layer. If you only push back at the executor, the scheduler queues grow unbounded.

---

## 9. Persistence + consistency model

```
   write path:
   client → API → leader.append(entry) ──► Raft replicate ──► quorum ack
                                                  │
                                                  ▼
                                      state machine applies entry:
                                        - mutate in-mem queues
                                        - upsert Postgres jobs row
                                                  │
                                                  ▼
                                            ack to client

   recovery path:
   leader boot
       │
       ▼
   load latest snapshot (in-mem queue state)
       │
       ▼
   replay Raft log from snapshot.index
       │
       ▼
   in-mem queues reconstructed, dispatcher resumes
```

- **Raft log = WAL = source of truth.** Postgres is a *materialization* for queryability (status, list, dashboards). If Postgres is lost, rebuild from the log.
- **Snapshots** of the in-memory queue state are taken every N entries (e.g. 10k) to bound replay time. Snapshot truncates the log up to its index.
- **Why not etcd as the job store?** etcd is excellent for small (KB-sized) strongly-consistent metadata — leader election, config, service discovery. It is *not* designed for high-volume mutable state and *not* designed for arbitrary indexed queries. K8s uses etcd but explicitly keeps object count bounded and lifts heavy state out (see the K8s scalability docs). Putting weeks of job history in etcd would beat it up. Use Raft directly (e.g. `hashicorp/raft`) for the consensus log; use Postgres for the materialized job table. They serve different roles — don't conflate.
- **Why not Kafka as the queue?** Kafka is great for ordered append-only logs, bad for the operations we need: O(1) cancel by job_id, reprio (deletion + reinsert), priority-ordered consumption. Building these on Kafka is fighting the tool.

---

## 10. Scaling story

| axis | v1 (10k/day) | 100× (1M/day, ~12 RPS) | 10000× (100M/day, ~1.2k RPS) |
|---|---|---|---|
| API gateway | 2 nodes | 10 nodes | autoscale |
| Scheduler | 1 leader, 5-node Raft | same; queues fit easily in mem | **shard by swimlane or tenant range**; multiple Raft groups |
| Postgres | single primary + read replicas | partition by `submitted_at`, archive cold | per-shard Postgres |
| Workers | 10s | 100s | 1000s, multi-region pools |
| First bottleneck | none — semantic correctness *is* the work | leader CPU on dispatch decision | leader fanout to workers; need sharding |

The single-leader Raft scheduler holds up far longer than people expect because dispatch decisions are cheap. The first thing that breaks at scale is **the leader's worker fanout**, not the queue. Solution: shard the scheduler by swimlane (or tenant range); each shard is its own Raft group; workers pull from the shard whose work they can run. This is exactly Borg's per-cell architecture — one cell ≈ one shard.

What does *not* change with scale: API contract, idempotency, fencing, lease semantics, DRF policy, cancel semantics. Get those right at v1 and they survive 4 orders of magnitude.

---

## 11. MVP vs v2

**v1 (MVP, ~6 weeks for a small team)**
- Coarse multi-tenant (per-tenant quotas, but all in one swimlane to start)
- **Postgres as source of truth, single primary** — `SELECT ... FOR UPDATE SKIP LOCKED` on a job table gives a working scheduler with crash-safety. This is the Sidekiq-Pro / `que` / `delayed_job` pattern. Accept the SPOF for v1.
- Strict priority + FIFO within tier
- Pull-based workers with leases + fencing tokens (do not skimp here — every bug in this layer is a duplicate-execution bug)
- Cancel via lazy tombstone; reprio = cancel + reinsert
- Priority aging with a single coefficient

**v2**
- Raft-replicated scheduler for HA
- Multiple swimlanes + DRF
- Per-tenant quotas + admission control (proper)
- Preemption of running low-priority jobs when high-priority arrives and capacity is tight
- Cross-region failover
- Sharding by swimlane

I'd start with Postgres + `SKIP LOCKED` and only move to Raft when the durability / HA story actually demands it. **Premature consensus is a tax** — most v1 systems can tolerate a 5-minute scheduler outage; the cost of getting Raft wrong is worse than the cost of a manual failover.

---

## 12. Prior art anchor

Closest prior art for v2:

| system | what I keep | what I change |
|---|---|---|
| **Borg / K8s scheduler** | pull-based workers, leases, controller pattern, per-cell sharding | K8s puts everything in etcd; I'm explicitly not — Postgres for materialization, Raft directly for the consensus log |
| **YARN / Mesos** | DRF for cross-tenant fairness | (just adopt) |
| **Temporal** | at-least-once + idempotency-key + fencing semantics | (just adopt) |
| **Slurm** | priority aging mechanics, swimlane (partition) model | replace homegrown HA with Raft |
| **Sidekiq Enterprise** | Postgres-backed pragmatic MVP, `SKIP LOCKED` pattern | add multi-tenant DRF when graduating from MVP |
| **AWS Batch** | the "you bring jobs, we manage compute" abstraction | keep tighter SLOs and per-tenant fairness |

---

## 13. Open questions / data I'd want before committing

- Actual distribution of job durations. If there's a long tail past 1h, lease timeout tuning matters a lot, and we may want checkpointing as a v2 feature.
- Distribution of priorities in production. If 80% of jobs are P0, "priority" is meaningless and we need a different signal (e.g. deadline-based scheduling).
- Real cancellation rate per swimlane. 1% globally may hide a 30% rate in ad-hoc, which would change the queue structure decision.
- Whether "exactly-once *effect*" is actually required, or whether "at-least-once with idempotent jobs" is the contract we're offering. This shapes what we tell users and how strict the fencing logic needs to be.
- Whether DAG-style dependencies live in this system or in a separate orchestrator (Airflow, Temporal). I assumed separate; worth confirming.
- Resource estimation accuracy. If callers under-declare GPU memory, we get OOMs and re-dispatch storms — DRF accounting becomes meaningless. May need a feedback loop that reconciles declared vs observed usage.

---

**Where I'd want to spend more interview time:** the cancel-after-dispatch race + fencing semantics, and the DRF + aging interaction (they can fight each other in pathological cases — DRF wants to throttle a swimlane that's over its share, but aging wants to promote the oldest job in that swimlane; need to define which wins).