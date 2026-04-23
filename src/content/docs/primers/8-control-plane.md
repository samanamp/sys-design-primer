---
title: Control plane 
description: Control plane 
---


## 1. What a staff engineer actually needs to know

**What matters in interviews**
- Crisp separation of control plane vs data plane
- Desired-state model and reconciliation loops
- Idempotency, retries, fencing
- Scheduler/controller separation
- Failure modes: leader failover, split brain, stale reads, thundering herd
- Tradeoff vocabulary: consistency vs availability, centralized vs sharded, push vs pull, static stability

**Expected depth**
- Whiteboard a control plane for a job scheduler / ML training platform / CDN edge orchestrator in 20 minutes
- Explain *why* each component exists and what breaks when it fails
- Reason about correctness under concurrent controllers, network partitions, and worker crashes

**What you can safely ignore**
- Raft/Paxos internals — say "back it with etcd/ZooKeeper/Spanner" and move on
- Specific Kubernetes CRD mechanics
- History of cluster managers (Borg → Omega → K8s trivia)
- Internals of specific schedulers (Sparrow, YARN). Know the archetypes, not the code.

---

## 2. Core mental model

**Control plane** = decides *what should be true*. Stores desired state, observes actual state, issues actions to reconcile the gap. Low QPS, high stakes.

**Data plane** = does the actual work. Serves traffic, runs jobs, moves packets. High QPS, low per-decision stakes.

```
        ┌──────────────────────────────────────────┐
        │            CONTROL PLANE                 │
        │  (decisions, desired state, policy)      │
        │                                          │
        │   API ──► State Store ──► Controllers    │
        │                    ▲          │          │
        │                    │          ▼          │
        │                  status    commands      │
        └────────────────────┬─────────┬───────────┘
                             │         │
                         observe    actuate
                             │         │
        ┌────────────────────┴─────────┴───────────┐
        │             DATA PLANE                   │
        │   Workers / Proxies / Runtime Agents     │
        │   (execute, serve traffic, run code)     │
        └──────────────────────────────────────────┘
```

**Scheduling** = deciding *where* and *when* work runs. A specialized controller whose job is placement under constraints.

**Reconciliation loop** = continuously compare desired vs actual; take corrective action; repeat. Not "do this once," but "keep this true."

**Why this model wins:** systems fail constantly. A one-shot imperative command loses information the moment it is issued. A reconciliation loop survives crashes, network blips, and partial failures because the *intent* is persisted and any healthy controller can resume convergence.

**Three properties that define a good control plane**
1. **Control** — clear authority over what actions are issued
2. **Convergence** — observed state approaches desired state over time
3. **Correctness under failure** — no corruption, no duplicate destructive actions, no permanent divergence

---

## 3. Essential components

### API / Desired-state store
- **What:** accepts writes of intent ("I want 10 replicas of service X"). Validates, authorizes, persists.
- **Why:** single source of truth. Decouples *wanting* from *achieving*.
- **Failure/scale:** becomes bottleneck under write load; stale reads if replicated async; large objects degrade watches.
- **Interview signal:** "Strongly consistent store (etcd/Spanner/FoundationDB); optimistic concurrency via resource versions."

### Controller / Operator
- **What:** runs the reconciliation loop for a resource type. Reads desired + observed, computes diff, issues actions.
- **Why:** separates policy (what) from mechanism (how). Each resource type gets its own controller.
- **Failure/scale:** if it crashes mid-action, something else must take over without duplicating destructive work.
- **Interview signal:** "One active leader per control loop, fencing via lease, idempotent actions."

### Scheduler
- **What:** assigns pending work to available capacity. A specialized controller.
- **Why:** placement is a global optimization problem (fairness, packing, constraints) that doesn't belong in per-resource controllers.
- **Failure/scale:** decisions go stale fast (node disappears, capacity changes); queue can grow unbounded.
- **Interview signal:** "Separate from executor; OCC so stale decisions get rejected, not applied."

### Worker / Agent
- **What:** runs on the data plane. Executes assigned work, reports status.
- **Why:** local execution with local knowledge (disk, GPU, processes).
- **Failure/scale:** worker crash mid-task → task in unknown state; heartbeats flood the control plane at scale.
- **Interview signal:** "Agents pull work, not push to; idempotent task execution; heartbeats with bounded frequency + jitter."

### State store / metadata store
- **What:** persists desired + observed state, leader election, locks.
- **Why:** durable ground truth; a crash without it wipes intent.
- **Failure/scale:** etcd-class stores cap at a few GB and tens of thousands of watches; hot keys kill performance.
- **Interview signal:** "Shard by resource type or tenant once you outgrow a single cluster. Not a general-purpose DB."

### Event stream / watch mechanism
- **What:** push-based notification on state change. Controllers watch rather than poll.
- **Why:** polling doesn't scale; watches give low-latency reconciliation.
- **Failure/scale:** watches drop (network, server restart); events can duplicate, reorder, or be lost.
- **Interview signal:** "Treat watches as cache-invalidation hints, not ground truth. Periodic full resync + per-event reconcile. Events must be idempotent in handling."

```
      ┌─────────┐  write desired   ┌──────────────┐
 User │  Client │ ───────────────► │  API Server  │
      └─────────┘                  └──────┬───────┘
                                          │ persist (w/ version)
                                          ▼
                                   ┌──────────────┐
                                   │ State Store  │◄──┐
                                   │  (etcd-like) │   │
                                   └──────┬───────┘   │
                                  watch   │           │ status
                                  events  ▼           │ updates
                                   ┌──────────────┐   │
                                   │  Controller  │───┘
                                   │  (loop)      │
                                   └──────┬───────┘
                                          │ actuate
                                          ▼
                                   ┌──────────────┐
                                   │   Worker     │
                                   └──────────────┘
```

---

## 4. Reconciliation loops

The single most important concept in this domain.

```
          ┌─────────────────────────────────┐
          │                                 │
          ▼                                 │
   ┌─────────────┐                          │
   │  OBSERVE    │  read actual state       │
   │  current    │  from workers/store      │
   └──────┬──────┘                          │
          │                                 │
          ▼                                 │
   ┌─────────────┐                          │
   │  COMPARE    │  desired vs actual       │
   │  compute    │  → diff / plan           │
   │  diff       │                          │
   └──────┬──────┘                          │
          │                                 │
          ▼                                 │
   ┌─────────────┐                          │
   │  ACT        │  issue idempotent,       │
   │  corrective │  bounded corrective      │
   │  action     │  action                  │
   └──────┬──────┘                          │
          │                                 │
          └───── wait / re-trigger ─────────┘
```

### Why reconciliation beats one-shot orchestration

| One-shot orchestration | Reconciliation |
|---|---|
| "Run this sequence of steps" | "Make reality match this spec" |
| If step 3 crashes, workflow is stuck | Loop resumes from current observed state |
| Hard to handle drift (manual changes, failures) | Drift is just another diff to reconcile |
| Retries require explicit plumbing | Retries are the default mode |
| State lives in the orchestrator | State lives in the store; orchestrator is stateless |

### Idempotency is non-negotiable
Every action must be safe to repeat.
- `create_if_not_exists(pod, spec)` not `create(pod)`
- `set_replicas(n)` not `add_replica()`
- IDs derived deterministically from desired state, so retries collide and dedupe at the store.

### Eventual convergence means
- At any instant, actual may diverge from desired
- Given enough time without new changes, it converges
- Strong invariant: no reconcile step leaves the system further from desired than before (monotonic progress, where possible)

### Duplicate events must be tolerated
Watch streams can deliver the same event twice, or an event may fire while a full resync is in flight. The controller must produce the same final action regardless of event cardinality. Rule: **key your actions off the observed state, not off the triggering event.**

### Retries with backoff

```
   event arrives ──► enqueue ──► dequeue ──► reconcile
                        ▲                       │
                        │   on error:           │
                        └── requeue with ───────┘
                            exp backoff + jitter
```

- Transient (network, rate limit) → exponential backoff + jitter
- Permanent (bad spec, auth) → surface in status, stop retrying
- Per-resource retry budget so one sick object can't starve the queue

---

## 5. Scheduling at interview depth

Scheduling = "I have N pending units of work and M units of capacity; match them under constraints."

### Placement decisions weigh
- **Resource requirements** — CPU, memory, GPU type, accelerator count
- **Affinity / anti-affinity** — "put these together" or "spread these apart"
- **Constraints** — zone, hardware class, license, data locality
- **Fairness** — no tenant monopolizes shared capacity (DRF, weighted shares)
- **Priority / preemption** — high-priority work can evict lower

### Bin packing vs spreading

```
  BIN PACKING                    SPREADING
  (maximize utilization)         (maximize availability)

  ┌────────┐ ┌────────┐          ┌────────┐ ┌────────┐ ┌────────┐
  │ A B C  │ │        │          │   A    │ │   B    │ │   C    │
  │ D E    │ │        │          │        │ │        │ │        │
  └────────┘ └────────┘          └────────┘ └────────┘ └────────┘
   Node 1     Node 2              Node 1     Node 2     Node 3

  + cost efficient               + failure isolation
  + fewer nodes running          + predictable tail latency
  − one node failure = big blast − wasted capacity
  − noisy neighbors              − higher infra cost
```

Real schedulers blend both: pack within a failure domain, spread across failure domains.

### Queueing of pending work
A scheduler is fed by one or more queues. Queues let you:
- Decouple arrival rate from decision rate
- Apply fairness across tenants
- Retry un-schedulable work without blocking the loop

### Preemption (high level)
If a higher-priority task can't be placed, find lower-priority tasks to evict. Must be done carefully — preempting mid-training-step is expensive. Most schedulers use *graceful* preemption with a drain period (PDBs, graceful shutdown signals).

### Why separate scheduling from execution
- Scheduler is a **decision engine**: "this pod → that node."
- Executor is a **mechanism**: "make this pod run on this node."
- Scheduler crashes shouldn't affect running workloads.
- Schedulers can be stateless and replaceable; executors keep local runtime state (mount points, process supervision).

### What makes scheduling hard at scale
1. **Decision rate** — 10K nodes × churn = many decisions/sec. Sequential schedulers cap out.
2. **Stale view** — by the time you place a pod, the chosen node may be full. Use OCC: write fails, you re-schedule.
3. **Constraint complexity** — NP-hard in general. Approximate: score + pick top-K.
4. **Heterogeneous hardware** — GPU types, TPU pods, FPGAs, mixed generations.
5. **Fairness vs throughput** — strict fairness starves latency-sensitive work; pure throughput starves small tenants.

### Scheduler architectures

```
  MONOLITHIC              TWO-LEVEL (Mesos-like)     SHARED STATE (Omega-like)
  single scheduler        framework schedulers       many schedulers
  sees everything         negotiate offers           contend via OCC

  ┌────────┐              ┌────────┐ ┌────────┐      ┌──────┐┌──────┐┌──────┐
  │ Sched  │              │Sched A │ │Sched B │      │Sched1││Sched2││Sched3│
  └───┬────┘              └──┬─────┘ └────┬───┘      └──┬───┘└──┬───┘└──┬───┘
      │                      └── offers ──┘             └───────┼───────┘
      ▼                            ▼                            ▼
   Cluster                      Cluster                    Shared State
                                (broker)                   (OCC writes)

  simple, limited throughput   scales horizontally        scales, conflicts
```

---

## 6. Must-know concepts

- **Desired state** — what the user asked for (declarative spec)
- **Observed state** — what the system currently reports (actual reality)
- **Convergence** — observed → desired over time, absent new changes
- **Control loop** — observe/compare/act, repeating
- **Idempotent action** — safe to apply 1 or N times with the same final effect
- **Work queue** — buffer of pending reconciliations, typically per-key rate-limited
- **Leader election** — exactly one active controller per loop; backed by a consensus store via a lease
- **Lease / fencing token** — monotonically increasing token attached to actions; state store rejects writes with stale tokens
- **Optimistic concurrency (OCC)** — read version → compute change → write-if-version-unchanged; retry on conflict
- **Retry with backoff** — exponential + jitter; distinguish transient from permanent
- **Stale state** — cached/watched data that lags reality; handled by resync and versioned writes
- **Drift** — observed diverges from desired without a new intent change (manual edit, partial failure, external actor)
- **Rollout** — apply new desired state gradually (canary, %-based, wave-by-wave)
- **Rollback** — revert desired state to previous; reconciler converges back

### Fencing in one picture

```
   t=0   Controller A acquires lease, token = 7
   t=1   Controller A gets GC-paused for 30s
   t=2   Lease expires; Controller B acquires lease, token = 8
   t=3   Controller B issues writes with token 8  ─► ACCEPTED
   t=4   Controller A wakes up, issues writes with token 7 ─► REJECTED
                                                              (store checks token)

   Without fencing: A and B both write. Corruption.
   With fencing:    A's stale writes are rejected at the store.
```

### OCC in one picture

```
   Reader:  GET obj  ─►  {value=X, version=42}
   Compute: new = f(X)
   Writer:  PUT obj if version==42  ─►  either SUCCESS (version=43)
                                        or CONFLICT (someone else wrote)
                                        on conflict: re-read and retry
```

---

## 7. Failure and scaling issues

### Controller crash
- Another replica takes over via leader election.
- In-flight action must be idempotent so the successor doesn't duplicate it.
- Work queue is rebuilt by a full resync on startup.

### Scheduler leader failover
- Active scheduler dies; standby wins lease after timeout (typically 5–15s).
- Pending decisions not yet persisted are lost; new leader re-derives from the queue.
- Decisions that *were* persisted are picked up by executors as normal.

### Duplicate work
- Two controllers each think they're leader during a partition.
- Or a single controller retries a non-idempotent action.
- Mitigations: fencing tokens, idempotency keys, exactly-once-**effect** via the store (not the network).

### Stale reads from state store
- Async replicas may lag the leader.
- Correctness-critical reads → leader. Status/list reads → followers OK.
- Use resource versions on conditional writes.

### Split brain

```
           Network Partition
                 │
       ┌─────────┴─────────┐
       ▼                   ▼
   ┌───────┐           ┌───────┐
   │ Ctrl A│           │ Ctrl B│
   │ lease │           │ lease │
   │ = 7   │           │ = 8   │
   └───┬───┘           └───┬───┘
       │                   │
       ▼                   ▼
  writes w/ 7         writes w/ 8
       │                   │
       └─────► Store ◄─────┘
                │
             accepts only
             highest token
             → A rejected
```

Root causes: lease timeout < max clock skew, long partition, GC pause.
Mitigations: fencing at the store, short leases + monotonic tokens, never trust "I am leader" without a fresh lease check just before writing.

### Worker crash during task execution
- Task is in unknown state: started? half-done? committed?
- Mitigations:
  - Tasks write checkpoints + status to control plane
  - Controller treats "unknown" as "retry after grace period"
  - Task operations keyed by task ID → idempotent
  - For non-idempotent external side effects, use idempotency tokens at the remote service or 2PC

### Task stuck in unknown state
- Timeout-and-assume-failed after a conservative deadline.
- Before declaring dead, send a fencing signal (revoke worker's lease/credentials) so the old worker can't still act.
- Only then restart elsewhere.

### Thundering herd after outage

```
   Outage ends →  all agents reconnect at once  →  ░░░░░░░░░  API melts
   With jitter →  reconnects spread over N min  →  ▒ ▒ ▒ ▒ ▒   steady load
```

Mitigations:
- Jittered reconnect backoff on agents
- Rate-limited admission at API server
- Watch resumes use `resourceVersion` to avoid full re-list
- Priority queues so critical reconciles happen first

### Control-plane overload
- Symptoms: watch lag, reconcile queue growth, 503s on API.
- Mitigations: shard controllers by namespace/tenant, separate read/write paths, back-pressure at API, bounded work queues with drop policies for low-priority items.

### Slow convergence
- Reconcile interval too long, or queue starvation on hot keys.
- Mitigation: per-key work queues with rate limiting; priority lanes for user-facing resources; watch-driven triggers instead of pure polling.

---

## 8. Multi-controller and coordination concerns

### One active leader per control loop, many replicas for HA

```
   ┌─────────────┐   ┌─────────────┐   ┌─────────────┐
   │ Replica 1   │   │ Replica 2   │   │ Replica 3   │
   │ [LEADER]    │   │ (standby)   │   │ (standby)   │
   │ holds lease │   │             │   │             │
   └─────────────┘   └─────────────┘   └─────────────┘
          │
          ▼
     actuates
```

Standby is cheap; failover is fast (seconds); correctness enforced via fencing.

### Sharding control responsibilities

```
  Resources A–H    I–P    Q–Z
       │            │      │
       ▼            ▼      ▼
    Ctrl 1       Ctrl 2  Ctrl 3
    (leader)     (leader)(leader)
```

Split the keyspace (hash / tenant / zone) across leader-per-shard. Each shard is an independent control loop with its own lease. A **resharding protocol** is needed when topology changes — hand off cleanly, fence the old owner before the new owner starts acting.

### Avoiding conflicting actuators
- Two controllers must never own the same field on the same object.
- Enforce via schema (field ownership) or single-writer-per-resource-type.
- Kubernetes solves this with server-side apply field ownership. Simpler systems just say "only the deployment controller writes replica counts."

### Centralized scheduling is fine when
- < ~10K decisions/sec
- Uniform constraints
- Single administrative domain
- You want strict global fairness/packing

### Distributed / hierarchical scheduling is needed when
- Federated multi-region systems
- Extreme throughput (Sparrow-style random sampling)
- Heterogeneous workloads with specialized schedulers (batch vs online vs interactive)

```
  HIERARCHICAL SCHEDULING

       ┌──────────────┐
       │ Global Sched │   (routes jobs to regions)
       └──────┬───────┘
              │
      ┌───────┼────────┐
      ▼       ▼        ▼
   Region1 Region2  Region3
    sched   sched    sched
      │       │        │
      ▼       ▼        ▼
    nodes   nodes    nodes
```

---

## 9. Interview reasoning patterns

**Q: Why reconciliation instead of imperative orchestration?**
Reconciliation makes failure recovery free. Intent is persisted once; the loop always picks up from current reality. An imperative workflow has to encode every failure branch explicitly and keeps state in the orchestrator, which becomes a SPOF. Reconciliation is also naturally tolerant of drift — external changes just look like a diff to fix.

**Q: What belongs in control plane vs data plane?**
Control plane: desired state, decisions, policy, placement, coordination. Data plane: the actual work — serving requests, running containers, moving packets. Rule of thumb: anything in the request hot path is data plane; anything that decides what should exist is control plane. Goal: control-plane outages must not take down the data plane (**static stability**).

**Q: How do you make control actions safe under retries?**
Make them idempotent. Use deterministic IDs derived from desired state so retries collide. For non-idempotent side effects, use idempotency tokens on the downstream API or 2PC if supported. Separate "decide" from "act": the decision can be logged and recovered; the act references the logged decision ID.

**Q: How do you avoid duplicate or conflicting work?**
Single-leader-per-loop via lease. Fencing tokens on all mutating actions — the store rejects writes with stale tokens, so a paused-then-resumed old leader can't corrupt state. Field ownership so two controllers never write the same field. OCC on all updates.

**Q: How do you design scheduler failover?**
Stateless schedulers with leader election. All pending work in the persisted queue, never in scheduler memory. New leader re-derives its view on startup. Decisions use OCC so stale decisions (from before failover) get rejected rather than applied. Target failover in < 30s.

**Q: How do you reason about convergence and correctness?**
Define desired and observed state precisely. Show each reconcile step makes monotonic progress or is a no-op. Identify invariants the system must never violate ("no two primaries for the same shard") and show they hold across any interleaving — usually by reducing to the lease/fencing mechanism.

**Q: What are the first bottlenecks and operational pain points?**
(1) State store write QPS and watch fan-out. (2) Reconcile queue backlog on popular resource types. (3) Scheduler decision latency as node count grows. (4) Thundering herd after control-plane restarts. Mitigations in order: watch caching, per-key rate limits, shard controllers, jittered agent reconnects, priority queues.

---

## 10. Common candidate mistakes

- **Drawing the control plane as just an API server.** API is one component. Without controllers, scheduler, and state store, nothing converges. Interviewers want to see the loop.
- **Ignoring desired vs actual state.** Treating the system as "submit job → run job" misses the entire reason control planes exist.
- **Hand-waving retries and idempotency.** "We'll retry on failure" without addressing dedup / idempotency is a red flag.
- **Ignoring stale state and race conditions.** Assuming your cache matches the store. Assuming your "leader" status is still valid 10s after you checked.
- **Coupling scheduler and executor.** If the scheduler dies, running jobs shouldn't care. If you can't explain why they're separate, you don't understand the architecture.
- **Ignoring leader election / fencing for mutating actions.** Two controllers mutating the same resource is a data-corruption incident waiting to happen.
- **Assuming events are reliable and unique.** Watches drop, duplicate, and reorder. The reconciler must be correct under any event sequence.
- **Ignoring backlog growth.** "The controller will reconcile it eventually" with no bound on queue depth or per-key fairness.
- **Treating the state store as a general-purpose DB.** etcd is not Postgres. Large objects, high write rates, and watches don't mix.
- **Mixing control-plane and data-plane SLAs.** Control plane: low-QPS / high-consistency. Data plane: high-QPS / high-availability. Designing both against the same tier causes misery.

---

## 11. Final cheat sheet

### Control plane vs data plane

| Aspect | Control plane | Data plane |
|---|---|---|
| Purpose | Decide what should exist | Execute the work |
| QPS | Low (ops/sec) | High (req/sec) |
| Consistency | Strong (for decisions) | Eventual is often fine |
| Latency budget | Seconds OK | ms or μs |
| Failure impact | Can't change state; existing work keeps running | User-visible outage |
| Typical store | etcd / Spanner / ZK | In-memory / local disk / sharded DB |
| Primary goal | Correctness & convergence | Throughput & availability |
| Scaling axis | Shard by resource / tenant | Shard by request / key |

### Scheduler vs controller

| Aspect | Scheduler | Controller |
|---|---|---|
| Job | Placement decisions (where/when) | Drive a resource to desired state |
| Input | Pending work + capacity view | Desired spec + observed status |
| Output | Binding (work → resource) | CRUD actions on managed objects |
| Scope | Cross-object global optimization | One resource type |
| Statefulness | Mostly stateless | Mostly stateless (state in store) |
| Failure mode | Queue grows until leader recovers | Drift grows until leader recovers |
| Typical count per system | 1 (or a few specialized) | Many (one per resource type) |

### Decision framework

```
   Given a request to design a distributed system:

   1. Is there intent separate from execution?
         → Yes: declarative control plane with desired state
         → No:  maybe a pure data plane (e.g., a KV store) — skip control plane

   2. Does state need to converge under failure?
         → Yes: reconciliation loop, not imperative workflow

   3. Is there placement/allocation of work across resources?
         → Yes: separate scheduler from controller
         → No:  single controller per resource type is enough

   4. Multiple writers to the same field possible?
         → Yes: leader election + fencing + field ownership
         → No:  single controller, OCC for concurrent clients

   5. What's the state store?
         → Strongly consistent: etcd / Spanner / FoundationDB
         → Size/QPS > single cluster: shard by tenant/resource type

   6. How do workers learn about work?
         → Small scale: push from control plane
         → Large scale: pull + watch, with jittered reconnect

   7. What's the blast radius of control-plane failure?
         → Data plane should keep serving (static stability)
         → Existing assignments persist; only new changes are blocked
```

### 10 likely interview questions with short strong answers

**1. Control plane vs data plane — what's the difference and why does it matter?**
Control plane decides; data plane does. Separating them lets data plane survive control-plane outages (static stability) and lets you scale them independently.

**2. Why reconciliation instead of orchestration?**
Reconciliation tolerates drift, crashes, and external changes by design. Orchestration has to encode every failure branch explicitly and keeps state in a fragile orchestrator.

**3. How do you guarantee a control action isn't applied twice?**
Idempotent actions keyed by deterministic IDs, fencing tokens rejected by the state store on stale leaders, idempotency keys on downstream APIs. Exactly-once *effect*, not exactly-once *delivery*.

**4. How do you handle a controller crash mid-action?**
A replica takes over via leader election. On startup it full-re-lists to rebuild queue state. The in-flight action was either idempotent (safe to redo) or its effect was persisted (recognized as done via observed state).

**5. What's a fencing token and why do you need it?**
A monotonic token attached to the lease. The state store rejects writes with outdated tokens. Without it, a GC-paused or partitioned old leader can corrupt state after a new one is elected.

**6. How does a scheduler handle stale node state?**
OCC: bind pod→node with a resource version; if the node's state changed, the write fails and the pod goes back in the queue for rescheduling.

**7. How do you scale a control plane beyond a single etcd cluster?**
Shard controllers by tenant/namespace/resource-type, each with its own state store. Route at the API layer. Avoid cross-shard transactions; make cross-shard operations eventually consistent with reconciliation.

**8. How do you avoid thundering herd after a control-plane restart?**
Jittered reconnect on agents, rate-limited API admission, watch resumes via resourceVersion (not full re-list), priority lanes for critical reconciles.

**9. How do two controllers avoid fighting over the same resource?**
Field ownership (schema enforces who writes what) or single-writer-per-resource-type (only one controller type mutates it). Fencing prevents stale leaders of the same type from conflicting.

**10. When would you use hierarchical scheduling instead of a single scheduler?**
Multi-region systems where a global scheduler can't meet decision-rate or locality needs; very large clusters where a single scheduler is the bottleneck; heterogeneous workloads where specialized schedulers (batch vs online) benefit from separation.

---

### Five phrases that signal staff-level thinking

1. "I'd separate decide-from-act so retries are safe."
2. "Fencing tokens so the store rejects stale writes."
3. "Static stability — data plane should survive a control-plane outage."
4. "OCC on the binding so a stale scheduling decision gets rejected, not applied."
5. "Per-key work queue with rate limiting to avoid hot-key starvation."