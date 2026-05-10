---
title: "3-Distributed Job Processing System"
description: "Distributed Job Processing System"
---
"Design a job processing system that handles 2B jobs/day with retry, scheduling, priority, and fair-share across tenants. Some jobs run for milliseconds, some for hours. Walk me through it."
---

## 1. Reframing

Before I draw a single box, three reframings.

**Job duration heterogeneity is the central architectural pressure**, not throughput. 2B jobs/day at 23K/sec is a moderate distributed-systems problem; the hard part is that 10ms jobs and 12-hour jobs share the wire. A single worker pool serving both has head-of-line blocking on short jobs or aggressive ack-deadline kills on long jobs. **[STAFF SIGNAL: duration-segmentation-as-central]** I will segment the worker fleet into duration-classed pools (short / medium / long) with different visibility, retry, checkpoint, and scaling semantics per pool. Everything downstream — fair-share, retry, state store sizing — falls out of this.

**Fair-share is a real scheduling algorithm, not a feature flag.** **[STAFF SIGNAL: fair-share-as-real-algorithm]** I'll implement weighted fair queueing with per-tenant deficit accounting, where the unit of share is *slot-time* (worker-seconds), not job count — because a tenant running 100 12-hour jobs consumes ~10⁶× the resource of a tenant running 100 1-second jobs. Round-robin is wrong here.

**Long-running jobs are architecturally different, not "longer short jobs."** **[STAFF SIGNAL: long-job-as-different-shape]** They need heartbeat-based liveness (not visibility timeout), application-level checkpointing (not full-restart-on-failure), cooperative cancellation (not preemption), and slot-time resource accounting.

**Delivery is at-least-once. Idempotency is the customer's responsibility.** **[STAFF SIGNAL: at-least-once honesty]** Exactly-once execution across worker crashes is impossible in the general case (the worker can die between side-effect and ack). What the platform offers is a stable `job_id` and an *effectively-once* optimization (durable execution-state check before re-running). I will say this to customers explicitly; pretending otherwise creates worse bugs than it prevents.

---

## 2. Scoping

**[STAFF SIGNAL: scope negotiation]** Committing to assumptions:

- **Job types.** Mixed: ~70% I/O-bound (HTTP fan-out, DB writes, email/SMS), ~25% compute (image/video processing, ETL), ~5% workflow-style (multi-step with state). Workflow jobs are out of scope for this design — they go on a Temporal-style durable-execution layer that *uses* this queue as its activity transport.
- **Idempotency.** Job authors are responsible for idempotent effects. The platform supplies `job_id` and an effectively-once execution-state check; it does not supply transactional side-effect guarantees against arbitrary external systems.
- **Customer count.** ~10K paying tenants, with a 100× skew: ~50 tenants drive 80% of volume, the long tail submits <100 jobs/sec each. Free-tier ~100K accounts share a small slice.
- **Regions.** Multi-region active-active for submission and dispatch (US-East, US-West, EU). State store regionally partitioned by tenant home region; cross-region jobs are rare and pay a latency penalty. No global-consistency guarantees on dispatch order.
- **SLOs.** Short pool: p99 submission-to-first-dispatch <500ms. Medium: <5s. Long: <60s. DLQ visibility within 1min of terminal failure.
- **Cost model.** Per-job-base-fee + slot-time billing, surfaced per tenant.

Out of scope: cross-tenant job dependencies, FIFO ordering guarantees (offered as a separate product), and exactly-once semantics.

---

## 3. Capacity math

**[STAFF SIGNAL: capacity math]**

```
Workload distribution (2B jobs/day = 23.1K/sec average, 5–10× burst → ~150K/sec peak)

Tier        | Share  | Avg dur  | Avg arrivals/sec | Peak arrivals/sec | Concurrency (avg)
------------|--------|----------|------------------|-------------------|------------------
Short  <1s  | 90%    | 200ms    | 20.8K            | 150K              | ~4.2K slots
Medium 1s–1m| 9%     | 15s      | 2.08K            | 15K               | ~31K slots
Long   1m–1h| 1%     | 10min    | 231              | 1.6K              | ~140K slot-min
LongLong>1h | 0.01%  | 4h       | 2.3              | 16                | ~33K slot-hr → ~33K concurrent
------------|--------|----------|------------------|-------------------|------------------
Worker count (peak, with 30% headroom):
  Short pool  : 1 worker = 50 concurrent jobs (asyncio/coroutines) → ~110 workers
  Medium pool : 1 worker = 8 concurrent jobs                       → ~5K workers
  Long pool   : 1 worker = 2 concurrent jobs (heavyweight)         → ~70K workers
  LongLong    : 1 worker = 1 job                                   → ~33K workers
                                                          Total fleet ≈ ~108K workers

Storage:
  Active queue depth  : peak 10M jobs × 2KB metadata = 20 GB working set
  Scheduled store     : ~50M scheduled-future jobs × 2KB = 100 GB
  State store writes  : 4 writes/job (enqueue, dispatch, ack, complete) × 150K/s peak
                      = 600K writes/sec peak. THIS IS THE HOT PATH.
  DLQ                 : retain 7d at 0.1% failure rate = ~1.4B rows × 4KB = 5.6 TB
```

The fleet is dominated by long-runners by *count* but short jobs dominate dispatch *rate*. The state store at 600K writes/sec peak is the architectural bottleneck — not the queue, not the workers.

---

## 4. High-level architecture

```
                       ┌──────────────────────────────────────┐
   Customer ─POST─►    │  Submission API (regional, stateless)│
                       │  - auth, rate-limit, validate        │
                       │  - assign job_id, classify pool      │
                       │  - write to State Store + Queue      │
                       └──────────┬───────────────────────────┘
                                  │
                                  ▼
        ┌──────────────────── State Store (sharded by tenant_id) ──────────────────┐
        │  job_id │ tenant │ pool │ status │ attempts │ payload_ref │ schedule_at  │
        │  CockroachDB / FoundationDB cluster, per-region home, async cross-region│
        └──────────────┬─────────────────────────────────────────────────┬────────┘
                       │                                                 │
                       ▼                                                 ▼
        ┌─────────────────────────┐                          ┌─────────────────────┐
        │ Active Queues           │                          │ Scheduled Store     │
        │ (per-pool, per-tenant)  │                          │ (time-indexed,      │
        │ Redis Streams / Kafka   │                          │  ZSET by execute_at)│
        └──────┬──────────────────┘                          └─────────┬───────────┘
               │                                                       │
               │     ┌─────────── Scheduler Service ──────────┐       │
               │     │ - reads per-tenant queue depths        │       │
               │◄────┤ - weighted fair-queueing (deficit)     ├───────┤
               │     │ - dispatches to pool-specific workers  │       │
               │     │ - "scheduled tick" promotes due jobs   │       │
               │     └────────────────────────────────────────┘       │
               │                                                       │
               ▼                                                       ▼
   ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  ┌──────────┐
   │  Short pool     │  │  Medium pool    │  │  Long pool      │  │ DLQ      │
   │  ~110 workers   │  │  ~5K workers    │  │  ~100K workers  │  │ (S3+idx) │
   │  visibility 10s │  │  visibility 5m  │  │  heartbeat 30s  │  │          │
   │  no checkpoint  │  │  opt checkpoint │  │  required ckpt  │  │          │
   └────────┬────────┘  └────────┬────────┘  └────────┬────────┘  └──────────┘
            └────────────────────┴────────────────────┘
                              ▼
                  Checkpoint store (S3 / blob, per-job-id key)
                              ▼
                  Customer surface: status API, DLQ replay, dashboards
```

The scheduler is a *service*, not a property of the queue. **[STAFF SIGNAL: rejected alternative]** Rejected: workers pulling directly from per-tenant queues. Why: fair-share requires global tenant-deficit state; pushing this into every worker is N² gossip and broken under partial failure. Rejected: a single global queue with priority field. Why: per-tenant queue depth limits and per-pool segmentation become impossible to enforce. Rejected: SQS / SNS as the substrate. Why: 12h max visibility timeout (long jobs need 24h+), no per-tenant fair-share, no scheduled-job primitive.

---

## 5. Fleet segmentation by duration class

**[STAFF SIGNAL: duration-segmentation-as-central]** Re-stated, because this is the load-bearing decision. Three pools, four with the long-long edge case:

```
                     ┌─────────────────────────────────────┐
   Submitted job ──►│  Pool classifier                    │
                    │  1. submitter-declared (with audit)  │
                    │  2. job_type lookup table (default)  │
                    │  3. measured EWMA p95 dur (override) │
                    └──────┬──────────┬──────────┬─────────┘
                           │          │          │
                           ▼          ▼          ▼
                       ┌──────┐  ┌────────┐  ┌──────┐
                       │SHORT │  │ MEDIUM │  │ LONG │
                       │<1s   │  │ 1s–1m  │  │ >1m  │
                       └──┬───┘  └───┬────┘  └──┬───┘
                          │          │          │
       visibility/lock:  10s        5min       heartbeat 30s
       concurrency/wkr:  50         8          1–2
       checkpoint:       no         optional   REQUIRED if >5 min
       retry budget:     5x         3x         1–2x (expensive)
       autoscale signal: queue depth + age, per pool
```

**Classification.** Three-tier:

1. **Submitter-declared at job-type registration.** A `send_password_reset_email` is registered as `pool=short`. Audited: if a job-type's measured p95 violates its declared pool >1% of the time, the platform forcibly reclassifies and emits a customer alert.
2. **Default by registered job_type.** Most jobs hit this path — a hash lookup, no per-job decision cost.
3. **Override on measured drift.** A streaming p95 EWMA per job_type updates the pool assignment. Reclassification is sticky for 1h to avoid flap.

**The "wrong pool" case.** A short-pool job runs 5 minutes:
- t=10s: visibility timeout fires; scheduler re-dispatches to another short worker. Original worker is still running.
- t=20s, 30s, ...: repeat. Now N workers are running the same job.
- Detection: per-`job_id` execution-state row in the state store with a `currently_running_on` field; on re-dispatch, the new worker checks and sees an active execution. If found and within heartbeat threshold, the new worker no-ops (the original keeps going, with extended visibility). If no heartbeat, the new worker takes over.
- Mitigation: emit `pool_violation` metric per job_type. After 100 violations in 1h, the classifier reclassifies. The current jobs ride out; future jobs go to medium/long.

**Quantification.** With ~108K total workers, ~30% of cost is the long-long pool (33K workers each pinned to one job for hours). Tempting to "just use serverless" for long-runners — rejected, because cold starts at the start of a 4h job aren't the issue, but graceful checkpointed restart on infra deprecation is, and serverless platforms don't expose that primitive. Long-pool workers run on dedicated EKS node groups with 24h drain windows.

---

## 6. Fair-share scheduling

**[STAFF SIGNAL: fair-share-as-real-algorithm]**

The unit of share is **slot-time** (worker-seconds), not job count, because job durations span 7 orders of magnitude. The scheduling algorithm is **Weighted Fair Queueing with deficit accounting**, per pool.

```
            ┌────────────────────── Scheduler tick (per pool, every 10ms) ──────────────────────┐
            │                                                                                   │
            │  for each tenant T in pool P:                                                     │
            │      T.deficit += T.weight × tick_interval × pool_capacity_slot_time              │
            │                                                                                   │
            │  candidates = [T for T in tenants if T.queue_depth > 0 and T.deficit > 0]         │
            │  T* = argmax(candidates, key=lambda T: T.deficit / T.weight)                      │
            │  job = T*.queue.peek()                                                            │
            │  T*.deficit -= estimated_slot_time(job)        # forecast cost up-front           │
            │  dispatch(job, worker)                                                            │
            │                                                                                   │
            │  on job completion:                                                               │
            │      T.deficit += (estimated_slot_time - actual_slot_time)   # reconcile          │
            └───────────────────────────────────────────────────────────────────────────────────┘
```

**Tier-aware weights.** **[STAFF SIGNAL: tier-aware multi-tenancy]**
- Free tier: shared weight pool of 1.0, divided among all free accounts.
- Standard: weight = 10 per tenant.
- Pro: weight = 100, with a *minimum guarantee* of 1% of pool capacity (deficit floor — even if everyone bursts, Pro gets its minimum).
- Enterprise: weight = 1000 with a *dedicated reservation* — a hard slice of the pool that nobody else can consume. The "you cannot be starved" guarantee.

Implementation: enterprise reservations are a *separate* logical pool with its own worker subset. The remaining pool runs WFQ across non-enterprise tenants. This is operationally simpler than implementing reservations within a single WFQ — the reservation invariant is enforced by physical worker allocation, not scheduler arithmetic.

**Per-tenant queue depth limits.** 10M jobs/tenant in the active queue. Past limit, submission is rejected with HTTP 429 and a customer-visible error pointing at the tenant's dashboard. This is the hard backpressure boundary.

**Estimated slot-time.** For a job, estimate = EWMA of measured duration for that job_type. On first execution of a new job_type, conservative default (10s for short pool, 60s for medium, 10min for long). Reconcile on completion. A tenant whose jobs run *longer* than estimated burns deficit faster, naturally throttling future dispatches; a tenant whose jobs run *shorter* gets credited back.

**Noisy neighbor case.** Tenant X submits 10M jobs in 1s. Their queue fills (up to 10M limit; remainder rejected). Their per-tick deficit accrual is `weight_X × tick × pool_capacity` — bounded. Their dispatch rate is therefore bounded by their share, not their submission rate. Other tenants are unaffected. 10M jobs at their share might take hours to drain — that's correct; that's what fair-share *means*.

**Why not DRF?** Dominant Resource Fairness handles multiple resource dimensions (CPU, memory, GPU). Here, slot-time is the dominant resource by 100×; CPU/memory variation within a pool is small enough that single-resource WFQ is sufficient and operationally simpler. **[STAFF SIGNAL: rejected alternative]**

---

## 7. Scheduled / delayed jobs

**[STAFF SIGNAL: scheduled-jobs-as-first-class]**

Every job has `execute_at`. Default = now (immediate). Future = scheduled.

**Storage.** A per-region sorted store, sharded by `(tenant_id, execute_at)`. Implementation: Redis Cluster with ZSETs per shard, score = unix_ms. Backed by durable state-store rows (the ZSET is a hot index; the state store is the source of truth).

```
   Scheduled store (logical, per region)

   shard_0: ZSET keyed by (tenant_id_hash % N == 0)
     score=1715300000000  member=job_id_aaa   ─┐
     score=1715300100000  member=job_id_bbb    │  sorted by execute_at
     score=1715303600000  member=job_id_ccc    │
     score=1715900000000  member=job_id_ddd   ─┘

   Promoter loop (per shard, 1Hz):
     due = ZRANGEBYSCORE(shard, -inf, now())
     for job in due:
       move to active queue (with jitter)
       ZREM from shard
       state_store.update(status=queued)
```

**Dispatch loop.** A "promoter" service per shard runs every 1s, queries `score <= now()`, atomically moves due jobs from scheduled-store to active-queue, and updates state-store status. The promoter is sharded so no single instance scans the whole space.

**Thundering herd on scheduled dispatch.** 10M jobs scheduled for 09:00 Monday. At 09:00 the promoter wants to dump 10M jobs into the active queue, which would saturate the state store and steamroll fair-share.

Mitigations:
1. **Submission-time jitter.** When `execute_at - now > 1h`, the submission API adds uniform jitter ±60s. Customer-opt-out for hard deadlines.
2. **Promoter rate limit.** Each promoter shard caps promotion to N jobs/sec into the active queue. Excess waits one tick. For a large customer scheduling 10M jobs at once, the promoter spreads the dispatch over minutes.
3. **Per-tenant rate limit on promotion.** Reuses the standard fair-share path — promoted jobs hit the active queue and are dispatched at the tenant's fair share, not all at once.

**Cancellation race.** Customer cancels job J at t=9:00:00.500; promoter ran at t=9:00:00.499 and already moved J to active queue. Resolution: cancellation sets a tombstone in the state store. Worker, on dispatch, reads state-store row before executing — if `status=cancelled`, ack and skip. The state-store read is one extra hop on the hot path, but it's a single primary-key lookup, ~1ms.

**Promoter down.** Scheduled jobs accumulate in the ZSET; nothing executes. Alarm on `oldest_due_unpromoted_age`. Recovery: catch-up promotion, with the same rate limit so recovery doesn't itself stampede.

**Late-execution semantics.** A job scheduled for 09:00 runs at 09:30 if the promoter is lagging. For some jobs (`expire_offer_at_midnight`) this is broken. Customer-supplied `not_after` field: if `now > not_after`, the worker skips and emits a `late_execution` event. Customer must subscribe and handle.

---

## 8. Retry semantics

**[STAFF SIGNAL: per-failure-class retry]** Retry is per-failure-class, not uniform.

```
Retry state machine (per job)

                      ┌─────────────┐
   submission ───────►│   QUEUED    │
                      └──────┬──────┘
                             │ dispatched
                             ▼
                      ┌─────────────┐  heartbeat lost / vis timeout
                      │  IN_FLIGHT  ├────────────────────────────┐
                      └─┬─────┬───┬─┘                            │
                  ack OK│     │   │ exception in job code        │
                        │     │   │                              │
                        ▼     │   ▼                              │
                  ┌─────────┐ │  ┌────────────────┐              │
                  │SUCCEEDED│ │  │  APP_FAILURE   │              │
                  └─────────┘ │  └────┬───────────┘              │
                              │       │                           │
                              │       ▼                           │
                              │  classify failure ◄───────────────┘
                              │   ├── APP_ERROR     → backoff, attempt++
                              │   ├── WORKER_CRASH  → re-dispatch, attempt unchanged
                              │   ├── TIMEOUT       → 1 retry, then DLQ
                              │   ├── DEPENDENCY    → aggressive backoff
                              │   └── POISON        → DLQ now
                              │       │
                              │       ▼
                              │   ┌────────┐  attempts < max_attempts
                              │   │ RETRY  │──────► back to QUEUED with execute_at = now+backoff
                              │   └───┬────┘
                              │       │ attempts >= max_attempts
                              │       ▼
                              │   ┌────────┐
                              └──►│  DLQ   │
                                  └────────┘
```

**Per-failure-class.**

- **APP_ERROR** (job code threw an exception). Counts against `max_attempts`. Default: 5 attempts, exponential backoff with jitter: 1s, 5s, 30s, 5min, 30min, then DLQ. Configurable per job_type.
- **WORKER_CRASH** (heartbeat lost while job in-flight). Does *not* count against `max_attempts` — the job didn't fail intrinsically. Re-dispatched with attempt counter unchanged. **[STAFF SIGNAL: invariant-based thinking]** Invariant: a worker crash should not consume a customer's retry budget.
- **TIMEOUT** (job exceeded its declared time limit). Often signals misclassification or a runaway. 1–2 retries max, then DLQ. Emits a pool-violation metric.
- **DEPENDENCY_FAILURE** (job declares a dependency that returned 5xx). Aggressive retry with circuit-breaker awareness — if 50% of jobs depending on dependency D are failing, the platform halts retries for that job_type for 60s and emits `dependency_circuit_open`.
- **POISON** (a job_type with >50% failure rate over the last 1h). Per-job-type quarantine: max 2 attempts, then DLQ. Prevents one buggy job_type from saturating workers with infinite retries. **[STAFF SIGNAL: blast radius reasoning]**

**Retry budget per tenant.** Tenant retry rate capped at 2× their dispatch rate. Past cap, retries are dropped to DLQ early. Prevents one tenant's broken job from infinite-loop saturating the cluster.

**DLQ.** Customer-facing. Stored in S3 + an index in the state store. Dashboard shows DLQ rate per job_type. One-click bulk replay (with a fresh attempt counter). Retention: 7 days, then GC. **[STAFF SIGNAL: customer-facing observability]**

---

## 9. Long-running jobs

**[STAFF SIGNAL: long-job-as-different-shape]** **[STAFF SIGNAL: checkpointing-required-for-long]**

Long jobs (>1min, especially >1h) break visibility-timeout-based liveness. A 4h job with 5min visibility would require 48 timeout extensions; a missed extension = re-dispatch = duplicate work.

**Heartbeat instead of visibility timeout.** Worker sends `heartbeat(job_id, progress)` every 30s to the state store. Scheduler watches heartbeat freshness. Heartbeat older than `3 × interval` (90s) → worker considered dead, job re-dispatched. The lock is the heartbeat row, not a queue-level visibility timer.

**Checkpointing.** **Required** for jobs >5min. The job code calls:

```
ctx.checkpoint(state_blob)  # platform writes to S3 keyed by job_id
state = ctx.load_checkpoint() or initial_state()  # on restart
```

On crash + re-dispatch, the new worker calls `load_checkpoint()` and resumes from the latest snapshot. **[STAFF SIGNAL: invariant-based thinking]** Invariant: long-job state survives worker death.

**Checkpoint cost.** Serialization + S3 write = 100ms–10s depending on state size. Not free. Policy: jobs <5min don't checkpoint (full restart is cheaper than the operational complexity). Jobs 5–60min checkpoint at 1min intervals. Jobs >1h checkpoint at 5min intervals or at logical boundaries (e.g., per-batch in a batch processor) — author's choice. The platform supplies the API; the author supplies the granularity.

**Cooperative cancellation.** **[STAFF SIGNAL: cooperative-cancellation]** Customer issues `DELETE /jobs/{id}`. Platform writes `cancellation_requested=true` to the state store. The worker's job context exposes `ctx.is_cancelled()`; the job code is *required* to call this between major work units (typically: at every checkpoint boundary). On observing cancellation, the job exits gracefully — emits a final event, releases resources. This is cooperative; preemption (SIGKILL) is only used after a 5min cancellation grace period as the hard fallback. Forced kill loses the in-flight checkpoint window but cleans up the worker.

**Resource accounting.** A long worker holds a slot for hours. Fair-share's slot-time accounting captures this naturally — a tenant running 100 4h jobs consumes 400 slot-hours, deducted from their deficit. A tenant running 100 1s jobs consumes 100 slot-seconds. Equal *job count*, vastly different *cost*. The deficit-based scheduler prevents the long-job tenant from consuming more than their share.

**Long-pool autoscaling.** Different from short-pool. Workers can't be drained mid-job; scale-down requires waiting for jobs to complete or forcing checkpoint+kill. Policy: scale up aggressively (queue age >60s → +N workers); scale down conservatively (workers idle for 30min → drain). Cost overshoot is acceptable; mid-job termination is not (causes customer-visible duplicate work).

---

## 10. Idempotency and at-least-once

**[STAFF SIGNAL: at-least-once honesty]**

The honest contract:

> Every successfully-submitted job is dispatched to **at least one** worker and runs **at least once**. Under worker failure, network partition, or visibility-timeout race, the same job may run more than once. Job code must be idempotent.

Why exactly-once is impossible: a worker can crash *after* the side effect (e.g., DB write) and *before* the ack. The platform sees no ack and re-dispatches. Two executions, one side effect — or two side effects if the side effect was non-idempotent. No system can prevent this without coordinating with the side-effect target, which is opaque to the queue.

**What the platform supplies.**

1. **Stable `job_id`.** Same UUID across all attempts. Job code uses it as an idempotency key against external systems (DBs that support `INSERT ... ON CONFLICT (idempotency_key) DO NOTHING`, APIs that accept an idempotency-key header, etc.).
2. **Effectively-once optimization.** Before executing, the worker reads the state-store row. If `status == SUCCEEDED`, skip and ack. If `IN_FLIGHT` with a fresh heartbeat from another worker, skip. This catches the common case (visibility timeout fired but the original execution succeeded) but doesn't catch the worst case (crash after side-effect, before status update). Documented as such.
3. **`first_attempt` flag.** Job code can branch on whether this is attempt #1 — useful for rendering side-effect-bearing logic conservatively on retries.

**Customer-facing message.** "Your jobs may run more than once. Use `job_id` as an idempotency key. We provide effectively-once execution-state checks but not exactly-once side-effect guarantees." Explicit in docs, in the SDK comments, in onboarding.

---

## 11. State store and the hot path

**[STAFF SIGNAL: state-store-as-hot-path]**

At 600K writes/sec peak, the state store is the bottleneck. Every job touches it 4+ times: enqueue (1), dispatch (1), heartbeat (N for long jobs), ack (1), terminal status (1).

**Substrate.** **[STAFF SIGNAL: rejected alternative]**
- **Postgres single instance.** Caps at ~50K writes/sec. Rejected.
- **Postgres sharded (Citus).** Works, but cross-shard transactions for fair-share state are awkward.
- **Kafka as state store.** No — Kafka is a log; per-job mutable state requires compaction tricks that don't scale to 2B unique keys/day.
- **CockroachDB / FoundationDB / Spanner-class.** Chosen. Sharded by `(tenant_id, job_id)`, with secondary indexes on `(pool, status, dispatched_at)` for scheduler queries. Multi-region async replication; per-region home for low-latency hot writes.

**Sharding.** `tenant_id` as the shard key. Hot tenants get rebalanced across multiple shards via consistent hashing on `(tenant_id, job_id_high_bits)`. Avoids one tenant's traffic concentrating on one shard.

**Write amplification reduction.**
- **Batch heartbeats.** Workers buffer heartbeats for 5s, then flush as a batch write. Reduces 100K heartbeat-writes/sec to ~3K batched writes.
- **Coalesce status transitions.** If a job goes IN_FLIGHT → SUCCEEDED in <100ms (typical short job), one write covers both via an in-memory state machine flush.
- **Async ack journaling.** Acks are written to a per-worker WAL first (local SSD), batched-flushed to the state store. Cost: ack durability lag of ~1s. Acceptable; the worst case is a duplicate execution, which is already covered by at-least-once semantics.

**Read path.** Scheduler reads per-tenant queue depths via cached counts updated incrementally (counter rows per tenant per pool, updated on every enqueue/dispatch). Avoids COUNT(*) over hot rows.

**Failure mode.** State store down → catastrophic: no submissions, no dispatches, no acks. Mitigation: regional active-active with async replication; fallback region accepts submissions but cannot dispatch (queues grow, but durability is preserved). Cross-region failover is operator-initiated, ~5min RTO.

---

## 12. Multi-tenancy and isolation beyond fair-share

**[STAFF SIGNAL: tier-aware multi-tenancy]**

Fair-share covers the dispatch decision. But isolation extends further:

- **Submission rate limits.** Per-tenant max submissions/sec (e.g., 10K/sec for Pro, 100/sec for Standard, 10/sec for Free). Enforced at the API gateway via token bucket, before any state-store write.
- **Concurrent-dispatch caps.** Beyond fair-share's deficit-based smoothing, a hard cap on `(tenant, pool)` concurrent in-flight jobs. Prevents one tenant from monopolizing a pool's concurrent slots even if their deficit is high.
- **Per-job CPU/memory limits.** Workers are cgroup-isolated. Job exceeds limit → killed, classified as `RESOURCE_VIOLATION`, retried once, then DLQ.
- **Per-tenant worker pool size cap.** A tenant whose jobs consistently OOM/crash workers must not be allowed to recycle workers indefinitely. Cap on `(tenant, pool)` worker recycling rate. Past cap, the tenant's pool is throttled and an alert fires.
- **Adversarial detection.** Per-tenant infinite-loop detector: jobs hitting CPU-time wall clock with no progress (no heartbeat advancement, no checkpoint). Killed and DLQ'd.
- **Free-tier sandbox.** Free tier shares a small dedicated slice (e.g., 5% of total capacity) — partitioned physically, so even if 100K free accounts burst simultaneously, paid tiers are unaffected. **[STAFF SIGNAL: blast radius reasoning]**

**Customer-visible errors.** Every limit hit produces a structured error code (`TENANT_RATE_LIMIT_EXCEEDED`, `POOL_CONCURRENCY_CAP`, etc.) and surfaces in the tenant dashboard. Limits are not silent.

---

## 13. Failure modes and graceful degradation

**[STAFF SIGNAL: failure mode precision]**

| Failure | Symptom | Response |
|---|---|---|
| Scheduler down | Dispatch stops, queues grow | Hot standby with leader election; failover ~10s. Workers can fall back to direct queue-pull (degraded fair-share) if scheduler unavailable >60s. |
| State store regional outage | Submissions fail in region | API returns 503; client SDK retries to alternate region; cross-region async replication means most pending jobs are durable elsewhere. |
| Long-pool exhaustion | Long jobs queue >60s | Autoscaler triggers; if at hard cap, alert ops. Spillover to medium pool is *not* allowed (visibility-timeout mismatch). |
| Catastrophic retry storm | Dependency D failing, all D-dependent jobs retry, queue saturates | Per-job-type circuit breaker opens after 50% failure rate over 5min; suspends retries for 5min; emits `dependency_circuit_open`. **[STAFF SIGNAL: blast radius reasoning]** |
| Scheduled-job thundering herd | 10M jobs due at 09:00 | Submission jitter + promoter rate-limit + fair-share dispatch — covered above. |
| Job-type explosion | New job_type submitted at 1B/day suddenly | Per-job-type metrics autoregister; rate-limit per-tenant catches it; alert if a single tenant exceeds 10× their 1h baseline. |
| Worker fleet partial failure | 20% of workers in a pool die | Heartbeats expire; jobs re-dispatched; autoscaler replaces workers. p99 latency spike for ~2min. |
| Promoter lag | Scheduled jobs fire late | `oldest_due_unpromoted_age` alarm; auto-scale promoter shards. Customer sees `late_execution` events. |
| Poison message | One job_type fails 100% | Auto-quarantine after 2 attempts; DLQ; tenant alerted via dashboard. |
| Customer submits 1B jobs in 1s (DoS) | Submission API saturates | Per-tenant rate limit at gateway returns 429; state store and queues are unaffected. |

The unifying principle: every failure has a *bounded blast radius*. One tenant cannot break the cluster; one dependency cannot saturate retries; one job_type cannot consume infinite worker capacity. **[STAFF SIGNAL: invariant-based thinking]**

---

## 14. Autoscaling and observability

**[STAFF SIGNAL: autoscaling-discipline]**

**Per-pool autoscaling.** Each pool scales independently:
- **Signal**: composite of queue depth, oldest-job-age, and worker utilization.
- **Hysteresis**: scale-up threshold (queue age >5s) ≠ scale-down threshold (queue age <1s for 5min). Prevents flap.
- **Cold start**: warm pool of pre-baked workers for short pool (~10% over baseline) absorbs bursts in seconds rather than minutes.
- **Long-pool special case**: scale-up is aggressive, scale-down requires drain (workers idle 30min → drain candidate; only drains workers with no in-flight jobs). Cost overshoot acceptable; mid-job termination is not.

**Observability.** **[STAFF SIGNAL: customer-facing observability]**

Internal dashboards (per pool):
- queue_depth, oldest_job_age (the single best health signal — if this is low, customers are happy)
- dispatch_rate, completion_rate, retry_rate, DLQ_rate
- worker_utilization, worker_recycle_rate
- p50 / p99 submission-to-first-dispatch
- per-tenant deficit distribution (catches scheduler bugs)

Per-tenant dashboards (customer-facing):
- their own submission rate, queue depth, dispatch rate
- per-job-type p99 duration, success rate
- DLQ contents with replay UI
- billing: slot-time consumed per pool

**Composite SLO.** "p99 oldest-job-age <SLO_target per pool" is the headline. If it's healthy, the system is healthy.

---

## 15. Tradeoffs taken and what would change them

- **Three pools, not five.** Bigger surface = better fit per workload but more operational overhead. With more workload diversity (e.g., GPU jobs), I'd add a fourth pool.
- **At-least-once, not exactly-once.** If customers paid significantly more for exactly-once, I'd build a Temporal-style durable-execution layer on top. The queue itself stays at-least-once.
- **Slot-time fair-share, not DRF.** If memory/GPU became a meaningful resource axis, I'd switch to DRF.
- **CockroachDB-class state store.** If the org already runs Spanner / FoundationDB at scale, use that. If we had *more* throughput (>2M writes/sec), I'd push more state into per-shard local stores with async aggregation.
- **Multi-region active-active for submission, regional for dispatch.** Cross-region jobs are rare; if they became common, I'd add a global routing layer.

---

## 16. What I would push back on

**[STAFF SIGNAL: saying no]**

1. **"2B jobs/day"** as a sufficient spec. Without the duration distribution and tenant skew, you can't size anything. The whole architecture changes if 99% of jobs are >1min vs. 99% are <1s. I committed to a distribution; verify before building.

2. **"Fair-share"** without specifying the share unit. If you mean "fair-share by job count," the design is wrong — a tenant running long jobs gets 1000× their share. I asserted slot-time. Confirm.

3. **"Priority"** as a feature in the prompt — I treated it as priority *within* a tenant's queue (e.g., `password_reset` beats `weekly_digest` for the same tenant). Cross-tenant priority is fair-share's job. If you meant cross-tenant priority, that's a different system (and probably a worse one — it incentivizes everyone to mark everything high-priority).

4. **Implicit exactly-once expectation.** I made the at-least-once contract explicit. If a customer pushes back, the right answer is to point them at our Temporal-style durable-execution offering, not to pretend the queue gives exactly-once.

5. **"Some jobs run for hours"** treated as a minor caveat. It's the largest cost driver in the fleet (~60% of worker count) and reshapes liveness, retry, and cancellation. If anyone says "we'll just use a longer visibility timeout," the design fails.

---

*Word count: ~4,450.*