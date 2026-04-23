---
title: Stream Processing & Stateful Computation
description: Stream Processing & Stateful Computation
---


## 1. What a staff engineer actually needs to know

**What matters in interviews**
- Choosing streaming vs batch with a defensible reason.
- Drawing clean boundaries between source, engine, sink.
- Talking about event time, watermarks, late data without hand-waving.
- Reasoning about **state**: partitioning, size, recovery, skew.
- Discussing delivery semantics honestly (not "we'll use exactly-once").
- Naming the failure modes and having a plausible mitigation.

**Expected depth**
- You can sketch a pipeline, place windows/keys/state, identify bottlenecks, and discuss tradeoffs.
- You do **not** need to implement a watermark algorithm, explain RocksDB internals, or recall Flink API specifics.

**Safe to ignore for most roles**
- Specific APIs (Flink DataStream vs Table, KStreams DSL).
- Exact checkpoint barrier alignment algorithm.
- Query planner details in Spark Structured Streaming.
- Vendor feature matrices.

---

## 2. Core mental model

**Stream processing**: process an unbounded input incrementally, emitting results continuously. Input never ends; you commit to a latency budget and tolerate partial information.

**Batch**: process a bounded input all at once, emit when done. You get a consistent snapshot and full information, at the cost of latency equal to batch period.

```
BATCH:                             STREAM:

  ┌──────────────────┐               e e e e e e e e e e e ...
  │ day's data       │───►           │ │ │ │ │ │ │ │ │ │ │
  │ (bounded)        │               ▼ ▼ ▼ ▼ ▼ ▼ ▼ ▼ ▼ ▼ ▼
  └──────┬───────────┘              ─────────────────────────►
         │                           processed incrementally
         ▼   one big job
     [result]                       [r][r][r][r][r][r][r]
     (once/day)                     (continuous output)
```

**Stateful computation**: an operator carries memory across events — counts, running aggregates, join buffers, sessionization, ML features. Output at time *t* depends on history, not just the current event.

**Why state makes things hard**
- State must be **partitioned** (so it scales), **checkpointed** (so it survives failure), **recoverable** (so restarts don't lose results), **bounded** (so it doesn't grow forever), and **resharded** when you rescale.
- State size dominates operational cost, not throughput.
- Every correctness guarantee (exactly-once, event-time windows) ultimately reduces to "did we snapshot state consistently with source offsets."

**When streaming is actually justified**
- Latency SLA measured in seconds (sub-minute to a few minutes).
- Continuous, incremental output is a product requirement (dashboards, alerts, feature stores, fraud).
- Unbounded data with windowed semantics (rolling 1h CTR, session analytics).
- Event-driven architecture where downstream consumers expect a stream.

**When it's overkill**
- Hourly/daily latency acceptable → batch is simpler, cheaper, more debuggable.
- Small volume → a cron job and Postgres beat Flink every time.
- Simple stateless transforms → a Kafka consumer + a function is enough.
- "We might want real-time later" is not a reason.

> Default answer in interviews: **"Start with batch. Move to streaming only when the latency SLA or semantics force it."**

---

## 3. Essential concepts

### Event time vs processing time
- **Event time**: when the event occurred (timestamp in the payload).
- **Processing time**: when the engine sees it (wall clock at the operator).
- Event time gives consistent results under reprocessing, delays, or backfill. Processing time is simple but produces different answers every run.

```
event time  ─►  10:00   10:01   10:02   10:03   10:04
                  e1      e2      e3      e4      e5
                               (e3 delayed by network)
                  │        │        │        │        │
                  ▼        ▼        ▼        ▼        ▼
proc time   ─►  10:00   10:01          10:04   10:05
                                           ▲
                                    e3 arrives late,
                                    AFTER e4 and e5
```
*If you key off processing time here, e3 lands in the wrong window. Event time + watermarks fix this.*

- **What interviewers want**: "Event time for any correctness-sensitive logic; processing time only for SLO-style monitoring."

### Windows

```
TUMBLING (fixed, non-overlapping):
  ├──W1──┤├──W2──┤├──W3──┤├──W4──┤
   e e e   e e e   e e e   e e e

SLIDING (fixed size, slides by < size, overlaps):
  ├────W1────┤
       ├────W2────┤
            ├────W3────┤
   e e  e e  e e  e e  e e

SESSION (dynamic, closed by inactivity gap):
  ├─S1─┤             ├─S2─┤        ├─S3─────┤
   e e e  <- gap ->   e e e  gap    e e e e
```

- State cost: tumbling cheapest; sliding ≈ window_size / slide ratio; session grows with concurrent open sessions.
- Tradeoff: smaller windows = lower latency, less state, more output; larger = smoother, more state, delayed results.

### Watermarks

A watermark *W(t)* is the engine's assertion: "I don't expect events with event-time < *t*." Drives window closing and timer firing.

```
event time axis ─────────────────────────────────────►

events arriving:   e(10:03)  e(10:05)  e(10:02)  e(10:07)  e(10:04)
                                         ▲ out-of-order         ▲ late
watermark:               ──── W(10:00) ─── W(10:03) ─── W(10:05)
                                                           │
                                                watermark lags max
                                                event time by, say,
                                                2 min; past W(10:04)
                                                e(10:04) is "late"
```

- It's a **heuristic**, not a guarantee. Typically `max_observed_event_time − allowed_lag`.
- Tradeoff: aggressive watermark → low latency, more events arrive "late"; conservative → high latency, fewer late events, state held longer.
- **What interviewers want**: you acknowledge watermarks are heuristic, you pick the lag based on observed source distribution, and you have a plan for stragglers.

### Late and out-of-order data

```
window [10:00, 10:05)    allowed lateness: 2 min
┌──────────────────────┐├── +2 min grace ──┤
│   e   e   e   e  e   │  e   e   e       │   e(very late)
│                      │                  │      │
└──────────┬───────────┘                  │      ▼
           │                              │   side-output
      fires at W=10:05                    │   (DLQ / slow
      (initial result)                    │    pipeline)
                                          │
                                 still open: emits
                                 retraction/update
                                 (if sink supports)
```

Options, cheapest to most correct:
1. **Drop** past watermark (simple, loses data).
2. **Side-output** late events to a DLQ or separate slow path.
3. **Allowed lateness**: keep window state open past watermark for N minutes; emit updates.
4. **Retractions / updates**: emit corrections; requires idempotent or upsert-capable sink.

### Stateful operators
Operators that persist data between events: `reduce`, `aggregate`, windowed ops, joins, sessionization, custom `process` functions with keyed state.

### Checkpoints

Periodic, consistent snapshot of **(all operator state) + (source offsets)**. In Flink, barriers are injected into the stream; every operator snapshots its state when the barrier passes through.

```
  source              op A              op B              sink
    │                   │                 │                 │
    │──event──event──B──event──event──► │                 │
    │                   │                 │                 │
    │               barrier B             │                 │
    │               triggers              │                 │
    │               snapshot(A)           │                 │
    │                   │                 │                 │
    │                   │──event──event──B──event──►       │
    │                   │               barrier B           │
    │                   │               triggers            │
    │                   │               snapshot(B)         │
    │                   │                 │                 │
    ▼                   ▼                 ▼                 ▼
  offset Oₙ        state Sₙ(A)       state Sₙ(B)      committed
                                                      output ≤ Oₙ

    checkpoint Cₙ = { Oₙ, Sₙ(A), Sₙ(B), ... }  ──►  durable store (S3/HDFS)
```

- Enables restart to a consistent point. Frequency trades recovery lag against steady-state overhead.
- **What interviewers want**: "I'd start at 30s–1min checkpoint interval and tune."

### Replay

```
                       crash at t=T
                            │
                            ▼
  ──e──e──e──e──Cₙ──e──e──e──e── ✗
                │
                │ restore from Cₙ
                ▼
   source rewinds to offset Oₙ
   operators reload state Sₙ
                │
                ▼
  ──────e──e──e──e──e──────── ► resume (these events get re-processed)
```

- Requires a **replayable source** (Kafka, Kinesis, Pulsar). A REST webhook is not replayable — common interview trap.
- Anything emitted between *Cₙ* and the crash will be re-emitted on replay → sink must be idempotent or transactional.

### Backpressure

Downstream can't keep up → signals propagate upstream → eventually source reads slow.

```
healthy:
  source ──►  op A ──►  op B ──►  sink
          ~ok       ~ok       ~ok

backpressure (sink is slow):
  source ──▶  op A ──▶  op B ──▶  sink
         slow ◀──  slow ◀── slow ◀── (pressure upstream)
           │           │           │
       buffers     buffers     buffers
       growing     growing     FULL
           │
           ▼
    consumer lag grows at source
    checkpoint barriers stall (can't align)
    recovery window widens silently
```

- Healthy: absorbs transient spikes.
- Pathological: stalls checkpoints, grows buffers, lag explodes, can cascade into source retention loss.

---

## 4. Stateful computation (the high-weight section)

### Keyed state

State partitioned by a key (user_id, account_id, device_id). One logical entry per key per operator. Keys hash to tasks.

```
input stream (keyed by user_id):
   (u7,…) (u3,…) (u9,…) (u1,…) (u5,…) (u3,…) …
       │      │      │      │      │      │
       └──────┼──────┴──────┼──────┘      │
              │             │             │
         hash(u3)      hash(u7,u9)   hash(u1,u5)
              │             │             │
              ▼             ▼             ▼
         ┌─────────┐   ┌─────────┐   ┌─────────┐
         │ task 0  │   │ task 1  │   │ task 2  │
         │ state:  │   │ state:  │   │ state:  │
         │  u3→…   │   │  u7→…   │   │  u1→…   │
         │         │   │  u9→…   │   │  u5→…   │
         └─────────┘   └─────────┘   └─────────┘
```

- Scales horizontally: more keys → spread across more tasks.
- **Key choice is the single most consequential decision in a streaming design.**

### Operator state
Shared within an operator instance, not keyed. Typically small: source offsets, connection pools, buffered batches. Rescaling semantics differ from keyed state (list-redistributed or union-redistributed).

### Aggregations over time
Running count/sum/avg/min/max, top-K, percentiles. For large cardinality, use **sketches**:
- HyperLogLog — distinct count, state O(log log n).
- Count-Min — frequency estimation.
- t-digest — quantiles.

Bounded state at the cost of approximation. Mentioning these is a strong signal.

### Joins over streams

**Stream–stream** (both sides unbounded, needs a window):
```
left:  ─ L1 ───── L2 ───────── L3 ───── L4 ───── L5 ──►
                                                        join window: 10 min
right: ───── R1 ───── R2 ────────── R3 ───── R4 ──────►

buffer per key:
  [left side]   L1  L2   L3    L4    L5      ← aged out at >10 min
  [right side]     R1  R2    R3    R4        ← aged out at >10 min

match L3 ↔ R2 if within 10 min, both sides buffered until expiry
```
State = buffered events from both sides within window. Memory-heavy; skew on join key kills you.

**Stream–table** (enrichment):
```
stream of events ─► [ join operator ] ─► enriched events
                          │
                          ▼
                     table snapshot
                   (users, products, …)
                          ▲
                          │
                   CDC / broadcast updates
```
Much cheaper: state = table snapshot per task, not buffered events.

**Temporal join**: join stream to a versioned table *as-of* event time. Requires table history; often the right answer for enrichment with correct point-in-time semantics.

### Timers and triggers
- Timers fire at a future event-time or processing-time. They drive window closure, session timeouts, state TTL, alert emission.
- Triggers decide *when* a window emits: on watermark, on count, on processing time, or a custom mix.

### Why partitioning and key choice matter
- Key determines which task owns the state. Skew → one task is the bottleneck regardless of cluster size.
- Key should be **high cardinality**, **uniformly distributed**, and **aligned with aggregation/join semantics**.
- Classic failure: keying by `country` for a global service → US/IN partitions melt, everything else idle.

### State size and recovery cost
Steady-state throughput is usually not the limit. **Recovery time is.**
- TB-scale keyed state on RocksDB → restore can take tens of minutes.
- Incremental checkpoints help steady-state but restore still loads the working set.
- Mitigations: state TTL, compaction, sharding, smaller checkpoint intervals, hot-standby replicas.

### What makes stateful systems operationally hard
- **Rescaling**: requires repartitioning state (expensive, usually offline).
- **Schema evolution**: state encoded with old schema; migrations are painful.
- **Debugging**: "why is this count wrong" requires reasoning over state + watermark + late data.
- **Checkpoint storage** cost and lifecycle.
- **Hot keys** producing silent tail latency.

---

## 5. Correctness and delivery semantics

### At-least-once vs exactly-once
- **At-least-once**: each event processed ≥1 time. Duplicates possible. Requires idempotent sinks to be correct.
- **Exactly-once (a.k.a. effectively-once)**: each event's *effect* observed once end-to-end. The engine can guarantee this internally via checkpointed state + committed offsets. End-to-end requires sink cooperation.

### Checkpoint + replay mental model
> At checkpoint *Cₙ*: saved state *Sₙ* and source offsets *Oₙ*. On failure, restore *Sₙ*, rewind source to *Oₙ*, resume. Output committed between *Cₙ* and the crash must either be rolled back (transactional sink) or be safe to redo (idempotent sink).

### Correctness boundaries (draw this in the interview)

```
  [source]  ──►  [stream engine]  ──►  [sink]
     │                │                   │
   replayable?   checkpoint+state   idempotent or
   committed     atomic with        transactional
   offsets?      offsets?           commit?

   ─ ─ ─ every link must hold for E2E exactly-once ─ ─ ─
```

The engine is rarely the weak link; **sources and sinks are**.

### Transactional commit (two-phase) — the mechanism behind "exactly-once sinks"

```
   engine                                   sink (e.g. Kafka txn)
     │                                          │
     │  1. begin txn, write events              │
     │─────────────────────────────────────────►│
     │                                          │ staged, uncommitted
     │  2. snapshot state, include txn id       │
     │     in checkpoint Cₙ                     │
     │                                          │
     │  3. checkpoint Cₙ durable → pre-commit   │
     │─────────────────────────────────────────►│
     │                                          │
     │  4. on next "checkpoint complete":       │
     │     commit txn                           │
     │─────────────────────────────────────────►│
     │                                          │ committed, visible

  crash between 2 and 4:  restore Cₙ₋₁, abort txn, reprocess → no duplicates
  crash after 4:          consumer sees committed events exactly once
```

### Dedup / idempotency (practical)
- Assign a stable event ID at the source.
- Sink-side dedup with a bounded window (hash set keyed by event ID, TTL'd).
- Or transactional writes keyed by event ID (Postgres `ON CONFLICT`, Kafka transactions, object-store atomic rename).

### When exactly-once is realistic
- Kafka → Flink → Kafka (Kafka transactions).
- Kafka → Flink → transactional DB with 2PC or idempotent upserts.
- Kafka → Flink → object store with atomic rename/commit.

### When it's mostly marketing
- Sinks that hit external APIs with side effects (payments, emails, push notifications).
- Multi-sink fanout without a coordinator.
- Non-replayable sources.
- "Exactly-once" claims that quietly mean "exactly-once within the engine only."

> Strong interview answer: **"I want at-least-once delivery with idempotent sinks. I reach for exactly-once only when the sink supports transactions and the cost is justified."**

---

## 6. Failure and scaling issues

| Problem | What happens | Mitigation |
|---|---|---|
| Slow consumer | Backpressure stalls upstream | Scale out, decouple with buffer topic, async sink |
| Checkpoint stalls | Barriers can't align under backpressure → no new checkpoint → recovery window grows | Unaligned checkpoints, smaller state, reduce backpressure root cause |
| Large state recovery | Restart takes minutes–hours | State TTL, incremental checkpoints, hot standby, local state on fast disk |
| Skewed / hot keys | One task is the bottleneck; p99 explodes | Salt keys, custom partitioner, pre-aggregate, split hot keys |
| Late data | Windows close before events arrive → wrong results | Larger watermark lag, allowed lateness, side-output |
| Replay storms | Backfill saturates sink or downstream | Throttled replay, shadow pipeline, separate backfill cluster |
| Sink overload | Writes time out, retries amplify load | Batching, rate limiting, circuit breaker, DLQ |
| Stuck watermark | One slow partition holds global watermark back → nothing emits | Per-partition watermarks, idle source detection, timeout advancement |

### Hot-key salting (memorize this pattern)

Before:
```
keyBy(user_id)
   │
   ▼
┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐
│ task 0   │  │ task 1   │  │ task 2   │  │ task 3   │
│ user=X   │  │  idle    │  │  idle    │  │  idle    │
│ 90% load │  │          │  │          │  │          │
└──────────┘  └──────────┘  └──────────┘  └──────────┘
```

After (two-phase aggregation with salt):
```
keyBy((user_id, salt % N))        then  re-keyBy(user_id)
   │                                           │
   ▼  stage 1 (parallelized)                   ▼  stage 2 (combine)
┌──────────┐  ┌──────────┐                ┌──────────┐
│ (X, 0)   │  │ (X, 1)   │ … (X, N-1)     │ final X  │
│ partial  │  │ partial  │                │ sum      │
└────┬─────┘  └────┬─────┘                └──────────┘
     │             │                            ▲
     └─────────────┴─────────┬──────────────────┘
                             │
                  shuffled to one task
                  for the small # of partials
```
Trades a second shuffle for even load. Final-combine sees only *N* partials per hot key, not millions of events.

---

## 7. Interview reasoning patterns

**"Do I need streaming instead of batch?"**
Latency SLA? >1h → batch. Seconds–minutes with continuous output → streaming. If a 5-min micro-batch works, say so; that's often the right answer.

**"Is Flink / Spark Streaming / Kafka Streams justified?"**
- **Flink**: large state, event-time correctness, exactly-once, low latency. Default for serious stateful streaming.
- **Kafka Streams**: library, not a cluster; embedded in your service; great when everything is already Kafka and state is modest. Rescaling is awkward.
- **Spark Structured Streaming**: micro-batch; fine when latency ≥ tens of seconds and your org already runs Spark.
- **None of the above**: a Kafka consumer + Postgres + a cron is often the right answer. Don't reach for a cluster you don't need.

**"When is streaming overkill?"**
Small data, hourly latency, simple transforms, no windowed semantics, no event-time need. Batch wins on every axis.

**"How do I handle late / out-of-order events?"**
Watermark strategy (lag from observed source skew), allowed lateness, side-output past that. Perfect is the enemy of shipped.

**"How do I reason about correctness?"**
Draw source → engine → sink. Name the guarantee at each link. Identify where duplicates can enter. Default to at-least-once + idempotent sinks.

**"How do I choose keys / partitions?"**
High cardinality, uniform distribution, aligned with aggregation or join. Plan for skew before it happens.

**"How do I keep state manageable?"**
TTL, bounded windows, sketches for large-cardinality aggregates, incremental checkpoints, budget state size explicitly.

**"What are the first bottlenecks / failure modes?"**
Hot keys, checkpoint duration under backpressure, sink throughput, recovery time. Name them unprompted.

---

## 8. Common candidate mistakes

- **"Use Flink"** without describing state, keying, or windowing.
- Using **processing time** when event-time correctness matters.
- Treating **late data as nonexistent**. Mobile events arrive hours late.
- Claiming **exactly-once** without naming the sink's commit mechanism.
- Forgetting that **state must be checkpointed and recovered**, and that recovery dominates downtime.
- Ignoring **skew / hot keys** — the first follow-up destroys the design.
- Treating **replay as free** — ignores sink load, downstream duplicates, time.
- Reaching for **streaming where a 15-min batch is simpler, cheaper, more correct**.
- Conflating **micro-batch** with true streaming; different latency/correctness profiles.
- Not distinguishing **keyed state vs operator state** when asked.

---

## 9. Stream processing as a lens on LLM inference serving

This section is for when an interviewer asks about inference serving (vLLM, SGLang, TensorRT-LLM, disaggregated prefill/decode) and you want to speak in system-design terms they already understand. The mapping is tight enough that most stream-processing concepts carry over with renamed parts.

### The core mapping

| Stream processing | LLM inference serving |
|---|---|
| Unbounded event source | Incoming request queue |
| Event | Request (one prefill "event" + many decode "events") |
| Keyed state | Per-sequence KV cache, keyed by `request_id` |
| State backend (RocksDB) | Paged attention block manager / HBM + CPU offload |
| Operator | Scheduler + model engine step |
| Stateless transform | Tokenizer, detokenizer, sampling |
| Stream–table join | Prefix cache / shared-prefix attention |
| Micro-batch window | Continuous batching iteration |
| Watermark | SLA deadline (TTFT / TPOT budget) |
| Backpressure | GPU HBM pressure → admission control, preemption |
| Hot key | Very long sequence dominating one replica |
| Checkpoint | Session persistence (optional, for long chats) |
| Replay | Prefill recompute after KV cache eviction |
| Exactly-once | Idempotent request handling via `request_id` |

### Request flow as a streaming DAG

```
clients ─► [ingress queue] ─► [scheduler] ─► [prefill engine] ─► [decode engine] ─► [detokenizer] ─► stream tokens back
                  │                 │                │                   │
                  │             batching             │          continuous batching
                  │             decisions            │          step (micro-window)
                  │                 │                │                   │
                  │                 ▼                ▼                   ▼
                  │           per-request      KV cache for         KV cache for
                  │           metadata         active prefills      active decodes
                  │           (keyed state)    (keyed state)        (keyed state)
                  │                                       │
                  ▼                                       ▼
            admission control                    eviction / offload
            (backpressure)                       (state management)
```

### Continuous batching = scheduled micro-windowing

```
time ────────────────────────────────────────────────────────────►
step t₀:  [ req A (decode)   req B (decode)   req C (prefill) ]
step t₁:  [ req A (decode)   req B (decode)   req D (decode) ]    ← C finished prefill, B finished, D admitted
step t₂:  [ req A (decode)   req D (decode)   req E (prefill)   req F (prefill chunk) ]
step t₃:  [ req A (decode)   req D (decode)   req E (decode)    req F (prefill chunk) ]
            ▲                                                    ▲
            long-running "key"                                   chunked prefill = chunked micro-batch
            (stays across many steps)
```

Each step is a tiny window: the scheduler packs whatever fits in the compute + memory budget, emits one token per active request, and re-decides every step. This is **stateful streaming with a strict per-step latency budget**, where the budget is the TPOT SLO.

### KV cache = keyed state, with the same failure modes

- **Partitioning**: sequences map to replicas. Sequence-affinity pinning = keyed partitioning. Naive load-balancing destroys cache locality and is wrong for long chats.
- **State size dominates**: a single 128K-context request can hold multiple GB of KV. HBM is the budget; eviction is the TTL story.
- **Hot keys**: very long sequences monopolize a replica's memory and per-step compute — same hot-partition problem, same mitigation (isolate, split, schedule separately).
- **Recovery cost**: on eviction, the only way to rehydrate KV is to re-run prefill — this is **replay** in the streaming sense, and it is not free (prefill is compute-bound, can cost seconds). Prefix caching reduces recompute cost the same way incremental checkpoints reduce restore cost.

### Prefix caching = stream–table join

A shared system prompt or few-shot context is a slowly changing "table" materialized in the KV store. Incoming requests "join" against it: if their prefix hash matches cached block range, attention reads those blocks directly instead of recomputing.

```
request: [ sys_prompt  |  user_query_1 ]
              │                │
              ▼                ▼
       prefix cache hit    new prefill
       (table side)        (stream side)
              │                │
              └────────┬───────┘
                       ▼
               attention over merged KV
```

Same tradeoffs as stream–table joins: staleness (cache invalidation on system prompt change), memory cost of the "table," skew (popular prefixes concentrate).

### Disaggregated prefill/decode = pipeline stages with heterogeneous resource profiles

```
┌─────────────────┐       KV transfer        ┌─────────────────┐
│ prefill cluster │─────(NVLink/RDMA)─────► │ decode cluster  │
│ compute-bound   │                          │ memory-bandwidth│
│ large batch OK  │                          │ many small seqs │
│ TTFT critical   │                          │ TPOT critical   │
└─────────────────┘                          └─────────────────┘
```

Exactly the pattern of a two-stage streaming pipeline where each stage has different scaling characteristics — optimize each independently, manage the handoff explicitly. **The KV transfer is the inter-stage shuffle; bandwidth there is what a network shuffle is in Flink.**

### Speculative decoding = optimistic processing with rollback

Draft model proposes *k* tokens; target model verifies in one forward pass. Accept the longest verified prefix, roll back the rest.

```
draft:   t̂₁   t̂₂   t̂₃   t̂₄   t̂₅      ← proposed speculatively
target:  t₁   t₂   t₃   t̂₄≠t₄         ← verification forks at t₄
accept:  t₁   t₂   t₃                  ← commit
reject:                t̂₄   t̂₅        ← discard, resume from t₃
```

Analogous to optimistic execution in streaming engines: proceed under an assumption, verify, roll back the un-committed portion. Same state-commit mental model.

### Backpressure in inference

GPU HBM is the scarce resource, not CPU or network. When it fills:

```
  [queue growing] ─► admission control rejects (429 / queue depth limit)
                         OR
                     preempt lowest-priority in-flight sequence
                     (swap KV to CPU, or recompute on resume)
                         OR
                     chunked prefill to cap per-step cost
```

Same propagation pattern as a Flink sink slowing down; the mitigation vocabulary is identical: shed load, buffer with bounded depth, decouple with a queue, scale out.

### Things that don't translate cleanly
- **Event time / watermarks**: requests have no semantic event time. TTFT budgets are processing-time SLOs.
- **Out-of-order / late data**: not applicable; requests are independent.
- **End-to-end exactly-once**: almost always reduces to `request_id` idempotency at the client — standard API-level dedup, not a pipeline-level concern.

### Interview-ready framings for inference

- **"Continuous batching is a stream scheduler with a per-iteration micro-window"** — the scheduler's job is the same as Flink's: pack work under a compute and memory budget while respecting per-key (per-sequence) state.
- **"KV cache is keyed state with brutal size budgets"** — eviction is TTL, recomputation is replay, recovery cost is prefill time. Prefix caching is the incremental-checkpoint equivalent.
- **"Disaggregated prefill/decode is pipeline stages with different resource profiles"** — the KV handoff is the shuffle, and its bandwidth is what you optimize first.
- **"Admission control is backpressure"** — GPU memory is the scarce resource that propagates the signal, and the mitigation toolbox is the same: shed, buffer, scale, preempt.
- **"Hot keys show up as long sequences"** — one 128K-context request can starve a replica. Mitigations parallel streaming: isolate on a dedicated replica, split across tensor-parallel ranks, route to a different queue class.

---

## 10. Cheat sheet

### Batch vs stateless streaming vs stateful streaming
| Axis | Batch | Stateless stream | Stateful stream |
|---|---|---|---|
| Latency | hours | seconds | seconds |
| Complexity | low | low–medium | high |
| State story | re-derive each run | none | checkpoint + recover |
| Correctness | easy (snapshot input) | easy | hard (watermarks, replay, semantics) |
| Failure cost | rerun | restart consumer | restore state, replay |
| Best for | analytics, ML training, reports | ETL, routing, enrichment by lookup | windows, sessions, joins, features |
| Default choice | **yes, unless latency forces otherwise** | when transforms are trivial | when semantics require it |

### Processing time vs event time
| Axis | Processing time | Event time |
|---|---|---|
| Definition | wall clock at operator | timestamp in event |
| Simplicity | trivial | requires watermarks |
| Reproducibility | different every run | deterministic |
| Late data handling | impossible | first-class |
| Correct for | monitoring, SLOs | everything else |
| Use when | you truly don't care when events happened | almost always |

### Decision framework
```
1. Latency SLA?
   > 1h           -> batch. Stop.
   minutes–hours  -> micro-batch or periodic job.
   seconds        -> streaming.

2. Stateful?
   No  -> Kafka consumer + function.
   Yes -> Flink (default) / Kafka Streams (Kafka-native, small state).

3. Event-time correctness matters?
   Yes -> watermarks + windows + allowed lateness + side-output.
   No  -> processing time is fine; say so explicitly.

4. Delivery guarantee?
   At-least-once + idempotent sink -> default.
   Exactly-once                     -> only if sink supports transactions.

5. Key / partitioning?
   Highest-cardinality field aligned with aggregation/join.
   Plan salting for skew before it happens.

6. State size?
   Budget it. TTL it. Sketch it where possible. Recovery cost scales with it.

7. Failure modes named?
   Hot keys, checkpoint stalls, sink overload, late data, replay storms.
```

### 10 likely interview questions with strong short answers

1. **"Stream or batch for this?"**
   > "Batch by default. Streaming only if latency SLA is sub-minute, output must be continuous, or we need event-time windowed semantics over unbounded data."

2. **"Event time or processing time?"**
   > "Event time for correctness-sensitive logic; processing time only for monitoring. Event time requires a watermark strategy and a plan for late data."

3. **"How do watermarks work and what's the tradeoff?"**
   > "Heuristic assertion that no earlier events remain. More lag → fewer late events but higher latency. Tune from observed source skew percentiles; add allowed lateness + side-output for stragglers."

4. **"Can you guarantee exactly-once?"**
   > "Inside the engine, yes, via checkpointed state + committed offsets. End-to-end only if the sink supports transactions or is idempotent. Default: at-least-once + idempotent sinks."

5. **"How do you handle late data?"**
   > "Watermark with realistic lag, allowed lateness for the window, side-output to DLQ past that. Emit retractions only if the sink is upsert-capable."

6. **"How do you pick a key?"**
   > "High cardinality, uniform distribution, aligned with aggregation/join semantics. Plan for skew: salting and two-phase aggregation if a small number of keys dominate."

7. **"What happens during a task failure?"**
   > "Restore keyed state from the last checkpoint, rewind source to committed offsets, resume. Downtime is dominated by state restore time, not checkpoint interval."

8. **"What breaks first at scale?"**
   > "Hot keys, checkpoint duration under backpressure, sink throughput. Then recovery time as state grows."

9. **"Stream–stream join at high volume — concerns?"**
   > "Join window drives buffered state on both sides; skew on the join key amplifies it; late data on one side breaks matches. Consider stream–table or temporal join if one side is slowly changing."

10. **"When is Flink the wrong tool?"**
    > "Small scale, hourly latency, stateless transforms, no event-time needs, or sinks that don't support replay/transactions. A Kafka consumer + Postgres + a cron beats Flink for 80% of 'real-time' requests."

---

**Final rule of thumb**: talk like someone who has been paged at 3am by a stream pipeline. Name failure modes before you're asked. Be honest about what exactly-once costs. Default to batch unless the problem forces streaming. And if the interview pivots to inference serving, use the same vocabulary — it's the same problem with renamed parts.