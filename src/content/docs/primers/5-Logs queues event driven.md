---
title: Logs, queues, and event-driven systems
description: Logs, queues, and event-driven systems
---


## 1. What a staff engineer actually needs to know

**What matters in interviews:**
- Clear mental model of queue vs log vs pub/sub — and *why you'd pick each*.
- Delivery semantics stated precisely, without hand-waving.
- Failure modes: poison messages, backlog, hot partitions, retry storms, duplicate delivery.
- Tradeoffs between sync APIs and event-driven decoupling.
- Ordering, partitioning, idempotency — these come up in almost every design.

**Expected depth:**
- Explain how a consumer group rebalances and why.
- Reason about what "exactly-once" actually costs and where it's real.
- Justify partition-key choice against skew and ordering needs.
- Design a retry + DLQ policy end-to-end.

**What you can ignore for most interviews:**
- Kafka/RabbitMQ/SQS internals (ISR details, Raft, segment files, Erlang mailboxes).
- Exact config flags.
- Protocol-level framing.
- Storage engines underneath the broker.

Only go deep if the role is explicitly "distributed data infra." Otherwise, design-level reasoning wins.

---

## 2. Core mental model

### Log
An **append-only, durable, ordered** sequence of records. Consumers read by **offset**. Records are **retained independent of consumption** (time- or size-bounded). Multiple independent consumers can read the same log at their own pace.

```
producers ──► [ r0 r1 r2 r3 r4 r5 r6 ... ]  ◄── consumer A (offset=3)
                                             ◄── consumer B (offset=6)
                                             ◄── consumer C (offset=0, replaying)
```

### Queue
A **point-to-point** buffer. Producers enqueue; **one consumer per message** dequeues. Typically messages are **deleted after ack**. No replay. Designed for **work distribution**.

```
producers ──► [ m1 m2 m3 m4 ] ──► worker1 (took m1)
                                  worker2 (took m2)
                                  worker3 (took m3)
```

### Pub/Sub
A **broadcast primitive**. One publish → N subscribers each get a copy. The *substrate* can be ephemeral (e.g., Redis pub/sub, no durability) or durable (a log with consumer groups is how Kafka does pub/sub).

```
publisher ──► topic ──► subscriber A (gets copy)
                    ──► subscriber B (gets copy)
                    ──► subscriber C (gets copy)
```

### Event-driven architecture
A system design style where **services communicate by emitting events** rather than calling each other synchronously. Each service reacts to events it cares about. Coupling moves from service-to-service to service-to-topic.

### Axes that distinguish these

| Axis              | Queue         | Pub/Sub (ephemeral) | Durable log             |
|-------------------|---------------|---------------------|-------------------------|
| Retention         | until acked   | until delivered     | time/size bounded       |
| Replay            | no            | no                  | yes                     |
| Ordering          | FIFO-ish      | none/weak           | per-partition strict    |
| Fanout            | no (1 reader) | yes                 | yes (via groups)        |
| Consumer model    | compete       | each gets all       | compete within a group  |

### Why "just use Kafka" is a shallow answer
- Kafka is a **log**, not a task queue. Using it for small-volume RPC-like work is overkill and has poor per-message latency at low throughput.
- Replay is great — until you accidentally replay a billion events in prod.
- Kafka does not give you per-message ack/retry semantics; you commit offsets. Mixing slow and fast messages in one partition causes head-of-line blocking.
- Partitioning decisions are sticky and hard to change.

### Why "just use a queue" is a shallow answer
- No replay. If a downstream consumer has a bug and drops messages, they're gone.
- Hard to fanout to multiple independent consumers without architectural gymnastics.
- Usually weaker ordering than you think.

---

## 3. Essential models

### 3a. Point-to-point work queue (SQS, RabbitMQ work queue, Redis list)

```
             ┌──────────────────┐
producers ──►│ [m1 m2 m3 m4 m5] │──► workers compete for messages
             └──────────────────┘           │
                                            ▼
                                      ack → delete
                                      no-ack → redelivered after visibility timeout
```

- **Producer path:** enqueue, get durability ack.
- **Consumer path:** receive → process → ack. Until ack, message is "invisible" to other consumers (visibility timeout).
- **Delivery:** at-least-once by default. Duplicates possible on timeout or crash.
- **Strengths:** load balancing across workers, simple retry via visibility timeout, horizontal scaling by adding workers.
- **Weaknesses:** no replay, weak ordering (FIFO queues exist but constrain throughput), no fanout to independent subscribers.
- **Use when:** tasks are idempotent, you need work distribution, you don't need to rewind.
- **Interviewer wants to hear:** visibility timeout, DLQ after N retries, idempotency keys, when ordering breaks.

### 3b. Pub/Sub messaging (Redis pub/sub, SNS, Google Pub/Sub)

```
publisher ──► topic ──► subscriber A
                    ──► subscriber B
                    ──► subscriber C
```

- **Producer path:** publish to topic; broker delivers to all current subscribers.
- **Consumer path:** subscribe; receive pushed messages.
- **Delivery:** varies. Ephemeral pub/sub (Redis) = at-most-once, no durability. Durable pub/sub (SNS→SQS, Google Pub/Sub) = at-least-once with ack.
- **Strengths:** simple fanout, decoupled subscribers.
- **Weaknesses:** ephemeral variants lose messages on subscriber downtime; no replay unless backed by a log.
- **Use when:** multiple independent consumers need the same event, and you don't need replay beyond what the broker retains.
- **Interviewer wants to hear:** the distinction between push vs pull, what happens if a subscriber is down, whether the system is durable.

### 3c. Durable append-only log (Kafka, Pulsar, Kinesis)

```
                    partition 0: [r0 r1 r2 r3 r4 r5 r6 ...]
producers ──► topic  partition 1: [r0 r1 r2 r3 r4 r5 ...]
                    partition 2: [r0 r1 r2 r3 ...]

consumer group G1: {c1 reads p0, c2 reads p1+p2}  ← offsets tracked per (group, partition)
consumer group G2: {c1 reads p0+p1+p2}            ← independent offsets, full fanout via groups
```

- **Producer path:** choose partition (usually via key hash), append; broker persists to disk, replicated.
- **Consumer path:** consumer joins a group; group members split partitions among themselves; each member reads and commits offsets.
- **Delivery:** at-least-once by default. Exactly-once is possible within Kafka (transactions + idempotent producer) but **does not extend to arbitrary side effects**.
- **Strengths:** replay, multiple independent consumer groups, high throughput, per-partition ordering, long retention enables rebuilds and backfills.
- **Weaknesses:** no per-message ack; head-of-line blocking within a partition; partition count is hard to change after the fact; operational cost higher than a managed queue.
- **Use when:** replay matters, multiple services consume the same event stream, you need high throughput, or you want a "source of truth" event backbone.
- **Interviewer wants to hear:** partition key selection, consumer groups, offset commit strategy, retention policy, what happens on consumer lag.

---

## 4. Must-know concepts

**Ordering and partitioning.**
Global total order is expensive and usually not what you need. Logs give **per-partition order**, which means: pick your partition key so that messages that must be ordered relative to each other land on the same partition. Common key: `user_id`, `order_id`, `tenant_id`. Tradeoff: more grouping = more risk of hot partitions.

**Offsets / acknowledgments.**
- Queues: per-message ack. Ack = delete.
- Logs: offset commits. Committing offset N means "I've processed everything up to N." Commit *after* successful processing, not before, unless you want at-most-once.

**Retention.**
- Queues: retention until ack (plus a bounded max age, e.g., 14 days on SQS).
- Logs: time- or size-based. E.g., 7 days, or 1 TB per partition. Independent of consumption.

**Replay.**
Only logs support replay cleanly. Reset consumer offset → reprocess from any point. Required for: rebuilding a derived view, recovering from a bug, onboarding a new consumer.

**Consumer groups.**
Kafka's mechanism for load-balancing a log across N consumers *while still allowing fanout across groups*. Within a group, each partition is assigned to exactly one consumer. Adding a consumer triggers a **rebalance**; all consumers briefly pause. Too many rebalances = instability.

**Backpressure.**
The system's ability to slow producers when consumers can't keep up. Queues naturally provide it (enqueue blocks or errors when full). Logs push it to consumers via lag metrics — you must explicitly monitor and react. In event-driven systems, backpressure across service boundaries is an active design concern.

**Retries.**
- Immediate retry: fine for transient network blips, dangerous for overload (retry storms).
- Exponential backoff with jitter: the correct default.
- Retry budget: cap retries in time or count, then DLQ.
- Never retry forever on a poison message.

**Dead-letter queue (DLQ).**
A side channel for messages that have failed N times. Purpose: unblock the main pipeline, preserve the failing message for investigation. Must have: alerting on DLQ depth, a replay path, a way to inspect messages.

**Idempotency.**
The property that processing the same message twice produces the same result as processing it once. **Required** under at-least-once delivery, which is the practical default. Implementations:
- Natural idempotency (set x = 5, not x += 1).
- Idempotency key + dedup table (`INSERT ... ON CONFLICT DO NOTHING`).
- Conditional writes (`if version = N, set version = N+1`).

**Deduplication (interview depth).**
Dedup window in the broker (SQS FIFO has 5-minute dedup) or app-level (hash → seen table). Broker-level dedup is bounded and doesn't replace idempotent consumers.

**Delivery semantics (practical depth):**
- **At-most-once:** fire-and-forget. Lose messages on failure. Use when loss is acceptable (telemetry, metrics samples).
- **At-least-once:** default for durable systems. Duplicates possible. **Consumers must be idempotent.**
- **Exactly-once:** a property of an end-to-end system, not a transport guarantee. Achievable when (1) the broker supports transactional writes (Kafka EOS) *and* (2) the consumer's side effect is either transactional with offset commit or idempotent. If the side effect is "send email" or "call external API," exactly-once is a lie — aim for at-least-once + idempotency.

**Visibility timeout (queues).**
When a consumer receives a message, the broker hides it for T seconds. If the consumer acks within T, it's deleted. If not, it reappears. Set T larger than your p99 processing time; too small → duplicate work, too large → slow recovery from crashes.

**Fanout vs work distribution.**
- **Fanout:** every subscriber/group gets every message. Use for "notify N systems of this event."
- **Work distribution:** each message goes to exactly one worker. Use for "process this task."
  Logs + consumer groups give you both: within a group = work distribution; across groups = fanout.

---

## 5. Logs vs queues

This is the single most interview-relevant decision.

**When a queue is a better fit:**
- Task distribution to a pool of workers.
- Per-message retry and DLQ semantics matter.
- You don't need multiple independent consumers of the same stream.
- Low-to-moderate throughput with variable processing time per message.
- You don't need replay.

**When a durable log is a better fit:**
- Multiple independent services consume the same event stream.
- You need replay — for backfills, rebuilding state, new consumers joining after the fact, recovering from a bug.
- High throughput with reasonably uniform per-message cost.
- The stream is the source of truth for downstream systems (event sourcing, CDC).
- You care about ordering within a key.

**Why ordering guarantees are usually limited:**
- Queues: FIFO-strict variants exist but cut throughput significantly and still don't give you cross-partition order.
- Logs: only per-partition. Global order needs a single partition = single-writer-bottleneck.
  Staff-level move: don't promise global order. Ask what order actually needs to be preserved (usually per-entity), pick a partition key that guarantees it.

**How partitioning changes semantics:**
- Adds parallelism.
- Breaks global order.
- Introduces risk of hot partitions (one key dominates traffic).
- Makes rebalancing non-trivial when partition count changes.

**What a staff-level candidate should say when choosing:**
> "If multiple downstream services need this data, or we need the ability to replay, I'd lean toward a durable log like Kafka, partitioned by `<key>` so ordering per `<entity>` is preserved. If this is a one-producer one-consumer-pool task distribution problem with bounded retries and DLQ, a managed queue like SQS is simpler and cheaper. The decision comes down to: do I need replay, do I need fanout, and do I need per-message ack."

---

## 6. Event-driven systems

### What you get
- **Async decoupling:** producers don't know who consumes. New consumers plug in without touching producers.
- **Producer/consumer independence:** services deploy, scale, and fail independently.
- **Workflow chaining:** service A emits `OrderPlaced` → service B reacts, emits `PaymentAuthorized` → service C reacts, emits `InventoryReserved`.

### What you pay
- **Eventual consistency.** State is correct *eventually*, not immediately. User reads may see stale data.
- **Debuggability cost.** "Why didn't X happen?" becomes a trace across topics, consumers, and retries.
- **Correctness reasoning is harder.** Invariants that span services can't be enforced with a transaction.

### Integration events vs command messages
- **Command:** "do this" — imperative, addressed to one consumer (e.g., `ChargeCard`). Usually a queue.
- **Integration event:** "this happened" — fact, past tense, broadcast (e.g., `OrderPlaced`). Usually a log or pub/sub.
  Mixing them causes confusion. Staff-level clarity: events describe facts; commands request actions. They go on different channels.

### When event-driven helps
- Multiple consumers for the same fact.
- Decoupled deploys across teams.
- Long-running workflows with natural stages.
- Audit trail / event sourcing requirements.
- High fanout with heterogeneous consumers.

### When event-driven hurts
- Simple request/response flows. Adding a broker adds latency, ops, and failure modes.
- Transactional correctness across steps (e.g., "authorize and capture must both succeed or both fail"). Async = sagas = compensating actions = complexity.
- Tight latency budgets (< 10 ms end-to-end).
- Small teams/small systems where the coupling cost isn't yet painful.

### When synchronous APIs are still better
- User-facing read paths needing immediate consistency.
- Flows where the caller genuinely needs to wait for the outcome.
- Simple CRUD with no fanout.
  Don't eventify everything. The failure mode is a system you can't reason about locally.

---

## 7. Common failure and scaling issues

**Poison messages.**
A message that always fails (malformed, hits a bug). Without a retry cap, it blocks the partition/queue forever.
*Mitigation:* retry budget → DLQ → alert → investigate.

**Stuck consumers.**
Consumer receives, doesn't ack, doesn't crash. Visibility timeout keeps hiding the message or the partition offset doesn't advance.
*Mitigation:* processing timeout < visibility timeout; health checks; lag alerts per partition.

**Slow consumers.**
Consumer keeps up on average but p99 processing is high. Backlog grows during bursts.
*Mitigation:* autoscale on queue depth / lag; shed load; increase parallelism within the partition if safe; move expensive work off the hot path.

**Backlog growth.**
Sustained producer rate > consumer rate. Latency grows unboundedly.
*Mitigation:* alert on lag, autoscale consumers, shed low-priority producers, increase partitions (for logs, planned ahead).

**Replay storms.**
Someone resets a consumer group offset to 0 on a month-old topic. Downstream systems get hammered.
*Mitigation:* rate-limit replays, separate replay consumer group, document replay runbooks, protect downstream with idempotency + throttles.

**Duplicate delivery.**
At-least-once means duplicates. If the consumer isn't idempotent, you get double-charges, double-emails, duplicate rows.
*Mitigation:* idempotency keys end-to-end.

**Out-of-order processing.**
Across partitions, messages for the same entity interleave. Stale event overwrites fresh state.
*Mitigation:* partition by entity key; version/timestamp each event; reject writes where `incoming.version <= current.version`.

**Hot partitions.**
One key (big tenant, celebrity user) dominates. That partition lags while others idle.
*Mitigation:* composite keys (`tenant_id:hash(sub_id)`); detect and split; rate-limit the hot producer; overprovision partitions to spread.

**Retry storms.**
Downstream is degraded. Consumers retry aggressively, amplifying load, preventing recovery.
*Mitigation:* exponential backoff with jitter, retry budgets, circuit breakers, load shedding.

**Downstream failure propagation.**
Consumer depends on service X; X slows; consumer slows; backlog builds; producer side sees lag.
*Mitigation:* bulkhead (isolate by consumer group), timeouts + circuit breakers, DLQ for unrecoverable errors, graceful degradation (skip non-critical side effects).

---

## 8. Interview reasoning patterns

**When should I use a queue vs a log?**
> Queue if it's task distribution, per-message retry, no replay needed. Log if multiple consumers need the same stream, or replay/backfill is a requirement, or the stream is a source of truth.

**When is Kafka the wrong answer?**
> Low volume with no fanout. Pure request/response. When ops cost and operational skill aren't there. When per-message retry/DLQ semantics matter more than replay. When head-of-line blocking within a partition would be a problem.

**When is a queue the wrong answer?**
> When multiple independent services need the same event. When replay matters. When you need to preserve a stream as a source of truth. When throughput demands exceed what per-message-ack systems handle comfortably.

**How do I handle retries safely?**
> Exponential backoff with jitter, bounded retry count, DLQ on exhaustion, circuit breaker on the downstream, idempotent consumer so retries are safe.

**How do I reason about idempotency?**
> Assume at-least-once. Every consumer needs either a naturally idempotent operation, an idempotency key with a dedup store, or conditional/versioned writes. Idempotency is a per-operation design concern, not a transport setting.

**How do I handle ordering requirements?**
> Ask: ordering of what, relative to what? Usually per-entity, not global. Partition by that entity's key. If global order is truly required, you have a single-writer bottleneck and need to justify it.

**How do I handle backpressure?**
> Queues: bounded queue + producer sees errors. Logs: monitor consumer lag, autoscale consumers, shed load at the producer if lag exceeds threshold. Across services: circuit breakers, timeouts, rate limits at the boundary.

**When does event-driven help vs hurt?**
> Helps when you have multiple independent consumers, decoupled deploys, long workflows, or audit/replay needs. Hurts when you need strong cross-service consistency, when latency budgets are tight, or when the team is small and the coupling cost hasn't materialized yet.

---

## 9. Common candidate mistakes

- Saying "use Kafka" with no justification. Always pair the choice with the property that drove it (replay, fanout, throughput, source of truth).
- Confusing a queue with a pub/sub log. Queue = compete for messages. Log = read at your offset. Pub/sub = broadcast to subscribers.
- Hand-waving delivery guarantees. "We'll use exactly-once" without saying what the side effect is or whether the broker's EOS extends to it.
- Assuming exactly-once is free. It isn't. It's a property of the *end-to-end system*, and it costs throughput, complexity, or both.
- Forgetting idempotency. At-least-once + non-idempotent consumer = incidents.
- Ignoring ordering limits. Claiming "ordered" without specifying the partition key or acknowledging cross-partition interleaving.
- Ignoring consumer lag and backlog growth. "We'll scale up" isn't an answer — name the metric, name the threshold, name the action.
- Using async for flows that need strong synchronous correctness. A user clicking "pay" and getting "we'll let you know" is almost always wrong.
- Ignoring dead-letter handling. DLQ is mandatory, not optional. Also name: alerting on depth, replay path, investigation workflow.
- Forgetting partition key selection. This is one of the most common staff-level probes.

---

## 10. Final cheat sheet

### Queue vs Pub/Sub vs Durable Log

| Property              | Queue (SQS, Rabbit)      | Pub/Sub (SNS, Redis)        | Durable Log (Kafka, Kinesis)            |
|-----------------------|--------------------------|-----------------------------|------------------------------------------|
| Primary use           | Work distribution        | Fanout                      | Event stream / source of truth           |
| Retention             | Until ack (max ~days)    | Until delivered (or lost)   | Time/size bounded (days–months)          |
| Replay                | No                       | No (or limited)             | Yes                                      |
| Ordering              | FIFO variant only        | None to weak                | Per-partition                            |
| Fanout                | No                       | Yes                         | Yes via consumer groups                  |
| Ack model             | Per message              | Per subscription            | Offset commit (bulk, per partition)      |
| Backpressure          | Natural (bounded)        | Variable                    | Manual (lag-based)                       |
| Typical delivery      | At-least-once            | At-most- or at-least-once   | At-least-once (EOS for in-Kafka writes)  |
| Scaling unit          | Workers                  | Subscribers                 | Partitions + consumer group size         |
| Operational cost      | Low (managed)            | Low (managed)               | High (self-hosted); moderate (managed)   |

### Delivery semantics

| Semantic        | What it means                                | Cost                                      | When to pick                           |
|-----------------|-----------------------------------------------|-------------------------------------------|-----------------------------------------|
| At-most-once    | May lose, never duplicate                     | Cheapest, simplest                        | Telemetry, sampled metrics              |
| At-least-once   | Never lose, may duplicate                     | Requires idempotent consumer              | Default for durable systems             |
| Exactly-once    | Never lose, never duplicate (end-to-end)      | Requires transactional broker + transactional/idempotent sink; limited to systems the transaction covers | In-broker pipelines (Kafka→Kafka), financial flows where idempotency keys are tracked |

### Decision framework

```
                      Does the consumer need replay or
                      do multiple services consume this stream?
                                 │
                       ┌─────────┴─────────┐
                       │                   │
                     YES                  NO
                       │                   │
                Durable log         Is it fanout to
               (Kafka, etc.)      many subscribers?
                                         │
                                ┌────────┴────────┐
                                │                 │
                              YES                NO
                                │                 │
                        Pub/Sub            Work distribution
                       (SNS, etc.)         to worker pool?
                                                  │
                                        ┌─────────┴─────────┐
                                        │                   │
                                      YES                  NO
                                        │                   │
                                     Queue           Sync API / RPC
                                   (SQS, Rabbit)
```

Then, regardless of choice, answer:
1. **Partition / routing key:** by what property?
2. **Delivery semantics:** at-least-once; how is the consumer idempotent?
3. **Retries:** backoff policy, retry budget, DLQ threshold.
4. **Ordering:** what ordering is required, and how does partitioning preserve it?
5. **Backpressure:** what do producers do when consumers can't keep up?
6. **Failure modes:** poison message, hot partition, backlog — what's the runbook?

### 10 likely interview questions with short strong answers

**1. "Queue or Kafka for this async job system?"**
Queue. It's task distribution with retries and DLQ. No fanout, no replay. Kafka is overkill, and head-of-line blocking on a slow job hurts more than it helps. I'd use SQS with DLQ after N retries.

**2. "How do you guarantee exactly-once processing?"**
You don't, not as a transport property. You design for at-least-once + idempotent consumer: idempotency key per message, dedup store with TTL matching retention, or conditional writes keyed on message ID. True end-to-end exactly-once only works when the sink participates in the same transaction as the offset commit.

**3. "How do you handle ordering?"**
Clarify: ordering of what, relative to what? Usually per-entity. Partition by that entity's key so all its events land on the same partition, preserving order there. Global ordering is rarely required and expensive.

**4. "How do you handle a poison message?"**
Bounded retries with exponential backoff and jitter. After the budget is exhausted, route to DLQ. Alert on DLQ depth. Keep a replay tool to reprocess after the bug is fixed.

**5. "Consumers are falling behind. What do you do?"**
Diagnose: is lag per-partition uniform (throughput issue) or skewed (hot partition)? If uniform: scale consumers, check downstream for slowness, shed low-priority work. If skewed: rebalance partition keys, split hot tenants, add backpressure upstream.

**6. "How do you design retries?"**
Exponential backoff with jitter, bounded count or time budget, DLQ on exhaustion. Pair with a circuit breaker on the downstream to avoid retry storms during incidents. Idempotent consumer so retries are always safe.

**7. "When is event-driven a bad fit?"**
When you need synchronous correctness (e.g., "did the payment go through"), when latency budgets are tight, when cross-service invariants need transactional enforcement, or when the team is small enough that the coupling hasn't yet become painful. Async adds real complexity; it should earn its place.

**8. "How do you pick a partition key?"**
Must preserve the ordering that matters (usually per-entity). Must distribute load evenly (avoid hot keys). Must be stable (don't pick something that churns). Common choices: `user_id`, `order_id`, `tenant_id`. If one tenant is huge, use a composite like `tenant_id:hash(sub_id)` and reconstruct order at the consumer via a sequence number.

**9. "How do you bound memory and avoid OOM in the consumer?"**
Cap in-flight messages (prefetch limit). Commit offsets after successful processing. Don't batch more than you can process within the visibility timeout / session timeout. Lag-based autoscaling rather than in-process unbounded buffering.

**10. "Describe a pipeline where CDC feeds a search index."**
Source DB → Debezium → Kafka topic partitioned by primary key → search-indexer consumer group → idempotent upsert into the search index keyed by primary key, with version checks so stale events don't overwrite fresher state. Retention ~7 days on the topic to allow index rebuilds. New consumer groups can replay to bootstrap new derived views. DLQ for schema-incompatible events. Monitor consumer lag per partition; alert on hot partitions from large tenants.

---

**Closing heuristics:**
- Name the property (replay, fanout, ordering, durability) before naming the tech.
- Always state delivery semantics and how the consumer is idempotent.
- Always specify the partition key and why.
- Always design the retry + DLQ path.
- Always define "what does backpressure do here."
- Events describe facts; commands request actions — keep them on different channels.