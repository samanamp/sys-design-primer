---
title: RPC, Service Communication, and Resilience
description: RPC, Service Communication, and Resilience
---


## 1. What a Staff Engineer Actually Needs to Know

**What matters in staff interviews:**
- Choosing the right communication pattern (sync RPC vs async vs streaming) and defending it
- Reasoning about latency budgets, p99, and fanout amplification
- Anticipating failure modes *before* the interviewer asks ("what if this is slow / down / partitioned?")
- Designing systems that **degrade** rather than collapse
- Trading off consistency, availability, and latency explicitly

**Expected depth:**
- You can discuss timeouts, retries, idempotency, circuit breakers, load shedding, and backpressure without being led
- You know *why* each exists (what failure mode it addresses) — not just *what* it is
- You can draw a call graph, mark where failures happen, and walk through mitigation
- You can pick REST vs gRPC vs async queue and justify in 2 sentences

**What can be ignored:**
- Protocol wire formats (HTTP/2 framing, protobuf encoding internals)
- Service mesh internals (Envoy xDS, iptables tricks)
- Load balancer algorithm math (you just need to know the names and tradeoffs)
- Building a consensus protocol from scratch (unless you're interviewing for a database team)

**The mental shift:** Staff-level is not "I know more APIs." It's "I predict what breaks, at what scale, and how to contain the blast radius."

---

## 2. Core Mental Model

### What RPC Is

**RPC = Remote Procedure Call.** You make a function call that *looks* local but crosses a network boundary:

```
  Client                                Server
  ------                                ------
  result = svc.getUser(id)  ────────►   def getUser(id): ...
                            ◄────────   return user
```

The abstraction is convenient and **dangerous**. Every local function call is ~nanoseconds, deterministic, and cannot fail independently of the caller. Every remote call can be slow, drop, partially succeed, or return stale data.

### Sync vs Async Service Communication

```
  SYNC (RPC)                          ASYNC (queue / event)
  ----------                          -----------------------
  A ──req──► B                        A ──event──► [QUEUE] ──► B
  A ◄──resp── B                         (A does not wait)
   (A blocks)                            (B consumes when ready)
```

- **Sync**: caller waits for response. Tight coupling in time. Low latency on success, but caller's latency = callee's latency + network + queueing.
- **Async**: caller hands off work. Decoupled in time. Caller finishes fast; completion happens later, possibly much later.

### Request/Response vs Streaming (high level)

```
  UNARY (req/resp)              SERVER STREAM           CLIENT STREAM           BIDI STREAM
  ----------------              -------------           -------------           -----------
  C ──req──► S                  C ──req──► S            C ──msg──► S            C ◄──msg──► S
  C ◄──resp── S                 C ◄──msg── S            C ──msg──► S            C ◄──msg──► S
                                C ◄──msg── S            C ──msg──► S            (full duplex)
                                C ◄──msg── S            C ◄──resp── S
                                (one req, many resps)
```

Streaming matters when: large payloads, long-lived server-push, progressive results (LLM token streaming), or bidirectional protocols (chat, trading).

### Why Network Calls Are Fundamentally Different from In-Process Calls

| Property | Local call | Network call |
|---|---|---|
| Latency | ~ns | ~ms to ~seconds |
| Failure mode | Crashes together | Independent partial failure |
| Partial success | Impossible | Normal |
| Bandwidth | Unlimited | Finite, shared |
| Ordering | Program order | Not guaranteed across connections |
| Observability | Stack trace | Requires distributed tracing |
| Security | Trust boundary | Must authn/authz every call |

The eight fallacies of distributed computing are the TL;DR: the network is not reliable, latency is not zero, bandwidth is not infinite, the network is not secure, topology changes, there is more than one admin, transport cost is not zero, the network is not homogeneous.

### Why Service Communication Is About Latency, Failure, and Load — Not API Shape

Interview candidates over-focus on "REST vs gRPC, JSON vs protobuf." That's the easy 10%. The hard 90% is:

- **Latency**: what's the p99 under load, not the happy-path latency
- **Failure**: what happens when the callee is slow, down, or lying
- **Load**: what happens when request rate doubles — does the system degrade gracefully or collapse?

Pick the API shape in 30 seconds; spend the rest of the design discussing these three.

---

## 3. Essential Communication Patterns

### 3.1 Synchronous RPC

**How it works:** caller sends request, blocks (or awaits), callee responds, caller continues.

```
  C ──req──► S   (C is blocked / awaiting)
  C ◄──resp── S
```

**When it fits:**
- Low-latency reads where the caller cannot make progress without the answer
- Request-scoped operations (authn, authz, config lookup)
- Small, fast, highly-available dependencies

**Strengths:** simple mental model, strong consistency, easy tracing, easy error propagation.

**Weaknesses:** tight temporal coupling. Caller's availability ≤ product of all sync dependencies' availabilities.

**Failure behavior:** caller sees timeouts, connection errors, or 5xx. Must decide: retry, fallback, or propagate.

**What interviewers want to hear:** "Sync RPC couples availability. If I call three 99.9% services synchronously, my ceiling is ~99.7%. I'll use it only where the caller truly needs the answer inline."

### 3.2 Async Messaging (queues / event streams)

**How it works:** producer writes message to broker; consumer pulls (or is pushed) and processes. Decoupled in time.

```
  Producer ──msg──► [BROKER] ──msg──► Consumer
                    (Kafka,
                     SQS,
                     RabbitMQ)
```

**When it fits:**
- Long-running work (video transcode, batch jobs, ML training kicks)
- Fanout to many consumers (event-driven architecture)
- Smoothing load spikes (broker acts as buffer)
- Producer doesn't need the result immediately (or ever)

**Strengths:** temporal decoupling, natural backpressure (queue depth), retry is cheap, easy to add consumers.

**Weaknesses:** eventual consistency, harder tracing, duplicate delivery is the default (at-least-once), ordering is often per-partition only.

**Failure behavior:** producer succeeds if broker accepts. Consumer failures retry via redelivery. Poison messages need dead-letter queues (DLQ).

**What interviewers want to hear:** "I'll use async for anything the user doesn't have to wait on, and whenever the downstream might be slow or flaky. The broker absorbs load and decouples failure domains."

### 3.3 One-Way Fire-and-Forget

**How it works:** caller sends, doesn't wait for ack, doesn't care about the result.

```
  C ──msg──► S   (C moves on immediately, no resp)
```

**When it fits:** metrics, logs, telemetry, hints, cache invalidations where loss is tolerable.

**Strengths:** lowest caller latency, zero coupling.

**Weaknesses:** **no delivery guarantee**. If you need the message to arrive, this is the wrong pattern — use async messaging with a durable broker instead.

**Interview trap:** candidates sometimes say "fire and forget" when they mean "async messaging." Those are different. Fire-and-forget = UDP vibes, loss is acceptable.

### 3.4 Streaming RPC

**How it works:** one or both sides send a sequence of messages over a persistent connection (HTTP/2 or HTTP/3).

```
  LLM TOKEN STREAMING EXAMPLE
  ---------------------------
  C ──"prompt"──► S
  C ◄── tok1 ───  S
  C ◄── tok2 ───  S
  C ◄── tok3 ───  S
  C ◄── EOS ────  S
```

**When it fits:**
- Incremental results (LLM inference, search-as-you-type, live video)
- Large payloads that don't fit in memory
- Long-lived subscriptions (price feeds, notifications)
- Bidirectional protocols (chat, collaborative editing)

**Strengths:** low latency to first byte (TTFB), progressive delivery, one connection amortized across many messages.

**Weaknesses:** harder to load-balance (sticky connections), harder to retry mid-stream, complicates timeouts, head-of-line blocking risk per connection.

**What interviewers want to hear:** "For LLM inference, streaming is table stakes because TTFT matters more than TTLT. For a request where the user can't use partial output, unary is simpler and I default to that."

---

## 4. Must-Know Concepts

### Timeout / Deadline

- **Timeout**: "fail this call if it takes longer than X ms."
- **Deadline**: "this entire request must complete by wall-clock time T." Deadlines propagate through the call graph; timeouts don't.

```
  User req  ───► Gateway (deadline = now + 500ms)
                   │
                   ├──► AuthSvc     (deadline propagated, 500ms remaining)
                   │                (auth takes 20ms, 480ms left)
                   │
                   ├──► SearchSvc   (deadline = 480ms)
                   │                (takes 200ms, 280ms left)
                   │
                   └──► RankerSvc   (deadline = 280ms)  ◄── deadline budget shrinks down the graph
```

**Rule:** every RPC has a deadline. The caller owns the deadline, not the callee. Deadlines > timeouts because they prevent wasted work far down the stack.

### Retry

Retry is only safe if:
1. The operation is **idempotent** (or you have an idempotency key), and
2. You have a **budget** (to avoid retry storms), and
3. You use **backoff + jitter**

Retry the following error classes: transient network errors, 503s, 429s (with `Retry-After`). **Never** retry 4xx validation errors, and be careful with 500s (may have partially succeeded).

### Retry Budget

Cap retries as a fraction of requests (e.g., "retries ≤ 10% of successful requests in a sliding window"). Prevents a wave of retries from DDoSing a sick dependency.

```
  WITHOUT BUDGET                      WITH BUDGET
  --------------                      -----------
  Requests ━━━━━━━━                   Requests ━━━━━━━━
  Retries  ━━━━━━━━ ◄── amplifies     Retries  ━━ ◄── capped
  Total load: 2x                      Total load: 1.1x
```

### Exponential Backoff

Wait longer between successive retries: `delay = base * 2^attempt`. Prevents hammering a struggling service.

### Jitter

Add randomness to backoff. Without jitter, all clients retry at synchronized moments, creating thundering-herd spikes.

```
  NO JITTER                            WITH FULL JITTER
  ---------                            ----------------
  all clients retry at t=1s            clients retry at random in [0, 1s]
  ↓                                    ↓
  ┃╏╏╏╏╏  ← spike                      ┆╌┆╌┆╌┆ ← spread
```

Use **full jitter**: `sleep = random(0, base * 2^attempt)`. AWS's blog post on this is the canonical reference.

### Idempotency

An operation is idempotent if applying it N times has the same effect as applying it once. `GET`, `PUT`, `DELETE` are idempotent by spec; `POST` usually isn't.

For non-idempotent operations, use an **idempotency key**: the client generates a UUID per logical operation; the server deduplicates based on the key (stored with TTL in Redis or the DB).

```
  Client                                    Server
  ------                                    ------
  POST /charge  Idempotency-Key: abc123 ──►
                                            check store for "abc123"
                                            ├─ not found → process, store result
                                            └─ found     → return stored result
```

### Circuit Breaker (interview depth)

Three states:
```
   ┌─────────┐   failures > threshold   ┌──────┐
   │ CLOSED  │ ────────────────────────►│ OPEN │
   │(normal) │                          │(fail │
   │         │◄─────────────────────────│ fast)│
   └─────────┘     success in probe     └──┬───┘
        ▲                                  │ after cooldown
        │     probe succeeds               ▼
        │                           ┌────────────┐
        └────────────────────────── │ HALF-OPEN  │
                                    │(1 req test)│
                                    └────────────┘
```

- **Closed**: normal, counting failures
- **Open**: reject immediately without calling downstream (fail fast, shed load from the sick service)
- **Half-open**: after a cooldown, let one request through to test recovery

**Why it matters:** without it, your threads/goroutines pile up waiting on a dead dependency → resource exhaustion → cascade.

### Load Balancing

Distribute requests across backend instances. Named algorithms to know:

- **Round robin**: simple, ignores backend load
- **Least connections**: good for variable request costs
- **Least request / least loaded**: better
- **Power of two choices (P2C)**: pick 2 random backends, send to the less loaded one — very effective, simple
- **Consistent hashing**: for cache affinity (Memcached, Cassandra)

**Client-side LB** (Finagle, gRPC): client maintains pool, picks backend. Low latency, needs service discovery.
**Proxy-side LB** (Envoy, NGINX, AWS ALB): centralized, easier ops, one extra hop.

### Service Discovery

How clients find backends. Options:

- **DNS**: simple, cached, stale on changes
- **Registry** (Consul, etcd, Zookeeper): clients query on startup, subscribe to changes
- **Service mesh** (Istio, Linkerd): sidecar handles it transparently
- **Platform-native** (Kubernetes `Service`, AWS Cloud Map): abstracted behind VIP or DNS

**What interviewers want:** know that it exists, know staleness is the main failure mode (clients calling dead endpoints).

### Connection Pooling

Opening TCP/TLS connections is expensive (1-3 RTTs + handshake). Keep a pool of persistent connections per upstream. Size = f(latency, throughput, target concurrency).

**Pool exhaustion** is a common incident trigger: slow upstream → requests hold connections → pool empties → new requests block → caller latency spikes → cascade.

### Head-of-Line Blocking (high level)

Multiple requests sharing a connection/queue, and one slow request blocks the rest.

```
  HTTP/1.1 pipelined requests on one connection:
  [req1: slow] [req2] [req3] [req4]
       ▲
       └── blocks everything behind it
```

HTTP/2 solves this at the connection layer (multiplexed streams), but can still HOL-block at the TCP layer (one packet loss stalls all streams). HTTP/3 (QUIC) solves that.

### Request Hedging

Send the same request to multiple backends; take the first response; cancel the others. Reduces tail latency at the cost of extra load.

```
  t=0     send req to backend A
  t=100ms  ← still waiting, send duplicate to backend B
  t=110ms  ← B responds, cancel A
```

Rule of thumb: only hedge **idempotent reads**, only after p95 latency has elapsed, cap hedge rate at ~5% of traffic. Google's "The Tail at Scale" paper is the reference.

### Backpressure

Signal to upstream: "slow down, I can't keep up."

- TCP does this at the transport layer (receive window)
- App-layer: bounded queues, 429 responses, gRPC flow control
- Without it, a slow consumer's queues grow unbounded → memory pressure → OOM → outage

### Partial Failure

Some dependencies up, others down or slow. The defining characteristic of distributed systems. Your job is to keep the product functional — even degraded — when parts fail.

### Cascading Failure

One component's failure overloads its neighbors, which fail and overload *their* neighbors. The canonical death spiral:

```
  Svc A slow → A's callers' threads pile up → callers' pools exhaust
            → callers' callers pile up → cluster-wide brownout
```

Prevented by: timeouts, circuit breakers, load shedding, bulkheads, retry budgets.

---

## 5. Latency and Tail Behavior

### Why p99 Matters

Averages lie. A service with 10ms avg latency and 5s p99 will feel broken. In a fanout system, **a single user request touches dozens of backends**, so the slowest one dominates.

If each of 100 backends has p99 = 1s and you hit all 100, the probability that *at least one* hits its p99 on a given request is `1 - 0.99^100 ≈ 63%`. You've effectively made p99 the median.

```
  Latency distribution:
         p50    p90    p99        p99.9
          │      │      │           │
  ▁▂▃▄▅▆▇█▇▆▅▄▃▂▁▁▁▁▁▁▂▁▁▁▁▁▁▁▁▁▁▁▁▁▃
          10ms  80ms  500ms        3s
          ↑             ↑
          "the system   "what users
           usually       actually
           looks like"   experience"
```

### Fanout Amplification

```
  Gateway ──► Svc A  (p99 = 50ms)
          ──► Svc B  (p99 = 50ms)
          ──► Svc C  (p99 = 50ms)
          ...
          ──► Svc J  (p99 = 50ms)

  End-to-end latency ≈ max(all p99s) ≈ p99 of max-of-10 samples
                    ≫ 50ms
```

Mitigations: fewer hops, hedged requests, reduce fanout (cache, pre-compute), tighter deadlines to fail fast and fall back.

### Queueing Delay

Little's Law: `L = λW` (avg items in queue = arrival rate × avg wait time).

As utilization approaches 100%, queueing delay → ∞. An M/M/1 queue at 80% utilization has 4x the wait time of one at 50%. **Target ~50-70% utilization** for latency-sensitive services. If you're at 95%, you're one traffic spike from a meltdown.

```
  Latency vs utilization (hockey stick):

   latency
     │                             ╱
     │                          ╱
     │                       ╱
     │                    ╱
     │              ╱
     │ ___________╱
     └────────────────────────────── utilization
     0%     50%    70%    90%   100%
```

### Slow Downstreams

When a dependency gets slow, your caller gets slow *and* accumulates in-flight requests. Without circuit breakers and timeouts, slow = down from the caller's perspective — except worse, because sockets and memory stay pinned.

### Retry Amplification

```
  Baseline: 1000 rps to Svc A
  A starts failing 50% of requests
  Clients retry 3x → 1000 + 500 + 250 + 125 = 1875 rps
  A gets *more* load while it's sick → death spiral
```

Fix: retry budgets, circuit breakers, jittered backoff.

### Why One Slow Dependency Dominates

A sync call graph's latency is additive (along the critical path) and its p99 is dominated by the worst link. You cannot be faster than your slowest *required* dependency. This is why staff engineers push to:
- Move non-critical dependencies off the critical path (async, background, fire-after-respond)
- Cache or pre-compute
- Degrade gracefully (fall back to a less-good answer instead of waiting)

### Latency Budgets Across Hops

State the budget explicitly:

```
  User-visible SLO: p99 ≤ 300ms end-to-end

  Budget breakdown:
    client→gateway network:  20ms
    gateway processing:      10ms
    auth (parallel):         30ms  ◄── runs in parallel with rest
    data fetch:             150ms
    rank/compose:            50ms
    gateway→client network:  20ms
    slack:                   20ms
    ─────────────────────────────
    total on critical path: 270ms   (+20 slack → 290, under 300ms)
```

Saying this out loud in an interview signals staff-level thinking. Then: "if ranker slips, I degrade to unranked results rather than exceed the budget."

---

## 6. Resilience Patterns

### Deadlines vs Timeouts

**Use deadlines**, propagate them through the call graph (gRPC does this natively; REST needs a header convention or tracing context). Timeouts per-hop work but let orphaned work keep running downstream after the caller gave up.

### Retries Only When Safe

Retry logic checklist:
1. Is the operation idempotent? (If not, skip retry or use idempotency key.)
2. Is the error transient? (Network, 503, 429 — yes. 400 — no. 500 — maybe.)
3. Do I have budget left?
4. Am I using jittered exponential backoff?
5. Have I capped max attempts (typically 2-3)?

### Idempotent APIs

Design endpoints so retries are safe:
- Use `PUT` with the client-chosen ID, not `POST`
- Accept an idempotency key on mutations
- Make sure writes are atomic and keyed by a unique request ID

### Fallback / Graceful Degradation

When a dependency fails, return something useful instead of failing the whole request:
- Stale cached value
- Default answer
- Partial page ("recommendations unavailable" vs blank screen)
- Slower/cheaper path

**Interview line:** "Ranker is down → return chronological feed. Personalization is down → show trending. We never show a blank screen."

### Circuit Breakers

Already covered. The key staff-level point: **they protect you from waiting on dead things**, which is what causes cascades. They also give the downstream breathing room to recover.

### Bulkheads / Isolation

Partition resources so one bad dependency can't consume all your threads/connections.

```
  ONE SHARED POOL                     BULKHEADED POOLS
  ---------------                     ----------------
  [================]                  [======] svcA pool (50 conns)
  all 200 conns                       [======] svcB pool (50 conns)
  if svcA hangs, A fills              [======] svcC pool (50 conns)
  all 200 → everyone dies             [======] svcD pool (50 conns)
                                      svcA hang only exhausts its pool
```

Implement with: separate thread pools, separate connection pools, separate queues per downstream or per tenant.

### Rate Limiting / Admission Control

Reject requests at the edge before they consume backend resources. Algorithms: token bucket, leaky bucket, sliding window. Dimensions: per-user, per-tenant, per-API, global.

### Load Shedding

When overloaded, drop lowest-priority work to preserve the rest.

```
  incoming ──► [admission control] ──► server
                    │
                    ├─ priority 0 (health checks)   ◄── always admit
                    ├─ priority 1 (paid users)
                    ├─ priority 2 (free users)      ◄── shed first
                    └─ priority 3 (batch/crawlers)  ◄── shed earliest
```

Better to serve 80% of users well than 100% poorly.

### Backpressure

If you can't keep up, tell upstream. Mechanisms: bounded queues + reject-on-full, HTTP 429, gRPC `RESOURCE_EXHAUSTED`, TCP receive-window tightening, explicit flow-control credits.

Anti-pattern: unbounded queues. They just move the failure from "drop requests" to "OOM and take down everything."

### Fail Open vs Fail Closed

When a dependency is down, which default?

- **Fail open**: allow the action. Use when the dep is advisory (e.g., fraud-check timeout → allow the purchase, flag for review).
- **Fail closed**: deny the action. Use when the dep is a hard invariant (e.g., authz down → deny by default).

**Interview line:** "I'll fail closed for auth, fail open for personalization. The cost of a wrong answer differs."

---

## 7. Failure Scenarios

### Slow Dependency

**Symptom:** latency spike, thread/connection pool fills, upstream clients time out.
**Mitigation:** tight deadlines, circuit breaker, bulkhead, parallel hedged request (for reads), degrade to cached/fallback.

### Dependency Outage (full down)

**Symptom:** 100% errors from that dep.
**Mitigation:** circuit breaker trips fast → fail fast → fallback path. The circuit breaker is what keeps you up while the dep recovers.

### Network Partition (practical)

**Symptom:** some clients can reach a service, others can't. Or two data centers can't see each other.
**Mitigation:** retry against alternate endpoints, multi-AZ/region failover, accept eventual consistency on the partitioned side (AP systems), or reject writes on one side (CP systems). State which you chose.

### Partial Response

**Symptom:** aggregator got 7/10 subcomponents, 3 failed.
**Mitigation:** return partial with flags ("ads: unavailable"), don't fail the whole request. This is the core of graceful degradation.

### Duplicate Request Due to Retry

**Symptom:** same order placed twice, same charge twice.
**Mitigation:** idempotency keys stored server-side with TTL; make mutations keyed by request ID.

### Stale Endpoint / Failed Service Discovery

**Symptom:** client dials a dead IP, gets connection refused or timeout.
**Mitigation:** aggressive health checks, short TTL, connection error → remove backend → retry to sibling, service-mesh sidecar for fresh state.

### Connection Exhaustion

**Symptom:** "too many open files," "dial tcp: can't assign requested address," new requests block indefinitely.
**Mitigation:** bounded connection pools, keepalives, reject-on-full, separate pools per upstream, timeouts on idle conns.

### Retry Storm

**Symptom:** dep gets sick, load doubles instead of halving, spiral to death.
**Mitigation:** retry budgets, circuit breakers, jittered exponential backoff, cap attempts at 2-3.

### Cascading Failure

**Symptom:** one service degrades, its callers degrade, their callers degrade, cluster-wide brownout.
**Mitigation:** every pattern above in combination: timeouts + circuit breakers + bulkheads + load shedding + retry budgets. **No single pattern is enough**; they're a defense-in-depth stack.

```
  Cascading failure cross-section:

  User → Gateway → Svc A → Svc B → Svc C (slow)
                     │        │        │
                     │        │        └─ latency 50ms → 5s
                     │        └─ threads held 5s, pool empty,
                     │           new reqs block
                     └─ same, amplified
         └─ same, amplified to user
```

---

## 8. Service Communication Design Choices

### REST vs gRPC (interview depth)

| Dimension | REST (JSON/HTTP) | gRPC (proto/HTTP2) |
|---|---|---|
| Wire format | JSON (text) | Protobuf (binary) |
| Schema | Optional (OpenAPI) | Required (.proto) |
| Latency | Higher (JSON parse) | Lower |
| Payload size | Larger | ~30-50% smaller |
| Streaming | SSE / WebSocket bolt-on | Native (4 modes) |
| Browser support | Native | Needs gRPC-Web proxy |
| Tooling | Universal | Strong in Go/Java/C++/Python |
| Human debuggability | Easy (curl) | Harder (needs grpcurl) |
| Default for | External/public APIs | Internal service-to-service |

**Interview heuristic:** "gRPC for internal, REST for external. gRPC when latency/throughput matters or streaming is needed. REST when browsers, debugability, or polyglot clients dominate."

### Unary vs Streaming

- **Unary**: default. Simpler. Easier to retry, LB, trace.
- **Streaming**: use when partial/progressive results help the user (LLM tokens), when payloads are huge, or when you need server push.

### Sync RPC vs Async Workflow

**Sync RPC** when:
- User is waiting for the answer
- Call is fast (<~100ms expected, <~500ms p99)
- Availability of the dep is high

**Async workflow** when:
- User doesn't need to wait
- Work is long (>1s) or variable
- Failure should be retried over hours, not milliseconds
- Multiple consumers need the event

**Hybrid:** sync accept + async process (202 Accepted + callback/webhook/polling). Common for job submission, long-running ML.

```
  HYBRID PATTERN
  --------------
  POST /jobs           ───► 202 Accepted, {jobId, pollUrl}
  GET /jobs/{id}       ───► 200 {status: running}
  GET /jobs/{id}       ───► 200 {status: done, result: ...}
       or
  webhook to callback  ◄─── POST {jobId, status, result}
```

### Client-side vs Proxy/Mesh Load Balancing

- **Client-side**: lower latency, more complex clients, every language needs LB logic
- **Proxy (L7 LB)**: centralized, extra hop (~1ms), single ops story
- **Service mesh (sidecar)**: proxy per pod, transparent to app, adds complexity + a bit of latency, great observability

### When Request/Response Is the Wrong Model

- Long-running jobs (hours) → job API or async workflow
- Pub/sub of events to many consumers → message broker
- Live progressive updates → streaming or SSE/WebSocket
- High-fanout writes to one datastore → batch/async ingest

### When Chatty Service Boundaries Are Dangerous

N+1 RPCs: for each of N items, call a service → N network hops per request. Kills p99.

```
  CHATTY                              COALESCED
  ------                              ---------
  for item in cart:                   cart_ids = [i.id for i in cart]
    svc.getPrice(item.id)             svc.getPrices(cart_ids)  ◄── one call
  (N round trips)                     (1 round trip)
```

Fix: batch APIs, BFFs (backend-for-frontend) that aggregate, GraphQL for client-driven selection.

### When Aggregation/Fanout Services Become Fragile

An aggregator that synchronously calls 10 backends has availability ≤ product of 10 backends' availability. At 99.9% each, that's 99.0%. Fixes:

- Make many of those calls **optional** (degrade on failure)
- Run them in parallel (not serial) to share the deadline
- Cache aggressively
- Pre-compute when the aggregation is stable

---

## 9. Interview Reasoning Patterns

### "Should this be sync RPC or async?"

Ask:
1. Does the caller need the result *now* to proceed?
2. Is the callee fast (<500ms p99) and highly available?
3. Is the work short?

All three yes → sync RPC. Any no → strong signal for async.

### "Where should timeouts live?"

- At **every** RPC (no unbounded waits, ever)
- Set by the **caller** (they own the deadline)
- Shorter than the caller's remaining deadline
- Propagated via deadline (gRPC) or header (REST)

Example: gateway has 300ms budget → child call gets deadline = now + 200ms (reserves 100ms slack).

### "How do I set retry policy safely?"

1. Only retry idempotent ops (or use idempotency keys)
2. Retry only on transient errors
3. Max 2-3 attempts
4. Exponential backoff with **full jitter**
5. Enforce a retry budget (e.g., ≤10% of successes)
6. Retry at the **closest** layer to the failure — never at every layer (multiplies)

### "When do I need idempotency keys?"

Any mutation that might be retried: payments, order placement, message sending, resource creation. Client generates UUID; server dedups by key with TTL.

### "How do I prevent cascading failure?"

Defense in depth:
1. Timeouts/deadlines everywhere
2. Circuit breakers on every remote call
3. Bulkheaded connection/thread pools per upstream
4. Retry budgets
5. Load shedding at the edge
6. Backpressure on queues
7. Graceful degradation paths

"No single one is enough; they stack."

### "How do I reason about end-to-end latency budgets?"

1. Start from user-visible SLO (e.g., p99 ≤ 300ms)
2. Decompose into hops, identify critical path (serial) vs parallel
3. Assign budgets with slack (10-15%)
4. For each hop, state: "if it slips, I degrade to X"
5. Name the biggest risks: fanout amplification, slow dep on critical path, queueing at hotspots

### "When is gRPC better than REST?"

- Internal service-to-service where you control both ends
- Latency-sensitive, high-throughput paths
- Need for streaming
- Strong typing/codegen valued

Stay with REST when: public APIs, browser clients, heterogeneous stack, human debuggability matters.

### "When should I use streaming?"

- Large/unbounded payloads
- Progressive user-visible results (LLM tokens, search)
- Server push (price feeds, notifications)
- Bidirectional (chat, collab)

Avoid when: simple req/resp works, LB/retry needs to be easy, short payloads.

### "What are the first bottlenecks and operational pain points?"

Name them unprompted:
1. Hotspot partitions / hot keys
2. Connection pool exhaustion under slow dep
3. Retry storms after a dep blip
4. Fanout p99 amplification
5. Queue buildup / unbounded queues
6. Thundering herd on cache miss / cold start
7. Cross-region latency / cost
8. Schema-change rollouts without backward compatibility

---

## 10. Common Candidate Mistakes

- **Treating RPC like a normal function call.** Ignoring that it can be slow, fail, or partial-succeed.
- **"Just retry."** Without idempotency, backoff, jitter, or budget. This is how you make things worse.
- **No deadlines.** Saying "I'll call Service X" without stating a budget for the call.
- **No retry storm consideration.** Every layer retries 3 times → 3^N amplification through N layers.
- **Hand-waving service discovery/LB.** "The load balancer handles it" — okay, but *which* algorithm, who owns it, how does it find backends, what's the staleness window?
- **Ignoring tail latency and fanout.** Quoting averages. Forgetting that calling 50 backends makes their p99 your p50.
- **Using sync RPC for long-running work.** "The user clicks export, then we run a 5-minute job on the request thread." → timeouts, wasted resources, bad UX.
- **Ignoring connection pooling.** Designing assuming connections are free; getting surprised by socket exhaustion.
- **No graceful degradation.** Designing only for happy path; "what if ranker is down?" → blank screen.
- **Mentioning circuit breakers without explaining what they prevent.** Jargon without mechanism.
- **Conflating at-least-once and exactly-once.** "Kafka gives exactly-once" — only with careful consumer idempotency, not by default.
- **Ignoring data consistency under partition.** CAP trade-offs get glossed.

---

## 11. Final Cheat Sheet

### Sync RPC vs Async Messaging vs Streaming

| | Sync RPC | Async Messaging | Streaming RPC |
|---|---|---|---|
| Caller waits? | Yes | No | Yes (for messages) |
| Coupling | Tight (temporal) | Loose | Tight (connection) |
| Typical latency | ms | ms–hours | ms, progressive |
| Delivery | 1:1, synchronous | broker buffers, async | 1:1 over stream |
| Failure handling | Retry or fail | Redelivery, DLQ | Reconnect, resume |
| Good for | Reads, inline deps | Long work, fanout events, decoupling | Progressive results, push, large payloads |
| Bad for | Long work, unstable deps | Work needing instant answer | Simple req/resp |
| Ordering | Per call | Per-partition (Kafka) | In-order within stream |
| Typical tech | gRPC, REST, Thrift | Kafka, SQS, RabbitMQ, NATS | gRPC streaming, SSE, WebSocket |
| Retry cost | High (user waiting) | Low (broker retries) | Complex (mid-stream) |

### REST vs gRPC

| | REST (JSON/HTTP1.1) | gRPC (proto/HTTP2) |
|---|---|---|
| Wire | JSON text | Protobuf binary |
| Schema | Optional (OpenAPI) | Required (.proto), codegen |
| Streaming | SSE/WebSocket add-ons | Native (4 modes) |
| Perf | Lower | Higher (2-5x throughput typical) |
| Browser | Native | Needs gRPC-Web proxy |
| Debug | curl, Postman | grpcurl, BloomRPC |
| Deadline propagation | Via header convention | Native |
| Best for | External/public APIs, browser, polyglot | Internal services, latency-sensitive, streaming |
| Learning curve | Low | Medium |

### Resilience Checklist for Service-to-Service Calls

For every RPC, ask and answer:

- [ ] Deadline set and propagated?
- [ ] Timeout < caller's remaining deadline?
- [ ] Retries: only idempotent? Max 2-3? Jittered backoff? Budget enforced?
- [ ] Circuit breaker on this call?
- [ ] Bulkheaded connection pool?
- [ ] What's the fallback if this fails?
- [ ] Fail open or fail closed? Justified?
- [ ] Is this on the critical path? Can it move off?
- [ ] p99 latency of callee vs remaining budget?
- [ ] Backpressure path if callee slows?
- [ ] Service discovery freshness / health checks?
- [ ] Load-shed priority for this caller?
- [ ] Tracing + metrics (p50/p99, error rate, saturation)?

### 10 Likely Interview Questions + Strong Short Answers

**Q1. When would you pick async messaging over sync RPC?**
When the caller doesn't need the answer inline, the work is long or variable, the callee's availability is lower than I want to inherit, or I need to fan out to multiple consumers. Async decouples availability and smooths load; the price is eventual consistency and harder tracing.

**Q2. How do you prevent a retry storm?**
Retry budget (retries ≤ ~10% of successes), exponential backoff with full jitter, cap attempts at 2-3, retry at only one layer in the stack, and circuit-breaker so retries stop when the dep is clearly down.

**Q3. Why does p99 matter more than average latency in a fanout system?**
Because with N parallel backends, the user waits for max(all), so their experience is closer to p99 of the individual services. If I have 100 backends each at p99 = 1s, ~63% of user requests will hit at least one 1s call.

**Q4. Where do you put timeouts?**
Everywhere. Caller-owned, shorter than the remaining request deadline, and propagated as a deadline (wall-clock time) rather than a per-hop timeout so downstream work stops when the user has already given up.

**Q5. When is a circuit breaker the wrong choice?**
For infrequent calls where the failure-rate signal is noisy, for operations where failing fast is worse than a slow retry (rare), or when the "downstream" is something like a cache whose failure should just fall through to the source of truth without circuit logic.

**Q6. How do you handle duplicate requests from retries?**
Idempotency keys: client generates a UUID per logical op, server dedups with TTL. For natural idempotency (`PUT` by ID), make that the API shape. For mutations like payments, the key is mandatory.

**Q7. How would you design a user-facing endpoint with a 300ms p99 SLO that calls 8 backends?**
Parallelize what I can, identify the critical path, set a deadline at the gateway and propagate. Make as many backends optional/degradable as possible — fallback to cached or default on timeout. Use hedged requests on idempotent reads past p95. Tight per-call deadlines (maybe 150ms for criticals, 80ms for optionals). Consider pre-computing or caching the aggregation.

**Q8. Sync RPC vs async workflow for an email-sending feature?**
Async. The user doesn't need the SMTP ack inline; email delivery can take seconds or minutes. Put the send on a queue, retry with backoff for hours, DLQ after max attempts. Return 202 to the user immediately.

**Q9. What causes cascading failure and how do you prevent it?**
A slow or failing dep pins caller resources (threads, connections) → caller exhausts → its callers pin *their* resources, and so on. Prevent with defense in depth: deadlines, circuit breakers, bulkheaded pools per upstream, retry budgets, load shedding at the edge, and graceful degradation paths.

**Q10. When is gRPC streaming the right choice?**
When the client can use partial results before the end (LLM token streaming, search-as-you-type), when the payload is too large for one message, when the server needs to push updates (notifications, feeds), or when the protocol is truly bidirectional (collab, trading). For simple req/resp, unary is easier to retry, LB, and trace — default to unary unless streaming earns its keep.

---

### One-page "call-and-response" to memorize

- "Is this sync or async?" → "Async unless the caller must have the answer inline."
- "What's the deadline?" → "Caller-owned, propagated, tighter than the user-facing SLO minus slack."
- "What if it fails?" → "Circuit breaker trips; we fall back to [cache / default / degraded mode]."
- "What if it's slow?" → "Deadline fires; we shed or degrade; bulkhead prevents pool exhaustion."
- "What if retries pile up?" → "Budget + jittered backoff + breaker stops them."
- "What about p99?" → "Watching it, not the average. Fanout amplifies tails; I minimize serial hops and degrade non-critical ones on timeout."
- "Idempotent?" → "Yes, via [natural PUT-by-id / idempotency key stored with TTL]. Safe to retry."