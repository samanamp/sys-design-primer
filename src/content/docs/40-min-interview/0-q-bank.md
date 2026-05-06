---
title: LLM Inference Q Bank
description: LLM Inference Q Bank
---

Two contextual notes that will affect how you should approach the screen: at staff level, the interviewer gives a prompt and minimal direction — you scope the problem, decide what to focus on, and manage the conversation. And the model is treated as a black box; the round evaluates infrastructure and distributed systems skills, not inference internals — though for an Inference Engineer screen specifically, expect deeper KV cache / batching follow-ups than a generic SWE round.

---

**1. Single-GPU inference batching system [Reported — most common Anthropic prompt]**

"You have a single GPU that can process up to 100 inputs per batch. Users submit requests synchronously and wait for results. Design the system that receives inputs, batches them, processes them on the GPU, and returns responses to the correct users."

Follow-ups to expect:
- Batching policy: max-batch-size vs max-wait-time trade-off; how do you tune the wait window?
- What's your queue data structure? FIFO vs priority? How do you handle a user who's been waiting too long while a full batch keeps forming?
- The GPU dies mid-batch. What happens to the 100 in-flight requests? Idempotency story?
- Now requests have variable lengths (LLM-style). How does that change everything?
- A bad input causes the GPU to OOM/crash. How do you isolate the offender without retrying all 100?
- Backpressure: queue is full and growing. What do you return to clients?
- How do you prove to me your batching window is correctly tuned? What metric do you watch?

---

**2. End-to-end LLM inference API [Reported]**

"Design an inference API for serving large language models. Variable-length prompts and outputs. Concurrent users. Streaming responses."

Follow-ups (these are reported as the actual probes from candidate write-ups):
- How do you handle variable-length requests, GPU memory under concurrent requests, priority-based queueing, and streaming responses?
- Continuous batching vs static batching — when does each lose? Walk me through how requests join an in-flight batch.
- KV cache memory accounting: how do you admission-control when memory is the bottleneck, not compute?
- Streaming: SSE vs WebSocket vs gRPC streaming — pick one and defend it.
- A request asks for 100k output tokens. Another asks for 50. How do you keep the long one from starving the short one?
- Now add prefill vs decode separation (disaggregated serving). What changes in your design?

---

**3. Distributed inference fleet with routing [Reported variant]**

"Same as #2, but now you have N GPU nodes across one region. Design the routing layer."

Follow-ups:
- How do you determine which GPU has capacity, and how do you handle failover?
- Sticky routing for KV cache reuse vs load-balanced routing for utilization — pick.
- A node is degraded but not failed (3x slower). How does your system detect and respond?
- Hot-spotting: 80% of traffic hits 20% of prompts (shared system prompt). How do you exploit that?
- Multi-region: how do you route, and what do you replicate?

---

**4. Performance debugging — p95 spike [Reported]**

"A system's 95th percentile latency spiked from 100ms to 2000ms. How would you investigate and fix it?"

Follow-ups:
- Walk me through your diagnostic tree before you touch anything.
- p50 is unchanged, only p95/p99 moved. What does that tell you?
- TTFT is fine but TPOT (time per output token) is bad. Where do you look?
- You discover one customer's traffic pattern shifted. How do you confirm causation, not correlation?
- What metrics/traces do you wish you had instrumented before this happened?

---

**5. Model binary / weights distribution [Reported]**

"A large file needs to reach thousands of machines from a single bandwidth-constrained source" — framed as "distribute a 500GB model to 1000 inference nodes as fast as possible."

Follow-ups:
- BitTorrent-style P2P vs tree distribution vs CDN — trade-offs.
- Half the fleet has the previous version. Incremental delta distribution?
- Atomic version cutover across the fleet — how?
- A node fails mid-download. Resume strategy?
- How does your design change if this is a hot rollback (need to revert in 60 seconds)?

---

**6. Multi-tenant inference platform with SLOs and quotas [Reported / Variant]**

"Design a production inference platform serving multiple ML models backed by GPUs. Strict latency SLOs for online traffic, high throughput via dynamic batching, model versioning with A/B routing, autoscale across heterogeneous GPU nodes (A10/A100/H100), isolation and quotas for multiple tenants, fault tolerant, two-region global deployment."

Follow-ups:
- How do you allocate H100s vs A100s? Per-model affinity or per-request?
- Tenant A's burst is starving Tenant B. Mechanism?
- A/B routing with KV cache: how do you not double your memory footprint?
- TTFT P95 SLO is 200ms. You're at 250ms. What's your debug-then-fix order of operations?

---

**7. Prefix / KV cache management system [Extrapolation, but high-probability for Inference Engineer specifically]**

"Design the prefix-caching subsystem behind a multi-tenant LLM API. Many requests share long system prompts. Cache must be correct, fair across tenants, and respect GPU memory limits."

Follow-ups:
- Cache key — exact-match or token-prefix tree (radix)? Memory cost of each.
- Eviction policy: LRU vs LFU vs cost-aware (longer prefixes are more expensive to recompute)?
- Cross-request, cross-tenant: does Tenant A's cache benefit Tenant B? What about privacy?
- CPU↔GPU cache tiering: when do you offload, when do you evict outright?
- A misbehaving client sends 10k unique long prompts/sec, polluting cache. Defenses?
- How does prefix caching interact with continuous batching (different requests at different cache states in one batch)?

---

**8. Speculative decoding system [Extrapolation, role-specific]**

"Design a serving system that uses speculative decoding (draft model + verifier) to improve TPOT. Requests have varying acceptance rates."

Follow-ups:
- When is speculative decoding a net loss? How do you decide per-request?
- Adaptive `k` (lookahead): how do you tune online?
- Memory accounting: now you have two models resident. How do you batch?
- Failure mode: draft model is much worse than expected on a workload shift — your system has been wasting compute. How do you detect?
- EAGLE vs Medusa vs vanilla draft model — be ready to argue trade-offs if pushed.

---

**9. Inference platform with safety/moderation layer [Reported variant]**

"Design serving such that every output passes a safety classifier before user delivery. Maintain low latency."

Follow-ups:
- Safety check in series adds latency. Run in parallel with inference finalization?
- Streaming case: you've already streamed 200 tokens when the classifier flags. What now?
- Classifier is itself a model on GPU. How do you co-locate without contention?
- Audit trail requirements: durable logging without latency hit.

---

**10. High-concurrency request admission and overload control [Variant]**

"You have a fixed inference fleet. Traffic is bursty — 10x spikes for 30 seconds. Design the admission control / load shedding layer."

Follow-ups:
- Reject vs queue vs degrade (smaller model fallback)?
- What scaling signal do you use? Raw GPU utilization is misleading because latency can be out of control while utilization looks fine. Queue depth weighted by estimated token count is a better signal. Defend or attack that.
- How do you estimate token count before generation starts?
- Cold-start: scaling up means loading weights (minutes). What do you do in the meantime?
- A single big customer is 60% of traffic. Per-tenant rate limiting algorithm — token bucket, sliding window, or something else?

---

**General preparation notes from the reports**

A few cross-cutting things that interviewers consistently push on regardless of question:

- Abstraction ability — can you cut through AI framing and identify the core infrastructure problem? Interviewers specifically watch for whether you get intimidated by unfamiliar terminology or can decompose the problem into components you know how to solve. For you specifically, the inverse risk: don't go too deep into kernel-level detail when they want infra. Read the room.
- "Design a batch inferencing API for a GPU cluster" should become "design a batching system with constrained compute resources." The infrastructure patterns (queuing, load balancing, async processing) are the same patterns you'd use in any distributed system.
- Anthropic interviews are reported as deliberately open-ended; the interviewer may not have a single correct answer in mind. They want to see how you think through an unsolved problem, not whether you reproduce a known architecture. So expect novel twists; don't pattern-match too hard.
- The retrieval-under-pressure failure mode you identified from PI applies here too. Recommend doing 2–3 of these timed (50 min, whiteboard or excalidraw) before the screen, not just reading them. The one most worth drilling cold is #1 — it's the highest-frequency reported prompt and #2/#3 are extensions of it.

---
## 2nd set

## Category 1: Rate-Limiting / Quota / Concurrency Control

**1. Distributed rate limiter**
*"Design a global rate limiter for an API that serves 5M requests/sec across 3 regions. Per-user limits, per-API-key limits, fairness across noisy neighbors."*

The canonical "do you understand distributed counters and consistency tradeoffs" question. Test of: local-vs-global enforcement, fail-open vs fail-closed policy, hot-key handling.

**2. Distributed quota/budget enforcement**
*"Design a system that enforces a monthly spending budget across many concurrent API consumers. Each request consumes a variable amount of the budget. Once the budget is exhausted, no further requests should succeed. Multi-region, low-latency."*

A meatier variant — combines rate limiting with the harder problem of enforcing a *cumulative* limit that can't be approximated as easily as a rate.

---

## Category 2: Job Orchestration / Async Workflows

**3. Distributed job queue / scheduler**
*"Design a job processing system that handles 2B jobs/day with retry, scheduling, priority, and fair-share across tenants. Some jobs run for milliseconds, some for hours."*

Tests: queueing primitives, delivery semantics, isolation, gang scheduling for big jobs, the head-of-line blocking problem.

**4. Webhook delivery system**
*"Design the system that delivers webhooks to customer endpoints — at-least-once delivery, exponential backoff, dead-lettering, isolation between fast and slow customers, signed payloads."*

The classic Stripe-style question. Tests: per-tenant queueing, retry policy, the "one slow customer can't block others" problem.

---

## Category 3: Real-Time Data / Counting / Metrics

**5. Real-time metrics / counting system**
*"Design a system that counts events at scale — think view counts, like counts, ad impressions. 100B events/day, near-real-time read freshness, queries that aggregate across counters."*

Tests: write-path aggregation, hot-counter handling, idempotency, the streaming pipeline.

**6. Time-series / observability database**
*"Design the storage and query layer for a metrics system: 10M points/sec ingest, 100M active series, queries from 'last 5 minutes one series' to 'last 30 days, sum 10K series.'"*

Tests: cardinality, compression, query planning, index structure. More TSDB-flavored.

---

## Category 4: Pub/Sub, Notifications, Streaming

**7. Notification / fan-out system**
*"Design a notification system that delivers messages to users across web, mobile push, email, SMS. 100M users, support for personalized batching ('don't send more than 3 notifications/hour to a user'), preference management, retry."*

Tests: fan-out, deduplication, per-user state, channel routing, the cross-channel-rate-limit problem.

**8. Real-time pub/sub for collaborative apps**
*"Design the real-time messaging layer for a collaborative app — think Figma-style multiplayer presence and document edits. Sub-100ms propagation, 100K concurrent rooms, rooms ranging from 2 to 1000 participants."*

Tests: WebSocket fan-out, presence/heartbeat, ordering, the room-sharding problem.

---

## Category 5: Storage / Caching / Data Systems

**9. Distributed cache / KV store**
*"Design a distributed cache serving 10M ops/sec with sub-ms latency. Eviction, replication, hot-key mitigation, consistency model."*

Tests: classical caching, replication topology, the hot-key problem, eviction policy under memory pressure.

**10. Idempotency / deduplication system**
*"Design the idempotency layer for a payments API. Clients send an idempotency key with each request; the system guarantees that a request with the same key is executed exactly once even across retries, network partitions, and our own restarts."*

Tests: storage of idempotency keys, concurrent same-key handling, the partial-execution case (side-effect happened, response storage failed).

---

## Bonus — JD-Aligned Variants

If Rajan does pick a question with ML-infra flavor (less likely given the description, but possible given the role):

**11. Multi-tenant model serving (research-flavored)**
*"Design an internal serving system for 200+ research model variants on a shared GPU pool. Variable load per model, fast checkpoint loading, hibernation of cold models, per-researcher cost attribution."*

**12. Inference benchmarking harness**
*"Design a system that benchmarks new model variants automatically — varying batch sizes, sequence lengths, hardware types — producing comparable performance numbers over months."*

---

## Prep strategy

The safe play is to prepare cold on the 10 generalist questions above. Each has well-known patterns, but each has a depth of follow-up where staff signal lives:
- The hot-key problem (in rate limiting, counting, caching, queueing — same shape, different domain)
- Fail-open vs fail-closed policy (rate limiting, idempotency, webhook delivery)
- Per-tenant isolation under noisy neighbor (every question on this list)
- Multi-region eventual consistency (every question that has "global" in it)
- Idempotency and exactly-once semantics (the universal hard problem)

For each question, prep:
1. The 1-minute "happy path" architecture
2. The specific scaling math
3. The 3-4 hardest follow-ups and how you'd handle them
4. The failure modes and graceful degradation
5. One thing you'd push back on in the prompt

If you want, I can write the full staff-level prompt (in the format from your question bank) for any of these so you can practice cold against a strong reasoning model. Which 2-3 do you want me to start with?