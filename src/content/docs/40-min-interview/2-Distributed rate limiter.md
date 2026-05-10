
---
title: "1-Distributed rate limiter"
description: "Distributed rate limiter"
---

"Design a global rate limiter for an API that serves 5M requests/sec across 3 regions. Per-user limits, per-API-key limits, fairness across noisy neighbors. Walk me through it."

---

# Distributed Rate Limiter — Staff-Level Design

## 1. Reframing

A rate limit ("100 req/sec per user") and a cumulative budget ("$1000/month") look isomorphic but are architecturally distinct. **Rate limits self-heal at window boundaries; cumulative budget errors accumulate forever until reset.** If my regional counter is 15 requests stale at second boundary T, the limit re-enforces correctly at T+1. If a budget counter is $50 stale, that $50 is permanently over-spent until the month rolls.

**[STAFF SIGNAL: rate-vs-cumulative]** This single observation justifies the entire architectural simplification. Local-region enforcement with bounded staleness is acceptable. Probabilistic counters (Count-Min Sketch) are acceptable. Fail-open is often acceptable. None of these are acceptable for cumulative budgets — those need leases, monotonic audit logs, and synchronous sub-budget allocation.

**[STAFF SIGNAL: contrast with cumulative-budget framing]** Rate limiting needs none of: lease management, audit log integrity, sub-budget allocation, charge-back on failure. Recognizing this is the difference between a 5-day implementation and a 5-month one. (See §15 for the architecture inversions a budget would force.)

The two central constraints in this design are:

1. **Latency budget**: API has ~50ms p99 total. The rate limiter gets 1–5ms p99. Multiple synchronous network round trips are unaffordable.
2. **Hot keys**: ~0.1% of keys drive ~50% of traffic. Uniform-shard designs melt under realistic skew.

A naive "Redis cluster with token bucket sharded by user_id" satisfies neither. The rest of this document earns its space by addressing both.

## 2. Scoping

**[STAFF SIGNAL: scope negotiation]** Committed assumptions:

- **Limit policy classes**: *comfort* (per-user, per-API-key fairness; tolerate ±5% over-allowance); *security* (login, password reset; over-allowance is an incident); *cost-protection* (LLM tokens, expensive endpoints).
- **Enforcement**: hard cutoff. Over-limit returns 429 with `Retry-After`. No "soft" mode beyond shadow-evaluation for new rules.
- **Global semantics**: best-effort global with bounded over-allowance. Strict global reserved for ≤1% of rules (signup, security-adjacent).
- **Rule lifecycle**: largely static (per-tier templates) with dynamic overrides. Propagation SLO 5s globally.
- **Cardinality**: 100M active keys aggregate across all dimensions. Top-100 keys ≥10K req/sec each.
- **Tiers**: free / pro / enterprise. Enterprise gets priority during admission control.

Out of scope: DDoS mitigation (upstream), bot detection (separate system), billing-grade quota enforcement (sibling system; §15).

## 3. Capacity and Latency Math

**[STAFF SIGNAL: capacity math]**

```
Workload
─────────────────────────────────────────────────────────────────
Total requests              5,000,000 req/sec
Regions                     3 (us-east, us-west, eu-west)
Per-region requests         ~1,667,000 req/sec
Limit checks per request    4 (user, api_key, customer, endpoint)
Per-region counter ops      ~6,667,000 ops/sec (worst case)
Aggregate counter ops       ~20,000,000 ops/sec (worst case)

Memory
─────────────────────────────────────────────────────────────────
Active keys (aggregate)     100M
Bytes per counter (GCRA)    ~50 B (key + TAT + metadata)
Per-region state            ~5 GB
Aggregate state             ~15 GB     (fits in distributed in-mem)

Latency budget (p99)
─────────────────────────────────────────────────────────────────
API total budget            50 ms
Rate limiter slice          2–5 ms
Intra-DC RTT                0.5–1 ms
Cross-region RTT            30–80 ms   (unaffordable on hot path)
Local fast-path lookup      <0.1 ms    (in-process map)

Effective latency target with 95% local fast-path hit:
  p50 = 0.95 × 0.1 ms + 0.05 × 1.4 ms ≈ 0.16 ms
  p99 = miss-path-dominated ≈ 1.4 ms
─────────────────────────────────────────────────────────────────
```

Hot-key concentration: top-100 keys at ~50K req/sec each = ~5M ops/sec from 100 keys. Uniform sharding would route this to ~100 shards; realistic skew melts the unlucky ones (§7).

## 4. High-Level Architecture

```
┌──────────┐
│  Client  │
└────┬─────┘
     │
     ▼
┌─────────────────────────────────────────────────────────────────┐
│  API Gateway (per region, ~1000 instances)                      │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │  Rate Limit SDK (in-process)                              │  │
│  │  ┌────────────────┐    ┌────────────────────────────────┐ │  │
│  │  │ Local fast-path│───▶│ Rule cache (push-updated, 5s)  │ │  │
│  │  │  cache <0.1ms  │    └────────────────────────────────┘ │  │
│  │  └────────┬───────┘                                       │  │
│  │           │ miss / near-limit                             │  │
│  │           ▼                                               │  │
│  │  ┌──────────────────┐                                     │  │
│  │  │ Regional client  │                                     │  │
│  │  └────────┬─────────┘                                     │  │
│  └───────────┼───────────────────────────────────────────────┘  │
└──────────────┼──────────────────────────────────────────────────┘
               │ ~1ms RTT
               ▼
┌────────────────────────────────────────────────┐
│ Regional Counter Store (Dragonfly/Redis        │     ┌─────────────┐
│ Cluster, ~32 shards, GCRA Lua primitive)       │◀───▶│ Rule Store  │
│  • per-key TAT                                 │     │ Postgres    │
│  • hot-key sub-keys (split)                    │     │ + pub/sub   │
│  • heavy-hitters sketch                        │     └─────────────┘
└──────────┬─────────────────────────────────────┘
           │ async delta gossip (~200ms)
           ▼
┌────────────────────────────────────────────────┐
│  Cross-region replication (CRDT additive)      │
└────────────────────────────────────────────────┘

Hot-path latency:
  fast-path hit  → 0.05–0.1 ms
  fast-path miss → 0.1 + 1 (RTT) + 0.3 (Lua) ≈ 1.4 ms
  hot-key split  → same as miss but on a different shard
```

**[STAFF SIGNAL: layered-limit evaluation]** The SDK runs in-process inside the API gateway. The hot path: rule lookup (local) → fast-path check (local) → optional remote increment (regional). Cross-region sync is async and off the hot path entirely.

The Rule Store and Counter Store are deliberately separate. Rules are read-heavy with ~zero write QPS; counters are write-dominated at 6.7M ops/sec/region. Co-locating them sacrifices independent scaling.

**[STAFF SIGNAL: rejected alternative]** Rejected: a single global Redis cluster serving all 3 regions. Cross-region RTT (30–80ms) destroys the latency budget on every miss, and a partition between regions takes the entire system down. Regional counter stores with async sync localize blast radius and respect the budget. Rejected: counter state in Postgres. Write throughput insufficient by 2 orders of magnitude.

## 5. Algorithm Choice

**Committed: GCRA (Generic Cell Rate Algorithm) for steady limits; sliding window counter as an opt-in for window-aligned analytics-flavored limits.**

GCRA stores a single timestamp per key — the *theoretical arrival time* (TAT). Each request compares `now` to TAT, decides allow/deny, and atomically advances TAT if allowed. O(1) memory per key, O(1) compute, supports burst capacity and steady rate from a single primitive. Used in production by RabbitMQ; Stripe describe a near-identical primitive in public engineering posts.

**[STAFF SIGNAL: rejected alternative]**

```
Algorithm           Memory/key   Accuracy  Burst       Verdict
───────────────────────────────────────────────────────────────────
Token bucket        ~16 B        Exact     Native      OK; 2 fields
Leaky bucket        ~16 B        Exact     Smooths     Reject (no burst)
Sliding window log  O(N) per key Highest   Exact       Reject (memory)
Sliding window ctr  ~32 B        ±0.003%   Approximate OK for windows
GCRA                ~16 B        Exact     Native      Chosen
```

Sliding window log is rejected hard. At 100 req/sec limits across 100M keys, worst case is ~10B timestamps in flight — terabytes of memory. Disqualifying.

Token bucket is acceptable but less elegant: separate fields for token count and last-refill, two reads + two writes per check. GCRA is one read, one CAS via Lua. At 6.7M ops/sec/region, the constant factor matters.

Sliding window counter is reserved for limits where the customer wants window-aligned semantics ("100 per calendar minute" for billing-style rules) rather than rolling. Default is GCRA.

**[STAFF SIGNAL: invariant-based thinking]** Invariant: for any (key, rule), exactly one TAT exists. All checks against the rule mutate the same TAT atomically. This invariant is what keeps the counter store sane under hot-key splitting (§7) — splits create N parallel TATs that we sum.

## 6. The Local Fast-Path

**[STAFF SIGNAL: latency-budget on hot path]** The local fast-path is the difference between a 2ms p99 limiter and a 10ms one. Central optimization, not afterthought.

**Mechanism**: each gateway instance maintains an in-process LRU cache mapping `(key, rule_id) → (recent_count_estimate, last_remote_sync_ns)`. On request:

1. Look up `(key, rule_id)` locally. Miss → fall through to remote.
2. If `recent_count_estimate + 1 < limit × safety_margin`, allow locally; increment local count; do not touch remote.
3. Else fall through: increment remote counter atomically; cache result.
4. Async background: every 100ms, push local deltas to remote and pull authoritative count.

```
Local fast-path decision tree
─────────────────────────────────────────────────────────────
                 ┌─ local_count < limit × 0.2 ─→ ALLOW (local only)
request ─ rule ──┤
                 ├─ local_count < limit × 0.8 ─→ ALLOW + async push
                 ├─ local_count near limit    ─→ REMOTE check (auth)
                 └─ remote DENY               ─→ REJECT (cache 100ms)
─────────────────────────────────────────────────────────────
```

The safety margin matters. With 1000 gateway instances per region and a 5× safety margin (allow locally only when `local_count < limit/5`), worst-case over-allowance is bounded: each instance might independently allow up to `limit/5` without consulting remote, but a user's traffic doesn't fan out uniformly — it lands on one or two instances behind a load balancer with sticky-ish hashing. Real over-allowance is well under 2× even at peak.

**Hit-rate math**: the typical user is far from their limit (uses ~5% of allotment). They hit the "obviously allowed" branch. The cache only misses on (a) cold keys, (b) keys near limit, (c) keys with high churn across instances. Empirically (Stripe's published numbers, Cloudflare's blog) this rate is 90–97%. Target 95%.

```
Effective latency with 95% local hit rate:
  p50: 0.95 × 0.05 ms + 0.05 × 1.4 ms ≈ 0.12 ms
  p99: dominated by miss path = 1.4 ms (well under 5 ms budget)
  p99.9: cold cache + slow shard = 5–8 ms (rare; tail mitigated
         via hedged second request after 3 ms)
```

**[STAFF SIGNAL: invariant-based thinking]** Invariant: a local "ALLOW" decision must be recoverable into the global count within one sync window (100ms). If a local instance crashes before sync, over-allowance is bounded by what one instance had buffered (~100ms × per-instance QPS for that key). For a hot key on one instance, ~100 requests; for cold keys, ~1. Both acceptable.

The fast-path is what makes single-digit-ms rate limiting feasible at 5M req/sec. Without it, every request hits remote, the counter store needs ~4× capacity, and p99 latency doubles.

## 7. Hot-Key Detection and Mitigation

**[STAFF SIGNAL: hot-key as central]** Hot keys are not a footnote. They are the operational test of this design.

**Detection**: per-shard Count-Min Sketch (4 hash functions × 1M counters per shard, ~16 MB/shard) tracking key→QPS estimates. Every 1s, each shard reports its top-100 heavy hitters to a regional aggregator. The aggregator reconciles into a global hot-key list (top-1000 keys per region) and pushes to gateway SDKs. Latency from "key becomes hot" to "mitigation activated": ~3s.

**Activation thresholds**:

```
Key QPS           Action
─────────────────────────────────────────────────────────────
< 1,000           Standard path
1,000 – 10,000    Local fast-path with tighter sync (50ms)
10,000 – 50,000   Key splitting (N=8)
> 50,000          Key splitting (N=32) + local-only enforcement
> 200,000         Edge-cached deny decisions (1s TTL)
─────────────────────────────────────────────────────────────
```

**Mitigation 1 — Key splitting**:

```
Standard:    user:42 ──▶ shard_h(user:42) = shard_7   (single shard)

Split N=8:   user:42 ──▶ random suffix [0..7]
                  │
                  ├─ user:42:0 ─▶ shard_3
                  ├─ user:42:1 ─▶ shard_19
                  ├─ user:42:2 ─▶ shard_22
                  ├─ user:42:3 ─▶ shard_5
                  ├─ user:42:4 ─▶ shard_11
                  ├─ user:42:5 ─▶ shard_28
                  ├─ user:42:6 ─▶ shard_8
                  └─ user:42:7 ─▶ shard_14

  Per sub-key TAT enforces limit/N
  Aggregate enforcement: sum(sub-counters) ≤ limit
                         (verified async; each sub-shard independent)
```

Each sub-key has its own GCRA TAT enforcing `limit/N`. A request hashes uniformly to one sub-key. Per-sub-shard load drops to `total_QPS / N`, bringing a 50K req/sec key down to ~1.5K req/sec per shard at N=32.

**Mitigation 2 — Local-only enforcement for the very hottest**: at >50K req/sec, even the round-trip to a split shard wastes budget. Each gateway instance enforces a local sub-limit (`total_limit / num_instances × 1.5`). Sync globally every 1s. Over-allowance bound: `1.5 × total_limit` for ≤1s. For comfort limits, acceptable.

**Mitigation 3 — Request coalescing at the shard**: when N requests arrive at the same shard for the same key within a 100µs window, the shard runs the GCRA check once and applies the decision to all N (TAT advanced accordingly). Reduces lock contention from N×check to 1×check.

**Mitigation 4 — Edge-cached deny for emergencies**: at >200K req/sec (viral content scenario), even split shards strain. The gateway SDK caches DENY decisions for hot keys with 1s TTL. Most requests for that key never leave the gateway. We accept 1s of stale-deny propagation.

**[STAFF SIGNAL: blast radius reasoning]** Without mitigation, one viral key melts a shard, cascading to all other keys on it. With detection + splitting + local-only, the blast radius is contained to the originating customer (their requests are served at degraded accuracy; other customers on those shards are unaffected). Quantified: at 50K req/sec with N=32, per-shard added load is ~1.5K ops/sec — well within shard capacity (~200K ops/sec).

## 8. Multi-Dimensional Limit Evaluation

**[STAFF SIGNAL: layered-limit evaluation]** A single request triggers checks against: per-user, per-API-key, per-customer-org, per-endpoint. Sequential evaluation = 4 × hot-path latency. Unacceptable.

**Strategy**: parallel dispatch + short-circuit on local fast-path + composite keys for the most common combinations.

```
Request arrives
  │
  ▼
┌─────────────────────────────────────────────────────────────┐
│ Resolve rules from local rule cache:                        │
│   user:42        → comfort, 100 rps                         │
│   apikey:abc     → comfort, 1000 rps                        │
│   customer:7     → comfort, 10000 rps                       │
│   endpoint:/v1/x → comfort, 50000 rps (per-region)          │
└─────────────────────────────┬───────────────────────────────┘
                              │
       ┌──────────────────────┼─────────────────────────────┐
       │                      │                             │
       ▼                      ▼                             ▼
┌────────────┐         ┌────────────┐               ┌────────────────┐
│ Local fp   │         │ Local fp   │      ...      │ Local fp       │
│ user:42    │         │ apikey:abc │               │ endpoint:/v1/x │
└─────┬──────┘         └─────┬──────┘               └────────┬───────┘
      │                      │                               │
      ▼                      ▼                               ▼
   ALLOW?                 ALLOW?                          ALLOW?
      │                      │                               │
      └─────────── AND ──────┴───────────────────────────────┘
                              │
                  ┌───────────┴────────────┐
                  ▼                        ▼
              ALL ALLOW                ANY DENY
              proceed                  return 429
                                       (X-RateLimit-Denied-By: <dim>)

Latency: max(check_i), not Σ.
```

**Short-circuit on local cache**: any local "obviously allowed" decision is instant. We only fall through to remote for limits near threshold. In practice ~90% of requests have all 4 limits in "obviously allowed" state — total local-path latency for the whole evaluation is ~0.2ms.

**Composite keys**: for the most common pair (user × api_key), collapse to a single counter `user:42|apikey:abc` enforcing the tighter of the two limits. Memory: O(pairs) instead of O(users + keys). Worthwhile for the top-1000 most-active pairs; not for the long tail.

**[STAFF SIGNAL: capacity math]** Per-request remote ops with 95% local hit per dimension: `4 × 0.05 = 0.2`. At 5M req/sec, aggregate remote counter ops/sec = 1M, not 20M. The local fast-path is what makes this affordable.

**Denial attribution**: the response identifies which dimension denied (`X-RateLimit-Denied-By: user`). Without this, customers can't diagnose 429s and retry blindly.

## 9. Cross-Region Semantics

**[STAFF SIGNAL: global-rate-limit honesty]** Most "global rate limits" are not literally global. Synchronous cross-region quorum on every check costs 30–80ms RTT — incompatible with a 5ms latency budget. What we offer is **best-effort global with bounded over-allowance**.

```
Region us-east           Region us-west          Region eu-west
┌─────────────┐         ┌─────────────┐         ┌─────────────┐
│ Counter     │         │ Counter     │         │ Counter     │
│ Store       │         │ Store       │         │ Store       │
└──────┬──────┘         └──────┬──────┘         └──────┬──────┘
       │                       │                       │
       └───────────┬───────────┴───────────┬───────────┘
                   │  async delta gossip   │
                   │  ~200 ms cadence      │
                   ▼                       ▼
   CRDT-style additive sum across regions
   each region knows: own_count + last-known peer_counts

Hot-path enforcement:
  effective_count = own_regional + Σ(last_known_peer)

Over-allowance bound during sync lag (200ms) for limit L rps:
  worst case = L × gossip_lag_s × (regions − 1)
             = 100 × 0.2 × 2 = 40 requests on a 100-rps limit
  bounded; self-heals at next window boundary
```

**Three regimes**, selected per limit:

1. **Best-effort global (default for comfort limits)**: gossip-based eventual consistency. Over-allowance bounded by gossip lag; self-heals at window. ~95% of rules.

2. **Regional sub-limits**: assign `limit / 3` to each region. User can use up to `limit/3` per region simultaneously, totaling `limit`. No over-allowance possible; slight under-allowance if one region is unused. Useful when over-allowance is unacceptable but synchronous is too slow.

3. **Strict global (≤1% of rules)**: synchronous owner — a single regional shard owns the key; all other regions forward to it. Latency cost: 30–80ms per check. Reserved for trial signups (must be globally unique), hard cost ceilings, security-sensitive limits.

**[STAFF SIGNAL: failure mode precision]** **The malicious cross-region user**: an attacker parallelizes requests across all 3 regions to exploit gossip lag. With best-effort global, they achieve up to 3× the limit during sync lag — 300 req/sec on a 100 req/sec limit for ≤200ms. We document this bound. For comfort limits, acceptable. For security limits (login attempts), we use strict global despite the latency cost — the security requirement justifies it.

**[STAFF SIGNAL: invariant-based thinking]** Invariant: cross-region over-allowance is bounded by `limit × gossip_lag × (regions − 1)` and self-heals at the next window boundary. Gossip lag is a SLI; we alarm at >500ms.

## 10. Fail-Policy Per Limit Type

**[STAFF SIGNAL: fail-policy by limit type]** "Fail-open or fail-closed" is not a system-wide policy. It is per-limit-type.

```
Limit type            Failure mode            Rationale
────────────────────────────────────────────────────────────────────
Comfort (per-user     Fail-open               Brief over-allowance
fairness, per-key                             self-heals; worse to
RPS)                                          break customer's API.

Security (login,      Fail-closed             Brute-force window opens
password reset,                               during outage; security
2FA verify)                                   incident outweighs avail.

Cost-protection       Tier-aware:             Free-tier abuse during
(LLM tokens,          fail-closed for free,   outage costs real $;
expensive endpoints)  fail-open for paid      enterprise has SLA.

Strict global         Fail-closed             Strict-global limits
(signup,                                      exist *because of* the
billing-adjacent)                             guarantee; failing open
                                              violates the contract.
────────────────────────────────────────────────────────────────────
```

**Layered fallback** for comfort limits (the common case):

1. Global counter reachable → enforce global (best-effort).
2. Global unreachable, regional reachable → enforce regional (state="degraded"; alarm).
3. Regional unreachable, local cache populated → enforce local-only (state="local-only"; over-allowance bounded by per-instance traffic).
4. All unreachable → fail-open with priority-class consideration (security limits and a coarse last-resort filter still apply).

Each transition has bounded timeout (50ms, 20ms, 5ms). Customers see degradation, not outage.

**[STAFF SIGNAL: blast radius reasoning]** Counter-store outage blast radius: the limiter does not become an availability SPOF for the API because of the fallback chain. A complete counter-store loss degrades enforcement quality (over-allowance rises) but does not stop traffic. Without this, the limiter would be a worse availability risk than the API itself.

## 11. Multi-Tenant Fairness

The threat: customer X sends 80% of traffic. Even within their own rate limit, they consume worker capacity, queueing customer Y.

**Mechanisms**:

1. **Per-customer concurrent-request limits**: orthogonal to rate limits. "Customer X may have at most 1000 in-flight requests." Prevents one customer from monopolizing the worker pool. Enforced at the gateway via per-customer semaphore.

2. **Per-customer worker reservations**: under contention, each customer's tier guarantees a minimum worker share. Free shares one pool; pro/enterprise have reserved capacity. Implemented via priority queues at the worker level.

3. **Adaptive admission control**: when system metric (worker queue depth, p99 latency) crosses threshold, *temporarily reduce* limits for high-volume customers. Free reduced first, then pro, then enterprise. Reduction is reversed when load subsides. Distinct from rate limiting — this is a system-protection layer.

4. **Free-tier abuse detection**: a free customer hits their limit; signs up for 100 free accounts; striped traffic. Signal: per-IP, per-payment-method, per-fingerprint counters in addition to per-account. Flag for review when one IP correlates with many accounts under heavy load.

**[STAFF SIGNAL: noisy-neighbor protection]** These are not the rate limiter. They complement it. The rate limiter alone cannot solve noisy-neighbor because rate limits are per-key — a single customer at 10K req/sec under their 10K limit is "compliant" but still drowning others. Concurrent limits and adaptive admission are the actual mitigation.

## 12. Rule Distribution

Rules change. New tier rolls out; emergency tightening during incident; per-customer custom limit. Changes must propagate to all enforcement points within 5s.

```
Rule write path:

Operator ─▶ Rule API ─▶ Postgres (source of truth)
                          │
                          ├─▶ Pub/Sub (push: ~1s propagation)
                          │     └─▶ Gateway SDK rule cache
                          │
                          └─▶ Periodic snapshot (S3, every 60s)
                                └─▶ SDK pull on startup +
                                    5-min reconciliation poll
```

Push for speed; pull for reliability. If pub/sub lags, the 5-min reconcile catches stragglers. Staleness SLO: 5s p99, 60s p999.

**Emergency override**: a separate path with higher pub/sub priority and 1s SLO. Used for incident response (tighten a customer's limit immediately; block a runaway script). Audit-logged; auto-expires after 1 hour unless renewed.

**Guardrails**: a rule deploy that would block >5% of recent traffic (tested in shadow mode before commit) requires explicit operator confirmation. >50% requires multi-party approval. Prevents the "set customer limit to 0 by typo" failure mode.

**[STAFF SIGNAL: rule-distribution discipline]** In-flight rule change race: a request mid-evaluation when rules update reads whichever version is local. Rule version is captured in the decision log for support debugging.

## 13. Failure Modes

**[STAFF SIGNAL: failure mode precision]**

```
Failure                     Detection             Response
──────────────────────────────────────────────────────────────────────
Counter store partition     Heartbeat fail        Fall to regional-only;
(region isolated)           >2s                   alarm; over-allowance
                                                  bounded by single-region.

Counter store down          Connection failures   Fall through chain
(regional)                                        (§10); alarm.

Hot-key flood               QPS sketch threshold  Auto-activate splitting
                            crossed               + local-only enforcement.

Rule misconfig blocking     Per-customer 429      Auto-rollback if 429
all customer traffic        rate >10× baseline    rate spikes >10× within
                            for >60s              5min of rule change;
                                                  page on-call.

Limiter SDK overloaded      In-process queue      Shed checks in priority
                            depth                 order (security >
                                                  cost > comfort); brief
                                                  fail-open on comfort.

Cross-region gossip         Lag SLI               Surface to dashboards;
delayed                                           tighten local-only
                                                  thresholds dynamically.

Botnet overwhelming         Aggregate rate        Not the rate limiter's
the limiter itself          >2× capacity          job — DDoS layer
                                                  upstream handles this.

Cache poisoning (corrupt    Sanity bounds on      Cache TTL ≤1s reduces
local fast-path entry)      cached values         blast radius; full purge
                                                  on detection.
──────────────────────────────────────────────────────────────────────
```

**The malicious-user case**: a customer generates 1M req/sec from a botnet. Pure rate limiting can't help — the limiter needs known customer identity to enforce. DDoS protection upstream (L3/L4 + L7 with bot detection) is the right layer. The rate limiter is the *next* layer after DDoS, not a DDoS mitigation itself.

## 14. Customer-Facing Surface

**[STAFF SIGNAL: customer-facing surface]**

**Response headers** on every response:

```
X-RateLimit-Limit:       100
X-RateLimit-Remaining:   42
X-RateLimit-Reset:       1715000000     (epoch s)
X-RateLimit-Policy:      "100;w=1"      (RFC 9331 draft format)
X-RateLimit-Denied-By:   user           (only on 429)
Retry-After:             1              (only on 429)
```

Not optional. Without them, well-behaved clients can't back off correctly; poorly-behaved clients hammer in retry loops, amplifying the very problem the limiter solves.

**Decision log** (sampled at 0.1%, full for 429s): every decision logged with `(timestamp, key, rule_version, counter_value, decision, region)`. Indexed for support: when a customer asks "why was I rate limited at 14:32:17?", we answer with the actual counter value and rule version.

**Adaptive surfacing**: when a customer is consistently >80% of their limit for a sustained period, surface this in the dashboard with an upgrade recommendation. Reduces support tickets *and* moves customers up tiers.

**Dry-run mode** for new rules: deploy a rule in observe-only; emit "would-have-denied" metrics without actually denying. Operator validates impact before flipping to enforce. The right antidote to "the rule misconfig blocked all our customers."

## 15. Tradeoffs Taken — and What Would Change Them

**[STAFF SIGNAL: contrast with cumulative-budget framing]** This entire design assumes self-healing rate limits. **For cumulative budgets, the architecture inverts on nearly every axis:**

- Best-effort global → strict global with per-region leases (sub-budget allocation per region from a master quota).
- Local fast-path with bounded over-allowance → unacceptable; every spend is permanent. Local cache becomes write-through only, with synchronous lease checkout.
- Fail-open on counter-store outage → unacceptable; a budget must reject during outage (or hold in a durable queue).
- Probabilistic counters / sketches → unacceptable; need exact counts for billing-grade audit.
- Eventual-consistency CRDT → replaced by CP-side store (Spanner/CockroachDB) with synchronous quorum.
- 5s rule propagation → fine for rate, but budget changes (especially tier-up mid-month) need transactional semantics with the customer's actual spend at switchover.

The siblings converge on observability and customer surface; they diverge on every consistency, accuracy, and failure-mode decision. Building one architecture that "handles both" produces a system that's overkill for rate limits and inadequate for budgets.

## 16. What I'd Push Back On

**[STAFF SIGNAL: saying no]**

1. **"Global rate limiter" framing**: misleading. We provide best-effort global with bounded over-allowance, plus strict global as opt-in for the few rules that justify the latency cost. Pretending we offer literal global enforcement on all rules sets customer expectations we can't keep.

2. **"5M req/sec across 3 regions" without specifying limit cardinality**: a workload of "100 limits × 50K customers" is a different problem from "10M limits × 0.5 customer/limit." The former has high hot-key concentration; the latter has a long tail. Sharding strategy and hot-key activation thresholds pivot on this.

3. **Implicit assumption that one rate limiter design serves all limit types**: comfort, security, and cost-protection limits have meaningfully different policies (consistency, fail mode, latency tolerance). The system is policy-driven, not monolithic. A rule's `policy_class` determines which path it takes.

4. **Implicit assumption that the rate limiter is the API's only protection layer**: it isn't. Upstream DDoS (Cloudflare/Fastly/L3-L4 scrubbing) handles volumetric attacks; downstream adaptive admission control protects against in-budget-but-still-overwhelming traffic. The rate limiter sits between these. Designing it as a load shield in isolation produces an overbuilt system that solves problems other layers solve better.