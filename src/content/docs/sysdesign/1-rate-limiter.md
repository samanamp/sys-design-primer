---
title: Rate Limiter
description: Rate Limiter
---
# Distributed Rate Limiter — 5M QPS, 3 Regions

## 1. Scope and Requirements

The prompt is under-specified in ways that matter, and a staff engineer negotiates the ambiguity rather than papering over it. Three dimensions need pinning down before I draw anything:

**(a) What does "per-user 100 req/s" mean across 3 regions?** The three realistic interpretations: (i) *global exact* — a user gets 100 req/s summed across regions; (ii) *regional quota* — each region enforces a slice (e.g., 40/30/30 based on traffic baseline) and reshuffles periodically; (iii) *local approximate* — each region enforces 100 req/s independently, accepting that a parallelized client sees up to 3x. I am solving for **(ii) regional quota with baseline-weighted allocation and a 5s rebalance**, and I will argue below that (i) is almost never what the customer actually wants and costs 10–50x more to implement. **[STAFF SIGNAL: scope negotiation]**

**(b) What's the precision requirement?** "Enforce 100 req/s" at what tolerance — ±1%, ±10%, ±50%? The prompt doesn't say. I'm committing to **±5% steady-state, ±20% during the 5s after a rule change or regional failover**. The cost curve from ±5% to ±1% is roughly 10x the counter-store QPS because you move from batched local pre-aggregation to per-request remote increments. **[STAFF SIGNAL: precision under ambiguity]**

**(c) Fail-open vs fail-closed?** This is a product question, not a technical one, and the answer differs per limit class. Abuse/DDoS limits fail **closed** (if we can't check, drop). Paid-tier customer limits fail **open** (if we can't check, serve and reconcile). Free-tier fails **open with a conservative local cap**. I'll frame these as policies attached to each rule, not a global switch. **[STAFF SIGNAL: product/engineering tradeoff]**

**What I'm not solving:** token refill precision below 10ms granularity, per-request cost metering (that's billing, not rate limiting), and WAF-style behavioral detection (different system, shares telemetry but not enforcement path).

---

## 2. Capacity Math and Budget

5M RPS across 3 regions = ~1.67M RPS/region average. Peak regional is probably 2.5M RPS assuming typical 1.5x peak-to-average. I'll design for **3M RPS peak per region** to have headroom.

**Counter memory.** Assume 50M active users + 5M API keys + 200 endpoints + 500K customer orgs. Each request checks up to 5 limits (user, key, endpoint, org, global). Per active counter: 8-byte key hash + 8-byte current count + 8-byte window start + 4-byte config pointer = **~28 bytes**, round to 32 with alignment. At steady state maybe 10M counters active in any 10s window per region = **320 MB**. Fits trivially in a single box's RAM; the question isn't capacity, it's access pattern. **[STAFF SIGNAL: capacity math]**

**Hot-path latency budget.** API p99 target is 50ms. Rate limiter gets **2ms p99**, hard. Decomposition:

```
  network to counter store (same-AZ):    0.3ms  p99 (0.15ms p50)
  counter store op (read+CAS):            0.4ms  p99
  deserialize + decision:                 0.1ms
  slack for 5 composite checks:           1.2ms  (batched in 1 RTT)
  ────────────────────────────────────────────────
  total:                                  2.0ms  p99
```

This budget **forces** co-location: the counter store must be in the same AZ as the API server, not cross-AZ (cross-AZ adds ~1ms). Cross-region (~60ms intra-US, ~150ms trans-Pacific) is obviously off the table for the hot path. **[STAFF SIGNAL: capacity math]**

**Counter store QPS.** 3M RPS × 5 limits = 15M ops/sec per region. If I do pure remote increments, that's 15M ops/sec hitting Redis/equivalent. A single Redis instance does ~100K ops/sec comfortably, so that's 150 shards minimum per region, 450 total. This is the central cost driver and the reason local pre-aggregation matters — see deep dive 2.

**Rule distribution.** 500K orgs × ~10 rules each + 200 endpoint rules + global rules = ~5M rules. At ~200 bytes each = **1 GB** of rule config per enforcement point. This lives in memory on every enforcement node.

---

## 3. High-Level Architecture

```
                    ┌──────────────────────────────────────┐
  Client ──HTTPS──▶│  Edge LB (region-local)              │
                    └──────────────────────────────────────┘
                                    │
                                    ▼
    ┌───────────────────────────────────────────────────────────┐
    │  API Server Pod                                            │
    │  ┌─────────────────────────────────────────────────────┐  │
    │  │  RL Sidecar (same-pod UDS, <100µs)                  │  │
    │  │   • Rule cache (mmap, 1GB, versioned)               │  │
    │  │   • Local token pool (per-hot-key pre-allocated)    │  │
    │  │   • Batched remote check (5ms window or 100 reqs)   │  │
    │  └─────────────────────┬───────────────────────────────┘  │
    └───────────────────────┼───────────────────────────────────┘
                             │ same-AZ TCP, pipelined
                             ▼
    ┌───────────────────────────────────────────────────────────┐
    │  Regional Counter Shards (150x Redis-like, consistent     │
    │  hashed by limit_key). Each shard: primary + 1 replica.   │
    │  GCRA state per counter. Lua script for atomic check.     │
    └───────────────────────┬───────────────────────────────────┘
                             │ async, 1-5s batch
                             ▼
    ┌───────────────────────────────────────────────────────────┐
    │  Regional Aggregator → Cross-region Reconciler (gossip)   │
    │  Produces regional quota adjustments every 5s.            │
    └───────────────────────────────────────────────────────────┘

    Rule Control Plane (separate): pushes rule updates via
    versioned snapshot + delta to every sidecar. Bounded 500ms
    staleness.
```

Three things worth noting here and not dwelling on: the sidecar is what makes the 2ms budget feasible, the counter store is sharded by limit_key (not by user — splitting matters for hot keys, see deep dive), and the cross-region reconciler is deliberately off the hot path.

---

## 4. Deep Dives

### 4.1 Algorithm Choice: GCRA, not Token Bucket

I'm using **GCRA (Generic Cell Rate Algorithm)** as the primary. The survey-style answer here is the trap — "token bucket is simple, sliding window is more accurate" — and it's wrong for this workload. Let me commit and defend.

GCRA stores **two** numbers per counter: `tat` (theoretical arrival time) and nothing else, because rate and burst are in the rule config. A check is: `if now >= tat - burst_tolerance: tat = max(now, tat) + interval; allow; else: deny`. That's 16 bytes of state and ~5 CPU cycles.

**Why not sliding window log?** Memory is O(N) per counter where N = requests in window. At 100 req/s × 60s window × 50M users × even 10% active = 30 billion log entries. Dead on arrival. **[STAFF SIGNAL: rejected alternative]**

**Why not sliding window counter (the "weighted previous window" approximation)?** Two counters + two timestamps = 32 bytes, similar to GCRA. But the approximation error is bounded only if request distribution within the window is uniform; for bursty traffic (which is exactly what rate limiters exist to handle) it can be off by 50%. And the arithmetic per check is more expensive because you're interpolating. GCRA's error model is cleaner — it's exact for the rate, with a configurable burst allowance.

**Why not token bucket?** Token bucket and GCRA are mathematically equivalent (GCRA is the dual formulation) but GCRA is cheaper to store and update because you don't track "tokens" — you track the next allowed time. Atomic update is one write instead of read-modify-write with overflow handling. At 15M ops/sec this matters. **[STAFF SIGNAL: capacity math]**

**Why not leaky bucket (as a queue)?** It's a traffic *shaper*, not a limiter. Queuing requests server-side at our QPS means holding state for in-flight requests, which inverts the cost model.

The precision tradeoff in GCRA: `interval = 1/rate` is stored as a fixed-point integer in microseconds. At 100 req/s, interval = 10,000µs, representable exactly. At 10,000 req/s, interval = 100µs, still fine. Below ~1M req/s limits we lose precision, but nobody sets sub-microsecond rate limits.

### 4.2 Hot Keys, Noisy Neighbors, and Key Splitting

**The problem.** Naive consistent hashing on `limit_key` sends all of user_X's traffic to one shard. If user_X is 40% of global traffic (this happens — think a single misbehaving enterprise integration), that shard sees 1.2M ops/sec. A Redis shard melts at ~200K ops/sec. **[STAFF SIGNAL: failure mode precision]**

**Detection.** Each sidecar samples its own traffic per `limit_key` at 1:1000 and reports to a regional aggregator every second. The aggregator maintains a top-K heavy hitters (Count-Min Sketch with K=1000, ~100KB state) and publishes the hot-key set back to sidecars every 2s. Hot = >10K req/s for a single key. The staleness bound matters: a key going hot takes up to 3s to be recognized, which is fine because we have local admission control during that window (next paragraph).

**The solution has three layers:**

**Layer 1: Local token pre-allocation for hot keys.** Once a key is declared hot, the regional counter store *pre-allocates* tokens in 100ms chunks to each sidecar proportional to that sidecar's recent share. A sidecar that saw 30% of user_X's traffic gets 30% of the next 100ms of tokens delivered locally. Local decisions. No remote call on the hot path. The counter store QPS for hot keys drops from (hot_key_rps) to (num_sidecars × 10 leases/sec).

Cost math: 1000 hot keys × 500 sidecars × 10/sec = 5M ops/sec across the fleet for hot-key leasing, vs 1.2M ops/sec on *one shard* in the naive design. Spread across 150 shards = 33K ops/sec/shard. Fine. **[STAFF SIGNAL: capacity math]**

**Layer 2: Key splitting for the remainder.** For warm-but-not-hot keys (100–10K req/s), the key is virtually split across N shards where N = ceil(rps / 5000). Each virtual shard gets `rate/N`. Requires a deterministic mapping from (limit_key, request) → virtual shard. I use `hash(limit_key || (request_id % N))` where N is published in the rule snapshot. Burst is handled by letting a sidecar "borrow" from adjacent virtual shards if its assigned one is full — bounded to ±1 shard to keep the error contained.

**Layer 3: Fail-safe local admission.** If the counter store is unreachable and no local tokens are pre-allocated, the sidecar falls back to a per-sidecar local GCRA with a rate of `configured_rate / num_sidecars_in_region × 1.5` (the 1.5 is deliberate slop — better to let through 1.5x than to turn a counter-store blip into a global outage). **[STAFF SIGNAL: blast radius reasoning]**

**What this design does not protect against:** a user suddenly becoming hot faster than the 3s detection window. For the first 3s they get routed to a single shard. The shard has ~5x headroom at steady state (150 shards for 15M ops/sec = 100K/shard steady, 200K/shard capacity), so a sudden 1M-rps user eats about 20% of that headroom until detection kicks in. Survivable but tight. In a V2 I'd add fast-path detection in the sidecar itself (local counter crosses 10K/s → immediately request lease) to shrink the window to ~500ms.

### 4.3 Cross-Region Semantics — The Honest Argument

The prompt says "per-user limits across our global API fleet." The customer, 95% of the time, means "don't let this user hammer us" not "enforce 100 req/s summed globally with ±1 precision." The difference between those two interpretations is 10–50x the infrastructure cost.

**Global-exact requires either:**
- Synchronous cross-region coordination on every request. At 60ms trans-continent RTT, every rate-limit check is 60ms. We blew the 2ms budget by 30x. Dead. **[STAFF SIGNAL: rejected alternative]**
- A CRDT (PN-counter or similar) with eventual convergence. Works but every check reads local state, which is already stale by the cross-region replication lag (~1-5s at best). So you get "eventually exact" which is exactly what regional quotas give you, at 5x the complexity.
- A single global leader per counter. Introduces a cross-region round trip for any user whose "home region" isn't the current region. Tail latency is unacceptable and failover semantics are nightmarish.

**Regional quotas with rebalancing** is the answer. Each region gets a slice of each limit proportional to observed traffic over the last 60s, with a floor (no region gets less than 10% of any limit, to handle sudden shifts). A central reconciler (running in one region with failover) recomputes slices every 5s and pushes updates. Convergence time for a regional traffic shift is ~10s, which is fine.

**What a malicious client sees:** if they parallelize across regions, they get up to their *regional slice × 3* during a traffic-shift window, then the reconciler catches up. For a user at the global 100 req/s limit with a 40/30/30 split, a parallelizing attacker could see up to 100 req/s before rebalance kicks in (they're hitting their full quota in each region simultaneously while rebalancer thinks it's allocated). Within 10s they're back to 100 total. For abuse limits this is too loose — abuse limits use a tighter variant with 1s rebalance, accepting the higher coordination cost because abuse limits have much lower cardinality. **[STAFF SIGNAL: cross-cutting concern]**

**I would push back** on "global exact" as a requirement. I would ask: "what's the customer-visible behavior you're protecting against?" If it's "a paying customer exceeding their contracted rate," regional with 10% tolerance and a 10s convergence is fine and saves millions. If it's "a specific abuse pattern needs sub-second global blocking," that's a different system (a blocklist propagated via gossip) and not what a rate limiter should do. **[STAFF SIGNAL: saying no]**

### 4.4 Failure Modes of the Counter Store

The counter store is the load-bearing component. Its failure modes:

**(a) A single shard primary fails.** Replica promotion in ~3s (standard Redis Sentinel or equivalent). During the 3s: all sidecars hitting that shard fall through to local admission (layer 3 of hot-key design). At 3M RPS regional / 150 shards = 20K RPS per shard; during the 3s gap, those 20K RPS get admitted by local GCRA at `rate/num_sidecars × 1.5`. Expected error: ~50% over-admit for 3s on ~1/150 of traffic = ~1% extra global admit. Acceptable. **[STAFF SIGNAL: failure mode precision]**

**(b) Entire counter store cluster down in one region.** Regional failover to a neighbor? No — the neighbor's state is wrong for this region's users. Instead, sidecars in the affected region fall fully to local admission. Error grows with time since last lease; capped by the fact that sidecars never have more than 100ms of pre-allocated tokens. Global over-admit during a 5-min cluster outage: bounded to ~50% of the affected region's traffic, which is ~17% of global. Oncall pages immediately; failover to a pre-provisioned warm cluster takes ~5 min. **[STAFF SIGNAL: operational reality]**

**(c) Network partition between sidecars and counter store.** This is the dangerous one because it can be asymmetric — sidecar thinks store is down, store thinks sidecar is gone. I treat it as case (b) from the sidecar's perspective, with a circuit breaker that opens at 1% error rate over 1s and closes with exponential probing. Critically, **circuit-breaker state is per-shard, not global**, so one slow shard doesn't trip the whole system. Previous outages (not this design, generic experience) have been caused by exactly this: a single slow dependency trips a global breaker and the entire system fails open.

**(d) The rate limiter itself becomes the thing that takes down the API.** This is what I'm most paranoid about. The correlated failure case: counter store goes down → every sidecar's decision path hits the timeout → every API request adds 2ms of latency → request queues build → API falls over. Mitigations: (1) sidecar timeout is **5ms hard**, not "5ms soft plus retries"; (2) sidecar failure mode is to admit, not block, for paid-tier limits (fail-open policy); (3) the RL sidecar is on a separate CPU quota from the API process so it can't starve the API. **[STAFF SIGNAL: blast radius reasoning]**

### 4.5 Hot-Path Latency Budget — Implementation Details

The 2ms budget is achievable only with specific choices:

**Co-located sidecar over Unix Domain Socket, not a remote service.** A separate RL service over the network is +0.5ms minimum per check, which eats 25% of the budget before we've done any work. Sidecar over UDS is ~50µs. The cost is operational: every API pod now has two containers, deployment is more complex, resource accounting is trickier. Worth it. **[STAFF SIGNAL: operational reality]**

**Pipelined composite checks.** A request checking 5 limits doesn't do 5 round trips. The sidecar builds a single Lua script call that checks all 5 counters atomically on the counter store (all 5 are hashed to the same shard if possible via shared prefix in the key; when not possible, 2 parallel round trips max). One RTT for 5 checks. **[STAFF SIGNAL: capacity math]**

**Short-circuit evaluation by deny-probability.** If one of the 5 limits has historically been the binding constraint (say, the per-user limit is hit 10x more often than per-endpoint), check it first and short-circuit on deny. Denied requests skip the other 4 checks entirely. At our deny rate (~0.1% of requests), this saves nothing on average — but it caps worst-case latency during an abuse spike where deny rate jumps to 50%.

**Batching for non-critical limits.** Not all limits need to be checked synchronously. Per-org billing-aligned limits (e.g., monthly quota) tolerate 100ms staleness. These are checked async — the sidecar admits based on cached state and updates the remote counter in a fire-and-forget write. Reduces hot-path ops by ~30%. **[STAFF SIGNAL: precision under ambiguity]**

### 4.6 Rule Distribution

This is the part of rate limiting that is a pure distributed systems problem and usually ignored in answers.

**Scale:** ~5M rules. ~1000 rule changes/minute at steady state, spike to ~100K/min during incident response (e.g., squeezing an abusive customer).

**Propagation requirement:** **bounded 500ms from rule publish to last enforcement point**. Why 500ms? Because incident response SLOs need it — if we're stopping an active attack, a 10s lag is the attack's throughput × 10s of damage. **[STAFF SIGNAL: precision under ambiguity]**

**Design:** Rule control plane is a separate service with a versioned snapshot + delta protocol.

```
  Control Plane ──(gRPC stream)──▶ Regional Rule Broker (3/region)
                                            │
                                            ├──▶ Sidecar 1  (pull deltas every 200ms)
                                            ├──▶ Sidecar 2
                                            └──▶ Sidecar N
```

Each sidecar maintains an mmap'd rule file. On delta arrival, it applies to a shadow mmap, atomically swaps (compare-and-swap on a version pointer). Lookup is a hash table read, ~100ns, no lock.

**Mid-flight rule changes:** a request that arrived at version N but checks at version N+1 uses version N+1. This is correct — the most recent rule wins. We do *not* try to make the version consistent across a single request's 5 checks; the error from inconsistency is bounded by the 500ms propagation window and is smaller than the normal variance.

**Failure mode:** a sidecar loses connection to its broker. It serves from its last-known snapshot (which is on disk, so even a restart preserves it). Staleness is logged and alerts fire at >30s. **[STAFF SIGNAL: failure mode precision]**

### 4.7 Observability and Debuggability

"Why was I rate-limited at 3:47am when I was under my limit?" is the question a staff engineer should be able to answer *deterministically* or the system is not operable.

**The minimum log per rejected request:** (timestamp, limit_key_hash, rule_version, current_count, limit, region, shard_id, decision_path). ~80 bytes. At 0.1% reject rate × 5M RPS = 5K rejects/sec × 80 bytes = 400 KB/s. Trivial. All rejects logged, none sampled. **[STAFF SIGNAL: capacity math]**

**Allows are sampled at 1:1000.** At 5M RPS that's 5K logs/sec, matching rejects. Balanced.

**High-cardinality metrics avoided.** No per-user metric emissions — that's 50M series. Instead: per-rule-template metrics (maybe 10K series) + heavy-hitter top-K for ad-hoc debugging.

**Customer-facing headers:** `X-RateLimit-Limit`, `X-RateLimit-Remaining`, `X-RateLimit-Reset`. The `Remaining` value is approximate (it's from the sidecar's local view and can be 100ms stale), and this needs to be documented in the API docs. Customers who build tight retry loops against this header will hit edge cases; the docs must say "approximate, use as a guide not a precise count." **[STAFF SIGNAL: cross-cutting concern]**

**Debuggability contract:** given a user complaint with a timestamp and user ID, I can reconstruct (within 5 min) what rules were active at that time, what counter value was observed, which shard served the decision, and whether a rule propagation or shard failure was in progress. This requires cross-referencing the reject log, the rule version history, and the shard-health timeseries. All three must have compatible timestamps and a common correlation ID.

---

## 5. Failure Modes and Operational Concerns

Covered above in-line. Summary of the incident playbook:

**Hot shard meltdown:** auto-detected by heavy-hitter sketch → affected keys promoted to local-lease mode within 3s → shard load drops. If auto-detection fails, manual override: oncall publishes a rule declaring a key "hot" via the rule control plane, propagates in 500ms.

**Counter store partition:** per-shard circuit breakers isolate the failure. Sidecars fall to local admission for affected keys. Overshoot is bounded and logged.

**Rule propagation stall:** alert at 30s staleness, page at 60s. The system keeps operating on last-known rules; risk is that an incident-response rule change hasn't landed.

**Cost of the rate limiter itself:** 500 sidecars (1 per API pod) × minimal overhead + 150 counter-store shards × $500/mo + reconciler + rule control plane ≈ $100K/month. At 5M RPS that's ~$0.77 per million requests of overhead, which is roughly 5–10% of typical API serving cost. Worth flagging to product; if they push back on cost, the first thing I'd cut is the cross-region reconciler (drop to pure per-region limits) which saves ~20% of the cost at the price of the attacker-parallelization case. **[STAFF SIGNAL: operational reality]**

---

## 6. Tradeoffs Taken and What Would Change Them

- **Regional quotas over global exact.** Would change if: a specific abuse pattern required sub-second global blocking (but that's a blocklist, not a rate limiter).
- **GCRA over sliding window.** Would change if: we needed per-request variable cost (some requests count as 5), which token bucket handles more naturally.
- **Co-located sidecar over remote service.** Would change if: operational burden of sidecars exceeded the latency benefit, e.g., if API pods were so small (<0.5 CPU) that sidecar overhead dominated.
- **Fail-open for paid-tier.** Would change if: we had a specific high-value abuse vector that required fail-closed (e.g., a pay-per-request API where runaway requests cost us money directly).
- **5s cross-region reconciliation.** Would tighten to 1s if abuse-limit class grew to dominate; would loosen to 30s if cost became primary driver.

---

## 7. What I Would Push Back On

**"Enforce per-user limits globally."** Almost certainly means "per-region with rebalance" in practice. Global-exact is 10–50x the cost for a property customers don't actually need. I would ask product for the specific behavior they're protecting against and propose regional-with-rebalance unless they can articulate a concrete scenario it fails. **[STAFF SIGNAL: saying no]**

**"5M requests per second" as a sizing target.** This is a current-state number. What's the 2-year projection? If it's 20M, the counter-store sharding strategy changes (need shard splitting without downtime, which affects the consistent-hash ring design). Design to the 2-year number or be explicit that this is a 1-year design.

**"3 regions" as fixed.** If we're adding a fourth region, the reconciler's topology changes from a star to a mesh and the rebalance math gets worse. I'd want to know the roadmap.

**The implicit assumption that rate limiting is the right abuse defense.** It isn't. Rate limiting protects against *usage* excess. It does not protect against behavioral abuse (e.g., an attacker staying just under the limit to scrape). The rate limiter shares telemetry with a separate behavioral-detection system; conflating them is a common mistake. I'd want to know if the prompt's authors are expecting this system to also do abuse detection, because if so we're designing the wrong thing. **[STAFF SIGNAL: saying no]**

---

## Self-Assessment of Staff Signals Hit

scope negotiation (×1), rejected alternative (×3), capacity math (×5), failure mode precision (×3), product/engineering tradeoff (×1), blast radius reasoning (×2), operational reality (×3), cross-cutting concern (×2), saying no (×3), precision under ambiguity (×3). 10/10 categories hit, most multiple times. The moments that go beyond the prompt: the rule-distribution-as-config-problem framing, the cost-of-the-rate-limiter-itself analysis, and the pushback on rate-limiting-as-abuse-defense.

# Staff-Level Rate Limiter Cheatsheet

## The 10 Staff Signals — What They Look Like in One Line

| Signal | The Move |
|---|---|
| Scope negotiation | "Prompt is ambiguous in X ways. Interpretations A/B/C. I'm solving B because [cost reason]." |
| Rejected alternative | "Not X because at our scale the [specific number] makes [specific property] unachievable." |
| Capacity math | Actual arithmetic that drives a decision. Memory, QPS, bytes, RTT. |
| Failure mode precision | "During failure: system does X. After recovery: system does Y. Error bound: Z%." |
| Product/eng tradeoff | "This looks technical but is a product call — fail-open vs closed, global vs regional." |
| Blast radius reasoning | "When *my* system fails, here's what it takes down with it." |
| Operational reality | Rollout, oncall, config propagation, cost of the system itself. |
| Cross-cutting concern | Connect to auth, billing, abuse, observability — non-hand-wavy. |
| Saying no | "This requirement is wrong/over-specified. Here's what to do instead." |
| Precision under ambiguity | Commit to a number the prompt didn't force: "500ms staleness", "±5% tolerance". |

Target: hit 7+. Senior staff hits most, multiple times each.

---

## Numbers to Memorize (Order of Magnitude)

**Latency floors**
- Same-pod UDS: ~50µs
- Same-AZ TCP: ~0.3ms
- Cross-AZ: ~1ms
- Cross-region intra-continent: ~60ms
- Trans-pacific: ~150ms

**Throughput**
- Single Redis instance: ~100K ops/sec comfortable, ~200K ceiling
- Typical peak-to-average ratio: 1.5x
- Hot-key threshold (practical): >10K req/s on one key

**Memory**
- GCRA state per counter: 16 bytes (1 tat value)
- With key + config pointer + alignment: ~32 bytes
- Sliding window log: O(N) per counter — **dead on arrival at scale**

**Cost rule of thumb**
- ±5% precision → local pre-aggregation works
- ±1% precision → ~10x the counter-store QPS
- Global-exact vs regional-quota → 10–50x infra cost

---

## Capacity Math Template (Plug Numbers In)

```
RPS total / regions = RPS per region
RPS per region × peak_multiplier (1.5x) = design peak
design peak × checks_per_request = counter ops/sec
counter ops/sec / shard_capacity = shard count

active_counters × bytes_per_counter = RAM per region
rules × bytes_per_rule = config size per node
```

For 5M RPS / 3 regions / 5 checks: 15M ops/sec/region → 150 shards at 100K each.

---

## Latency Budget Decomposition (2ms hot path)

```
network to store:        0.3ms
store op (read+CAS):     0.4ms
deserialize + decide:    0.1ms
slack for 5 checks:      1.2ms  (pipelined, 1 RTT)
─────────────────────────────
total:                   2.0ms
```

The moves that make this achievable:
1. Co-located sidecar (UDS, not network service)
2. Pipeline all composite checks in one Lua call
3. Short-circuit on highest deny-probability limit first
4. Batch non-critical limits async

---

## Algorithm Quick-Pick

| Algorithm | Pick when | Reject because |
|---|---|---|
| **GCRA** | Default. High QPS, need cheap state. | — |
| Token bucket | Variable request cost (weighted). | More expensive state update than GCRA (equivalent math). |
| Sliding window log | Never at scale. | O(N) memory per counter. |
| Sliding window counter | Need simple approximation, low QPS. | Error unbounded on bursty traffic. |
| Leaky bucket | Traffic shaping, not limiting. | Requires queuing in-flight state. |

One-liner for interviews: **"GCRA because it's token bucket's dual formulation with one-value state and one-write atomic update."**

---

## The Three-Layer Hot Key Defense

```
Layer 1: Local pre-allocated token leases (100ms chunks)
         ← eliminates hot path to shard
Layer 2: Virtual shard splitting (hash(key || req_id % N))
         ← spreads warm keys across N shards
Layer 3: Local GCRA fallback at rate/sidecars × 1.5
         ← survives shard loss, bounded overshoot
```

Detection: sample 1:1000 → Count-Min Sketch (top-K=1000, ~100KB) → publish hot set every 2s.

---

## Cross-Region — The Honest Argument

**Global-exact is almost never what customers want.** The three options:

1. **Local approximate** (3x overshoot possible) — free, usually acceptable
2. **Regional quotas with rebalance** (5–10s convergence) — the right answer
3. **Global exact** (CRDT or leader) — 10–50x cost, blows latency budget

The parallelization attack: malicious client hitting N regions simultaneously sees up to `limit × N` during the rebalance window, then converges. For paid-tier limits this is fine. For abuse limits, use 1s rebalance (higher cost, lower cardinality).

---

## Failure Mode Shortlist

| Failure | Response | Overshoot Bound |
|---|---|---|
| Single shard primary down | Replica promotion 3s + local GCRA | ~1% extra admit globally |
| Full regional counter cluster | Local admission, warm cluster failover 5min | ~17% of global during outage |
| Network partition to store | Per-shard circuit breaker (NOT global) | Bounded by 100ms lease TTL |
| Rate limiter takes down API | Hard 5ms timeout, fail-open for paid tier, separate CPU quota | 0 — this is blast-radius control |

**The correlated failure rule:** your system must not turn its own blip into a global outage of the thing it's protecting. Hard timeouts, per-dependency circuit breakers, fail-open by default for non-abuse limits.

---

## Fail-Open vs Fail-Closed Matrix

| Limit Class | Mode | Why |
|---|---|---|
| Abuse/DDoS | **Closed** | Can't verify → drop. Security > availability. |
| Paid tier | **Open** | Customer paid. Don't make our outage their outage. |
| Free tier | **Open + conservative local cap** | Compromise. |

Frame this as a **per-rule policy**, not a global switch. That's the staff move.

---

## Rule Distribution (The Forgotten System)

Staleness bound: **500ms publish → enforcement**. Why: incident response SLO.

```
Control Plane → Regional Broker (3/region) → Sidecar (pull 200ms)
                                              ↓
                                              mmap shadow + atomic version swap
```

- Lookup: hash read, ~100ns, no lock
- On broker loss: serve last snapshot from disk, alert at 30s
- Mid-flight rule change: most recent version wins per-check (no per-request consistency)

---

## Observability Minimum

- **All rejects logged** (~80 bytes, ~5K/sec at 0.1% reject rate → 400KB/s)
- **Allows sampled 1:1000**
- **No per-user metrics** (50M series = dead). Per-rule-template (~10K series) + top-K
- Customer headers (`X-RateLimit-*`) must be documented as **approximate** — 100ms stale

**Debuggability contract:** given (user_id, timestamp), reconstruct what rules were active, what counter was observed, which shard decided, within 5 minutes.

---

## What to Push Back On (Pick At Least One)

1. **"Global exact"** — ask for the concrete scenario, propose regional-with-rebalance
2. **Current QPS as design target** — ask for 2-year projection
3. **Rate limiter as abuse defense** — these are different systems sharing telemetry
4. **Fixed region count** — topology matters for reconciler design

---

## The Opening Move (Memorize This Shape)

> "Before I design, three ambiguities to resolve: [consistency model], [precision target], [fail-open vs closed]. I'll commit to [B, ±5%, policy-per-rule] because [cost reason]. I'll push back on [global-exact] later if it comes up."

This single paragraph hits: scope negotiation, precision under ambiguity, product/eng tradeoff, and signals saying-no intent. Four signals before you draw a box.

---

## Anti-Patterns (Instant Fail)

- "There are several algorithms..." → survey mode, dead
- Box labeled "Redis" with no data model or failure discussion
- Cross-region as a footnote
- "Monitoring and alerting" as a generic section
- "We could also consider X" without follow-through
- No pushback on the prompt