---
title: "2-Distributed quota enforcement"
description: "Distributed quota enforcement"
---

"Design a system that enforces a monthly spending budget across many concurrent API consumers. Each request consumes a variable amount of the budget. Once the budget is exhausted, no further requests should succeed. Multi-region, low-latency. Walk me through it."

---

## 1. Reframing

This is not a rate limiter. **[STAFF SIGNAL: cumulative-vs-rate]** A rate limit — "100 req/sec" — is self-healing: errors at one window boundary do not propagate to the next. Local approximation works because each window is independent. A cumulative monthly budget — "spend $1000 by month-end" — accumulates errors that never reset. If five regions each leak $50 of overspend through stale local views, the customer is billed $250 over budget at month-end. There is no boundary that erases the error. That single property reshapes the architecture: the rate-limiter playbook does not transfer.

The central mechanism is **sub-budget allocation with leases [STAFF SIGNAL: sub-budget-allocation-as-central]**: divide the global budget into regional sub-budgets, enforce locally against each, have a coordinator periodically rebalance. This trades strict global correctness for low latency. The size of that trade is controlled by lease size — the central tuning knob.

Over-spend tolerance is a **policy decision, not a bug [STAFF SIGNAL: over-spend-as-policy]**. Zero over-spend in a multi-region system requires synchronous global coordination on every request — ~150ms cross-region floor, incompatible with the latency target. Realistic systems commit to a bounded over-spend budget: "we will exceed customer budget by no more than $X under steady state, $Y under partition." That bound is per-tier: prepaid/free-tier customers get tighter enforcement (over-spend is non-recoverable revenue loss); postpaid/enterprise customers get looser enforcement (over-spend is captured in the next invoice).

Variable cost per request is the third reshaping force. We don't know cost until the request finishes — naive post-hoc decrement allows unbounded overshoot at the boundary; naive worst-case decrement denies legitimate requests. The architecture must explicitly handle reserve / commit / refund.

## 2. Scoping

I am committing to: **[STAFF SIGNAL: scope negotiation]**

- **Cardinality:** 1M customers total; ~10K active in any given second. Long-tail: top 1% generate 80% of traffic.
- **QPS:** mean customer ~1 req/sec; whales ~1K req/sec; aggregate peak ~1M req/sec.
- **Cost variance:** wide. LLM-style: $0.0001 (cached short response) to $10 (long completion with tools). Median ~$0.001. Worst-case-per-request ≈ 10⁵× median.
- **Budget magnitude:** $5–$10,000/month. Free tier $5/month, hard cutoff.
- **Latency:** budget check adds <5ms p99 to request path. Synchronous global coordination (~50ms+) is incompatible.
- **Regions:** 5 (us-east, us-west, eu-west, ap-southeast, ap-northeast).
- **Settlement:** hybrid. Free/prepaid: hard cutoff, over-spend = platform loss. Postpaid enterprise: soft cutoff with grace, over-spend recoverable through invoicing.
- **Period:** monthly is the headline; the design generalizes to hierarchical (daily / hourly / per-minute) caps composed via the same mechanism. **[STAFF SIGNAL: saying no]** I'm not solving "monthly only" — the prompt's framing is too narrow.

## 3. Capacity math

**[STAFF SIGNAL: capacity math]**

```
AGGREGATE SCALE
  customers:        1e6 total, 1e4 concurrently active
  peak QPS:         1e6 req/sec aggregate
  per region (×5):  2e5 req/sec each at peak

STATE PER CUSTOMER
  hot fields:  budget_total, spent, period_start, tier, currency,
               sub_budget_remaining, lease_id, lease_expiry,
               local_spend_since_lease ≈ 200 bytes
  hot working set:  1e4 active × 200B = 2 MB / region (trivial RAM)
  cold storage:     1e6 × 200B × 5 regions = 1 GB total

AUDIT LOG VOLUME
  1e6 req/sec × ~150 B/event = 150 MB/sec
  ~13 TB/day, ~390 TB/month
  → durable append-only: Kafka tier-1 + S3 tier-2

CROSS-REGION REPLICATION
  spend deltas only, batched per-customer per 100ms
  active deltas:  1e4/sec × 5 regions = 5e4 cross-region writes/sec
  payload:        ~50 B/delta = 2.5 MB/sec aggregate
  → single replication stream per region pair handles it

LEASE ECONOMICS (the central tuning math)
  lease size L, whale spend rate R = $1/sec, 5 regions
  coordinator RTT/whale = R/L
  L=$60  → 1 RTT/min,   max overshoot ≈ $300 across regions
  L=$1   → 60 RTT/min,  max overshoot ≤ $5
  → adaptive L: large when remaining/rate > 1 hour;
    shrinks linearly to $0 (synchronous mode) at exhaustion.
```

## 4. High-level architecture

```
                          ┌─────────────────────┐
                          │  CUSTOMER CONSOLE   │
                          │ spend, alerts,      │
                          │ denial debug, API   │
                          └──────────▲──────────┘
                                     │ reads from coord + audit log
       ┌─────────────────────────────┴─────────────────────────────┐
       │             GLOBAL BUDGET COORDINATOR (Raft, sharded)     │
       │  - shards customers by hash(customer_id) → coord pod      │
       │  - canonical budget, tier, period, lease registry         │
       │  - issues / reclaims leases, runs reconciliation          │
       │  - derived state; rebuilds from audit log on cold start   │
       └─▲────────▲────────▲────────▲────────▲─────────────────────┘
         │ leases │        │        │        │   gRPC, p50 30ms x-region
   ┌─────┴───┐┌───┴────┐┌──┴────┐┌──┴────┐┌──┴────┐
   │ us-east ││ us-west││ eu-w  ││ ap-se ││ ap-ne │   REGIONAL ENFORCERS
   │ + KV    ││ + KV   ││ + KV  ││ + KV  ││ + KV  │   - hold sub-budget
   └────▲────┘└───▲────┘└───▲───┘└───▲───┘└───▲───┘   - reserve / commit
        │         │         │        │        │       - emit audit
        │ gateway sidecar → enforcer (<2ms in-region)
   ┌────┴─────┐
   │ GATEWAY  │  per-request:
   │ + side-  │   1. estimate worst-case cost from params
   │   car    │   2. enforcer.reserve(cust, est) → res_id
   │          │   3. forward to backend
   │          │   4. on done: enforcer.commit(res_id, actual)
   │          │   5. enforcer sync-writes audit log; async-replicates
   └────▲─────┘
        │
   ┌────┴────────────────────────────────────────────────────────┐
   │ AUDIT LOG (Kafka acks=all → S3, partitioned by customer_id) │
   │ source of truth; coordinator + reconciler tail it           │
   └─────────────────────────────────────────────────────────────┘
```

The contract: **the audit log is the source of truth; everything else is a derived approximation. [STAFF SIGNAL: audit-log-as-source-of-truth]** A spend is "real" only when durably written. Running counters (regional sub-budget, coord view) are caches optimized for fast decisions.

Three components, three responsibilities:

1. **Regional enforcer** is the only thing on the request hot path. Owns sub-budgets, reserves on the way in, commits on the way out, emits audit events. Targets <2ms p99 for reserve+commit (in-memory).

2. **Coordinator** is off the hot path. Owns canonical per-customer budget, tier policy, period boundaries, lease registry. Hands out leases when regions request more, reclaims unused leases, runs reconciliation against the audit log.

3. **Audit log** is durable, append-only, partitioned by `customer_id` (per-customer total order). Synchronous write before request completion is the central correctness invariant **[STAFF SIGNAL: invariant-based thinking]**: *no spend is acked to the user until the audit log has it.*

## 5. Sub-budget allocation and the lease mechanism

The central mechanism. Three rejected alternatives first **[STAFF SIGNAL: rejected alternative]**:

- **Single global counter** (CAS on a shared key): every request hits one cross-region row. p99 50–150ms; whale-customer hot key; coordinator bottleneck at 1M req/sec. Rejected on latency and scaling.
- **Pure local enforcement, reconcile at month-end:** unbounded mid-month over-spend (regions don't see each other's spend until month-end). Rejected on over-spend bound.
- **Synchronous broadcast on every spend:** 5× write amplification, p99 floors at cross-region RTT. Rejected on latency.

Chosen: **token-bucket leases with adaptive sizing and demand-pull allocation.**

```
Coordinator                     Regional enforcer (us-east)
─────────                       ─────────────────────────
budget_total:    $1000          
budget_granted:  $800           sub_budget_remaining: $40 of last $200 lease
                                local_spend_since_lease: $160
                                lease_id: L7-9af3, expires: T+60s
                                spend rate ≈ $5/sec → exhausts in 8s
                                
        ◀─ lease_request(cust=C, lease=L7, used=$160,
                         rate=$5/sec, ask=$200) ──
        
  coord checks: total - granted_to_others = $1000 - $600 = $400 free
        
        ── grant(lease_id=L8, size=$200, ttl=120s) ──▶
                                
                                close L7 (report final spend $160)
                                open L8: sub_budget_remaining = $200 + $40
```

**Allocation strategy:** demand-pull. Regions hold zero initial sub-budget. Each region requests a lease *on first spend*, sized by recent local rate. Customers who only use us-east never tie up budget in idle regions. Rejected: equal split (wasteful), historical-weight split (cold-start fails on new customers).

**Lease size as the central tuning knob [STAFF SIGNAL: lease-size as tuning parameter]:**

```
Adaptive lease sizing rule:
  L = clamp(rate_local × headroom_seconds, L_min, L_max)
  
  headroom_seconds = f(remaining_budget / global_spend_rate):
    > 1 hour of spend remaining:  60s   (large leases)
    > 5 min:                      10s   (medium)
    > 30s:                        1s    (small)
    < 30s:                        SYNCHRONOUS  (per-request grant)
  
  L_min = max(p95 single-request cost, $0.10)
  L_max = remaining_budget / 10   (no region holds >10% of total)
```

Over-spend bound ≈ `5 regions × L + replication_lag × spend_rate`. With $1 leases at exhaustion, bound ≈ $5–$10. With $60 leases mid-month, bound ≈ $300 — but $300 on a $10K postpaid budget is invisible noise.

**Reclaim mechanism.** Every 30s the coordinator polls regions for `(local_spend, lease_age, utilization)`. If a lease is >5 min old and <20% utilized, the coordinator forcibly reclaims unused portion. This is the "high-traffic region starves while low-traffic region hoards" mitigation. Without it, demand-pull degrades into hostage-budget over time as customer traffic patterns shift.

**The sub-budget invariant [STAFF SIGNAL: invariant-based thinking]:** at any moment, `Σ active_leases ≤ budget_total - confirmed_spend`. The coordinator enforces this on every grant by reading the audit log's confirmed-spend high-water mark.

## 6. Variable-cost reservation

**[STAFF SIGNAL: variable-cost reservation]** Variable cost destroys naive enforcement. Three rejected alternatives:

- **Decrement after execution:** request executes, then deduct. Customer at $0.001 remaining + a $0.50 request = $0.499 over per request. Unbounded boundary overshoot. Rejected.
- **Decrement worst-case before:** reserve the max possible cost. Customer at $1 remaining is denied a $0.001 request because reservation is $10. ~99% false-deny rate near exhaustion. Rejected.
- **No reservation, accept overshoot:** fine when `max_cost ≪ budget`. For our spread ($10 max on $5 free tier = 200% overshoot possible), not fine. Rejected for our parameters.

**Chosen: reserve / execute / commit-refund with TTL.**

```
1. RESERVE                                2. EXECUTE
                                          
   gateway estimates from params:         backend processes
     est = predict_cost(params)           actual cost determined
     est = max(est, p95_for_params)         (e.g., output_tokens × price)
   
   enforcer.reserve(cust, est=$0.40):
     if sub_budget < est: REJECT
     sub_budget -= $0.40
     create reservation {
       res_id: R-7f3, reserved: $0.40,
       expires: T+30s, lease: L8 }
   ─ res_id ─▶ gateway

3. COMMIT (success)                       4. AUTO-REFUND (crash)
                                          
   actual = $0.05                         request crashed; res TTL fires
   enforcer.commit(R-7f3, $0.05):         sweeper at T+30s:
     refund = $0.40 - $0.05 = $0.35         sub_budget += $0.40
     sub_budget += $0.35                    audit_log.append(crash=R-7f3)
     audit_log.append(spend=$0.05)        no spend recorded; budget restored
```

**The reservation invariant [STAFF SIGNAL: invariant-based thinking]:** every reservation either commits with actual cost or expires with full refund. A reservation cannot leak. Implementation: reservations live in the regional enforcer's KV with TTL; a 1s heartbeat sweeps expirations and credits sub-budgets back. If the sweeper itself dies, alarm fires on `outstanding_reservation_value / lease_size > 0.5`.

**Estimation quality.** The unnecessary-deny rate near exhaustion ≈ `(p95_estimate − actual_mean) / p95_estimate`. For LLMs, `est = input_tokens × input_price + max_tokens × output_price` is tight (within 2×) for well-formed requests. Pathological case: `max_tokens=4096` but typical output 50 tokens → 95% of budget tied up unspent. **[STAFF SIGNAL: tier-aware policy]** We document this as a known free-tier limitation; postpaid tier uses `mean × 2` instead of p95 (over-spend is recoverable through invoicing).

**The audit log entry is the actual cost,** never the reserved cost. Reservations are in-memory state; the audit log only sees committed spend. This means a customer who issues a reservation and crashes never appears in the audit log — billing-correct.

## 7. Cross-region replication and convergence

Each region's spend is local; the *fact* of consumption must propagate. The model is a **G-Counter CRDT** per customer: global spend is the sum of per-region partial spends, each monotonically non-decreasing. Addition is commutative and associative — no two regions can disagree in a way that survives convergence.

```
Conceptual state for customer C:
  global_spend(C) = Σ own_spend(C, r) for r in regions

Per region maintains:
  own_spend(C)             // monotone-grows on local commits
  view_of(C, r')           // last-seen monotone value from r'

On local commit of $a in region r:
  own_spend(C) += $a
  emit replication event: (C, r, own_spend(C), ts)

On replication-receive at r from r':
  view_of(C, r') = max(view_of(C, r'), incoming_value)

Region r's belief about global spend:
  believed(C) = own_spend(C) + Σ view_of(C, r')
```

**Replication mechanism:** per-customer spend deltas batched at 100ms intervals, pushed to a Kafka-style replication bus partitioned by `customer_id` (per-customer order preserved). Each region tails the bus and applies updates to its `view_of` map.

**Replication lag budget:** within-region <10ms, cross-region p50 ~150ms, p99 ~500ms. With 1s batched replication, divergence ≤ `1s × per-region spend rate`. Whale at $10/sec → ~$10 staleness per region pair, ~$40 worst-case across the topology.

**Why not synchronous replication: [STAFF SIGNAL: rejected alternative]** writing 5× cross-region on every spend puts ~150ms RTT on the request path. Lag was paid into the data path; coordination cost was paid into the lease protocol (amortized over many requests). Explicit asymmetry.

**Convergence after partition.** During partition, replication events queue at the bus. On heal, the queue drains; each region's `view_of` catches up; the G-Counter converges. **No conflict resolution needed** because the data type is conflict-free by construction.

**The coordinator does not consume replication directly.** It rebuilds its global view from the audit log (durable, totally ordered, complete) at reconciliation time. Replication is the *fast path* for regions to know-roughly what others have done; the audit log is the *slow path* for billing-grade truth.

```
Replication topology (per customer_id partition):
  
  us-east ─emit─▶ ┌────────────────┐ ─tail─▶ us-west, eu-w, ap-se, ap-ne
  us-west ─emit─▶ │ Replication Bus│
  eu-w    ─emit─▶ │ Kafka per      │ ─tail─▶ all other regions
  ap-se   ─emit─▶ │   customer_id  │
  ap-ne   ─emit─▶ └────────┬───────┘
                           │ separate consumer group
                           ▼
                      S3 audit log
```

## 8. The exhaustion transition: progressive tightening

**[STAFF SIGNAL: progressive-tightening]** Enforcement strictness is not constant across the month. Most over-spend risk concentrates in the last few percent of budget. Be sloppy early, strict late, paying coordinator cost only where it matters.

```
Budget remaining → Enforcement mode:

 100% ┌─────────────────────────────────┐
      │ PHASE 1: GENEROUS               │  Lease:  $60 of spend
      │ - Large leases                  │  Coord:  ~1 RTT/min/region
      │ - Async repl, 1s batch          │  Lat:    1-2ms (local only)
      │ - Reservation = p95             │  O.S. bound: ~$300
  25% ├─────────────────────────────────┤
      │ PHASE 2: TIGHTENING             │  Lease:  $5
      │ - Smaller leases                │  Coord:  ~12 RTT/min/region
      │ - 100ms replication             │  Lat:    1-3ms
      │ - Reservation = p95             │  O.S. bound: ~$25
   5% ├─────────────────────────────────┤
      │ PHASE 3: STRICT                 │  Lease:  $0.50
      │ - Tiny leases                   │  Coord:  ~120 RTT/min/region
      │ - 10ms replication              │  Lat:    2-5ms
      │ - Reservation = max(p95, 1¢)    │  O.S. bound: ~$2.50
 0.5% ├─────────────────────────────────┤
      │ PHASE 4: SYNCHRONOUS            │  NO LEASES
      │ - Every request hits coord      │  Coord:  every request
      │ - Per-request grant             │  Lat:    30-80ms (x-region)
      │ - Replication: per event        │  O.S. bound: < 1 request
   0% ├─────────────────────────────────┤
      │ PHASE 5: EXHAUSTED              │  All requests rejected
      └─────────────────────────────────┘  with detailed denial msg
```

**Phase transitions are coordinator-driven.** Coord computes `remaining / global_rate` and pushes phase to all regions on lease grant. Regions ack new phase, adjust lease size, tighten replication. Transitions converge within one lease cycle.

**Cost economics.** For a customer with 1M requests/month, phase-3+4 covers ~the last 1% of volume. Coordinator load from synchronous-mode requests is 1% of what running-synchronously-all-month would cost. We pay strict-mode price only where strict-mode matters.

**Smoothed trigger, not instantaneous.** A customer who suddenly bursts (CI job kicks 10K requests at 3am) shouldn't flip into strict mode if the burst is short relative to remaining budget. Smoothing window = 5min EMA on spend rate.

**Phase 4 latency cost is real (~50ms p99 vs ~2ms).** Acceptable because (a) <1% of monthly volume, (b) customer is by construction near exhaustion and likely prefers accuracy over speed, (c) the alternative (uniform synchronous mode) would melt the coordinator at 1M req/sec. We surface the mode in the response (`X-Budget-Mode: strict`) so clients can detect and react.

## 9. Partition behavior

**[STAFF SIGNAL: partition-behavior policy]** Three cases, each with explicit tier-aware policy.

```
Case A: regional enforcer ↔ coordinator partition

  ┌──────────┐   ╳   ┌──────────┐
  │ us-east  │   ╳   │  coord   │
  └──────────┘       └──────────┘

  Region drains current lease, can't get new one.
  Tier policy:
    free / prepaid:  fail-closed when lease drained (over-spend = loss)
    postpaid:        continue at strict-phase rules using last-known
                     sub-budget; over-spend captured in next bill

Case B: regional enforcer ↔ replication bus partition

  Region's view_of stops updating; local enforcement uses stale
  view of remote spend; eventually drifts.
  
  Detection:    repl-bus lag > 30s
  Mitigation:   region requests synchronous mode from coord
                (skip local lease, every request hits coord).
                If coord also unreachable, fall back to tier policy.

Case C: coordinator ↔ all regions partition (or coord dies)
  
  All regions drain leases at their own pace, fall into tier policy.
  Coord is HA: 3+ replicas with Raft; quorum loss is the trigger.
  Lease-renewal stalls briefly during leader election (~5s).
```

**[STAFF SIGNAL: tier-aware policy]** The policy is grounded in revenue: over-spend on postpaid is recoverable (we invoice); over-spend on free-tier is non-recoverable (we eat it). We buy availability where cost is recoverable, correctness where it isn't.

**Heal protocol.** On reconnection, region sends `(lease_id, actual_spend, last_event_ts)` to coord. Coord reconciles against the audit log (which received the region's events all along through the replication bus, even during the coord partition — replication bus is independent infrastructure). Discrepancies logged for inspection; no automatic correction — the audit log is canonical, the coord's running counter is rebuilt from it.

## 10. Reconciliation and the audit log

**[STAFF SIGNAL: audit-log-as-source-of-truth]** The audit log is the only durable state. Everything else is a cache that can be rebuilt.

**Reconciliation cadence:**
- **Active customer (spending now):** every 1s, coord tails audit log partition for that customer, sums spend, compares with in-memory view. Discrepancies alarm.
- **Idle customer:** every hour or on first new spend.
- **Month-end:** full reconciliation against complete audit log; this number is what the customer is billed.

**Synchronous audit-log write invariant:** before the regional enforcer acks `commit` to the gateway, the spend event must be durably written. Kafka with `acks=all`, `min_insync_replicas=2`. Latency cost ~5ms — dominant cost on the request path, non-negotiable. If audit-log write fails (Kafka down), tier-aware fallback: postpaid → fail-open with WAL retry buffer (revenue-protective, captured on retry); free-tier → fail-closed (revenue-protective, no untracked spend).

**Reconstruction.** If coord state is lost (process crash, region failure), it rebuilds by replaying audit log from period-start. With Kafka log-compaction by `(customer_id, period)`, this is O(active customers) work, ~1M customers → bounded recovery 1–2 minutes.

**Billing-grade integrity.** At month-end the customer's bill is computed from a *closed* audit log, not from running counters. The running counter is allowed to be slightly wrong; the bill is not.

## 11. Multi-tenancy and high-cardinality

**[STAFF SIGNAL: hot-cold tenancy discipline]**

**Hot/cold tiering.** 1M customers, ~10K active in any given second. Regional enforcer keeps hot customers in-memory with LRU eviction. Cold customers' state in per-region KV (Cassandra-style). On first request for a cold customer: synchronous KV read (~2ms), instantiate in-memory entry, proceed. Subsequent requests are in-memory.

**Predictive pre-load** for high-frequency customers: offline job analyzes last 7 days of activity and pre-warms expected-active customers (e.g., a customer that bursts every Monday 9am gets pre-loaded at 8:55am). Avoids first-request latency penalty for predictable patterns.

**Coordinator sharding.** Customers sharded across coord pods by `hash(customer_id) % N`. A pod owns the lease registry and reconciliation for its shard. Failure of one pod affects only its customers; Raft-replicated state machine inside each pod gives single-shard HA.

**Noisy-neighbor protection. [STAFF SIGNAL: blast radius reasoning]** A customer making 1M req/sec to a single region hammers that region's enforcer with reservation traffic but not the coordinator — enforcer batches lease requests, doesn't make per-request coord calls. Coord-call rate per customer is bounded by `1 / lease_duration` ≈ 1/min/region in steady state. Even if all 1M customers were simultaneously near-exhaustion (synchronous mode), coord QPS = 1M req/sec spread across coord shards. Per-customer coord-call rate caps and tier-based admission control prevent any one customer from exhausting coord resources. Enterprise customers get dedicated coord capacity; free tier shares a pool with admission control.

## 12. Failure modes

**[STAFF SIGNAL: failure mode precision]**

- **Coordinator pod dies:** Raft quorum on remaining replicas; new leader; lease-renewal stalls ~5s. Existing leases unaffected.
- **Audit log (Kafka) unavailable:** tier-dependent. Free → fail-closed (refuse new requests, 503). Postpaid → fail-open with retry buffer in regional enforcer; spend events to local disk WAL, replayed on Kafka recovery. WAL bounded ~30 min of spend.
- **Region fully isolated:** see partition behavior.
- **Double-counting bug** (audit log gets a duplicate event): detected by reconciliation when coord-view diverges from sum-of-regional-views by >1%; alarm; manual replay from known-good checkpoint.
- **Reservation leak** (reservation neither commits nor expires): impossible by construction (TTL), but if the sweeper itself breaks, reservations accumulate; alarm on `outstanding_reservation_value / lease_size > 0.5`.
- **Estimator broken** (worst-case estimate is wildly wrong, denying legitimate requests): per-customer denial-rate alarm; circuit-break to historical-mean estimator.

**[STAFF SIGNAL: month-rollover discipline]** **Month rollover** is the hairiest event: at period boundary, all 1M customers' budgets reset simultaneously. Naive: every region's view of every customer changes at midnight UTC, producing thundering-herd refresh hits and inconsistency about which spend belongs to which period.

Protocol:

1. Coord pre-computes new-period allocations 15 min before rollover.
2. At T-0, coord publishes period boundary to replication bus.
3. Each region applies atomically: events with `ts < T-0` belong to closing period; events with `ts ≥ T-0` belong to new period.
4. Audit-log consumer for billing finalizes closing period at T+5 min (in-flight drain).

**Mid-period credit add:** coord increases `budget_total`; new lease grants reflect added budget within seconds; existing leases stay valid.

## 13. Customer-facing surface

**[STAFF SIGNAL: customer-facing surface]**

- **Dashboard** shows coord-view of `spent / total` with "as of N seconds ago" stamp. Customers must understand this is best-known, not exact — bill is canonical, dashboard is approximate.
- **Forecasting:** at current rate, projected exhaustion date. From EMA-smoothed spend rate.
- **Alerts** at 50%, 80%, 95% via configurable channel. Computed at coord; lag bounded by replication lag (~1s).
- **Denial debugging:** denied request returns
  ```
  HTTP 402 Payment Required
  X-Budget-Spent: $987.43
  X-Budget-Total: $1000.00
  X-Budget-Remaining: $12.57
  X-Request-Estimated-Cost: $15.00
  X-Budget-Mode: strict
  X-Period-End: 2026-05-31T23:59:59Z
  body: { reason: "request would exceed monthly budget",
          next_action: "increase budget, wait for reset, or
                        reduce max_tokens" }
  ```
- **Per-API breakdown** sourced from audit log (canonical), data freshness ~5 min through analytics pipeline.
- **Programmatic budget query** API: same numbers as dashboard, same staleness disclaimer.

## 14. Tradeoffs taken and what would change them

Optimized for: low p99 on the hot path, bounded over-spend (not zero), tier-differentiated policy. Traded off: strict global correctness mid-period.

- **If max-cost ≪ budget** (uniform small-cost workload): drop reservation, use post-hoc accounting. Simpler.
- **If latency budget allows 50ms** (batch/async workload): drop sub-budget allocation; synchronous global counter. Eliminates a category of failure modes.
- **If budgets are per-day** (tighter periods): leases shrink proportionally; coord load per customer ~30×; sharding becomes more important.
- **If over-spend is regulatory** (HIPAA-style hard legal caps): can't accept any over-spend. Synchronous mode all month; latency hit accepted; cost passed through. Different product entirely.
- **If cost variance is narrow** (compute-time billing, all requests within 2× of mean): mean-based reservation replaces p95; over-reservation penalty disappears.

## 15. What I'd push back on

**[STAFF SIGNAL: saying no]**

Three things in the prompt I'd push back on before architecting:

1. **"No further requests should succeed"** is unachievable in a multi-region low-latency system. The right SLA is "bounded over-spend per tier under bounded partition." I'd reframe this with the PM rather than architect around an impossible promise — the false promise gets us paged; the bounded promise can be designed to and met.

2. **"Multi-region, low-latency" without a tier breakdown.** Real platforms don't enforce identically across tiers. Free-tier strict enforcement is cheap when latency matters less; enterprise loose enforcement is sensible when over-spend is recoverable. The prompt's monolithic framing hides the most important policy lever.

3. **"Monthly"** is artificial. Real systems compose hierarchical budgets: per-minute rate limits for abuse, per-day caps for budget control, per-month for billing. The architecture I described generalizes (the lease mechanism nests trivially); the prompt should too. I'd ask whether "monthly" is a real product constraint or a simplification — and if it's a simplification, design for the real shape now to avoid retrofitting hierarchy later.