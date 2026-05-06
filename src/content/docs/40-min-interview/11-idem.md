---
title: Idempotency Layer for a Payments API
description: Idempotency Layer for a Payments API
---

# Idempotency Layer for a Payments API

## 1. Reframing: Kill "Exactly Once" Before It Kills You

**[STAFF SIGNAL: exactly-once-reframing]** "Exactly once" is marketing. In a distributed system with crashes, network partitions, and asynchronous downstream calls, exactly-once execution of a side effect is mathematically impossible in the general case. What we can build is a system that satisfies a precise external contract.

**Three invariants** **[STAFF SIGNAL: invariant-based thinking]**:

1. **At-most-once external side effect.** For a given idempotency key `K`, the externally observable side effect (a charge against a card, a transfer to a bank) happens zero or one time, never two or more, regardless of how many internal attempts occur.
2. **Reproducible successful response.** Once a request with key `K` has produced a response, every subsequent retry with key `K` and the same request fingerprint observes a byte-identical response — same status, same headers, same body, including any UUIDs and timestamps we minted.
3. **Fingerprint-bound replay.** A request with key `K` and a different fingerprint than the original is never replayed and never executed; it is rejected as misuse.

Internally the system may attempt the side effect 0, 1, or many times. What makes the contract holdable is that the side effect itself must be idempotent on a key we control. **[STAFF SIGNAL: end-to-end idempotency]** API-layer idempotency without downstream cooperation is a lie; it just relocates the double-charge from "our bug" to "the bank's bug." The PSP/issuer must dedupe on our key. This is a hard partner-integration requirement, not a nice-to-have.

**[STAFF SIGNAL: side-effect-ordering-as-central]** The central engineering problem is the ordering between the idempotency record write and the side-effect execution under arbitrary crashes. Every other concern — concurrency, response storage, retention, fingerprinting — is downstream of that. The whole engineering of an idempotency layer is the choreography that makes this ordering correct under arbitrary process death.

## 2. Scoping

**[STAFF SIGNAL: scope negotiation]** Committing:

- **API surface:** synchronous REST POST + JSON. `Idempotency-Key` header, UUIDv4-shape, ≤255 chars. Endpoints: `POST /v1/charges`, `POST /v1/transfers`. Async webhook delivery has its own related idempotency layer; out of scope here.
- **Side effect:** outbound HTTPS to a card network / PSP / issuer. Downstream supports an idempotency key we send, treated as a hard partner-integration requirement.
- **Retention:** 24h replay window + 24h tombstone for stale-key error messaging; hard delete at 48h.
- **Multi-region:** active-active across two regions. Each idempotency key is **pinned to a home region** at first write. Cross-region retries forwarded to home region; we accept ~80ms penalty on cross-region retries to avoid global consensus on every claim.
- **Throughput:** 50K req/s steady, 100K req/s peak. Replay rate ~3%. Concurrent same-key conflict rate empirically <0.05%.
- **Tenancy:** multi-tenant. Per-tenant rate limits upstream prevent one tenant saturating a shard.

## 3. Capacity Math

**[STAFF SIGNAL: capacity math]**

```
Per-record size:
  key (UUID)                        36 B
  fingerprint (SHA-256)             32 B
  request metadata                  ~64 B
  response status + headers         ~512 B
  response body (typical charge)    ~1.5 KB
  state, timestamps, lease, fence   ~64 B
  ───────────────────────────────────────
  total                             ~2.2 KB

Steady state (24h retention, 50K req/s):
  50,000 × 86,400 × 2.2 KB  ≈  9.5 TB

Peak window (48h tombstone, 100K req/s):
  upper bound                ≈  ~38 TB

Latency budget per request:
  Conditional claim write       1.5 ms p50,  5 ms p99
  Response durable write        2.0 ms p50,  6 ms p99
  Idempotency overhead total   ~3.5 ms p50, ~10 ms p99
  PSP side-effect              80–300 ms — dominates

Lease tuning:
  Initial lease                 30 s   (PSP timeout + buffer)
  Heartbeat renewal             10 s   (while side-effect inflight)
  Hard ceiling                 180 s
```

A 9.5TB working set at 50K req/s requires a horizontally sharded substrate; single-Postgres is excluded on this number alone.

## 4. State Machine of an Idempotency Record

**[STAFF SIGNAL: state-machine precision]**

```
                          ┌─────────┐
                          │ (none)  │  no record exists
                          └────┬────┘
                               │ conditional INSERT
                               │ ON CONFLICT DO NOTHING
                               │ + fence_token = N
                               ▼
                        ┌──────────────┐
      lease expires ◄───┤  IN_FLIGHT   │───► concurrent same-key arrives
      (orphan recovery: │  fence = N   │     → poll loop on this row
       fence → N+1)     │  lease_exp=T │
                        └──────┬───────┘
                               │
        ┌──────────────────────┼──────────────────────┐
        │ CAS on fence=N       │ CAS on fence=N       │ CAS on fence=N
        │ side-effect ok       │ recoverable failure  │ terminal failure
        ▼                      ▼                      ▼
  ┌───────────┐         ┌───────────────┐      ┌─────────────────┐
  │ COMPLETED │         │ FAILED_RETRY  │      │ FAILED_TERMINAL │
  │ resp body │         │ (next attempt │      │ resp body of    │
  │ stored    │         │  may proceed) │      │ failure stored  │
  └─────┬─────┘         └───────┬───────┘      └────────┬────────┘
        │                       │                       │
   retry → replay          retry → re-claim         retry → replay
   200/201 + body          (new fence)              4xx + body
                                │
                                └─► back to IN_FLIGHT (bounded)

  Terminal states age:
        ┌──────────────┐
        │  EXPIRED     │  24h after last update
        │  (tombstone) │  → retries get 410 Gone
        └──────┬───────┘
               │ +24h
               ▼
            (deleted, key reusable as new request)
```

**Transitions and atomicity:**

- `(none) → IN_FLIGHT`: conditional INSERT, atomic at the storage layer; issues fresh fence token. **Crash here:** insert either committed or didn't; next retry sees no row (re-claim) or IN_FLIGHT with dead lease (orphan path).
- `IN_FLIGHT → COMPLETED / FAILED_TERMINAL`: conditional UPDATE with `WHERE fence = N`. Fence prevents a stale-but-resumed worker (post GC pause) from clobbering a fresh worker's result. **[STAFF SIGNAL: lease/fencing discipline]**
- `IN_FLIGHT → IN_FLIGHT (orphan recovery)`: a fresh request finds `lease_exp < now()`, conditional-UPDATEs to bump fence to `N+1` and reset lease. The original holder, if it ever wakes, fails its CAS at `WHERE fence=N` and aborts.
- `FAILED_RETRY → IN_FLIGHT`: explicit re-claim. Bounded by an attempts counter to prevent infinite retry loops; exceeding it transitions to FAILED_TERMINAL with an operational alert.
- `* → EXPIRED`: explicit application-level check at read time (`if ttl < now() return 410`). Physical deletion via DynamoDB TTL within 48h after the ttl mark.

**The hardest case** is the IN_FLIGHT orphan path. Lease too short → false expiry while side effect is genuinely running → potential double-execute (mitigated only by PSP dedupe on our key). Lease too long → real failures cause user-visible stalls. We pick 30s base + 10s heartbeat, hard ceiling 180s, with the explicit assumption that the PSP also dedupes — the in-flight key from the original holder and the retry holder are the same key, so the PSP collapses both attempts to a single side effect and returns the canonical response to whichever attempt commits the CAS first.

## 5. High-Level Architecture

```
                ┌────────────────────────────────────┐
                │  Client (merchant)                 │
                │  sends Idempotency-Key header      │
                └─────────────┬──────────────────────┘
                              │ HTTPS
                              ▼
              ┌──────────────────────────────────────┐
              │  API Gateway                         │
              │  TLS, per-tenant rate limit,         │
              │  forwards to home region for key     │
              └─────────────┬────────────────────────┘
                            │
                            ▼
              ┌──────────────────────────────────────┐
              │  Payments App Server                 │
              │  ┌────────────────────────────────┐  │
              │  │ 1. Compute fingerprint          │ │
              │  │ 2. Conditional claim (DDB)      │ │
              │  │ 3. If COMPLETED → replay        │ │
              │  │ 4. If IN_FLIGHT → poll/wait     │ │
              │  │ 5. If claimed:                  │ │
              │  │    a. Outbox row in same txn    │ │
              │  │    b. Mint stable IDs           │ │
              │  │    c. Call PSP w/ our key       │ │
              │  │    d. CAS → COMPLETED + body    │ │
              │  │ 6. Reply                        │ │
              │  └────────────────────────────────┘  │
              └──────┬─────────────────────┬─────────┘
                     │                     │
       conditional   ▼                     ▼  outbox
       claim/CAS   ┌────────────────┐   ┌──────────────┐
                   │ Idempotency    │   │ Outbox Table │
                   │ Store (DDB,    │   │ (same DB,    │
                   │ sharded by key,│   │ same txn as  │
                   │ multi-AZ, PITR)│   │ claim)       │
                   └────────────────┘   └──────┬───────┘
                                               │ CDC / poll
                                               ▼
                                       ┌────────────────┐
                                       │ Outbox Worker  │
                                       │ Pool (PSP      │
                                       │ at-least-once  │
                                       │ with our key)  │
                                       └──────┬─────────┘
                                              │ HTTPS (idempotent)
                                              ▼
                                       ┌────────────────┐
                                       │ PSP / Network  │
                                       │ (dedupes on    │
                                       │ our key)       │
                                       └────────────────┘
```

The synchronous path is the fast path; the outbox is the recovery path. In steady state the app server completes the PSP call inline and CAS-writes COMPLETED. On crash mid-execute, the outbox worker retries the PSP call (which dedupes on the same key), receives the canonical response, and writes COMPLETED. The same downstream key serves both paths, which is what makes recovery determinism-preserving.

## 6. Side-Effect Ordering and the Outbox Pattern

**[STAFF SIGNAL: side-effect-ordering-as-central]** Walk the four candidate orderings.

**Option A — Side effect first, then store record.** Charge the card, then write the idempotency record. Crash between → next retry sees no record → charges again. **Reject: silent double-charge.**

**Option B — Store record first, then side effect (the naive textbook design).** Mark IN_FLIGHT, execute, mark COMPLETED. Crash mid-execution leaves IN_FLIGHT. Recovery requires (1) lease timeout for orphan detection and (2) the side effect to itself be idempotent so the next attempt doesn't double-charge. Without (2), unsafe under crash. With (2), it works for the synchronous path but has no story for "server got PSP 200 but died before writing COMPLETED" — the next retry sees IN_FLIGHT, eventually times out, re-attempts the PSP call (which dedupes and returns 200 again), and writes COMPLETED. Functional, but **the second PSP response need not byte-equal the first** (different PSP-side request_id, different timestamps), breaking the byte-identical-replay invariant. **Acceptable; inferior on determinism.**

**Option C — Two-phase commit between idempotency store and PSP.** Pre-allocate a transaction ID, pre-write a "preparing" record, call PSP with prepared ID, finalize. Adds round-trips; PSP must support prepare semantics, which most don't. **Reject: pushes coordination protocol onto a downstream we don't control.**

**Option D — Outbox pattern.** **[STAFF SIGNAL: outbox-pattern-or-equivalent]** In a single local DB transaction, atomically:

```
BEGIN;
  -- conditional on no existing row
  INSERT INTO idempotency
    (key, fingerprint, state='IN_FLIGHT',
     fence=1, lease_exp=now()+30s,
     stable_response_id=$generated_uuid);
  INSERT INTO outbox
    (key, payload, target='psp.charge',
     status='PENDING', attempts=0);
COMMIT;
```

Both rows commit or neither does. After commit, the synchronous path optimistically calls the PSP inline. On success: CAS UPDATE the idempotency row to COMPLETED + serialized body, mark the outbox row DONE in a single transaction. On any failure or crash: the outbox row remains PENDING; a worker pool tails the outbox (CDC or short poll) and retries the PSP call **with our key** as the downstream idempotency key. The PSP dedupes — we get the canonical response back — the worker writes COMPLETED + DONE.

**The unavoidable conclusion** **[STAFF SIGNAL: end-to-end idempotency]**: outbox works only because the PSP itself dedupes on our key. If the downstream is not idempotent, no API-layer design prevents double-charge under crash. Idempotency is a property of the entire pipeline; designing it as a single API-layer module is the mid-level mistake.

**Why inline-with-outbox-recovery, not pure async outbox?** **[STAFF SIGNAL: rejected alternative]** Pure async (always queue, never call PSP inline) is correct and simpler but adds 50–500ms tail latency on every charge. Merchants reject. We pay complexity to keep the median path inline.

**Why polling outbox, not transactional Kafka?** **[STAFF SIGNAL: rejected alternative]** Kafka transactions add a transaction coordinator and isolation-level reasoning; at our scale a polling outbox table on the same DB is simpler and the load is bounded by the failure rate, not the request rate. If our scale grew 10× we'd revisit.

**Stable IDs minted at claim, not response time.** Every response field that varies between executions — `charge_id`, `created_at`, server-generated reference numbers — is generated **at claim time** and stored in the idempotency row. The PSP call passes them in. The response object embeds them. On retry, replay regenerates nothing.

## 7. Concurrent Same-Key Handling

**[STAFF SIGNAL: concurrent-same-key]**

```
  Client (or retry storm) sends K twice → two app servers.

  ┌────────────┐                                ┌────────────┐
  │  Server A  │                                │  Server B  │
  └─────┬──────┘                                └─────┬──────┘
        │ CondPut(K, IN_FLIGHT, fence=1)              │
        │ if attribute_not_exists(K)                  │
        ├──────────────► Idempotency Store ◄──────────┤
        │ ◄ SUCCESS ─                                 │ CondPut(K, IN_FLIGHT, fence=1)
        │                                             ├─────► Store
        │                                             │ ◄ ConditionalCheckFailed
        │ executes side-effect (PSP)                  │
        │ ...                                         │ read existing row
        │                                             │ → IN_FLIGHT, lease_exp=T
        │                                             │
        │                                             │ poll loop:
        │                                             │  read every 50ms,
        │                                             │  bounded 5s + jitter
        │ CAS: IN_FLIGHT,fence=1 → COMPLETED          │
        ├──────────────► Store                        │
        │ ◄ SUCCESS ─                                 │ poll: row=COMPLETED
        │                                             │ ◄ row=COMPLETED + body
        │ reply to client                             │ replay stored response
        │                                             │ to its (retry) client
```

**Atomic claim primitive:** DynamoDB `PutItem` with `ConditionExpression: attribute_not_exists(pk)`. Strongly consistent on the partition. Postgres equivalent: `INSERT ... ON CONFLICT DO NOTHING RETURNING *`.

**Rejected:** **[STAFF SIGNAL: rejected alternative]** distributed lock service (ZooKeeper/etcd) — adds external dependency and moves the claim out of the same store as the data, breaking claim+outbox atomicity. Also rejected: Redis `SETNX` as source of truth — async replication can lose the most recent claim on failover, producing two winners.

**Losing-side behavior:** poll the row with bounded backoff, capped at 5s + small jitter. If it reaches COMPLETED, replay. If it reaches FAILED_TERMINAL, replay the failure response. If still IN_FLIGHT after 5s, return `409 Conflict` with `{"error":"idempotency_key_in_use","retry_after_ms":5000}`. We do **not** silently extend the wait; clients must distinguish "in flight" from "stuck" and SDK retries should not pile up.

**[STAFF SIGNAL: lease/fencing discipline]** **Fencing token discipline.** The row carries a monotonic `fence` counter. Every CAS to advance state requires `WHERE fence = N`. If server A GC-pauses for 60s, lease expires; server B reclaims via `WHERE fence=1 AND lease_exp<now()`, bumps to `fence=2`. When server A wakes and tries to write COMPLETED at `WHERE fence=1`, the CAS fails. Server A aborts and issues no further side effect. If server A had already issued a PSP call before pausing, the PSP dedupes our key — whichever attempt the PSP committed first is canonical, and server B's CAS captures the corresponding response. **[STAFF SIGNAL: blast radius reasoning]** This is the Kleppmann distributed-lock argument transplanted: locks alone are insufficient; the side-effect target must also enforce the fence (here, by deduping on our key).

## 8. Response Storage and Replay

**[STAFF SIGNAL: response-replay-with-determinism]** A retry must observe a byte-identical response, including any IDs we minted.

**Determinism capture at first execute.** Every non-deterministic field — server-minted `charge_id`, `created_at`, receipt numbers, anything that would differ between "compute again" and "remember from last time" — is generated **once** at claim time, stored in the idempotency row, and re-emitted on replay. A naive design that regenerates breaks the contract: clients caching by ID see two different "successful" charges and reconcile incorrectly.

Concrete: `charge_id` is generated before the PSP call so we can pass it as our reference. The full response object, including PSP-returned fields we surface, is serialized verbatim into the COMPLETED record.

**Ordering: durable response write before reply.** If we reply first and async-write the response, a crash in between loses the response. The next retry sees IN_FLIGHT, lease expires, outbox re-fires the PSP, gets the deduped response back — but the second PSP response need not byte-equal the first (different PSP-side `request_id`, different timestamp). Contract broken.

The correct ordering:

```
  PSP returns 200
        │
        ▼
  Serialize full response (using stable IDs minted at claim)
        │
        ▼
  CAS UPDATE: IN_FLIGHT,fence=N → COMPLETED, body=<bytes>
   (durably committed, multi-AZ)
        │
        ▼
  Reply to client with same bytes
```

We pay 2–6ms p99 for the durable write before reply. We do not optimize this away with an in-memory cache and async durability — failover loses recent responses, breaks determinism. The latency is the cost of the contract.

**Large responses.** Payments responses are <10KB; kept inline. For larger payloads (statement attachments), store a content-hash pointer plus blob in object storage with the same retention.

## 9. Fingerprint Validation

**[STAFF SIGNAL: fingerprint-validation]** Threat: client sends key `K` with body "charge $100" and gets a success. Later, the client (or an attacker holding the key) sends key `K` with body "charge $1000". Naive replay returns a fake success. Naive re-execute breaks the at-most-once invariant. Either way, broken.

**Mitigation.** At first claim, store `fingerprint = SHA-256(canonical(method | path | normalized_body | tenant_id))` in the row. On retry, recompute and compare:

- **Match → replay** stored response.
- **Mismatch → reject** with `422 Unprocessable Entity`, error code `idempotency_key_fingerprint_mismatch`. Do not execute, do not replay. Log for fraud analysis.

**What goes into `canonical(...)`:**

- `method`, `path` — yes.
- Request body — yes, after JSON canonicalization (sorted keys, no whitespace, integer/string normalization).
- `tenant_id` (or auth principal hash) — yes; same key from different tenants is a different request, full stop.
- `Idempotency-Key` itself — no; it's the key, not the payload.
- `User-Agent`, `X-Request-ID`, IP, accept-language, trace headers — no. Operationally varying; including them creates spurious mismatches when an SDK upgrades.
- `Content-Type` — yes if it changes parsing semantics.

**The grey zone — number normalization.** `{"amount": 100}` vs `{"amount": 100.0}` must canonicalize to the same fingerprint, or SDK serializer changes break replay. We canonicalize money as integer minor units (cents) before hashing.

**Rejected:** **[STAFF SIGNAL: rejected alternative]** comparing full request bytes — too brittle; any whitespace change breaks replay.

**Operational signal.** Sustained nonzero fingerprint mismatch rate is either (a) a client bug regenerating the same key for different requests, (b) an SDK serialization change, or (c) abuse. Surface to per-tenant dashboards; alert at 10/min/tenant.

## 10. Storage Substrate: DynamoDB

**Committed choice: DynamoDB** (Spanner where multi-region symmetric workloads dominate).

**Requirements check:**

- Conditional writes with strong per-partition consistency — yes.
- Horizontal scale at 100K req/s peak — yes, partitioned by hash of idempotency key.
- Multi-AZ durability — default.
- TTL — native (with up to 48h delete lag, which is why we tombstone in application code, not rely on TTL for correctness).
- Per-shard hot-key risk — keys are UUIDs, uniformly distributed; per-tenant rate limits prevent adversarial concentration on a partition.

**Rejected** **[STAFF SIGNAL: rejected alternative]**:

- **Single Postgres with partitioning.** Works to ~30K req/s with care. At our scale we'd be sharding by hand and managing failover at shard level — re-implementing DynamoDB worse. Rejected on operability.
- **Redis as source of truth (AOF + replication).** Sub-ms latency wins, but Redis failover under partition can drop the most recent fsync — and that's the most recent claim. For payments the worst case is "lost claim for an in-flight charge" → double-charge. Acceptable as a **read-through cache** in front of DynamoDB for the COMPLETED replay path (5–10× cost reduction on retry-heavy traffic) but never as source of truth for the claim.
- **CockroachDB / Spanner.** Strong consistency without sharding pain, multi-region native. Best for global active-active. We pick DynamoDB because home-region pinning gives us most of the multi-region benefit at lower cost; if the workload became truly region-symmetric we'd revisit.

**Schema (DynamoDB):**

```
PK: idempotency_key                      (string, hash key)
Attributes:
  state              NEW | IN_FLIGHT | COMPLETED |
                     FAILED_RETRY | FAILED_TERMINAL | EXPIRED
  fence              number, monotonic
  lease_exp          number, epoch ms
  fingerprint        binary, 32 B
  request_meta       map: method, path, tenant_id, created_at
  stable_ids         map: charge_id, created, etc. (minted at claim)
  response           map: status, headers, body  (set on COMPLETED/FAILED_TERMINAL)
  outbox_id          string, → outbox table
  attempts           number, bounded
  ttl                number, epoch s, drives 48h hard delete
```

## 11. Retention and Expiration

**[STAFF SIGNAL: expired-key contract]** Three windows, each with a precise contract:

- **0–24h: replay window.** Retries observe identical response.
- **24h–48h: tombstone window.** Record marked EXPIRED. Retries get `410 Gone` with `{"error":"idempotency_key_expired","original_request_at":"..."}`. We do **not** silently start a new execution; that would double-charge a client who cached a key.
- **>48h: deleted.** A new request with that key is treated as brand new and executed. Documented client contract.

**The expired-key race.** Without a tombstone, a record TTL'd at 24h plus a retry at 24h+50ms results in re-execution and double-charge. With the tombstone, the retry gets `410 Gone` and the client knows to use a fresh key. The tombstone is correctness, not nice-to-have.

**Mechanism.** Application checks `ttl < now()` at read time and returns `410` if so. DynamoDB TTL physically deletes within 48h after `ttl`. We do **not** rely on DynamoDB TTL for correctness — its delete latency is unbounded.

**Rejected:** **[STAFF SIGNAL: rejected alternative]** "extend retention forever" — storage cost grows unbounded; clients depend on bounded expiry semantics for their own bookkeeping. Bounded, documented window is correct.

**API contract documented:** "Idempotency keys are honored for replay for 24h after first use. Keys older than 24h that are reused will return `410 Gone`. After 48h, keys are eligible to be reused for new requests." This is part of the public API surface, not a hidden behavior.

## 12. Failure Mode Catalog

**[STAFF SIGNAL: failure mode precision]**

```
                    ┌──────────────────────────────┐
                    │  CRASH RECOVERY FLOW         │
                    └──────────────────────────────┘

  T0  Request arrives, fingerprint computed
       │
  T1  Conditional claim: IN_FLIGHT, fence=N, lease=T1+30s
       │     ┌── crash A (claim never committed) ──► retry: claim succeeds, normal
       │     ├── crash B (claim wrote, app died) ──► retry: sees IN_FLIGHT, polls,
       │                                              lease expires, orphan recovery,
       │                                              outbox re-fires PSP (dedupes),
       │                                              CAS COMPLETED
       ▼
  T2  Outbox INSERT (same txn as claim)
       │
  T3  PSP call (with our key as downstream idempotency key)
       │     ┌── crash C (PSP call mid-flight) ──► outbox re-calls; PSP dedupes;
       │                                            same canonical body → CAS COMPLETED
       ▼
  T4  PSP returns 200 + canonical body
       │
  T5  Serialize response (stable IDs already in row from T1)
       │     ┌── crash D (got 200, didn't store) ──► IN_FLIGHT until lease;
       │                                              outbox re-fires PSP;
       │                                              dedupe → same body → COMPLETED
       ▼
  T6  CAS UPDATE: IN_FLIGHT,fence=N → COMPLETED + body
       │
  T7  Reply to client
                ┌── crash E (replied, then died) ──► next retry: COMPLETED → replay
```

| Scenario | Detection | Response |
|---|---|---|
| Idempotency store unavailable | Conditional write throws | **[STAFF SIGNAL: fail-closed-policy]** Fail closed: `503` with `Retry-After`. Do **not** proceed without a claim. Refusing service for N seconds is bounded; double-charge is unbounded. |
| Crash between claim and outbox | Both-or-neither via single txn | If neither: retry re-claims. If both: outbox worker reattempts. |
| Crash mid-PSP-call | Lease expiry | Orphan recovery; outbox re-calls PSP; PSP dedupes our key. |
| Crash post-PSP, pre-COMPLETED | Lease expiry | Outbox worker re-calls PSP → same canonical body → COMPLETED. Determinism preserved because PSP returns the same bytes. |
| Crash post-COMPLETED, pre-reply | None needed | Next retry sees COMPLETED, replays. |
| Network partition app↔store | Conditional-write timeout | Fail closed. |
| Concurrent same-key, fingerprint match | Loser sees existing row | Wait/poll → replay. |
| Concurrent same-key, fingerprint mismatch | First writer's fingerprint vs second's body | Reject second with `422`. |
| Bad-actor key reuse, different payload | Fingerprint mismatch | `422` + abuse log. |
| Clock skew on lease | Use server-side `now()` from store, never client wall clock | Bounded skew within store. |
| GC pause longer than lease | Fence CAS fails | Late writer aborts; PSP dedupes any in-flight side effect. |
| Outbox worker pool down | Outbox row stays PENDING | Inline path may still succeed; alert on stuck-in-flight. |
| PSP returns ambiguous 5xx | Mark FAILED_RETRY, increment attempts | Bounded retries; after N → FAILED_TERMINAL + page. |
| Region failover | Cross-region forwarder detects | Promote standby; outbox replicated via Global Tables; same key still dedupes at PSP. |

## 13. Observability

**[STAFF SIGNAL: observability discipline]** Per-tenant and global, plus alerting:

- **Claim conflict rate** (concurrent same-key losses ÷ total claims). Baseline <0.05%. Spike → retry storm or load-balancer misbehavior.
- **Replay rate** (COMPLETED reads ÷ total claims). Baseline ~3%. Spike → upstream client retry storm or downstream incident causing client-side timeouts. Leading indicator for downstream issues.
- **Fingerprint mismatch rate** per tenant. Nonzero → client bug or abuse. Per-tenant alert at 10/min.
- **In-flight wait p50/p99/p999.** Latency the loser pays. p99 < 1s; >5s → stuck IN_FLIGHT, lease misconfigured, or outbox falling behind.
- **Stuck-in-flight gauge** (rows with `state=IN_FLIGHT AND lease_exp < now()-60s`). Should be near zero. Nonzero → page.
- **Outbox lag** (oldest PENDING age). p99 <5s; breach → page.
- **Fail-closed rate** (`503` from store-unavailable). Direct measure of customer impact during store incidents.
- **Audit log:** every claim, every state transition, every PSP attempt, every reply. Append-only; retained per compliance (typically 7 years for payments). Non-negotiable for chargeback investigations and SOX. This is part of correctness, not an "operations afterthought."

End-to-end p99 SLO ≤500ms (PSP-dominated); idempotency layer's contribution ≤15ms p99.

## 14. Tradeoffs Taken and What Would Change Them

We chose **inline-with-outbox-recovery** over **pure async outbox**, paying complexity for ~100ms median latency. SLO loosening to 500ms median → pure async is simpler and equally safe.

We chose **home-region pinning** over **global consensus on every claim**, paying ~80ms penalty on rare cross-region retries for ~5ms savings on the hot path. True symmetric active-active → Spanner replaces DynamoDB.

We chose **24h replay + 24h tombstone**. Stripe-class. Larger windows scale storage linearly (~9.5TB → ~30TB at 72h) — affordable but unjustified.

We chose **fail-closed** on store unavailability. For PSP integration where double-charge is six-figure dollars per hour, this is correct. For a system where double-execution is cheaply reconcilable, fail-open with reconciliation may dominate.

## 15. What I Would Push Back On

**[STAFF SIGNAL: saying no]**

- **"Exactly once."** Replaced with the three invariants. The phrase invites mid-level designs that ignore downstream cooperation.
- **"Idempotency at the API layer."** API-layer dedup without downstream dedup is theater. PSP idempotency support is a hard partner-onboarding requirement; reject PSPs that don't provide it.
- **"Single region is fine."** For real payments it isn't; commit to home-region pinning early. Retrofitting multi-region into a region-naive design is a 6-month project.
- **"Just use a Redis lock."** Locks aren't idempotency; locks plus a separate response store reintroduce the ordering problem less obviously. The idempotency record and the claim must be the same row in the same store.
- **"We'll add monitoring later."** Fingerprint-mismatch and stuck-in-flight are part of the correctness story. Without them, silent abuse or a leaking bug produces double-charges undetected for weeks.

The system above does not achieve "exactly once." It achieves at-most-once externally, byte-identical replay within 24h, and explicit failure modes everywhere else. That is the contract a payments API can actually keep.