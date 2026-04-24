---
title: Globally Distributed KV Store
description: Globally Distributed KV Store
---

```
Design a globally distributed KV store with configurable consistency per namespace — some namespaces need strong consistency (single-region leader), some need eventual multi-master. 5 regions, median latency target 10ms intra-region, 150ms cross-region. Walk me through it.
```
---
# Globally Distributed KV Store with Per-Namespace Consistency

## 1. Scope, Reframing, and CAP Posture

**[STAFF SIGNAL: scope negotiation]** Before I design anything, I'm pinning down what "KV store" means here because the answer bifurcates sharply on scope:

- **Key size**: bounded, ≤ 4KB. Larger is a blob store problem.
- **Value size**: two regimes. P50 ~1KB, p99 ~1MB. Values >1MB go to an object store with a pointer in the KV. I'm designing for the ≤1MB regime.
- **Durability**: fsync-per-write on strong namespaces (correctness-critical); batched group-commit with a ≤10ms ceiling for eventual.
- **Workload**: 100K writes/sec, 1M reads/sec steady-state, read-skewed, long tail of point reads and occasional TTL scans. No full-range scans as a first-class feature.
- **Multi-key**: single-key atomicity only in v1. Multi-key transactions are a yes/no product decision I'll flag in §12 — supporting them changes the storage format, shard placement rules, and client API. Committing to single-key for v1 and documenting the escape hatch.

**[STAFF SIGNAL: shared-substrate reframing]** The central difficulty in this prompt is not "build strong" or "build eventual." It is building a single substrate that honestly does both — sharing what can be shared, diverging only where it must. Two separate clusters glued by a routing tier is the mid-level answer and is a failure mode: doubled operational surface, two sets of on-call runbooks, two scaling stories, and no honest mode-change path.

**What's shared:**
- Storage engine (we're writing KV bytes either way — I'm committing to a Pebble/RocksDB-family LSM, specifically Pebble for predictable latency tail)
- Range model and sharding (every namespace is range-partitioned identically)
- Metadata service (placement, membership, mode config)
- Membership and failure detection (SWIM-style gossip + metadata-backed source of truth for range leases)
- Client routing (every client resolves `(namespace, key) → replica set` identically)
- Observability, deployment, security, mTLS, encryption at rest, snapshot/restore

**What diverges:**
- Replication protocol (Raft per range for strong; async pull-based replication with HLC ordering for eventual)
- Write path (cross-region quorum ack vs local ack)
- Read path (leaseholder/follower reads with closed timestamps vs local reads with conflict resolution on read)
- Conflict resolution (nonexistent for strong; HLC-LWW default + opt-in CRDTs for eventual)
- Recovery protocol after partition heal

**[STAFF SIGNAL: CAP honesty]** CAP is not negotiable. Committing explicitly:
- **Strong namespaces are CP.** During a partition, any range whose Raft group cannot form a quorum is **unavailable for writes, period**. Reads are available from followers with bounded staleness *if* closed-timestamp reads are enabled; otherwise also unavailable. No magic.
- **Eventual namespaces are AP.** During a partition, every region accepts writes locally. Divergence is permitted. On heal, reconcile per the namespace's conflict policy. The application must accept that reads can see a state not causally consistent with a write it just made in another region, unless it opts into session stickiness.
- There is no "both." Any customer asking for "strong and available during partitions" is wrong and I will say so in the product conversation.

**[STAFF SIGNAL: saying no]** I'll push back on one part of the prompt now: **"configurable consistency per namespace" is not free**. Each mode has its own operational runbooks, its own failure modes, its own capacity curves. Offering both doubles the surface area we debug at 3 AM. I'd ask the PM what fraction of customers actually need eventual — if it's <20%, push them to strong with follower reads and don't build the second path. Assuming the PM has validated demand, I continue.

---

## 2. Capacity Math and Latency Budget

**[STAFF SIGNAL: capacity math]** Numbers up front. They drive decisions.

**Latency targets (p50 / p99):**

| Operation | p50 | p99 | Notes |
|---|---|---|---|
| Strong single-region write (leader in region) | 4ms | 15ms | 1 intra-region Raft RTT (~1ms) + fsync (~2ms) + apply + ack |
| Strong cross-region write (3-of-5 quorum, 2 of 3 acks remote) | ~80ms | 180ms | Wait for 3rd-fastest ack — nearest 2 remotes, not slowest |
| Strong cross-region write (worst-case placement) | 155ms | 200ms | 1 cross-region RTT floor — nonnegotiable |
| Strong leaseholder read | 1-2ms | 5ms | Local read with lease validity check |
| Strong follower read (closed-ts, bounded staleness ~3s) | 1-2ms | 5ms | Local, no quorum |
| Eventual local write | 3ms | 10ms | Intra-region W=2 of 3 + fsync |
| Eventual cross-region replication delay | — | <1s | Async, measured as replication lag |
| Eventual read (local) | 1-2ms | 5ms | Local only |

**Cross-region bandwidth budget.** 100K writes/sec × 1KB avg = 100 MB/sec of raw writes. For eventual namespaces with all-regions replication, each write goes to 4 other regions → **400 MB/sec cross-region egress aggregate**, ~100 MB/sec per region-pair on average. For strong cross-region namespaces with leader co-located with writer region, each write hits 2 remote replicas → 200 MB/sec. Add ~30% protocol overhead: **~520 MB/sec worst case.** At cloud cross-region egress (~$0.02/GB), that's ~$45K/day = $16M/year just for replication egress. This is a product decision: **not every namespace should be all-regions.** Default should be 2-3 regions per namespace.

**Shard (range) sizing.** 100 TB / 256 MB per range = ~400K ranges. With RF=3, that's 1.2M range replicas across the fleet. Metadata at ~500 bytes/range = 200 MB of live metadata, fits comfortably in memory on the metadata service. Auto-split threshold: 512 MB or 10K QPS, whichever fires first.

**HLC overhead.** 5 regions. For eventual mode I'm using HLC timestamps (8 bytes physical + 2 bytes logical + 1 byte region_id = 11 bytes per write) — **not** full version vectors. A 5-entry version vector adds ~60 bytes per key and grows unboundedly with client IDs if we tracked those. HLC gives us causal ordering *within a region* and a deterministic tiebreak *across regions* without per-client state. Tradeoff flagged in §6.

**Read:write amplification.** 1M reads/sec served locally, 100K writes/sec generating cross-region traffic. This is why eventual mode is attractive for read-heavy global workloads: reads are 10× writes and pay zero cross-region cost.

---

## 3. High-Level Architecture

```
                        ┌─────────────────────────────────────┐
                        │  Control Plane (global)             │
                        │  ┌────────────┐  ┌───────────────┐  │
                        │  │ Metadata   │  │ Placement     │  │
                        │  │ Raft (5)   │  │ Driver        │  │
                        │  └────────────┘  └───────────────┘  │
                        │  namespace config, mode, ranges,    │
                        │  leases, replica placement          │
                        └──────▲──────────────────────▲───────┘
                               │ topology push/pull   │
      ┌────────────────────────┼──────────────────────┼──────────────────┐
      ▼                        ▼                      ▼                  ▼
  ┌───────┐  ┌───────┐  ┌───────┐  ┌───────┐  ┌───────┐
  │ US-E  │  │ US-W  │  │ EU    │  │ APAC  │  │ SA    │
  │       │  │       │  │       │  │       │  │       │
  │Gateway│  │Gateway│  │Gateway│  │Gateway│  │Gateway│
  │ ──┬── │  │ ──┬── │  │ ──┬── │  │ ──┬── │  │ ──┬── │
  │ Node  │  │ Node  │  │ Node  │  │ Node  │  │ Node  │
  │ pool  │  │ pool  │  │ pool  │  │ pool  │  │ pool  │
  │ (LSM) │  │ (LSM) │  │ (LSM) │  │ (LSM) │  │ (LSM) │
  └───┬───┘  └───┬───┘  └───┬───┘  └───┬───┘  └───┬───┘
      └──Raft / async-repl mesh across regions────┘

     Shared substrate: storage engine, ranges, routing, metadata
     Mode-specific:    replication protocol, write path, read path
```

Every node runs the **same binary**, **same storage engine**, **same membership protocol**, **same gateway**. The only thing that differs between a strong-namespace range and an eventual-namespace range hosted on the same node is the code path invoked for writes and reads, selected by the namespace's mode bit in the control plane.

**Control plane** is a 5-member Raft group placed odd-regionally to minimize cross-region quorum latency for metadata writes. Holds: namespace → (mode, replication policy, placement rule); range → (key range, replica set, leaseholder, mode-specific state); node → (region, health, capacity). Metadata reads served from a cached local replica on every node; writes go to control-plane Raft. Read-mostly, O(10-100) writes/sec steady-state (topology drifts slowly).

**Routing tier (gateway).** Thin per-region gateway that: (a) authenticates, (b) resolves `(namespace, key) → range → replica set` from cached metadata, (c) invokes the correct client protocol for the namespace's mode. Gateway is stateless; cache is eventually consistent with the control plane. Stale cache → retry with a hint from the server ("wrong node, leaseholder is now X").

**[STAFF SIGNAL: rejected alternative]** Rejected: **client-library-only routing** (smart client talks directly to nodes). Rejected because: (a) we support many language SDKs and the routing logic is complex — centralizing the hard parts in a gateway is the globally optimal engineering choice; (b) gateway is the natural place to terminate mTLS, enforce rate limits, and inject traces; (c) the extra hop is ~1ms intra-region, negligible against our budget. Smart clients are an optimization for latency-critical internal users later.

---

## 4. Partitioning and Placement

**[STAFF SIGNAL: rejected alternative]** Committing to **range partitioning with dynamic splits**, not consistent hashing. Reasons:
1. Per-namespace placement rules ("this namespace's replicas are only in US+EU, never APAC") are first-class — the product's whole point. Range partitioning gives us an explicit range-to-replica-set map; consistent hashing makes per-namespace placement awkward (restrict the hash ring? virtual-node migration during placement changes?).
2. Shard splits happen on keyspace boundaries atomically, affecting one range. Consistent hashing resharding requires token migration across the ring.
3. Range scans become feasible if we ever add them (cheap option preserved).
4. State of the art does this: Spanner, CockroachDB, TiKV, YugabyteDB. The industry has converged.

Rejected: **consistent hashing with vnodes** (Cassandra, Dynamo) — fine for eventual-only, fundamentally awkward for heterogeneous per-namespace placement. Rejected: **explicit static shards** (Vitess-style) — no dynamic splits, operator pain at scale.

### Range layout — strong namespace (all-region, RF=5, quorum=3):

```
Namespace "orders" (strong, 5-region, RF=5, quorum=3)

Key: orders/customer_42/order_abc
       │
       ▼
┌──────────────────────────────────────────────────┐
│ Range: orders/customer_40 .. orders/customer_45  │
│ Raft group r_7123                                │
│  Replicas:  US-E, US-W, EU, APAC, SA             │
│  Leaseholder: US-E  (follows write volume)       │
│  Quorum:     any 3 of 5                          │
└──────────────────────────────────────────────────┘
```

### Range layout — eventual namespace (3-region, intra-region RF=3):

```
Namespace "sessions" (eventual, regions=US-E,EU,APAC)

Key: sessions/u_99/token
       │
       ▼
┌────────────────────────────────────────────────────────────┐
│ Range: sessions/u_95 .. sessions/u_105                     │
│  US-E  replicas: 3 nodes, HLC-ordered, per-region leader   │
│  EU    replicas: 3 nodes, HLC-ordered, per-region leader   │
│  APAC  replicas: 3 nodes, HLC-ordered, per-region leader   │
│  Cross-region replication: async pull, per-region log      │
└────────────────────────────────────────────────────────────┘
```

Note the asymmetry: strong has **one global Raft group per range**; eventual has **per-region consensus (for durability) + async cross-region replication**. Storage engine and range metadata identical. The code that writes and reads differs.

**Hot-key protection.** For eventual, leaderless intra-region writes naturally spread across local replicas. For strong, a hot key is a hot leaseholder; a single-key split is impossible. Mitigation: push the app to shard at application layer or move to eventual mode. **[STAFF SIGNAL: saying no]** — the KV store cannot fix a fundamentally hot key.

---

## 5. Deep Dive A: Strong-Consistency Write and Read Path

**Commitment: Raft per range.** Multi-Paxos is equivalent; the ecosystem (etcd/raft, Hashicorp raft, dragonboat) makes Raft the pragmatic choice.

**[STAFF SIGNAL: rejected alternative]**
- **Chain replication**: throughput win at steady state, but head-of-line blocking under replica failure and more complex reconfiguration. Wrong tradeoff for interactive latency-sensitive workloads.
- **TrueTime-based ordering (Spanner)**: we don't own atomic clocks; cloud-provided time via NTP gives uncertainty intervals too wide for external consistency without prohibitively long commit-wait. HLC + closed timestamps is the post-Spanner state of the art (CockroachDB proved this model) and gives us bounded-staleness follower reads without atomic clocks.
- **Viewstamped Replication**: equivalent to Raft in safety/liveness, less tooling. No upside.

### Strong-mode write path:

```
Client   Gateway   Leaseholder    F2          F3          F4          F5
  │        │            │          │           │           │           │
  │──PUT──▶│            │          │           │           │           │
  │        │──Propose──▶│          │           │           │           │
  │        │            │─Append──▶│           │           │           │
  │        │            │─Append──────────────▶│           │           │
  │        │            │─Append──────────────────────────▶│           │
  │        │            │─Append──────────────────────────────────────▶│
  │        │            │◀─ack─────│           │           │           │
  │        │            │◀─ack─────────────────│           │           │
  │        │            │  (3 of 5 achieved → commit)                  │
  │        │            │─Apply to state machine                       │
  │        │◀──OK───────│                                              │
  │◀──OK───│                                                           │
```

For an in-region leaseholder with 2 in-region followers, quorum (3) is achieved intra-region: p50 ~4ms. For a leaseholder placed in US-E with replicas in all 5 regions, quorum = US-E + any 2 others; **we only wait for the 3rd-fastest ack**, not the 5th. With intra-region (US-E) + nearest remote (~40ms one-way) + next-nearest (~80ms one-way), p50 cross-region write is ~80ms *not* 150ms. The 150ms figure applies to pessimistic placements where the leaseholder lacks in-region followers.

### Read path options:

1. **Leaseholder read (linearizable).** Read from leaseholder after verifying lease validity. 1-2ms local. Default for strong reads.
2. **Quorum read (linearizable, no lease).** Read from a quorum of replicas, return newest. Used when leaseholder is down. Higher latency, rarely used.
3. **Closed-timestamp follower read (serializable at past T, not linearizable).** Leaseholder periodically (every ~200ms) publishes a closed timestamp `T_closed`: "no write at ts ≤ T_closed will ever commit to this range." Any follower with log applied through T_closed serves reads at T_closed. Staleness bounded by publish interval (~3s typical, tunable). **This is the critical mechanism for low-latency global reads on strong namespaces** — the post-Spanner breakthrough.

**[STAFF SIGNAL: consistency precision]** To be precise about what each gives:
- Leaseholder reads are **linearizable**.
- Quorum reads are **linearizable**.
- Closed-timestamp reads are **serializable at T_closed**, a consistent snapshot from the recent past. They are **not linearizable** (a newer write may have committed). Clients opt in with a `max_staleness` parameter.

### Leaseholder placement and failover

Leaseholder is elected via Raft; the placement driver can hint/constrain (e.g., "keep leaseholder in US-E, where 80% of writes originate"). Failover is standard Raft election (~1-3s including failure detection). Fencing uses the Raft term: a deposed leaseholder whose heartbeats lapsed cannot commit writes — its AppendEntries are rejected by followers with a higher term.

**[STAFF SIGNAL: invariant-based thinking]** Invariant: *no two leaseholders can commit writes in the same Raft term.* Enforced by Raft's log-matching and term-monotonicity properties.

### The zombie leaseholder (GC pause) case

A leaseholder that GC-pauses for 10s wakes up and tries to serve a read using its cached lease. Without fencing, it returns stale data. Fix: **epoch-based leases with wall-clock expiration**. Leaseholder checks its liveness epoch is live before serving any read. Each node maintains a liveness record in the metadata Raft that expires every 9s, renewed every 3s. A partitioned-off node's epoch expires; it stops serving; new leader takes over. CockroachDB uses this scheme; Spanner uses TrueTime commit-wait. Invariant preserved: *no leaseholder serves a read without a live epoch.*

### Availability envelope

A strong namespace's availability is bounded by (a) its Raft group's ability to form quorum, (b) metadata service reachability for routing, (c) gateway availability. (a) dominates. With RF=5 across 5 regions: survives 2 region failures. With RF=3 across 3 regions: survives 1 region failure. RF=3 is the sane default; RF=5 is for compliance/SLA-critical namespaces at doubled storage + bandwidth cost.

---

## 6. Deep Dive B: Eventual-Consistency Write Path and Conflict Resolution

### Eventual-mode write path:

```
Client   Gateway   Local-R1          Local-R2         Local-R3          Remote regions
  │        │          │                  │                 │                    │
  │──PUT──▶│          │                  │                 │                    │
  │        │─Write───▶│ (HLC stamp)      │                 │                    │
  │        │          │─repl─in-region──▶│                 │                    │
  │        │          │─repl─in-region────────────────────▶│                    │
  │        │          │◀─ack─────────────│                                      │
  │        │          │ (W=2 of 3 local, fsync complete)   │                    │
  │        │◀─OK──────│                                                         │
  │◀──OK───│                                                                    │
  │                   │ async background:                                       │
  │                   │ replicate to remote region logs via pull cursor────────▶│
```

Intra-region durability (W=2 of 3, fsync) before ack. Cross-region replication is async via pull-based per-region logs — remote region's replication worker polls a cursor on the local log and applies missed writes.

### HLC (Hybrid Logical Clock) mechanics

Each write gets timestamp `(physical_ts, logical_counter, region_id)`. On receive from another region, advance local HLC:
- `local.physical = max(local.physical, msg.physical, wallclock())`
- If physical ties, `local.logical = max(local.logical, msg.logical) + 1`

This gives us: per-region monotonicity; a total order across regions (region_id as final tiebreak) correlated with real time to within clock skew; correct ordering of causally related writes *if* they flow through the system in causal order, which single-key writes do by construction.

**[STAFF SIGNAL: rejected alternative]** Rejected: **version vectors / dotted version vectors**. They give true per-key causal history at the cost of unbounded growth with client identity. At 5 regions × millions of clients, pruning is hard and state dominates the value itself. HLC+LWW is coarser but bounded. For applications needing true causal divergence tracking, we offer CRDT types (below); plain register keys use HLC+LWW.

### Conflict resolution — per-namespace policy with per-key-prefix override:

1. **HLC-LWW (default).** Two concurrent writes resolved by HLC order; tiebreak region_id. Loses one write silently. Appropriate for last-writer-wins semantics (session state, user preferences).
2. **CRDTs (opt-in per key prefix).** Supported types: G-Counter (monotonic increment), PN-Counter (increment/decrement), LWW-Set, OR-Set (add/remove with observed-remove semantics), G-Map, MV-Register (returns all concurrent values). Encoded as CRDT header + payload; storage engine is opaque to this; read/write path deserializes and merges.
3. **App-defined merge (opt-in).** Namespace registers a sandboxed merge function `merge(v1, v2) → v3` required to be deterministic, commutative, associative, idempotent. Powerful, dangerous; used sparingly for e.g. shopping-cart merge. Versioned; breaking changes require a migration.

**[STAFF SIGNAL: conflict resolution discipline]** The trap: CRDTs are often pitched as "just use CRDTs, conflicts go away." In practice:
- CRDTs restrict your operations to those expressible commutatively. A bank transfer is not a CRDT.
- OR-Set has unbounded tombstone growth without coordination. Tombstone GC requires a causal barrier (must know all regions have observed the remove).
- PN-Counter over-counts under network replay unless you deduplicate by operation ID.
- A LWW-Register CRDT is no better than our default HLC-LWW.

We offer CRDTs as a principled tool for workloads they fit; we don't pretend they're a silver bullet. Riak bet the farm on CRDTs and hit exactly these operational limits.

### Concurrent-write scenario (ASCII):

```
Time
  │
  │  US-E writes K=A          EU writes K=B
  │  HLC=(100,0,US-E)         HLC=(100,0,EU)
  │        │                       │
  │  [regional replication complete locally, clients ACK'd]
  │        │                       │
  │        └──── async repl ───────┘
  │                  │
  │         Each region now has both writes in its per-region log.
  │         On read: resolve by HLC tiebreak on region_id.
  │         region_id(EU) > region_id(US-E)  ⇒  EU wins.
  │         US-E's write silently lost unless CRDT or session-token.
  │
  │  If MV-Register:
  │         Read returns BOTH {A, B}. App chooses.
  │
  │  If app-defined merge:
  │         Read returns merge(A, B).
  ▼
```

### Convergence and anti-entropy

Two regions converge after: (a) cross-region replication lag clears (target <1s p99 steady-state); (b) anti-entropy detects and repairs any gaps — Merkle-tree comparison per range, hourly baseline, on-demand on replication-lag spike. Convergence is **not** bounded under sustained partition. This is the AP tradeoff stated honestly.

### Read-your-writes across regions

**Not provided by default.** Client writes in EU, reads in APAC, may not see the write for up to replication lag. Opt-in mechanisms:

1. **Session stickiness.** Client pins to region until timeout.
2. **HLC-token session.** Client carries the HLC returned by its write. On subsequent read against another region, server waits until local replica has applied through that HLC or returns 409 "not caught up" with the current HLC. Upper-bounded wait; backpressure is explicit. (Pattern: Zanzibar's "zookie.")
3. **Accept staleness.**

---

## 7. Deep Dive C: The Mode-Change Protocol

**[STAFF SIGNAL: mode-change invariant]** This is the hardest problem in the system and the likeliest place for a subtle correctness bug. Treating it as a protocol, not a feature.

**Invariants preserved across mode change:**
1. No acknowledged write is lost.
2. No read returns a value inconsistent under the target mode's guarantees *from the moment the mode change is announced to clients*.
3. Protocol is resumable and reversible until a **point-of-no-return**, which is explicit and logged.

### Protocol: eventual → strong (the harder direction):

```
Stage 0: ANNOUNCE
  ├─ Control plane sets namespace.target_mode = STRONG
  ├─ Mode = TRANSITIONING_TO_STRONG
  ├─ Gateways receive topology push; clients get a warning header
  └─ Writes still accepted (eventual semantics)

Stage 1: FREEZE CONFLICT CREATION
  ├─ Switch writes to "single-region-only":
  │   all writes for this namespace route to a designated primary region.
  ├─ Non-primary regions return 503 "mode-transition, retry against US-E"
  ├─ Reads still local-eventual.
  └─ Now no new conflicts can be created. Backlog of prior conflicts remains.

Stage 2: DRAIN REPLICATION
  ├─ Wait for cross-region replication lag → 0.
  ├─ Run anti-entropy sweep: Merkle-tree compare all replicas across regions.
  ├─ Any mismatches: repair via HLC-LWW.
  └─ Bounded wait; if exceeds threshold (e.g., 10 min), ROLLBACK.

Stage 3: RESOLVE CONFLICTS
  ├─ Enumerate all keys with concurrent versions (CRDT or app-merge mode).
  ├─ CRDTs: merge deterministically.
  ├─ App-merge: invoke registered merge function.
  ├─ MV-Register with >1 value: ESCALATE — no algorithm resolves correctly.
  │   Halt migration; require human choice via resolution UI, OR
  │   pick by policy (latest HLC) and log as lossy. Customer chooses.
  └─ Now every key has a single canonical value.

Stage 4: INSTALL RAFT    ◀──── POINT OF NO RETURN
  ├─ For each range: bootstrap Raft group with current replica set.
  │  Canonical value becomes Raft state-machine's initial state.
  ├─ Writes now go through Raft; eventual-mode write path disabled.
  ├─ Reads immediately become leaseholder reads.
  └─ Mode = STRONG.

Stage 5: CLEANUP
  ├─ Delete eventual-mode metadata (per-region logs, HLC tombstones).
  ├─ Announce mode-change complete.
  └─ Clients resume normal operation.

ROLLBACK (stages 0-3 only):
  ├─ Revert target_mode; re-enable cross-region writes.
  ├─ Any writes accepted in primary-only mode are durable and visible
  │  per eventual semantics; no data lost.
  └─ After stage 4, rollback requires a separate strong→eventual migration.
```

**The subtle case.** In stage 3, if we have MV-Register keys with semantically incompatible concurrent values (two different shopping carts for the same user), no algorithm resolves them correctly. We surface to the customer via resolution API, pause the migration, and wait. This is a product feature, not a correctness escape hatch: **eventual-mode customers must understand that switching to strong requires resolving the divergence that eventual mode permitted.**

### Protocol: strong → eventual (easier, not trivial):
- Stage 0: Announce.
- Stage 1: For each range, take snapshot at a specific Raft log index. Record it as eventual-mode genesis.
- Stage 2: Bootstrap per-region leaders from snapshot. All replicas share identical state at genesis.
- Stage 3: Switch writes to eventual path. Tear down Raft group.
- Subtle: clients mid-flight with an in-flight strong write may have an uncertain outcome. Protocol drains in-flight Raft writes first. A client with successful strong-write ack sees that write; a client with uncertain response must retry under eventual semantics.

**Duration.** eventual→strong ~30 min for a 100 GB namespace; strong→eventual ~10 min. Scheduled during customer-approved windows.

---

## 8. Deep Dive D: Partition Behavior

**[STAFF SIGNAL: partition behavior precision]** Scenario: **US-E is partitioned from {US-W, EU, APAC, SA}.** US-E can still talk to its own nodes and to clients inside US-E.

### Strong namespace, RF=5, leaseholder in US-E:
- Raft group has 1 of 5 replicas reachable (itself). Cannot form quorum. **Range unavailable for writes.**
- After ~9s the leaseholder's liveness epoch expires. The other 4 regions elect a new leaseholder among themselves. Writes resume on the {US-W, EU, APAC, SA} side.
- US-E clients see "no leaseholder reachable" for this namespace. Must fail over via application-level DR strategy (read-only mode, or route to another region).
- US-E clients **can** do closed-timestamp reads locally against the ex-leaseholder replica, bounded staleness ≤ T_closed at partition time. After ~3s, no newer closed timestamp arrives; reads become increasingly stale and eventually stop (staleness budget exceeded).

### Strong namespace, RF=5, leaseholder outside US-E:
- Raft group has 4 of 5 replicas on the outside. Writes proceed, p50 latency rises slightly (different RTT mix in new quorum).
- US-E replica is a follower not receiving AppendEntries; cannot publish new closed timestamps. US-E clients see bounded-staleness reads up to the last closed timestamp.
- US-E client experience: stale reads up to staleness budget, then errors. No writes.

### Eventual namespace, multi-region:

```
Before partition:
  US-E  ◀── async repl ──▶  EU  ◀── async repl ──▶  APAC ...

During partition (US-E isolated):
  US-E   ←────── X ──────→   {EU, APAC, US-W, SA}
   │                                  │
   │ accepts writes                   │ accepts writes
   │ HLC=(T+n, *, US-E)               │ HLC=(T+n, *, {EU,APAC,...})
   │                                  │
   │ local reads: US-E writes only    │ local reads: non-US-E state

Heal (after 10 minutes):
  US-E accumulated 10K local writes the other regions don't have.
  {EU, APAC, US-W, SA} accumulated 40K writes US-E doesn't have.
  Replication logs drain in both directions.
  For each key with writes on both sides:
    - HLC-LWW:  later HLC wins, other silently lost
    - CRDT:     merge deterministically
    - App-merge: invoke function
  Anti-entropy confirms convergence via Merkle tree.
  Lag metric returns to <1s p99.
```

**Client experience in eventual mode during partition:** every client sees a locally-consistent view of its own region. A user who reads-then-writes in EU sees their own writes (intra-region, no replication lag). A user who bounces between regions sees per-region-consistent views but may appear to time-travel. The application either pins users to regions or accepts this. Document it.

**The heal is non-blocking.** New writes continue during reconciliation. Replication protocol handles out-of-order application via HLC.

---

## 9. Deep Dive E: Cross-Region Replication Protocol and Cost

Per-region replication log (Kafka-like, internal): each range has a monotonic per-region log keyed by HLC. Remote regions pull via long-poll with a cursor. Batch size tuned to amortize network overhead: 4KB-64KB batches, target 100 batches/sec per region-pair.

### Backpressure and region-down behavior

If EU is unreachable, US-E's EU-bound cursor stops advancing. US-E continues accepting writes (eventual mode); the per-region log accumulates. Log retention capped at 7 days (configurable per namespace) — beyond that, EU requires a bootstrap from snapshot. The 7-day figure covers most operational incidents without unbounded storage.

### Bandwidth math — worst-case eventual namespace
- 100K writes/sec × 1KB × 4 remote regions = 400 MB/sec aggregate.
- zstd compression (~3× on JSON-like values) → ~130 MB/sec.
- Protocol overhead + heartbeats → ~160 MB/sec.
- Per region-pair: ~16 MB/sec typical (traffic is asymmetric — US-E→EU heavier than APAC→SA).

### Read-your-writes across regions — API options
1. **Session stickiness** — client pins to region until timeout.
2. **HLC-token wait** — client passes write's HLC; server waits up to `max_wait` for local replica to catch up.
3. **Accept staleness** — no guarantee, fastest.

All three documented; customer picks per session.

---

## 10. Failure Modes and Correctness Recovery

**[STAFF SIGNAL: failure mode precision]**

1. **Shard loses Raft quorum.** Detect: gateway 503s from leaseholder; metadata sees expired lease. User-visible: writes fail "range unavailable." Recovery: if majority replicas merely unreachable, wait; if permanently lost (≥ quorum nodes' disks dead), invoke **unsafe recovery** — operator confirms data loss, rebuilds range from remaining replicas, fences writes during rebuild, new Raft group bootstrapped with best-available state.

2. **Permanent region loss.** Detect: region-wide heartbeat loss + out-of-band confirmation. Strong namespaces with one replica in lost region: degraded to RF-1, quorum reduced proportionally; operator must add replica in a new region to restore durability. Strong namespaces whose quorum crossed the lost region: **unavailable** until unsafe recovery. Eventual namespaces: lose writes that hadn't replicated out (bounded by replication lag, typically ≤1s = ≤100K writes in the worst case).

3. **Silent disk corruption.** Detect: hourly Merkle-tree anti-entropy detects replica divergence; checksum mismatch on read. Recovery: minority-value replica wiped, re-replicated from majority. Invariant: *every persisted KV pair has a checksum computed at write and verified on read.*

4. **Leaseholder zombie (GC pause).** Covered in §5. Invariant via epoch-based liveness.

5. **Buggy client writes malformed data.** For strong: bad write persisted, replicated, visible. Recovery: application-level compensation. The KV store is not a sanitizer. Mitigation: per-namespace validation hooks (schema, size limits) at gateway.

6. **Failed Raft membership change (joint consensus).** Raft uses joint consensus for adding/removing replicas: old and new configs must both reach quorum during transition. Failure mid-transition leaves range in joint configuration. Tooling: operator can force-complete or force-abort joint configs via a metadata-Raft-write.

7. **Metadata Raft unavailable.** **[STAFF SIGNAL: blast radius reasoning]** Catastrophic: no lease transitions, no range splits, no placement changes, no mode changes. But the **data plane degrades gracefully** — existing leases continue to serve as long as leaseholder is alive and cached metadata is valid. Eventual namespaces continue accepting writes. Strong namespaces continue at current leaseholders. What breaks: any operation requiring metadata write (new range, lease renewal beyond cached epoch, failover). We size metadata Raft for 99.999% availability (5 members, 3 regions) and keep write rate low. The whole system is designed so data plane survives control plane outages for ~epoch-lifetime (9s+) without user-visible impact.

---

## 11. Consistency Guarantees in Precise Terms

**[STAFF SIGNAL: consistency precision]**

### Strong namespace guarantees:
- **Single-key reads**: linearizable (leaseholder or quorum path).
- **Closed-timestamp follower reads**: serializable at a past timestamp. **Not** linearizable. Monotonically consistent within a client session (enforced by client token).
- **Monotonic reads**: yes (via session token).
- **Read-your-writes**: yes within session + region; cross-region requires leaseholder read or HLC-token wait.
- **Monotonic writes**: yes (linearizability implies it).
- **Multi-key atomicity**: not in v1.

### Eventual namespace guarantees:
- **Durability**: W=2 of 3 intra-region, fsync, before ack.
- **Eventual convergence**: assuming partition heals and replication catches up, all replicas converge.
- **Per-region linearizability**: a client talking to one region's replicas sees a linearizable view of that region's writes (within-region consensus is present).
- **Causal consistency across regions**: **NOT** provided by default. A client reading from EU then writing to US-E has no guarantee the US-E write is ordered after what it read. Opt-in via HLC-token session.
- **Read-your-writes across regions**: not by default; session stickiness or HLC-token wait opt-in.
- **Multi-key atomicity**: not meaningful under this consistency model.

Being precise about these is the difference between a product customers can build on and a product that generates incident tickets for a decade.

---

## 12. Multi-Key Operations

Committed to single-key in v1. The escape hatch:

- **Co-location hint**: client declares "these keys must live on the same range" via a key-prefix hint. Placement driver honors subject to range-size limits. Single-range multi-key operations then execute atomically via Raft (one log entry per batch). Good enough for 80% of multi-key use cases.
- **Cross-range transactions (v2)**: 2PC with a transaction coordinator, Percolator-style. Significant new protocol. Adds latency (2 Raft rounds), adds failure modes (prepared-but-not-committed transactions need recovery), requires MVCC storage layout (move from Pebble to MVCC-aware layout — nontrivial). Quarter-plus of work.
- **Eventual namespaces**: multi-key atomicity is not meaningful — the consistency model cannot support atomicity across keys with independent concurrent writers. We offer best-effort batching (N keys in one request: durability atomic, visibility not) and document the limitation.

**[STAFF SIGNAL: saying no]** If the product requirement becomes "serializable multi-key transactions across any namespace," this design is wrong and we should instead build a Spanner/CockroachDB-style MVCC transactional store from day one. The substrate does not cleanly extend. Kill the ambiguity now before we're 6 months in.

---

## 13. Recent Developments and Why They Matter Here

**[STAFF SIGNAL: modern awareness]**

- **Spanner (2012) + TrueTime**: proved globally-consistent transactions at scale via bounded clock uncertainty. Commit-wait requires atomic-clock infrastructure. Without that, we use HLC + closed timestamps — the post-Spanner practical design.
- **CockroachDB closed timestamps + follower reads**: proved you don't need TrueTime to give users bounded-staleness consistent reads on strongly-consistent data. The single most important technique I've borrowed. Transforms the global read path from "cross-region hop to leaseholder" to "local follower read" for the huge class of read workloads tolerating seconds of staleness.
- **FoundationDB**: separation of concerns (transactions, storage, resolution in separate layers) and deterministic simulation testing. Our correctness-testing story should crib their simulation approach — deterministic simulation of Raft + replication + partitions is the only way to find the bugs we otherwise won't. Snowflake, iCloud, and Tigris running on FDB is existence proof that one substrate can serve wildly different workloads.
- **DynamoDB Global Tables**: multi-master eventually consistent, and the public postmortems (LWW silently losing writes, the move to stricter per-item guarantees) are a warning. Default-LWW without application awareness causes data loss; we surface conflict policies explicitly per namespace and default to HLC-LWW only when the customer has opted in.
- **Aurora DSQL** (announced late 2024): active-active Postgres-compatible with separation of storage, compute, and transaction log. Shows the industry moving toward log-centric designs where the log is the truth and compute is stateless. Our replication log is the same pattern.
- **TiKV / YugabyteDB**: Raft-per-range is the de-facto standard for strongly-consistent KV; I'm following their model.
- **CRDTs in practice (Riak's pivot)**: Riak bet on CRDTs and hit exactly the issues flagged in §6 — unbounded tombstone growth, app restrictions. Antidote and recent bounded-CRDT work matter for customers who truly need them; most don't.
- **HLC paper (Kulkarni et al., 2014)**: foundational for the eventual-mode ordering design.
- **Google Zanzibar**: demonstrated how to combine a strongly-consistent core with bounded-staleness reads at massive scale. The "zookie" token is the pattern I'm proposing for session consistency.

---

## 14. Tradeoffs Taken and What Would Force a Redesign

- **Single-key atomicity only.** If we need multi-key transactions: MVCC + 2PC + transaction coordinator. Quarter-plus of work. Storage layout changes.
- **HLC + LWW for eventual (no true causal tracking).** If customers need causal consistency guarantees: add version-vector tracking with pruning protocol. State blows up 10-60×. Offer as opt-in per namespace.
- **Range partitioning.** If we suddenly need uniform load distribution without per-namespace placement: consistent hashing. But lose per-namespace placement — the product's whole point.
- **Gateway tier adds 1ms hop.** Latency-sensitive internal users who can't pay 1ms: ship a smart-client library that resolves ranges directly. Added maintenance cost per SDK language.
- **Metadata Raft is a single-point-of-governance.** Multi-tenant hard isolation requirement: shard metadata Raft by tenant. Operational complexity.

---

## 15. What I Would Push Back On

**[STAFF SIGNAL: saying no]**

1. **"Configurable consistency per namespace" as a headline feature.** Operationally expensive. Ask the PM: what fraction of customers need eventual? If <20%, push them to strong+follower-reads and don't build the eventual path. If "all of them, team A wants strong and team B wants eventual," we're solving an org problem with a product.

2. **"150ms cross-region p50 target for strong writes."** That is a **floor**, not a target. Any product requirement below ~170ms p99 for cross-region strong writes is unmeetable. Product must accept this, or choose single-region-leader placement (with associated availability cost).

3. **Implicit assumption that namespaces replicate to all 5 regions.** Doubles cost unnecessarily. Default should be 2-3 regions per namespace, opt-in to 5.

4. **The ask for "live mode change" without specifying acceptable downtime or latency-spike budget.** As designed, mode change is minutes to hours with partial availability. Zero-downtime, zero-latency-spike mode change is not achievable given the invariants we preserve — the drain-and-install step takes time proportional to replication lag and conflict count.

5. **"KV store" without scope clarification.** If customers actually need range scans, secondary indexes, or transactions, this design is the wrong starting point. Be honest about this before we commit.

---

**[STAFF SIGNAL: operational longevity]** Year-5 outlook: adding a 6th region requires rebalancing ~20% of ranges, done by placement driver incrementally at capped bandwidth. Removing a region requires range-by-range replica replacement before decommission. Rolling protocol upgrade requires feature-gated dual-write/dual-read phases for any on-disk format change — feature-gate flags live in range metadata so individual ranges upgrade independently and can roll back. Storage engine swap (Pebble → something) is a 6-month project with shadow replicas running in parallel and a flip-switch per range. Deterministic simulation testing (FoundationDB-style) is the only tractable way to maintain correctness across this change set.

The system runs for a decade. We plan for it.