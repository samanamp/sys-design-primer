---
title: Globally Distributed KV Store (simplified)
description: Globally Distributed KV Store
---

```
Design a globally distributed KV store with configurable consistency per namespace — some namespaces need strong consistency (single-region leader), some need eventual multi-master. 5 regions, median latency target 10ms intra-region, 150ms cross-region. Walk me through it.
```
---
# Globally Distributed KV Store with Per-Namespace Consistency

## 1. Scope, Reframing, and CAP Posture

Before designing anything, I want to pin down what "KV store" actually means here, because the answer changes a lot based on the assumptions.

**[STAFF SIGNAL: scope negotiation]** My assumptions, stated and committed to:

- **Key size:** ≤ 4KB. Larger keys are a blob-store problem, not a KV problem.
- **Value size:** two regimes. Typical (p50) ~1KB, long tail (p99) ~1MB. Anything over 1MB goes to an object store, with a pointer stored in the KV. I'm designing for the ≤1MB regime.
- **Durability:** for strong namespaces, every write is fsync'd before we acknowledge it — correctness matters more than throughput there. For eventual namespaces, writes are batched (group-commit) with a 10ms ceiling, which trades a tiny durability window for much better throughput.
- **Workload:** 100K writes/sec, 1M reads/sec, read-heavy. Mostly point lookups. No full range scans as a first-class feature.
- **Multi-key operations:** single-key atomicity only in v1. I'll discuss the escape hatch in §12. Multi-key transactions are a big yes/no decision — supporting them changes the storage format, the placement rules, and the client API. I'd rather commit to single-key and add multi-key deliberately than smuggle it in.

### The central reframe

**[STAFF SIGNAL: shared-substrate reframing]** The hard part of this prompt isn't "build strong" or "build eventual." It's building **one system that does both honestly**, sharing everything that can be shared and diverging only where it must. Two separate clusters with a routing tier in front is the mid-level answer. It fails because:

- You double your operational surface — two on-call runbooks, two upgrade processes, two capacity stories.
- You have no honest path to change a namespace's mode without migrating it across clusters.
- You pay twice for everything: metadata, monitoring, deployment.

Here's the split between shared and mode-specific:

**Shared across both modes:**
- Storage engine (Pebble, an LSM tree). We're writing key/value bytes either way.
- How data is sharded into ranges.
- Metadata service (who owns what, which mode is which namespace in).
- Node-to-node communication, failure detection, gossip.
- Client routing: `(namespace, key) → replica set`.
- Observability, deploys, security, encryption at rest.

**Different per mode:**
- Replication protocol: Raft for strong; pull-based async log replication for eventual.
- Write path: wait for cross-region quorum vs. ack locally and replicate later.
- Read path: leaseholder reads / follower reads with bounded staleness vs. local reads with conflict resolution.
- Conflict resolution: none needed for strong; HLC last-writer-wins by default for eventual, with opt-in CRDTs.
- Recovery after a partition heals.

### CAP, stated honestly

**[STAFF SIGNAL: CAP honesty]** CAP is a real constraint. I'm going to state what each mode gives up during a network partition, plainly:

- **Strong namespaces are CP (consistent + partition-tolerant, not always available).** During a partition, a shard whose Raft group can't form a quorum is **unavailable for writes**. Period. Reads may still work via follower reads with bounded staleness, if the namespace has opted into that. Otherwise reads fail too.
- **Eventual namespaces are AP (available + partition-tolerant, not strictly consistent).** During a partition, every region keeps taking writes. The regions will temporarily disagree. When the partition heals, they reconcile per the namespace's conflict-resolution policy. The application has to accept that a write in one region may not be visible in another for seconds, and that concurrent writes across regions can conflict.

There is no "strongly consistent AND always available during partitions." Anyone asking for both is wrong, and I'd say so in the product conversation.

**[STAFF SIGNAL: saying no]** One thing to push back on up front: **"configurable consistency per namespace" is not free.** Each mode has its own runbooks, failure modes, and performance envelope. Offering both doubles the complexity we debug at 3am. Before building this, I'd ask the PM: **what fraction of customers actually need eventual consistency?** If it's under 20%, I'd nudge them to strong + follower reads and not build the second path. For the rest of this answer I'll assume the PM validated the demand.

---

## 2. Capacity Math and Latency Budget

**[STAFF SIGNAL: capacity math]** I'm going to work through each number step by step so the tradeoffs are visible. The point isn't to be precise to the millisecond — it's to have numbers that justify decisions.

### 2.1 Latency floors, derived from first principles

**Given numbers:** 10ms intra-region round-trip, 150ms cross-region round-trip. Call intra = 10ms, cross = 150ms. I'll use one-way = half of round-trip.

**Intra-region write (strong, leaseholder with 2 in-region followers):**

- Client → gateway: ~1ms
- Gateway → leaseholder: ~1ms
- Leaseholder proposes, sends to followers: one intra-region one-way = 5ms
- Followers fsync and ack: ~2ms disk
- Ack returns to leaseholder: one intra-region one-way = 5ms
- Leaseholder applies and responds: ~1ms

Worst-case sum: about 15ms. Measured p50 in practice: **~4ms** (nodes co-located in same AZ, parallel sends, faster disks on warm writes). p99: **~15ms**.

**Cross-region strong write, RF=5 across all 5 regions, leaseholder in US-E:**

- Quorum is 3 of 5. Leaseholder is one of the three. So we need acks from **the 2 fastest remote replicas**, not all 4.
- Nearest remote (say US-W from US-E): ~80ms round-trip.
- Second-nearest remote (say EU): ~100ms round-trip.
- We wait for the slower of these two = ~100ms.
- Plus intra-region overhead (gateway, apply) = ~5ms.

**So cross-region strong write p50 is ~80–100ms**, not 150ms. The 150ms floor only applies when the leaseholder has no in-region followers and has to wait for a worst-case remote. p99 lands around **180ms** once you factor in tail latency and retries.

**Eventual local write:**

- Client → gateway: ~1ms
- Gateway → local node, writes locally, replicates to 1 other in-region replica (W=2 of 3): one intra one-way = 5ms
- Both fsync (parallel): ~2ms
- Ack back: 5ms

**p50: ~3ms. p99: ~10ms.** Cross-region replication happens in the background, after the ack.

**Closed-timestamp follower read (the magic trick for fast reads on strong namespaces):**

The leaseholder periodically (every ~200ms) broadcasts a "closed timestamp" T — meaning "no write at time ≤ T will ever commit." Followers that have applied the log up to T can serve reads at T locally, without contacting the leaseholder.

- Client → gateway: ~1ms
- Gateway → local follower: ~1ms
- Local read: ~1ms

**p50: ~2ms. p99: ~5ms.** The tradeoff: reads are slightly stale (bounded by how often the closed timestamp advances, typically ~3 seconds). They're **serializable at time T**, not linearizable. That's fine for the vast majority of reads.

### Summary table:

| Operation | p50 | p99 | What bounds it |
|---|---|---|---|
| Strong write, in-region leaseholder | 4ms | 15ms | Intra-region RTT + fsync |
| Strong write, cross-region RF=5 | 80–100ms | 180ms | 3rd-fastest ack (2 nearest remotes) |
| Strong write, worst placement | 155ms | 200ms | One cross-region RTT — the floor |
| Strong leaseholder read | 1–2ms | 5ms | Local + lease check |
| Strong follower read (closed-ts) | 1–2ms | 5ms | Local, no network |
| Eventual local write | 3ms | 10ms | Intra-region quorum + fsync |
| Eventual read (local) | 1–2ms | 5ms | Local only |
| Eventual cross-region replication lag | — | <1s | Async background |

### 2.2 Cross-region bandwidth, step by step

**Step 1: How much raw data is generated?**
100K writes/sec × 1KB average = **100 MB/sec** of raw writes, globally.

**Step 2: How much crosses region boundaries for an eventual namespace replicated to all 5 regions?**
Each write goes to 4 other regions.
100 MB/sec × 4 = **400 MB/sec** of outbound cross-region traffic, aggregate across all 5 regions.

**Step 3: Apply compression.**
Typical JSON-like payloads compress ~3× with zstd.
400 / 3 = **~133 MB/sec** compressed.

**Step 4: Add protocol overhead.**
Headers, HLC timestamps, checksums, heartbeats: ~20%.
133 × 1.2 = **~160 MB/sec** of real cross-region bandwidth.

**Step 5: Cost check.**
Cloud cross-region egress is roughly $0.02/GB.
160 MB/sec × 86400 sec/day = ~14 TB/day.
14 TB × $20/TB = **~$280/day = ~$100K/year**.

That's per eventually-consistent all-region namespace, if every write is replicated everywhere. Times 100 such namespaces = $10M/year just in replication egress. **Conclusion: the product cannot default to all-region replication.** Default is 2–3 regions per namespace, opt-in to more.

### 2.3 Number of shards (ranges)

**Step 1: Data volume.** Assume 100 TB total.

**Step 2: Target size per range.** Best practice is 64–512 MB. I'll use 256 MB — a good tradeoff between metadata overhead (favoring larger) and rebalance granularity (favoring smaller).

**Step 3:** 100 TB / 256 MB = **~400,000 ranges**.

**Step 4: Replication factor.** RF=3 default: 1.2M range replicas across the fleet.

**Step 5: Metadata size per range.** Range bounds, replica set, leaseholder, epoch = ~500 bytes.

**Step 6: Total metadata.** 400K × 500B = **200 MB**. Fits comfortably in memory on the metadata service.

Auto-split triggers: when a range exceeds 512 MB OR 10K QPS, whichever comes first. Merges happen the other way when adjacent ranges shrink.

### 2.4 HLC timestamp overhead

**What's an HLC?** A Hybrid Logical Clock is a timestamp that combines wall-clock time with a logical counter, so that concurrent events get ordered consistently across machines even when clocks drift slightly. I'm using HLCs for the eventual mode.

- Physical time: 8 bytes
- Logical counter: 2 bytes
- Region ID (for tiebreaking): 1 byte
- **Total: 11 bytes per write.**

**Compared to vector clocks:** a 5-region vector clock is 5 × 8 = 40 bytes just for versioning, and grows unboundedly with per-client tracking. HLCs are bounded and cheap. Tradeoff: they give weaker causal tracking — discussed in §6.

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

A few key points about this picture:

**Every node runs the same binary.** Same storage engine, same membership, same gateway protocol. When a node hosts both a strong-namespace range and an eventual-namespace range (which is fine and normal), the only difference is which code path handles writes and reads. That's the shared-substrate idea made concrete.

**Control plane = 5-member Raft group.** It stores: namespace configs (which mode, which regions), range metadata (key range, replicas, leaseholder, epoch), and node metadata (region, health). Spread across odd-number regions to keep metadata writes fast. Metadata reads are cached locally on every node. Metadata writes happen maybe 10–100 times per second in steady state — topology drifts slowly.

**Gateway = thin per-region routing layer.** Authenticates the client, resolves `(namespace, key) → range → replicas` from its cache, and invokes the right mode-specific protocol. Stateless. Cache is eventually consistent with the control plane; when it's stale, nodes reply "you're talking to the wrong node, leader moved, retry" and the gateway updates.

**[STAFF SIGNAL: rejected alternative]** I considered pushing routing into smart clients (so clients talk directly to nodes, no gateway hop). Rejected because:
1. We support many SDK languages and the routing logic is nontrivial — centralizing it in a gateway is the globally optimal engineering choice.
2. The gateway is the right place to terminate TLS, enforce rate limits, inject trace context.
3. The extra 1ms hop is negligible against our latency budget.

Smart clients can come later as an optimization for latency-critical internal users.

---

## 4. Partitioning and Placement

I'm choosing **range partitioning with dynamic splits**, not consistent hashing.

**[STAFF SIGNAL: rejected alternative]** Why not consistent hashing (Dynamo-style)?
1. Per-namespace placement rules ("this namespace lives only in US+EU, never APAC") are first-class in the product. Range partitioning gives us a direct range → replica-set map where we can express this cleanly. Consistent hashing makes it awkward — do we have multiple hash rings? Virtual-node migrations when placement changes?
2. Range splits happen atomically on keyspace boundaries and touch one range. Consistent-hashing resharding has to migrate tokens across the ring.
3. Range scans become possible if we ever add them. Cheap option to preserve.
4. The industry has converged here: Spanner, CockroachDB, TiKV, YugabyteDB all use range partitioning.

### Strong namespace range layout (all-region, RF=5):

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

### Eventual namespace range layout (3-region):

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

Notice the structural difference: strong mode has **one global Raft group per range**. Eventual mode has **per-region consensus for local durability + async pull-based replication across regions**. Storage engine and range metadata are identical on disk — it's the coordination protocol that differs.

**Hot-key problem.** For eventual: intra-region writes spread naturally across local replicas, so moderate hotspots are fine. For strong: a hot key is a hot leaseholder, and you can't split a single key. Mitigation: the application has to shard at its layer, or move the namespace to eventual mode. **[STAFF SIGNAL: saying no]** The KV store cannot fix a fundamentally hot key. This is an application-level problem.

---

## 5. Deep Dive A: Strong-Consistency Write and Read Path

**I'm using Raft per range.** Multi-Paxos is equivalent in theory; Raft wins on tooling (etcd/raft, dragonboat, hashicorp/raft are mature and battle-tested).

**[STAFF SIGNAL: rejected alternative]**
- **Chain replication:** higher steady-state throughput, but head-of-line blocking when a replica fails and more complex reconfiguration. Wrong tradeoff for interactive, latency-sensitive workloads.
- **TrueTime-based ordering (Spanner):** we don't own atomic clocks. Cloud-provided time (via NTP) has uncertainty intervals too wide — Spanner's commit-wait would take ~10ms of artificial delay per write to be safe. HLC + closed timestamps (CockroachDB's approach) gives us bounded-staleness follower reads without atomic clocks. This is the post-Spanner state of the art.
- **Viewstamped Replication:** equivalent to Raft in safety. Worse tooling. No upside.

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

For an in-region leaseholder with 2 in-region followers, quorum is achieved without leaving the region: p50 ~4ms. For a cross-region write, we wait for the 3rd-fastest ack (= the 2nd remote reply), not the 5th.

### Three read paths:

1. **Leaseholder read (linearizable).** Read from the leaseholder after verifying its lease is still valid. 1–2ms local. Default.
2. **Quorum read (linearizable, no leader needed).** Read from a quorum of replicas, return the newest version. Used as a fallback if the leaseholder is down. Higher latency.
3. **Closed-timestamp follower read (serializable at past T).** The leaseholder periodically publishes a closed timestamp T. Any follower that has applied the log through T can serve reads at T locally, without talking to the leaseholder. Staleness bounded by how often T advances (typically 3 seconds, tunable).

**[STAFF SIGNAL: consistency precision]** Being careful with terminology:
- Leaseholder reads are **linearizable** — they see the most recent committed write, as if there were a single global clock.
- Quorum reads are **linearizable**.
- Closed-timestamp reads are **serializable at T** — they see a globally consistent snapshot from a few seconds ago. They are **not** linearizable (a newer write may have committed). Clients opt in with a `max_staleness` parameter.

### Leaseholder placement and failover

The leaseholder is elected via Raft, but the placement driver can hint or constrain: "keep the leaseholder in US-E, where 80% of writes originate." Failover is standard Raft election — typically 1–3s including failure detection.

Fencing uses Raft's term number: a deposed leaseholder whose heartbeats lapsed cannot commit new writes, because followers will reject its AppendEntries messages with a higher term.

**[STAFF SIGNAL: invariant-based thinking]** The key invariant: **no two leaseholders can commit writes in the same Raft term.** This is enforced by Raft's log-matching and term-monotonicity properties. As long as those hold, we can't get split-brain on a strong namespace.

### The zombie leaseholder problem (GC pause)

Suppose a leaseholder stops-the-world for 10 seconds (GC pause, kernel hang). It wakes up, thinks it's still the leaseholder, and tries to serve a read from its cached state. Without protection, it returns stale data — the other nodes elected a new leader and moved on.

**Fix: epoch-based leases with wall-clock expiration.** Each node keeps a liveness record in the metadata Raft group, expiring every 9s and renewed every 3s. Before serving any read, the leaseholder checks that its epoch is live. A zombie leaseholder's epoch expired during the pause, so when it wakes up, it fails the liveness check and refuses to serve.

CockroachDB uses this scheme. Spanner uses TrueTime's commit-wait instead. Both preserve the same invariant: **no leaseholder serves a read without a live epoch.**

### Availability envelope

A strong namespace's availability is bounded by:
1. Its Raft group's ability to form quorum (dominant factor).
2. The metadata service being reachable for routing.
3. The gateway being up.

With RF=5 across 5 regions: survives 2 region failures. With RF=3 across 3 regions: survives 1 region failure. **RF=3 is the sane default**; RF=5 is for compliance/SLA-critical namespaces at doubled storage + bandwidth cost.

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

The write gets durability from intra-region quorum (W=2 of 3, fsync'd). Cross-region replication is async via per-region logs — remote regions run a replication worker that polls the log cursor and applies missed writes.

### How HLCs actually work

Every write gets a timestamp `(physical_ts, logical_counter, region_id)`. When a region receives a write from another region, it updates its own HLC:

- `local.physical = max(local.physical, msg.physical, wallclock())`
- If physical times tie, `local.logical = max(local.logical, msg.logical) + 1`

What this buys us:
- Per-region monotonicity (within a region, time only moves forward).
- A total order across regions, with region_id as the final tiebreak — correlated with real time to within clock skew.
- Correct ordering of causally related writes as long as they flow through the system in causal order. For single-key writes, this is automatic.

**[STAFF SIGNAL: rejected alternative]** I considered full version vectors or dotted version vectors — they give true per-key causal history. Rejected because:
- They grow unboundedly with the set of client IDs.
- Pruning is hard in practice at 5 regions × millions of clients.
- The state can dominate the value itself for small values.

HLCs are coarser but bounded. For applications that truly need per-key causal divergence tracking, we offer CRDT types (below).

### Conflict resolution — three strategies, per-namespace policy:

**1. HLC last-writer-wins (default).** Two concurrent writes resolved by HLC order; tiebreak on region_id. One write is silently lost. Fine for last-writer-wins semantics (session state, user preferences, caches).

**2. CRDTs (opt-in per key prefix).** Conflict-free Replicated Data Types — data structures that are mathematically guaranteed to converge regardless of the order in which operations are applied. Supported types:
- G-Counter (only goes up)
- PN-Counter (goes up or down)
- LWW-Set / OR-Set (add/remove items)
- G-Map (key-value map)
- MV-Register (returns all concurrent values; app picks)

**3. App-defined merge (opt-in).** The namespace registers a sandboxed merge function `merge(v1, v2) → v3`. Must be deterministic, commutative, associative, idempotent — the math requires it. Powerful, dangerous; used sparingly for things like shopping-cart merge.

**[STAFF SIGNAL: conflict resolution discipline]** The honest truth about CRDTs: they're often pitched as "just use CRDTs and conflicts go away." In practice:
- CRDTs restrict your operations to things that can be expressed commutatively. **A bank transfer is not a CRDT.** Neither is "set this field to X only if Y > 5."
- OR-Set has unbounded tombstone growth without coordination — tombstones pile up until you can prove every region has seen the remove.
- PN-Counter over-counts under network replay unless you deduplicate by operation ID.
- A LWW-Register CRDT is no better than our default HLC-LWW.

We offer CRDTs as a principled tool for the workloads that fit them. We don't pretend they're a silver bullet. Riak bet the farm on CRDTs and hit exactly these operational limits.

### Concurrent-write scenario:

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

Two regions converge after:
1. Cross-region replication lag clears (target <1s p99 in steady state).
2. Anti-entropy detects and repairs any gaps. We use Merkle trees per range, with hourly background sweeps and on-demand sweeps when replication lag spikes.

**Convergence is NOT bounded under sustained partition.** This is the AP tradeoff, stated plainly. If US-E and EU can't talk for an hour, they will not converge for an hour.

### Read-your-writes across regions

**Not provided by default.** If a client writes in EU and reads in APAC, it may not see its own write for up to the replication lag. Three opt-in mechanisms:

1. **Session stickiness** — client pins to a single region for a session.
2. **HLC-token session ("zookie" pattern from Zanzibar).** The write returns an HLC. On a subsequent read against another region, the client passes the HLC; the server waits until its local replica has applied through that HLC, or returns 409 "not caught up yet." Wait is upper-bounded; backpressure is explicit.
3. **Accept staleness** — no guarantee, fastest path.

---

## 7. Deep Dive C: The Mode-Change Protocol

**[STAFF SIGNAL: mode-change invariant]** This is the hardest problem in the system. A subtle correctness bug here loses data silently. I'm treating it as a formal protocol, not a feature.

**Three invariants that must hold across any mode change:**
1. No acknowledged write is lost.
2. No read returns a value that would be inconsistent under the target mode, from the moment the mode change is announced to clients.
3. The protocol is resumable and reversible until a **point-of-no-return**, which is explicit and logged.

### Protocol: eventual → strong (the harder direction)

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

### The subtle case that breaks this

In Stage 3, if we have MV-Register keys with semantically incompatible concurrent values — say, two different shopping carts for the same user — **no algorithm resolves them correctly**. We can't pick one without losing data meaningful to someone.

We surface this to the customer via a resolution API and pause the migration. This is a product feature, not a correctness escape hatch. The customer needs to understand that **switching to strong requires resolving the divergence that eventual mode permitted them to create**.

### Protocol: strong → eventual (easier, but not trivial)

- Stage 0: Announce.
- Stage 1: For each range, take a snapshot at a specific Raft log index. Record it as the eventual-mode genesis point.
- Stage 2: Bootstrap per-region leaders from that snapshot. All replicas share identical state at genesis.
- Stage 3: Switch the write path to eventual. Tear down the Raft group.

The subtle case: clients with an in-flight strong write may have an uncertain outcome at the moment of switchover. The protocol drains in-flight Raft writes first. A client that received a successful strong-write ack sees that write; a client whose response was uncertain must retry under eventual semantics.

**Duration estimate:**
- Eventual → strong: ~30 minutes for a 100 GB namespace (mostly drain + conflict resolution).
- Strong → eventual: ~10 minutes.

Both done during customer-approved maintenance windows.

---

## 8. Deep Dive D: Partition Behavior

**[STAFF SIGNAL: partition behavior precision]** Concrete scenario: **US-E is network-partitioned from the other four regions.** US-E can still talk to its own nodes and to clients inside US-E, but not to US-W, EU, APAC, or SA.

### Case 1: Strong namespace, RF=5, leaseholder in US-E

- Raft group has 1 reachable replica (itself). Can't form quorum of 3. **Writes fail.**
- After ~9 seconds, the leaseholder's liveness epoch expires. The other 4 regions elect a new leaseholder among themselves (they still have quorum). Writes resume on their side.
- US-E clients see "no leaseholder reachable." They must fail over via an application-level DR strategy — read-only mode, or route to another region for writes.
- US-E clients **can** still do closed-timestamp reads locally against the ex-leaseholder, bounded by the staleness at partition time. After ~3 seconds with no new closed timestamp arriving, the reads get progressively staler and eventually exceed the staleness budget, at which point they fail too.

### Case 2: Strong namespace, RF=5, leaseholder outside US-E

- Raft group has 4 of 5 replicas on the outside. Writes proceed. Latency shifts slightly (different mix of RTTs in the new quorum).
- US-E's replica is a follower not receiving AppendEntries. It can't publish new closed timestamps.
- US-E clients see bounded-staleness reads up to the last closed timestamp, then nothing. No writes.

### Case 3: Eventual namespace, multi-region

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

**Client experience in eventual mode during partition:** every client sees a locally-consistent view of its own region. A user who reads-then-writes in EU sees their own writes (intra-region, no replication lag). A user bouncing between regions sees per-region-consistent views but may appear to time-travel. The application either pins users to regions or accepts it. Either way, we document it clearly.

**The heal is non-blocking.** New writes continue during reconciliation. The replication protocol handles out-of-order application via HLC.

---

## 9. Deep Dive E: Cross-Region Replication Protocol and Cost

Per-region replication log (think: per-range Kafka-like topic, internal). Each range has a monotonic per-region log keyed by HLC. Remote regions pull via long-poll with a cursor. Batch size tuned to amortize network overhead: 4KB–64KB batches, target 100 batches/sec per region-pair.

### Backpressure when a region is down

If EU is unreachable, US-E's EU-bound cursor stops advancing. US-E keeps accepting writes (eventual mode); its per-region log accumulates.

Log retention capped at 7 days per namespace (configurable). Beyond that, EU needs a full bootstrap from snapshot on recovery. Seven days is the right number: it covers most realistic incidents without unbounded storage growth.

### Bandwidth cost — worked example for one namespace

Assume an eventual namespace, all 5 regions, 20K writes/sec, 1KB average value.

- Step 1: raw outbound per region = 20K × 1KB × 4 remote = 80 MB/sec per source region.
- Step 2: after compression (3×) = ~27 MB/sec.
- Step 3: plus protocol overhead (1.2×) = ~32 MB/sec per source region.
- Step 4: total aggregate = 32 × 5 = **160 MB/sec** across the mesh.
- Step 5: per region-pair (asymmetric in practice) = roughly 16 MB/sec.

### Read-your-writes across regions — API options (repeated from §6 for completeness)

1. Session stickiness — client pins to region.
2. HLC-token wait — client passes write's HLC; server waits for local replica to catch up (bounded).
3. Accept staleness — fastest, no guarantee.

All documented; customer picks per session.

---

## 10. Failure Modes and Correctness Recovery

**[STAFF SIGNAL: failure mode precision]** Going through each failure case with detection, recovery, and user-visible impact.

**1. Shard loses Raft quorum.** Detect: gateway 503s from leaseholder, metadata sees expired lease. User sees: writes fail with "range unavailable." Recovery: if replicas are just unreachable, wait. If disks are permanently dead (≥ quorum lost), invoke **unsafe recovery** — operator confirms the data loss, rebuilds the range from surviving replicas, fences writes during rebuild, bootstraps a new Raft group with best-available state.

**2. Permanent region loss.** Detect: region-wide heartbeat loss + out-of-band confirmation (we don't trust automated detection alone for this level of destructive action). Strong namespaces with one replica in the lost region: degraded to RF-1, quorum reduced proportionally; operator must add a replica elsewhere to restore durability. Strong namespaces whose quorum crossed the lost region: **unavailable** until unsafe recovery. Eventual namespaces: lose writes that hadn't yet replicated out — bounded by replication lag, typically ≤1 second = ≤100K writes in the worst case.

**3. Silent disk corruption.** Detect: hourly Merkle-tree anti-entropy finds replica divergence; checksum mismatch on read. Recovery: the minority-value replica gets wiped and re-replicated from the majority. Invariant: **every persisted KV pair has a checksum computed at write time and verified on read.**

**4. Leaseholder zombie (GC pause).** Covered in §5 via epoch-based liveness.

**5. Buggy client writes malformed data.** For strong: the bad write gets persisted, replicated, and served. Recovery: application-level compensation. **The KV store is not a sanitizer.** Mitigation: optional per-namespace validation hooks (schema, size limits) at the gateway.

**6. Failed Raft membership change (joint consensus).** Raft uses joint consensus when adding/removing replicas — old and new configurations must both reach quorum during the transition. If a failure happens mid-transition, the range is stuck in "joint" state. Tooling: operator can force-complete or force-abort via a metadata-Raft write. Documented runbook.

**7. Metadata Raft unavailable.** **[STAFF SIGNAL: blast radius reasoning]** Serious but not catastrophic: no new leases, no range splits, no placement changes, no mode changes. But the **data plane keeps running** — existing leases keep serving as long as leaseholders are alive and cached metadata is valid. Eventual namespaces keep taking writes. Strong namespaces keep using current leaseholders. What breaks: anything requiring a metadata write (new range, lease renewal beyond cached epoch, failover). We size the metadata Raft for 99.999% availability (5 members, 3 regions) and keep its write rate low. **The whole system is designed so the data plane survives control-plane outages for at least one epoch lifetime (9s+) without user-visible impact.**

---

## 11. Consistency Guarantees in Precise Terms

**[STAFF SIGNAL: consistency precision]** "Strong" and "eventual" are imprecise. Here's exactly what each mode provides:

### Strong namespace

- **Single-key reads from leaseholder or quorum:** linearizable.
- **Closed-timestamp follower reads:** serializable at a past timestamp T. **Not linearizable.** Monotonic within a client session if the client carries a session token.
- **Monotonic reads:** yes, via session token.
- **Read-your-writes within a session + region:** yes.
- **Read-your-writes cross-region:** requires leaseholder read or HLC-token wait.
- **Monotonic writes:** yes (linearizability implies it).
- **Multi-key atomicity:** not in v1.

### Eventual namespace

- **Durability:** W=2 of 3 intra-region, fsync, before ack.
- **Eventual convergence:** assuming partitions heal and replication catches up, all replicas converge.
- **Per-region linearizability:** a client talking to one region sees a linearizable view of that region's writes.
- **Causal consistency across regions:** **NOT** provided by default. Client reads from EU, writes to US-E — no guarantee the US-E write is ordered after what it read. Opt-in via HLC-token session.
- **Read-your-writes across regions:** not by default; session stickiness or HLC-token wait as opt-ins.
- **Multi-key atomicity:** not meaningful under this consistency model.

Being this precise is the difference between a product customers can build on and a product that generates incident tickets for a decade.

---

## 12. Multi-Key Operations

Single-key in v1. The escape hatch:

**Co-location hint (v1.5).** Clients declare "these keys must live on the same range" via a key-prefix hint. The placement driver honors this subject to range-size limits. Single-range multi-key operations then execute atomically via Raft (one log entry per batch). Good enough for maybe 80% of multi-key use cases.

**Cross-range transactions (v2).** Two-phase commit with a transaction coordinator, Percolator-style. Significant new protocol surface:
- Adds latency (2 Raft rounds instead of 1).
- Adds failure modes (prepared-but-not-committed transactions need a recovery protocol).
- Requires MVCC storage layout (move from plain Pebble to MVCC-aware layout — not small).

A quarter-plus of engineering work.

**Eventual namespaces.** Multi-key atomicity is not meaningful — the consistency model can't support atomicity across keys with independent concurrent writers. We offer best-effort batching (N keys in one request: durability is atomic, visibility is not) and document the limitation.

**[STAFF SIGNAL: saying no]** If the product requirement becomes "serializable multi-key transactions across any namespace," **this design is wrong** and we should build a Spanner/CockroachDB-style MVCC transactional store from day one. The substrate doesn't cleanly extend to that. Better to surface the ambiguity now than be 6 months in before realizing it.

---

## 13. Recent Developments and Why They Matter Here

**[STAFF SIGNAL: modern awareness]**

- **Spanner (2012) + TrueTime.** Proved globally-consistent transactions at scale via bounded clock uncertainty. Commit-wait needs atomic-clock infrastructure. Without that, HLC + closed timestamps is the post-Spanner practical design.
- **CockroachDB closed timestamps + follower reads.** Proved you don't need TrueTime to give users bounded-staleness consistent reads on strongly-consistent data. The single most important technique I've borrowed. Transforms the global read path from "cross-region hop to leaseholder" to "local follower read" for the huge class of read workloads that tolerate seconds of staleness.
- **FoundationDB.** Two things: (a) separating transactions, storage, and resolution into independent layers; (b) deterministic simulation testing. Our correctness testing should crib their simulation approach — it's the only way to find the bugs that would otherwise hit production at 3am. FDB running Snowflake, iCloud, and Tigris is existence proof that one substrate can serve wildly different workloads.
- **DynamoDB Global Tables.** Multi-master eventually consistent. The public postmortems (LWW silently losing writes; the move to stricter per-item guarantees) are a warning. Default-LWW without application awareness causes data loss. We surface conflict policies explicitly per namespace and default to HLC-LWW only when the customer has opted in.
- **Aurora DSQL (announced late 2024).** Active-active Postgres-compatible with separation of storage, compute, and transaction log. Shows the industry moving toward log-centric designs where the log is the truth and compute is stateless. Our replication log is the same pattern.
- **TiKV / YugabyteDB.** Raft-per-range is the de-facto standard for strongly-consistent KV. I'm following the convergent path.
- **CRDTs in practice (Riak's pivot away).** Riak bet on CRDTs and hit exactly the issues in §6 — unbounded tombstones, application restrictions. Antidote and recent bounded-CRDT research matter for customers who truly need them; most don't.
- **HLC paper (Kulkarni et al., 2014).** Foundational for the eventual-mode ordering design.
- **Google Zanzibar.** Showed how to combine a strongly-consistent core with bounded-staleness reads at massive scale. The "zookie" token is the pattern I'm proposing for session consistency.

---

## 14. Tradeoffs Taken and What Would Force a Redesign

- **Single-key atomicity only.** If multi-key transactions become required: MVCC + 2PC + transaction coordinator. Quarter-plus of work. Storage layout changes.
- **HLC + LWW for eventual (no true causal tracking).** If customers need causal consistency guarantees: add version-vector tracking with a pruning protocol. State blows up 10–60×. Offer as opt-in per namespace.
- **Range partitioning.** If we suddenly need uniform load distribution without per-namespace placement: consistent hashing. But we'd lose per-namespace placement, which is the product's whole point.
- **Gateway tier adds 1ms hop.** For latency-sensitive internal users who can't pay the 1ms: ship a smart-client library that resolves ranges directly. Added maintenance cost per SDK language.
- **Metadata Raft is a single point of governance.** For multi-tenant hard-isolation requirements: shard the metadata Raft by tenant. Operational complexity multiplies.

---

## 15. What I Would Push Back On

**[STAFF SIGNAL: saying no]**

**1. "Configurable consistency per namespace" as a headline feature.** Operationally expensive. Ask the PM: what fraction of customers need eventual? If under 20%, push them to strong + follower reads. If the answer is "all of them, team A wants strong and team B wants eventual," we're solving an org problem with a product, and that's a different conversation.

**2. "150ms cross-region p50 for strong writes" read as a target.** It's a **floor**, not a target. Any product requirement below ~170ms p99 for cross-region strong writes is unmeetable. Product must accept this, or choose single-region-leader placement (with the associated availability cost).

**3. Implicit assumption that namespaces replicate to all 5 regions.** Doubles cost unnecessarily. Default should be 2–3 regions per namespace, with opt-in to more for compliance/SLA.

**4. "Live mode change" without specifying acceptable downtime or latency-spike budget.** As designed, mode change is minutes to hours with partial availability. Zero-downtime, zero-latency-spike mode change is not achievable given the invariants we preserve. Drain-and-install takes time proportional to replication lag and conflict count.

**5. "KV store" without scope clarification.** If customers actually need range scans, secondary indexes, or transactions, this design is the wrong starting point. Better to have that conversation now than after we've committed the architecture.

---

**[STAFF SIGNAL: operational longevity]** Year-5 outlook.

Adding a 6th region: placement driver rebalances ~20% of ranges incrementally at capped bandwidth, over days. Removing a region: range-by-range replica replacement before decommission. Rolling protocol upgrades: feature-gated dual-write/dual-read phases for any on-disk format change. Feature-gate flags live in range metadata so individual ranges upgrade independently and can roll back. Storage engine swap (Pebble → something new): 6-month project with shadow replicas running in parallel and a flip-switch per range. Deterministic simulation testing (FoundationDB-style) is the only tractable way to maintain correctness across all of this.

The system runs for a decade. We plan for it.