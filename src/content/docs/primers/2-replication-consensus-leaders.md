---
title: Replication, Consensus, and Leader Election
description: Replication, Consensus, and Leader Election
---


## 1. What a Staff Engineer Actually Needs to Know

**What matters in interviews:**

- Reasoning clearly about **tradeoffs** (consistency vs availability vs latency vs cost).
- Knowing the **write path** and **failure path** of whatever system you invoke.
- Being able to say "this can fail like X, which we mitigate with Y."
- Naming the right primitive (quorum, lease, fencing token, term) instead of hand-waving.

**Expected depth:**

- Mental models of leader-follower, quorum, and Raft-style consensus.
- Concrete failure scenarios and their implications.
- Knowing when a design needs consensus vs when it's overkill.

**What does NOT matter** (unless the role is explicitly infra/DB/consensus):

- Raft implementation details (log compaction specifics, snapshot formats).
- Paxos variants (Multi-Paxos, Fast Paxos, EPaxos).
- Byzantine fault tolerance.
- FLP impossibility proof details.
- CAP formalism beyond the intuitive version.

The bar: **you can defend any choice and predict how it breaks.**

---

## 2. Core Mental Model

### What each primitive solves

|Primitive|Problem it solves|
|---|---|
|**Replication**|"I don't want to lose data or availability when a node dies."|
|**Consensus**|"Multiple nodes must agree on a single value/order despite failures, with no contradictions."|
|**Leader election**|"Pick one node to coordinate, and make sure everyone agrees on who it is."|

### Replication ≠ Consensus

Replication is _mechanism_: copy bytes to N machines. Consensus is a _guarantee_: committed decisions survive any minority failure and are never contradicted.

You can replicate without consensus (async MySQL replicas: fast, can lose data). You can have consensus without scaling reads (Raft with all reads through leader). Modern systems usually combine both: **consensus decides the write order, replication distributes the data.**

### "Who can safely commit writes" > "who is leader"

Anyone with a laptop can _believe_ they're leader. The real question is:

> **Who can get a write acknowledged such that it will survive any future failure?**

In Raft: only a leader whose term is the highest the majority has seen, and only after a majority persists the entry.

In async leader-follower: the leader can "commit" locally and still lose the write on failover. The word "commit" is doing dangerous work.

Always ask: _committed where, visible to whom, and durable under what failure?_

---

## 3. The Essential Models

### 3.1 Leader-Follower Replication (async or sync)

```
           writes
Client --------> [Leader] ---------> [Follower 1]
                   |       replicate
                   |       -------->  [Follower 2]
                   |       -------->  [Follower 3]
  reads (maybe) <--+
  reads (stale) <-----------------------from any follower
```

**Writes:** client → leader → leader writes local log → replicates to followers.

- **Async**: leader acks client immediately after local write. Followers catch up eventually.
- **Sync**: leader waits for N followers to ack before acking client.
- **Semi-sync** (common): wait for at least 1 follower ack, not all.

**Reads:**

- From leader → fresh (usually).
- From follower → possibly stale (lagged by ms to seconds to minutes).

**Strengths:**

- Simple, fast writes (async).
- Easy to scale reads horizontally.
- Well-understood operationally.

**Weaknesses:**

- **Async can lose committed writes on failover.**
- **Sync = worst-case latency of slowest required follower**, and availability drops if followers are down.
- **No built-in protection against split brain** — needs external coordination (ZooKeeper, etcd, managed failover).

**Failure behavior:**

- Leader dies → must promote a follower → possible data loss (async) or stall (sync).
- Failover is often manual or supervised by an external coordinator.

**What interviewers want to hear:**

- "Async replication means the client ack does not imply the write survives a leader crash."
- "Failover either loses data or requires synchronous replication, which costs latency."
- "We still need an external mechanism to prevent two leaders."

---

### 3.2 Quorum / Majority Replication (Dynamo-style)

N replicas. Each write must be accepted by **W** replicas. Each read must contact **R** replicas.

**Key invariant:** `W + R > N` guarantees read and write quorums intersect → at least one replica in the read set has the latest write.

```
N = 5, W = 3, R = 3     (any write quorum ∩ any read quorum ≥ 1)

write → [x][x][x][ ][ ]   (W=3 acks)
read  →    [x][x][x]       (R=3; overlaps ≥1 with write set)
```

**Writes:** client sends to coordinator → coordinator fans out → acks after W. **Reads:** coordinator fans out → picks newest value by timestamp/version → often read-repairs lagging replicas.

**Strengths:**

- No single point of failure — no explicit leader.
- Tunable: `W=1,R=N` for write-heavy; `W=N,R=1` for read-heavy; `W=R=(N+1)/2` for balanced.
- Survives minority failures transparently.

**Weaknesses:**

- **`W + R > N` gives you last-write-wins, not linearizability.** Concurrent writes to the same key need conflict resolution (LWW timestamps, vector clocks, CRDTs).
- **Sloppy quorums** (Dynamo) accept W acks from _any_ nodes including hinted handoff targets → can violate the intersection property.
- Read repair and anti-entropy add background cost.

**Failure behavior:**

- Up to `N - W` replicas can be down and writes still succeed.
- Up to `N - R` replicas can be down and reads still succeed.
- No failover event — the system degrades gracefully.

**What interviewers want to hear:**

- "`W + R > N` is necessary but not sufficient for linearizability."
- "Concurrent writes need explicit conflict resolution — timestamps lose data, CRDTs converge."
- "Sloppy quorums trade correctness for availability under partition."

---

### 3.3 Leader-Based Consensus (Raft mental model)

One elected leader. All writes go through it. Leader replicates a log to a majority before committing.

```
              append entry
Client ----> [Leader @ term T] ----> [Follower] ack
                    |                [Follower] ack   <-- majority
                    |                [Follower] ack      reached
                    |                [Follower]
                    v
                 COMMIT (apply to state machine)
                    |
                    v
               ack client
```

**Writes:**

1. Client → leader.
2. Leader appends to its log (not yet committed).
3. Leader replicates to followers.
4. Once **majority** (including leader) have persisted the entry → entry is **committed**.
5. Leader applies to state machine and acks client.

**Reads:**

- Strongly consistent reads must go through the leader **and** the leader must confirm it's still leader (via heartbeat round-trip or a leader lease).
- Stale reads from followers are possible but must be explicitly opted into.

**Strengths:**

- **No committed write is ever lost** as long as a majority survives.
- Automatic, safe failover.
- Strong ordering guarantees — the log is the source of truth.

**Weaknesses:**

- Needs a majority alive → can't tolerate `≥ ⌈N/2⌉` failures.
- Write latency = round-trip to the slowest replica in the majority.
- Leader is a throughput bottleneck (all writes serialized).
- Cross-region consensus is brutal (inter-region RTT on every write).

**Failure behavior:**

- Leader crashes → followers time out → election → new leader with the most up-to-date log wins → no committed entry is lost.
- Minority partition stalls (can't form majority). Majority partition continues.

**What interviewers want to hear:**

- "Writes commit only after a majority persists them."
- "Terms/epochs prevent a stale leader from committing after a network partition heals."
- "Election requires the candidate's log to be at least as up-to-date as the majority."
- "Reads need a mechanism — lease or read-index — to avoid serving stale data from a deposed leader."

---

## 4. Must-Know Concepts

**Quorum** — any subset of replicas whose participation is required for an operation. Defined per operation (read quorum, write quorum).

**Majority** — quorum of size `⌊N/2⌋ + 1`. The smallest size where any two quorums intersect. Why consensus protocols pick it: intersection guarantees durability across failures.

**Commit** — a write is _committed_ when the system guarantees it survives the failures it's designed to tolerate. Local fsync is NOT a commit in a replicated system. "My disk has it" ≠ "the cluster has it."

**Replica lag** — time between a write being committed on the leader and visible on a follower. Typically ms in-region, can blow up to seconds under load or minutes under GC/network issues.

**Stale reads** — reads served from a replica that hasn't applied the latest committed writes. Bounded by replica lag. In quorum systems, requires `R < W + R - N` to be a risk.

**Failover** — promoting a replica to leader after the current leader fails. May or may not be safe depending on replication mode.

**Split brain** — two nodes simultaneously believe they are leader and accept writes. Requires external prevention (fencing, leases, consensus on leadership).

**Term / epoch** — monotonically increasing integer assigned to each leadership period. Every message carries it. Stale leaders carry stale terms → followers reject them. This is how Raft prevents old leaders from committing.

**Lease** — a time-bounded exclusive grant. "You are the leader until time T." After T, the lease must be renewed. Enables safe reads from leader (no round-trip per read) at the cost of an availability gap on failover (must wait out the old lease).

**Fencing token** — monotonically increasing ID attached to every operation sent to downstream systems. Downstream rejects any operation with a token lower than the highest it's seen.

```
Old leader (token=17) ---stale write---> [Storage: last seen token=23] --> REJECTED
New leader (token=24) ---write--------> [Storage: last seen token=23] --> ACCEPTED, updates to 24
```

Without fencing, a zombie leader recovering from a GC pause can corrupt state.

**Durability vs availability** — durable = will not be lost under the failure model. Available = can serve reads/writes now. Synchronous replication buys durability at the cost of availability (any slow replica blocks you). Async replication buys availability at the cost of durability under failover.

**Linearizable vs stale/eventual reads** —

- **Linearizable**: every read sees the latest committed write; system behaves like a single machine. Requires reads to go through the commit path (leader + confirmation, or read quorum in a consensus system).
- **Eventual**: reads may be stale by some bounded or unbounded amount; will converge if writes stop. Much cheaper, sufficient for most user-facing reads.

---

## 5. Leader Election and Failover

### The mechanics

```
[Follower] <--heartbeat-- [Leader]
    |
    | no heartbeat for election_timeout (e.g. 150-300ms)
    v
[Candidate: term++, vote for self, request votes]
    |
    | majority votes received (with up-to-date log check)
    v
[New Leader: sends heartbeats at term+1]
```

- **Heartbeats**: periodic no-op messages from leader (e.g. every 50ms).
- **Election timeout**: randomized per-follower (e.g. 150-300ms) to avoid simultaneous candidacies.
- **Vote requirement**: candidate must have a log at least as up-to-date as the voter's. This is what guarantees the new leader has every committed entry.

### Why old leaders are dangerous

A leader that's been network-partitioned or GC-paused may still believe it's leader. Meanwhile, the rest of the cluster elected a new one. If the old leader wakes up and tries to serve writes or reads, you have two leaders.

Consensus protocols handle this internally: the old leader's term is now stale, followers reject its messages, it steps down. But **downstream systems don't know about terms** unless you tell them.

### Split brain from the observer's POV

Even Raft can look like split brain _externally_ during the transition window:

```
t0: Old leader alive, happily serving
t1: Network partition
t2: Old leader still thinks it's leader, serves a stale read or a write to storage
t3: New leader elected on the other side
t4: Old leader gets a response and finally realizes it's deposed
```

Between t1 and t4, two "leaders" existed from the system's observational standpoint. The protocol guarantees no committed write is contradicted, but a naive downstream system (a cache, a storage layer, an external API) can still be corrupted by the old leader's actions.

### How leases and fencing protect downstream

- **Leader lease**: old leader refuses to serve after lease expiry, even if it hasn't heard it was deposed. Closes the reasoning gap.
- **Fencing token**: downstream rejects old-leader operations by token comparison. Closes the action gap.

Use **both**: lease bounds how long the old leader can _try_, fencing ensures downstreams reject what slips through.

### Strong interview answer during failover discussion

Mention, in order:

1. **Detection** — heartbeat timeout.
2. **Election** — randomized timeout, majority votes, up-to-date-log requirement.
3. **Data safety** — new leader has every committed entry; uncommitted entries may be truncated.
4. **Availability gap** — a few hundred ms to a few seconds, depending on timeouts.
5. **Downstream protection** — fencing tokens for anything the leader writes to outside the consensus group.
6. **Client impact** — in-flight writes need idempotent retries; clients see transient errors.

---

## 6. Common Failure Scenarios

### Leader crash

- **Async replication**: the last few writes the leader acked may not have reached followers → lost on promotion.
- **Sync replication**: no data loss, but the cluster stalled while the leader was slow/dead until timeout kicks in.
- **Raft**: no committed write lost. Non-committed writes may be truncated from the new leader's log.

### Follower crash

- Mostly invisible. Leader continues. Cluster health degrades (one less vote / one less read source).
- In quorum systems, approaching `N - W + 1` failures kills writes.

### Network partition

- **Symmetric**: both sides can't talk. Minority side stalls (consensus) or diverges (async/quorum). Majority side continues (consensus) or both sides accept writes (async/quorum).
- **Asymmetric** (one-way): particularly nasty — leader thinks followers are fine but can't hear them. Hard to detect quickly.

### Slow node vs dead node

- **Impossible to distinguish** in a purely asynchronous network — this is a core tenet of distributed systems.
- Timeouts conflate the two. Tune them to your environment.
- A slow **leader** is worst: it holds up the cluster but doesn't trigger election quickly enough.

### Stale leader continuing to send writes

- Within the consensus group: term check rejects it.
- To external systems: _only fencing tokens save you._ If you're not passing a monotonically increasing token, you will eventually have a corruption incident.

### Failover while replicas are behind

- Async: new leader may lack the last N writes the old leader acked. Data loss, and the client thinks those writes succeeded.
- This is why "committed" must mean "on a majority" for strong guarantees.

### Lost ack vs lost write confusion

- A write timed out. Did it happen or not?
- **You don't know.** The write may have been applied just before the leader died.
- Clients MUST treat writes as **idempotent** (use a client-generated write ID) so retries are safe.

---

## 7. Interview Reasoning Patterns

### When is simple leader-follower enough?

- Read-heavy workload, stale reads acceptable.
- Failover downtime measured in seconds-to-minutes is OK.
- You can accept rare data loss on leader crash (or can use semi-sync for the critical writes).
- Examples: read-replica MySQL/Postgres for analytics or app reads, cache clusters, session stores with regeneration.

### When do you need consensus?

- Losing a committed write is unacceptable (payments, account state, configuration).
- You need a **single coordinator/metadata store** that must never disagree with itself (scheduler, shard manager, lock service).
- You need automatic failover without operator intervention.
- Examples: etcd/ZooKeeper/Consul, Spanner's directory metadata, Kafka controller (now KRaft), database cluster membership.

### When are stale reads acceptable?

- User-facing reads where "recent" is fine: feeds, timelines, search results, recommendations, dashboards.
- Anything where the cost of a stale read is "user sees slightly old data for a few seconds."

### When are stale reads NOT acceptable?

- Read-your-own-writes expectations (use session stickiness or read from leader).
- Financial balances immediately after a transaction.
- Lock/lease checks (must be linearizable).
- Any decision-making based on current state (scheduling, resource allocation).

### How do you prevent split brain?

- **Consensus on leadership** (Raft/Paxos in a dedicated coordination service).
- **Leader leases** with clock bounds.
- **Fencing tokens** for all external side effects.
- _Never_ rely on "we use a leader" alone. Be specific about the mechanism.

### How do you reason about quorum?

- Durability: a write is safe if it's on a write quorum.
- Consistency: reads see latest write iff `W + R > N` AND no concurrent writes OR conflict resolution is defined.
- Availability: writes succeed iff `≥ W` replicas are reachable. Reads succeed iff `≥ R` reachable.
- Pick `W, R` based on read/write ratio and consistency needs.

### How do you trade off latency vs durability?

- Sync replication to K replicas → `p99_write ≈ p99_slowest_of_K`.
- Async → `p99_write ≈ local_write_time`, but window of potential loss.
- Per-write tunable (some systems let clients pick sync vs async per request).
- **Rule of thumb**: sync within an AZ, async across regions, unless you really need cross-region durability (and you'll pay for it in latency).

### What changes in multi-AZ vs multi-region?

**Multi-AZ** (within one region, ~1-2 ms RTT):

- Consensus works fine. Put replicas in 3 AZs → survive any single AZ failure.
- Default for serious production.

**Multi-region** (tens to hundreds of ms RTT):

- Cross-region consensus adds that RTT to every write — often unacceptable.
- Common patterns:
    - **Regional leader, async replication to other regions**: fast writes, DR-capable, possible data loss on region failure.
    - **Per-region consensus groups, shard by region**: each region owns its data, cross-region coordination only for the rare cross-shard case.
    - **Spanner-style global consensus with TrueTime**: expensive, correct, use only if you need global linearizability.

The mental trap: **saying "we'll use Raft across regions" without acknowledging the latency cost is an immediate credibility hit.**

---

## 8. Common Mistakes

- **Treating replication and consensus as the same thing.** "We'll replicate to 3 nodes so it's consistent." No — replication gives durability, consensus gives agreement.
- **Saying "use Raft" without explaining the write path, election, or failover.** Name-dropping is not an answer.
- **Assuming there is always exactly one leader.** During elections, partitions, and GC pauses, there can be zero or two from the observer's POV.
- **Ignoring stale reads and replica lag.** "We read from a follower" without saying how stale that read can be, or whether the feature can tolerate it.
- **Hand-waving split brain.** "We'll just make sure there's only one leader." How? With what mechanism? What happens during a partition?
- **Ignoring leases and fencing.** The moment a consensus-backed leader writes to an external system (S3, a SQL DB, a cache), fencing is mandatory. Candidates skip this constantly.
- **Confusing local fsync with replicated durability.** "The leader wrote it to disk" does not mean it'll survive the leader crashing permanently.
- **Forgetting idempotency.** Clients must retry on timeout, and retries must be safe. If you don't design for this, duplicate writes will happen.
- **Quorum math errors.** Saying `W + R > N` is sufficient for linearizability (it isn't). Forgetting that `W = 1, R = N` still gives you read-your-writes but not much else.
- **Using a single consensus group for too much.** A single Raft group has throughput limits (~10k-50k ops/s depending on implementation). Scale via sharding / multi-raft, not by scaling up one group.

---

## 9. Cheat Sheet

### Comparison table

|Property|Async Leader-Follower|Quorum Replication (Dynamo)|Consensus-Backed Leader (Raft)|
|---|---|---|---|
|**Consistency**|Eventual (followers), linearizable on leader|Tunable; LWW by default, not linearizable|Linearizable (through leader)|
|**Write latency**|Local write only|`W`-th fastest replica|Majority of replicas|
|**Write throughput**|High (leader-bound)|High (no single bottleneck)|Leader-bound, moderate|
|**Data loss on failover**|Possible (last few writes)|None if `W ≥ ⌈N/2⌉+1`|None|
|**Failover time**|Manual/supervised, seconds+|No failover (no leader)|Automatic, sub-second possible|
|**Split brain risk**|High without external coord|Lower; no leader to duplicate|Internal: none; external: need fencing|
|**Tolerates failures**|Any number of followers|Up to `N - W` / `N - R`|Up to `⌊(N-1)/2⌋`|
|**Complexity**|Low|Medium|High|
|**Best for**|Read scaling, relaxed durability|High availability, tunable consistency|Metadata, coordination, critical state|

### Decision framework

```
Can I lose a committed write?
├─ Yes  → async leader-follower (cheap, fast)
└─ No   → need majority-based replication
          │
          Does every write need global order?
          ├─ Yes → consensus (Raft/Paxos)
          └─ No  → quorum replication (Dynamo-style) with conflict resolution

Does write throughput exceed one leader's capacity?
├─ Yes → shard, then apply the above per shard
└─ No  → single group is fine

Cross-region?
├─ Can tolerate cross-region RTT per write → global consensus
└─ Cannot → regional leaders + async cross-region, shard by region if possible
```

### Failure-scenario checklist

For any replicated/coordinated component in your design, answer:

1. What happens if the leader crashes? (Detection, election, data loss window.)
2. What happens if a follower crashes? (Capacity, quorum, replica lag.)
3. What happens during a network partition? (Which side serves? Which stalls?)
4. What happens during a slow node / GC pause? (Timeouts, false-positive elections.)
5. How are stale leaders prevented from corrupting external state? (Terms, leases, fencing.)
6. What's the replica lag under normal and pathological load? (Stale read bound.)
7. How do clients handle a write timeout? (Idempotency, retry policy.)
8. What's the availability during failover? (Seconds of no writes? No reads?)
9. How does this scale? (Shard count, per-shard consensus group, rebalancing.)
10. What's the blast radius if coordination fails? (Entire service? One shard? Degraded mode?)

### 10 Likely Interview Questions with Strong Short Answers

**Q1: What's the difference between replication and consensus?** Replication copies data to multiple nodes for durability and read scaling. Consensus is an agreement protocol ensuring a set of nodes decides on a value that won't be contradicted even under failures. Replication is the mechanism; consensus provides the guarantee. You can replicate without consensus (async MySQL) but then you lose the "no contradictions" property.

**Q2: Why does Raft need a majority?** Because any two majorities intersect in at least one node. That intersection node carries any committed entry into the next term, so no committed entry is ever lost when a new leader is elected from a majority.

**Q3: How does Raft prevent a partitioned old leader from committing writes?** Every message carries a term number. When a new leader is elected, the term increments. The old leader, on rejoining, discovers a higher term and steps down. Followers reject any append from a lower term. External systems need fencing tokens to get the same guarantee.

**Q4: What's a fencing token and why do you need one?** A monotonically increasing ID the leader attaches to every operation it sends to external systems. The external system tracks the highest token it's seen and rejects any operation with a lower one. Without this, a GC-paused old leader that wakes up can corrupt external state before it realizes it was deposed.

**Q5: When is `W + R > N` enough, and when isn't it?** It's enough for "last-write-wins" semantics if writes are well-ordered (e.g. monotonic timestamps) and there are no concurrent writes to the same key. It's not enough for linearizability: without coordination, concurrent writes produce conflicts that LWW silently resolves by dropping data. You need CRDTs, vector clocks, or consensus to handle concurrent writes safely.

**Q6: How do you prevent split brain in a leader-based system?** Three layers: (1) elect leaders via consensus so only one is legitimate at a time, (2) give the leader a time-bounded lease so an old leader self-deposes after expiry, (3) attach a fencing token to every external operation so downstream systems reject stale-leader writes. All three are needed — consensus alone doesn't stop a GC-paused leader from writing to S3.

**Q7: What's the tradeoff between synchronous and asynchronous replication?** Sync waits for acks from `K` replicas before acking the client → no data loss up to `K-1` failures, but write latency is bounded by the slowest of those `K` replicas. Async acks the client immediately and replicates in background → lowest latency, but a leader crash before replication loses writes. Semi-sync (wait for one remote ack) is a common middle ground.

**Q8: How does failover work without losing data?** Need synchronous replication to a quorum (e.g. Raft): a write only commits after the majority persists it. On leader failure, the new leader must have a log at least as up-to-date as the majority, which by quorum intersection means every committed entry. Uncommitted entries may be truncated, which is fine because they were never acked to the client.

**Q9: Why doesn't a single Raft cluster scale to arbitrary throughput?** Every write serializes through the leader: one process, one network pipe, one disk. Throughput caps at tens of thousands of ops/sec depending on implementation. To scale, you shard the keyspace and run one Raft group per shard. Cross-shard transactions then need 2PC or similar.

**Q10: You have a multi-region deployment. How do you handle writes?** Cross-region consensus adds RTT (50-200ms) to every write, usually unacceptable. Common patterns: (1) Regional leader with async cross-region replication — fast writes, possible data loss on region failure, good for most user-facing workloads. (2) Shard by region so each region owns its data and cross-region coordination is rare. (3) Global consensus with something like Spanner's TrueTime — correct, expensive, reserved for cases that truly need global linearizability like global uniqueness or financial ledgers.

---

### Final meta-advice for the interview

- **Name the mechanism, don't wave.** "Fencing token," "leader lease," "majority quorum," "monotonic term" — these words are signals.
- **Walk the write path end-to-end.** Client → coordinator → replicas → commit condition → ack.
- **Then walk the failure path.** What dies, what recovers, what's lost, what's served.
- **Explicitly state what you're not providing.** "This gives eventual consistency, not linearizability — that's fine because reads are for a feed." Showing you know what you're giving up is more impressive than claiming everything works.
- **Default to consensus for metadata, replication for data.** Most large systems do exactly this. Saying it out loud shows you understand the architecture of real systems.