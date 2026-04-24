---
title: Object store metadata layer (simplified)
description: Object store metadata layer
---
# S3 Metadata Layer — The Simpler Walkthrough

```
"Build the metadata layer for an S3-like object store. Blob data is handled elsewhere. We need list operations, prefix queries, versioning, and strong consistency for a single key. Target: 10M ops/sec peak, p99 < 50ms for point lookups."
```
---

## Part 1: What this system actually does

An object store has two layers. The **blob layer** stores bytes — give it an ID, get back the data. The **metadata layer** is everything else: "does this key exist," "what's the current version," "list all keys under this prefix," "atomically create-if-not-exists." The blob layer is essentially a giant hash table. The metadata layer is where the system gets interesting.

The question gives us four operations:
- **GET** `bucket/key` → the metadata for one object
- **PUT** `bucket/key, metadata` → write
- **DELETE** `bucket/key` → remove
- **LIST** `bucket, prefix` → all keys starting with that prefix

Plus **versioning** (every PUT creates a new version, old ones remain) and **strong consistency** for a single key (if you PUT then GET, you see your write).

Target: **10 million ops/sec, p99 under 50ms** for point lookups.

---

## Part 2: The naive design — and why it fails

The obvious first move is: hash the key, send it to one of N shards. This is what a mid-level engineer would draw.

**Why it works for GET/PUT/DELETE:** the key hashes to exactly one shard; you do one RPC; done.

**Why it destroys LIST:** imagine the key `tenant-a/logs/2026-04-17/event-1234`. Under hash partitioning, this key lands on shard 47. The key `tenant-a/logs/2026-04-17/event-1235` lands on shard 198. Keys with the same prefix are **scattered across every shard in the fleet** — that's literally what hashing does.

So when a customer runs `LIST tenant-a/logs/`, you have to ask **every shard** "do you have any keys matching this prefix?" That's a scatter-gather.

Let's do the math. Say we have 250 shards (we'll justify this number shortly). LIST is 5% of 10M ops/sec = **500,000 LISTs/sec**. Each LIST hits 250 shards = **125 million internal RPCs/sec** just for LIST traffic. That's 250× the total ops/sec budget spent on the scatter for one op type. The network melts. The p99 melts. The design is dead.

> **Staff signal #1: workload-driven partitioning.** Why this matters: most candidates pick a data structure (hash map, B-tree, LSM) and ask "what can it do?" Staff engineers look at the workload and ask "what does it *demand*?" LIST is only 5% of ops, but it's 100% of the design risk. If you don't open with "LIST forces the partitioning decision," you've already failed the question — everything after that is mid-level.

So hash partitioning is out. What's the alternative?

---

## Part 3: Range partitioning — the forced choice, and its consequences

**Range partitioning** means we store keys in sorted order, split the keyspace into contiguous ranges, and each range lives on one shard.

```
shard 1: keys from "" to "bucketA/logs/m..."
shard 2: keys from "bucketA/logs/m..." to "bucketA/logs/z..."
shard 3: keys from "bucketA/logs/z..." to "bucketB/..."
...
shard 250: keys from "bucketZ/..." to infinity
```

Now `LIST bucketA/logs/` hits **at most 2-3 adjacent shards** — whoever owns that range. Usually 1. LIST is now fast.

But range partitioning creates three new problems, and solving them is the rest of the design.

### Problem 3a: How many shards, and how much per shard?

Let's size this. Target: 10M ops/sec. The mix is roughly 70% GET, 20% PUT, 5% DELETE, 5% LIST.

Per shard, we want to cap at **~40k ops/sec sustained**. Why 40k? Because a single LIST can internally scan 1000 keys (LIST amplification — you fetch many rows to return a page), and we need headroom for that amplification plus Raft overhead plus compaction. Staff engineers pick this number with a reason; mid-level engineers pick "50k" because it's round.

So: 10M / 40k = **250 active shards**. With 3-way replication (RF=3 for durability — one replica can die without data loss), that's **750 replica processes**. At 3 shards per machine, that's **~250 machines** for the metadata layer.

Per-key storage: each metadata row is about 550 bytes (the key itself is ~80B, the value is ~230B of pointers/timestamps/etags, plus ~2× overhead for LSM indexing and bloom filters). At 2.5M writes/sec, that's 1.4 GB/sec of write ingress across the fleet, or ~5.5 MB/sec per shard. That's well within a RocksDB-class engine's budget.

> **Staff signal #2: capacity math.** Why this matters: "it scales" is unfalsifiable. Actual numbers are either right or wrong, and they tell the interviewer you think in orders of magnitude. "250 shards, 40k ops each, 550B per row" is a committable design. "Several shards, fast enough" is a coward's answer.

### Problem 3b: Hot shards from sequential keys

Range partitioning has a nasty failure mode: **monotonic keys melt one shard**.

A customer writes `bucketX/events/2026-04-17T14:35:22.123-uuid`. Every new event has a timestamp slightly larger than the last. Under range partitioning, they all land in the rightmost shard of that range. One shard takes 100% of that customer's write traffic.

If that customer is writing 300k events/sec (a realistic top-tier tenant — think a logging platform), one shard gets 300k ops/sec. Our per-shard budget is 40k. That shard is 7.5× overloaded. It dies.

**The fix is three layered defenses:**

**Layer 1: Automatic range splitting.** Each shard monitors its own load by sub-region. If any sub-region of its key space exceeds **30k ops/sec for 60 seconds**, the shard proposes a split to the placement driver. The split takes about 5 seconds — atomically, via a Raft log entry that says "as of this moment, this range is now two ranges." We'll do the correctness of this in Part 5.

**Layer 2: Salted sharding for monotonic workloads.** If splitting doesn't help — because the hot spot is always at the *tail* of the range (new timestamps keep pushing right) — we add a salt byte to the key: `bucketX/events/<salt>/...` where `salt` is `hash(key) mod 16`. This scatters writes across 16 sub-ranges.

The cost: LIST on `bucketX/events/` now has to merge results from 16 ranges. But this is per-tenant (only applied to tenants with the problem), per-prefix (not global), and the tenant chose to create this pattern. They pay the LIST cost.

**Layer 3: Admission control.** Routers run per-tenant token buckets. Default: ~3,500 PUT/sec per prefix. Exceed the budget → 503 SlowDown response. This is the S3-like rate limit you see as a customer. It protects other tenants from any single runaway workload.

> **Staff signal #3: hot-key discipline.** Why this matters: in any real system, the top 0.1% of customers cause more than half the incidents. If your design treats them as a footnote, your design doesn't survive contact with production. Interviewers pattern-match on whether you name detection, mitigation, *and* fallback — three layers, not one.

### Problem 3c: Correctness when a shard splits

Range splits are the most dangerous operation in this system. During a split, writes are in-flight, the old shard is still serving, the new shards are being populated, ownership is being handed over. **If we screw this up, LIST returns duplicates or missing entries, or a PUT lands in the wrong shard.**

The invariant we must preserve: **for every key K and every moment T, exactly one Raft group is authoritative for K at T.** Not "eventually" — exactly.

How we enforce it:

1. Placement driver decides to split shard G at key M.
2. Placement driver creates a new empty shard G2 on three machines.
3. Shard G's leader writes a **Raft log entry** that says: "as of this committed entry, I no longer own keys ≥ M; G2 does."
4. The moment that Raft entry commits, G starts rejecting writes to keys ≥ M, even if G2 isn't yet serving.
5. G2 bootstraps by copying the snapshot of keys [M, ∞) from G.
6. G2 starts serving.
7. Routers get the new shard map and route accordingly.

The magic is step 3-4. The ownership handover is a single linearization point in G's Raft log. Before it commits: G owns everything. After it commits: G owns less. There's no overlap, no ambiguity. A client that routes to G for a key ≥ M after step 4 gets a "wrong range" response and retries.

> **Staff signal #4: invariant-based thinking.** Why this matters: when you design by listing mechanisms, you miss the failure modes *between* mechanisms. When you design by stating an invariant ("exactly one group is authoritative per key") and then showing each mechanism preserves it, you catch yourself. This is the difference between an engineer who reads papers and one who has shipped consensus systems. Naming the invariant explicitly before the mechanism is the move.

---

## Part 4: Versioning — why "add a column" is wrong

A mid-level answer to versioning is "add a version_id column to the metadata row." This is wrong because it misses how versioning changes the *read path*.

The question isn't "store multiple versions." The question is: **"Given a key, give me the current version in one seek."**

If versions are just rows with a version_id column, finding the current version requires either (a) a secondary index mapping key → current version ID, which you now have to keep consistent with the primary, or (b) scanning all versions of the key and picking the newest. Both are bad.

The trick is to make the storage itself sort newest-first. We encode keys like this:

```
row key = bucket_id | \0 | object_key | \0 | reverse_timestamp | version_id

where reverse_timestamp = MAX_UINT64 − commit_time_micros
```

Because we subtract the timestamp from a big constant, **newer timestamps produce smaller bytes**, which sort first in lexicographic order. So for any given object key, the row you'd find first when scanning is automatically the newest version.

Now reads are simple:
- **GET current version**: seek to `bucket_id | \0 | object_key | \0`, take the first row. One seek.
- **GET specific version**: seek to the full row key. One seek.
- **LIST without versions**: scan, but after emitting one version per key, skip to the next key. One seek per logical key.
- **LIST with all versions**: scan everything. Newest-first per key falls out for free.
- **DELETE**: insert a new row that's a "delete marker." Subsequent GETs without a version ID return 404 because the delete marker is newest.

```
Storage for one key with 3 versions + a delete:

[bucketA | obj.jpg | rev_ts=v4 | vid_4]  → delete marker     ← newest
[bucketA | obj.jpg | rev_ts=v3 | vid_3]  → version 3 data
[bucketA | obj.jpg | rev_ts=v2 | vid_2]  → version 2 data
[bucketA | obj.jpg | rev_ts=v1 | vid_1]  → version 1 data    ← oldest
```

The garbage collection problem is real: a buggy or malicious client writes 1M versions of one key. Storage bloat, slow list-versions, compaction pain. Three defenses:

1. Hard cap at 1M versions per key — further PUTs fail with TooManyVersions.
2. Lifecycle rules — "delete non-current versions after 30 days" runs as a background job.
3. Default soft lifecycle on versioned buckets — 90 day retention of non-current if no rule specified. Opinionated but prevents the common footgun.

> **Staff signal #5: "the layout is the index."** Why this matters: candidates who think in terms of "table + index" fail to see that in a range-partitioned key-value store, **the way you encode the key IS the index**. If you need newest-first, you encode time in reverse and the sort order gives it to you for free. Recognizing this is the difference between bolt-on engineering and native design.

---

## Part 5: Consistency — what "strong" actually means

"Strong consistency for a single key" sounds clear but it's actually five different things. Being precise about which is which is the signal.

| Scenario | What "strong" means here |
|---|---|
| PUT(K,v) then GET(K) | GET returns v. (Read-after-write) |
| PUT(K,v1), PUT(K,v2), GET(K) | GET returns v2. (Read-after-overwrite) |
| DELETE(K), GET(K) | Returns 404 (or next version). (Read-after-delete) |
| Two concurrent PUTs to K | One wins; GET returns a coherent result. (Linearizable) |
| `If-Match: etag1` PUT(K) | Atomic compare-and-swap. (Conditional write) |

These all need to hold. LIST is **deliberately not on this list** — LIST is read-committed at a snapshot, which is weaker. Why? Because to make LIST linearizable with PUT, every PUT would have to synchronize with a global timestamp authority (adds ~20ms per PUT) or every LIST would have to coordinate across shards with a fresh read barrier (adds tens of ms per LIST). S3 itself has never been strictly linearizable for LIST. Don't promise what isn't worth building.

**How we implement strong consistency for the single-key operations:** each shard is a Raft group (3 replicas, one leader). The leader serves reads under a time-bounded lease — it's confident no new leader has been elected in the last few seconds, so it can serve reads directly without consulting followers. Writes go through Raft consensus: propose, replicate to a majority, commit, respond. Standard.

Conditional writes (`If-Match`) are implemented as CAS inside the Raft state machine — the write op carries the expected etag, and the state machine rejects if the current etag doesn't match. **No read-then-write round trip**. The check happens atomically inside a single Raft log entry.

> **Staff signal #6: consistency precision.** Why this matters: "strong consistency" is the phrase most abused by mid-level engineers. They say it without knowing whether it means linearizable, sequential, read-committed, or read-my-writes. If you can cleanly separate "linearizable for single-key ops, snapshot-read for LIST, conditional writes are atomic CAS" — the interviewer knows you've read the Raft paper, not just the Wikipedia summary.

---

## Part 6: The metadata-blob contract — the hidden question

The prompt says "blob data is handled elsewhere." This sentence is a trap. Most candidates skip it entirely. It's actually the hardest distributed systems problem in the design.

If metadata and blob storage are separate services, they can disagree. Metadata says "object X exists, blob ID is B7." Blob service says "B7? Never heard of it." What does the client see? What does the system do about it?

**The protocol for PUT:**

1. Router authenticates the client, generates a blob ID.
2. Router writes the bytes to the blob service. Waits for durable confirmation.
3. Router writes the metadata row to the metadata layer (Raft commit).
4. On success, return 200 to the client.

**The invariant:** if a committed metadata row references a blob ID, that blob exists durably. This is guaranteed by ordering: step 3 only runs if step 2 succeeded, and step 3 is itself durable via Raft.

**Failure modes:**

| Where it fails | Client sees | System state | Recovery |
|---|---|---|---|
| Blob write fails | 5xx | No metadata, maybe a partial blob | Client retries. Blob GC sweep collects orphans after 24h. |
| Blob succeeds, metadata fails | 5xx or timeout | **Orphan blob**, no metadata | GC scan: blobs with no metadata reference after 24h → deleted. |
| Metadata commits, response lost | Timeout, client retries | Actually correct. Retry is idempotent. | Client retries; CAS or version_id makes it safe. |
| Concurrent PUTs to same key | One wins via Raft | Exactly one metadata row; losing blob is orphaned | GC reclaims losing blob. |

For **DELETE**, we flip the order: write metadata delete marker first, leave the blob alone. The blob only gets reclaimed when a lifecycle rule or explicit version-delete runs later. Why? Because deleting a delete marker is how you "undelete" in a versioned bucket. If we nuked the blob immediately, undelete wouldn't work.

> **Staff signal #7: cross-system contract.** Why this matters: the prompt deliberately says "blob is handled elsewhere" to see if you'll notice that the metadata-blob consistency protocol is the actual interesting distributed systems question. Most candidates don't notice. Noticing — and then specifying the protocol with named invariants and concrete failure modes — is what separates senior staff from staff. This is probably the single biggest signal in the question after the LIST→partitioning move.

---

## Part 7: Things I'd push back on

The prompt is deliberately under-specified in several places. A staff engineer names them and proposes defaults; a senior engineer silently accepts them and builds the wrong thing.

1. **"Strong consistency for a single key" is silent on LIST.** I'd ask: "Do you need linearizable LIST, or is snapshot-read OK?" The first is much more expensive. My default is the second.

2. **"10M ops/sec peak" doesn't say peak over what window, or across how many tenants.** Peak over 1 second vs 1 minute vs 1 hour changes the admission-control design. I'd push for the burst profile.

3. **"p99 < 50ms" as a single number is incoherent.** A GET touching one shard in one AZ, a LIST spanning 5 shards, and a PUT going through Raft quorum have wildly different latency profiles. 50ms is generous for point lookups and tight for cross-shard LIST. I'd ask for separate budgets per op type.

4. **The prompt assumes one design fits all workloads.** S3 itself now has a separate "Express" tier for latency-sensitive small-object workloads because the general-purpose design can't meet single-digit-ms latencies. I'd ship the design I've drawn, then plan a specialized tier within a year.

> **Staff signal #8: saying no.** Why this matters: the prompt is giving you requirements that a junior engineer would accept and a mid-level engineer would caveat. A staff engineer pushes back, proposes a better-specified version, and commits to defaults where the spec is ambiguous. Agreeing to everything the prompt says is evidence you don't know what to push on.

---

## What makes this a staff answer, in one paragraph

The whole thing hinges on one move: **recognizing that LIST — which looks like a footnote in the requirements — forces range partitioning, which cascades through every other decision.** Range partitioning gives you efficient prefix scans but creates hot shards; so you need automatic splits, which means you need Raft-based ownership transfer with a provable invariant. It makes versioning trivially fast if you encode reverse-timestamps into the sort order. It makes the shard map into a live topology that must evolve online. It makes LIST inherently a multi-shard operation when prefixes span ranges, which forces read-committed semantics rather than linearizable ones. Every design detail falls out of the partitioning choice. **A mid-level engineer designs the components and hopes the system works. A staff engineer identifies the one decision that determines the shape of the whole system and defends it through every downstream consequence.** That's what this question is actually testing.