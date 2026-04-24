---
title: Object store metadata layer
description: Object store metadata layer
---

# S3-like Object Store: Metadata Layer Design

## 1. Scope and Reframing

Before I draw anything, let me commit to what we're solving.

**In scope** (I'll own these end-to-end):
- GET / HEAD / PUT / DELETE for a single key, with strong read-after-write consistency
- LIST with `prefix`, `delimiter`, `start-after`, `max-keys` — full S3 API semantics
- Versioning: versioned PUT, versioned GET, delete markers, LIST-versions
- Conditional writes: `If-Match`, `If-None-Match` (needed for CAS patterns)
- Multipart upload **metadata** (the UploadId → parts manifest); the parts themselves are blob-layer

**Explicitly deferred** (I'll flag where they'd bend the design):
- Object Lock / legal holds (needs a separate retention ACL layer)
- Lifecycle rules (needs an offline rule engine driving background DELETEs; orthogonal)
- Tags, ACLs, IAM policy evaluation (orthogonal control plane)
- Cross-region replication (will discuss topology briefly; full multi-region active-active is a different talk)
- Full-text / content search over metadata (would force a denormalized secondary index and a redesign — §12)

**The central forcing function.** **[STAFF SIGNAL: workload-driven partitioning decision]** The question gives us four workload axes: point lookups, prefix LIST, versioned reads, cross-prefix scans. Point lookups are satisfied by almost any scheme. **Prefix LIST is the one that kills you.** If I hash-partition by key, `LIST /tenant-a/logs/2026-04-17/` becomes a scatter-gather across every shard in the fleet. At 500k LIST/sec against 400 shards that's 200M fanned-out RPCs/sec on the metadata network, before pagination. That alone blows the p99 budget. So I'm committing upfront: **the namespace is range-partitioned on the lexicographic key, and every subsequent decision cascades from that.** The rest of the talk is defending that choice against the problems it creates (hot shards, split correctness, monotonic keys).

**[STAFF SIGNAL: scope negotiation]** I'm also calling out one piece of under-specification now: "strong consistency for a single key" is precise for GET/PUT/DELETE but silent on LIST. LIST cannot reasonably be strongly consistent with the rest of the system at 10M ops/sec — it's a scan over a moving target. I'll implement LIST as **read-committed at a bounded snapshot** and come back to whether that's what the customer actually wants in §13.

## 2. Capacity Math and Budget

Assume a workload mix (typical for large tenants — logs, backups, ML datasets):

| Op       | % of ops | QPS         | Notes |
|----------|---------:|------------:|-------|
| GET/HEAD | 70%      | 7.0 M/s     | point lookup, ideally 1 RPC |
| PUT      | 20%      | 2.0 M/s     | single-key write, Raft commit |
| DELETE   | 5%       | 0.5 M/s     | in versioned buckets, this is a delete-marker PUT |
| LIST     | 5%       | 0.5 M/s     | each returns up to 1000 keys; scan amplification is real |

**Per-shard target.** I'll target 40k ops/sec sustained per shard, with headroom for LIST amplification (a single LIST that scans 1000 keys costs ~1000 local reads, even if it only returns a handful after delimiter collapse). That gives budget for a scan-heavy burst without pushing the shard to GC jail. At 10M ops/sec with ~40k/shard, I need **~250 active shards** in the working set.

**Replication.** RF=3 with Raft per shard. 750 replica processes. On 32-core nodes co-locating 3 shards each, that's ~250 machines for the metadata KV, not counting routers, placement driver, and GC.

**Per-key storage.** **[STAFF SIGNAL: capacity math]** Each version row:
- Key (bucket_id + object_key + version_suffix): avg 80 B
- Value: blob_pointer (32 B) + size (8) + etag (16) + content-type (16) + storage-class (1) + sys timestamps (16) + checksum (16) + user metadata (avg 128) ≈ 230 B
- LSM overhead (block index, bloom filters, compression headroom): ~1.8×
- **Effective: ~550 B per version row**

At 2.5M PUT/s (including delete markers), uncompacted, that's ~1.4 GB/s of WAL+memtable ingress fleet-wide, or ~5.5 MB/s per shard. Very manageable for RocksDB-class engines.

**Hot-customer worst case.** Top 0.1% of tenants carry ~30% of traffic (power law). That's 3M ops/sec across ~1000 tenants, mean 3k ops/sec per top tenant, **p99 tenant at ~300k ops/sec**. No single shard can hold 300k ops/sec for one tenant — I'll need sub-range splits. More in §5.

**p99 latency budget (50 ms).** For a GET:
- Client → router: 2 ms (intra-region)
- Router → shard leader: 2 ms
- Shard leader read (memtable + block cache hit): p99 ~5 ms (cold-block read on SSD: ~15 ms)
- Router → client: 2 ms
- **Happy path: ~11 ms. Budget for Raft lease renewal, retries, shard-map misses: ~40 ms.**

For PUT, add Raft quorum latency (~6 ms p99 to the slower follower intra-region), pushing happy-path PUT to ~18 ms. Still fits.

## 3. High-Level Architecture

```
             ┌─────────────────────────────────────────────────────┐
             │              Clients (S3 SDK)                       │
             └─────────────────────────────────────────────────────┘
                            │  (HTTPS, SigV4)
                            ▼
     ┌──────────────────────────────────────────────────────────┐
     │  Frontend Routers (stateless, ~200 nodes)                │
     │  • Caches shard map (TTL 30s, invalidated on redirect)   │
     │  • Per-tenant token buckets (admission control)          │
     │  • Parses request → (bucket, key, op)                    │
     └──────────────────────────────────────────────────────────┘
                            │
        ┌───────────────────┼───────────────────────┐
        ▼                   ▼                       ▼
  ┌──────────┐        ┌──────────┐            ┌──────────┐
  │ Shard 17 │        │ Shard 18 │    ...     │ Shard N  │
  │ (Raft:   │        │ (Raft:   │            │          │
  │  3 repl) │        │  3 repl) │            │          │
  │ RocksDB  │        │ RocksDB  │            │ RocksDB  │
  │ range    │        │ range    │            │ range    │
  │ [k0,k1)  │        │ [k1,k2)  │            │ [kN,∞)   │
  └──────────┘        └──────────┘            └──────────┘
        │
        │     ┌─────────────────────────────────┐
        └────▶│   Placement Driver (PD)         │
              │   • Owns shard map              │
              │   • Orchestrates splits/merges  │
              │   • Raft-replicated, 5 nodes    │
              └─────────────────────────────────┘

     ┌──────────────────────────────────────────────────────────┐
     │  Auxiliary services (async, not on critical path)        │
     │  • GC workers (orphan blobs, expired versions)           │
     │  • Quota aggregator (per-shard deltas → tenant counts)   │
     │  • Audit log shipper                                     │
     └──────────────────────────────────────────────────────────┘
                            │
     ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─│─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─
     metadata layer boundary │ blob layer (separate service)
                             ▼
                      ┌──────────────┐
                      │ Blob Service │  content-addressed or ID-addressed
                      └──────────────┘
```

The **shard map** is the only piece of global state on the hot path, and routers cache it aggressively. On miss (router stale, client retries), the shard responds "not my range" with a redirect hint and the router refreshes. The PD is not on the critical path of any read or write — it only orchestrates topology.

**[STAFF SIGNAL: rejected alternative]** I considered a **two-tier design** (a directory tier mapping bucket-prefix → shard-set, with leaf shards being hash-partitioned). Rejected: the directory tier becomes the bottleneck for the LIST workload it's designed to accelerate; at 500k LIST/s and the tier being globally consistent, you've just moved the problem. Flat range-partition gives us O(1) shard lookups given a cached map, with no additional hop on the hot path.

## 4. Partitioning Deep-Dive

### Key encoding

```
Primary row key (lexicographic, ordered):
┌──────────────┬────┬──────────────────┬────┬──────────────────────┐
│  bucket_id   │ \0 │    object_key    │ \0 │  version_suffix      │
│  (8B, u64)   │    │   (variable)     │    │  (1 + 8 + 16 B)      │
└──────────────┴────┴──────────────────┴────┴──────────────────────┘

version_suffix layout:
  byte 0:       kind byte (0x10 = version row, 0xFF = per-key sidecar)
  bytes 1-8:    reverse timestamp (u64, big-endian, = MAX_U64 - commit_ts_micros)
  bytes 9-24:   version_id (UUIDv7-like, 128-bit)
```

**Why `\0` separator.** Zero-byte is the minimum value, so `\0` between fields guarantees `bucket_id=X, key="a"` sorts strictly before `bucket_id=X, key="ab"`. Without it, `key="a"` followed by the version-suffix's first byte (0x10) would collide with `key="a\x10..."`.

**Why reverse timestamp.** For a given object key, the newest version has the smallest version_suffix. A prefix scan on `bucket_id\0object_key\0` yields versions newest-first in a single sequential scan — this is the path for "GET current version" and it's a single RocksDB `Seek` + one `Next`.

### Shard map

```
shard_map : sorted list of (range_start_key, range_end_key, raft_group_id)

  [\0\0            , bkt7/\0e...    ) → group_0x4a
  [bkt7/\0e...     , bkt7/\0u...    ) → group_0x4b   ← sub-splits within
  [bkt7/\0u...     , bkt12/\0...    ) → group_0x4c   ←  bucket 7
  [bkt12/\0...     , bkt45/\0log/m.) → group_0x4d
  [bkt45/\0log/m.  , bkt45/\0log/q.) → group_0x4e   ← hot prefix sub-split
  ...
```

Range boundaries can fall **inside** a bucket, even inside a prefix. A hot tenant like bucket 45 has its `log/` prefix split into multiple shards.

### Split mechanics

**[STAFF SIGNAL: invariant-based thinking]** **Invariant under split: for every key K and every wall-clock instant T, there is exactly one Raft group authoritative for K at T.** The split is a linearization point, not a data-copy event.

```
  Before:  group_G owns [A, Z)

  1. PD decides split at key M (based on heat telemetry)
  2. PD issues ADD_PEERS to create group_G2 with empty state,
     replicas on 3 new nodes.
  3. group_G leader proposes, via its own Raft log, entry:
        { type: SPLIT, at: M, new_group: G2, snapshot_idx: N }
  4. Entry N commits. From this moment, group_G's state machine
     rejects any write with key ≥ M.
  5. group_G2 bootstraps by ingesting the snapshot of keys [M, Z)
     from group_G (streaming SST ingest, not logical replay).
  6. group_G2 catches up to Raft log index N on those keys.
  7. PD publishes new shard map. Routers pick it up lazily.

  Client that routes to group_G for key ≥ M after step 4:
    → group_G responds with WRONG_RANGE + hint (G2)
    → router refreshes map, retries
```

Step 3 is the linearization point. Before it commits, all writes go to G. After, writes to [M, Z) are rejected by G even before G2 is serving — the client retries until G2 is up (subsecond window in practice). This is the **fencing** analog: the Raft log gives us a single serial order for ownership changes, with zero ambiguity about who owns what at any moment.

### Hot-shard mitigation

**[STAFF SIGNAL: hot-key / hot-prefix discipline]** The pathological case for range partitioning is **monotonic keys**. Customer writes `s3://mybkt/events/2026-04-17T14:35:22.123456Z-uuid`. Every write lands at the tail of the keyspace. At 300k PUT/sec from one customer, one shard melts.

Three layers of defense:

1. **Automatic sub-range splitting.** Each shard tracks per-key-range heat (QPS in 1-second windows over 64 internal sub-ranges of its key space). **[STAFF SIGNAL: precision under ambiguity]** If any sub-range exceeds **30k ops/sec for 60 seconds**, the shard proposes a split to PD at the median-load key. Split takes ~5 seconds end-to-end.

2. **Tail-split heuristic.** When a shard's hottest sub-range is the **rightmost** one (monotonic workload), splitting at the median just moves the hot spot to the new right shard. Detector flag: if post-split, the right child's QPS re-exceeds threshold within 5 minutes, PD applies **salted sharding** to that tenant's monotonic prefix: `bucket_id/<salt_byte>/<original_key>`. Cost: LIST on that prefix now scatter-gathers across 256 sub-shards. This is a per-tenant, per-prefix feature flag, not the default.

3. **Admission control.** Router-layer token buckets per tenant. Default: 3,500 PUT/s + 5,500 GET/s per prefix (mirrors S3's historical per-prefix limits), auto-raised on sustained success. Overflow → 503 SlowDown with `Retry-After`.

```
  Hot-shard scenario:

  t=0:     Tenant X hits 300k PUT/s on keys bkt_X/evt/2026-04-17T*
           Shard S_3 owns [bkt_X/evt/, bkt_X/evt/\xff)
           S_3 saturates at ~50k/s, p99 latency balloons

  t=60s:   S_3 proposes split at bkt_X/evt/2026-04-17T14:20
           → S_3a: [bkt_X/evt/, bkt_X/evt/2026-04-17T14:20)  [cold]
           → S_3b: [bkt_X/evt/2026-04-17T14:20, bkt_X/evt/\xff)  [all new writes still here]

  t=120s:  S_3b also saturates. PD detects pattern = tail-hot.
           Salted sharding enabled for bkt_X/evt/.
           New writes go to bkt_X/evt/<hash(key)[:1]>/<key>
           Spreads writes across 16 ranges.
           LIST on bkt_X/evt/ now does 16-way merge (paid by this tenant only).
```

## 5. Deep Dive: LIST Operation Design

LIST is the centerpiece. Full semantics:

```
LIST(bucket, prefix, delimiter, start_after, max_keys=1000)
  → (keys[], common_prefixes[], continuation_token)
```

### Execution when the prefix lives in one shard

Happy path:

```
  Router:
    shard = shard_map.find(bucket + "\0" + prefix)
    response = shard.Scan(
       start  = max(bucket+"\0"+prefix, bucket+"\0"+start_after),
       end    = bucket+"\0"+prefix_upper_bound,
       limit  = max_keys,
       filter = {first_version_per_key, skip_delete_markers,
                 delimiter_collapse}
    )

  Shard (RocksDB iterator):
    Iter.Seek(start)
    for each row until limit:
      if row.is_delete_marker or row.kind != VERSION_ROW: skip
      if delimiter: collapse to common_prefix, emit once, skip-past
      emit row
    return (keys, common_prefixes, last_key_for_token)
```

**Pushdown is critical.** The scan filters (first-version-per-key, delimiter collapse, delete-marker skip) run **on the shard**, not the router. If the shard returned raw version rows, a LIST on a tenant with 1M versions per key would transfer 1M rows per key over the wire to filter to 1. At ~550 B/row that's 550 MB of wire traffic for a trivial response. Filter on the shard.

### Execution when the prefix spans multiple shards

```
  Client             Router              Shard A        Shard B        Shard C
    │                    │                    │              │              │
    │ LIST prefix=p/     │                    │              │              │
    ├───────────────────▶│                    │              │              │
    │                    │ find shards        │              │              │
    │                    │ intersecting       │              │              │
    │                    │ [p/, p/\xff) → A,B,C              │              │
    │                    │                    │              │              │
    │                    │ Scan(p/, p/\xff, 1000) ──────────▶│              │
    │                    │                    │              │ Scan ───────▶│
    │                    │                    ◀───── rows ────┤              │
    │                    │◀────── rows ───────┤              │◀─── rows ────┤
    │                    │                                                   
    │                    │ k-way merge by key order, take first 1000         
    │                    │                                                   
    │◀────── keys[1000], continuation_token ────────────────────────────────
    │
    │ LIST ..., continuation_token=T
    ├───────────────────▶│  (token encodes {last_key, per_shard_cursors})
    │                    │  continues scan only on shards with rows ≥ last_key
```

The continuation token is **opaque to the client but structured**: it encodes the last-emitted key and per-shard cursors. On resumption, the router queries only shards whose range still contains keys ≥ last_key.

### Consistency of LIST

**[STAFF SIGNAL: consistency precision]** LIST is **read-committed, not linearizable**. Specifically:
- Each shard's scan is snapshot-consistent within that shard (RocksDB snapshot taken at scan start).
- Across shards, scans may be at slightly different snapshots (up to ~seconds skew).
- A PUT that committed at time T0 may or may not appear in a LIST started at T0+ε on a different shard than the PUT's shard.

This is **weaker** than per-key strong consistency. It's defensible: S3's own LIST has never been strictly linearizable with PUT. Strengthening it would require global timestamps (TrueTime-style) or a witness service adding ~20 ms to every PUT. Not worth the cost for LIST's value proposition.

### Pagination across splits

If a shard splits mid-LIST, the continuation token must survive. The token stores `{last_key, last_range_id}`. On resumption, if last_range_id no longer exists (split happened), the router consults the shard map using last_key — the key itself is stable under splits, so this always resolves.

### The billion-object-under-one-prefix case

If a tenant has 10B objects under `logs/` and issues `LIST logs/`, we **can't** enumerate all of them in one response. The API is paginated: each response returns 1000 keys. Tenant that wants "total object count" pays for N/1000 RPCs. Fair.

But a `LIST logs/` with `delimiter=/` that collapses to 10k common prefixes should complete quickly. The shard-side delimiter collapse pushes the scan to the iterator level. Worst case: 10B keys all collapse to 1 common prefix — we still have to scan 10B rows to know there's only one. **This is why I bound `max_keys_scanned` separately from `max_keys_returned`**: the shard emits a continuation token after scanning 1M rows regardless of how many results it found. The client resumes. This prevents one unbounded LIST from starving the shard.

## 6. Deep Dive: Versioning Layout

```
Logical key "photos/vacation.jpg" in bucket 42 with 3 versions:

┌────────────────────────────────────────────────────────────────┬───────────────────┐
│ row key (lex-ordered, newest-first within a key)               │ value (metadata)  │
├────────────────────────────────────────────────────────────────┼───────────────────┤
│ 42\0photos/vacation.jpg\0\x10<rev_ts=3><vid_3>                 │ {blob_ptr, size,  │
│                                                                │  etag, ...}       │
│ 42\0photos/vacation.jpg\0\x10<rev_ts=2><vid_2>                 │ {..., is_delete_  │
│                                                                │  marker=true}     │
│ 42\0photos/vacation.jpg\0\x10<rev_ts=1><vid_1>                 │ {blob_ptr, ...}   │
└────────────────────────────────────────────────────────────────┴───────────────────┘

Where rev_ts_N = MAX_U64 - commit_ts_N  (newest ⇒ smallest suffix)
```

**Get current version.** `Seek(42\0photos/vacation.jpg\0\x10)` → returns row with rev_ts=3 (newest). Check `is_delete_marker`: if true, return 404. Else return metadata, client fetches blob.

**Get specific version.** Client supplies version_id. Row key is fully determined; point lookup.

**LIST without versions.** Scan with pushed-down filter "emit first row per key, skip delete-marker". Single seek per logical key.

**LIST with versions (`list-object-versions`).** Emit every row. Ordering: by key, then by rev_ts (newest-first within key) — falls out of the encoding for free.

**Delete-marker semantics.** `DELETE photos/vacation.jpg` without a version-id inserts a new row with `is_delete_marker=true` at the top of the version stack. GET without version-id returns 404. GET with version-id of a previous version returns that version. `DELETE` of a specific version removes that version row (subject to GC, see below).

### The GC problem

**[STAFF SIGNAL: failure mode precision]** A malicious or buggy client writes 1M versions of the same key. Implications:
- Storage: 550 MB for one logical key. Per-customer billing pain.
- **LIST-with-versions performance for the containing prefix**: scans 1M rows per key.
- **Compaction**: RocksDB rewrites these rows repeatedly at every level.

Mitigations:
1. **Hard per-key version cap** (default: 1M versions per key). Beyond that, PUT fails with `TooManyVersions`. Prevents unbounded growth.
2. **Lifecycle-rule-driven GC.** Customer configures "expire non-current versions after 30 days." A background job scans, emits DELETE-version RPCs for qualifying rows. Runs per-shard, throttled to <10% of shard's QPS budget.
3. **Soft default lifecycle.** Bucket with versioning enabled but no lifecycle rule → default 90-day non-current expiry. Opinionated but prevents the common footgun.

### LIST latency when one key has 1M versions

With the layout above, `LIST prefix=photos/` with `first-version-per-key` filter: the iterator seeks to the first version of each key, then `Seek(next_key_prefix)` to skip past the remaining 999,999 versions. RocksDB supports this efficiently via `iterate_upper_bound` and block-level bloom skipping — costs ~1 extra block read per key. Bounded.

`LIST list-object-versions prefix=photos/` against the pathological key: returns 1M rows, paginated 1000 at a time. Client hates us but correctness holds. This is the right API-level answer — we shouldn't invent a special path.

## 7. Deep Dive: Consistency Semantics

**[STAFF SIGNAL: consistency precision]**

**What "strong consistency for a single key" delivers here:**

| Scenario | Guarantee |
|---|---|
| PUT(K, v1) → GET(K) | Returns v1 (read-after-write). |
| PUT(K, v1); PUT(K, v2) → GET(K) | Returns v2 (read-after-overwrite). |
| DELETE(K) → GET(K) | Returns 404 (or next-newest version if versioned). |
| Concurrent PUT(K, v1), PUT(K, v2) | One wins; GET returns a coherent winner (linearizable). |
| `If-Match: etag1` PUT(K, v2) when K.etag ≠ etag1 | Fails with 412. Atomic CAS. |
| `If-None-Match: *` PUT(K, v) when K exists | Fails with 412. Atomic create-only. |

**Implementation.** Each key's row lives on exactly one shard at any moment (§4 invariant). Within that shard, Raft gives us linearizable writes. Reads go to the Raft leader with a **read-lease** (leader is confident no new leader exists for the next L ms, serves reads without a fresh log entry). Lease renewal every ~3s. Standard TiKV / CockroachDB practice.

Conditional writes (`If-Match`) are implemented as **CAS inside the state machine**: the write op includes the expected etag; the state machine rejects the op if the current etag doesn't match. No read-then-write round trip — atomic inside a single Raft log entry.

**What we give up without strong consistency.** A tenant that sets a per-bucket flag `consistency: eventual` gets:
- Reads may hit followers with bounded staleness (up to replica lag, typically ~50 ms).
- PUTs can return after append to leader log but before quorum ack. Reduces PUT p99 by ~4 ms; sacrifices durability on leader crash before quorum replication.
- Conditional writes still require quorum.

Per-bucket opt-in, not the default.

## 8. Metadata-Blob Contract

**[STAFF SIGNAL: cross-system contract]** "Blob data is handled elsewhere" is the most dangerous sentence in the prompt. The contract:

**PUT protocol (blob-first, metadata-commit-last):**
1. Router authenticates, generates `blob_id = UUIDv7`.
2. Router → blob service: `PutBlob(blob_id, bytes)`. Returns durable on 2-of-3 replicas.
3. Router → shard leader: `CommitMetadata(key, blob_id, size, etag)` via Raft.
4. On Raft commit, respond 200 to client.

**Invariant.** If a committed metadata row references blob_id B, then B exists durably. (Provable by construction: step 3 only runs after step 2 returned durable; step 3 is itself durable via Raft.)

**Failure modes:**

| Failure point | Client sees | System state | Recovery |
|---|---|---|---|
| Blob write fails (step 2) | 5xx | No metadata, partial/aborted blob | Client retries; blob service has a 24h orphan-cleanup sweep |
| Blob succeeds, metadata Raft fails (step 3) | 5xx or timeout | Orphan blob, no metadata | **GC scan**: blob_ids unreferenced after 24h → delete |
| Metadata Raft commits, response lost | Timeout, retries | State is correct; retry with same key+blob_id is idempotent by conditional PUT | Expected. |
| Concurrent PUTs to same key | One wins via Raft ordering | Exactly one metadata row committed; other blob becomes orphan | GC reclaims losing blob |

**DELETE protocol.** Metadata-first: write delete-marker row in Raft, return success. Blob stays referenced until lifecycle GC or explicit version-delete. Different from PUT because deletion is logically reversible (undo delete marker by deleting it).

**Version-delete (hard delete of a specific version).** Two-phase: (1) mark row `pending_delete=true` in metadata; (2) async worker verifies no references, removes blob_id from blob service; (3) remove metadata row. Between (1) and (3), GETs of that version return 410 Gone. Protects against races where an in-flight GET has already resolved the blob_id — hard-deleting the blob underneath would produce a correctness bug.

## 9. Failure Modes and Recovery

**[STAFF SIGNAL: blast radius reasoning]**

**Single shard failure.** Raft handles leader failure transparently (~200 ms failover). Follower failure: no impact. All-replicas failure (rack or AZ failure): the shard is unavailable. **Blast radius: a contiguous key range**, which with range partitioning means **contiguous tenants**. This is worse than hash partitioning's random blast radius — a single AZ loss can take down one big tenant's entire contiguous range. Mitigations:
- **Replica placement across AZs** (RF=3, one per AZ, single-region).
- For multi-region DR: 2 additional cross-region learner replicas; Raft quorum stays intra-region (3), cross-region is async catch-up.

**LIST across a failed shard.** LIST spans [A, B, C]. B is down. Options:
- **Fail fast**: return 5xx. Simple but breaks apps doing iterative LIST.
- **Partial result + continuation token marking "refetch B"**: completes the LIST for A and C; token encodes that B's range is pending. Client retries the token later. Ship this — it's what S3 actually does during gray failures.

**Split rollback.** If a split is proposed but G2's bootstrap fails (node dies mid-snapshot-transfer), PD aborts: group_G commits a new Raft entry (`SPLIT_ABORT`). Since no writes to the split range were served by G2 before completion, no data is lost. Splits have a 2-minute SLO; PD pages on any split stuck longer.

**PD failure.** PD is Raft-replicated (5 nodes). Its failure stops new splits/rebalances but doesn't block data plane — routers use cached shard maps, existing shards keep serving. Design principle: **PD is off the hot path**; its availability requirement is 99.9%, not 99.999%.

**Correlated metadata-blob unavailability.** Blob service down during a PUT: step 2 fails, clean failure. Blob service down during a GET: metadata returns the blob_id, router attempts fetch, blob service returns 5xx, router returns 5xx to client with `Retry-After`. Metadata is correct; blob is temporarily unreachable. No cross-layer cleanup needed.

## 10. Recent Developments That Matter Here

**[STAFF SIGNAL: modern awareness]**

**S3's 2020 strong-consistency transition.** Before 2020, S3 was eventually consistent for overwrite PUTs and LIST. The change was enabled by a new witness/barrier layer that tracks recent writes with strong consistency; reads consult the witness before returning to see if there's a newer version to wait for. The rest of S3 stayed eventually consistent. Architectural lesson: **retrofitting consistency onto an EC core is more expensive than designing for it**. Our Raft-per-shard design gets strong consistency natively, no witness sidecar needed. Also worth calling out: S3 exposed zero customer-visible API changes for this migration — a masterclass in operational evolution.

**S3 Express One Zone (2023).** Single-AZ directory buckets with flat namespaces (no hierarchical prefix semantics) and session-authenticated access, for single-digit-ms latency. The design point: by giving up multi-AZ durability and full prefix hierarchy, they cut metadata ops to near-local-disk latencies. Relevant as a **scope-cutting lesson**: if a customer doesn't need prefix LIST, you can be dramatically faster. We could offer an "express mode" sub-product on top of this design with a different shard configuration (smaller shards, in-memory index, single-AZ placement).

**S3 Tables + Iceberg (2024).** Object storage that natively understands Iceberg table metadata (manifests, partition stats, snapshot trees). Pushes data-aware metadata into the object store. If this design had to support S3 Tables semantics, I'd add a **typed-row** path in the shard: some rows aren't object metadata but table metadata (manifests), with schema-aware compaction that merges old snapshots. Range partitioning still works; storage engine needs typed-row awareness.

**FoundationDB as a metadata substrate.** Snowflake moved metadata to FDB (~2019); iCloud and CloudKit are built on it. FDB provides: ordered keyspace, ACID multi-key transactions, automatic range rebalancing, strict serializable isolation. For this design, **FDB is the most realistic "buy, don't build" option**. If staffing this with 4 engineers and 9 months, I'd put metadata on FDB and write shard-specific logic (pushdown filters, versioning layout, router) on top. A custom shard engine is only justified if FDB's per-op overhead becomes the bottleneck — plausible at 10M ops/sec, but worth measuring before committing to custom. The design I've drawn is effectively "what you'd build if you decided FDB wasn't enough" — the structure is the same.

**TiKV / CockroachDB range-based sharding.** The PD pattern, Raft-per-range, SST snapshot transfer on split — this whole design borrows heavily. Known issue: hot-range detection latency is the bottleneck on hot-shard mitigation (PD telemetry is typically polled at seconds granularity). For sub-second hot spots, push detection into the shard itself with local auto-split proposals — which is what I specified in §4.

**Disaggregated lakehouse metadata (Delta, Iceberg).** Different problem (table metadata, not object metadata), but the theme — metadata as a first-class service separate from data — is what this question is about. If our tenants run lakehouse workloads, expect LIST patterns dominated by manifest-list queries (many small prefix scans, high read-to-write ratio), which our design handles well.

## 11. Operational Evolution

**[STAFF SIGNAL: operational reality]**

- **Adding a metadata field.** Value is a Protobuf with reserved field tags. New field added with a new tag number; old readers ignore; no migration needed.
- **Changing partitioning strategy.** Shard map is indirection; can add new shards, rebalance ranges, all online. Converting a bucket from non-salted to salted sharding: write new entries under salted keys, background-copy old entries, flip read path via bucket flag, delete old entries. Takes days for a big bucket; runs at 5% of the bucket's steady-state QPS to avoid interference.
- **Storage engine upgrade (RocksDB minor version).** One replica at a time: drain via Raft membership change, upgrade, rejoin. Fleet-wide upgrade ~1 week.
- **Rolling out a new consistency mode.** Per-bucket flag. New buckets opt in at creation; existing buckets migrate by customer request (no-op: flag flip).
- **Rollback path.** Every schema change is behind a flag. Every data migration is idempotent with a reverse script. Observability includes a per-shard "staleness" metric — if a rollout shows staleness spikes, flag flip reverts in under a minute.

## 12. Tradeoffs and What Would Force a Redesign

- **If LIST were not required**: hash partition, done. 10× simpler. S3 Express points the way.
- **If full-text or attribute search on metadata were required**: range partitioning alone can't serve "find all objects with content-type=image/png." I'd add a Lucene-backed secondary index with eventual consistency to metadata. Whole new data plane. Don't pretend this fits in the current design.
- **If multi-region strong consistency were required**: Raft quorum across regions → PUT p99 ~80 ms intercontinental. I'd instead offer **single-region strong + async cross-region** as the default, with a **multi-region consistency tier** (Spanner-like, TrueTime-equivalent) as a premium SKU. Explicit tradeoff exposed to customers via bucket config.
- **If storage cost dominated over latency**: longer compactions, higher compression (ZSTD instead of LZ4), move cold version rows to object storage itself (metadata-of-metadata, with a pointer table in hot tier). Feasible because cold versions are rarely LIST'd.
- **If per-object latency SLO dropped to p99 < 10 ms**: Raft quorum overhead is too big; would need single-replica leases (like Express) or move to an in-memory index tier with async durability. Redesign.

## 13. What I'd Push Back On

**[STAFF SIGNAL: saying no]**

1. **"Strong consistency for a single key" is under-specified.** Silent on LIST consistency, on multi-key transactions (rename? copy-object?), on cross-region. I'd press: "Is linearizable GET-after-PUT enough, or do you want linearizable LIST too?" If the latter, the design changes materially (global timestamps or a witness service) and the cost model shifts. My default is to ship the former and be explicit that LIST is read-committed.

2. **"10M ops/sec peak" is inadequately specified.** Peak over what window? Per-bucket? Per-tenant? Determines whether we need global admission control or per-shard local. I'd push for p99 and p99.9 sustained rates plus a burst profile (e.g., "2× average for 10 seconds").

3. **Implicit "one metadata design fits all workloads" assumption.** The Express-vs-standard split in S3 exists for a reason — latency-sensitive small-object workloads and list-heavy big-data workloads benefit from different shard configurations and possibly different storage engines. I'd ship the general-purpose design first and plan a specialized tier within 12 months.

4. **"p99 < 50 ms" as a single number.** p99 GET vs p99 LIST vs p99 PUT are wildly different problems. Strong-consistency GET at 11 ms happy-path, 50 ms p99 is generous. LIST at 50 ms p99 is tight for cross-shard fans — I'd ask if LIST can get a separate budget (say p99 200 ms for cross-shard LISTs).

5. **Why isn't versioning optional in the scope?** Most S3 buckets don't use versioning. The design cost (key layout, GC, LIST filter) is paid by every bucket if universal. Real S3 makes it a per-bucket opt-in; I've implicitly assumed the same, worth confirming.

---

**Closing.** The single most important move here is committing to range partitioning the moment the LIST requirement lands, and then defending that choice with the hot-shard mitigation stack, the split-correctness protocol, and the versioning key layout that makes current-version GET a single-seek operation. Everything else is a detail that falls out once the keyspace itself is the index.