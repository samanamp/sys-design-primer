---
title: Storage Engines & Data Layouts
description: Storage Engines & Data Layouts
---

> Goal: enough storage-engine fluency to make strong design choices, defend tradeoffs, and not sound shallow. Not enough to build one.

---

## 1. What a staff engineer actually needs to know

**What interviewers test**

- Can you pick the right engine/layout for a given workload and defend it?
- Can you name the bottleneck that appears first at scale?
- Can you articulate tradeoffs in the language of amplifications (read/write/space), tail latency, and ops burden?
- Can you distinguish durability, replication, and consistency?

**What does not matter**

- Memorizing specific B-tree variants (B+, B*, Bw), exact RocksDB flags, specific page sizes.
- Pseudocode of compaction algorithms.
- Vendor trivia ("Aurora vs Spanner internals").

**Expected depth**

- Explain WAL, LSM vs B-tree, compaction, bloom filters, row vs column, and hot-partition mitigation **without stalling**.
- Give a credible number when asked (e.g., "LSM write amp is typically 10–30x under leveled compaction"). Order of magnitude is enough.
- Know when to stop: "I'd benchmark this" is a valid terminal move for anything past back-of-envelope.

---

## 2. Core mental model

**Storage engine vs database.** The storage engine is the component that turns keys and values into durable bytes on disk and back. A database adds query parsing, planning, transactions, schema, replication, and client protocols on top. Postgres uses a B-tree engine; Cassandra uses LSM; ClickHouse uses columnar (MergeTree). Same engine family can power very different databases.

**The three paths.**

```
WRITE PATH (typical durable engine)
 client -> parse/validate -> WAL append (fsync) -> in-memory structure -> ack
                                                         |
                                                         v
                                                  (async) flush to disk

READ PATH
 client -> cache (app/CDN) -> DB query cache -> buffer/block cache
        -> index lookup -> data page/block -> decode -> return

DURABILITY PATH
 write to WAL -> fsync -> OS durably on disk -> only then ack
 (replication is a SEPARATE dimension: durable-here vs durable-elsewhere)
```

**Memory vs disk vs cache — the only ratios that matter in interviews.**

```
L1/L2 cache         : ~1 ns
DRAM                : ~100 ns          (~100x slower than cache)
NVMe SSD random 4KB : ~100 us          (~1000x slower than DRAM)
NVMe sequential     : ~1-3 GB/s
HDD random          : ~10 ms           (~100x slower than SSD random)
Network RTT (DC)    : ~500 us
Network RTT (region): ~10-100 ms
```

Implication: random I/O is the enemy; sequential I/O is cheap; network within a DC is roughly SSD-latency; cross-region RTT dominates everything.

**Why workload shape matters.** Engines optimize for different shapes. You must state the shape early:

- Read/write ratio (95/5 vs 50/50 vs 5/95).
- Key distribution (uniform vs zipfian vs monotonic).
- Value size (100 B vs 1 MB).
- Access pattern (point vs range vs full scan).
- Freshness (can reads be stale? by how much?).
- Working set vs total data (fits in RAM? 10%? 1%?).

If you don't state these, you can't justify any engine choice.

---

## 3. The big 3 design families

### 3.1 B-tree / page-oriented (Postgres, MySQL/InnoDB, SQL Server)

**Shape.** Balanced tree of fixed-size pages (typically 4–16 KB). Internal nodes = routing keys. Leaf nodes = data (clustered) or pointers (non-clustered). Leaves usually linked for range scans.

```
              [ 40 | 80 ]              <- internal
             /     |     \
        [..30]  [50..70] [90..]        <- leaves (linked list)
         data    data      data
```

**Writes.** Locate leaf page → log change to WAL → modify page in the buffer pool → page eventually flushed. Updates are **in-place**. Splits when full; merges when underfull.

**Reads.** Tree traversal: ~3–4 hops for billions of rows with high fanout. Point reads and small range scans are excellent.

**Range scans.** Sequential via linked leaves. Very good if clustering order matches the scan.

**Strengths**

- Low read amplification (few page reads per lookup).
- Mature concurrency (MVCC, row locking, rich SQL).
- Good mixed OLTP behavior; predictable latencies.

**Weaknesses**

- Write amp from page-level updates: modifying a 100-byte row can rewrite a 16 KB page.
- Random I/O under heavy write churn.
- Page splits cause fragmentation; periodic VACUUM/rebuild needed.
- Scaling writes vertically has a ceiling.

**Best fit.** OLTP, read-heavy, mixed workloads where working set mostly fits in memory, transactions with rich secondary indexes.

**What interviewers want to hear**

- "It's page-oriented with in-place updates."
- "Random writes + fsync-per-commit are the bottleneck; group commit and WAL buffering matter."
- "Write amp = page/row size; problematic for tiny hot rows in a huge table."

### 3.2 LSM engines (RocksDB, LevelDB, Cassandra, ScyllaDB, HBase, BadgerDB)

**Shape.** In-memory sorted buffer (memtable) + WAL + a stack of immutable sorted files (SSTables) on disk, organized into levels.

```
memtable (in RAM, mutable, sorted)
    |
    v (flush when full)
L0: [SST][SST][SST]   <- may overlap
L1: [  SST SST SST ]  <- non-overlapping, ~10x L0
L2: [       ...      ] <- ~10x L1
...
```

**Writes.** Append to WAL → insert into memtable → ack. Memtable fills → made immutable → flushed as a new SSTable. Background **compaction** merges SSTables, discards old versions and tombstones.

**Reads.** Check memtable → check L0 SSTables → descend levels. Each SSTable has a bloom filter and an index block; bloom filter skips SSTables that definitely don't contain the key.

```
Point read cost (worst case, leveled):
  1 memtable probe + N_L0 bloom checks + 1 read per level beyond L0
```

**Range scans.** Merge-iterate across memtable and SSTables. Higher cost than B-tree because you must merge from multiple sources.

**Compaction strategies (know the two)**

- **Leveled** (RocksDB default): each level is non-overlapping and ~10x larger than the previous. Low space amp, low read amp, **high write amp** (10–30x).
- **Tiered/size-tiered** (Cassandra default legacy): groups of similar-sized SSTables compacted together. Lower write amp, higher space and read amp.

**Strengths**

- Writes are sequential → huge write throughput.
- Compression-friendly (SSTables are sorted blocks).
- Embeddable and simple to operate in single-node form (RocksDB).
- Snapshots are cheap (SSTables are immutable).

**Weaknesses**

- Read amp: may touch multiple SSTables per key.
- Write amp: compaction rewrites data many times; eats SSD endurance and IOPS.
- Space amp: dead versions and tombstones linger until compaction.
- Compaction causes tail-latency spikes and IO backpressure; p99.9 suffers.
- Range scans are measurably slower than B-tree equivalents.

**Best fit.** Write-heavy, log-like ingest, time-series, KV stores, metadata stores where throughput and write cost matter more than p99 read tail.

**What interviewers want to hear**

- "Sequential writes, merged compaction. Bloom filters gate SSTable reads."
- "Tradeoff triangle: write amp vs read amp vs space amp — you can't minimize all three; compaction strategy picks two."
- "Compaction is not free background work — it's a capacity-planning input."

### 3.3 Columnar (ClickHouse, Parquet, Druid, BigQuery, Redshift, Snowflake, Vertica)

**Shape.** Store values of a single column contiguously rather than rows contiguously. Typically chunked into row-groups/parts with per-column compressed blocks and min/max stats.

```
Row layout   : [r1c1 r1c2 r1c3][r2c1 r2c2 r2c3]...
Column layout: [r1c1 r2c1 r3c1 ...] [r1c2 r2c2 r3c2 ...] [r1c3 r2c3 ...]
```

**Writes.** Usually batch/append into new parts. Point updates are expensive — often implemented as "write a delete marker + new version, merge later" (ClickHouse ReplacingMergeTree, Snowflake micro-partitions). Avoid columnar for high-rate point updates.

**Reads.** Only read the columns the query touches. Compression ratios are very high (RLE, dictionary, delta, frame-of-reference) because adjacent values in one column are similar. Block-level min/max stats enable **predicate pushdown** and block skipping.

**Range scans.** The native case. Vectorized execution processes columns in SIMD-friendly chunks.

**Strengths**

- Massive scan throughput on selected columns (often 10–100x vs row store for analytics).
- Compression reduces both storage and I/O.
- Predicate pushdown + block skipping → queries touch a small fraction of data.

**Weaknesses**

- Bad for point lookups (must stitch row from many column files).
- Bad for high-rate mutations.
- Not transactional in the OLTP sense.

**Best fit.** OLAP, analytics, metrics, logs, data warehouse, time-series aggregates.

**What interviewers want to hear**

- "Row reconstruction is the cost you pay for scan efficiency."
- "Compression is a primary feature, not an optimization — it's what makes columnar fast."
- "Use columnar for read-heavy aggregations on wide tables with selective columns; not for OLTP."

---

## 4. Must-know concepts

**WAL (write-ahead log).** Append-only sequential log. All mutations are logged before being applied to in-memory/disk structures. `fsync` on the WAL before ack is what gives you durability. On crash, replay WAL to rebuild state. Key interview points:

- Group commit batches fsyncs to amortize cost.
- WAL is local durability; it does **not** give you replication.
- WAL size controls recovery time — truncated by checkpoints.

**Memtable.** In-memory sorted structure (skip list, B-tree, or similar) holding recent writes in an LSM. When it hits a size threshold, it's frozen and flushed. Typically 64–256 MB.

**SSTable (Sorted String Table).** Immutable on-disk file containing sorted key-value pairs, a sparse index block (one entry per data block), and a bloom filter. Immutability is the magic: concurrent reads need no locks; compaction produces new files and atomically swaps.

```
SSTable layout:
  [ data blocks ][ index block ][ bloom filter ][ footer ]
   (compressed)   (key -> offset)
```

**Compaction.** Merges SSTables to:

1. reclaim space from overwritten keys and tombstones,
2. reduce the number of files a read must consult,
3. move data to larger levels.

Two flavors:

- **Leveled**: keeps each level non-overlapping. Each key exists at most once per level (beyond L0). Low read/space amp, high write amp.
- **Tiered**: runs of similar-sized files merged together when enough accumulate. Lower write amp, worse read/space amp.

Universal/hybrid policies exist but interview-depth is "leveled vs tiered."

**Tombstones.** Deletes in LSMs are writes of a marker. The key is not physically gone until compaction reaches a level that has no older version. Problem: tombstone accumulation in wide-row workloads (Cassandra's classic footgun — scanning a partition with many tombstones is O(tombstones), not O(live rows)).

**Bloom filters.** Bit array + k hashes. `contains(k)` returns "definitely not" or "probably yes." Per-SSTable filters let readers skip files without reading them. Typical sizing: 10 bits/key → ~1% false positive rate. Cost: memory and per-lookup hashing. Interview line: "Bloom filters turn LSM point reads from O(levels × SSTables) disk seeks into O(1–2) most of the time."

**MVCC (multi-version concurrency control).** Each write creates a new version stamped with a transaction ID or timestamp. Readers at snapshot T see the newest version with stamp ≤ T. Writers don't block readers; readers don't block writers. Old versions garbage-collected once no snapshot needs them. Postgres does it by keeping old row versions in the heap (→ VACUUM). LSMs get it nearly for free because SSTables are immutable and already versioned.

**Primary vs secondary indexes.**

- **Primary (clustered)**: determines physical storage order. Lookup by primary key → data. Range scans on PK are cheap.
- **Secondary**: separate structure mapping secondary key → primary key (or row ID). Lookup by secondary key requires a second hop to fetch the row (unless it's a **covering index** that stores needed columns inline).

Staff-level insight: every secondary index doubles or triples your write cost. Index count is a first-class capacity decision.

**Clustering / sorted order.** Rows with adjacent keys stored adjacently. Enables efficient range scans and good cache locality. Wide-column stores like Cassandra and Bigtable let you pick a **partition key** (shard) and **clustering key** (sort order within the partition). The clustering key determines what ranges you can scan cheaply.

**Read amplification.** Disk reads (or blocks) per logical read.

- B-tree: ~height of tree (often 3–4).
- LSM leveled: 1 memtable + 1 per level, mitigated by bloom filters.
- Columnar: scan of relevant blocks, typically dominant over index.

**Write amplification.** Bytes written to disk per byte of user data.

- B-tree: ~page_size / row_size (can be 100x for tiny rows).
- LSM leveled: ~10–30x (each key rewritten ~once per level).
- LSM tiered: ~5–10x but with worse space/read amp.
- Columnar (append-only): ~1x until you need to rewrite parts, then spikes.

Why it matters: SSD endurance (TBW budget) and IOPS cost. A 10x write amp means 10x the storage budget for the same user-write rate.

**Space amplification.** On-disk bytes / logical live bytes.

- B-tree: ~1.1–1.5x (fragmentation, fillfactor).
- LSM leveled: ~1.1x.
- LSM tiered: ~2x+ (multiple live copies between compactions).
- MVCC systems: depends on snapshot retention and update rate.

**Cache layers.** Not free. Each layer has coherence cost.

- OS page cache: implicit, shared, no app knowledge.
- Engine block/buffer cache: explicit, engine-aware (eviction, pinning).
- Query/result cache: invalidation is the hard part.
- App cache / external cache (Redis, memcached): network hop, stale-read risk, thundering herd on miss.
- CDN: for read-mostly static content.

The cache-miss cliff is the thing: working set fits → everything is fast; exceeds cache → latency jumps by 10–100x.

---

## 5. Data layout decisions

**Row vs column.** Decide on access pattern first:

- Reading most columns of few rows → row.
- Reading few columns of most rows → column.
- Writing full records at high rate → row.
- Updating individual fields rarely, scanning often → column.

Hybrid "PAX"-style layouts (Parquet row groups) blend both; interview answer "Parquet is columnar within row groups" is sufficient.

**Key-value vs document vs wide-column.**

- **KV**: opaque bytes, simplest, scales horizontally, no server-side filtering beyond key. Use for caches, session stores, blob-ish data.
- **Document**: structured values (JSON/BSON), optional schema, indexable paths. Use for product catalogs, user profiles, content where access is mostly by ID with some secondary lookups.
- **Wide-column (Cassandra, Bigtable, HBase)**: keyed by partition key + clustering key, with sparse columns per row. Use for time-series and event data with natural partitioning (per-user, per-device) and per-partition range scans.

**Append-only vs in-place update.**

- Append-only: simpler concurrency, natural MVCC, cheap snapshots, higher space amp, needs compaction/GC.
- In-place: lower space amp, harder concurrency (locks/latches), fragmentation over time.

LSMs and columnar systems are append-only. B-trees are in-place. Event-sourced systems are append-only by design.

**Clustered vs non-clustered access.**

- Clustered: rows near in key are near on disk. Cache-hot for range scans. Dangerous with monotonic keys: writes hit one shard (hot partition). Common fix: hash the key or prepend a short hash prefix.
- Non-clustered: random physical placement. Uniform load, bad range behavior.

**Hot keys / hot partitions.** The single most common real-world failure mode at scale.

- Symptom: one shard at 100% while the rest idle; tail latency detonates.
- Causes: monotonic timestamp keys, celebrity users, sequential IDs, per-minute buckets.
- Fixes:
    - Salt/hash the partition key (lose range scans on it).
    - Use compound keys (tenant_id, timestamp).
    - Add a read-through cache for hot items.
    - Write-through with async fanout for hot writers.

Say this unprompted in any sharded-store design. It demonstrates operational awareness.

**Locality and range scans.** If the product requires "get all X for user Y ordered by time," your clustering key must be `(Y, time)`, your partition key should group by Y, and your engine must support ordered scans. Choosing a pure hash-sharded KV here is a red flag.

---

## 6. Interview reasoning patterns

**"Why RocksDB-like storage here?"** Say it when:

- Write-heavy ingest (logs, events, metadata updates).
- Key-based access with occasional range scans.
- Need embeddable/library-level storage (one-binary service).
- Snapshot/backup friendliness matters.
- You're fine eating read-side bloom-filter work and compaction ops.

**"Why not Postgres/MySQL here?"** Say it when:

- Sustained write rate > ~10–50k rows/s on one node starts to hurt (vacuum, WAL pressure, lock contention).
- Data is fundamentally append-oriented logs, not updatable entities.
- Schema churn would be painful (rigid DDL at scale).
- You need horizontal write scaling without sharding pain (though Postgres + Citus/Aurora narrows this).
- Working set is many TB with mixed cold data — buffer pool can't help.

Always add: "Postgres is a good default for OLTP up to single-node ceilings; I'd push toward LSM or dedicated systems only when specific workload pressure demands it."

**"When is an LSM engine a bad fit?"**

- Strict p99/p99.9 read latency SLAs (compaction spikes, deep reads).
- Workload dominated by large range scans with predictable order (columnar or clustered B-tree wins).
- Very small datasets fully in RAM (B-tree/in-memory index simpler).
- Rich relational joins and transactions across many tables.

**"When does columnar help?"**

- Analytics, aggregations, BI dashboards over wide tables.
- Log analytics and metrics (high cardinality, selective column queries).
- Read-heavy; bulk ingest patterns; tolerable latency for mutations.
- "We're computing `SELECT sum(x) WHERE y > ...` over billions of rows daily" → columnar.

**"What bottlenecks appear first?"** Order of appearance in a growing system:

1. Single hot partition / hot key.
2. Write fsync throughput (or compaction IOPS for LSM).
3. Working set exceeding buffer/page cache → read latency cliff.
4. Secondary index write amplification.
5. Replication lag under write bursts.
6. Backup/restore window exceeding the RPO.
7. Schema migration time on huge tables.

**"What operational pain points matter?"**

- Compaction backpressure (LSM): surfaces as IO saturation and tail spikes.
- Vacuum / bloat (Postgres MVCC): long-running transactions prevent cleanup → table bloat.
- Rebuild time after failure: WAL replay, snapshot restore, index rebuild.
- Schema change at scale: online DDL, shadow tables, backfills.
- Capacity planning: write amp × user-write-rate × retention = real SSD bytes.

---

## 7. Common mistakes candidates make

- **"LSM is faster."** It's faster for writes. Reads often hit multiple SSTables; p99 is worse than B-tree in steady state.
- **"Kafka is storage."** Kafka is an ordered append-only log. No random reads by key, no secondary indexes, no updates. Use it as a durable pipe or event log, not as a database.
- **Confusing durability and replication.** Durability = survives crash on this machine (fsync). Replication = survives loss of this machine (copies). You need both; they solve different failures. Say this distinction out loud.
- **Hand-waving compaction.** "Background compaction handles it" is a red flag. Quantify: "Leveled compaction rewrites each byte ~N times where N ≈ number of levels, so write amp is ~10–30x and I need to budget SSD IOPS for it."
- **Ignoring range scans.** If the product has "list my recent X" or "show items between A and B," layout matters. Pure hash partitioning kills range scans.
- **Ignoring write amplification.** 10x write amp on a 100 MB/s user write rate means 1 GB/s to disk. That's a capacity decision, not a footnote.
- **Treating cache as free.** "We'll cache it" without sizing the working set, discussing invalidation, or mentioning the cache-miss cliff.
- **Conflating OLTP and OLAP engines.** Proposing Postgres for 100 TB analytical scans, or ClickHouse for per-user transactional updates.
- **Saying "eventually consistent" without a bound.** "Eventually" is not a number. Give a window (ms, seconds, minutes) and a convergence mechanism.
- **Forgetting the hot partition.** Any sharded design needs a hot-partition story. Missing it is a staff-level red flag.

---

## 8. Final cheat sheet

### 8.1 Comparison table

|Dimension|B-tree|LSM|Columnar|
|---|---|---|---|
|Write path|In-place, random I/O|Append to WAL + memtable, sequential|Batch append, sequential|
|Read (point)|Tree traversal, low amp|Memtable + SSTables, bloom-gated|Expensive; must reconstruct row|
|Range scan|Excellent (linked leaves)|Good (merge-iterate)|Excellent for column subsets|
|Write amp|Low–medium (page-level)|High (10–30x leveled)|Low until part rewrites|
|Read amp|Low (~tree height)|Medium (levels, mitigated by bloom)|Scan-oriented; column-selective|
|Space amp|Low (~1.1–1.5x)|Medium (compaction-gated)|Low (compression helps)|
|Mutations|Cheap in place|Cheap append, cost paid in compaction|Expensive|
|p99 tail|Predictable|Spiky under compaction|Query-bound|
|Best for|OLTP, mixed, read-heavy|Write-heavy KV, time-series, logs|OLAP, analytics, metrics|
|Examples|Postgres, MySQL, SQLite|RocksDB, Cassandra, Scylla|ClickHouse, Druid, Parquet|

### 8.2 Decision framework (use in interviews)

```
1. State the workload shape:
   - R/W ratio, value size, access pattern, freshness, working set size.

2. Ask the killer question:
   - "Point reads or range scans dominant?"
   - "Mutations per second and per key?"
   - "p99 latency target vs throughput target?"

3. Eliminate:
   - Analytical scans, many columns unused -> columnar.
   - Sustained high write rate, simple keyed access -> LSM.
   - Mixed OLTP, transactions, moderate scale -> B-tree (Postgres/MySQL).

4. Address the usual suspects:
   - Sharding key and hot partitions.
   - Index strategy (PK, secondary, covering).
   - Durability (WAL + fsync policy) and replication (sync/async, quorum).
   - Caching layer and working-set sizing.
   - Backup/restore window vs data size.

5. Name the first bottleneck you expect and your mitigation.
```

### 8.3 Ten likely interview questions, short strong answers

**1. Walk me through a write in an LSM engine.** Append to WAL and fsync for durability. Insert into in-memory sorted memtable. Ack. When memtable fills, it's frozen and flushed as an immutable SSTable. Background compaction merges SSTables, removes obsolete versions and tombstones, and pushes data to deeper levels.

**2. Walk me through a read in an LSM engine.** Check memtable. Then, for each level, consult the bloom filter per SSTable; on a "maybe," read the index block, then the data block. Return the newest version found. Bloom filters keep most point reads to O(1–2) disk reads.

**3. When would you pick Cassandra over Postgres?** Very high sustained write throughput with partitionable keys; multi-region active-active with tunable consistency; data naturally wide-column and time-ordered within a partition; willingness to give up joins, FKs, and rich transactions. Not when you need strong consistency, transactions across partitions, or complex ad-hoc queries.

**4. Explain write amplification and why it matters.** Ratio of bytes written to disk vs bytes written by the user. Each rewrite (compaction, page update) multiplies it. Matters because SSDs have finite endurance and IOPS budgets; a 20x write amp turns 100 MB/s of user writes into 2 GB/s of device writes — that drives disk type, count, and cost.

**5. What's the difference between durability and replication?** Durability protects against crash of this machine — WAL + fsync. Replication protects against loss of this machine — copies on others. Synchronous replication provides durability-across-failures but adds latency. You need a policy for both; they don't substitute.

**6. Why are bloom filters useful in LSM engines?** They let a reader skip an SSTable entirely when the key is definitely absent. Without them, point reads would touch every SSTable at every level. With them, most SSTables are skipped cheaply, keeping point reads at a handful of disk reads.

**7. You have a hot partition. What do you do?** Identify the cause — monotonic key, celebrity entity, burst writer. Mitigations: salt/hash the partition key (trades off range scans on that key), move to a compound key that distributes load, add a read-through cache for hot items, shed load with rate limits, promote hot data to a separate tier, or denormalize to spread writes across shards.

**8. Row vs columnar — how do you decide?** Access pattern. If queries read most columns of a few rows and mutate records, row. If queries read few columns over many rows (aggregations, scans), columnar. Rate of mutation matters too: columnar handles point updates badly. Mixed workloads often run both, with ingest to a row store and a background export to a columnar warehouse.

**9. Why not just use Postgres for everything?** Fine default up to single-node write ceilings (usually single-digit TB, tens of thousands of writes/s). Pressure points: vacuum overhead under heavy update churn, WAL fsync bandwidth, buffer-pool misses as working set exceeds RAM, index write amp on wide indexes, schema-change latency on huge tables, and analytical scans that want columnar. Fork to specialized systems when a specific pressure dominates.

**10. What's the first bottleneck you'd expect in a sharded OLTP system at 100k writes/s?** Hot partition driven by key distribution, long before aggregate throughput. After that, fsync bandwidth on the WAL of individual shards and secondary-index write amplification. I'd measure per-shard QPS distribution and p99, not just the aggregate.

---

### Quick mental triggers

- "Durable" → WAL + fsync.
- "Replicated" → copies + consistency model.
- "Write-heavy" → LSM, sequential I/O, compaction budget.
- "Read-heavy OLTP" → B-tree, working set in RAM, cache.
- "Analytical" → columnar, compression, predicate pushdown.
- "Range scans" → clustering key, layout matters.
- "Point lookups" → bloom filters / hash / primary index.
- "Hot partition" → hash/salt key, rethink partitioning.
- "p99 tail" → compaction pressure, GC, cache miss cliff.
- "TB-scale" → amplifications × retention = capacity plan.