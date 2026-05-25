---
title: "Log-structured append-only KV (LSM lite)"
description: "Log-structured append-only KV (LSM lite)"
---



**Prompt:** `put(k, v)`, `get(k)`, `delete(k)`. Writes go to an append-only log; reads must be fast. Support compaction.

**What's tested:** can you reason about write-amp vs read-amp, and do you know what an SSTable is.

**The structure:**
- **MemTable:** in-memory sorted dict (red-black tree, skiplist, or just `SortedDict`). All writes go here first.
- **Write-ahead log:** every write also appended to disk for crash recovery. Drop on memtable flush.
- **SSTables on disk:** when memtable hits size threshold, flush as a sorted immutable file. Each SSTable has a sparse index in memory (every Nth key → file offset) and optionally a Bloom filter.
- **Read path:** check memtable → check SSTables newest-to-oldest until hit or all checked. Bloom filter short-circuits negative lookups.
- **Compaction:** periodically merge multiple SSTables into one, dropping superseded entries and tombstones.

**Deletes are tombstones**, same as the nested-transaction problem. They get cleaned up at compaction.

**Staff signal moves:**
- Name the write-amp / read-amp / space-amp tradeoff explicitly. LSM trades read-amp for write-amp; B-trees do the opposite.
- Mention leveled vs tiered compaction (RocksDB vs Cassandra defaults). Don't implement; just know the names.
- For the in-memory interview version, skip the WAL and the disk — keep memtable + a list of immutable sorted layers in memory. Compaction merges layers. The structure is identical; you've stripped the I/O.
- Bloom filter sizing: `~10 bits/key gives ~1% FP rate`. Memorize this.

---

```py
import bisect

TOMBSTONE = object()
THRESH = 4  # flush memtable at this size

class LSM:
    def __init__(self):
        self.mem = {}
        self.layers = []  # newest first; each is sorted list[(k, v)]

    def put(self, k, v):
        self.mem[k] = v
        if len(self.mem) >= THRESH:
            self.layers.insert(0, sorted(self.mem.items()))
            self.mem = {}

    def delete(self, k):
        self.put(k, TOMBSTONE)

    def get(self, k):
        if k in self.mem:
            v = self.mem[k]
        else:
            v = TOMBSTONE  # "not found" collapses into same return path
            for layer in self.layers:
                i = bisect.bisect_left(layer, (k,))
                if i < len(layer) and layer[i][0] == k:
                    v = layer[i][1]; break
        return None if v is TOMBSTONE else v

    def compact(self):  # full compaction: merge all layers, drop tombstones
        seen, out = set(), []
        for layer in self.layers:           # newest to oldest
            for k, v in layer:
                if k not in seen:
                    seen.add(k)
                    if v is not TOMBSTONE:
                        out.append((k, v))
        out.sort()
        self.layers = [out] if out else []
```


