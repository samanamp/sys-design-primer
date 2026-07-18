---
title: "External Sort & K-Way Merge: When the Data Doesn't Fit"
description: "Sort a 100 GB file with 4 GB of RAM: run generation, heap-based k-way merge, the I/O cost model, buffer arithmetic, multi-pass merging, and where this shows up in MapReduce, LSM compaction, and database spills."
---

# External Sort & K-Way Merge: When the Data Doesn't Fit

"Sort a 100 GB file on a machine with 4 GB of RAM." This is the staff-level escalation Google interviewers reach for after any in-memory sorting or merging question, and it has a canonical two-phase answer: **run generation** (produce sorted chunks that each fit in RAM) then **k-way merge** (stream them back together). Everything else — buffer sizing, pass counting, fan-in limits — is arithmetic on top of that skeleton.

## The two-phase picture

```
Phase 1: run generation (1 read + 1 write of everything)
┌──────────────┐   read 4 GB      ┌─────────────┐   write     ┌──────────────────────┐
│ input.dat    │ ──chunk at a──▶  │ RAM: 4 GB   │ ──sorted──▶ │ run_00 … run_24      │
│ 100 GB       │   time, ×25      │ sort chunk  │   chunk     │ 25 sorted 4 GB runs  │
└──────────────┘                  └─────────────┘             └──────────────────────┘

Phase 2: k-way merge (1 read + 1 write of everything)
run_00 ─┐
run_01 ─┤   25 streams into a      ┌─────────────────────┐      ┌────────────┐
  ⋮     ├──────────────────────▶   │ RAM: min-heap of 25 │ ───▶ │ output.dat │
run_24 ─┘   (sequential reads)     │ heads + buffers     │      │ 100 GB     │
                                   └─────────────────────┘      └────────────┘
```

Total: every byte crosses the disk **twice per phase** — read + write — so this is 2 passes × 2 × 100 GB = 400 GB of I/O, all sequential.

## Phase 1: run generation

Simplest version: read a RAM-sized chunk (4 GB), `sort()` it in memory, write it out as a sorted run, repeat. 100 GB / 4 GB → 25 runs. CPU cost is the same O(n log n) as any sort; the point is each chunk fits.

The classic upgrade is **replacement selection**: keep a heap of the whole RAM budget, pop the minimum, write it, and refill from input — accepting a new record into the *current* run only if it's ≥ the last value written (otherwise it's frozen for the next run). On random input this produces runs averaging **2× RAM** (~8 GB), halving the run count — which matters exactly when run count is what forces extra merge passes. One sentence of this in an interview signals you've seen the literature; don't derive it.

## Phase 2: k-way merge — LeetCode 23 scaled up

This is exactly [23. Merge k Sorted Lists](https://leetcode.com/problems/merge-k-sorted-lists/), except the "lists" are file streams. Min-heap of one head element per run; pop the global min, emit it, refill from whichever run it came from. Trace with 4 runs:

```
run A: 3, 9, 12, …      run B: 1, 4, 20, …
run C: 5, 6, 7, …       run D: 2, 8, 15, …

step  heap (min first)        emit   refill from
 1    [1B, 2D, 3A, 5C]         1     B → push 4
 2    [2D, 3A, 4B, 5C]         2     D → push 8
 3    [3A, 4B, 5C, 8D]         3     A → push 9
 4    [4B, 5C, 8D, 9A]         4     B → push 20
 5    [5C, 8D, 9A, 20B]        5     C → push 6
 6    [6C, 8D, 9A, 20B]        6     C → push 7
      output so far: 1 2 3 4 5 6 …
```

Each of the n elements costs one heap pop + push: **O(n log k)** compares total. When a run is exhausted, the heap just shrinks; merge ends when it's empty.

## The buffering arithmetic

You don't read runs one record at a time — you give each run an input buffer and read whole blocks; likewise one output buffer flushed in blocks. All buffers must fit the RAM budget:

```
RAM = 4 GB
┌──────────────────────────────────────────────────────────┐
│ in-buf run_00 │ in-buf run_01 │ … │ in-buf run_24 │ out  │
│    ~155 MB    │    ~155 MB    │   │    ~155 MB    │ buf  │
└──────────────────────────────────────────────────────────┘
   25 × in-buf  +  1 × out-buf  ≤ 4 GB
   → buffer ≈ 4 GB / 26 ≈ 157 MB each
```

157 MB reads are enormous relative to seek time, so the merge stays effectively sequential. This inequality is the real limit on fan-in k: bigger k → more runs merged per pass, but smaller per-run buffers → more, smaller reads. Below ~1–10 MB per buffer on HDD, seek overhead starts eating you; that's the practical ceiling on k, not the heap.

## I/O cost model and multi-pass merges

**Total I/O = 2 × data × passes** (each pass reads and writes everything once). Disk bandwidth dwarfs CPU cost here, so you minimize *passes*, which means maximizing k. Passes = 1 (run generation) + ⌈**log_k(runs)**⌉ (merging).

If runs > affordable k, merge in waves. Say 2.5 TB with 4 GB RAM → 625 runs, and buffer math caps k at 25:

```
625 runs ──25-way merges──▶ 25 runs (each 100 GB) ──25-way merge──▶ 1 sorted file
              pass 2                                    pass 3
passes = 1 + log₂₅(625) = 1 + 2 = 3   →  total I/O = 2 × 2.5 TB × 3 = 15 TB
```

With k = 25 and one pre-pass, k² × RAM-sized runs = 625 × 4 GB = 2.5 TB sorts in two merge passes; a single merge pass handles up to k × RAM = 100 GB. The reachable data size grows exponentially in passes — which is why real datasets almost never need more than 2–3.

## This is the real world, not a puzzle

- **MapReduce / Spark shuffle**: mappers spill sorted runs to local disk; reducers fetch and k-way merge them. "Sort a 100 GB file" *is* the shuffle, in miniature.
- **LSM-tree compaction**: merging sorted SSTables is exactly this k-way merge — see the [KV Store](/coding/1-kvstore/) design.
- **Database `ORDER BY` / merge joins**: when the sort exceeds `work_mem` (Postgres) the planner switches to "external merge" — this algorithm, verbatim, in `EXPLAIN` output.
- **Log merging**: combining time-ordered log streams from many machines into one timeline is the merge phase alone.

## The 60-second spoken escalation

You've just solved merge-k-lists or "sort this array," and the interviewer says *"now it's a 100 GB file, 4 GB of RAM."* Say:

> "The data doesn't fit, so I'll do an external merge sort. Pass one: stream the file in 4 GB chunks, sort each in memory, write 25 sorted runs to disk. Pass two: open all 25 runs, k-way merge with a min-heap of the head elements — exactly merge-k-lists, but the lists are files — writing the output as a stream. I'd give each run a big input buffer, roughly 4 GB / 26 ≈ 150 MB, so all disk I/O is sequential. Total cost: two passes, so 4× the data in I/O, and O(n log k) compares. If there were too many runs to merge at once, I'd merge in waves — log_k(runs) passes — which is why you maximize fan-in."

## Minimal Python

```python
import heapq

def sorted_runs(path, mem_records):
    """Pass 1: yield paths of sorted run files."""
    with open(path) as f:
        i = 0
        while chunk := f.readlines(mem_records):
            chunk.sort()
            run = f"run_{i:03d}"
            with open(run, "w") as out:
                out.writelines(chunk)
            yield run
            i += 1

def kway_merge(run_paths, out_path):
    """Pass 2: heap-based merge of sorted file streams."""
    files = [open(p) for p in run_paths]
    with open(out_path, "w") as out:
        out.writelines(heapq.merge(*files))   # streaming, O(n log k)
    for f in files:
        f.close()

kway_merge(list(sorted_runs("input.txt", 4 * 2**30)), "sorted.txt")
```

`heapq.merge` does the pop/emit/refill cycle lazily over iterators — the whole phase-2 heap in one call. In an interview, sketching it manually (heap of `(value, run_index)`, refill on pop) shows you own it.

## Complexity

| | Cost |
|---|---|
| Compares | O(n log n) total — run sort O(n log n) + merge O(n log k) per pass |
| I/O | 2 × data × passes; passes = 1 + ⌈log_k(runs)⌉ |
| Memory | RAM budget B; fan-in k ≈ B / buffer_size − 1 |
| One-merge-pass capacity | k × B (≈ 100 GB at B = 4 GB, k = 25) |

## Interview Q&A

**Q: Sort 1 TB with 1 GB of RAM — walk the numbers.**
Run generation: 1 TB / 1 GB = 1,000 runs. Can we merge 1,000 at once? Buffers: 1 GB / 1,001 ≈ 1 MB each — marginal on HDD, fine on SSD. Conservatively, two merge waves: 1,000 → ~32 runs (k=32, 31 MB buffers) → 1. Three passes total, 6 TB of I/O; at ~200 MB/s sequential, on the order of 8 hours on one spindle — which is the cue to say "and this is why you'd shard it across machines, i.e., MapReduce."

**Q: Why maximize fan-in k?**
I/O scales with passes = 1 + log_k(runs); CPU heap cost O(n log k) grows only logarithmically and disk is the bottleneck anyway. Bigger k → fewer passes → less total I/O. The limit is buffer size: k + 1 buffers must fit in RAM and each must stay big enough to amortize seeks.

**Q: What changes on SSD vs HDD?**
No seek penalty, so tiny buffers stop hurting and you can push k much higher — often merging everything in one pass. Sequential and random bandwidth converge; the pass-count formula still rules, you just hit 1 merge pass sooner. (Write endurance is the new soft constraint at scale.)

**Q: How does MapReduce sort a petabyte?**
Partition first: mappers range-partition or hash keys so each reducer owns a key range; each mapper spill is sorted and spilled as runs (this algorithm, per-node), reducers k-way merge the fetched runs. External sort per node + partitioning across nodes — the distributed answer is this doc plus a shard function.
