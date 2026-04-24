---
title: KV store
description: KV store
---
```
Solve this coding/system-design interview question in Python:

Design an in-memory time-based key-value store with:
- set(key, value, ts)
- get(key, ts): return the latest value at or before ts
Assume timestamps for a given key are initially non-decreasing.

Then extend it with:
- delete(key, ts)
- range_query(key, start_ts, end_ts)
- TTL support
```

## Reference answer — structure it like this

### 1. API contract (state before coding)

```
set(key, value, ts, ttl=None)
  - Writes version (ts, value). ts is caller-supplied logical time.
  - TTL in seconds, measured against SERVER wall-clock at insert. TTL is
    decoupled from ts on purpose: mixing caller clocks into expiry invites
    drift bugs.
  - Assumption: ts is non-decreasing per key. If violated we fall back to
    insort (O(n)) rather than crash.

get(key, ts) -> value | None
  - Latest LIVE version with ts' <= ts. Live = not tombstone and not expired.
  - Tombstone at ts' <= ts masks older versions.

delete(key, ts)
  - Writes a tombstone at ts. Log-structured: history is preserved so range
    queries and time-travel reads stay coherent. Compactor reclaims later.

range_query(key, start_ts, end_ts) -> List[(ts, value)]
  - All LIVE versions with start_ts <= ts <= end_ts, ascending.

Duplicate-ts rule: last writer wins at query time.
```

### 2. Implementation

```python
import bisect, heapq, threading, time
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

_TOMBSTONE = object()

@dataclass
class _Entry:
    ts: int
    value: Any           # or _TOMBSTONE
    expires_at: float    # wall-clock; inf if no TTL


class TimeKVStore:
    def __init__(self, reaper_interval: float = 1.0):
        self._data: dict = defaultdict(list)               # key -> [_Entry] sorted by ts
        self._key_locks: dict = defaultdict(threading.RLock)
        self._ttl_heap: list = []                          # (expires_at, key, ts)
        self._ttl_lock = threading.Lock()
        self._stop = threading.Event()
        self._reaper = threading.Thread(
            target=self._reap_loop, args=(reaper_interval,), daemon=True
        )
        self._reaper.start()

    # ---- public ----
    def set(self, key: str, value: Any, ts: int, ttl: Optional[float] = None) -> None:
        self._write(key, ts, value, ttl)

    def delete(self, key: str, ts: int) -> None:
        self._write(key, ts, _TOMBSTONE, None)

    def get(self, key: str, ts: int) -> Optional[Any]:
        with self._key_locks[key]:
            entries = self._data.get(key)
            if not entries:
                return None
            i = self._bisect_right_ts(entries, ts) - 1
            now = time.time()
            # Walk back past expired entries (amortized O(1) with reaper).
            # A tombstone short-circuits to None.
            while i >= 0:
                e = entries[i]
                if e.value is _TOMBSTONE:
                    return None
                if e.expires_at > now:
                    return e.value
                i -= 1
            return None

    def range_query(self, key: str, start_ts: int, end_ts: int) -> List[Tuple[int, Any]]:
        if start_ts > end_ts:
            return []
        with self._key_locks[key]:
            entries = self._data.get(key)
            if not entries:
                return []
            lo = self._bisect_left_ts(entries, start_ts)
            hi = self._bisect_right_ts(entries, end_ts)
            now = time.time()
            return [(e.ts, e.value) for e in entries[lo:hi]
                    if e.value is not _TOMBSTONE and e.expires_at > now]

    def close(self) -> None:
        self._stop.set()
        self._reaper.join(timeout=2.0)

    # ---- internals ----
    def _write(self, key: str, ts: int, value: Any, ttl: Optional[float]) -> None:
        expires_at = time.time() + ttl if ttl is not None else float("inf")
        entry = _Entry(ts, value, expires_at)
        with self._key_locks[key]:
            entries = self._data[key]
            if entries and ts < entries[-1].ts:
                # Out-of-order. Under stated assumptions this shouldn't happen;
                # we degrade to O(n) insort rather than corrupt the index.
                idx = bisect.bisect_right([e.ts for e in entries], ts)
                entries.insert(idx, entry)
            else:
                entries.append(entry)
        if ttl is not None:
            with self._ttl_lock:
                heapq.heappush(self._ttl_heap, (expires_at, key, ts))

    @staticmethod
    def _bisect_left_ts(entries, ts):
        lo, hi = 0, len(entries)
        while lo < hi:
            mid = (lo + hi) // 2
            if entries[mid].ts < ts: lo = mid + 1
            else:                    hi = mid
        return lo

    @staticmethod
    def _bisect_right_ts(entries, ts):
        lo, hi = 0, len(entries)
        while lo < hi:
            mid = (lo + hi) // 2
            if entries[mid].ts <= ts: lo = mid + 1
            else:                     hi = mid
        return lo

    def _reap_loop(self, interval: float) -> None:
        while not self._stop.wait(interval):
            self._reap_once(time.time())

    def _reap_once(self, now: float) -> None:
        victims: List[Tuple[str, int]] = []
        with self._ttl_lock:
            while self._ttl_heap and self._ttl_heap[0][0] <= now:
                _, k, ts = heapq.heappop(self._ttl_heap)
                victims.append((k, ts))
        # Drop heap lock before taking key locks to keep lock order consistent.
        for k, ts in victims:
            with self._key_locks[k]:
                entries = self._data.get(k)
                if not entries:
                    continue
                idx = self._bisect_left_ts(entries, ts)
                while idx < len(entries) and entries[idx].ts == ts:
                    if entries[idx].expires_at <= now:
                        del entries[idx]
                        break
                    idx += 1
                if not entries:
                    del self._data[k]
```

### 3. Complexity

| op | time | space |
|---|---|---|
| `set` in-order | O(log n) lookup + O(1) append | +1 entry |
| `set` out-of-order | O(n) | +1 entry |
| `get` | O(log n + s) where s = expired suffix; amortized O(log n) with reaper | O(1) |
| `delete` | O(log n) + O(1) append (tombstone) | +1 entry |
| `range_query` | O(log n + r) | O(r) |

### 4. What to say out loud (staff signals)

- **Why tombstones, not destructive delete:** log-structured history makes time-travel reads consistent (`get(k, old_ts)` keeps returning the old value unless a tombstone precedes it) and makes `delete` O(log n). Mirrors Bigtable / RocksDB semantics. Destructive delete forces O(n) and breaks snapshot isolation.
- **Why TTL uses wall-clock, not the caller's `ts`:** decouples versioning from expiry. Caller `ts` may come from Lamport clocks, external event times, or replayed logs — none of which should decide when memory is freed.
- **Why two-layer expiry (lazy + reaper):** lazy check on `get` is the correctness invariant (readers never return expired data even if the reaper is behind). The reaper is a memory-reclamation optimization, not a correctness mechanism. This separation is a standard staff pattern (cache invalidation in the read path, GC in the background).
- **Lock granularity:** per-key `RLock` avoids global contention; reaper takes each key lock briefly. Next step is **sharded locks** (`hash(key) % N`) to bound the number of mutexes under millions of keys. Beyond that: copy-on-write per-key lists for lock-free readers — readers snapshot the list reference, writers publish new lists atomically. Mention this, don't implement it.
- **Compaction strategy for tombstones and shadowed versions:** background compactor walks keys, for each key finds the latest tombstone ts and drops every entry with ts ≤ that tombstone (after a grace period for in-flight reads). Also caps per-key history at a configurable `max_versions`. The reaper handles TTL; the compactor handles tombstones and version bloat. They're different jobs.
- **Memory model under pressure:** per-key growth is unbounded by default. Production needs (a) max versions per key, (b) max age, (c) max total bytes with LRU eviction across keys.
- **Assumption on non-decreasing ts:** stated up front, handled gracefully (insort) rather than assumed away.

### 5. Test strategy

- **Invariant tests:** after any sequence of ops, per-key `entries` is sorted by ts; `get(k, t)` return is either `None`, a tombstone-shadowed None, or a value from some entry with `ts' ≤ t`.
- **Property-based** (Hypothesis): random op sequences, cross-checked against a reference O(n) impl.
- **TTL edge cases:** TTL = 0 (immediate expiry), TTL exactly at read time, clock going backwards (monotonic clock discussion), reaper lag.
- **Concurrency:** N writer threads + M reader threads, assert no torn reads and no indices diverging. Stress the reaper lock ordering.
- **Benchmarks:** p50/p99 for `get` vs entries-per-key; reaper throughput; memory ceiling under adversarial TTL patterns (many short-TTL writes to one key).

### 6. Rejected alternatives

- **Single global sorted list keyed by `(key, ts)`** — O(n) scan per op when many keys, terrible cache behavior, no per-key concurrency. Rejected.
- **Skip-list / red-black tree per key** — better for heavy out-of-order writes, but the stated assumption is non-decreasing ts, so `list + append + bisect` beats them on constants and code simplicity. Mention, don't use.
- **Store TTL-expiry in the caller's `ts` space** — couples expiry to external clocks. Your code correctly avoided this; state it explicitly instead of as a comment.
- **Eager delete on TTL with no lazy check on reads** — race window where reader sees a still-present expired entry before the reaper gets to it. Must be lazy-first.
- **Single global lock** — serializes the whole store. Dead on arrival at any real QPS.

---

**The blunt takeaway for you specifically:** your instincts are staff-level (wall-clock TTL, heap reaper, per-entry metadata, compaction awareness). Your CoderPad execution isn't. Binary search templates and "is this mutation consistent across all my parallel arrays" are the kinds of things that cost you interviews even when the design discussion is right. Drill the `bisect_left`/`bisect_right` template until it's automatic, and stop maintaining parallel arrays indexed in lockstep — use a single list of record objects. That one change removes most of your correctness risk.