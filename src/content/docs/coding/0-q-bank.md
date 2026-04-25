---
title: Question Bank
description: Question Bank
---

### 1. Time-versioned key-value store
**The most-reported question, in some form.** Implement a time-based KV store using real timestamps; follow-ups cover how to write tests, mock timestamps, ensure strictly increasing timestamps, handle locks under multithreading, and compare lock implementations.

```
set(key, value)              # uses current timestamp
get(key)                     # latest value
get(key, ts)                 # value at-or-before ts
get(key, version=N)          # Nth set
```
- **P1**: Single-threaded. Pick the per-key data structure that makes `get(key, ts)` log-time. Justify.
- **P2**: Make timestamps *injectable* (clock abstraction) so tests don't need `time.sleep`. Show how you'd test out-of-order writes, ties, and reads at gaps.
- **P3**: Multi-threaded writes + reads. Compare global lock, per-key lock, optimistic (version check + retry). Pick one and defend.
- **P4**: 1B keys, 99% idle. How does memory change?

### 2. Spreadsheet with formulas
```
set_cell(addr, expr)   # number or "=A1+B2*3"
get_cell(addr) -> float
```
- **P1**: Numeric values + `=` expressions over `+ - * /` and cell refs. Evaluate lazily on `get_cell`.
- **P2**: Make get_cell O(1); set_cell must proactively update all dependent downstream cells when a value changes. Switch to eager propagation via dep graph.
- **P3**: Cycle detection (DFS + visiting set). Reject the offending `set_cell` and preserve prior state — no half-updates.
- **P4**: Concurrent `set_cell` on cells in the same connected component. Minimal locking that's still correct.

### 3. Resumable iterator over a large dataset
Implement a resumable iterator for a large dataset.

```
class ResumableIterator:
    def __init__(self, source): ...
    def next_batch(self, n) -> list
    def checkpoint(self) -> bytes
    @classmethod
    def restore(cls, source, blob): ...
```
- **P1**: Wrap a generator/source; produce items in chunks of `n`; track position.
- **P2**: `checkpoint()` returns enough state to resume after process restart. `restore()` reconstructs without re-reading consumed items.
- **P3**: Source size = 10B items. `checkpoint()` must be O(1) in size, not O(consumed).
- **P4**: Two threads call `next_batch` on the same iterator. Define semantics (serialized? interleaved?) and enforce.

### 4. In-memory SQL subset
Implement an in-memory database supporting insert and query, with where filtering and order by sorting; provide select(table_name, where=None, order_by=None); multiple where conditions only support AND logic; original requirements simulate SQL using a map, support comparison operators, and order by.

```
create_table(name, columns)
insert(name, row: dict)
select(name, where=None, order_by=None) -> list[dict]
```
`where` is a list of `(col, op, val)` joined by AND, with ops in `{=, !=, <, <=, >, >=}`. `order_by` is a list of `(col, "asc"|"desc")`.

- **P1**: Build it. Tests: empty where, multi-column where, single-col sort, multi-col sort.
- **P2**: 10M rows, point queries on indexed column. Add a hash index. Trade-offs vs sorted index for range queries.
- **P3**: API stability: extend `where` to support OR without breaking existing callers.
- **P4**: Concurrent inserts during a select — what consistency model do you offer? Snapshot? Read-committed? How do you implement it cheaply?

### 5. KV store with custom serialization to disk
Implement serialization and deserialization for a key-value store where both keys and values can contain any characters including delimiters; you can't use simple delimiters because they might appear in the data; most candidates land on length-prefix encoding (3:key5:value), the same pattern used in Redis protocol. Custom serialization/deserialization must be implemented (no Python built-in libraries like json).

```
class KVStore:
    def set(self, key: str, value: str): ...
    def get(self, key: str) -> str | None: ...
    def save(self, path: str): ...
    @classmethod
    def load(cls, path: str) -> "KVStore": ...
```
- **P1**: Length-prefix encoding. Show on paper why a delimiter approach (e.g. `key=value\n`) breaks for arbitrary-content keys/values.
- **P2**: Round-trip property test: random keys/values containing `\n`, `:`, `\0`, unicode — must survive.
- **P3**: Append-only log instead of full rewrite on every `save()`. Compaction strategy.
- **P4**: Crash mid-write. How do you detect a torn record on `load()`? (Length prefix + checksum, or length prefix + final-marker.)

### 6. Path normalization with symlinks
Implement a function to normalize filesystem paths, resolving . and .. components and handling symbolic links; handle . by ignoring it, handle .. by popping from the stack, resolve symbolic links using a provided mapping dictionary. (Also reported as: implement `cd` command navigation with cycle detection.)

```
normalize(path: str, symlinks: dict[str, str]) -> str
# normalize("/home/user/docs/../photos/./img.png", {}) -> "/home/user/photos/img.png"
# normalize("/photos/../docs", {"/photos": "/media/external/photos"}) -> ?
```
- **P1**: `.`, `..`, repeated slashes, no symlinks. Stack-based.
- **P2**: Add symlink resolution. Define when symlinks resolve (eager on each component? lazy at end?). Pick one and justify against POSIX behavior.
- **P3**: Detect symlink cycles (`/a -> /b`, `/b -> /a`) and bound resolution.
- **P4**: Relative symlinks: `/foo/link -> ../bar`. Handle correctly.

### 7. Expiring credit ledger
A complex data-structure problem involving time-based credit allocation and expiration; implement a system that supports adding credits, expiring credits, and consuming credits; consumption rule: always consume the credits that expire the soonest (FIFO/queue); core logic: when performing Cost(t, x) (consume x credits at time t), if the balance is sufficient, the system must correctly update the totals for future expiration events so that consumption is fully reflected.

```
add(t_grant, t_expire, amount)   # grant credits available [t_grant, t_expire)
cost(t, x) -> bool               # consume x at time t, FIFO by earliest expiry; False if insufficient
balance(t) -> int                # available credits at time t
```
- **P1**: Build with a min-heap keyed by expiry. `cost` walks the heap consuming earliest-expiring first, dropping already-expired entries lazily.
- **P2**: `balance(t)` for arbitrary `t` — including times in the future and past. What structure makes this cheap?
- **P3**: 1M `cost` ops/sec. Profile: where's the cost? (Heap rebalance vs bookkeeping.)
- **P4**: A query asks "total credits expiring in [t1, t2]." Add this without scanning the whole heap.

### 8. Concurrent web crawler
Build a web crawler that uses multiple threads to fetch pages concurrently. You need to coordinate between threads, avoid crawling the same URL twice, handle failures, and probably implement some kind of rate limiting.

```
class Crawler:
    def __init__(self, fetch_fn, max_workers, rate_per_sec): ...
    def crawl(self, seed_url, max_pages) -> list[str]
```
`fetch_fn(url) -> (html, links)`. Mock it in tests with a fake graph + injected failures.
- **P1**: Single-threaded BFS with visited set. Get correctness first.
- **P2**: Thread pool of N workers, shared queue + visited set. Fix the obvious races.
- **P3**: Rate limit at `rate_per_sec` *globally* (not per-thread). Token bucket or semaphore? What happens when workers idle waiting for tokens?
- **P4**: `fetch_fn` fails 5% of the time. Retry with exponential backoff, but cap total attempts. Make sure a permanent failure doesn't deadlock other workers.

### 9. Counter with a race condition (debug + fix)
Identify the race condition in this code and modify it to prevent the issue; explain race conditions conceptually, identify the specific vulnerability (read-modify-write is not atomic), fix using appropriate primitives (locks, atomics, thread-safe queues).

```python
class Counter:
    def __init__(self): self.count = 0
    def increment(self):
        current = self.count
        self.count = current + 1   # race: read-modify-write not atomic
```
- **P1**: Identify the race. Write a test that *deterministically* exposes it (not just "run with N threads and pray"). Hint: monkey-patch a sleep between read and write.
- **P2**: Fix with `Lock`. Measure throughput at 1, 4, 16 threads.
- **P3**: Replace `Lock` with atomic CAS (`itertools.count` or per-shard counters). Compare contention.
- **P4**: Now `increment` also writes to a list of recent updates. Avoid the lock holding the list-append — separate hot path from cold path.

### 10. Mini ORM
Build a simple ORM (Object-Relational Mapping) layer.

```python
class Model:
    # subclasses define class attributes typed as Field(...)
    # framework provides: save(), delete(), .objects.filter(...).order_by(...).all()
```
Backing store: in-memory dict-of-dicts (don't write to a real DB).

- **P1**: Field descriptors (str, int, datetime). Type validation on assignment. `save()` inserts or updates by primary key.
- **P2**: `Manager` class exposing `filter(**kwargs)` returning a lazy QuerySet. Chainable: `.filter(...).filter(...).order_by('field').all()`. Evaluation deferred until iteration.
- **P3**: `__repr__` / `__eq__` / hashing on PK. What breaks if a user mutates a field after `__hash__` is computed?
- **P4**: Migrations: a model adds a field. Existing in-memory rows lack it. Define and implement the upgrade path.