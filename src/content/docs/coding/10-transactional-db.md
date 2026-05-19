---
title: transactional kv db
description: transactional kv db
---

**Prompt:** `set(k, v)`, `get(k)`, `delete(k)`, `begin()`, `commit()`, `rollback()`. Transactions can nest arbitrarily.

**What's tested:** transaction semantics. This is the OpenAI canonical from the question bank — drill it.

**The right structure: a stack of overlay dicts.**

```
class KV:
    def __init__(self):
        self.layers = [{}]   # bottom = committed state

    def begin(self):
        self.layers.append({})

    def get(self, k):
        for layer in reversed(self.layers):
            if k in layer:
                v = layer[k]
                return None if v is _TOMBSTONE else v
        return None

    def set(self, k, v):
        self.layers[-1][k] = v

    def delete(self, k):
        self.layers[-1][k] = _TOMBSTONE   # critical

    def commit(self):
        if len(self.layers) == 1: raise
        top = self.layers.pop()
        self.layers[-1].update(top)

    def rollback(self):
        if len(self.layers) == 1: raise
        self.layers.pop()
```

The tombstone is non-obvious. Without it, a `delete` inside a transaction doesn't suppress a value from a lower layer on `get`. Candidates ship the bug constantly.

**Staff signal moves:**
- Walk through the tombstone reasoning *before* writing code. Show you anticipated it.
- Complexity: `get` is O(depth) — fine for nesting depth 1–10, ugly at 1000. Optimization is a single dict + a per-key undo log; mention it as the next step.
- Commit is "merge into parent," not "merge into root." Nested commits flatten one layer at a time. This is the second canonical bug.
- Snapshot isolation is the P4: multiple concurrent transactions seeing a stable view. Now you need MVCC — versioned values, transactions tagged with a read timestamp. Different problem entirely; signal that you know it's different.

## Explanation
Let me walk through each piece with concrete traces. The code is correct but the "why" only clicks once you simulate it.

**The tombstone**

Without it, `delete` inside a transaction is silently broken. Trace:

```
kv = KV()
kv.set("x", 1)        # layers = [{"x": 1}]
kv.begin()            # layers = [{"x": 1}, {}]
kv.delete("x")        # naive: top layer stays {} (nothing to delete there)
                      # with tombstone: layers = [{"x": 1}, {"x": TOMBSTONE}]
kv.get("x")           # naive: walks down, finds 1, returns 1. BUG.
                      # tombstone: hits top, sees sentinel, returns None.
```

The mental trap is thinking `delete` means "remove the entry." In a layered model it means "shadow whatever's below." `set(k, v)` shadows with a value; `delete(k)` shadows with "not present." Both are writes to the top layer. That symmetry is the key insight — `delete` is just `set(k, TOMBSTONE)`.

**Commit merges into parent, not root**

```
kv.set("x", 1)        # [{"x": 1}]
kv.begin()            # [{"x": 1}, {}]
kv.set("x", 2)        
kv.begin()            # [{"x": 1}, {"x": 2}, {}]
kv.set("x", 3)        # [{"x": 1}, {"x": 2}, {"x": 3}]
kv.commit()           # [{"x": 1}, {"x": 3}]   <- merged into middle, not bottom
kv.rollback()         # [{"x": 1}]             <- outer rollback discards x=3
```

The inner commit is *provisional*. It promotes the inner's writes one level up, where they're still inside the outer transaction. If the outer rolls back, everything the inner "committed" is discarded — because semantically, the inner committed *to its parent*, not to durable state. Only when the outermost transaction commits do the writes actually land in `layers[0]`. The bug candidates ship is `self.layers[0].update(top)`, which makes inner commits unconditionally durable and breaks the rollback above.

**O(depth) get and the undo-log alternative**

`get` walks layers top-down because that's where shadowing lives. Fine for human-scale nesting. At depth 1000 you're doing 1000 dict lookups per get.

The optimization: keep a single authoritative dict `data = {}` and a stack of undo logs. Each `begin` pushes a new empty log. Each `set`/`delete` records the prior value (or a "key didn't exist" marker) into the top log, then mutates `data` directly. `rollback` pops the top log and replays the recorded restores in reverse. `commit` pops the top log and *prepends* its entries onto the parent's log — so the parent can still undo them on its own rollback. `get` is now O(1). The tombstone problem disappears because `data` itself is always the ground truth.

You mention this as a follow-up, not the initial solution, because the layered version makes the semantics obvious. Reaching for the optimized version first signals you're solving for benchmarks instead of correctness.

**Why snapshot isolation is a different problem**

The layered/undo design handles *nested* transactions: one client, scoped sub-transactions, strictly sequential. Snapshot isolation handles *concurrent* transactions: many clients, each needing a stable view of the database as of when their transaction started, regardless of what others commit in the meantime.

Concretely: T1 begins, reads x=1. T2 begins, sets x=2, commits. T1 reads x again — and must still see 1, because T1's snapshot is frozen at its begin timestamp.

You can't get that from overlay dicts. You need versioned values: each key holds a list of `(commit_ts, value)` tuples. Each transaction has a `read_ts` (timestamp at `begin`). A read for key k returns the value with the largest `commit_ts ≤ read_ts`. Writes go to a per-transaction write buffer; at commit, you assign a fresh `commit_ts` and append `(commit_ts, value)` to each touched key's version chain. Conflict detection (write-write or write-read depending on isolation level) happens at commit time. Garbage collection trims versions older than the oldest active read_ts.

The signal to give in an interview: the moment they ask about multiple concurrent transactions seeing stable views, say "this is a different data structure — MVCC with per-key version chains and timestamp-tagged transactions, not nested overlays." That distinction is the whole point of the P4 — they want to know you don't try to bolt concurrency onto the overlay design.