---
title: Expiring credit ledger
description: Expiring credit ledger
---
## Question
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

---
You're right on both counts. Let me reset.

## What's actually expected in 70 minutes

P1 with a heap. P2 with whatever you can build from `collections` and `bisect`. P3 as a discussion. P4 as either "I'd add a structure for this" or a simple linear scan with the caveat stated.

## What you actually have in stdlib

- `heapq` — min-heap, fine for P1.
- `bisect` — binary search over a sorted `list`. This is the SortedDict substitute, with the caveat that `list.insert` is O(n) due to the shift. For interview-scale data that's acceptable; you state the tradeoff.
- `collections.deque` — O(1) ends, useful if grants arrive monotonically by expiry.

The realistic interview answer uses `heapq` for P1 and a sorted `list` via `bisect` for P2/P4. You acknowledge the O(n) insert cost out loud and move on. No interviewer faults you for it; many will nod.

## The 70-minute solution

```python
import heapq
import bisect

class CreditLedger:
    """
    Credits granted over [t_grant, t_expire), consumed FIFO by earliest expiry.

    Two structures:
    - heap of (t_expire, unique_id, remaining): the consumption queue for cost().
    - parallel sorted list of (t_expire, t_grant, amount): for balance() and
      range-expiry queries. Insertions are O(n) due to list shift; acceptable
      for interview-scale workloads. In production I'd reach for a balanced BST
      or a Fenwick tree over compressed times.

    A cost() that partially consumes an entry rewrites the entry in BOTH structures.
    The insight worth saying out loud: partial consumption is equivalent to reducing
    the amount on that (t_grant, t_expire) interval — nothing about the times changes.
    """

    def __init__(self):
        self._heap = []          # (t_expire, id, remaining_ref)
        self._counter = 0
        self._entries = {}       # id -> [t_grant, t_expire, remaining]; mutable, shared with heap
        self._sorted = []        # sorted list of (t_expire, t_grant, id) for range queries
        self._live = 0           # running total of un-expired, un-consumed credits

    # ---------- public ----------

    def add(self, t_grant, t_expire, amount):
        if amount <= 0 or t_grant >= t_expire:
            return
        eid = self._counter
        self._counter += 1
        entry = [t_grant, t_expire, amount]
        self._entries[eid] = entry
        heapq.heappush(self._heap, (t_expire, eid))
        bisect.insort(self._sorted, (t_expire, t_grant, eid))
        self._live += amount

    def cost(self, t, x) -> bool:
        if x <= 0:
            return True
        self._evict_expired(t)
        if self._live < x:
            return False

        remaining = x
        while remaining > 0:
            t_expire, eid = self._heap[0]
            entry = self._entries.get(eid)
            # Stale heap entry (already drained or evicted): drop and continue.
            if entry is None or entry[2] == 0:
                heapq.heappop(self._heap)
                continue

            take = min(entry[2], remaining)
            entry[2] -= take
            remaining -= take
            self._live -= take

            if entry[2] == 0:
                heapq.heappop(self._heap)
                # Remove from sorted list.
                key = (entry[1], entry[0], eid)
                i = bisect.bisect_left(self._sorted, key)
                if i < len(self._sorted) and self._sorted[i] == key:
                    self._sorted.pop(i)
                del self._entries[eid]
        return True

    def balance(self, t) -> int:
        # Sum amounts where t_grant <= t < t_expire, over current (un-consumed) entries.
        # Linear scan over the sorted list. We can prune the right side by t_expire <= t,
        # but on the left we still scan everything with t_expire > t and check t_grant.
        # In an interview I'd state this is O(n) and that the heap-only design can't do
        # better without auxiliary structure.
        i = bisect.bisect_right(self._sorted, (t, float('inf'), float('inf')))
        # Everything before i has t_expire <= t — already expired at time t (half-open).
        total = 0
        for j in range(i, len(self._sorted)):
            _, t_grant, eid = self._sorted[j]
            entry = self._entries[eid]
            if t_grant <= t:
                total += entry[2]
        return total

    def expiring_in(self, t1, t2) -> int:
        # P4: credits with t_expire in [t1, t2]. Binary search both ends, sum the slice.
        # O(log n + k) where k is entries in range. The slice itself is what we want;
        # we just sum amounts over it.
        lo = bisect.bisect_left(self._sorted, (t1, -1, -1))
        hi = bisect.bisect_right(self._sorted, (t2, float('inf'), float('inf')))
        total = 0
        for j in range(lo, hi):
            _, _, eid = self._sorted[j]
            total += self._entries[eid][2]
        return total

    # ---------- internals ----------

    def _evict_expired(self, t):
        # Drop entries with t_expire <= t (half-open interval).
        while self._heap and self._heap[0][0] <= t:
            t_expire, eid = heapq.heappop(self._heap)
            entry = self._entries.pop(eid, None)
            if entry is None:
                continue
            self._live -= entry[2]
            key = (entry[1], entry[0], eid)
            i = bisect.bisect_left(self._sorted, key)
            if i < len(self._sorted) and self._sorted[i] == key:
                self._sorted.pop(i)
```

## What to say while you're writing this

Three things, in order:

**Minute 5, after the spec is clear:** "I'll use a heap for the FIFO-by-expiry consumption — that's the natural fit. For `balance` and the range query in P4, I need to look up by time, which a heap can't do. Without `sortedcontainers`, my options are a parallel sorted `list` with `bisect` — O(log n) lookup, O(n) insert — or a hand-rolled BST. For 70 minutes the sorted list is the right call; I'll note the insert cost." This sentence alone tells the interviewer you understand the tradeoff space.

**Minute 25, when you hit partial consumption:** "Partial consumption doesn't change `t_expire` or `t_grant` — it just reduces the remaining amount on that interval. So I can mutate the entry in place and keep both structures pointing at it. Stale heap entries from fully-drained items I clean up lazily."

**Minute 50, on P3:** Same answer as before. Interpreter overhead dominates at 1M ops/s in Python. The structural optimization, if grants happen to arrive in monotonic expiry order, is to replace the heap with a `deque` — O(1) on both ends, no log factor. State this as a *conditional* optimization, not a default.

## The honest tradeoffs in this version

- `add` is O(n) due to `bisect.insort` shifting the list. At interview scale (think 10⁴–10⁵ entries) this is fine. State it.
- `balance` is O(n) in the worst case. Without an order-statistic structure or a prefix-sum tree, you can't beat that with stdlib alone. State it.
- `cost` is amortized O((k + s) log n) where k is entries touched and s is stale heap entries skipped. The lazy cleanup is standard and uncontroversial.
- `expiring_in` is O(log n + k), which is genuinely good and is the part that justifies maintaining the sorted list at all.

## What I'd cut if running short

If you're at minute 50 and the sorted list is buggy, drop it. Keep the heap. Make `balance` walk a flat list of all live entries — O(n) but bulletproof. Tell the interviewer: "Given more time I'd add a sorted index for sub-linear `balance` and the P4 range query, but I want to ship something correct first." That's a strong signal. A half-built sorted list with subtle bugs is a bad signal.
