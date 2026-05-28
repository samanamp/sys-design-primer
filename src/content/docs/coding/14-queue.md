---
title: "concurrent producer-consumer queue"
description: "concurrent producer-consumer queue"
---

**Prompt:** Implement a thread-safe bounded queue with `put(item)` (blocks if full) and `get()` (blocks if empty). Multiple producers, multiple consumers.

**What's tested:** can you reason about condition variables correctly, and do you know the canonical pitfalls.

**The right answer — two condition variables, one lock:**

```
class BoundedQueue:
    def __init__(self, capacity):
        self.q = collections.deque()
        self.cap = capacity
        self.lock = threading.Lock()
        self.not_full = threading.Condition(self.lock)
        self.not_empty = threading.Condition(self.lock)

    def put(self, item):
        with self.not_full:
            while len(self.q) == self.cap:   # while, not if
                self.not_full.wait()
            self.q.append(item)
            self.not_empty.notify()           # not notify_all

    def get(self):
        with self.not_empty:
            while not self.q:
                self.not_empty.wait()
            item = self.q.popleft()
            self.not_full.notify()
            return item
```

**Canonical bugs to call out by name:**
1. **`if` instead of `while` on the wait condition.** Spurious wakeups exist. After waking, re-check. This is the textbook bug.
2. **One condition variable for both directions.** Producer blocked on full, consumer blocked on empty — if you share a CV and use `notify`, you might wake the wrong side. Two CVs avoid this. `notify_all` masks the bug but wastes wakeups.
3. **Lock not held during the predicate check.** `len(self.q) == self.cap` must be evaluated under the lock; otherwise it's a data race.

**Staff signal moves:**
- Volunteer "while, not if" before they ask.
- Mention `queue.Queue` exists in the stdlib and does this correctly; you're reimplementing it because they asked.
- For high contention, mention lock-free SPSC ring buffers (single-producer single-consumer with atomic head/tail indices). LMAX Disruptor for MPMC. Don't implement; just name them.
- Timeout / cancellation: real production queues need `put(item, timeout)`. The pattern: `cv.wait(timeout)` returns False on timeout, return False from `put`.