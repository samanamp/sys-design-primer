---
title: Rate limiter
description: Rate limiter
---
**Prompt:** Implement `allow(client_id) -> bool` that rate-limits each client to N requests per T seconds.

**What's tested:** which algorithm you reach for, and whether you understand the precision/memory tradeoff between them.

**The four canonical algorithms — know all of them:**

| Algorithm | Memory/client | Smoothness | When to use |
|---|---|---|---|
| Fixed window counter | O(1) | bursty at boundary | cheapest, tolerates 2× burst |
| Sliding window log | O(N) | exact | when fairness matters more than memory |
| Sliding window counter | O(1) | approximate | the practical default |
| Token bucket | O(1) | smooth + burstable | API gateways, network shaping |

**Right default for an in-memory interview answer: token bucket.**

```
state per client: (tokens: float, last_refill: timestamp)

allow(id):
    now = time()
    s = state[id]
    elapsed = now - s.last_refill
    s.tokens = min(capacity, s.tokens + elapsed * refill_rate)
    s.last_refill = now
    if s.tokens >= 1:
        s.tokens -= 1
        return True
    return False
```

O(1) time, O(1) memory per client. Lazy refill — don't run a background thread to top up.

**Staff signal moves:**
- Volunteer the four algorithms by name, then justify token bucket
- Bring up GC: clients vanish, you accumulate stale entries. Either lazy-evict on touch + occasional sweep, or use a TTL'd hashmap. Don't pretend the dict grows forever.
- Distributed case is the obvious P4: shared state via Redis (`INCR` with TTL for fixed window, Lua script for token bucket), or per-shard limits with `N/shards` per shard (cheaper, less fair).
- Concurrency: per-client lock, not global. Or atomic CAS on the `(tokens, last_refill)` pair.

```py
import threading
import time
from dataclasses import dataclass, field

@dataclass
class _Bucket:
    tokens: float
    last_refill: float
    lock: threading.Lock = field(default_factory=threading.Lock)

class RateLimiter:
    def __init__(self, n: int, t_seconds: float):
        self.capacity = float(n)
        self.refill_rate = n / t_seconds  # tokens/sec
        self._buckets: dict[str, _Bucket] = {}
        self._dict_lock = threading.Lock()

    def _bucket(self, cid: str) -> _Bucket:
        b = self._buckets.get(cid)          # fast path, no lock
        if b is not None:
            return b
        with self._dict_lock:                # double-checked init
            b = self._buckets.get(cid)
            if b is None:
                b = _Bucket(tokens=self.capacity, last_refill=time.monotonic())
                self._buckets[cid] = b
            return b

    def allow(self, client_id: str) -> bool:
        b = self._bucket(client_id)
        with b.lock:
            now = time.monotonic()
            b.tokens = min(self.capacity,
                           b.tokens + (now - b.last_refill) * self.refill_rate)
            b.last_refill = now
            if b.tokens >= 1.0:
                b.tokens -= 1.0
                return True
            return False
```

```py
import threading, time

class FixedWindowLimiter:
    def __init__(self, n: int, t_seconds: float):
        self.n, self.t = n, t_seconds
        self._state: dict[str, list[int]] = {}  # cid -> [window_id, count]
        self._lock = threading.Lock()

    def allow(self, cid: str) -> bool:
        window = int(time.monotonic() // self.t)
        with self._lock:
            s = self._state.get(cid)
            if s is None or s[0] != window:
                s = [window, 0]
                self._state[cid] = s
            if s[1] >= self.n:
                return False
            s[1] += 1
            return True
```

```py
from collections import deque

class SlidingWindowLogLimiter:
    def __init__(self, n: int, t_seconds: float):
        self.n, self.t = n, t_seconds
        self._logs: dict[str, deque[float]] = {}
        self._lock = threading.Lock()

    def allow(self, cid: str) -> bool:
        now = time.monotonic()
        cutoff = now - self.t
        with self._lock:
            log = self._logs.get(cid)
            if log is None:
                log = deque()
                self._logs[cid] = log
            while log and log[0] <= cutoff:
                log.popleft()
            if len(log) >= self.n:
                return False
            log.append(now)
            return True
```

```py
class SlidingWindowCounterLimiter:
    def __init__(self, n: int, t_seconds: float):
        self.n, self.t = n, t_seconds
        # cid -> [window_id, curr_count, prev_count]
        self._state: dict[str, list[int]] = {}
        self._lock = threading.Lock()

    def allow(self, cid: str) -> bool:
        now = time.monotonic()
        window = int(now // self.t)
        p = (now % self.t) / self.t  # position within current window, [0, 1)

        with self._lock:
            s = self._state.get(cid)
            if s is None or window - s[0] >= 2:
                s = [window, 0, 0]                # cold or stale: reset both
                self._state[cid] = s
            elif s[0] == window - 1:
                s[0], s[1], s[2] = window, 0, s[1]  # roll: curr -> prev
            # else: s[0] == window, no roll

            estimate = s[2] * (1.0 - p) + s[1]
            if estimate >= self.n:
                return False
            s[1] += 1
            return True
```