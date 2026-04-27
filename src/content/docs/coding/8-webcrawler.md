---
title: Web crawler
description: Web crawler
---
## Question

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

---
```python
import time
import random
import threading
from queue import Queue, Empty
from concurrent.futures import ThreadPoolExecutor


class TokenBucket:
    """Lazy-refill bucket. Workers wait on a Condition; no polling thread."""
    def __init__(self, rate, capacity=None):
        self.rate = rate
        self.capacity = capacity if capacity is not None else rate
        self.tokens = float(self.capacity)
        self.last = time.monotonic()
        self.cv = threading.Condition()

    def acquire(self):
        with self.cv:
            while True:
                now = time.monotonic()
                self.tokens = min(self.capacity,
                                  self.tokens + (now - self.last) * self.rate)
                self.last = now
                if self.tokens >= 1:
                    self.tokens -= 1
                    return
                self.cv.wait(timeout=(1 - self.tokens) / self.rate)


class Crawler:
    def __init__(self, fetch_fn, max_workers, rate_per_sec, max_attempts=4):
        self.fetch_fn = fetch_fn
        self.max_workers = max_workers
        self.bucket = TokenBucket(rate_per_sec)
        self.max_attempts = max_attempts

    def crawl(self, seed_url, max_pages):
        q = Queue()
        seen = {seed_url}        # dedupe at enqueue time
        visited = []             # successfully fetched, in order
        lock = threading.Lock()
        done = threading.Event()
        # pending = items in queue + items currently being processed.
        # Termination: queue empty AND pending == 0.
        pending = [1]
        q.put(seed_url)

        def worker():
            while not done.is_set():
                try:
                    url = q.get(timeout=0.1)
                except Empty:
                    with lock:
                        if pending[0] == 0:
                            return
                    continue

                try:
                    with lock:
                        if len(visited) >= max_pages:
                            done.set()
                            return

                    html, links = self._fetch_with_retry(url)
                    if html is None:
                        continue  # permanent failure; pending-- in finally

                    new_links = []
                    with lock:
                        if len(visited) >= max_pages:
                            done.set()
                            return
                        visited.append(url)
                        if len(visited) >= max_pages:
                            done.set()
                            return
                        for l in links:
                            if l not in seen:
                                seen.add(l)
                                new_links.append(l)
                        pending[0] += len(new_links)  # bump BEFORE put
                    for l in new_links:
                        q.put(l)
                finally:
                    with lock:
                        pending[0] -= 1

        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            futs = [pool.submit(worker) for _ in range(self.max_workers)]
            for f in futs:
                f.result()  # surface worker exceptions

        return visited

    def _fetch_with_retry(self, url):
        for attempt in range(self.max_attempts):
            self.bucket.acquire()  # token per attempt; sleep does NOT hold one
            try:
                return self.fetch_fn(url)
            except Exception:
                if attempt == self.max_attempts - 1:
                    return None, None
                time.sleep((2 ** attempt) * 0.1 + random.random() * 0.05)
        return None, None
```

Quick test driver with a deterministic graph and injected failures:

```python
def make_fetch_fn(graph, fail_rate=0.0, seed=42):
    rng = random.Random(seed)
    rlock = threading.Lock()
    def fetch(url):
        with rlock:
            fail = rng.random() < fail_rate
        time.sleep(0.01)  # simulate I/O
        if fail:
            raise IOError("transient")
        return "<html>", list(graph.get(url, []))
    return fetch

graph = {f"u{i}": [f"u{2*i+1}", f"u{2*i+2}"] for i in range(200)}
c = Crawler(make_fetch_fn(graph, fail_rate=0.05), max_workers=4, rate_per_sec=50)
result = c.crawl("u0", 50)
assert len(result) == 50 and len(set(result)) == 50
```

**Key invariants the design rests on:**

1. **Pending counter** is incremented *before* `q.put` (inside the lock with `seen`), decremented in `finally`. Guarantees a worker only exits when no other worker is mid-flight or about to enqueue.
2. **Token acquired per attempt, not per URL.** Backoff sleep happens *outside* both the bucket and any lock. A URL stuck in retry loop can't deadlock other workers or hog rate.
3. **`seen` (dedupe) is separate from `visited` (success count).** A failed fetch doesn't count toward `max_pages`, but its URL is still in `seen` so we don't keep retrying it from neighbors.
4. **Early-exit via `done.Event()`** lets all workers bail out fast once `max_pages` is hit, without each waiting on a queue timeout.

**Things I'd deliberately punt on in 1hr (and say so out loud):**

- URL canonicalization (scheme/host case, trailing slash, fragment stripping).
- Distinguishing transient vs permanent failures — currently retries everything up to `max_attempts`.
- Per-host politeness / robots.txt — global rate limit only.
- Cancellation of in-flight fetches when `done` is set; we just let them finish.
- Memory bound on `seen` for very large crawls.

If you want to drill this further, the failure modes worth stress-testing are: (a) `fail_rate=1.0` on the seed (must terminate cleanly with empty result), (b) `max_workers=1` (regression check that single-threaded path still works), and (c) a graph with a cycle (must terminate at `max_pages` even though links are infinite).