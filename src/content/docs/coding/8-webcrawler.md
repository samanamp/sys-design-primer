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

# Added per host politeness
```python
import time
import random
import threading
from queue import Queue, Empty
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser


class TokenBucket:
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


class HostInfo:
    """Per-host state: robots rules, crawl-delay, next-available slot."""
    def __init__(self):
        self.init_lock = threading.Lock()   # serializes robots.txt load
        self.initialized = False
        self.rules = None                   # RobotFileParser, or None = allow-all
        self.crawl_delay = 0.0
        self.slot_lock = threading.Lock()   # protects next_slot
        self.next_slot = 0.0


class Crawler:
    def __init__(self, fetch_fn, max_workers, rate_per_sec,
                 max_attempts=4, default_crawl_delay=0.0, user_agent="*"):
        self.fetch_fn = fetch_fn
        self.max_workers = max_workers
        self.bucket = TokenBucket(rate_per_sec)
        self.max_attempts = max_attempts
        self.default_crawl_delay = default_crawl_delay
        self.user_agent = user_agent
        self._hosts = {}
        self._hosts_lock = threading.Lock()

    # ---------- per-host politeness ----------

    def _host_info(self, host):
        # Get-or-create the HostInfo, then lazily load robots.txt exactly once.
        with self._hosts_lock:
            info = self._hosts.get(host)
            if info is None:
                info = HostInfo()
                self._hosts[host] = info

        if not info.initialized:
            with info.init_lock:
                if not info.initialized:
                    self._load_robots(host, info)
                    info.initialized = True
        return info

    def _load_robots(self, host, info):
        # robots.txt fetch counts against the global bucket but skips per-host delay.
        try:
            self.bucket.acquire()
            body, _ = self.fetch_fn(f"http://{host}/robots.txt")
            parser = RobotFileParser()
            parser.parse(body.splitlines() if isinstance(body, str) else [])
            info.rules = parser
            delay = parser.crawl_delay(self.user_agent)
            info.crawl_delay = float(delay) if delay else self.default_crawl_delay
        except Exception:
            # Network error on robots.txt => allow-all with default delay.
            # (Stricter policy would disallow on 5xx; we don't have status codes.)
            info.rules = None
            info.crawl_delay = self.default_crawl_delay

    def _allowed(self, info, url):
        return info.rules is None or info.rules.can_fetch(self.user_agent, url)

    def _reserve_slot(self, info):
        # Reserve a per-host slot atomically, then sleep until it OUTSIDE the lock.
        # Concurrent workers hitting the same host queue up on next_slot.
        with info.slot_lock:
            now = time.monotonic()
            slot = max(now, info.next_slot)
            info.next_slot = slot + info.crawl_delay
        wait = slot - time.monotonic()
        if wait > 0:
            time.sleep(wait)

    # ---------- fetch ----------

    def _fetch_with_retry(self, url):
        host = urlparse(url).netloc
        info = self._host_info(host)
        if not self._allowed(info, url):
            return None, None  # disallowed by robots.txt

        for attempt in range(self.max_attempts):
            self._reserve_slot(info)   # per-host crawl-delay
            self.bucket.acquire()      # global rate limit
            try:
                return self.fetch_fn(url)
            except Exception:
                if attempt == self.max_attempts - 1:
                    return None, None
                time.sleep((2 ** attempt) * 0.1 + random.random() * 0.05)
        return None, None

    # ---------- crawl loop (unchanged from prior version) ----------

    def crawl(self, seed_url, max_pages):
        q = Queue()
        seen = {seed_url}
        visited = []
        lock = threading.Lock()
        done = threading.Event()
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
                            done.set(); return
                    html, links = self._fetch_with_retry(url)
                    if html is None:
                        continue
                    new_links = []
                    with lock:
                        if len(visited) >= max_pages:
                            done.set(); return
                        visited.append(url)
                        if len(visited) >= max_pages:
                            done.set(); return
                        for l in links:
                            if l not in seen:
                                seen.add(l)
                                new_links.append(l)
                        pending[0] += len(new_links)
                    for l in new_links:
                        q.put(l)
                finally:
                    with lock:
                        pending[0] -= 1

        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            futs = [pool.submit(worker) for _ in range(self.max_workers)]
            for f in futs:
                f.result()
        return visited
```

Test driver with two hosts, one with a Disallow and a Crawl-delay:

```python
def make_fetch_fn(graph, robots, fail_rate=0.0, seed=42):
    rng = random.Random(seed)
    rlock = threading.Lock()
    def fetch(url):
        if url.endswith("/robots.txt"):
            host = urlparse(url).netloc
            return robots.get(host, ""), []
        with rlock:
            fail = rng.random() < fail_rate
        time.sleep(0.005)
        if fail:
            raise IOError("transient")
        return "<html>", list(graph.get(url, []))
    return fetch

graph = {
    "http://a.com/":      ["http://a.com/1", "http://a.com/2", "http://b.com/"],
    "http://a.com/1":     ["http://a.com/3", "http://b.com/secret"],
    "http://a.com/2":     ["http://a.com/4"],
    "http://a.com/3":     [], "http://a.com/4": [],
    "http://b.com/":      ["http://b.com/x", "http://b.com/secret"],
    "http://b.com/x":     [],
    "http://b.com/secret":["http://b.com/leaked"],
}
robots = {
    "a.com": "User-agent: *\nCrawl-delay: 0\n",
    "b.com": "User-agent: *\nDisallow: /secret\nCrawl-delay: 0.05\n",
}
c = Crawler(make_fetch_fn(graph, robots), max_workers=4, rate_per_sec=50)
result = c.crawl("http://a.com/", 20)
assert "http://b.com/secret" not in result        # disallowed
assert "http://b.com/leaked" not in result        # unreachable via blocked link
```

**Design points worth saying out loud:**

1. **Two locks per host, on purpose.** `init_lock` serializes the one-time robots.txt fetch (double-checked `initialized` flag avoids re-locking on the hot path). `slot_lock` is held only long enough to bump `next_slot` — workers sleep *outside* the lock so the next request can already reserve its slot.

2. **`next_slot` reservation pattern beats "lock + sleep".** N workers arriving simultaneously at a host with crawl_delay=1s get assigned slots t, t+1, t+2, … atomically; each then sleeps until its own slot. No thundering herd, no lock held during sleep.

3. **Order of gates per attempt: robots check → host slot → global bucket → fetch.** Robots check is free and rejects early. Host slot reserves a future time without blocking. Global bucket is acquired last so we don't burn a token on a request that's about to sleep 10s on per-host delay.

4. **Robots fetch goes through the global bucket but skips per-host delay.** It's a one-shot per host; making it wait on its own crawl-delay would be circular.

5. **Failure policy on robots.txt: allow-all.** Real-world: 4xx → allow, 5xx → disallow. We don't have status codes via `fetch_fn`, so the simple policy is allow-all on exception. Worth flagging in the interview.

**Still punted:**

- URL canonicalization (host case, default ports, trailing slash, fragment).
- Transient vs permanent error distinction (we retry everything).
- Robots.txt staleness — cached for the lifetime of the crawler.
- Sitemap discovery from robots.txt.
- Memory bound on `seen` and `_hosts` for very large crawls.
- Cancellation of in-flight fetches when `done` fires.

The likely interview probe on the politeness layer: *what happens if one host has crawl_delay=60s and the queue is dominated by URLs from that host?* Answer: workers serialize on `next_slot` and most of them sleep idle while other-host work starves the queue. Real fix is per-host queues with workers picking the host with the earliest available slot — explicitly out of scope for 1hr but good to name.