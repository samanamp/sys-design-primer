---
title: "TTFT Optimization Program: 70B Chat"
description: "TTFT Optimization Program: 70B Chat"
---

Job scheduler with DAG dependencies

**Prompt:** `add_job(id, deps=[...])`, `run()` executes all jobs respecting dependencies. Detect cycles. Bonus: parallel execution with K workers.

**What's tested:** topological sort + cycle detection + (if asked) worker pool coordination.

**Single-threaded version: Kahn's algorithm.** Maintain `in_degree[job]` and `dependents[job]`. Push all `in_degree == 0` jobs into a ready queue. Pop one, run it, decrement `in_degree` of its dependents, push any that hit zero. If at the end any job has `in_degree > 0`, you have a cycle.

Don't use DFS-coloring here unless asked specifically about cycle detection without execution. Kahn's gives you topo order *and* cycle detection in one pass and parallelizes naturally.

**Parallel version with K workers.** Replace the ready queue with a thread-safe queue. Workers pull jobs, run them, then under a lock decrement dependent counters and push newly-ready jobs. Main thread waits until all jobs are accounted for.

Subtle: "all done" is not "queue is empty." Queue can be empty while workers are still running and about to enqueue more. Track `jobs_remaining` (atomic counter), terminate when it hits zero.

**Staff signal moves:**
- Volunteer Kahn's vs DFS topo sort tradeoff (Kahn's parallelizes; DFS gives reverse-postorder cheaply).
- For the parallel version, the right primitive is `concurrent.futures.ThreadPoolExecutor` + a counter, not raw threads. Mention you'd use it but can write the loop by hand if they want.
- Mention dynamic dependency addition (jobs that spawn child jobs at runtime) as a known harder variant — this is what real workflow engines do (Airflow's dynamic tasks, Dagster, Temporal).
- Cycle detection at `add_job` time vs `run()` time is a design choice; volunteer it.

---

```py
from collections import deque, defaultdict
import queue
import threading
from concurrent.futures import ThreadPoolExecutor
import time

class SchedulerConcurrent:
    def __init__(self):
        self._deps = defaultdict(int)
        self._children = defaultdict(list)
        self._q = queue.Queue()
        self._LOCK = threading.Lock()
        
    def reset(self):
        self._deps = defaultdict(int)
        self._children = defaultdict(list)
        self._q = queue.Queue()
        
    def add(self, id, deps):
        self._deps[id] = len(deps)
        for n in deps:
            self._children[n].append(id)
            if n not in self._deps:
                self._deps[n] = 0
                
    def worker(self):
        while not self._done.is_set():

            
            try:
                p = self._q.get(timeout=1)
                with self._LOCK:
                    self._inflight.add(p)
                    print(f"increasing in flight! {p} -> {self._inflight}")
            except queue.Empty:
                print(f"Q is empty!->{self._inflight}")
                if len(self._inflight) == 0:
                    self._done.set()
                continue
            
            if p is None:
                print("returning as p is none")
                return
            
            
                
            time.sleep(1)
            # print(f"{threading.get_ident()} got {p}")
                
            with self._LOCK:
                
                print(f"{threading.get_ident()} running {p}")
                if p in self._children:
                    for child in self._children[p]:
                        self._deps[child] -= 1
                        if self._deps[child] == 0:
                            self._q.put(child)
                # print(f"{threading.get_ident()} before done {p}")
                self._q.task_done()
                # print(f"{threading.get_ident()} done {p}")
                self.remaining -= 1
                self._inflight.remove(p)
                print(f"decreasing in flight! {p} -> {self._inflight}")
                del self._deps[p]
                

    def run(self):
        self._done = threading.Event()
        self.remaining = len(self._deps)
        self._inflight = set()
        for key, val in self._deps.items():
            if val == 0:
                self._q.put(key)
        with ThreadPoolExecutor(max_workers=4) as e:
            e.submit(self.worker)
            e.submit(self.worker)
            e.submit(self.worker)
        print("now waiting on join!")
        self._q.join()
        print(self._children)
        print(self._deps)
        # print(self._q)
        
        
            
        print(self._children)
        print(self._deps)
        if len(self._deps) > 0:
            raise ValueError(f"cycle detected: {self._deps.keys()}")
        print("all done!")

sch = SchedulerConcurrent()
sch.add(1, [2, 3, 5, 8, 10, 12])
sch.add(2, [4])

sch.run()

sch.reset()
sch.add(1, [2, 3])
sch.add(2, [4, 1])
sch.run()
```