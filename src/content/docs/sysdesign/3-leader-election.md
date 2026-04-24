---
title: Leader election
description: Leader election
---

Fair. Let me rewrite this so you actually understand it. I'll build up from the problem, show the math explicitly, and keep the staff signals — but explain *why* each one matters instead of just tagging them.

---

# Leader Election — Explained Simply

## The core problem in one paragraph

You have a control plane (the "brain" that tells worker machines what to do). If two brains are active at once, they issue conflicting orders and corrupt the system. So we want one brain. But here's the catch: **in a distributed system, you literally cannot guarantee "exactly one brain" at any moment in time.** Messages are delayed, clocks drift, processes freeze. The best you can do is guarantee that **only one brain's orders actually get accepted by the workers.** That reframing is the whole interview in one sentence.

**[STAFF SIGNAL: problem reframing]** — Mid-level engineers try to prevent two leaders from existing. Staff engineers accept that two leaders *will* briefly exist and design so it doesn't matter.

---

## Part 1: Why "exactly one leader" is impossible — a concrete story

Imagine Leader A is happily running. Then:

1. Leader A's computer freezes for 45 seconds (garbage collection pause — real thing, happens in Java/Go).
2. The rest of the system notices A stopped responding, elects Leader B.
3. B starts issuing orders.
4. A wakes up. From A's perspective, **no time has passed**. A still thinks it's the leader. A issues orders.
5. Now two leaders are issuing conflicting orders. Workers are confused. State is corrupted.

You cannot prevent step 4. You can't tell A's CPU to know it was frozen. So instead, you make step 5 impossible: **you make sure A's orders get rejected by the workers.** That's what fencing does.

---

## Part 2: Fencing tokens — the actual solution (with the formula)

A fencing token is just **a counter that goes up every time there's a new leader.**

```
Leader A gets elected → token = 42
Leader A freezes
Leader B gets elected → token = 43
Leader B sends orders with token=43
Worker records: "highest token I've seen = 43"
Leader A wakes up, sends orders with token=42
Worker checks: is 42 >= 43? NO. Reject.
```

**The rule every worker enforces:**

```
accept(order) if and only if order.token >= worker.highest_seen_token
```

That's it. That one inequality is what makes the system safe. Everything else is about making failover *fast*; this is about making it *correct*.

**[STAFF SIGNAL: fencing discipline]** — If you don't have this check on every write, nothing else matters. Fencing isn't a feature; it's the foundation.

---

## Part 3: The 30-second budget — does it add up?

We have 30 seconds from "leader dies" to "new leader is serving traffic." Let's break it down:

```
Step                              Time     Running total
─────────────────────────────────────────────────────────
1. Detect the leader is dead      6s       6s
2. Elect a new leader             1s       7s
3. New leader loads state         2s       9s
4. New leader warms up            13s      22s
5. Clients find new leader        5s       27s
─────────────────────────────────────────────────────────
                                  Total:   27s  ✓ fits in 30s
```

### The formula for detection time

Detection works by heartbeats. The leader sends "I'm alive" every `H` seconds. If we miss `N` in a row, we declare it dead.

```
detection_time = N × H
```

I'm picking `H = 2 seconds` and `N = 3`, giving `detection_time = 6 seconds`.

**Why these numbers?**

- If `H` is too small (say 0.5s): the network has tiny hiccups all the time. We'd false-alarm constantly.
- If `N` is too small (say 1): one dropped packet = false failover.
- If `N × H` is too big: we blow the 30s RTO.

**The fundamental tradeoff, as a formula:**

```
false_failover_rate  ∝  1 / (N × H)       ← shorter timeout = more false alarms
real_failover_time   =   N × H            ← shorter timeout = faster recovery
```

You cannot make both good. You pick a point on the curve.

**[STAFF SIGNAL: capacity math]** — Most candidates say "we'll tune it." Staff engineers show the formula and commit to `H=2s, N=3` with a reason.

### The lease formula

The leader holds a "lease" on leadership — permission to be leader, valid for `L` seconds. It must renew the lease before it expires.

```
L (lease duration) > N × H (detection time)
```

Why? If the lease expires *before* we detect failure, there's a gap where nobody holds leadership and nobody knows. So I pick `L = 8s` (just barely bigger than 6s detection).

---

## Part 4: The architecture (simple version)

```
           ┌─────────────────────────┐
           │   etcd (the referee)    │
           │   — holds the lease     │
           │   — assigns tokens      │
           └──────┬───────────┬──────┘
                  │           │
          renews  │           │  watches
          every   │           │  for changes
          2s      │           │
              ┌───▼───┐   ┌───▼────┐
              │Leader │   │Standby │  ← ready to take over
              │token=42│   │(warm)  │
              └───┬───┘   └────────┘
                  │
                  │ sends orders with token=42
                  ▼
             ┌────────────┐
             │  Workers   │  ← check token, reject old ones
             └────────────┘
```

**Three roles:**
1. **etcd** is the referee. It's a separate system (3 or 5 machines running the Raft algorithm) whose only job is to answer "who is the leader?" and hand out tokens.
2. **Leader + Standby** are your control plane. One is active, one is waiting.
3. **Workers** do the real work and enforce the token check.

**Why etcd and not "build our own"?** Writing your own consensus algorithm is a multi-year project where the bugs only show up at 3am during a network partition. etcd has had a decade of those bugs already fixed. The rule: **don't build what you can rent, unless the rental is the bottleneck.**

**[STAFF SIGNAL: rejected alternative]**

---

## Part 5: The GC pause story, with the math

Let's trace what happens when Leader A freezes for 45 seconds.

```
Time    Leader A          etcd              Standby B        Workers
────────────────────────────────────────────────────────────────────────
t=0     healthy           lease valid       watching         seen_token=42
        token=42          until t=8
        
t=1     *FREEZES*         ...               ...              ...

t=8     still frozen      LEASE EXPIRES     sees lease gone  ...
                          
t=9                       grants lease      becomes leader   ...
                          to B, token=43    token=43         

t=10                      ...               sends order      seen_token=43
                                            with token=43    (order accepted)
                                            
t=45    WAKES UP          ...               ...              ...
        thinks it's
        still leader
        
t=45.1  tries to send                                        receives order
        order token=42                                       with token=42
                                                             42 < 43 → REJECT
                                                             
t=45.2  tries to renew    "your lease
        its lease         expired, you're
                          not leader"
        
t=45.3  self-terminates
```

**The key insight:** Leader A's internal clock is wrong (it thinks only a moment passed). But **the token check doesn't care about A's clock** — it only cares about the number 42 vs. 43. Math doesn't lie, even when clocks do.

**[STAFF SIGNAL: failure mode precision]**

---

## Part 6: What happens to workers during failover?

For 27 seconds, there's no leader to issue new orders. What do workers do?

**Wrong answer:** Pause and wait. (Now your entire system is down for 27s every time a leader hiccups.)

**Right answer:** Keep executing the last orders they received. Don't pause. Just don't accept *new* orders until the new leader shows up with a valid token.

```
Worker's state machine:
  - Receive new order → check token → accept or reject
  - No new orders → keep doing current work
  - Leader gap → invisible to me, I just have nothing new to do
```

**Why this matters:** It decouples the "brain failing" from the "body failing." Your data plane (the actual work) keeps running even when the control plane (decisions) is momentarily down.

**[STAFF SIGNAL: blast radius reasoning]** — Mid-level design: leader failure = system failure. Staff design: leader failure = brief pause in new decisions, nothing else.

---

## Part 7: The one question that separates staff from senior staff

**Do we actually need a leader at all?**

Most "we need exactly one leader" requirements are really "we need exactly one decider for a few specific operations." Those are very different problems.

Example: a control plane might handle 100 different operations, but only 3 of them actually need global serialization (like "reconfigure the cluster"). The other 97 can use **optimistic concurrency** — each operation carries a version number, and conflicting operations fail with "try again."

```
Old approach: 1 leader decides all 100 operations → whole system stops during failover
New approach: leader only for 3 operations → 97 keep working during failover
```

This is the question to raise in the real interview. Even if you stick with the leader design, showing you considered it proves you're not just pattern-matching on "distributed system = need Raft."

**[STAFF SIGNAL: saying no, challenging premises]**

---

## Part 8: What would force me to redesign?

- **RTO drops from 30s to 5s:** this design can't hit 5s. Detection alone is 6s. I'd have to shard the leadership across 10 pieces so each one fails over in parallel.
- **Ops rate goes from 100/sec to 10,000/sec:** the single leader becomes the bottleneck. Same fix: shard.
- **Multi-region:** network latency across regions adds ~50ms to every etcd operation. The math changes. I'd need regional leaders.

**[STAFF SIGNAL: knowing when your design breaks]** — A design without stated breaking points is a design you don't understand.

---

## The one-paragraph summary to memorize

*"You can't guarantee one leader, but you can guarantee that only one leader's writes are accepted — using a monotonic counter called a fencing token that workers check on every write. Election gives you liveness (someone is leader); fencing gives you safety (only one leader's orders count). With a 2-second heartbeat and a 3-miss threshold, detection takes 6 seconds; with an 8-second lease and a warm standby, total RTO is about 27 seconds, fitting in the 30-second budget. Workers keep executing old orders during the gap, so the data plane doesn't stop when the control plane fails over. And the real question is whether you need a global leader at all — most of the time, per-shard leadership or optimistic concurrency solves the same problem with smaller blast radius."*

If you can say that paragraph and then explain each sentence with the formulas above, you're delivering a staff-level answer.