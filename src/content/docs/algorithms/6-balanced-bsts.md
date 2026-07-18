---
title: "Balanced BSTs: AVL, Red-Black, and Splay — Explain, Don't Code"
description: "The interview-bar treatment of balanced binary search trees: the invariant each maintains, why rotations work, the spoken AVL insert story, when each wins, and a practice path."
---

# Balanced BSTs: AVL, Red-Black, Splay — Explain, Don't Code

Google's prep guide lists balanced BSTs, but the realistic bar is **explanation, not live implementation** — nobody codes red-black insertion in 45 minutes. What interviewers probe: do you know *why* these exist, *what invariant* each maintains, *why rotations* restore it cheaply, and *when you'd actually use one*. This doc is sized to that bar.

## Why they exist (the 15-second version)

A plain BST degrades to a linked list under sorted insertion — O(n) everything. Balanced BSTs add an invariant that caps height at O(log n), enforced by local O(1) restructurings (rotations) on the insert/delete path. You pay a constant factor on writes to guarantee logarithmic everything, plus the thing a hashtable can't give you: **sorted order** — range queries, predecessor/successor, ordered iteration.

## The rotation — the one mechanism everything shares

A rotation is a local pointer swap that changes heights but preserves the BST ordering invariant. It's O(1): three pointer updates.

```
    y                x
   / \    right     / \
  x   C   ---->    A   y
 / \      <----       / \
A   B     left       B   C
```

In-order traversal of both: A, x, B, y, C — identical. That's the whole trick: rotations re-shape without re-ordering. Every balanced tree is just a policy for *when* to rotate.

## AVL — strict height balance

**Invariant:** for every node, |height(left) − height(right)| ≤ 1.

**The spoken insert story (rehearse this once, out loud):** "Insert like a plain BST. Walk back up the path updating heights. At the first node where the balance factor hits ±2, look at which grandchild subtree the insert went into: outside case (left-left or right-right) needs one rotation; inside case (left-right or right-left) needs two — rotate the child first to convert it to the outside case, then rotate the unbalanced node. Heights above are then restored, so **insert fixes everything with at most two rotations**. Delete is the harder one — it can cascade rotations all the way to the root, O(log n) of them."

**Character:** strictest balance → shortest trees → fastest lookups, more rotation work on writes. Height ≤ ~1.44·log₂(n).

## Red-black — looser balance, cheaper writes

**Invariants:** every node is red or black; the root is black; a red node never has a red child; every root-to-leaf path has the same number of black nodes (black-height).

Those rules force the longest path (alternating red-black) to be at most 2× the shortest (all black) → height ≤ 2·log₂(n+1). Insert fixes violations with recoloring (cheap, may cascade up) plus **at most two rotations**; delete needs at most three. That write-cheapness is why red-black is the production default: C++ `std::map`/`std::set`, Java `TreeMap`, the classic Linux CFS scheduler runqueue.

Useful mental model: a red-black tree is a 2-3-4 B-tree in disguise — a black node with its red children is one "fat" B-tree node. If B-trees make sense to you, red-black rules stop being arbitrary.

## Splay — no invariant, amortized balance

No stored balance data at all. Every access **splays** the touched node to the root through double rotations (zig-zig, zig-zag). Worst-case single operation is O(n), but any sequence of m operations is O(m log n) amortized, and recently/frequently accessed keys sit near the top — a self-optimizing cache-like structure. Rarely the production answer (bad worst case per op, writes on reads kill concurrency), but a great interview answer for "design a structure that exploits access locality."

## The comparison you'll actually be asked

| | AVL | Red-black | Splay | Hashtable |
|---|---|---|---|---|
| Lookup | O(log n), shortest tree | O(log n) | amortized O(log n), hot keys ~O(1) | O(1) expected |
| Insert/delete | O(log n), more rotations | O(log n), ≤2–3 rotations | amortized O(log n) | O(1) amortized |
| Ordered iteration / range / min-max | ✔ | ✔ | ✔ | ✘ (the reason trees exist) |
| Use when | read-heavy, latency-sensitive lookups | mixed workload — the default | skewed access pattern | no ordering needed |

**The production-honest closer:** "In an interview or production I'd say *sorted container* — `std::map`, Java `TreeMap` — which is a red-black tree underneath; I'd only hand-roll if I needed order-statistics augmentation (k-th element, rank), and then I'd augment each node with subtree size."

That augmentation point is the staff-level flourish: interval trees (max endpoint per subtree) and order-statistic trees (subtree sizes) are the same red-black machinery with one extra field maintained through rotations.

## Where they show up in system design

- Ordered in-memory index: memtables in LSM engines (see [KV Store](/coding/1-kvstore/)) — though production memtables often prefer skiplists for lock-free concurrency; saying *why* (CAS-friendly, no rotations to lock) is a great aside.
- Anything "give me the smallest/next": schedulers, timer wheels' fallback, rate limiters with ordered eviction.
- On disk, the answer changes: B-trees/B+trees win because node fan-out matches page size — if the interviewer shifts to disk, shift structures.

## Practice path

1. **Say the AVL insert story out loud** (above) twice, once with a drawn example: insert 1,2,3 → right-right case → single left rotation. Ten minutes, done.
2. **Explain red-black to someone** (or to a wall) via the 2-3-4 correspondence. If you can answer "why at most 2 rotations on insert?" (recoloring handles the cascade; rotation terminates it) you're at the bar.
3. **Code the neighbors, not the beast** — these LeetCode problems give the hands-on feel without full RB machinery:
   - [108. Sorted Array → height-balanced BST](https://leetcode.com/problems/convert-sorted-array-to-binary-search-tree/) — build balanced directly
   - [1382. Balance a BST](https://leetcode.com/problems/balance-a-binary-search-tree/) — global rebalance (flatten + rebuild)
   - [98. Validate BST](https://leetcode.com/problems/validate-binary-search-tree/) + [110. Balanced Binary Tree](https://leetcode.com/problems/balanced-binary-tree/) — check both invariants
   - [230. Kth Smallest in BST](https://leetcode.com/problems/kth-smallest-element-in-a-bst/) — then answer the follow-up "what if it's queried often?" with the subtree-size augmentation
   - [729. My Calendar I](https://leetcode.com/problems/my-calendar-i/) — the "I need a sorted container" recognition rep
4. **Optional, only if curious:** implement AVL insert (not delete) once in ~40 lines. Skip splay/red-black implementation entirely — negative ROI for interview prep.

## Interview Q&A

**Q: Why not just use a hashtable?**
Ordering. Range queries, floor/ceiling, ordered iteration, min/max — O(n) on a hashtable, O(log n) or better on a tree. If none of those are needed, the hashtable wins.

**Q: Red-black vs AVL — when each?**
AVL is more strictly balanced → faster lookups, costlier updates → read-heavy. Red-black bounds rotation count per write → better mixed workloads, and it's what standard libraries ship.

**Q: What does a rotation cost and what does it preserve?**
O(1) — three pointer updates. Preserves in-order sequence; changes subtree heights. Balanced trees are policies for when to rotate.

**Q: Why do databases use B-trees instead?**
Disk pages. A B-tree node with fan-out of hundreds turns a lookup into 3–4 page reads instead of ~20 binary-tree hops. Same balancing idea, adapted to the memory hierarchy — say this and you've connected it back to performance engineering.
