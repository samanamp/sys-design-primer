---
title: "Heapsort: Free Once You Own a Heap"
description: "The array-is-a-tree insight, an honest O(n) heapify argument, a fully traced sort phase, and why the heap structure — not the sort — is the interview star."
---

# Heapsort: Free Once You Own a Heap

Heapsort is two ideas glued together: (1) an array can *be* a complete binary tree with zero pointers, and (2) if the tree is a max-heap, the maximum is sitting at index 0, so repeatedly plucking it off sorts the array in place. Once you can build and repair a heap, the sort is almost an afterthought — which is exactly the right framing, because in interviews the **heap** shows up constantly (top-k, streaming median, k-way merge) while heapsort itself is mostly a "compare and contrast" question.

## The whole insight: the array IS the tree

Take `[4, 10, 3, 5, 1, 8]`. No nodes, no pointers — just arithmetic. For the element at index `i`: children live at `2i+1` and `2i+2`, parent at `(i-1)//2`.

```
index:   0    1    2    3    4    5              tree view (same data):
value: [ 4 , 10 ,  3 ,  5 ,  1 ,  8 ]
                                                        4          ← idx 0
children of 0 → idx 1,2  (10, 3)                      /   \
children of 1 → idx 3,4  (5, 1)                     10      3      ← idx 1,2
children of 2 → idx 5    (8)                       /  \    /
parent of 5   → (5-1)//2 = 2  (3)                 5    1  8        ← idx 3,4,5
```

Because a heap is a *complete* tree (every level full, last level filled left to right), the mapping has no holes. That's the entire data structure: an array plus an indexing convention. The **max-heap invariant** we want: every parent ≥ its children. The picture above violates it (4 < 10, and 3 < 8) — fixing that is heapify.

## Sift-down, and building the heap in O(n)

**Sift-down** repairs one node assuming its subtrees are already valid heaps: swap the node with its larger child until it's ≥ both children (or hits the bottom). **Build-max-heap** runs sift-down on every parent, from the *last parent* up to the root. Last parent = `n//2 - 1 = 2`.

```
[4, 10, 3, 5, 1, 8]
sift-down(i=2, val 3):  child idx 5 = 8 > 3 → swap        → [4, 10, 8, 5, 1, 3]
sift-down(i=1, val 10): children 5, 1; 10 ≥ both → no swap → [4, 10, 8, 5, 1, 3]
sift-down(i=0, val 4):  children 10, 8; larger = 10 → swap → [10, 4, 8, 5, 1, 3]
    4 now at idx 1: children 5, 1; larger = 5 → swap       → [10, 5, 8, 4, 1, 3]
    4 now at idx 3: no children → stop

max-heap:  [10, 5, 8, 4, 1, 3]        10
                                     /  \
                                    5    8
                                   / \  /
                                  4  1 3      every parent ≥ children ✓
```

**Why is this O(n), not O(n log n)?** The lazy bound says "n nodes × log n sift" — true but loose. The honest argument: sift-down cost is the node's *height*, and almost all nodes are near the bottom where height is tiny. In a heap of n nodes, about n/2 nodes have height 0 (cost 0), n/4 have height 1, n/8 have height 2, ... So total work is roughly

```
Σ  (n / 2^(h+1)) · h   for h = 0..log n   =   n · Σ h/2^(h+1)   ≤   n · 1   →  O(n)
```

The series Σ h/2^h converges to 2 (a constant), so the whole build is linear. Contrast with building by n insertions + **sift-up**: there, most nodes are near the bottom and may bubble *up* the full log n — that's genuinely O(n log n). Sift-down from the bottom is the cheap direction. (Rule of thumb: sift-up for one incoming element, sift-down for a broken root or bulk build.)

## The sort phase: swap, shrink, sift

Max is at index 0. Swap it to the end, pretend the array is one shorter, sift-down the new root. Repeat. Three iterations traced from `[10, 5, 8, 4, 1, 3]`:

```
iter 1: swap a[0]↔a[5]        → [3, 5, 8, 4, 1 | 10]      heap size 5
        sift-down 3: children 5, 8 → swap 8 → [8, 5, 3, 4, 1 | 10]
                     3 at idx 2: no children in heap → stop

iter 2: swap a[0]↔a[4]        → [1, 5, 3, 4 | 8, 10]      heap size 4
        sift-down 1: children 5, 3 → swap 5 → [5, 1, 3, 4 | 8, 10]
                     1 at idx 1: child 4    → swap 4 → [5, 4, 3, 1 | 8, 10]

iter 3: swap a[0]↔a[3]        → [1, 4, 3 | 5, 8, 10]      heap size 3
        sift-down 1: children 4, 3 → swap 4 → [4, 1, 3 | 5, 8, 10]
```

The sorted suffix (right of `|`) grows one element per iteration, largest-first, and it's exactly the final positions — no merge, no second array. Two more iterations yield `[1, 3, 4, 5, 8, 10]`.

## Guaranteed n log n, in place — so why isn't it the default?

Heapsort is the only mainstream sort with **all three** of: worst-case O(n log n), O(1) extra space, no pathological input. Quicksort can go quadratic; mergesort wants O(n) scratch. Yet quicksort beats it in practice, usually by 2–3×, because heapsort is **cache-hostile**: sift-down jumps from index `i` to `2i+1` — child indices double every level, so consecutive accesses land pages apart, missing cache on nearly every hop, while quicksort's partition streams sequentially through memory. Heapsort also does more comparisons per element on average and every element travels root-to-leaf twice-ish.

The production reconciliation is **introsort** (C++ `std::sort`): run quicksort, track recursion depth, and if it exceeds ~2·log n (someone fed you a killer input), switch that subproblem to heapsort. Quicksort's average speed, heapsort's worst-case guarantee — heapsort as insurance policy, which is its honest modern role.

## The heap is the star; heapsort is the demo

Interviews rarely ask you to heapsort. They constantly ask you to *use a heap* — often via `heapq`, keeping k elements instead of sorting all n:

- **Top-k / k-th largest**: min-heap of size k over n elements → O(n log k), beats full sort's O(n log n). [215. Kth Largest](https://leetcode.com/problems/kth-largest-element-in-an-array/), [347. Top K Frequent](https://leetcode.com/problems/top-k-frequent-elements/), [703. Kth Largest in a Stream](https://leetcode.com/problems/kth-largest-element-in-a-stream/) — 703 is literally "maintain a size-k min-heap forever."
- **Streaming median**: max-heap of the lower half + min-heap of the upper half, rebalance to keep sizes within 1 — [295. Find Median from Data Stream](https://leetcode.com/problems/find-median-from-data-stream/).
- **K-way merge**: heap of k head elements, pop-min then push that list's next — the merge step of external sort and LSM compaction. [23. Merge k Sorted Lists](https://leetcode.com/problems/merge-k-sorted-lists/).

Say "heap" the moment you hear *k-th*, *top-k*, *median so far*, or *merge k streams* — that recognition is worth more than the sort.

## Implementation

```python
def heapsort(a: list) -> None:
    n = len(a)

    def sift_down(i: int, size: int) -> None:
        while True:
            largest, l, r = i, 2 * i + 1, 2 * i + 2
            if l < size and a[l] > a[largest]:
                largest = l
            if r < size and a[r] > a[largest]:
                largest = r
            if largest == i:
                return
            a[i], a[largest] = a[largest], a[i]
            i = largest

    for i in range(n // 2 - 1, -1, -1):   # build-max-heap: O(n)
        sift_down(i, n)

    for end in range(n - 1, 0, -1):       # sort phase: O(n log n)
        a[0], a[end] = a[end], a[0]
        sift_down(0, end)
```

## Complexity box

| | |
|---|---|
| **Time** | O(n log n) worst, average, *and* best (heapify O(n) + n sifts of O(log n)) |
| **Space** | O(1) extra — iterative sift, no recursion |
| **Stable** | No — the root↔end swap teleports elements past equals |
| **In-place** | Yes |

## How it loses

- **To quicksort** on real hardware: cache-hostile index jumps + more comparisons → typically 2–3× slower on random data despite the same big-O.
- **To mergesort** when stability matters, or on linked lists / external data — heaps need random access.
- **To O(n log k) heap-*selection*** when you only need top-k: full heapsort is overkill; keep a size-k heap instead.
- **To counting/radix** when keys are small bounded integers — see the next doc.

## Interview Q&A

**Q: Why is building the heap O(n) when each sift-down is O(log n)?**
Because sift cost equals node height, and node counts halve as height grows: ~n/2 nodes cost 0, n/4 cost 1, n/8 cost 2. The sum n·Σh/2^(h+1) is bounded by a constant times n — the geometric series converges. The O(n log n) bound assumes every node pays full log n; only the root does.

**Q: If quicksort is faster, why does anyone use heapsort?**
The guarantee. Worst-case O(n log n) with O(1) space and no adversarial input — so it's the fallback inside introsort (`std::sort`) when quicksort's recursion runs deep, and a fit for hard-real-time systems where "average case" isn't a promise you can ship.

**Q: What's the relationship to priority queues?**
A binary heap *is* the standard priority-queue implementation: push = append + sift-up, pop = swap root with last + sift-down, both O(log n), peek O(1). Heapsort is just "build a priority queue over the array, then pop everything" — done in place by parking each popped max in the vacated slot.

**Q: Is heapsort stable?**
No. Sift-down and the root-to-end swap move elements across arbitrary distances with no memory of original order — two equal keys can easily finish reversed. If stability is required, that's mergesort's (or Timsort's) territory.
