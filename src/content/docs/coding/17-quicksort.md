---
title: "Quicksort: The Fast One With a Temper"
description: "In-place, cache-friendly, fastest in practice — until a bad pivot makes it O(n²). Lomuto partition traced step by step, good vs bad recursion trees, 3-way partitioning for duplicates, introsort, and quickselect."
---

# Quicksort: The Fast One With a Temper

Quicksort is the practical champion: in-place, cache-friendly, and typically the fastest comparison sort on real hardware. Its temper: pick pivots badly and O(n log n) collapses to **O(n²)** — and "badly" includes the extremely common case of already-sorted input, if you're naive about it. Interviews probe exactly this edge.

One breath: pick a pivot, **partition** the array so everything smaller is left of it and everything larger is right (the pivot lands in its final position), recurse on both sides. All the work is in partition; there's no merge step.

## Lomuto partition — [7, 2, 9, 4, 3, 8, 6], pivot = 6

Pivot is the last element. `i` marks the end of the "< pivot" region (starts at -1); `j` scans. When `a[j] < pivot`, grow the region: `i += 1`, swap `a[i], a[j]`.

```
start:      [7, 2, 9, 4, 3, 8, |6]      i=-1, pivot=6

j=0: 7<6? no                            [7, 2, 9, 4, 3, 8, 6]   i=-1
j=1: 2<6? yes → i=0, swap a[0],a[1]     [2, 7, 9, 4, 3, 8, 6]   i=0
j=2: 9<6? no                            [2, 7, 9, 4, 3, 8, 6]   i=0
j=3: 4<6? yes → i=1, swap a[1],a[3]     [2, 4, 9, 7, 3, 8, 6]   i=1
j=4: 3<6? yes → i=2, swap a[2],a[4]     [2, 4, 3, 7, 9, 8, 6]   i=2
j=5: 8<6? no                            [2, 4, 3, 7, 9, 8, 6]   i=2

final: swap a[i+1], pivot → a[3]↔a[6]   [2, 4, 3, |6|, 9, 8, 7]
                                         <──less──  ^p  ──more──>
```

6 is now in its final sorted position (index 3), and we recurse on `[2,4,3]` and `[9,8,7]`. Note the long-distance swaps (9 jumped from index 2 to 4): that's what destroys stability — more below.

## Pivot luck — the two recursion trees

Left: pivots keep splitting near the middle → **log n depth**. Right: sorted input `[1,2,3,4,5,6,7]` with last-element pivot — every partition strips off one element → **n depth**, and each level still scans its whole range: n + (n−1) + … = O(n²).

```
GOOD (balanced pivots)            BAD (sorted input, last-elem pivot)
[7,2,9,4,3,8,6] p=6               [1,2,3,4,5,6,7] p=7
   /        \                     [1,2,3,4,5,6] p=6 │ (nothing right)
[2,4,3]   [9,8,7]                 [1,2,3,4,5] p=5   │
 p=3       p=7                    [1,2,3,4] p=4     │
 /  \       /  \                  [1,2,3] p=3       │
[2] [4]  [] [9,8]                 [1,2] p=2         │
                                  [1]               ▼
depth ≈ log n → O(n log n)        depth = n → O(n²)
```

**The fix: randomize the pivot** (or median-of-three). A random pivot makes every input behave like the average case — expected O(n log n) — and, just as important, means no *adversarial* input can be pre-constructed to trigger n². Fixed pivot rules are DoS-able: an attacker who knows your pivot scheme can craft killer inputs (this has been done against real libraries).

## Many duplicates — 3-way (Dutch national flag) partition

Plain Lomuto on `[4, 1, 4, 4, 2, 4, 4]` puts one `4` in place and recurses on a region still full of `4`s — arrays of mostly-equal keys degrade toward O(n²). Three-way partitioning maintains `< | == | >` regions and never recurses into the equal block:

```
[4, 1, 4, 4, 2, 4, 4]  pivot = 4    (lt | eq | gt regions)

scan 4 → eq            [ |4| ]
scan 1 → lt            [1 |4| ]
scan 4,4 → eq          [1 |4,4,4| ]
scan 2 → lt            [1,2 |4,4,4| ]
scan 4,4 → eq          [1,2 |4,4,4,4,4| ]

result: [1, 2 | 4, 4, 4, 4, 4]
recurse on [1,2] and [] — the five 4s are DONE, never touched again
```

With few distinct keys this drops to O(n · #distinct-keys) — effectively linear. Standard follow-up; the classic drill is [LeetCode 75 (Sort Colors)](https://leetcode.com/problems/sort-colors/).

## Why it wins in practice — and why it isn't stable

Quicksort partitions **in place**: sequential scans over contiguous memory, no auxiliary buffer, no copy-back. Mergesort's O(n) scratch buffer doubles memory traffic; quicksort's working set streams through cache. Identical O(n log n) average, but quicksort usually wins wall-clock by a healthy constant factor.

**Not stable:** partition swaps elements across long distances (watch 9 fly past 7 in the trace above) with no memory of original order — two equal keys can leapfrog each other. Making quicksort stable costs the O(n) space that was its whole advantage, so nobody does.

## Introsort — how libraries tame the temper

C++ `std::sort` runs **introsort**: quicksort, but track recursion depth; if it exceeds ~2·log₂ n (pivot luck has clearly gone bad), switch that subproblem to **heapsort** — in-place, guaranteed n log n, just slower constants. Tiny partitions fall through to insertion sort. Result: quicksort's speed on real data, a hard O(n log n) worst-case ceiling, immune to adversarial inputs.

## Quickselect — the same partition, half the work

"K-th largest element" ([LeetCode 215](https://leetcode.com/problems/kth-largest-element-in-an-array/)) doesn't need a full sort. Partition once: the pivot lands at its final index `p`. If `p == k`, done; otherwise recurse into **only the side containing k**. Halving one side instead of both gives n + n/2 + n/4 + … = **O(n) average** (O(n²) worst, same cure: random pivot). Same code, one recursive call instead of two.

## Implementation

```python
import random

def quick_sort(values: list[int], low: int = 0, high: int | None = None) -> None:
    if high is None:
        high = len(values) - 1
    if low >= high:
        return
    pivot_index = partition(values, low, high)
    quick_sort(values, low, pivot_index - 1)
    quick_sort(values, pivot_index + 1, high)

def partition(values: list[int], low: int, high: int) -> int:
    rand = random.randint(low, high)                  # randomized pivot
    values[rand], values[high] = values[high], values[rand]
    pivot = values[high]
    boundary = low - 1                                # end of "< pivot" region
    for j in range(low, high):
        if values[j] < pivot:
            boundary += 1
            values[boundary], values[j] = values[j], values[boundary]
    values[boundary + 1], values[high] = values[high], values[boundary + 1]
    return boundary + 1
```

## Complexity box

| | |
|---|---|
| Time — best / average / worst | O(n log n) / O(n log n) / **O(n²)** |
| Space | O(log n) expected stack (O(n) worst; recurse smaller side first to cap it) |
| Stable? | **No** — partition swaps across long distances |
| In-place? | **Yes** — its core advantage |

## How it loses

- **Bad pivots → O(n²):** sorted/reverse-sorted input with a fixed pivot, or crafted adversarial input. Randomization or introsort is mandatory in anything production-facing.
- **Duplicate-heavy input:** 2-way partition degrades; needs the 3-way variant.
- **Needs stability:** disqualified outright — hand the job to mergesort/Timsort.
- **Hard latency guarantees:** "usually fast, occasionally quadratic" is unacceptable in real-time paths; heapsort or mergesort's flat n log n wins.
- **Linked lists / external data:** partitioning wants random access; mergesort owns both.

## Interview Q&A

**Q: What's the worst case and how do you prevent it?**
O(n²), when pivots repeatedly land near an extreme — depth n, each level scanning almost everything. Sorted input with first/last-element pivot is the everyday trigger. Prevent: randomized pivot or median-of-three (makes expected cost n log n on all inputs), or introsort's depth limit for a hard guarantee.

**Q: Why doesn't C++ std::sort use plain quicksort?**
Plain quicksort has no worst-case guarantee and fixed pivot schemes are exploitable (adversarial inputs forcing n²). Introsort keeps quicksort's average speed but monitors recursion depth and bails to heapsort past ~2 log n, plus insertion sort on tiny ranges — guaranteed O(n log n) with quicksort constants.

**Q: How do you handle an array with many duplicates?**
3-way Dutch-flag partition into `< pivot | == pivot | > pivot`; recurse only on the outer regions, so equal keys are placed once and never revisited. Runtime becomes proportional to n times the number of distinct keys.

**Q: Quicksort vs quickselect?**
Same partition primitive. Quicksort recurses on both sides → O(n log n) full sort. Quickselect recurses only on the side containing the target index → O(n) average, and returns just the k-th element in its place. It's the standard "k-th largest without sorting" answer (LeetCode 215); the heap alternative is O(n log k) but worst-case-safe.
