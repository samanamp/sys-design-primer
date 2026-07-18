---
title: "Mergesort: The Reliable One"
description: "Guaranteed n log n, stable, predictable: the recursion tree traced on a real array, the two-pointer merge step by step, why stability matters, the O(n) space tax, and where mergesort quietly runs everything (Timsort, external sort)."
---

# Mergesort: The Reliable One

Mergesort's pitch is boring on purpose: **O(n log n) always** — best case, worst case, adversarial case — plus **stability**. No pivot luck, no degenerate inputs. The price is O(n) auxiliary space. When an interviewer says "sort a linked list," "sort a huge file," or "sort by X, ties by Y," mergesort is the answer they're fishing for.

The algorithm in one breath: split the array in half until pieces are size 1 (trivially sorted), then merge sorted halves back together in linear time.

## The recursion tree — [38, 27, 43, 3, 9, 82, 10]

Split down, merge up. Every node shows real values:

```
SPLIT (top → bottom)
                [38, 27, 43, 3, 9, 82, 10]
                /                        \
        [38, 27, 43]                [3, 9, 82, 10]
        /         \                  /           \
    [38]       [27, 43]         [3, 9]        [82, 10]
                /    \           /   \          /    \
             [27]   [43]       [3]   [9]     [82]   [10]

MERGE (bottom → top)
             [27]   [43]       [3]   [9]     [82]   [10]
                \    /           \   /          \    /
    [38]       [27, 43]         [3, 9]        [10, 82]
        \         /                  \           /
        [27, 38, 43]                [3, 9, 10, 82]
                \                        /
                [3, 9, 10, 27, 38, 43, 82]
```

Size-1 arrays are the base case — a single element is already sorted. All the actual work happens on the way back up, in `merge`.

## The merge — two pointers, traced

Merging `[3, 27, 38, 43]` and `[9, 10, 82]`. Compare the fronts, take the smaller, advance that pointer:

```
Left:  [3, 27, 38, 43]    Right: [9, 10, 82]
        i                         j              Output

step 1: 3 vs 9   → take 3,  i→27      out = [3]
step 2: 27 vs 9  → take 9,  j→10      out = [3, 9]
step 3: 27 vs 10 → take 10, j→82      out = [3, 9, 10]
step 4: 27 vs 82 → take 27, i→38      out = [3, 9, 10, 27]
step 5: 38 vs 82 → take 38, i→43      out = [3, 9, 10, 27, 38]
step 6: 43 vs 82 → take 43, i done    out = [3, 9, 10, 27, 38, 43]
step 7: left exhausted → drain right  out = [3, 9, 10, 27, 38, 43, 82]
```

Each element is looked at once → merge is O(n). One detail carries the whole stability story: on a **tie**, take from the **left** array (`<=` in code). Left elements came earlier in the original array, so equal keys keep their original order.

## Why n log n — the picture

Halving 7 elements bottoms out in ⌈log₂ 7⌉ = 3 splits. At every level, the merges collectively touch all n elements once:

```
Level 0:  [3,9,10,27,38,43,82]                    ← 1 merge  × 7 elems = n work
Level 1:  [27,38,43]      [3,9,10,82]             ← 2 merges, 7 elems  = n work
Level 2:  [27,43] [3,9] [10,82]  (+ [38] idle)    ← 3 merges, 6 elems  ≈ n work
Level 3:  [38][27][43][3][9][82][10]              ← base case, no work

          log n levels × O(n) per level = O(n log n)
```

No input can make the tree deeper — the split is always by index, never by value. That's the structural difference from quicksort, whose tree depth depends on pivot luck.

## Stability — why anyone cares

Stable = equal keys keep their original relative order. This is what makes **multi-key sorting composable**: sort employees by name, then stable-sort by department — within each department, names stay alphabetized. With an unstable sort, the second pass scrambles the first. Real systems do this constantly (ORDER BY dept, name; spreadsheet column sorts), which is why Python's `sorted` and Java's object sort are stable — both mergesort derivatives.

## The space tax — and the linked-list loophole

Array mergesort needs an O(n) scratch buffer: you can't merge two adjacent sorted runs in place in linear time without one (in-place merging exists but is complicated and slow in practice). Plus O(log n) recursion stack.

On a **linked list**, the tax vanishes: merging is just pointer re-splicing — no buffer at all. That's why [LeetCode 148 (Sort List)](https://leetcode.com/problems/sort-list/) with its O(1)-extra-space follow-up is a mergesort problem: split with fast/slow pointers, merge by relinking nodes. Quicksort is awkward on lists (no random access for pivots); mergesort is natural.

**Bottom-up variant, in two sentences:** skip recursion entirely — merge adjacent runs of width 1, then 2, then 4, doubling until one run covers the array. Same O(n log n), no call stack, and it's the natural shape for linked lists and external sorting.

## Where it runs in production

- **Timsort** (Python `sorted`, Java `Arrays.sort` for objects) is bottom-up mergesort that first detects already-sorted runs in the input, then merges them galloping-style — O(n) on nearly-sorted data, stable, n log n worst case.
- **External sort:** when data exceeds RAM, you sort chunks in memory and k-way merge sorted runs from disk — mergesort's merge is the only step that works on streams you can't hold in memory. Full treatment in [External Sort & K-Way Merge](/coding/20-external-sort-kway-merge/).

## Implementation

```python
def merge_sort(values: list[int]) -> list[int]:
    if len(values) <= 1:
        return values
    mid = len(values) // 2
    left = merge_sort(values[:mid])
    right = merge_sort(values[mid:])
    return merge(left, right)

def merge(left: list[int], right: list[int]) -> list[int]:
    merged: list[int] = []
    i = j = 0
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:          # <= keeps it stable
            merged.append(left[i])
            i += 1
        else:
            merged.append(right[j])
            j += 1
    merged.extend(left[i:])
    merged.extend(right[j:])
    return merged
```

## Complexity box

| | |
|---|---|
| Time — best / average / worst | O(n log n) / O(n log n) / O(n log n) |
| Space | O(n) auxiliary (arrays); O(1) extra on linked lists; O(log n) stack |
| Stable? | **Yes** (with `<=` on the left) |
| In-place? | **No** |

## How it loses

- **Memory:** the O(n) buffer disqualifies it in tight-memory / embedded contexts where quicksort or heapsort sort in place.
- **Constant factors:** copying to and from the scratch buffer means more memory traffic and worse cache behavior than quicksort's in-place partitioning — quicksort typically wins wall-clock on random arrays despite identical average complexity.
- **Overkill for primitives:** when stability is meaningless (raw ints), libraries often pick something in-place instead — historically why C++/Java used quicksort/introsort variants for primitives.

## Interview Q&A

**Q: Why is mergesort stable?**
On tied keys, merge takes from the left half first (`<=`). Left-half elements preceded right-half elements in the original array, so equal keys never reorder — at any level of the recursion. Flip that to `<` and stability is gone; that one character is the whole property.

**Q: Sort a 10 GB file with 1 GB of RAM?**
External mergesort. Pass 1: read ~1 GB chunks, sort each in memory, write ~10 sorted runs to disk. Pass 2: k-way merge the 10 runs with a min-heap over their front elements, streaming output — only one buffered block per run in memory at a time. Merge is the only sort step that works on streams, which is why this is always mergesort. Details: [External Sort & K-Way Merge](/coding/20-external-sort-kway-merge/).

**Q: Why does Python use a mergesort derivative (Timsort) instead of quicksort?**
Python sorts arbitrary objects where stability is a documented guarantee (multi-key sorts must compose), and real-world data is often partially sorted — Timsort exploits existing runs to hit O(n) on nearly-sorted input while keeping the n log n worst-case guarantee. Quicksort offers none of the three.

**Q: Sort a linked list in O(n log n) with O(1) extra space?**
Bottom-up mergesort on the list: merge runs of width 1, 2, 4, … by re-splicing pointers. No auxiliary buffer (merging lists is relinking) and no recursion stack. This is the intended answer to LeetCode 148's follow-up.
