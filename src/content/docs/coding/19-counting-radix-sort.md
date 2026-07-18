---
title: "Counting & Radix: Beating n log n When Keys Are Bounded"
description: "Full traces of counting sort and LSD radix sort, why the Ω(n log n) lower bound doesn't apply, why stability is the load-bearing property, and when these actually win."
---

# Counting & Radix: Beating n log n When Keys Are Bounded

Comparison sorts can't beat Ω(n log n) — but counting sort isn't a comparison sort. It never asks "is a < b?"; it uses the key *as an array index* and reads off the answer. That side-steps the lower bound entirely, at the price of space proportional to the key range k. Radix sort is the fix for when k is huge: chop keys into digits and run several small-k counting sorts. Together they're the standard answer to "can you sort faster than n log n?" — *yes, when keys are bounded integers (or behave like them)*.

## Counting sort, traced end to end

Sort `[4, 2, 2, 8, 3, 3, 1]`, keys in range 0..8 (k = 9).

**Pass 1 — count occurrences:**

```
value:   0  1  2  3  4  5  6  7  8
count:  [0, 1, 2, 2, 1, 0, 0, 0, 1]     e.g. two 2s, two 3s, one 8
```

**Pass 2 — prefix sums.** Each cell becomes "how many elements are ≤ this value" — i.e. one past the last slot this value may occupy in the output:

```
value:   0  1  2  3  4  5  6  7  8
count:  [0, 1, 3, 5, 6, 6, 6, 6, 7]     count[3]=5 → the 3s end at index 5 (slots 3,4)
```

**Pass 3 — stable placement.** Walk the *input from right to left*; for each element, decrement its counter and drop it at that index. Right-to-left plus pre-decrement means equal keys keep their relative order:

```
input right→left      counter used            output [ _, _, _, _, _, _, _ ]
1  → count[1]: 1→0 → out[0]=1              [1, _, _, _, _, _, _]
3  → count[3]: 5→4 → out[4]=3  (later 3)   [1, _, _, _, 3, _, _]
3  → count[3]: 4→3 → out[3]=3  (earlier 3) [1, _, _, 3, 3, _, _]
8  → count[8]: 7→6 → out[6]=8              [1, _, _, 3, 3, _, 8]
2  → count[2]: 3→2 → out[2]=2              [1, _, 2, 3, 3, _, 8]
2  → count[2]: 2→1 → out[1]=2              [1, 2, 2, 3, 3, _, 8]
4  → count[4]: 6→5 → out[5]=4              [1, 2, 2, 3, 3, 4, 8]
```

Note the two 3s: the later one landed at index 4, the earlier at index 3 — original order preserved. Total: O(n + k) time, O(n + k) space, zero comparisons.

## LSD radix sort, traced

When k is large, sort digit by digit, **least significant first**, using a stable small-k sort (counting sort with k = 10) for each pass. Sort `[170, 45, 75, 90, 802, 24, 2, 66]` — 3 digits, 3 passes:

```
pass 1 — ones digit (stable):
  bucket 0: 170, 90    2: 802, 2    4: 24    5: 45, 75    6: 66
  → [170, 90, 802, 2, 24, 45, 75, 66]

pass 2 — tens digit (stable):
  bucket 0: 802, 2     2: 24        4: 45    6: 66        7: 170, 75    9: 90
  → [802, 2, 24, 45, 66, 170, 75, 90]

pass 3 — hundreds digit (stable):
  bucket 0: 2, 24, 45, 66, 75, 90        1: 170        8: 802
  → [2, 24, 45, 66, 75, 90, 170, 802]   ✓ sorted
```

Watch 170 vs 75 through pass 2: both have tens digit 7, and stability keeps 170 before 75 — which is exactly the ones-digit order from pass 1. That's the invariant: **after pass i, the array is sorted by the last i digits.** Pass 3 groups by hundreds digit; within bucket 0, the earlier passes' order (already correct on tens+ones) survives *only because the sort is stable*. Total: O(d·(n + b)) for d digits in base b.

## The decision picture

```
                    keys comparable only ──────────►  Ω(n log n) floor. Quicksort/
                    (floats, strings, objects)        mergesort/Timsort. No way out.
   what are
   your keys? ──►   integers, range k ≈ O(n) ──────►  counting sort: O(n + k). Done.
                    (ages, bytes, small enums)
                                                       counting sort alone: count
                    integers, k huge ──────────────►  array of 4 BILLION cells for
                    (32-bit ints: k = 2³²)            32-bit keys — absurd.
                                                              │
                                                              ▼ radix rescues it:
                                                       4 passes of byte-wise counting
                                                       (b = 256): 4·(n + 256) = O(n)
```

Concretely, n = 1M random 32-bit ints: naive counting sort needs a 4-billion-entry count array (16 GB) to sort 4 MB of data. Radix by bytes: 4 passes, each with a 256-cell count array — total work ~4n, total scratch one n-sized buffer. That's the whole trick: **trade one impossible k for d easy ones.**

Why no contradiction with the lower bound: Ω(n log n) is proved by counting leaves of a *comparison* decision tree — n! orderings need log(n!) ≈ n log n yes/no questions. Counting sort asks zero comparisons; it extracts log k bits of information per key by indexing. Different model, different bound. (Equivalently: radix costs O(n · d), and d ≥ log_b k — for arbitrary-precision keys where k grows with n, you're back to n log n. The win requires *bounded* keys.)

## Where this is real

- **GPU sorting is radix sorting.** CUB/Thrust `sort` on NVIDIA GPUs is a radix sort — counting histograms and scatter passes parallelize beautifully; comparison sorts don't. When an accelerator "sorts", this is how.
- **Bucketing by a small key** in serving systems — group requests by shard id, length class, or priority (0..255): that's one counting-sort pass, and engineers write it without calling it sorting.
- **Suffix arrays**: the classic O(n log n) construction (and linear DC3) sorts rank-pairs each round with radix/counting sort — stability is what makes the rank composition correct.
- **Fixed-width records**: phone numbers, IPs, timestamps bucketed by day — LSD radix on the raw digits/bytes.

## Implementation

```python
def counting_sort_by(a: list, key, k: int) -> list:
    """Stable counting sort of a by key(x) in range [0, k)."""
    count = [0] * k
    for x in a:
        count[key(x)] += 1
    for v in range(1, k):                 # prefix sums: end positions
        count[v] += count[v - 1]
    out = [None] * len(a)
    for x in reversed(a):                 # right-to-left → stable
        count[key(x)] -= 1
        out[count[key(x)]] = x
    return out


def radix_sort(a: list) -> list:
    """LSD radix sort of non-negative ints, one byte per pass."""
    if not a:
        return a
    passes = (max(a).bit_length() + 7) // 8
    for p in range(passes):
        shift = 8 * p
        a = counting_sort_by(a, lambda x: (x >> shift) & 0xFF, 256)
    return a
```

## Complexity box

| | Counting sort | LSD radix sort |
|---|---|---|
| **Time** | O(n + k) | O(d · (n + b)) — d digits, base b |
| **Space** | O(n + k) | O(n + b) |
| **Stable** | Yes (right-to-left placement) | Yes — and it *requires* a stable inner sort |
| **In-place** | No | No (ping-pong buffers) |

## How it loses

- **k ≫ n** with no digit decomposition available (sparse arbitrary keys, floats without bit tricks, general objects): the count array dwarfs the data. Comparison sorts don't care about key range.
- **Comparator-defined order** ("sort by name, then by custom collation"): keys aren't small ints; radix on strings exists (MSD) but is niche.
- **Small n or few wide keys**: d passes over the data with a scatter each — quicksort's single cache-friendly streaming partition wins when d·n passes cost more than n log n comparisons (log n < d·c is common for modest n and 64-bit keys).
- **Memory-tight, in-place requirements**: the output buffer is O(n); heapsort/quicksort need O(1)/O(log n).

## Interview Q&A

**Q: Can you sort in O(n)? Doesn't that violate the lower bound?**
Yes, when keys are integers in a bounded range — counting sort O(n + k), radix O(d·n). No violation: Ω(n log n) applies only to comparison-based sorts (a decision tree distinguishing n! orderings). Counting sort indexes by key instead of comparing. Say the qualifier out loud: "for bounded integer keys" — that's the point being tested.

**Q: Why must radix's inner sort be stable?**
The pass-i invariant is "sorted by the last i digits." Pass i+1 groups by digit i+1; within a group, correctness depends on the previous passes' order surviving untouched — exactly what stability guarantees. Unstable inner passes scramble it: in the trace, 170 stays ahead of 75 in the tens pass only because pass 1 put it there and pass 2 didn't reorder equals.

**Q: When does radix lose to quicksort?**
Wide keys with small n (d passes of scatter beat log n comparisons only when n is large or d small), non-integer/comparator keys, tight memory (radix needs an O(n) buffer), and cache behavior — each radix pass scatters to 256 destinations while quicksort streams. Rule of thumb: radix wins on millions of narrow fixed-width keys; quicksort wins on general workloads, which is why it's the library default.

**Q: How would you sort 1 billion phone numbers?**
Ten-digit numbers → fixed-width integer keys, ideal for LSD radix: 5 passes at 2 digits each (b = 100), ~5 linear scans, no comparisons. Follow-ups worth volunteering: if they must be *distinct*, a bitmap of 10^10 bits ≈ 1.25 GB and one scan suffices (the classic *Programming Pearls* answer); if the data doesn't fit in RAM, shift to external sort — partition by leading digits into files, radix-sort each, concatenate.
