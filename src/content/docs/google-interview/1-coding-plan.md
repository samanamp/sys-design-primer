---
title: "Google Coding Round: 17-Day Daily Plan"
description: "A 17-day LeetCode practice plan built from the Google company tag's 30-day and 3-month frequency data (pulled July 17, 2026), fully interleaved across domains so every problem forces cold pattern recognition."
---

# Google Coding Round: 17-Day Daily Plan

Built from the LeetCode Premium Google tag, frequency-sorted, 30-day and 3-month windows (pulled 2026-07-17). Google rotates questions, so this trains the *distribution*, not the paper: arrays/two-pointer, sliding window, binary search (especially on-answer), intervals/monotonic stack, heaps, graphs, light-to-medium DP, and design-lite. The plan is **interleaved, not blocked**: consecutive problems come from different domains, so recognizing the pattern is part of every rep — blocked practice pre-loads the answer to that question and produces fake confidence.

**Protocol per problem (this is the actual interview skill):** restate + ask one clarifying question about constraints → state brute force and its complexity in one sentence → name the target approach and complexity *before coding* → code cleanly, narrating → test on one normal + one edge case out loud → then ask yourself the Google follow-up: "what if the input doesn't fit in memory / arrives as a stream / must be answered in O(1) per query?" Target: 25 min per medium, then stop and read the editorial regardless.

**Daily shape (~75–90 min):** warm-up easy (10 min) + 2 core mediums (50 min) + review yesterday's misses (15 min). Hards are marked — attempt 15 minutes for the idea, then study the solution; Google rarely requires a full hard implementation, but the ideas (binary-search-on-answer, monotonic stack) show up inside mediums.

## Days 1–15 — interleaved

Domains are deliberately mixed within and across days — no "binary-search week." Each problem should force the *which pattern is this?* decision cold, because that decision is most of the interview. If two problems in a row feel like the same tool, that's a bug; tell me and I'll reshuffle.

| Day | Warm-up | Core |
|---|---|---|
| 1 | [1. Two Sum](https://leetcode.com/problems/two-sum/) (hash) | [33. Search in Rotated Sorted Array](https://leetcode.com/problems/search-in-rotated-sorted-array/) (binary search) · [102. Binary Tree Level Order Traversal](https://leetcode.com/problems/binary-tree-level-order-traversal/) (tree/BFS) |
| 2 | [206. Reverse Linked List](https://leetcode.com/problems/reverse-linked-list/) (linked list) | [560. Subarray Sum Equals K](https://leetcode.com/problems/subarray-sum-equals-k/) (prefix-sum hash) · [253. Meeting Rooms II](https://leetcode.com/problems/meeting-rooms-ii/) (intervals/heap) |
| 3 | [20. Valid Parentheses](https://leetcode.com/problems/valid-parentheses/) (stack) | [3. Longest Substring Without Repeating Characters](https://leetcode.com/problems/longest-substring-without-repeating-characters/) (sliding window) · [200. Number of Islands](https://leetcode.com/problems/number-of-islands/) (graph/DFS) · **implement:** [912. Sort an Array](https://leetcode.com/problems/sort-an-array/) — write mergesort by hand, no library sort |
| 4 | [70. Climbing Stairs](https://leetcode.com/problems/climbing-stairs/) (DP) | [11. Container With Most Water](https://leetcode.com/problems/container-with-most-water/) (two pointers) · [875. Koko Eating Bananas](https://leetcode.com/problems/koko-eating-bananas/) (BS-on-answer) |
| 5 | [110. Balanced Binary Tree](https://leetcode.com/problems/balanced-binary-tree/) (tree) | [15. 3Sum](https://leetcode.com/problems/3sum/) (two pointers) · [322. Coin Change](https://leetcode.com/problems/coin-change/) (DP) · **hard idea:** [42. Trapping Rain Water](https://leetcode.com/problems/trapping-rain-water/) |
| 6 | [268. Missing Number](https://leetcode.com/problems/missing-number/) (bit/math) | [146. LRU Cache](https://leetcode.com/problems/lru-cache/) (design) · [209. Minimum Size Subarray Sum](https://leetcode.com/problems/minimum-size-subarray-sum/) (sliding window) |
| 7 | [35. Search Insert Position](https://leetcode.com/problems/search-insert-position/) (binary search) | [56. Merge Intervals](https://leetcode.com/problems/merge-intervals/) (intervals) · [198. House Robber](https://leetcode.com/problems/house-robber/) (DP) · **implement:** [208. Implement Trie](https://leetcode.com/problems/implement-trie-prefix-tree/) from scratch |
| 8 | [121. Best Time to Buy and Sell Stock](https://leetcode.com/problems/best-time-to-buy-and-sell-stock/) (array) | [394. Decode String](https://leetcode.com/problems/decode-string/) (stack) · [994. Rotting Oranges](https://leetcode.com/problems/rotting-oranges/) (graph/BFS) |
| 9 | [1046. Last Stone Weight](https://leetcode.com/problems/last-stone-weight/) (heap) | [5. Longest Palindromic Substring](https://leetcode.com/problems/longest-palindromic-substring/) (string) · [162. Find Peak Element](https://leetcode.com/problems/find-peak-element/) (binary search) · **hard idea:** [410. Split Array Largest Sum](https://leetcode.com/problems/split-array-largest-sum/) (BS-on-answer — high frequency at Google) |
| 10 | [169. Majority Element](https://leetcode.com/problems/majority-element/) (array) | [2. Add Two Numbers](https://leetcode.com/problems/add-two-numbers/) (linked list) · [347. Top K Frequent Elements](https://leetcode.com/problems/top-k-frequent-elements/) (heap/hash) · **implement:** [703. Kth Largest in a Stream](https://leetcode.com/problems/kth-largest-element-in-a-stream/) — write the min-heap (sift-up/sift-down) yourself, no heapq |
| 11 | [496. Next Greater Element I](https://leetcode.com/problems/next-greater-element-i/) (monotonic stack) | [904. Fruit Into Baskets](https://leetcode.com/problems/fruit-into-baskets/) (sliding window) · [215. Kth Largest Element in an Array](https://leetcode.com/problems/kth-largest-element-in-an-array/) (quickselect/heap) |
| 12 | [700. Search in a Binary Search Tree](https://leetcode.com/problems/search-in-a-binary-search-tree/) (tree) | [238. Product of Array Except Self](https://leetcode.com/problems/product-of-array-except-self/) (array) · [17. Letter Combinations of a Phone Number](https://leetcode.com/problems/letter-combinations-of-a-phone-number/) (backtracking) · **implement:** redo 912 as quicksort (in-place partition, randomized pivot) + [589. N-ary Tree Preorder](https://leetcode.com/problems/n-ary-tree-preorder-traversal/) iteratively |
| 13 | [69. Sqrt(x)](https://leetcode.com/problems/sqrtx/) (binary search) | [662. Maximum Width of Binary Tree](https://leetcode.com/problems/maximum-width-of-binary-tree/) (tree/BFS) · [8. String to Integer (atoi)](https://leetcode.com/problems/string-to-integer-atoi/) (messy-spec simulation, very Google) · **hard idea:** [239. Sliding Window Maximum](https://leetcode.com/problems/sliding-window-maximum/) (deque) or [84. Largest Rectangle in Histogram](https://leetcode.com/problems/largest-rectangle-in-histogram/) |
| 14 | [88. Merge Sorted Array](https://leetcode.com/problems/merge-sorted-array/) (two pointers) | [31. Next Permutation](https://leetcode.com/problems/next-permutation/) (array) · [300. Longest Increasing Subsequence](https://leetcode.com/problems/longest-increasing-subsequence/) (DP + binary search) |
| 15 | [359. Logger Rate Limiter](https://leetcode.com/problems/logger-rate-limiter/) (design-lite, Google staple) | [54. Spiral Matrix](https://leetcode.com/problems/spiral-matrix/) (simulation) · [1004. Max Consecutive Ones III](https://leetcode.com/problems/max-consecutive-ones-iii/) (sliding window) · **hard idea:** [295. Find Median from Data Stream](https://leetcode.com/problems/find-median-from-data-stream/) (two heaps) |

## Days 16–17 — Timed mocks

| Day | Format |
|---|---|
| 16 | **45-min mock, out loud:** 2 unseen mediums from the 3-month list — suggested: [49. Group Anagrams](https://leetcode.com/problems/group-anagrams/) + [1631. Path With Minimum Effort](https://leetcode.com/problems/path-with-minimum-effort/). Then full review. |
| 17 | **45-min mock:** 1 unseen medium ([152. Maximum Product Subarray](https://leetcode.com/problems/maximum-product-subarray/)) + 15-min hard-idea ([41. First Missing Positive](https://leetcode.com/problems/first-missing-positive/)) + redo your two worst misses of the 17 days. |

## CS fundamentals checklist (from Google's official prep guide)

Google's guide asks for more than tagged problems measure. Be able to do each of these cold — the **implement** slots above cover the hands-on ones:

- **Sorting:** write mergesort AND quicksort from scratch (912, days 3 and 12); state their complexities, stability, in-place-ness, and when each loses (quicksort worst case, mergesort allocation). Know why Python/C++ use hybrids (Timsort/introsort).
- **Heaps:** implement sift-up/sift-down and heapify (703, day 10); know heapify is O(n) and why.
- **Hashtables:** explain collision resolution (chaining vs open addressing), resize/amortization, and what makes a bad hash. Be ready to say when a hashtable is the WRONG choice (ordered iteration, range queries → tree).
- **Tries:** implement insert/search/startsWith (208, day 7); know the space tradeoff vs hashing and when tries win (prefix queries, autocomplete — a Google favorite domain).
- **Balanced BSTs (red-black / AVL / splay):** the bar is *explain, not code*: the invariant each maintains, why rotations restore it in O(1), why height stays O(log n), and the practical answer — "in production I'd use the language's sorted container, which is typically a red-black tree." Learn and practice from the dedicated primer: [Balanced BSTs: AVL, Red-Black, Splay](/google-interview/2-balanced-bsts/).
- **Trees generally:** binary, n-ary (589, day 12), and traversals both recursive AND iterative (interviewers ask for the iterative version to test stack fluency).
- **Big-O:** for every problem in this plan, state time AND space before coding — including the recursion-stack space people forget.

**Code-quality bar (their words: clean, bug-free, edge cases, maintainability):** real language, no pseudo-code; name variables like production code; handle empty/single/duplicate/overflow inputs unprompted; after coding, walk one test through the code line by line — that's the "testing whiteboard code" bullet, and skipping it is the most common flag.

## After day 17

Re-pull the 30-day tag (it shifts) and convert to maintenance: one timed medium daily, one 45-min two-problem mock weekly. Anything missed twice goes on a spaced-repetition list: redo at +3 days and +10 days.

Cut from the compressed plan but high-frequency — use as maintenance-mode pool: [22. Generate Parentheses](https://leetcode.com/problems/generate-parentheses/), [39. Combination Sum](https://leetcode.com/problems/combination-sum/), [45. Jump Game II](https://leetcode.com/problems/jump-game-ii/), [50. Pow(x, n)](https://leetcode.com/problems/powx-n/), [148. Sort List](https://leetcode.com/problems/sort-list/), [189. Rotate Array](https://leetcode.com/problems/rotate-array/), [332. Reconstruct Itinerary](https://leetcode.com/problems/reconstruct-itinerary/), [424. Longest Repeating Character Replacement](https://leetcode.com/problems/longest-repeating-character-replacement/), [503. Next Greater Element II](https://leetcode.com/problems/next-greater-element-ii/), [540. Single Element in a Sorted Array](https://leetcode.com/problems/single-element-in-a-sorted-array/), [1288. Remove Covered Intervals](https://leetcode.com/problems/remove-covered-intervals/), [1944. Number of Visible People in a Queue](https://leetcode.com/problems/number-of-visible-people-in-a-queue/), [25. Reverse Nodes in k-Group](https://leetcode.com/problems/reverse-nodes-in-k-group/).

## Appendix: raw frequency order (top of each window, 2026-07-17)

**30 days:** Two Sum · Add Two Numbers · Palindrome Number · Median of Two Sorted Arrays · Longest Substring w/o Repeating · Longest Consecutive Sequence · Trapping Rain Water · Sqrt(x) · Best Time to Buy/Sell · Container With Most Water · Merge Two Sorted Lists · Majority Element · Longest Common Prefix · 3Sum · Valid Parentheses · Subarray Sum Equals K · Pow(x,n) · Maximum Subarray · Climbing Stairs · Rotate Array · Min Size Subarray Sum · Longest Palindromic Substring · atoi · Search in Rotated Sorted Array · Combination Sum · Spiral Matrix · LRU Cache · House Robber · 540 · Next Permutation · Jump Game II · N-Queens · Kth Largest · Decode String · Split Array Largest Sum · Koko · Fruit Into Baskets · Sliding Window Maximum · Coin Change · 1944 · 994 · 1631 · 662 · 875…

**3 months adds/emphasizes:** Merge Intervals · Group Anagrams · Number of Islands · Meeting Rooms II · Top K Frequent · Jump Game · Largest Rectangle in Histogram · Level Order Traversal · Find Median from Data Stream · Next Greater Element I/II · LIS · Product of Array Except Self · Sort Colors · Logger Rate Limiter · Longest Repeating Character Replacement · First Missing Positive · Permutations · 81. Search Rotated II.

The 30-day list also contained a tail of very recent additions ([1291](https://leetcode.com/problems/sequential-digits/), [1358](https://leetcode.com/problems/number-of-substrings-containing-all-three-characters/), [1846](https://leetcode.com/problems/maximum-element-after-decreasing-and-rearranging/), [2007](https://leetcode.com/problems/find-original-array-from-doubled-array/), [2484](https://leetcode.com/problems/count-palindromic-subsequences/), [2812](https://leetcode.com/problems/find-the-safest-path-in-a-grid/)) — worth one pass in maintenance mode as "recently reported" signals.
