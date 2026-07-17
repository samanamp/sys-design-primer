---
title: "Google Coding Round: 17-Day Daily Plan"
description: "A 17-day LeetCode practice plan built from the Google company tag's 30-day and 3-month frequency data (pulled July 17, 2026), organized by pattern with the interview protocol baked in."
---

# Google Coding Round: 17-Day Daily Plan

Built from the LeetCode Premium Google tag, frequency-sorted, 30-day and 3-month windows (pulled 2026-07-17). Google rotates questions, so this trains the *distribution*, not the paper: arrays/two-pointer, sliding window, binary search (especially on-answer), intervals/monotonic stack, heaps, graphs, light-to-medium DP, and design-lite.

**Protocol per problem (this is the actual interview skill):** restate + ask one clarifying question about constraints → state brute force and its complexity in one sentence → name the target approach and complexity *before coding* → code cleanly, narrating → test on one normal + one edge case out loud → then ask yourself the Google follow-up: "what if the input doesn't fit in memory / arrives as a stream / must be answered in O(1) per query?" Target: 25 min per medium, then stop and read the editorial regardless.

**Daily shape (~75–90 min):** warm-up easy (10 min) + 2 core mediums (50 min) + review yesterday's misses (15 min). Hards are marked — attempt 15 minutes for the idea, then study the solution; Google rarely requires a full hard implementation, but the ideas (binary-search-on-answer, monotonic stack) show up inside mediums.

## Days 1–5 — Arrays, strings, two pointers, sliding window

| Day | Warm-up | Core |
|---|---|---|
| 1 | [1. Two Sum](https://leetcode.com/problems/two-sum/) | [3. Longest Substring Without Repeating Characters](https://leetcode.com/problems/longest-substring-without-repeating-characters/) · [15. 3Sum](https://leetcode.com/problems/3sum/) |
| 2 | [121. Best Time to Buy and Sell Stock](https://leetcode.com/problems/best-time-to-buy-and-sell-stock/) | [11. Container With Most Water](https://leetcode.com/problems/container-with-most-water/) · [560. Subarray Sum Equals K](https://leetcode.com/problems/subarray-sum-equals-k/) |
| 3 | [88. Merge Sorted Array](https://leetcode.com/problems/merge-sorted-array/) | [209. Minimum Size Subarray Sum](https://leetcode.com/problems/minimum-size-subarray-sum/) · [904. Fruit Into Baskets](https://leetcode.com/problems/fruit-into-baskets/) |
| 4 | [169. Majority Element](https://leetcode.com/problems/majority-element/) | [5. Longest Palindromic Substring](https://leetcode.com/problems/longest-palindromic-substring/) · [1004. Max Consecutive Ones III](https://leetcode.com/problems/max-consecutive-ones-iii/) |
| 5 | [268. Missing Number](https://leetcode.com/problems/missing-number/) | [238. Product of Array Except Self](https://leetcode.com/problems/product-of-array-except-self/) · [31. Next Permutation](https://leetcode.com/problems/next-permutation/) · **hard idea:** [42. Trapping Rain Water](https://leetcode.com/problems/trapping-rain-water/) |

## Days 6–9 — Binary search, intervals, stacks/monotonic structures

| Day | Warm-up | Core |
|---|---|---|
| 6 | [35. Search Insert Position](https://leetcode.com/problems/search-insert-position/) | [33. Search in Rotated Sorted Array](https://leetcode.com/problems/search-in-rotated-sorted-array/) · [162. Find Peak Element](https://leetcode.com/problems/find-peak-element/) |
| 7 | [69. Sqrt(x)](https://leetcode.com/problems/sqrtx/) | [875. Koko Eating Bananas](https://leetcode.com/problems/koko-eating-bananas/) · **hard idea:** [410. Split Array Largest Sum](https://leetcode.com/problems/split-array-largest-sum/) (BS-on-answer — high frequency at Google) |
| 8 | [20. Valid Parentheses](https://leetcode.com/problems/valid-parentheses/) | [56. Merge Intervals](https://leetcode.com/problems/merge-intervals/) · [253. Meeting Rooms II](https://leetcode.com/problems/meeting-rooms-ii/) |
| 9 | [496. Next Greater Element I](https://leetcode.com/problems/next-greater-element-i/) | [394. Decode String](https://leetcode.com/problems/decode-string/) · **hard idea:** [239. Sliding Window Maximum](https://leetcode.com/problems/sliding-window-maximum/) (deque) or [84. Largest Rectangle in Histogram](https://leetcode.com/problems/largest-rectangle-in-histogram/) |

## Days 10–13 — Linked lists, trees, graphs, heaps

| Day | Warm-up | Core |
|---|---|---|
| 10 | [206. Reverse Linked List](https://leetcode.com/problems/reverse-linked-list/) | [2. Add Two Numbers](https://leetcode.com/problems/add-two-numbers/) · [146. LRU Cache](https://leetcode.com/problems/lru-cache/) (design staple) |
| 11 | [110. Balanced Binary Tree](https://leetcode.com/problems/balanced-binary-tree/) | [102. Binary Tree Level Order Traversal](https://leetcode.com/problems/binary-tree-level-order-traversal/) · [662. Maximum Width of Binary Tree](https://leetcode.com/problems/maximum-width-of-binary-tree/) |
| 12 | [700. Search in a Binary Search Tree](https://leetcode.com/problems/search-in-a-binary-search-tree/) | [200. Number of Islands](https://leetcode.com/problems/number-of-islands/) · [994. Rotting Oranges](https://leetcode.com/problems/rotting-oranges/) |
| 13 | [1046. Last Stone Weight](https://leetcode.com/problems/last-stone-weight/) | [215. Kth Largest Element in an Array](https://leetcode.com/problems/kth-largest-element-in-an-array/) · [347. Top K Frequent Elements](https://leetcode.com/problems/top-k-frequent-elements/) · **hard idea:** [295. Find Median from Data Stream](https://leetcode.com/problems/find-median-from-data-stream/) (two heaps) |

## Days 14–15 — DP, backtracking, design-lite, simulation

| Day | Warm-up | Core |
|---|---|---|
| 14 | [70. Climbing Stairs](https://leetcode.com/problems/climbing-stairs/) | [198. House Robber](https://leetcode.com/problems/house-robber/) · [322. Coin Change](https://leetcode.com/problems/coin-change/) · [300. Longest Increasing Subsequence](https://leetcode.com/problems/longest-increasing-subsequence/) |
| 15 | [359. Logger Rate Limiter](https://leetcode.com/problems/logger-rate-limiter/) (Google design-lite staple) | [17. Letter Combinations of a Phone Number](https://leetcode.com/problems/letter-combinations-of-a-phone-number/) · [8. String to Integer (atoi)](https://leetcode.com/problems/string-to-integer-atoi/) (messy-spec simulation, very Google) · [54. Spiral Matrix](https://leetcode.com/problems/spiral-matrix/) |

## Days 16–17 — Timed mocks

| Day | Format |
|---|---|
| 16 | **45-min mock, out loud:** 2 unseen mediums from the 3-month list — suggested: [49. Group Anagrams](https://leetcode.com/problems/group-anagrams/) + [1631. Path With Minimum Effort](https://leetcode.com/problems/path-with-minimum-effort/). Then full review. |
| 17 | **45-min mock:** 1 unseen medium ([152. Maximum Product Subarray](https://leetcode.com/problems/maximum-product-subarray/)) + 15-min hard-idea ([41. First Missing Positive](https://leetcode.com/problems/first-missing-positive/)) + redo your two worst misses of the 17 days. |

## After day 17

Re-pull the 30-day tag (it shifts) and convert to maintenance: one timed medium daily, one 45-min two-problem mock weekly. Anything missed twice goes on a spaced-repetition list: redo at +3 days and +10 days.

Cut from the compressed plan but high-frequency — use as maintenance-mode pool: [22. Generate Parentheses](https://leetcode.com/problems/generate-parentheses/), [39. Combination Sum](https://leetcode.com/problems/combination-sum/), [45. Jump Game II](https://leetcode.com/problems/jump-game-ii/), [50. Pow(x, n)](https://leetcode.com/problems/powx-n/), [148. Sort List](https://leetcode.com/problems/sort-list/), [189. Rotate Array](https://leetcode.com/problems/rotate-array/), [332. Reconstruct Itinerary](https://leetcode.com/problems/reconstruct-itinerary/), [424. Longest Repeating Character Replacement](https://leetcode.com/problems/longest-repeating-character-replacement/), [503. Next Greater Element II](https://leetcode.com/problems/next-greater-element-ii/), [540. Single Element in a Sorted Array](https://leetcode.com/problems/single-element-in-a-sorted-array/), [1288. Remove Covered Intervals](https://leetcode.com/problems/remove-covered-intervals/), [1944. Number of Visible People in a Queue](https://leetcode.com/problems/number-of-visible-people-in-a-queue/), [25. Reverse Nodes in k-Group](https://leetcode.com/problems/reverse-nodes-in-k-group/).

## Appendix: raw frequency order (top of each window, 2026-07-17)

**30 days:** Two Sum · Add Two Numbers · Palindrome Number · Median of Two Sorted Arrays · Longest Substring w/o Repeating · Longest Consecutive Sequence · Trapping Rain Water · Sqrt(x) · Best Time to Buy/Sell · Container With Most Water · Merge Two Sorted Lists · Majority Element · Longest Common Prefix · 3Sum · Valid Parentheses · Subarray Sum Equals K · Pow(x,n) · Maximum Subarray · Climbing Stairs · Rotate Array · Min Size Subarray Sum · Longest Palindromic Substring · atoi · Search in Rotated Sorted Array · Combination Sum · Spiral Matrix · LRU Cache · House Robber · 540 · Next Permutation · Jump Game II · N-Queens · Kth Largest · Decode String · Split Array Largest Sum · Koko · Fruit Into Baskets · Sliding Window Maximum · Coin Change · 1944 · 994 · 1631 · 662 · 875…

**3 months adds/emphasizes:** Merge Intervals · Group Anagrams · Number of Islands · Meeting Rooms II · Top K Frequent · Jump Game · Largest Rectangle in Histogram · Level Order Traversal · Find Median from Data Stream · Next Greater Element I/II · LIS · Product of Array Except Self · Sort Colors · Logger Rate Limiter · Longest Repeating Character Replacement · First Missing Positive · Permutations · 81. Search Rotated II.

The 30-day list also contained a tail of very recent additions ([1291](https://leetcode.com/problems/sequential-digits/), [1358](https://leetcode.com/problems/number-of-substrings-containing-all-three-characters/), [1846](https://leetcode.com/problems/maximum-element-after-decreasing-and-rearranging/), [2007](https://leetcode.com/problems/find-original-array-from-doubled-array/), [2484](https://leetcode.com/problems/count-palindromic-subsequences/), [2812](https://leetcode.com/problems/find-the-safest-path-in-a-grid/)) — worth one pass in maintenance mode as "recently reported" signals.
