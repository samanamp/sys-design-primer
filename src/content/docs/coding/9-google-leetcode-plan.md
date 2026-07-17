---
title: "Google Coding Round: 4-Week Daily Plan"
description: "A daily LeetCode practice plan built from the Google company tag's 30-day and 3-month frequency data (pulled July 17, 2026), organized by pattern with the interview protocol baked in."
---

# Google Coding Round: 4-Week Daily Plan

Built from the LeetCode Premium Google tag, frequency-sorted, 30-day and 3-month windows (pulled 2026-07-17). Google rotates questions, so this trains the *distribution*, not the paper: arrays/two-pointer, sliding window, binary search (especially on-answer), intervals/monotonic stack, heaps, graphs, light-to-medium DP, and design-lite.

**Protocol per problem (this is the actual interview skill):** restate + ask one clarifying question about constraints → state brute force and its complexity in one sentence → name the target approach and complexity *before coding* → code cleanly, narrating → test on one normal + one edge case out loud → then ask yourself the Google follow-up: "what if the input doesn't fit in memory / arrives as a stream / must be answered in O(1) per query?" Target: 25 min per medium, then stop and read the editorial regardless.

**Daily shape (~75 min):** 1 warm-up easy (10 min) + 2 mediums (50 min) + review yesterday's misses (15 min). Hards appear ~2×/week — attempt for 15 minutes for the idea, then study the solution; Google rarely requires a full hard implementation, but the *ideas* (binary-search-on-answer, monotonic stack) show up inside mediums.

## Week 1 — Arrays, strings, two pointers, sliding window

| Day | Warm-up | Core |
|---|---|---|
| 1 | 1. Two Sum | 3. Longest Substring Without Repeating Characters · 15. 3Sum |
| 2 | 121. Best Time to Buy/Sell Stock | 11. Container With Most Water · 560. Subarray Sum Equals K |
| 3 | 88. Merge Sorted Array | 209. Minimum Size Subarray Sum · 904. Fruit Into Baskets |
| 4 | 169. Majority Element | 5. Longest Palindromic Substring · 189. Rotate Array |
| 5 | 14. Longest Common Prefix | 1004. Max Consecutive Ones III · 424. Longest Repeating Character Replacement |
| 6 | 268. Missing Number | 238. Product of Array Except Self · 31. Next Permutation |
| 7 | review day | 42. Trapping Rain Water (hard, two-pointer idea) · redo the week's worst two |

## Week 2 — Binary search, intervals, stacks/monotonic structures

| Day | Warm-up | Core |
|---|---|---|
| 1 | 35. Search Insert Position | 33. Search in Rotated Sorted Array · 162. Find Peak Element |
| 2 | 69. Sqrt(x) | 875. Koko Eating Bananas · 540. Single Element in Sorted Array |
| 3 | 20. Valid Parentheses | 56. Merge Intervals · 253. Meeting Rooms II |
| 4 | 496. Next Greater Element I | 503. Next Greater Element II · 1944. Visible People in a Queue (hard, monotonic stack) |
| 5 | 202. Happy Number | 410. Split Array Largest Sum (hard, BS-on-answer — high-frequency at Google) · 1288. Remove Covered Intervals |
| 6 | 27. Remove Element | 239. Sliding Window Maximum (hard, deque) · 394. Decode String |
| 7 | review day | 84. Largest Rectangle in Histogram (hard) · redo worst two |

## Week 3 — Linked lists, trees, graphs, heaps

| Day | Warm-up | Core |
|---|---|---|
| 1 | 206. Reverse Linked List | 2. Add Two Numbers · 141. Linked List Cycle → follow-up: find cycle start |
| 2 | 21. Merge Two Sorted Lists | 146. LRU Cache (design staple) · 148. Sort List |
| 3 | 110. Balanced Binary Tree | 102. Level Order Traversal · 662. Maximum Width of Binary Tree |
| 4 | 700. Search in a BST | 200. Number of Islands · 994. Rotting Oranges |
| 5 | 1046. Last Stone Weight | 215. Kth Largest Element · 347. Top K Frequent Elements |
| 6 | 100. Same Tree | 1631. Path With Minimum Effort (Dijkstra/BS) · 332. Reconstruct Itinerary (hard) |
| 7 | review day | 295. Find Median from Data Stream (hard, two heaps) · 25. Reverse Nodes in k-Group (hard) |

## Week 4 — DP, backtracking, design, simulation + mixed mocks

| Day | Warm-up | Core |
|---|---|---|
| 1 | 70. Climbing Stairs | 198. House Robber · 322. Coin Change |
| 2 | 509. Fibonacci Number | 300. Longest Increasing Subsequence · 152. Maximum Product Subarray |
| 3 | 412. Fizz Buzz | 17. Letter Combinations · 39. Combination Sum |
| 4 | 359. Logger Rate Limiter (Google design-lite staple) | 22. Generate Parentheses · 54. Spiral Matrix |
| 5 | 50. Pow(x, n) | 8. String to Integer (atoi) — messy-spec simulation, very Google · 45. Jump Game II |
| 6 | — | **Timed mock:** 2 unseen mediums from the 3-month list, 45 min, out loud |
| 7 | — | **Timed mock:** 1 medium + 15-min hard-idea (41. First Missing Positive) + review |

## After week 4

Re-pull the 30-day tag (it shifts) and convert to maintenance: one timed medium daily, one 45-min two-problem mock weekly. Anything you missed twice goes on a spaced-repetition list: redo at +3 days and +10 days.

## Appendix: raw frequency order (top of each window, 2026-07-17)

**30 days:** Two Sum · Add Two Numbers · Palindrome Number · Median of Two Sorted Arrays · Longest Substring w/o Repeating · Longest Consecutive Sequence · Trapping Rain Water · Sqrt(x) · Best Time to Buy/Sell · Container With Most Water · Merge Two Sorted Lists · Majority Element · Longest Common Prefix · 3Sum · Valid Parentheses · Subarray Sum Equals K · Pow(x,n) · Maximum Subarray · Climbing Stairs · Rotate Array · Min Size Subarray Sum · Longest Palindromic Substring · atoi · Search in Rotated Sorted Array · Combination Sum · Spiral Matrix · LRU Cache · House Robber · 540 · Next Permutation · Jump Game II · N-Queens · Kth Largest · Decode String · Split Array Largest Sum · Koko · Fruit Into Baskets · Sliding Window Maximum · Coin Change · 1944 · 994 · 1631 · 662 · 875…

**3 months adds/emphasizes:** Merge Intervals · Group Anagrams · Number of Islands · Meeting Rooms II · Top K Frequent · Jump Game · Largest Rectangle in Histogram · Level Order Traversal · Find Median from Data Stream · Next Greater Element I/II · LIS · Product of Array Except Self · Sort Colors · Logger Rate Limiter · Longest Repeating Character Replacement · First Missing Positive · Permutations · 81. Search Rotated II.

The 30-day list also contained a tail of very recent additions (1291 Sequential Digits, 1358, 1846, 2007, 2484, 2812, 3534, 3658, 3699) — worth one pass in maintenance mode as "recently reported" signals.
