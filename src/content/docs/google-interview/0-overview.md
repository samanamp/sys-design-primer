---
title: "Google Loop: Overview & Battle Plan"
description: "The interview loop for Staff SWE ML Performance (Core ML), what each round tests, and where in this site the prep for each round lives."
sidebar:
  label: "Overview & Battle Plan"
---

# Google Loop: Overview & Battle Plan

Target: **Staff SWE, ML Performance (TPU / GPU), Google Core ML.** Loop per recruiter (July 2026): **1× coding, 2× system design, 1× Googleyness.** No dedicated domain deep-dive round — domain depth must surface *through* the system-design rounds, which makes them the rounds where the offer is won. The coding round is a filter to pass cleanly; Googleyness is a real staff-level behavioral, not a formality.

## Round 1 — Coding

- **[The 17-day interleaved plan](/google-interview/1-coding-plan/)** — built from LeetCode's Google tag (30-day + 3-month frequency), mixed across domains so every problem forces cold pattern recognition, with implement-from-scratch slots (mergesort, quicksort, trie, heap) matching Google's official prep guide.
- **[Balanced BSTs primer](/algorithms/6-balanced-bsts/)** — the explain-not-code treatment of AVL/red-black/splay from the same guide.
- Bar: clean working code in a real language, unprompted edge cases, complexity stated before coding, one test walked through line by line at the end. Do at least one mock in a Google Doc — that's the actual medium.

## Rounds 2–3 — System design (the offer rounds)

At least one will be ML-performance-flavored ("design a model that runs well on TPU"-shaped). Research across reported loops says expect one of: LLM fine-tune-and-serve optimizing throughput/memory, a Gemini-scale serving system with latency SLOs, a batch-inference API for an accelerator cluster, weight distribution to a fleet, or compute-resource allocation. The second round is likelier a classic distributed-systems staple or an ML-product design, often steered by your resume.

Prep, in order:

1. **[Optimization track](/optimization/0-overview/)** — the substrate: serving, KV cache, parallelism, quantization, TPU/XLA. The [mental-math drills](/optimization/15-mental-math-drills/) are the per-day habit; interviewers use fumbled arithmetic as a shallowness signal.
2. **[TTFT Optimization Program](/optimization/1-ttft-optim/)** — the template for how a staff-level design answer *sounds*: measurement first, budget decomposition, sequenced plan, explicit pushback.
3. **[Trace-reading track](/trace-reading/0-overview/)** — for the "here's a profile, what's wrong" moment inside a design round.
4. L6 rubric, condensed from reported loops: *you* drive — propose scope, do the estimates unprompted, pick the two components worth deep-diving, name failure modes and tradeoffs before being asked, and push back on at least one requirement with arithmetic.

The interview script for any perf design prompt: **state the regime** (prefill/decode, compute/memory-bound, latency/throughput) → **do the roofline/bytes math out loud** → **name the bottleneck** → **design around it** → **say what you'd measure to check yourself**.

## Round 4 — Googleyness (don't wing it)

Staff-level behavioral: leading without authority, conflict with a peer/TL, navigating ambiguity, disagree-and-commit, mentoring, a failure you owned. Prepare 6–8 stories in STAR form from real projects, each with a version that fits in 90 seconds; rehearse the two hardest out loud. Positioning note: hands-on experience is presented **GPU-first** (Nsight, CUDA, NCCL); TPU knowledge is framed as studied depth + transferable performance fundamentals, which is honest and lands fine for a TPU team.

## Working agreements

- Daily habit: mental-math drills + 2–3 trace drills + the day's coding-plan problems.
- Everything Google-loop-specific lives in this folder; the technical depth stays in the topic tracks and gets linked, not duplicated.
