---
title: "Trace Reading Track: Overview"
description: "How to build the perceptual skill of reading GPU profiler timelines — vocabulary, guided walkthroughs, hands-on labs, and an interactive trainer — for staff-level ML performance interviews."
sidebar:
  label: "Overview"
---

# Trace Reading Track: Overview

Performance interviews at the staff level almost always include a moment where someone puts a profile in front of you: "here's the trace — what's wrong?" On the job it's the daily loop. The [optimization track](/optimization/0-overview/) trains the arithmetic and the vocabulary; this track trains the *perceptual* skill — looking at a timeline and seeing the story: gaps, stragglers, serialization, starvation.

Everything here is GPU-framed: Nsight Systems, `torch.profiler`, Perfetto, CUDA streams, NCCL.

## The three tiers

1. **[The Vocabulary](/trace-reading/1-trace-reading-vocabulary/)** — what every row in a trace means, the fixed reading order (step boundary → gaps → overlap → longest kernel → zoom), and the canonical pathology signatures. Read this first; everything else assumes it.
2. **[Guided Walkthroughs](/trace-reading/2-guided-walkthroughs/)** — six case studies narrated the way an interviewer runs them, wrong hypotheses included, with the confirming measurement and the arithmetic for each fix.
3. **[Hands-On Labs](/trace-reading/3-hands-on-profiling/)** — capture your own traces: `torch.profiler` → Perfetto runs on a laptop CPU today; the `nsys`/`ncu` workflow is there for when you rent a GPU box. Companion notebook included.

**[▶ The interactive trace trainer](/tools/trace-drills.html)** — randomized synthetic Nsight-style timelines with an injected pathology (or none — healthy traces are in the rotation, and two scenarios are deliberate lookalikes of others, because recognizing a clean trace and telling twins apart is part of the skill). Two-minute clock, diagnosis-first, full reveal with signature, confirming measurement, and fix.

## How to use it

- Read tier 1 once, carefully. Then drill the trainer a few traces per day alongside the [mental-math drills](/optimization/15-mental-math-drills/) — same protocol: out loud, committed diagnosis *before* the reveal, under 2:00.
- Do one guided walkthrough per day and re-narrate it from the timeline alone the next day.
- Run Lab A this week so the Perfetto UI is muscle memory; do Labs B and C on a rented GPU box at least once before interview loops — "I ran nsys last week" lands very differently from "I've read about it".

The interview script the whole track builds toward: state the step time → quantify the gap → name the signature → name the one measurement that confirms it → give the fix and the expected win with arithmetic. Five sentences, spoken over a timeline. That's the rep.
