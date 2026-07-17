---
title: "Interactive Trace Drills"
description: "The randomized trace-reading trainer: synthetic Nsight-style timelines with injected pathologies, a two-minute clock, Perfetto export, and full diagnosis reveals — plus how to run the drill protocol."
sidebar:
  label: "Interactive Drills"
---

# Interactive Trace Drills

**▶ [Open the trace trainer](/tools/trace-drills.html)** · **▶ [Open the mental-math trainer](/tools/perf-drills.html)**

The trace trainer generates a randomized synthetic Nsight-style timeline — CPU/dataloader, CUDA API, GPU compute, and NCCL rows — with one of eleven scenarios injected: unoverlapped all-reduce, input starvation, launch-bound inference, straggler rank, memory-bound dominant kernel, sync stalls, pipeline bubbles, allocator stalls, a genuinely bandwidth-bound collective, recompilation gaps, or a perfectly healthy trace (weighted double, so you can't assume something is always wrong). Each scenario rotates through several framings — different hardware, different symptoms, sometimes a colleague's wrong theory baked into the prompt — so you learn the signature, not the story.

Two of the eleven are deliberate **lookalikes**: the bandwidth-bound collective renders almost identically to a straggler (the discriminator is arithmetic — bytes ÷ bus bandwidth — plus per-rank symmetry), and recompilation gaps mimic input starvation (the discriminator is intermittency: starvation hits every step, recompiles only on shape changes). Getting these right means you're reading the timeline, not pattern-matching the vibe.

## The protocol

1. **New trace.** Read it out loud in the fixed order from [the vocabulary](/trace-reading/1-trace-reading-vocabulary/): step boundary → gap analysis → overlap → longest kernel.
2. **Commit to a diagnosis before revealing** — say the signature, the one measurement that would confirm it, the fix, and the expected win with arithmetic. Under 2:00 or it's a miss.
3. **Reveal** and compare against all three sections: signature, confirming measurement, fix.

## Practicing in real Perfetto

Interviews may put you in front of the real tool, so drill the navigation too: **Export to Perfetto (.json)** downloads the current timeline as a Chrome-trace file — open it at [ui.perfetto.dev](https://ui.perfetto.dev) (drag the file in, or "Open trace file"). Then do the read in Perfetto itself: W/S to zoom, A/D to pan, drag-select a region to get exact durations for your gap analysis, click slices for details. Harder mode: have the trainer generate a trace, export it *without looking at the rendered timeline or scenario text*, and diagnose purely inside Perfetto.

For capturing traces of your own code (the real thing), see [Hands-On Profiling](/trace-reading/3-hands-on-profiling/).
