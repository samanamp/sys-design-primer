---
title: "Worked Answer: Fleet-Wide Perf Regression Detection"
description: "A full staff-level model answer with arithmetically consistent artifacts: capacity math, tiered benchmarking, changepoint statistics with FDR control, automated bisection, and a worked XLA fusion-split regression with before/after traces."
---

# Worked Answer: Fleet-Wide ML Performance Regression Detection

**▶ [Open the full model answer](/answers/perf-regression-detection.html)** — a self-contained long-form document (inline SVG diagrams, Perfetto-style before/after traces, alert JSON, bisection log, HLO diff, staff-signal callouts per section, and a 60-minute delivery run sheet).

The question: design a system that catches perf regressions (TTFT/TPOT/goodput/MFU/cost-per-token) across compiler releases and hardware generations, attributes them to root cause, and scales to hundreds of models × platforms × weekly releases.

The spine of the answer, for review without opening the full document:

1. **Capacity math first**: the naive matrix is ~72,000 chip-hours/week — unaffordable — which *forces* the tiered design (~6% of naive at ≥95% spend-weighted coverage).
2. **The detector needs its own SLOs**: 24 h verdict, ≤5 false pages/week, 1% MDE on the top-20 (because 1% of fleet ≈ $10M/yr).
3. **Benchmark honesty**: exact metric definitions, the three ways MFU accounting lies (remat FLOPs, causal ½, MoE total-vs-active), geometry-matched vs production-shaped strata, open-loop load generation.
4. **Statistics, not vibes**: variance tables (TPU σ≈0.4% via determinism vs GPU σ≈2%), MAD noise floors, changepoint detection + BH-FDR, and the power formula that derives repeat counts.
5. **Attribution as a two-customer product**: cost-model-filtered bisection with verification, per-op/HLO diffs, and a fully worked example — an XLA VMEM-cap CL splits a fused attention kernel, +8.2% decode TPOT on Trillium only, with the roofline arithmetic explaining the platform- and phase-selectivity.
6. **Org mechanics**: dollar-threshold gating, pre-staged rollback, a triage rotation with a false-page KPI — alert fatigue treated as a system-design failure.
