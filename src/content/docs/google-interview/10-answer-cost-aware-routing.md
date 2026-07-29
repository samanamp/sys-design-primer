---
title: "Worked Answer: Cost-Aware LLM Request Routing"
description: "Staff-level model answer for 'design smart, cost-aware routing of LLM requests' over a heterogeneous TPU (v5e/Trillium/Ironwood) + GPU (H100/B200) fleet: the $/token cost table, lever ranking, cascades, cache-aware P2C placement, and the feedback plane."
---

# Worked Answer: Smart, Cost-Aware Routing of LLM Requests

**▶ [Open the full model answer](/answers/cost-aware-routing.html)** — self-contained: the three-timescale decomposition diagram, the 3×4 prefill/decode $/Mtok cost table with live arithmetic, the occupancy sweep, cache-affinity break-even chart, KV-transfer pairing matrix, feedback control loop, escalation-storm anatomy, an 8-row pathology table, and the 60-minute run sheet.

The spine:

1. **Reframe first**: "routing" is a joint optimization at three timescales — variant/pool/replica per-request (≤1 ms, executes pre-computed policy only), pool control (minutes–hours), fleet planning (weeks). The fast path never optimizes; it reads tables.
2. **Cost model before architecture**: prefill (compute-bound) vs. decode (bandwidth-bound) $/token formulas, computed across {70B BF16, 70B FP8, 8B distilled} × {Trillium, Ironwood, H100, B200}, with one cell derived end-to-end and sanity-checked against the compute roofline.
3. **The lever ranking that drives everything**: occupancy (~8×, from the batch sweep on identical silicon) ≥ variant (~5×) ≫ hardware generation (~1.6×) — so the router is designed primarily as an *occupancy-shaping machine*, with hardware arbitrage as a sunk-fleet opportunity-cost argument.
4. **Variant selection**: upfront difficulty routing for class A, cascades for B/C, with the cascade economics ($3.0e-4 → $1.3e-4/req at 25% escalation, +270 ms TTFT tax) and escalation-rate drift monitored *in both directions* as a first-class SLI.
5. **Placement**: work-based (not request-count) load signals, hierarchical routing (global tables + cell-local cache-aware P2C), the cache-affinity coefficient derived from a break-even calculation per SLO class, disaggregated prefill/decode with a KV-transfer cost matrix (NVLink/ICI cheap, cross-DCN banned for long-context class A), and brownout as an explicit product-visible state machine.
6. **The feedback plane**: measured $/Mtok overrides the analytic table, every learned component ships with a dumb fallback + kill switch + counterfactual decision logging, and policy changes travel replay → shadow → staged rollout, promoted on Δ$ at equal SLO attainment.
7. **Proving the savings**: holdback cells as the honest headline number, runner-up logging and replay as attribution detail — always denominated at fixed SLO attainment.
