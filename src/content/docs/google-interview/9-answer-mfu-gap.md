---
title: "Worked Answer: MFU Gap Investigation"
description: "Staff-level model answer for 'expected 50% MFU, getting 25%': interrogate the metric first, the L0-L4 decomposition hierarchy, three stacked causes with reconciling arithmetic, real artifacts, and the incident-to-runbook process design."
---

# Worked Answer: MFU Gap Investigation ("Expected 50%, Getting 25%")

**▶ [Open the full model answer](/answers/mfu-gap-investigation.html)** — self-contained: the L0–L4 hierarchy with decision rules, before/after timelines, per-op roofline table, HLO remat diff, step-time strip chart, the 25→45% bridge chart, runbook flow, and the 60-minute run sheet.

The spine:

1. **Interrogate the metric before the system**: analytic-6ND numerator (traced FLOPs absolve remat), bf16 dense denominator, stalls included; triangulate via step time + tokens/s/chip + HBM BW; and *audit the expectation* — a geometry check shows "expected 50%" was ~3 points optimistic (head_dim 96 vs the comparable's 128).
2. **The frame**: 6·70e9·2.1M ÷ (256×918 TF) = 3.75 s ideal; observed 15.0 s = 25.0%. Every hypothesis cashes out in seconds against that line.
3. **L0–L4 hierarchy** with per-level tools, exit criteria, and descend-by-bucket-size decision rules — the part an L4 can run next time.
4. **Three stacked causes** (compound, not one villain): quant-library barrier exposing 3.6 s of collectives (+8.0 pts), remat-opaque custom op forcing FFN recompute (+7.1 pts), periodic checkpoint/input stalls (+4.9 pts) → 45.0% vs corrected 47% target; residual reconciled, not hand-waved.
5. **Incident → process**: runbook with time budgets and escalation boundaries, pre-launch geometry review (catches head_dim 96 at config time for free), derived-not-folklore expectations wired into the regression-detection system, and the ROI ledger (~3,400 chip-days recovered for ~3 engineer-weeks).
6. **When to stop**: the marginal MFU point priced against the engineer's best alternative use — park the Pallas kernel, write it down, move on.
