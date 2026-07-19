---
title: "torch.compile Primer for Perf Engineers"
description: "Concept-level torch.compile for design interviews: Dynamo/guards/graph breaks, dynamic shapes and the TPU collision, Inductor fusion via roofline, the openxla path vs JAX, production piecewise compilation, an 8-entry failure bestiary, and ten sayable statements."
---

# torch.compile Primer for ML Performance Engineers

**▶ [Open the full primer](/answers/torch-compile-primer.html)** — self-contained long-form document: full-stack pipeline diagram (GPU and TPU paths side by side), graph-break and recompilation-storm diagrams, the compiler-first decision tree, an 8-entry failure-mode bestiary keyed to diagnosis commands, and a 10-statement interview deployment sheet.

The spine, for quick review:

1. **The stack**: Dynamo (bytecode → FX graph + guards) → AOTAutograd (joint fw/bw, functionalized ATen) → Inductor→Triton (GPU) or openxla→StableHLO→XLA (TPU). Every production pathology attributes to exactly one stage.
2. **Graph breaks are structural**: fusion and cudagraphs can't cross them — serving runs `fullgraph=True` so breaks fail the build instead of silently costing latency.
3. **Dynamic shapes**: symbolic after first change, but production serving pre-compiles a bucket lattice anyway — and on TPU bucketing is the programming model, not an optimization (whole-program static-shape XLA compiles).
4. **Fusion = roofline arithmetic**: unfused elementwise chains run at AI≈0.25; fusing N ops divides HBM bytes by N. Check `output_code.py` kernel counts before believing any speedup claim.
5. **Production pattern**: piecewise compilation with attention outside as a custom op, cudagraphs per bucket, warm-up over the lattice, cache artifacts shipped in the image (`save_cache_artifacts`).
6. **Compiler-first, custom-kernel-last**: profile → roofline-place → inspect codegen → fix capture → only then hand-write, because hand kernels are re-owned at every compiler/hardware rev.

Version-anchored to the PyTorch 2.10–2.12 era (Jan 2026 release, two-month cadence).
