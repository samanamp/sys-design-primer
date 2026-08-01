# Prompt: Multimodal/Video Training Data Pipeline — Staff+ System Design Document

You are a Staff+ ML infrastructure engineer answering the following system design interview question, in written form, as a single self-contained HTML reference document. Write as a practitioner who has built this, not as a survey author or blogger. **Audience:** a staff-level ML systems engineer preparing for interviews — fluent in transformers, training systems, and GPU/TPU performance, but not a data-infrastructure specialist. Calibrate depth accordingly: never explain what attention is; always explain what a shard manifest is.

## The interview question

> Design the end-to-end data pipeline for training a large video world model / video generation model. Assume a raw corpus on the order of **100 PB of video with audio** (mixed sources: licensed catalogs, web-crawled, synthetic), plus **image and image-text corpora for joint training** (state your assumed proportions), a training cluster of **~10k accelerators**, and a target of training on the order of **10^12–10^13 video tokens** (or latent-frame equivalents). The pipeline must cover everything from raw bytes landing in object storage to tensors arriving on accelerator HBM, including all offline preprocessing, labeling, curation, storage, and the online loading path. Design it as a **continuously operating system** (new data keeps landing, filters and recipes evolve), not a one-shot dataset build.

## Before writing: research requirement

Do a genuine survey of **current (2025–2026) industry practice** before writing. Prioritize: technical reports and engineering blogs from teams shipping video/world models (e.g., video generation model reports, world-model papers, VLM data papers), data-infrastructure talks/papers, and open-source dataloading and preprocessing systems actually used at scale. Then, throughout the document:

- **Mark recency explicitly.** Tag claims as one of: `[established practice]` (stable for years), `[current practice 2025–26]` (what frontier teams do now, with source), or `[emerging/contested]` (proposed or partially adopted, note who claims it). If something I might know from ~2023 has been superseded (e.g., older captioning approaches, older dataset-scale assumptions, older codec/decode tradeoffs), say so explicitly: "this used to be done via X; current systems do Y because Z."
- Cite the source (team/paper/system name) inline for every `[current practice]` claim. If you can't find a current source, say the practice is inferred and mark it as such — do not silently present training-data-era knowledge as current.
- **Enforcement:** every stage section must contain at least one tagged claim. Do not let tagging decay after the first sections — it applies uniformly through the final section.

## Scope: complete design, connected

Cover the **full pipeline** as one coherent system. Every section must state what it receives from the previous stage and what contract it exposes to the next (formats, granularity, throughput, metadata). Required stages, in order:

1. **Ingestion & raw storage** — source tiers (licensed / crawled / synthetic), object storage layout, metadata catalog, dedup at ingest (perceptual hashing for video), legal/provenance tagging, and **held-out eval set carving at ingest time** (why decontamination requires reserving eval material before dedup/filtering ever runs).
2. **Preprocessing fleet (offline)** — decode strategy (codecs, hardware decode vs CPU), **robustness to corrupt/adversarial files** (decoder crashes, quarantine, retry budgets at 100 PB scale), scene detection & splitting, resampling (fps/resolution ladders — designed to serve **progressive-resolution curricula**, i.e., the same clip at multiple resolutions across training phases), **audio path** (keep/drop decision, audio codec/encoding, sync with visual stream), quality filtering (aesthetic, motion, OCR/text-overlay, NSFW), and the flagship decision: **offline VAE/latent encoding vs decode-on-the-fly** — analyze this as a cost problem with numbers, including re-encoding risk when the VAE or video tokenizer changes.
3. **Labeling & captioning as an embedded inference system** — VLM captioning fleet sizing, caption density/recaptioning strategies, structured metadata extraction (camera motion, object tracks if used), synthetic data generation loops, quality control on labels.
4. **Curation & mixture management** — dedup at semantic level (embedding clustering), quality-tier bucketing, mixture weights across sources **and modalities (video vs image vs image-text co-training proportions)**, data recipe versioning and reproducibility, filtering-model feedback loops.
5. **Training-ready storage & sharding** — shard formats (and why: seekability, sample size variance), bucketing by resolution/duration, shuffle strategy at shard vs sample granularity, storage tier choice and read-bandwidth math.
6. **Online loading path** — host-side dataloader design, prefetch depth, decode/transform placement (CPU vs accelerator), determinism & resumability under elastic restarts and reshards, multi-epoch and mixture-sampling correctness, and the **throughput proof**: bytes/step × steps/sec vs storage and host budgets at 10k accelerators, showing the pipeline is not the bottleneck.
7. **Lifecycle: continuous operation & versioning** — incremental processing of newly landed data, **dataset snapshots as frozen manifests** (what "dataset v3" means physically), backfill strategy when a filter or the tokenizer changes (recompute everything vs lazy migration, and the cost of each), reproducing a months-old training run.
8. **Observability & data quality in production** — per-stage throughput/cost dashboards, sample-level lineage, contamination/eval-leakage checks against the reserved eval sets, how a bad-data incident is detected and rolled back mid-training-run.

## Cost as a first-class axis

For each major stage, give order-of-magnitude cost estimates (compute-hours, storage $/PB/month, egress) and identify the top 2–3 cost drivers of the whole pipeline. Explicitly compare: preprocessing fleet cost vs training cluster cost; store-latents vs re-decode; caption-everything vs caption-on-demand. Two costs that must not be omitted: (a) **total storage footprint including all derived artifacts** — raw + intermediates + latents + captions/metadata + multiple resolution ladders (state the multiplication factor over raw); (b) **placement/colocation** — where the preprocessing fleet, object storage, and training cluster physically sit, and what cross-region movement of PB-scale data costs; treat egress as a first-class line item, not a footnote. Show the arithmetic, state assumptions, use round numbers.

## Depth bar

- Every design choice must include: the alternative(s), why this one, and what breaks if the scale changes 10×. Never just name a technique — explain its mechanism in 2–4 sentences and give the number that justifies it.
- Include worked napkin math wherever a claim depends on scale (decode FLOPs/throughput per stream, latent compression ratios, shard sizes, aggregate read bandwidth, captioning fleet size).
- Anticipate and answer **at least** 6 of the hardest interviewer follow-ups inline in a dedicated section. Examples of the caliber expected (do not limit yourself to these): "what if the VAE changes mid-project," "how do you shuffle 100 PB," "why not train from raw pixels," "how do you keep eval sets uncontaminated," "what changes for interactive world models with action labels," "a loss spike is traced to data — walk me through the investigation."
- For the **3–4 highest-leverage design decisions** in the document (your pick), append a boxed **"defend this in 60 seconds"** paragraph: the compressed spoken version of the argument, with its one key number — exactly what a candidate would say aloud at a whiteboard.

## Format

- Single self-contained HTML file, desktop-width, clean typography, no external dependencies.
- **Diagram-dense**: an SVG diagram for every stage plus one end-to-end architecture diagram with data volumes annotated on the edges (PB → TB → GB/step). Diagrams must carry information (numbers, formats, fan-in/fan-out), not decorate.
- Tables for every comparison (codec choices, shard formats, storage tiers, cost breakdowns).
- No filler, no motivational intros, no "in conclusion" padding, no restating the question. Every paragraph must contain a decision, a number, or a mechanism.
- Length: whatever completeness requires; cut fluff, not content. **If output limits force cuts, cut in this order: prose elaboration → repeated examples → never the diagrams, tables, napkin math, follow-ups, or 60-second defenses.** Do not silently thin the later sections; the online loading path and follow-ups matter as much as the early stages.
