---
title: "Text Pretraining Data Pipeline"
description: ""
---

**Question:** design the end-to-end data pipeline for pretraining a frontier LLM. ~30 PB compressed web crawl + licensed corpora + code + academic/books + synthetic generation → 10–50T training tokens → 10k–100k accelerators. Continuously operating, not a one-shot build.

**Format note:** the source prompt asked for self-contained HTML with SVG diagrams. This is Markdown per the delivery request, so diagrams are ASCII/box figures inside fenced blocks. They still carry the numbers, formats, and fan-in/fan-out — nothing was dropped, only re-rendered. Tables, napkin math, follow-ups, and the 60-second defenses are all present.

**Recency tags used throughout:**

| Tag | Meaning |
|---|---|
| `[established]` | Stable practice for 3+ years, uncontroversial |
| `[current 2025–26]` | What teams do now; source named inline |
| `[emerging/contested]` | Proposed or partially adopted; who claims it is named |
| `[inferred]` | No public source found; this is my reasoning, flagged as such |

**Open vs frontier caveat, stated once and assumed everywhere:** the open-data community (FineWeb / DCLM / Nemotron-CC / Dolma / OLMo) documents its methods completely; frontier labs publish data sections that are one to three paragraphs of prose with no ablations. Where I describe a mechanism in detail, it is almost always the open version. I mark the places where I believe frontier practice diverges and label it `[inferred]`.

---

## 0. Global assumptions and the reference funnel

Every number in this document derives from this one set of assumptions. Change these and re-derive; do not mix them with numbers from a different assumption set.

| Quantity | Value | Note |
|---|---|---|
| Raw crawl, compressed WARC | 30 PB | ≈100 PB uncompressed; includes CC archive + own crawl, ~4 years accumulated |
| Raw fetch records | 1.0 × 10¹² | Recrawls dominate: same URL fetched many times |
| Unique URL-content pairs after recrawl collapse | 150 × 10⁹ | ~6.7 fetches per surviving document |
| Bytes/char/token constants | 4.3 chars/token; 1 token ≈ 4.3 B UTF-8 English | BPE, 128k–256k vocab |
| Training run | 100k accelerators × 30 days | ≈7.2 × 10⁷ accelerator-hours ≈ $150M at $2/hr |
| Target training budget | 30T presented tokens | From ~26T unique tokens + selective repetition |
| CPU spot cost | $0.02 / core-hour | |
| GPU cost | $2.00 / GPU-hour | H100-class |
| Object storage | $20 / TB-month | |
| Cross-region egress | $20 / TB | The single most avoidable line item |

### End-to-end yield funnel

```
                            RAW ACQUISITION
  ┌───────────────────────────────────────────────────────────────────────┐
  │ own crawl + CommonCrawl   licensed   code repos   academic/books      │
  │   30 PB comp. WARC         ~40 TB     ~20 TB        ~15 TB            │
  │   1.0e12 fetch records      text       tarballs      PDF/LaTeX        │
  └───────────────┬───────────────────────────────────────────────────────┘
                  │  URL + content-hash collapse of recrawls   (-85% records)
                  ▼
        150e9 unique docs / 20 PB uncompressed HTML
                  │  §2 EXTRACTION (trafilatura-class) + langID + encoding
                  │  HTML→text keeps ~7.5% of bytes
                  ▼
        150e9 docs / 1.5 PB plain text          ◄── contract: JSONL+zstd,
                  │                                  1 doc = 1 record, provenance
                  │  langID keep target languages (-40% docs)
                  ▼
         90e9 docs /  0.90 PB
                  │  §3 EXACT dedup (doc SHA-256 + line-level)   -25% docs
                  ▼
         68e9 docs /  0.65 PB
                  │  §3 MINHASH-LSH near-dedup (14×8)            -50% docs
                  ▼
         34e9 docs /  0.34 PB           ◄── 40 TB of signatures shuffled here
                  │  §4 heuristic filters (Gopher/C4 rules)      -20% docs
                  ▼
         27e9 docs /  0.27 PB
                  │  §4 model quality classifier, keep top ~35%
                  ▼
        9.5e9 docs /   95 TB  ≈  22.0T web tokens
                  │  §4 safety/PII  -0.5%   §5 decontamination  -1.0%
                  ▼
                        21.5T web tokens
                              │
      ┌───────────────────────┼──────────────────────────────────┐
      │        │              │            │            │        │
   code 1.5T  math/sci 0.3T  books/lic 0.5T  multiling.  synthetic 2.0T
      │        │              │        (inside 21.5T)      │
      └───────────────────────┴──────────────────────────────────┘
                              ▼
                  ~25.8T UNIQUE TOKENS in the pool
                              │  §7 mixture + multi-epoch policy
                              ▼
                  30T PRESENTED TOKENS (1.16 avg epochs, 4× on best 5%)
                              │  §8 tokenize, pack, shard  → 120 TB uint32
                              ▼
                  §9 loader: 11.5M tok/s aggregate = 46 MB/s on the wire
```

**Read the funnel twice.** Post-extraction retention is 95 TB / 1.5 PB = **6.3%**, consistent with published FineWeb-Edu-style full-pipeline pass rates of 5–7% `[current 2025–26]` (Spheron 2026 curation guide; consistent with FineWeb's own reported per-stage yields). The much scarier-looking 0.3% figure against raw compressed WARC is almost entirely recrawl collapse, not curation.

### Where the money goes

| Cost bucket | Order of magnitude / run | % of $150M training run | Notes |
|---|---|---|---|
| **Storage (all artifacts, 12 mo)** | **$8.4M/yr** | 5.6% | 35 PB × $20/TB-mo. #1 recurring cost |
| **CPU extraction + normalization** | **$300K** | 0.2% | 11M core-hours, 4.6 days on 100k cores |
| Dedup (MinHash + shuffle) | $60K | 0.04% | 1.25M core-hours + 50 TB distributed sort |
| Encoder quality classifier (150M) | $10K | 0.007% | 3.2k GPU-hours. Effectively free |
| LLM-as-judge classifier (8B, if used) | $400K | 0.27% | 190k GPU-hours |
| **Synthetic generation (8B gen, 2T tok)** | **$500K** | 0.33% | 240k GPU-hours |
| Synthetic generation (70B teacher, 2T tok) | $5M | 3.3% | The version that actually hurts |
| Tokenization | $200 | ~0 | Genuinely negligible; the *rewrite* isn't |
| Cross-region egress (if misplaced) | $600K–$2M | 0.4–1.3% | Entirely avoidable; see §12 |
| **Crawl fleet (own crawler, annualized)** | **$3–6M/yr** | 2–4% | See §1 build-vs-buy |
| Mixture ablation compute | $1–3M | 0.7–2% | See §7; the best-spent money here |

**Top three cost drivers: storage footprint, the crawl fleet, and mixture-ablation compute.** Note what is *not* on that list: the actual data processing. Extraction, dedup, filtering, and tokenization together are under $1M against a $150M training run. This is the single most counterintuitive fact about text pipelines and it drives most of the design: **compute spent on curation is nearly free relative to training, so spend it aggressively; storage and human iteration time are the real constraints.**

---

## 1. Acquisition & raw storage

**Receives:** nothing (system boundary). **Exposes to §2:** immutable WARC/WAT/WET-equivalent objects in a partitioned object store, plus a per-record provenance row in a metadata table.

### 1.1 Source tiers have different trust, legal handling, and reprocessing rights

| Tier | Volume | Trust | Legal handling | Reprocess freely? |
|---|---|---|---|---|
| Own crawl | 25 PB comp. | Low (adversarial) | robots/opt-out at fetch time; jurisdiction rules | Yes |
| CommonCrawl | 5 PB comp. | Low | Inherits CC's robots compliance; contested (§1.4) | Yes |
| Licensed corpora | ~40 TB | High | Per-contract: term limits, model-scope limits, deletion clauses | **No** — must honor expiry |
| Code repos | ~20 TB | Medium | Per-repo SPDX license; copyleft policy decision | Yes, within license |
| Academic / books | ~15 TB | High | Mixed: arXiv OK, books often licensed or public-domain-only | Depends |
| Synthetic | generated | Medium | Generator model's ToS; provenance of *source* doc propagates | Yes |

The licensing tier is the one that breaks naive pipeline design. A licensed corpus with a 3-year term and a "models trained before expiry may be retained, no new training after" clause means **your dataset manifests must be able to exclude a source retroactively and prove which past runs contained it.** That requirement alone forces per-document provenance to survive all the way into the shard manifests (§8), which is a design constraint most teams discover late.

### 1.2 Storage layout

```
s3://corpus-raw/
  crawl/own/          dt=2026-07-15/host_bucket=0037/part-00412.warc.zst
  crawl/cc/           snapshot=CC-MAIN-2026-26/segment=00891/...
  licensed/           vendor=acme/contract=A-2024-11/v=1/...
  code/               forge=github/repo_bucket=0a3f/repo=owner__name.tar.zst
  academic/           source=arxiv/yymm=2607/...
s3://corpus-meta/
  provenance/         Iceberg table, 1 row per fetch record
  evalholdout/        WRITE-ONCE, separate account, separate credentials
```

**Partitioning choice:** partition raw crawl by `(date, hash(host) % 4096)`. Alternatives were date-only and host-only.
- Date-only: fine for ingest, catastrophic for dedup and for host-level policy actions (you must scan everything to purge one domain).
- Host-only: gives you host locality for free but produces massive skew — the top 100 hosts are ~5% of the crawl, so a few partitions become 100× the size of the median.
- Hash-bucketed host with a date prefix gives you (a) incremental daily processing, (b) bounded partition skew, (c) host-local grouping inside a bucket for host-level dedup and policy.

**At 10× scale (300 PB):** the partition count is fine; what breaks is the metadata table. 1 row per fetch record at 10¹³ records is a 10-trillion-row Iceberg table. At that point you move provenance to a two-level scheme: a compact per-document row (doc_id → source_id, url_hash, ts, license_id) and a separate append-only fetch log that is queryable but not joined in the hot path.

### 1.3 Provenance record — the contract everything downstream depends on

```
doc_id          uint128   content-addressed: blake3(normalized_text)
url             string
url_hash        uint64
host_id         uint32    → host dimension table (policy, robots state, quality prior)
fetch_ts        timestamp
snapshot_id     string
source_tier     enum      {own_crawl, cc, licensed, code, academic, synthetic}
license_id      uint32    → license table (SPDX or contract ID, expiry, scope)
robots_state    enum      {allowed, disallowed_at_fetch, unknown, retro_optout}
lang            char[3] + confidence
content_type    enum      {html, pdf, latex, code, plaintext}
derived_from    uint128?  non-null for synthetic: points at the source doc
```

`derived_from` is the field people forget. Without it, synthetic data breaks decontamination (a rephrase of a contaminated doc is still contaminated) and breaks license expiry (a rephrase of a licensed doc inherits the contract). `[inferred]` — I have not seen a public dataset that documents this field, but the failure mode is mechanical and I'd expect any lab that has run synthetic generation for two model generations to have it.

### 1.4 Crawler build vs buy

`[current 2025–26]` This has moved from a cost question to a legal and control question. Three things changed since 2023:

1. **Blocking rose sharply.** Reputable sites blocking AI crawlers went from ~23% (Sept 2023) to roughly 60% (May 2025), with the most defensive sites disallowing 15+ distinct AI user agents (Digital Applied / Playwire tracking, 2026).
2. **Infrastructure flipped the default.** On 1 July 2025 Cloudflare made AI scraping opt-in — block by default — and reported over a million customers subsequently choosing to block AI crawlers.
3. **CommonCrawl itself became contested.** On 29 April 2026 the News/Media Alliance sent a formal demand letter to CommonCrawl demanding removal of publisher content, terms explicitly prohibiting AI-training use, and enforceable opt-out; signatories included NBCUniversal, CNN, Vox Media, Ziff Davis, and USA Today.

That last point is the design-relevant one: **CommonCrawl is no longer a stable, legally inert dependency.** A dataset built on CC snapshots may need retroactive document removal on a timescale you do not control.

| | Own crawl fleet | CommonCrawl only | Hybrid (recommended) |
|---|---|---|---|
| Freshness | Hours–days | 4–8 week snapshot cadence | Own for fresh, CC for archive |
| Coverage control | You choose seeds, depth, recrawl policy | Fixed, breadth-first, shallow | Both |
| Opt-out control | You honor robots at fetch, defensible record | Inherit CC's; retroactive risk | Own crawl is your defensible record |
| Historical depth | Only since you started | 2008→present | CC is irreplaceable for pre-2023 web |
| Cost | $3–6M/yr (below) | ~$0 + egress | $3–6M/yr |
| Deduped against | Your own state | Nothing | Must dedup CC against own crawl |

**Fleet cost arithmetic.** Target 5B fetches/day (a CC-snapshot-equivalent every 12 hours). At 20 fetches/sec/worker (politeness-limited, not bandwidth-limited), you need 5e9/86400/20 ≈ **2,900 concurrent workers**. At 4 vCPU each on spot: 11,600 vCPU × 24 × 365 × $0.02 = **$2.0M/yr compute**. Bandwidth: 5e9 × 60 KB = 300 TB/day ingress. Ingress is typically free; storage of the delta after dedup is the cost. Add DNS, IP reputation management (residential/datacenter IP pools, CAPTCHA and WAF handling), and a small team → **$3–6M/yr all-in**. That is 2–4% of a single training run, and it buys you freshness, seed control, and a defensible robots-compliance audit trail.

**Recommendation and what breaks at 10×:** build the fleet, keep CC for historical depth. At 10× (50B fetches/day) politeness limits, not compute, bind — you cannot ethically hit a single host 10× harder, so scale comes from breadth (more hosts) not depth, and the marginal host is much lower quality. This is a hard ceiling: the crawl budget stops converting into useful tokens well before the compute budget runs out.

### 1.5 Robots / opt-out as a pipeline filter, not just a fetch-time check

`[current 2025–26]` Three enforcement points, because one is insufficient:

1. **Fetch time** — respect `robots.txt` for your training user-agent. Note that major providers now split training from search/answer agents (GPTBot vs OAI-SearchBot; ClaudeBot; Google-Extended is a *directive token*, not a crawler, and never appears in server logs). Your fleet needs a documented training user-agent so sites can opt out specifically.
2. **Ingest time** — a nightly job re-reads robots.txt for every host in the corpus and updates `host.robots_state`. Sites that newly disallow are marked `retro_optout`.
3. **Manifest time** — the mixture builder excludes any doc whose host is `retro_optout` at build time. This is the mechanism that makes retroactive opt-out cheap: you never delete data, you exclude it from future manifests, and old manifests remain reproducible with an audit note.

The reason this is a three-point design rather than "just don't fetch it": robots state changes after you fetch. Roughly 12.9% of bots ignored robots.txt entirely in Q1 2025 (up from 3.3%), which is exactly why being able to *prove* your compliance posture through manifest lineage matters more than the fetch-time check alone.

### 1.6 Held-out eval carving at ingest — do this before anything else touches the data

**Why it must be first.** Decontamination (§5) compares training data against eval sets. If your eval sets are themselves drawn from the corpus *after* dedup and filtering, three things go wrong:
- Dedup has already collapsed near-duplicates of eval items into single documents, so you cannot tell which surviving doc is the eval item's twin.
- Quality filtering has a selection effect: your held-out set is now conditioned on passing your own filter, so held-out loss is optimistically biased and stops tracking real-world text.
- You have no clean record of what the eval item *was* before normalization, so fuzzy matching (§5) has nothing stable to match against.

**Mechanism.** At ingest, before extraction:
- Copy all public benchmark corpora (MMLU, GSM8K, HumanEval, GPQA, RULER, the long-context suites, and your internal evals) into `s3://corpus-meta/evalholdout/` in a **separate cloud account with separate credentials and write-once (object-lock) semantics.** The separate account is not paranoia theater — it is the only structural protection against a future pipeline engineer "helpfully" adding the eval directory to a glob.
- Additionally carve a **random 0.1% of raw crawl by `hash(url_hash) % 1000 == 0`** as a perpetual held-out web sample. Hash-based, not date-based, so it stays stable as new snapshots land and gives you a distribution-matched loss probe forever.
- Also carve a **time-forward slice**: everything with `fetch_ts > T_cutoff` for a rolling T. This is your only defense against contamination-by-construction on benchmarks published after your cutoff.

**Contract to §2:** eval holdout is written and its manifest hash recorded before any extraction job is allowed to start. The extraction job's config references that hash. If it does not, the job does not launch.

---

## 2. Extraction & normalization

**Receives:** raw WARC objects + provenance rows. **Exposes to §3:** one JSONL-per-shard, zstd level 6, one document per line, schema `{doc_id, text, lang, lang_conf, provenance_ref, extract_meta}`. Target shard size 512 MB compressed (≈2 GB text, ≈200k docs). Throughput contract: **250k docs/sec sustained** across the fleet.

### 2.1 The extractor choice is a model-quality decision, not a plumbing decision

`[established, quantified 2024]` This is the finding that reframed the stage. FineWeb's ablations showed that replacing CommonCrawl's own WET (raw text dump) with trafilatura-based extraction from WARC produced a **measurably better model at equal token count** — the WET files retain navigation, footers, and cookie banners that survive downstream filters and consume training budget. The practical rule: **never train on WET; always re-extract from WARC.**

| Extractor | Throughput (docs/s/core) | Boilerplate removal | Structure preserved | Verdict |
|---|---|---|---|---|
| CC WET (no-op) | ∞ | None | None | Never. Costs model quality. |
| `resiliparse` | ~120 | Good | Minimal | Best throughput/quality point; DCLM's choice |
| `trafilatura` | ~25 | Very good | Headings, lists, tables | FineWeb's choice; the quality reference |
| `justext` | ~60 | Moderate | Little | Superseded |
| LLM-based rewrite | ~0.05 (GPU) | Excellent | Excellent | Only for math/PDF (§2.4) |

**Choice:** trafilatura for the general path, accepting ~5× the CPU of resiliparse. The arithmetic says you can afford it: 1e12 records at 25 docs/s/core = 4e10 core-seconds = **11.1M core-hours = $222K**. On a 100k-core fleet that is **4.6 days wall clock.** Resiliparse would be $46K and 1 day. The $176K delta is 0.1% of the training run and buys a documented downstream quality gain — take it.

`[emerging/contested]` Several 2025–26 pipelines use a **two-pass extractor**: cheap resiliparse on everything, then trafilatura (or an LLM cleanup pass) only on documents that a cheap classifier flags as likely-high-value. This gets most of the quality at 20% of the CPU. NVIDIA's Nemotron-CC-Math pipeline is the clearest public instance of the expensive-second-pass idea, using layout-aware rendering (lynx) plus a targeted LLM cleaning stage for math specifically. Whether it's worth it for general web is unresolved publicly.

**What breaks at 10×:** nothing in the algorithm; the fleet just gets 10× bigger and the wall clock is what you defend. At 300 PB, 4.6 days becomes 46 days on the same fleet, which is longer than your snapshot cadence — you are no longer keeping up with ingest. The fix is the two-pass extractor, not more cores, because the cheap pass discards 60%+ before the expensive one runs.

### 2.2 Language identification and the multilingual fork

`[established]` fastText `lid.176` at ~3,000 docs/s/core; cost is noise (30 core-hours for the whole corpus). `[current 2025–26]` GlotLID and OpenLID extend coverage to 1,600+ languages with better low-resource behavior; FineWeb2 uses this class of model to build per-language pipelines.

The design decision is the **confidence threshold**, and it differs by language:
- English at conf > 0.65: high recall, and the downstream quality filter cleans up misfires.
- Low-resource languages at conf > 0.65: catastrophic. A 0.65-confidence "Tigrinya" document is usually a code page, a spam page, or Amharic. For languages under ~1B tokens in the corpus, use conf > 0.90 **and** require a script-consistency check (fraction of characters in the expected Unicode block > 0.8).

Documents that fail langID entirely (conf < threshold for all languages) go to quarantine, not to /dev/null — they are disproportionately code, math, tabular data, and multilingual mixtures, all of which have their own high-value pipelines.

### 2.3 Robustness: quarantine and retry budgets

Adversarial and corrupt documents are not an edge case at 10¹² records; they are a daily occurrence.

| Failure | Detection | Action |
|---|---|---|
| Zip bomb / 2 GB HTML | Size cap 8 MB pre-parse | Truncate + tag, don't drop (some are legit dumps) |
| Parser hang (pathological nesting) | 5-second per-doc watchdog | Kill, quarantine, continue shard |
| Invalid UTF-8 | `ftfy` mojibake repair, then validate | Repair; drop if >2% replacement chars |
| Crawler trap / tarpit (infinite link maze) | Host-level: >10⁶ URLs, <0.01 unique-content ratio | Blocklist host, purge from future manifests |
| Deliberate poisoning (hidden text, prompt injection) | Zero-width char density, `display:none` text extraction | Strip invisible content at extraction; density > 5% → quarantine |

**Retry budget:** 2 retries per shard on transient errors, then the shard is marked `partial` with a per-doc failure list. A single poison document must never be able to fail a 200k-document shard — the per-doc watchdog exists specifically to make shard-level progress independent of document-level failures. `[inferred]` — this is standard batch-systems practice rather than a documented LLM-pipeline practice, but the alternative (shard-level atomicity) demonstrably does not survive adversarial web input.

**Poisoning is a real and growing threat vector.** Some publishers deploy tarpits and honeypots explicitly to waste crawler resources and pollute training corpora (documented practice as of 2026). Treat host-level anomaly detection as a security control, not a hygiene control.

### 2.4 PDF and math paths

`[current 2025–26]` This is where the biggest measurable delta of 2025 landed. Two named systems:

- **olmOCR / Dolma 3** (Ai2, Nov 2025): Olmo 3 is pretrained on Dolma 3, a ~9.3T-token corpus that explicitly includes **science PDFs processed with olmOCR** as a first-class source alongside web, code, and math. Before this, PDFs were mostly discarded or handled with `pdftotext`, which destroys equation and table structure.
- **Nemotron-CC-Math** (NVIDIA, 2025): recovers math across MathJax, KaTeX, and MathML by layout-aware rendering with `lynx` plus a targeted LLM cleaning stage that standardizes to LaTeX. Result: **Nemotron-CC-Math-4+ at 52B tokens is 5.5× larger than FineMath-4+**, the previous best open math corpus, and yields +4.8 to +12.6 on MATH and +4.6 to +14.3 on MBPP+ when pretraining an 8B model.

The generalizable lesson: **for structured content, extraction quality is worth GPU dollars.** An LLM cleanup pass over 100M math documents at 50 docs/s/GPU is 550 GPU-hours = $1,100. The measured downstream gain is double-digit points on MATH. This is the best-ROI line item anywhere in the pipeline and it is a 2025 discovery — a 2023-era pipeline would have thrown these documents away.

### 2.5 Code as its own sub-pipeline

Code is not "text with different tokens." Four things differ:

**(a) Repo-level grouping, not file-level.** A single `.py` file is meaningless out of context. The unit of processing is the repository: files concatenated in a deterministic order (README first, then by dependency-ish heuristic or path sort), with path headers as separators. `[established]` — pioneered by CodeLlama, which concatenated code from the same project to build ultra-long training sequences.

**Why repo-level dedup rather than file-level, concretely:** file-level dedup on GitHub removes `LICENSE`, `__init__.py`, `setup.py`, and every vendored dependency copy — which sounds right until you notice it also destroys the *relationship* between files in a forked repo. Two forks of the same project share 95% of files. File-level dedup keeps one copy of each shared file and both copies of the diverging files, producing a Frankenstein corpus where no repo is coherent. Repo-level dedup (MinHash over the set of file content hashes, threshold ~0.7 Jaccard) keeps whichever fork has the higher signal (stars, recency, test coverage) and drops the other **entirely**, preserving coherence. Cost: you keep more total bytes; benefit: every retained repo is a valid, complete, long-context training document.

**(b) License filtering.** Parse SPDX from `LICENSE`, `package.json`, `Cargo.toml`, `pyproject.toml`, and file headers. Policy decision:

| Policy | Tokens available | Legal exposure | Who does this |
|---|---|---|---|
| Permissive only (MIT/Apache/BSD/ISC) | ~1.0T | Lowest | The Stack v2 default set; most open efforts |
| Permissive + weak copyleft (LGPL/MPL) | ~1.2T | Moderate | Some open efforts |
| All OSI licenses incl. GPL/AGPL | ~1.5T | Contested | `[emerging/contested]` — no major open corpus does this; frontier practice undisclosed |
| No license file present | +0.8T | Highest (default = all rights reserved) | Not recommended |

**Recommendation: permissive-only for the base mix, with an explicit `license_class` tag so the policy is a manifest-time decision, not a pipeline-time one.** If legal posture changes, you rebuild manifests, not the corpus. This is worth 0.5T tokens of optionality for essentially zero storage cost.

**(c) Code-specific quality signals.** Popularity heuristics (stars, forks) are cheap and weakly correlated with quality; they also have a severe head/tail problem (99% of repos have <5 stars).

| Signal | Cost | Signal quality | Use |
|---|---|---|---|
| Stars/forks | Free (API metadata) | Weak, biased to JS/web | Tie-break only |
| Lint/parse success (tree-sitter) | 500 files/s/core | Strong negative filter | **Yes** — reject unparseable |
| Test presence + ratio | Cheap (path patterns) | Moderate positive | Yes, as a feature |
| Actual execution in sandbox | ~$0.001/repo, 30s | Strongest | Only on a 1% sample to calibrate cheaper signals |
| LLM-judged quality | $0.0002/file | Strong | Yes, on the permissive subset |

`[current 2025–26]` The practical stack is: tree-sitter parse as a hard gate (removes ~8% of files that are minified, generated, or corrupt), then an LLM or encoder classifier for quality scoring, with execution-based filtering used only to *calibrate* the classifier because it doesn't scale. Full execution-based filtering of the corpus is `[emerging/contested]` — attractive in principle, and I have not seen a public corpus built this way at scale.

**(d) Generated and vendored code must die.** `node_modules/`, `vendor/`, `.min.js`, `*_pb2.py`, `*.generated.*`, lockfiles. These are ~30% of raw GitHub bytes and near-zero training value; they also poison dedup statistics because they are the most-duplicated content on the platform. Path-pattern rejection before hashing.

---
## 3. Deduplication

**Receives from §2:** 90e9 docs / 0.90 PB of language-tagged JSONL. **Exposes to §4:** 34e9 docs / 0.34 PB, same schema plus `{dup_cluster_id, dup_cluster_size}` — the cluster size is retained deliberately because it is a *quality signal* (a document duplicated 10,000 times is either canonical reference text or template spam, and the classifier in §4 can learn which).

### 3.1 Dedup runs before quality filtering. This ordering is not negotiable.

If you filter first and dedup second, you keep duplicates that happen to score well — and you have systematically over-amplified high-quality boilerplate (Wikipedia infobox templates, Stack Overflow answer scaffolding, license headers, cookie-consent text that survived extraction) across the entire corpus. `[established]` The reverse order is the standard FineWeb / RefinedWeb / DCLM recipe and the reasoning is documented in all three.

There is a real cost: you run the expensive dedup shuffle on 90B documents instead of on the 27B that would survive filtering. That is roughly 3× the shuffle volume. Pay it.

### 3.2 The three levels

```
 90e9 docs
   │
   ├─► LEVEL 1: EXACT                     removes 25%  →  68e9 docs
   │     doc-level: blake3(normalize(text)) → sort/group
   │     line-level: hash every line, drop lines seen in >N docs
   │     cost: O(N), 1 pass, ~400k core-hours = $8k
   │
   ├─► LEVEL 2: NEAR-DUPLICATE (MinHash-LSH)  removes 50%  →  34e9 docs
   │     5-gram shingles → 112 permutations → 14 bands × 8 rows
   │     40 TB signatures, 50 TB band-key records shuffled
   │     cost: 1.25M core-hours + 50 TB distributed sort = $60k
   │
   └─► LEVEL 3: SEMANTIC (embedding clustering)   [OPTIONAL — see 3.6]
         ModernBERT-small embeddings → k-means → intra-cluster prune
         cost: 200k GPU-hours = $400k
```

**Level 1, doc-level:** blake3 over normalized text (NFKC, collapse whitespace, lowercase, strip punctuation-only lines). Sort 90e9 × 24-byte records = 2.2 TB — a trivial distributed sort. Removes 20–30% of a typical web crawl.

**Level 1, line-level:** hash each line, count global occurrences, drop any line appearing in more than N documents. This is the single highest-value cheap trick in the pipeline: it removes nav bars, cookie banners, "Related Posts", and license footers *without* removing the documents that contain them. Set N by corpus size — at 90e9 docs, N ≈ 10,000 works (roughly one-in-10⁷). Too aggressive and you delete legitimately repeated content (song choruses, code idioms, common sentences); too lax and boilerplate survives. `[established]` — this is C4's original contribution and it has not been improved on.

**Level 2, MinHash-LSH.** Mechanism, since the prompt asks for it explicitly:

1. **Shingling.** Represent each document as the *set* of its overlapping word 5-grams. `[current 2025–26]` FineWeb uses 5-grams; 3-grams over-match short documents (too many false positives on common phrases), 8-grams under-match reformatted text. The set representation is what makes Jaccard similarity the right metric: two documents that share the same content in a different order still share shingles.

2. **MinHash signature.** For each of K=112 independent hash permutations h_i, record `min over shingles s of h_i(s)`. The key property: **P(min_h(A) == min_h(B)) = Jaccard(A,B)** exactly. So a K-length signature is an unbiased estimator of Jaccard with standard error ≈ 1/√K ≈ 9.4% at K=112. That's the whole trick — you replaced a set-intersection over thousands of shingles with 112 integers.

3. **Banding.** Split the 112 hashes into b=14 bands of r=8 rows. Two documents are *candidates* if any band matches exactly. The probability of being a candidate at Jaccard s is:

   **P(candidate) = 1 − (1 − s^r)^b**

   This is an S-curve with inflection near **s\* ≈ (1/b)^(1/r)**. For 14×8: (1/14)^(1/8) = **0.719**.

| Jaccard s | s^8 | P(candidate) at 14×8 | Interpretation |
|---|---|---|---|
| 0.50 | 0.0039 | **0.054** | Different docs, correctly kept |
| 0.60 | 0.0168 | 0.212 | Borderline |
| 0.70 | 0.0576 | 0.566 | At threshold |
| 0.72 | 0.0722 | 0.650 | Inflection point |
| 0.80 | 0.1678 | **0.922** | Near-dup, correctly caught |
| 0.90 | 0.4305 | **0.9991** | Duplicate, caught |
| 0.95 | 0.6634 | 0.99999 | Caught |

**The knobs, with their consequences:**

| Config | b × r | Threshold s\* | Effect |
|---|---|---|---|
| FineWeb default | 14 × 8 | 0.72 | Balanced; the current reference `[current 2025–26]` |
| Aggressive | 20 × 5 | 0.55 | Catches paraphrase-ish dups; kills ~15% more docs; risks diversity |
| Conservative | 9 × 13 | 0.845 | Only near-verbatim; keeps 20% more docs |
| More permutations (K=256, 32×8) | 32 × 8 | 0.63 | Lower variance, 2.3× signature storage |

`[current 2025–26]` FineWeb applies MinHash **per-crawl-snapshot rather than globally**, and this is a deliberate, contested choice. Per-snapshot dedup is O(snapshots × n²/bucket) instead of O(N²/bucket) and parallelizes perfectly, but it leaves cross-snapshot duplicates in place. FineWeb's own ablation found that *global* dedup across all snapshots produced a *worse* model — because it disproportionately removed the content that appears in every snapshot, which is exactly the persistent, canonical, high-quality web. See §3.5.

**Napkin math for signature storage:**
- 90e9 docs × 112 hashes × 4 bytes = **40.3 TB of signatures.** Fits in object storage trivially; does not fit in any single machine's RAM, which is why this is a sort-shuffle problem and not a hash-table problem.
- Band-key records: `(band_id, band_hash[16B], doc_id[16B])` = 36 B × 14 bands × 90e9 docs = **45 TB to shuffle and sort.** At 10 GB/s/node aggregate on a 2,000-node cluster, that's 45e12 / 2e13 B/s ≈ **~1 hour of pure shuffle**, realistically 4–6 hours with skew. This is the dominant wall-clock cost of dedup.
- Union-find over candidate pairs: candidate pairs at these settings ≈ 3–5× the number of documents ≈ 3e11 edges. Do **not** run global union-find; use per-band-bucket connected components with a bounded bucket size (cap at 10⁴ members, split oversized buckets by a secondary hash), then a single global pass to merge cluster IDs.

**What breaks at 10×:** the band shuffle is 450 TB, which is still fine. What actually breaks is the **quadratic blowup inside hot buckets**. A single template used by 50M sites produces one bucket with 50M members and 1.25e15 pairs. The bucket cap is mandatory, not an optimization.

### 3.3 Cross-snapshot dedup for a continuously ingesting system

The one-shot formulation (`dedup(all_data)`) is wrong for a system where a new snapshot lands every 4 weeks. Reprocessing 90e9 documents every month is $60K/month in shuffle you don't need to spend and, worse, changes the dedup decision for *already-processed* documents, breaking reproducibility of prior manifests.

**Incremental design:**

```
  Persistent state:  BAND INDEX  — 14 tables, band_hash → representative_doc_id
                     Size: 14 × 34e9 surviving docs × 24 B = 11 TB
                     Storage: sorted immutable SSTables in object store + bloom filters
                     Bloom: 34e9 × 14 keys at 1% FPR = 8.2 bits each = 700 GB
                            → sharded across workers, ~700 MB per worker at 1000 workers

  New snapshot (3e9 docs) arrives:
    1. Intra-snapshot dedup      (normal MinHash, 3e9 docs, ~20 min)
    2. Probe bloom filters       (cheap reject of ~97% of band keys)
    3. Probe SSTables for hits   (only ~3% survive to a real lookup)
    4. New uniques → append to band index as new SSTable generation
    5. Compact SSTables monthly
```

The bloom-filter front-end is what makes this affordable: `[current 2025–26]` LSHBloom (Khan et al. 2024) replaced the MinHash-LSH index with bloom filters and reported a **12× speedup at petascale**, and this is now the standard pattern for incremental ingest.

**The critical policy question: which copy survives?** When a new snapshot's document matches an existing one, you must choose. Three options:

| Policy | Behavior | Consequence |
|---|---|---|
| Keep-first (oldest) | New copy dropped | Corpus never refreshes; stale content persists forever |
| Keep-last (newest) | Old copy dropped | **Breaks reproducibility** — old manifests reference deleted docs |
| Keep-first, tag-new | New copy stored but marked `dup_of=X` | Recommended |

**Keep-first with tagging.** Nothing is ever deleted, so old manifests stay valid. The new copy carries a pointer, and the *manifest builder* decides whether to prefer the fresher copy for freshness-weighted mixes (§10). Deletion is never a pipeline operation; it is a manifest operation. This principle recurs everywhere in this design.

### 3.4 Semantic dedup — cost it honestly, then probably don't do it

Mechanism (SemDeDup, Abbas et al. 2023): embed every document, k-means into n clusters, prune any document within cosine distance τ of another in the same cluster.

**Cost at our scale.** Embedding 34e9 documents with a small encoder. ModernBERT-base-class, 150M params, 512-token chunks, ~2 chunks/doc average: 150e6 × 512 × 2 FLOPs × 2 chunks = 3.1e11 FLOPs/doc. At 4e14 effective FLOP/s: **1,300 docs/s/GPU**. 34e9 / 1300 = 2.6e7 GPU-s = **7,300 GPU-hours = $15K.** Plus embedding storage: 34e9 × 768 dims × 2 B (fp16) = **52 TB**, and k-means over 52 TB is 10–20 full passes = another ~$50K in I/O and compute. Call it **$100K all-in** — an order of magnitude cheaper than the folk wisdom that semantic dedup is unaffordable. `[current 2025–26]` The cost objection is now largely obsolete; cheap embedders (ModernBERT, GTE-small) made this tractable.

**So why not do it?** Because the ablations don't clearly support it. Abbas et al. showed modest gains; other work shows neutral-to-slightly-negative results, because semantic dedup removes legitimate diversity — five news outlets covering the same event from different angles is exactly the redundancy that teaches a model what is *invariant* about a fact versus what is stylistic. Published open corpora at scale (FineWeb, DCLM, Nemotron-CC, Dolma 3) do **not** use semantic dedup on the general web path. It shows up in domain-specific corpora (e.g., ManufactuBERT: MinHash 20×20 followed by SemDeDup with n=1000 clusters and τ=0.15, removing ~80% of documents in total) where the domain is narrow and redundancy is genuinely pathological.

**Recommendation:** skip it on the general web path; **use it on the synthetic path** (§6), where mode collapse produces genuinely redundant output that MinHash misses because the rephrasings are lexically distinct. That is the one place where the cost is clearly justified. Mark this as `[emerging/contested]`.

### 3.5 The over-deduplication tradeoff — say this number out loud

**What over-deduplication destroys:** frequency information. The web's duplication structure is not noise; it encodes importance. A fact stated on 10,000 pages is more likely to be true and more likely to be worth memorizing than a fact stated once. Aggressive global dedup flattens this distribution and measurably hurts knowledge benchmarks.

The empirical anchor `[current 2025–26]`: **FineWeb found that per-snapshot MinHash outperformed global cross-snapshot MinHash**, specifically because global dedup preferentially removes the persistent, canonical, high-quality web that appears in every snapshot, while leaving the ephemeral spam that appears in exactly one. The intuition "more dedup is better" is inverted at the top end.

The other half of the tradeoff, from the opposite direction: duplicated data causes memorization and privacy leakage — measured at roughly **8× increased verbatim leakage** in models trained on un-deduplicated corpora, plus ~18% wasted training compute on redundant tokens.

**Where to sit:** aggressive exact + line-level dedup (cheap, unambiguously good, no diversity cost), moderate near-dedup at s\*≈0.72 applied per-snapshot with a lightweight cross-snapshot pass that only catches near-verbatim (s\*≈0.85), and no semantic dedup on web. Retain `dup_cluster_size` so the quality classifier and the mixture weighting can use duplication as a feature rather than having it destroyed.

> ### ⬛ Defend this in 60 seconds: dedup order and strength
>
> "Dedup before quality filtering, always. If you filter first, every surviving duplicate is a *high-scoring* duplicate — you've amplified Wikipedia infobox templates and Stack Overflow scaffolding across the whole corpus. The cost is running MinHash on 90 billion documents instead of 27 billion, about 3× the shuffle, which is $60K against a $150M training run. On strength: MinHash at 14 bands of 8 rows gives a threshold of 0.72 Jaccard, and I run it per-snapshot, not globally — **FineWeb showed global cross-snapshot dedup makes the model worse**, because the content that appears in every snapshot is the canonical high-quality web, and the content that appears once is spam. Duplication frequency is signal, not noise. I keep cluster size as a feature and let the classifier decide."

---

## 4. Quality filtering & safety

**Receives from §3:** 34e9 docs / 0.34 PB with dedup cluster metadata. **Exposes to §5:** 9.5e9 docs / 95 TB ≈ 22.0T tokens, with `{quality_scores[], safety_flags[], pii_spans[]}` attached. Throughput contract: the classifier fleet must clear a full corpus pass in under 48 hours so that a threshold change is a two-day experiment, not a two-week one.

### 4.1 Heuristic filters: still there, greatly demoted

`[established → superseded in emphasis]` This is the clearest case of drift since 2023. In 2022–23, heuristics *were* the quality filter (Gopher rules, C4 rules, RefinedWeb rules): mean word length in [3,10], symbol-to-word ratio < 0.1, fraction of lines ending in "..." < 0.3, bullet-point line fraction < 0.9, stopword presence, "lorem ipsum" rejection, curly-brace rejection for prose, a minimum document length.

**Current role:** a cheap pre-filter that removes ~20% of documents at essentially zero cost, whose job is to reduce the volume entering the model-based classifier — *not* to determine quality. Every one of these rules is a proxy for something a classifier learns better. Keep them because they are free and they catch pathological inputs that would waste classifier throughput; do not tune them and do not expect model quality from them.

The one heuristic that still earns its keep independently is **perplexity filtering** against a small reference LM trained on high-quality text — Nemotron-CC uses perplexity scoring as one component of its ensemble `[current 2025–26]`. It is a cheap continuous signal rather than a binary rule.

### 4.2 Model-based quality classifiers — the actual filter, as an embedded inference system

`[current 2025–26]` This is the defining shift of 2024–26. Modern pipelines use neural classifiers to score every document, pioneered publicly by FineWeb-Edu (educational-value classifier) and DCLM (fastText classifier on instruction-formatted positives), and extended by Nemotron-CC into **classifier ensembling**.

**How training data is bootstrapped — the part that is not obvious.** You cannot hand-label 500k documents. The bootstrap is:

1. **LLM annotation.** Prompt a strong instruct model to score sampled web documents on a rubric (typically educational value, 0–5). Nemotron-CC used **the same 460k-document FineWeb-Edu annotation set**, scored by both Mistral-8x22B-Instruct and Nemotron-340B-Instruct. Cost: 460k docs × ~1500 tokens × ~$1/M tokens ≈ **$700.** This is the entire labeling budget.
2. **Distill into a small encoder.** Train a 150M-parameter encoder (or a fastText model) on those labels. This is what actually runs at scale, because you cannot afford the 340B model on 34 billion documents (see the cost comparison below).
3. **Ensemble.** `[current 2025–26]` Nemotron-CC's reported gains come specifically from ensembling multiple classifiers with different labelers and rubrics rather than thresholding one. Their high-quality subset shows **+5.6 MMLU and +3.1 average over DCLM** on an 8B model at 1T tokens.
4. **Next generation bootstraps from the previous model.** Once you have model N, it becomes the annotator for model N+1's classifier — better rubric adherence, cheaper than a third-party API, and the labels reflect what your model actually needs. This feedback loop is the thing that compounds across model generations, and it is why data pipeline quality is a durable advantage rather than a one-time build.

**Fleet sizing, with arithmetic.** 34e9 documents to score:

| Scorer | FLOPs/doc | Docs/s/GPU | GPU-hours for 34e9 | Cost | Wall clock on 2,000 GPUs |
|---|---|---|---|---|---|
| fastText (CPU) | ~0 | 3,000/s/**core** | 3,200 core-h | **$65** | 2 hours on 5k cores |
| Encoder 150M @ 512 tok | 1.5e11 | ~2,700 | 3,500 GPU-h | **$7K** | **1.8 hours** |
| Encoder 400M @ 1024 tok | 8.2e11 | ~490 | 19,300 GPU-h | $39K | 9.6 hours |
| LLM 8B @ 500 tok in/out | 8e12 | ~50 | 189,000 GPU-h | $378K | 3.9 days |
| LLM 70B judge | 7e13 | ~6 | 1.6M GPU-h | $3.2M | 33 days — **infeasible** |

**The design conclusion falls right out of the table.** An encoder classifier costs 0.005% of the training run and clears the corpus in under two hours. An 8B LLM judge costs 0.25% and takes four days. A 70B judge is infeasible. So: **LLM judges label the training set for the encoder; the encoder scores the corpus.** Nobody runs a large LLM over 34 billion documents, and if an interviewer suggests it, the 33-day wall clock is the answer, not the dollar cost.

**Threshold selection via downstream ablation — the only valid method.** Classifier score thresholds cannot be chosen by inspecting the score distribution or by precision/recall against held-out labels, because the objective is downstream model quality, not label agreement. The procedure:

```
  For each candidate threshold τ ∈ {top 10%, 20%, 35%, 50%, 70%}:
      build a token pool at that threshold
      train a 1B model on 30B tokens from that pool   (~1,500 GPU-h = $3K)
      evaluate on the development benchmark suite
  Pick τ, then VALIDATE at 8B / 300B tokens          (~120,000 GPU-h = $240K)
```

5 thresholds × $3K + one validation = **$255K, about 0.17% of the training run.** This is the cheapest high-leverage experiment in the entire pipeline.

**The 2025 correction on where to set it.** `[current 2025–26]` FineWeb-Edu's aggressive filtering (keeping a few percent) maximizes short-horizon benchmark scores but destroys unique-token yield — up to **90% of the data eliminated** in DCLM/FineWeb-Edu-style pipelines. Nemotron-CC's central finding is that for **long-horizon training (≥15T tokens) this is the wrong tradeoff**: they achieve **4× more unique real tokens and +5 MMLU over Llama 3.1 after 15T tokens** by balancing quality against quantity rather than maximizing quality. If you are training a 15T-token run, a filter tuned on 30B-token ablations will systematically over-filter. **Ablate at a token horizon proportional to your real run, or you will pick the wrong threshold.**

### 4.3 Multilingual: why English-trained filters fail, and what to do

An educational-value classifier trained on English documents and applied to, say, Vietnamese fails in three distinct ways, and it is worth separating them because they have different fixes:

1. **Representation failure.** The encoder's tokenizer and embeddings are English-tuned; Vietnamese text is over-fragmented and lands in a poorly-modeled region of representation space. Scores become near-random. *Fix: multilingual encoder backbone (mDeBERTa, XLM-R, or a multilingual ModernBERT).*
2. **Label transfer failure.** "Educational value" is defined against an English web whose composition is different. In many languages the high-quality web is dominated by news and government sites rather than tutorials and reference; an English rubric penalizes exactly the good content. *Fix: per-language rubrics and per-language annotation sets.*
3. **Threshold failure.** Even with a working classifier, the score distribution shifts per language. A global threshold keeps 60% of German and 4% of Swahili. *Fix: per-language quantile thresholds — "keep the top 35% *of this language*" — not a global score cutoff.*

`[current 2025–26]` FineWeb2 is the clearest public instance of this design: per-language pipelines with per-language filter configuration and thresholds, rather than one global filter. This is a direct correction of the 2023-era approach where multilingual data was filtered with English-derived rules.

**Low-resource handling.** Below roughly 1B tokens available in a language, filtering is counterproductive — you are removing 65% of a corpus that is already too small to matter, and the residual is not diverse enough to teach the language. Policy tiers:

| Tier | Available tokens | Filter policy | Epoch policy |
|---|---|---|---|
| High (en, zh, es, de, fr, ru, ja…) | >100B | Full classifier, top 35% | 1–2 epochs |
| Medium (~40 languages) | 5–100B | Multilingual classifier, top 60% | 2–4 epochs |
| Low (~1,500 languages) | <5B | Heuristics + script check only | 4–8 epochs, capped weight |

`[emerging/contested]` The multi-epoch policy for low-resource languages follows directly from data-constrained scaling (§7.4) but I have not found a frontier report that states its per-language epoch counts. Treat the specific numbers as reasoned, not sourced.

### 4.4 Toxicity, safety, PII: filter vs scrub vs keep-with-tags

The naive instinct — filter everything objectionable — is wrong, and the reason is worth stating precisely: **a model that has never seen toxic content cannot recognize, refuse, or classify it.** Removing all toxicity from pretraining data degrades the safety behavior of the post-trained model. The current consensus is a three-way split by category:

| Category | Action | Rationale |
|---|---|---|
| CSAM, terrorism instructionals, credible CBRN uplift | **Remove**, log hash, report per policy | Non-negotiable; no training value justifies retention |
| General toxicity / hate speech | **Keep with tag**, downweight to ~1/10 natural rate | Needed for safety classifiers and refusal training; tag enables removal at manifest time |
| Spam / SEO junk / MFA sites | **Remove** (quality filter usually catches it) | No value at any weight |
| PII — emails, phones, SSNs, IPs | **Scrub in place** | Document remains useful; the identifier does not |
| PII — names in public/journalistic context | **Keep** | Removing it makes the model unable to discuss public figures |

`[current 2025–26]` NeMo Curator's `PiiModifier` is the reference open implementation: regex plus NER-backed detection for emails, phone numbers, SSNs, and IPs, replacing rather than dropping. Scrubbing rather than dropping matters at scale — dropping every document containing an email address would remove most of the forum, mailing-list, and documentation web.

**Tag, don't delete, for anything ambiguous.** Same principle as §3.3: the safety policy becomes a manifest-time decision. When the policy changes (and it will, between model generations and across jurisdictions), you rebuild a manifest in an hour instead of reprocessing 34 billion documents.

> ### ⬛ Defend this in 60 seconds: the curation-economics tradeoff
>
> "The whole pipeline's economics hinge on one asymmetry: an encoder quality classifier costs about **$7,000 to score 34 billion documents** — 3,500 GPU-hours — against a training run that costs $150 million. That's 0.005%. So the question is never 'can we afford to filter more carefully'; it's always yes. What you *can't* afford is an 8-billion-parameter LLM judge over the whole corpus — that's four days of wall clock on 2,000 GPUs — and you definitely can't afford a 70B judge, which is 33 days. So the pattern is: **LLM labels 460,000 documents for about $700, distill into a 150M encoder, encoder scores the corpus.** The one thing worth spending real money on is threshold ablation — five 1B-model runs plus one 8B validation, about $255K — because Nemotron-CC showed that a threshold tuned at a short horizon over-filters badly at 15T tokens, and getting that wrong costs you 5 MMLU points on the real run."

---

## 5. Decontamination

**Receives from §4:** 9.5e9 docs / 22.0T tokens with quality and safety metadata, plus the write-once eval holdout from §1.6. **Exposes to §6/§7:** 21.5T tokens with `{contamination_flags[]}`, plus a per-benchmark contamination report that is a required artifact of every dataset version.

### 5.1 Why it runs here — after dedup and filtering, before mixing

- **After dedup:** dedup would otherwise collapse a contaminated document into a cluster with clean ones, and you would lose track of which cluster members were contaminated. Running after means every surviving document is a distinct object you can flag individually.
- **After filtering:** contamination check is O(tokens), so running it on 22T tokens instead of 200T is a 9× saving on a stage that touches everything.
- **Before mixing:** because upsampling is the amplifier. A contaminated document upsampled 8× in the annealing mix (§7.5) does 8× the damage, and the annealing mix is exactly where high-quality-looking benchmark-adjacent content concentrates. Decontaminating after mixing means you've already built the manifest you have to throw away.

### 5.2 N-gram matching — mechanism and sizing

`[established, current implementation 2025–26]` The standard: build an n-gram index over the eval sets, then reject any training document containing a run of matching n-grams.

- **n = 13.** Below ~10, false positives explode on common English ("in the context of the following"). Above ~20, trivial reformatting evades. 13 is the de-facto standard.
- **Match rule: 10+ *consecutive* matching 13-grams**, not a single hit. A single 13-gram match is often coincidental or a shared quotation; ten consecutive means ~22 verbatim tokens of overlap.

**Sizing the filter.** Eval sets: ~200 benchmarks, ~2B tokens total including all splits and prompt templates. That is ~2e9 13-grams. A bloom filter at 1% FPR needs ≈9.6 bits/element:

> 2e9 × 9.6 bits = 1.92e10 bits = **2.4 GB**

which fits in RAM on **every** worker. That is the whole reason this stage is cheap: the eval side is tiny, so decontamination is a single streaming pass over the training data with an in-memory probe. Cost: ~34e9 docs at ~2,000 docs/s/core = 4,700 core-hours = **$95.** With a 1% bloom FPR and a 10-consecutive-match rule, the effective false-positive rate on documents is negligible; confirm hits against the exact n-gram set (2 GB on disk) before flagging.

### 5.3 Fuzzy and the paraphrase arms race

N-gram matching catches verbatim copies. It does not catch:
- The same GSM8K problem with the numbers changed.
- An MMLU question translated to another language and back.
- A tutorial that explains exactly the reasoning pattern a benchmark tests, written independently.
- Model-generated rephrases of benchmark items that ended up in your synthetic pipeline (§6) — the nastiest case, because you generated it yourself.

**Escalating defenses, with honest cost/benefit:**

| Method | Catches | Cost | Verdict |
|---|---|---|---|
| 13-gram exact | Verbatim | $95 | Always |
| MinHash vs eval set (s\*=0.6) | Reformatted, partial | $2K (reuse §3 signatures) | Always |
| Embedding similarity vs eval set | Paraphrase, translation | $15K (reuse §3.4 embeddings) | Yes — cheap since embeddings exist |
| LLM-judged semantic equivalence | Deep paraphrase | $50K on top-1M candidates | Yes, as a final pass on the flagged tail |
| Nothing | "Independently written tutorial on the same topic" | — | **Unsolvable, and that's fine** |

The last row deserves emphasis because it is where candidates get tangled up. **You cannot and should not remove all content topically related to a benchmark** — that is just removing the capability you are trying to measure. The line is: *this specific item, or a transformation of it*, is contamination; *the general knowledge the item tests* is the training objective. When an interviewer pushes on the arms race, the honest answer is that the arms race is unwinnable at the semantic end, which is why the real defense is held-out evaluation on benchmarks constructed after your data cutoff (§1.6's time-forward slice), not better matching.

`[current 2025–26]` Olmo 3 / Dolma 3 makes decontamination a headline property of the corpus ("much stronger decontamination via extensive deduplication, quality filtering, and careful control over data mixing") — but note that even the most-documented open corpus describes it qualitatively. The quantitative contamination-audit reports are, as far as I can find, not published by anyone. `[inferred]` — I would expect frontier labs to hold per-benchmark contamination rates internally as a release gate.

### 5.4 The contamination report — a required artifact

Every dataset version emits:

```
dataset_version: dolma-like-v7.2
per_benchmark:
  MMLU:      docs_flagged=1,247   tokens_flagged=3.1M   rate=1.4e-7   max_run=41 ngrams
  GSM8K:     docs_flagged=89      tokens_flagged=0.2M   rate=9.1e-9   max_run=18
  HumanEval: docs_flagged=14,203  tokens_flagged=8.8M   rate=4.0e-7   max_run=200+
  ...
fuzzy_flagged_pending_review: 4,109 docs
policy: flagged docs excluded from all mixes; retained in corpus with flag
```

HumanEval will always be the ugly row — its problems are on GitHub in a hundred forms — which is exactly why you want the number in front of you rather than discovering it when the benchmark score looks too good.

---
## 6. Synthetic & augmented data as a generation pipeline

**Receives from §5:** decontaminated real documents with quality scores. **Exposes to §7:** synthetic documents in the same JSONL schema, with `derived_from` populated, `source_tier=synthetic`, `generator_model_id`, and `prompt_template_id`. Synthetic output re-enters the pipeline at §3 (dedup) and §5 (decontamination) — **it is not exempt from either.**

### 6.1 What changed since 2023

`[current 2025–26]` In 2023 synthetic pretraining data meant Phi-style "textbooks" — a small, curated, high-effort corpus. It is now a **bulk pipeline stage that produces trillions of tokens**, and it is used primarily to *recover* data destroyed by aggressive filtering rather than to add new knowledge.

The pivotal public result: Nemotron-CC generates **2 trillion tokens of high-quality synthetic data** to expand its final dataset, explicitly to restore content eliminated by filtering — noting that DCLM and FineWeb-Edu discard up to **90%** of the data, which makes them unsuitable for long-horizon (15T-token) pretraining. That reframes synthetic data's role entirely: it is the counterweight to over-filtering, not a substitute for the web.

Named approaches, with what each does:

| System | Transformation | Note |
|---|---|---|
| WRAP (Maini et al. 2024) | Stylistic rewriting into Wikipedia/QA/textbook styles | The origin of bulk rephrasing |
| Nemotron-CC (NVIDIA 2025) | QA-pair extraction, knowledge lists, distillation, wrap | 2T tokens; production scale |
| BeyondWeb (DatologyAI 2025) | Continuation and summarization rephrasing | |
| REWIRE (Nguyen et al. 2025) | Guided transformation of *low-quality* documents | Targets the discard pile directly |
| FinePhrase (2026) | FineWeb → 4 structured formats, 1.35B samples / **486B tokens** | Publishes its cost; see below |

`[current 2025–26]` A 2026 systematic comparison found that the synthetic datasets (Nemotron-HQ-Synth, REWIRE) land **within 0.28 macro-average points of DCLM** as a primary baseline — i.e., roughly at parity with the best real-data pipeline, not above it. Read that carefully: synthetic data currently **buys you token *quantity* at real-data *quality***, which is precisely what a data-constrained long-horizon run needs, but it does not yet beat good real data.

### 6.2 Cost per synthetic token, with real numbers

The only public end-to-end figure I found: **FinePhrase generated 486B completion tokens using 100 H100s with SmolLM2-1.7B and suffix-32 speculative decoding, at ~9,200 tokens/sec/GPU, for 612 GPU-days ≈ 14,700 GPU-hours.** That is **33M tokens per GPU-hour**, or **$0.06 per million synthetic tokens** at $2/GPU-hour.

Scale that up by generator size:

| Generator | Throughput (tok/s/GPU) | GPU-h for 2T tokens | Cost | % of $150M run |
|---|---|---|---|---|
| 1.7B + spec decode (measured) | 9,200 | 60,000 | **$120K** | 0.08% |
| 8B, batched, no spec decode | ~2,300 | 240,000 | **$480K** | 0.32% |
| 8B + spec decode | ~5,000 | 111,000 | $222K | 0.15% |
| 70B | ~500 | 1,100,000 | **$2.2M** | 1.5% |
| 70B, 2 epochs of rewriting | ~500 | 2,200,000 | $4.4M | 2.9% |

**The design conclusion:** small generators with speculative decoding are so cheap that generating 2T synthetic tokens is a rounding error. A 70B teacher is 18× more expensive and starts to be a real budget line. The question is whether the larger generator's output is *18× more valuable*, and the honest answer from the systematic comparison above is: probably not for bulk rephrasing, possibly yes for reasoning traces and math/code solutions where the generator's actual capability is the constraint. **Split the budget: 1.7–8B for bulk rephrasing at trillion scale; 70B+ for a few hundred billion tokens of reasoning and domain content.**

Speculative decoding is not optional here. At 2T output tokens, going from 2,300 to 5,000 tok/s/GPU saves $258K and halves wall clock. This is one of the few places where inference-engine optimization directly shows up in a data-pipeline budget.

### 6.3 Quality control and collapse

Three distinct failure modes, often conflated:

**(a) Mode collapse in output distribution.** The generator has stylistic tics; 2T tokens of it means the model learns the tics. *Detection:* type-token ratio and n-gram entropy of the synthetic corpus vs the real corpus it was derived from; a distinguisher classifier (real vs synthetic) — if it hits >95% AUC on held-out text, the synthetic data is too identifiable. *Mitigation:* rotate prompt templates (FinePhrase uses 4 structured formats), rotate generators, and run **semantic dedup (§3.4) on the synthetic path specifically** — this is the one place it clearly earns its cost, because rephrasings of the same source are lexically distinct (MinHash misses them) but semantically identical.

**(b) Factual drift / hallucination amplification.** A rephrase can introduce facts the source did not contain. *Mitigation:* constrain the task. Extraction (QA pairs, knowledge lists) and style transfer are much safer than free continuation, because the output is grounded in the input. `[emerging/contested]` NLI-based faithfulness scoring of rephrase against source, sampled at 0.1%, is the obvious check; I have not seen it documented in a production pretraining pipeline.

**(c) Recursive self-consumption.** Model N's output trains model N+1. This is the widely-discussed "model collapse" risk. In practice the mitigation is structural rather than clever: **synthetic data is capped as a fraction of the mix (see below), and every synthetic document is grounded in a real source document via `derived_from`.** Ungrounded generation ("write a textbook chapter about X") is where collapse risk actually lives, and it should be a small, separately-budgeted, separately-ablated slice.

**(d) Contamination laundering — the one people miss.** If a benchmark item leaked into the source corpus and you rephrase it, the rephrase evades your 13-gram filter. Two defenses: decontaminate *before* generating (which is why §5 precedes §6 in this pipeline), and re-run decontamination on synthetic output including the embedding-similarity pass.

### 6.4 Where synthetic sits in the mixture, and how much

`[current 2025–26]` Nemotron-CC's 2T synthetic tokens sit alongside its real tokens in a corpus intended for 15T-token training — so roughly **10–15% of presented tokens** in the open reference. Frontier fractions are undisclosed; `[inferred]` I would expect frontier labs to be higher, in the 15–30% range, because they have both more filtering pressure and cheaper generation, but this is reasoning, not a source.

Placement matters as much as fraction: synthetic data is disproportionately clean, structured, and instruction-like, which makes it a natural fit for the **mid-training and annealing phases** (§7.5) rather than uniform distribution across the run. Olmo 3's Dolmino mid-training mix explicitly includes synthetic instruction-following and reasoning-trace data — 100B training tokens sampled from a ~2.2T-token pool.

---

## 7. Tokenization & mixture management

**Receives from §5/§6:** ~25.8T unique tokens' worth of scored, decontaminated, deduplicated documents across all sources. **Exposes to §8:** a **recipe** (a versioned declarative document) plus tokenized token streams. This section contains the single highest-leverage decision in the pipeline, so it gets the most space.

### 7.1 Tokenization at scale

**Throughput.** HuggingFace `tokenizers` fast BPE: ~2–4 MB/s/core. For 95 TB of final text:

> 9.5e13 B / 3e6 B/s = 3.17e7 core-seconds = **8,800 core-hours = $176**

Tokenization is free. Say this clearly because interviewers sometimes expect it to be a bottleneck: it is 0.08% of the extraction cost.

**What is *not* free is the rewrite.** Tokenized output is 30T tokens × 4 bytes (uint32; vocab is 128k–256k so uint16 is unavailable) = **120 TB**, and it must be written, shuffled, and re-sharded. At 20 GB/s aggregate write bandwidth that's 6,000 seconds of pure write, but the shuffle (§8.3) is the real cost — call it 6–12 hours and ~$5K of I/O for a full retokenization.

**The tokenizer-changes-mid-build problem.** Options:

| Strategy | Cost | Latency | When |
|---|---|---|---|
| **Recompute everything** | $176 CPU + ~$5K I/O + 12 h | 12 h | Almost always. It's cheap. |
| **Lazy / tokenize-on-the-fly in the loader** | ~0 storage, +CPU per host | 0 | Only if host CPU is idle |
| **Dual-write both tokenizers** | 2× storage: +120 TB = +$2.4K/mo | 0 | During a transition window |
| Keep old tokenizer, live with it | 0 | 0 | If change is cosmetic |

**The answer to "the tokenizer changes after 60% of the data is processed" is: recompute, and it is not a close call** — 12 hours and $5K. The trap in the question is that the *interesting* cost is not tokenization; it is that **you cannot mix tokenizers within a run**, so the real decision is whether to restart the training run. If 60% of *training* (not data prep) is done, you eat the tokenizer you have or you restart at a cost of ~$90M. If 60% of *data prep* is done, you retokenize and lose half a day. Distinguishing those two readings of the question is the answer.

**Lazy tokenization deserves a fair hearing.** Loader-side tokenization at 11.5M tokens/s aggregate needs 11.5e6 / (3e6 B/s / 4.3 B/tok) ≈ **16 cores** across the whole fleet. Sixteen cores. It is genuinely viable, it eliminates the 120 TB artifact and the retokenization problem entirely, and it makes the tokenizer a runtime parameter. The reason it isn't standard: it makes **exact-position resumability (§9.3) harder**, because your shard offsets are byte offsets in text, and the token count of a shard is not known until you tokenize it, so you cannot compute "which token index am I at" without a pre-pass. `[emerging/contested]` — a pre-pass storing per-shard token counts solves this and I'd expect some teams do exactly that, but I have no source.

### 7.2 The recipe is the artifact

```yaml
recipe_id: pretrain-v7.2
corpus_snapshot: dolma-like-v7.2   # frozen manifest hash, §10
tokenizer: bpe-256k-v3             # hash-pinned
seed: 20260731
phases:
  - name: main
    tokens: 25.0e12
    seq_len: 8192
    weights:                       # sampling probability, sums to 1
      web_en_hq:      0.34
      web_en_mid:     0.16
      web_multiling:  0.14
      code:           0.16
      math_sci:       0.06
      books_licensed: 0.04
      synthetic:      0.10
    max_epochs: {web_en_hq: 4, web_en_mid: 1, code: 2, math_sci: 4,
                 books_licensed: 3, synthetic: 1, web_multiling: 3}
  - name: midtrain
    tokens: 4.0e12
    seq_len: 8192
    weights: {math_sci: 0.25, code: 0.25, web_en_hq: 0.20,
              synthetic_reasoning: 0.20, instruct_like: 0.10}
  - name: longctx
    tokens: 0.6e12
    seq_len: 131072
    weights: {long_docs: 0.60, shortmix_replay: 0.40}
  - name: anneal
    tokens: 0.4e12
    weights: {curated_hq: 0.70, instruct_like: 0.20, math_sci: 0.10}
```

The recipe is version-controlled, hash-pinned to a corpus snapshot and a tokenizer, and is the *only* input to §8. Everything about a run's data is reproducible from `(recipe_id, seed)`.

### 7.3 How mixture weights are actually chosen

This is the question with the most drift since 2023, so here is the honest current state, in three tiers:

**Tier 1 — manual + intuition + prior generation.** `[established, still dominant]` Most weights in most real recipes are inherited from the previous model generation with adjustments driven by observed weaknesses. Nobody's published recipe looks like the output of an optimizer. This is not laziness: the search space is huge, the objective is a whole benchmark suite, and the previous generation's mixture is a strong prior.

**Tier 2 — proxy-model scaling experiments.** `[current 2025–26, this is the real workhorse]` Train small models on candidate mixtures, measure, extrapolate.

- **RegMix** (Liu et al., ICLR 2025 Spotlight) is the canonical form: train many small proxies on diverse mixtures, fit a regression from mixture weights → validation loss, optimize the regression. They trained **512 models at 1M parameters on 1B tokens each** to predict the best mixture among 64 models that were **1000× larger and trained 25× longer**, achieving **+6.3% on HellaSwag over human selection at 2% extra training FLOPs**. They also report that mixture choice moves single-task performance by up to **14.6%** — which is the number that justifies the whole exercise.
- **DoReMi** (Xie et al. 2023): group-DRO on a proxy to learn domain weights without downstream tasks. Earlier, still cited, generally superseded by regression approaches on cost grounds.
- **Micro-annealing** `[current 2025–26]`: Olmo 3's actual production method. Rather than optimizing a full mixture, they run **lightweight training runs that ablate each candidate data source in parallel**, then combine the promising ones into a centralized 100B-token annealing "integration test." This is a much more practical shape than a global optimizer: it is embarrassingly parallel, each experiment answers a human-legible question ("is this source worth including"), and the integration run catches interactions.

**Tier 3 — learned / gradient-based methods.** `[emerging/contested]` A dense 2025–26 literature: CLIMB, MixMin (convex minimization), Chameleon (leverage scores), FastMix (gradient-based bilevel, claiming ~550× less compute than RegMix at 1.3 GPU-hours), CausalMix, capacity-aware mixture laws. Real results on small models; **I have found no frontier technical report stating that its production mixture came from a learned optimizer.** Treat this tier as promising and not yet load-bearing.

**One finding from Tier 3 worth carrying anyway** `[emerging/contested]`: the optimal mixture *shifts with model size* — as models get larger, the optimal proportion of knowledge-heavy data grows while math and code shrink (capacity-aware mixture law work, 2026). If true, it means proxy-model mixtures systematically over-weight math and code, and you should correct in that direction when transferring from a 1B proxy to a 100B target. This is exactly the kind of claim to test cheaply rather than believe.

**Budget for mixture experiments.** This is the money you should actually fight for:

```
  Micro-anneal ablations:  40 sources × (1B model, 20B tokens)
                           = 40 × 1,000 GPU-h = 40,000 GPU-h  = $80K
  RegMix-style regression: 200 × (0.5B model, 10B tokens)
                           = 200 × 250 GPU-h  = 50,000 GPU-h  = $100K
  Integration runs:        6 × (8B model, 100B tokens)
                           = 6 × 40,000 GPU-h = 240,000 GPU-h = $480K
  Scale validation:        1 × (30B model, 500B tokens)       = $600K
                                                    TOTAL  ≈ $1.26M
```

**$1.26M — 0.84% of the training run — to de-risk the single decision that RegMix measured at up to 14.6% on individual tasks.** If any line item in this document is under-funded in practice, it is this one.

### 7.4 Multi-epoch policy under data constraints

`[established 2023, now standard practice]` The framing has completely inverted since the Chinchilla era. Early LLMs trained one epoch and the literature warned against reuse (Hernandez 2022, Lee 2022) — this is exactly why aggressive dedup became standard. Then Muennighoff et al. (2023, JMLR 2025) ran 400+ experiments from 10M to 9B parameters, up to 1500 epochs, and established:

- Repeating data for a **small number of epochs (up to ~4) is nearly as good as fresh unique data**; the value of repeated tokens decays exponentially toward a ceiling.
- Beyond ~4 epochs, returns collapse; at high repetition counts, test loss can *increase mid-training*.
- Under data constraints, **allocate additional compute to epochs faster than to parameters** — the opposite of Chinchilla's equal-scaling prescription.
- Their headline combination: doubling data with code and then repeating 4× gives 8× more training tokens with performance matching 8× more unique data.

`[current 2025–26]` The field has moved from *whether* to reuse to *how*: repetition budgets are now set **per source**, not globally, which is the practically important refinement (and an explicit limitation of the original paper, which only modeled uniform repetition of the whole corpus).

**Per-source epoch policy for our pool:**

| Source | Unique tokens | Epochs | Presented tokens | Reasoning |
|---|---|---|---|---|
| web_en_hq (top 5% by classifier) | 1.1T | 4 | 4.4T | Highest value, scarcest; well within the 4-epoch limit |
| web_en_mid | 12.0T | 1 | 12.0T | Abundant; no reason to repeat |
| web_multiling | 5.0T | 1–3 (per tier) | 6.5T | Low-resource tiers repeat more (§4.3) |
| code (permissive) | 1.5T | 2 | 3.0T | Scarce and high-value |
| math_sci | 0.4T | 4 | 1.6T | Very scarce, very high value |
| books/licensed | 0.5T | 3 | 1.5T | Scarce, high quality, license-limited |
| synthetic | 2.0T | 1 | 2.0T | **Never repeat synthetic** — see below |
| | **22.5T unique** | | **31.0T presented** | avg 1.38 epochs |

**Why synthetic never repeats:** it is already a lossy re-expression of real data you are also training on. Repeating it compounds the generator's distributional bias without adding information. If you need more synthetic tokens, generate more — at $0.06–0.24/M tokens that is cheaper than the risk. `[inferred]`

### 7.5 Staged schedules: pretraining → mid-training → long-context → annealing

`[current 2025–26]` The uniform-mixture assumption is dead. The standard shape is now multi-stage:

- **Definition anchor:** Olmo 2 defined mid-training as a distinct stage consuming **5–10% of training FLOPs**, and released Dolmino Mix 1124 as the first systematic mid-training dataset (843B token pool). Olmo 3 refined this: **Dolma 3 Mix at ~5.9T tokens for main pretraining, Dolma 3 Dolmino at 100B training tokens sampled from a ~2.2T pool of high-quality math, science, code, instruction-following, and reading comprehension — including reasoning traces that enable RL directly on the base model — then a separate 100B-token long-context stage.**
- Adopted broadly: OLMo 2/3, Phi-4, LongCat-Flash, Yi, EuroLLM-9B (trapezoid/WSD schedule with progressively higher-quality data), SmolLM2 (4 stages).
- The learning-rate interaction is the mechanism: with a WSD or trapezoid schedule, the decay phase is where the model's weights are most malleable toward the final data distribution, so high-quality data placed there has outsized effect.

`[emerging/contested]` A 2025–26 caution worth knowing, because it inverts the naive intuition: work on training re-evaluation curves and on LR-decay/curriculum interaction argues that **decaying the LR to near zero during the high-quality phase can waste that data** — the model can't move far enough to absorb it. One line of work advocates holding LR constant through mid-training (α_mid = 1.0) rather than decaying. This is unresolved; if you present the staged design, be ready to say that the placement-at-low-LR assumption is currently being questioned.

**Recommended stage structure for our 30T budget:**

| Stage | Tokens | % | Seq len | Character |
|---|---|---|---|---|
| Main pretraining | 25.0T | 83% | 8k | Broad web-dominant mix, WSD stable phase |
| Mid-training | 4.0T | 13% | 8k | Math/code/reasoning-heavy; ~5–10% FLOPs per OLMo definition |
| Long-context extension | 0.6T | 2% | 32k → 128k | See §7.6 |
| Annealing | 0.4T | 1.3% | 8k | Curated highest-quality; LR → 0 |

### 7.6 Long-context data curation

The core problem, stated plainly: **documents longer than 32k tokens are rare and are almost entirely books and code.** You cannot get a 128k-context model by upsampling the natural length distribution because the natural distribution is nearly empty out there.

`[current 2025–26]` The established findings:

- **Source composition.** ProLong's ablations (Gao et al. 2024) found code repositories (all files per repo concatenated) and narrative books are the strongest long-context sources — code hitting 99.2 on recall metrics, books 94.9, best when mixed. This is the strongest argument for the repo-level code grouping in §2.5: **it is simultaneously your best long-context source.**
- **Long/short ratio.** ProLong: roughly **60% long / 40% short** tokens in the continued-pretraining mix. Training *only* on long data collapses short-context ability; training only on short data confers no long-context ability. SmolLM2 used 40% long (≥8k, from DCLM 10% / FineWeb-Edu 10% / Dolma books 20%) with 60% following the prior stage's mix.
- **Upsampling factor.** Llama-3-lineage practice: documents >32k tokens upsampled ~5× **without altering the overall domain mixture ratio** — i.e., you upsample the long tail within each domain, not by shifting toward book-heavy domains.
- **Budget.** Small. Yi extended to 64k with **5B tokens** (100 optimization steps at 4M batch). ProLong used 20B at 64k plus 20B at 512k, resetting the LR schedule and raising the RoPE base frequency at the transition. Olmo 3 used 100B. Long-context extension is a 0.1–2% of budget stage, not a major consumer.
- **Synthetic long documents.** Concatenating *related* short documents (ICLM/In-Context Pretraining builds a document similarity graph and orders by an approximate traveling-salesman path; CodeLlama concatenates same-project files) beats concatenating random documents, because random concatenation teaches the model that distant context is irrelevant — the exact opposite of the objective.

**Interaction with packing (§8.2), which is the part people miss.** If you pack documents to fill a 128k sequence and use a naive causal mask, most of your "long-context" sequences are 30 unrelated 4k documents in a trench coat, and the model learns to ignore anything before the last document boundary. **Long-context stages require either (a) genuinely long single documents, or (b) intra-document attention masking so cross-document attention is blocked, or (c) deliberately *related* concatenation.** The Llama-3 lineage treats an inter-document attention mask as essential during long-context continued pretraining. Option (b) plus (c) is the practical answer: mask by default, and use similarity-ordered concatenation to create genuine long-range dependencies where real long documents don't exist.

**Length-distribution management contract:** the corpus metadata carries `token_count` per document, and the mixture sampler can express weights as a joint distribution over (source, length bucket). Length buckets: `[0,1k), [1k,4k), [4k,16k), [16k,64k), [64k,∞)`. Without this, long-context recipes are unexpressible.

> ### ⬛ Defend this in 60 seconds: the mixture is the highest-leverage decision
>
> "Mixture is where the leverage is. **RegMix measured up to 14.6% swing on individual downstream tasks purely from mixture weights**, and it predicted the right mixture for 1B-parameter models by training 512 one-million-parameter proxies — a thousand times smaller — at 2% extra FLOPs. So I'd spend about **$1.26 million, under 1% of the run**, on proxy ablations: parallel micro-anneals per source the way Olmo 3 does it, a regression fit over ~200 proxy mixtures, then six 8B integration runs and one 30B validation. And I would not run a uniform mixture — that assumption is dead. Olmo 2 defined mid-training as **5–10% of training FLOPs** on a high-quality mix, and everyone from Phi-4 to SmolLM2 does some version of it. One caveat I'd raise unprompted: recent work argues that decaying the learning rate to zero during the high-quality phase wastes that data, so the 'save the best for last' intuition is currently contested."

---

## 8. Training-ready storage & sharding

**Receives from §7:** a recipe plus tokenized token streams per source. **Exposes to §9:** immutable shard files plus a manifest. Contract: given `(recipe_id, seed, global_step, rank)`, the loader can compute exactly which bytes it needs without coordination.

### 8.1 Shard format

| Format | Random access | Metadata | Streaming | Verdict |
|---|---|---|---|---|
| Raw `.bin` + separate `.idx` | Yes (mmap) | External | Yes | **Recommended.** Megatron-style. Dead simple. |
| WebDataset (`.tar`) | Sequential only | In-band | Yes | Good for multimodal, unnecessary overhead here |
| Parquet | Yes | Rich, columnar | Yes | Nice for analysis, decode overhead in the hot path |
| MDS / Mosaic StreamingDataset | Yes | Rich | Yes, designed for it | Strong alternative; more machinery than needed |
| TFRecord | Sequential | Weak | Yes | Legacy |

**Choice: flat uint32 token arrays + a sidecar index.** Reasons: (a) zero decode cost — the loader `mmap`s and slices; (b) exact token positions are computable arithmetic, which is what makes §9.3 resumability trivial; (c) it is trivially seekable, which the deterministic sampler requires.

```
shard_00417.bin      120 MB, 30e6 uint32 tokens, no framing, no compression
shard_00417.idx      doc boundaries: uint64 offsets, one per document (≈ 4k docs)
shard_00417.meta     source_id, recipe-relevant tags, doc_id range,
                     token_count, sha256(bin), license_ids[], build_timestamp
```

**Do not compress the .bin.** Tokenized data compresses only ~15% (it's already entropy-coded by BPE) and compression destroys `mmap` and exact offset arithmetic. The 15% is not worth it.

**Manifest** (one per dataset version, ~50 MB for 1M shards):

```
manifest_v7.2.json:
  recipe_id, corpus_snapshot_hash, tokenizer_hash
  shards: [ {path, source_id, n_tokens, sha256, length_bucket_hist} × 1e6 ]
  source_totals: {web_en_hq: 1.10e12, code: 1.50e12, ...}
```

Shard size: **120 MB / 30M tokens**. Smaller means manifest bloat and per-file overhead at object-store scale; larger means coarse shuffling granularity and expensive skip-a-bad-shard (§9.5).

### 8.2 Packing and document boundaries

Sequences must be exactly `seq_len` tokens. Three strategies:

| Strategy | Waste | Boundary handling | Use |
|---|---|---|---|
| Pad to seq_len | 40–70% of compute | Clean | Never at pretraining scale |
| **Concatenate + split (naive)** | ~0% | Documents split mid-way; cross-doc attention allowed | Standard for main pretraining |
| **Best-fit packing + intra-doc mask** | ~1% | Documents intact, cross-doc attention blocked | **Long-context and mid-training** |

**The attention-masking implication, which is a real systems cost.** Blocking cross-document attention requires a block-diagonal mask. With FlashAttention this is a variable-length (`varlen`/`cu_seqlens`) call rather than a dense mask — you pass cumulative sequence lengths and the kernel skips the off-block tiles entirely. That means it is not merely free, it is *faster* than full attention on the same sequence, because you skip work: packing k documents of equal length into one sequence turns O(L²) attention into O(L²/k). The cost is loader complexity (the loader must emit `cu_seqlens` alongside tokens) and a bin-packing step at shard-build time.

**When does cross-document attention actually hurt?** At 8k sequence length with a mean document length of ~1k tokens, a naive packed sequence contains ~8 unrelated documents and the model spends attention capacity learning that document boundaries are boundaries. Empirically this is tolerable for main pretraining (everyone did it for years) and clearly harmful for long-context, where it defeats the entire purpose. **Recommendation: naive concatenation for the 25T-token main phase (simpler, and the attention waste is small relative to the token budget); varlen masking for mid-training, long-context, and annealing.**

### 8.3 Shuffle strategy

Three levels, and you need all three:

1. **Build-time global shuffle.** Documents assigned to shards by `hash(doc_id, seed)` so that any single shard is a random sample of its source. This is the 120 TB shuffle and the expensive one — a full sort-shuffle of 120 TB at 20 GB/s aggregate is ~1.7 hours of I/O, realistically 6 hours. Done once per dataset version.
2. **Epoch-level shard permutation.** The loader visits shards in an order determined by `PRF(seed, epoch)`. Free.
3. **Host-side buffer shuffle.** A ring buffer of ~10⁴ sequences per rank, sampled without replacement. Covers within-shard correlation. Costs 10⁴ × 8192 × 4 B = **328 MB of host RAM per rank** — trivial.

You need level 1 because levels 2 and 3 cannot fix a shard whose contents are all from one website. You need level 3 because level 1 is per-dataset-version and level 2 is coarse.

### 8.4 The bandwidth math — and why this is trivial compared to video

```
  30e12 tokens  /  (30 days = 2.59e6 s)   =  11.6e6 tokens/sec globally
  × 4 bytes/token (uint32)                =  46 MB/s  aggregate
  / 12,500 hosts (100k accel / 8)         =  3.7 KB/s per host
  read amplification (block reads, shuffle buffer, prefetch): 8×
                                          =  370 MB/s aggregate, 30 KB/s per host
```

**Thirty kilobytes per second per host.** A single modern NVMe drive could feed the entire 100,000-accelerator cluster's data needs sixty times over. The 120 TB dataset fits on ~15 NVMe drives.

**Contrast with video, since the prompt asks explicitly.** A video pretraining run at the same accelerator count consumes raw frames: 100k accelerators processing, say, 2M frames/s at 224×224×3 bytes = **300 GB/s**, ~6,500× the text figure — and that is *after* decode, with the JPEG/H.264 decode itself consuming meaningful host CPU. Video pipelines are dominated by storage bandwidth and decode; **text pipelines are dominated by nothing at the loading layer at all.** The entire §9 design is therefore optimized for *correctness properties* — determinism, resumability, mixture fidelity — not for throughput. This is the single most important framing difference between text and multimodal data infrastructure, and it is worth saying out loud in an interview because it explains why the text loader is a state machine rather than a pipeline.

**Storage tier:** standard object storage (S3 Standard / GCS Standard) with a per-host local NVMe cache. No need for a parallel filesystem, no need for hot tiers. Prefetch depth of 2–3 shards per rank hides object-store latency (~50 ms p99) completely at these rates.

### 8.5 Total storage footprint

| Artifact | Size | Monthly cost @ $20/TB |
|---|---|---|
| Raw WARC (compressed) | 30 PB | $600K |
| Extracted text (zstd) | 0.45 PB | $9K |
| Metadata / provenance / scores | 0.30 PB | $6K |
| MinHash signatures + band index | 0.05 PB | $1K |
| Embeddings (if semantic dedup) | 0.05 PB | $1K |
| Tokenized shards, current version | 0.12 PB | $2.4K |
| Tokenized shards, 3 retained prior versions | 0.36 PB | $7.2K |
| Synthetic data (text + tokenized) | 0.03 PB | $0.6K |
| **Total** | **≈31.4 PB** | **≈$627K/month = $7.5M/yr** |

**Multiplication factor over raw: 1.05×.** This is the most reassuring number in the document and it has a single cause: **raw crawl dwarfs everything derived.** All the derived artifacts combined are 4.6% of the raw. Which means the storage optimization that matters is not "compress the tokenized shards" — it is "do we need 30 PB of raw WARC online, or can 80% of it live in Glacier-class storage at $1/TB-month?"

**Tiering decision:** keep the most recent 12 months of crawl (≈8 PB) in standard storage; archive the rest. 22 PB at $1/TB-mo = $22K instead of $440K. **Saves $418K/month — $5M/year — and is the single largest cost optimization available anywhere in this pipeline.** The cost is retrieval latency (hours) when you need to reprocess old crawl, which happens roughly once per model generation and can be scheduled. Do this.

---
## 9. Online loading path

**Receives from §8:** immutable shards + manifest + recipe. **Exposes to the trainer:** batches of `(tokens[B, L], cu_seqlens[])` plus a resumable position token. Contract: **the loader is a pure function of `(recipe_id, manifest_hash, seed, global_step, dp_rank, dp_world_size)`.** Nothing else. No mutable state, no coordination, no central sampler service.

That purity is the entire design, and every requirement below falls out of it.

### 9.1 Architecture

```
   MANIFEST (immutable, 50 MB)  +  RECIPE (immutable)  +  SEED
                        │
                        ▼
   ┌──────────────────────────────────────────────────────────┐
   │  DETERMINISTIC MIXTURE SAMPLER   (pure function, no I/O)  │
   │  index i  →  (source_id, shard_id, offset_in_shard)       │
   └──────────────────────────────────────────────────────────┘
                        │  per dp_rank: i ∈ {rank, rank+W, rank+2W, ...}
                        ▼
   ┌──────────────┐   prefetch 3 shards    ┌────────────────────┐
   │ local NVMe   │ ◄───────────────────── │  object store      │
   │ cache 200 GB │                        │  120 TB, 1e6 shards│
   └──────┬───────┘                        └────────────────────┘
          │  mmap + slice, zero copy
          ▼
   ┌──────────────────────────┐
   │ shuffle ring buffer 10^4 │  328 MB RAM
   └──────┬───────────────────┘
          ▼
   pack → (tokens[B,L], cu_seqlens[])  →  pinned buffer → H2D
```

### 9.2 Deterministic mixture sampling, and proving the realized mixture matches the recipe

**Mechanism.** Naive per-step multinomial sampling from the recipe weights is wrong for two reasons: it doesn't reproduce under a different `dp_world_size`, and it has O(√n) variance so the realized mixture drifts from the target within any finite window.

Use a **stratified deterministic sequence** instead:

```
  For global sample index i:
      j       = PRF(seed, i)                 # keyed pseudorandom permutation, e.g. Philox
      src     = inverse_cdf(recipe.weights, j / 2^64)
      k       = PRF(seed ^ src, i)           # independent stream per source
      pos     = k mod source_total_positions[src]
      shard, offset = manifest.locate(src, pos)
```

Properties this buys you:
- **Reshard-invariant.** Rank r takes indices `{r, r+W, r+2W, …}`. Change W from 12,500 to 10,000 and the *set* of samples consumed by step N is identical; only the assignment to ranks changes. This is what makes elastic restart possible.
- **Bounded mixture error.** With a keyed permutation over a stratified index rather than i.i.d. draws, the realized fraction of each source over any window of n samples deviates by O(1/n) rather than O(1/√n). Over a 4M-token batch, the mixture is correct to ~5 decimal places.
- **Epoch accounting is arithmetic.** `epochs_consumed[src] at step N = (N × weight[src]) / source_unique_tokens[src]`. No counters, no state.

**Proving the realized mixture matches the recipe.** Three independent checks, because this is exactly the kind of bug that silently costs you a model generation:

1. **Offline replay.** Run the sampler for the full 30T-token schedule with no I/O (it's a pure function; ~10 minutes on one machine), histogram the source IDs, diff against the recipe. **This is a unit test, and it should gate every recipe merge.** Tolerance: 1e-4 relative.
2. **Online counter.** Each rank increments a per-source token counter; all-reduce every 1,000 steps; assert against the recipe within 1e-3. Costs one 16-element all-reduce per 1,000 steps — free. This catches manifest corruption and shard-skip drift that the offline replay can't see.
3. **Post-hoc audit.** The training log records `(step, source_histogram)`; the final dataset card reports realized vs target weights per phase. This is what you show when someone asks six months later whether the model actually saw 16% code.

The reason all three exist: (1) catches recipe bugs, (2) catches runtime bugs, (3) catches the case where someone changed the manifest mid-run.

### 9.3 Resumability and exact token-position recovery

Because the sampler is a pure function of `global_step`, **the checkpoint's data state is one integer.** Not a shard list, not a shuffle buffer dump, not an iterator state — one integer.

```
  checkpoint.data_state = {
      global_step:     412_337,
      recipe_id:       "pretrain-v7.2",
      manifest_hash:   "sha256:9f3c…",
      seed:            20260731,
      skipped_shards:  [17_442, 391_006],   # see 9.5
  }
```

Restart procedure: read `global_step`, re-derive every index from it, resume. Works under:
- **Elastic restart with a different world size.** Sample *set* is invariant (§9.2); only the rank assignment changes. Note the caveat: the shuffle ring buffer contents differ after a reshard, so the *order* within a small window changes. This is a real but negligible difference — you have not re-shown or skipped any token, only reordered within ~10⁴ samples. If bit-exactness across a reshard is a hard requirement, drain and disable the ring buffer for the boundary step.
- **Different topology / parallelism strategy.** The loader knows only `dp_rank` and `dp_world_size`; tensor/pipeline/expert parallel degree is invisible to it. Changing TP from 8 to 4 doesn't touch the data path.
- **Mid-epoch restart.** There is no epoch boundary in the sampler; `global_step` is the only clock.

**The alternative and why it loses.** The common design is a stateful sampler that checkpoints its RNG state, its shard queue, and its buffer. It is simpler to write and it breaks on every reshard, because the RNG state is entangled with the world size. Teams then bolt on "reshard means restart the epoch," which silently re-shows data and corrupts epoch accounting. The pure-function design costs a day of extra work at build time and eliminates an entire category of incident.

### 9.4 Epoch accounting per source

With per-source epoch caps in the recipe (§7.4), the sampler enforces them structurally: once `pos` would exceed `max_epochs[src] × source_total_positions[src]`, that source's weight is redistributed. To keep this deterministic and reshard-invariant, **the redistribution schedule is precomputed from the recipe at build time** — a list of `(step_range, effective_weights)` — rather than decided at runtime. Runtime decisions based on runtime state are how purity dies.

### 9.5 Skip-and-continue on bad shards

A shard fails its sha256, or the object store returns a persistent 500, or a shard is discovered to contain a contamination leak mid-run. You cannot stop a 100k-accelerator run for this.

```
  Detection:  loader verifies sha256 on first read of each shard (once, cached)
  On failure:
    1. Rank logs (shard_id, error) to the run's control plane
    2. Rank substitutes: shard' = manifest.next_shard_same_source(shard_id, seed)
       — deterministic successor, so all ranks that would hit this shard agree
    3. Control plane broadcasts shard_id into the run-wide skip set
    4. skipped_shards[] enters the next checkpoint
  On restart: skip set is replayed from the checkpoint, so the substitution
              is reproduced exactly.
```

**The subtlety worth calling out:** substitution must be deterministic and *recorded*, not random. If rank 3 quietly reads a different shard than the one the sampler specified and nobody writes it down, the run is no longer reproducible and the mixture audit (§9.2) will show a discrepancy nobody can explain. The skip set in the checkpoint is what preserves the purity guarantee in the presence of failures.

**Budget:** if more than 0.1% of shards enter the skip set, halt the run — that indicates systemic corruption, not bad luck.

> ### ⬛ Defend this in 60 seconds: the loader is a pure function
>
> "The whole loading path is designed around one property: the loader is a pure function of recipe, manifest hash, seed, global step, and data-parallel rank. Nothing else. That means the data state in a checkpoint is **one integer** — the global step — instead of an RNG state, a shard queue, and a buffer dump. And it means a reshard from 12,500 hosts to 10,000 doesn't change *which* samples get consumed, only which rank consumes them, so elastic restart is free instead of being an incident. I can afford this design because text loading is trivially cheap: **30 trillion tokens over 30 days is 46 megabytes per second aggregate, about 30 kilobytes per second per host** — a video run at the same accelerator count needs on the order of 300 gigabytes per second. So there is zero throughput pressure and I spend the entire design budget on correctness: determinism, exact-position resumability, and a mixture audit that proves the realized mix matches the recipe to one part in ten thousand."

---

## 10. Lifecycle: continuous operation & versioning

### 10.1 Incremental processing of new snapshots

A new snapshot (≈3e9 documents) lands every 4 weeks. The incremental path:

```
  Day 0   new snapshot lands              3.0e9 docs raw
  Day 0-1 extraction (§2)                 4 h on 20k cores       $16K
  Day 1   langID + heuristics             1 h                    $1K
  Day 1   intra-snapshot MinHash          30 min                 $2K
  Day 1   cross-snapshot bloom probe (§3.3)  1 h                 $3K
                                          → 1.1e9 survive
  Day 1   quality classifier              20 min on 500 GPUs     $350
  Day 1   decontamination                 15 min                 $10
  Day 2   append to corpus, update band index, emit yield report
  ────────────────────────────────────────────────────────────
  Total: ~2 days, ~$22K per snapshot, ~250B new tokens
```

Two days and $22K per snapshot. **The reason to care about this number is iteration speed, not cost:** if incremental ingest took two weeks, you could not respond to a discovered data problem within a model generation.

**Frozen manifests are the versioning primitive.** A "dataset version" is not a copy of data; it is a manifest — a list of shard paths, hashes, and source totals — pinned to a corpus snapshot hash. Creating a new version costs the manifest write (50 MB). Data is append-only and never deleted; versions differ by which data they *reference* and by the filter/classifier versions used to produce the derived fields.

### 10.2 Freshness and what a "knowledge cutoff" actually is

**Operationally, a knowledge cutoff is a snapshot-selection predicate in the manifest builder.** It is not a training-time switch, it is not a post-training behavior, and it is not a single date. Concretely:

```
  manifest_filter:
    crawl:     fetch_ts <= 2026-06-30
    licensed:  contract_effective_range overlaps [.., 2026-06-30]
    code:      commit_ts <= 2026-06-30
    synthetic: generator_model_cutoff <= 2026-06-30   # ← the one people miss
```

That last line is the subtle one: **synthetic data generated by a model with a later cutoff smuggles post-cutoff knowledge into your corpus.** If your generator was trained on data through August and you claim a June cutoff, the claim is false. Track the generator's cutoff as a property of every synthetic document.

**Recency weighting.** Fresh data is disproportionately valuable (the model should know about recent events) and disproportionately risky (less time for the web to correct errors, more likely to be AI-generated slop, and — increasingly — more likely to be *your own model's output*, which is the self-consumption loop in §6.3 arriving through the front door).

`[current 2025–26]` The practical scheme is to upweight the most recent snapshots in the **mid-training and annealing phases** rather than in the main phase — Llama-3-lineage practice explicitly places "recent" data in the annealing stage. Concretely: main phase samples crawl uniformly across all snapshots; annealing upweights the last 6 months by ~3×. Rationale: recency matters most for the knowledge the model retrieves most readily, and the late-stage placement is where retrieval-readiness is highest.

`[emerging/contested]` **AI-generated web content is now a first-class filtering problem.** A growing fraction of new crawl is model output. There is no reliable detector at scale and the classifier-based detectors have unacceptable false-positive rates on non-native-English human writing. The mitigations available: prefer pre-2023 snapshots for the "high-quality general web" slice, weight domains with strong editorial provenance higher, and treat the freshness upweight as a knob you tune down over successive generations. I have not found a frontier report that states a policy here; treat this as an open problem, and say so if asked.

### 10.3 Backfill when a filter, classifier, or tokenizer changes

| What changed | Must reprocess | Cost | Wall clock | Decision |
|---|---|---|---|---|
| Mixture weights | Nothing — rebuild manifest | $0 | minutes | Always |
| Quality **threshold** | Nothing — scores are stored | $0 | minutes | **Always store raw scores, never just the pass/fail bit** |
| Quality **classifier** | Re-score 34e9 docs | $7K | 2 h | Do it |
| Heuristic filter rules | Re-run filters on extracted text | $5K | 3 h | Do it |
| Extractor | Re-extract from WARC | $222K | 5 days | Only at a generation boundary |
| Dedup parameters | Re-MinHash + re-shuffle | $60K | 8 h | Do it |
| Tokenizer | Re-tokenize + re-shard | $5K | 12 h | Do it (§7.1) |
| Legal/opt-out change | Rebuild manifest with exclusion | $0 | minutes | Always |

**The design rule this table encodes: store the continuous value, not the decision.** Store the classifier score, not `passed`. Store the dup cluster ID and size, not `is_duplicate`. Store the license ID, not `allowed`. Every stage that stores a decision instead of a value converts a $0 manifest rebuild into a multi-thousand-dollar reprocess. This one principle is worth more than any individual optimization in this document.

Only the extractor is genuinely expensive to change, because it is the one stage upstream of all derived fields. That is the argument for spending the extra $176K on trafilatura in §2.1 rather than economizing there: you don't want to revisit that decision.

### 10.4 Reproducing a months-old run bit-for-bit

Required pins, all recorded in the run config:

```
  manifest_hash        sha256 of the manifest (which pins every shard sha256)
  recipe_id            + its content hash
  tokenizer_hash
  seed
  sampler_version      the code version of the deterministic sampler
  skipped_shards[]     from the final checkpoint
```

Given those six things, the token stream is bit-identical, **regardless of world size, topology, or hardware.** That is the payoff of §9's purity constraint.

What breaks reproducibility in practice, in order of frequency:
1. **Data was deleted.** Prevented by append-only + manifest-based exclusion (§3.3, §4.4). Never delete.
2. **The sampler code changed.** Prevented by versioning the sampler and refusing to load a checkpoint whose `sampler_version` doesn't match, unless explicitly overridden.
3. **A shard was rewritten in place.** Prevented by per-shard sha256 in the manifest, verified on read (§9.5).
4. Non-determinism in the trainer (kernel nondeterminism, reduce order) — outside data-pipeline scope, but worth stating that data reproducibility is necessary and not sufficient for run reproducibility.

**The legal-deletion exception.** "Never delete" collides with contract expiry and legal takedown. The resolution: deletion is a *physical* operation that is decoupled from *logical* reproducibility. When data must be physically deleted, the manifest entry is retained as a tombstone recording that the shard existed, its hash, its size, and why it was removed. The run is no longer bit-reproducible, and the tombstone is the honest record of why. This is strictly better than silent deletion, which produces an unexplainable reproduction failure two years later.

---

## 11. Observability & data quality in production

### 11.1 Per-stage yield dashboard

The single most useful dashboard in the system. One row per stage per snapshot, with the *retention percentage* as the primary metric and an alert on deviation from the trailing median.

| Stage | Docs in | Docs out | Retention | Typical | Alert if |
|---|---|---|---|---|---|
| Fetch → unique | 1.0e12 | 150e9 | 15.0% | 14–17% | outside ±3pp |
| Extraction success | 150e9 | 149e9 | 99.3% | >99% | <98% |
| langID (target langs) | 149e9 | 90e9 | 60.4% | 58–63% | outside ±4pp |
| Exact dedup | 90e9 | 68e9 | 75.6% | 70–80% | outside ±6pp |
| MinHash near-dedup | 68e9 | 34e9 | 50.0% | 45–55% | outside ±6pp |
| Heuristic filters | 34e9 | 27e9 | 79.4% | 78–82% | outside ±3pp |
| Quality classifier | 27e9 | 9.5e9 | 35.2% | 33–37% | outside ±3pp |
| Safety + PII | 9.5e9 | 9.45e9 | 99.5% | >99% | <99% |
| Decontamination | 9.45e9 | 9.36e9 | 99.0% | >98.5% | <98% |
| **End-to-end (docs)** | 1.0e12 | 9.36e9 | **0.94%** | | |
| **End-to-end (post-extraction bytes)** | 1.5 PB | 95 TB | **6.3%** | 5–7% | outside ±1.5pp |

**Retention deviation is your best early-warning signal, and it fires before anything else does.** A langID retention drop from 60% to 45% means an encoding regression in extraction. A dedup retention *rise* from 50% to 70% means the band index failed to load and you are about to append 20 billion duplicates. A classifier retention drop means the classifier was silently swapped or the score distribution shifted. Every one of these is a several-million-dollar mistake caught by a percentage on a dashboard.

Also track, per snapshot: token count by source and language, mean document length, mean quality score distribution (not just the pass rate — the full histogram), top 1,000 hosts by surviving token count, and new-host rate.

### 11.2 Sample-level lineage: training token → source URL

Required capability: given `(recipe_id, seed, global_step, dp_rank, position_in_sequence)`, return the source URL.

```
  1. global_step, rank, position → sample index i          (§9.2 arithmetic)
  2. i → (source_id, shard_id, offset)                     (sampler, pure fn)
  3. shard_id → shard.idx, binary search offset → doc ordinal
  4. (shard_id, doc ordinal) → doc_id                       (shard .meta)
  5. doc_id → provenance row → url, fetch_ts, license, derived_from
  6. if derived_from is non-null (synthetic), recurse to step 5
```

Cost: two object reads and one metadata point-lookup. **Sub-second, no scan.** This falls out of the pure-function sampler for free — a stateful sampler makes step 1 impossible, which is the strongest practical argument for the design in §9.

You will use this exactly when it matters most: a loss spike at step 412,337, and you need to know what the model was looking at, in the next ten minutes rather than the next three days.

### 11.3 Detecting and responding to a data-caused loss spike

**Signature discrimination — how to tell data from everything else:**

| Cause | Signature |
|---|---|
| **Data** | Spike localized to specific ranks/steps; reproducible on replay of those samples; grad norm spikes with the loss |
| Numerical (fp8/bf16 overflow) | Correlated with a specific layer's activation stats; reproducible without the data |
| Hardware (silent data corruption) | Single node; not reproducible on replay elsewhere |
| Optimizer / LR | Global, gradual onset, correlates with schedule inflection |

**Response runbook:**

```
  T+0    Loss spike detected (>4σ over trailing 1k steps)
  T+1m   Log which dp_ranks saw elevated loss; record (step, rank) pairs
  T+3m   Lineage lookup (§11.2) → the ~500 documents involved
  T+5m   Human reads 20 of them.
         Typical findings: a base64 blob, a 200k-token repeated character
         run, a corrupted-encoding document, a single non-target-language
         shard, or a pathological code file
  T+10m  Decision:
           localized to ≤5 shards  → add to skip set, continue (§9.5)
           systemic (a whole source is bad) → pause, rebuild manifest
                                              excluding source, resume
           model already damaged   → roll back to last good checkpoint
                                     (typically ≤2h of training = ~$400K)
  T+1h   Root-cause the pipeline stage that let it through; add a filter;
         add a regression test to the shard-build validation suite
```

**Shard-build-time validation is what keeps this rare.** Before a shard is admitted to a manifest, assert: max repeated-token run < 1,000; token entropy > 2.0 bits; no single token > 30% of the shard; language distribution within tolerance for the declared source; token count matches the index. These five checks cost microseconds per shard and catch the overwhelming majority of the documents that cause loss spikes.

### 11.4 Benchmark regression mid-run

Distinguish three cases, because the responses are completely different:

1. **Contamination (score too *high*).** Take the benchmark items the model gets right, run the §5.3 fuzzy pipeline against the *realized* token stream for this run specifically (not the corpus — the run), and check whether the model's per-token loss on the benchmark items is anomalously low relative to comparable held-out text. See follow-up Q4.
2. **Mixture drift (score too *low* on one domain).** Check the online mixture counters (§9.2). If realized code fraction is 12% instead of 16%, you have a manifest or skip-set bug.
3. **Genuine capability regression.** Compare against the proxy-model ablation predictions from §7.3. If the 8B integration run predicted this and you shipped anyway, that's a recipe problem, not a pipeline problem.

---

## 12. Placement, colocation, and egress

Physical placement is a first-class cost decision and the easiest large mistake to make.

```
  REGION A (cheap power, cold climate)     REGION B (accelerator capacity)
  ┌────────────────────────────────┐       ┌──────────────────────────────┐
  │ raw WARC object store  30 PB   │       │  TRAINING CLUSTER 100k accel │
  │ CPU extraction fleet   100k co │       │  ▲                           │
  │ dedup shuffle          45 TB   │       │  │ 120 TB tokenized shards   │
  │ GPU scoring fleet      2k GPU  │       │  │ read at 370 MB/s          │
  │ synthetic gen fleet    5k GPU  │       │  └── local object store      │
  └───────────┬────────────────────┘       └──────────────────────────────┘
              │
              └──────── 120 TB per dataset version ────────►
                        egress @ $20/TB = $2.4M... no.
```

**The rule: everything upstream of tokenization colocates with the raw data; only the tokenized shards cross regions.** The arithmetic:

| What crosses regions | Volume | Egress cost @ $20/TB |
|---|---|---|
| Naive: extracted text to the training region | 450 TB | $9.0M |
| Naive: raw WARC to the training region | 30,000 TB | **$600M** (absurd) |
| **Correct: tokenized shards only** | 120 TB | **$2.4M per full transfer** |
| Correct + incremental (only new/changed shards) | ~15 TB/version | **$300K per version** |
| Best: negotiate a private interconnect or same-provider transfer | 120 TB | $0–$200K |

**$2.4M for one full dataset transfer is a top-five cost line in this entire document** — larger than extraction, dedup, filtering, synthetic generation, and tokenization *combined*. It is also entirely avoidable. Three mitigations, in order of preference:

1. **Colocate the training cluster with the data.** Usually impossible — accelerator capacity dictates region, not you.
2. **Ship deltas, not full datasets.** Because shards are content-addressed and immutable, a new dataset version shares most shards with the previous one. Only new and changed shards transfer: ~15 TB per version = $300K.
3. **Rebuild in the training region from a compact intermediate.** Ship the *filtered, pre-tokenization* text (95 TB, $1.9M) — worse. Or ship nothing and run tokenization in the training region from a locally-cached filtered corpus — this only helps if you retokenize often, which you don't.

**Take option 2 and make shard immutability a hard invariant precisely so that delta shipping works.** This is a case where an architectural property adopted for correctness reasons (§9's purity) pays for itself in dollars.

---

## 13. The hardest interviewer follow-ups

### Q1. "How do you dedup 100 billion documents without O(n²)?"

You never compare pairs. You convert similarity search into an exact-match sort, which is O(N log N).

Three steps: (1) MinHash reduces each document from thousands of shingles to 112 integers, and the min-hash collision probability *is* the Jaccard similarity, so the signature is an unbiased similarity estimator. (2) Banding — split 112 hashes into 14 bands of 8 — turns approximate similarity into exact equality: two docs are candidates iff some 8-hash band matches exactly. That's a group-by, which is a distributed sort. (3) The sort is 14 × 90e9 × 36 bytes = **45 TB**, about 4–6 hours on a 2,000-node cluster.

The threshold is `(1/b)^(1/r) = (1/14)^(1/8) = 0.72`, and the S-curve is sharp: 92% catch rate at Jaccard 0.8, 5% at 0.5.

Two things that actually break in production, which is what the question is really probing: **hot buckets** — a template used by 50M sites produces one bucket with 50M members and 1.25e15 implied pairs, so you cap bucket size at 10⁴ and split by secondary hash; and **global union-find**, which you avoid by computing connected components per bucket and doing a single global cluster-ID merge pass.

For the continuous case, you don't re-dedup: you keep a persistent band index (11 TB of SSTables) fronted by bloom filters, and each new snapshot probes it. LSHBloom reported a **12× speedup** from exactly this substitution.

### Q2. "The tokenizer changes after 60% of the data is processed. Walk me through it."

First, disambiguate, because the two readings have answers that differ by four orders of magnitude.

**If 60% of *data prep* is done:** retokenize everything. BPE tokenization of the full 95 TB corpus is **8,800 core-hours, $176**. The real cost is rewriting and re-shuffling 120 TB of shards — about 12 hours and $5K. Do it without discussion. Retain the old shards for one version so any in-flight experiment isn't invalidated (+$2.4K/month).

**If 60% of *training* is done:** you cannot mix tokenizers within a run — the embedding table and the token IDs are fixed at initialization. Options are (a) finish with the old tokenizer, (b) restart, costing ~$90M of burned compute, or (c) `[emerging/contested]` embedding-surgery approaches that map the old vocabulary into the new one and continue. The answer is almost always (a) unless the tokenizer change fixes something that would render the model unusable — e.g. it can't represent a target language's script, or a critical token is missing.

The design lesson to volunteer: this is why the tokenizer is **hash-pinned in the recipe** and why tokenizer changes are gated to generation boundaries. And it's why lazy loader-side tokenization is worth considering — it needs about **16 cores across the whole 100k-accelerator fleet** — with the honest caveat that it complicates exact-position resumability because shard token counts aren't known without a pre-pass.

### Q3. "How do you know your mixture is right before spending the full training budget?"

You don't know; you bound the risk for about 1% of the budget.

Layered, from cheapest: (1) **per-source micro-anneals** — 1B models, 20B tokens each, one per candidate source, embarrassingly parallel, ~$2K apiece. This is Olmo 3's production method and it answers "is this source worth including" rather than solving for a global optimum. (2) **Regression over proxy mixtures** — RegMix trained **512 models at 1M parameters** and predicted the best mixture for models 1000× larger at 2% extra FLOPs, beating human selection by 6.3% on HellaSwag. (3) **Integration runs** — 6 × 8B × 100B tokens, which catch interactions the per-source ablations miss. (4) **One scale-validation run** at 30B × 500B tokens to check that the mixture ranking survives the scale gap.

Total ≈ **$1.26M against a $150M run**, and the thing it's protecting against is RegMix's measured **14.6% swing on individual tasks** from mixture alone.

Two limits I'd state without being asked. First, **proxy rankings are not scale-invariant**: 2026 work on capacity-aware mixture laws finds the optimal knowledge-data fraction *rises* with model size while math and code *fall*, so a 1B proxy systematically over-weights math and code. Second, **token horizon matters as much as model size**: Nemotron-CC's central finding is that filters and mixtures tuned at short horizons over-filter badly at 15T tokens — they got **+5 MMLU over Llama 3.1 at 15T** specifically by preserving 4× more unique tokens. So ablate at a horizon proportional to the real run, or your ablation will lie to you in a predictable direction.

### Q4. "A benchmark jump is suspected to be contamination. Prove or disprove it."

Five tests, cheapest first, and none of them is conclusive alone.

1. **Exact match against the realized stream.** Not the corpus — the actual tokens this run consumed, reconstructed from `(recipe, seed, manifest, skip set)`. 13-gram bloom probe, ~2.4 GB in RAM. If you find verbatim items: proven, done.
2. **Per-token loss comparison.** Compute the model's loss on the benchmark items versus on held-out text of matched domain and difficulty. Memorized text has *characteristically* low loss — sharply lower than the model's general competence predicts. A model that genuinely learned the skill has normal loss on the item and gets it right anyway.
3. **Canary check.** Evaluate on the time-forward held-out slice from §1.6 — data crawled *after* the cutoff, which cannot be in training. If the jump replicates there, it's real capability.
4. **Perturbation test.** Rewrite the benchmark items — change names, numbers, and surface form while preserving the reasoning. Contaminated models collapse; capable models don't. This is the strongest single test, and it's cheap: an LLM does the rewriting.
5. **Fuzzy/embedding sweep.** Reuse the §3.4 embeddings to find paraphrase-level matches the n-gram filter missed.

**What I'd say about the outcome:** if 1 and 5 come back clean and 3 and 4 replicate the gain, I'd report the gain as real *with the contamination audit attached*. And I'd note the structural point — you can't fully disprove contamination on a public benchmark, which is why the time-forward held-out slice is carved at ingest in the first place. That carve is the only test in this list that is robust by construction.

### Q5. "You're data-constrained. Repeat high-quality data or add lower-quality fresh data?"

Both, and the split is quantifiable.

Muennighoff et al. established that repetition is nearly free up to ~4 epochs — the value of repeated tokens decays exponentially toward a ceiling, and under data constraints you should scale epochs *faster* than parameters, which directly contradicts Chinchilla. Beyond ~4 epochs returns collapse and test loss can rise mid-training.

So: **repeat the scarce high-value data up to 4 epochs, then add fresh lower-quality data for the remainder.** Concretely in my recipe, top-5% web gets 4 epochs, math and science get 4, code gets 2, books get 3, and the mid-quality web tail gets 1. That yields 31T presented tokens from 22.5T unique.

But there's a third option that dominates both at our filter strength, and it's the 2025 answer rather than the 2023 one: **regenerate.** Nemotron-CC's whole thesis is that DCLM and FineWeb-Edu discard up to **90%** of the data, which makes them unusable for 15T-token horizons — and their fix is 2 trillion tokens of synthetic rephrasing to recover it. At **$0.06 to $0.24 per million tokens**, generating a trillion fresh-ish tokens costs $60K to $240K. Compared to a fifth epoch on your best data, or to admitting genuine slop, that is obviously the right move. The honest caveat: a 2026 systematic comparison found synthetic corpora land within **0.28 macro-average points of DCLM** — parity, not superiority. Synthetic buys you *quantity at quality*, which is precisely what data-constrained scaling needs; it does not buy you better data than the good real data you already have.

One thing I would not do: repeat synthetic data. It's already a lossy re-expression of real data that's also in the mix, so repeating it compounds the generator's bias without adding information. Generate more instead.

### Q6. "A loss spike is traced to data. Walk me through the investigation."

The whole investigation is ten minutes because of a design decision made much earlier: the loader is a pure function of `(recipe, manifest, seed, global_step, rank)`.

Step 1, discriminate the cause. Data spikes are **localized to specific ranks and steps and reproduce exactly on replay**; numerical spikes correlate with a layer's activation statistics and reproduce without the data; hardware corruption hits one node and doesn't reproduce elsewhere.

Step 2, lineage. Feed `(step, rank)` into the sampler in reverse: sample index → shard and offset → doc ordinal via the shard index → `doc_id` → provenance row → source URL. Two object reads and a point lookup, sub-second, no scan. That gives me the ~500 documents those ranks consumed.

Step 3, read twenty of them. In my experience the finding is one of five things: a base64 or hex blob that survived extraction, a document with a 200,000-token repeated-character run, a mojibake encoding failure, a shard that's entirely the wrong language, or a pathological minified code file.

Step 4, decide. If it's confined to five shards or fewer, add them to the skip set and continue — the skip set is deterministic and checkpointed, so the run stays reproducible. If a whole source is bad, pause, rebuild the manifest excluding it, resume. If the model is already damaged, roll back; two hours of training is about $400K, which is worth it against the alternative.

Step 5, close the loop. Add the failure signature to shard-build validation — max repeated-token run under 1,000, token entropy above 2 bits, no single token exceeding 30% of a shard. Those checks cost microseconds and catch most of this class before it ever ships.

### Q7. "CommonCrawl removes 15% of its archive tomorrow under legal pressure. What breaks?"

Not hypothetical: the News/Media Alliance sent CommonCrawl a formal demand letter on **29 April 2026** — NBCUniversal, CNN, Vox Media, Ziff Davis, USA Today — demanding content removal and enforceable opt-out.

**What doesn't break:** my corpus. I hold my own copy; CC removing content from *their* distribution doesn't reach into my object store. And my own crawl covers roughly 80% of the volume, with its own robots-compliance record at fetch time, which is the defensible artifact.

**What does break, in order:** (1) *Future* snapshots lose that content — a coverage problem, not a corpus problem. (2) If my legal position is that I should honor the removal, I need retroactive exclusion, which is exactly the `retro_optout` host state in §1.5 — a manifest rebuild in minutes, not a reprocess. (3) The models already trained on it stay trained on it; that's a legal and policy question, not an engineering one, and the engineering answer is that per-run manifests let me state precisely which runs contained which sources. (4) Reproducibility of past manifests, if physical deletion is compelled — handled by tombstones (§10.4), which preserve an honest record instead of producing an unexplainable failure later.

**The design principle that makes all of this cheap:** never encode policy as a data operation. Data is append-only; policy lives in the manifest builder. When the policy changes — and it changes every few months now — you rebuild a 50 MB manifest instead of reprocessing 34 billion documents.

I'd also flag the strategic read: publishers blocking AI crawlers went from ~23% in Sept 2023 to ~60% by May 2025, and Cloudflare flipped to block-by-default on 1 July 2025. The open web's availability curve is bending down. That argues for crawling breadth *now*, for weighting the licensed tier and synthetic generation up in future generations, and for treating pre-2023 archives as a genuinely appreciating asset.

### Q8. "Your quality classifier is trained on labels from your previous model. Isn't that a doom loop?"

It's a real risk with a specific shape, and the mitigations are structural rather than clever.

The mechanism: model N labels data → model N+1 trains on the filtered result → model N+1 labels for N+2. Each generation's filter inherits the previous generation's blind spots and amplifies them, and the corpus narrows toward whatever the model family already finds legible.

Four defenses. **First, ground the rubric in human-written criteria, not model preference.** Nemotron-CC scored educational value on an explicit 0–5 rubric with two different labelers — Mistral-8x22B and Nemotron-340B — over the same 460k documents. Multiple independent labelers with an explicit rubric is measurably different from "ask the model what it likes."

**Second, keep an unfiltered control.** I train a proxy model on a fixed unfiltered sample every generation and check that the filtered pipeline still beats it by the expected margin. If the gap narrows across generations, the filter is collapsing and I've measured it rather than guessed.

**Third, monitor corpus diversity directly.** Track type-token ratio, host diversity, topic-cluster entropy, and length distribution per generation. A narrowing corpus is visible in these before it's visible in benchmarks.

**Fourth — and this is the one people skip — never filter the entire corpus with one classifier.** My main mix keeps 16% mid-quality web that the classifier scored *below* threshold, deliberately, as a diversity floor. This is directly supported by Nemotron-CC: their finding is that maximizing classifier score costs you 4× in unique tokens and loses 5 MMLU at a 15T horizon. Aggressive filtering has a measured cost, so keeping a deliberate low-score fraction isn't hedging — it's the empirically better recipe.

The honest limit: none of this is proven over five model generations, because nobody has published that. `[inferred]` — this is my reasoning about a failure mode the field agrees exists and has not yet quantified longitudinally.

---

## 14. Summary of the load-bearing decisions

| # | Decision | Alternative rejected | Key number |
|---|---|---|---|
| 1 | Re-extract from WARC with trafilatura | Use CC WET | $222K, 4.6 days; measurable model-quality gain |
| 2 | Dedup **before** quality filtering | Filter first (3× cheaper shuffle) | Avoids amplifying high-scoring boilerplate |
| 3 | Per-snapshot MinHash at 14×8 (s\*=0.72) | Global dedup | FineWeb: global dedup made the model *worse* |
| 4 | LLM labels 460k docs → distill to 150M encoder | LLM scores the corpus | $7K / 1.8 h vs $378K / 3.9 days (or 33 days at 70B) |
| 5 | Keep a deliberate 16% mid-quality web slice | Maximize classifier score | Nemotron-CC: 4× unique tokens, +5 MMLU at 15T |
| 6 | Staged mixture (main / mid / long-ctx / anneal) | Uniform mixture | OLMo: mid-training = 5–10% of FLOPs |
| 7 | Per-source epoch caps, ≤4 | Single epoch, or global repetition | Muennighoff: ~4 epochs ≈ free; collapse beyond |
| 8 | Loader as a pure function of (recipe, seed, step, rank) | Stateful sampler | Checkpoint data state = one integer; reshard-safe |
| 9 | Store scores, not decisions; policy lives in the manifest | Bake decisions into the data | Filter change: $0 rebuild vs $222K reprocess |
| 10 | Archive crawl older than 12 months to cold storage | All-hot | **Saves $5M/year** — largest single cost win |
| 11 | Ship only tokenized shard *deltas* across regions | Ship full datasets | $300K/version vs $2.4M/version |
| 12 | Eval holdout carved at ingest, write-once, separate account | Carve after processing | The only contamination defense robust by construction |

**If you remember three things:** the pipeline's compute is nearly free relative to training, so filter and curate aggressively; store continuous values rather than decisions so that policy changes are manifest rebuilds rather than reprocesses; and make the loader a pure function so that lineage, reproducibility, and elastic restart all fall out for free instead of each needing its own machinery.

---

## Sources consulted (2025–26)

Nemotron-CC (Su et al., NVIDIA, 2025) and the NeMo Curator integration blog · Nemotron-CC-Math (NVIDIA, 2025) · FineWeb / FineWeb-Edu / FineWeb2 and DataTrove (Penedo et al., HuggingFace) · DCLM (Li et al.) · Olmo 3 / Dolma 3 / Dolma 3 Dolmino (Ai2, Nov 2025) and OLMo 2 / Dolmino Mix 1124 · Scaling Data-Constrained Language Models (Muennighoff et al., JMLR 2025) and follow-on prescriptive scaling work (2026) · RegMix (Liu et al., ICLR 2025) and the 2026 mixture-optimization literature (CLIMB, MixMin, FastMix, CausalMix, capacity-aware mixture laws) · ProLong (Gao et al. 2024) and long-context data-curation surveys (2025–26) · SmolLM2, Yi / Yi-Lightning, EuroLLM-9B · WRAP, BeyondWeb, REWIRE, FinePhrase, and the 2026 systematic synthetic-data comparison · SemDeDup (Abbas et al.) and LSHBloom (Khan et al. 2024) · AI-crawler and robots.txt landscape reporting (Cloudflare July 2025 policy change; News/Media Alliance demand letter, 29 April 2026) · practitioner curation guides (Spheron 2026, Zilliz/Milvus 2025).

Claims sourced to these are tagged inline. Claims tagged `[inferred]` have no source and are my reasoning.
