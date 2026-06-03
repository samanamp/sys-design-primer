---
title: RAG
description: RAG
---

# Internal Knowledge Assistant — ML System Design

*Staff-level design for a permission-aware, cited Q&A system over a large, fast-changing private corpus (docs, wikis, tickets, code, chat, email). ~10s of millions of documents, strict per-user ACLs.*

---

## 1. Research pass — State of the art (2026)

I did a read pass before designing. Key sources: Anthropic's **Contextual Retrieval** (Sept 2024); the 2025–2026 RAG-vs-long-context literature (LightOn "RAG is Dead, Long Live RAG"; RAGFlow's 2025 year-end review; VentureBeat's Q1-2026 RAG infra tracker); **RAFT** (Zhang et al., arXiv 2403.10131, Berkeley/Meta/Microsoft); and the access-control literature (Pinecone/SpiceDB, Oso, Squirro, RheinInsights, and the HONEYBEE RBAC-partitioning paper, arXiv 2505.01538).

**Where the field actually is.** The 2023 baseline — embed chunks, top-k cosine, stuff the prompt — is dead in enterprise. The 2026 consensus:

- **Hybrid retrieval is now table stakes.** Dense (semantic) + sparse (BM25, exact lexical) fused by reciprocal rank fusion, then a **cross-encoder reranker**. VentureBeat's Q1-2026 tracker reports enterprise intent to adopt hybrid retrieval *tripled* (≈10%→33%) in one quarter, and retrieval optimization overtook evaluation as the top investment priority.
- **Contextual Retrieval** is the dominant chunking fix: an LLM prepends a chunk-situating blurb before embedding *and* before BM25 indexing. Anthropic's published numbers: contextual embeddings cut top-20 retrieval-failure rate 35% (5.7%→3.7%), +contextual BM25 → 49% (→2.9%), +reranking → 67% (→1.9%). Their recommended dense:sparse fusion weight is ~4:1.
- **RAG didn't die from long context; it matured into a retrieval *policy*.** With 1M-token windows, "just dump everything" fails at tens-of-millions-of-docs scale (you still must select what to dump) and is 8–82× more expensive per query than retrieval for typical workloads, with worse latency and more distractor noise. Long context changes *how* you chunk (bigger chunks, retrieve generously, let the model sort) but doesn't remove selection.
- **Agentic / conditional retrieval** is the frontier: the model decides *if, what, and how* to retrieve, calls retrieval as a tool, and iterates (self-RAG / CRAG critique-and-correct loops). Agents make far more retrieval calls than humans, which reshapes the serving tier.
- **Fine-tuning has settled into discipline.** The reflex to fine-tune the generator on company facts is recognized as the wrong move; the ROI lives in embedding/reranker domain adaptation and **RAFT** (teach the generator to use noisy context, cite, and abstain) — not in baking knowledge into weights.
- **Access control is the unsolved-in-practice hard part.** The literature converges on early-binding (pre-filter the ANN search with ACL tokens) vs post-filter, late-binding against a live permission service, and dedicated principal/relationship stores (ReBAC, SpiceDB-style).

Direction of travel: retrieval is an attention policy inside an agent loop; the grounding corpus stays the single source of truth; weights stay knowledge-free.

---

## 2. Problem framing, metrics, abstention

**[SIGNAL: scope-and-framing]** Before boxes: what is success, and what happens when the answer isn't there?

**User goals** split into three modes with different retrieval profiles:
1. **Factual lookup** ("what's the VPN config for contractors?") — single authoritative chunk, high precision, abstain if absent.
2. **How-to / procedural** ("how do I request prod access?") — one current runbook; *recency and authority dominate* (a 2021 superseded runbook is worse than nothing).
3. **Synthesis across sources** ("summarize every incident touching service X last quarter") — multi-hop, multi-doc, benefits from agentic/graph retrieval.

**Success metric hierarchy** (retrieval first, because retrieval *is* the system):

| Tier | Metric | Why it's the gate |
|---|---|---|
| Retrieval | recall@k, context-precision, context-recall | If the right chunk isn't retrieved, nothing downstream can recover |
| Grounding | faithfulness / groundedness (claims supported by context) | Confident-but-ungrounded answers cause real harm |
| Answer | answer-relevance, citation-correctness | Every claim traceable to an openable, authorized source |
| Safety | **abstention accuracy** on out-of-corpus questions | "I couldn't find this" must fire when it should |
| Security | **zero unauthorized retrievals** (hard constraint, not a metric to optimize) | One leak = lawsuit-grade incident |

**[SIGNAL: calibrated-abstention]** Calibrated abstention is a *feature*. In enterprise, a confidently wrong answer about a security policy or a customer contract is worse than no answer — someone acts on it. The system must say *"I couldn't find an authoritative source for this"* when groundedness is low or retrieval returns nothing above threshold. This is enforced at generation (groundedness check) **and** at retrieval (if top reranked score < τ, short-circuit to abstain). The eval set therefore deliberately includes **known-unanswerable** questions; abstention recall on those is a first-class number, not an afterthought.

---

## 3. Scope and capacity / cost math

**[SIGNAL: scope-and-framing]** Committed scope: **single-org internal tool** (not multi-tenant product), interactive chat (**<2s to first token**), **answer + cite first; agentic actions later**. Source rollout in difficulty order: **docs + wiki (Confluence/Notion) first** (clean, authoritative) → **Jira tickets** → **code (GitHub)** → **Slack/email last** (worst signal/noise, hardest ACLs, most PII). We do *not* boil the ocean on day one.

**[SIGNAL: capacity-and-cost-math]**

| Quantity | Estimate | Notes |
|---|---|---|
| Documents | 30M | |
| Chunks (avg ~10/doc) | **~300M chunks** | structure-aware; ~500–800 tokens each |
| Chunk text storage | 300M × ~1.5KB ≈ **450 GB** | raw + contextual blurb |
| Embedding storage (1024-dim fp32 = 4KB) | 300M × 4KB ≈ **1.2 TB** | → ~300 GB at int8 / MRL-512 |
| ANN index (HNSW overhead ~1.5×) | **~1.8 TB** dense + BM25 inverted index | sharded across nodes; RAM-resident hot tiers |
| Initial embedding cost | 300M chunks × ~700 tok = 210B tok @ ~$0.02/M ≈ **$4.2K** | one-time; re-embed on model upgrade |
| Contextualization (LLM blurb per chunk) | 300M LLM calls; **prompt-cache the parent doc** | dominant ingestion cost; cache makes it ~10–20× cheaper |
| Daily incremental | ~0.5–2% churn → 1.5–6M chunks/day re-embed | trivial vs full re-embed |

**Latency budget (interactive, <2s to first token):**

```
query understanding (rewrite/expand)   ~150 ms   (small/cheap model or cached)
hybrid retrieve (dense ANN ∥ BM25)      ~120 ms   (parallel, ACL-prefiltered)
ACL late-bind check (bulk)              ~ 30 ms   (permission service, batched)
rerank top-100 → top-15 (cross-enc)     ~200 ms   (GPU cross-encoder)
generation TTFT (stream)                ~600 ms
------------------------------------------------
first token                            ~1.1 s     ✓ under budget
```

Reranking the top-100 is the single highest-ROI quality lever and it fits the budget. We do **not** rerank 1,000 candidates (latency) nor skip reranking (quality collapse).

---

## 4. High-level architecture

```
                          INGESTION (async, CDC-driven)
┌───────────────────────────────────────────────────────────────────────────┐
│ Connectors      Parse        Chunk          Contextualize     Embed   Index │
│ Confluence ─┐   ┌────────┐   ┌──────────┐   ┌────────────┐   ┌─────┐  ┌────┐│
│ Notion     ─┤   │format- │   │structure-│   │ LLM blurb  │   │dense│  │ANN ││
│ Jira       ─┼──▶│aware   │──▶│aware     │──▶│ +parent ctx│──▶│ +   │─▶│ +  ││
│ GitHub     ─┤   │parsers │   │small→big │   │(prompt-    │   │BM25 │  │BM25││
│ Slack/email─┘   └────────┘   └──────────┘   │ cached)    │   └─────┘  └─┬──┘│
│        │                          │         └────────────┘             │   │
│        └──────── ACL EXTRACTION ──┴──── attach allow/deny tokens ───────┘   │
│                  + source authority + effective/expiry dates + content hash │
└───────────────────────────────────────────────────────────────────────────┘
                          │                                    │
                 ┌────────▼─────────┐                ┌─────────▼──────────┐
                 │  Principal store  │                │  Chunk index (dense │
                 │  (groups, ReBAC,  │                │  + BM25 + metadata) │
                 │  live ACL source) │                └─────────┬──────────┘
                 └────────┬─────────┘                           │
                          │            QUERY TIME               │
┌─────────────────────────┼───────────────────────────────────┼─────────────┐
│ user+query              │                                    │             │
│   │                     │                                    │             │
│   ▼                     ▼                                    ▼             │
│ ┌──────────┐   ┌─────────────────┐   ┌──────────────┐   ┌─────────────┐    │
│ │ query     │  │ ACL PRE-FILTER  │   │ hybrid       │   │ rerank      │    │
│ │ understand│─▶│ (allow/deny     │──▶│ retrieve     │──▶│ cross-enc   │──┐ │
│ │ rewrite/  │  │ tokens injected │   │ dense ∥ BM25 │   │ top100→15   │  │ │
│ │ decompose │  │ into ANN+BM25)  │   │ + RRF        │   └─────────────┘  │ │
│ └──────────┘   └─────────────────┘   └──────────────┘                    │ │
│                         ▲                                                 │ │
│              ┌──────────┴──────────┐         ┌──────────────────┐         │ │
│              │ LATE-BIND ACL CHECK │◀────────│ generation +     │◀────────┘ │
│              │ vs live principal   │  passed │ grounding check  │           │
│              │ store (re-verify)   │  chunks │ + citation +     │           │
│              └─────────────────────┘         │ abstain-if-weak  │           │
│                                              └────────┬─────────┘           │
│                                  audit log ◀──────────┴── answer + cites    │
└──────────────────────────────────────────────────────────────────────────-┘
```

The permission filter appears **twice** (pre-filter the search space; late-bind re-verify before generation) — that double-gate is deliberate and explained in §5.

---

## 5. Permission-aware retrieval — the enterprise core

**[SIGNAL: permission-aware-retrieval]** This is the question that separates enterprise RAG from a demo. If the system ever surfaces a chunk the asker can't see — or even *reveals that such a chunk exists* ("I found 5 results but can't show 3") — that's a lawsuit-grade leak. ACLs are not a one-line filter.

**Ingestion-time:** every chunk inherits its source document's ACLs as **allow-tokens** and **deny-tokens** (user IDs + group IDs), plus the document's authority tier, effective/expiry dates, and content hash. Following the enterprise-search pattern (RheinInsights, Azure AI Search security trimming), the query becomes:

```
retrieve(query_vector)  WHERE  (allow ∩ user.principals ≠ ∅)
                          AND   (deny ∩ user.principals = ∅)
```

**Pre-filter vs post-filter — the central tradeoff:**

```
                 PRE-FILTER (early binding)          POST-FILTER
                 ───────────────────────────         ───────────────────────────
 recall          high — ANN searches only            LOSSY — ANN returns top-k
                 the authorized subspace             globally, then drops unauth;
                                                      if user can see 2% of corpus,
                                                      top-100 may yield ~2 visible
 leak risk       low (search never sees forbidden)   "N results hidden" leaks
                                                      existence → must suppress
 cost            ACL filter must be expressible       cheap to query, expensive
                 inside ANN (metadata filtering /     in wasted recall
                 partitioning); hard when ACLs
                 don't partition cleanly
 verdict         DEFAULT for selective corpora        only viable at high hit-rate
```

**[SIGNAL: rejected-alternative]** We choose **pre-filter as default** (Pinecone/SpiceDB's guidance: pre-filter wins on large corpora with low per-user hit-rate — exactly the enterprise case, where a given user can see a small fraction). Post-filter is rejected as primary: with strict ACLs the visible fraction is small, so post-filtering top-k starves recall and forces existence-leak suppression. We *also* reject naive per-user index partitioning (HONEYBEE shows static partitioning blows up storage when documents are shared across many overlapping groups) in favor of **ACL-token metadata filtering on a shared index**, with dynamic partitioning only for the hottest role clusters.

**Late binding — why we check twice.** Index ACL metadata goes stale: a user removed from a project at 09:00 must stop retrieving its docs at 09:00:01, not at the next reindex. So after retrieval+rerank, we **re-resolve the surviving chunks' ACLs against the live principal store** (a SpiceDB/ReBAC-style service, bulk `CheckPermission`) before they reach the generator. Pre-filter uses possibly-slightly-stale index tokens for *recall efficiency*; late-bind uses live truth for *correctness*. Stale index → at worst a chunk is dropped at late-bind (safe failure); it can never leak.

**Hard cases:** nested groups (resolve transitively in the principal store, cache the closure with short TTL), document-inherited permissions (a chunk inherits its space/folder ACL; re-derive on parent ACL change via CDC), and time-bounded access (expiry date in metadata + live check). Deny-tokens always win over allow.

```
removed-from-group event ──▶ principal store updated (live, instant)
                          ──▶ CDC re-stamps affected chunk tokens (async, minutes)
   late-bind check uses LIVE store ⇒ access revoked instantly regardless of index lag
```

---

## 6. Ingestion, parsing, chunking, contextualization

**[SIGNAL: contextual-retrieval]** The classic failure: a chunk reading *"revenue grew 3% over the previous quarter"* doesn't say *which company* or *which quarter*, so it's unretrievable for "ACME Q2-2023 growth." Anthropic's **Contextual Retrieval** fixes this by prepending an LLM-generated situating blurb to each chunk *before* both embedding and BM25 indexing.

```
INGESTION PIPELINE (per source, CDC-triggered)
 connector ─▶ fetch doc + ACLs + metadata
            │
            ▼
 format-aware parse:
   PDF/tables → layout-aware extract (preserve table structure)
   code       → AST-aware split (function/class boundaries, keep symbols)
   Slack/email→ thread reconstruction (group reply chains, strip quoting)
   wiki       → section/heading structure preserved
            │
            ▼
 structure-aware chunk  (NOT fixed 512-token windows)
   small→big / parent-document: index small precise chunks,
   carry pointer to larger parent for generation context
            │
            ▼
 contextualize:  LLM writes 1–2 sentence blurb situating chunk in its doc
   PROMPT-CACHE the parent document  ⇒ ~10–20× cheaper than naive
   blurb prepended to chunk for BOTH embedding AND BM25 index
            │
            ▼
 embed (dense) ∥ tokenize (BM25)  ─▶  index with {ACL, authority, dates, hash}
```

**[SIGNAL: rejected-alternative]** Fixed-size chunking is rejected — it splits tables mid-row, functions mid-body, and Slack threads mid-conversation, destroying retrievability. We use **structure-aware chunking + small-to-big**: index small high-precision chunks for matching, but feed the *parent* section to the generator so it has room to reason. For deep multi-doc synthesis we layer **RAPTOR-style hierarchical summaries** (cluster + summarize bottom-up) so a query can hit a summary node instead of 40 leaf chunks. Tables, code, and threaded chat get **distinct chunkers** — treating a Jira thread like a PDF paragraph is a guaranteed quality loss.

---

## 7. Retrieval architecture — hybrid + rerank (+ multi-vector)

**[SIGNAL: retrieval-is-the-system]** On a real corpus the LLM almost never fails to write a good answer *given the right context* — it fails because the right context wasn't retrieved. So the bulk of the engineering lives here; the generator is the commodity layer.

**[SIGNAL: hybrid-and-rerank]** **Dense alone is insufficient in enterprise.** Semantic embeddings miss exact tokens that matter constantly here: ticket IDs (`PROJ-4821`), error codes (`0x80070005`), code symbols (`getUserACL()`), internal product names, and acronyms. **BM25 nails exact match; dense nails meaning.** We run both and fuse with **Reciprocal Rank Fusion**:

```
                ┌──────────────────────────┐
   query ──────▶│ dense ANN (HNSW)          │──▶ ranked list A ─┐
                │  ACL-prefiltered          │                  │
                └──────────────────────────┘                  ├─ RRF fuse
                ┌──────────────────────────┐                  │  (score = Σ 1/(k+rank))
   query ──────▶│ sparse BM25 (contextual)  │──▶ ranked list B ─┘   weight dense:sparse ≈ 4:1
                │  ACL-prefiltered          │                       │
                └──────────────────────────┘                       ▼
                                                          top-100 candidates
                                                                   │
                                          ┌────────────────────────▼─────────────┐
                                          │ CROSS-ENCODER RERANKER                │
                                          │ scores (query, chunk) jointly —       │
                                          │ full attention, not just cosine       │
                                          │ top-100 → top-15  (~200ms on GPU)     │
                                          └────────────────────────┬─────────────┘
                                                          dedup + authority/recency
                                                          re-weight ──▶ top-8 to LLM
```

**Why the reranker is the highest-ROI lever.** Bi-encoder ANN scores query and chunk *independently* (cheap, scalable, but coarse). A cross-encoder reads query+chunk *together* with full attention — far more accurate relevance, too slow to run over 300M chunks but perfect over the top-100. Anthropic's numbers: reranking takes the contextual-retrieval failure rate from 2.9% to 1.9%. We retrieve broad (top-100) and rerank narrow rather than trusting raw ANN order.

**[SIGNAL: rejected-alternative]** **Multi-vector / late-interaction (ColBERT)** is a strong alternative — token-level matching beats single-vector on hard queries — but it inflates the index ~10–100× (one vector per token) which at 300M chunks is prohibitive. We **reject ColBERT as the primary index** and keep single-vector dense + BM25 + cross-encoder rerank, revisiting ColBERT only for a hard-query subset if rerank plateaus. **ANN choice:** HNSW for recall/latency at the cost of RAM, sharded; IVF-PQ for the cold tier to control the 1.8 TB footprint.

---

## 8. Query understanding

**[SIGNAL: retrieval-is-the-system]** The raw user question is often a poor retrieval query. Cheap fixes, large gains:

- **Rewriting / coreference resolution** — multi-turn "how do I configure *it*?" → resolve *it* from history to "configure the VPN client." Without this, retrieval matches on pronouns.
- **Acronym / jargon expansion** — internal "PRR" → "Production Readiness Review" using a maintained company glossary (also feeds the BM25 side).
- **Decomposition** — multi-hop "who owns the service that depends on X?" → sub-queries (find dependents of X → find owner) executed and fused.
- **HyDE** — generate a *hypothetical* answer doc, embed *that*, retrieve against it; helps when the question's vocabulary differs from the corpus's.
- **Multi-query fusion** — issue 3 paraphrases, RRF the union; cheap recall boost.

```
"how do I configure it for contractors?"  (turn 3)
        │ coreference + expand
        ▼
"configure VPN client for external contractors"  ─┬─▶ retrieval query
        │ decompose (if multi-hop)                 ├─▶ sub-q: contractor VPN policy
        ▼                                          └─▶ sub-q: VPN client setup steps
```

**[SIGNAL: rejected-alternative]** We gate the expensive steps: HyDE and full decomposition fire only when a cheap classifier flags the query as hard/multi-hop. Running them on every "what's the wifi password" wastes latency for zero gain.

---

## 9. Advanced paradigms + long-context-vs-RAG

**[SIGNAL: long-context-vs-rag]** Million-token windows do **not** kill RAG at 30M-doc scale — you still cannot fit the corpus, so you must *select*, and selection *is* retrieval. The 2025–2026 evidence: RAG is ~8–82× cheaper per query than dumping long context, with better latency and less distractor noise. What long context *does* change: we chunk **bigger** and **retrieve more generously** (top-15 parent sections instead of top-5 tight chunks), letting the long-context generator sort signal from noise — "retrieve generously, let the model adjudicate."

**GraphRAG** earns its cost on **relationship / multi-hop** questions: "who owns the service that depends on X, and what incidents hit it last quarter?" Flat chunks can't traverse dependency edges; a knowledge graph (entities = services/people/tickets, edges = owns/depends-on/resolved) can. **[SIGNAL: rejected-alternative]** But GraphRAG is expensive to build and maintain (entity extraction, community detection, graph freshness), so we **reject it as the default index** and deploy it as a *secondary* retriever for the synthesis-mode queries that demonstrably need traversal — not for factual lookup.

**Agentic / iterative retrieval (self-RAG, CRAG):** the model retrieves, critiques whether the context actually answers the question, and re-retrieves with a refined query if not. This is the right tool for complex synthesis and substantially lifts hard-query quality — at the cost of multiple retrieval round-trips and higher latency/spend. We make it **conditional**: single-shot for simple lookups (the majority), agentic loop only when the groundedness check on the first pass fails. This matches the 2026 "conditional over automatic" consensus: the system decides *if/what/how* to retrieve rather than retrieving blindly every turn.

---

## 10. Generation, grounding, citation, contradiction

**[SIGNAL: retrieval-is-the-system]** With the right context retrieved, generation is the easy part — but grounding discipline is non-negotiable.

**Prompt construction:** retrieved parent-sections are presented with explicit source IDs and metadata (title, author, **date**, authority tier). The system instruction: answer *only* from provided context, cite every claim, and if the context doesn't contain the answer, **say so** — do not fall back to parametric knowledge.

**Faithfulness / groundedness:** after generation, a check verifies each claim is supported by a cited chunk (NLI-style entailment or LLM-judge). Ungrounded sentences are stripped or the answer is downgraded to abstention. This is what stops the model answering a security-policy question from its 2024 pretraining instead of the company's actual current policy.

**Citation:** every claim links to a chunk the user can open **and is authorized to see** (the cited source passed the §5 late-bind check by construction). We never cite a document the user can't open — that both frustrates and leaks existence.

**[SIGNAL: messy-corpus-realism]** **Contradiction handling at answer time:** when retrieved chunks disagree (the wiki says X, a recent Slack thread says Y), the system does not silently pick one. It surfaces both with their authority and dates: *"The official policy doc (2024) says X; a more recent engineering thread (last week) suggests Y — confirm with the policy owner."* Authority + recency drive which leads, but the conflict is made visible rather than resolved by coin flip.

---

## 11. The fine-tuning decision tree — the explicit ask

**[SIGNAL: fine-tuning-discipline]** **[SIGNAL: saying-no]** The reflex answer — "fine-tune the LLM on company data" — is the **most common and most wrong** move, and I'm pushing back on it explicitly. It bakes knowledge into weights, which (a) goes stale the instant a doc changes, defeating RAG's entire freshness purpose, (b) is the highest-cost lowest-ROI option, and (c) *increases* confident hallucination. The disciplined answer is a tree, not a reflex:

```
                      ┌─────────────────────────────────────────────┐
                      │ Is the answer quality problem a RETRIEVAL    │
                      │ problem or a GENERATION problem?             │
                      │ (instrument both — see §12)                  │
                      └───────────────┬─────────────────────────────┘
                                      │
            ┌─────────────────────────┴──────────────────────────┐
            ▼ retrieval (the usual case)                          ▼ generation
 ┌──────────────────────────────┐               ┌────────────────────────────────┐
 │ STEP 0 — DON'T FINE-TUNE YET. │               │ Is the base model bad at        │
 │ Exhaust retrieval quality:    │               │ grounded QA over NOISY context  │
 │ hybrid + RRF + rerank +       │               │ (uses distractors, won't cite,  │
 │ contextual retrieval + query  │               │ won't abstain)?                 │
 │ understanding. Most "wrong    │               └───────────┬─────────────────────┘
 │ answers" die here.            │                    yes ▼          ▼ no
 └──────────────┬───────────────┘            ┌──────────────────┐  ┌─────────────┐
                │ still short on recall?      │ STEP 2 — RAFT     │  │ prompt-only │
                ▼                             │ fine-tune the     │  │ + better    │
 ┌──────────────────────────────┐            │ generator: train  │  │ retrieval.  │
 │ STEP 1 — HIGHEST ROI:         │            │ on oracle+        │  │ Done.       │
 │ fine-tune EMBEDDING model     │            │ distractor docs,  │  └─────────────┘
 │ (+ reranker) on in-domain     │            │ teach: cite       │
 │ query–doc pairs mined from    │            │ verbatim, ignore  │
 │ usage logs + golden set.      │            │ distractors,      │
 │ Company jargon/acronyms/code  │            │ abstain when      │
 │ symbols are OOD for general   │            │ answer absent.    │
 │ embedders. Cheap, independent │            └───────────────────┘
 │ of fast-changing knowledge.   │
 └──────────────┬───────────────┘
                │
                ▼
 ┌────────────────────────────────────────────────────────────────────────┐
 │ STEP 3 — FINE-TUNE GENERATOR *KNOWLEDGE*:  ESSENTIALLY NEVER.            │
 │ Baking company facts into weights → stale on first doc change, costly to │
 │ keep current, raises confident hallucination. Knowledge lives in the     │
 │ retrievable corpus, not the weights. RAG's whole point.                  │
 └────────────────────────────────────────────────────────────────────────┘
```

**Costs / criteria per step:**

- **Step 1 (embeddings + reranker):** data = query–doc pairs from click logs + golden set (cheap to mine); compute = small (contrastive fine-tune, hours on a few GPUs); payoff = recall lift on company-specific vocabulary that no general embedder has seen. **Highest ROI; do this first if retrieval plateaus.** Caveat: it triggers a **re-embed migration** (§13).
- **Step 2 (RAFT):** per the RAFT recipe (Zhang et al. 2024), train on questions with **oracle + distractor** passages and chain-of-thought answers that cite verbatim; include a fraction of *answer-absent* examples so the model learns to **abstain**. Warranted only when the base model is measurably weak at grounded, noisy-context QA. Cost = synthetic data generation + SFT + eval + redeploy.
- **Step 3 (generator knowledge):** the criterion to justify it is essentially never met for a fast-changing corpus. Reject.

---

## 12. Evaluation over a private corpus

**[SIGNAL: private-corpus-eval]** There is no public benchmark for *your* corpus — so build one.

**Golden set:** curated `Question → {gold answer, gold source-doc(s)}` triples, sampled across the three query modes, source types, and authority tiers — plus a deliberate slice of **known-unanswerable** questions to measure abstention. SMEs validate. Grow it continuously from production thumbs-down and escalations.

**[SIGNAL: private-corpus-eval]** **Metric stack** (RAGAS-style, automated + LLM-judge):

```
                          GOLDEN SET (Q → answer, source docs, +unanswerable slice)
                                          │
        ┌─────────────────────────────────┼──────────────────────────────────┐
        ▼ COMPONENT (retrieval)            │              ▼ END-TO-END (answer)
 ┌──────────────────────────┐             │      ┌────────────────────────────┐
 │ recall@k                  │             │      │ faithfulness / groundedness │
 │ context-precision         │             │      │ answer-relevance            │
 │ context-recall            │             │      │ citation-correctness        │
 │ (did we fetch gold doc?)  │             │      │ abstention-accuracy         │
 └────────────┬─────────────┘             │      │  (on unanswerable slice)    │
              │                            │      └─────────────┬──────────────┘
              └──── SPLIT THE BLAME ───────┴────────────────────┘
   gold doc retrieved but answer wrong  ⇒ GENERATION failure
   gold doc NOT retrieved               ⇒ RETRIEVAL failure (fix retrieval, not the LLM)
```

**[SIGNAL: retrieval-is-the-system]** That component-vs-end-to-end split is the whole game: it tells you whether a bad answer is a retrieval or a generation failure, which is exactly the input to the §11 fine-tuning tree. Most teams skip it and "fix" the generator when the gold doc was never retrieved.

**LLM-as-judge pitfalls** (use it, don't trust it blindly): position bias, verbosity bias, self-preference. Mitigate with randomized order, calibration against human labels on a held-out slice, and reporting judge–human agreement. **Online:** thumbs up/down, answer-acceptance rate, escalation-to-human rate, and per-source quality dashboards. Offline gates every deploy; online catches drift offline misses.

---

## 13. Freshness, incremental indexing, deletion

**[SIGNAL: freshness-and-deletion]** Internal knowledge changes hourly; nightly full re-embed is unacceptable on correctness *and* cost.

```
SOURCE CHANGE (CDC: webhook / change-feed / poll)
   │
   ├─ CREATE/UPDATE ─▶ re-parse → re-chunk → re-contextualize (parent prompt-cached)
   │                   → re-embed ONLY changed chunks (content-hash diff) → upsert
   │                   → refresh ACL tokens + dates           (minutes end-to-end)
   │
   └─ DELETE/ARCHIVE ─▶ TOMBSTONE immediately in index  ◀── security-critical
                        → hard-purge async
        A deleted-but-still-retrievable doc is BOTH a correctness bug
        AND a data-leak (it can be cited after it was meant to be gone).
        Deletion propagates on the same change-feed as edits, prioritized.
```

**Content-hash diffing** means we only re-embed chunks whose text actually changed, so a 1-word edit doesn't re-embed the doc. New policies are answerable within minutes; archived docs leave the retrievable set immediately (tombstone first, purge later).

**[SIGNAL: re-embedding-migration]** **The re-embed migration.** Upgrading the embedding model (or fine-tuning it per §11 Step 1) **changes the vector space** — old and new vectors are not comparable, so you must **re-embed the entire 300M-chunk corpus**. That's a planned, costly, multi-day migration (~$4K+ compute plus pipeline time), run as a **dual-index blue-green cutover**: build the new index in the background, shadow-eval it against the golden set, then atomically switch reads. This cost is *why* embedding fine-tuning isn't free even though it's "highest ROI" — every retrain pays the migration tax, so we batch model upgrades.

---

## 14. Authority, staleness, contradiction, dedup

**[SIGNAL: messy-corpus-realism]** A system that confidently answers from a 3-year-old deprecated runbook is worse than no system. The corpus is messy by nature, and the design must treat that as first-class:

- **Source-authority ranking:** a tier attached at ingestion — official policy doc > maintained wiki > Jira ticket > Slack hot-take. Authority is a re-ranking signal *after* relevance, so an authoritative-but-slightly-less-similar policy can outrank a chatty exact match.
- **Recency weighting:** effective/expiry dates in metadata; a decay factor down-weights stale chunks. A superseded 2021 runbook with a newer replacement is demoted hard. For how-to queries, recency is near-decisive.
- **Contradiction detection:** at retrieval, flag when top chunks disagree (entailment check across the set); pass the conflict to generation to surface (§10) rather than hide.
- **Deduplication:** the same policy copied into four places wastes the context window on four identical chunks and crowds out diverse evidence. Near-duplicate detection (embedding cosine + MinHash on text) collapses them to the most-authoritative, most-recent copy before the context is assembled.

---

## 15. Serving, caching, cost, multi-tenancy

End-to-end is the §3 budget, streamed (TTFT ~1.1s). Throughput scales by sharding the ANN + BM25 indexes and autoscaling the reranker GPUs (the latency-critical tier).

**[SIGNAL: cache-acl-safety]** **Caching must be ACL-aware or it leaks.** Three layers, each with a permission caveat:

- **Embedding cache** (query → vector): safe, content-only, no ACL concern.
- **Retrieval cache** (query → chunk IDs): **must key on `(query, user-principal-set)`** or be re-filtered through late-bind on read — otherwise user A's retrieval is served to unauthorized user B.
- **Full-answer cache** (query → answer): the dangerous one. A cached answer **must never be served to a user who couldn't have retrieved its sources.** Key on the authorized principal set, and **invalidate on any ACL change or source edit** affecting the cited chunks. A cached answer that outlives a permission revocation is a leak.

**Cost model:** dominant lines are reranker GPU time (per query) and generation tokens (per query); embedding is a one-time + incremental cost. Caching repeated questions (a real pattern in enterprise — "how do I request access?" asked daily) cuts both meaningfully, *if* done ACL-safely.

**Multi-tenancy:** out of scope (single-org, §3). Were it a product, tenant isolation would be a hard index-partition boundary, not just an ACL token — cross-tenant leakage is existential.

---

## 16. Monitoring, governance, feedback loop

**[SIGNAL: permission-aware-retrieval]** **Audit logging** is mandatory *and* itself sensitive: who asked what, what was retrieved, what was cited. It's the forensic trail for "an answer was wrong and someone acted on it," and a compliance requirement — but the log contains confidential retrieved content, so it inherits its own strict ACLs and retention policy.

**Production monitoring** beyond "we'll add monitoring":
- **Groundedness/hallucination monitor** sampling live answers (LLM-judge + spot human review); alert on faithfulness drops.
- **Drift detection:** corpus drift (distribution of new docs), query drift (new topics/acronyms appearing), embedding staleness (golden-set recall declining → schedule a re-embed migration).
- **PII / sensitive-content handling:** detect and redact PII in retrieved context and answers; route sensitive sources through stricter governance.
- **Incident path:** "answer was wrong and acted upon" → trace via audit log to the exact retrieved chunks → classify retrieval vs generation failure → add to golden set → fix the responsible component.

**Feedback loop:** thumbs-down + escalations → triaged → (a) grow the golden set, (b) mine hard query–doc pairs for embedding/reranker fine-tuning (§11 Step 1), (c) flag stale/contradictory sources for SME review. The loop feeds retrieval improvement, which is where quality actually moves.

---

## 17. Tradeoffs taken and what would change them

- **Pre-filter ACLs + shared index** over per-user partitioning — would flip to dynamic partitioning (HONEYBEE-style) if a few role clusters dominate traffic and metadata filtering becomes the latency bottleneck.
- **Single-vector dense + BM25 + cross-encoder** over ColBERT — would add late-interaction for a hard-query subset if reranking plateaus and the index-size budget grows.
- **GraphRAG as secondary, not primary** — would promote it if synthesis/relationship queries become the dominant mode.
- **Conditional agentic retrieval** over always-agentic — would loosen toward more iteration if latency budget relaxes (e.g., async/email-style answering vs interactive chat).
- **Don't fine-tune the generator's knowledge** — nothing changes this for a fast-changing corpus; it's a principle, not a tunable.

---

## 18. What I'd push back on

**[SIGNAL: saying-no]** Three pushbacks on the prompt's implicit framing:

1. **"Whether you'd fine-tune anything"** invites the reflex to fine-tune the LLM on company facts. I'm rejecting that outright (§11) — it's the highest-cost, lowest-ROI, freshness-destroying move. Fine-tuning belongs on the *retriever* (embeddings/reranker) and, only if grounding is weak, on the generator's *skill* via RAFT — never on its knowledge.
2. **"Just use a million-token context"** — rejected at this scale (§9): you still must select from 30M docs, retrieval is ~8–82× cheaper, and long context doesn't give you ACLs, citations, freshness, or deletion. Long context augments retrieval; it doesn't replace it.
3. **Treating permissions as a filter** — the single highest-risk assumption. ACLs are a *double-gated, late-bound* concern (§5), and caching/audit/deletion all inherit the leak risk. If I had to cut scope, I'd cut sources (drop Slack/email) before I'd cut any ACL rigor — a smaller correct system beats a broad one that leaks.