---
title: Recomm system
description: Recomm system
---

Design a real-time recommendation system for a large marketplace. Cover retrieval, ranking, features, training data, serving, and feedback loops.

---

# Real-time Recommendations for a Marketplace

## 1. Scoping

Before designing anything, the things I'd want to nail down with the interviewer. "Marketplace" can mean five different products with five different bottlenecks.

**Vertical and inventory shape.** Products vs services vs housing vs labor. I'll commit to a *general-purpose e-commerce marketplace*, Etsy/eBay-style: many small sellers, heterogeneous SKUs, mostly long-tail, no SKU-level dedup. This matters because it means: (a) brutal cold-start on the item side — millions of new listings/day from independent sellers; (b) we cannot rely on rich structured product taxonomies the way Amazon Retail can; (c) seller-side fairness is a real constraint, not a nice-to-have.

**Scale.** ~100M items in catalog, ~50M DAU, ~500M sessions/day. ~1k QPS p50, 5k p99 for the recs endpoint. This sizing drives every infra choice downstream.

**Surface.** Home feed. I'm picking this specifically because it's the *hardest* — weak intent, no query, the model is doing all the work. Search recs and PDP "similar items" share most of the machinery but with different prior signal. If we crack the home feed, the others fall out as variants.

**Latency budget.** 200ms p99 server-side, ~300ms total wall-clock to first paint. Tight but achievable.

**Mixing.** Organic only for v1. Ads blended in v2 via a separate ad auction whose CTRs are calibrated against the organic ranker. I'll come back to why calibration is what makes that mixing possible.

**Cold start.** Both sides — new buyers (first-session users) and new items (just-listed). I'll assume cold-start is *severe* on the item side because of the long tail.

[STAFF SIGNAL: refusing to design until I've committed to one scenario; the rest of this answer doesn't branch into "well if it were services instead..."]

Things I'll *not* design: search relevance, query understanding, ad auction mechanics, payments, seller onboarding signals. I'll mention them where they interact with recs but not deep-dive.

## 2. Success Metrics

Defined before any model. The metric tree is what keeps you honest later when offline AUC goes up 2pp and online GMV goes down.

**North star:** GMV per session, measured at 28 days post-impression to capture delayed conversions. Not CTR — CTR is hackable by surfacing junk.

**Online proxies** (what we actually look at in A/B tests because GMV has slow signal):
- Click-through rate on recommended slots
- Add-to-cart rate
- Purchase rate per session
- Dwell time on clicked items (proxy for relevance, not just clickbait)

**Guardrails** (any of these regressing kills the launch regardless of north-star):
- Seller concentration (Gini coefficient over impressed sellers) — popularity collapse detector
- Novel-seller exposure rate (% sessions with ≥1 impression from a seller with <30d on platform)
- Refund / complaint rate per impression
- Coverage: % of catalog impressed in a week
- Price-distribution drift (recs collapsing toward only cheap items)

[STAFF SIGNAL: distinguishing north-star from proxies from guardrails before designing — the failure mode is shipping a model that wins CTR and loses GMV, and you don't notice for a quarter]

The offline metric I'd track on the ranker is NDCG@10 on buy-labels, plus calibration ECE per head. Offline AUC on click is almost useless for ranking-quality prediction; I've seen too many wins evaporate.

## 3. High-Level Architecture

```
                    ┌────────────────────────────────────────────┐
   Client ─────────▶│  Edge / API Gateway  (auth, rate-limit)     │
                    └─────────────────┬──────────────────────────┘
                                      │ userId, context, sessionId
                                      ▼
                    ┌──────────────────────────────────────────┐
                    │       Recommender Orchestrator           │
                    │  (request fanout, timeout/budget mgmt)   │
                    └──────┬──────────┬──────────┬─────────────┘
                           │          │          │
              ┌────────────▼─┐  ┌─────▼─────┐  ┌─▼─────────────┐
              │  Retrievers  │  │  Feature  │  │   User-state  │
              │  (parallel)  │  │   Store   │  │   service     │
              │              │  │  (online  │  │ (session,     │
              │  - 2-tower   │  │   KV)     │  │  recents)     │
              │  - graph     │  └─────┬─────┘  └───────┬───────┘
              │  - heuristic │        │                │
              │  - content   │        │                │
              └──────┬───────┘        │                │
                     │ ~1500 cands    │ features       │
                     ▼                ▼                ▼
              ┌──────────────────────────────────────────────┐
              │   Coarse Ranker (cheap MLP, CPU)             │
              │   1500 → 300                                 │
              └──────────────────┬───────────────────────────┘
                                 ▼
              ┌──────────────────────────────────────────────┐
              │   Fine Ranker (DLRM, GPU, multi-head)        │
              │   300 → 100 scored                           │
              └──────────────────┬───────────────────────────┘
                                 ▼
              ┌──────────────────────────────────────────────┐
              │   Re-ranker (diversity, business rules,      │
              │   exploration injection)                     │
              │   100 → top 30                               │
              └──────────────────┬───────────────────────────┘
                                 ▼
                    Response  +  async logging → Kafka
                                                  │
                                                  ▼
                                       Training data pipeline
```

The orchestrator owns the latency budget. Each downstream call has a deadline; partial responses are acceptable for some stages (retrieval, not ranking).

## 4. Deep Dive: Retrieval

The job of retrieval is *recall, not precision* — get the right 1500 from 100M, fast. Precision is the ranker's job. Mixing those two up is the most common mistake I see.

```
   100M items
       │
       ▼
   ┌─────────────────────────────────────────────┐
   │  Parallel retrievers (each returns top-K)    │
   ├─────────────────────────────────────────────┤
   │  2-tower ANN  (HNSW)        →  1000 items   │
   │  Item-item co-engagement     →  200 items    │
   │  Recently-viewed complements →  50 items     │
   │  Content-based (cold items)  →  100 items    │
   │  Seller-diverse explore pool →  150 items    │
   └────────────────────┬────────────────────────┘
                        ▼
                 Dedup + merge
                        │
                        ▼
                  ~1500 candidates
```

**Two-tower model.** User tower consumes profile features, recent sequence of engaged items (last 100, transformer encoder), context (time, device, geo). Item tower consumes item ID embedding, category/attribute embeddings, text encoder over title+description, image embedding (precomputed). Output: 128-dim L2-normalized embedding on each side, dot product trained with sampled-softmax.

Training labels are *positive engagement* (clicked, carted, bought) with weights — bought >> carted >> clicked. Hard negatives matter enormously here. Random negatives teach the model to separate "kitchen mixer" from "running shoes" — which is trivially easy and uses up model capacity that should be spent on hard distinctions.

**Hard negative mining.**
- In-batch negatives: free, gets you most of the way, but biases toward popular items because popular items appear in more batches.
- Mixed negatives: sample from yesterday's impressions where the user did *not* engage. These are items the previous ranker thought were good but the user rejected — exactly the boundary you want the next model to learn.
- Cap mixed-negative ratio at ~30% of batch; pure hard-negative training collapses precision.

[STAFF SIGNAL: knowing that pure random negatives produce a model that looks good on offline retrieval recall@1000 but is useless because the negatives are too easy — and that pure hard negatives create gradient pathologies]

**ANN index choice.** HNSW for the main two-tower index. Reasoning: at 100M items, FP32 embeddings of 128 dims = ~50GB raw, which fits in memory on a beefy box with replication. HNSW gives me ~98% recall@1000 vs exhaustive at sub-10ms p99 on commodity hardware. IVF-PQ would compress 4-8x but costs ~5-10pp recall and adds quantization noise that interacts badly with the learned embeddings. ScaNN is a fine alternative if we're in the Google ecosystem. The recall/latency/memory tradeoff for our scale lands on HNSW.

I'd revisit this at 1B items where memory becomes the dominant cost and IVF-PQ or RaBitQ-style binary quantization starts to matter.

[STAFF SIGNAL: naming the tradeoff explicitly with the cost — "5-10pp recall" not "some recall loss" — and the scale at which I'd flip the decision]

**Embedding refresh.** Item tower retrained nightly on the previous N days of data; embeddings recomputed in a batch job and atomically swapped into the index. Within a day, new items are bootstrapped via the content-based path: title + image + category go through the item tower (which only needs content features for new items — no ID lookup needed because we use ID embeddings additively, with content features as the backbone). This is a deliberate choice: pure ID-based two-towers can't handle cold start at all.

User embeddings computed at request time from session state — they're cheap, and we want them fresh. Stale user embeddings are the #1 cause of "I just clicked X and it's still showing me Y."

**Complementary retrievers.** Two-tower alone undercovers. The graph retriever (item-item co-purchase / co-view graph, PinSage-style or just a sparse matrix factorization for v1) catches "people who bought X also bought Y" patterns the two-tower misses. The heuristic retriever (recently-viewed, cart-complements) covers strong-intent signals that the model under-weights because they're rare in the training distribution. Content retriever covers cold items.

I'd build the two-tower first, add graph as v2, heuristic from day one because it's free and very high precision on a small slice.

## 5. Deep Dive: Ranking

```
  1500 candidates
       │
       ▼
  ┌───────────────────────────┐
  │  Coarse Ranker            │
  │  - 2-layer MLP            │
  │  - user/item embeddings + │
  │    ~20 cheap features     │
  │  - distilled from fine    │
  │  - CPU, batched           │
  │  - 1500 → 300             │
  └───────────────┬───────────┘
                  ▼
  ┌───────────────────────────────────────┐
  │  Fine Ranker (DLRM-style)             │
  │                                       │
  │  sparse cat features ──┐              │
  │  dense features ────┐  ▼              │
  │                     ▼  emb tables     │
  │                  ┌─────────┐          │
  │                  │ deep+cross│         │
  │                  │  network │          │
  │                  └────┬─────┘         │
  │                       ▼               │
  │              ┌──────────────────┐     │
  │              │ multi-head       │     │
  │              │ - p(click)       │     │
  │              │ - p(cart|click)  │     │
  │              │ - p(buy|cart)    │     │
  │              │ - E[dwell]       │     │
  │              └──────────────────┘     │
  │                                       │
  │  GPU, batch=300, ~60ms                │
  └───────────────┬───────────────────────┘
                  ▼
            scored candidates
```

**Model class.** For v0, GBDT (LightGBM) with ~500 hand-crafted features. It's a strong baseline, fast to train, easy to debug, and gives you a feature-importance readout that surfaces data bugs in week one. I've shipped GBDT rankers at scale; they're not embarrassing.

For v1, DLRM-style with explicit cross network (DCN-v2) for feature interactions plus a deep tower. Reasoning: marketplace data has heavy categorical sparsity (seller_id, category, brand, geo) where embedding tables genuinely beat one-hot + GBDT. Cross network captures the explicit "this user × this category" interactions that GBDTs fake via deep trees but neural nets do more efficiently at this scale.

For v2, transformer-based sequence model over the user's engagement history. Adds 1-2pp NDCG in my experience but doubles training cost and complicates serving. Worth it only after the simpler pieces are landed.

[STAFF SIGNAL: starting with the simple thing that works, naming why, and saying when I'd upgrade — not jumping to "we'd train a 7B foundation model"]

**Multi-objective.** Click is a noisy proxy for the thing we care about. We want clicks, carts, purchases, and not-too-short dwells. Cheap clickbait listings game pure-CTR rankers; we've all seen it.

Architecture: shared backbone, four heads, each predicting a calibrated probability or expected value. The serving score is a weighted combination:

```
score = w_click * log(p_click)
      + w_cart  * log(p_cart_given_click)
      + w_buy   * log(p_buy_given_cart)
      + w_price * log(expected_price | bought)
      + w_dwell * log(E[dwell])
```

Equivalent to a log-prob of expected GMV per impression if you set weights right. Weights tuned via Pareto sweep against the offline metric tree, then validated online. *Don't* tune weights to maximize offline AUC — those weights will not be optimal online, because offline AUC is computed on a biased sample of items the previous ranker chose to show.

**Calibration.** Per-head Platt scaling fit on a held-out time window. Why it matters: the moment we mix organic with ads, we're comparing $-value of an ad impression to $-value of an organic conversion, which requires absolute probabilities, not just rankings. If we only sort organic and never mix, calibration is theoretically unnecessary — but in practice you always end up mixing eventually (with ads, with editorial slots, with sponsored sellers), so I'd build calibration in from v1.

[STAFF SIGNAL: knowing calibration matters when you mix and is moot when you only sort — and that "we'll mix eventually" is the right design assumption]

**Coarse ranker.** A distilled student of the fine ranker, 2-layer MLP, no embedding tables (just consumes pre-computed user/item embeddings + ~20 dense features). Runs on CPU, 1500 items in <15ms batched. Distillation target: the fine ranker's logits on the same examples. This is how you get cheap top-of-funnel pruning without losing too much precision — running the fine ranker on 1500 items would blow the latency budget.

## 6. Deep Dive: Features

This is where most prod recommenders actually live or die. The model is 20% of the work.

```
                        OFFLINE                              ONLINE
   ┌───────────────────────────────────┐      ┌──────────────────────────────────┐
   │  Event logs (Kafka → warehouse)   │      │  Online feature service (KV)      │
   │  - impressions, clicks, carts, buys│      │  - user real-time feats (Redis)  │
   │  - feature snapshots @ event time │      │  - item near-RT feats (Redis)    │
   │                                    │      │  - item static feats (Redis)     │
   │  Batch feature jobs (Spark)        │      │  - cross feats: computed online  │
   │  - daily item aggregates           │◀────▶│                                  │
   │  - daily user aggregates           │ sync │  Updated by:                     │
   │  - graph features                  │      │   - streaming jobs (Flink)       │
   │                                    │      │   - batch loaders (nightly)      │
   │  Training data builder:            │      │                                  │
   │  - join labels to feature snapshot │      │  Read by ranker per request:     │
   │  - point-in-time correct           │      │   batched mget, ~40ms p99        │
   └───────────────────────────────────┘      └──────────────────────────────────┘
                 │                                          ▲
                 │ training set                             │ same transform code
                 ▼                                          │ (shared library)
            Train ranker ─────────────────────────────────▶ Serve ranker
```

**Feature classes and freshness budgets.**

| Class | Examples | Freshness | Storage |
|---|---|---|---|
| User real-time | last-N events, session state | <1s | Redis, written by Flink |
| User profile | 7d / 30d engagement counts, preferred categories | hourly | Redis, nightly + hourly delta |
| Item real-time | price, inventory, last-hour CTR | 1-5 min | Redis, streaming |
| Item static | category, attributes, image emb | daily | Redis, nightly load |
| Cross (user×item) | has-user-engaged-this-seller, category-affinity-score | request-time | Computed in ranker |
| Context | time, device, geo, app version | request-time | From request |

Cross features are the most valuable single category and the most expensive to precompute (N_users × N_items combinatorial). The right move is to compute them online from primitives that *are* precomputed: e.g., store the user's seller-engagement set and the item's seller_id, then compute "has user engaged this seller" at scoring time. This is a couple-µs lookup per candidate, fine for a 300-candidate batch.

**Training-serving parity.** This is non-negotiable and gets botched constantly. The fix:

1. *Single feature-transform library.* The same code computes features for training (against historical KV snapshots) and serving (against live KV). No reimplementation in two languages.
2. *Feature snapshots logged at impression time.* When the orchestrator logs an impression, it logs the full feature vector that the ranker actually saw. We train on those snapshots, not on reconstructions from the warehouse "as it looks now."
3. *Online/offline parity tests.* Sample 0.1% of live requests, replay through the offline pipeline, compute feature-by-feature diff. Alert on drift > 0.1% of values disagreeing.

[STAFF SIGNAL: training-serving skew detection via shadow replay with a tight alert threshold, not just "we have unit tests on the feature library"]

**Point-in-time correctness.** The trap: training joins look like

```sql
SELECT impressions.*, items.rating  -- ← items.rating AS OF NOW, not as-of-impression
FROM impressions JOIN items ON ...
```

That leaks future state. An item that became great *after* the impression looks great in the training data even when it was mediocre at the time of impression. The model learns a spurious correlation and overfits.

Fix: features are joined as-of impression time. Either via temporal joins in the warehouse, or — better — via the snapshot-logging approach above. The snapshot approach is dramatically easier to maintain correctly than temporal joins, because point-in-time joins on dozens of feature tables is a perpetual source of subtle bugs.

[STAFF SIGNAL: distinguishing point-in-time correctness from "I joined the tables." This bug class is one of the most common causes of "offline metrics looked great, online launch was a wash"]

**Leakage cases specific to marketplaces.**
- Item-side features computed using engagements *from the impression we're predicting on.* Always exclude the current impression from any aggregated item feature.
- "Will this item sell out in the next hour" — feature engineered from future inventory state. Tempting and disastrous.
- User-side: "user's average dwell time" computed including the current session.

## 7. Deep Dive: Training Data

```
   Impressions (with feature snapshots)
                │
                ▼
   ┌──────────────────────────────────┐
   │  Wait window: 7 days for delayed │
   │  conversion attribution           │
   └──────────────┬───────────────────┘
                  ▼
   Join clicks (immediate), carts, buys
                  │
                  ▼
   ┌──────────────────────────────────┐
   │  Position-bias correction:        │
   │  - position fed as feature        │
   │  - dropped at serving (PAL)       │
   │  OR                               │
   │  - IPS reweighting by 1/p(shown) │
   └──────────────┬───────────────────┘
                  ▼
   ┌──────────────────────────────────┐
   │  Negative sampling:               │
   │  - in-impression non-clicks (hard)│
   │  - sampled global negatives       │
   │    (weighted by exposure)         │
   └──────────────┬───────────────────┘
                  ▼
         Training examples
```

**From events to examples.** Every impression becomes a candidate training example. Label depends on the head: click head gets {0,1} click label; cart head conditions on click; buy head conditions on cart. Conditional heads train only on the subset that satisfies the condition — this fixes the gradient-imbalance problem of trying to predict a 0.5%-rate event with cross-entropy from scratch.

**Position bias.** The slot a user clicks on a 10-item feed isn't uniform across positions — top positions get ~3-5x the CTR of bottom positions, mostly because users look there, not because those items are better. Train naively on impression logs and you bake "being shown at position 1" into the click prediction.

Two corrections worth knowing:
- **PAL / position-as-feature**: include position as a feature during training, set to a fixed constant (e.g., median or 0) at serving. The model learns to factor out position. Simple, works well, my default.
- **IPS / counterfactual reweighting**: weight each example by 1 / p(item shown at that position | logging policy). Requires knowing the logging policy's propensities, which you can if you log them. Cleaner theory, more variance, requires careful clipping of small propensities.

I'd ship PAL in v1 and consider IPS in v2 once the logging policy is fully instrumented. The honest answer is PAL covers 80% of the gain.

[STAFF SIGNAL: causal-inference-aware framing of position bias and a clear opinion on which fix is right for v1 vs v2]

**Selection bias.** We only have labels for items we showed. The model's training distribution is the *current ranker's choices*, not the catalog. This creates a feedback loop where the model learns to imitate the prior ranker plus epsilon, and items the prior ranker never showed never get labels.

Mitigations:
- Random-exposure data: dedicate 1-5% of impressions to uniformly-sampled candidates from retrieval (skipping the ranker). Tiny utility cost, enormous diagnostic value. Use it to train an unbiased component, or just to evaluate the production ranker's calibration on un-cherry-picked items.
- IPS-style reweighting using the production ranker's score as the propensity.
- Doubly-robust estimation when comparing two rankers offline.

[STAFF SIGNAL: explicit selection-bias framing and a concrete cost — "1-5% of impressions, paid as utility cost, repaid as off-policy evaluation capability"]

**Negative sampling.** For ranking, the "negatives" are the items in the same impression that the user didn't click. These are the *right* hard negatives because they were shown next to the clicked item. Augment with sampled global negatives (~5x ratio) so the model doesn't overfit to the specific items the retriever produces.

**Label delay.** Clicks: seconds. Carts: minutes. Purchases: hours to days. Refunds: weeks. We can't wait 30 days to train. Approach:

- Maintain multiple labels per impression: `click_label` (closed within minutes), `buy_7d_label`, `buy_28d_label`, `refund_30d_label`. Each label closes at a different time, so each training run uses whatever labels have matured.
- Delayed-feedback model: predict $p(\text{eventually converts})$ using an exponential time-to-conversion model jointly with the conversion-rate model. This lets us include partially-observed positives.
- For purely online learning, use the cached click stream to keep CTR fresh and accept that buy-head updates lag a week.

## 8. Deep Dive: Serving

```
Latency budget: 200ms p99 server-side

  Edge ingress + auth            :  ~10ms
  Orchestrator dispatch          :   ~5ms
  Retrieval (parallel fanout)    :  ~30ms   ┐
    - HNSW lookup                 :  10ms   │ all run
    - graph lookup                :  15ms   │ in parallel,
    - heuristic                   :  ~5ms   │ deadline = 30ms
    - content                     :  20ms   ┘
  Feature fetch (batched mget)   :  ~40ms
  Coarse rank (CPU, batch 1500)  :  ~15ms
  Fine rank (GPU, batch 300)     :  ~60ms
  Re-rank                        :  ~15ms
  Response serialize + log       :  ~15ms
  Network/jitter buffer          :  ~10ms
  ────────────────────────────────────────
  Total p99                      :  ~200ms
```

The arithmetic has to work. If you can't draw this and have it sum, you don't have a design.

[STAFF SIGNAL: explicit latency budget arithmetic per stage with real numbers, not "we'd optimize for low latency"]

**Where the time actually goes.** In practice the surprises are (a) feature fetch tail latency due to KV cache misses, which I'd attack with co-locating hot user state and pre-warming on session start, and (b) GPU batching overhead at the ranker — you only get good GPU utilization at batch sizes ~100+, which is why coarse-rank down to 300 first.

**CPU vs GPU.** Coarse ranker is CPU because the model is small and batching at 1500 items already saturates a CPU core's matmul. Fine ranker is GPU because the embedding tables are big and the deep+cross network has enough FLOPs that GPU batching wins. We pay the CPU→GPU transfer cost once per request; with batch 300 of ~1KB feature vectors each, that's 300KB across PCIe — negligible compared to the kernel time.

**Caching strategy.**
- *Cache.* User profile features (per session, with TTL = session length). Item static features (in-process LRU at each ranker node, 90%+ hit rate on the long tail of impressions for popular items). Item embeddings (cached in the HNSW node).
- *Don't cache.* Final rankings. Every request has different context (time, recency, exploration noise), and caching rankings is how you get stale-feeling recs and break exploration.
- *Cache invalidation.* User-side cache invalidates on any cart/buy event via pub-sub from the event stream — this is the difference between "I just bought a coffee maker" and the model still recommending coffee makers for the next hour.

**Graceful degradation.** Failures are when staff signals show up:
- *Feature store partial failure*: use last-known cached features + a freshness penalty in the score. Don't fail the request.
- *Fine ranker down*: fall back to coarse ranker output directly. Quality drops measurably but the surface still works.
- *Retrieval partial failure* (e.g., HNSW node out): merge whatever retrievers returned, accept lower recall. Log it.
- *Total fallback*: a precomputed top-N-by-category list keyed by coarse demographic. Boring, but the home feed renders.

The principle: never serve an error to the home feed. Degrade quality silently and alert internally.

[STAFF SIGNAL: predicting where the design will fail and naming the silent-degrade path for each — this is the difference between a system that has 99.9% uptime and one that has 99.99%]

**Autoscaling.** Ranker fleet scaled on QPS with 30% headroom; fine-ranker GPU pool has a longer scale-out lag (cold-starting GPU pods is slow), so we keep a warm buffer. We pay for it. It's worth it.

## 9. Deep Dive: Feedback Loops

This is where most marketplace recommenders quietly fail over months.

**Popularity collapse.** The naive system trains on what users engage with → engagement concentrates on items the model promotes → those items get more impressions and more labels → the model becomes even more confident → tail items never get a fair shot. Within a quarter, your top 1% of items eat 50% of impressions.

Detection: monitor the Gini of impressions over items week-over-week, plot impression-share of items that were unseen 30 days ago. If it trends down monotonically, you have a collapse.

Mitigation: explicit exposure regularization. Add a term to the re-ranker score that penalizes items already shown to this user this week, and (at the population level) add a counterfactual-coverage bonus for under-impressed items. IPS during training weights tail items more heavily so the loss isn't dominated by head items.

[STAFF SIGNAL: naming the second-order effect (popularity collapse), giving the detector, and giving the fix — not just listing "we should have diversity"]

**Filter bubbles.** Same dynamic but per-user. User clicks 3 dresses → model floods them with dresses → user has forgotten our marketplace also has home goods. Counter in re-ranker: diversity constraints (e.g., MMR over category embeddings, cap of K items per category in the top 30), small slot reservation for cross-category exploration.

**Seller-side fairness.** This is the marketplace-specific version. Sellers who win the algorithm early get a flywheel; new sellers can't break in even with good listings. We owe sellers (and the long-term marketplace health) explicit fairness:
- New-seller exposure quota: a small slot reservation for sellers with <30d on platform, scored by content-based retrieval since they have no engagement history.
- Per-seller impression caps within a session.
- Monitoring: percentile of sellers in the impression distribution, week-over-week.

**Exploration.** Pure greedy training-on-logs collapses because the model never observes outcomes for items it doesn't show. Counters:
- *Position-based exploration*: reserve 1-3 slots in the top 30 for explore candidates (e.g., from a UCB or Thompson-sampling pool over high-uncertainty items).
- *Epsilon-random*: 1-5% of impressions per session are uniformly retrieved (we already need this for the unbiased-evaluation data above; it pays double duty).
- *Thompson sampling on the ranker*: maintain a posterior over item utility, sample at scoring time. Expensive at our scale; I'd defer it.

Evaluation that we've broken the loop: counterfactual evaluation using the random-exposure slice. If the production model's score correlates well with engagement *on uniformly-shown items* (not just on items it already chose to show), it's not just imitating itself.

[STAFF SIGNAL: explicit causal-inference framing for "how do we know exploration is working" — measuring on the slice where the system isn't selecting]

## 10. Failure Modes, Monitoring, Rollout

**Monitoring (per ranker model in prod):**
- Online: CTR, cart rate, conversion rate, GMV/session, all bucketed by user cohort (new/returning) and session position
- Calibration: ECE on each head, weekly
- Coverage: % catalog impressed, Gini of seller impressions
- Latency: p50/p99/p99.9 per stage
- Feature freshness: oldest feature in any served request, per class
- Online/offline feature parity: drift per feature, alerted at >0.1% mismatch
- Training-data freshness: lag from event to ingested label

**Rollout.** New ranker:
1. *Offline*: NDCG and per-head calibration on a held-out forward time window (no random splits — random splits leak temporal info and make you optimistic). Compare to incumbent.
2. *Shadow*: serve in parallel for 1% of traffic, score but don't return. Compare score distributions and feature distributions to incumbent.
3. *Interleaving*: small-traffic interleaved test (TDI / team-draft interleaving) for fast directional signal.
4. *A/B*: 1% → 10% → 50% over ~3 weeks, gated on north-star + guardrails. Conversion-rate effects need 2+ weeks for the delayed labels.
5. *Holdback*: keep a 1% control on the old ranker indefinitely. The long-term value of this is enormous — it's the only way to measure the population-level effect of changes that influence what users come back for.

[STAFF SIGNAL: insisting on time-based splits not random, and on a permanent holdback for measuring long-term effects that A/B tests can't capture]

## 11. What I'd Build First vs Defer

**v0 (weeks 1-6).** Two-tower retrieval with random + in-batch negatives. HNSW index. GBDT ranker on hand-crafted features, single objective (click). Heuristic re-rank for category diversity and seller cap. Snapshot-based feature logging from day one (this is the thing you cannot retrofit). Position-as-feature for bias correction. 1% random-exposure slice for off-policy eval.

This will be roughly competitive with whatever incumbent exists. The point is to land the *pipeline*: logging, training, serving, monitoring. Once the pipeline is there, model iteration is fast.

**v1 (months 2-4).** DLRM-style ranker with multi-head (click, cart, buy, dwell). Per-head calibration. Graph retriever. Mixed hard-negative mining in the two-tower. Online/offline parity testing. Permanent holdback established.

**v2 (months 5-8).** Transformer sequence model for user representation. Cold-start retriever for new items based on content. Seller-fairness explicit constraints in re-rank. Thompson-sampling exploration. Delayed-feedback model for conversion head.

**v3 and beyond.** Ad mixing with calibrated auction. Multi-task with cross-surface (search, PDP) sharing. RL-based re-ranker on session-level rewards.

[STAFF SIGNAL: sequencing decisions — explicit about what's worth building first not because it's easiest, but because the pipeline-shape it forces (snapshot logging, holdback) constrains all future iteration]

**What I'd defer aggressively.** Real-time online learning of the ranker (rarely worth the operational cost when daily retraining is fine). Per-user models. Reinforcement learning before the supervised baseline is solid. Foundation-model retrievers — interesting research but the ROI vs a well-tuned two-tower is small for the cost.

**Things I'd want to revisit at scale.** ANN choice at 1B+ items (HNSW memory becomes a problem). Ranker model class at 10B+ params (DLRM saturates, transformers help). Feature store sharding strategy when single-region KV stops scaling.

---

The honest summary: 70% of getting this right is the data plumbing — snapshot logging, point-in-time correct training data, training-serving parity, holdback measurement. 20% is the modeling choices. 10% is the architectural pieces (which ANN, which feature store, which orchestrator). The interview tends to optimize for talking about the 10% because it's flashy, but the 70% is what separates systems that work from systems that ship and then quietly regress.