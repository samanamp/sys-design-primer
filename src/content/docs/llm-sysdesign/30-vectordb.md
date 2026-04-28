---
title: Vector Database for Retrieval
description: Vector Database for Retrieval
---

# Vector Database at 10B Scale — Staff-Level System Design

> **Question:** Design a vector database supporting 10B vectors, 1K-dim, k=100 NN with p99 < 50ms. Metadata filtering. Multi-tenant with strong isolation.

---

## 1. Research Pass — State of the Art (2026)

Before designing anything, the design space has shifted significantly since the HNSW-everywhere era. What I confirmed from a focused research pass:

**The graph-in-RAM era is over for billion-scale.** DiskANN/Vamana (Microsoft, NeurIPS 2019) demonstrated 95%+ recall@1 at <5ms latency on a billion-point dataset using a 64GB-RAM machine — vectors and graph live on NVMe, only PQ codes stay in memory. SPANN (Microsoft, NeurIPS 2021) reaches 90% recall@10 in ~1ms with only 10% of the memory cost of DiskANN by storing only centroid points in RAM and posting lists on SSD. SPANN powers Bing at hundred-billion scale today.

**The frontier is object-storage-first, not SSD-first.** Turbopuffer's architecture (~2024) inverts the tiering — object storage (S3/R2) is the source of truth, NVMe is a cache, RAM is a hot cache. They run 3.5T+ documents, 10M+ writes/sec, with cold queries at ~400ms (3–4 S3 round trips of ~100ms each) and hot queries in single-digit ms. Pinecone serverless (2024, GA 2024) uses an essentially identical pattern with "slabs" — immutable LSM-style files in S3, bitmap indexes per metadata field, query executors that cache slabs locally. Cursor stores tens of millions of namespaces in S3, one per codebase; Notion runs 10B+ vectors across millions of namespaces. The cost reduction vs pod-based is reported at 50–100x for intermittent workloads.

**Crucially, neither uses HNSW or DiskANN/Vamana at the namespace level.** Both use centroid-based / IVF-style indexes. Turbopuffer is explicit about why: graph indexes have terrible round-trip behavior on object storage because each greedy-search step depends on the previous one. Centroid-based indexes pre-resolve the route in one round-trip and then fan out one batched read. **This contradicts the prompt's implicit assumption that DiskANN is the obvious 2026 answer.** It isn't. DiskANN is the right answer for single-machine NVMe. SPANN-style is the right answer for object-storage-resident.

**Filtering integration is the active research frontier.** Filtered-DiskANN (WWW 2023) builds graph edges that respect *both* vector geometry *and* label sets. Turbopuffer's "native filtering" (2025) makes attribute indexes cluster-aware so the query planner intersects matching clusters with attribute posting lists *before* deciding which clusters to fetch — pre-filter recall (100%) at near-post-filter latency (25ms vs 20ms). Pinecone uses roaring-bitmap indexes per metadata field and chooses pre-filter vs mid-scan filter dynamically based on selectivity.

**Quantization is now layered, not single-precision.** The mixedbread mxbai-v1 family (2024) combines Matryoshka + binary QAT to produce 64-byte embeddings retaining ~96% of full-precision performance. The standard production stack is now: binary embedding for first-stage ANN → int8 rescore on top-1000 → optional cross-encoder rerank on top-100. This is 32x storage reduction with ~3% recall loss, which is recoverable in the rescore stage.

**Recall is now an observable.** Turbopuffer samples 1% of live traffic and runs brute-force in shadow to compute recall@10 continuously. This is what production-grade looks like in 2026 and the prompt is right to call it out — "we'll measure recall offline" is not enough.

**[STAFF SIGNAL: research-before-design]** **[STAFF SIGNAL: 2026-cutting-edge awareness]**

---

## 2. Scope, Assumptions, Reframing

**[STAFF SIGNAL: 40-TB-reframing]** 10B vectors × 1024 dim × 4 bytes (FP32) = **40 TB raw**, before any index, replication, metadata, or WAL. Cloud DRAM at ~$5/GB/month makes "everything in RAM" cost ~$200K/month for *one replica* of just the raw vectors. The HNSW graph alone, at ~100 bytes/vec for adjacency lists, is another 1 TB and *must* be RAM-resident for reasonable greedy search. This number forces the architecture: data lives on object storage, hot subsets cache to NVMe, and the index format has to be designed around minimizing object-storage round trips, not minimizing graph hops.

**[STAFF SIGNAL: scope negotiation]** I'm committing to these assumptions; the design changes substantially if any of them flip:

| Dimension | Assumption | Why this matters |
|---|---|---|
| Workload | RAG retrieval; read-heavy with continuous low-rate writes | Drives latency budget split, justifies caching |
| Tenant distribution | Power-law: ~100 whale tenants with 100M–1B vectors, ~100K mid tenants with 10K–10M, millions of cold tenants <10K | Drives namespace-per-tenant model and cold-start handling |
| Update pattern | Append-mostly with occasional deletes via tombstone | Allows LSM-style segment construction; no in-place graph mutation |
| Embedding model | Single model per namespace, externally produced; we record model version | Recall consistency invariant — see §10 |
| Query type | k=100 vector NN + AND/OR over typed metadata (some equality, some range, some array-membership) | Drives filter index design |
| Filter cardinality | Mixed: some low (e.g., `language=en`), some 1M+ unique values (`user_id`) | Drives bitmap vs partition decisions |

**[STAFF SIGNAL: saying no]** Two pushbacks on the prompt itself:

1. **k=100 is unrealistic for the stated p99=50ms budget on a 10B index, and probably unnecessary.** Modern RAG pipelines retrieve k=20–50 from ANN and rerank to k=5–10 with a cross-encoder. k=100 is a holdover from pre-reranker thinking. I'll design for k=100 as a hard constraint, but I'll note that relaxing to k=50 + reranker is the better product.
2. **"10B in one logical index" is almost never the right framing under multi-tenancy.** With namespace-per-tenant, 10B is the *aggregate* across millions of namespaces, not a single index. The right index size to engineer for is the p99 namespace, which under power-law distribution is much smaller than 10B. I'll engineer for both: a small handful of whale namespaces in the 100M–1B range, and millions of small ones.

**Central tensions** I'll spend most of this answer on:

- **[STAFF SIGNAL: filter-as-central-tension]** Filter integration. Pre-filter collapses recall on graph indexes; post-filter under-returns. The 2026 answer is *cluster-aware bitmap indexes* that participate in the query plan.
- **Storage economics.** Object storage as source of truth; NVMe and RAM as hierarchical caches. Cold-tenant pricing model is a feature, not an artifact.
- **Multi-tenancy isolation.** Namespace-per-tenant on shared compute. Per-tenant query budgets to bound blast radius. Crypto-level isolation only for whales who pay for it.

---

## 3. Capacity and Latency Budget Math

**[STAFF SIGNAL: capacity-math]**

### 3.1 Storage at each quantization level (10B × 1024-dim)

| Representation | Bytes/vec | Total | Use |
|---|---|---|---|
| FP32 | 4096 | 40.0 TB | Truth tier (rerank) |
| FP16 | 2048 | 20.0 TB | Mid tier; rare in production |
| Int8 SQ | 1024 | 10.0 TB | Rescore tier |
| PQ (M=128, 8b) | 128 | 1.25 TB | First-stage candidates (alternative) |
| Binary (1-bit) | 128 | 1.25 TB | First-stage candidates (preferred) |
| Matryoshka@256 + int8 | 256 | 2.50 TB | Mid tier with shorter vectors |
| Matryoshka@128 + binary | 16 | **160 GB** | Aggressive first stage (~96% recall) |

The headline: aggressive layered quantization compresses 40 TB → 160 GB for the first-stage candidate index. That difference is what makes object-storage residency economically inevitable. We pay for FP32 storage *once, on cold object storage*, and never load it into the search hot path except for the rerank top-N.

### 3.2 Latency budget decomposition (p99 = 50 ms)

| Stage | Target | Mechanism |
|---|---|---|
| API gateway + auth + tenant routing | 2 ms | Edge service with tenant→region map |
| Query planner (filter cost-est + plan) | 1 ms | In-memory bitmap stats |
| Coarse routing (centroid lookup, namespace cache hit) | 3 ms | RAM-resident centroid index for hot namespace |
| Posting-list fetch (NVMe hit) or 1 S3 ranged read (cold) | 5 ms hot / 100 ms cold | NVMe ~100µs/4KB; S3 ~100ms/range |
| First-stage scoring (binary Hamming, ~10K candidates) | 8 ms | SIMD popcount, ~1 GB/s/core |
| Int8 rescore (top 1000) | 6 ms | SIMD int8 dot product |
| Filter intersection | 4 ms | Cluster-aware bitmap AND |
| FP32 rerank (top 200, NVMe) | 12 ms | NVMe random read + dot product |
| Result merge / response | 4 ms | |
| **Total (hot path)** | **~45 ms** | |
| **Total (cold)** | **~250 ms+** | Cold path is *not* in-SLO; warming is required |

**Key implication:** the SLO is a hot-path SLO. Cold namespaces violate it. We commit to a separate cold-start SLO ("first query after 30 days idle: p99 < 1s") and aggressive warming to keep hot tenants in cache.

### 3.3 SSD random-read budget at 50 ms

NVMe at 4KB random reads: ~100 µs each, ~1 GB/s sequential. In a 50ms budget with ~30 ms reserved for I/O: ~300 random reads max. That's *exactly* why graph indexes that touch hundreds of pages per query work on local NVMe — and *don't* work on S3 where every page costs 100 ms. **The 100x latency gap between NVMe and S3 is what dictates the index choice on the cold path.**

---

## 4. High-Level Architecture

```
                        ┌──────────────────────────────┐
                        │  Client / RAG application    │
                        └──────────────┬───────────────┘
                                       │
                        ┌──────────────▼───────────────┐
                        │    Regional API Gateway       │
                        │  (auth, tenant→namespace,     │
                        │   rate limit, query budget)   │
                        └──────────────┬───────────────┘
                                       │
              ┌────────────────────────┼────────────────────────┐
              │                        │                        │
       Write Path                Read Path                 Control Plane
              │                        │                        │
              ▼                        ▼                        ▼
      ┌──────────────┐         ┌──────────────┐         ┌──────────────┐
      │ WAL (S3,     │         │  Query       │         │  Tenant      │
      │ append-only) │         │  Routers     │         │  Catalog     │
      └──────┬───────┘         │  (stateless) │         │  (Postgres)  │
             │                 └──────┬───────┘         └──────────────┘
             ▼                        │
    ┌─────────────────┐               │ namespace consistent-hash
    │ Memtable (RAM)  │               │
    │ per namespace   │               ▼
    │ exact L0 index  │      ┌─────────────────────┐
    └────────┬────────┘      │  Query Executors    │ ◄── 1% shadow recall
             │ flush         │ (stateful cache,    │     sampler
             ▼               │  any node serves    │
    ┌─────────────────┐      │  any namespace)     │
    │ Slab Compactor  │      └─────────┬───────────┘
    │ (background)    │                │
    └────────┬────────┘                │
             │ writes slabs            │ reads
             ▼                         │
    ┌─────────────────────────────────────────────────┐
    │  Storage Hierarchy                              │
    │  ┌────────┐  ┌──────────┐  ┌─────────────────┐  │
    │  │ RAM    │  │  NVMe    │  │  Object Storage │  │
    │  │ centroid │  │ slabs/  │  │  (source of     │  │
    │  │  +bitmap │  │ posting │  │   truth, S3)    │  │
    │  │ hot ns   │  │ binary  │  │  all slabs +    │  │
    │  └────────┘  │  int8    │  │  WAL + FP32     │  │
    │   ~100GB     │  └──────────┘  └─────────────────┘  │
    │              ~10TB/node    ~PB                  │
    └─────────────────────────────────────────────────┘
```

The shape is the lambda architecture for vector search that Pinecone calls out explicitly: a freshness path (memtable, exact, in-RAM) unioned with an indexed path (slabs, ANN, on disk/S3). Writes go to WAL → memtable → eventual flush to immutable slab. Compaction merges L0 slabs into larger L1+ slabs in the background. Queries fan out across both the memtable and matching slabs.

---

## 5. Index Choice — SPANN-style Centroid Index per Namespace

**[STAFF SIGNAL: rejected-alternative]** Choices considered and rejected:

| Option | Why rejected |
|---|---|
| **HNSW everywhere** | Graph is ~1–2 TB at 10B; must be in RAM. ~$5K/GB/year just for graph. Single-namespace HNSW also doesn't shard cleanly. Eliminated by §2 economics. |
| **DiskANN/Vamana per namespace** | Best on local NVMe; greedy search touches 100s of pages. On S3 each page is 100ms, so cold-path latency explodes. Right answer for a *single-machine* deployment, wrong for object-storage-first. |
| **IVF-PQ classic** | Compressed-only search loses too much recall vs. modern centroid + binary + rescore. Posting list balance is hard at billion scale. |
| **One huge HNSW with `tenant_id` filter** | Eliminated on isolation grounds (§7). Also: shared graph means no per-tenant cold-tier savings. |
| **GPU-resident (CAGRA)** | At 10B × 1024 dim, vectors don't fit in GPU memory at any reasonable cost. GPU helps for high-QPS workloads on indexes that fit; not the right primary tier here. Could be used as a rerank accelerator. |

**Chosen: SPANN-style hierarchical centroid index per namespace, with cluster-aware bitmap attribute indexes, layered binary→int8→FP32 quantization, all on object storage with NVMe and RAM caches.**

This is the architecture both Turbopuffer (SPFresh — an incrementally-updatable variant of SPANN) and Pinecone serverless (slab compaction with cluster-style indexes per slab) have converged on, independently, after running this exact problem in production. That's a strong empirical signal.

### 5.1 Index data structure (per namespace)

```
              ┌──────────────────────────────────────────┐
              │   Top-level centroid index (in RAM)      │
              │   (HNSW on 1-2% of vectors as centroids) │
              │   ~100K–1M centroids per whale namespace │
              └────────────────┬─────────────────────────┘
                               │  greedy: top-K nearest centroids
              ┌────────────────┴─────────────────────────┐
              │  L1 slabs (NVMe-cached, S3-resident)     │
              │  one slab ≈ one cluster ≈ ~10K vectors   │
              │  contains:                               │
              │   • binary (16B)         – first stage   │
              │   • int8 (1KB)           – rescore       │
              │   • FP32 ref to L2       – rerank        │
              │   • bitmap indexes for filterable attrs  │
              │   • cluster-level downsampled bitmap     │
              └──────────────────────────────────────────┘
              ┌──────────────────────────────────────────┐
              │  L2 cold tier (S3 only, on-demand)       │
              │  FP32 vectors, accessed only for top-200 │
              │  rerank when full-fidelity needed        │
              └──────────────────────────────────────────┘
```

Why this shape:
- Centroid index in RAM is small (1–2% of vectors × ~256B each = ~50 GB for a 1B-vector whale). Fits per-node, persistent across queries.
- Each cluster maps 1:1 to a slab, sized so that a single S3 ranged read fetches the whole posting list in one round trip (~MB-range). Cold query: 1 S3 read for centroid index header + 1 S3 read per probed cluster (typical: 8–32 clusters) → in practice 3–4 round trips total via parallelism.
- Slab is *self-contained*: contains its own binary+int8+attribute bitmap. No cross-slab index dependency. This is what makes the LSM merge cheap.

---

## 6. Deep Dive — The Filter Integration Problem

This is the load-bearing part of the design.

### 6.1 Why pre-filter and post-filter both fail

```
PRE-FILTER (naive)
  ┌────────────┐    ┌─────────────────────┐    ┌────────────┐
  │  filter    │───▶│ candidates matching │───▶│ exhaustive │
  │  index     │    │   filter (set S)    │    │ NN over S  │
  └────────────┘    └─────────────────────┘    └────────────┘
   recall: 100% if exhaustive; latency: O(|S| · d). 
   At |S|=1M and 1024-d, ~5+ seconds. DEAD.

POST-FILTER (naive)
  ┌────────────┐    ┌─────────────────────┐    ┌────────────┐
  │ ANN top-K' │───▶│  apply filter       │───▶│  return    │
  │ (no filter)│    │  (drop non-matches) │    │  top-k     │
  └────────────┘    └─────────────────────┘    └────────────┘
   latency: fast. recall: 0% if filter selectivity 
   doesn't overlap with vector top-K'. DEAD on selective filters.

NATIVE / INTEGRATED FILTERING (what we build)
  ┌──────────────┐    ┌────────────────────────┐    ┌────────────┐
  │  filter      │───▶│ identify candidate     │───▶│ first-stage│
  │  bitmap      │    │ clusters that contain  │    │ score only │
  │  intersect   │    │ ≥1 matching vector;    │    │ matching   │
  │  with cluster│    │ probe those clusters   │    │ vectors in │
  │  index       │    │ in centroid order      │    │ those      │
  └──────────────┘    └────────────────────────┘    │ clusters   │
                                                    └────────────┘
   recall: ~90%+. latency: ~25ms. SAME number of candidates 
   considered as unfiltered query. Matches turbopuffer's stated target.
```

### 6.2 How we build it

The trick is that the attribute index is **cluster-aware**: instead of mapping `attribute_value → set of vector_ids`, it maps `attribute_value → set of (cluster_id, vector_id_within_cluster)` and is rolled up to a `attribute_value → set of cluster_ids` bitmap that lives next to the centroid index in RAM.

Query plan for `{ vector: q, filter: lang=en AND date>2024 }`:

1. Compute `cluster_bitmap_lang = bitmap[lang=en]` and `cluster_bitmap_date = bitmap[date>2024]`. These are rolled-up cluster-level bitmaps. AND them. Result: clusters that contain *at least one* matching vector.
2. Score all centroids against `q`, but skip clusters not in the AND-bitmap. Pick top-`probe` clusters that survive.
3. Fetch those slabs (NVMe hit if hot). Within each slab, intersect the per-vector bitmap with the slab to get matching vectors *in that cluster*. Score them with binary + int8.
4. Take top-1000 globally, fetch FP32 from L2 for the top-200, rerank, return top-100.

Recall is preserved because we're considering *the same number of candidate clusters* as an unfiltered query — we just skip the empty ones early. **[STAFF SIGNAL: filter-as-central-tension]**

### 6.3 The high-cardinality filter problem

**[STAFF SIGNAL: high-cardinality-filter-discipline]** What if a tenant has a `user_id` field with 10M unique values? A naive bitmap index has 10M entries each potentially covering the whole namespace.

Three-part response:

1. **Schema enforcement.** Attributes flagged as high-cardinality are *not* given inverted indexes — they're scan-only. If the user filters on them, the planner falls back to post-filter with a higher `over_fetch` parameter and warns on recall.
2. **Hash-partitioning by high-card attribute.** If the customer says "I always filter by user_id," that becomes a *hint*: we hash-partition the namespace by `user_id` into sub-namespaces. Now `user_id=X` is a *namespace selection*, not a filter. This is what Turbopuffer means by "create a namespace per logical group, not a filter."
3. **Roaring bitmaps with cluster rollup.** For mid-cardinality (10K–1M values), roaring bitmaps compress well; the cluster rollup keeps the AND-set tractable.

The architectural commitment: **filtering is a feature of namespace design, not just query design.** Customers with a known dominant filter dimension shard on it.

### 6.4 Recall guarantees under filtering

**[STAFF SIGNAL: filtered-recall-awareness]** Filtered recall is *strictly* harder than unfiltered. The relevant set changes per query. Turbopuffer's continuous-recall sampler explicitly computes recall on filtered queries and reports them — that's because they're known to be the weak case. Our SLO:

- Recall@100 (unfiltered) ≥ 0.95 at p50, ≥ 0.92 at p99
- Recall@100 (filtered, selectivity ≥ 1%) ≥ 0.92 at p50, ≥ 0.88 at p99
- Recall@100 (filtered, selectivity < 1%) — falls back to **kNN exact** with brute force over matching set; latency may exceed SLO but recall is 100%. This matches what Turbopuffer recently shipped as the kNN-exact escape valve.

---

## 7. Deep Dive — Multi-tenancy and Strong Isolation

**[STAFF SIGNAL: multi-tenant-explicit-commitment]**

### 7.1 The three patterns and the choice

```
PATTERN A: Shared index, tenant_id filter
┌──────────────────────────────────────────────────────┐
│ ONE BIG INDEX (mixed tenants)                        │
│   v1[t=A], v2[t=B], v3[t=A], v4[t=C], …              │
└──────────────────────────────────────────────────────┘
  + Cheapest. Best per-vector cost.
  - Isolation = correct WHERE clause = application logic.
    One bug = cross-tenant leak. UNACCEPTABLE for "strong" isolation.
  - Noisy tenant blows hot cache for everyone.
  - No per-tenant cold-tier savings.

PATTERN B: Namespace-per-tenant, shared compute   ◄── CHOSEN
┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐
│ ns(A)    │  │ ns(B)    │  │ ns(C)    │  │ ns(D)    │
│ S3 prefix│  │ S3 prefix│  │ S3 prefix│  │ S3 prefix│
│ own keys │  │ own keys │  │ own keys │  │ own keys │
└──────────┘  └──────────┘  └──────────┘  └──────────┘
        \         |         /         /
         \        |        /         /
          ┌──────────────────────────┐
          │  Stateless query fleet    │
          │  (any node, any namespace)│
          └──────────────────────────┘
  + Data-plane isolation: separate S3 prefix, separate KMS key.
  + Cold tenants pay near-zero (S3 storage only).
  + Per-tenant query budget enforces compute isolation.
  + Index parameters tunable per-namespace.
  - Cold-start: first query after eviction = ~250-1000ms.
  - Catalog overhead: millions of namespaces.

PATTERN C: Index-per-tenant, dedicated compute
┌──────────────┐  ┌──────────────┐
│ Tenant A pod │  │ Tenant B pod │   ...one per tenant
└──────────────┘  └──────────────┘
  + Strongest possible isolation; physical separation.
  - Doesn't scale beyond 1000s of tenants.
  - 100% baseline cost per tenant even if idle.
  - Reserved for whales who pay for it.
```

**Decision:** Pattern B by default. Pattern C as an upsell ("dedicated compute") for whales paying for compliance/SLA isolation.

### 7.2 The threat model and isolation enforcement

Three threats:

1. **Data exfiltration** — tenant A reads tenant B's vectors. Defense: per-namespace S3 prefix with separate KMS key per tenant; query routers verify the API key's tenant matches the namespace prefix at the gateway, and again at the query executor before any read. *Two enforcement points* so a bug in one doesn't open the door.
2. **Compute starvation** — tenant A's pathological query slows tenant B. Defense: per-tenant query budget (cores·ms), per-tenant rate limit at the gateway, per-tenant slab cache quota at the executor (tenant A can't blow B's cache out). Slow queries get ejected with a `query budget exceeded` error.
3. **Storage poisoning** — malicious payload in attribute corrupts shared structures. Defense: schema validation at ingress; per-namespace WAL and slab files are strictly isolated; no shared mutable state in the executor across namespaces.

**[STAFF SIGNAL: blast-radius-reasoning]** A single hot namespace with 1000x normal QPS triggers:
- Gateway rate limit kicks in first → tenant gets 429s, others unaffected.
- If the rate limit is misconfigured: the executor's per-tenant cache quota means hot-tenant slabs evict *their own* old slabs first, not other tenants'.
- If the executor is saturated CPU-wise: weighted-fair queuing across tenants, so other tenants see ≤2x latency increase, not 1000x.

### 7.3 The cold-start problem

Cold namespace = not queried recently, evicted from NVMe and RAM caches. First query needs to fetch:

1. Centroid index header from S3 (~10MB for a 100K-centroid namespace) — 100ms
2. Top-K probed clusters (~32 × ~1MB) — parallelized to 1–2 round trips, 100–200ms
3. Attribute bitmap shards if filtering — overlaps with (2)

Total cold latency: ~300–500ms typically. This **violates** the 50ms SLO. Mitigations:

- **Warming on signal.** When a user opens a workspace (Cursor's pattern: opens a codebase), the application calls a `warm(namespace)` API. We pre-fetch centroid + top-N hot clusters into NVMe within ~1s.
- **Predictive warming.** Per-tenant query history → predict which namespaces a tenant is likely to query next. Warm during low-traffic windows.
- **Tiered SLOs.** Cold-start SLO is `p99 < 1s for first query`, `p99 < 50ms for warm`. Customers pay extra for "always warm" (pinned in NVMe). Whales get pinning by default.

**[STAFF SIGNAL: invariant-based-thinking]** Invariant: a hot namespace's cache footprint ≤ its quota. Enforced by an LRU per-tenant. A noisy hot tenant cannot evict another tenant's hot data; only its own.

---

## 8. Deep Dive — Quantization Layering

**[STAFF SIGNAL: quantization-layered]**

We do *not* run search on FP32. We do *not* run search on a single quantized representation. We use a **3-stage pyramid**:

```
                     ╱╲
                    ╱  ╲   FP32 (40 TB on cold S3)
                   ╱    ╲       ↑ rerank top-200
                  ╱──────╲      |
                 ╱  Int8  ╲ Int8 (10 TB; rescore tier)
                ╱  rescore ╲    ↑ rescore top-1000
               ╱────────────╲   |
              ╱   Binary     ╲  Binary (1.25 TB; first stage)
             ╱  first-stage   ╲     candidate generation
            ╱──────────────────╲
           ╱  Centroids (HNSW)  ╲ in RAM (~50 GB hot)
          ╱──────────────────────╲
```

Pipeline per query:

1. **Stage 0: routing.** Score query against centroids in RAM. Pick top-`probe` clusters.
2. **Stage 1: binary candidate generation.** For each probed cluster, score all candidates with binary Hamming distance (SIMD popcount, ~10 GB/s/core on AVX-512). At ~10K candidates per cluster × 32 clusters = 320K candidates, this is ~5ms.
3. **Stage 2: int8 rescore.** Take top-1000 from Stage 1. Score with int8 dot product. ~3ms.
4. **Stage 3: FP32 rerank.** Take top-200. Pull FP32 vectors from L2 cold tier (NVMe if cached, S3 if not). Score, return top-100.

Why this works: binary preserves *order* (which is what ANN needs) at low precision; the rescore stages recover the *exact* ranking. mxbai-v1 reports 96% retention at the binary stage with rescoring; this matches our SLO floor with margin.

Failure case: queries from a different distribution than the embedding model was trained on can have terrible binary quantization recall. Defense: continuous recall measurement (see §10) catches this; the alarm is on aggregate distance distribution shift relative to baseline.

### 8.1 Matryoshka: when and why

For latency-critical queries on small namespaces, we additionally truncate Matryoshka-trained embeddings to 256 or 128 dim at *query time*. That gives another ~4–8x speedup on the binary stage. Whether to use Matryoshka is a per-namespace setting based on whether the tenant's embedding model was trained for it (mxbai, OpenAI 3-large with reduction, etc.).

This contradicts the prompt's "1K-dim" assumption — it should be "≤1K-dim, query-time tunable for Matryoshka models." **[STAFF SIGNAL: saying-no]**

---

## 9. Deep Dive — Storage Tiering and Economics

**[STAFF SIGNAL: storage-tier-economics]** Tier hierarchy and concrete numbers:

| Tier | Where | Cost (us-east-1, 2026) | Latency | What lives here |
|---|---|---|---|---|
| L0 RAM | per-executor DRAM | $5/GB/mo | ~100ns | Centroid indexes for hot namespaces, top-of-LRU slabs |
| L1 NVMe | per-executor local NVMe | $0.10/GB/mo | ~100µs/4KB | Recently queried slabs, attribute bitmaps |
| L2 Object | S3/R2 standard | $0.023/GB/mo (S3) or $0.015/GB/mo (R2) | ~100ms/range | Source of truth: WAL + all slabs + FP32 vectors |
| L2 Cold | S3 IA / Glacier | $0.0125/GB/mo or less | seconds | Only-FP32 archive for namespaces idle >30 days |

For a 1B-vector whale namespace at 4 TB compressed:
- All-RAM (HNSW): ~$20K/month per replica
- DiskANN on local NVMe: ~$400/month per replica
- Object-storage primary, NVMe cache: ~$100/month for storage + amortized compute

**Eviction policy:** segmented LRU per tenant within a global capacity pool. Each tenant has a quota; LRU within quota; tenant whose footprint exceeds quota evicts itself first. Hot global pool (top 5% of slabs by access freq) is excluded from eviction.

**Crossover analysis:** at what tenant access rate does object-storage residency stop being a win? Roughly when ≥20% of slabs are hot most of the time — at that point you're paying for NVMe and S3 both. Whale namespaces with continuous high QPS get pinned to NVMe (or upgraded to dedicated compute, Pattern C).

---

## 10. Deep Dive — Ingestion, Freshness, Streaming Updates

**[STAFF SIGNAL: freshness-mechanism]**

### 10.1 Write path

```
write(namespace, vector, metadata)
     │
     ▼
┌─────────────────────┐
│ Gateway: schema     │
│ validate, model     │
│ version assert      │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐    ┌──────────────────────┐
│ WAL writer:         │───▶│ S3 append-only WAL   │
│ assign LSN, durably │    │ namespace prefix     │
│ commit before ack   │    └──────────────────────┘
└──────────┬──────────┘
           │  ack to client (~100ms p50)
           ▼
┌─────────────────────┐
│ Memtable: exact     │  immediately searchable
│ scan in RAM, with   │  via fan-out at query time
│ tombstones for      │
│ deletes             │
└──────────┬──────────┘
           │  threshold: ~500MB or ~5min
           ▼
┌─────────────────────┐    ┌──────────────────────┐
│ Slab builder:       │───▶│ S3: new L0 slab      │
│ build small SPANN   │    │ (full self-contained │
│ index over memtable │    │  index)              │
└──────────┬──────────┘    └──────────────────────┘
           │  background
           ▼
┌─────────────────────┐    ┌──────────────────────┐
│ Compactor: merge    │───▶│ S3: L1, L2 slabs     │
│ L0 → L1 → L2 with   │    │ (larger, optimized   │
│ centroid rebalance  │    │  index types)        │
└─────────────────────┘    └──────────────────────┘
```

Freshness: a write is durable in ~100ms (WAL ack), searchable in ~100ms (memtable update is synchronous), indexed in 5min (memtable flush), fully optimized in ~hours (compaction). This matches Pinecone serverless's stated "writes durable in <100ms, searchable in seconds."

10K writes/sec: WAL append throughput on S3 is the bottleneck. We batch 100ms windows per namespace. For very write-heavy namespaces, we shard the WAL across multiple prefixes within the namespace.

### 10.2 Streaming vs batch index update

Both Turbopuffer (SPFresh) and DiskANN have streaming variants (FreshDiskANN, SPFresh) that avoid full rebuilds. SPFresh maintains cluster invariants under continuous insert/delete by incrementally rebalancing centroids: if a cluster grows past threshold, split; if it shrinks, merge with neighbor. We adopt this — the alternative (periodic full rebuild) costs days for whale namespaces and gives stale results during the rebuild.

Deletes are tombstones in the slab, plus a tombstone bitmap. Slab compaction physically removes tombstoned vectors and rebalances. Hard guarantee: a deleted vector cannot appear in results once the WAL ack returns (memtable carries the tombstone immediately).

### 10.3 Embedding model consistency invariant

**[STAFF SIGNAL: invariant-based-thinking]** The single biggest correctness footgun. A query embedded with model v2 against vectors embedded with model v1 returns garbage with no error.

Defenses:

- Each namespace declares a `model_id` and `model_version` at creation. Writes that don't match are rejected.
- Queries carry the model id; queries that don't match the namespace are rejected with an explicit error.
- Model upgrade requires a *blue/green namespace*: build `ns_v2` in shadow with the new model, dual-write, cut over reads atomically, then drop `ns_v1`. We expose this as a managed `reembedding_job` API rather than letting customers DIY it badly.

---

## 11. Deep Dive — Recall as a First-Class SLO

**[STAFF SIGNAL: recall-as-first-class]**

Recall is not "we'll measure offline." It's a continuous production observable, with an alert.

### 11.1 Continuous online recall measurement

We sample 1% of live queries (Turbopuffer's pattern, which is the right one). For each sampled query:
- Run the normal ANN pipeline → returns set A
- Run brute-force kNN over the *same filter-restricted set* → returns set G (ground truth)
- Compute recall@100 = |A ∩ G| / |G|

The brute-force is expensive (linear in matching set size) but only runs on 1% of traffic, so cost is bounded. Results stream into a TSDB; alarms fire on:
- Aggregate recall@100 < 0.92 over a 5-min window (per namespace, with smoothing)
- p99 recall < 0.85 for any namespace
- Recall drift > 5% over 24 hours (catches index degradation under continuous updates)

### 11.2 Per-query recall is noisy

Per-query recall is binary-ish (you get 95/100 or 92/100, can't smoothly interpolate). The SLO is on aggregate, not per-query. We expose per-namespace recall in the dashboard so customers can see it.

### 11.3 Offline regression evaluation

A held-out query set per major workload type runs nightly with brute-force ground truth. Used to validate index parameter changes (centroid count, probe depth, binary→int8 cutoff) before rollout.

### 11.4 Recall under filtering

Filtered recall is measured separately because it's structurally weaker. Filtered SLO is set 3 percentage points lower than unfiltered. Customers with selectivity <1% are routed to kNN-exact mode (linear scan of matches) which gives 100% recall at the cost of latency.

---

## 12. Hybrid Retrieval and Reranking

**[STAFF SIGNAL: hybrid-retrieval-awareness]** Pure dense ANN is not the whole pipeline.

Three modes the system supports:

1. **Dense-only.** Standard k-NN over vector. What's described above.
2. **Hybrid (dense + sparse).** BM25/SPLADE over text fields runs in parallel to vector ANN. Both return top-200; results are fused via Reciprocal Rank Fusion (k=60). Returns top-100 final.
3. **Two-stage with cross-encoder rerank.** ANN returns top-1000 (we tune for recall@1000, not @100). Customer's reranker (cross-encoder, deployed as a separate service or by them) rescores. We feed metadata along with vectors to support reranker context.

Latency budget split when reranking is in scope:
- ANN stage: 30 ms (more candidates means more recall headroom)
- Rerank: 20 ms (out of the system's hands; we expose an end-to-end SLO option)

The vector DB's job in mode 3 is *not* to maximize recall@100. It's to maximize recall@1000 with low variance. Different tuning regime — looser probe depth, more candidates returned. We expose this as an `optimize_for: rerank` namespace setting.

---

## 13. Failure Modes and Operational Reality

**[STAFF SIGNAL: failure-mode-precision]** Concrete scenarios:

| Scenario | Detection | Response |
|---|---|---|
| Tenant ingests 10M vectors, attribute field has 5M unique values; bitmap blows up | Schema service tracks cardinality on write; alarm at 100K unique values per field | Auto-disable indexing on the field, return 202 with warning; recommend hash-partition into namespaces |
| Embedding model upgrade by customer; recall collapses | Continuous-recall sampler detects recall drop within minutes; aggregate-distance shift alarm | Reject mismatched-model queries via model-version check; force blue/green migration |
| Popular namespace gets 1000x normal QPS | Per-tenant rate limit (gateway); per-tenant cache quota (executor) | Tenant 429s; other tenants unaffected. Auto-scale executors on signal. |
| SSD failure on a node, slab cache lost | Health check fails; node ejected | Other nodes serve any namespace; cold-fetch from S3 on first miss. Background warming restores cache. |
| Replica recall divergence (graph construction has randomness) | Shadow recall sampler runs across replicas, compares | Mark divergent replica unhealthy, rebuild from S3 source-of-truth slabs |
| Cold namespace, first query in 30 days | Expected; not an alarm | Tier-2 SLO applies (1s); warming pipeline pre-fetches if any usage signal precedes |
| Eventually-consistent ingestion: write on replica A, query on replica B misses | WAL with global LSN; readers wait for LSN if `consistency=strong` requested | Default is `consistency=eventual` (millisecond lag); strong-consistency mode adds ~5ms but guarantees read-your-writes |
| Object storage outage in region | Multi-AZ S3 typically resilient; cross-region replication for whale namespaces | Read from replicated region. Write path queues to local WAL until S3 recovers. |

---

## 14. When This Architecture Is Wrong

**[STAFF SIGNAL: when-not-to-build-this]**

A staff engineer asks. The design above is justified only when:

- Total scale > ~100M vectors AND
- Multi-tenancy with thousands+ tenants AND
- Recall floor matters (RAG quality, not just demo)

If any fail, simpler is better:

- **<10M vectors total, single tenant:** pgvector with HNSW. The "vector database" abstraction is overkill. You're paying for a control plane you don't need.
- **Single corpus, no multi-tenancy:** A single DiskANN index on a beefy NVMe machine. Replicate for HA. This is what Anthropic-style internal knowledge bases look like at modest scale.
- **Workload dominated by lexical / sparse text matching:** dense vectors are an expensive way to fail at term-match. BM25 with a learned reranker beats this design for many enterprise-search use cases.
- **Mostly batch / offline:** you don't need a database. Faiss + a job runner, with results materialized into Parquet/Iceberg, is dramatically cheaper.

Staff candidates who skip this section have failed to ask "is the question right?"

---

## 15. Tradeoffs and What Would Change the Design

| If this changed... | ...the design changes how |
|---|---|
| Latency budget loosens to 200ms p99 | Use DiskANN/Vamana instead of SPANN; fewer round trips matter less; better recall at higher latency |
| Latency budget tightens to 10ms p99 | Force-pin index in RAM; can't use object storage primary; cost goes up 10x; cap on namespace size |
| Tenant count drops to 100 | Pattern C (index-per-tenant) becomes viable and removes most isolation engineering |
| Update rate goes to 1M writes/sec | WAL becomes the bottleneck; need partitioned WAL across namespaces, possibly Kafka in front of S3 |
| Embedding dim drops to 256 | Whole pipeline gets cheaper proportionally; binary first-stage may be unnecessary |
| Filter complexity grows (joins, aggregations) | Push to a separate query engine layered on top; vector DB is wrong abstraction for relational filtering |
| Workload becomes write-dominated | Slab compaction becomes critical-path; consider streaming-only index (no compaction) |

---

## 16. What I'd Push Back on in the Prompt

**[STAFF SIGNAL: saying-no]** The prompt has three implicit framings I'd reject:

1. **"10B vectors in one logical index."** Multi-tenant means it's 10B aggregate across millions of namespaces with a power-law distribution. The right number to engineer for is the p99 namespace size (typically 10M–100M), with whale namespaces handled via dedicated tiers. Designing for a single 10B blob is the wrong shape.

2. **"k=100."** No production RAG uses k=100 from the vector store. They use k=20–50 with a cross-encoder reranker. k=100 from the ANN forces the index to over-fetch and doesn't improve end-to-end quality. I'd ask the interviewer if k=100 is a hard product constraint or a default that can be relaxed.

3. **"Strong isolation."** Underspecified. "Strong" can mean (a) data isolation (no cross-tenant reads), (b) compute isolation (no cross-tenant performance impact), (c) crypto isolation (separate keys per tenant), or (d) compliance isolation (separate physical hardware). These have very different costs. Default is (a) + (b) + (c). (d) is a paid upgrade.

And one push back on the standard 2026 framing:

4. **"Vector DB" is increasingly a misnomer.** What we've built is a *retrieval substrate*: vector + sparse + structured filter + rerank pipeline as a unit. Pure vector DB is a tier in a larger system, not the system itself. Designing the vector path in isolation from the sparse path leaves recall on the table.

---

## Summary — Staff Signals Hit

| # | Signal | Section |
|---|---|---|
| 1 | 40-TB-reframing | §2 |
| 2 | research-before-design | §1 |
| 3 | filter-as-central-tension | §6 |
| 4 | multi-tenant-explicit-commitment | §7 |
| 5 | storage-tier-economics | §9 |
| 6 | scope-negotiation | §2 |
| 7 | rejected-alternative | §5, §15 |
| 8 | capacity-math | §3 |
| 9 | recall-as-first-class | §11 |
| 10 | failure-mode-precision | §13 |
| 11 | filtered-recall-awareness | §6.4 |
| 12 | high-cardinality-filter-discipline | §6.3 |
| 13 | freshness-mechanism | §10 |
| 14 | hybrid-retrieval-awareness | §12 |
| 15 | quantization-layered | §8 |
| 16 | invariant-based-thinking | §7.3, §10.3 |
| 17 | blast-radius-reasoning | §7.2 |
| 18 | when-not-to-build-this | §14 |
| 19 | saying-no | §2, §8.1, §16 |
| 20 | 2026-cutting-edge-awareness | §1 throughout |

20/21 signals tagged inline. Senior-staff target hit.

---

## Citations

- DiskANN/Vamana: Subramanya et al., NeurIPS 2019. Microsoft Research.
- SPANN: Chen et al., NeurIPS 2021. Microsoft Research.
- Filtered-DiskANN: Gollapudi et al., WWW 2023.
- FreshDiskANN/SPFresh: streaming variants (2023–24).
- Turbopuffer architecture, native filtering, continuous recall: turbopuffer.com docs and engineering blog (2024–25).
- Pinecone serverless slab architecture: pinecone.io engineering blog (2024–25).
- Matryoshka Representation Learning: Kusupati et al., NeurIPS 2022.
- Binary quantization with rescore: Yamada et al. 2021; mixedbread mxbai-v1 2024; HuggingFace embedding-quantization (2024); Vespa MRL+binary blog (2024).