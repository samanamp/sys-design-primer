---
title: Public API Versioning Strategy
description: Public API Versioning Strategy
---

# API Versioning & Evolution Strategy for a 6-Year-Old, 100K-Customer Public API

## 1. Scope, Reframing, and Pushback Before I Start

**[STAFF SIGNAL: saying no]** Before I design anything, I want to push back on the framing of the prompt. "We cannot break customers" is not a literally honorable invariant. Every API in production breaks some customer at some point — through a security-driven CVE patch, a bug fix that the customer was depending on, an infrastructure change that shifts latency. The honest invariant is: **we will not silently break customers, we will not break customers without published notice and a published policy that they could have planned around, and we will reserve a small, well-defined emergency lane for cases where we must.** That reframing is the first staff move on this question, because every subsequent decision flows from what kind of contract we are actually willing to commit to.

**[STAFF SIGNAL: scope negotiation]** I'm going to commit to a concrete scope so the design has teeth. Assume:

- **Surface**: REST/JSON public API. (gRPC and GraphQL change some answers; I'll note where.)
- **Workload**: Mixed read/write, ~50K req/s peak across the customer base, with a webhook delivery surface as a separate but related contract.
- **SDK**: We publish official SDKs in 5-7 languages and ~60% of customer traffic flows through them. The other 40% is raw HTTP from a long tail of integrators.
- **Domain**: Commerce/platform-adjacent — high cost-of-breakage, but not a regulated financial API where contracts are externally enforced. (If we were Plaid or a payments processor under PSD2, deprecation timelines roughly double and "frozen v1" becomes a regulatory artifact, not a choice.)
- **v1 history**: 6 years live. v1 has accumulated approximately 240 endpoints, 4 major behavioral revisions that we shipped as "non-breaking" and discovered weren't, and an unknown number of customers depending on undocumented behaviors.

**[STAFF SIGNAL: de-facto contract awareness]** That last point is the central forcing function of this entire design. Six years of v1 means our real contract is not the OpenAPI spec — it's **Hyrum's Law in full force**: every observable behavior of the system is now a contract that some customer somewhere depends on. Field ordering in JSON arrays. The exact text of error messages. The latency distribution of a particular endpoint. Whether a 404 returns immediately or after a 200ms DB lookup. The timezone of timestamps when the `Accept-Language` header is absent. The fact that pagination is stable across a single cursor session even though we never promised it would be. **The design problem is not "how do we version going forward"; it is "how do we evolve a system whose true contract is roughly 10x larger than its documented contract."**

**[STAFF SIGNAL: customer segmentation]** And the 100K customers are not a uniform population. From day one I'm going to treat them as three cohorts with different strategies, different timelines, different communication channels, and different escape valves. A deprecation that works for the active middle is hostile to the dormant tail and insulting to the strategic head.

## 2. The Customer Contract and Deprecation Policy — Before Mechanism

**[STAFF SIGNAL: policy-before-mechanism]** I am deliberately writing this section before I touch URL-vs-header. The mechanism exists to enforce the policy. Pick mechanism first and you'll discover six months later that it doesn't support the policy you actually need.

### What "won't break you" means, precisely

We commit to the following, and we publish it as a versioned policy document on our developer site:

1. **No silent breakage.** Any change that alters the wire-level behavior of a documented endpoint receives, at minimum, an in-band response header warning and a written deprecation notice with a deadline.
2. **Documented behavior is contract.** Anything in the OpenAPI spec, the published reference docs, or written explicitly in our deprecation policy is enforceable by customers.
3. **Undocumented behavior is best-effort.** If you depend on the order of fields in a JSON object, or the precise text of an error message, or the relative latency of two endpoints, we will try not to break you, but we make no formal commitment. If enough customers complain about an undocumented break, we treat that as evidence we should have called it out and we apologize and roll back. (This is the practical Hyrum response — we promise documented behavior, we don't pretend undocumented behavior is unobservable, and we treat the gap with humility, not hostility.)
4. **Tiered change classification, published.** We define four tiers of change:
   - **Tier 0 — Safe**: additive (new endpoint, new optional response field, new optional request param with default that preserves prior behavior). Ships any time, no notice.
   - **Tier 1 — Observable but compatible**: latency improvements, new error subcodes within an existing 4xx/5xx class, new enum values in a closed set with a documented "treat unknown values as X" guidance. Shipped with changelog notice but no deprecation.
   - **Tier 2 — Breaking with notice**: removing a field, changing a field type, changing default behavior, changing rate limit semantics. Requires the full deprecation timeline.
   - **Tier 3 — Security/integrity emergency**: forced change with reduced timeline, governed by a separate published emergency policy.

### Deprecation timelines, published as policy

| Change tier | Announcement | In-product nudges | Sunset header | Brownouts | Hard cutoff |
|---|---|---|---|---|---|
| Major version (v1 → v2) | T−24mo | T−12mo | T−6mo | T−3mo | T = 0 |
| Tier 2 within version | T−12mo | T−6mo | T−3mo | T−1mo | T = 0 |
| Tier 3 emergency | T−2wk | T−2wk | T−1wk | T−3d | T = 0 |
| SDK-only breaking change | T−6mo | n/a | n/a | n/a | next major SDK release |

**[STAFF SIGNAL: policy as published artifact]** The critical move is publishing this *before* we need to use it. Once a strategic customer is angry about a deprecation, the worst possible time to negotiate the policy is in the moment. A policy that exists in the developer docs at deprecation time-zero, that customers signed up under, is far easier to enforce consistently than a policy invented under pressure. This is also the only way to say "no" to escape-valve requests without it being a relationship event — you're enforcing a published rule, not making an arbitrary call.

### Escape valves and emergency provisions, also published

- **Paid extended support**: a customer can purchase extended support for a deprecated version at a published price (e.g., $X/month for v1 after T=0, with a published max duration of 24 months post-sunset). This is not a free favor; it's a product line.
- **Migration assistance grants**: for customers who commit to a migration plan with milestones, we will fund migration engineering on our side or theirs. Published criteria (revenue threshold or strategic flag), not ad hoc.
- **Emergency Tier 3 protocol**: pre-published, so when v2 has a critical bug at T+18mo we don't invent the response.

## 3. Versioning Mechanism — Date-Based with Per-Account Pinning

**[STAFF SIGNAL: rejected alternative]** The choice is between four real options, and I'll name what I'm rejecting and why:

- **URL versioning (`/v1/`, `/v2/`)** — explicit, debuggable, cacheable. *Rejected as primary mechanism* because every breaking change creates a global migration event. With 100K customers and 6 years of v1, "global migration event" means a 24-month campaign with measurable churn. URL versioning is honest but expensive. We keep `/v1/` as the legacy mount point but new evolution does not happen through `/v2/`.
- **Header versioning (`Accept: application/vnd.company.v2+json`)** — slightly more granular than URL but has the same all-or-nothing migration property and is harder to debug from a curl one-liner. *Rejected* for the same reasons as URL versioning.
- **Date-based versioning, Stripe-style (`X-API-Version: 2024-04-10`) with per-account pinning** — each customer is implicitly pinned to the version that was current when they integrated; new customers get the latest; existing customers experience zero forced migrations until we deprecate their pinned version. *Chosen.* The cost is high implementation complexity in the translation layer; the benefit is that the cost-of-breakage on the customer side is dramatically lower, and the mechanism mirrors the policy I committed to above.
- **No versioning, additive only (GitHub's older model)** — never break, only add. *Rejected* because the v1 surface already contains decisions we know are wrong (e.g., a field that conflates two concepts; an enum that should have been a sub-resource). Pure additive evolution leaves these forever and the API surface gets worse over time, not better.

We adopt **date-based versioning with per-account pinning, with v1 retained as a legacy URL prefix that we will eventually deprecate**. New evolution happens via dated versions on a `/v2/`-or-unversioned base path; v1 is a separate facade that we maintain for the existing dormant tail.

### Request flow with version handling

```
Client request
  Headers: X-API-Version: 2024-04-10  (optional override)
  Auth:    api_key=sk_live_...
  
        |
        v
+-------------------------+
| Edge Gateway            |
|  - terminate TLS        |
|  - lookup api_key       |
|  - resolve account_id   |
|  - fetch account.pin    |
+-------------------------+
        |
        v
+-------------------------+
| Version Resolver        |
|  effective_version =    |
|    header_override      |
|    ?? account.pin       |
|    ?? account.created_at|
|    ?? "latest stable"   |
+-------------------------+
        |
        v
+-------------------------+
| Compat Library (UP)     |
|   external_request_v_X  |
|     -> canonical_v_INT  |
|   (additive transforms, |
|    field renames,       |
|    default injections)  |
+-------------------------+
        |
        v
+-------------------------+
| Internal RPC dispatch   |
|   gRPC -> charges v9    |
|   gRPC -> customers v4  |
|   gRPC -> ledger v7     |
+-------------------------+
        |
        v
+-------------------------+
| Compat Library (DOWN)   |
|   canonical_v_INT       |
|     -> external_resp_v_X|
+-------------------------+
        |
        v
Client response
  Sunset: Wed, 01 May 2027 00:00:00 GMT  (if applicable)
  Deprecation: true                       (if applicable)
  X-API-Version-Used: 2024-04-10
```

The `X-API-Version-Used` response header is non-negotiable. It tells the customer exactly what version we evaluated their request as, which is critical for support — when a customer says "the API broke," the first question is "which version did you actually hit," and we want that answer to be in their logs without them having to instrument anything.

## 4. Internal vs External Versioning — The Translation Layer Is Where the Complexity Lives

**[STAFF SIGNAL: internal/external decoupling]** This is the section most candidates miss entirely, and it's the section that actually matters for cost.

The external API version is a **facade**. It is not the version of any single backend service. v2 of our public API may be served by version 9 of the charges service, version 4 of the customers service, and version 7 of the ledger service. Internal services evolve at their own cadence under their own backward-compatibility discipline (typically additive-only protobuf, with `reserved` slots for removed fields and never recycling tag numbers).

### Translation layer architecture

```
External versions (date-pinned, ~12 supported concurrently)
  2019-05  2020-08  2021-03  2022-01  2023-08  2024-04  2025-09 ...
     \        \        \        \        \        \        /
      \        \        \        \        \        \      /
       \        \        \        \        \        \    /
        v        v        v        v        v        v  v
       +--------------------------------------------------+
       |              COMPAT LIBRARY                       |
       |   per-version transforms, both directions         |
       |   small, declarative, heavily tested              |
       +--------------------------------------------------+
                            |
                  canonical internal model
                  (one shape, the "current truth")
                            |
            +---------------+---------------+
            |               |               |
            v               v               v
       charges-svc     customers-svc     ledger-svc
        v9 (gRPC)       v4 (gRPC)         v7 (gRPC)
       additive,        additive,         additive,
       per-service      per-service       per-service
       evolution        evolution         evolution
```

The critical architectural insight: **we do not write `n × m` translation paths for `n` external versions and `m` internal versions.** We write a chain. Every external version translates *up* to the latest canonical internal model and every internal response translates *down* to the requested external version. New external version = add one new pair of transforms (request-up, response-down). Old external versions don't change — their transforms are immutable, well-tested, and just keep running.

This is the Stripe pattern in essence: each version transition is a single "compat patch" that knows how to translate the diff between version `N` and version `N+1`. Stack them and you can translate any version to any version. The patches are intentionally small — usually 1-5 field renames, defaults, type coercions — and reviewed under a much higher bar than ordinary code changes.

### Where the translation lives — gateway vs service

I am putting the compat library at **the gateway/edge tier, not in each service**. The argument:

- **Gateway placement (chosen)**: services see only the canonical internal contract. Service teams don't need to know the external version exists. Cleaner service code, easier service evolution, single place to audit version-translation correctness. The cost is that the gateway becomes thicker — it runs request-shaped logic, not just routing — and gateway team becomes a high-contention shared resource.
- **In-service placement (rejected)**: every service understands every external version. Distributes the load but multiplies the surface where version bugs can hide and forces every service team to learn the external version model. We've seen companies do this; the failure mode is that a service team backports a fix to v_external_2024 but forgets v_external_2023 and a small percentage of traffic silently corrupts.

The gateway-placement choice has a corollary: **we invest hard in the compat library**. It is its own first-class service with its own team, code review standards higher than feature work, mandatory contract tests, and a published deprecation pipeline. It is the most-leveraged code in the company.

### What about webhooks?

Webhooks are an inverted version problem. The customer is now the server and we are the client — and they cannot pin a version on inbound traffic the way a request can. Webhook versioning is per-endpoint-registration: when a customer creates a webhook endpoint, they pin the schema version for that endpoint. Our delivery service reads the pin, runs the response through the compat library *down* to that version, and delivers. Same library, opposite direction.

## 5. Capacity Math and the Version Tax

**[STAFF SIGNAL: capacity math]** Numbers, with the arithmetic shown.

### Customer segmentation, estimated

| Cohort | Size | API call share | Behavior | Migration feasibility |
|---|---|---|---|---|
| Dormant | ~70,000 | ~5% | No calls in 90 days; integration written, deployed, forgotten | Will not migrate voluntarily; will only respond to brownouts or hard cutoff |
| Active middle | ~28,000 | ~60% | Calls weekly, has engineers who maintain the integration | Will migrate with 12mo notice and decent migration tooling |
| Strategic head | ~2,000 | ~35% | Top revenue / strategic accounts, named CSMs | Will migrate, but on their schedule, with assistance |

Numbers are illustrative; the actual cohort sizes come from the telemetry pipeline (Section 6, deep dive 2). The point of writing them down is that **the deprecation campaign for the dormant 70K is operationally completely different from the campaign for the strategic 2K**, and conflating them is the central mid-level mistake.

### Version tax — engineering hours per supported version per year

Approximate budget per concurrent supported version:

| Cost line | Hours/year |
|---|---|
| Compat library maintenance (rule reviews, occasional bug fixes) | ~80 |
| Test matrix maintenance (per-version contract tests) | ~120 |
| Bug-fix backporting (avg 3-4 bugs/yr in shared code paths) | ~60 |
| Documentation generation, review, accuracy checks | ~40 |
| Customer support training/refresher on this version's quirks | ~20 |
| **Per-version annual cost** | **~320 hours ≈ 0.18 FTE** |

If we support 12 concurrent date-pinned versions plus v1-legacy: **~13 × 320 = ~4,160 hours/year ≈ 2.3 FTE of pure version tax.** If we support 24 versions, it doubles. The tax is not literally linear — some shared infrastructure scales sub-linearly — but it is also not free, and many companies discover at year 4 that they are paying 5+ FTE in version tax and never put it on a roadmap.

**Policy implication**: we set a **bounded version-count policy of 12 concurrent supported versions** (plus the legacy v1 mount). Each new version published forces the deprecation announcement of the oldest. This makes the version tax bounded and predictable, and it makes deprecation a routine operational event rather than a crisis.

## 6. Deep Dives

### 6.1 What counts as a breaking change — the de-facto contract problem

The four-tier classification I described in Section 2 is necessary but not sufficient. The real problem is the **secret-breaking-change** category — changes that look safe by every documented criterion but break a real customer because of Hyrum's Law.

Examples we've actually seen in v1:
- Reducing p99 latency from 800ms to 200ms. A customer had a hardcoded 500ms timeout. Previously their requests always failed and they had downstream retry logic. After the latency improvement their requests started succeeding, and their retry logic created duplicate orders.
- Stabilizing a previously-flaky pagination cursor. Customers had written code that assumed they'd see duplicates across cursor pages and deduplicated client-side. After we stabilized the cursor, a small number of customers' dedup logic started discarding legitimate distinct entities.
- Fixing an off-by-one in a list count returned in metadata. Customers depended on the wrong count for their internal accounting reconciliation; the fix made their books not balance until they rewrote the reconciliation.

**Our doctrine**: we do not pretend this category does not exist. Three operational responses:
1. **Canary every change, including "non-breaking" ones**, against synthetic-customer fleets that exercise common usage patterns. We catch obvious latency and behavioral changes in canary.
2. **For high-traffic endpoints, run a dual-execution shadow** — for a small percentage of traffic, run both old and new code paths and diff the responses byte-by-byte. Diffs go to a dashboard. A staff engineer reviews the dashboard before any "non-breaking" change ships to 100%.
3. **Customer-driven contract testing**, where we encourage customers to publish their own contract expectations (via a CDC framework or just documented test fixtures), and we run those against pre-release versions. This works for the strategic 2K. It does not work for the dormant 70K.

The hard truth for the dormant tail: we cannot defend against every secret-breaking-change for customers we have no contact with. The brownout strategy (Section 6.4) is partly about generating that contact.

**[STAFF SIGNAL: invariant-based thinking]** The invariants we are committing to enforce: (a) no silent change to documented behavior, (b) any change to undocumented behavior is reviewed for plausible Hyrum-class impact and shadowed before rollout, (c) every change ships behind a feature flag we can revert in seconds, not hours.

### 6.2 Usage measurement — including the field-level problem

**[STAFF SIGNAL: usage-measurement discipline]** You cannot deprecate what you cannot measure. The measurement stack:

**Endpoint-level / version-level / customer-level** is straightforward — every request gets logged with `(account_id, api_key, endpoint, version_evaluated, status, latency, request_id)` and aggregated into an OLAP store. Standard. We've already done this for v1 for years.

**Field-level usage** is the genuinely hard problem. We need to know: of the 240 endpoints, which response fields are customers actually reading? Because if we want to deprecate a field, we need to know who depends on it.

Three mechanisms in order of fidelity:
1. **SDK telemetry** (highest fidelity, only 60% coverage). Our official SDKs deserialize JSON into typed objects. We can instrument the deserializer to record which fields each customer's code path actually accesses. This requires customer opt-in for telemetry and an SDK update cycle, but for the 60% of traffic on official SDKs, it's the gold standard.
2. **Request-side proxy** (full coverage, lower fidelity for response reads). For request fields, we know exactly what the customer sent. We can deprecate request fields with high confidence based on this alone.
3. **Canary-with-removal** (full coverage, expensive to run). For a tiny fraction of traffic, we ship a canary that has the field removed/zeroed/null. Errors and customer complaints surface dependence. This is a last resort because it briefly breaks customers, but for a field we believe is unused, it's the most reliable test. We schedule these explicitly, with announcements ("we will canary-remove field X on date Y"), so any customer who actually reads documentation can self-identify.

**The "set-and-never-read" pattern**: a customer that PUTs a field but never reads response data. This is common in webhook senders and write-mostly integrations. Removing the field from the response is safe; removing it from the request is not. Our usage telemetry distinguishes the two.

**Per-customer migration dashboards**: we expose to each customer (in their developer dashboard) a view of "you are using these endpoints at these versions; this version is sunsetting on date X; here are the migration steps relevant to your usage." This is the highest-leverage piece of customer-facing tooling we ship. It moves the migration work from "customer reads docs" to "customer follows a personalized checklist."

### 6.3 Per-customer pinning and rollout-without-rollout

**[STAFF SIGNAL: rejected alternative]** Per-customer pinning vs global default with override.

- **Global default with override (rejected)**: there's a single "current version" that all customers hit unless they explicitly pin. New version becomes default = forced migration for everyone simultaneously.
- **Per-customer pinning (chosen)**: every account has an effective version. Default-on-account-creation is the version current at that time. Customers can override per-request via header, but their account pin is the truth.

The data model: `account.api_version_pin` is set at first API key creation time. Each API key inherits the account pin but can override (so a customer's dev API key can be on a newer version than their prod key). Header on a request can override the key's version (so they can test a specific version with curl). The override hierarchy is request-header > api-key-pin > account-pin > "latest at integration time."

**Rollout-without-rollout**: when we ship a new version, no existing customer is affected. They keep their pin. New customers who integrate today get the new version automatically. This is the magic of the model — we ship continuously and customers experience zero forced churn until we explicitly deprecate their pinned version.

**Voluntary upgrade**: customers can update their pin via dashboard or API (`POST /v1/account/api_version` with the desired version). They do this when they're ready to adopt new features. We make this a first-class workflow with a "what's new since your version" diff tool.

**Forced deprecation**: at T=0 of a deprecation, we batch-update pins from the deprecated version to the next supported version. This is a database-level operation for the dormant tail; for the active middle, they've usually upgraded voluntarily by then; for the strategic head, we've negotiated the timing.

### 6.4 Brownout strategy

**[STAFF SIGNAL: brownout strategy]** Brownouts are the most underused tool in API deprecation, and they're the only thing that reliably surfaces dormant integrations.

Mechanism: in the final 3 months before a version's hard cutoff, we periodically return errors for short windows for that specific version. Schedule (published in advance, in the developer changelog and via email):

```
T-90 days:  1 hour brownout, every Tuesday 14:00-15:00 UTC
T-60 days:  2 hours, twice a week
T-30 days:  4 hours, three times a week
T-14 days:  whole-day brownouts on Mon, Wed, Fri
T-7 days:   continuous except 09:00-17:00 UTC for support window
T = 0:      sunset; HTTP 410 Gone with migration link
```

What this does:
- **Forces dormant integrations to surface**. A customer who hasn't read an email in 6 months gets paged when their integration breaks for an hour. They migrate or contact support. Either way, they exit "dormant."
- **Distributes the migration load**. Without brownouts, 100% of dormant migrations land at T=0, customer support is overwhelmed for 2 weeks. With brownouts, the load spreads over the final 90 days.
- **Provides a known, documented schedule**. This is critical — brownouts are not surprise outages. Customers can plan around them, and the pattern is discoverable in our changelog.

The error returned during a brownout is a 503 with a structured body explaining what's happening, the sunset date, the migration link, and a "to disable brownouts for this account, upgrade your version pin" CTA. The 503 (vs 410) is intentional — 503 is recoverable, the customer's retry logic will handle it. 410 is reserved for the actual sunset.

### 6.5 The strategic-customer escape valve

**[STAFF SIGNAL: business-decision boundary]** This is the depth area where staff engineers separate from senior engineers.

Scenario: at T=0 of v1 deprecation, 200 customers are still on v1, including 3 that represent 20% of company revenue. What do we do?

The 200 dormant tail: the brownouts have caught the catchable ones. The remaining 197 are likely abandoned integrations — services nobody owns anymore. We white-glove them: dedicated engineering rep reaches out, offers free migration consulting, in many cases offers to write the migration PR for them. Most convert. Some go silent and we eventually shut off v1 for them with full notice; they were going to churn anyway.

The 3 strategic customers: this is **not an engineering decision**. Engineering's job is to articulate cost and option, not to make the call.

The cost data we put in front of leadership:
- Maintaining v1 in its current frozen state for 3 customers: ~0.4 FTE/year (compat library maintenance, test matrix, security patches).
- Maintaining v1 with feature parity to current versions: not viable; we wouldn't do this.
- Risk: any v1 security CVE in a shared dependency requires us to patch v1 even if we'd rather not.
- Operational: v1 runs on its own isolated stack; an incident there has separate paging and on-call.

The options for leadership:
1. **Frozen v1, indefinitely**. We isolate v1 on minimal dedicated infrastructure, freeze the feature set, accept the ~0.4 FTE cost as a strategic-account-retention investment. Cost: ~$300K/yr. Decision criterion: is the strategic value of these 3 accounts worth this annual spend?
2. **Paid extended support**. Charge the strategic customers for v1 maintenance at a rate that covers our cost plus margin. Forces a real conversation about migration urgency on their side.
3. **Funded migration**. We pay for or staff their migration effort. This is the right answer if their stated blocker is "we don't have engineering capacity," and we have reason to believe that.
4. **Negotiated multi-year migration commitment**. Joint plan with milestones, regular check-ins, and a hard end date 18-24 months out. This is usually the right answer, combined with option 3.

The published policy explicitly enumerates these options, so the conversation with the strategic customer isn't "please don't break us" but "which of the four published options would you like to engage with?" That's a totally different negotiation.

**The hard line**: option 1 indefinitely is a slippery slope. After 5 years of "frozen v1," we've accumulated 5 years of additional security patches, the team that built v1 is gone, and the cost of maintenance is 5x what we projected. **I will recommend to leadership a published maximum frozen-version lifetime of 36 months post-sunset, after which the version is shut down regardless of remaining customers.** This is the only way to make the cost actually bounded.

## 7. Emergency Change Handling — The 2-Week-v3 Case

**[STAFF SIGNAL: emergency-exception policy]** Scenario: v2 has a critical bug — pick the bad one: a security issue that allows account A to read account B's data through a misconfigured authorization check on a specific endpoint. The fix is itself a breaking change because it requires reshaping the response to remove a field that should never have been there. Two-week timeline.

The wrong response is to invent the policy in the moment. The right response is to execute the pre-published Tier 3 protocol:

**Day 0** — incident declared. Leadership informed. Comms plan activated.
**Day 0-2** — **mitigation in v2**: we patch v2 to return 403 on the affected endpoint. This is itself a breaking change (clients depending on success now get 403), but it's strictly less harmful than the data leak. Customers are notified within 24 hours via mandatory email + dashboard banner + Sunset header on every response from v2's affected endpoint.
**Day 2-7** — **v3 (or, more precisely, a new dated version that supersedes v2)** is shipped with the redesigned response. Compat library entry is added. New customers get v3 by default; v2-pinned customers continue to receive 403 on the affected endpoint with a migration link.
**Day 7-14** — active migration support, white-glove for high-traffic customers, automated migration tooling for SDK users, brownouts on remaining v2 traffic for the affected endpoint to force surfacing.
**Day 14** — affected endpoint on v2 hard-cutoff. Other v2 endpoints continue to function normally; the deprecation is **scoped to the affected surface, not the entire version**.

Two critical features of this response:
1. **Scoped deprecation**, not whole-version. We don't sunset v2 — we sunset the broken endpoint of v2 and force migration for customers who use that endpoint. This dramatically reduces the blast radius.
2. **Emergency exceptions are rare and well-justified.** If we exercise this protocol more than once a year, the deprecation policy becomes meaningless. Every emergency exception triggers a postmortem that interrogates whether we could have caught the issue earlier (security review, contract testing, fuzz testing of authorization paths).

**[STAFF SIGNAL: blast radius reasoning]** What if the compat library itself has a bug? Every external version flows through it. A bug in a single transform corrupts data for one specific version pair. We mitigate via: (a) per-transform versioning and the same diff-shadow approach as Section 6.1 — every new transform runs in shadow mode against production traffic and the diffs are reviewed before promotion; (b) instant per-transform rollback at the gateway, so if a transform misbehaves we revert it without rolling back any service.

## 8. Operational Realities at Year 3

**[STAFF SIGNAL: operational longevity]** A staff engineer thinks about this in year 3, not at launch.

- **Documentation rot**. We commit to per-version doc generation directly from the OpenAPI specs that drive the compat library. Manual doc maintenance for 12 versions is impossible; auto-generation with version-aware examples is mandatory. Every PR that adds a transform adds the doc diff in the same PR or it doesn't merge.

- **SDK matrix**. Each official SDK pins a default API version in its release. SDK v3.x targets API version 2024-04-10; SDK v3.5 targets 2025-09-01. The matrix of supported `(SDK_version, API_version)` combinations gets gnarly. Mitigation: SDKs support "any API version within the SDK's release window," but we publish a compatibility table and we deprecate SDK versions on their own clock that's roughly aligned with API version deprecations but not identical.

- **Customer support training**. Support engineers get a per-version reference card and the customer dashboard shows the support engineer the customer's current version pin so they don't waste time debugging the wrong contract.

- **Postmortem culture**. Every version-related incident — a transform bug, a misconfigured pin, a brownout that hit the wrong cohort, a customer who experienced silent corruption — produces a public postmortem. The postmortems improve the policy. Year 3 of this is when the policy starts being actually mature.

- **The 5-year regret list**. Things I'd do differently for a hypothetical v3 design from scratch: separate API key creation from version pinning (so customers can rotate keys without re-pinning), version webhooks separately from the request API from day one, build the per-customer migration dashboard before we need it, and **invest in the field-level usage measurement before we have customers**, not after. The field-level usage problem in v1 is fundamentally unsolvable for the dormant tail because we never instrumented it; in v_next we instrument from request 1.

## 9. Real-World References, with Architectural Substance

**[STAFF SIGNAL: real-world reference]**

- **Stripe**: date-based versioning (`Stripe-Version: 2024-04-10`), per-account pinning, compat library handles all translations. Public commitments to never-break-pinned-versions in practice — they have versions still working from 2011. Architectural significance: Stripe makes the cost trade explicitly — they pay enormous compat-library tax in exchange for zero forced customer migration, and for a payments API where customer migration cost is "we have to re-certify our PCI integration," that trade is correct. For a developer-tools API where customer migration is cheap, the same trade is overkill.

- **GitHub**: started with header/media-type versioning (`Accept: application/vnd.github.v3+json`), evolved toward additive-only changes with feature opt-in via Accept headers (preview features). Architectural significance: GitHub's REST API surface is extensible enough that additive evolution works for most changes; for the genuinely-breaking cases they introduced GraphQL as the new surface, leaving REST mostly frozen. This is the "fork the API" strategy and it's underrated — sometimes the right answer to "we need to break things" is "ship a parallel surface and let customers migrate at will."

- **Kubernetes**: API group versioning with explicit lifecycle stages (`v1alpha1` → `v1beta1` → `v1`), with explicit guarantees per stage. Alpha = may change without notice, off by default. Beta = enabled by default, may have breaking changes with notice. GA = formal deprecation policy. Architectural significance: the per-stage *contract* is the key insight. Your customers know what kind of stability they're getting based on the stage label. Our equivalent is the `Tier 0/1/2/3` change classification, which is the same insight applied to changes rather than to whole APIs.

- **AWS SDK / API**: per-service versioning (each AWS service has its own version), almost-never-deprecate culture, additive-heavy. Architectural significance: AWS is on the extreme "compat tax is a cost of being AWS" end of the spectrum. They support versions effectively forever. The cost is borne in the compat layer and is accepted as a structural cost of the business.

- **The contrast — companies that got it wrong**: I won't name-and-shame, but the failure modes are well-documented. Common patterns: (a) header-versioning with a default-version-that-changes-silently — customers built integrations without setting the header, the default got bumped, integrations broke. (b) Whole-API versioning with no per-customer pinning — every version bump is a global migration event with no graceful path. (c) Versioning the API but not the SDK independently — customers are forced to upgrade SDKs to get unrelated bug fixes. The lessons are baked into the design above.

## 10. Tradeoffs Taken and What Would Change Them

- **If we owned the SDK comprehensively (>90% of traffic on official SDKs)**, field-level usage measurement becomes solved and we can deprecate fields aggressively. We'd shorten Tier 2 timelines from 12mo to 6mo.
- **If we were a regulated financial API**, all timelines roughly double, "frozen v1" becomes a regulatory artifact, and the version count goes from 12 supported to 30+ supported because regulatory contracts often outlive technical contracts.
- **If we were earlier-stage (10K customers, 2 years of v1)**, I'd consider URL versioning with hard cutovers — the customer-support burden is bearable at that scale and the simplicity of the implementation pays for itself. Date-based versioning is the right answer at our scale; it'd be over-engineered at 1/10th our scale.
- **If we had self-service migration tooling** (customer-facing tools that auto-rewrite their integration), Tier 2 timelines could shrink to 3-6mo and the dormant tail could be migrated automatically.
- **If our domain were truly stable** (the data model is genuinely correct and unlikely to need breaking evolution), GitHub's additive-only model becomes attractive — but for any domain with active product evolution, the cost of "we can never fix our wrong abstractions" eventually exceeds the cost of versioning.

## 11. What I'm Pushing Back On

**[STAFF SIGNAL: saying no]** Three things in the prompt I think are wrong, over-specified, or evidence of an X-Y problem:

1. **"We cannot break customers."** Cannot be honored literally. Every Tier 3 emergency, every security CVE, every fundamental data-model fix breaks someone. The honest commitment is "we won't break customers silently or without published policy." If leadership wants the literal version, that's a request for option 1 above (frozen versions indefinitely) and the cost should be put in front of them with a number.

2. **"100K customers" without segmentation** is operationally meaningless for this design. The strategy for 70K dormant customers is fundamentally different from the strategy for 2K strategic customers, and any plan that treats them uniformly is a plan that fails one of the two cohorts. If the prompt is "we have 100K customers and we want one strategy," I push back on the premise.

3. **The implicit framing that engineering owns this decision.** Deprecation timing, strategic-customer treatment, and frozen-version policy are business decisions that engineering informs with cost and risk data. If the engineering team is making these decisions unilaterally, the company has a leadership-clarity problem, not an engineering problem. My job is to surface the cost, articulate the options, recommend, and execute the chosen path — not to pretend I have authority I don't.

---

## Final Note on Approach

The reason this answer is structured policy-first is that **the technical mechanism is downstream of the contract**. URL vs header vs date-based is a choice with three reasonable answers depending on the contract you're committing to. Pick the contract first and the mechanism falls out. Pick the mechanism first and you'll spend two years discovering it doesn't fit the contract. The 6-year-v1, 100K-customer, "cannot break" constraint set forces date-based-with-pinning; in a different constraint set, URL versioning would be the right answer. The staff move is recognizing which constraint set you're actually in and committing to the matching design with full conviction — not retreating to "it depends," not enumerating without committing, and not pretending the policy questions are someone else's job.