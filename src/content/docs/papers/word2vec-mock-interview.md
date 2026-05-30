---
title: word2vec (Skip-gram + Negative Sampling) — Paper-to-Code Mock Interview
description: A full combined mock (read paper, explain the benefit, implement in Colab) using word2vec skip-gram with negative sampling as the worked example.
sidebar:
  order: 9
  label: word2vec
---

> **Paper:** *Distributed Representations of Words and Phrases and their Compositionality* — Mikolov et al., 2013. arXiv: [1310.4546](https://arxiv.org/abs/1310.4546)
>
> **Format:** Read the paper (~15 min) → explain the *real* benefit → implement the core idea in Colab → sanity-check it.
>
> **Companion notebook:** [`word2vec_mock.ipynb`](/notebooks/word2vec_mock.ipynb) (download) — synthetic-corpus toy task + an `SGNS` stub to fill in, plus verification cells. Open in [Google Colab](https://colab.research.google.com/) via *File → Upload notebook*. A reference solution is included at the bottom of this page.
>
> **Difficulty:** 🟡 Medium. The model is tiny, but the *negative-sampling* trick and the "two embedding tables" detail are where interviews live.

---

## How to run this as a timed drill (~60 min)

Treat this like the real thing. Set a timer and don't look at the answers below until each block is done.

| Time | Block | What you produce |
|------|-------|------------------|
| 0:00–0:15 | **Read** (use the [three-pass method](/papers/lora-mock-interview/#part-0--how-to-read-a-paper-in-15-minutes-the-three-pass-method)) | The SGNS loss + why softmax was too expensive |
| 0:15–0:20 | **Explain the benefit** out loud (cover Part 2) | 1-paragraph pitch + "why negative sampling", "two tables?", "how many negatives" |
| 0:20–0:50 | **Implement** from the stub (Part 3) | A working `SGNS` + same-topic cosine ≫ cross-topic cosine on the toy corpus |
| 0:50–1:00 | **Sanity-check** (Part 4) | All 6 checks passing, talked through out loud |

### Self-grading rubric — "what good looks like"
- ✅ Explained negative sampling as **turning an O(V) softmax into O(k) binary classification**, not just "we sample some words."
- ✅ Knew there are **two embedding tables** (center `v` and context `u`) and could say why they're separate.
- ✅ Wrote the SGNS loss with the **`-` inside the second sigmoid** (`σ(-u_n·v_c)`) — push negatives apart, pull positives together.
- ✅ Demonstrated the benefit with a **number** (same-group vs different-group cosine), not "it runs."
- ✅ Sampled negatives from the **unigram^0.75** distribution and could explain the exponent.
- ⚠️ Red flags: claiming word2vec "uses a softmax" (the whole point is it doesn't), forgetting the minus sign on negatives, using one shared table without justification, no number behind the benefit claim.

---

## Part 1 — Structured read of THIS paper

### The 30-second summary (the "benefit")
You want **dense vectors for words, learned from raw text with no labels**, such that similar words land near each other and analogies fall out of vector arithmetic (the famous `king − man + woman ≈ queen`). The skip-gram objective does this by predicting context words from a center word. The problem: a proper softmax over the whole vocabulary `V` (often millions of words) costs **O(V)** per training example — far too expensive. The payoff of this paper:

- **Negative sampling (SGNS)** replaces the full softmax with a **binary logistic classifier**: distinguish the true `(center, context)` pair from `k` randomly-sampled "negative" pairs. Cost drops from **O(V) → O(k)** with `k` typically 5–20.
- It learns embeddings that capture **semantic and syntactic** regularities (similar words cluster; analogies are linear directions).
- Two extra tricks — **subsampling frequent words** and **learning phrases** ("New_York") — sharpen the vectors further.

### The core idea (Method — you implement this)
Skip-gram maximises the probability of context words given a center word. The expensive full-softmax form is:

$$p(o \mid c) = \frac{\exp(u_o^\top v_c)}{\sum_{w=1}^{V} \exp(u_w^\top v_c)}$$

where `v_c` is the **center** embedding of word `c` and `u_o` is the **context** embedding of word `o`. That denominator is the O(V) cost. **Negative sampling** sidesteps it: instead of normalising over all words, train a logistic classifier that says "real pair" for the true context `o` and "fake pair" for `k` sampled negatives `n`. For one center `c`, true context `o`, and negatives `n_1..n_k`, minimise:

$$\mathcal{L} = -\log \sigma(u_o^\top v_c) \; - \; \sum_{i=1}^{k} \log \sigma(-\,u_{n_i}^\top v_c)$$

where $\sigma(z) = 1/(1+e^{-z})$. The first term **pulls** the true context and center together (dot product up → σ up); each negative term **pushes** a random word away (note the **minus sign** — we want `σ(-u_n·v_c)` large, i.e. `u_n·v_c` small).

Key details (the things an interviewer probes):
- **Two embedding tables.** A word has a **center** vector `v` (when it's the input) and a **context** vector `u` (when it's a predicted neighbour). Keeping them separate makes the optimisation cleaner; people typically keep `v` (or the average) as "the" word vector at the end.
- **Where do negatives come from?** Sampled from the **unigram distribution raised to the 0.75 power**: $P(w) \propto \text{count}(w)^{0.75}$. The exponent damps very frequent words and boosts rare ones relative to plain frequency — empirically better than uniform or raw unigram.
- **`k` is the number of negatives**, ~5–20 for small data, ~2–5 for large. More negatives = better gradient signal but more compute; the cost is **O(k)**, independent of `V`.
- **No labels.** Training signal is purely **co-occurrence**: which words show up near which, inside a context window.

### Where the evidence lives (tables that matter)
*(Figures/numbers below are approximate — confirm against the PDF.)*
- **Analogy accuracy table:** SGNS with negative sampling matches or beats hierarchical softmax on the semantic/syntactic analogy benchmark, far faster than full softmax → the headline efficiency-vs-quality claim.
- **Effect of `k`:** accuracy rises then plateaus as `k` grows → negatives give signal with diminishing returns.
- **Subsampling + phrases:** subsampling frequent words and adding phrase tokens both lift analogy accuracy → the auxiliary tricks earn their place.

### The honest limitations (have an opinion)
- **One vector per word type** — no sense disambiguation. "bank" (river vs money) collapses into one point. Contextual models (ELMo/BERT) later fixed this.
- **Ignores morphology** — "run"/"running"/"ran" are unrelated ids; fastText (subword n-grams) addressed this.
- **Bag-of-context, no order inside the window** — and it can't represent out-of-vocabulary words at all.
- **Sensitive to hyperparameters** — window size, `k`, subsampling rate, and the 0.75 exponent all matter; a lot of word2vec's quality is careful tuning, not just the objective.

---

## Part 2 — The interview dialogue (interviewer ⇄ interviewee)

> **🧑‍💼 Interviewer:** One paragraph — what does word2vec actually give me, and what's the key trick?
>
> **🧑‍💻 Interviewee:** It gives me dense word vectors learned from raw text with no labels, where similar words are close and analogies are linear directions. The skip-gram objective predicts context words from a center word, but a real softmax over the whole vocabulary is O(V) per example — way too expensive. The key trick is negative sampling: instead of normalising over every word, I train a binary logistic classifier to tell the true (center, context) pair apart from `k` randomly-sampled fake pairs. That turns the cost from O(V) into O(k), and the resulting vectors are just as good.

> **🧑‍💼 Interviewer:** Why two embedding tables instead of one?
>
> **🧑‍💻 Interviewee:** Each word plays two roles — as the center word being conditioned on (`v`) and as a context word being predicted (`u`). Giving each role its own vector decouples the gradients and trains more stably; if you tie them, a word's vector appears on both sides of its own dot products which complicates the optimisation. At the end I usually keep the center table `v` as "the" embedding, or average the two.

> **🧑‍💼 Interviewer:** Walk me through the loss — and why is there a minus sign on the negatives?
>
> **🧑‍💻 Interviewee:** For a center `c` and true context `o` it's `-log σ(u_o·v_c)`, which pushes that dot product up so the pair scores as "real." Then for each negative `n` I add `-log σ(-u_n·v_c)`. The minus inside the sigmoid means I'm maximising `σ(-u_n·v_c)`, i.e. driving `u_n·v_c` *down* — pushing random words away from the center. Positives get pulled in, negatives get pushed out, and it's all cheap binary logistic regression.

> **🧑‍💼 Interviewer:** Where do the negatives come from, and why the 0.75 exponent?
>
> **🧑‍💻 Interviewee:** From the unigram distribution raised to the 0.75 power — `P(w) ∝ count(w)^0.75`. Raw unigram over-samples ultra-frequent words like "the"; uniform over-samples rare junk. The 0.75 exponent is a middle ground that damps the frequent words and lifts the rare ones, and Mikolov found it empirically best. `k` is how many negatives per positive, usually 5–20 on small data.

> **🧑‍💼 Interviewer:** Implement it and show that words in the same topic end up with high cosine similarity.

---

## Part 3 — Implementation

The whole model is two embedding tables plus the SGNS loss; negatives are drawn by `torch.multinomial`.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class SGNS(nn.Module):
    """Skip-gram with Negative Sampling: two embedding tables + binary logistic loss."""

    def __init__(self, vocab_size, dim):
        super().__init__()
        self.center = nn.Embedding(vocab_size, dim)    # v: word as the input/center
        self.context = nn.Embedding(vocab_size, dim)   # u: word as a predicted neighbour
        nn.init.uniform_(self.center.weight, -0.5 / dim, 0.5 / dim)
        nn.init.uniform_(self.context.weight, -0.5 / dim, 0.5 / dim)

    def loss(self, center_ids, context_ids, neg_ids):
        # center_ids:(B,)  context_ids:(B,)  neg_ids:(B,k)
        v_c = self.center(center_ids)               # (B, d)
        u_o = self.context(context_ids)             # (B, d)
        u_n = self.context(neg_ids)                 # (B, k, d)

        pos = F.logsigmoid((u_o * v_c).sum(-1))                          # pull true pair together
        neg = F.logsigmoid(-torch.bmm(u_n, v_c.unsqueeze(-1)).squeeze(-1))  # push negatives apart
        return -(pos + neg.sum(-1)).mean()          # scalar


def draw_negatives(num, k, sampling_weights, generator=None):
    """Sample (num, k) negatives from the unigram^0.75 distribution."""
    flat = torch.multinomial(sampling_weights, num * k, replacement=True, generator=generator)
    return flat.view(num, k)
```

### Why each line matters (talk through it)
- **`self.center` / `self.context`** — the two tables. `v` is the word as a center, `u` is the word as a context; separating the roles is the standard SGNS formulation.
- **small `uniform_` init** — so dot products start near 0 and `σ ≈ 0.5` (no structure yet); the model has to *learn* the topic geometry.
- **`(u_o * v_c).sum(-1)`** — the positive logit `u_o·v_c` per example; `F.logsigmoid` is the numerically-stable `log σ`.
- **`torch.bmm(u_n, v_c.unsqueeze(-1))`** — batched matrix-vector product giving each of the `k` negative logits `u_n·v_c`; the **minus** in `logsigmoid(-...)` is what pushes negatives away.
- **`-(pos + neg.sum(-1)).mean()`** — sum the `k` negative terms, add the positive, negate, average over the batch → the scalar to minimise.
- **`torch.multinomial(..., replacement=True)`** — draws `O(k)` negatives per positive; cost is independent of vocab size `V`.

### Demonstrating the benefit (synthetic-corpus toy task)
We build a corpus with **clear co-occurrence structure**: 4 "topics" of 5 words each. Each sentence picks a topic and emits its words (shuffled), so words **co-occur only within their topic, never across topics**. We train SGNS with **no labels** — only `(center, context)` pairs from a sliding window — and check whether same-topic words end up with high cosine similarity.

```python
def build_corpus(seed=0):
    g = torch.Generator().manual_seed(seed)
    topics = [list(range(t * 5, t * 5 + 5)) for t in range(4)]   # 4 topics x 5 words = vocab 20
    vocab_size = 20
    sentences = []
    for _ in range(800):
        t = torch.randint(0, len(topics), (1,), generator=g).item()
        words = topics[t][:]
        perm = torch.randperm(len(words), generator=g).tolist()
        sentences.append([words[i] for i in perm])
    window, centers, contexts = 2, [], []
    for s in sentences:
        for i, c in enumerate(s):
            for j in range(max(0, i - window), min(len(s), i + window + 1)):
                if j != i:
                    centers.append(c); contexts.append(s[j])
    centers, contexts = torch.tensor(centers), torch.tensor(contexts)
    weights = torch.bincount(centers, minlength=vocab_size).float().pow(0.75)  # unigram^0.75
    return centers, contexts, weights / weights.sum(), topics, vocab_size


def train_sgns(seed=0, dim=16, k=5, epochs=60, lr=0.05, batch=256):
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)
    centers, contexts, weights, topics, V = build_corpus(seed)
    model = SGNS(V, dim)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    n, losses = centers.shape[0], []
    for _ in range(epochs):
        perm = torch.randperm(n, generator=g)
        ep, nb = 0.0, 0
        for start in range(0, n, batch):
            idx = perm[start:start + batch]
            neg = draw_negatives(idx.shape[0], k, weights, generator=g)
            loss = model.loss(centers[idx], contexts[idx], neg)
            opt.zero_grad(); loss.backward(); opt.step()
            ep += loss.item(); nb += 1
        losses.append(ep / nb)
    return model, topics, losses


def group_similarities(emb, topics):
    sim = F.normalize(emb, dim=-1) @ F.normalize(emb, dim=-1).t()
    wt = {w: ti for ti, ws in enumerate(topics) for w in ws}
    same, diff = [], []
    for a in range(emb.shape[0]):
        for b in range(a + 1, emb.shape[0]):
            (same if wt[a] == wt[b] else diff).append(sim[a, b])
    return torch.stack(same).mean().item(), torch.stack(diff).mean().item()


model, topics, losses = train_sgns(seed=0)
same, diff = group_similarities(model.center.weight.detach(), topics)
print(f"loss {losses[0]:.3f} -> {losses[-1]:.3f}")
print(f"same-topic cos = {same:+.3f}   diff-topic cos = {diff:+.3f}")
```

You should see the loss fall and **same-topic cosine clearly above cross-topic cosine** (roughly `+0.64` vs `+0.22` with this seed) — the model recovered the topic geometry from co-occurrence alone, no labels. Exact numbers are seed-dependent; the *direction* — same ≫ different — is the point.

---

## Part 4 — Sanity checks (don't skip)

### Check 1 — The SGNS loss is a scalar
```python
m = SGNS(20, 16)
c, o = torch.tensor([0, 1, 2]), torch.tensor([1, 2, 3])
neg = torch.tensor([[4, 5], [6, 7], [8, 9]])
L = m.loss(c, o, neg)
assert L.dim() == 0
print("OK: loss is scalar =", round(L.item(), 4))
```

### Check 2 — At init there is no structure (same ≈ diff ≈ 0)
```python
m = SGNS(20, 16)
topics = [list(range(t * 5, t * 5 + 5)) for t in range(4)]
s0, d0 = group_similarities(m.center.weight.detach(), topics)
assert abs(s0) < 0.2 and abs(d0) < 0.2
print(f"OK: init same={s0:+.3f}, diff={d0:+.3f} (both ~0)")
```

### Check 3 — Negative sampling draws the requested count from the vocab
```python
w = torch.ones(20) / 20
neg = draw_negatives(num=7, k=5, sampling_weights=w)
assert neg.shape == (7, 5)
assert int(neg.min()) >= 0 and int(neg.max()) < 20
print("OK: drew", tuple(neg.shape), "ids in [0,20)")
```

### Check 4 — The loss decreases over training
```python
_, _, losses = train_sgns(seed=0)
assert losses[-1] < losses[0]
print(f"OK: loss {losses[0]:.3f} -> {losses[-1]:.3f}")
```

### Check 5 — After training, same-topic cosine > different-topic cosine (the demo)
```python
model, topics, _ = train_sgns(seed=0)
same, diff = group_similarities(model.center.weight.detach(), topics)
assert same > diff + 0.2
print(f"OK: same={same:+.3f} >> diff={diff:+.3f}")
```

### Check 6 — Gradient flows to BOTH embedding tables
```python
m = SGNS(20, 16)
m.loss(torch.tensor([0, 1]), torch.tensor([1, 2]), torch.tensor([[3, 4], [5, 6]])).backward()
assert m.center.weight.grad.abs().sum() > 0
assert m.context.weight.grad.abs().sum() > 0
print("OK: gradient flows to center and context tables")
```

> **Relationship to full-softmax skip-gram:** if you replaced the negative-sampling loss with a true softmax `p(o|c) = exp(u_o·v_c) / Σ_w exp(u_w·v_c)` and minimised `-log p(o|c)`, you'd get the same kind of embeddings but pay **O(V)** per step. SGNS is the cheap approximation: each step touches only `1 + k` context vectors instead of all `V`.

---

## Part 5 — Likely follow-up questions

- *"SGNS vs hierarchical softmax?"* — Both avoid the full O(V) softmax. Hierarchical softmax replaces it with a binary tree of O(log V) sigmoids; negative sampling replaces it with `k` independent sigmoids (O(k)). SGNS is simpler and usually competitive; HS can help for rare words.
- *"Why is SGNS (almost) implicit matrix factorisation?"* — Levy & Goldberg (2014) showed SGNS implicitly factorises a shifted pointwise-mutual-information (PMI) matrix: `u_w·v_c ≈ PMI(w,c) − log k`. That's why the geometry encodes co-occurrence statistics.
- *"CBOW vs skip-gram?"* — CBOW predicts the center from the averaged context (faster, better for frequent words); skip-gram predicts each context word from the center (slower, better for rare words and small data).
- *"What does the 0.75 exponent do?"* — It interpolates between uniform (exponent 0) and raw unigram (exponent 1) negative sampling, damping frequent words and boosting rare ones; 0.75 was found empirically best.
- *"How do analogies work?"* — Linear structure: `vec(king) − vec(man) + vec(woman)` lands near `vec(queen)` because consistent semantic/syntactic relations become roughly constant offset vectors.
- *"Limitations / what came next?"* — One vector per word (no senses), no morphology, no OOV. fastText added subword n-grams; ELMo/BERT made embeddings contextual.

---

## TL;DR cheat sheet
| Thing | Answer |
|---|---|
| Core idea | Learn word vectors from co-occurrence; predict context from center word |
| Key trick | Negative sampling: O(V) softmax → O(k) binary logistic classification |
| Loss | `-log σ(u_o·v_c) − Σ_n log σ(-u_n·v_c)` |
| Two tables | center `v` (input) + context `u` (predicted neighbour) |
| Negatives from | unigram^0.75: `P(w) ∝ count(w)^0.75`, `k`≈5–20 |
| Training signal | co-occurrence only — **no labels** |
| Benefit | Cheap, high-quality embeddings; similar words near, analogies linear |
| #1 bug | Forgetting the minus sign on the negative term `σ(-u_n·v_c)` |
| Limitation | One vector/word (no senses), no morphology, no OOV |
