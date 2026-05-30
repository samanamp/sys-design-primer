---
title: Mixture-of-Experts — Paper-to-Code Mock Interview
description: A combined mock (read paper, explain benefit, implement in Colab) using Sparsely-Gated MoE / Switch Transformer top-k routing as the worked example.
sidebar:
  order: 17
  label: MoE
---

> **Papers:** *Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer* — Shazeer et al., 2017 ([arXiv:1701.06538](https://arxiv.org/abs/1701.06538)) · *Switch Transformers* — Fedus et al., 2021 ([arXiv:2101.03961](https://arxiv.org/abs/2101.03961)). We frame around **top-k routing** as used in Switch / standard MoE.
>
> **Format:** Read (~15 min) → explain the *real* benefit → implement the core idea in Colab → sanity-check it.
>
> **Companion notebook:** [`moe_mock.ipynb`](/notebooks/moe_mock.ipynb) (download) — a clustered toy task + an `MoE` stub to fill in, plus verification cells. Open in [Google Colab](https://colab.research.google.com/) via *File → Upload notebook*.
>
> **Difficulty:** 🟡🔴 Medium-hard. Routing + dispatch + an auxiliary loss makes this chunkier than Dropout/RMSNorm.

---

## How to run this as a timed drill (~40 min)

| Time | Block | What you produce |
|------|-------|------------------|
| 0:00–0:12 | **Read** (use the [three-pass method](/papers/lora-mock-interview/#part-0--how-to-read-a-paper-in-15-minutes-the-three-pass-method)) | Why conditional compute decouples params from FLOPs/token |
| 0:12–0:17 | **Explain the benefit** out loud (cover Part 2) | Top-k routing + the load-balancing loss |
| 0:17–0:33 | **Implement** from the stub (Part 3) | A working `MoE` + a router that specializes per cluster |
| last 5 min | **Sanity-check** (Part 4) | All checks passing, narrated out loud |

### Self-grading rubric — "what good looks like"
- ✅ Framed MoE as **conditional compute**: grow *total* parameters while keeping *FLOPs per token* roughly constant.
- ✅ Knew routing = **router logits → softmax → top-k → renormalize → weighted combine** cold.
- ✅ Could explain why an **auxiliary load-balancing loss** exists (the router collapses onto a few experts otherwise).
- ✅ Demonstrated the benefit with **specialization** (clusters → distinct experts) and **active ≪ total** params, not just "it runs."
- ⚠️ Red flags: claiming MoE makes per-token compute cheaper than a same-width dense layer (it adds *capacity*, not speed), forgetting to renormalize top-k gates, ignoring load balancing entirely.

---

## Part 1 — Structured read of THIS paper

### The 30-second summary (the "benefit")
A dense layer spends **every** parameter on **every** token. MoE replaces one big feed-forward block with `N` expert sub-networks plus a tiny **router** that sends each token to only its **top-k** experts (often `k=1` or `k=2`). The payoff is **conditional computation**:

- **Scale parameters massively while keeping FLOPs-per-token ~constant.** Total capacity grows with `N`; per-token cost grows with `k` (which stays small). This *decouples model size from compute*.
- **Experts specialize.** Different parts of the input distribution learn to use different experts, so the model behaves like many specialists sharing one router.
- Switch/MoE reports large quality gains at fixed compute budget by trading "more parameters" for "same FLOPs" (the headline scaling tables — exact figures vary by setup).

### The core idea (Method — you implement this)
For a token `x`, the router produces logits, a softmax gives a distribution over experts, we keep the **top-k** and **renormalize** those gate values, dispatch `x` to those experts, and combine their outputs by the renormalized weights:

$$g = \mathrm{softmax}(W_r\, x) \in \mathbb{R}^{N}, \qquad \mathcal{T} = \text{top-}k(g), \qquad \tilde g_e = \frac{g_e}{\sum_{j \in \mathcal{T}} g_j}$$

$$y = \sum_{e \in \mathcal{T}} \tilde g_e \; E_e(x)$$

The router must not **collapse** onto a few experts, so we add an **auxiliary load-balancing loss** (Switch-style). With `f_e` = fraction of tokens routed to expert `e` and `p_e` = mean router probability for `e`:

$$\mathcal{L}_{\text{aux}} = N \sum_{e=1}^{N} f_e \, p_e$$

This is minimized (`= 1`) when both the routing fractions and the mean probabilities are **uniform**, and grows when traffic concentrates — it nudges the router toward balanced usage.

Key details (the things an interviewer probes):
- **`k` controls compute, `N` controls capacity.** Per-token FLOPs scale with `k` (plus the cheap router); total params scale with `N`. That's the whole point.
- **Renormalize the top-k gates.** After dropping the non-selected experts, the surviving gate values must sum to 1 so the combine is a proper convex average.
- **Why the aux loss?** Without it, the router finds it easiest to pick the same one or two experts for everything; the rest never get gradients and die. The aux term keeps traffic spread out.
- **It's not a speedup over a same-width dense layer.** A single token still does `k` expert FLOPs; the win is that you can afford `N ≫ k` experts of capacity at the cost of only `k`.

### Where the evidence lives (tables that matter)
- **Quality-vs-compute scaling tables:** MoE/Switch matches or beats dense baselines at **equal FLOPs** by adding parameters → the conditional-compute claim.
- **Expert-utilization / load-balancing plots:** show traffic spread across experts once the aux loss is on → the collapse-prevention mechanism.
- **Ablations on `k` and `N`:** more experts help capacity; `k` trades quality for cost. (Exact numbers depend heavily on data/scale — treat headline figures as directional.)

### The honest limitations (have an opinion)
- **Load balancing is fragile.** Without (or with mistuned) aux loss, experts collapse; with too much, you over-regularize routing.
- **Systems complexity.** At scale, dispatch means all-to-all communication and **expert-capacity** limits (tokens dropped when an expert is full) — a real engineering cost.
- **Memory.** You hold all `N` experts' parameters even though each token touches `k`; total memory is large even if per-token compute isn't.
- **Training instability / fine-tuning quirks** are commonly reported for large sparse models.

---

## Part 2 — The interview dialogue (interviewer ⇄ interviewee)

> **🧑‍💼 Interviewer:** One paragraph — what does a Mixture-of-Experts layer actually buy me?
>
> **🧑‍💻 Interviewee:** Conditional compute. Instead of one dense feed-forward block where every parameter touches every token, I have `N` expert networks and a small router that sends each token to only its top-k experts. So total parameters — capacity — scale with `N`, but FLOPs per token scale with `k`, which stays at 1 or 2. That decouples model size from compute: I can make the model much bigger without making each token more expensive. Experts end up specializing on different regions of the input.

> **🧑‍💼 Interviewer:** Walk me through the routing math.
>
> **🧑‍💻 Interviewee:** The router is a linear map from the token to `N` logits, softmax gives a distribution over experts. I take the top-k experts, renormalize just those gate values so they sum to 1, run the token through each selected expert, and combine their outputs as a weighted average using the renormalized gates. The renormalization matters — otherwise the combine isn't a proper convex average and magnitudes drift.

> **🧑‍💼 Interviewer:** Why do you need an auxiliary loss at all?
>
> **🧑‍💻 Interviewee:** Because the router collapses. Picking the same one or two experts for everything is a local optimum: those experts get all the gradient and get better, which makes the router pick them even more, and the rest die. The load-balancing loss, `N * Σ_e f_e * p_e`, is minimized when routing fractions and mean probabilities are both uniform, so it pushes traffic to spread across all experts and keeps them all trained.

> **🧑‍💼 Interviewer:** Is MoE faster than a dense layer of the same width?
>
> **🧑‍💻 Interviewee:** No — that's the common misconception. A token still pays `k` experts' worth of FLOPs plus the router. MoE isn't cheaper per token than a same-width dense layer; it lets me add a lot more *capacity* (`N` experts) for only `k`-experts of per-token cost. The trade is more parameters and memory and systems complexity, not less compute.

> **🧑‍💼 Interviewer:** Implement it and show the router specialize.

---

## Part 3 — Implementation

The method is: router → softmax → top-k → renormalize → dispatch → weighted combine, plus an auxiliary load-balancing loss.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class MoE(nn.Module):
    """N expert MLPs + a learned router. Each token goes to its top-k experts;
    outputs are combined by renormalized gate weights. Switch-style aux loss
    discourages router collapse."""

    def __init__(self, dim, hidden, n_experts, k=1):
        super().__init__()
        assert 1 <= k <= n_experts
        self.n_experts, self.k = n_experts, k
        self.router = nn.Linear(dim, n_experts)
        self.experts = nn.ModuleList(
            nn.Sequential(nn.Linear(dim, hidden), nn.ReLU(), nn.Linear(hidden, dim))
            for _ in range(n_experts)
        )

    def forward(self, x):                              # x: (B, dim)
        logits = self.router(x)                        # (B, N)
        probs = F.softmax(logits, dim=-1)             # (B, N) distribution over experts
        topk_w, topk_idx = probs.topk(self.k, dim=-1) # (B, k) gates + ids
        topk_w = topk_w / topk_w.sum(dim=-1, keepdim=True)  # renormalize the survivors

        out = torch.zeros_like(x)
        for slot in range(self.k):
            idx = topk_idx[:, slot]                    # (B,) expert id in this slot
            w = topk_w[:, slot].unsqueeze(-1)          # (B, 1) combine weight
            for e in range(self.n_experts):
                mask = idx == e
                if mask.any():
                    out[mask] += w[mask] * self.experts[e](x[mask])  # dispatch only these tokens

        aux = self.load_balance_loss(probs, topk_idx)
        return out, aux, probs, topk_idx

    def load_balance_loss(self, probs, topk_idx):
        # Switch-style: N * sum_e (fraction routed to e) * (mean prob for e).
        # = 1.0 when uniform; grows under collapse.
        one_hot = F.one_hot(topk_idx, self.n_experts).sum(dim=1).clamp(max=1).float()
        frac = one_hot.mean(dim=0)                     # (N,) token fraction per expert
        mean_prob = probs.mean(dim=0)                  # (N,) mean router prob per expert
        return self.n_experts * (frac * mean_prob).sum()
```

### Why each line matters (talk through it)
- `probs = softmax(logits)` — turns router scores into a distribution; top-k acts on probabilities.
- `probs.topk(k)` — selects each token's `k` best experts; **this is where sparsity / conditional compute happens** (we never call the other `N-k` experts).
- `topk_w / topk_w.sum(...)` — renormalize so the chosen gates sum to 1; the combine is a convex average.
- `for e ... mask = idx == e` — **dispatch**: gather the tokens assigned to expert `e`, run only those through it, scatter back. Each token touches `k` experts, not `N`.
- `load_balance_loss` — `frac` (hard routing fractions) times `mean_prob` (soft router probabilities); multiplying by `N` normalizes the uniform case to 1.0. Added to the task loss with a small coefficient.

### Demonstrating the benefit (clustered toy task)
Synthetic data with several latent **clusters**, where each cluster's target is a **different** nonlinear map. A specializing router should learn to send each cluster to its own expert — and each token only pays for `k=1` expert.

```python
def make_clustered_data(n_per=400, dim=8, n_clusters=4, seed=0):
    g = torch.Generator().manual_seed(seed)
    centers = torch.randn(n_clusters, dim, generator=g) * 4.0
    maps = [torch.randn(dim, dim, generator=g) for _ in range(n_clusters)]  # distinct map per cluster
    X, Y, C = [], [], []
    for c in range(n_clusters):
        xc = centers[c] + 0.5 * torch.randn(n_per, dim, generator=g)
        yc = torch.tanh(xc @ maps[c])
        X.append(xc); Y.append(yc); C.append(torch.full((n_per,), c))
    return torch.cat(X), torch.cat(Y), torch.cat(C)


torch.manual_seed(0)
dim, n_clusters = 8, 4
X, Y, C = make_clustered_data(dim=dim, n_clusters=n_clusters, seed=0)

moe = MoE(dim=dim, hidden=32, n_experts=n_clusters, k=1)
opt = torch.optim.Adam(moe.parameters(), lr=1e-2)
for _ in range(800):
    out, aux, _, _ = moe(X)
    loss = F.mse_loss(out, Y) + 0.01 * aux
    opt.zero_grad(); loss.backward(); opt.step()

# Cluster -> expert assignment matrix
moe.eval()
with torch.no_grad():
    _, _, _, topk_idx = moe(X)
assign = torch.zeros(n_clusters, moe.n_experts)
for cl in range(n_clusters):
    chosen = topk_idx[C == cl, 0]
    for e in range(moe.n_experts):
        assign[cl, e] = (chosen == e).float().mean()
print("cluster -> expert routing matrix:\n", assign)
print("dominant expert per cluster:", assign.argmax(dim=1).tolist())

total = sum(p.numel() for p in moe.parameters())
active = sum(p.numel() for p in moe.router.parameters()) + moe.k * sum(p.numel() for p in moe.experts[0].parameters())
print(f"TOTAL params {total}   ACTIVE/token {active}   ratio {active/total:.3f}")
```

Verified output (seed 0):

```
cluster -> expert routing matrix:
 tensor([[0.0000, 1.0000, 0.0000, 0.0000],
         [0.9975, 0.0000, 0.0000, 0.0025],
         [0.0025, 0.0000, 0.0000, 0.9975],
         [0.0000, 0.0000, 1.0000, 0.0000]])
dominant expert per cluster: [1, 0, 3, 2]
TOTAL params 2244   ACTIVE/token 588   ratio 0.262
```

Each cluster routes ~100% of its tokens to a **distinct** expert (the router specialized), and each token uses only **588 of 2244** parameters — active ≪ total. With more experts the ratio shrinks further: that's the params-vs-FLOPs decoupling. (The *direction* — specialization + active ≪ total — is the point; exact assignments are seed-dependent.)

---

## Part 4 — Sanity checks (don't skip)

### Check 1 — Top-k routing selects exactly k experts per token
```python
moe = MoE(dim=6, hidden=16, n_experts=5, k=2)
x = torch.randn(7, 6)
_, _, _, topk_idx = moe(x)
assert topk_idx.shape == (7, 2)
for row in topk_idx:
    assert len(set(row.tolist())) == 2          # k distinct experts
print("OK: exactly k distinct experts per token")
```

### Check 2 — Combine weights are normalized (sum to 1 per token)
```python
_, _, probs, _ = moe(x)
tw, _ = probs.topk(moe.k, dim=-1)
tw = tw / tw.sum(dim=-1, keepdim=True)
assert torch.allclose(tw.sum(dim=-1), torch.ones(7), atol=1e-6)
print("OK: gate weights sum to 1")
```

### Check 3 — Output shape matches the input-derived shape
```python
out, _, _, _ = moe(x)
assert out.shape == x.shape
print("OK: output shape == input shape")
```

### Check 4 — Aux loss is lower when balanced than when collapsed
```python
N = moe.n_experts
balanced = torch.full((100, N), 1.0 / N)
bal_idx = torch.arange(100).remainder(N).unsqueeze(-1)
collapsed = torch.zeros(100, N); collapsed[:, 0] = 1.0
col_idx = torch.zeros(100, 1, dtype=torch.long)
loss_bal = moe.load_balance_loss(balanced, bal_idx)
loss_col = moe.load_balance_loss(collapsed, col_idx)
print(f"aux balanced={loss_bal.item():.3f}  collapsed={loss_col.item():.3f}")
assert loss_bal < loss_col                       # ~1.0 vs N
```

### Check 5 — Active params/token < total, and gradient flows to selected experts + router
```python
total = sum(p.numel() for p in moe.parameters())
active = sum(p.numel() for p in moe.router.parameters()) + moe.k * sum(p.numel() for p in moe.experts[0].parameters())
print(f"active/token={active}  total={total}  ratio={active/total:.3f}")
assert active < total

moe.zero_grad()
out, aux, _, topk_idx = moe(x)
(out.sum() + aux).backward()
assert moe.router.weight.grad.abs().sum() > 0    # router gets gradient
for e in set(topk_idx.flatten().tolist()):
    assert moe.experts[e][0].weight.grad.abs().sum() > 0   # selected experts get gradient
print("OK: active < total; grads to router + selected experts")
```

### Check 6 — Router specialization after training (the demonstration), asserted
```python
# (re-run the Part 3 training, then assert)
dominant = assign.argmax(dim=1)
assert len(set(dominant.tolist())) == n_clusters          # distinct expert per cluster
for c in range(n_clusters):
    assert assign[c, dominant[c]] > 0.9                   # cluster routes (almost) entirely to one expert
print("OK: router specialized; distinct expert per cluster")
```

All six checks pass with the reference implementation under a fixed seed.

---

## Part 5 — Likely follow-up questions

- *"`k=1` (Switch) vs `k=2` (classic MoE)?"* — `k=1` is the cheapest and simplest (Switch's key simplification); `k=2` gives the router a fallback and smoother gradients at higher per-token cost. Compute scales with `k`.
- *"What is expert capacity / token dropping?"* — At scale each expert has a fixed buffer; if too many tokens route to it, the overflow is dropped (skipped or passed through residually). A capacity factor trades wasted compute vs dropped tokens.
- *"How does this work across devices?"* — Experts are sharded across accelerators, so dispatch is an **all-to-all** communication. The router decides which device each token's compute happens on — a major systems cost.
- *"Why multiply `f_e` (hard) by `p_e` (soft) in the aux loss?"* — `f_e` (argmax counts) has no gradient; `p_e` (softmax) is differentiable. Their product gives the router a gradient that discourages over-loading any expert.
- *"Noisy top-k gating?"* — Shazeer et al. add tunable Gaussian noise to router logits before top-k to encourage exploration and smoother load balancing early in training.
- *"Is MoE the same as ensembling?"* — No. An ensemble runs *all* models per input and averages; MoE runs only `k` of `N` per token via a learned router — sparse, not dense.

---

## TL;DR cheat sheet
| Thing | Answer |
|---|---|
| Core idea | Route each token to its top-k of N experts; combine by renormalized gates |
| Routing | `g = softmax(W_r x)`, take top-k, renormalize, `y = Σ g̃_e E_e(x)` |
| Benefit | **Conditional compute**: total params scale with N, FLOPs/token with k |
| Aux loss | `N · Σ_e f_e · p_e` (= 1 when balanced) prevents router collapse |
| `k` vs `N` | `k` = per-token compute; `N` = total capacity |
| Active vs total | each token touches router + k experts ≪ all N experts |
| #1 misconception | "MoE is faster than a same-width dense layer" (it adds capacity, not speed) |
| Limitations | Load balancing is fragile; all-to-all comms; large memory; instability |
