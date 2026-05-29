---
title: DPO — Paper-to-Code Mock Interview
description: A combined mock (read paper, explain benefit, implement in Colab) using Direct Preference Optimization as the worked example.
sidebar:
  order: 16
  label: DPO
---

> **Paper:** *Direct Preference Optimization: Your Language Model is Secretly a Reward Model* — Rafailov et al., 2023. arXiv: [2305.18290](https://arxiv.org/abs/2305.18290)
>
> **Format:** Read (~15 min) → explain the *real* benefit → implement the core idea in Colab → sanity-check it.
>
> **Companion notebook:** [`dpo_mock.ipynb`](/notebooks/dpo_mock.ipynb) (download) — a preference-alignment toy task + a `dpo_loss` stub to fill in, plus verification cells. Open in [Google Colab](https://colab.research.google.com/) via *File → Upload notebook*. A reference solution is included at the bottom of this page.
>
> **Difficulty:** 🟡🔴 Medium-hard. The loss is short, but the *conceptual* leap (why this replaces a whole RL pipeline) is what's tested.

---

## How to run this as a timed drill (~60 min)

| Time | Block | What you produce |
|------|-------|------------------|
| 0:00–0:15 | **Read** (use the [three-pass method](/papers/lora-mock-interview/#part-0--how-to-read-a-paper-in-15-minutes-the-three-pass-method)) | The DPO loss + why it removes the reward model and the RL loop |
| 0:15–0:20 | **Explain the benefit** out loud (cover Part 2) | The "policy is secretly a reward model" reparameterization |
| 0:20–0:50 | **Implement** from the stub (Part 3) | A working `dpo_loss` + a policy whose implicit reward orders preferences |
| last 10 min | **Sanity-check** (Part 4) | All 6 checks passing, narrated out loud |

### Self-grading rubric — "what good looks like"
- ✅ Explained DPO as **RLHF without a separate reward model and without RL/PPO** — a single classification-style loss on preference pairs.
- ✅ Knew the loss is **on log-prob *differences* relative to a frozen reference**, not on raw log-probs.
- ✅ Could state the **implicit reward** `r(y) = β·log(πθ(y|x)/πref(y|x))` and why ordering by it recovers the preferences.
- ✅ Knew the reference policy is **frozen** and why (it anchors the KL constraint).
- ⚠️ Red flags: describing a PPO loop, training a reward model, forgetting the reference term, claiming DPO needs online sampling (it's offline on a fixed preference dataset).

---

## Part 1 — Structured read of THIS paper

### The 30-second summary (the "benefit")
Standard RLHF aligns a language model in **three** stages: supervised fine-tune, then **train a separate reward model** on human preference pairs, then **optimize the policy against that reward with RL (PPO)** plus a KL penalty to a reference model. That pipeline is fiddly — reward-model overfitting, unstable on-policy sampling, lots of hyperparameters. DPO collapses the last two stages into **one supervised loss**:

- **No reward model** — the policy's own log-probs (relative to a frozen reference) *are* the implicit reward.
- **No RL loop** — no PPO, no online rollouts, no value network. It's a plain, stable, offline classification-style objective over preferred-vs-dispreferred pairs.
- **Same optimum** as the RLHF objective it replaces, but far simpler and cheaper to run.

### The core idea (Method — you implement this)
RLHF maximizes expected reward minus a KL penalty to the reference policy. That constrained problem has a **closed-form optimal policy**: $\pi^*(y\mid x) \propto \pi_{\text{ref}}(y\mid x)\,\exp\!\big(\tfrac{1}{\beta} r(x,y)\big)$. Invert it and the reward is expressible **through the policy itself**:

$$r(x, y) = \beta \, \log \frac{\pi_\theta(y \mid x)}{\pi_{\text{ref}}(y \mid x)} + \beta \log Z(x)$$

Plug that into the Bradley–Terry preference model $P(y_w \succ y_l) = \sigma\big(r(x,y_w) - r(x,y_l)\big)$. The partition term $Z(x)$ **cancels** (it's the same for both responses), leaving a loss with no reward model and no RL — for a preferred response $y_w$ and dispreferred $y_l$:

$$\mathcal{L}_{\text{DPO}} = -\log \sigma\!\Big( \beta \big[ \big(\log \pi_\theta(y_w\mid x) - \log \pi_{\text{ref}}(y_w\mid x)\big) - \big(\log \pi_\theta(y_l\mid x) - \log \pi_{\text{ref}}(y_l\mid x)\big) \big] \Big)$$

Key details (the things an interviewer probes):
- **It's a logistic-regression-style loss** on the *difference of log-ratios*. You push the policy to raise the log-prob of $y_w$ and lower that of $y_l$ — **relative to the frozen reference**.
- **`β`** controls how far the policy may drift from the reference (the KL strength). Larger `β` = stay closer to ref.
- **The reference `πref` is frozen** (usually the SFT model). It anchors the KL constraint; if you let it move, you lose the regularizer.
- **The implicit reward** is $r(x,y) = \beta\,\log\frac{\pi_\theta(y\mid x)}{\pi_{\text{ref}}(y\mid x)}$. After training, **ranking responses by this implicit reward should match the human preferences** — "your language model is secretly a reward model."
- **Offline & stable:** trains on a fixed dataset of $(x, y_w, y_l)$ triples, no sampling from the current policy.

### Where the evidence lives (tables/figures that matter)
*(Hedge on exact numbers — quote the shape of the result, not memorized digits.)*
- **Sentiment / summarization / dialogue experiments:** DPO matches or beats PPO-based RLHF on the reward-vs-KL frontier — same alignment, simpler method. This is the core benefit claim.
- **Reward-vs-KL frontier figure:** DPO reaches higher reward at the same KL divergence from the reference → it's not just simpler, it's competitive on the actual tradeoff.
- **Stability/ablations:** DPO is less sensitive to the sampling temperature and hyperparameters that make PPO finicky.

### The honest limitations (have an opinion)
- **Needs a good reference & a preference dataset:** it's offline, so it can only exploit the pairs you give it; no exploration of new responses.
- **Can over-optimize / degenerate** if `β` is too small (policy drifts far from ref) — you still need the KL anchor.
- **Distribution shift:** because it never samples on-policy, the preference data must cover responses near where the policy ends up; off-distribution pairs help less than in online RLHF.
- **Length / reward-hacking biases** in the preference data get baked in directly, just as they would into a reward model.

---

## Part 2 — The interview dialogue (interviewer ⇄ interviewee)

> **🧑‍💼 Interviewer:** One paragraph — what does DPO actually buy me over RLHF?
>
> **🧑‍💻 Interviewee:** It removes two of the three RLHF stages. Classic RLHF trains a separate reward model on preference pairs, then runs PPO to optimize the policy against it with a KL penalty. DPO shows that the optimal RLHF policy has a closed form, which lets you express the reward *through the policy's own log-ratio against a frozen reference*. Substituting that into the Bradley–Terry preference likelihood gives a single supervised loss — basically logistic regression on preferred-vs-dispreferred pairs. No reward model, no RL loop, no online sampling. Same objective, much simpler and more stable.

> **🧑‍💼 Interviewer:** Write the loss. What's actually being compared?
>
> **🧑‍💻 Interviewee:** `L = -log σ(β·[(logπθ(y_w|x) − logπref(y_w|x)) − (logπθ(y_l|x) − logπref(y_l|x))])`. For each response I take the log-ratio of policy to reference — that's the implicit reward up to a constant. I take the chosen response's log-ratio minus the rejected one's, scale by β, and push it through a log-sigmoid. So I'm maximizing the margin by which the policy prefers `y_w` over `y_l` *more than the reference does*.

> **🧑‍💼 Interviewer:** Where did the partition function / reward model go?
>
> **🧑‍💻 Interviewee:** The closed-form optimal policy has a per-prompt normalizer `Z(x)`. In the preference model you only ever take a *difference* of rewards for two responses to the same prompt, so `Z(x)` is identical in both terms and cancels. That's the trick — it's why you never have to compute or train the reward; the policy's relative log-probs carry it.

> **🧑‍💼 Interviewer:** Why is the reference policy frozen, and what does β do?
>
> **🧑‍💻 Interviewee:** The reference is the KL anchor — the original RLHF objective penalizes drift from it, and that penalty becomes the `−logπref` terms. If I let the reference move, I lose the regularizer and the policy can collapse. `β` is the KL strength: large β keeps the policy close to the reference; small β lets it drift further and chase the preferences harder, at the risk of degenerating.

> **🧑‍💼 Interviewer:** Implement it and show the policy's implicit reward recovers a known preference ranking — with no reward model trained.

---

## Part 3 — Implementation

The whole method is a few lines: take per-sequence log-probs under the policy and the frozen reference for the chosen and rejected responses, form the difference of log-ratios, and pass it through `-logsigmoid`.

```python
import torch
import torch.nn.functional as F


def dpo_loss(logp_pol_chosen, logp_pol_rejected,
             logp_ref_chosen, logp_ref_rejected, beta=0.1):
    """DPO loss for a batch of (chosen, rejected) preference pairs.

    Each argument is a per-sequence log-prob log pi(y|x): the chosen/rejected
    response under the policy and under the FROZEN reference. No reward model.
    """
    pol_logratio = logp_pol_chosen - logp_pol_rejected   # policy's relative pref
    ref_logratio = logp_ref_chosen - logp_ref_rejected   # reference's relative pref
    logits = beta * (pol_logratio - ref_logratio)        # margin over the reference
    return -F.logsigmoid(logits).mean()


def implicit_reward(logp_pol, logp_ref, beta=0.1):
    """The reward DPO implicitly optimizes: r(y) = beta * log(pi_theta / pi_ref)."""
    return beta * (logp_pol - logp_ref)
```

### Why each line matters (talk through it)
- `logp_pol_chosen - logp_pol_rejected` — the policy's log-odds of preferring chosen over rejected. This is what we *push up*.
- `logp_ref_chosen - logp_ref_rejected` — the same quantity under the frozen reference. We optimize the **margin over the reference**, not the raw policy preference — that's the KL anchor showing up.
- `beta * (...)` — scales the margin; this is the inverse-KL-strength `β`. At init (policy == reference) the bracket is 0, so `logits == 0` and `loss == -log σ(0) == log 2`.
- `-F.logsigmoid(logits)` — Bradley–Terry NLL. Numerically stable log-sigmoid (don't write `-log(sigmoid(...))`).
- `implicit_reward` — never used in the loss directly; it's how we *read off* the learned reward afterward to check the ordering. No separate reward model is ever trained.

### Demonstrating the benefit (preference-alignment toy task)
A real DPO run needs an LLM. To isolate the *method*, we use a **tiny discrete policy**: a categorical over a small set of "responses," with learnable logits, conditioned on a couple of prompts. The reference is a **frozen copy of the initial (uniform) policy**. We feed it synthetic preference pairs from a known ranking and train only with `dpo_loss`.

```python
class ToyPolicy(torch.nn.Module):
    """pi(y|x): a categorical over n_responses, per prompt. Logits ARE the policy."""
    def __init__(self, n_prompts, n_responses):
        super().__init__()
        self.logits = torch.nn.Parameter(torch.zeros(n_prompts, n_responses))  # uniform

    def logprobs(self, prompt_idx):
        return F.log_softmax(self.logits[prompt_idx], dim=-1)


def run_toy_task(seed=0, beta=0.1, steps=400):
    torch.manual_seed(seed)
    n_prompts, n_responses = 2, 6
    policy = ToyPolicy(n_prompts, n_responses)

    # Frozen reference = a copy of the INITIAL policy.
    ref = ToyPolicy(n_prompts, n_responses)
    ref.load_state_dict(policy.state_dict())
    for p in ref.parameters():
        p.requires_grad_(False)

    # Known rankings (best -> worst); build all chosen>rejected pairs from them.
    rankings = {0: [0, 1, 2, 3, 4, 5], 1: [5, 4, 3, 2, 1, 0]}
    pairs = [(pr, o[a], o[b]) for pr, o in rankings.items()
             for a in range(len(o)) for b in range(a + 1, len(o))]
    prompt_t = torch.tensor([p for p, _, _ in pairs])
    chosen_t = torch.tensor([c for _, c, _ in pairs])
    reject_t = torch.tensor([r for _, _, r in pairs])

    before = policy.logprobs(torch.arange(n_prompts)).exp().detach()

    opt = torch.optim.Adam(policy.parameters(), lr=0.05)
    for _ in range(steps):
        lp_pol = policy.logprobs(prompt_t)
        with torch.no_grad():
            lp_ref = ref.logprobs(prompt_t)
        loss = dpo_loss(lp_pol.gather(1, chosen_t[:, None]).squeeze(1),
                        lp_pol.gather(1, reject_t[:, None]).squeeze(1),
                        lp_ref.gather(1, chosen_t[:, None]).squeeze(1),
                        lp_ref.gather(1, reject_t[:, None]).squeeze(1), beta=beta)
        opt.zero_grad(); loss.backward(); opt.step()

    with torch.no_grad():
        after = policy.logprobs(torch.arange(n_prompts)).exp()
        rewards = implicit_reward(policy.logprobs(torch.arange(n_prompts)),
                                  ref.logprobs(torch.arange(n_prompts)), beta=beta)
    return dict(before=before, after=after, rewards=rewards,
                rankings=rankings, ref=ref, policy=policy)


out = run_toy_task(seed=0)
for pr, order in out["rankings"].items():
    ranked = sorted(range(6), key=lambda y: out["rewards"][pr, y].item(), reverse=True)
    print(f"prompt {pr}: top response prob {out['before'][pr, order[0]]:.3f} -> "
          f"{out['after'][pr, order[0]]:.3f}   reward ranking {ranked}  "
          f"(target {order}, match={ranked == order})")
```

You should see the most-preferred response's probability climb well above its uniform `1/6 ≈ 0.167` start, and the **implicit-reward ranking exactly match the target preference order** — all without ever training a reward model or running an RL loop. (Exact probabilities are seed/step dependent; the *ordering* is the point.)

---

## Part 4 — Sanity checks (don't skip)

### Check 1 — At initialization the DPO loss is `log 2`
When `πθ == πref` the bracket is 0, so `loss = -log σ(0) = log 2 ≈ 0.693`.
```python
import math
z = torch.zeros(8)
loss0 = dpo_loss(z, z, z, z, beta=0.1)
print("loss at init:", loss0.item(), "(expected log2 =", math.log(2), ")")
assert abs(loss0.item() - math.log(2)) < 1e-6
```

### Check 2 — The reference policy is frozen
```python
ref = out["ref"]
assert all(not p.requires_grad for p in ref.parameters())          # no grad
assert torch.equal(ref.logits, torch.zeros_like(ref.logits))        # never moved
print("OK: reference has no grad and is unchanged after training")
```

### Check 3 — Gradient flows to the policy
```python
pol = ToyPolicy(2, 6)
with torch.no_grad(): pol.logits[0, 0] += 0.3      # nudge off uniform
lp, lpr = pol.logprobs(torch.tensor([0])), torch.zeros(1, 6).log_softmax(-1)
dpo_loss(lp[:, 0], lp[:, 1], lpr[:, 0], lpr[:, 1], beta=0.1).backward()
assert pol.logits.grad is not None and pol.logits.grad.abs().sum() > 0
print("OK: |grad| =", pol.logits.grad.abs().sum().item())
```

### Check 4 — Bigger margin lowers the loss; swapping chosen/rejected raises it
```python
def L(pc, pr): return dpo_loss(torch.tensor([pc]), torch.tensor([pr]),
                               torch.tensor([0.0]), torch.tensor([0.0]), beta=1.0).item()
base, big, swap = L(0.0, 0.0), L(3.0, -3.0), L(-3.0, 3.0)
print(f"base={base:.4f}  big_margin={big:.4f}  swapped={swap:.4f}")
assert big < base < swap and big < 0.01
print("OK: favoring chosen -> loss ~0; favoring rejected -> loss grows")
```

### Check 5 — After training, the implicit-reward ranking matches the preferences
```python
for pr, order in out["rankings"].items():
    ranked = sorted(range(6), key=lambda y: out["rewards"][pr, y].item(), reverse=True)
    assert ranked == order
    assert out["after"][pr, order[0]] > out["before"][pr, order[0]]   # mass moved up
print("OK: implicit reward orders responses exactly like the preferences")
```

### Check 6 — The implicit reward scales linearly with `β`
`r(y) = β·log(πθ/πref)`, so doubling `β` doubles the reward.
```python
lp, lr = torch.tensor([math.log(0.5)]), torch.tensor([math.log(0.25)])
r1, r2 = implicit_reward(lp, lr, beta=0.1), implicit_reward(lp, lr, beta=0.2)
assert torch.allclose(r2, 2 * r1, atol=1e-6)
print("OK: beta 0.1 ->", r1.item(), " beta 0.2 ->", r2.item())
```

---

## Part 5 — Likely follow-up questions

- *"How does DPO relate to PPO-based RLHF?"* — Same underlying KL-regularized reward objective; DPO uses the closed-form optimal policy to turn it into a supervised loss, skipping both the reward model and the on-policy RL optimization.
- *"Why does the partition function `Z(x)` cancel?"* — Preferences compare two responses to the *same* prompt, so the per-prompt normalizer is identical in both reward terms and drops out of the difference.
- *"What if you don't have a reference / use a bad one?"* — The reference is the KL anchor; a weak SFT reference means the policy can drift into degenerate text. People sometimes set the reference to the SFT model and keep it fixed.
- *"What breaks if `β` is too small or too large?"* — Too small: policy over-optimizes the preferences and drifts far from the reference (degeneration, reward hacking). Too large: it barely moves from the reference and under-fits the preferences.
- *"Online vs offline?"* — Vanilla DPO is offline on a fixed preference dataset. Variants (e.g., iterative/online DPO) sample fresh responses and re-label to combat distribution shift.
- *"Other variants?"* — IPO (fixes an over-optimization issue), KTO (works from unpaired good/bad labels), and length-regularized DPO all build on this loss.

---

## TL;DR cheat sheet
| Thing | Answer |
|---|---|
| Core idea | RLHF without a reward model or RL — one supervised loss on preference pairs |
| Loss | `-log σ(β·[(logπθ(y_w)−logπref(y_w)) − (logπθ(y_l)−logπref(y_l))])` |
| Implicit reward | `r(y) = β·log(πθ(y|x)/πref(y|x))` — rank by it to recover preferences |
| Why no reward model | The policy's log-ratio *is* the reward; `Z(x)` cancels in the difference |
| Reference policy | Frozen (the KL anchor, usually the SFT model) |
| `β` | KL strength: large = stay near ref, small = drift / over-optimize |
| Benefit | Simpler, cheaper, more stable than PPO RLHF; offline |
| Limitation | Offline (no exploration); sensitive to ref/β; bakes in data biases |
