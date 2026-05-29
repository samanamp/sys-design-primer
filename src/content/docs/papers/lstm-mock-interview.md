---
title: LSTM — Paper-to-Code Mock Interview
description: A full combined mock (read paper, explain the benefit, implement in Colab) using the LSTM as the worked example — the constant error carousel that beats vanishing gradients.
sidebar:
  order: 18
  label: LSTM
---

> **Paper:** *Long Short-Term Memory* — Hochreiter & Schmidhuber, 1997. *Neural Computation* 9(8):1735–1780. [Canonical PDF](https://www.bioinf.jku.at/publications/older/2604.pdf) (no arXiv). A famously clear explainer is Chris Olah's [*Understanding LSTM Networks*](https://colah.github.io/posts/2015-08-Understanding-LSTMs/).
>
> **Format:** Read the paper (~15 min) → explain the *real* benefit → implement the core idea in Colab → sanity-check it.
>
> **Companion notebook:** [`lstm_mock.ipynb`](/notebooks/lstm_mock.ipynb) (download) — a long-range memory task + an `LSTMCell` stub to fill in, plus verification cells. Or open it straight in [Google Colab](https://colab.research.google.com/) via *File → Upload notebook*. A reference solution is included at the bottom of this page.
>
> **Difficulty:** 🟡🔴 Medium–hard. More moving parts than dropout (four gates, two recurrent states, BPTT). Do the warm-ups first.

---

## How to run this as a timed drill (~60 min)

Treat this like the real thing. Set a timer and don't look at the answers below until each block is done.

| Time | Block | What you produce |
|------|-------|------------------|
| 0:00–0:15 | **Read** (use the [three-pass method](/papers/lora-mock-interview/#part-0--how-to-read-a-paper-in-15-minutes-the-three-pass-method)) | Why vanilla RNNs forget + the cell-state / gates idea |
| 0:15–0:20 | **Explain the benefit** out loud (cover Part 2) | The constant error carousel + what each gate does |
| 0:20–0:50 | **Implement** from the stub (Part 3) | A working `LSTMCell` + a long-range task the LSTM solves and a vanilla RNN doesn't |
| last 10 min | **Sanity-check** (Part 4) | All 6 checks passing, narrated out loud |

### Self-grading rubric — "what good looks like"
- ✅ Explained the failure being fixed: **BPTT multiplies many Jacobians**, so gradients vanish/explode and distant inputs are forgotten.
- ✅ Named the mechanism precisely: a **cell state** with **gated, additive updates** — the **constant error carousel** — not just "it has gates."
- ✅ Knew what each gate does (forget, input, output) and the **additive** `c' = f⊙c + i⊙g` update (vs the RNN's *multiplicative* `tanh(W·)`).
- ✅ Demonstrated the benefit with a **long-range task** where the LSTM wins and a vanilla RNN doesn't — and could point at the **gradient norm** to early inputs.
- ⚠️ Red flags: calling the cell state "the hidden state" (they're different), forgetting the update is additive, claiming LSTMs *cannot* explode (they still can; that's why grad clipping exists), reciting gate equations with no idea why the additive path matters.

---

## Part 1 — Structured read of THIS paper

### The 30-second summary (the "benefit")
A vanilla RNN unrolls in time and is trained with **backprop through time (BPTT)**. The gradient that reaches an input `T` steps in the past is a **product of `T` Jacobians**; if their magnitudes are consistently below 1 the gradient **vanishes**, above 1 it **explodes**. Either way the net can't learn dependencies across long gaps — it forgets distant inputs. The LSTM adds a **cell state** `c` updated **additively** through **gates**:

- When the **forget gate ≈ 1** and the **input gate ≈ 0**, the cell state is carried forward **almost unchanged** — a **constant error carousel (CEC)**.
- Because the update is additive (not a fresh `tanh(W·)` every step), gradients flow back across many timesteps **without vanishing**, so the network can learn **long-range memory**.
- The gates let the net *learn when* to write, keep, and read memory — selective, content-dependent storage.

### The core idea (Method — you implement this)
At each step, from the previous hidden state `h` and the input `x` (concatenated), compute three gates and a candidate update:

$$
\begin{aligned}
i &= \sigma(W_i[h, x] + b_i) &\text{(input gate)} \\
f &= \sigma(W_f[h, x] + b_f) &\text{(forget gate)} \\
o &= \sigma(W_o[h, x] + b_o) &\text{(output gate)} \\
g &= \tanh(W_g[h, x] + b_g) &\text{(candidate)}
\end{aligned}
$$

Then update the **cell state additively** and produce the new hidden state:

$$ c' = f \odot c + i \odot g, \qquad h' = o \odot \tanh(c') $$

Here `σ` is the sigmoid (gates live in `(0,1)` — soft on/off switches) and `⊙` is elementwise product. The whole sequence model just **loops this cell** over the timesteps and reads out from the final `h`.

Why this beats the vanilla RNN — the thing an interviewer probes:
- **The carousel is the additive path.** `∂c'/∂c = f` (elementwise). If `f ≈ 1`, the per-step backward factor is `≈ 1`, so error flows back through many steps undamped. Contrast the vanilla RNN: `∂h'/∂h = diag(1 - tanh²) · W`, repeatedly multiplied → vanishes (or explodes).
- **Gates are learned, content-dependent switches.** The net learns *when* a value is worth keeping (`f→1, i→0`), *when* to overwrite (`f→0, i→1`), and *when* to expose it to the rest of the net (`o`).
- **Forget-gate bias matters in practice.** Initializing `b_f` positive (e.g. `+1`) starts the cell "remembering," which makes long-range learning much more reliable (a well-known follow-up to the original paper, which used input/output gates only; the forget gate was added by Gers et al., 2000).

### Where the evidence lives
- **The long-time-lag benchmarks** (e.g. the embedded Reber grammar and the "2-sequence" / latching problems) — tasks engineered so the answer depends on an input many steps earlier. LSTM solves lags that vanilla RNNs / earlier methods fail on.
- **The error-flow analysis** — the paper's argument that, without the gated constant-error path, backpropagated error decays (or blows up) exponentially in the time lag. That analysis *is* the motivation for the architecture.
- **Treat the exact 1997 numbers as historical.** The architecture is what's load-bearing; benchmarks have moved on, and modern code uses the forget-gate variant.

### The honest limitations (have an opinion)
- **LSTMs can still explode** — the CEC controls vanishing, not exploding. Real training uses **gradient clipping**.
- **Sequential, not parallel.** The recurrence is inherently step-by-step; you can't parallelize over time the way Transformers do, so long sequences are slow.
- **Limited effective context.** Memory is finite-dimensional and gated; very long-range or precise-lookup tasks favor **attention/Transformers**, which largely replaced LSTMs for language.
- **Many parameters per step** (four `[h,x]→H` maps). The original is also a *cell*; production code uses `nn.LSTM` with a different (fused) weight layout.

---

## Part 2 — The interview dialogue (interviewer ⇄ interviewee)

> **🧑‍💼 Interviewer:** One paragraph — what problem does the LSTM actually fix?
>
> **🧑‍💻 Interviewee:** Vanilla RNNs can't learn long-range dependencies. With backprop through time, the gradient reaching an input `T` steps back is a product of `T` Jacobians; their magnitudes are either consistently under 1 (gradient vanishes) or over 1 (it explodes), so distant inputs effectively get no learning signal and the net forgets them. The LSTM adds a separate cell state updated *additively* through gates: `c' = f⊙c + i⊙g`. When the forget gate is near 1 and the input gate near 0, the cell is carried forward almost unchanged — a constant error carousel — so gradients flow across many steps without vanishing, and the net can hold information for a long time.

> **🧑‍💼 Interviewer:** Walk me through the gates and which one is the "memory."
>
> **🧑‍💻 Interviewee:** Three sigmoid gates and one tanh candidate, all from `[h, x]`. The **forget gate** `f` decides how much of the old cell state to keep; the **input gate** `i` decides how much of the new candidate `g = tanh(...)` to write; the **output gate** `o` decides how much of the cell to expose as the hidden state. The actual memory is the **cell state** `c`, updated additively as `c' = f⊙c + i⊙g`. The hidden state is a gated, squashed *view* of the cell: `h' = o⊙tanh(c')`. People conflate `c` and `h`, but `c` is the long-term store and `h` is what the rest of the network reads.

> **🧑‍💼 Interviewer:** Why does additive `c' = f⊙c + i⊙g` help gradients when `h' = tanh(W[h,x])` doesn't?
>
> **🧑‍💻 Interviewee:** Differentiate the carousel: `∂c'/∂c = f` elementwise. If `f ≈ 1`, each step contributes a backward factor near 1, so chaining `T` of them stays near 1 — no vanishing. The vanilla RNN's recurrent Jacobian is `diag(1 − tanh²)·W`; the `1 − tanh²` term is ≤ 1 and usually well under it, and you multiply `T` of those together, which decays to zero fast. The additive path with a near-open forget gate is exactly the "+1 highway" idea — the same trick residual connections use for depth, here applied across time.

> **🧑‍💼 Interviewer:** So gradients can never vanish or explode with an LSTM?
>
> **🧑‍💻 Interviewee:** No — that's a common overclaim. The CEC *protects against vanishing* when the forget gate stays open, but if the net learns `f < 1` the memory decays, and gradients can still **explode** (the cell update has no upper bound and `tanh` saturates the readout, not the storage). In practice you still use **gradient clipping**, and you often init the forget-gate bias positive so the cell defaults to remembering and long-range learning is reliable.

> **🧑‍💼 Interviewer:** Implement the cell and show it beats a vanilla RNN on a long-range task.

---

## Part 3 — Implementation

The core is one cell: four gate computations from `[h, x]`, an additive cell update, a gated readout. A thin wrapper loops it over the sequence. We also implement a vanilla `RNNCell` for the comparison.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class LSTMCell(nn.Module):
    """One LSTM step. Gates from concatenated [h, x]; additive cell update (CEC)."""

    def __init__(self, in_dim, hid_dim, forget_bias=1.0):
        super().__init__()
        self.hid_dim = hid_dim
        # one big Linear over [h, x] producing the 4 gate pre-activations: i, f, g, o
        self.W = nn.Linear(in_dim + hid_dim, 4 * hid_dim)
        # start the forget gate "open" so the cell defaults to remembering
        with torch.no_grad():
            self.W.bias[hid_dim:2 * hid_dim].fill_(forget_bias)

    def forward(self, x, state):
        h, c = state
        z = self.W(torch.cat([h, x], dim=-1))          # (B, 4H)
        i, f, g, o = z.chunk(4, dim=-1)                 # split into 4 gates
        i = torch.sigmoid(i)                            # input gate   (0,1)
        f = torch.sigmoid(f)                            # forget gate  (0,1)
        g = torch.tanh(g)                              # candidate    (-1,1)
        o = torch.sigmoid(o)                            # output gate  (0,1)
        c_new = f * c + i * g                           # additive cell update
        h_new = o * torch.tanh(c_new)
        return h_new, c_new


class RNNCell(nn.Module):
    """Vanilla tanh RNN step: h' = tanh(W[h, x] + b). Same (h, c) interface."""

    def __init__(self, in_dim, hid_dim):
        super().__init__()
        self.hid_dim = hid_dim
        self.W = nn.Linear(in_dim + hid_dim, hid_dim)

    def forward(self, x, state):
        h, _ = state
        h_new = torch.tanh(self.W(torch.cat([h, x], dim=-1)))
        return h_new, h_new  # c is unused


class SeqModel(nn.Module):
    """Loops a cell over a sequence and reads out from the final hidden state."""

    def __init__(self, cell_cls, in_dim, hid_dim, out_dim):
        super().__init__()
        self.cell = cell_cls(in_dim, hid_dim)
        self.hid_dim = hid_dim
        self.readout = nn.Linear(hid_dim, out_dim)

    def forward(self, x):                                # x: (B, T, in_dim)
        B, T, _ = x.shape
        h = x.new_zeros(B, self.hid_dim)
        c = x.new_zeros(B, self.hid_dim)
        for t in range(T):
            h, c = self.cell(x[:, t, :], (h, c))
        return self.readout(h)
```

### Why each line matters (talk through it)
- **`nn.Linear(in_dim + hid_dim, 4 * hid_dim)`** — one fused matrix produces all four gate pre-activations from `[h, x]` in a single matmul; `chunk(4)` splits them. (`nn.LSTM` uses a different fused layout — order `i, f, g, o` here is a convention.)
- **`forget_bias=1.0` on `b_f`** — starts the cell *remembering*; without it the forget gate starts at `σ(0)=0.5` and memory decays like `0.5^T`, which kills long-range learning. This is the single most important practical detail.
- **`f * c + i * g`** — the **additive** update is the whole point. `∂c'/∂c = f`, so with `f ≈ 1` gradients survive across many steps. The RNN's `tanh(W[h,x])` has no such path.
- **`o * tanh(c_new)`** — the hidden state is a gated, squashed *view* of the cell; the cell `c` itself is the long-term store and is **not** squashed before being carried forward.
- **The loop in `SeqModel`** — initial `h, c` are zeros; we read out from the **final** `h`. This is BPTT: autograd unrolls the loop and multiplies the per-step Jacobians for you.

### Demonstrating the benefit (long-range memory: delayed XOR)
Two cue bits arrive at `t=0` and `t=1`; then `T−2` steps of pure noise; the target is the **XOR** of the two early bits. To answer, the net must **hold both bits across the whole noisy gap** *and* combine them non-linearly — exactly the long-range dependency vanilla RNNs can't learn. We train a vanilla-RNN model and an LSTM model on the same data and compare.

```python
def make_xor_data(n, T, seed):
    g = torch.Generator().manual_seed(seed)
    x = torch.zeros(n, T, 2)
    a = (torch.rand(n, generator=g) < 0.5).float()
    b = (torch.rand(n, generator=g) < 0.5).float()
    x[:, 0, 0] = a * 2 - 1                                # first cue bit  (+/-1)
    x[:, 1, 0] = b * 2 - 1                                # second cue bit (+/-1)
    x[:, 2:, 1] = torch.randn(n, T - 2, generator=g)      # distractor noise
    y = (a.long() ^ b.long())                            # XOR target
    return x, y


def train_model(cell_cls, T, hid=32, steps=600, lr=3e-3, seed=0):
    torch.manual_seed(seed)
    Xtr, ytr = make_xor_data(256, T, seed=1)
    Xte, yte = make_xor_data(512, T, seed=2)
    model = SeqModel(cell_cls, 2, hid, 2)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(steps):
        model.train()
        loss = F.cross_entropy(model(Xtr), ytr)
        opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    with torch.no_grad():
        logits = model(Xte)
        acc = (logits.argmax(-1) == yte).float().mean().item()
    return model, acc


T = 60
rnn_model, rnn_acc = train_model(RNNCell, T, seed=0)
lstm_model, lstm_acc = train_model(LSTMCell, T, seed=0)
print(f"vanilla RNN : test acc {rnn_acc:.3f}")   # ~0.49 — chance
print(f"LSTM        : test acc {lstm_acc:.3f}")   # ~1.00 — solved
```

With `T=60` the LSTM reaches **~1.00** test accuracy while the vanilla RNN sits at **~0.49** — chance, i.e. it never learned the dependency. (Exact numbers are seed-dependent; the *direction* — LSTM solves it, RNN doesn't — is the point, and it's asserted in Part 4. If you crank `T` up, even the LSTM eventually needs more optimization budget.)

---

## Part 4 — Sanity checks (don't skip)

### Check 1 — Output shapes are correct
```python
cell = LSTMCell(2, 8)
h0 = torch.zeros(4, 8); c0 = torch.zeros(4, 8)
h1, c1 = cell(torch.randn(4, 2), (h0, c0))
assert h1.shape == (4, 8) and c1.shape == (4, 8)
print("OK: hidden/cell shapes", tuple(h1.shape))
```

### Check 2 — Gates are in (0,1) and the candidate in (−1,1)
```python
cell = LSTMCell(2, 16)
z = cell.W(torch.cat([torch.randn(100, 16), torch.randn(100, 2)], -1))
i, f, g, o = z.chunk(4, -1)
i, f, o = torch.sigmoid(i), torch.sigmoid(f), torch.sigmoid(o)
g = torch.tanh(g)
assert i.min() > 0 and i.max() < 1 and f.min() > 0 and f.max() < 1 and o.min() > 0 and o.max() < 1
assert g.min() > -1 and g.max() < 1
print("OK: gates in (0,1), candidate in (-1,1)")
```

### Check 3 — Constant error carousel: forget=1, input=0 ⇒ cell carried UNCHANGED
```python
cell = LSTMCell(2, 8); H = 8
with torch.no_grad():
    cell.W.weight.zero_()
    cell.W.bias.zero_()
    cell.W.bias[0:H] = -50.0     # input gate  i -> sigmoid(-50) ~ 0
    cell.W.bias[H:2 * H] = 50.0  # forget gate f -> sigmoid(50)  ~ 1
c_prev = torch.randn(3, 8)
_, c_next = cell(torch.randn(3, 2), (torch.randn(3, 8), c_prev))
assert torch.allclose(c_next, c_prev, atol=1e-5)
print("OK: forget=1, input=0 carries cell state unchanged (CEC)")
```

### Check 4 — LSTM solves the long-range task; vanilla RNN fails
```python
# uses the trained models from Part 3
assert lstm_acc > 0.9, lstm_acc
assert rnn_acc < 0.75, rnn_acc
assert lstm_acc - rnn_acc > 0.2
print(f"OK: LSTM {lstm_acc:.3f} >> RNN {rnn_acc:.3f} on long-range task")
```

### Check 5 — Gradient flows to the EARLIEST input (non-vanishing vs the RNN)
```python
Tg = 60
def early_grad(cell_cls, seed=0):
    torch.manual_seed(seed)
    m = SeqModel(cell_cls, 2, 16, 2)
    x = torch.randn(8, Tg, 2, requires_grad=True)
    m(x).sum().backward()
    return x.grad[:, 0, :].abs().mean().item()      # grad wrt the t=0 input

g_lstm, g_rnn = early_grad(LSTMCell), early_grad(RNNCell)
print(f"early-input grad  LSTM {g_lstm:.3e}   RNN {g_rnn:.3e}")
assert g_lstm > g_rnn * 10
print("OK: LSTM gradient to earliest input >> vanilla RNN (non-vanishing)")
```
With `Tg=60` you'll see the LSTM's gradient to the first step is roughly `~1e-6` while the vanilla RNN's has **vanished to ~1e-17** — about ten orders of magnitude smaller. That gap *is* the vanishing-gradient problem, and the carousel fixing it.

### Check 6 — A full-sequence forward is finite and right-shaped
```python
out = lstm_model(torch.randn(5, T, 2))
assert out.shape == (5, 2) and torch.isfinite(out).all()
print("OK: full-sequence forward finite, shape", tuple(out.shape))
```

---

## Part 5 — Likely follow-up questions

- *"Original LSTM vs the modern one?"* — The 1997 paper had **input and output gates** and the CEC; the **forget gate** was added by **Gers et al. (2000)** so the cell can learn to *reset* itself. Today "LSTM" almost always means the forget-gate variant, and initializing `b_f > 0` is standard.
- *"LSTM vs GRU?"* — The **GRU** (Cho et al., 2014) merges the cell and hidden state and uses two gates (update, reset) instead of three. Fewer parameters, often comparable accuracy; the LSTM's separate cell can be better on tasks needing precise long memory.
- *"Why did Transformers replace LSTMs for language?"* — Recurrence is **sequential** (no parallelism over time) and effective context is limited; **self-attention** gives `O(1)` path length between any two positions and parallelizes across the sequence, so it scales better and learns longer-range structure.
- *"How do you stop LSTM gradients exploding?"* — **Gradient clipping** (clip the global norm), sensible init, and not letting forget gates push the cell unbounded. The CEC only addresses *vanishing*.
- *"`nn.LSTM` vs your cell — same numbers?"* — Functionally equivalent, but PyTorch fuses the weights with a **different layout** (separate input/hidden matrices, gate order `i,f,g,o` with two bias vectors), so an exact numerical match to `nn.LSTMCell` isn't expected without remapping weights — a functional check (it learns the task) is enough. Use `nn.LSTM` for real work; it's far faster.

---

## TL;DR cheat sheet
| Thing | Answer |
|---|---|
| Problem fixed | Vanilla RNNs forget: BPTT multiplies many Jacobians → gradients vanish/explode |
| Core idea | A **cell state** with **gated, additive** updates — the constant error carousel |
| Gates | forget `f` (keep old), input `i` (write new), output `o` (expose); candidate `g=tanh(·)` |
| Update | `c' = f⊙c + i⊙g`,  `h' = o⊙tanh(c')` |
| Why grads survive | `∂c'/∂c = f`; with `f≈1` the backward factor ≈ 1 across many steps |
| cell vs hidden | `c` = long-term store (not squashed when carried); `h = o⊙tanh(c)` = the read-out view |
| Practical must-do | Init `b_f > 0` (default to remembering) + **gradient clipping** |
| Limitations | Still explodes; sequential/slow; limited context → Transformers took over |
