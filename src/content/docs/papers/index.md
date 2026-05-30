---
title: "Paper Mocks: Overview & Study Plan"
description: A difficulty-ordered library of paper-to-code mock interviews — read a paper, explain its real benefit, implement it in Colab, and sanity-check it.
sidebar:
  label: Overview & Study Plan
  order: 0
---

A library of **paper-to-code mock interviews** for the increasingly common ML/research interview format: *you're handed a paper, asked to explain its actual benefit, then implement the core idea in Google Colab.*

Each page is a self-contained, timed drill with the same shape:

- **How to read the paper in 15 minutes** — the [three-pass method](/papers/lora-mock-interview/#part-0--how-to-read-a-paper-in-15-minutes-the-three-pass-method) (map → method+evidence → pressure-test).
- **Explain the real benefit** — a structured read + an interviewer⇄interviewee dialogue with model answers.
- **Implement it** — a clean PyTorch reference, a "why each line matters" walkthrough, and a **companion Colab notebook** (a fill-in-the-blank stub + reference solution).
- **Sanity-check it** — 6 checks that *prove* the code is correct, plus a toy task that demonstrates the claim. Every toy task and check in this library was **executed before being written down**.

> **How to use a page:** set a timer, read on the real PDF, talk your answer out loud, fill in the notebook stub *without* peeking, then run the sanity checks. The honest goal isn't a perfect implementation — it's reading critically, narrating your reasoning, and verifying as you go.

---

## The 20 papers, by difficulty

| 🟢 = warm-up · 🟢🟡 = easy–medium · 🟡 = medium · 🟡🔴 = medium–hard · 🔴 = hard |

| # | Paper | Difficulty | The one-line benefit / what you implement |
|---|-------|:---:|---|
| 1 | [Dropout](/papers/dropout-mock-interview/) | 🟢 | Randomly zero units to break co-adaptation; train/eval + inverted-dropout scaling |
| 2 | [RMSNorm](/papers/rmsnorm-mock-interview/) | 🟢 | LayerNorm minus the mean-subtraction; scale-invariance |
| 3 | [ResNet](/papers/resnet-mock-interview/) | 🟢🟡 | Identity skip connections make very deep nets trainable |
| 4 | [SwiGLU](/papers/swiglu-mock-interview/) | 🟢🟡 | Gated FFN; the 2/3 param-matching + an honest "no toy win" |
| 5 | [AdamW](/papers/adamw-mock-interview/) | 🟡 | Decouple weight decay from the adaptive step (L2 ≠ weight decay) |
| 6 | [BatchNorm](/papers/batchnorm-mock-interview/) | 🟡 | Normalize per batch; train/eval running stats |
| 7 | [LoRA](/papers/lora-mock-interview/) | 🟡 | Freeze the weights, train a tiny low-rank update |
| 8 | [Knowledge Distillation](/papers/knowledge-distillation-mock-interview/) | 🟡 | A student learns the teacher's soft "dark knowledge" |
| 9 | [word2vec](/papers/word2vec-mock-interview/) | 🟡 | Word embeddings via skip-gram + negative sampling (a cheap softmax) |
| 10 | [Focal Loss](/papers/focal-loss-mock-interview/) | 🟡 | Down-weight easy examples to fight class imbalance |
| 11 | [RoPE](/papers/rope-mock-interview/) | 🟡 | Rotary positions — relative position for free |
| 12 | [GQA / MQA](/papers/gqa-mock-interview/) | 🟡 | Share KV heads to shrink the inference KV cache |
| 13 | [ViT](/papers/vit-mock-interview/) | 🟡 | An image as a sequence of patches through a plain Transformer |
| 14 | [SimCLR](/papers/simclr-mock-interview/) | 🟡🔴 | Contrastive representations with no labels (NT-Xent) |
| 15 | [VAE](/papers/vae-mock-interview/) | 🟡🔴 | Reparameterization trick + ELBO → a sample-able latent space |
| 16 | [DPO](/papers/dpo-mock-interview/) | 🟡🔴 | Preference alignment with no reward model and no RL |
| 17 | [MoE](/papers/moe-mock-interview/) | 🟡🔴 | Route tokens to top-k experts: capacity ≫ FLOPs/token |
| 18 | [LSTM](/papers/lstm-mock-interview/) | 🟡🔴 | A gated cell state beats vanishing gradients (long-range memory) |
| 19 | [Attention](/papers/attention-mock-interview/) | 🔴 | Scaled dot-product + multi-head self-attention |
| 20 | [DDPM](/papers/ddpm-mock-interview/) | 🔴 | Denoising diffusion: learn to reverse a noising process |

> **⚠️ One dependency to know:** *Attention* sorts near the end because it's the hardest to implement fully — but **RoPE, GQA, and ViT all build on it**. If you're heading for those three, do Attention first regardless of its difficulty rating.

---

## Study plans

### The recommended ladder (1 paper/day)
A prerequisite-aware path — close to the difficulty order, but with **Attention pulled earlier** so the things that depend on it come after.

**Week 1 — fundamentals (training & building blocks)**
1. Dropout → 2. RMSNorm → 3. ResNet → 4. SwiGLU → 5. AdamW → 6. BatchNorm → 7. LoRA

**Week 2 — sequences & attention**
8. **Attention** (the keystone) → 9. RoPE → 10. GQA/MQA → 11. ViT → 12. word2vec → 13. LSTM

**Week 3 — objectives, alignment & generative**
14. Focal Loss → 15. Knowledge Distillation → 16. SimCLR → 17. VAE → 18. DPO → 19. MoE → 20. DDPM

At one a day that's ~3 weeks. Two rest/review days fit naturally at the week boundaries.

### Short on time?
- **Have ~1 week:** do the **bold keystones** — Attention, LoRA, RMSNorm, AdamW, ResNet, BatchNorm, Dropout. These are the highest-probability "implement this" papers and cover the most reusable mechanics.
- **The single best rep:** do **one full timed mock end-to-end** (read → explain → implement → sanity-check) rather than skimming five. The combined drill is what actually predicts interview performance.
- **Day before:** re-read the *cheat sheet* table at the bottom of each page you've done; re-run one notebook from scratch to confirm muscle memory.

### Themed tracks (pick by the role)
- **Modern LLM:** Attention → RoPE → GQA/MQA → RMSNorm → SwiGLU → MoE → LoRA → DPO
- **Generative modeling:** VAE → DDPM (+ how they relate to GANs / score matching)
- **Computer vision:** ResNet → BatchNorm → ViT → SimCLR
- **Training & optimization:** Dropout → BatchNorm → AdamW → ResNet → Focal Loss → Knowledge Distillation
- **NLP foundations:** word2vec → LSTM → Attention → RoPE

---

## What "good" looks like (the cross-cutting rubric)
Every page has its own rubric, but these habits transfer to all of them:

- ✅ Anchor a benefit claim to a **specific table/figure** ("the ablation shows…"), not the abstract.
- ✅ State the **tradeoff / limitation** unprompted, not only the upside.
- ✅ Implement **without copy-pasting** — and narrate decisions while coding.
- ✅ Write **≥2 sanity checks** before being asked (shapes + "only X trains" are free wins).
- ✅ Be honest when a toy **can't** show the headline benefit (e.g. SwiGLU, AdamW, DDPM at scale) — show the *mechanism* instead of overselling a number.
