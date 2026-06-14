---
title: Diffusion interview map
description: What to memorize cold for diffusion and world-model interviews, with links to the best local notes.
---

# Diffusion Interview Map

This is the memorization map for Waymo-style diffusion and world-model interviews. The goal is not to name every paper. The goal is to have the architecture substrate cold, know where each technique sits in the stack, and be able to explain the tradeoffs without reaching for notes.

Use this page as the checklist. Each row tells you what to memorize, the one sentence you should be able to say out loud, and where to review it in this primer.

---

## What to memorize cold

| Priority | Topic | Memorize this | Where to find it |
|---|---|---|---|
| P0 | DDPM | Forward process adds Gaussian noise; reverse model learns to denoise, often by predicting epsilon with an MSE loss. | [Diffusion Basics](/ml-modeling-fundamentals/diffusion_basics/) |
| P0 | DDIM | Same trained model, deterministic non-Markovian sampler, fewer NFEs than ancestral DDPM. | [Training-Free Diffusion Solvers](/diffusion/7-training-free-solvers/) |
| P0 | Score SDE / probability-flow ODE | Diffusion sampling can be seen as solving an SDE or deterministic ODE from noise to data. | [Training-Free Diffusion Solvers](/diffusion/7-training-free-solvers/) |
| P0 | NFE accounting | Cost is denoiser calls times guidance, horizon, candidate actions, and samples per action. | [Fast Diffusion Sampling](/diffusion/6-diff-optimization/) |
| P0 | Classifier-free guidance | Combine unconditional and conditional predictions; guidance improves conditioning but doubles NFE and can reduce diversity. | [Fast Diffusion Sampling](/diffusion/6-diff-optimization/) |
| P0 | Training-free solvers | DDIM, DPM-Solver++, and UniPC change only the sampler/integrator; they preserve the teacher better than distillation but bottom out around low double-digit NFE. | [Training-Free Diffusion Solvers](/diffusion/7-training-free-solvers/) |
| P0 | Latent diffusion | Diffuse in a compressed latent instead of pixels; this is the core efficiency move for video/world models. | [Diffusion World Model](/diffusion/3-diff-world-model/) |
| P0 | Tokenizers: VAE vs VQ | Continuous latents pair naturally with diffusion/flow matching; discrete tokens pair naturally with autoregressive transformers. | [Diffusion World Model](/diffusion/3-diff-world-model/) |
| P0 | Flow matching / rectified flow | Learn a velocity field along straighter paths; straighter paths need fewer solver steps. | [Fast Diffusion Sampling](/diffusion/6-diff-optimization/) |
| P1 | U-Net | Original diffusion backbone: convolutional encoder-decoder with skips and attention. Know it as the baseline architecture. | [U-Net and DiT Backbones](/diffusion/6-unet-dit-backbones/) |
| P1 | DiT | Patchify latent tokens and use transformer blocks; modern diffusion backbone, especially for image/video/world models. | [U-Net and DiT Backbones](/diffusion/6-unet-dit-backbones/) |
| P1 | Conditioning injection | Action, map, route, agent boxes, text, and event controls enter through cross-attention, AdaLN/FiLM, concatenation, or ControlNet-style branches. | [Diffusion World Model](/diffusion/3-diff-world-model/) |
| P1 | Video latent world model | History is encoded into latents; a denoiser samples future latents; decoder renders future video. | [Training-Free Diffusion Solvers](/diffusion/7-training-free-solvers/) |
| P1 | BEV / occupancy world model | Predict future occupancy or scene tokens instead of pixels; easier to validate and more directly plannable. | [Diffusion World Model](/diffusion/3-diff-world-model/) |
| P1 | Trajectory diffusion | Sample future agent trajectories conditioned on history, map, and ego action; cheap but abstracts away perception. | [Training-Free Diffusion Solvers](/diffusion/7-training-free-solvers/) |
| P1 | Distillation | Retrain a student for 1-8 NFE; faster than solvers, but risks quality and tail-mode loss. | [Fast Diffusion Sampling](/diffusion/6-diff-optimization/) |
| P1 | Diversity / tail coverage | For world models, lost modes are lost futures; a fast sampler that drops rare dangerous futures is a safety failure. | [Fast Diffusion Sampling](/diffusion/6-diff-optimization/) |
| P1 | Validation hierarchy | Looks-real is not enough; validate physical consistency, reactivity, distributional coverage, and downstream predictive validity. | [Diffusion Validation](/diffusion/5-diff-validation/) |
| P2 | GAIA-2 | AV-specific controllable multi-view latent diffusion world model; know it as the driving-video reference architecture. | [Diffusion World Model](/diffusion/3-diff-world-model/) |
| P2 | Vista | Open driving world model for controllable high-fidelity video prediction. | [Diffusion World Model](/diffusion/3-diff-world-model/) |
| P2 | OccWorld | 3D occupancy tokenizer plus GPT-like spatiotemporal transformer for scene and ego-token rollout. | [Diffusion World Model](/diffusion/3-diff-world-model/) |
| P2 | Cosmos | World foundation model platform: tokenizer plus diffusion and autoregressive models for physical AI. | [Diffusion World Model](/diffusion/3-diff-world-model/) |

---

## The architecture you should be able to whiteboard

If you memorize only one diagram, memorize this:

```text
history / scene state
      |
      v
encoder or tokenizer
      |
      v
history latent / context tokens
      |
      |              action, map, route, agent boxes, text/event controls
      |                    |
      v                    v
random noisy future x_T -> denoiser f_theta(x_t, t, c)
                              |
                              v
                       sampler / solver
                              |
                              v
                       clean future x_0
                              |
                              v
                         decoder
                              |
                              v
              future video, occupancy, or trajectories
```

The key interview sentence:

> The tokenizer and denoiser are architecture. The solver is inference-time numerical integration. The conditioning stack is what makes the generator a world model instead of just a video model.

---

## Why U-Net and DiT came next

I made [U-Net and DiT Backbones](/diffusion/6-unet-dit-backbones/) the next study article because it is the missing middle between diffusion math and world-model systems design.

Priority for your study:

```text
P0: diffusion math and sampling
    DDPM, DDIM, ODE view, CFG, NFE accounting

P1: denoiser architecture
    U-Net, latent U-Net, DiT, conditioning injection, video extensions

P1: tokenization and representation
    VAE vs VQ, latent size, continuous vs discrete state

P2: named world-model systems
    GAIA-2, Vista, OccWorld, Cosmos
```

Reason: if an interviewer asks you to design the model, the backbone is the first real architecture whiteboard. You need to explain what the denoiser is, why U-Net was the original default, why DiT became the modern scalable default, where action/map/history conditioning enters, and how this differs from the sampler. Named systems like GAIA-2 or Vista become easier once this substrate is automatic.

---

## The model taxonomy

Use this taxonomy to place any architecture they mention:

| Family | Representation | Strength | Weakness | Examples to know |
|---|---|---|---|---|
| Video latent diffusion | Future camera/video latents | Sensor realism, perception stress tests | Expensive, hard to validate physically | GAIA-2, Vista, Cosmos Diffusion |
| Occupancy / BEV world model | 3D occupancy, BEV grids, scene tokens | Geometric, checkable, planning-relevant | Less appearance detail | OccWorld, Drive-OccWorld-style models |
| Autoregressive token world model | Discrete video/scene/action tokens | Natural long rollout, token-level prediction | Drift, quantization, exposure bias | Cosmos AR, Genie-style models |
| Trajectory world model | Agent future trajectories | Cheap, directly useful for sim agents | Assumes perception and geometry upstream | Motion forecasting / sim-agent models |

The interview move is to pick the representation from the use case:

- Use **video latent diffusion** when you need sensor-realistic scenario generation.
- Use **occupancy or BEV** when you need checkable dynamics and planning relevance.
- Use **trajectory models** when you need cheap closed-loop agent behavior.
- Use **autoregressive token models** when long rollout and tokenized state are the natural fit.

---

## End-to-end system design questions (in ML Design)

The articles above are the *substrate* — math, backbones, taxonomy. The three problems below apply that substrate end to end; they live in the [ML Design](/ml-design/) section but are listed here because each is one AV specialization (perception/forecasting, simulation, end-to-end driving). Each walks the same arc: **clarify → frame (real-world vs. training objective) → metrics → fallback when uncertain → data & long-tail → baseline → modeling → serving within budget → offline+online eval → shadow/canary/rollback with drift monitoring.**

| Question | Specialization | The core thing it tests |
|---|---|---|
| [Onboard motion forecasting](/ml-design/5-motion-forecasting/) | Motion forecasting | Calibrated multimodal multi-agent prediction under a hard onboard latency budget; never drop the dangerous mode. |
| [Closed-loop driving simulator](/ml-design/6-closed-loop-simulation/) | Simulation | Realistic, reactive, diverse sim-agents + scenario gen at massive throughput, validated by sim-to-real correlation. |
| [End-to-end driving with a world model](/ml-design/7-e2e-driving/) | End-to-end driving | Joint optimization via interpretable-intermediate end-to-end + world-model planning, bounded by a hard safety layer. |

These three deliberately reuse the same engine in different regimes: a **scene-conditioned (often diffusion) generator** of trajectories/futures, served once **onboard with bounded latency** (forecasting), once **offline at huge throughput** (simulation), and once **in the planning loop** (end-to-end). The contrasts in serving strategy across the three are themselves a strong interview point.

---

## Suggested dedicated articles

This is the article plan for the new diffusion section. Some pages already exist elsewhere and can be moved or rewritten into this folder.

| Article | Status |
|---|---|
| `1-ddpm-basics.md` | Exists as [Diffusion Basics](/ml-modeling-fundamentals/diffusion_basics/). |
| `2-ddim-and-ode-view.md` | Covered in [Training-Free Diffusion Solvers](/diffusion/7-training-free-solvers/). |
| `3-training-free-solvers.md` | Exists as [Training-Free Diffusion Solvers](/diffusion/7-training-free-solvers/). |
| `4-classifier-free-guidance.md` | Covered in [Fast Diffusion Sampling](/diffusion/6-diff-optimization/). |
| `5-latent-diffusion-and-tokenizers.md` | Covered in [Diffusion World Model](/diffusion/3-diff-world-model/). |
| `6-unet-dit-backbones.md` | Exists as [U-Net and DiT Backbones](/diffusion/6-unet-dit-backbones/). |
| `7-flow-matching-rectified-flow.md` | Covered in [Fast Diffusion Sampling](/diffusion/6-diff-optimization/). |
| `8-video-diffusion-architectures.md` | Planned. |
| `9-driving-world-models.md` | Covered in [Diffusion World Model](/diffusion/3-diff-world-model/). |
| `10-speed-distillation-and-tradeoffs.md` | Covered in [Fast Diffusion Sampling](/diffusion/6-diff-optimization/). |
| `11-validation-for-world-models.md` | Exists as [Diffusion Validation](/diffusion/5-diff-validation/). |
