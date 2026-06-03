---
title: "ML Modeling Fundamentals: Overview and Study Plan"
description: "A practical study plan for ML modeling interviews focused on autonomous driving simulation, robotics, metrics, losses, diffusion, and debugging."
---

# ML Modeling Fundamentals: Overview and Study Plan

This section is built for a hybrid ML Modeling & Fundamentals interview: explain theory from first principles, reason about modeling choices, and implement small NumPy/PyTorch snippets.

The target domain is autonomous driving simulation, but the ideas also transfer to robotics, ranking, perception, prediction, and model debugging.

## What you should be able to do

After this section, you should be able to:

- Choose the right metric instead of optimizing blindly.
- Explain precision, recall, F1, PR-AUC, calibration, ADE/FDE, and ranking metrics.
- Explain why loss functions create incentives.
- Use weighted cross entropy, focal loss, and multi-modal trajectory losses.
- Explain diffusion from first principles: forward noise, reverse denoising, epsilon prediction.
- Explain why diffusion helps generate diverse future scenes.
- Evaluate simulation outputs for realism, safety, map validity, and physical feasibility.
- Debug a model that is not learning using a systematic checklist.
- Write small PyTorch snippets in CoderPad.

## Recommended study order

### 1. Start with metrics

Read: [ML Modeling Metrics](./ml_modeling_metrics/)

Why first: metrics tell you what "good" means. Without them, losses and models are easy to misuse.

Focus on:

- confusion matrix,
- precision vs recall,
- thresholding,
- PR-AUC for rare events,
- calibration,
- ADE/FDE limitations.

Notebook: [ml_modeling_metrics_colab.ipynb](/notebooks/ml_modeling_metrics_colab.ipynb)

### 2. Learn custom losses

Read: [Custom Losses](./custom_losses/)

Why next: losses are how you make the model care about the right mistakes.

Focus on:

- weighted cross entropy,
- focal loss,
- why hard examples are not always good examples,
- why MSE fails for multi-modal futures,
- multi-modal trajectory loss.

Notebook: [custom_losses_colab.ipynb](/notebooks/custom_losses_colab.ipynb)

### 3. Learn diffusion basics

Read: [Diffusion Basics](./diffusion_basics/)

Why next: diffusion is easier once you understand loss design and multi-modal prediction.

Focus on:

- forward noising,
- reverse denoising,
- epsilon prediction,
- timestep conditioning,
- why sampling gives multiple plausible outputs.

Notebook: [diffusion_basics_colab.ipynb](/notebooks/diffusion_basics_colab.ipynb)

### 4. Apply diffusion to simulation

Read: [Diffusion for Simulation](./diffusion_for_simulation/)

Why next: this connects the math to autonomous driving and robotics.

Focus on:

- conditioning on map/history/lights/route/intent,
- controllability,
- rare scenario generation,
- realism vs safety tradeoffs.

Notebook: [diffusion_for_simulation_colab.ipynb](/notebooks/diffusion_for_simulation_colab.ipynb)

### 5. Evaluate generated scenarios

Read: [Simulation Metrics](./simulation_metrics/)

Why next: generated scenarios need different evaluation than ordinary prediction.

Focus on:

- collision,
- offroad,
- wrong-way,
- kinematic infeasibility,
- log divergence,
- realism vs safety.

Notebook: [simulation_metrics_colab.ipynb](/notebooks/simulation_metrics_colab.ipynb)

### 6. Finish with debugging

Read: [Debugging a Model That Is Not Learning](./debugging_model_not_learning/)

Why last: debugging requires knowing what the model, loss, and metrics are supposed to do.

Focus on:

- overfit-one-batch test,
- data skew,
- bad labels,
- normalization bugs,
- learning rate problems,
- leakage,
- gradient inspection.

Notebook: [debugging_model_not_learning_colab.ipynb](/notebooks/debugging_model_not_learning_colab.ipynb)

## One-week study plan

### Day 1: Metrics

Goal: never say "accuracy" blindly.

Tasks:

- Read the metrics article.
- Run the notebook.
- Explain precision and recall out loud using pedestrian crossing.
- Practice threshold tradeoffs.

### Day 2: Custom losses

Goal: understand how losses change model incentives.

Tasks:

- Read the custom losses article.
- Run the notebook.
- Implement focal loss from memory.
- Explain why MSE fails for multi-modal futures.

### Day 3: Diffusion basics

Goal: explain DDPM without paper notation overload.

Tasks:

- Read diffusion basics.
- Run the noising notebook.
- Write the equation for $x_t$ from memory.
- Explain why predicting epsilon is convenient.

### Day 4: Diffusion for simulation

Goal: connect diffusion to future-scene generation.

Tasks:

- Read diffusion for simulation.
- Run the conditional trajectory notebook.
- Explain map/history/light/route conditioning.
- Discuss rare scenario generation tradeoffs.

### Day 5: Simulation metrics

Goal: evaluate generated driving scenarios rigorously.

Tasks:

- Read simulation metrics.
- Run the metrics notebook.
- Implement ADE/FDE and kinematic checks.
- Explain realism vs safety.

### Day 6: Debugging

Goal: have a reliable debugging playbook.

Tasks:

- Read debugging article.
- Run the overfit-one-batch notebook.
- Practice diagnosing train/eval loss patterns.
- List autonomous-driving-specific data bugs.

### Day 7: Mock interview loop

Goal: combine theory, modeling judgment, and code.

Practice prompts:

- "Why is accuracy bad for rare event detection?"
- "Implement focal loss."
- "Why does MSE fail for multi-modal trajectories?"
- "Explain diffusion forward and reverse processes."
- "How would you condition a diffusion model on map and traffic lights?"
- "How do you evaluate generated scenarios?"
- "Your model is not learning. Walk me through your debugging process."

## 60-minute cram plan

If you only have one hour:

```text
0-10 min   Metrics: precision, recall, PR-AUC, calibration
10-20 min  Losses: weighted CE, focal loss
20-30 min  Multi-modal trajectory loss and MSE failure
30-40 min  Diffusion basics: x_t equation and epsilon prediction
40-50 min  Simulation: conditioning and scenario metrics
50-60 min  Debugging: overfit-one-batch and gradient inspection
```

## CoderPad checklist

Be ready to implement:

- confusion matrix metrics,
- precision/recall/F1,
- focal loss,
- weighted cross entropy call,
- multi-modal trajectory loss,
- ADE/FDE,
- kinematic speed/acceleration checks,
- overfit-one-batch training loop,
- gradient norm helper,
- diffusion `q_sample`.

## Interview stance

A strong answer usually has this shape:

1. Define the task and the cost of mistakes.
2. Choose metrics that reflect that cost.
3. Choose a loss that optimizes toward the metric.
4. Discuss tradeoffs and failure modes.
5. Give a small implementation.
6. Say how you would debug it.

That structure works for most ML modeling questions in autonomous driving simulation.

