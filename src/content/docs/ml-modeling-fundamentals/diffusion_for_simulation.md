---
title: "Diffusion for Autonomous Driving Simulation"
description: "Interview-focused guide to using diffusion models for future scene and trajectory generation in autonomous driving simulation."
---

# Diffusion for Autonomous Driving Simulation

## 1. Interview-level intuition

Autonomous driving simulation needs diverse, realistic, and controllable futures. Diffusion models are attractive because they can sample many plausible futures from the same initial scene.

Given the same history:

- One car yields.
- Another merges.
- A pedestrian waits.
- A cyclist turns.
- A vehicle cuts in.

A deterministic model tends to average these. A diffusion model can sample them.

In simulation, the goal is not just prediction accuracy. The goal is useful generated scenarios for testing planning, prediction, and autonomy behavior.

## 2. Mathematical formulation

Let $c$ be conditioning context:

$$
c = \{\text{map}, \text{agent history}, \text{traffic lights}, \text{route}, \text{intent}, \text{text prompt}\}
$$

Let $x_0$ be the future scene or trajectory to generate.

Conditional diffusion trains:

$$
\mathcal{L} =
\mathbb{E}_{x_0, c, t, \epsilon}
\left[
\|\epsilon - \epsilon_\theta(x_t,t,c)\|_2^2
\right]
$$

where:

$$
x_t = \sqrt{\bar{\alpha}_t}x_0 + \sqrt{1-\bar{\alpha}_t}\epsilon
$$

The model learns:

$$
p_\theta(x_0 | c)
$$

For trajectory generation:

$$
x_0 \in \mathbb{R}^{A \times T \times D}
$$

where:

- $A$ is number of agents.
- $T$ is future timesteps.
- $D$ might include $(x,y,\theta,v)$.

Conditioning can be injected through:

- Concatenation.
- Cross-attention.
- Graph neural networks.
- Map encoders.
- Agent-history encoders.
- Classifier-free guidance.

Classifier-free guidance uses conditional and unconditional predictions:

$$
\hat{\epsilon} =
\epsilon_\theta(x_t,t,\varnothing)
+ s \left[
\epsilon_\theta(x_t,t,c) -
\epsilon_\theta(x_t,t,\varnothing)
\right]
$$

where $s$ controls conditioning strength.

## 3. Why this matters for autonomous driving simulation

Driving simulation needs more than replaying logs. It needs controlled variation:

- Same intersection, different pedestrian behavior.
- Same merge, more aggressive vehicle.
- Same route, rare cut-in.
- Same scene, changed traffic light timing.
- Same map, language instruction: "generate a near-miss merge."

Diffusion can help because:

- Sampling gives diversity.
- Conditioning gives controllability.
- Rare scenarios can be oversampled or guided.
- Multi-agent futures can be generated jointly.
- Generated scenarios can stress-test planners.

Tradeoffs:

- High realism may reduce rare-event frequency.
- Strong guidance may reduce realism.
- More denoising steps increase latency.
- Generated scenes need strict validity metrics.

## 4. Common interview questions and strong answers

**Q: What would the conditioning input include?**  
A: Map geometry, lanes, traffic light state, route, agent history, object types, velocities, interactions, and optional scenario intent or text prompt.

**Q: Why not just use a regression model?**  
A: Regression with MSE tends to average multi-modal futures. Diffusion can sample diverse plausible futures from the same context.

**Q: How do you make diffusion controllable?**  
A: Condition on structured inputs like route or intent, use classifier-free guidance, filter samples with constraints, or add cost/guidance terms during sampling.

**Q: How do you generate rare scenarios?**  
A: Condition on rare-event labels or prompts, oversample rare contexts, guide toward risk metrics, or search over diffusion seeds and retain samples that meet criteria.

## 5. Minimal NumPy or PyTorch implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class ConditionalDenoiser(nn.Module):
    def __init__(self, traj_dim, cond_dim, hidden=128, steps=100):
        super().__init__()
        self.t_emb = nn.Embedding(steps, hidden)
        self.net = nn.Sequential(
            nn.Linear(traj_dim + cond_dim + hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, traj_dim),
        )

    def forward(self, xt, t, cond):
        # xt: [B, traj_dim], cond: [B, cond_dim]
        emb = self.t_emb(t)
        return self.net(torch.cat([xt, cond, emb], dim=-1))

def diffusion_loss(model, x0, cond, alpha_bar):
    B = x0.size(0)
    t = torch.randint(0, alpha_bar.numel(), (B,), device=x0.device)
    noise = torch.randn_like(x0)
    a = alpha_bar[t].view(B, 1)
    xt = a.sqrt() * x0 + (1 - a).sqrt() * noise
    pred = model(xt, t, cond)
    return F.mse_loss(pred, noise)

# Toy: one agent, 5 future steps, x/y => traj_dim = 10
B, traj_dim, cond_dim, steps = 4, 10, 6, 100
betas = torch.linspace(1e-4, 0.02, steps)
alpha_bar = torch.cumprod(1 - betas, dim=0)
model = ConditionalDenoiser(traj_dim, cond_dim, steps=steps)

x0 = torch.randn(B, traj_dim)
cond = torch.randn(B, cond_dim)
loss = diffusion_loss(model, x0, cond, alpha_bar)
loss.backward()
```

## 6. Failure modes and debugging checklist

- Model ignores conditioning.
- Generated trajectories leave drivable area.
- Agents collide unrealistically.
- Rare scenarios are not generated.
- Guidance creates unrealistic behavior.
- Samples are diverse but low quality.
- Samples are realistic but not controllable.
- Denoising is too slow for target workflow.

Checklist:

- Compare conditional vs shuffled-conditioning performance.
- Measure collision/offroad/kinematic violations.
- Plot multiple samples for same scene.
- Track diversity and realism together.
- Evaluate rare scenario recall.
- Test different guidance strengths.
- Use map-based validity checks.

## 7. A 60-second explanation I can say out loud

For simulation, diffusion models learn a conditional distribution over future scenes or trajectories. The conditioning includes map, agent history, traffic lights, route, and possibly intent or language. During training, we add noise to the future trajectory and train the model to predict the noise given the noisy future and context. At generation time, we start from noise and denoise into a plausible future. This is useful because driving futures are multi-modal: the same scene can lead to yielding, merging, braking, or cutting in. Diffusion gives diversity, and conditioning gives controllability.

## 8. 3 practice exercises with answers

**Exercise 1:** What does $p_\theta(x_0|c)$ mean?  
**Answer:** The model distribution over clean future scenes or trajectories $x_0$ conditioned on context $c$.

**Exercise 2:** How would you test whether the model uses traffic lights?  
**Answer:** Change only the traffic light condition and see whether generated trajectories change appropriately; also test shuffled-light conditioning.

**Exercise 3:** What is the tradeoff in increasing guidance strength?  
**Answer:** More controllability, but higher risk of unrealistic or low-diversity samples.

