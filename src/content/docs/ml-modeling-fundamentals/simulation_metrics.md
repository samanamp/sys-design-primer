---
title: "Simulation Metrics: A 1-Hour Interview Learning Session"
description: "A practical one-hour lesson on autonomous driving simulation metrics: collision, offroad, wrong-way, trajectory error, realism, safety, log divergence, and kinematic feasibility."
---

# Simulation Metrics: A 1-Hour Interview Learning Session

Companion notebook: [simulation_metrics_colab.ipynb](/notebooks/simulation_metrics_colab.ipynb)

Simulation metrics answer a deceptively hard question:

> Did we generate a scenario that is realistic, physically valid, controllable, and useful for safety evaluation?

One metric cannot answer that. You need a dashboard.

## 0. One-hour plan

```text
0-10 min   Why trajectory error is not enough
10-25 min  Safety and map-validity metrics
25-40 min  Realism, log divergence, and kinematics
40-50 min  Evaluating generated scenarios
50-60 min  Interview answers and drills
```

---

## 1. Why you should care

Autonomous driving simulation can fail in two opposite ways:

```text
Very realistic, but boring:
  good log replay, poor long-tail safety coverage

Very challenging, but fake:
  lots of collisions, impossible actor motion
```

Good metrics distinguish:

- realism,
- safety relevance,
- physical feasibility,
- map compliance,
- diversity,
- controllability.

In robotics, the same idea applies: a generated manipulation rollout must be physically plausible and task-relevant, not just different.

---

## 2. Trajectory error

Average displacement error:

$$
ADE =
\frac{1}{T}\sum_{t=1}^{T}\|\hat{p}_t-p_t\|_2
$$

Final displacement error:

$$
FDE =
\|\hat{p}_T-p_T\|_2
$$

For $K$ predicted modes:

$$
minADE =
\min_k
\frac{1}{T}
\sum_t
\|\hat{p}^{(k)}_t-p_t\|_2
$$

Why trajectory error is useful:

- Simple.
- Easy to compare.
- Good for prediction against logs.

Why it is insufficient:

- Penalizes plausible alternatives.
- Does not check collision.
- Does not check offroad.
- Does not check wrong-way.
- Does not check generated mode probabilities.

Interview phrase:

> ADE is a prediction metric, not a full simulation-quality metric.

---

## 3. Safety and map-validity metrics

### Collision

For agents $i,j$ with bounding boxes $B_i(t)$ and $B_j(t)$:

$$
\text{collision}_{ij}(t)
=
\mathbf{1}[B_i(t)\cap B_j(t)\ne \varnothing]
$$

Use oriented boxes when possible, not just center distance.

### Offroad

Let $\mathcal{D}$ be drivable area:

$$
\text{offroad}(t)=\mathbf{1}[p_t\notin\mathcal{D}]
$$

### Wrong-way

Let $h_t$ be actor heading and $d_t$ be lane direction:

$$
h_t\cdot d_t < \tau
$$

flags wrong-way behavior.

### Traffic-rule violations

Examples:

- red-light running,
- stop-sign violation,
- illegal turn,
- lane-boundary crossing,
- speed-limit violation.

These require map and signal state, not just trajectories.

---

## 4. Kinematic feasibility

A generated trajectory can be map-valid but physically impossible.

Speed:

$$
v_t=
\frac{\|p_t-p_{t-1}\|}{\Delta t}
$$

Acceleration:

$$
a_t=
\frac{v_t-v_{t-1}}{\Delta t}
$$

Jerk:

$$
j_t=
\frac{a_t-a_{t-1}}{\Delta t}
$$

For vehicles, also inspect:

- yaw rate,
- curvature,
- lateral acceleration,
- reverse motion,
- discontinuities.

Kinematic metrics catch teleporting actors and unrealistic sudden turns.

---

## 5. Realism vs safety

Realism and safety challenge are in tension.

```text
Log-likelihood high:
  likely realistic, but may be common/boring

Risk high:
  useful for safety, but may be unrealistic
```

A useful generated scenario should sit in the middle:

```text
rare enough to test the system
plausible enough to matter
controllable enough to reproduce
```

This is why collision rate alone is not enough. A high collision rate may mean the generator is broken.

---

## 6. Log divergence

If starting from a logged scene, divergence from log is:

$$
D(t)=\|\hat{p}_t-p_t^{log}\|_2
$$

This helps measure how quickly a rollout departs from recorded behavior.

Interpretation:

- Low divergence: close replay.
- Moderate divergence: plausible variation.
- Huge early divergence: likely unrealistic or uncontrolled.

But divergence is not always bad. Simulation often wants counterfactuals. The question is whether divergence is plausible and controlled.

---

## 7. Evaluating generated scenarios

Use a dashboard:

```text
Prediction:
  ADE, FDE, minADE, miss rate

Map validity:
  offroad, wrong-way, rule violation

Safety:
  collision, near-miss, time-to-collision

Physics:
  speed, acceleration, jerk, yaw rate

Realism:
  discriminator score, log divergence, human review

Diversity:
  unique modes, coverage, pairwise distance

Control:
  prompt/intent success rate

Downstream:
  planner failure rate, intervention rate
```

The strongest interview answer is to discuss metric tradeoffs, not just list metrics.

---

## 8. Minimal PyTorch implementation

```python
import torch

def ade(pred, target):
    return torch.norm(pred - target, dim=-1).mean()

def fde(pred, target):
    return torch.norm(pred[:, -1] - target[:, -1], dim=-1).mean()

def kinematic_flags(traj, dt=0.1, max_speed=40.0, max_accel=8.0):
    vel = (traj[:, 1:] - traj[:, :-1]) / dt
    speed = torch.norm(vel, dim=-1)
    accel = (speed[:, 1:] - speed[:, :-1]) / dt
    return (speed > max_speed).any(dim=1), (accel.abs() > max_accel).any(dim=1)

def rectangular_offroad(traj, x_min, x_max, y_min, y_max):
    x, y = traj[..., 0], traj[..., 1]
    off = (x < x_min) | (x > x_max) | (y < y_min) | (y > y_max)
    return off.any(dim=1)
```

For real systems, replace rectangular offroad with map polygon checks.

---

## 9. Common interview questions and strong answers

**Q: Why is ADE insufficient?**  
A: It compares against one logged future and ignores multi-modality, collision, offroad, wrong-way, and physical feasibility.

**Q: Is collision rate always bad?**  
A: No. For rare scenario generation, collisions or near-misses may be intentional, but they must be physically plausible.

**Q: How do you evaluate a generated scenario?**  
A: Use a dashboard covering realism, safety relevance, map compliance, kinematics, diversity, controllability, and downstream planner impact.

**Q: What is log divergence?**  
A: It measures how far the generated rollout departs from the logged trajectory over time. It helps distinguish replay from counterfactual generation.

---

## 10. A 60-second explanation you can say out loud

Simulation metrics need to measure more than trajectory error. ADE and FDE tell me how close I am to one logged future, but driving is multi-modal and simulation needs plausible alternatives. I also need collision, offroad, wrong-way, traffic-rule, and kinematic feasibility metrics. For generated scenarios, I evaluate realism, diversity, controllability, and safety relevance. A collision is useful only if it is physically plausible, not if it comes from impossible actor motion. The best evaluation is a dashboard plus visual inspection of high-risk cases.

---

## 11. Practice exercises with answers

**Exercise 1:** Why can minADE hide bad probability calibration?  
**Answer:** It only checks whether one mode matches the log, not whether the model assigned that mode high probability.

**Exercise 2:** Why are oriented boxes better than center distance for collision?  
**Answer:** Vehicles have size and heading. Center distance can miss side-swipe or corner collisions.

**Exercise 3:** A generated car goes from 0 to 30 m/s in 0.1s. Which metric catches it?  
**Answer:** Acceleration or jerk feasibility.

**Exercise 4:** What does high early log divergence suggest?  
**Answer:** The generated rollout may be uncontrolled or unrealistic, unless the goal explicitly requested a strong counterfactual.

