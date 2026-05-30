---
title: "Simulation Metrics for Autonomous Driving Models"
description: "Interview-focused guide to collision, offroad, wrong-way, trajectory error, realism, safety, log divergence, kinematic infeasibility, and evaluating generated driving scenarios."
---

# Simulation Metrics for Autonomous Driving Models

## 1. Interview-level intuition

Simulation metrics answer two questions:

1. Is the generated scenario realistic?
2. Is it useful for evaluating safety?

These are not identical. A scenario can be realistic but boring. A scenario can be safety-critical but physically impossible. Good simulation evaluation balances realism, diversity, controllability, and safety relevance.

For autonomous driving, metrics should catch:

- Collisions.
- Offroad behavior.
- Wrong-way driving.
- Kinematic impossibility.
- Log divergence.
- Poor trajectory prediction.
- Unrealistic interactions.

## 2. Mathematical formulation

### Trajectory error

Average displacement error:

$$
ADE = \frac{1}{T}\sum_{t=1}^{T}\|\hat{p}_t - p_t\|_2
$$

Final displacement error:

$$
FDE = \|\hat{p}_T - p_T\|_2
$$

For multi-modal predictions:

$$
minADE = \min_k \frac{1}{T}\sum_t \|\hat{p}^{(k)}_t - p_t\|_2
$$

### Collision

For agents $i,j$ with bounding boxes $B_i(t), B_j(t)$:

$$
\text{collision}_{ij}(t) =
\mathbf{1}[B_i(t) \cap B_j(t) \ne \varnothing]
$$

Collision rate:

$$
\frac{\text{number of scenarios with collision}}{\text{number of scenarios}}
$$

### Offroad

Let $p_t$ be agent position and $\mathcal{D}$ be drivable area:

$$
\text{offroad}(t) = \mathbf{1}[p_t \notin \mathcal{D}]
$$

### Wrong-way

Let heading vector be $h_t$ and lane direction be $d_t$:

$$
\text{wrong-way}(t) = \mathbf{1}[h_t \cdot d_t < \tau]
$$

where $\tau$ is a threshold, often near zero or negative.

### Kinematic infeasibility

Speed:

$$
v_t = \frac{\|p_t - p_{t-1}\|}{\Delta t}
$$

Acceleration:

$$
a_t = \frac{v_t - v_{t-1}}{\Delta t}
$$

Jerk:

$$
j_t = \frac{a_t - a_{t-1}}{\Delta t}
$$

Flag if speed, acceleration, yaw rate, or jerk exceed feasible thresholds.

### Log divergence

If a generated rollout starts from a logged scene, log divergence measures how quickly it departs from logged behavior:

$$
D(t) = \|\hat{p}_t - p_t^{log}\|_2
$$

Some divergence is expected if generating alternatives. Too much divergence too early may indicate unrealistic dynamics.

## 3. Why this matters for autonomous driving simulation

Simulation is used to test autonomy systems before real-world deployment. Bad metrics can create false confidence.

Examples:

- Low ADE but many offroad trajectories: not acceptable.
- High realism but no rare events: poor safety coverage.
- Many collisions caused by physically impossible actors: not useful.
- Diverse trajectories that violate lane topology: low quality.

Metrics must separate:

- Prediction accuracy.
- Physical validity.
- Map compliance.
- Interaction realism.
- Safety-criticality.
- Scenario diversity.

## 4. Common interview questions and strong answers

**Q: Is collision rate always bad?**  
A: Not necessarily. For generated rare scenarios, collisions or near-collisions may be intentional. But they must be physically plausible and controllable, not artifacts.

**Q: Why is ADE insufficient?**  
A: ADE compares to one logged future. It penalizes plausible alternative futures and does not catch offroad, wrong-way, collision, or interaction realism.

**Q: How do you evaluate generated scenarios?**  
A: Use a dashboard: realism metrics, safety metrics, map compliance, kinematic feasibility, diversity, controllability, and downstream planner impact.

**Q: What is the realism vs safety tradeoff?**  
A: Real logs are mostly normal driving, but safety testing needs rare events. We need scenarios that are rare and challenging while still physically plausible.

## 5. Minimal NumPy or PyTorch implementation

```python
import torch

def ade(pred, target):
    # pred, target: [B, T, 2]
    return torch.norm(pred - target, dim=-1).mean()

def fde(pred, target):
    return torch.norm(pred[:, -1] - target[:, -1], dim=-1).mean()

def speed_accel_violations(traj, dt=0.1, max_speed=40.0, max_accel=8.0):
    # traj: [B, T, 2]
    vel = (traj[:, 1:] - traj[:, :-1]) / dt
    speed = torch.norm(vel, dim=-1)
    accel = (speed[:, 1:] - speed[:, :-1]) / dt

    speed_bad = speed > max_speed
    accel_bad = accel.abs() > max_accel
    return speed_bad.any(dim=1), accel_bad.any(dim=1)

def point_offroad(traj, x_min, x_max, y_min, y_max):
    x, y = traj[..., 0], traj[..., 1]
    off = (x < x_min) | (x > x_max) | (y < y_min) | (y > y_max)
    return off.any(dim=1)

pred = torch.tensor([[[0., 0.], [1., 0.], [2., 0.]]])
target = torch.tensor([[[0., 0.], [1., 1.], [2., 1.]]])
print(ade(pred, target))
print(fde(pred, target))
```

## 6. Failure modes and debugging checklist

- ADE rewards average but unrealistic futures.
- Collision metric uses points instead of boxes.
- Offroad metric ignores map topology.
- Wrong-way metric fails at intersections.
- Kinematic thresholds too strict for unusual but valid behavior.
- Log divergence penalizes valid alternative futures.
- Safety metrics reward impossible adversarial behavior.

Checklist:

- Visualize metric failures.
- Use bounding boxes for collision.
- Slice metrics by agent type.
- Separate generated rare events from accidental invalidity.
- Track diversity and realism together.
- Evaluate downstream autonomy performance.
- Manually inspect high-risk generated scenarios.

## 7. A 60-second explanation I can say out loud

Simulation metrics need to measure realism and safety usefulness. ADE and FDE compare predicted trajectories to logs, but they are not enough because driving is multi-modal. I also need collision, offroad, wrong-way, and kinematic feasibility metrics. For generated scenarios, I care whether the scene is physically plausible, map-compliant, diverse, controllable, and challenging for the planner. A rare collision scenario is useful only if it is realistic, not if it comes from impossible actor motion.

## 8. 3 practice exercises with answers

**Exercise 1:** Why can minADE be misleading?  
**Answer:** It only checks whether one mode matches the log and may ignore low probability assigned to that mode or invalid other modes.

**Exercise 2:** Why use boxes for collision instead of center distance?  
**Answer:** Vehicles have dimensions and orientation. Center distance can miss or falsely report collisions.

**Exercise 3:** What metric catches impossible sudden motion?  
**Answer:** Kinematic infeasibility: speed, acceleration, yaw rate, and jerk thresholds.

