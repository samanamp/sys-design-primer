---
title: "Transformer Motion Forecasting for Autonomous Driving"
description: "Senior/staff ML system design answer for transformer-based multimodal trajectory prediction in autonomous driving simulation."
---

# Transformer Motion Forecasting for Autonomous Driving

## Research Pass / State of the Art as of 2026

**[SIGNAL: research-before-design]** The transformer motion-forecasting lineage has converged on vectorized, interaction-aware, multimodal prediction rather than single-agent regression. **VectorNet** established the core representation shift: encode HD map lanes and agent histories as vectors/polylines rather than raster BEV images, avoiding lossy rendering and heavy CNN compute while preserving geometric structure ([VectorNet](https://arxiv.org/abs/2005.04259)). **Scene Transformer** showed a unified scene-centric transformer for multiple prediction tasks and emphasized permutation-equivariant multi-agent modeling ([Scene Transformer](https://waymo.com/research/scene-transformer-a-unified-architecture-for-predicting-multiple-agent-trajectories/)). **Wayformer**, Waymo's own ICRA 2023 work, simplified the architecture into attention-based scene encoder + trajectory decoder, compared early/late/hierarchical fusion, and studied factorized and latent-query attention to keep forecasting practical under real-time constraints ([Wayformer](https://waymo.com/research/wayformer/)).

The strongest leaderboard-style models then moved toward query/intention decoders. **MTR** frames forecasting as global intention localization plus local trajectory refinement using learnable motion queries ([MTR](https://arxiv.org/abs/2209.13508)); **MTR++** adds symmetric scene modeling and guided intention querying ([MTR++](https://arxiv.org/abs/2306.17770)). **QCNet** pushes query-centric coordinates and relative encodings, a key symmetry idea for translation/rotation invariance ([QCNet](https://openaccess.thecvf.com/content/CVPR2023/papers/Zhou_Query-Centric_Trajectory_Prediction_CVPR_2023_paper.pdf)). **MotionLM** casts multi-agent forecasting as language modeling over discrete motion tokens and directly models joint interactive futures, ranking first on the WOMD interactive leaderboard ([MotionLM](https://arxiv.org/abs/2309.16534), [Waymo blog](https://blog.waymo.com/intl/jp/research/motionlm)).

WOMD remains the reference benchmark: 1s history + current state, 8s future at 10Hz, map data, and metrics including minADE, minFDE, miss rate, mAP, and overlap rate ([WOMD](https://waymo.com/intl/it/open/data/motion/)). Recent work broadens the input substrate: **MoST** tokenizes multi-modal scene evidence using image foundation features and LiDAR encoders, addressing limitations of symbolic-only perception outputs ([MoST](https://waymo.com/research/most-multi-modality-scene-tokenization-for-motion-prediction/)). Waymo's 2025 scaling-laws work reports power-law-like improvements for encoder-decoder autoregressive transformer models in forecasting and planning, with inference compute helping difficult scenarios ([Scaling Laws](https://waymo.com/research/scaling-laws-of-motion-forecasting-and-planning/)). Frontier direction: vector/query-centric transformers, joint scene-level prediction, richer sensor-token context, and closed-loop evaluation, not just open-loop minADE.

## Problem Framing and Task Definition

**[SIGNAL: problem-framing-first]** I would frame this before drawing any transformer. The system runs onboard the AV at roughly 10Hz. At each planning cycle, it receives the current tracked scene and predicts a distribution over future motion for surrounding agents.

Input:

- 1s of history for each tracked agent: position, velocity, heading, size, type, validity mask.
- HD map around the AV: lane centerlines, boundaries, crosswalks, stop signs, speed limits, connectivity.
- Traffic light/sign state over the recent history and current time.
- AV state and route/context, because nearby agents react to the AV and road intent.
- Optional perception uncertainty: track confidence, occlusion state, object covariance.

Output:

- For each relevant agent $i$, $K$ plausible future trajectories over 8 seconds at 10Hz or a lower planning resolution:

$$
\{(\pi_{i,k}, \hat{Y}_{i,k})\}_{k=1}^{K}, \quad \hat{Y}_{i,k} \in \mathbb{R}^{T \times d}
$$

where $\pi_{i,k}$ is mode probability and $d$ includes at least $(x,y,\theta,v)$ or enough state for planning.

**[SIGNAL: multimodality-as-core]** The output is a distribution, not a single trajectory. A car at an intersection may go straight, turn left, or turn right; a pedestrian may wait or cross. A single MSE trajectory learns the conditional mean, which can be physically meaningless.

The planner consumes the distribution. **[SIGNAL: planning-aware]** Success is not "best minADE"; success is whether the planner can make safe, comfortable, legal decisions under uncertainty. If the forecaster is uncertain, that uncertainty must be honest and calibrated so the planner can slow down, yield, increase clearance, or request fallback behavior.

Several seconds is the hard horizon: under one second, constant velocity/turn-rate often works; over 3-8 seconds, intent, route, traffic rules, and interaction dominate.

## Baseline Before Complexity

**[SIGNAL: baseline-first]** I would start with baselines:

1. **Constant velocity / constant turn-rate.** Extrapolate the latest kinematic state. This is hard to beat for short horizons, parked vehicles, and simple lane following.
2. **Lane-following heuristic.** Project agents along candidate lane centerlines with speed estimates. Strong on highways and structured roads.
3. **Simple learned recurrent or MLP baseline.** Agent history + nearest lane features, independent per agent, multimodal head.

The transformer must pay rent. It adds value when:

- agent-agent interactions matter,
- map geometry is complex,
- futures are multimodal,
- long-horizon intent matters,
- scene context contains many heterogeneous elements,
- marginal independent predictions become inconsistent.

I would ship no transformer until these baselines are measured by slice. If the transformer only improves aggregate minADE on easy highway cases and not pedestrians/intersections/merges, it is not the right model.

## Scene Representation and Tokenization

**[SIGNAL: symmetry-aware representation]** This is the transformer-depth core. A driving scene is not naturally a sequence like text. We must define tokens and position encodings that respect geometry.

I would use a mostly vectorized representation, following VectorNet/Wayformer/MTR/QCNet lineage. Raster BEV can work, especially for perception-like cues, but it is lossy for map topology and expensive. Forecasting needs exact lane geometry, connectivity, and agent states; vectors preserve that.

Token types:

```text
Agent-history token:
  type, size, current state, encoded history polyline

Lane-polyline token:
  lane segment points, direction, speed limit, turn type, connectivity

Crosswalk/stop/sign token:
  polygon/polyline geometry + semantic type

Traffic-light token:
  controlled lane id + state + timing/history

AV-route token:
  route lane sequence / intended path context
```

For each polyline, a small local encoder processes points into a token:

```text
lane points:  p1 -> p2 -> ... -> pm
                 local polyline encoder
                         |
                         v
                   lane token

agent history: s(t-10) ... s(t0)
                 temporal/polyline encoder
                         |
                         v
                   agent token
```

Then a transformer attends across tokens:

```text
Raw scene
  |
  +-- agent histories ---- local temporal encoder --- agent tokens
  +-- lane polylines ----- local polyline encoder --- map tokens
  +-- lights/signs ------- embedding/projection ----- signal tokens
  +-- route/AV state ----- embedding/projection ----- route tokens
                                                        |
                                                        v
                                                scene transformer
```

Coordinate frame matters. Options:

- **Scene-centric/global frame:** encode everything once. Efficient, but not invariant to translation/rotation.
- **Agent-centric frame:** recenter/rotate the scene around each target agent. Strong invariance, but re-encoding per agent is expensive.
- **Query-centric/relative frame:** maintain shared scene tokens but use relative position/heading encodings between query and key tokens.

I prefer query-centric or agent-centric for modeled agents. Forecasting should be equivariant to global translation and rotation: if the same intersection is moved or rotated, behavior should not change. Relative geometry is the true signal. This parallels relative position encodings/RoPE in LLMs: absolute token index is less important than relations among tokens.

For batching, variable token counts are handled with padding and masks. I also cap far-away or irrelevant map tokens by radius/topological reachability to control latency.

## Attention Architecture and Interaction Modeling

**[SIGNAL: interaction-modeling depth]** I would use an encoder-decoder transformer:

- Encoder fuses agents, map, lights, route, and AV context.
- Decoder uses mode/intention queries to produce multimodal futures.

Attention structure:

```text
                 map tokens
                    ^
                    | agent-map attention
                    |
agent tokens <---- scene attention ----> signal/route tokens
     ^
     |
agent-agent attention
```

Agent-agent attention captures yielding, merging, following, right-of-way, and social interaction. Agent-map attention captures lane following, crosswalks, stop control, and feasible paths. Signal attention connects traffic lights to controlled lanes and agents.

Dense attention over all tokens is expensive. With 100 agents and hundreds of map polylines, naive all-pairs attention can exceed the 100ms cycle budget. **[SIGNAL: real-time-onboard-constraint]** I would use locality and factorization:

- agent tokens attend to nearby agents, not all agents globally;
- agents attend to reachable map polylines within radius/topological corridor;
- map-map attention can be precomputed or factorized;
- route/light tokens use sparse connectivity;
- latent-query attention, as in Wayformer, can compress scene context.

Marginal vs joint prediction is central. Marginal prediction gives each agent independent modes:

```text
agent A: modes A1, A2, A3
agent B: modes B1, B2, B3
```

This is cheap but can be inconsistent: two vehicles may both occupy the same gap. Joint prediction models scene-level futures:

```text
scene mode 1: A yields, B goes
scene mode 2: A goes, B yields
```

Joint prediction is harder because combinations scale badly, but it is more aligned with planning. My production design would start with marginal per-agent modes plus interaction-aware scoring and overlap penalties, then move high-risk interactive slices toward joint scene-level prediction or MotionLM-style autoregressive joint decoding.

## Multimodal Output Head and Training Problem

The output head is the design center. Options:

1. **Single regression:** reject except as a baseline. It averages modes.
2. **Anchor/intention-based:** cluster future trajectories or goals; classify anchor and regress residual.
3. **GMM head:** output mixture weights, means, covariances over future trajectories.
4. **Query-based decoder:** learned mode queries attend to scene embeddings and each emits one trajectory/mode probability.
5. **Goal-conditioned:** predict endpoints/goals first, then trajectory conditioned on each goal.
6. **Motion-token LM:** discretize motion into tokens and autoregressively generate joint futures, as in MotionLM.

For a transformer system, I would use query-based multimodal decoding with optional goal/intention priors:

```text
scene embedding tokens
       |
       v
 K learned mode queries per target agent
       |
       +-- query 1 -> trajectory 1 + probability
       +-- query 2 -> trajectory 2 + probability
       +-- query 3 -> trajectory 3 + probability
       ...
```

The loss must confront the label problem. **[SIGNAL: single-sample-label honesty]** In logs, we observe one future. We do not observe all counterfactuals that could have happened. For a car that went straight, the logged label does not say whether a left turn was also plausible.

Standard best-of-K / winner-takes-all:

$$
k^* = \arg\min_k ADE(\hat{Y}_k, Y)
$$

$$
\mathcal{L} =
\mathcal{L}_{reg}(\hat{Y}_{k^*}, Y)
+ \lambda \cdot CE(\pi, k^*)
$$

This encourages at least one mode to match the observed future and assigns probability to that mode. It is practical but imperfect.

Failure modes:

- **Mode collapse:** all modes become similar.
- **Probability miscalibration:** right mode exists but low probability.
- **Single-sample bias:** plausible unobserved modes are not rewarded.
- **Anchor coverage gaps:** rare maneuvers absent from anchors.

Mitigations:

- initialize modes with trajectory/goal clusters;
- use diversity or repulsion regularizers carefully;
- use anchor/intention auxiliary losses;
- train with hard scenario mining and rare maneuver oversampling;
- use EM-style soft assignment or NLL mixture training;
- calibrate mode probabilities post-training;
- add scene-consistency losses: collision/overlap penalties for joint modes;
- evaluate probability-aware metrics, not just minADE.

For high-interaction areas, I would consider a scene-level decoder: a set of scene mode queries emits coordinated futures for the relevant agents, avoiding post-hoc collision between marginal modes.

## Metrics: Slice-Based and Safety-Aware

**[SIGNAL: slice-based safety metrics]** I would never report one aggregate minADE. Standard open-loop metrics:

- minADE$_K$, minFDE$_K$;
- miss rate;
- mAP / soft mAP for probability-ranked modes;
- overlap rate / collision consistency;
- calibration error for mode probabilities.

minADE is useful but insufficient: it rewards mode coverage but can hide probability errors. A model can include the true future as its 6th mode with 1% probability and get good minADE while being unsafe for planning.

Slices:

- agent type: pedestrian, cyclist, vehicle, emergency vehicle;
- scenario: intersection, merge, unprotected left, crosswalk, highway cut-in;
- interaction density: isolated vs dense;
- speed regime: parked, low-speed, high-speed;
- geography/weather/time;
- occlusion and perception uncertainty;
- rare long-tail: jaywalkers, red-light runners, wrong-way cyclists.

Planning-relevant metrics:

- Did the planner choose a safe trajectory using these forecasts?
- Did forecast uncertainty lead to appropriate yielding/slowing?
- Did missed modes correlate with near-miss or hard-brake events?
- Closed-loop sim safety: collisions, near-misses, comfort, progress.

Calibration is safety-critical. If pedestrian crossing is predicted at 10% when empirically it happens 50% in similar contexts, the planner will underreact.

## Data Strategy and Long-Tail Coverage

Labels come from logs: the future trajectory is what actually happened. That is cheap and high-volume, but biased.

Data sources:

- production driving logs;
- manual driving logs where useful;
- simulation-generated perturbations;
- mined rare/hard scenarios;
- map and traffic signal state;
- perception uncertainty metadata.

Long-tail coverage is the main data problem. Highway cruising dominates logs; rare dangerous interactions are sparse. I would mine:

- near misses;
- hard braking/yielding;
- planner discomfort;
- prediction disagreement;
- high entropy;
- high forecast error;
- unusual agent behavior;
- geography/weather-specific failures.

Then stratify training/evaluation by slices. I would upweight interactive and rare slices but protect against noisy labels. The AV's own behavior biases the data: a conservative AV may cause other agents to behave conservatively, and it may avoid the very risky interactions we need to model. This self-selection bias must be tracked by geography, AV behavior policy version, and scenario type.

## Real-Time Onboard Serving

The model runs every ~100ms with the planner waiting. A late prediction is useless.

Serving constraints:

- bounded token count;
- bounded number of predicted agents;
- bounded K modes;
- sparse/local attention;
- quantized/distilled model for onboard compute;
- incremental updates across frames;
- deterministic latency budget and fallback.

I would precompute map token embeddings for local map tiles where possible. Across frames, agent histories shift slightly, so we can cache static map context and reuse recent scene encodings if the tracking set changes modestly. I would prioritize agents by relevance: nearby, on collision-relevant routes, vulnerable road users, high uncertainty. Far-away parked agents can use cheap baselines.

Model budget is part of model design: a 2% minADE gain that doubles p99 latency is not deployable.

## Offline/Online Mismatch and Safe Rollout

**[SIGNAL: offline-online-mismatch]** Open-loop minADE does not guarantee better driving. In closed loop, the planner reacts to forecasts, changes the future, and may create new interactions. Small forecast errors can compound.

Evaluation ladder:

1. Offline validation on WOMD-like and internal slices.
2. Counterfactual closed-loop simulation, e.g. Waymax-style evaluation, with the full planner using the new forecast.
3. Stress scenarios: rare pedestrians, merges, occlusions, construction, weather.
4. Shadow mode onboard: model runs but does not control; log disagreements and latency.
5. Canary rollout: geofenced, limited fleet, safety monitors.
6. Gradual expansion with rollback.

Rollback triggers:

- slice regression above threshold;
- latency p99 violation;
- increased hard braking/near-miss in sim or shadow;
- calibration degradation on vulnerable road users;
- planner conservatism regression: too timid or too aggressive.

## Monitoring, Drift, and Retraining

**[SIGNAL: full-system-design]** Monitoring must be slice-aware:

- forecast error when future labels become available;
- mode entropy and calibration;
- missed-mode events;
- overlap/collision inconsistency;
- latency and fallback rate;
- scenario distribution drift.

Drift sources:

- new cities/geographies;
- weather/season;
- road layout changes;
- new agent behaviors;
- construction;
- perception model changes upstream.

Retraining triggers:

- drift detected in a slice;
- cluster of high-error failures;
- new geography launch;
- upstream perception schema/model change;
- repeated planner interventions tied to forecasting.

Failure mining should feed a curated long-tail set with stable holdouts to prevent regressions.

## The Safety Fallback

**[SIGNAL: safety-fallback]** The planner should not treat the top mode as truth. It consumes the full distribution.

When uncertainty is high:

- high entropy over modes;
- low confidence;
- OOD agent behavior;
- perception uncertainty;
- conflicting joint futures;

the planner uses conservative behavior: slow, increase buffers, yield, request more evidence, or fall back to rule/physics predictions. An overconfident wrong forecast is more dangerous than a humble uncertain forecast. I would calibrate uncertainty and expose it explicitly to planning.

## End-to-End System Diagram

```text
ONBOARD LOOP (~10Hz)

Sensors/perception/tracking/map/light state
        |
        v
Scene tokenization
  agent history tokens + map polyline tokens + signal/route tokens
        |
        v
Transformer scene encoder
  local/factorized agent-agent and agent-map attention
        |
        v
Multimodal decoder
  K mode queries per agent or joint scene queries
        |
        v
Trajectories + probabilities + uncertainty
        |
        v
Planner
  risk-aware planning over forecast distribution


OFFLINE LOOP

Logs -> scenario mining/slicing -> training set
        |                         |
        v                         v
 labels from future          model training
        |                         |
        v                         v
 offline metrics -> closed-loop sim -> shadow -> canary -> rollout
        ^                                           |
        |                                           v
 monitoring/drift/failures <---------------- retraining
```

## What I Would Push Back On

**[SIGNAL: saying no]** I would push back on the assumption that "better forecasting benchmark metrics imply safer driving." The real objective is closed-loop safety and planner performance. A model can improve minADE while making probabilities less calibrated or missing a low-probability dangerous pedestrian mode.

I would also push back on purely marginal prediction if the planner depends on scene consistency in dense interactions. Marginal modes are cheaper and often good enough for low-interaction slices, but intersections and merges need joint consistency or at least interaction-aware scoring.

Finally, I would question whether forecasting should remain a separate module everywhere. Modular forecasting is interpretable and safety-testable; end-to-end driving/world-model approaches may capture richer feedback. For a safety-critical product, I would keep the modular forecaster unless the end-to-end alternative can beat it in closed-loop safety, interpretability, rollback, and monitoring.
