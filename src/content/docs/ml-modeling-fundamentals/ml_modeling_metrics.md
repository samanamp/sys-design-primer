---
title: "ML Modeling Metrics: A 1-Hour Interview Learning Session"
description: "A practical one-hour lesson on precision, recall, F1, ROC-AUC, PR-AUC, calibration, regression metrics, ranking metrics, and thresholding for autonomous driving and robotics interviews."
---

# ML Modeling Metrics: A 1-Hour Interview Learning Session

Companion notebook: [ml_modeling_metrics_colab.ipynb](/notebooks/ml_modeling_metrics_colab.ipynb)

Metrics tell you whether your model is improving the thing you actually care about. Without the right metric, you can optimize blindly.

For autonomous driving, the key point is:

> Accuracy is rarely enough. The cost of a false negative pedestrian event is not the same as a false positive normal-driving event.

This article covers the core metrics you need for ML modeling interviews: precision, recall, F1, confusion matrix, ROC-AUC, PR-AUC, calibration, regression metrics, ranking metrics, and thresholding.

## 0. One-hour plan

```text
0-10 min   Confusion matrix and accuracy
10-25 min  Precision, recall, F1, false positives/negatives
25-35 min  Thresholds, ROC-AUC, PR-AUC
35-45 min  Calibration and probability quality
45-55 min  Regression/ranking metrics
55-60 min  Interview drills
```

---

## 1. Why you should care

Imagine a model predicts whether a pedestrian will enter the crosswalk.

If the dataset is:

```text
does not cross: 99%
crosses:         1%
```

A model that always predicts "does not cross" gets 99% accuracy. It is useless for safety.

Metrics should answer:

- What mistakes are we making?
- Which mistakes are expensive?
- Does the score threshold match the product need?
- Are predicted probabilities trustworthy?
- Does performance hold on rare slices?

In robotics, the same issue appears in grasp success, collision prediction, failure detection, and anomaly detection.

---

## 2. Confusion matrix

For binary classification:

| | Predicted positive | Predicted negative |
| --- | ---: | ---: |
| Actually positive | TP | FN |
| Actually negative | FP | TN |

Definitions:

- **TP:** predicted event, event happened.
- **FP:** predicted event, event did not happen.
- **FN:** missed event.
- **TN:** correctly predicted no event.

Driving example:

```text
positive = pedestrian crosses
negative = pedestrian does not cross
```

False negative:

```text
model says no crossing, pedestrian crosses
```

False positive:

```text
model says crossing, pedestrian waits
```

The false negative is often more safety-critical.

---

## 3. Accuracy

$$
Accuracy =
\frac{TP + TN}{TP + FP + FN + TN}
$$

Accuracy is useful when:

- classes are balanced,
- mistake costs are similar,
- the metric is only a rough sanity check.

Accuracy is dangerous when:

- classes are imbalanced,
- rare positives matter,
- false positives and false negatives have different costs.

Interview phrase:

> Accuracy can be high while the model is useless on the minority class.

---

## 4. Precision and recall

Precision:

$$
Precision = \frac{TP}{TP + FP}
$$

Question it answers:

> When the model predicts positive, how often is it right?

Recall:

$$
Recall = \frac{TP}{TP + FN}
$$

Question it answers:

> Of all real positives, how many did the model catch?

Driving interpretation:

- High precision: few false alarms.
- High recall: few missed events.

Tradeoff:

```text
lower threshold:
  more positives predicted
  recall up
  precision often down

higher threshold:
  fewer positives predicted
  precision up
  recall often down
```

For safety-critical detection, you often start by requiring high recall, then manage false positives.

---

## 5. F1 and F-beta

F1 is harmonic mean of precision and recall:

$$
F1 =
2 \cdot
\frac{Precision \cdot Recall}
{Precision + Recall}
$$

It is useful when you want one number that balances both.

If recall matters more than precision, use $F_\beta$:

$$
F_\beta =
(1+\beta^2)
\frac{Precision \cdot Recall}
\beta^2 Precision + Recall}
$$

If $\beta > 1$, recall matters more.

If $\beta < 1$, precision matters more.

Autonomous driving example:

- Pedestrian crossing detection may use high-recall operating points.
- Scenario mining may tolerate lower precision if humans or filters review candidates.
- Planner intervention prediction may need high precision to avoid noisy alarms.

---

## 6. Thresholds

Most classifiers output a score or probability:

$$
p = P(y=1|x)
$$

To make a binary decision:

$$
\hat{y} =
\mathbf{1}[p \ge \tau]
$$

where $\tau$ is threshold.

Changing $\tau$ changes precision and recall.

```text
threshold 0.1:
  many positives
  high recall
  low precision

threshold 0.9:
  few positives
  low recall
  high precision
```

Do not assume threshold 0.5 is correct. Pick thresholds based on validation metrics and product costs.

---

## 7. ROC-AUC and PR-AUC

ROC curve plots:

$$
TPR = \frac{TP}{TP + FN}
$$

against:

$$
FPR = \frac{FP}{FP + TN}
$$

ROC-AUC measures ranking quality across thresholds.

PR curve plots precision vs recall. PR-AUC is often better for imbalanced rare-event problems because it focuses on positive-class performance.

Interview answer:

> For rare events, I prefer PR-AUC over ROC-AUC because ROC-AUC can look strong when negatives dominate.

Example:

```text
rare cut-in detection:
  PR-AUC is more informative than accuracy
```

---

## 8. Calibration

A model is calibrated if predicted probabilities match observed frequencies.

If the model predicts 0.8 probability for 1,000 examples, about 800 should be positive.

Calibration matters when:

- downstream systems consume probabilities,
- thresholds are safety-critical,
- multiple model scores are compared,
- uncertainty matters.

Expected Calibration Error:

$$
ECE =
\sum_{b=1}^{B}
\frac{|S_b|}{n}
\left|
acc(S_b) - conf(S_b)
\right|
$$

where:

- $S_b$ is confidence bin $b$,
- $acc(S_b)$ is accuracy in that bin,
- $conf(S_b)$ is average confidence in that bin.

Driving example:

If a planner treats 0.9 crossing probability differently from 0.6, those probabilities must mean something.

---

## 9. Multi-class and multi-label

Multi-class:

```text
one label among many
example: maneuver = straight / left / right / stop
```

Multi-label:

```text
multiple labels can be true
example: scenario = rainy + night + pedestrian + occlusion
```

For multi-class:

- confusion matrix by class,
- macro F1,
- weighted F1,
- per-class precision/recall.

For multi-label:

- per-label precision/recall,
- micro average,
- macro average,
- label co-occurrence analysis.

Macro average treats each class equally. Micro average pools decisions and can be dominated by common classes.

---

## 10. Regression metrics

For trajectory or continuous prediction:

Mean absolute error:

$$
MAE =
\frac{1}{n}\sum_i |\hat{y}_i-y_i|
$$

Mean squared error:

$$
MSE =
\frac{1}{n}\sum_i(\hat{y}_i-y_i)^2
$$

Root mean squared error:

$$
RMSE = \sqrt{MSE}
$$

Trajectory metrics:

$$
ADE =
\frac{1}{T}
\sum_t
\|\hat{p}_t-p_t\|_2
$$

$$
FDE =
\|\hat{p}_T-p_T\|_2
$$

Tradeoff:

- MAE is more robust to outliers.
- MSE/RMSE penalize large errors more.
- ADE/FDE are intuitive but ignore map validity and multi-modality.

---

## 11. Ranking metrics

Many autonomous-driving systems rank candidates:

- top likely trajectories,
- risky scenarios,
- objects to review,
- retrieval results,
- mined rare events.

Useful metrics:

- Top-k accuracy.
- Recall@k.
- Precision@k.
- Mean reciprocal rank (MRR).
- NDCG.

Recall@k:

$$
Recall@k =
\frac{\text{relevant items in top k}}
{\text{total relevant items}}
$$

Scenario mining example:

If the review team can inspect only 100 scenarios, precision@100 matters.

---

## 12. Minimal PyTorch implementation

```python
import torch

def binary_metrics(scores, labels, threshold=0.5):
    pred = scores >= threshold
    labels = labels.bool()

    tp = (pred & labels).sum().item()
    fp = (pred & ~labels).sum().item()
    fn = (~pred & labels).sum().item()
    tn = (~pred & ~labels).sum().item()

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    accuracy = (tp + tn) / max(tp + fp + fn + tn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "accuracy": accuracy,
        "f1": f1,
    }
```

---

## 13. Common interview questions and strong answers

**Q: Why is accuracy bad for rare event detection?**  
A: Because negatives dominate. A model can get high accuracy by predicting the majority class and missing rare positives.

**Q: Precision or recall for pedestrian crossing?**  
A: I would usually prioritize recall because missing a crossing is safety-critical, then manage false positives through thresholding and downstream logic.

**Q: ROC-AUC or PR-AUC for rare events?**  
A: PR-AUC is usually more informative because it focuses on positive-class performance under imbalance.

**Q: Why does calibration matter?**  
A: If downstream planning uses probabilities, 0.9 should mean roughly 90% likelihood. Uncalibrated scores can lead to bad thresholds and risk decisions.

**Q: What is macro vs micro averaging?**  
A: Macro averages metrics per class equally. Micro pools all examples and can be dominated by common classes.

---

## 14. Failure modes and debugging checklist

- Accuracy high, minority recall near zero.
- ROC-AUC high, PR-AUC poor.
- Threshold chosen arbitrarily at 0.5.
- Model score ranks well but probabilities are uncalibrated.
- Aggregate metric hides geography/weather/agent-type failures.
- Regression metric improves while safety metrics worsen.
- Top-k metric good but top-1 behavior poor.

Checklist:

- Always inspect confusion matrix.
- Report per-class precision/recall.
- Sweep thresholds.
- Plot PR curve for rare events.
- Check calibration.
- Slice by scenario type.
- Match metric to product cost.

---

## 15. A 60-second explanation you can say out loud

Metrics define what model improvement means. Accuracy is often insufficient in autonomous driving because rare events matter. I start with a confusion matrix, then look at precision and recall. Precision tells me how reliable positive predictions are; recall tells me how many real positives I catch. For rare safety events, PR-AUC is often more useful than ROC-AUC. I also care about calibration if downstream systems use probabilities. For trajectories, ADE and FDE are useful but incomplete because they ignore multi-modality, collision, offroad, and physical validity. I would always slice metrics by scenario type and choose thresholds based on the cost of false positives and false negatives.

---

## 16. Practice exercises with answers

**Exercise 1:** A model has 99% accuracy on a 1% positive dataset. What metric do you ask for next?  
**Answer:** Positive-class recall and precision, plus PR-AUC.

**Exercise 2:** What happens to recall when you lower the threshold?  
**Answer:** Recall usually increases because more examples are predicted positive.

**Exercise 3:** Why can ROC-AUC be misleading for rare events?  
**Answer:** It includes true-negative behavior, and negatives dominate. PR-AUC focuses on positive-class retrieval.

**Exercise 4:** A model predicts 0.9 confidence but is correct 60% of the time in that bin. What is wrong?  
**Answer:** It is overconfident and poorly calibrated.

