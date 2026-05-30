---
title: "Accuracy and Numerics: Optimizing Models Without Silent Regressions"
description: "A staff-level guide to accuracy preservation during model optimization: numeric drift, saliency, kurtosis, activation outliers, layer sensitivity, verbosity regressions, and debugging workflows."
---

# Accuracy and Numerics: Optimizing Models Without Silent Regressions

Every serious model optimization project has the same hidden contract:

> Make the model cheaper or faster without changing the behavior users care about.

That contract is harder than it sounds. A model can keep the same headline benchmark score while becoming worse at code, longer-winded in chat, less calibrated in classification, more fragile on long context, or subtly unsafe in refusal behavior. A model can preserve perplexity and still regress in tool use. It can preserve MMLU and become more verbose, more repetitive, or more likely to collapse on rare tokens.

This article is about the engineering layer beneath pruning, quantization, distillation, low-rank compression, sparse kernels, and mixed precision. It covers how to reason about accuracy regressions, numerical sensitivity, activation and weight distributions, saliency, kurtosis, layer-wise debugging, and behavior drift.

The staff-level goal is not "run the benchmark." The goal is:

> Build a process that tells us where the model changed, why it changed, whether the change matters, and which mitigation is cheapest.

---

## 1. The Optimization Contract

An optimization is not successful because it reduces FLOPs, memory, or latency. It is successful only if it improves a system metric while staying inside a quality budget.

A useful optimization contract has four parts:

```text
Optimization proposal
    |
    +-- System target
    |      latency, throughput, memory, cost, power
    |
    +-- Quality budget
    |      max allowed regression by task slice
    |
    +-- Behavior budget
    |      verbosity, refusal rate, formatting, tool-use, safety
    |
    +-- Debug path
           layer sensitivity, numeric diffing, rollback, mitigation
```

Examples:

- "Reduce p95 latency by 25% with less than 0.3% absolute drop on routing accuracy."
- "Fit the model on one GPU with no statistically significant regression on production evals."
- "Improve tokens/sec/GPU by 40% while keeping average response length within 5%."
- "Quantize to W4A16 with less than 1 point drop on code evals and no increase in invalid JSON."

The "no increase in verbosity" clause is not cosmetic. For LLM serving, verbosity is cost. If an optimized model emits 20% more tokens, the decode path gets more expensive. A speedup on each token can disappear if the model now generates more tokens per request.

---

## 2. Accuracy Is Not One Number

Accuracy regression can show up in several layers.

| Regression type | Example | Common cause |
| --- | --- | --- |
| Task accuracy | More wrong answers | Lost capacity, bad pruning, quantization error |
| Calibration | Confidence no longer matches correctness | Logit scale shift, temperature drift |
| Ranking quality | Worse retrieval/reranking order | Embedding drift, score compression |
| Generation quality | More hallucination or repetition | Distribution shift in logits |
| Verbosity | Longer answers for same prompt | Decoding/logit entropy changes |
| Formatting | More invalid JSON/tool calls | Token-level probability shifts |
| Safety | More unsafe compliance or over-refusal | Distillation/filtering mismatch |
| Long-tail behavior | Rare tasks regress | Calibration data undercoverage |

For LLMs, the worst mistake is evaluating only aggregate benchmark score. Aggregate metrics hide slice regressions. A 0.2% overall drop may contain a 10% regression on one high-value domain.

Use a scorecard:

```text
Quality scorecard
-------------------------------------------------
General capability:      MMLU, GPQA, BBH, etc.
Product tasks:           real production evals
Format reliability:      JSON / tool schema validity
Safety:                  refusal and unsafe compliance
Length behavior:         output tokens, verbosity ratio
Calibration:             ECE, confidence, logit margins
Long context:            retrieval, needle, document QA
Cost:                    generated tokens per request
Latency:                 TTFT, ITL, p95/p99
```

The exact metrics depend on the product. The important point is that the optimization must be evaluated on the behavior it can change.

---

## 3. Numeric Drift: The Small Difference That Grows

Most optimization methods introduce numeric drift. Quantization rounds values. Pruning removes terms. Kernel changes alter reduction order. Mixed precision changes accumulation. Low-rank factorization approximates matrices. Distillation changes the training target.

For a linear layer:

$$
y = Wx
$$

An optimized model uses:

$$
\hat{y} = \hat{W}\hat{x}
$$

The output error is:

$$
\Delta y = \hat{W}\hat{x} - Wx
$$

Add and subtract $\hat{W}x$:

$$
\Delta y = \hat{W}(\hat{x} - x) + (\hat{W} - W)x
$$

This separates activation error from weight error. Across many layers, those errors can amplify:

$$
\|\Delta h_{l+1}\| \leq \|J_l\| \|\Delta h_l\| + \|\epsilon_l\|
$$

where $J_l$ is the layer's local Jacobian and $\epsilon_l$ is the new error introduced at layer $l$.

This is why "the average quantization error is tiny" is not enough. If the error lands in a sensitive direction, or if a later nonlinearity amplifies it, the final behavior can change.

---

## 4. Distribution Diagnostics

Before optimizing a model, inspect the weights and activations. Accuracy problems often announce themselves as distribution problems.

For a tensor $x$, track:

$$
\mu = \mathbb{E}[x]
$$

$$
\sigma^2 = \mathbb{E}[(x - \mu)^2]
$$

$$
\text{skewness} = \mathbb{E}\left[\left(\frac{x-\mu}{\sigma}\right)^3\right]
$$

$$
\text{kurtosis} = \mathbb{E}\left[\left(\frac{x-\mu}{\sigma}\right)^4\right]
$$

High kurtosis means heavy tails: most values are near the center, but rare values are very large. Heavy-tailed activations are dangerous for low precision because the quantization range must cover outliers, leaving fewer effective levels for normal values.

For each layer, log:

- Mean and variance.
- Min, max, p99, p99.9, p99.99.
- Fraction of zeros.
- Fraction of NaN/Inf.
- Kurtosis and skew.
- Channel-wise max values.
- Token-position-wise activation norms.
- Logit norm and entropy.
- Attention score range before softmax.

An ASCII shape of the problem:

```text
Nice distribution:

          ***
        *******
      ***********
    ***************
      ***********
        *******
          ***

Heavy-tailed distribution:

          ***
        *******
      ***********
    ***************
                     *
                              *
                                        *

Quantization must cover the far tail, so normal values get fewer useful bins.
```

This is why activation-aware methods exist. Papers such as **[LLM.int8](https://arxiv.org/abs/2208.07339)**, **[SmoothQuant](https://arxiv.org/abs/2211.10438)**, and **[AWQ](https://arxiv.org/abs/2306.00978)** are all, in different ways, responses to the fact that weights and activations do not behave like friendly Gaussian blobs.

---

## 5. Outliers Are Not Just Noise

Outliers are tempting to dismiss as numerical inconvenience. In LLMs, some outliers are functional.

LLM.int8 showed that large transformer models develop systematic large-magnitude activation features. The practical result was a mixed-precision scheme: most values can be handled in int8, but outlier dimensions are isolated and computed in higher precision.

The engineering lesson is:

> Do not clip or smooth outliers until you know whether they carry model behavior.

Outlier diagnostics:

- Which layers contain outliers?
- Are they weight outliers, activation outliers, or both?
- Are they channel-specific?
- Are they token-specific?
- Do they appear only on certain prompt types?
- Are they stable across calibration samples?
- Does ablating them damage perplexity, task accuracy, or attention patterns?

A simple ablation test:

```text
For each candidate outlier channel c:
  1. Run baseline model and record outputs.
  2. Clamp or zero channel c in one layer.
  3. Measure logit KL, perplexity delta, task delta.
  4. Rank channels by damage.
```

If a tiny set of channels causes large output changes, those channels need special handling: higher precision, excluded pruning, smoother scaling, or layer-specific treatment.

---

## 6. Saliency: What Matters If We Change It?

Saliency estimates how important a model component is. The component might be a scalar weight, row, column, head, MLP neuron, layer, activation channel, or token.

A first-order saliency approximation starts from:

$$
\Delta \mathcal{L} \approx g^T \Delta \theta
$$

where $g = \nabla_\theta \mathcal{L}$.

A second-order approximation is:

$$
\Delta \mathcal{L} \approx g^T \Delta \theta + \frac{1}{2}\Delta \theta^T H \Delta \theta
$$

Near a trained optimum, $g \approx 0$, so:

$$
\Delta \mathcal{L} \approx \frac{1}{2}\Delta \theta^T H \Delta \theta
$$

This is the same conceptual foundation behind Optimal Brain Damage, Optimal Brain Surgeon, GPTQ-style second-order quantization, and many pruning methods.

For a scalar weight $w_i$ set to zero:

$$
\Delta \mathcal{L}_i \approx \frac{1}{2} H_{ii} w_i^2
$$

For quantization, if $\hat{w}_i = w_i + \epsilon_i$, then:

$$
\Delta \mathcal{L} \approx \frac{1}{2}\epsilon^T H \epsilon
$$

The point is not that we compute the full Hessian. We usually cannot. The point is that **error direction matters**. A small numeric error in a sensitive direction can hurt more than a larger error in an insensitive direction.

Practical saliency signals:

- Weight magnitude.
- Activation magnitude.
- Weight times activation norm.
- Gradient times weight.
- Hessian diagonal approximation.
- Layer output reconstruction error.
- Logit KL after perturbation.
- Task metric delta after ablation.

For staff-level work, use saliency to prioritize experiments. Do not pretend it is ground truth.

---

## 7. Layer-Wise Sensitivity

Models are not uniformly sensitive. Some layers tolerate aggressive optimization. Others are fragile.

A layer-wise sensitivity study asks:

> If I apply the optimization only to this layer, how much does the model change?

Example workflow:

```text
Baseline model
      |
      +-- optimize layer 0 only -> eval delta
      +-- optimize layer 1 only -> eval delta
      +-- optimize layer 2 only -> eval delta
      ...
      +-- optimize layer L only -> eval delta
      |
      v
Sensitivity map
```

Measure several deltas:

- Perplexity delta.
- Logit KL:

$$
KL(p_{base}(\cdot|x) \| p_{opt}(\cdot|x))
$$

- Top-token agreement.
- Hidden-state cosine similarity:

$$
\cos(h, \hat{h}) = \frac{h \cdot \hat{h}}{\|h\|\|\hat{h}\|}
$$

- Task accuracy.
- Output length.
- Format validity.

The sensitivity map may look like:

```text
Layer:       0  1  2  3  4  5  6  7  8  9  ... 31
KL delta:   .  .  *  .  .  #  #  *  .  .      #
Risk:       L  L  M  L  L  H  H  M  L  L      H

. low sensitivity
* medium sensitivity
# high sensitivity
```

Use the map to assign policies:

- Fragile layers stay higher precision.
- Fragile layers get lower sparsity.
- Fragile modules use smaller quantization groups.
- Fragile heads/channels are excluded from pruning.
- Stable layers get more aggressive treatment.

Uniform optimization is easy. Layer-aware optimization is usually better.

---

## 8. Calibration Data Is a Numeric Dependency

Many optimization methods depend on calibration data:

- Activation quantization.
- SmoothQuant-style scaling.
- AWQ-style activation-aware weight protection.
- GPTQ-style reconstruction.
- Pruning saliency.
- Low-rank approximation with activation weighting.

The calibration set is not a detail. It defines what the optimizer thinks matters.

Bad calibration data causes predictable failures:

- No code prompts -> code regressions.
- No long-context prompts -> long-context regressions.
- No tool calls -> tool-use regressions.
- No safety prompts -> refusal drift.
- Short prompts only -> activation ranges too small for long prompts.
- English-only prompts -> multilingual regressions.

Calibration should be stratified:

```text
Calibration set
-------------------------------------------------
General chat                 20%
Code                         15%
Math/reasoning               15%
RAG / long context           15%
Tool use / structured output 10%
Safety / refusal             10%
Domain-specific traffic      15%
```

The exact mix should match production. For a coding assistant, code should dominate. For a support bot, product-specific conversations should dominate.

---

## 9. Verbosity and Generation Drift

Optimization can change generation behavior without obvious accuracy loss.

Track:

- Mean output tokens.
- p95 output tokens.
- Stop-reason distribution.
- Repetition rate.
- Invalid format rate.
- Refusal rate.
- Tool-call count.
- Entropy of next-token distribution.
- Average logit margin between top-1 and top-2 tokens.

Why verbosity changes:

- Logit scale changes after quantization or distillation.
- The optimized model becomes less confident and hedges.
- Stop-token probability decreases.
- Instruction-following behavior shifts.
- Decoding temperature/top-p interacts differently with the new logits.

For each prompt $x$, compare:

$$
\text{verbosity ratio} =
\frac{\text{tokens}_{opt}(x)}{\text{tokens}_{base}(x)}
$$

Then slice by prompt type. A global ratio of 1.03 may hide a 1.30 ratio on customer-support prompts.

Also compare stop-token rank:

```text
Prompt: "Return only valid JSON ..."

Base model:
  stop token rank after closing brace: 2
  invalid JSON rate: 0.4%

Optimized model:
  stop token rank after closing brace: 18
  invalid JSON rate: 4.8%
```

This is an accuracy regression even if benchmark score is unchanged.

---

## 10. Golden Numeric Diffing

For kernel changes, dtype changes, graph compilation, and inference engine migrations, build a numeric diff harness.

Run the same prompts through baseline and candidate models with deterministic decoding:

- Same tokenizer.
- Same prompt bytes.
- Same chat template.
- Greedy decoding or fixed seed.
- Same max tokens.
- Same stop conditions.

Collect:

- Layer input/output norms.
- Hidden-state cosine similarity.
- Per-layer max absolute error.
- Per-layer relative error:

$$
\frac{\|\hat{h}_l - h_l\|_2}{\|h_l\|_2 + \epsilon}
$$

- Logit KL.
- Top-k token overlap.
- First divergence token.

Debugging flow:

```text
Output changed
    |
    v
Did logits differ at token 1?
    |
    +-- No -> decoding / sampling / stop condition issue
    |
    +-- Yes
          |
          v
    Find first layer with large hidden-state diff
          |
          v
    Inspect that layer's dtype, scale, kernel, mask, layout
```

This is especially useful when the model "mostly works" but one class of prompts breaks.

---

## 11. Accuracy Debugging Playbook

When an optimization regresses quality, avoid random parameter sweeps. Localize.

### Step 1: Reproduce deterministically

Freeze:

- Model checkpoint.
- Tokenizer and template.
- Decoding settings.
- Runtime version.
- GPU type.
- Random seed.
- Prompt set.

If you cannot reproduce the regression, you cannot debug it.

### Step 2: Identify the regression slice

Ask:

- Which tasks failed?
- Which prompt lengths?
- Which languages?
- Which domains?
- Which output formats?
- Which safety categories?
- Which decoding settings?

### Step 3: Compare distributions

Look for:

- Activation range expansion.
- Saturation or clipping.
- NaN/Inf.
- Heavy-tailed channels.
- Logit norm changes.
- Entropy changes.
- Stop-token probability changes.

### Step 4: Run layer-wise sensitivity

Apply the optimization one module at a time. Find the fragile layers.

### Step 5: Apply targeted mitigation

Mitigations:

- Keep fragile layers higher precision.
- Increase bit width or group size only where needed.
- Protect salient channels.
- Use activation smoothing.
- Recalibrate on better data.
- Add recovery fine-tuning or distillation.
- Adjust decoding only if the model quality is otherwise intact.

### Step 6: Re-evaluate system value

Do not accept a mitigation that erases the optimization win. If protecting 5% of channels recovers quality with small cost, good. If protecting half the model is required, the optimization may not be worth it.

---

## 12. Common Failure Patterns

### The benchmark stayed flat, but users complain

Likely cause: benchmark does not cover product behavior. Add production evals, output length metrics, and format reliability.

### Perplexity is fine, but reasoning got worse

Likely cause: aggregate next-token loss is insensitive to multi-step task success. Add reasoning and task-level evals.

### Only long-context prompts regress

Likely cause: calibration undercovered long contexts, activation ranges grow with sequence length, or attention logits became numerically unstable.

### JSON validity dropped

Likely cause: small logit shifts around punctuation, quotes, braces, or stop tokens. Track constrained-decoding failures and token-level divergence.

### Model became verbose

Likely cause: logit scale, stop-token probability, or uncertainty changed. Measure output token ratio and entropy.

### Safety behavior changed

Likely cause: safety prompts underrepresented in calibration/recovery data, or compression damaged refusal-specific features. Evaluate both unsafe compliance and over-refusal.

---

## 13. Important Papers to Read

These are not all "accuracy debugging" papers, but they expose the numerical issues that show up during optimization.

1. **[LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale](https://arxiv.org/abs/2208.07339)** — Dettmers et al., 2022.  
   Read for outlier features and mixed-precision treatment of fragile dimensions.

2. **[SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models](https://arxiv.org/abs/2211.10438)** — Xiao et al., 2022 / ICML 2023.  
   Read for activation outliers and mathematically equivalent smoothing between activations and weights.

3. **[GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers](https://arxiv.org/abs/2210.17323)** — Frantar et al., 2022.  
   Read for Hessian-aware post-training optimization and layer-wise reconstruction.

4. **[AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration](https://arxiv.org/abs/2306.00978)** — Lin et al., 2023 / MLSys 2024.  
   Read for activation-aware saliency and protecting important weights.

5. **[SpQR: A Sparse-Quantized Representation for Near-Lossless LLM Weight Compression](https://arxiv.org/abs/2306.03078)** — Dettmers et al., 2023.  
   Read for handling outlier weights with a sparse-quantized representation.

6. **[QuIP: 2-Bit Quantization of Large Language Models With Guarantees](https://arxiv.org/abs/2307.13304)** — Chee et al., 2023.  
   Read for incoherence processing and why coordinate alignment affects low-bit accuracy.

7. **[Optimal Brain Damage](https://papers.nips.cc/paper/1989/hash/6c9882bbac1c7093bd25041881277658-Abstract.html)** and **[Optimal Brain Surgeon](https://authors.library.caltech.edu/54981/)**.  
   Read for the original saliency and second-order sensitivity framing.

---

## 14. The Staff Engineer Summary

Optimization changes models. The job is not to pretend it does not. The job is to bound, measure, localize, and mitigate the change.

The practical checklist:

- Define a quality budget before optimizing.
- Track behavior metrics, not only benchmark accuracy.
- Inspect weight and activation distributions.
- Treat high-kurtosis and outlier channels carefully.
- Run layer-wise sensitivity before applying uniform compression.
- Use calibration data that reflects production traffic.
- Compare hidden states, logits, entropy, output length, and stop-token behavior.
- Debug regressions by localization, not random sweeps.
- Keep fragile layers or channels in safer formats when the system win survives.

The interview answer:

> Any optimization that changes numerics needs an accuracy-debugging plan. I want distribution diagnostics, saliency estimates, layer-wise sensitivity, deterministic diffing, and product-slice evals before I trust a speedup.

