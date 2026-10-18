# SANA Projection Debugging: Short Progress Report

## Objective

The goal is to replace SANA's native Gemma text conditioning with projected LLaVA features while preserving image-generation performance on GenEval.

## Work completed

### 1. Restored SANA's prompt context

The first correction was to add SANA's full `chi_prompt` context to the prompts used by the projection pipeline. This made the projected features use the same textual context and token structure as native SANA conditioning.

### 2. Validated the pipeline with exact official prompt pairs

As a sanity check, the projector was fitted directly on the official GenEval prompt pairs. The reconstruction fit was essentially perfect, and the resulting image evaluation matched the native model:

| Experiment | Correct images | Correct prompts | Overall GenEval |
|---|---:|---:|---:|
| Native SANA | 65.19% | 78.30% | **0.66732** |
| Exact official-pair sanity check | 66.32% | 79.75% | **0.67781** |

This established that the conditioning injection and evaluation paths were working. Because the official evaluation prompts were used for fitting, this experiment was only a pipeline sanity check and not a generalization result.

### 3. Expanded the projection calculation set

An alternate 80-class fitting set improved over the early projections, reaching an overall GenEval score of `0.08723`, but remained far below native SANA. Calculation sets with 1,000 and 2,000 classes were then tested. The 2,000-class variant produced the better held-out cosine result and was selected for further evaluation.

The initial full evaluation of the 2,000-class artifact scored only `0.00454`. This result was later found to be invalid as a measure of projection quality because the exported conditioning had the wrong magnitude.

## Main debugging result: missing embedding scale

To isolate the failure, evaluation was reduced to small, separate prompt-family slices of five prompts each. All families initially failed, including simple single-object prompts. Inspection of the saved artifact then revealed that every projected token vector had norm approximately `1.0`, while native SANA/Gemma token vectors had a median norm of approximately `113.15` and varied substantially by token.

Cosine similarity did not expose this error because cosine measures direction and ignores magnitude. The projected directions could therefore have high cosine similarity while entering SANA at roughly 1/113 of the expected scale.

For a diagnostic test, one global multiplier was applied to every projected value:

```text
scaled_embedding = projected_embedding * 113.150364844
```

This preserved every vector direction while restoring the typical native magnitude. It is a constant-scale diagnostic, not yet a learned per-token norm reconstruction.

## Results after constant-scale correction

Thirty prompts were tested: five prompts from each of the six GenEval families, with one image per prompt.

| Prompt family | Our method | Native full-benchmark reference |
|---|---:|---:|
| Single object | **5/5 (100%)** | 99.38% |
| Two objects | **4/5 (80%)** | 78.28% |
| Counting | **3/5 (60%)** | 70.00% |
| Colors | **5/5 (100%)** | 87.23% |
| Position | **1/5 (20%)** | 25.50% |
| Color attributes | **1/5 (20%)** | 40.00% |
| **Combined** | **19/30 (63.33%)** | **66.73% overall** |

The corrected 30-prompt result is only about one successful image below the native aggregate rate. The task pattern also resembles native SANA: object identity and simple colors are strong, while position and multi-object color binding are the most difficult. Because this diagnostic contains only five prompts per family, each image changes a family score by 20 percentage points; the result is promising but not a replacement for the full benchmark.

## Current conclusion and next steps

The catastrophic 2,000-class result was primarily caused by a scale/export error rather than unusable projected directions. Global scale restoration recovered performance from `0%` to `63.33%` on the 30-prompt diagnostic, close to the native `66.73%` reference.

The full constant-scale benchmark has now been completed. The remaining work is:

1. Replace the global constant with a norm model learned only from the calculation set, so each projected token receives an appropriate individual magnitude.
2. Re-run the full benchmark with per-token norm restoration and compare it directly with native SANA.

## Full GenEval evaluation

The complete evaluation used all 553 prompts and four images per prompt, for a total of 2,212 generated images.

| Metric | Our method | Native SANA baseline |
|---|---:|---:|
| Correct images | **47.38%** (1,048/2,212) | **65.19%** |
| Correct prompts | **64.38%** (356/553) | **78.30%** |
| Counting | **47.81%** (153/320) | **70.00%** |
| Color attributes | **25.75%** (103/400) | **40.00%** |
| Colors | **69.95%** (263/376) | **87.23%** |
| Position | **20.75%** (83/400) | **25.50%** |
| Single object | **82.81%** (265/320) | **99.38%** |
| Two objects | **45.71%** (181/396) | **78.28%** |
| **Overall GenEval** | **0.48796** | **0.66732** |

Next, I will learn a token-specific scale projection from the calculation set so that each projected direction receives an appropriate predicted magnitude instead of using one global x113 factor.

