# GenEval Reproduction Report

## Status

- Baseline evaluation: complete
- Candidate/custom-model evaluation: pending
- Evaluation date: 2026-09-05

## Headline result

| Run | GenEval overall | Difference from official Sana-0.6B |
| --- | ---: | ---: |
| Official Sana-0.6B checkpoint | 0.68000 | -- |
| This reproduction | **0.66732** | -0.01268 |
| Ovis-Image paper reference | 0.84000 | +0.17268 vs. this reproduction |
| Candidate/custom model | _TBD_ | _TBD_ |

The paper-comparable result from this run is **0.66732**, which can be reported as
**0.67**. The repository's [performance table](../README.md#performance) reports
**0.68** for the same downloadable Sana-0.6B checkpoint.

## Task breakdown

| Task | Score | Correct / total |
| --- | ---: | ---: |
| Single object | 99.38% | 318 / 320 |
| Two object | 78.28% | 310 / 396 |
| Counting | 70.00% | 224 / 320 |
| Colors | 87.23% | 328 / 376 |
| Position | 25.50% | 102 / 400 |
| Color attribution | 40.00% | 160 / 400 |
| **Overall (mean across tasks)** | **66.732%** | -- |

Additional diagnostics:

- Correct images: 65.19% across 2,212 images
- Correct prompts: 78.30% across 553 prompts

The headline GenEval score is the mean across the six task scores, not the
percentage of correct images.

## Evaluation configuration

| Setting | Value |
| --- | --- |
| Model | Sana-0.6B, 1024 px, multilingual checkpoint |
| Config | `configs/sana_config/1024ms/Sana_600M_img1024.yaml` |
| Checkpoint | `hf://Efficient-Large-Model/Sana_600M_1024px/checkpoints/Sana_600M_1024px_MultiLing.pth` |
| Prompts | Standard GenEval set, 553 prompts |
| Images per prompt | 4 |
| Total images | 2,212 |
| Resolution | 1024 x 1024 |
| Sampler | `flow_dpm-solver` |
| Sampling steps | 20 |
| CFG scale | 4.5 |
| Flow shift | 4.0 |
| Seed | 0 |
| Precision | float16 |
| Generation batch size | 1 |
| GPUs | 1 |
| Scoring runtime | 7m 28s |
| Generation runtime | Not recorded |
| Evaluation-code commit | `b521a02` |
| PyTorch | 2.7.1+cu126 |

## Commands

Generate all images and score them:

```bash
bash scripts/bash_run_inference_metric_geneval.sh \
  configs/sana_config/1024ms/Sana_600M_img1024.yaml \
  hf://Efficient-Large-Model/Sana_600M_1024px/checkpoints/Sana_600M_1024px_MultiLing.pth \
  --np=1 \
  --step=20 \
  --sample_nums=553 \
  --log_geneval=false
```

Score existing generated images without regenerating them:

```bash
bash scripts/bash_run_inference_metric_geneval.sh \
  configs/sana_config/1024ms/Sana_600M_img1024.yaml \
  hf://Efficient-Large-Model/Sana_600M_1024px/checkpoints/Sana_600M_1024px_MultiLing.pth \
  --np=1 \
  --inference=false \
  --geneval=true \
  --sample_nums=553 \
  --log_geneval=false
```

## Output location

The run directory is under:

```text
output/Sana_600M_1024px/vis/
  GenEval_epochunknown_stepunknown_scale4.5_step20_size1024_bs1_sampflow_dpm-solver_seed0_float16_flowshift4.0_imgnums553/
```

The detailed JSONL results and text summary use the same experiment name with
`_geneval.jsonl` and `_geneval_result.txt` suffixes.

## Candidate/custom-model results

Fill in this section after running the candidate model with the same GenEval
prompts and settings.

| Field | Value |
| --- | --- |
| Evaluation date | _TBD_ |
| Git commit | _TBD_ |
| Model/checkpoint | _TBD_ |
| Hardware | _TBD_ |
| Generation runtime | _TBD_ |
| Scoring runtime | _TBD_ |

| Metric | Sana-0.6B reproduction | Candidate/custom model | Difference |
| --- | ---: | ---: | ---: |
| Overall | 0.66732 | _TBD_ | _TBD_ |
| Single object | 0.9938 | _TBD_ | _TBD_ |
| Two object | 0.7828 | _TBD_ | _TBD_ |
| Counting | 0.7000 | _TBD_ | _TBD_ |
| Colors | 0.8723 | _TBD_ | _TBD_ |
| Position | 0.2550 | _TBD_ | _TBD_ |
| Color attribution | 0.4000 | _TBD_ | _TBD_ |

Notes:

- Keep the prompt set, four images per prompt, sampler, step count, CFG scale,
  resolution, seed, and scorer identical for a direct comparison.
- Record any intentional configuration differences here: _TBD_.
