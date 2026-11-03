# SANA with projected LLaVA conditioning

This fork replaces SANA's native Gemma text conditioning at GenEval inference time with LLaVA features projected into SANA's token-conditioning space.

For the original SANA installation, models, training, and inference documentation, see [README (base repo).md](README%20%28base%20repo%29.md).

## Projection workflow

### 1. Build and extract native SANA embeddings

The repository changes used to prepare the SANA/Gemma target embeddings are:

- [`tools/build_geneval_augmented_class_calculation.py`](tools/build_geneval_augmented_class_calculation.py): builds the GenEval-shaped calculation vocabulary and prompt banks.
- [`tools/export_exact_geneval_conditioning.py`](tools/export_exact_geneval_conditioning.py): extracts exact native SANA/Gemma token conditioning, including the CHI prompt, token IDs, masks, offsets, and active-token embeddings.

The evaluated 1,080-class setup can be reproduced with:

```bash
python tools/build_geneval_augmented_class_calculation.py \
  --target-classes 1080 \
  --output-dir output/text_embeddings/geneval_vocab1080

python tools/export_exact_geneval_conditioning.py \
  --metadata output/text_embeddings/geneval_vocab1080/evaluation_metadata_augmented_classes.jsonl \
  --output-dir output/text_embeddings/geneval_vocab1080 \
  --output-prefix geneval_vocab1080 \
  --compact
```

The main Kaggle input produced here is `geneval_vocab1080_native_exact.pt`.

### 2. Extract LLaVA embeddings and project on Kaggle

Run [`vlm-embeddings-for-image-generation (8).ipynb`](<vlm-embeddings-for-image-generation%20%288%29.ipynb>) on Kaggle. This notebook performs both remaining stages:

1. Extracts contextual LLaVA hidden states and aligns them with SANA/Gemma token positions using the exact CHI-prefixed prompts.
2. Fits the SEAM and polynomial-kernel projection, learns target token scale, and exports SANA-compatible conditioning.

Required Kaggle inputs:

- `geneval_vocab1080_native_exact.pt`
- `sana_llava_geneval_exact_pairs.pt`

Notebook outputs:

- `sana_llava_geneval_vocab1080_projector_token_scale.pt`: fitted projector and statistics.
- `sana_llava_geneval_vocab1080_projected_compact_token_scale.pt`: compact projected conditioning consumed during GenEval inference.

Generated `.pt` files are ignored by Git and must be downloaded or kept locally.

## Where projected conditioning enters SANA

| File | Change |
|---|---|
| [`scripts/inference_geneval.py`](scripts/inference_geneval.py) | Loads and validates projected conditioning, injects it in place of native text-encoder output, and preserves the projected attention mask. |
| [`scripts/bash_run_inference_metric_geneval.sh`](scripts/bash_run_inference_metric_geneval.sh) | Accepts and forwards `--projected_text_embeddings`. |
| [`scripts/infer_run_inference_geneval.sh`](scripts/infer_run_inference_geneval.sh) | Passes the projected bundle to each inference worker and supports prompt ranges. |
| [`scripts/infer_metric_run_inference_metric_geneval.sh`](scripts/infer_metric_run_inference_metric_geneval.sh) | Connects projected inference with GenEval scoring. |
| [`diffusion/utils/conditioning_debugger.py`](diffusion/utils/conditioning_debugger.py) | Compares native and projected conditioning through SANA's projection, attention, and denoiser stages. |
| [`tools/metrics/geneval/evaluation/evaluate_images.py`](tools/metrics/geneval/evaluation/evaluate_images.py) | Reports per-prompt-family GenEval results. |

Supporting projection utilities are in [`tools/build_sana_llava_geneval_projection.py`](tools/build_sana_llava_geneval_projection.py) and [`tools/compact_sana_conditioning.py`](tools/compact_sana_conditioning.py).

## GenEval setup and evaluation

Evaluation requires [GenEval](https://github.com/djghosh13/geneval). An adapted copy is included under [`tools/metrics/geneval/`](tools/metrics/geneval/), but its separate `geneval` Conda environment and detector dependencies are still required. Follow [`tools/metrics/geneval/geneval_env.md`](tools/metrics/geneval/geneval_env.md) to install the environment. The scoring launcher downloads detector weights into `output/pretrained_models/geneval/` when needed.

Run the full 553-prompt, four-images-per-prompt evaluation with:

```bash
conda activate geneval

bash scripts/bash_run_inference_metric_geneval.sh \
  configs/sana_config/1024ms/Sana_600M_img1024.yaml \
  hf://Efficient-Large-Model/Sana_600M_1024px/checkpoints/Sana_600M_1024px_MultiLing.pth \
  --np=1 \
  --step=20 \
  --sample_nums=553 \
  --img_nums_per_sample=4 \
  --add_label=_llava_vocab1080 \
  --projected_text_embeddings=sana_llava_geneval_vocab1080_projected_compact_token_scale.pt \
  --log_geneval=false
```

Use a unique `--add_label` for every experiment so existing images are not reused.

## Results

The full benchmark contains 553 prompts and 2,212 generated images.

| Metric | Vocab-1080 projection | Previous projection | Native SANA |
|---|---:|---:|---:|
| Correct images | **48.46%** | 47.38% | 65.19% |
| Correct prompts | **60.04%** | 64.38% | 78.30% |
| Counting | **65.00%** | 47.81% | 70.00% |
| Color attributes | **5.75%** | 25.75% | 40.00% |
| Colors | **85.37%** | 69.95% | 87.23% |
| Position | **14.25%** | 20.75% | 25.50% |
| Single object | **97.81%** | 82.81% | 99.38% |
| Two objects | **37.88%** | 45.71% | 78.28% |
| **Overall GenEval** | **0.51011** | 0.48796 | 0.66732 |

See [`docs/sana_projection_debug_report_short.md`](docs/sana_projection_debug_report_short.md) for debugging history, scale analysis, and detailed conclusions.
