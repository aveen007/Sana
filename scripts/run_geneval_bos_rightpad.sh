#!/bin/bash
set -euo pipefail

# Physical GPU to use on the server. Override with SANA_GPU_ID=<id>.
gpu_id="${SANA_GPU_ID:-1}"
projector="${SANA_PROJECTOR:-sana_llava_token_aligned_nystrom_projector_bos_rightpad.pt}"
conditioning="${SANA_CONDITIONING:-output/text_embeddings/sana_llava_token_aligned_bos_rightpad_geneval_native.pt}"

config="configs/sana_config/1024ms/Sana_600M_img1024.yaml"
checkpoint="hf://Efficient-Large-Model/Sana_600M_1024px/checkpoints/Sana_600M_1024px_MultiLing.pth"

if [ ! -f "$projector" ]; then
  echo "Missing corrected projector: $projector" >&2
  exit 1
fi

if [ ! -f "$conditioning" ]; then
  echo "Building corrected 553-prompt conditioning on physical GPU $gpu_id..."
  CUDA_VISIBLE_DEVICES="$gpu_id" python tools/build_sana_llava_geneval_projection.py \
    --projector "$projector" \
    --output "$conditioning" \
    --device cuda \
    --source-batch-size 2 \
    --batch-size 512
else
  echo "Using existing conditioning: $conditioning"
fi

echo "Running GenEval on physical GPU $gpu_id..."
CUDA_VISIBLE_DEVICES="$gpu_id" bash scripts/bash_run_inference_metric_geneval.sh \
  "$config" \
  "$checkpoint" \
  --np=1 \
  --step=20 \
  --sample_nums=553 \
  --add_label=_token_aligned_bos_rightpad \
  --projected_text_embeddings="$conditioning" \
  --log_geneval=false
