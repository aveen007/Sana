#!/bin/bash
set -euo pipefail

# Physical GPU to use on the server. Override with SANA_GPU_ID=<id>.
gpu_id="${SANA_GPU_ID:-1}"
prompt_index="${SANA_DEBUG_PROMPT_INDEX:-0}"
projector="${SANA_PROJECTOR:-sana_llava_token_aligned_nystrom_projector_bos_rightpad.pt}"
conditioning="${SANA_CONDITIONING:-output/text_embeddings/sana_llava_token_aligned_bos_rightpad_geneval_native.pt}"
report="${SANA_DEBUG_REPORT:-reports/conditioning_debug_prompt_$(printf '%05d' "$prompt_index")_bos_rightpad.json}"

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

echo "Debugging prompt $prompt_index on physical GPU $gpu_id; no images will be sampled..."
CUDA_VISIBLE_DEVICES="$gpu_id" DPM_TQDM=True python scripts/inference_geneval.py \
  --config="$config" \
  --model_path="$checkpoint" \
  --sampling_algo=flow_dpm-solver \
  --step=20 \
  --cfg_scale=4.5 \
  --sample_nums=1 \
  --n_samples=1 \
  --batch_size=1 \
  --gpu_id=0 \
  --start_index="$prompt_index" \
  --end_index="$((prompt_index + 1))" \
  --projected_text_embeddings="$conditioning" \
  --conditioning_debug=true \
  --conditioning_debug_only=true \
  --conditioning_debug_prompt_index="$prompt_index" \
  --conditioning_debug_output="$report" \
  --conditioning_debug_per_token=true \
  --conditioning_debug_top_n=20 \
  --conditioning_debug_worst_tokens=5

echo "Debug report: $report"
echo "Flattened comparison: ${report%.json}.csv"
