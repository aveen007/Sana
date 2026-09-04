#!/bin/bash

# ===================== hyper =================
geneval=true

default_np=8    # number of GPUs to use
py=tools/metrics/geneval/evaluation/evaluate_images.py
default_sample_nums=553
report_to=wandb
default_log_suffix_label=''

# parser
img_path=$1
exp_names=$2
job_name=$(basename $(dirname "$img_path"))

for arg in "$@"
do
    case $arg in
        --np=*)
        np="${arg#*=}"
        shift
        ;;
        --sample_nums=*)
        sample_nums="${arg#*=}"
        shift
        ;;
        --suffix_label=*)
        suffix_label="${arg#*=}"
        shift
        ;;
        --log_geneval=*)
        log_geneval="${arg#*=}"
        shift
        ;;
        --tracker_project_name=*)
        tracker_project_name="${arg#*=}"
        shift
        ;;
        *)
        ;;
    esac
done

sample_nums=${sample_nums:-$default_sample_nums}
np=${np:-$default_np}

if ! [[ "$np" =~ ^[1-9][0-9]*$ ]]; then
  echo "--np must be a positive integer, got: $np" >&2
  exit 2
fi
samples_per_gpu=$((sample_nums / np))
log_suffix_label=${suffix_label:-$default_log_suffix_label}
log_geneval=${log_geneval:-true}
tracker_project_name=${tracker_project_name:-"t2i-evit-baseline"}
echo "sample_nums: $sample_nums"
echo "GPU count: $np"
echo "log_geneval: $log_geneval"
echo "wandb_project_name: $tracker_project_name"

mask2former_path=output/pretrained_models/geneval
if [ ! -d "$mask2former_path" ]; then
  echo "Model path does not exist. Running download_models.sh..."
  if ! bash tools/metrics/geneval/evaluation/download_models.sh $mask2former_path; then
    echo "Failed to download the GenEval detector weights." >&2
    exit 1
  fi
fi

cmd_template="python $py --img_path {img_path} --exp_name {exp_name} \
            --model-path  $mask2former_path \
            --report_to $report_to --name {job_name} --tracker_project_name $tracker_project_name"

wait_for_evaluation_jobs() {
  local job_failed=0
  local pid
  for pid in "$@"; do
    if ! wait "$pid"; then
      job_failed=1
    fi
  done
  return "$job_failed"
}

if [ "$geneval" = true ]; then
  # =============== compute GenEval from json ==================
  echo "==================== computing geneval ===================="

  if [[ "$exp_names" != *.txt ]]; then
    exp_name=$(basename "$exp_names")
    cmd="${cmd_template//\{img_path\}/$img_path}"
    cmd="${cmd//\{exp_name\}/$exp_name}"
    cmd="${cmd//\{job_name\}/$job_name}"
    cmd="${cmd//\{gpu_id\}/0}"
    evaluation_failed=0
    if ! eval CUDA_VISIBLE_DEVICES=0 $cmd >> "${img_path}/${exp_name}_geneval_result.txt" 2>&1; then
      evaluation_failed=1
    fi
    cat "${img_path}/${exp_name}_geneval_result.txt"
    if [ "$evaluation_failed" -ne 0 ]; then
      echo "GenEval scoring failed for: $exp_name" >&2
      exit 1
    fi
  else

    if [ ! -f "$exp_names" ]; then
      echo "Model paths file not found: $exp_names"
      exit 1
    fi

    gpu_id=0
    max_parallel_jobs=$np
    job_count=0
    evaluation_failed=0
    pids=()
    echo "" >> "$exp_names"   # add a new line to the file avoid skipping last line dir

    while IFS= read -r exp_name; do
      echo $exp_name
      if [ -n "$exp_name" ] && ! [[ $exp_name == \#* ]]; then
        exp_name=$(basename "$exp_name")
        cmd="${cmd_template//\{img_path\}/$img_path}"
        cmd="${cmd//\{exp_name\}/$exp_name}"
        cmd="${cmd//\{job_name\}/$job_name}"
        echo "Running on GPU $gpu_id: $cmd"
        eval CUDA_VISIBLE_DEVICES=$gpu_id $cmd >> "${img_path}/${exp_name}_geneval_result.txt" 2>&1 &
        pids+=("$!")

        gpu_id=$(( (gpu_id + 1) % np ))
        job_count=$((job_count + 1))

        if [ $job_count -ge $max_parallel_jobs ]; then
          if ! wait_for_evaluation_jobs "${pids[@]}"; then
            evaluation_failed=1
          fi
          pids=()
          job_count=0
        fi
      fi
    done < "$exp_names"
    if ! wait_for_evaluation_jobs "${pids[@]}"; then
      evaluation_failed=1
    fi
    # show the results
    while IFS= read -r exp_name; do
      if [ -n "$exp_name" ] && ! [[ $exp_name == \#* ]]; then
        cat "${img_path}/${exp_name}_geneval_result.txt"
      fi
    done < "$exp_names"
    if [ "$evaluation_failed" -ne 0 ]; then
      echo "One or more GenEval scoring jobs failed." >&2
      exit 1
    fi
  fi
fi

# =============== log GenEval result online after the above result saving ==================
if [ "$log_geneval" = true ] && [ "$geneval" = true ]; then
  echo "==================== logging onto $report_to ===================="

  if [ -n "${log_suffix_label}" ]; then
    echo "log_suffix_label: $log_suffix_label"
    cmd_template="${cmd_template} --suffix_label ${log_suffix_label}"
  fi

  if [[ "$exp_names" != *.txt ]]; then
    exp_name=$(basename "$exp_names")
    cmd="${cmd_template//\{img_path\}/$img_path}"
    cmd="${cmd//\{exp_name\}/$exp_name}"
    cmd="${cmd//\{job_name\}/$job_name}"
    echo $cmd
    eval $cmd --log_geneval
  else
    while IFS= read -r exp_name; do
      if [ -n "$exp_name" ] && ! [[ $exp_name == \#* ]]; then
        exp_name=$(basename "$exp_name")
        cmd="${cmd_template//\{img_path\}/$img_path}"
        cmd="${cmd//\{exp_name\}/$exp_name}"
        cmd="${cmd//\{job_name\}/$job_name}"
        eval $cmd --log_geneval
      fi
    done < "$exp_names"
    wait
  fi
fi

echo GenEval finally done
