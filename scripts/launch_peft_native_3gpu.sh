#!/usr/bin/env bash
# Launch the three source-only PEFT LoRA Direct-VLM runs in isolated outputs.
set -euo pipefail

workspace=${WORKSPACE:-/raid/home/sy-sai/intern_study}
run_root=${RUN_ROOT:-/workspace/direct_vlm/fair_tuned/fair_runs/peft_native_formal_v1}
epochs=${EPOCHS:-3}
model_path=${MODEL_PATH:-/workspace/vlm_service/qwen3_8_27b/cache/huggingface/hub/models--Qwen--Qwen3.8-27B/snapshots/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0}

launch() {
  local container=$1 source=$2 log_name=$3
  docker exec -d -u 1006:985 -w /workspace "$container" bash -lc "
    mkdir -p ${run_root}/logs ${run_root}/runtime_cache/${source}/home ${run_root}/runtime_cache/${source}/xdg ${run_root}/runtime_cache/${source}/triton
    exec env HOME=${run_root}/runtime_cache/${source}/home XDG_CACHE_HOME=${run_root}/runtime_cache/${source}/xdg TRITON_CACHE_DIR=${run_root}/runtime_cache/${source}/triton PYTHONPATH=/workspace/direct_vlm/fair_tuned/vendor:/workspace/UR-DMU /workspace/vlm_service/qwen3_8_27b/env/bin/python /workspace/direct_vlm/fair_tuned/scripts/train_peft_mil.py --workspace /workspace --run-root ${run_root} --source ${source} --model-path ${model_path} --epochs ${epochs} --lr 1e-4 --topk 2 --clips-per-video 8 --grad-accumulation 4 --native-resolution > ${run_root}/logs/${log_name}.log 2>&1
  "
}

launch "${GPU1_CONTAINER:-direct-zeroshot-qwen-gpu1}" 01 lora01
launch "${GPU2_CONTAINER:-dense-lora-gpu2}" 08 lora08
launch "${GPU3_CONTAINER:-dense-lora-gpu3}" mixed loramixed
echo "Launched PEFT native Direct-VLM source-only jobs under ${run_root}"
