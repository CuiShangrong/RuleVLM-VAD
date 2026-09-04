#!/usr/bin/env bash
# Native-resolution, locked-prediction LoRA Direct-VLM inference.  No GT is read.
set -euo pipefail

run_root=${RUN_ROOT:-/workspace/direct_vlm/fair_tuned/fair_runs/peft_native_formal_v1/inference}
model_path=${MODEL_PATH:-/workspace/vlm_service/qwen3_8_27b/cache/huggingface/hub/models--Qwen--Qwen3.8-27B/snapshots/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0}
adapter_root=${ADAPTER_ROOT:-/workspace/direct_vlm/fair_tuned/fair_runs/peft_native_formal_v1/adapters}

launch_sequence() {
  local container=$1 tag=$2 setup_a=$3 adapter_a=$4 setup_b=${5:-} adapter_b=${6:-}
  docker exec -d -u 1006:985 -w /workspace "$container" bash -lc "
    set -euo pipefail
    mkdir -p ${run_root}/logs ${run_root}/runtime_cache/${tag}/home ${run_root}/runtime_cache/${tag}/xdg ${run_root}/runtime_cache/${tag}/triton
    export HOME=${run_root}/runtime_cache/${tag}/home
    export XDG_CACHE_HOME=${run_root}/runtime_cache/${tag}/xdg
    export TRITON_CACHE_DIR=${run_root}/runtime_cache/${tag}/triton
    export PYTHONPATH=/workspace/direct_vlm/fair_tuned/vendor:/workspace/UR-DMU
    /workspace/vlm_service/qwen3_8_27b/env/bin/python /workspace/direct_vlm/fair_tuned/scripts/run_unified_lora.py --workspace /workspace --run-root ${run_root} --setup ${setup_a} --adapter ${adapter_a} --model-path ${model_path} > ${run_root}/logs/${setup_a}.log 2>&1
    if [ -n '${setup_b}' ]; then
      /workspace/vlm_service/qwen3_8_27b/env/bin/python /workspace/direct_vlm/fair_tuned/scripts/run_unified_lora.py --workspace /workspace --run-root ${run_root} --setup ${setup_b} --adapter ${adapter_b} --model-path ${model_path} > ${run_root}/logs/${setup_b}.log 2>&1
    fi
  "
}

launch_sequence "${GPU1_CONTAINER:-direct-zeroshot-qwen-gpu1}" gpu1 01_to_01 "${adapter_root}/LoRA_01/selected_adapter" 01_to_08 "${adapter_root}/LoRA_01/selected_adapter"
launch_sequence "${GPU2_CONTAINER:-dense-lora-gpu2}" gpu2 08_to_08 "${adapter_root}/LoRA_08/selected_adapter" 08_to_01 "${adapter_root}/LoRA_08/selected_adapter"
launch_sequence "${GPU3_CONTAINER:-dense-lora-gpu3}" gpu3 mixed_to_mixed "${adapter_root}/LoRA_mixed/selected_adapter"
echo "Launched GT-isolated LoRA inference under ${run_root}"
