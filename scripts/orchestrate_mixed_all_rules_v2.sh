#!/usr/bin/env bash
# Source-only Mixed rule generation is already running on gpu3.  This script
# continues only after it has completed, and invokes GT evaluation only after
# the inference writer has created prediction_lock.json.
set -euo pipefail

host_root=/raid/home/sy-sai/intern_study/direct_vlm/fair_tuned/rule_prompt_runs/v2_all_source_rules_full
root=/workspace/direct_vlm/fair_tuned/rule_prompt_runs/v2_all_source_rules_full
model=/workspace/vlm_service/qwen3_8_27b/cache/huggingface/hub/models--Qwen--Qwen3.8-27B/snapshots/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0
adapter=/workspace/direct_vlm/fair_tuned/fair_runs/peft_native_formal_v1/adapters/LoRA_mixed/selected_adapter

while [[ ! -f "$host_root/source_rules/domain_mixed/description_summary.json" ]]; do sleep 60; done

docker exec official-methods-qwen-gpu3 bash -lc "PYTHONPATH=/workspace/UR-DMU /workspace/vlm_service/qwen3_8_27b/env/bin/python /workspace/direct_vlm/fair_tuned/scripts/aggregate_source_normal_rules.py --description-dir $root/source_rules/domain_mixed --source mixed --endpoint http://127.0.0.1:8000/v1/chat/completions"

# The service runtime pid file is shared among service containers; select only
# the vLLM parent actually visible inside the gpu3 container.
docker exec official-methods-qwen-gpu3 bash -lc 'pid=$(ps -eo pid,args | awk "/[v]llm serve/{print \$1; exit}"); test -n "$pid"; kill -TERM "$pid"'
for _ in $(seq 1 60); do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 3 | tr -dc '0-9')
  [[ ${used:-999999} -lt 20000 ]] && break
  sleep 2
done

docker exec -w /workspace dense-lora-gpu3 bash -lc "
  set -euo pipefail
  mkdir -p $root/variants/mixed_to_mixed/logs $root/runtime_cache/gpu3/home $root/runtime_cache/gpu3/xdg $root/runtime_cache/gpu3/triton
  export HOME=$root/runtime_cache/gpu3/home XDG_CACHE_HOME=$root/runtime_cache/gpu3/xdg TRITON_CACHE_DIR=$root/runtime_cache/gpu3/triton PYTHONPATH=/workspace/direct_vlm/fair_tuned/vendor:/workspace/UR-DMU
  /workspace/vlm_service/qwen3_8_27b/env/bin/python /workspace/direct_vlm/fair_tuned/scripts/run_unified_lora_all_rules.py --workspace /workspace --run-root $root/variants/mixed_to_mixed --source mixed --target-name mixed --adapter $adapter --model-path $model --rules-jsonl $root/source_rules/domain_mixed/all_source_rules.jsonl > $root/variants/mixed_to_mixed/logs/inference_gpu3.log 2>&1
  /workspace/vlm_service/qwen3_8_27b/env/bin/python /workspace/direct_vlm/fair_tuned/scripts/finalize_all_source_rules_locked.py --workspace /workspace --output-dir $root/variants/mixed_to_mixed/targets/mixed > $root/variants/mixed_to_mixed/logs/evaluation_gpu3.log 2>&1
  /workspace/vlm_service/qwen3_8_27b/env/bin/python /workspace/direct_vlm/fair_tuned/scripts/summarize_all_source_rules_v2.py --workspace /workspace --run-root $root > $root/logs_results_summary.log 2>&1
"
