#!/usr/bin/env bash
# Complete the four Direct-VLM 01/08 -> Mixed fills.  Every VLM phase runs
# the same target in three workload-balanced, mutually exclusive shards.
set -euo pipefail

root=/workspace/direct_vlm/fair_tuned/cross_mixed_fill/v1
# This launcher runs on the host while workers run in containers.  Keep the
# two path namespaces explicit; `/workspace` is intentionally only used in
# docker-exec commands below.
host_root=/raid/home/sy-sai/intern_study/direct_vlm/fair_tuned/cross_mixed_fill/v1
model=/workspace/vlm_service/qwen3_8_27b/cache/huggingface/hub/models--Qwen--Qwen3.8-27B/snapshots/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0
adapter_root=/workspace/direct_vlm/fair_tuned/fair_runs/peft_native_formal_v1/adapters
runner=/workspace/direct_vlm/fair_tuned/scripts/run_unified_lora_all_rules_shard.py
merger=/workspace/direct_vlm/fair_tuned/scripts/merge_unified_lora_all_rules_shards.py
evaluator=/workspace/direct_vlm/fair_tuned/scripts/finalize_all_source_rules_locked.py

run_phase() {
  local tag=$1 source=$2 mode=$3 rules=${4:-}
  local run=$root/$tag host_run=$host_root/$tag adapter=$adapter_root/LoRA_${source}/selected_adapter
  local rule_arg=''
  [[ $mode != none ]] && rule_arg="--rules-jsonl $rules"
  [[ -f $host_run/targets/mixed/evaluation.json ]] && { echo "$tag already evaluated"; return; }
  # A detached host scheduler can be interrupted while the three container
  # jobs keep running.  In that case do not launch duplicate VLM inference:
  # the GPU3 job already owns merge/evaluation after its shard completes.
  local active=0 container
  for container in dense-lora-rules-gpu1 dense-lora-gpu2 dense-lora-gpu3; do
    # `[r]un...` deliberately prevents pgrep from matching its own command
    # line.  The old literal pattern made an absent phase look active.
    if docker exec "$container" bash -lc "pgrep -f '[r]un_unified_lora_all_rules_shard.py.*--run-root $run' >/dev/null"; then
      active=$((active + 1))
    fi
  done
  if [[ $active -gt 0 ]]; then
    echo "$tag already has $active active shard job(s); waiting without relaunch"
    while [[ ! -f $host_run/targets/mixed/evaluation.json ]]; do sleep 120; done
    echo "completed $tag"
    return
  fi
  for spec in 'dense-lora-rules-gpu1 0 gpu1' 'dense-lora-gpu2 1 gpu2'; do
    set -- $spec; local container=$1 shard=$2 gpu=$3
    docker exec -d -w /workspace "$container" bash -lc "
      set -euo pipefail; mkdir -p $run/logs $root/runtime_cache/$tag/$gpu/home $root/runtime_cache/$tag/$gpu/xdg $root/runtime_cache/$tag/$gpu/triton
      export HOME=$root/runtime_cache/$tag/$gpu/home XDG_CACHE_HOME=$root/runtime_cache/$tag/$gpu/xdg TRITON_CACHE_DIR=$root/runtime_cache/$tag/$gpu/triton PYTHONPATH=/workspace/direct_vlm/fair_tuned/vendor:/workspace/UR-DMU
      /workspace/vlm_service/qwen3_8_27b/env/bin/python $runner --workspace /workspace --run-root $run --source $source --target-name mixed --rule-mode $mode --shard-index $shard --shard-count 3 --adapter $adapter --model-path $model $rule_arg > $run/logs/shard_${shard}_${gpu}.log 2>&1"
  done
  docker exec -d -w /workspace dense-lora-gpu3 bash -lc "
    set -euo pipefail; mkdir -p $run/logs $root/runtime_cache/$tag/gpu3/home $root/runtime_cache/$tag/gpu3/xdg $root/runtime_cache/$tag/gpu3/triton
    export HOME=$root/runtime_cache/$tag/gpu3/home XDG_CACHE_HOME=$root/runtime_cache/$tag/gpu3/xdg TRITON_CACHE_DIR=$root/runtime_cache/$tag/gpu3/triton PYTHONPATH=/workspace/direct_vlm/fair_tuned/vendor:/workspace/UR-DMU
    /workspace/vlm_service/qwen3_8_27b/env/bin/python $runner --workspace /workspace --run-root $run --source $source --target-name mixed --rule-mode $mode --shard-index 2 --shard-count 3 --adapter $adapter --model-path $model $rule_arg > $run/logs/shard_2_gpu3.log 2>&1
    while [[ ! -f $run/shards/shard_00/shard_manifest.json || ! -f $run/shards/shard_01/shard_manifest.json ]]; do sleep 60; done
    /workspace/vlm_service/qwen3_8_27b/env/bin/python $merger --workspace /workspace --run-root $run --target-name mixed --shard-count 3 > $run/logs/merge.log 2>&1
    /workspace/vlm_service/qwen3_8_27b/env/bin/python $evaluator --workspace /workspace --output-dir $run/targets/mixed > $run/logs/evaluation.log 2>&1"
  echo "started $tag"
  while [[ ! -f $host_run/targets/mixed/evaluation.json ]]; do sleep 120; done
  echo "completed $tag"
}

run_phase lora_01_to_mixed 01 none
run_phase lora_08_to_mixed 08 none
run_phase rules_01_to_mixed 01 all /workspace/direct_vlm/fair_tuned/rule_prompt_runs/v1_domain01_normal_rules/source_rules/all_source_rules.jsonl
run_phase rules_08_to_mixed 08 all /workspace/direct_vlm/fair_tuned/rule_prompt_runs/v2_all_source_rules_full/source_rules/domain_08/all_source_rules.jsonl
