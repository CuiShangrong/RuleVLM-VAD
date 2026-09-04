#!/usr/bin/env bash
# GT remains unread until each independently written prediction lock exists.
set -euo pipefail
base=/workspace/direct_vlm/fair_tuned/rule_prompt_runs/v1_domain01_normal_rules/variants
for variant in all_source_rules general_rules_only; do
  out=$base/$variant/targets/domain_08
  while [[ ! -f $out/prediction_lock.json ]]; do sleep 60; done
  PYTHONPATH=/workspace/direct_vlm/fair_tuned/vendor:/workspace/UR-DMU \
    /workspace/vlm_service/qwen3_8_27b/env/bin/python \
    /workspace/direct_vlm/fair_tuned/scripts/evaluate_unified_lora_locked.py \
    --workspace /workspace --output-dir $out > $base/$variant/logs/evaluation.log 2>&1
done
