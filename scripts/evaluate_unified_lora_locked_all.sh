#!/usr/bin/env bash
# Run only after all five prediction_lock.json files exist and their hashes verify.
set -euo pipefail

run_root=${RUN_ROOT:-/workspace/direct_vlm/fair_tuned/fair_runs/peft_native_formal_v1/inference}

evaluate() {
  local setup=$1 target=$2
  python /workspace/direct_vlm/fair_tuned/scripts/evaluate_unified_lora_locked.py --workspace /workspace --output-dir "${run_root}/${setup}/targets/${target}"
}

evaluate 01_to_01 domain_01
evaluate 01_to_08 domain_08
evaluate 08_to_08 domain_08
evaluate 08_to_01 domain_01
evaluate mixed_to_mixed mixed
