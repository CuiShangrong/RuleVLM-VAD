#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" && pwd)
ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)
for setup in 01_to_01 01_to_08 08_to_08 08_to_01 mixed_to_mixed; do "$ROOT/zero shot/scripts/run_direct_vlm.sh" "$setup"; done
for setup in 01_to_01 01_to_08 08_to_08 08_to_01 mixed_to_mixed; do "$ROOT/tuned/scripts/run_tuned_vlm_setup.sh" "$setup"; done
