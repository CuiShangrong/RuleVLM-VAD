#!/usr/bin/env bash
set -euo pipefail
SETUP=${1:?Usage: run_direct_vlm.sh <01_to_01|01_to_08|08_to_08|08_to_01|mixed_to_mixed>}
SCRIPT_DIR=$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" && pwd)
ROOT=$(cd "$SCRIPT_DIR/../../.." && pwd)
RUN=/workspace/runs/vlm_baselines_shanghaitech
REV=1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0
BASE=vlm-qwen3-8-27b
PY=/workspace/vlm_service/qwen3_8_27b/env/bin/python
docker start "$BASE" >/dev/null 2>&1 || true
docker exec -w /workspace "$BASE" bash -lc '/workspace/vlm_service/qwen3_8_27b/scripts/start_server.sh' || true
docker exec "$BASE" curl -fsS -H 'Authorization: Bearer EMPTY' http://127.0.0.1:8000/health >/dev/null
docker exec -w /workspace/UR-DMU "$BASE" "$PY" tools/run_direct_vlm.py --workspace /workspace --run-root "$RUN" --setup "$SETUP" --endpoint http://127.0.0.1:8000/v1/chat/completions --model Qwen/Qwen3.8-27B --revision "$REV" --window-length "${WINDOW_LENGTH:-32}" --stride "${WINDOW_STRIDE:-32}" --sample-count "${SAMPLE_COUNT:-4}"
docker exec -w /workspace/UR-DMU urdmu-shanghai python tools/evaluate_vlm_baseline.py --workspace /workspace --output-dir "$RUN/direct/setups/$SETUP"
