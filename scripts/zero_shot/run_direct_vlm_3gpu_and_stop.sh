#!/usr/bin/env bash
# Explicit full launcher: three local Qwen services, then cleanup.
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" && pwd)
ROOT=$(cd "$SCRIPT_DIR/../../.." && pwd)
RUN=/workspace/runs/vlm_baselines_shanghaitech
RUN_HOST="$ROOT/runs/vlm_baselines_shanghaitech"
REV=1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0
MODEL=Qwen/Qwen3.8-27B
MODEL_PATH=/workspace/vlm_service/qwen3_8_27b/cache/huggingface/hub/models--Qwen--Qwen3.8-27B/snapshots/$REV
PY=/workspace/vlm_service/qwen3_8_27b/env/bin/python
BASE=vlm-qwen3-8-27b
IMAGE=urdmu-gb200:ready
mkdir -p "$RUN_HOST/launcher_logs/direct_3gpu" "$RUN_HOST/services"

cleanup() {
  docker exec -w /workspace "$BASE" bash -lc '/workspace/vlm_service/qwen3_8_27b/scripts/stop_server.sh' >/dev/null 2>&1 || true
  for name in direct-vlm-qwen-gpu1 direct-vlm-qwen-gpu2; do
    docker exec "$name" pkill -f '[v]llm serve' >/dev/null 2>&1 || true
    docker stop "$name" >/dev/null 2>&1 || true
  done
}
trap cleanup EXIT INT TERM

wait_health() {
  for _ in $(seq 1 180); do
    docker exec "$1" curl -fsS -H 'Authorization: Bearer EMPTY' http://127.0.0.1:8000/health >/dev/null 2>&1 && return 0
    sleep 5
  done
  echo "Qwen service $1 did not become healthy" >&2
  return 1
}
start_base() {
  docker start "$BASE" >/dev/null 2>&1 || true
  docker exec -w /workspace "$BASE" bash -lc '/workspace/vlm_service/qwen3_8_27b/scripts/start_server.sh' || true
  wait_health "$BASE"
}
start_extra() {
  local gpu=$1 name="direct-vlm-qwen-gpu$1" service="/workspace/runs/vlm_baselines_shanghaitech/services/direct_gpu$1"
  mkdir -p "$RUN_HOST/services/direct_gpu$gpu"
  if ! docker container inspect "$name" >/dev/null 2>&1; then
    docker run -d --name "$name" --gpus "device=$gpu" -v "$ROOT:/workspace" "$IMAGE" sleep infinity >/dev/null
  else
    docker start "$name" >/dev/null 2>&1 || true
  fi
  docker exec "$name" bash -lc "mkdir -p '$service'; if ! pgrep -f '[v]llm serve.*--port 8000' >/dev/null; then nohup env CUDA_VISIBLE_DEVICES=0 HF_HOME=/workspace/vlm_service/qwen3_8_27b/cache/huggingface XDG_CACHE_HOME='$service/runtime' VLLM_CACHE_ROOT='$service/vllm_cache' $PY vllm serve '$MODEL_PATH' --served-model-name '$MODEL' --host 0.0.0.0 --port 8000 --api-key EMPTY --dtype bfloat16 --tensor-parallel-size 1 --max-model-len 32768 --gpu-memory-utilization 0.80 --limit-mm-per-prompt '{\"image\":8,\"video\":0}' --default-chat-template-kwargs '{\"enable_thinking\":false}' >'$service/server.log' 2>&1 & echo \$! >'$service/server.pid'; fi"
  wait_health "$name"
}
run_setup() {
  docker exec -w /workspace/UR-DMU "$1" "$PY" tools/run_direct_vlm.py --workspace /workspace --run-root "$RUN" --setup "$2" --endpoint http://127.0.0.1:8000/v1/chat/completions --model "$MODEL" --revision "$REV" --window-length "${WINDOW_LENGTH:-32}" --stride "${WINDOW_STRIDE:-32}" --sample-count "${SAMPLE_COUNT:-4}"
  docker exec -w /workspace/UR-DMU urdmu-shanghai python tools/evaluate_vlm_baseline.py --workspace /workspace --output-dir "$RUN/direct/setups/$2"
}

start_base
start_extra 1
start_extra 2
( run_setup "$BASE" 01_to_01; run_setup "$BASE" 01_to_08 ) >"$RUN_HOST/launcher_logs/direct_3gpu/gpu0.log" 2>&1 & p0=$!
( run_setup direct-vlm-qwen-gpu1 08_to_08; run_setup direct-vlm-qwen-gpu1 08_to_01 ) >"$RUN_HOST/launcher_logs/direct_3gpu/gpu1.log" 2>&1 & p1=$!
( run_setup direct-vlm-qwen-gpu2 mixed_to_mixed ) >"$RUN_HOST/launcher_logs/direct_3gpu/gpu2.log" 2>&1 & p2=$!
wait "$p0" "$p1" "$p2"
docker exec -w /workspace/UR-DMU urdmu-shanghai python tools/summarize_vlm_baselines.py --workspace /workspace --run-root "$RUN" | tee "$RUN_HOST/launcher_logs/direct_3gpu/summary.log"
echo "Direct VLM finished; Qwen services are being stopped."
