#!/usr/bin/env bash
# Target-only Direct VLM zero-shot protocol.  Uses GPUs 1,2,3 only.
set -euo pipefail
MODE=${1:---full}
if [[ "$MODE" != '--smoke' && "$MODE" != '--full' ]]; then
  echo "Usage: bash $0 [--smoke|--full]" >&2; exit 2
fi
SCRIPT_DIR=$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" && pwd)
ROOT=$(cd "$SCRIPT_DIR/../../.." && pwd)
RUN_HOST="$ROOT/direct_vlm/zero shot/runs/zeroshot_10s_8frame"
RUN='/workspace/direct_vlm/zero shot/runs/zeroshot_10s_8frame'
WORKSPACE=/workspace
IMAGE=urdmu-gb200:ready
PY=/workspace/vlm_service/qwen3_8_27b/env/bin/python
VLLM=/workspace/vlm_service/qwen3_8_27b/env/bin/vllm
MODEL=Qwen/Qwen3.8-27B
REV=1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0
MODEL_PATH=/workspace/vlm_service/qwen3_8_27b/cache/huggingface/hub/models--Qwen--Qwen3.8-27B/snapshots/$REV
WINDOW_FRAMES=${WINDOW_FRAMES:-240}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.65}
mkdir -p "$RUN_HOST/launcher_logs" "$RUN_HOST/services"
names=(direct-zeroshot-qwen-gpu1 direct-zeroshot-qwen-gpu2 direct-zeroshot-qwen-gpu3)

stop_service() {
  local name=$1
  docker exec "$name" bash -lc '
    pidfile=/workspace/direct_vlm/zero\ shot/runs/zeroshot_10s_8frame/services/server.pid
    if [ -s "$pidfile" ]; then
      pid=$(cat "$pidfile")
      if kill -0 "$pid" 2>/dev/null; then kill -TERM "$pid" || true; fi
      for i in $(seq 1 30); do kill -0 "$pid" 2>/dev/null || break; sleep 1; done
      if kill -0 "$pid" 2>/dev/null; then kill -KILL "$pid" || true; fi
    fi
  ' >/dev/null 2>&1 || true
}
cleanup() { for name in "${names[@]}"; do stop_service "$name"; done; }
trap cleanup EXIT INT TERM

wait_health() {
  local name=$1
  for _ in $(seq 1 180); do
    docker exec "$name" curl -fsS -H 'Authorization: Bearer EMPTY' http://127.0.0.1:8000/health >/dev/null 2>&1 && return 0
    sleep 5
  done
  echo "Qwen service $name did not become healthy" >&2; return 1
}
start_service() {
  local gpu=$1
  local name="direct-zeroshot-qwen-gpu$gpu"
  local service='/workspace/direct_vlm/zero shot/runs/zeroshot_10s_8frame/services'
  if ! docker container inspect "$name" >/dev/null 2>&1; then
    docker run -d --name "$name" --gpus "device=$gpu" -v "$ROOT:/workspace" "$IMAGE" sleep infinity >/dev/null
  else
    docker start "$name" >/dev/null 2>&1 || true
  fi
  # A PID file, rather than pgrep, avoids accidentally matching this launcher.
  docker exec -e SERVICE="$service" -e GPU="$gpu" -e SERVICE_BIN="$VLLM" -e MODEL_PATH="$MODEL_PATH" -e MODEL="$MODEL" -e GPU_MEMORY_UTILIZATION="$GPU_MEMORY_UTILIZATION" "$name" bash -lc '
    mkdir -p "$SERVICE"
    pidfile="$SERVICE/server.pid"
    if ! curl -fsS -H "Authorization: Bearer EMPTY" http://127.0.0.1:8000/health >/dev/null 2>&1; then
      if [ -s "$pidfile" ]; then old=$(cat "$pidfile"); kill -TERM "$old" 2>/dev/null || true; fi
      nohup env CUDA_VISIBLE_DEVICES=0 HF_HOME=/workspace/vlm_service/qwen3_8_27b/cache/huggingface XDG_CACHE_HOME="$SERVICE/runtime_gpu$GPU" VLLM_CACHE_ROOT="$SERVICE/vllm_cache_gpu$GPU" "$SERVICE_BIN" serve "$MODEL_PATH" --served-model-name "$MODEL" --host 127.0.0.1 --port 8000 --api-key EMPTY --dtype bfloat16 --tensor-parallel-size 1 --max-model-len 32768 --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" --limit-mm-per-prompt "{\"image\":8,\"video\":0}" --default-chat-template-kwargs "{\"enable_thinking\":false}" >"$SERVICE/server_gpu$GPU.log" 2>&1 &
      echo $! > "$pidfile"
    fi
  '
  wait_health "$name"
}
run_target() {
  local name=$1 target=$2
  docker exec -w /workspace/UR-DMU "$name" "$PY" '/workspace/direct_vlm/zero shot/scripts/run_zeroshot_10s.py' --workspace "$WORKSPACE" --run-root "$RUN" --target "$target" --endpoint http://127.0.0.1:8000/v1/chat/completions --model "$MODEL" --revision "$REV" --window-frames "$WINDOW_FRAMES"
  # Evaluation is deliberately independent and uses the existing baseline
  # environment, which contains scikit-learn; Qwen's isolated vLLM env does not.
  docker exec -w /workspace/UR-DMU urdmu-shanghai python '/workspace/direct_vlm/zero shot/scripts/evaluate_zeroshot_locked.py' --workspace "$WORKSPACE" --output-dir "$RUN/targets/$target"
}
run_smoke() {
  local name=$1 target=$2 video=$3
  docker exec -w /workspace/UR-DMU "$name" "$PY" '/workspace/direct_vlm/zero shot/scripts/run_zeroshot_10s.py' --workspace "$WORKSPACE" --run-root "$RUN" --target "$target" --endpoint http://127.0.0.1:8000/v1/chat/completions --model "$MODEL" --revision "$REV" --window-frames "$WINDOW_FRAMES" --video-id "$video" --smoke
}

# Mapping is deliberately GPU1 -> domain01, GPU2 -> domain08, GPU3 -> Mixed.
start_service 1; start_service 2; start_service 3
# Smoke prediction uses no GT and independently validates exact per-video reconstruction.
run_smoke direct-zeroshot-qwen-gpu1 domain_01 01_0015
run_smoke direct-zeroshot-qwen-gpu2 domain_08 08_0058
run_smoke direct-zeroshot-qwen-gpu3 mixed 02_0161
if [[ "$MODE" == '--smoke' ]]; then
  echo 'Zero-shot Direct VLM sanity test completed; services on GPUs 1, 2, and 3 are stopping.'
  exit 0
fi
( run_target direct-zeroshot-qwen-gpu1 domain_01 ) >"$RUN_HOST/launcher_logs/gpu1_domain01.log" 2>&1 & p1=$!
( run_target direct-zeroshot-qwen-gpu2 domain_08 ) >"$RUN_HOST/launcher_logs/gpu2_domain08.log" 2>&1 & p2=$!
( run_target direct-zeroshot-qwen-gpu3 mixed ) >"$RUN_HOST/launcher_logs/gpu3_mixed.log" 2>&1 & p3=$!
wait "$p1" "$p2" "$p3"
docker exec -w /workspace direct-zeroshot-qwen-gpu1 "$PY" '/workspace/direct_vlm/zero shot/scripts/summarize_zeroshot_10s.py' --run-root "$RUN" | tee "$RUN_HOST/launcher_logs/summary.log"
echo 'Zero-shot Direct VLM completed; services on GPUs 1, 2, and 3 are stopping.'
