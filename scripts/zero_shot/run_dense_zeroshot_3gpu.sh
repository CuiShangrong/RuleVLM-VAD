#!/usr/bin/env bash
set -euo pipefail
MODE=${1:---full}; [[ "$MODE" == --smoke || "$MODE" == --full ]] || { echo "Usage: $0 [--smoke|--full]"; exit 2; }
SCRIPT=$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" && pwd); ROOT=$(cd "$SCRIPT/../../.." && pwd)
RUN='/workspace/direct_vlm/zero shot/runs/dense_tokenprob_10s'; HOST_RUN="$ROOT/direct_vlm/zero shot/runs/dense_tokenprob_10s"
PY=/workspace/vlm_service/qwen3_8_27b/env/bin/python; VLLM=/workspace/vlm_service/qwen3_8_27b/env/bin/vllm; MODEL=Qwen/Qwen3.8-27B; REV=1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0
MODEL_PATH=/workspace/vlm_service/qwen3_8_27b/cache/huggingface/hub/models--Qwen--Qwen3.8-27B/snapshots/$REV; IMAGE=urdmu-gb200:ready; MEM=${GPU_MEMORY_UTILIZATION:-0.65}
mkdir -p "$HOST_RUN/launcher_logs" "$HOST_RUN/services"
health(){ docker exec "$1" curl -fsS -H 'Authorization: Bearer EMPTY' http://127.0.0.1:8000/health >/dev/null 2>&1; }
start(){ local g=$1; local n="direct-zeroshot-qwen-gpu$g"; docker start "$n" >/dev/null 2>&1 || true
  if health "$n"; then return; fi
  docker exec -e SVC='/workspace/direct_vlm/zero shot/runs/dense_tokenprob_10s/services' -e G="$g" -e BIN="$VLLM" -e MP="$MODEL_PATH" -e M="$MODEL" -e MEM="$MEM" "$n" bash -lc '
    mkdir -p "$SVC"; nohup env CUDA_VISIBLE_DEVICES=0 HF_HOME=/workspace/vlm_service/qwen3_8_27b/cache/huggingface XDG_CACHE_HOME="$SVC/runtime_gpu$G" VLLM_CACHE_ROOT="$SVC/vllm_cache_gpu$G" "$BIN" serve "$MP" --served-model-name "$M" --host 127.0.0.1 --port 8000 --api-key EMPTY --dtype bfloat16 --tensor-parallel-size 1 --max-model-len 32768 --gpu-memory-utilization "$MEM" --limit-mm-per-prompt "{\"image\":8,\"video\":0}" --default-chat-template-kwargs "{\"enable_thinking\":false}" >"$SVC/server_gpu$G.log" 2>&1 & echo $! >"$SVC/server_gpu$G.pid"'
  for i in $(seq 1 240); do health "$n" && return; sleep 5; done; echo "service gpu$g failed" >&2; return 1; }
run(){ local n=$1 t=$2; docker exec -w /workspace/UR-DMU "$n" "$PY" '/workspace/direct_vlm/zero shot/scripts/run_dense_zeroshot.py' --workspace /workspace --run-root "$RUN" --target "$t" --endpoint http://127.0.0.1:8000/v1/chat/completions --revision "$REV"; }
smoke(){ local n=$1 t=$2 v=$3; docker exec -w /workspace/UR-DMU "$n" "$PY" '/workspace/direct_vlm/zero shot/scripts/run_dense_zeroshot.py' --workspace /workspace --run-root "$RUN" --target "$t" --endpoint http://127.0.0.1:8000/v1/chat/completions --revision "$REV" --video-id "$v" --smoke; }
eval_locked(){ docker exec -w /workspace/UR-DMU urdmu-shanghai python '/workspace/direct_vlm/zero shot/scripts/evaluate_zeroshot_locked.py' --workspace /workspace --output-dir "$RUN/targets/$1"; }
start 1; start 2; start 3
smoke direct-zeroshot-qwen-gpu1 domain_01 01_0015; smoke direct-zeroshot-qwen-gpu2 domain_08 08_0058; smoke direct-zeroshot-qwen-gpu3 mixed 02_0161
[[ "$MODE" == --smoke ]] && exit 0
(run direct-zeroshot-qwen-gpu1 domain_01; eval_locked domain_01) >"$HOST_RUN/launcher_logs/gpu1_domain01.log" 2>&1 & a=$!
(run direct-zeroshot-qwen-gpu2 domain_08; eval_locked domain_08) >"$HOST_RUN/launcher_logs/gpu2_domain08.log" 2>&1 & b=$!
(run direct-zeroshot-qwen-gpu3 mixed; eval_locked mixed) >"$HOST_RUN/launcher_logs/gpu3_mixed.log" 2>&1 & c=$!
wait "$a" "$b" "$c"
docker exec direct-zeroshot-qwen-gpu1 "$PY" '/workspace/direct_vlm/zero shot/scripts/summarize_zeroshot_10s.py' --run-root "$RUN" >"$HOST_RUN/launcher_logs/summary.log"
