#!/usr/bin/env bash
set -euo pipefail
ROOT=/workspace/vlm_service/qwen3_8_27b
MODEL="$ROOT/cache/huggingface/hub/models--Qwen--Qwen3.8-27B/snapshots/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
SESSION=qwen3_8_27b_vllm
OLD_PID=$(cat "$ROOT/runtime/server.pid" 2>/dev/null || true)
if [ -n "$OLD_PID" ] && kill -0 "$OLD_PID" 2>/dev/null && [ "$(ps -o stat= -p "$OLD_PID" | cut -c1)" != Z ]; then echo "already running pid=$OLD_PID"; exit 0; fi
export HF_HOME="$ROOT/cache/huggingface"
export XDG_CACHE_HOME="$ROOT/cache/runtime"
export VLLM_CACHE_ROOT="$ROOT/cache/runtime/vllm"
export CUDA_VISIBLE_DEVICES=0
export VLLM_LOGGING_LEVEL=INFO
ARGS=("$ROOT/env/bin/vllm" serve "$MODEL" --served-model-name Qwen/Qwen3.8-27B --host 0.0.0.0 --port 8000 --api-key EMPTY --dtype bfloat16 --tensor-parallel-size 1 --max-model-len 32768 --gpu-memory-utilization 0.80 --limit-mm-per-prompt '{"image":8,"video":0}' --default-chat-template-kwargs '{"enable_thinking":false}')
printf '%q ' "${ARGS[@]}" > "$ROOT/runtime/start_command.txt";printf '\n' >> "$ROOT/runtime/start_command.txt"
if command -v tmux >/dev/null; then
 tmux new-session -d -s "$SESSION" "exec $(cat "$ROOT/runtime/start_command.txt") >>$ROOT/logs/server.log 2>&1"
 tmux list-panes -t "$SESSION" -F '#{pane_pid}' > "$ROOT/runtime/server.pid";echo "started tmux=$SESSION pid=$(cat "$ROOT/runtime/server.pid")"
else
 nohup "${ARGS[@]}" >>"$ROOT/logs/server.log" 2>&1 & echo $! > "$ROOT/runtime/server.pid";echo "started background pid=$(cat "$ROOT/runtime/server.pid")"
fi
echo "log=$ROOT/logs/server.log"
