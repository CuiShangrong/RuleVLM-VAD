#!/usr/bin/env bash
set -euo pipefail
SESSION=qwen3_8_27b_vllm
ROOT=/workspace/vlm_service/qwen3_8_27b
if command -v tmux >/dev/null && tmux has-session -t "$SESSION" 2>/dev/null; then tmux kill-session -t "$SESSION"; echo "stopped $SESSION"
elif [ -f "$ROOT/runtime/server.pid" ] && kill -0 "$(cat "$ROOT/runtime/server.pid")" 2>/dev/null; then kill "$(cat "$ROOT/runtime/server.pid")"; echo "stopped pid=$(cat "$ROOT/runtime/server.pid")"
else echo "not running"; fi
