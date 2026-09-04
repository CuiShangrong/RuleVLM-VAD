#!/usr/bin/env bash
set -euo pipefail
curl -fsS -H 'Authorization: Bearer EMPTY' http://127.0.0.1:8000/health >/dev/null
curl -fsS -H 'Authorization: Bearer EMPTY' http://127.0.0.1:8000/v1/models
echo
