#!/usr/bin/env python3
"""Run the independent evaluator only after an immutable prediction lock exists."""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--workspace", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--poll-seconds", type=int, default=60)
    args = p.parse_args(); out = Path(args.output_dir)
    lock = out / "prediction_lock.json"
    while not lock.exists():
        time.sleep(args.poll_seconds)
    evaluator = Path(__file__).with_name("evaluate_unified_lora_locked.py")
    result = subprocess.run([sys.executable, str(evaluator), "--workspace", args.workspace, "--output-dir", str(out)], check=True, text=True, capture_output=True)
    (out / "evaluation_stdout.log").write_text(result.stdout)
    print(result.stdout, end="")


if __name__ == "__main__":
    main()
