#!/usr/bin/env python3
"""Verify released locked prediction files against their recorded SHA256 hashes."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    locks = sorted(ROOT.glob("artifacts/**/prediction_lock.json"))
    checked = 0
    for lock_path in locks:
        lock = json.loads(lock_path.read_text())
        prediction = lock_path.parent / "frame_scores.npy"
        if not prediction.exists():
            raise FileNotFoundError(f"missing prediction for {lock_path}: {prediction}")
        actual = sha256(prediction)
        expected = lock["sha256"]
        if actual != expected:
            raise RuntimeError(f"SHA256 mismatch for {prediction}: {actual} != {expected}")
        checked += 1
    print(f"Release verification PASSED: {checked} locked prediction files verified")


if __name__ == "__main__":
    main()
