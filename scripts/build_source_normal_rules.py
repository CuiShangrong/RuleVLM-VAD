#!/usr/bin/env python3
"""Build resumable, source-normal-only descriptions for 01, 08, or mixed."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "UR-DMU"))
from vlm_baselines.fair_unified import WeakShanghaiTrainData
from vlm_baselines.dense_zeroshot import context_bounds, temporal_centers
from vlm_baselines.common import uniform_indices_fixed_count
from build_domain01_normal_rules import MODEL, REVISION, SYSTEM, PROMPT, request_description


EXPECTED_NORMAL = {"01": 42, "08": 25, "mixed": 175}


def source_normal_ids(data: WeakShanghaiTrainData, source: str) -> list[str]:
    ids = list(data.train_normal) if source == "mixed" else [x for x in data.train_normal if x.startswith(source + "_")]
    if len(ids) != EXPECTED_NORMAL[source]:
        raise ValueError(f"expected {EXPECTED_NORMAL[source]} {source} normal videos, got {len(ids)}")
    return ids


def representative_centers(frame_count: int) -> list[int]:
    centers = temporal_centers(frame_count, 16)
    return [centers[round((i + 1) * (len(centers) - 1) / 3)] for i in range(2)]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--workspace", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--endpoint", required=True)
    p.add_argument("--source", choices=["01", "08", "mixed"], required=True)
    p.add_argument("--seed", type=int, default=2022)
    args = p.parse_args()
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    data = WeakShanghaiTrainData(args.workspace)
    normal_ids = source_normal_ids(data, args.source)
    (out / "rule_generation_prompt.txt").write_text(SYSTEM + "\n\n" + PROMPT)
    (out / "source_normal_video_ids.txt").write_text("\n".join(normal_ids) + "\n")
    raw_path = out / "raw_descriptions.jsonl"
    existing: dict[str, dict] = {}
    if raw_path.exists():
        for line in raw_path.read_text().splitlines():
            if line.strip():
                row = json.loads(line); existing[row["description_id"]] = row
    descriptions, rows, new_calls = [], [], 0
    for video_id in normal_ids:
        frames = data.weak_train_frames(video_id)
        for clip_index, center in enumerate(representative_centers(len(frames))):
            lo, hi = context_bounds(center, len(frames), 240)
            indices = uniform_indices_fixed_count(lo, hi, 8)
            description_id = f"D{args.source.replace('mixed', 'MIX')}N_{video_id}_C{clip_index:02d}"
            row = existing.get(description_id)
            if row is not None:
                if row.get("source_domain") != args.source or row.get("source_label") != "normal":
                    raise ValueError(f"cached description provenance mismatch: {description_id}")
            else:
                parsed, raw = request_description(args.endpoint, [frames[i] for i in indices], indices)
                row = {"description_id": description_id, "source_domain": args.source, "source_label": "normal", "source_split": "SH_Train.txt", "source_video_id": video_id, "clip_index": clip_index, "center_frame": center, "context_start": lo, "context_end_exclusive": hi, "sample_indices": indices, "sample_count": 8, "parsed_description": parsed, **raw}
                with raw_path.open("a") as handle:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                new_calls += 1
            rows.append(row)
            descriptions.append({"description_id": description_id, "source_domain": args.source, "source_label": "normal", "source_split": "SH_Train.txt", "source_video_id": video_id, "clip_index": clip_index, "center_frame": center, "context_start": lo, "context_end_exclusive": hi, "sample_indices": indices, "sample_count": 8, "description": row["parsed_description"], "generation_prompt_sha256": hashlib.sha256((SYSTEM + PROMPT).encode()).hexdigest()})
    (out / "normal_descriptions.jsonl").write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in descriptions))
    summary = {"model": MODEL, "revision": REVISION, "endpoint": args.endpoint, "seed": args.seed, "source_domain": args.source, "source_label": "normal", "source_split": "SH_Train.txt", "source_normal_videos": len(normal_ids), "clips_per_video": 2, "description_count": len(descriptions), "source_abnormal_videos_used": 0, "target_test_or_gt_used": False, "generation_prompt_sha256": hashlib.sha256((SYSTEM + PROMPT).encode()).hexdigest(), "vlm_calls": new_calls, "description_rows_total": len(rows), "sum_vlm_elapsed_seconds": sum(float(x.get("elapsed_seconds", 0)) for x in rows)}
    (out / "description_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
