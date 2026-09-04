#!/usr/bin/env python3
"""Describe reproducibly sampled Domain01 normal clips; never opens target data."""
from __future__ import annotations

import argparse, base64, hashlib, json, time, urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
import sys
sys.path.insert(0, str(ROOT / "UR-DMU"))
from vlm_baselines.fair_unified import WeakShanghaiTrainData
from vlm_baselines.dense_zeroshot import context_bounds, temporal_centers
from vlm_baselines.common import uniform_indices_fixed_count

MODEL = "Qwen/Qwen3.8-27B"
REVISION = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
SYSTEM = "You describe visible normal surveillance activity. Return JSON only; do not invent unseen events or anomalies."
PROMPT = """Eight chronological RGB frames come from a normal source-domain surveillance video.
Describe only observable normal content in the following fields. Keep each field concise and factual.
Return strict JSON only:
{"human_normal_behaviors":["..."],"normal_human_interactions":["..."],"normal_motion_patterns":["..."],"human_scene_relations":["..."],"stable_normal_activity_patterns":["..."],"evidence_frame_indices":[integers]}
Do not propose anomaly categories. Do not infer events that are not visible."""
SAMPLE_CLIP_COUNT = 2


def data_uri(path: Path) -> str:
    return "data:image/jpeg;base64," + base64.b64encode(path.read_bytes()).decode()


def parse_json(text: str) -> dict:
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            raise
        return json.loads(text[start:end + 1])


def request_description(endpoint: str, paths: list[Path], indices: list[int]) -> tuple[dict, dict]:
    content = [{"type": "text", "text": PROMPT}]
    for index, path in zip(indices, paths):
        content += [{"type": "text", "text": f"Original frame index: {index}"},
                    {"type": "image_url", "image_url": {"url": data_uri(path)}}]
    payload = {"model": MODEL, "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": content}],
               "temperature": 0, "max_tokens": 384, "chat_template_kwargs": {"enable_thinking": False}}
    started = time.monotonic()
    request = urllib.request.Request(endpoint, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", "Authorization": "Bearer EMPTY"})
    with urllib.request.urlopen(request, timeout=300) as response:
        body = json.loads(response.read())
    raw_text = body["choices"][0]["message"]["content"]
    return parse_json(raw_text), {"raw_text": raw_text, "response": body, "elapsed_seconds": time.monotonic() - started}


def clip_centers(frame_count: int) -> list[int]:
    centers = temporal_centers(frame_count, 16)
    # Deterministic temporal thirds: representative early and late activity,
    # with no target-aware selection or random retries.
    positions = [round((i + 1) * (len(centers) - 1) / (SAMPLE_CLIP_COUNT + 1)) for i in range(SAMPLE_CLIP_COUNT)]
    return [centers[int(position)] for position in positions]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", required=True); parser.add_argument("--output", required=True)
    parser.add_argument("--endpoint", required=True); parser.add_argument("--seed", type=int, default=2022)
    args = parser.parse_args(); out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    data = WeakShanghaiTrainData(args.workspace)
    normal_ids = [video_id for video_id in data.train_normal if video_id.startswith("01_")]
    if len(normal_ids) != 42:
        raise ValueError(f"expected 42 Domain01 normal training videos, got {len(normal_ids)}")
    (out / "rule_generation_prompt.txt").write_text(SYSTEM + "\n\n" + PROMPT)
    (out / "source_normal_video_ids.txt").write_text("\n".join(normal_ids) + "\n")
    raw_path, desc_path = out / "raw_descriptions.jsonl", out / "normal_descriptions.jsonl"
    existing = {json.loads(line)["description_id"]: json.loads(line) for line in raw_path.read_text().splitlines()} if raw_path.exists() else {}
    descriptions, raw_rows = [], []
    for video_id in normal_ids:
        frames = data.weak_train_frames(video_id)
        for clip_index, center in enumerate(clip_centers(len(frames))):
            lo, hi = context_bounds(center, len(frames), 240)
            indices = uniform_indices_fixed_count(lo, hi, 8)
            description_id = f"D01N_{video_id}_C{clip_index:02d}"
            if description_id in existing:
                row = existing[description_id]
            else:
                parsed, raw = request_description(args.endpoint, [frames[i] for i in indices], indices)
                row = {"description_id": description_id, "source_video_id": video_id, "clip_index": clip_index,
                       "center_frame": center, "context_start": lo, "context_end_exclusive": hi,
                       "sample_indices": indices, "sample_count": 8, "parsed_description": parsed, **raw}
                with raw_path.open("a") as handle: handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            raw_rows.append(row)
            descriptions.append({"description_id": description_id, "source_domain": "01", "source_label": "normal",
                "source_split": "SH_Train.txt", "source_video_id": video_id, "clip_index": clip_index,
                "center_frame": center, "context_start": lo, "context_end_exclusive": hi, "sample_indices": indices,
                "sample_count": 8, "description": row["parsed_description"],
                "generation_prompt_sha256": hashlib.sha256((SYSTEM + PROMPT).encode()).hexdigest()})
    desc_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in descriptions))
    summary = {"model": MODEL, "revision": REVISION, "endpoint": args.endpoint, "seed": args.seed,
               "source_domain": "01", "source_label": "normal", "source_split": "SH_Train.txt",
               "source_normal_videos": len(normal_ids), "clips_per_video": SAMPLE_CLIP_COUNT,
               "description_count": len(descriptions), "source_abnormal_videos_used": 0,
               "domain08_train_used": False, "target_test_or_gt_used": False,
               "generation_prompt_sha256": hashlib.sha256((SYSTEM + PROMPT).encode()).hexdigest(),
               "vlm_calls": len(raw_rows), "sum_vlm_elapsed_seconds": sum(float(x.get("elapsed_seconds", 0)) for x in raw_rows)}
    (out / "description_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__": main()
