#!/usr/bin/env python3
"""Aggregate source-normal descriptions into provenance-backed all-source rules."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from aggregate_domain01_normal_rules import MODEL, REVISION, AGG_SYSTEM, CLASS_SYSTEM, call

EXPECTED_DESCRIPTIONS = {"01": 84, "08": 50, "mixed": 350}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--description-dir", required=True)
    p.add_argument("--endpoint", required=True)
    p.add_argument("--source", choices=["01", "08", "mixed"], required=True)
    args = p.parse_args(); out = Path(args.description_dir)
    rows = [json.loads(line) for line in (out / "normal_descriptions.jsonl").read_text().splitlines() if line.strip()]
    expected = EXPECTED_DESCRIPTIONS[args.source]
    if len(rows) != expected or any(x.get("source_domain") != args.source or x.get("source_label") != "normal" or x.get("source_split") != "SH_Train.txt" for x in rows):
        raise ValueError(f"descriptions do not meet {args.source}-normal-only provenance")
    agg_prompt = f'''The following descriptions are from {args.source} normal training clips only. Do not add facts beyond them.
Deduplicate, merge, and abstract them into exactly 12 short independent normal-behavior rules. Each rule must express exactly one normality condition, contain at most 12 English words, not be a frame-level description or anomaly type, and preserve one or two supporting description IDs only.
Return strict JSON only: {{"rules":[{{"rule_id":"R001","rule":"...","source_description_ids":["..."]}}]}}

Descriptions:
'''
    (out / "rule_aggregation_prompt.txt").write_text(AGG_SYSTEM + "\n\n" + agg_prompt)
    text = "\n".join(f"{x['description_id']}: {json.dumps(x['description'], ensure_ascii=False)}" for x in rows)
    aggregated, raw_agg = call(args.endpoint, AGG_SYSTEM, agg_prompt + text, 4096)
    (out / "raw_rule_aggregation_response.json").write_text(json.dumps(raw_agg, ensure_ascii=False, indent=2) + "\n")
    known = {x["description_id"] for x in rows}; rules = []
    for index, item in enumerate(aggregated.get("rules", []), 1):
        ids = [x for x in item.get("source_description_ids", []) if x in known]
        rule = str(item.get("rule", "")).strip()
        if rule and ids:
            rules.append({"rule_id": f"R{index:03d}", "rule": rule, "source_description_ids": ids, "source_domain": args.source, "source_label": "normal", "source_split": "SH_Train.txt"})
    if not rules:
        raise RuntimeError("aggregation returned no provenance-backed rules")
    (out / "all_source_rules_preclassification.jsonl").write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in rules))
    class_prompt = '''Classify every supplied rule exactly once.
domain_general: not dependent on a camera, background, road shape, area position, scene layout, fixed movement direction, or camera view; theoretically transferable.
scene_specific: depends on one or more of those scene properties.
Return strict JSON only: {"classifications":[{"rule_id":"R001","category":"domain_general or scene_specific","reason":"one short reason"}]}

Rules:
'''
    (out / "rule_classification_prompt.txt").write_text(CLASS_SYSTEM + "\n\n" + class_prompt)
    classified, raw_class = call(args.endpoint, CLASS_SYSTEM, class_prompt + "\n".join(f"{x['rule_id']}: {x['rule']}" for x in rules), 900)
    (out / "raw_rule_classification_response.json").write_text(json.dumps(raw_class, ensure_ascii=False, indent=2) + "\n")
    labels = {x.get("rule_id"): x for x in classified.get("classifications", [])}; final = []
    for row in rules:
        item = labels.get(row["rule_id"], {}); category = item.get("category")
        if category not in {"domain_general", "scene_specific"}:
            raise RuntimeError(f"missing category for {row['rule_id']}")
        final.append({**row, "category": category, "category_reason": str(item.get("reason", "")).strip(), "aggregation_prompt_sha256": hashlib.sha256((AGG_SYSTEM + agg_prompt).encode()).hexdigest(), "classification_prompt_sha256": hashlib.sha256((CLASS_SYSTEM + class_prompt).encode()).hexdigest()})
    (out / "all_source_rules.jsonl").write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in final))
    general = [x for x in final if x["category"] == "domain_general"]; scene = [x for x in final if x["category"] == "scene_specific"]
    (out / "general_rules_only.jsonl").write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in general))
    (out / "scene_specific_rules.jsonl").write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in scene))
    summary = {"model": MODEL, "revision": REVISION, "endpoint": args.endpoint, "description_count": len(rows), "all_rule_count": len(final), "domain_general_count": len(general), "scene_specific_count": len(scene), "source_domain": args.source, "source_label": "normal", "source_abnormal_videos_used": 0, "target_test_or_gt_used": False, "vlm_calls": 2, "sum_vlm_elapsed_seconds": float(raw_agg["elapsed_seconds"] + raw_class["elapsed_seconds"])}
    (out / "rule_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
