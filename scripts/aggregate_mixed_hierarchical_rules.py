#!/usr/bin/env python3
"""Context-safe, source-only hierarchical aggregation for 350 Mixed descriptions."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from aggregate_domain01_normal_rules import MODEL, REVISION, AGG_SYSTEM, CLASS_SYSTEM, call

CHUNK_SIZE = 40


def rule_prompt(scope: str) -> str:
    return f'''The following {scope} are from Mixed normal training clips only. Do not add facts beyond them.
Deduplicate, merge, and abstract them into exactly 12 short independent normal-behavior rules. Each rule must express exactly one normality condition, contain at most 12 English words, not be a frame-level description or anomaly type, and preserve one or two supporting IDs only.
Return strict JSON only: {{"rules":[{{"rule_id":"R001","rule":"...","source_description_ids":["..."]}}]}}

Items:
'''


def main() -> None:
    p = argparse.ArgumentParser(); p.add_argument("--description-dir", required=True); p.add_argument("--endpoint", required=True)
    args = p.parse_args(); out = Path(args.description_dir)
    rows = [json.loads(x) for x in (out / "normal_descriptions.jsonl").read_text().splitlines() if x.strip()]
    if len(rows) != 350 or any(x.get("source_domain") != "mixed" or x.get("source_label") != "normal" or x.get("source_split") != "SH_Train.txt" for x in rows):
        raise ValueError("Mixed descriptions fail normal-only provenance")
    raw_batches, intermediate, provenance = [], [], {}
    batch_prompt = rule_prompt("descriptions")
    for batch_index, start in enumerate(range(0, len(rows), CHUNK_SIZE), 1):
        chunk = rows[start:start + CHUNK_SIZE]; known = {x["description_id"] for x in chunk}
        text = "\n".join(f"{x['description_id']}: {json.dumps(x['description'], ensure_ascii=False)}" for x in chunk)
        parsed, raw = call(args.endpoint, AGG_SYSTEM, batch_prompt + text, 4096)
        kept = []
        for index, item in enumerate(parsed.get("rules", []), 1):
            ids = [x for x in item.get("source_description_ids", []) if x in known]; rule = str(item.get("rule", "")).strip()
            if rule and ids:
                rid = f"B{batch_index:02d}_R{index:03d}"; kept.append({"rule_id": rid, "rule": rule, "source_description_ids": ids}); provenance[rid] = ids
        if not kept: raise RuntimeError(f"batch {batch_index} returned no source-backed rules")
        raw_batches.append({"batch_index": batch_index, "description_ids": sorted(known), "raw": raw, "intermediate_rules": kept})
        intermediate.extend(kept)
    (out / "rule_aggregation_batches.jsonl").write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in raw_batches))
    (out / "intermediate_source_rules.jsonl").write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in intermediate))
    final_prompt = rule_prompt("intermediate rules")
    (out / "rule_aggregation_prompt.txt").write_text(AGG_SYSTEM + "\n\n" + final_prompt)
    final_input = "\n".join(f"{x['rule_id']}: {x['rule']}" for x in intermediate)
    parsed, raw_final = call(args.endpoint, AGG_SYSTEM, final_prompt + final_input, 4096)
    (out / "raw_rule_aggregation_response.json").write_text(json.dumps(raw_final, ensure_ascii=False, indent=2) + "\n")
    known_intermediate = set(provenance); final = []
    for index, item in enumerate(parsed.get("rules", []), 1):
        chosen = [x for x in item.get("source_description_ids", []) if x in known_intermediate]; rule = str(item.get("rule", "")).strip()
        expanded = sorted({desc for rid in chosen for desc in provenance[rid]})
        if rule and chosen and expanded:
            final.append({"rule_id": f"R{index:03d}", "rule": rule, "source_description_ids": expanded, "source_intermediate_rule_ids": chosen, "source_domain": "mixed", "source_label": "normal", "source_split": "SH_Train.txt"})
    if not final: raise RuntimeError("final aggregation returned no source-backed rules")
    (out / "all_source_rules_preclassification.jsonl").write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in final))
    class_prompt = '''Classify every supplied rule exactly once.
domain_general: not dependent on a camera, background, road shape, area position, scene layout, fixed movement direction, or camera view; theoretically transferable.
scene_specific: depends on one or more of those scene properties.
Return strict JSON only: {"classifications":[{"rule_id":"R001","category":"domain_general or scene_specific","reason":"one short reason"}]}

Rules:
'''
    (out / "rule_classification_prompt.txt").write_text(CLASS_SYSTEM + "\n\n" + class_prompt)
    classified, raw_class = call(args.endpoint, CLASS_SYSTEM, class_prompt + "\n".join(f"{x['rule_id']}: {x['rule']}" for x in final), 900)
    (out / "raw_rule_classification_response.json").write_text(json.dumps(raw_class, ensure_ascii=False, indent=2) + "\n")
    labels = {x.get("rule_id"): x for x in classified.get("classifications", [])}; completed = []
    for row in final:
        item = labels.get(row["rule_id"], {}); category = item.get("category")
        if category not in {"domain_general", "scene_specific"}: raise RuntimeError(f"missing category for {row['rule_id']}")
        completed.append({**row, "category": category, "category_reason": str(item.get("reason", "")).strip(), "aggregation_prompt_sha256": hashlib.sha256((AGG_SYSTEM + final_prompt).encode()).hexdigest(), "classification_prompt_sha256": hashlib.sha256((CLASS_SYSTEM + class_prompt).encode()).hexdigest()})
    (out / "all_source_rules.jsonl").write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in completed))
    general = [x for x in completed if x["category"] == "domain_general"]; scene = [x for x in completed if x["category"] == "scene_specific"]
    (out / "general_rules_only.jsonl").write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in general))
    (out / "scene_specific_rules.jsonl").write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in scene))
    summary = {"model": MODEL, "revision": REVISION, "source_domain": "mixed", "source_label": "normal", "source_abnormal_videos_used": 0, "target_test_or_gt_used": False, "description_count": len(rows), "all_rule_count": len(completed), "domain_general_count": len(general), "scene_specific_count": len(scene), "aggregation_protocol": {"type": "fixed_order_hierarchical_context_safe", "chunk_size": CHUNK_SIZE, "batch_count": len(raw_batches), "intermediate_rule_count": len(intermediate)}, "vlm_calls": len(raw_batches) + 2, "sum_vlm_elapsed_seconds": sum(x["raw"]["elapsed_seconds"] for x in raw_batches) + raw_final["elapsed_seconds"] + raw_class["elapsed_seconds"]}
    (out / "rule_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__": main()
