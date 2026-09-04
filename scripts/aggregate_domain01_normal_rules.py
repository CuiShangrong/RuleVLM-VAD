#!/usr/bin/env python3
"""Aggregate source-normal descriptions then classify rules, with full provenance."""
from __future__ import annotations

import argparse, hashlib, json, time, urllib.request
from pathlib import Path

MODEL = "Qwen/Qwen3.8-27B"
REVISION = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
AGG_SYSTEM = "You consolidate factual normal surveillance descriptions into transferable normal-behavior rules. Return JSON only."
AGG_PROMPT = """The following descriptions are from Domain01 normal training clips only. Do not add facts beyond them.
Deduplicate, merge, and abstract them into exactly 12 short independent normal-behavior rules. Each rule must express exactly one normality condition, contain at most 12 English words, not be a frame-level description or anomaly type, and preserve one or two supporting description IDs only.
Return strict JSON only: {"rules":[{"rule_id":"R001","rule":"...","source_description_ids":["..."]}]}\n\nDescriptions:\n"""
CLASS_SYSTEM = "You classify surveillance normal-behavior rules for cross-domain transfer. Return JSON only."
CLASS_PROMPT = """Classify every supplied rule exactly once.
domain_general: not dependent on a camera, background, road shape, area position, scene layout, fixed movement direction, or camera view; theoretically transferable.
scene_specific: depends on one or more of those scene properties.
Return strict JSON only: {"classifications":[{"rule_id":"R001","category":"domain_general or scene_specific","reason":"one short reason"}]}\n\nRules:\n"""


def parse_json(text: str) -> dict:
    try: return json.loads(text.strip())
    except json.JSONDecodeError:
        left, right = text.find("{"), text.rfind("}")
        if left < 0 or right <= left: raise
        return json.loads(text[left:right + 1])


def call(endpoint: str, system: str, prompt: str, max_tokens: int) -> tuple[dict, dict]:
    payload = {"model": MODEL, "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
               "temperature": 0, "max_tokens": max_tokens, "chat_template_kwargs": {"enable_thinking": False}}
    started = time.monotonic(); request = urllib.request.Request(endpoint, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", "Authorization": "Bearer EMPTY"})
    with urllib.request.urlopen(request, timeout=600) as response: body = json.loads(response.read())
    text = body["choices"][0]["message"]["content"]
    return parse_json(text), {"raw_text": text, "response": body, "elapsed_seconds": time.monotonic() - started}


def main() -> None:
    p = argparse.ArgumentParser(); p.add_argument("--description-dir", required=True); p.add_argument("--endpoint", required=True)
    args = p.parse_args(); out = Path(args.description_dir)
    rows = [json.loads(line) for line in (out / "normal_descriptions.jsonl").read_text().splitlines() if line.strip()]
    if len(rows) != 84 or any(x.get("source_domain") != "01" or x.get("source_label") != "normal" or x.get("source_split") != "SH_Train.txt" for x in rows):
        raise ValueError("descriptions do not meet Domain01-normal-only provenance")
    (out / "rule_aggregation_prompt.txt").write_text(AGG_SYSTEM + "\n\n" + AGG_PROMPT)
    descriptions_text = "\n".join(f"{x['description_id']}: {json.dumps(x['description'], ensure_ascii=False)}" for x in rows)
    aggregated, raw_aggregation = call(args.endpoint, AGG_SYSTEM, AGG_PROMPT + descriptions_text, 4096)
    (out / "raw_rule_aggregation_response.json").write_text(json.dumps(raw_aggregation, ensure_ascii=False, indent=2) + "\n")
    known = {x["description_id"] for x in rows}; all_rules = []
    for index, item in enumerate(aggregated.get("rules", []), 1):
        ids = [x for x in item.get("source_description_ids", []) if x in known]; rule = str(item.get("rule", "")).strip()
        if rule and ids:
            all_rules.append({"rule_id": f"R{index:03d}", "rule": rule, "source_description_ids": ids,
                "source_domain": "01", "source_label": "normal", "source_split": "SH_Train.txt"})
    if not all_rules: raise RuntimeError("aggregation returned no provenance-backed rules")
    (out / "all_source_rules_preclassification.jsonl").write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in all_rules))
    (out / "rule_classification_prompt.txt").write_text(CLASS_SYSTEM + "\n\n" + CLASS_PROMPT)
    rules_text = "\n".join(f"{x['rule_id']}: {x['rule']}" for x in all_rules)
    classified, raw_classification = call(args.endpoint, CLASS_SYSTEM, CLASS_PROMPT + rules_text, 900)
    (out / "raw_rule_classification_response.json").write_text(json.dumps(raw_classification, ensure_ascii=False, indent=2) + "\n")
    labels = {x.get("rule_id"): x for x in classified.get("classifications", [])}; final = []
    for row in all_rules:
        label = labels.get(row["rule_id"], {}); category = label.get("category")
        if category not in {"domain_general", "scene_specific"}: raise RuntimeError(f"missing category for {row['rule_id']}")
        final.append({**row, "category": category, "category_reason": str(label.get("reason", "")).strip(),
                      "aggregation_prompt_sha256": hashlib.sha256((AGG_SYSTEM+AGG_PROMPT).encode()).hexdigest(),
                      "classification_prompt_sha256": hashlib.sha256((CLASS_SYSTEM+CLASS_PROMPT).encode()).hexdigest()})
    (out / "all_source_rules.jsonl").write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in final))
    general = [x for x in final if x["category"] == "domain_general"]; scene = [x for x in final if x["category"] == "scene_specific"]
    (out / "general_rules_only.jsonl").write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in general))
    (out / "scene_specific_rules.jsonl").write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in scene))
    summary = {"model": MODEL, "revision": REVISION, "endpoint": args.endpoint, "description_count": len(rows),
               "all_rule_count": len(final), "domain_general_count": len(general), "scene_specific_count": len(scene),
               "source_domain": "01", "source_label": "normal", "source_abnormal_videos_used": 0,
               "domain08_train_used": False, "target_test_or_gt_used": False, "vlm_calls": 2,
               "sum_vlm_elapsed_seconds": float(raw_aggregation["elapsed_seconds"] + raw_classification["elapsed_seconds"])}
    (out / "rule_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__": main()
