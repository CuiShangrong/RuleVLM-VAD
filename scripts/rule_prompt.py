"""Source-rule provenance validation and deterministic prompt construction."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


RULE_HEADER = (
    "Normal-behavior reference rules below were derived only from normal source-domain surveillance clips. "
    "Use them as context for visible actions and interactions; do not treat camera layout, background, lighting, "
    "or a source/target scene difference alone as abnormal.\n"
)


def load_rules(path: str | Path, mode: str, *, expected_source: str = "01") -> list[dict]:
    if mode not in {"all", "general"}:
        raise ValueError(f"unsupported rule mode: {mode}")
    entries = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    if not entries:
        raise ValueError("empty source rule list")
    for entry in entries:
        if entry.get("source_domain") != expected_source or entry.get("source_label") != "normal":
            raise ValueError(f"non-{expected_source}-normal rule provenance: {entry.get('rule_id')}")
        if entry.get("source_split") != "SH_Train.txt" or not entry.get("rule"):
            raise ValueError(f"invalid source rule: {entry.get('rule_id')}")
    selected = entries if mode == "all" else [x for x in entries if x.get("category") == "domain_general"]
    if not selected:
        raise ValueError(f"no selected rules for mode={mode}")
    return selected


def build_rule_prompt(base_prompt: str, rules: list[dict]) -> str:
    numbered = "\n".join(f"[{x['rule_id']}] {x['rule']}" for x in rules)
    return f"{RULE_HEADER}Source rules:\n{numbered}\n\n{base_prompt}"


def rule_metadata(path: str | Path, mode: str, rules: list[dict], prompt: str) -> dict:
    source = Path(path)
    return {
        "rule_mode": mode,
        "rules_file": str(source),
        "rules_file_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "rule_count": len(rules),
        "rule_ids": [x["rule_id"] for x in rules],
        "prompt_characters": len(prompt),
        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        "source_only_validated": True,
    }
