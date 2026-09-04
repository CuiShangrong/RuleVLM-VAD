#!/usr/bin/env python3
"""GT-isolated all-source-rule inference for each frozen fair LoRA adapter."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(Path(__file__).resolve().parents[1] / "vendor"), str(ROOT / "UR-DMU")]
from anomaly_ruler.core import sha256_file
from vlm_baselines.dense_zeroshot import PROMPT, SYSTEM, run_dense_target
from vlm_baselines.fair_unified import UnifiedScorer, fair_config
from rule_prompt import build_rule_prompt, load_rules, rule_metadata

TARGET_DOMAIN = {"domain_01": "01", "domain_08": "08", "mixed": "mixed"}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--workspace", required=True); p.add_argument("--run-root", required=True)
    p.add_argument("--adapter", required=True); p.add_argument("--model-path", required=True); p.add_argument("--rules-jsonl", required=True)
    p.add_argument("--source", choices=["01", "08", "mixed"], required=True)
    p.add_argument("--target-name", choices=sorted(TARGET_DOMAIN), required=True)
    args = p.parse_args(); target_domain = TARGET_DOMAIN[args.target_name]
    adapter = Path(args.adapter)
    train = json.loads((adapter.parent / "train_config.json").read_text())
    complete = json.loads((adapter.parent / "training_complete.json").read_text())
    if train.get("source") != args.source or train.get("frame_masks_used") is not False or train.get("target_gt_used") is not False or complete.get("selected_checkpoint") != "selected_adapter":
        raise ValueError("adapter is not the selected source-only frozen LoRA adapter")
    rules = load_rules(args.rules_jsonl, "all", expected_source=args.source)
    prompt = build_rule_prompt(PROMPT, rules)
    metadata = rule_metadata(args.rules_jsonl, "all", rules, prompt)
    cfg = fair_config(native_resolution=True)
    scorer = UnifiedScorer(args.model_path, adapter_path=str(adapter), cfg=cfg, prompt=prompt, system_prompt=SYSTEM)
    def chat_tokens(text: str) -> int:
        rendered = scorer.processor.apply_chat_template([{"role": "system", "content": SYSTEM}, {"role": "user", "content": [{"type": "text", "text": text}]}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
        return len(scorer.processor.tokenizer.encode(rendered, add_special_tokens=False))
    metadata.update({"base_prompt_text_token_count": chat_tokens(PROMPT), "actual_prompt_text_token_count": chat_tokens(prompt), "rule_prompt_insert_position": "before_original_lora_direct_prompt"})
    setup = f"{args.source}_to_{target_domain}"
    result = run_dense_target(args.workspace, args.run_root, args.target_name, scorer, context_frames=cfg["context_frames"], center_stride=cfg["center_stride"], extra_config={
        "method": "Weakly-Supervised PEFT LoRA Direct VLM + all source normal rules", "setup": setup, "source_domain": args.source, "source_data_used": True, "target_gt_used": False, "unified_transformers_pipeline": True, "unified_preprocessing": cfg, "system_prompt": SYSTEM, "prompt": prompt, "prompt_sha256": hashlib.sha256((SYSTEM + prompt).encode()).hexdigest(), "rule_prompt": metadata,
        "lora_adapter": {"path": str(adapter), "adapter_model_sha256": sha256_file(adapter / "adapter_model.safetensors"), "adapter_config_sha256": sha256_file(adapter / "adapter_config.json"), "training_complete_sha256": sha256_file(adapter.parent / "training_complete.json"), "train_config_sha256": sha256_file(adapter.parent / "train_config.json"), "selected_source_val_mil_loss": complete["selected_val_mil_loss"]},
    })
    print(result)


if __name__ == "__main__":
    main()
