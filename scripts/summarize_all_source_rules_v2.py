#!/usr/bin/env python3
"""Summarize locked All Source Rules results without accessing GT or predictions."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


SETUPS = ["01_to_01", "01_to_08", "08_to_08", "08_to_01", "mixed_to_mixed"]
TARGET = {"01_to_01": "domain_01", "01_to_08": "domain_08", "08_to_08": "domain_08", "08_to_01": "domain_01", "mixed_to_mixed": "mixed"}
SOURCE = {"01_to_01": "01", "01_to_08": "01", "08_to_08": "08", "08_to_01": "08", "mixed_to_mixed": "mixed"}


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def metrics(path: Path) -> dict:
    x = load(path)
    return {"frame_auroc": x["frame_auroc"], "frame_ap": x["frame_ap"],
            "macro_auroc": x.get("macro_auroc", x.get("macro_frame_auroc")),
            "macro_auc_valid_videos": x.get("macro_auc_valid_videos"),
            "prediction_frames": x.get("prediction_frames", x.get("frames")),
            "prediction_sha256": x.get("prediction_sha256")}


def main() -> None:
    p = argparse.ArgumentParser(); p.add_argument("--workspace", required=True); p.add_argument("--run-root", required=True)
    p.add_argument("--mixed-run-dir", help="optional completed parallel Mixed run directory")
    args = p.parse_args(); ws, root = Path(args.workspace), Path(args.run_root)
    old_rules_0108 = ws / "direct_vlm/fair_tuned/rule_prompt_runs/v1_domain01_normal_rules/variants/all_source_rules/targets/domain_08"
    old_lora = ws / "direct_vlm/fair_tuned/fair_runs/peft_native_formal_v1/inference"
    zero = ws / "direct_vlm/zero shot/runs/dense_tokenprob_10s/targets"
    ur = ws / "runs/urdmu_shanghaitech"
    new_rows, result_rows = {}, []
    for setup in SETUPS:
        target = TARGET[setup]
        if setup == "01_to_08": out = old_rules_0108
        elif setup == "mixed_to_mixed" and args.mixed_run_dir: out = Path(args.mixed_run_dir) / "targets" / target
        else: out = root / "variants" / setup / "targets" / target
        ev = metrics(out / "evaluation.json")
        lock = load(out / "prediction_lock.json")
        if lock["sha256"] != ev["prediction_sha256"]:
            raise ValueError(f"{setup}: evaluation prediction does not match lock")
        im = load(out / "inference_manifest.json")
        source = SOURCE[setup]
        if source == "01": rule_dir = ws / "direct_vlm/fair_tuned/rule_prompt_runs/v1_domain01_normal_rules/source_rules"
        else: rule_dir = root / "source_rules" / f"domain_{source}"
        rule = load(rule_dir / "rule_summary.json"); desc = load(rule_dir / "description_summary.json")
        row = {"method": "LoRA Direct VLM + All Source Rules", "setup": setup, **ev,
               "source_normal_videos": desc["source_normal_videos"], "source_rule_count": rule["all_rule_count"],
               "prompt_tokens": load(out / "inference_config.json")["rule_prompt"]["actual_prompt_text_token_count"],
               "temporal_centers": sum(int(x.get("centers", 0)) for x in im.get("videos", [])), "vlm_calls": im.get("vlm_calls"),
               "wall_elapsed_seconds": im.get("wall_elapsed_seconds"), "gpu": {"01_to_01": 1, "01_to_08": "reused", "08_to_08": 2, "08_to_01": 2, "mixed_to_mixed": 3}[setup],
               "prediction_path": str(out / "frame_scores.npy"), "rule_file": str(rule_dir / "all_source_rules.jsonl")}
        new_rows[setup] = row; result_rows.append(row)
        old = metrics(old_lora / setup / "targets" / target / "evaluation.json")
        result_rows.append({"method": "LoRA Direct VLM", "setup": setup, **old,
                            "delta_frame_auroc": 0.0, "delta_frame_ap": 0.0, "delta_macro_auroc": 0.0})
        row.update({"delta_frame_auroc": row["frame_auroc"] - old["frame_auroc"], "delta_frame_ap": row["frame_ap"] - old["frame_ap"], "delta_macro_auroc": row["macro_auroc"] - old["macro_auroc"]})
    for setup in SETUPS:
        target = TARGET[setup]
        z = metrics(zero / target / "evaluation.json")
        result_rows.append({"method": "Dense Direct Zero-shot Qwen", "setup": setup, **z})
        if setup == "mixed_to_mixed": u = metrics(ur / "metrics/evaluation.json")
        else: u = metrics(ur / "cross_domain" / setup / "seed_2022/evaluation.json")
        result_rows.append({"method": "UR-DMU WVAD", "setup": setup, **u})
    fields = ["method", "setup", "frame_auroc", "frame_ap", "macro_auroc", "macro_auc_valid_videos", "prediction_frames", "source_normal_videos", "source_rule_count", "prompt_tokens", "temporal_centers", "vlm_calls", "wall_elapsed_seconds", "gpu", "delta_frame_auroc", "delta_frame_ap", "delta_macro_auroc", "prediction_sha256", "prediction_path", "rule_file"]
    with (root / "results_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore"); writer.writeheader(); writer.writerows(result_rows)
    cross = {k: new_rows[k] for k in ["01_to_08", "08_to_01"]}
    summary = {"method": "Weakly-Supervised PEFT LoRA Direct VLM + All Source Rules", "results": new_rows,
               "cross_domain": {"01_to_08": cross["01_to_08"], "08_to_01": cross["08_to_01"],
                 "crossavg_frame_auroc": (cross["01_to_08"]["frame_auroc"] + cross["08_to_01"]["frame_auroc"]) / 2,
                 "crossavg_frame_ap": (cross["01_to_08"]["frame_ap"] + cross["08_to_01"]["frame_ap"]) / 2,
                 "crossavg_macro_auroc": (cross["01_to_08"]["macro_auroc"] + cross["08_to_01"]["macro_auroc"]) / 2},
               "reused_01_to_08": {"prediction_lock_verified": True, "path": str(old_rules_0108)}}
    (root / "results_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__": main()
