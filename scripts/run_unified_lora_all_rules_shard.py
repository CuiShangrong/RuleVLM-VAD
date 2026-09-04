#!/usr/bin/env python3
"""GT-isolated, deterministic workload-balanced shard inference for Mixed."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(Path(__file__).resolve().parents[1] / "vendor"), str(ROOT / "UR-DMU")]
from anomaly_ruler.core import atomic_json, append_jsonl, pad_last, sha256_file
from anomaly_ruler.data import ShanghaiData
from vlm_baselines.dense_zeroshot import PROMPT, SYSTEM, context_bounds, interpolate_centers, temporal_centers
from vlm_baselines.common import uniform_indices_fixed_count
from vlm_baselines.fair_unified import UnifiedScorer, fair_config
from rule_prompt import build_rule_prompt, load_rules, rule_metadata

TARGET_DOMAIN = {"domain_01": "01", "domain_08": "08", "mixed": "mixed"}


def clean(rec: dict) -> dict:
    return {"video_id": rec["video_id"], "frame_count": rec["frame_count"], "prediction_length": rec["gt_end"] - rec["gt_start"]}


def plan(manifest: list[dict], shards: int) -> tuple[dict[str, int], list[int]]:
    """LPT bin packing, with deterministic video-ID ties, balances center work."""
    work = [(x["video_id"], len(temporal_centers(x["frame_count"], 16))) for x in manifest]
    bins: list[list[str]] = [[] for _ in range(shards)]; loads = [0] * shards
    for video_id, centers in sorted(work, key=lambda x: (-x[1], x[0])):
        shard = min(range(shards), key=lambda i: (loads[i], i)); bins[shard].append(video_id); loads[shard] += centers
    return {video_id: index for index, videos in enumerate(bins) for video_id in videos}, loads


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--workspace", required=True); p.add_argument("--run-root", required=True); p.add_argument("--adapter", required=True); p.add_argument("--model-path", required=True); p.add_argument("--rules-jsonl")
    p.add_argument("--shard-index", type=int, required=True); p.add_argument("--shard-count", type=int, required=True)
    p.add_argument("--source", choices=["01", "08", "mixed"], required=True)
    p.add_argument("--target-name", choices=sorted(TARGET_DOMAIN), required=True)
    p.add_argument("--rule-mode", choices=["none", "all", "general"], required=True)
    args = p.parse_args()
    if not 0 <= args.shard_index < args.shard_count: raise ValueError("invalid shard index")
    adapter = Path(args.adapter); train = json.loads((adapter.parent / "train_config.json").read_text()); complete = json.loads((adapter.parent / "training_complete.json").read_text())
    if train.get("source") != args.source or train.get("frame_masks_used") is not False or train.get("target_gt_used") is not False or complete.get("selected_checkpoint") != "selected_adapter": raise ValueError("not the selected source-only LoRA adapter")
    if args.rule_mode == "none":
        if args.rules_jsonl:
            raise ValueError("rule file must not be supplied for rule-mode=none")
        rules, prompt = [], PROMPT
        meta = {"rule_mode":"none", "rule_count":0, "rule_ids":[], "source_only_validated":True,
                "prompt_characters":len(prompt), "prompt_sha256":hashlib.sha256(prompt.encode()).hexdigest()}
    else:
        if not args.rules_jsonl:
            raise ValueError("--rules-jsonl is required when rules are enabled")
        rules = load_rules(args.rules_jsonl, args.rule_mode, expected_source=args.source)
        prompt = build_rule_prompt(PROMPT, rules)
        meta = rule_metadata(args.rules_jsonl, args.rule_mode, rules, prompt)
    data = ShanghaiData(args.workspace); full = data.target_manifest(TARGET_DOMAIN[args.target_name]); assignments, loads = plan(full, args.shard_count)
    selected = [x for x in full if assignments[x["video_id"]] == args.shard_index]
    out = Path(args.run_root) / "shards" / f"shard_{args.shard_index:02d}"; out.mkdir(parents=True, exist_ok=True)
    cfg = fair_config(native_resolution=True); scorer = UnifiedScorer(args.model_path, adapter_path=str(adapter), cfg=cfg, prompt=prompt, system_prompt=SYSTEM)
    def tokens(text: str) -> int:
        rendered = scorer.processor.apply_chat_template([{"role":"system","content":SYSTEM},{"role":"user","content":[{"type":"text","text":text}]}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
        return len(scorer.processor.tokenizer.encode(rendered, add_special_tokens=False))
    meta.update({"base_prompt_text_token_count":tokens(PROMPT),"actual_prompt_text_token_count":tokens(prompt),"rule_prompt_insert_position":"before_original_lora_direct_prompt"})
    target_domain = TARGET_DOMAIN[args.target_name]
    method = "Weakly-Supervised PEFT LoRA Direct VLM" if args.rule_mode == "none" else f"Weakly-Supervised PEFT LoRA Direct VLM + {args.rule_mode} {args.source} normal rules"
    config = {"method":method, "setup":f"{args.source}_to_{target_domain}", "source_domain":args.source, "target_domain":target_domain, "target_name":args.target_name, "source_data_used":True, "target_gt_used":False, "gt_read":False, "sharded_inference":True, "shard_index":args.shard_index, "shard_count":args.shard_count, "shard_center_loads":loads, "unified_transformers_pipeline":True, "unified_preprocessing":cfg, "system_prompt":SYSTEM, "prompt":prompt, "prompt_sha256":hashlib.sha256((SYSTEM+prompt).encode()).hexdigest(), "rule_prompt":meta, "sampling":{"context_frames":cfg["context_frames"],"nominal_fps":24,"nominal_context_seconds":cfg["context_frames"]/24,"center_stride_frames":cfg["center_stride"],"sample_count":8,"boundary_policy":"truncate_valid_context_and_repeat_only_image_slots"}, "scoring":{"forced_choices":["0","1"],"continuous_score":"sigmoid(logit_1-logit_0)","temperature":0,"max_new_tokens":1}, "recovery":{"method":"linear_interpolation_at_temporal_centers","extra_smoothing":"none"}, "seed":2022, "target_manifest":[clean(x) for x in selected], "lora_adapter":{"path":str(adapter),"adapter_model_sha256":sha256_file(adapter/"adapter_model.safetensors"),"adapter_config_sha256":sha256_file(adapter/"adapter_config.json"),"training_complete_sha256":sha256_file(adapter.parent/"training_complete.json"),"train_config_sha256":sha256_file(adapter.parent/"train_config.json"),"selected_source_val_mil_loss":complete["selected_val_mil_loss"]}}
    atomic_json(out / "inference_config.json", config); atomic_json(out / "shard_plan.json", {"shard_index":args.shard_index,"shard_count":args.shard_count,"center_loads":loads,"video_ids":[x["video_id"] for x in selected]})
    videos=[]; calls=0; elapsed=0.; started=time.monotonic()
    for rec in selected:
        vid=rec["video_id"]; expected=rec["gt_end"]-rec["gt_start"]; file=out/f"{vid}_frame_scores.npy"
        if file.exists():
            scores=np.load(file)
            if scores.shape==(expected,) and np.isfinite(scores).all() and ((scores>=0)&(scores<=1)).all(): videos.append({"video_id":vid,"raw_frame_count":rec["frame_count"],"prediction_length":len(scores),"resumed":True}); continue
        paths=data.test_frames(vid); centers=temporal_centers(len(paths),16); values=[]
        for center in centers:
            lo,hi=context_bounds(center,len(paths),cfg["context_frames"]); indices=uniform_indices_fixed_count(lo,hi,8); result=scorer.score([paths[i] for i in indices],indices)
            if not result["success"]: raise RuntimeError(f"{vid} center {center}: {result}")
            calls+=1; elapsed+=float(result["elapsed_seconds"]); row={"video_id":vid,"center_frame":center,"context_start":lo,"context_end_exclusive":hi,"sample_indices":indices,"sample_count":8,**result}; append_jsonl(out/"center_manifest.jsonl",row); append_jsonl(out/"raw_vlm_responses.jsonl",row); values.append(result["score"])
        scores=pad_last(interpolate_centers(centers,values,len(paths)),expected).astype(np.float32); np.save(file,scores); videos.append({"video_id":vid,"raw_frame_count":len(paths),"prediction_length":len(scores),"centers":len(centers),"resumed":False})
    atomic_json(out / "shard_manifest.json", {"shard_index":args.shard_index,"shard_count":args.shard_count,"center_loads":loads,"videos":videos,"target_manifest":[clean(x) for x in selected],"vlm_calls":calls,"sum_vlm_elapsed_seconds":elapsed,"wall_elapsed_seconds":time.monotonic()-started,"gt_read":False})
    print(out)


if __name__ == "__main__": main()
