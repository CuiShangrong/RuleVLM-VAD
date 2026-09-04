#!/usr/bin/env python3
"""Merge completed GT-isolated Mixed shards into one canonical locked prediction."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "UR-DMU"))
from anomaly_ruler.core import atomic_json, sha256_file
from anomaly_ruler.data import ShanghaiData


def clean(rec: dict) -> dict:
    return {"video_id": rec["video_id"], "frame_count": rec["frame_count"], "prediction_length": rec["gt_end"] - rec["gt_start"]}


def main() -> None:
    p = argparse.ArgumentParser(); p.add_argument("--workspace", required=True); p.add_argument("--run-root", required=True); p.add_argument("--shard-count", type=int, default=3)
    p.add_argument("--target-name", choices=["domain_01", "domain_08", "mixed"], required=True)
    args = p.parse_args(); root = Path(args.run_root); shards = root / "shards"; data = ShanghaiData(args.workspace); target_domain = {"domain_01":"01","domain_08":"08","mixed":"mixed"}[args.target_name]; full = data.target_manifest(target_domain)
    manifests=[]; configs=[]
    for index in range(args.shard_count):
        directory=shards/f"shard_{index:02d}"; manifest=directory/"shard_manifest.json"; config=directory/"inference_config.json"
        if not manifest.exists() or not config.exists(): raise FileNotFoundError(f"incomplete shard {index}")
        manifests.append((directory,json.loads(manifest.read_text()))); configs.append(json.loads(config.read_text()))
    if any(x.get("gt_read") is not False or x.get("target_gt_used") is not False for x in configs): raise ValueError("a shard was not GT-isolated")
    reference=configs[0]
    for x in configs[1:]:
        for key in ["prompt_sha256","rule_prompt","lora_adapter","unified_preprocessing","sampling","scoring","recovery"]:
            if x.get(key)!=reference.get(key): raise ValueError(f"shard configuration mismatch: {key}")
    owners={}
    for directory, manifest in manifests:
        for video in manifest["videos"]:
            vid=video["video_id"]
            if vid in owners: raise ValueError(f"duplicate video {vid}")
            owners[vid]=(directory,video)
    expected_ids=[x["video_id"] for x in full]
    if set(owners)!=set(expected_ids): raise ValueError(f"shards do not cover canonical test set: missing={len(set(expected_ids)-set(owners))}")
    target=root/"targets"/args.target_name; target.mkdir(parents=True,exist_ok=True); scores=[]; videos=[]
    for rec in full:
        vid=rec["video_id"]; directory,video=owners[vid]; length=rec["gt_end"]-rec["gt_start"]; array=np.load(directory/f"{vid}_frame_scores.npy")
        if array.shape!=(length,) or not np.isfinite(array).all() or not ((array>=0)&(array<=1)).all(): raise ValueError(f"invalid {vid} shard score")
        scores.append(array.astype(np.float32)); videos.append({**video,"shard":json.loads((directory/"shard_manifest.json").read_text())["shard_index"]})
    combined=np.concatenate(scores); out=target/"frame_scores.npy"; np.save(out,combined)
    config={k:v for k,v in reference.items() if k not in {"shard_index","shard_count","shard_center_loads"}}
    config.update({"sharded_inference":True,"shard_count":args.shard_count,"target_name":args.target_name,"target_domain":target_domain,"target_manifest":[clean(x) for x in full],"gt_read":False,"target_gt_used":False})
    atomic_json(target/"inference_config.json",config); atomic_json(target/"config.json",config); (target/"target_test_video_ids.txt").write_text("\n".join(expected_ids)+"\n")
    calls=sum(int(x["vlm_calls"]) for _,x in manifests); elapsed=sum(float(x["sum_vlm_elapsed_seconds"]) for _,x in manifests); wall=max(float(x["wall_elapsed_seconds"]) for _,x in manifests)
    atomic_json(target/"inference_manifest.json",{"videos":videos,"prediction_length":len(combined),"frame_scores_sha256":sha256_file(out),"gt_read":False,"target_order":expected_ids,"sharded":True,"shard_count":args.shard_count,"vlm_calls":calls,"sum_vlm_elapsed_seconds":elapsed,"wall_elapsed_seconds":wall})
    atomic_json(target/"prediction_lock.json",{"target_name":args.target_name,"prediction":str(out),"sha256":sha256_file(out),"length":len(combined),"gt_read_before_lock":False,"sharded":True,"shard_count":args.shard_count})
    print(json.dumps({"prediction":str(out),"length":len(combined),"sha256":sha256_file(out)},indent=2))


if __name__ == "__main__": main()
