#!/usr/bin/env python3
import argparse,sys
from pathlib import Path
sys.path[:0]=[str(Path(__file__).resolve().parents[1]/'vendor'),str(Path(__file__).resolve().parents[3]/'UR-DMU')]
from vlm_baselines.fair_unified import UnifiedScorer,FAIR
from vlm_baselines.dense_zeroshot import run_dense_target
p=argparse.ArgumentParser();p.add_argument('--workspace',required=True);p.add_argument('--run-root',required=True);p.add_argument('--target',choices=['domain_01','domain_08','mixed'],required=True);p.add_argument('--model-path',required=True);p.add_argument('--max-pixels',type=int,default=336*336);p.add_argument('--native-resolution',action='store_true');p.add_argument('--video-id',action='append');p.add_argument('--smoke',action='store_true');a=p.parse_args()
cfg={**FAIR,'max_pixels':a.max_pixels}
if a.native_resolution: cfg.update({'min_pixels':65536,'max_pixels':16777216})
s=UnifiedScorer(a.model_path,cfg=cfg)
print(run_dense_target(a.workspace,a.run_root,a.target,s,context_frames=cfg['context_frames'],center_stride=cfg['center_stride'],video_ids=a.video_id,smoke=a.smoke,extra_config={'unified_transformers_pipeline':True,'unified_preprocessing':cfg,'source_data_used':False}))
