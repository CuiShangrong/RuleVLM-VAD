#!/usr/bin/env python3
import argparse,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[3]/'UR-DMU'))
from vlm_baselines.common import ZERO_SHOT_TARGETS
from vlm_baselines.dense_zeroshot import TokenProbabilityScorer,run_dense_target
p=argparse.ArgumentParser();p.add_argument('--workspace',required=True);p.add_argument('--run-root',required=True);p.add_argument('--target',choices=sorted(ZERO_SHOT_TARGETS),required=True);p.add_argument('--endpoint',required=True);p.add_argument('--model',default='Qwen/Qwen3.8-27B');p.add_argument('--revision',required=True);p.add_argument('--context-frames',type=int,default=240);p.add_argument('--center-stride',type=int,default=16);p.add_argument('--video-id',action='append');p.add_argument('--smoke',action='store_true');a=p.parse_args()
s=TokenProbabilityScorer(a.endpoint,a.model,a.revision,Path(a.run_root)/'cache')
print(run_dense_target(a.workspace,a.run_root,a.target,s,context_frames=a.context_frames,center_stride=a.center_stride,video_ids=a.video_id,smoke=a.smoke))
