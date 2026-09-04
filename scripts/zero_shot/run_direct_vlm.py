#!/usr/bin/env python3
import argparse,sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'UR-DMU'))
from vlm_baselines.common import EndpointScorer,run_direct
p=argparse.ArgumentParser(); p.add_argument('--workspace',required=True);p.add_argument('--run-root',required=True);p.add_argument('--setup',required=True);p.add_argument('--endpoint',required=True);p.add_argument('--model',default='Qwen/Qwen3.8-27B');p.add_argument('--revision',required=True);p.add_argument('--window-length',type=int,default=32);p.add_argument('--stride',type=int,default=32);p.add_argument('--sample-count',type=int,default=4);p.add_argument('--video-id',action='append');p.add_argument('--smoke',action='store_true')
a=p.parse_args(); s=EndpointScorer(a.endpoint,a.model,a.revision,Path(a.run_root)/'cache')
print(run_direct(a.workspace,a.run_root,a.setup,s,window_length=a.window_length,stride=a.stride,sample_count=a.sample_count,video_ids=a.video_id,smoke=a.smoke))
