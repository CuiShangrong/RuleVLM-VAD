#!/usr/bin/env python3
"""Target-only 10-second/8-frame Direct VLM inference; never reads GT."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'UR-DMU'))
from vlm_baselines.common import EndpointScorer, ZERO_SHOT_TARGETS, run_zeroshot_target

p = argparse.ArgumentParser()
p.add_argument('--workspace', required=True)
p.add_argument('--run-root', required=True)
p.add_argument('--target', choices=sorted(ZERO_SHOT_TARGETS), required=True)
p.add_argument('--endpoint', required=True)
p.add_argument('--model', default='Qwen/Qwen3.8-27B')
p.add_argument('--revision', required=True)
p.add_argument('--window-frames', type=int, default=240)
p.add_argument('--sample-count', type=int, default=8)
p.add_argument('--video-id', action='append')
p.add_argument('--smoke', action='store_true')
a = p.parse_args()
scorer = EndpointScorer(a.endpoint, a.model, a.revision,
                        Path(a.run_root) / 'cache' / 'zeroshot_10s_8frame')
print(run_zeroshot_target(a.workspace, a.run_root, a.target, scorer,
      window_length=a.window_frames, sample_count=a.sample_count,
      video_ids=a.video_id, smoke=a.smoke))
