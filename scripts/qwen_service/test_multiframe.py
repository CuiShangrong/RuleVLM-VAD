#!/usr/bin/env python3
import argparse,json
from pathlib import Path
from client_common import request_with_retry,sample_frames
def main():
 p=argparse.ArgumentParser();p.add_argument('--frame-dir',required=True);p.add_argument('--name',required=True);p.add_argument('--repeats',type=int,default=3);p.add_argument('--output-dir',required=True);a=p.parse_args();out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=True);samples=sample_frames(a.frame_dir);runs=[]
 for i in range(a.repeats):
  x=request_with_retry(samples);runs.append(x);(out/f'{a.name}_run_{i+1}.json').write_text(json.dumps(x,indent=2)+'\n');print(i+1,x['result']['valid'],x['result']['ttft_seconds'],x['result']['total_seconds'])
 summary={'name':a.name,'frame_dir':a.frame_dir,'sampled_frames':[{'original_frame_index':i+1,'path':str(p)} for i,p in samples],'runs':len(runs),'valid_runs':sum(x['result']['valid'] for x in runs),'retries':sum(len(x['attempts'])-1 for x in runs),'identical':len({json.dumps(x['result']['parsed'],sort_keys=True) for x in runs})==1,'ttft_seconds':[x['result']['ttft_seconds'] for x in runs],'total_seconds':[x['result']['total_seconds'] for x in runs]};(out/f'{a.name}_summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
