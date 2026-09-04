#!/usr/bin/env python3
import argparse,json,subprocess
from pathlib import Path
from client_common import request_with_retry,sample_frames
def gpu():return int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits','--id=0'],text=True).strip().splitlines()[0])
def main():
 p=argparse.ArgumentParser();p.add_argument('--frame-dir',required=True);p.add_argument('--requests',type=int,default=20);p.add_argument('--output',required=True);a=p.parse_args();samples=sample_frames(a.frame_dir);runs=[];baseline=gpu();peak=baseline
 for i in range(a.requests):
  x=request_with_retry(samples);used=gpu();peak=max(peak,used);runs.append({'index':i+1,'valid':x['result']['valid'],'attempts':len(x['attempts']),'ttft_seconds':x['result']['ttft_seconds'],'total_seconds':x['result']['total_seconds'],'gpu_memory_mib':used,'parsed':x['result']['parsed'],'raw_responses':[z['raw'] for z in x['attempts']]});print(i+1,runs[-1]['valid'],runs[-1]['total_seconds'],used)
 report={'requests':len(runs),'valid':sum(x['valid'] for x in runs),'success_rate':sum(x['valid'] for x in runs)/len(runs),'oom':False,'baseline_gpu_memory_mib':baseline,'peak_gpu_memory_mib':peak,'end_gpu_memory_mib':runs[-1]['gpu_memory_mib'],'memory_growth_mib':runs[-1]['gpu_memory_mib']-baseline,'mean_ttft_seconds':sum(x['ttft_seconds'] for x in runs)/len(runs),'mean_total_seconds':sum(x['total_seconds'] for x in runs)/len(runs),'runs':runs};Path(a.output).write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k!='runs'},indent=2))
if __name__=='__main__':main()
