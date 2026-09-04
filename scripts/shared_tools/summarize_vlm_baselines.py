#!/usr/bin/env python3
import argparse,csv,json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--workspace',required=True);p.add_argument('--run-root',required=True);a=p.parse_args();r=Path(a.run_root);w=Path(a.workspace)
rows=[]
for family,method in [('direct','Direct VLM'),('tuned','Tuned VLM LoRA')]:
 for setup in ['01_to_01','01_to_08','08_to_08','08_to_01','mixed_to_mixed']:
  f=r/family/'setups'/setup/'evaluation.json'
  if f.exists():
   x=json.loads(f.read_text());rows.append({'method':method,'setup':setup,'frame_auroc':x['frame_auroc'],'frame_ap':x['frame_ap'],'macro_auroc':x['macro_auroc'],'frames':x['prediction_frames']})
for setup in ['01_to_01','01_to_08','08_to_08','08_to_01']:
 f=w/'runs/urdmu_shanghaitech/cross_domain'/setup/'seed_2022/evaluation.json'
 if f.exists():
  x=json.loads(f.read_text());rows.append({'method':'UR-DMU WVAD','setup':setup,'frame_auroc':x['frame_auroc'],'frame_ap':x['frame_ap'],'macro_auroc':'','frames':x['prediction_frames']})
sf=w/'runs/urdmu_slowfast_agent_v2_local_entropy/results_summary.csv'
if sf.exists():
 for x in csv.DictReader(sf.open()):
  if x['variant'] in {'B1','B2','B4'}: rows.append({'method':'UR-DMU-SF-Agent '+x['name'],'setup':x['setup'],'frame_auroc':x['frame_auroc'],'frame_ap':x['frame_ap'],'macro_auroc':x['macro_frame_auroc'],'frames':x['frames']})
with (r/'results_summary.csv').open('w',newline='') as f:
 fields=['method','setup','frame_auroc','frame_ap','macro_auroc','frames'];o=csv.DictWriter(f,fieldnames=fields);o.writeheader();o.writerows(rows)
direct={x['setup']:x for x in rows if x['method']=='Direct VLM'};tuned={x['setup']:x for x in rows if x['method']=='Tuned VLM LoRA'}
summary={'rows':rows,'domain_shift_drop':{}}
for method,data in [('direct',direct),('tuned',tuned)]:
 for key,a_,b_ in [('01_to_01_to_01_to_08','01_to_01','01_to_08'),('08_to_08_to_08_to_01','08_to_08','08_to_01')]:
  if a_ in data and b_ in data:summary['domain_shift_drop'][method+'_'+key]=data[a_]['frame_auroc']-data[b_]['frame_auroc']
(r/'results_summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))
