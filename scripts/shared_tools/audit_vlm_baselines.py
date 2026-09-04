#!/usr/bin/env python3
import argparse,json,sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'UR-DMU'))
from anomaly_ruler.core import atomic_json
from anomaly_ruler.data import SETUPS,ShanghaiData
p=argparse.ArgumentParser();p.add_argument('--workspace',required=True);p.add_argument('--run-root',required=True);a=p.parse_args();d=ShanghaiData(a.workspace);report={'gt_read':False,'setups':{}}
for setup,(source,target) in SETUPS.items():
 source_ids=([x for x in d.train_abnormal+d.train_normal if source=='mixed' or x.split('_')[0]==source])
 target_ids=[x['video_id'] for x in d.target_manifest(target)];overlap=sorted(set(source_ids)&set(target_ids))
 report['setups'][setup]={'source_weak_train_videos':len(source_ids),'target_test_videos':len(target_ids),'source_target_overlap':overlap,'status':'PASSED' if not overlap else 'FAILED'}
 if overlap:raise SystemExit(f'{setup} leakage: {overlap}')
report['status']='PASSED';atomic_json(Path(a.run_root)/'leakage_audit.json',report);print(json.dumps(report,indent=2))
