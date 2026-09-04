#!/usr/bin/env python3
"""Independent evaluator.  This is the only zero-shot component allowed to read GT."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'UR-DMU'))
from anomaly_ruler.core import atomic_json, pad_last, sha256_file
from anomaly_ruler.data import ShanghaiData

p = argparse.ArgumentParser()
p.add_argument('--workspace', required=True)
p.add_argument('--output-dir', required=True)
a = p.parse_args()
out = Path(a.output_dir)
lock = json.loads((out/'prediction_lock.json').read_text())
pred_path = Path(lock['prediction'])
if sha256_file(pred_path) != lock['sha256']:
    raise ValueError('prediction changed after lock')
config = json.loads((out/'inference_config.json').read_text())
if config.get('source_data_used') is not False or config.get('gt_read') is not False:
    raise ValueError('not a locked target-only zero-shot prediction')
data = ShanghaiData(a.workspace); target = config['target_domain']; known = data.by_id
pred = np.load(pred_path); labels_all=[]; offset=0; per_video=[]
for rec in config['target_manifest']:
    vid=rec['video_id']; item=known[vid]; n=rec['prediction_length']
    if item['label']=='abnormal':
        raw=np.load(Path(a.workspace)/'datasets/shanghaitech/testing/test_frame_mask'/f'{vid}.npy').reshape(-1)
    else:
        raw=np.zeros(item['frame_count'],dtype=np.uint8)
    labels=pad_last(raw,n).astype(np.uint8); scores=pred[offset:offset+n]
    if len(scores)!=n or len(labels)!=n: raise ValueError(f'{vid}: video alignment error')
    auc=float(roc_auc_score(labels,scores)) if len(np.unique(labels))==2 else None
    ap=float(average_precision_score(labels,scores)) if labels.sum()>0 else None
    per_video.append({'video_id':vid,'frames':n,'anomalous_frames':int(labels.sum()),'auroc':auc,'ap':ap})
    labels_all.extend(labels.tolist()); offset += n
gt=np.asarray(labels_all,dtype=np.uint8)
if offset!=len(pred) or len(gt)!=len(pred): raise ValueError('concatenated GT/prediction mismatch')
valid=[x['auroc'] for x in per_video if x['auroc'] is not None]
metrics={'method':'Direct VLM zero-shot target-only','target_name':config['target_name'],'target_domain':target,
 'frame_auroc':float(roc_auc_score(gt,pred)), 'frame_ap':float(average_precision_score(gt,pred)),
 'macro_auroc':float(np.mean(valid)), 'macro_auc_valid_videos':len(valid),
 'gt_frames':len(gt),'prediction_frames':len(pred),'anomalous_frames':int(gt.sum()),'videos':len(per_video),
 'per_video':per_video,'prediction_sha256':lock['sha256'],'evaluation_after_lock':True}
atomic_json(out/'evaluation.json',metrics)
print(json.dumps(metrics,indent=2))
