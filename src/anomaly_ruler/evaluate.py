from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from .core import atomic_json, pad_last, sha256_file
from .data import SETUPS, ShanghaiData


def evaluate_locked(workspace, output_dir):
    output_dir=Path(output_dir); lock=json.loads((output_dir/'prediction_lock.json').read_text())
    pred_path=Path(lock['prediction'])
    if sha256_file(pred_path)!=lock['sha256']: raise ValueError('prediction changed after lock')
    config=json.loads((output_dir/'inference_config.json').read_text()); setup=config['setup']
    data=ShanghaiData(workspace); target=SETUPS[setup][1]
    by_id=data.by_id; gt=[]; per_video=[]; pred=np.load(pred_path); offset=0
    for rec in config['target_manifest']:
        vid=rec['video_id']; item=by_id[vid]; n=rec['prediction_length']
        if item['label']=='abnormal':
            raw=np.load(Path(workspace)/'datasets/shanghaitech/testing/test_frame_mask'/f'{vid}.npy').reshape(-1)
        else: raw=np.zeros(item['frame_count'],dtype=np.uint8)
        labels=pad_last(raw,n).astype(np.uint8); scores=pred[offset:offset+n]
        if len(scores)!=n: raise ValueError(f'{vid} prediction mismatch')
        auc=float(roc_auc_score(labels,scores)) if len(np.unique(labels))==2 else None
        ap=float(average_precision_score(labels,scores)) if labels.sum()>0 else None
        per_video.append({"video_id":vid,"frames":n,"anomalous_frames":int(labels.sum()),"auroc":auc,"ap":ap})
        gt.extend(labels.tolist()); offset+=n
    gt=np.asarray(gt,dtype=np.uint8)
    if offset!=len(pred) or len(gt)!=len(pred): raise ValueError(f'GT/pred mismatch {len(gt)} {len(pred)}')
    metrics={"setup":setup,"frame_auroc":float(roc_auc_score(gt,pred)),
      "frame_ap":float(average_precision_score(gt,pred)),"gt_frames":len(gt),"prediction_frames":len(pred),
      "anomalous_frames":int(gt.sum()),"videos":len(per_video),
      "macro_auroc":float(np.mean([x['auroc'] for x in per_video if x['auroc'] is not None])),
      "macro_auc_valid_videos":sum(x['auroc'] is not None for x in per_video),
      "per_video":per_video,"prediction_sha256":lock['sha256'],"evaluation_after_lock":True}
    atomic_json(output_dir/'evaluation.json',metrics); return metrics
