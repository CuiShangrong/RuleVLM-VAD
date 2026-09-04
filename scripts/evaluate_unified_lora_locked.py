#!/usr/bin/env python3
"""Post-lock evaluator for Weakly-Supervised PEFT LoRA Direct-VLM outputs."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def roc_auc_score(labels: np.ndarray, scores: np.ndarray) -> float:
    """Dependency-free binary AUROC with average ranks for tied scores.

    This is the Mann--Whitney formulation used by standard binary AUROC
    implementations.  Keeping it local lets post-lock evaluation run in the
    isolated Qwen environment, which intentionally does not install sklearn.
    """
    y = np.asarray(labels, dtype=np.uint8).reshape(-1)
    s = np.asarray(scores, dtype=np.float64).reshape(-1)
    positives = int(y.sum()); negatives = len(y) - positives
    if positives == 0 or negatives == 0:
        raise ValueError('AUROC requires both classes')
    order = np.argsort(s, kind='mergesort')
    ranks = np.empty(len(s), dtype=np.float64)
    sorted_scores = s[order]
    start = 0
    while start < len(s):
        end = start + 1
        while end < len(s) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2.0  # 1-indexed ranks
        start = end
    return float((ranks[y == 1].sum() - positives * (positives + 1) / 2.0) /
                 (positives * negatives))


def average_precision_score(labels: np.ndarray, scores: np.ndarray) -> float:
    """Dependency-free binary AP, grouping equal score thresholds exactly."""
    y = np.asarray(labels, dtype=np.uint8).reshape(-1)
    s = np.asarray(scores, dtype=np.float64).reshape(-1)
    positives = int(y.sum())
    if positives == 0:
        raise ValueError('AP requires at least one positive label')
    order = np.argsort(-s, kind='mergesort')
    y = y[order]; s = s[order]
    tps = np.cumsum(y, dtype=np.float64)
    # Retain only the final sample at every distinct score, as sklearn does.
    threshold = np.r_[s[:-1] != s[1:], True]
    tps = tps[threshold]
    fps = (np.nonzero(threshold)[0] + 1).astype(np.float64) - tps
    recall = tps / positives
    precision = tps / (tps + fps)
    return float(np.sum(np.diff(np.r_[0.0, recall]) * precision))

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'UR-DMU'))
from anomaly_ruler.core import atomic_json, pad_last, sha256_file
from anomaly_ruler.data import ShanghaiData


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--workspace', required=True)
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args()
    out = Path(args.output_dir)
    lock = json.loads((out/'prediction_lock.json').read_text())
    pred_path = Path(lock['prediction'])
    if sha256_file(pred_path) != lock['sha256']:
        raise ValueError('prediction changed after lock')
    config = json.loads((out/'inference_config.json').read_text())
    if config.get('gt_read') is not False or config.get('target_gt_used') is not False:
        raise ValueError('prediction was not GT-isolated before lock')
    if config.get('source_data_used') is not True or not config.get('lora_adapter'):
        raise ValueError('not a PEFT LoRA prediction')

    data = ShanghaiData(args.workspace)
    pred = np.load(pred_path)
    labels_all, per_video, offset = [], [], 0
    for rec in config['target_manifest']:
        vid = rec['video_id']; item = data.by_id[vid]; n = rec['prediction_length']
        if item['label'] == 'abnormal':
            raw = np.load(Path(args.workspace)/'datasets/shanghaitech/testing/test_frame_mask'/f'{vid}.npy').reshape(-1)
        else:
            raw = np.zeros(item['frame_count'], dtype=np.uint8)
        labels = pad_last(raw, n).astype(np.uint8); scores = pred[offset:offset+n]
        if len(scores) != n or len(labels) != n:
            raise ValueError(f'{vid}: video alignment error')
        auc = float(roc_auc_score(labels, scores)) if len(np.unique(labels)) == 2 else None
        ap = float(average_precision_score(labels, scores)) if labels.sum() else None
        per_video.append({'video_id':vid, 'frames':n, 'anomalous_frames':int(labels.sum()), 'auroc':auc, 'ap':ap})
        labels_all.extend(labels.tolist()); offset += n
    gt = np.asarray(labels_all, dtype=np.uint8)
    if offset != len(pred) or len(gt) != len(pred):
        raise ValueError('concatenated GT/prediction mismatch')
    valid = [x['auroc'] for x in per_video if x['auroc'] is not None]
    metrics = {
        'method': config['method'], 'setup': config['setup'], 'source_domain': config['source_domain'],
        'target_domain': config['target_domain'], 'frame_auroc':float(roc_auc_score(gt,pred)),
        'frame_ap':float(average_precision_score(gt,pred)), 'macro_auroc':float(np.mean(valid)),
        'macro_auc_valid_videos':len(valid), 'gt_frames':len(gt), 'prediction_frames':len(pred),
        'anomalous_frames':int(gt.sum()), 'videos':len(per_video), 'per_video':per_video,
        'prediction_sha256':lock['sha256'], 'adapter_model_sha256':config['lora_adapter']['adapter_model_sha256'],
        'evaluation_after_lock':True,
    }
    atomic_json(out/'evaluation.json', metrics)
    print(json.dumps(metrics, indent=2))


if __name__ == '__main__':
    main()
