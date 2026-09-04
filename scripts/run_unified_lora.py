#!/usr/bin/env python3
"""Locked-prediction inference for the native-resolution PEFT Direct-VLM runs.

This module intentionally does not import an evaluator, sklearn, a frame mask,
or a GT path.  It can only create deterministic frame-score predictions and a
prediction lock.  Evaluation is a separate post-lock process.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(Path(__file__).resolve().parents[1] / 'vendor'), str(ROOT / 'UR-DMU')]

from anomaly_ruler.core import sha256_file
from vlm_baselines.dense_zeroshot import run_dense_target
from vlm_baselines.fair_unified import UnifiedScorer, fair_config


SETUPS = {
    '01_to_01': ('01', 'domain_01'),
    '01_to_08': ('01', 'domain_08'),
    '08_to_08': ('08', 'domain_08'),
    '08_to_01': ('08', 'domain_01'),
    'mixed_to_mixed': ('mixed', 'mixed'),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--workspace', required=True)
    parser.add_argument('--run-root', required=True,
                        help='Root for all setup outputs; each setup is isolated below it.')
    parser.add_argument('--setup', choices=sorted(SETUPS), required=True)
    parser.add_argument('--adapter', required=True)
    parser.add_argument('--model-path', required=True)
    parser.add_argument('--video-id', action='append')
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()

    source, target_name = SETUPS[args.setup]
    adapter = Path(args.adapter)
    required = [adapter/'adapter_model.safetensors', adapter/'adapter_config.json',
                adapter.parent/'training_complete.json', adapter.parent/'train_config.json']
    missing = [str(x) for x in required if not x.is_file()]
    if missing:
        raise FileNotFoundError(f'missing selected adapter provenance: {missing}')
    complete = json.loads((adapter.parent/'training_complete.json').read_text())
    train_cfg = json.loads((adapter.parent/'train_config.json').read_text())
    if complete.get('selected_checkpoint') != 'selected_adapter':
        raise ValueError('adapter is not the source-validation-selected checkpoint')
    if train_cfg.get('source') != source:
        raise ValueError(f'{args.setup}: adapter source {train_cfg.get("source")!r} != {source!r}')
    if train_cfg.get('target_gt_used') is not False or train_cfg.get('frame_masks_used') is not False:
        raise ValueError('training provenance violates source-only weak-supervision protocol')

    cfg = fair_config(native_resolution=True)
    scorer = UnifiedScorer(args.model_path, adapter_path=str(adapter), cfg=cfg)
    setup_root = Path(args.run_root) / args.setup
    result = run_dense_target(
        args.workspace, setup_root, target_name, scorer,
        context_frames=cfg['context_frames'], center_stride=cfg['center_stride'],
        video_ids=args.video_id, smoke=args.smoke,
        extra_config={
            'method': 'Weakly-Supervised PEFT LoRA Direct VLM dense forced-choice token-probability',
            'setup': args.setup,
            'source_domain': source,
            'source_data_used': True,
            'target_gt_used': False,
            'unified_transformers_pipeline': True,
            'unified_preprocessing': cfg,
            'lora_adapter': {
                'path': str(adapter),
                'adapter_model_sha256': sha256_file(adapter/'adapter_model.safetensors'),
                'adapter_config_sha256': sha256_file(adapter/'adapter_config.json'),
                'training_complete_sha256': sha256_file(adapter.parent/'training_complete.json'),
                'train_config_sha256': sha256_file(adapter.parent/'train_config.json'),
                'selected_source_val_mil_loss': complete['selected_val_mil_loss'],
            },
        },
    )
    print(result)


if __name__ == '__main__':
    main()
