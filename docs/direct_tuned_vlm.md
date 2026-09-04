# Direct VLM and Tuned VLM: ShanghaiTech cross-domain

Both baselines use the local Qwen/Qwen3.8-27B revision
1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0. They are independent of
UR-DMU, AnomalyRuler and the SF-Agent, and they do not read I3D features.

## Shared inference

Videos use non-overlapping 32-frame windows with stride 32. Four RGB images
are sampled uniformly per window. A fixed target-label-free JSON prompt returns
an integer anomaly score from 0 to 100, which is divided by 100 and copied to
valid frames in the window. The final score is padded to the existing WVAD
prediction length. Prediction SHA256 is locked before the independent evaluator
reads GT to calculate micro AUROC/AP and macro video AUROC.

## Tuning

Tuned VLM uses BF16 LoRA, rank 8, alpha 16, on q/k/v/o and MLP projections.
Supervision is only weak video-level SH_Train labels: sampled clips from
abnormal source training videos target score 100; normal clips target score 0.
There is no source frame-GT fabrication, target training, target checkpoint
selection or early stopping. PEFT and Accelerate are installed in the separate
workspace directory vlm_tuning/deps.

## Smoke

On setup 01_to_08 and video 08_0058, Direct VLM produced AUROC 0.506869 and AP
0.666652; two-step source-only LoRA smoke produced AUROC 0.594451 and AP
0.702246. Both produced 352 predictions for 352 GT frames. These are smoke
results, not formal domain metrics.

## Per-setup launchers

From the workspace root, run:

    bash UR-DMU/tools/run_direct_vlm.sh 01_to_08
    bash UR-DMU/tools/run_tuned_vlm_01_to_08.sh

Equivalent scripts exist for 01_to_01, 08_to_08, 08_to_01 and mixed_to_mixed.
The per-setup tuned script uses one GPU by default (GPU 3), including training
and adapter inference.

## Three-GPU total launchers

These scripts are prepared but have not been executed.

    bash UR-DMU/tools/run_direct_vlm_3gpu_and_stop.sh
    bash UR-DMU/tools/run_tuned_vlm_train_1gpu_infer_3gpu_and_stop.sh

The first script starts or reuses three local Qwen services on GPUs 0, 1 and 2,
runs Direct VLM in parallel, evaluates and summarizes, then stops those Qwen
services. The second stops Qwen before starting, trains the five LoRA adapters
serially on one GPU (default GPU 3), then runs local-Transformers adapter
inference on GPUs 0, 1 and 2 in parallel. It also leaves Qwen stopped.

All results, adapters, logs, locks and summary CSV/JSON files are under
runs/vlm_baselines_shanghaitech.
