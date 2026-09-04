# DenseRuleLoRA-VAD

**Dense temporal Direct VLM baselines for weakly-supervised ShanghaiTech video anomaly detection.**

This release packages three completed Qwen3.8-27B experiments:

1. **DenseLogit-VAD (zero-shot)** — no ShanghaiTech source training data, rules,
   retrieval, I3D feature, or adaptation.
2. **Weakly-Supervised LoRA Direct VLM** — source-only video-level normal/abnormal
   weak labels with MIL; Qwen vision encoder and base language-model weights are frozen.
3. **RuleLoRA-VAD** — the same LoRA inference protocol, with all normal-behavior
   rules induced solely from source-domain normal training videos inserted before
   the unchanged Direct VLM prompt.

The repository name and display name are deliberately short enough for a GitHub
project while describing the final method: **DenseRuleLoRA-VAD**.

## Method in one diagram

```text
~10 s context (240 frames) → 8 chronological uniformly sampled RGB frames
       → Qwen3.8-27B (optional source-only LoRA / normal-behavior rules)
       → first-token log P(0), log P(1) → P(abnormal)
       → stride-16 dense temporal centres → linear interpolation → frame scores
```

At a boundary, only valid video frames are used; the last valid frame is repeated
only to fill an eight-image input slot. Each video prediction is separately
aligned to the snippet-padded length recorded by the WVAD manifest.

## Fixed Direct VLM prompt

System prompt:

```text
You are a visual anomaly detector. Output only 0 or 1.
```

User prompt:

```text
You are given eight chronological surveillance frames from one approximately 10-second clip. Based only on visible evidence, classify the clip. Focus on human actions, interactions, vehicles, falls, fights, abnormal motion, and abnormal objects. Do not infer unseen events. Output exactly one character: 0 for normal or 1 for abnormal.
```

The continuous score is `exp(logP(1)) / (exp(logP(0)) + exp(logP(1)))`, not a
verbalized confidence score. RuleLoRA-VAD adds a source-only normal-rule prefix
before this user prompt; no target label, GT interval, target rule, or target
adaptation is supplied.

## LoRA weak-supervision protocol

- Base model: `Qwen/Qwen3.8-27B`, revision
  `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`, BF16.
- PEFT LoRA targets: `q_proj`, `k_proj`, `v_proj`, `o_proj`; rank 16, alpha 32,
  dropout 0.05.
- Vision encoder and base language-model weights: frozen.
- Source training input: video-level WVAD labels from `SH_Train.txt` only.
- Normal bags: sampled clips are supervised toward normal.
- Weak-abnormal bags: detached top-2 current anomaly clips provide positive MIL
  supervision. The method does **not** label every abnormal-video clip positive.
- Source train/validation determines checkpoints; target test and frame masks are
  excluded from training, tuning, and checkpoint selection.

## Released content

| Directory | Contents |
| --- | --- |
| `src/` | Direct VLM, dense token-probability, unified LoRA, data/alignment and evaluation code. |
| `scripts/` | Rule construction, PEFT-MIL training, inference, evaluation, and local-service helpers. |
| `splits/` | Official RTFM-style 238-train / 199-test WVAD lists and the alignment manifest. |
| `artifacts/zero_shot/` | Locked dense zero-shot target predictions, configurations and evaluations. |
| `artifacts/lora/` | Formal LoRA adapters, source splits, logs, locks, dense predictions and evaluations. |
| `artifacts/lora_all_rules/` | Source-normal descriptions, induced rules, rule classifications, rule-prompt predictions and evaluations. |
| `results/` | Consolidated RuleLoRA-VAD results. |
| `env/` | Dependency and exact environment record. |

The release intentionally excludes ShanghaiTech raw videos/frames/GT, I3D
features, Qwen base-model weights, Hugging Face cache, and transient GPU/runtime
caches. Obtain ShanghaiTech through its authorised distribution and Qwen through
the model's official access route; point code to those local paths.

## Final locked results

Metrics are frame-level AUROC / AP / macro AUROC. Zero-shot is target-only and
therefore repeats by target rather than source→target setup.

| Method | 01→01 | 01→08 | 08→08 | 08→01 | Mixed→Mixed |
| --- | --- | --- | --- | --- | --- |
| Dense zero-shot | 0.715310 / 0.127525 / 0.473863 | 0.779870 / 0.163761 / 0.610319 | 0.779870 / 0.163761 / 0.610319 | 0.715310 / 0.127525 / 0.473863 | 0.766390 / 0.231664 / 0.583061 |
| Weakly-supervised LoRA | 0.902855 / 0.290461 / 0.520079 | 0.820182 / 0.222486 / 0.624405 | 0.815538 / 0.248949 / 0.571743 | 0.728582 / 0.124542 / 0.436631 | 0.897463 / 0.343179 / 0.589900 |
| RuleLoRA-VAD (all source rules) | 0.924567 / 0.378956 / 0.586415 | 0.862383 / 0.238870 / 0.728371 | 0.885392 / 0.271743 / 0.710591 | 0.887586 / 0.292733 / 0.533067 | 0.912386 / 0.377532 / 0.618048 |

`results/results_summary.csv` is the machine-readable all-rules table. All
published evaluation files were generated only after their corresponding
`prediction_lock.json` was written.

## Quick integrity check

```bash
python scripts/verify_release.py
```

It verifies every released `frame_scores.npy` against its locked SHA256 hash.

## Running locally

Set the code import path and point the supplied scripts at your ShanghaiTech
workspace, Qwen model path, and local OpenAI-compatible Qwen endpoint:

```bash
export PYTHONPATH="$PWD/src"
python scripts/run_unified_base.py --help
python scripts/train_peft_mil.py --help
python scripts/run_unified_lora.py --help
python scripts/run_unified_lora_all_rules.py --help
```

Run source-only training and inference only after verifying the data layout,
model revision and legal access conditions. The scripts preserve the protocol
above; do not substitute target GT for weak source supervision.

## Citation and scope

This is a ShanghaiTech WVAD adaptation and research artifact, not an official
Qwen or ShanghaiTech release. Respect the licenses and terms of the ShanghaiTech
dataset, Qwen model, PEFT, Transformers, and all upstream components.
