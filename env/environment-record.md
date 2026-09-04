# Recorded server environment

The released experiments used a local `Qwen/Qwen3.8-27B` service at revision
`1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`, BF16, vLLM 0.27.1. The training
environment recorded `torch==2.13.0`, `transformers==5.15.1`, `numpy==2.3.5`,
and `Pillow==12.3.0`; PEFT 0.19.1 and Accelerate 1.12.0 were vendored for the
formal LoRA run. CUDA and PyTorch must be installed for the target GPU before
installing the Python-level dependencies in `requirements-base.txt`.

No Qwen base-model weight, ShanghaiTech video/frame, frame mask, I3D feature,
or service cache is included in this repository.
