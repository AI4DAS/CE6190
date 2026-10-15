# Modern Extension Environment Report

Created: 2026-10-10

The existing `personalization-benchmark` Conda environment was not modified.
Two isolated environments were created under `/home/zwu/anaconda3/envs/`.

## DreamO environment

- Conda name: `environment_dreamo`
- Python: 3.10.22
- Torch: 2.6.0+cu124
- Torch CUDA runtime: 12.4
- torchvision: 0.21.0
- Diffusers: 0.33.1
- Transformers: 4.45.2
- Accelerate: 0.32.0
- PEFT: 0.21.2
- Approximate environment size: 6.3 GB
- Official pipeline import: PASS
- CUDA isolation test on one MIG: PASS
- Visible GPU: NVIDIA H200 NVL MIG 1g.35gb, 32.5 GiB

Complete package lock snapshot:
`results/system/modern_environments/dreamo_pip_freeze.txt`

Conda history export:
`results/system/modern_environments/environment_dreamo.yml`

## UNO environment

- Conda name: `environment_uno`
- Python: 3.10.22
- Torch: 2.4.0+cu124
- Torch CUDA runtime: 12.4
- torchvision: 0.19.0+cu124
- Diffusers: 0.30.1
- Transformers: 4.43.3
- Accelerate: 1.1.1
- UNO editable metadata: 0.0.1
- Approximate environment size: 5.5 GB
- Official pipeline import with repository on `PYTHONPATH`: PASS
- CUDA isolation test on one MIG: PASS
- Visible GPU: NVIDIA H200 NVL MIG 1g.35gb, 32.5 GiB

Complete package lock snapshot:
`results/system/modern_environments/uno_pip_freeze.txt`

Conda history export:
`results/system/modern_environments/environment_uno.yml`

## Installation commands used

```bash
conda create -n environment_dreamo python=3.10 -y
/home/zwu/anaconda3/envs/environment_dreamo/bin/pip install \
  -r third_party/modern/DreamO/requirements.txt

conda create -n environment_uno python=3.10 -y
/home/zwu/anaconda3/envs/environment_uno/bin/pip install \
  -r third_party/modern/UNO/requirements.txt
/home/zwu/anaconda3/envs/environment_uno/bin/pip install \
  -e third_party/modern/UNO --no-deps
/home/zwu/anaconda3/envs/environment_uno/bin/pip install accelerate==1.1.1
```

## Storage

The two environments consume approximately 11.8 GB. Repository checkouts use
about 52 MB. Full model checkpoints were deliberately not downloaded during
this audit. The filesystem retained approximately 4.8 TiB free afterward.

## Compatibility notes

- DreamO dependencies match the official pinned inference requirements.
- UNO's official CLI imports Accelerate, which is not listed in its base
  inference dependencies. Accelerate 1.1.1 was added because that is the
  version declared by the repository's optional dependency metadata.
- UNO must be launched from its repository directory or with an explicit
  `PYTHONPATH` because the upstream namespace lacks a top-level
  `uno/__init__.py`.
- No Nunchaku package was installed. The first approved DreamO test should use
  the official bf16 path on the 32.5-GiB MIG; quantization should be introduced
  only if measured memory requires it.
