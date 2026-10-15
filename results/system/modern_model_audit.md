# Modern Personalization Model Audit

Audit date: 2026-10-10

No inference or training was run during this audit. The completed SD1.5
benchmark and all existing outputs, metrics, and figures were left unchanged.

## Scope and interpretation

DreamO and UNO are modern reference-conditioned extensions, not additional
rows in the controlled SD1.5 experiment. They use FLUX-family diffusion
transformers, different pretrained data and scale, and different conditioning
mechanisms. Any later result must therefore be described as a capability
trade-off under a different conditioning paradigm, not as an isolated
architecture comparison.

## Summary

| Method | Official repository | Extension checkpoint | Architecture | Reference input | Text input | Estimated peak VRAM | Estimated runtime |
|---|---|---|---|---|---|---:|---:|
| DreamO v1.1 | `https://github.com/bytedance/DreamO` | `ByteDance/DreamO` plus FLUX.1-dev and auxiliary LoRAs | FLUX.1-dev DiT with DreamO task/index embeddings and fused LoRA adapters | One or two images; `IP` is the appropriate task for the selected non-human subjects | Natural-language prompt | About 24 GB for official bf16/offload path; official int8 path about 16 GB; Nunchaku/offload about 6.5 GB | Official turbo path uses 12 steps. Budget 10–30 s/image on a 1g.35gb H200 MIG until measured |
| UNO | `https://github.com/bytedance/UNO` | `bytedance-research/UNO/dit_lora.safetensors` plus FLUX.1-dev, AE, T5, and CLIP | FLUX.1-dev DiT with progressive cross-modal alignment, universal rotary position encoding, and rank-512 UNO LoRA | One or multiple images; official preprocessing uses reference long side 512 for one subject and 320 for multiple subjects | Natural-language prompt | Official fp8 mode is documented at about 16 GB; bf16 may exceed the practical margin of a 32.5-GiB MIG | Official default is 25 steps. Budget 15–45 s/image on a 1g.35gb H200 MIG until measured |

Runtime ranges above are planning estimates, not measurements. Later metadata
must replace them with synchronized per-image timings and measured CUDA peak
allocation.

## DreamO

### Official source and revision

- Repository: `https://github.com/bytedance/DreamO`
- Local checkout: `third_party/modern/DreamO`
- Audited commit: `461e35ba6b068f42ca53b43f289b3d91b7f911cd`
- License reported by the repository/model card: Apache-2.0
- Paper/model: `https://huggingface.co/ByteDance/DreamO`

### Installation requirements

The official requirements use Python 3.10, Torch 2.6.0, torchvision 0.21.0,
Diffusers 0.33.1, Transformers 4.45.2, Accelerate 0.32.0, PEFT, Quanto,
facexlib, and BEN2-related image preprocessing dependencies. These were
installed only in the `environment_dreamo` Conda environment.

### Architecture and inference requirements

`DreamOPipeline` extends the Diffusers `FluxPipeline`. The default v1.1 path
loads `black-forest-labs/FLUX.1-dev` in bf16, then applies DreamO, distilled
CFG, SFT, DPO, and optional FLUX Turbo LoRA weights. For the audited animal and
object subjects the reference task is `IP`, which performs background removal
before VAE-based reference conditioning. The official interface accepts up to
two reference images and returns a PIL image.

Expected components for the planned v1.1 turbo experiment include:

- `black-forest-labs/FLUX.1-dev`
- `ByteDance/DreamO/dreamo.safetensors`
- `ByteDance/DreamO/dreamo_cfg_distill.safetensors`
- `ByteDance/DreamO/v1.1/dreamo_sft_lora.safetensors`
- `ByteDance/DreamO/v1.1/dreamo_dpo_lora.safetensors`
- `alimama-creative/FLUX.1-Turbo-Alpha/diffusion_pytorch_model.safetensors`
- `PramaLLC/BEN2/BEN2_Base.pth`

### Checkpoint audit

- DreamO repository revision: `eb693ac03bc364d17ec4fe04a3c6ffcad038ac0c`
- DreamO repository size reported by the Hub API: 5.122 GiB, including ComfyUI
  duplicates. The four v1.1 files required by the Python path total roughly
  2.0 GiB.
- The DreamO model repository is public and a small-file access test passed.
- FLUX.1-dev revision: `3de623fc3c33e44ffbe2bad470d0f45bccf2eb21`
- FLUX.1-dev is gated. The current token received HTTP 403 for
  `model_index.json`; therefore the complete DreamO checkpoint stack is **not
  currently downloadable on this account**.

## UNO

### Official source and revision

- Repository: `https://github.com/bytedance/UNO`
- Local checkout: `third_party/modern/UNO`
- Audited commit: `a563432dcfb4caf98df4368facb8102bf97d376c`
- License reported by the repository/model card: Apache-2.0; the FLUX base
  model remains subject to its own terms.
- Model: `https://huggingface.co/bytedance-research/UNO`

### Installation requirements

The official inference requirements use Python 3.10–3.12, Torch 2.4.0 with
CUDA 12.4, torchvision 0.19.0, Diffusers 0.30.1, Transformers 4.43.3,
SentencePiece 0.2.0, and Gradio 5.22.0. `inference.py` imports Accelerate even
though it is omitted from the inference dependency list, so the repository's
documented Accelerate 1.1.1 was installed without the training-only DeepSpeed
dependency.

The upstream `uno/` directory has no top-level `__init__.py`; its editable
installation does not expose the namespace from arbitrary working
directories. The official pipeline imports successfully when run from the
repository root or with `PYTHONPATH=third_party/modern/UNO`. No upstream source
file was patched.

### Architecture and input/output

UNO uses a FLUX.1-dev diffusion transformer with an official rank-512 LoRA.
The command-line interface accepts either a prompt plus a list of image paths,
or an evaluation JSON containing those fields. It supports single-reference
and multi-reference conditioning in the same model. It writes one PNG and one
JSON configuration record per result.

Expected components are:

- `black-forest-labs/FLUX.1-dev/flux1-dev.safetensors`
- `black-forest-labs/FLUX.1-dev/ae.safetensors`
- `xlabs-ai/xflux_text_encoders`
- `openai/clip-vit-large-patch14`
- `bytedance-research/UNO/dit_lora.safetensors`

### Checkpoint audit

- UNO checkpoint revision: `cd4a5a8896243db22aaf4a2d3d2a9511c437fb12`
- UNO LoRA size: 1.781 GiB; the public repository access test passed.
- Official documentation estimates the complete required download at about
  37 GB when only required FLUX files are selected.
- The shared FLUX.1-dev access test failed with HTTP 403. Consequently UNO is
  also **not currently ready for inference**, despite its public LoRA being
  available.

## Selected extension subjects

| Subject | Category | Reference directory | Images | Challenge |
|---|---|---|---:|---|
| dog2 | Animal | `data/processed/dreambooth/dog2/images` | 6 | Identity and fur preservation |
| robot_toy | Rigid object | `data/processed/dreambooth/robot_toy/images` | 5 | Geometry and small structural details |
| colorful_sneaker | Wearable/product | `data/processed/dreambooth/colorful_sneaker/images` | 5 | Color, texture, and pattern preservation |

The six clean prompts and four seeds remain those in
`configs/main_prompts.yaml`. Class images will never be used as extension
references.

## Exact post-approval commands

These commands are documented but were **not run**. They will fail until the
Hugging Face account has accepted FLUX.1-dev access.

DreamO official v1.1 interface on one MIG:

```bash
cd /home/zwu/project/CE6190/personalization-benchmark
source scripts/mig_devices.sh
cd third_party/modern/DreamO
CUDA_VISIBLE_DEVICES="$MIG0" \
  /home/zwu/anaconda3/envs/environment_dreamo/bin/python app.py \
  --version v1.1 --device cuda --quant none
```

UNO single-reference smoke inference on one MIG:

```bash
cd /home/zwu/project/CE6190/personalization-benchmark
source scripts/mig_devices.sh
cd third_party/modern/UNO
CUDA_VISIBLE_DEVICES="$MIG1" \
  /home/zwu/anaconda3/envs/environment_uno/bin/python inference.py \
  --prompt "a photo of a dog on a beach" \
  --image_paths ../../../data/processed/dreambooth/dog2/images/00.jpg \
  --model_type flux-dev-fp8 --offload \
  --width 512 --height 512 --num_steps 25 --guidance 4 --seed 42 \
  --save_path ../../../outputs/modern_extension/smoke/uno_single
```

UNO multi-reference smoke inference uses the same model and two original
subject references:

```bash
cd /home/zwu/project/CE6190/personalization-benchmark
source scripts/mig_devices.sh
cd third_party/modern/UNO
CUDA_VISIBLE_DEVICES="$MIG1" \
  /home/zwu/anaconda3/envs/environment_uno/bin/python inference.py \
  --prompt "a photo of a dog on a beach" \
  --image_paths \
    ../../../data/processed/dreambooth/dog2/images/00.jpg \
    ../../../data/processed/dreambooth/dog2/images/01.jpg \
  --model_type flux-dev-fp8 --offload \
  --width 512 --height 512 --num_steps 25 --guidance 4 --seed 42 \
  --save_path ../../../outputs/modern_extension/smoke/uno_multi
```

Before either command is approved, verify access without downloading the full
checkpoint:

```bash
/home/zwu/anaconda3/envs/environment_dreamo/bin/python - <<'PY'
from huggingface_hub import hf_hub_download
print(hf_hub_download("black-forest-labs/FLUX.1-dev", "model_index.json"))
PY
```

## Readiness update (2026-10-12)

FLUX.1-dev access was subsequently granted and the small-file access test
passed. DreamO int8/offload, UNO single-reference fp8/offload, and UNO
two-reference fp8/offload each completed a one-image compatibility test. The
DreamO bf16 path exceeded the practical capacity of the shared 32.5-GiB MIG,
so the validated extension configuration is DreamO int8/offload. Measured
details and the preserved failure diagnosis are in
`results/system/modern_smoke_test_report.md`.

The environment and compatibility gates are now satisfied. The full modern
generation remains a long-running step and must be launched only after
explicit approval of the validated precision configurations.
