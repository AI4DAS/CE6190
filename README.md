# Benchmarking Subject-Driven Image Personalization

Reproducible Stable Diffusion 1.5 benchmark of Textual Inversion, full
DreamBooth, DreamBooth-LoRA, and IP-Adapter Plus. The repository also contains
the controlled rank, training-step, reference-count, prior-preservation, and
adapter-scale ablations used in the project. DreamO and UNO are implemented as
separately labelled modern-backbone extensions rather than controlled SD1.5
comparisons.

The release is source-only. It does **not** contain datasets, model weights,
personalized checkpoints, generated-image collections, logs, or third-party
repositories. The small `results/report_figures/` directory contains only the
six figure assets used in the final report. See `PACKAGE_CONTENTS.md` for the
complete inclusion/exclusion policy.

## Layout

- `configs/`: subjects, prompts, methods, Accelerate, and extension settings.
- `scripts/`: setup, download, training, generation, evaluation, and ablation
  entry points.
- `src/data/`: deterministic dataset preparation and class-image generation.
- `src/inference/`: SD1.5 and modern-extension inference pipelines.
- `src/evaluation/`: DINOv2, neutral-prompt CLIP, copying, cost, and ablation
  evaluation.
- `src/visualization/`: contact sheets, comparison grids, and plots.
- `results/report_figures/`: curated report result figures only.

All generated artifacts are written below the ignored `data/`, `models/`,
`outputs/`, `logs/`, and non-whitelisted `results/` paths.

## Environment and inputs

The validated environment uses Python 3.11, PyTorch with CUDA, Diffusers
0.35.1, Transformers, Accelerate, and PEFT. Create it and fetch the pinned
Diffusers examples:

```bash
bash scripts/setup_env.sh
conda activate personalization-benchmark
bash scripts/inspect_gpu.sh
```

Download the official DreamBooth and CustomConcept101 sources, deterministically
prepare all eight configured non-human subjects, and download the SD1.5 and
IP-Adapter Plus checkpoints:

```bash
bash scripts/download_datasets.sh
bash scripts/download_models.sh
```

`src/data/prepare_main_subjects.py` validates the expected image count, opens
every image with Pillow, renames files by sorted source order, and records the
source-to-prepared mapping locally. No dataset or model is committed.

## Subjects and protocol

The controlled benchmark uses `dog2`, `cat`, `pet_dog1`, `bear_plushie`,
`robot_toy`, `teapot`, `colorful_sneaker`, and `luggage_backpack1`. Subject and
dataset membership are defined in `configs/main_subjects.yaml`. Six semantic
prompt slots and seeds 42, 123, 456, and 789 are defined in
`configs/main_prompts.yaml`.

Fine-tuned methods use all configured reference images. The principal
IP-Adapter Plus condition uses deterministically selected reference image 00 at
scale 0.3. Class images used for prior preservation are training auxiliaries and
are never evaluation references.

## Training

Each job uses one independent visible GPU or MIG device; the code does not use
distributed training across MIG instances. Inspect and source the detected MIG
assignments before launch:

```bash
source scripts/mig_devices.sh
```

Prepare class images, then launch the seven non-pilot subject lanes. The
completed `dog2` pilot uses the three explicit pilot scripts shown below.

```bash
CUDA_VISIBLE_DEVICES="$MIG7" bash scripts/jobs/prepare_main_class_images.sh

CUDA_VISIBLE_DEVICES="$MIG0" bash scripts/jobs/train_dreambooth_pilot.sh
CUDA_VISIBLE_DEVICES="$MIG1" bash scripts/jobs/train_lora_pilot.sh
CUDA_VISIBLE_DEVICES="$MIG2" bash scripts/jobs/train_ti_pilot.sh

bash scripts/launch_main_training.sh
```

Full DreamBooth and LoRA use 800 steps with prior preservation; LoRA uses rank
16. Textual Inversion uses 3,000 steps. The exact prompts, learning rates,
precision, checkpoint paths, validation settings, and storage guards are in the
job scripts and YAML files. Scripts refuse unsafe output collisions and write
success/failure markers.

## Generation and evaluation

After all checkpoints pass their load tests:

```bash
bash scripts/launch_main_evaluation.sh
bash scripts/run_main_evaluation_parallel.sh
```

The generation pipeline writes one JSON sidecar per image with method, subject,
prompt, clean evaluation prompt, seed, checkpoint, scheduler, precision,
latency, GPU, and reference information. Evaluation uses only original subject
references:

- DINOv2 cosine similarity measures subject-reference proximity.
- CLIP image-text similarity uses the clean class-name evaluation prompt, not
  learned trigger tokens.
- Manual 0--2 labels separately describe background, pose, and composition
  carry-over for personalized outputs.
- Training time, peak VRAM, trainable parameters, storage, latency, and data
  requirements remain separate cost measurements.

Create the main report figures from the generated metrics and metadata with:

```bash
bash scripts/visualize_all.sh
python src/visualization/build_main_report.py --project-root .
```

## Ablations

The `scripts/jobs/run_task_*.sh` launchers reproduce the adapter-scale,
LoRA-rank, training-step, reference-count, inference-corruption, and
prior-preservation experiments. They use the predefined subsets, prompts, and
seeds encoded in the scripts; they do not overwrite the main benchmark.

Run `bash -n scripts/*.sh scripts/jobs/*.sh` before scheduling jobs. Long jobs
are designed to be launched explicitly by the user and are not started by setup
or download commands.

## Reproducibility notes

- The official Diffusers v0.35.1 training examples are fetched into the ignored
  `third_party/diffusers/` directory by `scripts/setup_env.sh`.
- Base and adapter revision hashes are recorded locally under
  `results/system/` by the download scripts.
- Experiment outputs record the Git commit, package versions, command, seed,
  GPU/MIG identifier, and runtime where supported.
- The modern DreamO/UNO code expects separately prepared environments and
  upstream repositories; it is not mixed into the controlled SD1.5 ranking.
- The source ZIP is created from the committed Git tree and contains no
  filesystem symbolic links.
