# Ablation and Diagnostic Results Summary

## Discovered validated defaults

These values were read from the existing training commands, generation metadata,
and local Diffusers scripts. They are not estimates.

| Setting | DreamBooth | DreamBooth-LoRA | Textual Inversion | IP-Adapter Plus / Base |
|---|---:|---:|---:|---:|
| Default training steps | 800 | 800 | 3000 | 0 |
| Learning rate | 5e-6 | 1e-4 | 5e-4 | N/A |
| Train batch size | 1 | 1 | 1 | N/A |
| Gradient accumulation | 1 | 1 | 4 | N/A |
| Precision | fp16 | fp16 (trainable LoRA weights fp32) | fp16 | fp16 |
| LoRA rank / alpha | N/A | 16 / 16 | N/A | N/A |
| Resolution | 512 | 512 | 512 | 512 |
| Inference steps | 50 | 50 | 50 | 50 |
| Guidance scale | 7.5 | 7.5 | 7.5 | 7.5 |
| Scheduler | PNDMScheduler | PNDMScheduler | PNDMScheduler | PNDMScheduler |
| Negative prompt | none | none | none | none |

The local Diffusers LoRA implementation sets `lora_alpha = rank`; rank-ablation
runs therefore preserve a constant alpha/rank ratio of 1 rather than fixing alpha
to 16.

Checkpoint locations:

- DreamBooth: `outputs/training/dreambooth/{subject}_db_s800_seed42/`
- LoRA: `outputs/training/lora/{subject}_lora_r16_s800_seed42/pytorch_lora_weights.safetensors`
- Textual Inversion: `outputs/training/textual_inversion/{subject}_ti_s3000_seed42/learned_embeds.safetensors`
- IP-Adapter Plus: `models/ip_adapter/IP-Adapter/models/ip-adapter-plus_sd15.bin`
- IP-Adapter plain: `models/ip_adapter/IP-Adapter/models/ip-adapter_sd15.bin`
- Base SD1.5: `models/base/stable-diffusion-v1-5/`

Reference-image audit:

| Subject | References | Deterministic first reference used by single-reference IP-Adapter |
|---|---:|---|
| dog2 | 6 | 00.jpg |
| cat | 5 | 00.jpg |
| pet_dog1 | 8 | 00.jpeg |
| bear_plushie | 5 | 00.jpg |
| robot_toy | 5 | 00.jpg |
| teapot | 5 | 00.jpg |
| colorful_sneaker | 5 | 00.jpg |
| luggage_backpack1 | 5 | 00.jpg |

Reference selection is lexicographic by filename. The main single-reference
IP-Adapter run received only the first file shown above. Prior-preservation class
images are never evaluation references.

## Repository-state caveat

Before this task began, an earlier run had overwritten the original main-run
IP-Adapter scale-0.7 PNG/JSON pairs in place with scale-0.3 outputs. No original
scale-0.7 image/metadata pairs remain locally. This task therefore records the
current main tree hashes before making changes and regenerates scale 0.7 into a
new variant directory; it does not claim that the overwritten files were
preserved.

## Task status

| Task | Status | Evidence / note |
|---|---|---|
| A1 scale 0.3 full run | Done and validated | 192 staged PNG/JSON; main tree unchanged |
| A2 Base baseline | Done and validated | 192 staged PNG/JSON; scale-zero max pixel difference = 0 |
| A3 scale/plain-vs-Plus sweep | Done | 432 subset images represented across six scales and two adapter components |
| A4 reference usage | Done | 36 all-reference-average outputs; single-reference scale-0.3 outputs reused without regeneration |
| A5 diagnostics | Done | `results/metrics/ip_diagnostics.csv` (972 rows) |
| A6 review grids | Done | 8 IP review grids and 32 four-method scale-0.3 grids |
| C steps ablation | Done | 1296 images; 3 subjects; DB 100–800, LoRA 100–1600, TI 250–3000 |
| B rank ablation | Done | 180 images; ranks 4/8/16/32/64; 3 subjects |
| E reference count | Done | 144 images; k=1/3/5; LoRA and IP-Adapter Plus |
| F prior preservation | Done | 48 images; LoRA ON vs OFF; dog2 and teapot |
| D corruption checks | Done | 96 inference-only images; 4 comparison grids |
| G consolidated metrics/reporting | Done | 3696-row master; 12 PNG/PDF report figures; validation PASS |

Measured tables, reproduction commands, validation results, failures, and known
issues will be appended immediately after each task completes.

## Task A — IP-Adapter diagnosis and controls

Status: **done**. Generation runtime was 644 seconds using five independent MIG
lanes. All 972 PNG/JSON pairs passed image readability, required metadata, and
variant-plus-subject duplicate checks.

- Full IP-Adapter Plus scale 0.3: 192 images (validated prior run, staged by hard link).
- Regenerated IP-Adapter Plus scale 0.7: 192 images in a new variant directory.
- Base SD1.5 no-personalization control: 192 images (validated prior run, staged by hard link).
- Plain/Plus scale sweep subset: scales 0.0, 0.2, 0.3, 0.5, 0.7, 1.0.
- Reference use at Plus scale 0.3: deterministic first reference versus the
  elementwise mean of prepared embeddings from every sorted reference image.
- Scale-zero sanity: 72 comparisons; maximum pixel difference = 0 and mean
  absolute pixel difference = 0 for both plain and Plus adapters.

Measured DINOv2 diagnostics are in:

- `results/metrics/ip_diagnostics.csv`
- `results/metrics/ip_diagnostics_summary.csv`
- `results/metrics/ip_scale0_sanity.csv`

The diagnostic table records, rather than interprets, the measured increase in
prompt-to-prompt image similarity and decrease in pixel difference as scale
increases. The no-personalization control contains 24 cross-subject duplicate
hashes because dog2 and pet_dog1 use identical class prompts and seeds with no
subject conditioning. No duplicates occur within a variant-plus-subject group.

Review figures:

- `results/figures/ablation/task_a/ip_scale03_review/` (8 PNG)
- `results/figures/ablation/task_a/four_method_scale03/` (32 PNG)

All review grids are 1865 pixels wide. Exact reproduction command:

```bash
tmux new-session -d -s ce6190_task_a \
  "cd /home/zwu/project/CE6190/personalization-benchmark && \
   bash scripts/jobs/run_task_a_ip_ablation.sh \
   >> logs/ablations/task_a_ip_adapter/launcher.log 2>&1"
```

## Task C — training-step ablation

Status: **done**. The nine independent trajectories completed in 8058 seconds
(2 h 14 min 18 s). Outputs contain exactly 1296 PNG and 1296 JSON files:

- DreamBooth: 288 images, steps 100–800 in increments of 100.
- DreamBooth-LoRA: 576 images, steps 100–1600 in increments of 100; the upper
  endpoint is 2× the validated 800-step default.
- Textual Inversion: 432 images, steps 250–3000 in increments of 250.

Each checkpoint used three subjects, six prompts, and seeds 42/123. All metric
columns are finite. The optional 2× DreamBooth extension was not run because it
was conditional on the time/disk budget; the complete default 100–800 trajectory
was retained. Temporary training-state checkpoint directories were removed only
after their output images were generated and validated.

Measured results:

- `results/metrics/steps_ablation.csv` (1296 per-image rows)
- `results/metrics/steps_ablation_summary.csv` (36 method/step rows)
- `results/figures/ablation/task_c/step_curves.png` and `.pdf`
- `results/figures/ablation/task_c/steps_grid_{method}_{subject}.png` (9 grids)

The measured trajectories are not monotonic. For example, mean LoRA DINOv2 is
0.623354 at step 800 and 0.484150 at step 1600, while its nearest-reference copy
proxy is 0.700821 and 0.551922 at those endpoints. These statements report the
numbers only; they do not assign a causal explanation or a manual overfitting
label.

Exact reproduction command:

```bash
tmux new-session -d -s ce6190_task_c \
  "cd /home/zwu/project/CE6190/personalization-benchmark && \
   bash scripts/jobs/run_task_c_steps_ablation.sh \
   >> logs/ablations/task_c_steps/launcher.log 2>&1"
```

## Task B — LoRA rank ablation

Status: **done**. Ten missing subject/rank adapters were trained and validated;
configuration-identical existing checkpoints were reused. Generation contains
exactly 180 PNG/JSON pairs (3 subjects × 5 ranks × 6 prompts × 2 seeds). The
task runtime was 1685 seconds. Every metric row is finite.

The local official Diffusers script explicitly sets `lora_alpha=args.rank`, so
alpha/rank remains 1 at ranks 4, 8, 16, 32, and 64. No other training or
generation setting was changed.

Measured files:

- `results/metrics/rank_ablation.csv` (180 per-image rows)
- `results/metrics/rank_ablation_summary.csv` (5 rank rows; storage/runtime are
  added by the consolidated report stage from actual files)

Exact reproduction command:

```bash
tmux new-session -d -s ce6190_task_b \
  "cd /home/zwu/project/CE6190/personalization-benchmark && \
   bash scripts/jobs/run_task_b_rank_ablation.sh \
   >> logs/ablations/task_b_rank/launcher.log 2>&1"
```

## Task E — training/reference-image-count ablation

Status: **done**. Five new LoRA checkpoints were trained; teapot k=5 reused
the main checkpoint because its complete five-image reference set and every
training option are identical. IP-Adapter Plus used the arithmetic mean of the
prepared embeddings of the first k lexicographically sorted files. The task
produced exactly 144 PNG/JSON pairs and completed in 774 seconds.

- `results/metrics/refcount_ablation.csv` (144 per-image rows)
- `results/metrics/refcount_ablation_summary.csv` (6 method/count rows)

All metric values are finite. The table reports, for example, the measured
DINOv2 means at k=1/3/5 as 0.492654/0.537161/0.631785 for LoRA and
0.562876/0.599565/0.646051 for IP-Adapter Plus. No conclusion beyond these
two subjects is assigned.

```bash
tmux new-session -d -s ce6190_task_e \
  "cd /home/zwu/project/CE6190/personalization-benchmark && \
   bash scripts/jobs/run_task_e_refcount_ablation.sh \
   >> logs/ablations/task_e_refcount/launcher.log 2>&1"
```

## Task F — prior-preservation component ablation

Status: **done**. Two prior-OFF LoRA checkpoints were trained; prior-ON reused
the main checkpoints. The task produced exactly 48 PNG/JSON pairs in 867
seconds, and all metric values are finite.

- `results/metrics/prior_preservation_ablation.csv` (48 per-image rows)
- `results/metrics/prior_preservation_ablation_summary.csv` (2 rows)

Measured OFF/ON means respectively are DINOv2 0.718680/0.643393, neutral
CLIP-T 0.258851/0.288612, and nearest-reference copy proxy
0.793827/0.715137. These are two-subject measurements, not a general causal
claim.

```bash
tmux new-session -d -s ce6190_task_f \
  "cd /home/zwu/project/CE6190/personalization-benchmark && \
   bash scripts/jobs/run_task_f_prior_ablation.sh \
   >> logs/ablations/task_f_prior/launcher.log 2>&1"
```

## Task D — color-blotch inference checks

Status: **done**. The inference-only task produced 96 PNG/JSON pairs in 169
seconds and four PNG/PDF grids for cat/dog2 × DreamBooth/LoRA.

- `results/metrics/face_corruption_checks.csv`
- `results/metrics/face_corruption_checks_summary.csv`
- `results/figures/ablation/task_d/face_corruption_{method}_{subject}.png`
- `results/figures/ablation/task_d/OBSERVATION_TEMPLATE.md`

At seed 42, lowering CFG from 7.5 to 5.0 visibly removes or strongly reduces
the conspicuous blotches in affected DreamBooth and LoRA slots. LoRA weights
0.8/0.6 also reduce them, but weight 0.6 changes dog2 identity in some outputs.
FP32 alone remains close to FP16 and does not consistently remove the blotches.
This is an inference-sensitivity observation on two subjects and one seed, not
a claim about the training-time cause.

```bash
tmux new-session -d -s ce6190_task_d \
  "cd /home/zwu/project/CE6190/personalization-benchmark && \
   bash scripts/jobs/run_task_d_face_corruption.sh \
   >> logs/ablations/task_d_face_corruption/launcher.log 2>&1"
```

## Task G — consolidated metrics, tables, figures, and validation

Status: **done**. `results/metrics/all_runs_per_image.csv` contains 3696 rows
and the exact requested columns. All DINOv2/nearest-reference/raw-CLIP/neutral-
CLIP values are finite. The report's primary prompt-alignment metric is
`clip_t_neutral`, which uses each sidecar's clean class-name evaluation prompt;
`clip_t_raw` is retained in the master table for audit.

### Main comparison including controls

| Variant | N | DINOv2 mean | Neutral CLIP-T mean | Nearest-train mean |
|---|---:|---:|---:|---:|
| Base (no personalization) | 192 | 0.287342 | 0.279860 | 0.345076 |
| DreamBooth | 192 | 0.660310 | 0.262708 | 0.755513 |
| DreamBooth-LoRA | 192 | 0.647814 | 0.266094 | 0.729614 |
| Textual Inversion | 192 | 0.532250 | 0.231977 | 0.604020 |
| IP-Adapter Plus (s=0.3) | 192 | 0.557757 | 0.266949 | 0.643738 |
| IP-Adapter Plus (s=0.7) | 192 | 0.668124 | 0.201804 | 0.816213 |

This table reports measured means only. It does not treat the Base control as a
personalization method or infer causality from differences.

Key tables:

- `results/metrics/main_comparison_with_ip_scales.csv`
- `results/metrics/ip_scale_ablation_summary.csv`
- `results/metrics/rank_ablation_summary.csv`
- `results/metrics/refcount_ablation_summary.csv`
- `results/metrics/prior_preservation_ablation_summary.csv`
- `results/metrics/manual_overfitting_annotation_ip_scale03_blank.csv` (192
  rows; every subjective score/annotator/blind cell is blank)

Report figures (each PNG + PDF, 200 dpi):

- `results/figures/ablation/task_g/tradeoff_scatter.*`
- `results/figures/ablation/task_g/ip_adapter_scale_curves.*`
- `results/figures/ablation/task_c/step_curves.*`
- `results/figures/ablation/task_g/rank_ablation.*`
- `results/figures/ablation/task_g/reference_count_ablation.*`
- `results/figures/ablation/task_g/prior_preservation_ablation.*`
- `results/figures/ablation/task_g/cost_training_time_and_storage.*`

Validation evidence:

- `results/system/ablation_validation_report.md`: all six experiment-directory
  count and integrity checks PASS.
- 0 corrupt PNG, 0 metadata schema errors, and 0 duplicate hash groups in Tasks
  B/C/D/E/F. Task A lists 48 expected duplicate groups caused by Base and
  scale-zero deterministic controls; none were removed.
- All 1920 pre-task main-run PNG/JSON hashes still match.
- Software/GPU/runtimes: `results/system/ablation_environment_report.md`.
- Git diff: `results/system/ablation_git_diff.patch`.

Consolidated metric/figure reproduction:

```bash
source scripts/mig_devices.sh
CUDA_VISIBLE_DEVICES="$MIG3" python src/evaluation/build_all_runs_master.py
python src/visualization/build_ablation_report.py
python src/evaluation/validate_ablation_deliverables.py
python src/visualization/package_ablation_review.py
```

## Known issues and bounded limitations

- The original pre-task main scale-0.7 IP-Adapter files had already been
  overwritten by an earlier scale-0.3 run. Scale 0.7 was regenerated into a new
  variant directory; it was not reconstructed by editing metadata.
- The Base and scale-zero controls intentionally have cross-subject duplicate
  pixels when subjects share the same class-name prompt and seed.
- DreamBooth was evaluated through its default 800 steps; the optional 2× full-
  model extension was skipped under the same-day time box. LoRA includes 2×.
- Task D visual observations cover two subjects and seed 42 only.
- No subjective annotation scores were fabricated; the scale-0.3 form is blank.
- DreamO and UNO were skipped as explicitly requested for this ablation stage.

Review archive (54 PNG grids, each width ≤2000 px, no datasets or weights):

- `results/exports/ablation_review_grids.zip`
- SHA-256: `98d431080281e132ed6d54ea6559159c173568daeb25f87d1f80bbf36ef1e3c6`
