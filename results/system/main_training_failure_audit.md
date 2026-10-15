# Main Training Failure Audit

Audit time: 2026-10-10 (America/Los_Angeles)

Scope: the seven `FAILED` markers reported for the main experiment. This was a
read-only diagnosis of training artifacts and scripts. No failed job was
restarted, no running job was interrupted, and no training output was deleted
or changed.

## Finding

All seven markers have the same root cause: a post-training storage-policy
check in `scripts/jobs/train_main_subject.sh`, not a model-training exception.

The wrapper intentionally exits with code 10 when it finds a top-level
`checkpoint-*` directory. The generated commands omitted
`--checkpointing_steps` under the mistaken assumption that omission disables
intermediate checkpoints. In the checked-in Diffusers scripts, however,
`--checkpointing_steps` defaults to 500 for Full DreamBooth, DreamBooth-LoRA,
and Textual Inversion. Consequently, each affected child process trained to its
requested final step and saved its final artifact, then the wrapper found the
default checkpoints and emitted:

```text
ERROR: storage policy violation: intermediate checkpoint found
```

The wrapper then exited with code 10 and its EXIT trap created `FAILED`.
There is no `CalledProcessError` or earlier child-process traceback in any of
the seven training logs.

Relevant implementation evidence:

- `scripts/jobs/train_main_subject.sh:180-182` rejects any `checkpoint-*`
  directory and exits 10.
- `third_party/diffusers/examples/dreambooth/train_dreambooth.py:347-349`
  sets `checkpointing_steps` to 500 by default.
- `third_party/diffusers/examples/dreambooth/train_dreambooth_lora.py:355-357`
  sets the same default, so running LoRA jobs are exposed to the same marker
  failure even though none was in the supplied failed-marker list.
- `third_party/diffusers/examples/textual_inversion/textual_inversion.py:411-413`
  sets the same default.

## Failed-job evidence

| Job | Method | Subject | Child completion evidence | First real failure |
|---|---|---|---|---|
| `cat_db_s800_seed42` | Full DreamBooth | cat | Reached 800/800; final `model_index.json` and UNet weights exist | No child exception; wrapper storage-policy error, exit 10 |
| `pet_dog1_db_s800_seed42` | Full DreamBooth | pet_dog1 | Reached 800/800; final `model_index.json` and UNet weights exist | No child exception; wrapper storage-policy error, exit 10 |
| `teapot_db_s800_seed42` | Full DreamBooth | teapot | Reached 800/800; final `model_index.json` and UNet weights exist | No child exception; wrapper storage-policy error, exit 10 |
| `cat_ti_s3000_seed42` | Textual Inversion | cat | Reached 3000/3000; final `learned_embeds.safetensors` exists | No child exception; wrapper storage-policy error, exit 10 |
| `pet_dog1_ti_s3000_seed42` | Textual Inversion | pet_dog1 | Reached 3000/3000; final `learned_embeds.safetensors` exists | No child exception; wrapper storage-policy error, exit 10 |
| `robot_toy_ti_s3000_seed42` | Textual Inversion | robot_toy | Reached 3000/3000; final `learned_embeds.safetensors` exists | No child exception; wrapper storage-policy error, exit 10 |
| `teapot_ti_s3000_seed42` | Textual Inversion | teapot | Reached 3000/3000; final `learned_embeds.safetensors` exists | No child exception; wrapper storage-policy error, exit 10 |

DreamBooth created `checkpoint-500` in each affected directory. Textual
Inversion created `checkpoint-500`, `checkpoint-1000`, `checkpoint-1500`,
`checkpoint-2000`, `checkpoint-2500`, and `checkpoint-3000`. These resumable
training-state directories explain both the marker failure and the unexpectedly
large output sizes (about 15 GiB per failed DreamBooth run and 4.5 GiB per
failed TI run).

## Log and command locations

For each job, the exact child command is in its `training_command.txt`, child
stdout/stderr is in `training.log`, and the wrapper summary is in
`failure_summary.txt` under its output directory:

- `outputs/training/dreambooth/cat_db_s800_seed42/`
- `outputs/training/dreambooth/pet_dog1_db_s800_seed42/`
- `outputs/training/dreambooth/teapot_db_s800_seed42/`
- `outputs/training/textual_inversion/cat_ti_s3000_seed42/`
- `outputs/training/textual_inversion/pet_dog1_ti_s3000_seed42/`
- `outputs/training/textual_inversion/robot_toy_ti_s3000_seed42/`
- `outputs/training/textual_inversion/teapot_ti_s3000_seed42/`

The wrapper-level first errors are recorded in:

- `logs/main_training/cat.lane.log` at lines 208 and 384
- `logs/main_training/pet_dog1.lane.log` at lines 208 and 384
- `logs/main_training/robot_toy.lane.log` at line 208
- `logs/main_training/teapot.lane.log` at lines 208 and 384

## Subject-data integrity

| Subject | Dataset | Reference count | Dimensions | Corrupt images |
|---|---|---:|---|---:|
| cat | DreamBooth | 5 | 1277-1282 px square | 0 |
| pet_dog1 | CustomConcept101 | 8 | 956-1298 px; one 965x968 image | 0 |
| robot_toy | DreamBooth | 5 | 1440x1440 | 0 |
| teapot | DreamBooth | 5 | 703-1282 px square | 0 |

All files opened and verified with Pillow. Formats and extensions are supported.

## Textual Inversion settings

| Subject | Placeholder token | Initializer token | Learnable property | Initializer is one tokenizer token? |
|---|---|---|---|---|
| cat | `<sks-cat>` | `cat` | object | Yes (ID 2368) |
| pet_dog1 | `<sks-dog-pet1>` | `dog` | object | Yes (ID 1929) |
| robot_toy | `<sks-robot-toy>` | `toy` | object | Yes (ID 5988) |
| teapot | `<sks-teapot>` | `teapot` | object | Yes (ID 40749) |

Each placeholder is absent from the base tokenizer before training, as
required. No TI token or learnable-property error was found.

## DreamBooth prior-preservation inputs

| Subject | Class directory | Class prompt | Instance prompt | Readable images | Non-images / zero-byte / corrupt |
|---|---|---|---|---:|---:|
| cat | `data/processed/class_images/cat_sd15_200` | `a photo of a cat` | `a photo of sks cat` | 200 | 0 |
| pet_dog1 | `data/processed/class_images/dog_sd15_200` | `a photo of a dog` | `a photo of sks dog` | 200 | 0 |
| teapot | `data/processed/class_images/teapot_sd15_200` | `a photo of a teapot` | `a photo of sks teapot` | 200 | 0 |

Each class directory is image-only, readable, and mode 0775. Prior
preservation was enabled and the class inputs are not the cause.

## Resource checks

- No child log contains CUDA OOM, `SIGKILL`, `Killed`, disk-full, permission,
  file-not-found, or Python traceback evidence.
- Root filesystem: 7.0 TiB total, 4.9 TiB free (26% used) during the audit.
- All seven active MIG workloads visible during the audit had approximately
  33,280 MiB each and were operating normally; one MIG remained idle.
- Output directories are writable by the experiment user.

## Classification and impact

Classification: **common pipeline bookkeeping/storage-policy defect**. It is
not subject-specific and does not indicate failed optimization.

The final artifacts from these seven runs appear complete by step count and
required-file checks. They should be load-tested before their markers are
reconciled, but retraining is not justified by the evidence in this audit.
The completed dog2 pilot is unaffected because it used separate validated
pilot scripts and already has successful artifacts.

The defect also applies prospectively to main-experiment LoRA and other running
or queued main jobs because their Diffusers scripts use the same default. The
running jobs should be allowed to continue: interrupting them would discard
valid progress, while the known failure occurs only after final artifact save.
They may receive false `FAILED` markers until the wrapper is corrected.

## Minimal proposed fix (not applied)

1. In `scripts/jobs/train_main_subject.sh`, explicitly pass a checkpoint
   interval greater than the corresponding total training steps:
   `--checkpointing_steps=801` for DreamBooth and LoRA, and
   `--checkpointing_steps=3001` for Textual Inversion. This preserves every
   training hyperparameter while preventing intermediate resume-state saves.
2. Keep the existing post-run assertion that rejects `checkpoint-*`; it then
   remains a useful storage-policy guard.
3. Syntax-check and run all 21 `--validate-only` preflights. Do not relaunch
   these seven completed optimizations.
4. Load-test each final DreamBooth pipeline and TI embedding. Only after those
   checks pass, remove the known intermediate resume-state directories and
   reconcile `FAILED` to `SUCCESS`. Checkpoint removal requires explicit
   approval because it is destructive; final models, embeddings, configs,
   logs, validation outputs, and reproducibility records must remain.
5. Apply the same explicit interval to LoRA before any new LoRA invocation.
   Running LoRA child commands cannot be altered in place; their final weights
   should be handled using the same validate/reconcile procedure if they
   receive the same false marker.

No fix and no cleanup were performed as part of this audit.
