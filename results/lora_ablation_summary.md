# LoRA Rank and Training-Step Ablation

## Scope

This controlled ablation uses the `dog2` subject, SD1.5, prior preservation,
200 class images, FP16, seed 42, six prompts, and four fixed generation seeds.
Only LoRA rank or training duration changes. The step comparison follows one
rank-16 trajectory: checkpoint 400, checkpoint 800, and a continuation from
checkpoint 800 to checkpoint 1200.

CLIP uses the clean category-level evaluation prompt for every image. It does
not contain `sks` or a learned placeholder token.

## Rank sensitivity at 800 steps

| Rank | DINO mean ± SD | CLIP mean ± SD | Trainable parameters | Checkpoint MiB |
|---:|---:|---:|---:|---:|
| 4 | 0.7304 ± 0.1511 | 0.2651 ± 0.0235 | 797,184 | 3.08 |
| 16 | 0.7134 ± 0.1395 | 0.2622 ± 0.0292 | 3,188,736 | 12.20 |
| 32 | 0.7495 ± 0.1418 | 0.2482 ± 0.0372 | 6,377,472 | 24.36 |

## Training-step trajectory at rank 16

| Steps | DINO mean ± SD | DINO max-reference mean ± SD | CLIP mean ± SD |
|---:|---:|---:|---:|
| 400 | 0.7649 ± 0.0827 | 0.8023 ± 0.0794 | 0.2627 ± 0.0222 |
| 800 | 0.7134 ± 0.1395 | 0.7547 ± 0.1398 | 0.2622 ± 0.0292 |
| 1200 | 0.6414 ± 0.1137 | 0.6956 ± 0.1180 | 0.2418 ± 0.0310 |

From 400 to 800 to 1200 steps, whole-image mean-reference DINO similarity was
0.7649, 0.7134, and
0.6414; neutral-prompt CLIP was
0.2627, 0.2622, and
0.2418. These automatic measurements do not show a
monotonic increase in training-image similarity with longer training; both
metrics fall at 1200 steps in this single-subject trajectory.

## RQ2 interpretation boundary

DINO similarity is supporting evidence, not a detector of background leakage,
pose copying, or composition copying. Therefore these results do **not** yet
establish whether longer training worsens qualitative overfitting. The review
grid and `results/metrics/lora_ablation_manual_annotation.csv` are ready for
blind human scoring on the existing 120 images. A manual-overfitting curve
must be produced only after those ratings are supplied.

## Limitations

- This is a one-subject ablation and is not a cross-dataset conclusion.
- Rank jobs ran concurrently on separate MIG instances that share a physical
  GPU; their measured wall times should not be interpreted as pure rank-scaling
  benchmarks.
- Whole-image DINO can respond to both subject and background similarity.
