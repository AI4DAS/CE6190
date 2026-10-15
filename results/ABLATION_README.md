# Ablation Run Order and Runtime Tracking

Priority order requested for the time-boxed run:

1. Task A: IP-Adapter diagnosis, full scale-0.3 validation, Base control,
   scale/component/reference-use ablations.
2. Task C: training-step trajectories for LoRA, DreamBooth, and Textual Inversion.
3. Task B: LoRA rank ablation.
4. Task E: reference-image-count ablation.
5. Task F: prior-preservation ablation.
6. Task D: inference-only corruption checks.
7. Task G: unified metrics, tables, plots, annotation templates, and validation.

Runtime estimates are printed from measured existing per-image/training runtimes
before each long stage. Actual runtimes are recorded after completion; estimates
are never reported as measurements.

All new outputs use variant-specific directories under `outputs/experiments/` or
`outputs/ablations/`. Existing main-run paths are read-only for this workflow.
Generation is resumable at the PNG/JSON-pair level. Training jobs use unique run
directories and completion markers.

## Completed run order and measured runtime

| Order | Task | Runtime (seconds) | Primary output count |
|---:|---|---:|---:|
| 1 | A: IP-Adapter diagnostics/controls | 644 | 972 |
| 2 | C: training-step ablation | 8058 | 1296 |
| 3 | B: LoRA rank ablation | 1685 | 180 |
| 4 | E: reference-count ablation | 774 | 144 |
| 5 | F: prior-preservation ablation | 867 | 48 |
| 6 | D: corruption inference checks | 169 | 96 |

The runtime column is parsed from completed task logs, not estimated. Metric and
figure post-processing is recorded separately in the environment and validation
reports.

## Reproduction entry points

Run each long task in the stated order on the configured independent MIGs:

```bash
tmux new-session -d -s ce6190_task_a \
  'cd /home/zwu/project/CE6190/personalization-benchmark && bash scripts/jobs/run_task_a_ip_ablation.sh'
tmux new-session -d -s ce6190_task_c \
  'cd /home/zwu/project/CE6190/personalization-benchmark && bash scripts/jobs/run_task_c_steps_ablation.sh'
tmux new-session -d -s ce6190_task_b \
  'cd /home/zwu/project/CE6190/personalization-benchmark && bash scripts/jobs/run_task_b_rank_ablation.sh'
tmux new-session -d -s ce6190_task_e \
  'cd /home/zwu/project/CE6190/personalization-benchmark && bash scripts/jobs/run_task_e_refcount_ablation.sh'
tmux new-session -d -s ce6190_task_f \
  'cd /home/zwu/project/CE6190/personalization-benchmark && bash scripts/jobs/run_task_f_prior_ablation.sh'
tmux new-session -d -s ce6190_task_d \
  'cd /home/zwu/project/CE6190/personalization-benchmark && bash scripts/jobs/run_task_d_face_corruption.sh'
```

Then build consolidated metrics, figures, validation, and the review archive:

```bash
source scripts/mig_devices.sh
CUDA_VISIBLE_DEVICES="$MIG3" python src/evaluation/build_all_runs_master.py
python src/visualization/build_ablation_report.py
python src/evaluation/validate_ablation_deliverables.py
python src/visualization/package_ablation_review.py
```

The packaging command intentionally refuses to overwrite an existing archive.
