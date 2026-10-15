# Ablation environment and runtime report

- Python: 3.11.17
- torch: 2.7.1+cu126
- torch CUDA: 12.6
- diffusers: 0.35.1
- transformers: 4.49.0
- accelerate: 1.4.0
- peft: 0.17.1
- Git commit: 0bd1d7d59d86e900f6f2ada7e3e47abf343069c0

## Measured task runtimes

| Task | Runtime (seconds) |
|---|---:|
| A | 644 |
| B | 1685 |
| C | 8058 |
| D | 169 |
| E | 774 |
| F | 867 |

## GPU information

```text
GPU 0: NVIDIA H200 NVL (UUID: GPU-eb8d3256-1cc8-e5eb-67cf-b13f75123901)
  MIG 1g.35gb     Device  0: (UUID: MIG-eb10c6cf-0474-58a4-8e0f-7afffc48b4b3)
  MIG 1g.35gb     Device  1: (UUID: MIG-3dc20912-6ea3-5b44-81e4-98481ef4c763)
  MIG 1g.35gb     Device  2: (UUID: MIG-b0e5ab9b-29b6-5fa3-9664-7eb91cf3ef84)
  MIG 1g.35gb     Device  3: (UUID: MIG-9d50c02d-6036-5ad7-8817-302b8d26d7ef)
GPU 1: NVIDIA H200 NVL (UUID: GPU-4d384add-c3a7-3f4c-9c55-94bf51de51f3)
  MIG 1g.35gb     Device  0: (UUID: MIG-8dceb4a0-261d-5b2f-9bb9-ef41f9f65765)
  MIG 1g.35gb     Device  1: (UUID: MIG-a4f5c260-0bbc-52cb-b5b9-98c4b9ac3251)
  MIG 1g.35gb     Device  2: (UUID: MIG-b02034a8-d6ef-57d6-a42c-305979b1bc94)
  MIG 1g.35gb     Device  3: (UUID: MIG-8c70a6f7-6e84-54fc-a316-08ee1d2e57f7)
```

## Git status (abbreviated)

```text
M configs/prompts.yaml
 M results/RESULTS.md
 M results/metrics/manual_overfitting_annotation.csv
 M results/report_tables/main_results.csv
 M results/report_tables/manual_overfitting.csv
 M scripts/check_training_status.sh
 M scripts/evaluate_all.sh
 M scripts/generate_all.sh
 M scripts/jobs/train_dreambooth_pilot.sh
 M scripts/jobs/train_lora_pilot.sh
 M scripts/visualize_all.sh
 M src/data/generate_class_images.py
 M src/inference/smoke_checkpoint_load.py
?? configs/main_prompts.yaml
?? configs/main_subjects.yaml
?? configs/modern_extension.yaml
?? data/processed/ablations/
?? data/processed/customconcept101/
?? data/processed/dreambooth/bear_plushie/
?? data/processed/dreambooth/cat/
?? data/processed/dreambooth/colorful_sneaker/
?? data/processed/dreambooth/robot_toy/
?? data/processed/dreambooth/teapot/
?? logs/
?? models/
?? results/ABLATION_README.md
?? results/RESULTS_SUMMARY.md
?? results/ScoringRules.docx
?? results/exports/
?? results/figures/ablation/
?? results/figures/comparison/background_leakage_inspection.pdf
?? results/figures/comparison/background_leakage_inspection.png
?? results/figures/comparison/dog2_four_method_grid.pdf
?? results/figures/comparison/dog2_four_method_grid.png
?? results/figures/comparison/dog2_four_method_grid_seed123.pdf
?? results/figures/comparison/dog2_four_method_grid_seed123.png
?? results/figures/comparison/dog2_four_method_grid_seed42.pdf
?? results/figures/comparison/dog2_four_method_grid_seed42.png
?? results/figures/comparison/dog2_four_method_grid_seed456.pdf
?? results/figures/comparison/dog2_four_method_grid_seed456.png
?? results/figures/comparison/dog2_four_method_grid_seed789.pdf
?? results/figures/comparison/dog2_four_method_grid_seed789.png
?? results/figures/comparison/failure_case_grid.pdf
?? results/figures/comparison/failure_case_grid.png
?? results/figures/comparison/grid_generation_summary.csv
?? results/figures/comparison/pilot_clip_alignment.pdf
?? results/figures/comparison/pilot_clip_alignment.png
?? results/figures/comparison/pilot_dino_fidelity.pdf
?? results/figures/comparison/pilot_dino_fidelity.png
?? results/figures/comparison/pilot_dino_vs_clip_scatter.pdf
?? results/figures/comparison/pilot_dino_vs_clip_scatter.png
?? results/figures/comparison/pilot_performance_by_prompt_category.pdf
?? results/figures/comparison/pilot_performance_by_prompt_category.png
?? results/figures/compute/
?? results/figures/debug/
?? results/figures/main_artifact_validation/
?? results/figures/main_experiment/
?? results/figures/main_subject_selection/
?? results/figures/modern_extension/
?? results/figures/smoke/lora_fp16_fix.json
?? results/figures/smoke/lora_fp16_fix.png
?? results/lora_ablation_summary.md
?? results/main_experiment_subject_audit.md
?? results/main_experiment_summary.md
?? results/manual_annotation_protocol.md
?? results/manual_annotation_scoring_rules.md
?? results/manual_overfitting_analysis.md
?? results/manual_overfitting_annotation_claude_provisional.csv
?? results/metrics/all_runs_per_image.csv
?? results/metrics/face_corruption_checks.csv
?? results/metrics/face_corruption_checks_summary.csv
?? results/metrics/ip_diagnostics.csv
?? results/metrics/ip_diagnostics_summary.csv
?? results/metrics/ip_scale0_sanity.csv
?? results/metrics/ip_scale_ablation_summary.csv
?? results/metrics/lora_ablation_manual_annotation.csv
?? results/metrics/lora_ablation_prompt_alignment.csv
?? results/metrics/lora_ablation_subject_fidelity.csv
?? results/metrics/main_comparison_with_ip_scales.csv
?? results/metrics/main_compute_cost.csv
?? results/metrics/main_evaluation.log
?? results/metrics/main_evaluation_scale03.log
?? results/metrics/main_evaluation_with_base_model.log
?? results/metrics/main_manual_overfitting_annotation.csv
?? results/metrics/main_manual_overfitting_annotation_provisional_clean.csv
?? results/metrics/main_prompt_alignment.csv
?? results/metrics/main_subject_fidelity.csv
?? results/metrics/manual_overfitting_annotation_ip_scale03_blank.csv
?? results/metrics/pilot_benchmark_metrics.txt
?? results/metrics/pilot_compute_cost.csv
?? results/metrics/pilot_prompt_alignment.csv
?? results/metrics/pilot_subject_fidelity.csv
?? results/metrics/prior_preservation_ablation.csv
?? results/metrics/prior_preservation_ablation_summary.csv
?? results/metrics/rank_ablation.csv
?? results/metrics/rank_ablation_summary.csv
?? results/metrics/refcount_ablation.csv
?? results/metrics/refcount_ablation_summary.csv
?? results/metrics/steps_ablation.csv
?? results/metrics/steps_ablation_summary.csv
?? results/pilot_prompt_protocol.csv
?? results/pilot_summary.csv
?? results/pilot_validation_report.md
?? results/report_tables/MANUAL_OVERFITTING_COMPATIBILITY.md
?? results/report_tables/category_analysis.csv
?? results/report_tables/compute_comparison.csv
?? results/report_tables/lora_ablation_all_variants.csv
?? results/report_tables/lora_rank_ablation.csv
?? results/report_tables/lora_training_steps.csv
?? results/report_tables/main_generation_prompt_catalog.csv
?? results/report_tables/main_generation_prompt_catalog.md
?? results/report_tables/main_generation_prompt_catalog_wide.csv
?? results/report_tables/main_results_with_manual_overfitting.csv
?? results/report_tables/main_subject_conditioning_catalog.csv
?? results/report_tables/manual_overfitting/
?? results/report_tables/pilot_compute_table.csv
?? results/report_tables/pilot_compute_table.md
?? results/summary_by_method_and_object.csv
?? results/system/ablation_duplicate_hashes.csv
?? results/system/ablation_environment_report.md
?? results/system/ablation_validation_counts.csv
?? results/system/ablation_validation_report.md
?? results/system/clip_neutral_prompt_audit.md
?? results/system/ip_adapter_scale03_regeneration_report.md
?? results/system/main_artifact_validation.log
?? results/system/main_artifact_validation.md
?? results/system/main_evaluation_validation.md
?? results/system/main_training_completion_audit.csv
?? results/system/main_training_failure_audit.md
?? results/system/modern_environment_report.md
?? results/system/modern_environments/
?? results/system/modern_model_audit.md
?? results/system/modern_smoke_test_report.md
?? results/system/pre_ablation_main_evaluation_sha256.txt
?? results/system/status_reconciliation.md
?? results/system/task_a_duplicate_hash_audit.csv
?? results/system/training_remaining_status.md
?? scripts/check_modern_extension_status.sh
?? scripts/jobs/continue_lora_steps_1200.sh
?? scripts/jobs/debug_dreambooth_boundary_20steps.sh
?? scripts/jobs/debug_lora_boundary_20steps.sh
?? scripts/jobs/generate_dreamo_extension.sh
?? scripts/jobs/generate_main_subject.sh
?? scripts/jobs/generate_uno_extension.sh
?? scripts/jobs/prepare_main_class_images.sh
?? scripts/jobs/repro_dreambooth_20steps.sh
?? scripts/jobs/repro_lora_20steps.sh
?? scripts/jobs/repro_lora_mixed_precision_20steps.sh
?? scripts/jobs/run_main_subject_lane.sh
?? scripts/jobs/run_modern_extension_parallel.sh
?? scripts/jobs/run_task_a_ip_ablation.sh
?? scripts/jobs/run_task_b_rank_ablation.sh
?? scripts/jobs/run_task_c_steps_ablation.sh
?? scripts/jobs/run_task_d_face_corruption.sh
?? scripts/jobs/run_task_e_refcount_ablation.sh
?? scripts/jobs/run_task_f_prior_ablation.sh
?? scripts/jobs/train_lora_rank_ablation.sh
?? scripts/jobs/train_lora_variant.sh
?? scripts/jobs/train_main_subject.sh
?? scripts/jobs/train_task_c_steps_lane.sh
?? scripts/launch_main_class_images.sh
?? scripts/launch_main_evaluation.sh
?? scripts/launch_main_training.sh
?? scripts/main_training_schedule.md
?? scripts/run_base_model_main.sh
?? scripts/run_ip_adapter_scale03_main.sh
?? scripts/run_lora_ablation_generation.sh
?? scripts/run_main_evaluation_parallel.sh
?? scripts/upload_seed_grids_to_gdrive.sh
?? src/data/prepare_main_subjects.py
?? src/data/prepare_refcount_ablation.py
?? src/evaluation/analyze_manual_overfitting.py
?? src/evaluation/build_all_runs_master.py
?? src/evaluation/create_blank_scale03_annotations.py
?? src/evaluation/evaluate_experiment_metrics.py
?? src/evaluation/evaluate_ip_diagnostics.py
?? src/evaluation/evaluate_lora_ablation.py
?? src/evaluation/evaluate_main_benchmark.py
?? src/evaluation/evaluate_pilot_benchmark.py
?? src/evaluation/evaluate_steps_ablation.py
?? src/evaluation/summarize_manual_overfitting.py
?? src/evaluation/validate_ablation_deliverables.py
?? src/inference/generate_face_corruption_checks.py
?? src/inference/generate_ip_adapter_ablation.py
?? src/inference/generate_lora_ablation.py
?? src/inference/generate_lora_variant.py
?? src/inference/generate_main_benchmark.py
?? src/inference/generate_modern_dreamo.py
?? src/inference/generate_modern_uno.py
?? src/inference/generate_pilot_benchmark.py
?? src/inference/generate_steps_checkpoint.py
?? src/inference/smoke_dreamo.py
?? src/inference/smoke_uno.py
?? src/inference/stage_task_a_existing.py
?? src/inference/validate_main_artifacts.py
?? src/utils/
?? src/visualization/build_ablation_report.py
?? src/visualization/build_face_corruption_grids.py
?? src/visualization/build_lora_ablation_report.py
?? src/visualization/build_main_qualitative_grids.py
?? src/visualization/build_main_report.py
?? src/visualization/build_modern_annotation_export.py
?? src/visualization/build_pilot_report.py
?? src/visualization/build_steps_grids.py
?? src/visualization/build_task_a_review_grids.py
?? src/visualization/main_subject_selection.py
?? src/visualization/package_ablation_review.py
```
