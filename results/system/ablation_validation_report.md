# Ablation validation report

| experiment_directory | expected_images | actual_png | actual_json | corrupt_png | metadata_schema_errors | duplicate_hash_groups | count_status | integrity_status |
|---|---|---|---|---|---|---|---|---|
| task_a | 972 | 972 | 972 | 0 | 0 | 48 | PASS | PASS |
| task_c_steps | 1296 | 1296 | 1296 | 0 | 0 | 0 | PASS | PASS |
| task_b_rank | 180 | 180 | 180 | 0 | 0 | 0 | PASS | PASS |
| task_e_refcount | 144 | 144 | 144 | 0 | 0 | 0 | PASS | PASS |
| task_f_prior | 48 | 48 | 48 | 0 | 0 | 0 | PASS | PASS |
| task_d_face_corruption | 96 | 96 | 96 | 0 | 0 | 0 | PASS | PASS |

- Main-run hash snapshot entries checked: 1920
- Main-run hash check exit code: 0
- Main-run hash failures: 0
- Master metric rows: 3696
- Master finite-metric status: PASS

Exact duplicate hashes are listed separately. Base/no-personalization and scale-zero variants can produce expected cross-subject duplicates when two subjects share the same class prompt and seed; these are reported rather than silently removed.
