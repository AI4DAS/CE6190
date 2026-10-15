# Main-experiment qualitative grids

These figures use the 768 validated main-experiment images and their JSON sidecars. The 192 IP-Adapter Plus images were regenerated in place at conditioning scale 0.3 on 2026-10-12; the other 576 images were not regenerated.

## Five-method comparison grids

`per_subject/{subject}_four_method_grid_seed{seed}.{png,pdf}` contains one figure for every subject and seed (`42`, `123`, `456`, and `789`).

- Columns retain the six semantic prompt classifications: identity, background change, context change, attribute change, composition change, and style change.
- Rows identify Reference, Base SD1.5 (no personalization), DreamBooth, DreamBooth-LoRA, Textual Inversion, and IP-Adapter Plus.
- The text below each generated image is the exact `generation_prompt` recovered from that image's JSON metadata.
- The text below each reference is its filename.

The column heading is therefore a prompt **classification**, not the literal model input. Exact model inputs differ by method: DreamBooth and LoRA use `sks`, Textual Inversion uses its learned placeholder token, and the base-model control and IP-Adapter Plus use the clean class prompt.

## Background-leakage review grids

`background_leakage_review/{subject}_background_leakage_seed{seed}.{png,pdf}` focuses on P1 (background change), P2 (context change), and P4 (composition change). All original training references are shown above the generated results to support manual comparison.

These sheets are for human annotation only. They do not automatically assert that leakage, pose copying, or composition copying occurred.

## Prompt records and manifests

- `results/report_tables/main_generation_prompt_catalog.csv`: machine-readable method-specific prompts and clean evaluation prompts.
- `results/report_tables/main_generation_prompt_catalog.md`: report-friendly prompt catalog grouped by subject and semantic condition.
- `results/report_tables/main_generation_prompt_catalog_wide.csv`: one row per subject/condition, with all four methods' literal generation prompts side by side.
- `results/report_tables/main_subject_conditioning_catalog.csv`: the training-time instance/class prompts and Textual Inversion tokens for every subject.
- `per_subject/grid_generation_summary.csv`: all 960 source image records used by the five-method grids.
- `background_leakage_review/background_leakage_review_manifest.csv`: the 480 background/context/composition records awaiting human review.

Regenerate these figures from existing outputs only with:

```bash
python src/visualization/build_main_qualitative_grids.py \
  --project-root /home/zwu/project/CE6190/personalization-benchmark \
  --overwrite
```

Without the explicit `--overwrite` option, the script refuses to overwrite existing figures or catalogs.
