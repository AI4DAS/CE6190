# Source release contents

This lightweight release contains the code and configuration needed to rebuild
the benchmark. It intentionally excludes downloaded or generated assets.

Included:

- `configs/`: controlled experiment, prompt, subject, and Accelerate settings.
- `scripts/`: environment setup, downloads, launchers, smoke tests, training,
  generation, evaluation, and ablation orchestration.
- `src/data/`: dataset normalization and class-image preparation.
- `src/inference/`: generation and checkpoint-loading pipelines.
- `src/evaluation/`: DINOv2, CLIP, copying, cost, and ablation evaluation.
- `src/visualization/`: contact sheets, comparison grids, and result figures.
- `requirements.txt` and `environment.yml`: Python environment definitions.
- `results/report_figures/`: the six figure assets used by the final report.

Excluded:

- datasets and processed reference images (`data/`);
- base models, adapters, and personalized checkpoints (`models/`);
- generated image/metadata trees and training artifacts (`outputs/`);
- logs, caches, third-party clones, and local environments;
- report LaTeX, report-writing materials, and audit/review artifacts;
- raw and intermediate result tables not needed for the source release.

The ZIP is built from the committed Git tree, contains regular files only, and
does not contain filesystem symbolic links.
