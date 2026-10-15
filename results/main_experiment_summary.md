# Main Experiment Summary

## Training completion status

All 24 subject-specific training jobs completed and passed artifact load and generation validation. Twenty `FAILED` markers were false post-training failures caused by the storage guard rejecting Diffusers `checkpoint-*` directories; they were reconciled only after validation. No training rerun was required.

## Methods compared

The non-personalized SD1.5 category-only baseline, Full DreamBooth, DreamBooth-LoRA, Textual Inversion, and IP-Adapter Plus were evaluated under the SD1.5 main protocol. IP-Adapter Plus used one fixed reference image per subject and conditioning scale 0.3.

## Dataset statistics

The benchmark contains 8 non-human subjects and 44 original reference images. Prior class images were excluded from quality evaluation.

## Quantitative results

| Method | DINOv2 mean ± std | CLIP mean ± std |
|---|---:|---:|
| Base SD1.5 (no personalization) | 0.2873 ± 0.2302 | 0.2799 ± 0.0388 |
| DreamBooth | 0.6603 ± 0.1613 | 0.2627 ± 0.0449 |
| DreamBooth-LoRA | 0.6478 ± 0.1291 | 0.2661 ± 0.0436 |
| Textual Inversion | 0.5323 ± 0.2246 | 0.2320 ± 0.0463 |
| IP-Adapter Plus | 0.5578 ± 0.1626 | 0.2669 ± 0.0413 |

## Efficiency analysis

Per-subject training, storage, and measured inference costs are reported in `results/metrics/main_compute_cost.csv`. IP-Adapter Plus has no per-subject optimization but retains shared-adapter and inference costs.

Mean per-subject training time was 10.50 minutes for DreamBooth, 9.47 minutes for LoRA, and 64.52 minutes for Textual Inversion. The base-model control and IP-Adapter Plus required no subject-specific training. Their missing training VRAM is reported as N/A because no per-subject training occurs.

## Failure cases

The failure-case figure uses a predetermined metric rule (lowest DINOv2-plus-CLIP sum), not visual cherry-picking. Manual leakage and copying fields remain blank pending human annotation.

## Limitations

This eight-subject benchmark is broader than the dog2 pilot but remains limited to selected non-human subjects, SD1.5 for the controlled comparison, whole-image DINOv2 similarity, and one fixed prompt/seed protocol. The IP-Adapter result represents the specific Plus checkpoint, scale 0.3, and one fixed reference image per subject; it must not be generalized to every IP-Adapter operating point. Metric differences do not by themselves establish perceptual quality or overfitting.

The dog2 pilot remains a separate completed experiment; these main-experiment aggregates must not be substituted for pilot results.
