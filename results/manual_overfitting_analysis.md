# Provisional Manual Overfitting Analysis

> **Stale IP-Adapter ratings:** the IP-Adapter Plus images were replaced on
> 2026-10-12 using conditioning scale 0.3. The IP-Adapter rows and any
> aggregates or figures containing them in this provisional analysis describe
> the previous scale-0.7 images and must not be reported for the current main
> benchmark. DreamBooth, DreamBooth-LoRA, and Textual Inversion images were not
> changed. Reannotation of the 192 current IP-Adapter images is required.
> The subsequently added 192-image non-personalized base-model control is also
> absent from this four-method annotation set and requires new ratings for a
> five-method manual comparison.

## Status and interpretation

This analysis covers all 768 classical main-experiment outputs. Scores are ordinal: 0 = none, 1 = partial/loose resemblance, and 2 = near-copy. Higher scores are worse and indicate more training-image copying; they do not measure output quality or subject fidelity.

The supplied annotations are explicitly provisional: one `claude_provisional` annotator, non-blind (`blind=no`). They are suitable for exploratory analysis and pipeline validation, not as final human-study evidence. Inter-annotator agreement cannot be calculated until at least two independent blind human ratings exist.

## Data validation

The table has 768 rows with no missing values, duplicate image keys, or invalid scores. It matches all 768 canonical output metadata records one-to-one across 8 subjects, 4 methods, 6 prompts, and 4 seeds. The supplied 32-row summary agrees with recomputation up to two-decimal rounding (maximum absolute difference 0.005).

## Method-level results

| Method | Background leakage | Probe leakage (P1/P2/P4) | Pose copying | Composition copying |
|---|---:|---:|---:|---:|
| DreamBooth | 0.896 ± 0.921 | 0.573 ± 0.805 | 1.344 ± 0.735 | 1.224 ± 0.777 |
| DreamBooth-LoRA | 0.641 ± 0.813 | 0.260 ± 0.508 | 1.042 ± 0.693 | 0.885 ± 0.707 |
| Textual Inversion | 0.271 ± 0.501 | 0.188 ± 0.392 | 0.219 ± 0.414 | 0.161 ± 0.369 |
| IP-Adapter Plus | 2.000 ± 0.000 | 2.000 ± 0.000 | 2.000 ± 0.000 | 2.000 ± 0.000 |

## Caveats

- The scoring rules state that IP-Adapter Plus outputs were judged as near-copies of one reference across prompts, producing 2/2/2 ratings. This is evidence under this provisional rubric of over-strong reference conditioning at the tested scale (0.7), not a general verdict on IP-Adapter.
- Background leakage is most directly probed by P1, P2, and P4. P0, P3, and P5 remain reported but should not replace the probe-only result.
- The three axes remain separate primary outcomes. Their arithmetic mean is included only as a descriptive plotting aid and is not a validated composite score.
- Automatic DINOv2/CLIP metrics do not establish background leakage or copying; joint plots are supporting comparisons only.
- Final reporting should add at least two blind human annotators and an agreement statistic before drawing strong overfitting conclusions.

