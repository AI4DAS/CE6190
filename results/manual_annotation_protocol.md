# Manual overfitting annotation protocol

## Purpose

This protocol evaluates visible background leakage, pose copying, and composition
copying in the completed `dog2` pilot. These ratings are qualitative evidence and
must not be inferred from DINOv2 or CLIP scores.

## Materials

- Original references: `data/processed/dreambooth/dog2/images/00.jpg` through
  `05.jpg`.
- Review sheet: `results/figures/comparison/background_leakage_inspection.png`.
- Generated images: `outputs/evaluation/{method}/dreambooth/dog2/{prompt_id}/`.
- Annotation table: `results/metrics/manual_overfitting_annotation.csv`.

The 200 generic prior-preservation dog images are not evaluation references and
must not be consulted when assigning these ratings.

## Procedure

1. Inspect all six original reference images before rating generated images.
2. For each annotation row, open the exact method, prompt ID, and seed image.
3. Compare the generated image with all six references, not only `00.jpg`.
4. Score each of the three criteria independently using only `0`, `1`, or `2`.
5. In `notes`, identify the reference filename and copied feature whenever a
   score is `1` or `2`. Leave subjective fields blank if the image has not yet
   been reviewed.
6. Do not use DINOv2 or CLIP results while rating. This reduces anchoring on the
   automatic metrics.

## Rating definitions

### Background leakage

- `0` — No recognizable training-background feature is reproduced, or the
  background is plausibly explained by the evaluation prompt.
- `1` — A training-background color, texture, or scene element appears similar,
  but the match is partial or ambiguous.
- `2` — A distinctive training-background feature or arrangement is clearly
  reproduced despite a prompt requesting another context.

Do not penalize a beach, snow, kitchen, or city-like background merely because
the prompt requested it. Leakage requires correspondence to a training image.

### Pose copying

- `0` — Pose and viewpoint are novel or only generically dog-like.
- `1` — Head direction, limb placement, or camera angle resembles a reference,
  but multiple plausible poses could produce the match.
- `2` — A distinctive pose and viewpoint clearly reproduce a particular
  reference image.

### Composition copying

- `0` — Subject scale, crop, position, and scene layout are materially different
  from the references.
- `1` — Some layout features match a reference, but the overall composition is
  not a close reproduction.
- `2` — Subject placement, crop, camera framing, and major scene layout clearly
  reproduce a reference composition.

## Quality control

- Complete one uninterrupted pass per annotator when possible.
- If multiple annotators participate, give each person a separate copy of the
  CSV and retain an `annotator_id` column in the combined file.
- Resolve data-entry errors, but do not force consensus on genuine judgment
  differences. Report per-method means and annotator agreement separately.
- The completed single-subject ratings remain pilot evidence and are not a
  cross-dataset conclusion.

## Aggregation

After all 96 rows contain valid integer ratings, run:

```bash
python src/evaluation/summarize_manual_overfitting.py \
  --input results/metrics/manual_overfitting_annotation.csv \
  --output results/metrics/pilot_manual_overfitting_summary.csv
```

The command refuses incomplete or out-of-range ratings and will not overwrite an
existing summary unless `--overwrite` is supplied explicitly.
