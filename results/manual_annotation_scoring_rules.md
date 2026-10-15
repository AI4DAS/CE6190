# Manual Training-Image Copying Scoring Rules

Source: `results/ScoringRules.docx`. This Markdown version is provided for reproducible analysis; the DOCX remains the authoritative supplied source.

## Interpretation

The annotations measure training-image copying, not image quality or subject fidelity. Higher scores are worse. A corrupted output may still receive high copying scores.

Each generated image is compared against all original reference images for the same subject. Prior-preservation class images are not references.

| Score | Meaning |
|---:|---|
| 0 | No resemblance to a training image on the assessed axis |
| 1 | Partial or loose resemblance, but clearly not a copy |
| 2 | Near-copy of a specific training image on the assessed axis |

## Axes

### Background leakage

- `0`: background follows the prompt or is unrelated to training scenes.
- `1`: some training-scene elements remain while mixed with new content.
- `2`: a training scene is clearly reproduced despite the requested change.

P1, P2, and P4 are the primary leakage probes. P0, P3, and P5 can reveal a training-like default background but should be reported separately from the primary probes.

### Pose copying

- `0`: clearly different pose or viewpoint.
- `1`: broadly similar posture family without matching one image.
- `2`: nearly identical limb arrangement and viewing angle.

For rigid objects, pose means orientation and viewpoint.

### Composition copying

- `0`: different scale, crop, placement, or camera distance.
- `1`: similar generic framing, but not a match.
- `2`: near-identical framing to one training image.

## Procedure and limitations

- Score the three axes independently.
- Compare against the complete reference set, not only the reference displayed in the same figure column.
- Report results across subjects, seeds, and prompt categories.
- The supplied annotations identify IP-Adapter outputs as near-copies at the tested scale; this should be interpreted as possible over-strong reference conditioning, not a general verdict on the method.
- The current ratings are provisional, produced by one non-blind annotator. At least two independent blind human raters are required before calculating inter-annotator agreement or making strong human-evaluation claims.
