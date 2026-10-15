# Main-Experiment IP-Adapter Plus Regeneration

Date: 2026-10-12

## Change

The 192 main-experiment IP-Adapter Plus image/metadata pairs were explicitly
overwritten using conditioning scale `0.3`. The checkpoint, fixed first
reference image, prompts, seeds, resolution, denoising steps, guidance scale,
scheduler, and FP16 policy were unchanged. No training was run. The 576
DreamBooth, DreamBooth-LoRA, and Textual Inversion images were not written.

## Validation

- IP-Adapter PNG files: 192
- IP-Adapter JSON sidecars: 192
- Metadata rows with `ip_adapter_scale == 0.3`: 192
- Outputs per subject: 24
- Distinct SHA-256 hashes per subject: 24
- DINOv2 rows: 768, with no NaN metric values
- CLIP rows: 768, with no NaN metric values
- Seed-specific four-method grids: 32 PNG and 32 PDF
- Background-leakage review grids: 32 PNG and 32 PDF

## Updated aggregate IP-Adapter results

| Metric | Previous scale 0.7 | Current scale 0.3 |
|---|---:|---:|
| DINOv2 mean-reference similarity | 0.6681 | 0.5578 |
| CLIP prompt alignment | 0.2018 | 0.2669 |

This observed change is consistent with the expected fidelity/editability
trade-off, but it is a comparison of two operating points rather than a full
scale ablation.

## Annotation status

The existing provisional human/Claude annotations were assigned to the former
scale-0.7 IP-Adapter images. They are not transferred to the new images.
Background-leakage review sheets were regenerated, but the 192 new IP-Adapter
outputs require fresh human annotation before manual-score tables or figures
can be updated validly.
