# Dog2 pilot validation report

## Scope and protocol

- Subject: DreamBooth `dog2`; evaluation references are exactly the six original images `00.jpg`–`05.jpg`.
- Methods: full DreamBooth, DreamBooth-LoRA rank 16, Textual Inversion, and IP-Adapter Plus at scale 0.7.
- Conditions: identity, beach, snow, kitchen, red hat, and watercolor.
- Seeds: 42, 123, 456, and 789.
- Shared inference settings: 512×512, 50 PNDM steps, guidance scale 7.5, FP16, and one untimed warm-up generation.
- IP-Adapter uses `00.jpg` as its one inference reference. The other methods use their subject-specific learned artifacts.
- CLIP uses the clean evaluation prompts in `results/pilot_prompt_protocol.csv`, never `sks` or `<sks-dog>`.

## Artifact checks

| Method | PNG | JSON | Checkpoint/load test | Prompt/seed coverage |
| --- | ---: | ---: | --- | --- |
| DreamBooth | 24 | 24 | PASS | 6 prompts × 4 seeds |
| DreamBooth-LoRA | 24 | 24 | PASS | 6 prompts × 4 seeds |
| Textual Inversion | 24 | 24 | PASS | 6 prompts × 4 seeds |
| IP-Adapter Plus | 24 | 24 | PASS | 6 prompts × 4 seeds |

- Total images: 96.
- Duplicate SHA-256 image groups: 0.
- DINOv2 rows: 96; NaN metric values: 0.
- CLIP rows: 96; NaN metric values: 0.
- Each JSON record identifies the correct checkpoint, generation/evaluation prompt, seed, precision, scheduler, MIG UUID, latency, and peak allocated inference memory.
- The 200 generic class images are used only by prior-preservation training for DreamBooth and LoRA; they are not evaluation references.

## Missing measurements

IP-Adapter Plus has no subject-specific training run, so subject-specific peak training VRAM is reported as N/A. Its shared adapter checkpoint is reported separately from per-subject storage. The separately shared image encoder is documented in the compute CSV notes and is not folded into the 150.71 MiB adapter-checkpoint value.

## Manual review

`results/metrics/manual_overfitting_annotation.csv` contains 96 unscored rows. Scores are deliberately blank pending human annotation. The background-leakage inspection sheet supports consistent review; no automatic subjective leakage, pose-copying, or composition-copying labels were assigned.
