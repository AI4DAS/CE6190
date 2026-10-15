# Main Evaluation Validation

Validation date: 2026-10-10

## Coverage

- Subjects: 8
- Methods: 4
- Prompt conditions per subject/method: 6
- Seeds per prompt: 4 (`42`, `123`, `456`, `789`)
- Expected canonical images: 768
- Canonical PNG images found: 768
- Metadata JSON records found: 768
- DINOv2 records: 768
- CLIP records: 768

Each method contains 192 canonical images and 192 metadata records. Every
method/subject/prompt/seed combination is present exactly once.

## Integrity checks

- All canonical images opened successfully with Pillow.
- All required metadata fields were present.
- No duplicate SHA-256 image groups were found.
- Neither metric table contains missing or NaN values.
- DINOv2 comparisons use only the original subject reference directories.
- CLIP comparisons use the clean `evaluation_prompt`, not `sks` or a Textual
  Inversion placeholder token.

## Preserved interrupted artifact

An image produced by an interrupted early generation attempt had no metadata
and was not a canonical benchmark sample. It was preserved, not deleted, at:

`outputs/debug/interrupted_main_evaluation/dreambooth/cat/P0/seed_789.interrupted.png`

It is excluded from the 768-image benchmark.
