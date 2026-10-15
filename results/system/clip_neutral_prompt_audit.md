# CLIP Neutral-Prompt Audit

Date: 2026-10-12

The main evaluation code reads `evaluation_prompt` from each image JSON and
passes that text directly to CLIP. It does not score the method-specific
`generation_prompt`.

Before adding the base-model control, all 768 personalized-method metadata
records were checked. There were zero `evaluation_prompt` occurrences of:

- `sks`
- Textual Inversion angle-bracket placeholder tokens
- any other method-specific learned identifier

Representative mappings:

| Method | Generation prompt | CLIP evaluation prompt |
|---|---|---|
| DreamBooth | `a photo of sks backpack` | `a photo of a backpack` |
| DreamBooth-LoRA | `a photo of sks backpack` | `a photo of a backpack` |
| Textual Inversion | `a photo of <sks-backpack>` | `a photo of a backpack` |
| IP-Adapter Plus | `a photo of a backpack` | `a photo of a backpack` |
| Base SD1.5 | `a photo of a backpack` | `a photo of a backpack` |

After the base-model control was added, the complete 960-row metric pipeline
was rerun. CLIP therefore compares every method against the same neutral
semantic text for a given subject and prompt condition. Fine-tuned methods are
not penalized for CLIP's inability to interpret `sks` or learned placeholder
tokens.

The corresponding implementation is
`src/evaluation/evaluate_main_benchmark.py`, where the batch text input is
constructed from `row["evaluation_prompt"]`.
