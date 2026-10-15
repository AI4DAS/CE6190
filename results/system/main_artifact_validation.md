# Main Artifact Validation

Validated artifacts: **24/24 passed**.

Each test loaded the final artifact on one visible MIG device and generated one 512×512 image with seed 42 and 10 denoising steps.

| Subject | Method | Status | Load time (s) | Generation (s) | Peak VRAM (GiB) | Validation image |
|---|---|---|---:|---:|---:|---|
| dog2 | dreambooth | PASS | 4.569 | 1.352 | 2.636 | `results/figures/main_artifact_validation/dog2_dreambooth.png` |
| dog2 | lora | PASS | 1.991 | 0.722 | 2.642 | `results/figures/main_artifact_validation/dog2_lora.png` |
| dog2 | textual_inversion | PASS | 5.200 | 0.715 | 2.636 | `results/figures/main_artifact_validation/dog2_textual_inversion.png` |
| cat | dreambooth | PASS | 4.048 | 0.662 | 2.636 | `results/figures/main_artifact_validation/cat_dreambooth.png` |
| cat | lora | PASS | 1.332 | 0.700 | 2.642 | `results/figures/main_artifact_validation/cat_lora.png` |
| cat | textual_inversion | PASS | 3.462 | 0.693 | 2.636 | `results/figures/main_artifact_validation/cat_textual_inversion.png` |
| pet_dog1 | dreambooth | PASS | 3.919 | 0.682 | 2.636 | `results/figures/main_artifact_validation/pet_dog1_dreambooth.png` |
| pet_dog1 | lora | PASS | 1.385 | 0.703 | 2.642 | `results/figures/main_artifact_validation/pet_dog1_lora.png` |
| pet_dog1 | textual_inversion | PASS | 3.484 | 0.698 | 2.636 | `results/figures/main_artifact_validation/pet_dog1_textual_inversion.png` |
| bear_plushie | dreambooth | PASS | 3.219 | 0.659 | 2.636 | `results/figures/main_artifact_validation/bear_plushie_dreambooth.png` |
| bear_plushie | lora | PASS | 1.392 | 0.699 | 2.642 | `results/figures/main_artifact_validation/bear_plushie_lora.png` |
| bear_plushie | textual_inversion | PASS | 4.442 | 0.708 | 2.636 | `results/figures/main_artifact_validation/bear_plushie_textual_inversion.png` |
| robot_toy | dreambooth | PASS | 2.575 | 0.660 | 2.636 | `results/figures/main_artifact_validation/robot_toy_dreambooth.png` |
| robot_toy | lora | PASS | 1.515 | 0.699 | 2.642 | `results/figures/main_artifact_validation/robot_toy_lora.png` |
| robot_toy | textual_inversion | PASS | 3.586 | 0.700 | 2.636 | `results/figures/main_artifact_validation/robot_toy_textual_inversion.png` |
| teapot | dreambooth | PASS | 3.801 | 0.658 | 2.636 | `results/figures/main_artifact_validation/teapot_dreambooth.png` |
| teapot | lora | PASS | 1.551 | 0.701 | 2.642 | `results/figures/main_artifact_validation/teapot_lora.png` |
| teapot | textual_inversion | PASS | 3.570 | 0.712 | 2.636 | `results/figures/main_artifact_validation/teapot_textual_inversion.png` |
| colorful_sneaker | dreambooth | PASS | 3.792 | 0.678 | 2.636 | `results/figures/main_artifact_validation/colorful_sneaker_dreambooth.png` |
| colorful_sneaker | lora | PASS | 1.374 | 0.703 | 2.642 | `results/figures/main_artifact_validation/colorful_sneaker_lora.png` |
| colorful_sneaker | textual_inversion | PASS | 3.419 | 0.688 | 2.636 | `results/figures/main_artifact_validation/colorful_sneaker_textual_inversion.png` |
| luggage_backpack1 | dreambooth | PASS | 1.662 | 0.657 | 2.636 | `results/figures/main_artifact_validation/luggage_backpack1_dreambooth.png` |
| luggage_backpack1 | lora | PASS | 1.367 | 0.699 | 2.642 | `results/figures/main_artifact_validation/luggage_backpack1_lora.png` |
| luggage_backpack1 | textual_inversion | PASS | 3.530 | 0.700 | 2.636 | `results/figures/main_artifact_validation/luggage_backpack1_textual_inversion.png` |

These are compatibility checks, not benchmark-quality samples. Intermediate `checkpoint-*` directories were not used.
