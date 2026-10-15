# Main Training Status Reconciliation

Markers were reconciled only after final-artifact load and generation tests passed.

| Subject | Method | Old marker | Actual status | Action |
|---|---|---|---|---|
| dog2 | dreambooth | SUCCESS | COMPLETED_VALIDATED | Preserved SUCCESS |
| dog2 | lora | SUCCESS | COMPLETED_VALIDATED | Preserved SUCCESS |
| dog2 | textual_inversion | SUCCESS | COMPLETED_VALIDATED | Preserved SUCCESS |
| cat | dreambooth | FAILED | COMPLETED_VALIDATED | Replaced false FAILED with SUCCESS |
| cat | lora | FAILED | COMPLETED_VALIDATED | Replaced false FAILED with SUCCESS |
| cat | textual_inversion | FAILED | COMPLETED_VALIDATED | Replaced false FAILED with SUCCESS |
| pet_dog1 | dreambooth | FAILED | COMPLETED_VALIDATED | Replaced false FAILED with SUCCESS |
| pet_dog1 | lora | FAILED | COMPLETED_VALIDATED | Replaced false FAILED with SUCCESS |
| pet_dog1 | textual_inversion | FAILED | COMPLETED_VALIDATED | Replaced false FAILED with SUCCESS |
| bear_plushie | dreambooth | FAILED | COMPLETED_VALIDATED | Replaced false FAILED with SUCCESS |
| bear_plushie | lora | NONE | COMPLETED_VALIDATED | Preserved SUCCESS |
| bear_plushie | textual_inversion | FAILED | COMPLETED_VALIDATED | Replaced false FAILED with SUCCESS |
| robot_toy | dreambooth | FAILED | COMPLETED_VALIDATED | Replaced false FAILED with SUCCESS |
| robot_toy | lora | FAILED | COMPLETED_VALIDATED | Replaced false FAILED with SUCCESS |
| robot_toy | textual_inversion | FAILED | COMPLETED_VALIDATED | Replaced false FAILED with SUCCESS |
| teapot | dreambooth | FAILED | COMPLETED_VALIDATED | Replaced false FAILED with SUCCESS |
| teapot | lora | FAILED | COMPLETED_VALIDATED | Replaced false FAILED with SUCCESS |
| teapot | textual_inversion | FAILED | COMPLETED_VALIDATED | Replaced false FAILED with SUCCESS |
| colorful_sneaker | dreambooth | FAILED | COMPLETED_VALIDATED | Replaced false FAILED with SUCCESS |
| colorful_sneaker | lora | FAILED | COMPLETED_VALIDATED | Replaced false FAILED with SUCCESS |
| colorful_sneaker | textual_inversion | FAILED | COMPLETED_VALIDATED | Replaced false FAILED with SUCCESS |
| luggage_backpack1 | dreambooth | FAILED | COMPLETED_VALIDATED | Replaced false FAILED with SUCCESS |
| luggage_backpack1 | lora | FAILED | COMPLETED_VALIDATED | Replaced false FAILED with SUCCESS |
| luggage_backpack1 | textual_inversion | FAILED | COMPLETED_VALIDATED | Replaced false FAILED with SUCCESS |

No training logs or checkpoint evidence were deleted.

## Launcher-update incident

While the launcher fix was applied, the already-running `bear_plushie` and
`colorful_sneaker` LoRA shells read the modified script after their completed
800-step child processes returned and began an unintended second invocation.
Those duplicate child process trees were stopped at approximately 30 steps.
The original final LoRA weights, first-run logs, runtime measurements, and
VRAM measurements were preserved. Both original final weights subsequently
passed the independent load-and-generation validation. Evidence remains in
each run directory as `training_duplicate_interrupted.log` and
`duplicate_interrupted_failure_summary.txt`.
