# Training pipeline smoke-test report

Date: 2026-10-10

## Original failures

All three original jobs exited before CUDA training because the command passed `--report_to=none`. Diffusers v0.35.1 passed the literal string `none` to Accelerate 1.4.0, whose tracker registry accepts `all`, `aim`, `tensorboard`, `wandb`, `comet_ml`, `mlflow`, `clearml`, or `dvclive`, but not `none`. Each log contains `ValueError: Unsupported logging capability: none`.

- Full DreamBooth: failed while constructing `Accelerator`; no model/data training step ran.
- DreamBooth-LoRA: failed while constructing `Accelerator`; no model/data training step ran.
- Textual Inversion: failed while constructing `Accelerator`; no model/data training step ran.

A second configuration risk was found during diagnosis: no Accelerate config existed, and an automatically generated config selected `MULTI_GPU` with two processes because the host exposes multiple MIG instances. The checked-in config explicitly uses `distributed_type: NO`, one process, one machine, and fp16.

## Dependency change

TensorBoard was absent. `tensorboard==2.19.0` was added because it is the official scripts' default supported local tracker. No existing package was downgraded or replaced.

## Smoke results

| Method | Steps | Prior preservation | Exit | Peak VRAM | Required artifact | Inference load |
|---|---:|---|---:|---:|---|---|
| Full DreamBooth | 5 | ON, 2 smoke class images | 0 | 18,592 MiB | complete Diffusers pipeline | PASS |
| DreamBooth-LoRA r16 | 5 | ON, 2 smoke class images | 0 | 4,706 MiB | `pytorch_lora_weights.safetensors` | PASS |
| Textual Inversion | 5 | N/A | 0 | 4,342 MiB | `learned_embeds.safetensors` | PASS |

Checkpoint-load images are stored under `results/figures/smoke/`. Full pilot training was not launched.
