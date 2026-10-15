#!/usr/bin/env python3
"""Generate standardized dog2 outputs for LoRA rank/step ablations."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

import torch
import yaml
from diffusers import StableDiffusionPipeline


VARIANTS = {
    "r4_s800": (4, 800, "outputs/ablations/lora_rank/dog2_lora_r4_s800_seed42/pytorch_lora_weights.safetensors"),
    "r16_s400": (16, 400, "outputs/training/lora/dog2_lora_r16_s800_seed42/checkpoint-400/pytorch_lora_weights.safetensors"),
    "r16_s800": (16, 800, "outputs/training/lora/dog2_lora_r16_s800_seed42/checkpoint-800/pytorch_lora_weights.safetensors"),
    "r16_s1200": (16, 1200, "outputs/ablations/lora_steps/dog2_lora_r16_s1200_seed42/checkpoint-1200/pytorch_lora_weights.safetensors"),
    "r32_s800": (32, 800, "outputs/ablations/lora_rank/dog2_lora_r32_s800_seed42/pytorch_lora_weights.safetensors"),
}


def render(template: str, subject: dict) -> str:
    return template.format(
        instance_token=subject["instance_token"],
        class_name=subject["class_name"],
        placeholder_token=subject["placeholder_token"],
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    args = parser.parse_args()
    root = args.project_root.resolve()
    if torch.cuda.device_count() != 1:
        raise RuntimeError(f"Expected one visible MIG device, got {torch.cuda.device_count()}")

    rank, training_steps, relative_weights = VARIANTS[args.variant]
    weights = root / relative_weights
    if not weights.is_file():
        raise FileNotFoundError(weights)
    base = root / "models/base/stable-diffusion-v1-5"
    subject = yaml.safe_load((root / "configs/main_subjects.yaml").read_text())["subjects"]["dog2"]
    config = yaml.safe_load((root / "configs/main_prompts.yaml").read_text())
    generation = config["generation"]
    output_root = root / "outputs/ablations/lora_generation" / args.variant

    started = time.perf_counter()
    pipe = StableDiffusionPipeline.from_pretrained(
        base, torch_dtype=torch.float16, safety_checker=None, local_files_only=True
    )
    pipe.load_lora_weights(weights.parent, weight_name=weights.name)
    pipe.to("cuda:0")
    torch.cuda.synchronize()
    loading_time = time.perf_counter() - started
    pipe(
        prompt="a photo of sks dog",
        num_inference_steps=2,
        guidance_scale=generation["guidance_scale"],
        height=generation["resolution"],
        width=generation["resolution"],
        generator=torch.Generator("cuda:0").manual_seed(999),
    )
    torch.cuda.synchronize()
    revision = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip()

    for prompt_id, template in config["prompt_templates"].items():
        generation_prompt = render(template["personalized"], subject)
        evaluation_prompt = render(template["clean"], subject)
        prompt_dir = output_root / prompt_id
        prompt_dir.mkdir(parents=True, exist_ok=True)
        for seed in generation["seeds"]:
            image_path = prompt_dir / f"seed_{seed}.png"
            metadata_path = image_path.with_suffix(".json")
            if image_path.exists() != metadata_path.exists():
                raise RuntimeError(f"Partial output pair: {image_path}")
            if image_path.is_file():
                print(f"SKIP variant={args.variant} prompt={prompt_id} seed={seed}")
                continue
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            tick = time.perf_counter()
            image = pipe(
                prompt=generation_prompt,
                num_inference_steps=generation["inference_steps"],
                guidance_scale=generation["guidance_scale"],
                height=generation["resolution"],
                width=generation["resolution"],
                generator=torch.Generator("cuda:0").manual_seed(seed),
            ).images[0]
            torch.cuda.synchronize()
            latency = time.perf_counter() - tick
            peak = torch.cuda.max_memory_allocated() / 1024**3
            image.save(image_path)
            metadata = {
                "experiment": "lora_ablation",
                "variant": args.variant,
                "method": "lora",
                "dataset": "dreambooth",
                "subject": "dog2",
                "prompt_id": prompt_id,
                "prompt_category": template["category"],
                "generation_prompt": generation_prompt,
                "evaluation_prompt": evaluation_prompt,
                "seed": seed,
                "base_model": "stable-diffusion-v1-5/stable-diffusion-v1-5",
                "checkpoint": str(weights.relative_to(root)),
                "training_steps": training_steps,
                "rank": rank,
                "resolution": generation["resolution"],
                "inference_steps": generation["inference_steps"],
                "guidance_scale": generation["guidance_scale"],
                "scheduler": pipe.scheduler.__class__.__name__,
                "generation_latency_sec": latency,
                "model_loading_time_sec": loading_time,
                "peak_inference_vram_gb": peak,
                "precision": "fp16",
                "gpu": torch.cuda.get_device_name(0),
                "mig_uuid": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
                "git_commit": revision,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            }
            metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
            print(
                f"GENERATED variant={args.variant} prompt={prompt_id} seed={seed} "
                f"latency={latency:.3f}s peak={peak:.3f}GiB",
                flush=True,
            )


if __name__ == "__main__":
    main()
