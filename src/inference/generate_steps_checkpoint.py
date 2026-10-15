#!/usr/bin/env python3
"""Generate the fixed Task C subset immediately after a training checkpoint."""

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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--method", choices=("dreambooth", "lora", "textual_inversion"), required=True)
    parser.add_argument("--subject", required=True)
    parser.add_argument("--step", type=int, required=True)
    parser.add_argument("--training-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    root = args.project_root.resolve()
    training_dir = args.training_dir if args.training_dir.is_absolute() else root / args.training_dir
    output_root = args.output_root if args.output_root.is_absolute() else root / args.output_root
    if torch.cuda.device_count() != 1:
        raise RuntimeError(f"Expected exactly one visible MIG, got {torch.cuda.device_count()}")

    subjects = yaml.safe_load((root / "configs/main_subjects.yaml").read_text())["subjects"]
    prompt_cfg = yaml.safe_load((root / "configs/main_prompts.yaml").read_text())
    subject = subjects[args.subject]
    generation = prompt_cfg["generation"]
    templates = prompt_cfg["prompt_templates"]
    base = root / "models/base/stable-diffusion-v1-5"
    model_path = training_dir if args.method == "dreambooth" else base
    started = time.perf_counter()
    pipe = StableDiffusionPipeline.from_pretrained(
        model_path, torch_dtype=torch.float16, safety_checker=None, local_files_only=True,
    )
    if args.method == "lora":
        weight = training_dir / "pytorch_lora_weights.safetensors"
        pipe.load_lora_weights(training_dir, weight_name=weight.name)
        checkpoint = weight
    elif args.method == "textual_inversion":
        weight = training_dir / "learned_embeds.safetensors"
        pipe.load_textual_inversion(training_dir, weight_name=weight.name)
        checkpoint = weight
    else:
        checkpoint = training_dir
    pipe.to("cuda:0")
    torch.cuda.synchronize()
    load_time = time.perf_counter() - started
    prompt_key = "textual_inversion" if args.method == "textual_inversion" else "personalized"
    values = {
        "instance_token": subject["instance_token"], "class_name": subject["class_name"],
        "placeholder_token": subject["placeholder_token"],
    }
    revision = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()

    warmup = next(iter(templates.values()))[prompt_key].format(**values)
    pipe(
        prompt=warmup, num_inference_steps=2, guidance_scale=generation["guidance_scale"],
        height=generation["resolution"], width=generation["resolution"],
        generator=torch.Generator("cuda:0").manual_seed(999),
    )
    torch.cuda.synchronize(); torch.cuda.empty_cache()

    for prompt_id, prompt in templates.items():
        generation_prompt = prompt[prompt_key].format(**values)
        evaluation_prompt = prompt["clean"].format(**values)
        for seed in (42, 123):
            directory = output_root / args.method / args.subject / f"step_{args.step}" / prompt_id
            image_path = directory / f"seed_{seed}.png"
            metadata_path = image_path.with_suffix(".json")
            if image_path.exists() != metadata_path.exists():
                raise RuntimeError(f"Partial pair: {image_path} / {metadata_path}")
            if image_path.is_file():
                if not args.resume:
                    raise FileExistsError(image_path)
                existing = json.loads(metadata_path.read_text())
                if existing.get("step") != args.step or existing.get("subject") != args.subject:
                    raise RuntimeError(f"Metadata mismatch: {metadata_path}")
                continue
            directory.mkdir(parents=True, exist_ok=True)
            torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize()
            image_started = time.perf_counter()
            image = pipe(
                prompt=generation_prompt,
                num_inference_steps=generation["inference_steps"],
                guidance_scale=generation["guidance_scale"],
                height=generation["resolution"], width=generation["resolution"],
                generator=torch.Generator("cuda:0").manual_seed(seed),
            ).images[0]
            torch.cuda.synchronize()
            latency = time.perf_counter() - image_started
            peak = torch.cuda.max_memory_allocated() / 1024**3
            image.save(image_path)
            row = {
                "method": args.method, "dataset": subject["dataset"], "subject": args.subject,
                "prompt_id": prompt_id, "prompt_category": prompt["category"],
                "generation_prompt": generation_prompt, "evaluation_prompt": evaluation_prompt,
                "seed": seed, "base_model": "stable-diffusion-v1-5/stable-diffusion-v1-5",
                "checkpoint": str(checkpoint.relative_to(root)), "training_steps": args.step,
                "resolution": generation["resolution"], "inference_steps": generation["inference_steps"],
                "guidance_scale": generation["guidance_scale"], "negative_prompt": None,
                "scheduler": pipe.scheduler.__class__.__name__, "precision": "fp16",
                "model_loading_time_sec": load_time, "generation_latency_sec": latency,
                "peak_inference_vram_gb": peak, "gpu": torch.cuda.get_device_name(0),
                "mig_uuid": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "git_commit": revision,
                "experiment": "task_c_steps_ablation",
                "variant": f"{args.method}_step_{args.step}", "scale": None,
                "rank": 16 if args.method == "lora" else None, "step": args.step,
                "n_refs": subject["expected_images"],
                "ref_files_used": sorted(path.name for path in (root / subject["image_dir"]).iterdir() if path.is_file()),
                "dtype": "float16",
            }
            metadata_path.write_text(json.dumps(row, indent=2) + "\n")
            print(f"SAVED {image_path} latency={latency:.3f}s", flush=True)
    print(f"CHECKPOINT_GENERATION_COMPLETE method={args.method} subject={args.subject} step={args.step}")


if __name__ == "__main__":
    main()
