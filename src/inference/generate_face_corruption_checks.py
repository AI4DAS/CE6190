#!/usr/bin/env python3
"""Task D: one-factor-at-a-time inference checks for visible color blotches."""

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


def checkpoint(root: Path, method: str, subject: str) -> Path:
    if method == "dreambooth":
        return root / "outputs/training/dreambooth" / f"{subject}_db_s800_seed42"
    return root / "outputs/training/lora" / f"{subject}_lora_r16_s800_seed42" / "pytorch_lora_weights.safetensors"


def variants(method: str) -> list[tuple[str, float, float, torch.dtype]]:
    common = [
        ("default_fp16_g7p5", 7.5, 1.0, torch.float16),
        ("guidance5_fp16", 5.0, 1.0, torch.float16),
        ("default_fp32_g7p5", 7.5, 1.0, torch.float32),
    ]
    if method == "lora":
        common.extend([
            ("lora_weight0p8_fp16", 7.5, 0.8, torch.float16),
            ("lora_weight0p6_fp16", 7.5, 0.6, torch.float16),
        ])
    return common


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--subject", choices=("dog2", "cat"), required=True)
    parser.add_argument("--method", choices=("dreambooth", "lora"), required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    root = args.project_root.resolve()
    if torch.cuda.device_count() != 1:
        raise RuntimeError(f"Expected one visible MIG device, got {torch.cuda.device_count()}")

    subjects = yaml.safe_load((root / "configs/main_subjects.yaml").read_text())["subjects"]
    prompts = yaml.safe_load((root / "configs/main_prompts.yaml").read_text())
    subject = subjects[args.subject]
    ckpt = checkpoint(root, args.method, args.subject)
    if not ckpt.exists():
        raise FileNotFoundError(ckpt)
    revision = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    values = {
        "instance_token": subject["instance_token"],
        "class_name": subject["class_name"],
        "placeholder_token": subject["placeholder_token"],
    }
    ref_files = sorted(p.name for p in (root / subject["image_dir"]).iterdir() if p.is_file())

    for variant, guidance, lora_weight, dtype in variants(args.method):
        if args.method == "dreambooth":
            pipe = StableDiffusionPipeline.from_pretrained(
                ckpt, torch_dtype=dtype, safety_checker=None, local_files_only=True
            )
        else:
            pipe = StableDiffusionPipeline.from_pretrained(
                root / "models/base/stable-diffusion-v1-5",
                torch_dtype=dtype, safety_checker=None, local_files_only=True,
            )
            pipe.load_lora_weights(ckpt.parent, weight_name=ckpt.name, adapter_name="task_d")
            pipe.set_adapters("task_d", adapter_weights=lora_weight)
        pipe.to("cuda:0")
        torch.cuda.synchronize()

        for prompt_id, prompt_cfg in prompts["prompt_templates"].items():
            generation_prompt = prompt_cfg["personalized"].format(**values)
            evaluation_prompt = prompt_cfg["clean"].format(**values)
            out_dir = root / "outputs/experiments/task_d_face_corruption" / args.method / args.subject / variant / prompt_id
            image_path = out_dir / "seed_42.png"
            metadata_path = out_dir / "seed_42.json"
            if image_path.exists() != metadata_path.exists():
                raise RuntimeError(f"Partial output pair: {image_path}")
            if image_path.exists():
                if not args.resume:
                    raise FileExistsError(image_path)
                continue
            out_dir.mkdir(parents=True, exist_ok=True)
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            started = time.perf_counter()
            image = pipe(
                prompt=generation_prompt,
                num_inference_steps=prompts["generation"]["inference_steps"],
                guidance_scale=guidance,
                height=prompts["generation"]["resolution"],
                width=prompts["generation"]["resolution"],
                generator=torch.Generator("cuda:0").manual_seed(42),
            ).images[0]
            torch.cuda.synchronize()
            latency = time.perf_counter() - started
            image.save(image_path)
            row = {
                "method": args.method, "dataset": subject["dataset"], "subject": args.subject,
                "prompt_id": prompt_id, "prompt_category": prompt_cfg["category"],
                "generation_prompt": generation_prompt, "evaluation_prompt": evaluation_prompt,
                "seed": 42, "base_model": "stable-diffusion-v1-5/stable-diffusion-v1-5",
                "checkpoint": str(ckpt.relative_to(root)), "training_steps": 800,
                "resolution": prompts["generation"]["resolution"],
                "inference_steps": prompts["generation"]["inference_steps"],
                "guidance_scale": guidance, "negative_prompt": None,
                "scheduler": pipe.scheduler.__class__.__name__,
                "generation_latency_sec": latency,
                "peak_inference_vram_gb": torch.cuda.max_memory_allocated() / 1024**3,
                "precision": "fp32" if dtype == torch.float32 else "fp16",
                "gpu": torch.cuda.get_device_name(0),
                "mig_uuid": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "git_commit": revision,
                "experiment": "task_d_face_corruption", "variant": variant,
                "scale": None, "rank": 16 if args.method == "lora" else None,
                "step": 800, "n_refs": len(ref_files), "ref_files_used": ref_files,
                "dtype": "float32" if dtype == torch.float32 else "float16",
                "lora_inference_weight": lora_weight if args.method == "lora" else None,
            }
            metadata_path.write_text(json.dumps(row, indent=2) + "\n")
            print(f"SAVED {image_path}", flush=True)
        del pipe
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
