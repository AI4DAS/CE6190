#!/usr/bin/env python3
"""Generate one subject/method lane for the eight-subject main benchmark."""

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
from PIL import Image


METHODS = ("base_model", "dreambooth", "lora", "textual_inversion", "ip_adapter_plus")


def git_revision(root: Path) -> str:
    return subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()


def training_dir(root: Path, subject: str, method: str) -> Path:
    if method == "dreambooth":
        return root / f"outputs/training/dreambooth/{subject}_db_s800_seed42"
    if method == "lora":
        return root / f"outputs/training/lora/{subject}_lora_r16_s800_seed42"
    return root / f"outputs/training/textual_inversion/{subject}_ti_s3000_seed42"


def load_pipeline(root: Path, subject: str, method: str, scale: float):
    base = root / "models/base/stable-diffusion-v1-5"
    trained = training_dir(root, subject, method) if method not in {"base_model", "ip_adapter_plus"} else None
    model_path = trained if method == "dreambooth" else base
    started = time.perf_counter()
    pipe = StableDiffusionPipeline.from_pretrained(
        model_path, torch_dtype=torch.float16, safety_checker=None, local_files_only=True
    )
    if method == "lora":
        checkpoint = trained / "pytorch_lora_weights.safetensors"
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        pipe.load_lora_weights(trained, weight_name=checkpoint.name)
    elif method == "textual_inversion":
        checkpoint = trained / "learned_embeds.safetensors"
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        pipe.load_textual_inversion(trained, weight_name=checkpoint.name)
    elif method == "ip_adapter_plus":
        adapter = root / "models/ip_adapter/IP-Adapter"
        checkpoint = adapter / "models/ip-adapter-plus_sd15.bin"
        pipe.load_ip_adapter(
            adapter,
            subfolder="models",
            weight_name="ip-adapter-plus_sd15.bin",
            image_encoder_folder="image_encoder",
            local_files_only=True,
        )
        pipe.set_ip_adapter_scale(scale)
    elif method == "dreambooth":
        checkpoint = trained
        if not (trained / "model_index.json").is_file():
            raise FileNotFoundError(trained / "model_index.json")
    else:
        checkpoint = base
    pipe.to("cuda:0")
    torch.cuda.synchronize()
    return pipe, checkpoint, time.perf_counter() - started


def render_prompts(entry: dict, template: dict) -> tuple[str, str]:
    values = {
        "instance_token": entry["instance_token"],
        "class_name": entry["class_name"],
        "placeholder_token": entry["placeholder_token"],
    }
    return template.format(**values), values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--subject", required=True)
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace complete existing image/metadata pairs for this lane.",
    )
    args = parser.parse_args()
    root = args.project_root.resolve()
    if torch.cuda.device_count() != 1:
        raise RuntimeError(f"Expected one visible MIG device, got {torch.cuda.device_count()}")
    subjects = yaml.safe_load((root / "configs/main_subjects.yaml").read_text())["subjects"]
    if args.subject not in subjects:
        raise KeyError(args.subject)
    subject = subjects[args.subject]
    prompt_config = yaml.safe_load((root / "configs/main_prompts.yaml").read_text())
    templates = prompt_config["prompt_templates"]
    generation = prompt_config["generation"]
    output_root = root / "outputs/main_evaluation" / args.method / subject["dataset"] / args.subject

    expected = {(prompt_id, seed) for prompt_id in templates for seed in generation["seeds"]}
    for prompt_id, seed in expected:
        image = output_root / prompt_id / f"seed_{seed}.png"
        metadata = image.with_suffix(".json")
        if image.exists() != metadata.exists():
            raise RuntimeError(f"Partial output pair; refusing substitution: {image} / {metadata}")
    pipe, checkpoint, load_time = load_pipeline(
        root, args.subject, args.method, generation["ip_adapter_scale"]
    )
    reference_path = root / subject["image_dir"] / sorted(
        path.name for path in (root / subject["image_dir"]).iterdir() if path.is_file()
    )[0]
    reference = Image.open(reference_path).convert("RGB")
    prompt_key = "textual_inversion" if args.method == "textual_inversion" else (
        "clean" if args.method in {"base_model", "ip_adapter_plus"} else "personalized"
    )
    first = next(iter(templates.values()))
    warmup_prompt, _ = render_prompts(subject, first[prompt_key])
    warmup = {
        "prompt": warmup_prompt,
        "num_inference_steps": 2,
        "guidance_scale": generation["guidance_scale"],
        "height": generation["resolution"],
        "width": generation["resolution"],
        "generator": torch.Generator("cuda:0").manual_seed(999),
    }
    if args.method == "ip_adapter_plus":
        warmup["ip_adapter_image"] = reference
    pipe(**warmup)
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    revision = git_revision(root)

    for prompt_id, template in templates.items():
        generation_prompt, _ = render_prompts(subject, template[prompt_key])
        evaluation_prompt, _ = render_prompts(subject, template["clean"])
        prompt_dir = output_root / prompt_id
        prompt_dir.mkdir(parents=True, exist_ok=True)
        for seed in generation["seeds"]:
            image_path = prompt_dir / f"seed_{seed}.png"
            metadata_path = image_path.with_suffix(".json")
            if image_path.is_file() and metadata_path.is_file() and not args.overwrite:
                print(f"SKIP existing subject={args.subject} method={args.method} prompt={prompt_id} seed={seed}")
                continue
            call = {
                "prompt": generation_prompt,
                "num_inference_steps": generation["inference_steps"],
                "guidance_scale": generation["guidance_scale"],
                "height": generation["resolution"],
                "width": generation["resolution"],
                "generator": torch.Generator("cuda:0").manual_seed(seed),
            }
            if args.method == "ip_adapter_plus":
                call["ip_adapter_image"] = reference
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            started = time.perf_counter()
            image = pipe(**call).images[0]
            torch.cuda.synchronize()
            latency = time.perf_counter() - started
            peak = torch.cuda.max_memory_allocated() / 1024**3
            image.save(image_path)
            metadata = {
                "method": args.method,
                "dataset": subject["dataset"],
                "subject": args.subject,
                "prompt_id": prompt_id,
                "prompt_category": template["category"],
                "generation_prompt": generation_prompt,
                "evaluation_prompt": evaluation_prompt,
                "seed": seed,
                "base_model": "stable-diffusion-v1-5/stable-diffusion-v1-5",
                "checkpoint": str(checkpoint.relative_to(root)),
                "training_steps": {"dreambooth": 800, "lora": 800, "textual_inversion": 3000}.get(args.method, 0),
                "rank": 16 if args.method == "lora" else None,
                "ip_adapter_scale": generation["ip_adapter_scale"] if args.method == "ip_adapter_plus" else None,
                "resolution": generation["resolution"],
                "inference_steps": generation["inference_steps"],
                "guidance_scale": generation["guidance_scale"],
                "scheduler": pipe.scheduler.__class__.__name__,
                "model_loading_time_sec": load_time,
                "generation_latency_sec": latency,
                "peak_inference_vram_gb": peak,
                "precision": "fp16",
                "gpu": torch.cuda.get_device_name(0),
                "mig_uuid": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "reference_images": [str(reference_path.relative_to(root))] if args.method == "ip_adapter_plus" else [],
                "git_commit": revision,
            }
            metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
            print(
                f"GENERATED subject={args.subject} method={args.method} prompt={prompt_id} "
                f"seed={seed} latency={latency:.3f}s peak={peak:.3f}GiB",
                flush=True,
            )


if __name__ == "__main__":
    main()
