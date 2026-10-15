#!/usr/bin/env python3
"""Resumable IP-Adapter plain/Plus generation for Task A ablations."""

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


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


def revision(root: Path) -> str:
    return subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()


def render(template: str, subject: dict) -> str:
    return template.format(
        instance_token=subject["instance_token"],
        class_name=subject["class_name"],
        placeholder_token=subject["placeholder_token"],
    )


def reference_paths(root: Path, subject: dict) -> list[Path]:
    directory = root / subject["image_dir"]
    paths = sorted(path for path in directory.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES)
    if len(paths) != subject["expected_images"]:
        raise RuntimeError(f"Reference count mismatch for {directory}: {len(paths)}")
    return paths


def averaged_image_embeds(pipe, images: list[Image.Image]) -> list[torch.Tensor]:
    prepared = [
        pipe.prepare_ip_adapter_image_embeds(
            ip_adapter_image=[image], ip_adapter_image_embeds=None,
            device="cuda:0", num_images_per_prompt=1,
            do_classifier_free_guidance=True,
        )[0]
        for image in images
    ]
    return [torch.stack(prepared, dim=0).mean(dim=0)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--subject", required=True)
    parser.add_argument("--adapter", choices=("plain", "plus"), required=True)
    parser.add_argument("--scale", type=float, required=True)
    parser.add_argument("--reference-mode", choices=("single", "average_all"), default="single")
    parser.add_argument(
        "--reference-count", type=int, default=0,
        help="Use the first k sorted references; zero uses the mode default/all available.",
    )
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    root = args.project_root.resolve()
    output_root = args.output_root if args.output_root.is_absolute() else root / args.output_root
    if torch.cuda.device_count() != 1:
        raise RuntimeError(f"Expected exactly one visible MIG, got {torch.cuda.device_count()}")

    subjects = yaml.safe_load((root / "configs/main_subjects.yaml").read_text())["subjects"]
    prompts = yaml.safe_load((root / "configs/main_prompts.yaml").read_text())
    if args.subject not in subjects:
        raise KeyError(args.subject)
    subject = subjects[args.subject]
    generation = prompts["generation"]
    templates = prompts["prompt_templates"]
    refs = reference_paths(root, subject)
    if args.reference_count < 0 or args.reference_count > len(refs):
        raise ValueError(f"Invalid reference count {args.reference_count} for {args.subject}")
    if args.reference_count:
        used_refs = refs[:args.reference_count]
    else:
        used_refs = refs[:1] if args.reference_mode == "single" else refs
    if args.reference_mode == "single" and len(used_refs) != 1:
        raise ValueError("single reference mode requires exactly one reference")

    adapter_root = root / "models/ip_adapter/IP-Adapter"
    weight_name = "ip-adapter_sd15.bin" if args.adapter == "plain" else "ip-adapter-plus_sd15.bin"
    checkpoint = adapter_root / "models" / weight_name
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)

    started = time.perf_counter()
    pipe = StableDiffusionPipeline.from_pretrained(
        root / "models/base/stable-diffusion-v1-5",
        torch_dtype=torch.float16, safety_checker=None, local_files_only=True,
    )
    pipe.load_ip_adapter(
        adapter_root, subfolder="models", weight_name=weight_name,
        image_encoder_folder="image_encoder", local_files_only=True,
    )
    pipe.set_ip_adapter_scale(args.scale)
    pipe.to("cuda:0")
    torch.cuda.synchronize()
    load_time = time.perf_counter() - started

    pil_refs = [Image.open(path).convert("RGB") for path in used_refs]
    embeds = averaged_image_embeds(pipe, pil_refs) if args.reference_mode == "average_all" else None
    conditioning = (
        {"ip_adapter_image": pil_refs[0]}
        if embeds is None else {"ip_adapter_image_embeds": embeds}
    )
    warmup_prompt = render(next(iter(templates.values()))["clean"], subject)
    pipe(
        prompt=warmup_prompt, num_inference_steps=2,
        guidance_scale=generation["guidance_scale"],
        height=generation["resolution"], width=generation["resolution"],
        generator=torch.Generator("cuda:0").manual_seed(999), **conditioning,
    )
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    git_commit = revision(root)

    for prompt_id, prompt_cfg in templates.items():
        clean_prompt = render(prompt_cfg["clean"], subject)
        for seed in args.seeds:
            directory = output_root / subject["dataset"] / args.subject / prompt_id
            image_path = directory / f"seed_{seed}.png"
            metadata_path = image_path.with_suffix(".json")
            if image_path.exists() != metadata_path.exists():
                raise RuntimeError(f"Partial output pair: {image_path} / {metadata_path}")
            if image_path.is_file() and metadata_path.is_file():
                if not args.resume:
                    raise FileExistsError(f"Refusing to overwrite {image_path}")
                existing = json.loads(metadata_path.read_text())
                expected = {
                    "experiment": args.experiment, "variant": args.variant,
                    "subject": args.subject, "prompt_id": prompt_id,
                    "seed": seed, "scale": args.scale,
                }
                mismatch = {k: (existing.get(k), v) for k, v in expected.items() if existing.get(k) != v}
                if mismatch:
                    raise RuntimeError(f"Existing metadata mismatch {metadata_path}: {mismatch}")
                print(f"SKIP {image_path}", flush=True)
                continue

            directory.mkdir(parents=True, exist_ok=True)
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            image_started = time.perf_counter()
            image = pipe(
                prompt=clean_prompt,
                num_inference_steps=generation["inference_steps"],
                guidance_scale=generation["guidance_scale"],
                height=generation["resolution"], width=generation["resolution"],
                generator=torch.Generator("cuda:0").manual_seed(seed),
                **conditioning,
            ).images[0]
            torch.cuda.synchronize()
            latency = time.perf_counter() - image_started
            peak = torch.cuda.max_memory_allocated() / 1024**3
            image.save(image_path)
            metadata = {
                "method": "ip_adapter_plus" if args.adapter == "plus" else "ip_adapter_plain",
                "dataset": subject["dataset"], "subject": args.subject,
                "prompt_id": prompt_id, "prompt_category": prompt_cfg["category"],
                "generation_prompt": clean_prompt, "evaluation_prompt": clean_prompt,
                "seed": seed, "base_model": "stable-diffusion-v1-5/stable-diffusion-v1-5",
                "checkpoint": str(checkpoint.relative_to(root)), "training_steps": 0,
                "ip_adapter_scale": args.scale, "resolution": generation["resolution"],
                "inference_steps": generation["inference_steps"],
                "guidance_scale": generation["guidance_scale"],
                "negative_prompt": None, "scheduler": pipe.scheduler.__class__.__name__,
                "model_loading_time_sec": load_time, "generation_latency_sec": latency,
                "peak_inference_vram_gb": peak, "precision": "fp16",
                "gpu": torch.cuda.get_device_name(0),
                "mig_uuid": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "reference_images": [str(path.relative_to(root)) for path in used_refs],
                "git_commit": git_commit,
                "experiment": args.experiment, "variant": args.variant,
                "scale": args.scale, "rank": None, "step": 0,
                "n_refs": len(used_refs),
                "ref_files_used": [path.name for path in used_refs],
                "dtype": "float16", "adapter_type": args.adapter,
                "reference_mode": args.reference_mode,
            }
            metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
            print(f"SAVED {image_path} latency={latency:.3f}s peak={peak:.3f}GiB", flush=True)


if __name__ == "__main__":
    main()
