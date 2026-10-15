#!/usr/bin/env python3
"""Generate the eight-subject, 192-image DreamO extension set."""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--dreamo-repo", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--resume", action="store_true", help="Skip complete, validated PNG/JSON pairs")
    parser.add_argument(
        "--subjects", nargs="+", default=None,
        help="Optional subject IDs to generate; omitted means all configured subjects.",
    )
    args = parser.parse_args()
    root, repo = args.project_root.resolve(), args.dreamo_repo.resolve()
    config = yaml.safe_load(args.config.read_text())
    sys.path.insert(0, str(repo))
    from dreamo_generator import Generator  # noqa: PLC0415

    if torch.cuda.device_count() != 1:
        raise RuntimeError("Exactly one MIG must be visible")
    settings = config["dreamo"]
    load_started = time.perf_counter()
    generator = Generator(
        version=settings["version"], offload=settings["offload"],
        no_turbo=False, quant=settings["quantization"], device="cuda",
    )
    torch.cuda.synchronize()
    load_seconds = time.perf_counter() - load_started

    configured_subjects = config["subjects"]
    selected_subjects = args.subjects or list(configured_subjects)
    unknown = sorted(set(selected_subjects) - set(configured_subjects))
    if unknown:
        raise ValueError(f"Unknown subject IDs: {unknown}")
    for subject in selected_subjects:
        subject_cfg = configured_subjects[subject]
        reference_path = sorted((root / subject_cfg["reference_dir"]).glob("*"))[0]
        reference = np.asarray(Image.open(reference_path).convert("RGB"))
        ref_conds, _, _ = generator.pre_condition(
            ref_images=[reference, None], ref_tasks=["ip", "ip"], ref_res=512, seed=42
        )
        for prompt_id, prompt_cfg in config["prompts"].items():
            prompt = prompt_cfg["template"].format(class_name=subject_cfg["class_name"])
            for seed in config["seeds"]:
                directory = root / "outputs/modern_extension/dreamo" / subject_cfg["dataset"] / subject / prompt_id
                image_path, json_path = directory / f"seed_{seed}.png", directory / f"seed_{seed}.json"
                if image_path.exists() and json_path.exists() and args.resume:
                    with Image.open(image_path) as existing_image:
                        existing_image.verify()
                    existing_metadata = json.loads(json_path.read_text())
                    expected = {
                        "method": "dreamo", "dataset": subject_cfg["dataset"],
                        "subject": subject, "prompt_id": prompt_id, "seed": seed,
                        "generation_prompt": prompt,
                    }
                    mismatches = {
                        key: (existing_metadata.get(key), value)
                        for key, value in expected.items() if existing_metadata.get(key) != value
                    }
                    if mismatches:
                        raise ValueError(f"Existing metadata mismatch for {json_path}: {mismatches}")
                    print(f"SKIP complete {image_path}", flush=True)
                    continue
                if image_path.exists() or json_path.exists():
                    raise FileExistsError(
                        f"Incomplete or protected output pair; refusing to overwrite: {image_path}, {json_path}"
                    )
                directory.mkdir(parents=True, exist_ok=True)
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()
                started = time.perf_counter()
                image = generator.dreamo_pipeline(
                    prompt=prompt, width=settings["resolution"], height=settings["resolution"],
                    num_inference_steps=settings["inference_steps"],
                    guidance_scale=settings["guidance_scale"], ref_conds=ref_conds,
                    generator=torch.Generator(device="cpu").manual_seed(seed),
                    true_cfg_scale=1.0, true_cfg_start_step=1, true_cfg_end_step=1,
                    negative_prompt="", neg_guidance_scale=3.5,
                    first_step_guidance_scale=settings["guidance_scale"],
                ).images[0]
                torch.cuda.synchronize()
                latency = time.perf_counter() - started
                peak = torch.cuda.max_memory_allocated() / 1024**3
                image.save(image_path)
                metadata = {
                    "method": "dreamo", "dataset": subject_cfg["dataset"], "subject": subject,
                    "category": subject_cfg["category"], "prompt_id": prompt_id,
                    "prompt_category": prompt_cfg["category"], "generation_prompt": prompt,
                    "evaluation_prompt": prompt, "seed": seed,
                    "reference_images": [str(reference_path.resolve())], "reference_count": 1,
                    "checkpoint": "ByteDance/DreamO@eb693ac03bc364d17ec4fe04a3c6ffcad038ac0c",
                    "base_model": "black-forest-labs/FLUX.1-dev@3de623fc3c33e44ffbe2bad470d0f45bccf2eb21",
                    "configuration": settings, "load_time_sec": load_seconds,
                    "generation_latency_sec": latency, "peak_inference_vram_gib": peak,
                    "gpu": torch.cuda.get_device_name(0),
                    "mig_uuid": __import__("os").environ.get("CUDA_VISIBLE_DEVICES"),
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
                json_path.write_text(json.dumps(metadata, indent=2) + "\n")
                print(f"SAVED {image_path} latency={latency:.3f}s peak={peak:.3f}GiB", flush=True)
    print("DREAMO_EXTENSION_COMPLETE")


if __name__ == "__main__":
    main()
