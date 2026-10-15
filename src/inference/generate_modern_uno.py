#!/usr/bin/env python3
"""Generate the eight-subject, 192-image single-reference UNO extension set."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
import yaml
from PIL import Image


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--uno-repo", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--resume", action="store_true", help="Skip complete, validated PNG/JSON pairs")
    parser.add_argument(
        "--subjects", nargs="+", default=None,
        help="Optional subject IDs to generate; omitted means all configured subjects.",
    )
    args = parser.parse_args()
    root, repo = args.project_root.resolve(), args.uno_repo.resolve()
    config = yaml.safe_load(args.config.read_text())
    sys.path.insert(0, str(repo))
    from uno.flux.pipeline import UNOPipeline, preprocess_ref  # noqa: PLC0415

    if torch.cuda.device_count() != 1:
        raise RuntimeError("Exactly one MIG must be visible")
    settings = config["uno"]
    load_started = time.perf_counter()
    pipeline = UNOPipeline(
        settings["model_type"], torch.device("cuda:0"), offload=settings["offload"],
        only_lora=True, lora_rank=settings["lora_rank"],
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
        reference = preprocess_ref(Image.open(reference_path).convert("RGB"), 512)
        for prompt_id, prompt_cfg in config["prompts"].items():
            prompt = prompt_cfg["template"].format(class_name=subject_cfg["class_name"])
            for seed in config["seeds"]:
                directory = root / "outputs/modern_extension/uno" / subject_cfg["dataset"] / subject / prompt_id
                image_path, json_path = directory / f"seed_{seed}.png", directory / f"seed_{seed}.json"
                if image_path.exists() and json_path.exists() and args.resume:
                    with Image.open(image_path) as existing_image:
                        existing_image.verify()
                    existing_metadata = json.loads(json_path.read_text())
                    expected = {
                        "method": "uno", "dataset": subject_cfg["dataset"],
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
                image = pipeline(
                    prompt=prompt, width=settings["resolution"], height=settings["resolution"],
                    guidance=settings["guidance_scale"], num_steps=settings["inference_steps"],
                    seed=seed, ref_imgs=[reference], pe="d",
                )
                torch.cuda.synchronize()
                latency = time.perf_counter() - started
                peak = torch.cuda.max_memory_allocated() / 1024**3
                image.save(image_path)
                metadata = {
                    "method": "uno", "dataset": subject_cfg["dataset"], "subject": subject,
                    "category": subject_cfg["category"], "prompt_id": prompt_id,
                    "prompt_category": prompt_cfg["category"], "generation_prompt": prompt,
                    "evaluation_prompt": prompt, "seed": seed,
                    "reference_images": [str(reference_path.resolve())], "reference_count": 1,
                    "checkpoint": "bytedance-research/UNO@cd4a5a8896243db22aaf4a2d3d2a9511c437fb12",
                    "base_model": "black-forest-labs/FLUX.1-dev@3de623fc3c33e44ffbe2bad470d0f45bccf2eb21",
                    "configuration": settings, "load_time_sec": load_seconds,
                    "generation_latency_sec": latency, "peak_inference_vram_gib": peak,
                    "gpu": torch.cuda.get_device_name(0),
                    "mig_uuid": os.environ.get("CUDA_VISIBLE_DEVICES"),
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
                json_path.write_text(json.dumps(metadata, indent=2) + "\n")
                print(f"SAVED {image_path} latency={latency:.3f}s peak={peak:.3f}GiB", flush=True)
    print("UNO_EXTENSION_COMPLETE")


if __name__ == "__main__":
    main()
