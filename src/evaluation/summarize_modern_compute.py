#!/usr/bin/env python3
"""Summarize measured modern-extension compute metadata and local shared weights."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.project_root.resolve()
    rows = []
    for method in ("dreamo", "uno"):
        metadata = [json.loads(path.read_text()) for path in sorted(
            (root / "outputs/modern_extension" / method).glob("*/*/P*/seed_*.json")
        )]
        if len(metadata) != 192:
            raise RuntimeError(f"Expected 192 metadata records for {method}, found {len(metadata)}")
        rows.extend(metadata)
    frame = pd.DataFrame(rows)

    dreamo_dir = root / "third_party/modern/DreamO/models"
    dreamo_files = [
        dreamo_dir / "dreamo.safetensors",
        dreamo_dir / "dreamo_cfg_distill.safetensors",
        dreamo_dir / "v1.1/dreamo_sft_lora.safetensors",
        dreamo_dir / "v1.1/dreamo_dpo_lora.safetensors",
        dreamo_dir / "diffusion_pytorch_model.safetensors",
    ]
    uno_files = list((Path.home() / ".cache/huggingface/hub/models--bytedance-research--UNO").glob(
        "snapshots/*/dit_lora.safetensors"
    ))
    if not all(path.is_file() for path in dreamo_files) or len(uno_files) != 1:
        raise RuntimeError("Expected local DreamO/UNO shared checkpoints were not found uniquely")
    shared_sizes = {
        "dreamo": sum(path.stat().st_size for path in dreamo_files) / 1024**2,
        "uno": uno_files[0].stat().st_size / 1024**2,
    }

    output = []
    for method, selected in frame.groupby("method"):
        configuration = selected.iloc[0]["configuration"]
        output.append({
            "method": method,
            "images": len(selected),
            "subjects": selected["subject"].nunique(),
            "subject_specific_training_required": False,
            "subject_specific_training_time_sec": 0,
            "per_subject_checkpoint_size_mib": 0,
            "shared_extension_weights_mib": shared_sizes[method],
            "reference_images_per_inference": int(selected["reference_count"].iloc[0]),
            "resolution": int(configuration["resolution"]),
            "inference_steps": int(configuration["inference_steps"]),
            "guidance_scale": float(configuration["guidance_scale"]),
            "precision_quantization": configuration.get("quantization", configuration.get("model_type")),
            "inference_latency_mean_sec": selected["generation_latency_sec"].mean(),
            "inference_latency_std_sec": selected["generation_latency_sec"].std(),
            "peak_inference_vram_gib": selected["peak_inference_vram_gib"].max(),
            "checkpoint": selected["checkpoint"].iloc[0],
            "base_model": selected["base_model"].iloc[0],
            "storage_note": "Shared extension weights only; FLUX.1-dev and preprocessing models excluded",
        })
    destination = root / "results/metrics/modern_compute_cost.csv"
    pd.DataFrame(output).to_csv(destination, index=False)
    print(pd.DataFrame(output).to_string(index=False))


if __name__ == "__main__":
    main()
