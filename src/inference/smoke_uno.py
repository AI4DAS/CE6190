#!/usr/bin/env python3
"""Single- and multi-reference UNO compatibility smoke tests."""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
from PIL import Image


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--uno-repo", type=Path, required=True)
    parser.add_argument("--references", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prompt", default="a photo of a dog on a beach")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    repo = args.uno_repo.resolve()
    sys.path.insert(0, str(repo))
    from uno.flux.pipeline import UNOPipeline, preprocess_ref  # noqa: PLC0415

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Smoke test requires exactly one visible CUDA device")
    if len(args.references) < 2:
        raise ValueError("Provide at least two references for single/multi smoke tests")
    for path in args.references:
        if not path.is_file():
            raise FileNotFoundError(path)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    torch.cuda.reset_peak_memory_stats()
    load_start = time.perf_counter()
    pipeline = UNOPipeline(
        "flux-dev-fp8", torch.device("cuda:0"), offload=True,
        only_lora=True, lora_rank=512,
    )
    torch.cuda.synchronize()
    load_seconds = time.perf_counter() - load_start
    load_peak_gib = torch.cuda.max_memory_allocated() / 1024**3

    modes = {
        "single": [args.references[0]],
        "multi": args.references[:2],
    }
    for mode, paths in modes.items():
        long_side = 512 if len(paths) == 1 else 320
        references = [
            preprocess_ref(Image.open(path).convert("RGB"), long_side)
            for path in paths
        ]
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        started = time.perf_counter()
        image = pipeline(
            prompt=args.prompt,
            width=512,
            height=512,
            guidance=4.0,
            num_steps=25,
            seed=args.seed,
            ref_imgs=references,
            pe="d",
        )
        torch.cuda.synchronize()
        latency = time.perf_counter() - started
        peak_gib = torch.cuda.max_memory_allocated() / 1024**3
        output = args.output_dir / f"uno_dog2_{mode}.png"
        image.save(output)
        metadata = {
            "method": "uno",
            "purpose": "compatibility_smoke_test",
            "reference_mode": mode,
            "reference_images": [str(path.resolve()) for path in paths],
            "reference_count": len(paths),
            "prompt": args.prompt,
            "seed": args.seed,
            "model_type": "flux-dev-fp8",
            "resolution": [512, 512],
            "inference_steps": 25,
            "guidance_scale": 4.0,
            "offload": True,
            "load_time_sec": load_seconds,
            "load_peak_allocated_vram_gib": load_peak_gib,
            "generation_latency_sec": latency,
            "generation_peak_allocated_vram_gib": peak_gib,
            "gpu": torch.cuda.get_device_name(0),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
        print(json.dumps(metadata, indent=2))
        print(f"UNO_{mode.upper()}_SMOKE_SUCCESS")


if __name__ == "__main__":
    main()
