#!/usr/bin/env python3
"""One-image DreamO compatibility smoke test; not a benchmark run."""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from PIL import Image


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dreamo-repo", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prompt", default="a photo of a dog on a beach")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--quant", choices=["none", "int8"], default="none")
    args = parser.parse_args()

    repo = args.dreamo_repo.resolve()
    sys.path.insert(0, str(repo))
    from dreamo_generator import Generator  # noqa: PLC0415

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Smoke test requires exactly one visible CUDA device")
    if not args.reference.is_file():
        raise FileNotFoundError(args.reference)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    torch.cuda.reset_peak_memory_stats()
    load_start = time.perf_counter()
    generator = Generator(
        version="v1.1", offload=True, no_turbo=False, quant=args.quant, device="cuda"
    )
    torch.cuda.synchronize()
    load_seconds = time.perf_counter() - load_start

    reference = np.asarray(Image.open(args.reference).convert("RGB"))
    ref_conds, _, seed = generator.pre_condition(
        ref_images=[reference, None], ref_tasks=["ip", "ip"], ref_res=512, seed=args.seed
    )
    torch.cuda.synchronize()
    generation_start = time.perf_counter()
    image = generator.dreamo_pipeline(
        prompt=args.prompt,
        width=768,
        height=768,
        num_inference_steps=12,
        guidance_scale=4.5,
        ref_conds=ref_conds,
        generator=torch.Generator(device="cpu").manual_seed(seed),
        true_cfg_scale=1.0,
        true_cfg_start_step=1,
        true_cfg_end_step=1,
        negative_prompt="",
        neg_guidance_scale=3.5,
        first_step_guidance_scale=4.5,
    ).images[0]
    torch.cuda.synchronize()
    generation_seconds = time.perf_counter() - generation_start
    peak_gib = torch.cuda.max_memory_allocated() / 1024**3
    image.save(args.output)
    metadata = {
        "method": "dreamo",
        "purpose": "compatibility_smoke_test",
        "reference": str(args.reference.resolve()),
        "prompt": args.prompt,
        "seed": seed,
        "version": "v1.1",
        "resolution": [768, 768],
        "inference_steps": 12,
        "guidance_scale": 4.5,
        "quantization": args.quant,
        "offload": True,
        "load_time_sec": load_seconds,
        "generation_latency_sec": generation_seconds,
        "peak_allocated_vram_gib": peak_gib,
        "gpu": torch.cuda.get_device_name(0),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    args.output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))
    print("DREAMO_SMOKE_SUCCESS")


if __name__ == "__main__":
    main()
