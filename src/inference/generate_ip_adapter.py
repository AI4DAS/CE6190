#!/usr/bin/env python3
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


def git_revision(root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "UNCOMMITTED"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--subject", default="dog2")
    parser.add_argument("--reference-image", type=Path)
    parser.add_argument("--scale", type=float, default=0.7)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456, 789])
    args = parser.parse_args()
    root = args.project_root.resolve()
    reference_path = args.reference_image or root / "data/processed/dreambooth/dog2/images/00.jpg"
    prompts = yaml.safe_load((root / "configs/prompts.yaml").read_text())["subjects"][args.subject]["prompts"]
    base_model = root / "models/base/stable-diffusion-v1-5"
    adapter = root / "models/ip_adapter/IP-Adapter"

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Select exactly one MIG UUID with CUDA_VISIBLE_DEVICES; it will be cuda:0.")
    pipe = StableDiffusionPipeline.from_pretrained(
        base_model,
        torch_dtype=torch.float16,
        safety_checker=None,
        local_files_only=True,
    )
    pipe.load_ip_adapter(
        adapter,
        subfolder="models",
        weight_name="ip-adapter-plus_sd15.bin",
        image_encoder_folder="image_encoder",
        local_files_only=True,
    )
    pipe.set_ip_adapter_scale(args.scale)
    pipe.to("cuda:0")
    reference = Image.open(reference_path).convert("RGB")
    output_root = root / "outputs/ip_adapter/dreambooth" / args.subject
    device_name = torch.cuda.get_device_name(0)
    mig_uuid = os.environ.get("CUDA_VISIBLE_DEVICES", "")

    for prompt_id, entry in prompts.items():
        prompt = entry["ip_adapter"]
        prompt_dir = output_root / prompt_id
        prompt_dir.mkdir(parents=True, exist_ok=True)
        for seed in args.seeds:
            image_path = prompt_dir / f"seed_{seed}.png"
            metadata_path = prompt_dir / f"seed_{seed}.json"
            if image_path.exists() or metadata_path.exists():
                raise FileExistsError(f"Refusing to overwrite existing output: {image_path}")
            generator = torch.Generator(device="cuda:0").manual_seed(seed)
            torch.cuda.reset_peak_memory_stats()
            started = time.perf_counter()
            image = pipe(
                prompt=prompt,
                ip_adapter_image=reference,
                num_inference_steps=50,
                guidance_scale=7.5,
                generator=generator,
                height=512,
                width=512,
            ).images[0]
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            image.save(image_path)
            metadata = {
                "method": "ip_adapter_plus",
                "dataset": "dreambooth",
                "subject": args.subject,
                "prompt_id": prompt_id,
                "prompt_category": entry["category"],
                "prompt": prompt,
                "seed": seed,
                "base_model": "stable-diffusion-v1-5/stable-diffusion-v1-5",
                "training_steps": None,
                "lora_rank": None,
                "ip_adapter_scale": args.scale,
                "generation_steps": 50,
                "guidance_scale": 7.5,
                "generation_time_sec": elapsed,
                "peak_vram_gb": torch.cuda.max_memory_allocated() / 1e9,
                "gpu": device_name,
                "mig_uuid": mig_uuid,
                "reference_images": [str(reference_path.relative_to(root))],
                "checkpoint_path": str(adapter.relative_to(root)),
                "git_commit": git_revision(root),
                "generation_command": " ".join(os.sys.argv),
            }
            metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
            print(f"saved {image_path.relative_to(root)} ({elapsed:.2f}s)", flush=True)


if __name__ == "__main__":
    main()

