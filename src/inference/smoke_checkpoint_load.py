#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import torch
from diffusers import StableDiffusionPipeline


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--method", choices=["dreambooth", "lora", "textual_inversion"], required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.project_root.resolve()
    run_dir = args.run_dir.resolve()
    base = root / "models/base/stable-diffusion-v1-5"
    output_dir = root / "results/figures/smoke"
    output_dir.mkdir(parents=True, exist_ok=True)
    if torch.cuda.device_count() != 1:
        raise RuntimeError(f"Expected exactly one visible GPU, got {torch.cuda.device_count()}")

    model_path = run_dir if args.method == "dreambooth" else base
    pipe = StableDiffusionPipeline.from_pretrained(
        model_path, torch_dtype=torch.float16, safety_checker=None, local_files_only=True
    ).to("cuda:0")
    if args.method == "lora":
        pipe.load_lora_weights(run_dir, weight_name="pytorch_lora_weights.safetensors")
        prompt = "a photo of sks dog"
    elif args.method == "textual_inversion":
        pipe.load_textual_inversion(run_dir, weight_name="learned_embeds.safetensors")
        prompt = "a photo of <sks-dog>"
    else:
        prompt = "a photo of sks dog"

    generator = torch.Generator(device="cuda:0").manual_seed(42)
    image = pipe(prompt, num_inference_steps=10, guidance_scale=7.5, generator=generator).images[0]
    output_path = output_dir / f"{args.method}.png"
    image.save(output_path)
    metadata = {
        "method": args.method,
        "smoke_run_dir": str(run_dir.relative_to(root)),
        "prompt": prompt,
        "seed": 42,
        "num_inference_steps": 10,
        "checkpoint_load_success": True,
    }
    output_path.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"CHECKPOINT_LOAD_SUCCESS method={args.method} output={output_path}")


if __name__ == "__main__":
    main()

