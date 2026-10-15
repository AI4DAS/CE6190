#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
import torch
from diffusers import StableDiffusionPipeline

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prompt", default="a photo of a dog")
    parser.add_argument("--num-images", type=int, default=200)
    parser.add_argument("--seed", type=int, default=10000)
    args = parser.parse_args()
    if torch.cuda.device_count() != 1:
        raise RuntimeError(f"Expected exactly one visible MIG GPU, got {torch.cuda.device_count()}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pipe = StableDiffusionPipeline.from_pretrained(args.model, torch_dtype=torch.float16, safety_checker=None, local_files_only=True).to("cuda:0")
    for index in range(args.num_images):
        path = args.output_dir / f"{index:05d}.jpg"
        if path.exists():
            continue
        generator = torch.Generator(device="cuda:0").manual_seed(args.seed + index)
        image = pipe(args.prompt, num_inference_steps=50, guidance_scale=7.5, generator=generator).images[0]
        image.save(path, quality=95)
        print(f"CLASS_IMAGE {index + 1}/{args.num_images} {path}", flush=True)
    # Diffusers DreamBooth examples enumerate every entry in class_data_dir and
    # attempt to open it with Pillow. Keep provenance beside, not inside, the
    # image-only directory.
    metadata_path = args.output_dir.parent / f"{args.output_dir.name}_metadata.json"
    metadata_path.write_text(json.dumps({"prompt": args.prompt, "num_images": args.num_images, "seed_start": args.seed, "model": str(args.model), "generation_steps": 50, "guidance_scale": 7.5}, indent=2) + "\n")
    count = len(list(args.output_dir.glob("*.jpg")))
    if count < args.num_images:
        raise RuntimeError(f"Expected {args.num_images} class images, found {count}")
    print("CLASS_IMAGE_PREPARATION_COMPLETED_SUCCESSFULLY")

if __name__ == "__main__":
    main()

