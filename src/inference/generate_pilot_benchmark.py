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


METHODS = ("dreambooth", "lora", "textual_inversion", "ip_adapter_plus")
SEEDS = (42, 123, 456, 789)


def revision(root: Path) -> str:
    return subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()


def load_pipeline(root: Path, method: str):
    base = root / "models/base/stable-diffusion-v1-5"
    db = root / "outputs/training/dreambooth/dog2_db_s800_seed42"
    lora = root / "outputs/training/lora/dog2_lora_r16_s800_seed42"
    ti = root / "outputs/training/textual_inversion/dog2_ti_s3000_seed42"
    started = time.perf_counter()
    model_path = db if method == "dreambooth" else base
    pipe = StableDiffusionPipeline.from_pretrained(
        model_path, torch_dtype=torch.float16, safety_checker=None, local_files_only=True
    )
    checkpoint = model_path
    if method == "lora":
        pipe.load_lora_weights(lora, weight_name="pytorch_lora_weights.safetensors")
        checkpoint = lora / "pytorch_lora_weights.safetensors"
    elif method == "textual_inversion":
        pipe.load_textual_inversion(ti, weight_name="learned_embeds.safetensors")
        checkpoint = ti / "learned_embeds.safetensors"
    elif method == "ip_adapter_plus":
        adapter = root / "models/ip_adapter/IP-Adapter"
        pipe.load_ip_adapter(
            adapter, subfolder="models", weight_name="ip-adapter-plus_sd15.bin",
            image_encoder_folder="image_encoder", local_files_only=True,
        )
        pipe.set_ip_adapter_scale(0.7)
        checkpoint = adapter / "models/ip-adapter-plus_sd15.bin"
    pipe.to("cuda:0")
    torch.cuda.synchronize()
    return pipe, checkpoint, time.perf_counter() - started


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--method", choices=METHODS, required=True)
    args = parser.parse_args()
    root = args.project_root.resolve()
    if torch.cuda.device_count() != 1:
        raise RuntimeError(f"Expected one visible MIG device, got {torch.cuda.device_count()}")
    prompts = yaml.safe_load((root / "configs/prompts.yaml").read_text())["subjects"]["dog2"]["prompts"]
    output_root = root / "outputs/evaluation" / args.method / "dreambooth/dog2"
    if output_root.exists() and any(output_root.rglob("seed_*.png")):
        raise FileExistsError(f"Refusing to overwrite evaluation outputs: {output_root}")
    pipe, checkpoint, load_time = load_pipeline(root, args.method)
    reference_path = root / "data/processed/dreambooth/dog2/images/00.jpg"
    reference = Image.open(reference_path).convert("RGB")
    prompt_key = "textual_inversion" if args.method == "textual_inversion" else (
        "ip_adapter" if args.method == "ip_adapter_plus" else "personalized"
    )
    # Untimed warm-up ensures reported values exclude one-time CUDA initialization.
    warmup_args = dict(prompt=prompts["P0"][prompt_key], num_inference_steps=2, guidance_scale=7.5,
                       height=512, width=512, generator=torch.Generator("cuda:0").manual_seed(999))
    if args.method == "ip_adapter_plus":
        warmup_args["ip_adapter_image"] = reference
    pipe(**warmup_args)
    torch.cuda.synchronize()
    torch.cuda.empty_cache()

    for prompt_id, entry in prompts.items():
        prompt_dir = output_root / prompt_id
        prompt_dir.mkdir(parents=True, exist_ok=True)
        for seed in SEEDS:
            generation_prompt = entry[prompt_key]
            call_args = dict(
                prompt=generation_prompt, num_inference_steps=50, guidance_scale=7.5,
                height=512, width=512, generator=torch.Generator("cuda:0").manual_seed(seed),
            )
            if args.method == "ip_adapter_plus":
                call_args["ip_adapter_image"] = reference
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            started = time.perf_counter()
            image = pipe(**call_args).images[0]
            torch.cuda.synchronize()
            latency = time.perf_counter() - started
            peak = torch.cuda.max_memory_allocated() / 1024**3
            image_path = prompt_dir / f"seed_{seed}.png"
            image.save(image_path)
            metadata = {
                "method": args.method, "dataset": "dreambooth", "subject": "dog2",
                "prompt_id": prompt_id, "prompt_category": entry["category"],
                "generation_prompt": generation_prompt, "evaluation_prompt": entry["evaluation"],
                "seed": seed, "base_model": "stable-diffusion-v1-5/stable-diffusion-v1-5",
                "checkpoint": str(checkpoint.relative_to(root)),
                "training_steps": {"dreambooth": 800, "lora": 800, "textual_inversion": 3000}.get(args.method),
                "rank": 16 if args.method == "lora" else None,
                "ip_adapter_scale": 0.7 if args.method == "ip_adapter_plus" else None,
                "resolution": 512, "inference_steps": 50, "guidance_scale": 7.5,
                "scheduler": pipe.scheduler.__class__.__name__,
                "model_loading_time_sec": load_time, "generation_latency_sec": latency,
                "peak_inference_vram_gb": peak, "precision": "fp16",
                "gpu": torch.cuda.get_device_name(0), "mig_uuid": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "reference_images": [str(reference_path.relative_to(root))] if args.method == "ip_adapter_plus" else [],
                "git_commit": revision(root),
            }
            image_path.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
            print(f"{args.method} {prompt_id} seed={seed} latency={latency:.4f}s peak={peak:.4f}GiB", flush=True)


if __name__ == "__main__":
    main()
