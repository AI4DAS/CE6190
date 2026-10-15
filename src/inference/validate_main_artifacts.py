#!/usr/bin/env python3
"""Load every final main-experiment artifact and generate one validation image."""

from __future__ import annotations

import argparse
import gc
import json
import time
import traceback
from pathlib import Path

import torch
import yaml
from diffusers import StableDiffusionPipeline


METHODS = ("dreambooth", "lora", "textual_inversion")


def run_dir(root: Path, subject: str, method: str) -> Path:
    if method == "dreambooth":
        return root / f"outputs/training/dreambooth/{subject}_db_s800_seed42"
    if method == "lora":
        return root / f"outputs/training/lora/{subject}_lora_r16_s800_seed42"
    return root / f"outputs/training/textual_inversion/{subject}_ti_s3000_seed42"


def validate(root: Path, output_dir: Path, subject: str, method: str, record: dict) -> dict:
    base = root / "models/base/stable-diffusion-v1-5"
    trained = run_dir(root, subject, method)
    image_path = output_dir / f"{subject}_{method}.png"
    json_path = image_path.with_suffix(".json")
    if image_path.is_file() and json_path.is_file():
        existing = json.loads(json_path.read_text())
        if existing.get("status") == "PASS":
            return existing

    if method == "dreambooth":
        required = [trained / "model_index.json", trained / "scheduler/scheduler_config.json",
                    trained / "unet/diffusion_pytorch_model.safetensors"]
        model_path = trained
    elif method == "lora":
        required = [trained / "pytorch_lora_weights.safetensors"]
        model_path = base
    else:
        required = [trained / "learned_embeds.safetensors"]
        model_path = base
    missing = [str(path) for path in required if not path.is_file() or path.stat().st_size == 0]
    if missing:
        raise FileNotFoundError(f"Missing or empty final artifacts: {missing}")

    started = time.perf_counter()
    pipe = StableDiffusionPipeline.from_pretrained(
        model_path, torch_dtype=torch.float16, safety_checker=None, local_files_only=True
    )
    if method == "lora":
        pipe.load_lora_weights(trained, weight_name="pytorch_lora_weights.safetensors")
        prompt = record["instance_prompt"]
    elif method == "textual_inversion":
        pipe.load_textual_inversion(trained, weight_name="learned_embeds.safetensors")
        prompt = f"a photo of {record['placeholder_token']}"
    else:
        prompt = record["instance_prompt"]
    pipe.to("cuda:0")
    load_time = time.perf_counter() - started
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    generated = time.perf_counter()
    image = pipe(
        prompt,
        num_inference_steps=10,
        guidance_scale=7.5,
        height=512,
        width=512,
        generator=torch.Generator("cuda:0").manual_seed(42),
    ).images[0]
    torch.cuda.synchronize()
    latency = time.perf_counter() - generated
    peak = torch.cuda.max_memory_allocated() / 1024**3
    image.save(image_path)
    result = {
        "subject": subject,
        "method": method,
        "status": "PASS",
        "run_dir": str(trained.relative_to(root)),
        "prompt": prompt,
        "seed": 42,
        "inference_steps": 10,
        "pipeline_load_time_sec": load_time,
        "generation_latency_sec": latency,
        "peak_allocated_vram_gb": peak,
        "output_image": str(image_path.relative_to(root)),
    }
    json_path.write_text(json.dumps(result, indent=2) + "\n")
    del pipe
    gc.collect()
    torch.cuda.empty_cache()
    return result


def write_report(path: Path, results: list[dict]) -> None:
    passed = sum(result["status"] == "PASS" for result in results)
    lines = [
        "# Main Artifact Validation",
        "",
        f"Validated artifacts: **{passed}/{len(results)} passed**.",
        "",
        "Each test loaded the final artifact on one visible MIG device and generated one 512×512 image with seed 42 and 10 denoising steps.",
        "",
        "| Subject | Method | Status | Load time (s) | Generation (s) | Peak VRAM (GiB) | Validation image |",
        "|---|---|---|---:|---:|---:|---|",
    ]
    for result in results:
        if result["status"] == "PASS":
            lines.append(
                f"| {result['subject']} | {result['method']} | PASS | "
                f"{result['pipeline_load_time_sec']:.3f} | {result['generation_latency_sec']:.3f} | "
                f"{result['peak_allocated_vram_gb']:.3f} | `{result['output_image']}` |"
            )
        else:
            error = result["error"].replace("|", "\\|").replace("\n", " ")
            lines.append(f"| {result['subject']} | {result['method']} | FAIL: {error} | N/A | N/A | N/A | N/A |")
    lines.extend([
        "",
        "These are compatibility checks, not benchmark-quality samples. Intermediate `checkpoint-*` directories were not used.",
        "",
    ])
    path.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.project_root.resolve()
    if torch.cuda.device_count() != 1:
        raise RuntimeError(f"Expected one visible MIG device, got {torch.cuda.device_count()}")
    subjects = yaml.safe_load((root / "configs/main_subjects.yaml").read_text())["subjects"]
    output_dir = root / "results/figures/main_artifact_validation"
    output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for subject, record in subjects.items():
        for method in METHODS:
            print(f"VALIDATING subject={subject} method={method}", flush=True)
            try:
                result = validate(root, output_dir, subject, method, record)
            except Exception as exc:
                traceback.print_exc()
                result = {"subject": subject, "method": method, "status": "FAIL",
                          "error": f"{type(exc).__name__}: {exc}"}
            results.append(result)
            print(f"RESULT subject={subject} method={method} status={result['status']}", flush=True)
    report = root / "results/system/main_artifact_validation.md"
    write_report(report, results)
    print(f"WROTE {report}")
    if any(result["status"] != "PASS" for result in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
