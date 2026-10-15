#!/usr/bin/env python3
"""Build the requested per-image metric table for main and ablation runs."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import torch
import yaml
from PIL import Image
from transformers import AutoImageProcessor, AutoModel, AutoProcessor, CLIPModel


def normalize(value: torch.Tensor) -> torch.Tensor:
    return value / value.norm(dim=-1, keepdim=True)


def records(root: Path) -> list[dict]:
    paths = sorted((root / "outputs/main_evaluation").glob("*/*/*/P*/seed_*.json"))
    paths += sorted((root / "outputs/experiments").glob("**/P*/seed_*.json"))
    output = []
    for path in paths:
        row = json.loads(path.read_text())
        image = path.with_suffix(".png")
        if not image.is_file():
            raise FileNotFoundError(image)
        if "subject" not in row or "prompt_id" not in row:
            continue
        if "main_evaluation" in path.parts:
            experiment = "main_benchmark"
            variant = {
                "base_model": "base_no_personalization",
                "ip_adapter_plus": "plus_s0p3",
            }.get(row["method"], "main_default")
        else:
            experiment = row.get("experiment", "")
            variant = row.get("variant", row["method"])
        output.append({**row, "experiment": experiment, "variant": variant, "image_file": image})
    if len({str(row["image_file"]) for row in output}) != len(output):
        raise RuntimeError("Duplicate image path in master input")
    return output


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    if torch.cuda.device_count() != 1:
        raise RuntimeError(f"Expected one visible MIG, got {torch.cuda.device_count()}")
    rows = records(root)
    if not rows:
        raise RuntimeError("No generated records found")
    subjects = yaml.safe_load((root / "configs/main_subjects.yaml").read_text())["subjects"]
    unknown = sorted({row["subject"] for row in rows} - set(subjects))
    if unknown:
        raise RuntimeError(f"Unknown subjects in generated records: {unknown}")
    device = "cuda:0"

    dino_processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
    dino_model = AutoModel.from_pretrained("facebook/dinov2-base").eval().to(device)
    reference_features, reference_names = {}, {}
    with torch.inference_mode():
        for subject in sorted({row["subject"] for row in rows}):
            paths = sorted(path for path in (root / subjects[subject]["image_dir"]).iterdir() if path.is_file())
            images = [Image.open(path).convert("RGB") for path in paths]
            values = dino_model(**dino_processor(images=images, return_tensors="pt").to(device)).last_hidden_state[:, 0]
            reference_features[subject] = normalize(values).cpu()
            reference_names[subject] = [path.name for path in paths]
    embeddings = []
    for start in range(0, len(rows), 24):
        batch = rows[start:start + 24]
        images = [Image.open(row["image_file"]).convert("RGB") for row in batch]
        with torch.inference_mode():
            values = dino_model(**dino_processor(images=images, return_tensors="pt").to(device)).last_hidden_state[:, 0]
        embeddings.extend(normalize(values).cpu())
    del dino_model
    torch.cuda.empty_cache()

    clip_processor = AutoProcessor.from_pretrained("openai/clip-vit-large-patch14")
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").eval().to(device)
    clip_raw, clip_neutral = [], []
    for start in range(0, len(rows), 16):
        batch = rows[start:start + 16]
        images = [Image.open(row["image_file"]).convert("RGB") for row in batch]
        image_inputs = clip_processor(images=images, return_tensors="pt").to(device)
        raw_inputs = clip_processor(
            text=[row["generation_prompt"] for row in batch], return_tensors="pt", padding=True
        ).to(device)
        neutral_inputs = clip_processor(
            text=[row["evaluation_prompt"] for row in batch], return_tensors="pt", padding=True
        ).to(device)
        with torch.inference_mode():
            image_features = normalize(clip_model.get_image_features(pixel_values=image_inputs["pixel_values"]))
            raw_features = normalize(clip_model.get_text_features(**raw_inputs))
            neutral_features = normalize(clip_model.get_text_features(**neutral_inputs))
        clip_raw.extend((image_features * raw_features).sum(1).float().cpu().tolist())
        clip_neutral.extend((image_features * neutral_features).sum(1).float().cpu().tolist())

    output = []
    for index, (row, embedding) in enumerate(zip(rows, embeddings)):
        similarities = embedding[None] @ reference_features[row["subject"]].T
        nearest = int(similarities.argmax())
        record = {
            "experiment": row["experiment"], "variant": row["variant"],
            "object": row["subject"], "method": row["method"], "seed": int(row["seed"]),
            "prompt_id": row["prompt_id"], "prompt_category": row["prompt_category"],
            "scale": row.get("scale", row.get("ip_adapter_scale", "")),
            "rank": row.get("rank", ""), "step": row.get("step", row.get("training_steps", "")),
            "n_refs": row.get("n_refs", len(row.get("reference_images", [])) if row.get("reference_images") else ""),
            "image_path": str(row["image_file"].relative_to(root)),
            "dino_ref_mean": float(similarities.mean()), "dino_ref_max": float(similarities[0, nearest]),
            "nearest_train_file": reference_names[row["subject"]][nearest],
            "nearest_train_sim": float(similarities[0, nearest]),
            "clip_t_raw": clip_raw[index], "clip_t_neutral": clip_neutral[index],
        }
        if any(isinstance(value, float) and not math.isfinite(value) for value in record.values()):
            raise RuntimeError(f"Non-finite metric: {record}")
        output.append(record)
    destination = root / "results/metrics/all_runs_per_image.csv"
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output[0]))
        writer.writeheader(); writer.writerows(output)
    print(f"ALL_RUNS_MASTER_COMPLETE rows={len(output)} path={destination}")


if __name__ == "__main__":
    main()
