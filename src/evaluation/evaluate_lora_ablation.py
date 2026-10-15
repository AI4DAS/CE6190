#!/usr/bin/env python3
"""Evaluate DINOv2 fidelity and neutral-prompt CLIP for LoRA ablations."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel, AutoProcessor, CLIPModel


def normalize(value):
    return value / value.norm(dim=-1, keepdim=True)


def write(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    if torch.cuda.device_count() != 1:
        raise RuntimeError(f"Expected one visible MIG device, got {torch.cuda.device_count()}")
    metadata_paths = sorted((root / "outputs/ablations/lora_generation").glob("*/P*/seed_*.json"))
    if len(metadata_paths) != 120:
        raise RuntimeError(f"Expected 120 metadata files, found {len(metadata_paths)}")
    rows = []
    for path in metadata_paths:
        row = json.loads(path.read_text())
        if "sks" in row["evaluation_prompt"] or "<" in row["evaluation_prompt"]:
            raise RuntimeError(f"Non-neutral evaluation prompt: {path}")
        row["image_path"] = str(path.with_suffix(".png").relative_to(root))
        rows.append(row)

    device = "cuda:0"
    references = [Image.open(path).convert("RGB") for path in sorted(
        (root / "data/processed/dreambooth/dog2/images").glob("*.*")
    )]
    dino_id = "facebook/dinov2-base"
    dino_processor = AutoImageProcessor.from_pretrained(dino_id)
    dino_model = AutoModel.from_pretrained(dino_id).eval().to(device)
    with torch.inference_mode():
        ref_features = normalize(
            dino_model(**dino_processor(images=references, return_tensors="pt").to(device)).last_hidden_state[:, 0]
        )
    dino_rows = []
    for start in range(0, len(rows), 16):
        batch = rows[start:start + 16]
        images = [Image.open(root / row["image_path"]).convert("RGB") for row in batch]
        with torch.inference_mode():
            features = normalize(
                dino_model(**dino_processor(images=images, return_tensors="pt").to(device)).last_hidden_state[:, 0]
            )
        for row, feature in zip(batch, features):
            similarities = (feature[None] @ ref_features.T).float().cpu().numpy()[0]
            dino_rows.append({
                "variant": row["variant"], "rank": row["rank"], "training_steps": row["training_steps"],
                "prompt_id": row["prompt_id"], "prompt_category": row["prompt_category"],
                "seed": row["seed"], "image_path": row["image_path"],
                "mean_reference_similarity": float(similarities.mean()),
                "max_reference_similarity": float(similarities.max()), "model": dino_id,
            })
    del dino_model
    torch.cuda.empty_cache()

    clip_id = "openai/clip-vit-large-patch14"
    clip_processor = AutoProcessor.from_pretrained(clip_id)
    clip_model = CLIPModel.from_pretrained(clip_id).eval().to(device)
    clip_rows = []
    for start in range(0, len(rows), 16):
        batch = rows[start:start + 16]
        images = [Image.open(root / row["image_path"]).convert("RGB") for row in batch]
        texts = [row["evaluation_prompt"] for row in batch]
        inputs = clip_processor(text=texts, images=images, return_tensors="pt", padding=True).to(device)
        with torch.inference_mode():
            image_features = normalize(clip_model.get_image_features(pixel_values=inputs["pixel_values"]))
            text_features = normalize(clip_model.get_text_features(
                input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"]
            ))
            scores = (image_features * text_features).sum(dim=1).float().cpu().numpy()
        for row, score in zip(batch, scores):
            clip_rows.append({
                "variant": row["variant"], "rank": row["rank"], "training_steps": row["training_steps"],
                "prompt_id": row["prompt_id"], "prompt_category": row["prompt_category"],
                "seed": row["seed"], "image_path": row["image_path"],
                "evaluation_prompt": row["evaluation_prompt"], "clip_similarity": float(score),
                "model": clip_id,
            })
    write(root / "results/metrics/lora_ablation_subject_fidelity.csv", dino_rows)
    write(root / "results/metrics/lora_ablation_prompt_alignment.csv", clip_rows)
    print(f"LORA_ABLATION_METRICS_COMPLETE dino={len(dino_rows)} clip={len(clip_rows)}")


if __name__ == "__main__":
    main()
