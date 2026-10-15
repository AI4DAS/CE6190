#!/usr/bin/env python3
"""Compute unchanged DINOv2 and CLIP metrics for the main benchmark."""

from __future__ import annotations

import argparse
import csv
import gc
import json
from pathlib import Path

import torch
import yaml
from PIL import Image
from transformers import AutoImageProcessor, AutoModel, AutoProcessor, CLIPModel


METHODS = ("base_model", "dreambooth", "lora", "textual_inversion", "ip_adapter_plus")


def normalize(tensor):
    return tensor / tensor.norm(dim=-1, keepdim=True)


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_records(root: Path, subjects: dict) -> list[dict]:
    rows = []
    for subject, record in subjects.items():
        for method in METHODS:
            base = root / "outputs/main_evaluation" / method / record["dataset"] / subject
            paths = sorted(base.glob("P*/seed_*.json"))
            if len(paths) != 24:
                raise RuntimeError(f"Expected 24 records for {subject}/{method}, found {len(paths)}")
            for path in paths:
                row = json.loads(path.read_text())
                image = path.with_suffix(".png")
                if not image.is_file():
                    raise FileNotFoundError(image)
                row["image_path"] = str(image.relative_to(root))
                rows.append(row)
    if len(rows) != 960:
        raise RuntimeError(f"Expected 960 total records, found {len(rows)}")
    return rows


def reference_paths(root: Path, record: dict) -> list[Path]:
    image_dir = root / record["image_dir"]
    paths = sorted(path for path in image_dir.iterdir() if path.is_file())
    if len(paths) != record["expected_images"]:
        raise RuntimeError(f"Reference count mismatch for {image_dir}: {len(paths)}")
    return paths


def evaluate_dino(root: Path, subjects: dict, rows: list[dict], device: str) -> list[dict]:
    model_id = "facebook/dinov2-base"
    processor = AutoImageProcessor.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id).eval().to(device)
    references = {}
    with torch.inference_mode():
        for subject, record in subjects.items():
            images = [Image.open(path).convert("RGB") for path in reference_paths(root, record)]
            references[subject] = normalize(
                model(**processor(images=images, return_tensors="pt").to(device)).last_hidden_state[:, 0]
            )
    output = []
    for start in range(0, len(rows), 16):
        batch = rows[start:start + 16]
        images = [Image.open(root / row["image_path"]).convert("RGB") for row in batch]
        with torch.inference_mode():
            features = normalize(
                model(**processor(images=images, return_tensors="pt").to(device)).last_hidden_state[:, 0]
            )
        for row, feature in zip(batch, features):
            similarity = (feature[None] @ references[row["subject"]].T).float().cpu().numpy()[0]
            output.append({
                "method": row["method"], "dataset": row["dataset"], "subject": row["subject"],
                "prompt_id": row["prompt_id"], "prompt_category": row["prompt_category"],
                "seed": row["seed"], "image_path": row["image_path"],
                "mean_reference_similarity": float(similarity.mean()),
                "max_reference_similarity": float(similarity.max()),
                "num_references": len(references[row["subject"]]), "model": model_id,
            })
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return output


def evaluate_clip(root: Path, rows: list[dict], device: str) -> list[dict]:
    model_id = "openai/clip-vit-large-patch14"
    processor = AutoProcessor.from_pretrained(model_id)
    model = CLIPModel.from_pretrained(model_id).eval().to(device)
    output = []
    for start in range(0, len(rows), 16):
        batch = rows[start:start + 16]
        images = [Image.open(root / row["image_path"]).convert("RGB") for row in batch]
        texts = [row["evaluation_prompt"] for row in batch]
        inputs = processor(text=texts, images=images, return_tensors="pt", padding=True).to(device)
        with torch.inference_mode():
            image_features = normalize(model.get_image_features(pixel_values=inputs["pixel_values"]))
            text_features = normalize(model.get_text_features(
                input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"]
            ))
            scores = (image_features * text_features).sum(dim=1).float().cpu().numpy()
        for row, score in zip(batch, scores):
            output.append({
                "method": row["method"], "dataset": row["dataset"], "subject": row["subject"],
                "prompt_id": row["prompt_id"], "prompt_category": row["prompt_category"],
                "seed": row["seed"], "image_path": row["image_path"],
                "evaluation_prompt": row["evaluation_prompt"], "clip_similarity": float(score),
                "model": model_id,
            })
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.project_root.resolve()
    if torch.cuda.device_count() != 1:
        raise RuntimeError(f"Expected one visible MIG device, got {torch.cuda.device_count()}")
    subjects = yaml.safe_load((root / "configs/main_subjects.yaml").read_text())["subjects"]
    rows = load_records(root, subjects)
    dino = evaluate_dino(root, subjects, rows, "cuda:0")
    clip = evaluate_clip(root, rows, "cuda:0")
    write_csv(root / "results/metrics/main_subject_fidelity.csv", dino)
    write_csv(root / "results/metrics/main_prompt_alignment.csv", clip)
    print(f"METRICS_COMPLETE dino={len(dino)} clip={len(clip)}")


if __name__ == "__main__":
    main()
