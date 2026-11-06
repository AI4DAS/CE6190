#!/usr/bin/env python3
"""Evaluate completed DreamO/UNO outputs with the main benchmark metrics."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
from pathlib import Path

import pandas as pd
import torch
import yaml
from PIL import Image
from transformers import AutoImageProcessor, AutoModel, AutoProcessor, CLIPModel


METHODS = ("dreamo", "uno")


def normalize(tensor: torch.Tensor) -> torch.Tensor:
    return tensor / tensor.norm(dim=-1, keepdim=True)


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise RuntimeError(f"No rows for {path}")
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_records(root: Path, subjects: dict) -> list[dict]:
    rows = []
    for subject, subject_cfg in subjects.items():
        for method in METHODS:
            directory = root / "outputs/modern_extension" / method / subject_cfg["dataset"] / subject
            metadata_paths = sorted(directory.glob("P*/seed_*.json"))
            if len(metadata_paths) != 24:
                raise RuntimeError(f"Expected 24 records for {method}/{subject}, found {len(metadata_paths)}")
            for metadata_path in metadata_paths:
                row = json.loads(metadata_path.read_text())
                image_path = metadata_path.with_suffix(".png")
                if not image_path.is_file():
                    raise FileNotFoundError(image_path)
                if row.get("subject") != subject or row.get("method") != method:
                    raise RuntimeError(f"Metadata mismatch: {metadata_path}")
                if row.get("evaluation_prompt") != row.get("generation_prompt"):
                    raise RuntimeError(f"Modern prompt is not already neutral: {metadata_path}")
                rows.append({**row, "image_file": image_path})
    expected = len(subjects) * len(METHODS) * 24
    if len(rows) != expected:
        raise RuntimeError(f"Expected {expected} total records, found {len(rows)}")
    return rows


def reference_paths(root: Path, subject_cfg: dict) -> list[Path]:
    directory = root / subject_cfg["image_dir"]
    paths = sorted(path for path in directory.iterdir() if path.is_file())
    if len(paths) != subject_cfg["expected_images"]:
        raise RuntimeError(f"Reference count mismatch for {directory}: {len(paths)}")
    return paths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.project_root.resolve()
    if torch.cuda.device_count() != 1:
        raise RuntimeError(f"Expected exactly one visible MIG, got {torch.cuda.device_count()}")

    subjects = yaml.safe_load((root / "configs/main_subjects.yaml").read_text())["subjects"]
    records = load_records(root, subjects)
    device = "cuda:0"

    dino_id = "facebook/dinov2-base"
    dino_processor = AutoImageProcessor.from_pretrained(dino_id)
    dino_model = AutoModel.from_pretrained(dino_id).eval().to(device)
    reference_features, reference_names = {}, {}
    with torch.inference_mode():
        for subject, subject_cfg in subjects.items():
            paths = reference_paths(root, subject_cfg)
            images = [Image.open(path).convert("RGB") for path in paths]
            output = dino_model(
                **dino_processor(images=images, return_tensors="pt").to(device)
            ).last_hidden_state[:, 0]
            reference_features[subject] = normalize(output).cpu()
            reference_names[subject] = [path.name for path in paths]

    generated_features = []
    for start in range(0, len(records), 16):
        batch = records[start:start + 16]
        images = [Image.open(row["image_file"]).convert("RGB") for row in batch]
        with torch.inference_mode():
            output = dino_model(
                **dino_processor(images=images, return_tensors="pt").to(device)
            ).last_hidden_state[:, 0]
        generated_features.extend(normalize(output).cpu())
    del dino_model
    gc.collect()
    torch.cuda.empty_cache()

    clip_id = "openai/clip-vit-large-patch14"
    clip_processor = AutoProcessor.from_pretrained(clip_id)
    clip_model = CLIPModel.from_pretrained(clip_id).eval().to(device)
    clip_scores = []
    for start in range(0, len(records), 16):
        batch = records[start:start + 16]
        images = [Image.open(row["image_file"]).convert("RGB") for row in batch]
        inputs = clip_processor(
            text=[row["evaluation_prompt"] for row in batch],
            images=images,
            return_tensors="pt",
            padding=True,
        ).to(device)
        with torch.inference_mode():
            image_features = normalize(clip_model.get_image_features(pixel_values=inputs["pixel_values"]))
            text_features = normalize(clip_model.get_text_features(
                input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"]
            ))
        clip_scores.extend((image_features * text_features).sum(1).float().cpu().tolist())

    dino_rows, clip_rows, combined_rows = [], [], []
    for row, feature, clip_score in zip(records, generated_features, clip_scores):
        similarities = feature[None] @ reference_features[row["subject"]].T
        nearest_index = int(similarities.argmax())
        mean_similarity = float(similarities.mean())
        max_similarity = float(similarities[0, nearest_index])
        common = {
            "method": row["method"], "dataset": row["dataset"], "subject": row["subject"],
            "prompt_id": row["prompt_id"], "prompt_category": row["prompt_category"],
            "seed": int(row["seed"]), "image_path": str(row["image_file"].relative_to(root)),
        }
        dino_rows.append({
            **common, "mean_reference_similarity": mean_similarity,
            "max_reference_similarity": max_similarity,
            "nearest_reference_file": reference_names[row["subject"]][nearest_index],
            "num_references": len(reference_names[row["subject"]]), "model": dino_id,
        })
        clip_rows.append({
            **common, "evaluation_prompt": row["evaluation_prompt"],
            "clip_similarity": float(clip_score), "model": clip_id,
        })
        combined_rows.append({
            **common, "evaluation_prompt": row["evaluation_prompt"],
            "dino_ref_mean": mean_similarity, "dino_ref_max": max_similarity,
            "nearest_reference_file": reference_names[row["subject"]][nearest_index],
            "clip_t_neutral": float(clip_score),
            "generation_latency_sec": float(row["generation_latency_sec"]),
            "peak_inference_vram_gib": float(row["peak_inference_vram_gib"]),
        })

    for output_row in combined_rows:
        numeric = ("dino_ref_mean", "dino_ref_max", "clip_t_neutral",
                   "generation_latency_sec", "peak_inference_vram_gib")
        if any(not math.isfinite(float(output_row[key])) for key in numeric):
            raise RuntimeError(f"Non-finite metric: {output_row}")

    metric_dir = root / "results/metrics"
    write_csv(metric_dir / "modern_subject_fidelity.csv", dino_rows)
    write_csv(metric_dir / "modern_prompt_alignment.csv", clip_rows)
    write_csv(metric_dir / "modern_metrics_per_image.csv", combined_rows)

    frame = pd.DataFrame(combined_rows)
    summary = frame.groupby("method", as_index=False).agg(
        images=("image_path", "count"), subjects=("subject", "nunique"),
        dino_mean=("dino_ref_mean", "mean"), dino_std=("dino_ref_mean", "std"),
        dino_max_mean=("dino_ref_max", "mean"),
        clip_mean=("clip_t_neutral", "mean"), clip_std=("clip_t_neutral", "std"),
        inference_latency_mean_sec=("generation_latency_sec", "mean"),
        inference_latency_std_sec=("generation_latency_sec", "std"),
        peak_inference_vram_gib=("peak_inference_vram_gib", "max"),
    )
    summary.to_csv(metric_dir / "modern_metrics_summary.csv", index=False)

    by_subject = frame.groupby(["method", "subject"], as_index=False).agg(
        images=("image_path", "count"), dino_mean=("dino_ref_mean", "mean"),
        dino_std=("dino_ref_mean", "std"), clip_mean=("clip_t_neutral", "mean"),
        clip_std=("clip_t_neutral", "std"),
    )
    by_subject.to_csv(metric_dir / "modern_metrics_by_subject.csv", index=False)
    by_prompt = frame.groupby(["method", "prompt_id", "prompt_category"], as_index=False).agg(
        images=("image_path", "count"), dino_mean=("dino_ref_mean", "mean"),
        dino_std=("dino_ref_mean", "std"), clip_mean=("clip_t_neutral", "mean"),
        clip_std=("clip_t_neutral", "std"),
    )
    by_prompt.to_csv(metric_dir / "modern_metrics_by_prompt.csv", index=False)
    print(summary.to_string(index=False))
    print(f"MODERN_METRICS_COMPLETE rows={len(frame)}")


if __name__ == "__main__":
    main()
