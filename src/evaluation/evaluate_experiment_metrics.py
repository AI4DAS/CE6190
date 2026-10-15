#!/usr/bin/env python3
"""Evaluate a metadata-defined experiment with DINOv2 and raw/neutral CLIP-T."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from PIL import Image
from transformers import AutoImageProcessor, AutoModel, AutoProcessor, CLIPModel


def normalize(value: torch.Tensor) -> torch.Tensor:
    return value / value.norm(dim=-1, keepdim=True)


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise RuntimeError(f"No rows for {path}")
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def load_records(root: Path, experiment: str) -> list[dict]:
    rows = []
    for path in sorted((root / "outputs/experiments").glob("**/P*/seed_*.json")):
        row = json.loads(path.read_text())
        if row.get("experiment") != experiment:
            continue
        image = path.with_suffix(".png")
        if not image.is_file():
            raise FileNotFoundError(image)
        rows.append({**row, "image_file": image})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--expected", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.project_root.resolve()
    if torch.cuda.device_count() != 1:
        raise RuntimeError(f"Expected one visible MIG device, got {torch.cuda.device_count()}")
    records = load_records(root, args.experiment)
    if len(records) != args.expected:
        raise RuntimeError(f"Expected {args.expected} records for {args.experiment}, found {len(records)}")
    subjects = yaml.safe_load((root / "configs/main_subjects.yaml").read_text())["subjects"]
    device = "cuda:0"

    dino_processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
    dino_model = AutoModel.from_pretrained("facebook/dinov2-base").eval().to(device)
    reference_features, reference_names = {}, {}
    with torch.inference_mode():
        for subject in sorted({row["subject"] for row in records}):
            paths = sorted(path for path in (root / subjects[subject]["image_dir"]).iterdir() if path.is_file())
            images = [Image.open(path).convert("RGB") for path in paths]
            output = dino_model(**dino_processor(images=images, return_tensors="pt").to(device)).last_hidden_state[:, 0]
            reference_features[subject] = normalize(output).cpu()
            reference_names[subject] = [path.name for path in paths]
    image_features = []
    for start in range(0, len(records), 16):
        batch = records[start:start + 16]
        images = [Image.open(row["image_file"]).convert("RGB") for row in batch]
        with torch.inference_mode():
            output = dino_model(**dino_processor(images=images, return_tensors="pt").to(device)).last_hidden_state[:, 0]
        image_features.extend(normalize(output).cpu())
    del dino_model
    torch.cuda.empty_cache()

    clip_processor = AutoProcessor.from_pretrained("openai/clip-vit-large-patch14")
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").eval().to(device)
    raw_scores, neutral_scores = [], []
    for start in range(0, len(records), 16):
        batch = records[start:start + 16]
        images = [Image.open(row["image_file"]).convert("RGB") for row in batch]
        image_inputs = clip_processor(images=images, return_tensors="pt").to(device)
        raw_inputs = clip_processor(
            text=[row["generation_prompt"] for row in batch], return_tensors="pt", padding=True
        ).to(device)
        neutral_inputs = clip_processor(
            text=[row["evaluation_prompt"] for row in batch], return_tensors="pt", padding=True
        ).to(device)
        with torch.inference_mode():
            features = normalize(clip_model.get_image_features(pixel_values=image_inputs["pixel_values"]))
            raw = normalize(clip_model.get_text_features(**raw_inputs))
            neutral = normalize(clip_model.get_text_features(**neutral_inputs))
        raw_scores.extend((features * raw).sum(1).float().cpu().tolist())
        neutral_scores.extend((features * neutral).sum(1).float().cpu().tolist())

    output_rows = []
    for index, (row, feature) in enumerate(zip(records, image_features)):
        similarities = feature[None] @ reference_features[row["subject"]].T
        nearest_index = int(similarities.argmax())
        output = {
            "experiment": row.get("experiment", args.experiment),
            "variant": row.get("variant", row.get("method", "")),
            "object": row["subject"], "method": row["method"], "seed": int(row["seed"]),
            "prompt_id": row["prompt_id"], "prompt_category": row["prompt_category"],
            "scale": row.get("scale", ""), "rank": row.get("rank", ""),
            "step": row.get("step", row.get("training_steps", "")),
            "n_refs": row.get("n_refs", ""),
            "image_path": str(row["image_file"].relative_to(root)),
            "dino_ref_mean": float(similarities.mean()),
            "dino_ref_max": float(similarities[0, nearest_index]),
            "nearest_train_file": reference_names[row["subject"]][nearest_index],
            "nearest_train_sim": float(similarities[0, nearest_index]),
            "clip_t_raw": raw_scores[index], "clip_t_neutral": neutral_scores[index],
            "prior_preservation": row.get("prior_preservation", ""),
            "lora_inference_weight": row.get("lora_inference_weight", ""),
            "guidance_scale": row.get("guidance_scale", ""), "dtype": row.get("dtype", row.get("precision", "")),
        }
        if any(isinstance(value, float) and not math.isfinite(value) for value in output.values()):
            raise RuntimeError(f"Non-finite metric: {output}")
        output_rows.append(output)
    destination = args.output if args.output.is_absolute() else root / args.output
    write_csv(destination, output_rows)

    frame = pd.DataFrame(output_rows)
    group_columns = {
        "task_b_rank_ablation": ["variant", "rank"],
        "task_e_refcount": ["method", "variant", "n_refs"],
        "task_f_prior": ["variant", "prior_preservation"],
        "task_d_face_corruption": ["method", "variant", "guidance_scale", "lora_inference_weight", "dtype"],
    }.get(args.experiment, ["method", "variant"])
    summary = frame.groupby(group_columns, dropna=False, as_index=False).agg(
        images=("image_path", "count"), subjects=("object", "nunique"),
        dino_mean=("dino_ref_mean", "mean"), dino_std=("dino_ref_mean", "std"),
        clip_neutral_mean=("clip_t_neutral", "mean"), clip_neutral_std=("clip_t_neutral", "std"),
        clip_raw_mean=("clip_t_raw", "mean"), clip_raw_std=("clip_t_raw", "std"),
        copy_proxy_mean=("nearest_train_sim", "mean"), copy_proxy_std=("nearest_train_sim", "std"),
    )
    summary.to_csv(destination.with_name(destination.stem + "_summary.csv"), index=False)
    print(f"EXPERIMENT_METRICS_COMPLETE experiment={args.experiment} rows={len(frame)} groups={len(summary)}")


if __name__ == "__main__":
    main()
