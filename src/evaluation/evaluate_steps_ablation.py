#!/usr/bin/env python3
"""Evaluate Task C step trajectories and create measured summaries/curves."""

from __future__ import annotations

import csv
import itertools
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from PIL import Image
from transformers import AutoImageProcessor, AutoModel, AutoProcessor, CLIPModel


EXPECTED = {"dreambooth": 8 * 3 * 12, "lora": 16 * 3 * 12, "textual_inversion": 12 * 3 * 12}
COLORS = {"dreambooth": "#0072B2", "lora": "#E69F00", "textual_inversion": "#009E73"}


def norm(x):
    return x / x.norm(dim=-1, keepdim=True)


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    source = root / "outputs/experiments/task_c_steps"
    subjects_cfg = yaml.safe_load((root / "configs/main_subjects.yaml").read_text())["subjects"]
    records = []
    for method, expected in EXPECTED.items():
        paths = sorted((source / method).glob("**/P*/seed_*.json"))
        if len(paths) != expected:
            raise RuntimeError(f"Expected {expected} {method} rows, found {len(paths)}")
        for path in paths:
            row = json.loads(path.read_text()); image = path.with_suffix(".png")
            if not image.is_file(): raise FileNotFoundError(image)
            records.append({**row, "image_file": image})

    device = "cuda:0"
    dino_processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
    dino_model = AutoModel.from_pretrained("facebook/dinov2-base").eval().to(device)
    refs, ref_names = {}, {}
    with torch.inference_mode():
        for subject in ("dog2", "teapot", "bear_plushie"):
            paths = sorted(path for path in (root / subjects_cfg[subject]["image_dir"]).iterdir() if path.is_file())
            images = [Image.open(path).convert("RGB") for path in paths]
            refs[subject] = norm(dino_model(**dino_processor(images=images, return_tensors="pt").to(device)).last_hidden_state[:, 0]).cpu()
            ref_names[subject] = [path.name for path in paths]
    embeddings = []
    for start in range(0, len(records), 16):
        batch = records[start:start + 16]
        images = [Image.open(row["image_file"]).convert("RGB") for row in batch]
        with torch.inference_mode():
            out = dino_model(**dino_processor(images=images, return_tensors="pt").to(device)).last_hidden_state[:, 0]
        embeddings.extend(norm(out).cpu())
    del dino_model; torch.cuda.empty_cache()

    clip_processor = AutoProcessor.from_pretrained("openai/clip-vit-large-patch14")
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").eval().to(device)
    clip_raw, clip_neutral = [], []
    for start in range(0, len(records), 16):
        batch = records[start:start + 16]
        images = [Image.open(row["image_file"]).convert("RGB") for row in batch]
        image_inputs = clip_processor(images=images, return_tensors="pt").to(device)
        raw_inputs = clip_processor(text=[row["generation_prompt"] for row in batch], return_tensors="pt", padding=True).to(device)
        neutral_inputs = clip_processor(text=[row["evaluation_prompt"] for row in batch], return_tensors="pt", padding=True).to(device)
        with torch.inference_mode():
            image_features = norm(clip_model.get_image_features(pixel_values=image_inputs["pixel_values"]))
            raw_features = norm(clip_model.get_text_features(**raw_inputs))
            neutral_features = norm(clip_model.get_text_features(**neutral_inputs))
        clip_raw.extend((image_features * raw_features).sum(1).float().cpu().tolist())
        clip_neutral.extend((image_features * neutral_features).sum(1).float().cpu().tolist())

    group_indices = defaultdict(list)
    nearest = []
    for index, (row, embedding) in enumerate(zip(records, embeddings)):
        similarity = embedding[None] @ refs[row["subject"]].T
        best = int(similarity.argmax())
        nearest.append((float(similarity.mean()), float(similarity[0, best]), ref_names[row["subject"]][best]))
        group_indices[(row["method"], row["subject"], int(row["step"]), int(row["seed"]))].append(index)
    sensitivity = {}
    for key, indices in group_indices.items():
        indices = sorted(indices, key=lambda i: records[i]["prompt_id"])
        if len(indices) != 6: raise RuntimeError(f"Expected six prompts: {key}")
        sensitivity[key] = float(np.mean([float(embeddings[a] @ embeddings[b]) for a, b in itertools.combinations(indices, 2)]))

    rows = []
    for index, row in enumerate(records):
        mean_sim, max_sim, filename = nearest[index]
        key = (row["method"], row["subject"], int(row["step"]), int(row["seed"]))
        result = {
            "experiment": row["experiment"], "variant": row["variant"],
            "method": row["method"], "subject": row["subject"], "step": row["step"],
            "seed": row["seed"], "prompt_id": row["prompt_id"],
            "prompt_category": row["prompt_category"],
            "image_path": str(row["image_file"].relative_to(root)),
            "dino_ref_mean": mean_sim, "dino_ref_max": max_sim,
            "nearest_train_file": filename, "nearest_train_sim": max_sim,
            "clip_t_raw": clip_raw[index], "clip_t_neutral": clip_neutral[index],
            "prompt_pairwise_dino_mean": sensitivity[key],
        }
        if any(isinstance(v, float) and not math.isfinite(v) for v in result.values()):
            raise RuntimeError(f"Non-finite result: {result}")
        rows.append(result)
    write_csv(root / "results/metrics/steps_ablation.csv", rows)

    frame = pd.DataFrame(rows)
    summary = frame.groupby(["method", "step"], as_index=False).agg(
        images=("image_path", "count"), subjects=("subject", "nunique"),
        dino_mean=("dino_ref_mean", "mean"), dino_std=("dino_ref_mean", "std"),
        clip_neutral_mean=("clip_t_neutral", "mean"), clip_neutral_std=("clip_t_neutral", "std"),
        copy_proxy_mean=("nearest_train_sim", "mean"), copy_proxy_std=("nearest_train_sim", "std"),
        prompt_sensitivity_mean=("prompt_pairwise_dino_mean", "mean"),
    )
    summary.to_csv(root / "results/metrics/steps_ablation_summary.csv", index=False)

    subject_means = frame.groupby(["method", "step", "subject"], as_index=False).agg(
        dino=("dino_ref_mean", "mean"), clip=("clip_t_neutral", "mean"), copy=("nearest_train_sim", "mean")
    )
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    for axis, metric, ylabel in zip(axes, ("dino", "clip", "copy"), ("DINOv2 similarity", "CLIP-T neutral", "Nearest-reference similarity")):
        for method in EXPECTED:
            selected = subject_means[subject_means.method == method]
            grouped = selected.groupby("step")[metric]
            x = np.array(sorted(grouped.groups))
            means = grouped.mean().reindex(x).to_numpy(); lows = grouped.min().reindex(x).to_numpy(); highs = grouped.max().reindex(x).to_numpy()
            axis.plot(x, means, marker="o", label=method, color=COLORS[method])
            axis.fill_between(x, lows, highs, color=COLORS[method], alpha=0.18)
        axis.set_xlabel("Training step"); axis.set_ylabel(ylabel); axis.grid(alpha=0.25)
    axes[0].legend(frameon=False)
    fig.tight_layout()
    out = root / "results/figures/ablation/task_c"
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "step_curves.png", dpi=200); fig.savefig(out / "step_curves.pdf")
    plt.close(fig)
    print(f"STEPS_EVALUATION_COMPLETE rows={len(rows)} summary={len(summary)}")


if __name__ == "__main__":
    main()
