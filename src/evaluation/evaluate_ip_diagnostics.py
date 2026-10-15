#!/usr/bin/env python3
"""Task A IP-Adapter diagnostics using measured images and DINOv2 embeddings."""

from __future__ import annotations

import csv
import itertools
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image
from transformers import AutoImageProcessor, AutoModel


def normalize(value: torch.Tensor) -> torch.Tensor:
    return value / value.norm(dim=-1, keepdim=True)


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise RuntimeError(f"No rows for {path}")
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def image_array(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.float32)


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    task_root = root / "outputs/experiments/task_a"
    subjects = yaml.safe_load((root / "configs/main_subjects.yaml").read_text())["subjects"]
    metadata_paths = sorted(task_root.glob("**/P*/seed_*.json"))
    if len(metadata_paths) != 972:
        raise RuntimeError(f"Expected 972 Task A records, found {len(metadata_paths)}")

    rows = []
    for metadata_path in metadata_paths:
        row = json.loads(metadata_path.read_text())
        image_path = metadata_path.with_suffix(".png")
        if not image_path.is_file():
            raise FileNotFoundError(image_path)
        rows.append({**row, "image_path": image_path, "metadata_path": metadata_path})

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
    model = AutoModel.from_pretrained("facebook/dinov2-base").eval().to(device)

    reference_embeddings: dict[str, torch.Tensor] = {}
    reference_names: dict[str, list[str]] = {}
    with torch.inference_mode():
        for subject, cfg in subjects.items():
            ref_paths = sorted(path for path in (root / cfg["image_dir"]).iterdir() if path.is_file())
            images = [Image.open(path).convert("RGB") for path in ref_paths]
            values = model(**processor(images=images, return_tensors="pt").to(device)).last_hidden_state[:, 0]
            reference_embeddings[subject] = normalize(values).cpu()
            reference_names[subject] = [path.name for path in ref_paths]

    generated_embeddings = []
    for start in range(0, len(rows), 16):
        batch = rows[start:start + 16]
        images = [Image.open(row["image_path"]).convert("RGB") for row in batch]
        with torch.inference_mode():
            values = model(**processor(images=images, return_tensors="pt").to(device)).last_hidden_state[:, 0]
        generated_embeddings.extend(normalize(values).cpu())

    grouped: dict[tuple[str, str, int], list[int]] = defaultdict(list)
    nearest = {}
    for index, (row, embedding) in enumerate(zip(rows, generated_embeddings)):
        similarities = embedding[None] @ reference_embeddings[row["subject"]].T
        best_index = int(similarities.argmax())
        nearest[index] = {
            "dino_ref_mean": float(similarities.mean()),
            "dino_ref_max": float(similarities[0, best_index]),
            "nearest_train_file": reference_names[row["subject"]][best_index],
            "nearest_train_index": best_index,
        }
        grouped[(row["subject"], row["variant"], int(row["seed"]))].append(index)

    sensitivity = {}
    for key, indices in grouped.items():
        if len(indices) != 6:
            raise RuntimeError(f"Expected six prompts for {key}, found {len(indices)}")
        indices = sorted(indices, key=lambda i: rows[i]["prompt_id"])
        dino_pairs, pixel_pairs = [], []
        for left, right in itertools.combinations(indices, 2):
            dino_pairs.append(float(generated_embeddings[left] @ generated_embeddings[right]))
            a, b = image_array(rows[left]["image_path"]), image_array(rows[right]["image_path"])
            if a.shape != b.shape:
                raise RuntimeError(f"Shape mismatch: {rows[left]['image_path']} / {rows[right]['image_path']}")
            pixel_pairs.append(float(np.abs(a - b).mean()))
        sensitivity[key] = {
            "prompt_pairwise_dino_mean": float(np.mean(dino_pairs)),
            "prompt_pairwise_pixel_mad_0_255": float(np.mean(pixel_pairs)),
        }

    output = []
    for index, row in enumerate(rows):
        key = (row["subject"], row["variant"], int(row["seed"]))
        record = {
            "subject": row["subject"], "dataset": row["dataset"],
            "variant": row["variant"], "method": row["method"],
            "adapter_type": row.get("adapter_type", "none"),
            "scale": row.get("scale", ""), "reference_mode": row.get("reference_mode", "none"),
            "n_refs": row.get("n_refs", 0), "seed": row["seed"],
            "prompt_id": row["prompt_id"], "prompt_category": row["prompt_category"],
            "image_path": str(row["image_path"].relative_to(root)),
            **sensitivity[key], **nearest[index],
        }
        if any(isinstance(v, float) and not math.isfinite(v) for v in record.values()):
            raise RuntimeError(f"Non-finite metric: {record}")
        output.append(record)
    write_csv(root / "results/metrics/ip_diagnostics.csv", output)

    summary_rows = []
    by_variant: dict[str, list[dict]] = defaultdict(list)
    for row in output:
        by_variant[row["variant"]].append(row)
    for variant, selected in sorted(by_variant.items()):
        summary_rows.append({
            "variant": variant, "images": len(selected),
            "subjects": len({row["subject"] for row in selected}),
            "dino_ref_mean": float(np.mean([row["dino_ref_mean"] for row in selected])),
            "dino_ref_max_mean": float(np.mean([row["dino_ref_max"] for row in selected])),
            "prompt_pairwise_dino_mean": float(np.mean([row["prompt_pairwise_dino_mean"] for row in selected])),
            "prompt_pairwise_pixel_mad_0_255": float(np.mean([row["prompt_pairwise_pixel_mad_0_255"] for row in selected])),
        })
    write_csv(root / "results/metrics/ip_diagnostics_summary.csv", summary_rows)

    sanity_rows = []
    baseline_root = task_root / "base_no_personalization"
    for variant_root in [task_root / "scale_sweep/plain_s0p0", task_root / "scale_sweep/plus_s0p0"]:
        for generated in sorted(variant_root.glob("**/P*/seed_*.png")):
            relative = generated.relative_to(variant_root)
            baseline = baseline_root / relative
            a = np.asarray(Image.open(generated).convert("RGB"), dtype=np.int16)
            b = np.asarray(Image.open(baseline).convert("RGB"), dtype=np.int16)
            delta = np.abs(a - b)
            sanity_rows.append({
                "variant": variant_root.name, "subject": relative.parts[1],
                "prompt_id": relative.parts[-2], "seed": int(generated.stem.split("_")[-1]),
                "max_pixel_difference": int(delta.max()),
                "mean_absolute_pixel_difference": float(delta.mean()),
                "matches_baseline_exactly": bool(np.array_equal(a, b)),
            })
    if len(sanity_rows) != 72:
        raise RuntimeError(f"Expected 72 scale-zero comparisons, found {len(sanity_rows)}")
    write_csv(root / "results/metrics/ip_scale0_sanity.csv", sanity_rows)
    print(
        f"IP_DIAGNOSTICS_COMPLETE rows={len(output)} variants={len(summary_rows)} "
        f"scale0_max_pixel_difference={max(row['max_pixel_difference'] for row in sanity_rows)}"
    )


if __name__ == "__main__":
    main()
