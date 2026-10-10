#!/usr/bin/env python3
import argparse
import csv
import gc
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel, AutoProcessor, CLIPModel


def normalize(tensor: torch.Tensor) -> torch.Tensor:
    return tensor / tensor.norm(dim=-1, keepdim=True)


def load_records(root: Path) -> list[dict]:
    records = []
    for metadata_path in sorted((root / "outputs/ip_adapter/dreambooth/dog2").glob("P*/seed_*.json")):
        record = json.loads(metadata_path.read_text())
        record["image_path"] = str(metadata_path.with_suffix(".png").relative_to(root))
        records.append(record)
    if len(records) != 24:
        raise RuntimeError(f"Expected 24 pilot records, found {len(records)}")
    return records


def dino_scores(root: Path, records: list[dict], device: str) -> list[dict]:
    model_id = "facebook/dinov2-base"
    processor = AutoImageProcessor.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id).eval().to(device)
    references = [Image.open(p).convert("RGB") for p in sorted((root / "data/processed/dreambooth/dog2/images").glob("*.jpg"))]
    with torch.inference_mode():
        inputs = processor(images=references, return_tensors="pt").to(device)
        reference_features = normalize(model(**inputs).last_hidden_state[:, 0])
    output = []
    for record in records:
        with Image.open(root / record["image_path"]) as image:
            inputs = processor(images=image.convert("RGB"), return_tensors="pt").to(device)
        with torch.inference_mode():
            feature = normalize(model(**inputs).last_hidden_state[:, 0])
        similarities = (feature @ reference_features.T).squeeze(0).float().cpu().numpy()
        output.append({
            "method": record["method"], "dataset": record["dataset"], "subject": record["subject"],
            "prompt_id": record["prompt_id"], "prompt_category": record["prompt_category"],
            "prompt": record["prompt"], "seed": record["seed"], "image_path": record["image_path"],
            "mean_reference_similarity": float(similarities.mean()),
            "max_reference_similarity": float(similarities.max()),
            "model": model_id,
        })
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return output


def clip_scores(root: Path, records: list[dict], device: str) -> list[dict]:
    model_id = "openai/clip-vit-large-patch14"
    processor = AutoProcessor.from_pretrained(model_id)
    model = CLIPModel.from_pretrained(model_id).eval().to(device)
    output = []
    for record in records:
        with Image.open(root / record["image_path"]) as image:
            inputs = processor(text=[record["prompt"]], images=[image.convert("RGB")], return_tensors="pt", padding=True).to(device)
        with torch.inference_mode():
            image_features = normalize(model.get_image_features(pixel_values=inputs["pixel_values"]))
            text_features = normalize(model.get_text_features(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"]))
        score = float((image_features @ text_features.T).item())
        output.append({
            "method": record["method"], "dataset": record["dataset"], "subject": record["subject"],
            "prompt_id": record["prompt_id"], "prompt_category": record["prompt_category"],
            "prompt": record["prompt"], "seed": record["seed"], "image_path": record["image_path"],
            "clip_similarity": score, "model": model_id,
        })
    return output


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.project_root.resolve()
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Select exactly one MIG UUID with CUDA_VISIBLE_DEVICES.")
    records = load_records(root)
    dino = dino_scores(root, records, "cuda:0")
    clip = clip_scores(root, records, "cuda:0")
    write_csv(root / "results/metrics/subject_fidelity.csv", dino)
    write_csv(root / "results/metrics/prompt_alignment.csv", clip)

    print("prompt_category,dino_mean,dino_std,clip_mean,clip_std")
    categories = sorted({r["prompt_category"] for r in records})
    for category in categories:
        d = np.array([r["mean_reference_similarity"] for r in dino if r["prompt_category"] == category])
        c = np.array([r["clip_similarity"] for r in clip if r["prompt_category"] == category])
        print(f"{category},{d.mean():.6f},{d.std(ddof=1):.6f},{c.mean():.6f},{c.std(ddof=1):.6f}")
    d = np.array([r["mean_reference_similarity"] for r in dino])
    c = np.array([r["clip_similarity"] for r in clip])
    print(f"ALL,{d.mean():.6f},{d.std(ddof=1):.6f},{c.mean():.6f},{c.std(ddof=1):.6f}")


if __name__ == "__main__":
    main()

