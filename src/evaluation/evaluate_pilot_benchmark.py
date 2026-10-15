#!/usr/bin/env python3
import argparse, csv, gc, json
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel, AutoProcessor, CLIPModel

METHODS = ("dreambooth", "lora", "textual_inversion", "ip_adapter_plus")

def normalize(x): return x / x.norm(dim=-1, keepdim=True)

def load_records(root):
    rows = []
    for method in METHODS:
        paths = sorted((root / f"outputs/evaluation/{method}/dreambooth/dog2").glob("P*/seed_*.json"))
        if len(paths) != 24: raise RuntimeError(f"Expected 24 {method} records, found {len(paths)}")
        for path in paths:
            row = json.loads(path.read_text()); row["image_path"] = str(path.with_suffix(".png").relative_to(root)); rows.append(row)
    return rows

def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

def dino(root, rows, device):
    model_id = "facebook/dinov2-base"; processor = AutoImageProcessor.from_pretrained(model_id); model = AutoModel.from_pretrained(model_id).eval().to(device)
    refs = [Image.open(p).convert("RGB") for p in sorted((root / "data/processed/dreambooth/dog2/images").glob("*.jpg"))]
    with torch.inference_mode(): ref = normalize(model(**processor(images=refs, return_tensors="pt").to(device)).last_hidden_state[:, 0])
    out = []
    for start in range(0, len(rows), 16):
        batch = rows[start:start + 16]; images = [Image.open(root / r["image_path"]).convert("RGB") for r in batch]
        with torch.inference_mode(): feat = normalize(model(**processor(images=images, return_tensors="pt").to(device)).last_hidden_state[:, 0])
        for r, sim in zip(batch, (feat @ ref.T).float().cpu().numpy()):
            out.append({"method": r["method"], "dataset": r["dataset"], "subject": r["subject"], "prompt_id": r["prompt_id"], "prompt_category": r["prompt_category"], "seed": r["seed"], "image_path": r["image_path"], "mean_reference_similarity": float(sim.mean()), "max_reference_similarity": float(sim.max()), "num_references": 6, "model": model_id})
    del model; gc.collect(); torch.cuda.empty_cache(); return out

def clip(root, rows, device):
    model_id = "openai/clip-vit-large-patch14"; processor = AutoProcessor.from_pretrained(model_id); model = CLIPModel.from_pretrained(model_id).eval().to(device); out = []
    for start in range(0, len(rows), 16):
        batch = rows[start:start + 16]; images = [Image.open(root / r["image_path"]).convert("RGB") for r in batch]; texts = [r["evaluation_prompt"] for r in batch]
        inputs = processor(text=texts, images=images, return_tensors="pt", padding=True).to(device)
        with torch.inference_mode(): scores = (normalize(model.get_image_features(pixel_values=inputs["pixel_values"])) * normalize(model.get_text_features(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"]))).sum(dim=1).float().cpu().numpy()
        for r, score in zip(batch, scores):
            out.append({"method": r["method"], "dataset": r["dataset"], "subject": r["subject"], "prompt_id": r["prompt_id"], "prompt_category": r["prompt_category"], "seed": r["seed"], "image_path": r["image_path"], "evaluation_prompt": r["evaluation_prompt"], "clip_similarity": float(score), "model": model_id})
    return out

def main():
    p = argparse.ArgumentParser(); p.add_argument("--project-root", type=Path, required=True); args = p.parse_args(); root = args.project_root.resolve()
    if torch.cuda.device_count() != 1: raise RuntimeError("Select exactly one MIG GPU")
    rows = load_records(root); d = dino(root, rows, "cuda:0"); c = clip(root, rows, "cuda:0")
    write_csv(root / "results/metrics/pilot_subject_fidelity.csv", d); write_csv(root / "results/metrics/pilot_prompt_alignment.csv", c)
    for method in METHODS:
        dv = np.array([x["mean_reference_similarity"] for x in d if x["method"] == method]); cv = np.array([x["clip_similarity"] for x in c if x["method"] == method])
        print(f"{method}: DINO {dv.mean():.6f} +/- {dv.std(ddof=1):.6f}; CLIP {cv.mean():.6f} +/- {cv.std(ddof=1):.6f}")

if __name__ == "__main__": main()
