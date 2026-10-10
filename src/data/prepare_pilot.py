#!/usr/bin/env python3
import argparse
import csv
import json
import shutil
from pathlib import Path

from PIL import Image


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.project_root.resolve()
    source = root / "data/dreambooth/upstream/dataset/dog2"
    destination = root / "data/processed/dreambooth/dog2/images"
    destination.mkdir(parents=True, exist_ok=True)
    source_images = sorted(p for p in source.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if not source_images:
        raise RuntimeError(f"No pilot images found in {source}")
    for image in source_images:
        target = destination / image.name
        if not target.exists():
            shutil.copy2(image, target)

    metadata = {
        "dataset": "dreambooth",
        "subject_id": "dog2",
        "class_name": "dog",
        "num_images": len(source_images),
        "instance_token": "sks",
        "instance_prompt": "a photo of sks dog",
        "source_repository": "https://github.com/google/dreambooth",
        "source_paths": [str(p.relative_to(root)) for p in source_images],
        "identifiable_humans_excluded": True,
    }
    subject_root = destination.parent
    (subject_root / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")

    rows = []
    for image_path in sorted(destination.iterdir()):
        if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        with Image.open(image_path) as image:
            width, height = image.size
        rows.append({
            "dataset": "dreambooth",
            "subject_id": "dog2",
            "class_name": "dog",
            "num_images": len(source_images),
            "width": width,
            "height": height,
            "license_note": "See upstream dataset/references_and_licenses.txt; dog2 images are attributed there to Unsplash photographer Lukasz Rawa.",
            "source_note": str(image_path.relative_to(root)),
        })
    summary = root / "results/metrics/dataset_summary.csv"
    summary.parent.mkdir(parents=True, exist_ok=True)
    with summary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()

