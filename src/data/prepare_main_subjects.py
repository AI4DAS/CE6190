#!/usr/bin/env python3
"""Deterministically prepare all configured benchmark subjects.

The script copies only supported images from the downloaded upstream datasets,
renames them by sorted source order, validates readability/counts, and writes
subject metadata plus a local dataset summary.  It never downloads data itself.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

import yaml
from PIL import Image


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/main_subjects.yaml")
    )
    args = parser.parse_args()
    root = args.project_root.resolve()
    config_path = args.config if args.config.is_absolute() else root / args.config
    subjects = yaml.safe_load(config_path.read_text())["subjects"]
    summary_rows: list[dict[str, object]] = []

    for subject_id, spec in subjects.items():
        source_dir = root / spec["source_dir"]
        image_dir = root / spec["image_dir"]
        if not source_dir.is_dir():
            raise FileNotFoundError(f"Missing source directory: {source_dir}")
        sources = sorted(
            p for p in source_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
        )
        expected = int(spec["expected_images"])
        if len(sources) != expected:
            raise RuntimeError(
                f"{subject_id}: expected {expected} source images, found {len(sources)}"
            )

        image_dir.mkdir(parents=True, exist_ok=True)
        expected_targets: list[Path] = []
        source_records: list[dict[str, str]] = []
        for index, source in enumerate(sources):
            target = image_dir / f"{index:02d}{source.suffix.lower()}"
            expected_targets.append(target)
            if not target.exists():
                shutil.copy2(source, target)
            with Image.open(target) as image:
                image.verify()
            with Image.open(target) as image:
                width, height = image.size
            source_records.append(
                {
                    "prepared_path": str(target.relative_to(root)),
                    "source_path": str(source.relative_to(root)),
                }
            )
            summary_rows.append(
                {
                    "dataset": spec["dataset"],
                    "subject_id": subject_id,
                    "class_name": spec["class_name"],
                    "image_path": str(target.relative_to(root)),
                    "width": width,
                    "height": height,
                    "source_path": str(source.relative_to(root)),
                }
            )

        extras = sorted(
            p for p in image_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
            and p not in expected_targets
        )
        if extras:
            raise RuntimeError(
                f"{subject_id}: unexpected prepared images present: "
                + ", ".join(str(p) for p in extras)
            )

        metadata = {
            "dataset": spec["dataset"],
            "subject_id": subject_id,
            "category": spec["category"],
            "class_name": spec["class_name"],
            "num_images": expected,
            "instance_token": spec["instance_token"],
            "instance_prompt": spec["instance_prompt"],
            "class_prompt": spec["class_prompt"],
            "placeholder_token": spec["placeholder_token"],
            "initializer_token": spec["initializer_token"],
            "identifiable_humans_excluded": True,
            "files": source_records,
        }
        (image_dir.parent / "metadata.json").write_text(
            json.dumps(metadata, indent=2) + "\n"
        )
        print(f"prepared {subject_id}: {expected} images -> {image_dir}")

    summary_path = root / "results/metrics/dataset_summary.csv"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"wrote {summary_path}")


if __name__ == "__main__":
    main()
