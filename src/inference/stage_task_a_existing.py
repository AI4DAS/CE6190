#!/usr/bin/env python3
"""Stage previously completed scale-0.3 and Base outputs without rewriting them."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import yaml


def stage(root: Path, source_method: str, destination: Path, experiment: str, variant: str) -> int:
    subjects = yaml.safe_load((root / "configs/main_subjects.yaml").read_text())["subjects"]
    count = 0
    for subject, cfg in subjects.items():
        source = root / "outputs/main_evaluation" / source_method / cfg["dataset"] / subject
        for metadata_path in sorted(source.glob("P*/seed_*.json")):
            source_image = metadata_path.with_suffix(".png")
            if not source_image.is_file():
                raise FileNotFoundError(source_image)
            relative = metadata_path.relative_to(source)
            target_metadata = destination / cfg["dataset"] / subject / relative
            target_image = target_metadata.with_suffix(".png")
            target_metadata.parent.mkdir(parents=True, exist_ok=True)
            row = json.loads(metadata_path.read_text())
            scale = row.get("ip_adapter_scale")
            expected_scale = 0.3 if source_method == "ip_adapter_plus" else None
            if scale != expected_scale:
                raise RuntimeError(f"Unexpected scale in {metadata_path}: {scale}")
            if target_image.exists() != target_metadata.exists():
                raise RuntimeError(f"Partial staged pair: {target_image} / {target_metadata}")
            enriched = {
                **row,
                "experiment": experiment, "variant": variant, "scale": scale,
                "rank": row.get("rank"), "step": row.get("training_steps", 0),
                "n_refs": len(row.get("reference_images", [])),
                "ref_files_used": [Path(p).name for p in row.get("reference_images", [])],
                "guidance_scale": row["guidance_scale"],
                "dtype": "float16" if row.get("precision") == "fp16" else row.get("precision"),
                "negative_prompt": None,
            }
            if target_image.exists():
                if json.loads(target_metadata.read_text()) != enriched:
                    raise RuntimeError(f"Staged metadata mismatch: {target_metadata}")
            else:
                try:
                    os.link(source_image, target_image)
                except OSError:
                    shutil.copy2(source_image, target_image)
                target_metadata.write_text(json.dumps(enriched, indent=2) + "\n")
            count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.project_root.resolve()
    base = root / "outputs/experiments/task_a"
    scale03 = stage(
        root, "ip_adapter_plus", base / "full_plus_s0.3",
        "task_a_full", "ip_adapter_plus_s0.3",
    )
    baseline = stage(
        root, "base_model", base / "base_no_personalization",
        "task_a_full", "base_no_personalization",
    )
    print(f"STAGED scale03={scale03} baseline={baseline}")


if __name__ == "__main__":
    main()
