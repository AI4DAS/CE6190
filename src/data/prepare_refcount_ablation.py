#!/usr/bin/env python3
"""Create deterministic image-only first-k reference directories using hard links."""

from __future__ import annotations

import os
from pathlib import Path

import yaml


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    subjects = yaml.safe_load((root / "configs/main_subjects.yaml").read_text())["subjects"]
    for subject in ("dog2", "teapot"):
        source = root / subjects[subject]["image_dir"]
        files = sorted(path for path in source.iterdir() if path.is_file())
        for count in (1, 3, 5):
            selected = files[:count]
            if len(selected) != count:
                raise RuntimeError(f"{subject} has fewer than {count} references")
            destination = root / f"data/processed/ablations/refcount/{subject}/k{count}/images"
            destination.mkdir(parents=True, exist_ok=True)
            existing = sorted(path for path in destination.iterdir() if path.is_file())
            if existing and [path.name for path in existing] != [path.name for path in selected]:
                raise RuntimeError(f"Refusing mismatched existing subset: {destination}")
            for path in selected:
                target = destination / path.name
                if not target.exists(): os.link(path, target)
            print(subject, count, ";".join(path.name for path in selected))


if __name__ == "__main__": main()
