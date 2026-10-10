#!/usr/bin/env python3
import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    metadata = json.loads((args.subject_dir / "metadata.json").read_text())
    images = sorted((args.subject_dir / "images").glob("*"))
    images = [p for p in images if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
    columns = min(3, len(images))
    rows = math.ceil(len(images) / columns)
    fig, axes = plt.subplots(rows, columns, figsize=(5 * columns, 4.2 * rows), squeeze=False)
    for axis in axes.flat:
        axis.axis("off")
    for axis, path in zip(axes.flat, images):
        with Image.open(path) as image:
            axis.imshow(image.convert("RGB"))
        axis.set_title(path.name, fontsize=12)
        axis.axis("off")
    fig.suptitle(f"{metadata['dataset']} / {metadata['subject_id']} ({metadata['class_name']})", fontsize=16)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{metadata['dataset']}_{metadata['subject_id']}_contact_sheet"
    fig.savefig(args.output_dir / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(args.output_dir / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()

