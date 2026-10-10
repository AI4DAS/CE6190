#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    root = args.project_root.resolve()
    cells = [("Reference", root / "data/processed/dreambooth/dog2/images/00.jpg")]
    for prompt_id in ["P0", "P1", "P2", "P3", "P4", "P5"]:
        path = root / f"outputs/ip_adapter/dreambooth/dog2/{prompt_id}/seed_{args.seed}.png"
        metadata = json.loads(path.with_suffix(".json").read_text())
        cells.append((f"{prompt_id}: {metadata['prompt_category']}", path))
    fig, axes = plt.subplots(1, len(cells), figsize=(3.4 * len(cells), 3.8))
    for axis, (title, path) in zip(axes, cells):
        with Image.open(path) as image:
            axis.imshow(image.convert("RGB"))
        axis.set_title(title, fontsize=10)
        axis.axis("off")
    fig.suptitle(f"IP-Adapter Plus pilot — dog2 — seed {args.seed}", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    output = root / "results/figures/comparison/ip_adapter_pilot_grid"
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()

