#!/usr/bin/env python3
"""Build Task D grids without assigning subjective artifact labels."""

from pathlib import Path
import json
import matplotlib.pyplot as plt
from PIL import Image


LABELS = {
    "default_fp16_g7p5": "FP16, CFG 7.5, weight 1.0",
    "guidance5_fp16": "FP16, CFG 5.0, weight 1.0",
    "default_fp32_g7p5": "FP32, CFG 7.5, weight 1.0",
    "lora_weight0p8_fp16": "FP16, CFG 7.5, weight 0.8",
    "lora_weight0p6_fp16": "FP16, CFG 7.5, weight 0.6",
}


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    source = root / "outputs/experiments/task_d_face_corruption"
    destination = root / "results/figures/ablation/task_d"
    destination.mkdir(parents=True, exist_ok=True)
    for method in ("dreambooth", "lora"):
        variants = ["default_fp16_g7p5", "guidance5_fp16", "default_fp32_g7p5"]
        if method == "lora":
            variants += ["lora_weight0p8_fp16", "lora_weight0p6_fp16"]
        for subject in ("dog2", "cat"):
            fig, axes = plt.subplots(len(variants), 6, figsize=(18, 3.15 * len(variants)))
            for row_index, variant in enumerate(variants):
                for column, prompt_id in enumerate(("P0", "P1", "P2", "P3", "P4", "P5")):
                    path = source / method / subject / variant / prompt_id / "seed_42.png"
                    meta = path.with_suffix(".json")
                    if not path.is_file() or not meta.is_file():
                        raise FileNotFoundError(path)
                    record = json.loads(meta.read_text())
                    axes[row_index, column].imshow(Image.open(path).convert("RGB"))
                    axes[row_index, column].axis("off")
                    if row_index == 0:
                        axes[row_index, column].set_title(f"{prompt_id}: {record['prompt_category']}", fontsize=10)
                    if column == 0:
                        axes[row_index, column].set_ylabel(LABELS[variant], fontsize=9)
            fig.suptitle(f"{subject}: {method} color-blotch inference checks (seed 42)", fontsize=15)
            fig.tight_layout(rect=(0, 0, 1, 0.975))
            stem = destination / f"face_corruption_{method}_{subject}"
            fig.savefig(stem.with_suffix(".png"), dpi=200, bbox_inches="tight")
            fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
            plt.close(fig)
    note = destination / "OBSERVATION_TEMPLATE.md"
    if not note.exists():
        note.write_text(
            "# Task D visual inspection\n\n"
            "No automatic causal claim is assigned. Compare each one-factor-at-a-time row and record "
            "whether visible color blotches are absent, reduced, unchanged, or increased.\n"
        )
    print("TASK_D_GRIDS_COMPLETE grids=4")


if __name__ == "__main__":
    main()
