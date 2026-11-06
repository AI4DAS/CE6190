#!/usr/bin/env python3
"""Plot classical and modern automatic metrics with backbone grouping explicit."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ORDER = ["base_model", "dreambooth", "lora", "textual_inversion", "ip_adapter_plus", "dreamo", "uno"]
LABELS = ["Base", "DreamBooth", "LoRA", "TI", "IP-Adapter", "DreamO", "UNO"]
COLORS = ["#8c8c8c", "#0072B2", "#56B4E9", "#009E73", "#CC79A7", "#E69F00", "#D55E00"]


def aggregate(path: Path, value: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    return frame.groupby("method", as_index=False)[value].agg(["mean", "std"]).reset_index()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.project_root.resolve()
    metrics = root / "results/metrics"

    classical_dino = pd.read_csv(metrics / "main_subject_fidelity.csv")
    modern_dino = pd.read_csv(metrics / "modern_subject_fidelity.csv")
    classical_clip = pd.read_csv(metrics / "main_prompt_alignment.csv")
    modern_clip = pd.read_csv(metrics / "modern_prompt_alignment.csv")
    dino = pd.concat([classical_dino, modern_dino], ignore_index=True).groupby("method")[
        "mean_reference_similarity"
    ].agg(["mean", "std"])
    clip = pd.concat([classical_clip, modern_clip], ignore_index=True).groupby("method")[
        "clip_similarity"
    ].agg(["mean", "std"])

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.7))
    x = np.arange(len(ORDER))
    for axis, data, ylabel, title in [
        (axes[0], dino, "DINOv2 reference similarity", "Subject-reference similarity"),
        (axes[1], clip, "Neutral-prompt CLIP-T", "Prompt alignment"),
    ]:
        means = [data.loc[method, "mean"] for method in ORDER]
        stds = [data.loc[method, "std"] for method in ORDER]
        bars = axis.bar(x, means, yerr=stds, capsize=3, color=COLORS, edgecolor="black", linewidth=0.6)
        for index in (5, 6):
            bars[index].set_hatch("//")
        axis.set_xticks(x, LABELS, rotation=28, ha="right")
        axis.set_ylabel(ylabel)
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.25)
        axis.set_axisbelow(True)
    fig.text(
        0.5, 0.01,
        "Hatched bars are FLUX extensions with different backbones and generation recipes; values are descriptive, not a controlled method ranking.",
        ha="center", fontsize=8.5,
    )
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    destination = root / "results/figures/modern_extension"
    destination.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination / "all_methods_automatic_metrics.png", dpi=200, bbox_inches="tight")
    fig.savefig(destination / "all_methods_automatic_metrics.pdf", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
