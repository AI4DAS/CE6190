#!/usr/bin/env python3
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.project_root.resolve()
    frame = pd.read_csv(root / "results/metrics/dataset_summary.csv")
    subjects = frame.drop_duplicates(["dataset", "subject_id"])
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    subjects.groupby("dataset")["subject_id"].nunique().plot.bar(ax=axes[0], color="#4472C4")
    axes[0].set_title("Pilot subjects per dataset"); axes[0].set_ylabel("subjects"); axes[0].tick_params(axis="x", rotation=0)
    subjects["num_images"].plot.hist(ax=axes[1], bins=range(1, int(subjects["num_images"].max()) + 2), color="#70AD47")
    axes[1].set_title("Images per subject"); axes[1].set_xlabel("images")
    subjects.groupby("class_name")["subject_id"].nunique().plot.bar(ax=axes[2], color="#ED7D31")
    axes[2].set_title("Pilot subject categories"); axes[2].set_ylabel("subjects"); axes[2].tick_params(axis="x", rotation=0)
    fig.suptitle("Dataset summary (one-subject pilot)", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    output = root / "results/figures/dataset/dataset_summary"
    fig.savefig(output.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()

