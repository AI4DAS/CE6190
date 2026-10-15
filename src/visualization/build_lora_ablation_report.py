#!/usr/bin/env python3
"""Build report tables, curves, and review grids for dog2 LoRA ablations."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from PIL import Image
from safetensors import safe_open


VARIANT_ORDER = ["r4_s800", "r16_s400", "r16_s800", "r16_s1200", "r32_s800"]
PROMPTS = ["P0", "P1", "P2", "P3", "P4", "P5"]
PROMPT_LABELS = ["Identity", "Background", "Context", "Attribute", "Composition", "Style"]


def save(fig, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def tensor_count(path: Path) -> int:
    with safe_open(path, framework="pt", device="cpu") as handle:
        return sum(int(np.prod(handle.get_slice(key).get_shape())) for key in handle.keys())


def weights_path(root: Path, variant: str) -> Path:
    paths = {
        "r4_s800": "outputs/ablations/lora_rank/dog2_lora_r4_s800_seed42/pytorch_lora_weights.safetensors",
        "r16_s400": "outputs/training/lora/dog2_lora_r16_s800_seed42/checkpoint-400/pytorch_lora_weights.safetensors",
        "r16_s800": "outputs/training/lora/dog2_lora_r16_s800_seed42/checkpoint-800/pytorch_lora_weights.safetensors",
        "r16_s1200": "outputs/ablations/lora_steps/dog2_lora_r16_s1200_seed42/checkpoint-1200/pytorch_lora_weights.safetensors",
        "r32_s800": "outputs/ablations/lora_rank/dog2_lora_r32_s800_seed42/pytorch_lora_weights.safetensors",
    }
    return root / paths[variant]


def curve(table: pd.DataFrame, x: str, xlabel: str, stem: Path, title_prefix: str) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4))
    specs = [
        ("dino_mean", "dino_std", "DINOv2 mean-reference similarity"),
        ("clip_mean", "clip_std", "CLIP neutral-prompt alignment"),
        ("dino_max_mean", "dino_max_std", "DINOv2 max-reference similarity"),
    ]
    for axis, (value, error, ylabel) in zip(axes, specs):
        axis.errorbar(table[x], table[value], yerr=table[error], marker="o", capsize=4, linewidth=2)
        axis.set_xlabel(xlabel)
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.25)
    fig.suptitle(title_prefix)
    save(fig, stem)


def qualitative_grid(root: Path, variants: list[str], row_labels: list[str], seed: int, stem: Path, title: str) -> None:
    refs = sorted((root / "data/processed/dreambooth/dog2/images").glob("*.*"))
    rows = len(variants) + 1
    fig, axes = plt.subplots(rows, len(PROMPTS), figsize=(15, 2.55 * rows))
    for column, prompt in enumerate(PROMPTS):
        axes[0, column].imshow(Image.open(refs[column % len(refs)]).convert("RGB"))
        axes[0, column].set_title(f"{prompt} {PROMPT_LABELS[column]}", fontsize=10)
        axes[0, column].axis("off")
        if column == 0:
            axes[0, column].text(
                -0.08, 0.5, "Reference", transform=axes[0, column].transAxes,
                rotation=90, va="center", ha="right", fontsize=10,
                fontweight="bold", clip_on=False,
            )
    for row, (variant, label) in enumerate(zip(variants, row_labels), start=1):
        for column, prompt in enumerate(PROMPTS):
            path = root / f"outputs/ablations/lora_generation/{variant}/{prompt}/seed_{seed}.png"
            axes[row, column].imshow(Image.open(path).convert("RGB"))
            axes[row, column].axis("off")
            if column == 0:
                axes[row, column].text(
                    -0.08, 0.5, label, transform=axes[row, column].transAxes,
                    rotation=90, va="center", ha="right", fontsize=10,
                    fontweight="bold", clip_on=False,
                )
    fig.suptitle(f"{title} (predetermined seed={seed})", fontsize=14)
    save(fig, stem)


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    dino = pd.read_csv(root / "results/metrics/lora_ablation_subject_fidelity.csv")
    clip = pd.read_csv(root / "results/metrics/lora_ablation_prompt_alignment.csv")
    merged = dino.merge(
        clip[["variant", "prompt_id", "seed", "clip_similarity"]],
        on=["variant", "prompt_id", "seed"], validate="one_to_one",
    )
    if len(merged) != 120 or merged.isna().any().any():
        raise RuntimeError("Expected 120 complete ablation metric rows")

    summary = merged.groupby(["variant", "rank", "training_steps"], as_index=False).agg(
        dino_mean=("mean_reference_similarity", "mean"),
        dino_std=("mean_reference_similarity", "std"),
        dino_max_mean=("max_reference_similarity", "mean"),
        dino_max_std=("max_reference_similarity", "std"),
        clip_mean=("clip_similarity", "mean"), clip_std=("clip_similarity", "std"),
        image_count=("image_path", "count"),
    )
    summary["trainable_parameters"] = summary.variant.map(
        {variant: tensor_count(weights_path(root, variant)) for variant in VARIANT_ORDER}
    )
    summary["checkpoint_size_mb"] = summary.variant.map(
        {variant: weights_path(root, variant).stat().st_size / 1024**2 for variant in VARIANT_ORDER}
    )
    summary["manual_overfitting_status"] = "pending_human_annotation"
    summary.to_csv(root / "results/report_tables/lora_ablation_all_variants.csv", index=False)
    rank_table = summary[summary.training_steps.eq(800)].sort_values("rank")
    step_table = summary[summary["rank"].eq(16)].sort_values("training_steps")
    rank_table.to_csv(root / "results/report_tables/lora_rank_ablation.csv", index=False)
    step_table.to_csv(root / "results/report_tables/lora_training_steps.csv", index=False)

    figure_root = root / "results/figures/ablation/lora"
    curve(rank_table, "rank", "LoRA rank", figure_root / "lora_rank_quality_curves", "LoRA rank sensitivity on dog2 (800 steps)")
    curve(step_table, "training_steps", "Training steps", figure_root / "lora_training_step_curves", "Training-step trajectory on dog2 (rank 16)")

    by_prompt = merged[merged["rank"].eq(16)].groupby(
        ["training_steps", "prompt_category"], as_index=False
    ).agg(dino=("mean_reference_similarity", "mean"), clip=("clip_similarity", "mean"))
    categories = list(yaml.safe_load((root / "configs/main_prompts.yaml").read_text())["prompt_templates"].values())
    category_order = [item["category"] for item in categories]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for category in category_order:
        selected = by_prompt[by_prompt.prompt_category.eq(category)].sort_values("training_steps")
        axes[0].plot(selected.training_steps, selected.dino, marker="o", label=category)
        axes[1].plot(selected.training_steps, selected["clip"], marker="o", label=category)
    axes[0].set_ylabel("DINOv2 mean-reference similarity")
    axes[1].set_ylabel("CLIP neutral-prompt alignment")
    for axis in axes:
        axis.set_xlabel("Training steps")
        axis.grid(alpha=0.25)
    axes[1].legend(fontsize=7, bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.suptitle("Prompt-category sensitivity along the rank-16 training trajectory")
    save(fig, figure_root / "lora_training_steps_by_prompt_category")

    for seed in [42, 123, 456, 789]:
        qualitative_grid(
            root, ["r4_s800", "r16_s800", "r32_s800"],
            ["Rank 4", "Rank 16", "Rank 32"], seed,
            figure_root / f"lora_rank_grid_seed{seed}", "LoRA rank comparison",
        )
        qualitative_grid(
            root, ["r16_s400", "r16_s800", "r16_s1200"],
            ["400 steps", "800 steps", "1200 steps"], seed,
            figure_root / f"lora_training_steps_grid_seed{seed}",
            "LoRA training-step comparison",
        )

    annotations = merged[["variant", "rank", "training_steps", "prompt_id", "prompt_category", "seed", "image_path"]].copy()
    annotations["background_leakage"] = ""
    annotations["pose_copying"] = ""
    annotations["composition_copying"] = ""
    annotations["notes"] = ""
    annotations.to_csv(root / "results/metrics/lora_ablation_manual_annotation.csv", index=False)

    rank_rows = "\n".join(
        f"| {int(row['rank'])} | {row.dino_mean:.4f} ± {row.dino_std:.4f} | {row.clip_mean:.4f} ± {row.clip_std:.4f} | {int(row.trainable_parameters):,} | {row.checkpoint_size_mb:.2f} |"
        for _, row in rank_table.iterrows()
    )
    step_rows = "\n".join(
        f"| {int(row.training_steps)} | {row.dino_mean:.4f} ± {row.dino_std:.4f} | {row.dino_max_mean:.4f} ± {row.dino_max_std:.4f} | {row.clip_mean:.4f} ± {row.clip_std:.4f} |"
        for _, row in step_table.iterrows()
    )
    report = f"""# LoRA Rank and Training-Step Ablation

## Scope

This controlled ablation uses the `dog2` subject, SD1.5, prior preservation,
200 class images, FP16, seed 42, six prompts, and four fixed generation seeds.
Only LoRA rank or training duration changes. The step comparison follows one
rank-16 trajectory: checkpoint 400, checkpoint 800, and a continuation from
checkpoint 800 to checkpoint 1200.

CLIP uses the clean category-level evaluation prompt for every image. It does
not contain `sks` or a learned placeholder token.

## Rank sensitivity at 800 steps

| Rank | DINO mean ± SD | CLIP mean ± SD | Trainable parameters | Checkpoint MiB |
|---:|---:|---:|---:|---:|
{rank_rows}

## Training-step trajectory at rank 16

| Steps | DINO mean ± SD | DINO max-reference mean ± SD | CLIP mean ± SD |
|---:|---:|---:|---:|
{step_rows}

From 400 to 800 to 1200 steps, whole-image mean-reference DINO similarity was
{step_table.iloc[0].dino_mean:.4f}, {step_table.iloc[1].dino_mean:.4f}, and
{step_table.iloc[2].dino_mean:.4f}; neutral-prompt CLIP was
{step_table.iloc[0].clip_mean:.4f}, {step_table.iloc[1].clip_mean:.4f}, and
{step_table.iloc[2].clip_mean:.4f}. These automatic measurements do not show a
monotonic increase in training-image similarity with longer training; both
metrics fall at 1200 steps in this single-subject trajectory.

## RQ2 interpretation boundary

DINO similarity is supporting evidence, not a detector of background leakage,
pose copying, or composition copying. Therefore these results do **not** yet
establish whether longer training worsens qualitative overfitting. The review
grid and `results/metrics/lora_ablation_manual_annotation.csv` are ready for
blind human scoring on the existing 120 images. A manual-overfitting curve
must be produced only after those ratings are supplied.

## Limitations

- This is a one-subject ablation and is not a cross-dataset conclusion.
- Rank jobs ran concurrently on separate MIG instances that share a physical
  GPU; their measured wall times should not be interpreted as pure rank-scaling
  benchmarks.
- Whole-image DINO can respond to both subject and background similarity.
"""
    (root / "results/lora_ablation_summary.md").write_text(report)
    print("LORA_ABLATION_REPORT_COMPLETE")


if __name__ == "__main__":
    main()
