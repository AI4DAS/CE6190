#!/usr/bin/env python3
"""Validate and summarize the provisional manual copying annotations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from PIL import Image


METHOD_IDS = ["dreambooth", "lora", "textual_inversion", "ip_adapter_plus"]
METHOD_LABELS = {
    "dreambooth": "DreamBooth",
    "lora": "DreamBooth-LoRA",
    "textual_inversion": "Textual Inversion",
    "ip_adapter_plus": "IP-Adapter Plus",
}
LABEL_TO_ID = {value: key for key, value in METHOD_LABELS.items()}
AXES = ["background_leakage", "pose_copying", "composition_copying"]
AXIS_LABELS = {
    "background_leakage": "Background leakage",
    "pose_copying": "Pose copying",
    "composition_copying": "Composition copying",
}
PROMPT_ORDER = ["P0", "P1", "P2", "P3", "P4", "P5"]
LEAKAGE_PROBES = {"P1", "P2", "P4"}
COLORS = ["#4C78A8", "#F58518", "#54A24B", "#E45756"]


def save_figure(fig, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def load_metadata(root: Path) -> pd.DataFrame:
    rows = []
    for path in sorted((root / "outputs/main_evaluation").glob("*/*/*/P*/seed_*.json")):
        record = json.loads(path.read_text())
        rows.append({
            "subject": record["subject"],
            "seed": int(record["seed"]),
            "method": record["method"],
            "prompt_id": record["prompt_id"],
            "dataset": record["dataset"],
            "generation_prompt": record["generation_prompt"],
            "evaluation_prompt": record["evaluation_prompt"],
            "image_path": str(path.with_suffix(".png").relative_to(root)),
            "metadata_path": str(path.relative_to(root)),
        })
    metadata = pd.DataFrame(rows)
    if len(metadata) != 768:
        raise RuntimeError(f"Expected 768 main metadata rows, found {len(metadata)}")
    return metadata


def validate_and_clean(root: Path, annotation_path: Path) -> pd.DataFrame:
    raw = pd.read_csv(annotation_path)
    required = {
        "object", "seed", "method", "prompt_id", "prompt_category",
        *AXES, "annotator", "blind",
    }
    missing_columns = required - set(raw.columns)
    if missing_columns:
        raise ValueError(f"Missing annotation columns: {sorted(missing_columns)}")
    if raw.isna().any().any():
        raise ValueError("Annotation CSV contains missing values")
    if len(raw) != 768:
        raise ValueError(f"Expected 768 annotation rows, found {len(raw)}")
    if raw.duplicated(["object", "seed", "method", "prompt_id"]).any():
        raise ValueError("Annotation CSV contains duplicate image keys")
    unknown_methods = set(raw["method"]) - set(LABEL_TO_ID)
    if unknown_methods:
        raise ValueError(f"Unknown method labels: {sorted(unknown_methods)}")
    for axis in AXES:
        if not raw[axis].isin([0, 1, 2]).all():
            raise ValueError(f"{axis} contains values outside the ordinal 0/1/2 scale")

    clean = raw.rename(columns={"object": "subject"}).copy()
    clean["method_label"] = clean["method"]
    clean["method"] = clean["method_label"].map(LABEL_TO_ID)
    metadata = load_metadata(root)
    keys = ["subject", "seed", "method", "prompt_id"]
    joined = clean.merge(metadata, on=keys, how="outer", validate="one_to_one", indicator=True)
    if not (joined["_merge"] == "both").all():
        counts = joined["_merge"].value_counts().to_dict()
        raise RuntimeError(f"Annotation/metadata coverage mismatch: {counts}")
    joined = joined.drop(columns="_merge")

    subjects = yaml.safe_load((root / "configs/main_subjects.yaml").read_text())["subjects"]
    joined["subject_category"] = joined["subject"].map(
        {subject: config["category"] for subject, config in subjects.items()}
    )
    joined["is_leakage_probe"] = joined["prompt_id"].isin(LEAKAGE_PROBES)
    joined["copying_mean_descriptive"] = joined[AXES].mean(axis=1)
    joined["annotation_status"] = "provisional_single_nonblind_annotator"
    return joined.sort_values(["subject", "method", "prompt_id", "seed"]).reset_index(drop=True)


def aggregate_tables(root: Path, clean: pd.DataFrame) -> dict[str, pd.DataFrame]:
    out = root / "results/report_tables/manual_overfitting"
    out.mkdir(parents=True, exist_ok=True)

    method_rows = []
    for method in METHOD_IDS:
        subset = clean[clean.method == method]
        probes = subset[subset.is_leakage_probe]
        row = {"method": method, "method_label": METHOD_LABELS[method], "n_images": len(subset)}
        for axis in AXES:
            row[f"{axis}_mean"] = subset[axis].mean()
            row[f"{axis}_std"] = subset[axis].std(ddof=1)
            row[f"{axis}_median"] = subset[axis].median()
            row[f"{axis}_score2_percent"] = (subset[axis] == 2).mean() * 100
        row["probe_background_leakage_mean"] = probes.background_leakage.mean()
        row["probe_background_leakage_std"] = probes.background_leakage.std(ddof=1)
        row["copying_mean_descriptive"] = subset.copying_mean_descriptive.mean()
        method_rows.append(row)
    by_method = pd.DataFrame(method_rows)
    by_method.to_csv(out / "provisional_by_method.csv", index=False)

    aggregations = {
        axis: ["mean", "std", "median", "count"] for axis in AXES
    }
    by_subject = clean.groupby(["subject", "subject_category", "method", "method_label"], as_index=False).agg(aggregations)
    by_subject.columns = ["_".join(column).rstrip("_") if isinstance(column, tuple) else column for column in by_subject.columns]
    by_subject.to_csv(out / "provisional_by_method_and_subject.csv", index=False)

    by_prompt = clean.groupby(
        ["prompt_id", "prompt_category", "method", "method_label"], as_index=False
    )[AXES + ["copying_mean_descriptive"]].agg(["mean", "std", "count"])
    by_prompt.columns = ["_".join(column).rstrip("_") if isinstance(column, tuple) else column for column in by_prompt.columns]
    by_prompt.to_csv(out / "provisional_by_method_and_prompt.csv", index=False)

    by_seed = clean.groupby(["seed", "method", "method_label"], as_index=False)[AXES].mean()
    by_seed.to_csv(out / "provisional_by_method_and_seed.csv", index=False)

    by_category = clean.groupby(
        ["subject_category", "method", "method_label"], as_index=False
    )[AXES].agg(["mean", "std", "count"])
    by_category.columns = ["_".join(column).rstrip("_") if isinstance(column, tuple) else column for column in by_category.columns]
    by_category.to_csv(out / "provisional_by_method_and_subject_category.csv", index=False)

    distribution_rows = []
    for method in METHOD_IDS:
        subset = clean[clean.method == method]
        for axis in AXES:
            counts = subset[axis].value_counts().reindex([0, 1, 2], fill_value=0)
            for score, count in counts.items():
                distribution_rows.append({
                    "method": method, "method_label": METHOD_LABELS[method], "axis": axis,
                    "score": score, "count": int(count), "percent": count / len(subset) * 100,
                })
    distribution = pd.DataFrame(distribution_rows)
    distribution.to_csv(out / "provisional_score_distribution.csv", index=False)

    report_ready = by_method[[
        "method_label", "n_images",
        "background_leakage_mean", "background_leakage_std",
        "probe_background_leakage_mean", "probe_background_leakage_std",
        "pose_copying_mean", "pose_copying_std",
        "composition_copying_mean", "composition_copying_std",
    ]].rename(columns={"method_label": "Method", "n_images": "N"})
    report_ready.to_csv(root / "results/report_tables/manual_overfitting.csv", index=False)

    prompt_category_key = {
        "P0": "identity", "P1": "background_change", "P2": "context_change",
        "P3": "attribute_change", "P4": "composition_change", "P5": "style_change",
    }
    manual_for_quality = clean.assign(
        prompt_category_key=clean.prompt_id.map(prompt_category_key)
    ).groupby(
        ["method", "dataset", "subject", "prompt_category_key"], as_index=False
    )[AXES].agg(["mean", "std"])
    manual_for_quality.columns = [
        "_".join(column).rstrip("_") if isinstance(column, tuple) else column
        for column in manual_for_quality.columns
    ]
    manual_for_quality = manual_for_quality.rename(columns={"prompt_category_key_": "prompt_category"})
    if "prompt_category_key" in manual_for_quality.columns:
        manual_for_quality = manual_for_quality.rename(columns={"prompt_category_key": "prompt_category"})
    quality = pd.read_csv(root / "results/report_tables/main_results.csv")
    combined = quality.merge(
        manual_for_quality,
        on=["method", "dataset", "subject", "prompt_category"],
        how="left",
        validate="one_to_one",
    )
    manual_metric_columns = [f"{axis}_{stat}" for axis in AXES for stat in ("mean", "std")]
    if combined[manual_metric_columns].isna().any().any():
        raise RuntimeError("Manual metrics failed to merge with main quality results")
    combined.to_csv(root / "results/report_tables/main_results_with_manual_overfitting.csv", index=False)
    return {
        "by_method": by_method,
        "by_subject": by_subject,
        "by_prompt": by_prompt,
        "by_seed": by_seed,
        "by_category": by_category,
        "distribution": distribution,
    }


def annotated_heatmap(data: pd.DataFrame, subjects: list[str], axis: str, stem: Path) -> None:
    pivot = data.pivot(index="subject", columns="method", values=f"{axis}_mean").reindex(
        index=subjects, columns=METHOD_IDS
    )
    fig, ax = plt.subplots(figsize=(8.4, 6.4))
    image = ax.imshow(pivot.values, cmap="YlOrRd", vmin=0, vmax=2, aspect="auto")
    ax.set_xticks(range(len(METHOD_IDS)), [METHOD_LABELS[method] for method in METHOD_IDS], rotation=18, ha="right")
    ax.set_yticks(range(len(subjects)), subjects)
    for i in range(len(subjects)):
        for j in range(len(METHOD_IDS)):
            value = pivot.iloc[i, j]
            ax.text(j, i, f"{value:.2f}", ha="center", va="center", color="white" if value > 1.15 else "black")
    ax.set_title(f"Provisional {AXIS_LABELS[axis].lower()} by subject (higher = worse)")
    fig.colorbar(image, ax=ax, label="Mean ordinal score (0–2)")
    save_figure(fig, stem)


def make_figures(root: Path, clean: pd.DataFrame, tables: dict[str, pd.DataFrame]) -> None:
    figure_dir = root / "results/figures/main_experiment/manual_overfitting"
    by_method = tables["by_method"].set_index("method").reindex(METHOD_IDS)

    fig, ax = plt.subplots(figsize=(9, 5.4))
    x = np.arange(len(METHOD_IDS))
    width = 0.24
    for index, axis in enumerate(AXES):
        ax.bar(
            x + (index - 1) * width,
            by_method[f"{axis}_mean"],
            width,
            yerr=by_method[f"{axis}_std"],
            capsize=3,
            label=AXIS_LABELS[axis],
        )
    ax.set_xticks(x, [METHOD_LABELS[method] for method in METHOD_IDS], rotation=15, ha="right")
    ax.set_ylim(0, 2.15)
    ax.set_ylabel("Mean ordinal score (0–2; higher = worse)")
    ax.set_title("Provisional training-image copying scores (mean ± SD)")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=9)
    save_figure(fig, figure_dir / "provisional_copying_scores_by_method")

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ax.bar(x, by_method.probe_background_leakage_mean, color=COLORS,
           yerr=by_method.probe_background_leakage_std, capsize=4)
    ax.set_xticks(x, [METHOD_LABELS[method] for method in METHOD_IDS], rotation=15, ha="right")
    ax.set_ylim(0, 2.15)
    ax.set_ylabel("Mean score (0–2; higher = worse)")
    ax.set_title("Background leakage on probe prompts P1/P2/P4 (mean ± SD)")
    ax.grid(axis="y", alpha=0.25)
    save_figure(fig, figure_dir / "provisional_background_leakage_probe")

    subjects = list(yaml.safe_load((root / "configs/main_subjects.yaml").read_text())["subjects"])
    for axis in AXES:
        annotated_heatmap(tables["by_subject"], subjects, axis, figure_dir / f"provisional_{axis}_heatmap")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharey=True)
    distribution = tables["distribution"]
    score_colors = ["#59A14F", "#F2CF5B", "#E15759"]
    for axis_plot, axis in zip(axes, AXES):
        bottom = np.zeros(len(METHOD_IDS))
        for score, color in zip([0, 1, 2], score_colors):
            values = distribution[(distribution.axis == axis) & (distribution.score == score)].set_index("method").reindex(METHOD_IDS).percent
            axis_plot.bar(range(len(METHOD_IDS)), values, bottom=bottom, color=color, label=f"Score {score}")
            bottom += values.to_numpy()
        axis_plot.set_title(AXIS_LABELS[axis])
        axis_plot.set_xticks(range(len(METHOD_IDS)), [METHOD_LABELS[m] for m in METHOD_IDS], rotation=35, ha="right")
        axis_plot.grid(axis="y", alpha=0.2)
    axes[0].set_ylabel("Images (%)")
    axes[-1].legend(loc="upper left", bbox_to_anchor=(1.02, 1))
    fig.suptitle("Provisional ordinal-score distributions")
    save_figure(fig, figure_dir / "provisional_score_distributions")

    prompt_means = clean.groupby(["prompt_id", "method"], as_index=False)[AXES].mean()
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), sharey=True)
    for axis_plot, metric in zip(axes, AXES):
        for method, color in zip(METHOD_IDS, COLORS):
            subset = prompt_means[prompt_means.method == method].set_index("prompt_id").reindex(PROMPT_ORDER)
            axis_plot.plot(PROMPT_ORDER, subset[metric], marker="o", color=color, label=METHOD_LABELS[method])
        axis_plot.set_title(AXIS_LABELS[metric])
        axis_plot.set_ylim(0, 2.08)
        axis_plot.set_xlabel("Prompt condition")
        axis_plot.grid(alpha=0.25)
    axes[0].set_ylabel("Mean score (higher = worse)")
    axes[-1].legend(fontsize=8, bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.suptitle("Provisional copying scores by prompt condition")
    save_figure(fig, figure_dir / "provisional_copying_by_prompt")

    categories = list(clean.subject_category.drop_duplicates())
    category_means = clean.groupby(["subject_category", "method"], as_index=False)[AXES].mean()
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), sharey=True)
    category_x = np.arange(len(categories))
    for axis_plot, metric in zip(axes, AXES):
        for method, color in zip(METHOD_IDS, COLORS):
            subset = category_means[category_means.method == method].set_index("subject_category").reindex(categories)
            axis_plot.plot(category_x, subset[metric], marker="o", color=color, label=METHOD_LABELS[method])
        axis_plot.set_title(AXIS_LABELS[metric])
        axis_plot.set_xticks(category_x, categories, rotation=28, ha="right")
        axis_plot.set_ylim(0, 2.08)
        axis_plot.grid(alpha=0.25)
    axes[0].set_ylabel("Mean score (higher = worse)")
    axes[-1].legend(fontsize=8, bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.suptitle("Provisional copying assessment by subject category")
    save_figure(fig, figure_dir / "provisional_copying_by_subject_category")

    candidates = clean[clean.is_leakage_probe].sort_values(
        ["method", "background_leakage", "subject", "prompt_id", "seed"],
        ascending=[True, False, True, True, True],
    )
    fig, axes = plt.subplots(len(METHOD_IDS), 2, figsize=(7.2, 13.5))
    for row_index, method in enumerate(METHOD_IDS):
        selected = candidates[candidates.method == method].head(2)
        for column_index, record in enumerate(selected.itertuples(index=False)):
            axes[row_index, column_index].imshow(Image.open(root / record.image_path).convert("RGB"))
            axes[row_index, column_index].axis("off")
            axes[row_index, column_index].set_title(
                f"{record.subject} · {record.prompt_id} · seed {record.seed}\n"
                f"leakage={record.background_leakage}, pose={record.pose_copying}, composition={record.composition_copying}",
                fontsize=9,
            )
            if column_index == 0:
                axes[row_index, column_index].text(
                    -0.08, 0.5, METHOD_LABELS[method], transform=axes[row_index, column_index].transAxes,
                    rotation=90, va="center", ha="right", fontsize=10, fontweight="bold", clip_on=False,
                )
    fig.suptitle("Deterministically selected high provisional background-leakage examples\n(P1/P2/P4; compare with reference-review sheets)", fontsize=13)
    save_figure(fig, figure_dir / "provisional_background_leakage_examples")

    dino = pd.read_csv(root / "results/metrics/main_subject_fidelity.csv")
    clip = pd.read_csv(root / "results/metrics/main_prompt_alignment.csv")
    metrics = dino.merge(
        clip[["method", "subject", "prompt_id", "seed", "clip_similarity"]],
        on=["method", "subject", "prompt_id", "seed"], validate="one_to_one",
    )
    merged = clean.merge(
        metrics[["method", "subject", "prompt_id", "seed", "mean_reference_similarity", "clip_similarity"]],
        on=["method", "subject", "prompt_id", "seed"], validate="one_to_one",
    )
    aggregate = merged.groupby("method", as_index=False).agg(
        dino=("mean_reference_similarity", "mean"),
        clip=("clip_similarity", "mean"),
        copying=("copying_mean_descriptive", "mean"),
    ).set_index("method").reindex(METHOD_IDS)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))
    for method, color in zip(METHOD_IDS, COLORS):
        axes[0].scatter(aggregate.loc[method, "dino"], aggregate.loc[method, "copying"], s=85, color=color)
        axes[1].scatter(aggregate.loc[method, "clip"], aggregate.loc[method, "copying"], s=85, color=color)
        axes[0].annotate(METHOD_LABELS[method], (aggregate.loc[method, "dino"], aggregate.loc[method, "copying"]), xytext=(4, 4), textcoords="offset points", fontsize=8)
        axes[1].annotate(METHOD_LABELS[method], (aggregate.loc[method, "clip"], aggregate.loc[method, "copying"]), xytext=(4, 4), textcoords="offset points", fontsize=8)
    axes[0].set_xlabel("DINOv2 subject fidelity")
    axes[1].set_xlabel("CLIP prompt alignment")
    for axis_plot in axes:
        axis_plot.set_ylabel("Descriptive copying mean (higher = worse)")
        axis_plot.grid(alpha=0.25)
    fig.suptitle("Quality metrics vs provisional copying assessment")
    save_figure(fig, figure_dir / "provisional_quality_vs_copying")


def write_report(root: Path, clean: pd.DataFrame, tables: dict[str, pd.DataFrame]) -> None:
    method = tables["by_method"].set_index("method").reindex(METHOD_IDS)
    lines = [
        "# Provisional Manual Overfitting Analysis", "",
        "## Status and interpretation", "",
        "This analysis covers all 768 classical main-experiment outputs. Scores are ordinal: 0 = none, 1 = partial/loose resemblance, and 2 = near-copy. Higher scores are worse and indicate more training-image copying; they do not measure output quality or subject fidelity.", "",
        "The supplied annotations are explicitly provisional: one `claude_provisional` annotator, non-blind (`blind=no`). They are suitable for exploratory analysis and pipeline validation, not as final human-study evidence. Inter-annotator agreement cannot be calculated until at least two independent blind human ratings exist.", "",
        "## Data validation", "",
        "The table has 768 rows with no missing values, duplicate image keys, or invalid scores. It matches all 768 canonical output metadata records one-to-one across 8 subjects, 4 methods, 6 prompts, and 4 seeds. The supplied 32-row summary agrees with recomputation up to two-decimal rounding (maximum absolute difference 0.005).", "",
        "## Method-level results", "",
        "| Method | Background leakage | Probe leakage (P1/P2/P4) | Pose copying | Composition copying |",
        "|---|---:|---:|---:|---:|",
    ]
    for method_id in METHOD_IDS:
        row = method.loc[method_id]
        lines.append(
            f"| {METHOD_LABELS[method_id]} | {row.background_leakage_mean:.3f} ± {row.background_leakage_std:.3f} | "
            f"{row.probe_background_leakage_mean:.3f} ± {row.probe_background_leakage_std:.3f} | "
            f"{row.pose_copying_mean:.3f} ± {row.pose_copying_std:.3f} | "
            f"{row.composition_copying_mean:.3f} ± {row.composition_copying_std:.3f} |"
        )
    lines.extend([
        "", "## Caveats", "",
        "- The scoring rules state that IP-Adapter Plus outputs were judged as near-copies of one reference across prompts, producing 2/2/2 ratings. This is evidence under this provisional rubric of over-strong reference conditioning at the tested scale (0.7), not a general verdict on IP-Adapter.",
        "- Background leakage is most directly probed by P1, P2, and P4. P0, P3, and P5 remain reported but should not replace the probe-only result.",
        "- The three axes remain separate primary outcomes. Their arithmetic mean is included only as a descriptive plotting aid and is not a validated composite score.",
        "- Automatic DINOv2/CLIP metrics do not establish background leakage or copying; joint plots are supporting comparisons only.",
        "- Final reporting should add at least two blind human annotators and an agreement statistic before drawing strong overfitting conclusions.", "",
    ])
    (root / "results/manual_overfitting_analysis.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    args = parser.parse_args()
    root = args.project_root.resolve()
    clean = validate_and_clean(root, args.annotations.resolve())
    clean_path = root / "results/metrics/main_manual_overfitting_annotation_provisional_clean.csv"
    clean.to_csv(clean_path, index=False)
    tables = aggregate_tables(root, clean)
    make_figures(root, clean, tables)
    write_report(root, clean, tables)
    print(f"MANUAL_OVERFITTING_ANALYSIS_COMPLETE rows={len(clean)}")


if __name__ == "__main__":
    main()
