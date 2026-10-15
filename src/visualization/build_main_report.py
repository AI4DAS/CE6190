#!/usr/bin/env python3
"""Build main-experiment compute tables, figures, and report summary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from PIL import Image
from safetensors import safe_open


METHODS = ["base_model", "dreambooth", "lora", "textual_inversion", "ip_adapter_plus"]
LABELS = {"base_model": "Base SD1.5 (no personalization)",
          "dreambooth": "DreamBooth", "lora": "DreamBooth-LoRA",
          "textual_inversion": "Textual Inversion", "ip_adapter_plus": "IP-Adapter Plus"}
COLORS = ["#9D9D9D", "#4C78A8", "#F58518", "#54A24B", "#E45756"]
PROMPT_TITLES = ["Identity", "Background", "Context", "Attribute", "Composition", "Style"]


def tensor_count(path: Path) -> int:
    with safe_open(path, framework="pt", device="cpu") as handle:
        return sum(int(np.prod(handle.get_slice(key).get_shape())) for key in handle.keys())


def mib(path: Path) -> float:
    return path.stat().st_size / 1024**2


def model_dir_mib(path: Path) -> float:
    total = 0
    for item in path.rglob("*"):
        if not item.is_file():
            continue
        relative = item.relative_to(path)
        if any(part.startswith("checkpoint-") or part == "logs" for part in relative.parts):
            continue
        if item.suffix in {".bin", ".safetensors", ".json", ".txt"}:
            total += item.stat().st_size
    return total / 1024**2


def run_dir(root: Path, subject: str, method: str) -> Path:
    if method == "dreambooth":
        return root / f"outputs/training/dreambooth/{subject}_db_s800_seed42"
    if method == "lora":
        return root / f"outputs/training/lora/{subject}_lora_r16_s800_seed42"
    return root / f"outputs/training/textual_inversion/{subject}_ti_s3000_seed42"


def runtime(path: Path) -> float:
    values = {}
    for line in path.read_text().splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            values[key] = value
    return float(values["training_time_seconds"])


def peak_vram(path: Path) -> float:
    for line in path.read_text().splitlines():
        if line.startswith("peak_vram_gb="):
            return float(line.split("=", 1)[1])
    return np.nan


def save_figure(fig, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def compute_table(root: Path, subjects: dict, metadata: pd.DataFrame) -> pd.DataFrame:
    base = root / "models/base/stable-diffusion-v1-5"
    total_base = sum(tensor_count(base / part) for part in [
        "unet/diffusion_pytorch_model.safetensors",
        "vae/diffusion_pytorch_model.safetensors",
        "text_encoder/model.safetensors",
    ])
    db_trainable = tensor_count(base / "unet/diffusion_pytorch_model.safetensors")
    adapter = root / "models/ip_adapter/IP-Adapter/models/ip-adapter-plus_sd15.bin"
    rows = []
    for subject, record in subjects.items():
        for method in METHODS:
            subset = metadata[(metadata.subject == subject) & (metadata.method == method)]
            if method in {"base_model", "ip_adapter_plus"}:
                shared = mib(adapter) if method == "ip_adapter_plus" else 0.0
                references = 1 if method == "ip_adapter_plus" else 0
                seconds, peak, trainable, personalized = 0.0, np.nan, 0, 0.0
                class_images, steps = 0, 0
            else:
                directory = run_dir(root, subject, method)
                seconds = runtime(directory / "runtime.txt")
                peak = peak_vram(directory / "peak_vram.txt")
                if method == "dreambooth":
                    trainable = db_trainable
                    personalized = model_dir_mib(directory)
                    class_images, steps = 200, 800
                elif method == "lora":
                    weights = directory / "pytorch_lora_weights.safetensors"
                    trainable, personalized = tensor_count(weights), mib(weights)
                    class_images, steps = 200, 800
                else:
                    weights = directory / "learned_embeds.safetensors"
                    trainable, personalized = tensor_count(weights), mib(weights)
                    class_images, steps = 0, 3000
                shared, references = 0.0, record["expected_images"]
            rows.append({
                "method": method, "dataset": record["dataset"], "subject": subject,
                "category": record["category"], "training_required": method != "ip_adapter_plus",
                "training_steps": steps, "training_time_seconds": seconds,
                "training_time_minutes": seconds / 60, "peak_training_vram_gb": peak,
                "trainable_parameters": trainable,
                "trainable_parameter_percent": trainable / total_base * 100,
                "personalized_checkpoint_size_mb": personalized,
                "shared_adapter_size_mb": shared, "reference_images": references,
                "class_images": class_images,
                "inference_latency_mean_sec": subset.generation_latency_sec.mean(),
                "inference_latency_std_sec": subset.generation_latency_sec.std(ddof=1),
                "peak_inference_vram_gb": subset.peak_inference_vram_gb.max(),
                "precision": "fp16", "base_model": "stable-diffusion-v1-5/stable-diffusion-v1-5",
            })
    result = pd.DataFrame(rows)
    result.to_csv(root / "results/metrics/main_compute_cost.csv", index=False)
    return result


def image_grid(paths, row_labels, column_labels, title, stem: Path) -> None:
    rows, columns = len(paths), len(paths[0])
    fig, axes = plt.subplots(rows, columns, figsize=(2.65 * columns, 2.5 * rows))
    axes = np.atleast_2d(axes)
    for i in range(rows):
        for j in range(columns):
            axes[i, j].imshow(Image.open(paths[i][j]).convert("RGB"))
            axes[i, j].axis("off")
            if i == 0:
                axes[i, j].set_title(column_labels[j], fontsize=11)
            if j == 0:
                axes[i, j].set_ylabel(row_labels[i], fontsize=11)
    fig.suptitle(title, fontsize=15, y=1.01)
    save_figure(fig, stem)


def method_bar(values, errors, ylabel, title, stem):
    fig, ax = plt.subplots(figsize=(7, 4.8))
    x = np.arange(len(METHODS))
    ax.bar(x, values, yerr=errors, capsize=4, color=COLORS)
    ax.set_xticks(x, [LABELS[method] for method in METHODS], rotation=15, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)
    save_figure(fig, stem)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.project_root.resolve()
    subjects = yaml.safe_load((root / "configs/main_subjects.yaml").read_text())["subjects"]
    generation = yaml.safe_load((root / "configs/main_prompts.yaml").read_text())["generation"]
    ip_adapter_scale = generation["ip_adapter_scale"]
    dino = pd.read_csv(root / "results/metrics/main_subject_fidelity.csv")
    clip = pd.read_csv(root / "results/metrics/main_prompt_alignment.csv")
    merged = dino.merge(
        clip[["method", "subject", "prompt_id", "seed", "clip_similarity"]],
        on=["method", "subject", "prompt_id", "seed"], validate="one_to_one"
    )
    merged["subject_category"] = merged["subject"].map(
        {subject: record["category"] for subject, record in subjects.items()}
    )
    if merged["subject_category"].isna().any():
        missing = sorted(merged.loc[merged["subject_category"].isna(), "subject"].unique())
        raise RuntimeError(f"Missing subject categories for: {missing}")
    metadata_rows = []
    for method in METHODS:
        for path in sorted((root / "outputs/main_evaluation" / method).glob("*/*/P*/seed_*.json")):
            metadata_rows.append(json.loads(path.read_text()))
    metadata = pd.DataFrame(metadata_rows)
    if len(metadata) != 960:
        raise RuntimeError(f"Expected 960 metadata records, found {len(metadata)}")
    compute = compute_table(root, subjects, metadata)

    quality = merged.groupby(["method", "dataset", "subject", "prompt_category"], as_index=False).agg(
        dino_fidelity=("mean_reference_similarity", "mean"),
        dino_std=("mean_reference_similarity", "std"),
        clip_alignment=("clip_similarity", "mean"),
        clip_std=("clip_similarity", "std"),
    )
    quality.to_csv(root / "results/report_tables/main_results.csv", index=False)
    category = merged.groupby(["method", "subject_category"], as_index=False).agg(
        dino_mean=("mean_reference_similarity", "mean"),
        dino_std=("mean_reference_similarity", "std"),
        clip_mean=("clip_similarity", "mean"),
        clip_std=("clip_similarity", "std"),
        image_count=("image_path", "count"),
    )
    category.to_csv(root / "results/report_tables/category_analysis.csv", index=False)
    compute.to_csv(root / "results/report_tables/compute_comparison.csv", index=False)

    figure_root = root / "results/figures/main_experiment"
    for subject, record in subjects.items():
        refs = sorted((root / record["image_dir"]).iterdir())
        paths = [[refs[index % len(refs)] for index in range(6)]]
        for method in METHODS:
            paths.append([
                root / "outputs/main_evaluation" / method / record["dataset"] / subject / f"P{index}/seed_42.png"
                for index in range(6)
            ])
        image_grid(paths, ["Reference"] + [LABELS[m] for m in METHODS], PROMPT_TITLES,
                   f"{subject}: five-method comparison (seed 42)", figure_root / "per_subject" / subject)

    representatives = {}
    for subject, record in subjects.items():
        representatives.setdefault(record["category"], subject)
    category_paths, category_labels = [], []
    for category_name, subject in representatives.items():
        record = subjects[subject]
        category_paths.append([
            root / "outputs/main_evaluation" / method / record["dataset"] / subject / "P0/seed_42.png"
            for method in METHODS
        ])
        category_labels.append(f"{category_name}\n({subject})")
    image_grid(category_paths, category_labels, [LABELS[m] for m in METHODS],
               "Representative identity results by subject category", figure_root / "category_comparison")

    overall = merged.groupby("method").agg(
        dino_mean=("mean_reference_similarity", "mean"), dino_std=("mean_reference_similarity", "std"),
        clip_mean=("clip_similarity", "mean"), clip_std=("clip_similarity", "std"),
    ).reindex(METHODS)
    method_bar(overall.dino_mean, overall.dino_std, "Cosine similarity",
               "Main experiment: DINOv2 subject fidelity", figure_root / "dino_fidelity")
    method_bar(overall.clip_mean, overall.clip_std, "Cosine similarity",
               "Main experiment: CLIP prompt alignment", figure_root / "clip_alignment")
    fig, ax = plt.subplots(figsize=(7, 5.5))
    for method, color in zip(METHODS, COLORS):
        subset = merged[merged.method == method]
        ax.scatter(subset.clip_similarity, subset.mean_reference_similarity, s=20, alpha=0.55,
                   label=LABELS[method], color=color)
    ax.set_xlabel("CLIP prompt alignment")
    ax.set_ylabel("DINOv2 subject fidelity")
    ax.set_title("Main experiment: fidelity–alignment trade-off")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    save_figure(fig, figure_root / "dino_vs_clip_scatter")

    costs = compute.groupby("method").agg(
        training_time_minutes=("training_time_minutes", "mean"),
        personalized_checkpoint_size_mb=("personalized_checkpoint_size_mb", "mean"),
        inference_latency=("inference_latency_mean_sec", "mean"),
    ).reindex(METHODS)
    for column, ylabel, title, filename in [
        ("training_time_minutes", "Minutes per subject", "Subject-specific training time", "training_cost"),
        ("personalized_checkpoint_size_mb", "MiB per subject", "Personalized storage", "storage_comparison"),
        ("inference_latency", "Seconds per image", "Inference latency", "latency_comparison"),
    ]:
        method_bar(costs[column], np.zeros(len(METHODS)), ylabel, title, figure_root / filename)

    failures = merged.assign(combined=merged.mean_reference_similarity + merged.clip_similarity).sort_values("combined")
    failures = failures.groupby("method", sort=False).head(2).set_index("method").loc[METHODS].reset_index()
    failure_paths = []
    for method in METHODS:
        rows = failures[failures.method == method]
        failure_paths.append([root / path for path in rows.image_path])
    image_grid(failure_paths, [LABELS[m] for m in METHODS], ["Lowest combined score 1", "Lowest combined score 2"],
               "Metric-selected failure cases (not manually annotated)", figure_root / "failure_cases")

    annotations = metadata[["method", "dataset", "subject", "prompt_id", "prompt_category", "seed"]].copy()
    annotations["background_leakage"] = ""
    annotations["pose_copying"] = ""
    annotations["composition_copying"] = ""
    annotations["notes"] = ""
    annotations.to_csv(root / "results/metrics/main_manual_overfitting_annotation.csv", index=False)

    report = [
        "# Main Experiment Summary", "", "## Training completion status", "",
        "All 24 subject-specific training jobs completed and passed artifact load and generation validation. Twenty `FAILED` markers were false post-training failures caused by the storage guard rejecting Diffusers `checkpoint-*` directories; they were reconciled only after validation. No training rerun was required.", "",
        "## Methods compared", "", f"The non-personalized SD1.5 category-only baseline, Full DreamBooth, DreamBooth-LoRA, Textual Inversion, and IP-Adapter Plus were evaluated under the SD1.5 main protocol. IP-Adapter Plus used one fixed reference image per subject and conditioning scale {ip_adapter_scale}.", "",
        "## Dataset statistics", "", f"The benchmark contains {len(subjects)} non-human subjects and {sum(r['expected_images'] for r in subjects.values())} original reference images. Prior class images were excluded from quality evaluation.", "",
        "## Quantitative results", "",
        "| Method | DINOv2 mean ± std | CLIP mean ± std |", "|---|---:|---:|",
    ]
    for method in METHODS:
        row = overall.loc[method]
        report.append(f"| {LABELS[method]} | {row.dino_mean:.4f} ± {row.dino_std:.4f} | {row.clip_mean:.4f} ± {row.clip_std:.4f} |")
    report.extend([
        "", "## Efficiency analysis", "",
        "Per-subject training, storage, and measured inference costs are reported in `results/metrics/main_compute_cost.csv`. IP-Adapter Plus has no per-subject optimization but retains shared-adapter and inference costs.", "",
        f"Mean per-subject training time was {costs.loc['dreambooth', 'training_time_minutes']:.2f} minutes for DreamBooth, {costs.loc['lora', 'training_time_minutes']:.2f} minutes for LoRA, and {costs.loc['textual_inversion', 'training_time_minutes']:.2f} minutes for Textual Inversion. The base-model control and IP-Adapter Plus required no subject-specific training. Their missing training VRAM is reported as N/A because no per-subject training occurs.", "",
        "## Failure cases", "",
        "The failure-case figure uses a predetermined metric rule (lowest DINOv2-plus-CLIP sum), not visual cherry-picking. Manual leakage and copying fields remain blank pending human annotation.", "",
        "## Limitations", "",
        f"This eight-subject benchmark is broader than the dog2 pilot but remains limited to selected non-human subjects, SD1.5 for the controlled comparison, whole-image DINOv2 similarity, and one fixed prompt/seed protocol. The IP-Adapter result represents the specific Plus checkpoint, scale {ip_adapter_scale}, and one fixed reference image per subject; it must not be generalized to every IP-Adapter operating point. Metric differences do not by themselves establish perceptual quality or overfitting.", "",
        "The dog2 pilot remains a separate completed experiment; these main-experiment aggregates must not be substituted for pilot results.", "",
    ])
    (root / "results/main_experiment_summary.md").write_text("\n".join(report))
    print("MAIN_REPORT_COMPLETE")


if __name__ == "__main__":
    main()
