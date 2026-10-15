#!/usr/bin/env python3
"""Create report-ready summary tables and figures from measured ablation CSVs."""

from __future__ import annotations

from pathlib import Path
import json
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PALETTE = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9", "#D55E00"]


def save(fig, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(stem.with_suffix(".png"), dpi=200, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def parse_runtime(path: Path) -> float | None:
    if not path.is_file():
        return None
    match = re.findall(r"training_time_seconds=(\d+(?:\.\d+)?)", path.read_text())
    return float(match[-1]) if match else None


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    master = pd.read_csv(root / "results/metrics/all_runs_per_image.csv")
    out = root / "results/figures/ablation/task_g"
    metrics_dir = root / "results/metrics"

    # Main comparison: neutral CLIP is the primary prompt metric.
    main = master[master.experiment == "main_benchmark"].copy()
    selected_main = main[main.method.isin(["dreambooth", "lora", "textual_inversion", "base_model", "ip_adapter_plus"])]
    old07 = master[(master.experiment == "task_a_ip_adapter") & (master.variant == "ip_adapter_plus_s0.7")]
    combined = pd.concat([selected_main, old07], ignore_index=True)
    combined["report_variant"] = combined.apply(
        lambda row: "IP-Adapter Plus (s=0.7)" if row["variant"] == "ip_adapter_plus_s0.7" else
        ("IP-Adapter Plus (s=0.3)" if row["method"] == "ip_adapter_plus" else
         {"base_model": "Base (no personalization)", "dreambooth": "DreamBooth", "lora": "DreamBooth-LoRA", "textual_inversion": "Textual Inversion"}.get(row["method"], row["method"])), axis=1
    )
    main_summary = combined.groupby("report_variant", as_index=False).agg(
        images=("image_path", "count"), dino_mean=("dino_ref_mean", "mean"), dino_std=("dino_ref_mean", "std"),
        clip_t_neutral_mean=("clip_t_neutral", "mean"), clip_t_neutral_std=("clip_t_neutral", "std"),
        nearest_train_sim_mean=("nearest_train_sim", "mean"), nearest_train_sim_std=("nearest_train_sim", "std"),
    )
    main_summary.to_csv(metrics_dir / "main_comparison_with_ip_scales.csv", index=False)

    fig, ax = plt.subplots(figsize=(8.5, 6.2))
    markers = ["o", "s", "^", "D", "P", "X"]
    low, high = main_summary.nearest_train_sim_mean.min(), main_summary.nearest_train_sim_mean.max()
    normalizer = plt.Normalize(low, high)
    for index, row in main_summary.iterrows():
        ax.scatter(row.clip_t_neutral_mean, row.dino_mean, s=115,
                   c=[row.nearest_train_sim_mean], cmap="viridis", norm=normalizer,
                   marker=markers[index % len(markers)], edgecolor="black", linewidth=0.5,
                   label=row.report_variant)
        ax.text(row.clip_t_neutral_mean + .0007, row.dino_mean + .0007, row.report_variant, fontsize=8)
    ax.set_xlabel("CLIP-T similarity (neutral class-name prompt)")
    ax.set_ylabel("DINOv2 mean reference similarity")
    ax.set_title("Personalization fidelity–prompt alignment trade-off")
    ax.grid(alpha=.25)
    scalar = plt.cm.ScalarMappable(norm=normalizer, cmap="viridis")
    fig.colorbar(scalar, ax=ax, label="Nearest-training-image DINOv2 similarity")
    ax.legend(frameon=False, fontsize=8, loc="best")
    save(fig, out / "tradeoff_scatter")

    # IP scale curves, subset and two fixed seeds only.
    subset = ["dog2", "teapot", "bear_plushie"]
    ip = master[(master.object.isin(subset)) & (master.seed.isin([42, 123]))].copy()
    ip = ip[
        ((ip.experiment == "task_a_ip_adapter") & ip.variant.str.match(r"ip_adapter_(plain|plus)_s[0-9.]+(_single)?$")) |
        ((ip.experiment == "task_a_full") & (ip.variant == "ip_adapter_plus_s0.3"))
    ]
    ip["adapter"] = ip.method.map({"ip_adapter_plain": "Plain", "ip_adapter_plus": "Plus"})
    ip["scale"] = pd.to_numeric(ip.scale)
    ip_summary = ip.groupby(["adapter", "scale"], as_index=False).agg(
        images=("image_path", "count"), dino_mean=("dino_ref_mean", "mean"),
        clip_neutral_mean=("clip_t_neutral", "mean"), copy_proxy_mean=("nearest_train_sim", "mean"),
    )
    diagnostics = pd.read_csv(metrics_dir / "ip_diagnostics.csv")
    diagnostics = diagnostics[diagnostics.subject.isin(subset) & diagnostics.seed.isin([42, 123])]
    diagnostics = diagnostics.copy()
    # The staged pre-task scale-0.3 sidecars inherited older main metadata that
    # predates adapter_type/reference_mode. Its variant and logged single
    # ref_files_used value identify the protocol unambiguously.
    diagnostics.loc[diagnostics.variant == "ip_adapter_plus_s0.3", "adapter"] = "Plus"
    diagnostics.loc[diagnostics.variant == "ip_adapter_plus_s0.3", "reference_mode"] = "single"
    diagnostics = diagnostics[diagnostics.reference_mode == "single"].copy()
    diagnostics["adapter"] = diagnostics["adapter"].fillna(
        diagnostics.adapter_type.map({"plain": "Plain", "plus": "Plus"})
    )
    sensitivity = diagnostics.groupby(["adapter", "scale"], as_index=False).prompt_pairwise_dino_mean.mean()
    ip_summary = ip_summary.merge(sensitivity, on=["adapter", "scale"], how="left", validate="one_to_one")
    ip_summary.to_csv(metrics_dir / "ip_scale_ablation_summary.csv", index=False)
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for adapter, color in zip(("Plain", "Plus"), PALETTE[:2]):
        data = ip_summary[ip_summary.adapter == adapter].sort_values("scale")
        for ax, metric, label in zip(axes.flat,
            ("dino_mean", "clip_neutral_mean", "copy_proxy_mean", "prompt_pairwise_dino_mean"),
            ("DINOv2 similarity", "CLIP-T neutral", "Nearest-reference similarity", "Mean pairwise prompt DINO (lower = more variation)")):
            ax.plot(data.scale, data[metric], marker="o", label=adapter, color=color)
            ax.set_xlabel("IP-Adapter scale"); ax.set_ylabel(label); ax.grid(alpha=.25)
    axes[0, 0].legend(frameon=False)
    save(fig, out / "ip_adapter_scale_curves")

    # Rank, reference count, prior preservation summaries and figures.
    if (metrics_dir / "rank_ablation.csv").is_file():
        rank = pd.read_csv(metrics_dir / "rank_ablation.csv")
        rank_summary = rank.groupby("rank", as_index=False).agg(
            images=("image_path", "count"), dino_mean=("dino_ref_mean", "mean"),
            clip_neutral_mean=("clip_t_neutral", "mean"), copy_proxy_mean=("nearest_train_sim", "mean"),
        )
        storage, runtimes = [], []
        for rank_value in rank_summary["rank"].astype(int):
            files, times = [], []
            for metadata in (root / f"outputs/experiments/task_b_rank/r{rank_value}").glob("**/P0/seed_42.json"):
                record = json.loads(metadata.read_text()); weight = root / record["checkpoint"]
                files.append(weight.stat().st_size / 1024**2)
                value = parse_runtime(weight.parent / "runtime.txt")
                if value is not None: times.append(value)
            storage.append(float(np.mean(files))); runtimes.append(float(np.mean(times)) if times else np.nan)
        rank_summary["adapter_size_mb"] = storage; rank_summary["training_time_sec"] = runtimes
        rank_summary.to_csv(metrics_dir / "rank_ablation_summary.csv", index=False)
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
        axes[0].plot(rank_summary["rank"], rank_summary.dino_mean, marker="o", label="DINOv2")
        axes[0].plot(rank_summary["rank"], rank_summary.clip_neutral_mean, marker="s", label="CLIP-T neutral")
        axes[0].set_xlabel("LoRA rank"); axes[0].set_ylabel("Cosine similarity"); axes[0].legend(); axes[0].grid(alpha=.25)
        axes[1].plot(rank_summary["rank"], rank_summary.adapter_size_mb, marker="o", color=PALETTE[2])
        axes[1].set_xlabel("LoRA rank"); axes[1].set_ylabel("Adapter size (MiB)"); axes[1].grid(alpha=.25)
        save(fig, out / "rank_ablation")

    if (metrics_dir / "refcount_ablation.csv").is_file():
        ref = pd.read_csv(metrics_dir / "refcount_ablation.csv")
        ref_summary = ref.groupby(["method", "n_refs"], as_index=False).agg(
            images=("image_path", "count"), dino_mean=("dino_ref_mean", "mean"),
            clip_neutral_mean=("clip_t_neutral", "mean"), copy_proxy_mean=("nearest_train_sim", "mean"),
        )
        ref_summary.to_csv(metrics_dir / "refcount_ablation_summary.csv", index=False)
        fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4))
        for index, (method, data) in enumerate(ref_summary.groupby("method")):
            data = data.sort_values("n_refs"); label = {"lora": "LoRA", "ip_adapter_plus": "IP-Adapter Plus"}.get(method, method)
            axes[0].plot(data.n_refs, data.dino_mean, marker="o", label=label, color=PALETTE[index])
            axes[1].plot(data.n_refs, data.clip_neutral_mean, marker="o", label=label, color=PALETTE[index])
        axes[0].set_ylabel("DINOv2 similarity"); axes[1].set_ylabel("CLIP-T neutral")
        for ax in axes: ax.set_xlabel("Reference images (count)"); ax.grid(alpha=.25); ax.legend(frameon=False)
        save(fig, out / "reference_count_ablation")

    if (metrics_dir / "prior_preservation_ablation.csv").is_file():
        prior = pd.read_csv(metrics_dir / "prior_preservation_ablation.csv")
        prior_summary = prior.groupby("prior_preservation", as_index=False).agg(
            images=("image_path", "count"), dino_mean=("dino_ref_mean", "mean"),
            clip_neutral_mean=("clip_t_neutral", "mean"), copy_proxy_mean=("nearest_train_sim", "mean"),
        )
        prior_summary.to_csv(metrics_dir / "prior_preservation_ablation_summary.csv", index=False)
        fig, ax = plt.subplots(figsize=(7.5, 4.8)); x = np.arange(len(prior_summary)); width=.25
        for offset, metric, label in [(-1, "dino_mean", "DINOv2"), (0, "clip_neutral_mean", "CLIP-T neutral"), (1, "copy_proxy_mean", "Copy proxy")]:
            ax.bar(x + offset * width, prior_summary[metric], width, label=label)
        ax.set_xticks(x, prior_summary.prior_preservation); ax.set_xlabel("Prior preservation")
        ax.set_ylabel("Cosine similarity"); ax.legend(frameon=False); ax.grid(axis="y", alpha=.25)
        save(fig, out / "prior_preservation_ablation")

    # Existing measured main-compute table; units remain separate in two panels.
    compute_path = metrics_dir / "main_compute_cost.csv"
    if compute_path.is_file():
        compute = pd.read_csv(compute_path)
        compute = compute.groupby("method", as_index=False).agg(
            training_time_minutes=("training_time_minutes", "mean"),
            personalized_checkpoint_size_mb=("personalized_checkpoint_size_mb", "mean"),
        )
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
        axes[0].bar(compute.method, compute.training_time_minutes, color=PALETTE[:len(compute)])
        axes[0].set_ylabel("Per-subject training time (min)")
        axes[1].bar(compute.method, compute.personalized_checkpoint_size_mb, color=PALETTE[:len(compute)])
        axes[1].set_ylabel("Personalized storage (MiB)"); axes[1].set_yscale("symlog", linthresh=0.01)
        for ax in axes: ax.tick_params(axis="x", rotation=25); ax.grid(axis="y", alpha=.25)
        save(fig, out / "cost_training_time_and_storage")
    print(f"ABLATION_REPORT_COMPLETE output={out}")


if __name__ == "__main__":
    main()
