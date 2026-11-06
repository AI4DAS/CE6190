#!/usr/bin/env python3
"""Build a report table spanning the SD1.5 benchmark and FLUX extensions."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


LABELS = {
    "base_model": "Base SD1.5",
    "dreambooth": "DreamBooth",
    "lora": "DreamBooth-LoRA",
    "textual_inversion": "Textual Inversion",
    "ip_adapter_plus": "IP-Adapter Plus (s=0.3)",
    "dreamo": "DreamO",
    "uno": "UNO",
}
ORDER = list(LABELS)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.project_root.resolve()
    metrics = root / "results/metrics"

    dino = pd.concat([
        pd.read_csv(metrics / "main_subject_fidelity.csv").rename(columns={"mean_reference_similarity": "dino"}),
        pd.read_csv(metrics / "modern_subject_fidelity.csv").rename(columns={"mean_reference_similarity": "dino"}),
    ]).groupby("method")["dino"].agg(["mean", "std"])
    clip = pd.concat([
        pd.read_csv(metrics / "main_prompt_alignment.csv").rename(columns={"clip_similarity": "clip"}),
        pd.read_csv(metrics / "modern_prompt_alignment.csv").rename(columns={"clip_similarity": "clip"}),
    ]).groupby("method")["clip"].agg(["mean", "std"])

    classical_cost = pd.read_csv(metrics / "main_compute_cost.csv").groupby("method", as_index=True).agg(
        training_time_min=("training_time_minutes", "mean"),
        inference_latency_sec=("inference_latency_mean_sec", "mean"),
        peak_inference_vram_gib=("peak_inference_vram_gb", "max"),
        personalized_storage_mib=("personalized_checkpoint_size_mb", "mean"),
        shared_extension_storage_mib=("shared_adapter_size_mb", "mean"),
    )
    modern_cost = pd.read_csv(metrics / "modern_compute_cost.csv").set_index("method")

    rows = []
    for method in ORDER:
        modern = method in ("dreamo", "uno")
        if modern:
            cost = modern_cost.loc[method]
            training_time = float(cost["subject_specific_training_time_sec"]) / 60
            latency = cost["inference_latency_mean_sec"]
            vram = cost["peak_inference_vram_gib"]
            personalized = cost["per_subject_checkpoint_size_mib"]
            shared = cost["shared_extension_weights_mib"]
        else:
            cost = classical_cost.loc[method]
            training_time = cost["training_time_min"]
            latency = cost["inference_latency_sec"]
            vram = cost["peak_inference_vram_gib"]
            personalized = cost["personalized_storage_mib"]
            shared = cost["shared_extension_storage_mib"]
        rows.append({
            "method": LABELS[method],
            "comparison_group": "FLUX modern extension" if modern else ("SD1.5 control" if method == "base_model" else "SD1.5 controlled benchmark"),
            "images": 192,
            "dino_mean": dino.loc[method, "mean"], "dino_std": dino.loc[method, "std"],
            "clip_mean": clip.loc[method, "mean"], "clip_std": clip.loc[method, "std"],
            "subject_specific_training_required": method in ("dreambooth", "lora", "textual_inversion"),
            "training_time_mean_min": training_time,
            "inference_latency_mean_sec": latency,
            "peak_inference_vram_gib": vram,
            "personalized_storage_mib": personalized,
            "shared_adapter_or_extension_storage_mib": shared,
            "comparison_note": "Different FLUX backbone/recipe; descriptive only" if modern else "Shared SD1.5 protocol",
        })
    destination = root / "results/report_tables/all_methods_comparison.csv"
    pd.DataFrame(rows).to_csv(destination, index=False)
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
