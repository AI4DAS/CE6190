#!/usr/bin/env python3
"""Validate counts, PNG integrity, metadata, hashes, and reproducibility evidence."""

from __future__ import annotations

import hashlib
import json
import subprocess
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
from PIL import Image


EXPECTED = {
    "task_a": 972,
    "task_c_steps": 1296,
    "task_b_rank": 180,
    "task_e_refcount": 144,
    "task_f_prior": 48,
    "task_d_face_corruption": 96,
}
REQUIRED_METADATA = {
    "experiment", "variant", "method", "subject", "prompt_id", "prompt_category",
    "seed", "scale", "rank", "step", "n_refs", "ref_files_used", "guidance_scale", "dtype",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    rows, duplicate_rows = [], []
    for directory, expected in EXPECTED.items():
        source = root / "outputs/experiments" / directory
        images = sorted(source.glob("**/P*/seed_*.png")) if source.is_dir() else []
        metadata = sorted(source.glob("**/P*/seed_*.json")) if source.is_dir() else []
        corrupt, schema_errors, hashes = [], [], defaultdict(list)
        for image in images:
            try:
                with Image.open(image) as value:
                    value.verify()
            except Exception as error:
                corrupt.append(f"{image.relative_to(root)}: {error}")
            hashes[sha256(image)].append(str(image.relative_to(root)))
        for path in metadata:
            record = json.loads(path.read_text())
            missing = REQUIRED_METADATA - set(record)
            if missing:
                schema_errors.append(f"{path.relative_to(root)}: {sorted(missing)}")
        duplicates = [paths for paths in hashes.values() if len(paths) > 1]
        for paths in duplicates:
            duplicate_rows.append({"experiment_directory": directory, "copies": len(paths), "paths": " | ".join(paths)})
        rows.append({
            "experiment_directory": directory, "expected_images": expected,
            "actual_png": len(images), "actual_json": len(metadata),
            "corrupt_png": len(corrupt), "metadata_schema_errors": len(schema_errors),
            "duplicate_hash_groups": len(duplicates),
            "count_status": "PASS" if len(images) == expected and len(metadata) == expected else "FAIL",
            "integrity_status": "PASS" if not corrupt and not schema_errors else "FAIL",
        })
    validation = pd.DataFrame(rows)
    system = root / "results/system"; system.mkdir(parents=True, exist_ok=True)
    validation.to_csv(system / "ablation_validation_counts.csv", index=False)
    pd.DataFrame(duplicate_rows, columns=["experiment_directory", "copies", "paths"]).to_csv(
        system / "ablation_duplicate_hashes.csv", index=False
    )

    snapshot = root / "results/system/pre_ablation_main_evaluation_sha256.txt"
    hash_result = subprocess.run(["sha256sum", "-c", str(snapshot)], cwd=root, text=True, capture_output=True)
    hash_lines = [line for line in hash_result.stdout.splitlines() if line]
    hash_failures = [line for line in hash_lines if not line.endswith(": OK")]
    master = root / "results/metrics/all_runs_per_image.csv"
    metric_status = "NOT_RUN"
    metric_rows = 0
    if master.is_file():
        metrics = pd.read_csv(master)
        metric_rows = len(metrics)
        numeric = ["dino_ref_mean", "dino_ref_max", "nearest_train_sim", "clip_t_raw", "clip_t_neutral"]
        metric_status = "PASS" if not metrics[numeric].isna().any().any() and np_isfinite(metrics[numeric]) else "FAIL"

    report = system / "ablation_validation_report.md"
    headers = list(validation.columns)
    markdown = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    markdown.extend("| " + " | ".join(str(row[column]) for column in headers) + " |" for _, row in validation.iterrows())
    lines = ["# Ablation validation report", "", *markdown, "",
             f"- Main-run hash snapshot entries checked: {len(hash_lines)}",
             f"- Main-run hash check exit code: {hash_result.returncode}",
             f"- Main-run hash failures: {len(hash_failures)}",
             f"- Master metric rows: {metric_rows}", f"- Master finite-metric status: {metric_status}", "",
             "Exact duplicate hashes are listed separately. Base/no-personalization and scale-zero variants can "
             "produce expected cross-subject duplicates when two subjects share the same class prompt and seed; "
             "these are reported rather than silently removed."]
    report.write_text("\n".join(lines) + "\n")
    print(f"VALIDATION_COMPLETE rows={len(validation)} main_hash_exit={hash_result.returncode} metric_status={metric_status}")


def np_isfinite(frame: pd.DataFrame) -> bool:
    values = frame.to_numpy(dtype=float)
    return bool((values == values).all() and (abs(values) != float("inf")).all())


if __name__ == "__main__":
    main()
