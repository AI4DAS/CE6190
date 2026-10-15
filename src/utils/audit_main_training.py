#!/usr/bin/env python3
"""Audit main-experiment training completion without trusting marker files."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import yaml


METHODS = {
    "dreambooth": ("dreambooth", "{subject}_db_s800_seed42", 800),
    "lora": ("lora", "{subject}_lora_r16_s800_seed42", 800),
    "textual_inversion": ("textual_inversion", "{subject}_ti_s3000_seed42", 3000),
}


def final_artifact(run_dir: Path, method: str) -> tuple[bool, str]:
    if method == "dreambooth":
        paths = [run_dir / "model_index.json", run_dir / "unet/diffusion_pytorch_model.safetensors"]
    elif method == "lora":
        paths = [run_dir / "pytorch_lora_weights.safetensors"]
    else:
        paths = [run_dir / "learned_embeds.safetensors"]
    return all(path.is_file() and path.stat().st_size > 0 for path in paths), ";".join(map(str, paths))


def completed_steps(log_path: Path, expected: int) -> int:
    if not log_path.is_file():
        return 0
    text = log_path.read_text(errors="replace").replace("\r", "\n")
    values = [int(match.group(1)) for match in re.finditer(r"Steps:\s+100%.*?(\d+)/(\d+)", text)
              if int(match.group(2)) == expected]
    return max(values, default=0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.project_root.resolve()
    subjects = yaml.safe_load((root / "configs/main_subjects.yaml").read_text())["subjects"]
    rows = []
    for subject in subjects:
        for method, (folder, pattern, expected) in METHODS.items():
            run_dir = root / "outputs/training" / folder / pattern.format(subject=subject)
            success = (run_dir / "SUCCESS").is_file()
            failed = (run_dir / "FAILED").is_file()
            marker = "BOTH" if success and failed else "SUCCESS" if success else "FAILED" if failed else "NONE"
            final_exists, artifact_paths = final_artifact(run_dir, method)
            log_path = run_dir / "training.log"
            steps = completed_steps(log_path, expected)
            checkpoints = sorted(path.name for path in run_dir.glob("checkpoint-*") if path.is_dir())
            complete = final_exists and steps >= expected
            if complete and failed and not success:
                actual = "FALSE_FAILED"
            elif complete:
                actual = "COMPLETED"
            elif run_dir.exists() and not final_exists:
                actual = "INCOMPLETE_OR_RUNNING"
            else:
                actual = "MISSING"
            rows.append({
                "subject": subject,
                "method": method,
                "marker_status": marker,
                "final_artifact_exists": final_exists,
                "checkpoint_exists": bool(checkpoints),
                "training_log_exists": log_path.is_file(),
                "training_completed_steps": steps,
                "expected_steps": expected,
                "actual_status": actual,
                "run_dir": str(run_dir.relative_to(root)),
                "final_artifact_paths": artifact_paths,
                "checkpoint_directories": ";".join(checkpoints),
            })
    output = root / "results/system/main_training_completion_audit.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    counts = {}
    for row in rows:
        counts[row["actual_status"]] = counts.get(row["actual_status"], 0) + 1
    print(f"WROTE {output}")
    print("STATUS_COUNTS", counts)


if __name__ == "__main__":
    main()
