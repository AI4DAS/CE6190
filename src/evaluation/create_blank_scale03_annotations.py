#!/usr/bin/env python3
"""Create a blank, deterministic annotation form for Task A scale-0.3 outputs."""

from pathlib import Path
import csv
import json


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    source = root / "outputs/experiments/task_a/full_plus_s0.3"
    metadata = sorted(source.glob("**/P*/seed_*.json"))
    if len(metadata) != 192:
        raise RuntimeError(f"Expected 192 metadata files, found {len(metadata)}")
    rows = []
    for path in metadata:
        record = json.loads(path.read_text())
        rows.append({
            "object": record["subject"], "seed": record["seed"],
            "method": "IP-Adapter Plus (s=0.3)", "prompt_id": record["prompt_id"],
            "prompt_category": record["prompt_category"], "background_leakage": "",
            "pose_copying": "", "composition_copying": "", "identity_loss": "",
            "artifact_severity": "", "annotator": "", "blind": "",
        })
    destination = root / "results/metrics/manual_overfitting_annotation_ip_scale03_blank.csv"
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    print(f"BLANK_ANNOTATION_COMPLETE rows={len(rows)} path={destination}")


if __name__ == "__main__":
    main()
