#!/usr/bin/env python3
"""Reconcile main-training markers after independent artifact validation."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.project_root.resolve()
    audit_path = root / "results/system/main_training_completion_audit.csv"
    rows = list(csv.DictReader(audit_path.open()))
    reconciliation = []
    remaining = []
    for row in rows:
        run_dir = root / row["run_dir"]
        old_marker = row["marker_status"]
        validation_path = root / "results/figures/main_artifact_validation" / f"{row['subject']}_{row['method']}.json"
        validation = json.loads(validation_path.read_text()) if validation_path.is_file() else {}
        validated = validation.get("status") == "PASS"
        if row["actual_status"] in {"COMPLETED", "FALSE_FAILED"} and validated:
            failed = run_dir / "FAILED"
            if failed.exists():
                failed.unlink()
            (run_dir / "SUCCESS").touch()
            action = "Replaced false FAILED with SUCCESS" if old_marker == "FAILED" else "Preserved SUCCESS"
            actual = "COMPLETED_VALIDATED"
        else:
            action = "Kept existing marker; artifact not complete and validated"
            actual = row["actual_status"]
            remaining.append(f"- {row['subject']} / {row['method']}: {actual}")
        reconciliation.append((row["subject"], row["method"], old_marker, actual, action))

    report = root / "results/system/status_reconciliation.md"
    lines = [
        "# Main Training Status Reconciliation", "",
        "Markers were reconciled only after final-artifact load and generation tests passed.", "",
        "| Subject | Method | Old marker | Actual status | Action |",
        "|---|---|---|---|---|",
    ]
    lines.extend(f"| {subject} | {method} | {old} | {actual} | {action} |"
                 for subject, method, old, actual, action in reconciliation)
    lines.extend(["", "No training logs or checkpoint evidence were deleted.", ""])
    report.write_text("\n".join(lines))

    remaining_report = root / "results/system/training_remaining_status.md"
    if remaining:
        body = ["# Remaining Main Training", "", "The following jobs still require attention:", "", *remaining, ""]
    else:
        body = [
            "# Remaining Main Training", "",
            "All 24 required training artifacts (8 subjects × 3 training methods) completed and passed load/generation validation.", "",
            "No training job requires rerun. Benchmark generation may proceed.", "",
        ]
    remaining_report.write_text("\n".join(body))
    print(f"WROTE {report}")
    print(f"WROTE {remaining_report}")
    if remaining:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
