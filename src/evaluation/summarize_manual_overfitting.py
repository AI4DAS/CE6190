#!/usr/bin/env python3
"""Validate and aggregate completed manual overfitting annotations."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


RATING_COLUMNS = [
    "background_leakage",
    "pose_copying",
    "composition_copying",
]
BASE_IDENTIFIER_COLUMNS = ["method", "prompt_id", "prompt_category", "seed"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing output file explicitly.",
    )
    return parser.parse_args()


def validate(df: pd.DataFrame) -> None:
    identifier_columns = [
        column for column in ["method", "dataset", "subject", "prompt_id", "prompt_category", "seed"]
        if column in df.columns
    ]
    required = BASE_IDENTIFIER_COLUMNS + RATING_COLUMNS
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")

    duplicate_count = int(df.duplicated(identifier_columns).sum())
    if duplicate_count:
        raise ValueError(f"Found {duplicate_count} duplicate annotation keys")

    incomplete = df[RATING_COLUMNS].isna().any(axis=1)
    if incomplete.any():
        raise ValueError(
            f"Annotations are incomplete: {int(incomplete.sum())} of "
            f"{len(df)} rows still contain blank ratings"
        )

    for column in RATING_COLUMNS:
        numeric = pd.to_numeric(df[column], errors="coerce")
        invalid = numeric.isna() | ~numeric.isin([0, 1, 2])
        if invalid.any():
            values = sorted({str(value) for value in df.loc[invalid, column]})
            raise ValueError(
                f"Column {column} contains ratings outside 0, 1, 2: {values}"
            )
        df[column] = numeric.astype(int)


def main() -> None:
    args = parse_args()
    if not args.input.is_file():
        raise FileNotFoundError(args.input)
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(
            f"Refusing to overwrite {args.output}; pass --overwrite explicitly"
        )

    annotations = pd.read_csv(args.input)
    validate(annotations)

    summary = (
        annotations.groupby("method", sort=True)[RATING_COLUMNS]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    summary.columns = [
        "method"
        if column[0] == "method"
        else f"{column[0]}_{column[1]}"
        for column in summary.columns
    ]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output, index=False)
    print(f"Validated {len(annotations)} annotations")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
