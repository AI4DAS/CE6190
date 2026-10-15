#!/usr/bin/env python3
"""Print a main-subject config as a fixed newline-delimited record."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml


FIELDS = [
    "dataset",
    "category",
    "class_name",
    "image_dir",
    "expected_images",
    "instance_prompt",
    "class_prompt",
    "class_dir",
    "placeholder_token",
    "initializer_token",
    "completed_pilot",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--subject", required=True)
    args = parser.parse_args()
    subjects = yaml.safe_load(args.config.read_text())["subjects"]
    if args.subject not in subjects:
        raise KeyError(f"Unknown subject: {args.subject}")
    record = subjects[args.subject]
    for field in FIELDS:
        value = record[field]
        if isinstance(value, bool):
            value = "true" if value else "false"
        rendered = str(value)
        if "\n" in rendered:
            raise ValueError(f"Newline is not allowed in {field}")
        print(rendered)


if __name__ == "__main__":
    main()
