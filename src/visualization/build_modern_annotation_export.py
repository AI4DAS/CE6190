#!/usr/bin/env python3
"""Build DreamO/UNO review grids and a blinded annotation ZIP.

This script only reads completed outputs. It does not run model inference.
"""

from __future__ import annotations

import csv
import argparse
import hashlib
import json
import random
import shutil
import textwrap
import zipfile
from pathlib import Path

import yaml
from PIL import Image, ImageDraw, ImageFont


METHOD_LABELS = {
    "ip_adapter_plus": "IP-Adapter Plus (scale=0.3)",
    "dreamo": "DreamO",
    "uno": "UNO",
}
PROMPT_IDS = ["P0", "P1", "P2", "P3", "P4", "P5"]
PROMPT_LABELS = {
    "P0": "Identity", "P1": "Background", "P2": "Context",
    "P3": "Attribute", "P4": "Composition", "P5": "Style",
}
LEAKAGE_IDS = ["P1", "P2", "P4"]
SEEDS = [42, 123, 456, 789]
COMMON_SUBJECTS = [
    "dog2", "cat", "pet_dog1", "bear_plushie", "robot_toy", "teapot",
    "colorful_sneaker", "luggage_backpack1",
]
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


def font(size: int, bold: bool = False):
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    return ImageFont.truetype(f"/usr/share/fonts/truetype/dejavu/{name}", size)


def fit(path: Path, width: int, height: int) -> Image.Image:
    with Image.open(path) as source:
        image = source.convert("RGB")
    image.thumbnail((width, height), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (width, height), "white")
    canvas.paste(image, ((width - image.width) // 2, (height - image.height) // 2))
    return canvas


def center(draw, box, text, text_font, fill="black"):
    left, top, right, bottom = box
    bounds = draw.textbbox((0, 0), text, font=text_font)
    draw.text(
        (left + (right - left - (bounds[2] - bounds[0])) / 2,
         top + (bottom - top - (bounds[3] - bounds[1])) / 2),
        text, font=text_font, fill=fill,
    )


def wrapped(draw, box, text, text_font, width_chars=38):
    left, top, right, bottom = box
    lines = textwrap.wrap(text, width=width_chars, break_long_words=False) or [""]
    line_height = text_font.size + 2
    y = top + max(0, (bottom - top - line_height * len(lines)) // 2)
    for line in lines:
        bounds = draw.textbbox((0, 0), line, font=text_font)
        draw.text((left + (right - left - (bounds[2] - bounds[0])) / 2, y), line, font=text_font, fill="black")
        y += line_height


def save(canvas: Image.Image, stem: Path):
    stem.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(stem.with_suffix(".png"), dpi=(300, 300))
    canvas.save(stem.with_suffix(".pdf"), "PDF", resolution=300.0)


def output_path(root: Path, method: str, dataset: str, subject: str, prompt: str, seed: int) -> Path:
    if method == "ip_adapter_plus":
        return root / f"outputs/main_evaluation/{method}/{dataset}/{subject}/{prompt}/seed_{seed}.png"
    return root / f"outputs/modern_extension/{method}/{dataset}/{subject}/{prompt}/seed_{seed}.png"


def load_records(root: Path, subjects: dict) -> dict:
    records = {}
    for subject, config in subjects.items():
        methods = ["ip_adapter_plus", "dreamo", "uno"]
        for method in methods:
            for prompt in PROMPT_IDS:
                for seed in SEEDS:
                    image = output_path(root, method, config["dataset"], subject, prompt, seed)
                    metadata = image.with_suffix(".json")
                    if not image.is_file() or not metadata.is_file():
                        raise FileNotFoundError(f"Missing pair: {image} / {metadata}")
                    row = json.loads(metadata.read_text())
                    expected = {"method": method, "subject": subject, "prompt_id": prompt, "seed": seed}
                    for key, value in expected.items():
                        if row.get(key) != value:
                            raise RuntimeError(f"Metadata mismatch {metadata}: {key}")
                    if method == "ip_adapter_plus" and row.get("ip_adapter_scale") != 0.3:
                        raise RuntimeError(f"Expected scale 0.3: {metadata}")
                    records[(subject, method, prompt, seed)] = {**row, "image_file": image}
    if len(records) != 576:
        raise RuntimeError(f"Expected 576 annotation records, found {len(records)}")
    return records


def references(root: Path, subject: dict) -> list[Path]:
    paths = sorted(
        path for path in (root / subject["image_dir"]).iterdir()
        if path.suffix.lower() in IMAGE_SUFFIXES
    )
    if len(paths) != subject["expected_images"]:
        raise RuntimeError(f"Reference count mismatch: {subject['image_dir']}")
    return paths


def comparison_grid(root: Path, stem: Path, subject_id: str, subject: dict, seed: int,
                    methods: list[str], prompts: list[str], title: str):
    cell_w, image_h, caption_h = 310, 250, 74
    row_h = image_h + caption_h
    label_w, title_h, header_h, footer_h = 310, 58, 62, 45
    rows = len(methods) + 1
    width = label_w + len(prompts) * cell_w
    height = title_h + header_h + rows * row_h + footer_h
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    center(draw, (0, 0, width, title_h), title, font(23, True))
    for column, prompt in enumerate(prompts):
        left = label_w + column * cell_w
        center(draw, (left, title_h, left + cell_w, title_h + header_h),
               f"{prompt}  {PROMPT_LABELS[prompt]}", font(16, True))
    refs = references(root, subject)
    labels = ["Evaluation references"] + [METHOD_LABELS[m] for m in methods]
    for row, label in enumerate(labels):
        top = title_h + header_h + row * row_h
        center(draw, (0, top, label_w - 8, top + image_h), label, font(16, True))
        for column, prompt in enumerate(prompts):
            left = label_w + column * cell_w
            if row == 0:
                path = refs[column % len(refs)]
                caption = path.name
            else:
                record = records_global[(subject_id, methods[row - 1], prompt, seed)]
                path = record["image_file"]
                caption = record["generation_prompt"]
            canvas.paste(fit(path, cell_w - 12, image_h - 10), (left + 6, top + 5))
            wrapped(draw, (left + 4, top + image_h, left + cell_w - 4, top + row_h), caption, font(12))
    footer = title_h + header_h + rows * row_h
    method_text = "The method uses" if len(methods) == 1 else "All displayed methods use"
    center(draw, (0, footer, width, height),
           f"{method_text} one fixed conditioning reference (00); compare against the complete reference row/set.",
           font(12), fill="#333333")
    save(canvas, stem)


def copy_export(root: Path, export: Path, records: dict, subjects: dict, figure_root: Path):
    export.mkdir(parents=True, exist_ok=False)
    (export / "annotation_images").mkdir()
    (export / "references").mkdir()
    (export / "unblinded_review_grids").mkdir()

    for subject_id, subject in subjects.items():
        target = export / "references" / subject_id
        target.mkdir()
        for source in references(root, subject):
            shutil.copy2(source, target / source.name)

    ordered = list(records.values())
    random.Random(42).shuffle(ordered)
    form_rows, key_rows = [], []
    for index, row in enumerate(ordered, start=1):
        image_id = f"IMG{index:04d}"
        suffix = Path(row["image_file"]).suffix.lower()
        exported = export / "annotation_images" / f"{image_id}{suffix}"
        shutil.copy2(row["image_file"], exported)
        form_rows.append({
            "image_id": image_id,
            "subject": row["subject"],
            "prompt_id": row["prompt_id"],
            "prompt_category": row["prompt_category"],
            "seed": row["seed"],
            "evaluation_prompt": row["evaluation_prompt"],
            "background_leakage": "", "pose_copying": "", "composition_copying": "",
            "matched_reference": "", "notes": "", "annotator_id": "", "blind": "yes",
        })
        key_rows.append({
            "image_id": image_id, "method": row["method"], "dataset": row["dataset"],
            "subject": row["subject"], "prompt_id": row["prompt_id"], "seed": row["seed"],
            "original_image_path": str(Path(row["image_file"]).relative_to(root)),
            "checkpoint": row.get("checkpoint", ""),
            "ip_adapter_scale": row.get("ip_adapter_scale", ""),
            "reference_count": row.get("reference_count", len(row.get("reference_images", []))),
        })
    for filename, rows in [("annotation_form_blinded.csv", form_rows), ("annotation_key_DO_NOT_OPEN_BEFORE_RATING.csv", key_rows)]:
        with (export / filename).open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)

    shutil.copy2(root / "results/manual_annotation_scoring_rules.md", export / "SCORING_RULES.md")
    for subdir in [
        "dreamo", "uno", "reference_conditioned_comparison",
        "background_leakage_review", "ip_adapter_scale03_background_review",
    ]:
        source = figure_root / subdir
        target = export / "unblinded_review_grids" / subdir
        target.mkdir(parents=True)
        for path in sorted(source.glob("*.png")):
            shutil.copy2(path, target / path.name)

    readme = """# Manual Annotation Export: IP-Adapter scale 0.3 + DreamO + UNO

This package contains 336 generated images: 192 IP-Adapter Plus scale-0.3
outputs across eight subjects, 72 DreamO outputs across three subjects, and 72
UNO outputs across three subjects. It also contains all original evaluation
references. Prior-preservation class images are excluded.

## Blind rating workflow

1. Do not open `annotation_key_DO_NOT_OPEN_BEFORE_RATING.csv` until ratings are complete.
2. Read `SCORING_RULES.md`.
3. For each row in `annotation_form_blinded.csv`, open the matching file under
   `annotation_images/` and compare it with every image under
   `references/{subject}/`.
4. Enter independent integer scores 0, 1, or 2 for background leakage, pose
   copying, and composition copying. Record the closest reference filename in
   `matched_reference` for nonzero scores.
5. Use P1, P2, and P4 as the primary background-leakage probes. Retain all six
   prompt categories for pose/composition analysis.
6. After completing the ratings, use the key to restore method identities.

The files under `unblinded_review_grids/` are report visualizations and
convenient review sheets. Do not consult them during a genuinely blind pass,
because their rows expose method names.

DreamO and UNO differ in backbone, resolution, inference steps, and guidance;
their manual scores are an extension comparison, not a controlled SD1.5
method ranking.
"""
    (export / "README.md").write_text(readme)


def zip_directory(source: Path, destination: Path):
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in sorted(source.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(source.parent))
    hasher = hashlib.sha256()
    with destination.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
    digest = hasher.hexdigest()
    destination.with_suffix(destination.suffix + ".sha256").write_text(f"{digest}  {destination.name}\n")


def main():
    global records_global
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--figures-only", action="store_true",
        help="Build the validated PNG/PDF grids without creating an annotation export.",
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    subjects = yaml.safe_load((root / "configs/main_subjects.yaml").read_text())["subjects"]
    records_global = load_records(root, subjects)
    figure_root = root / "results/figures/modern_extension"

    for subject_id in COMMON_SUBJECTS:
        subject = subjects[subject_id]
        for seed in SEEDS:
            for method in ["dreamo", "uno"]:
                comparison_grid(
                    root, figure_root / method / f"{subject_id}_{method}_seed{seed}",
                    subject_id, subject, seed, [method], PROMPT_IDS,
                    f"{subject_id}: {METHOD_LABELS[method]} (Seed={seed})",
                )
            comparison_grid(
                root, figure_root / "reference_conditioned_comparison" / f"{subject_id}_ip_dreamo_uno_seed{seed}",
                subject_id, subject, seed, ["ip_adapter_plus", "dreamo", "uno"], PROMPT_IDS,
                f"{subject_id}: Reference-Conditioned Methods (Seed={seed})",
            )
            comparison_grid(
                root, figure_root / "background_leakage_review" / f"{subject_id}_leakage_ip_dreamo_uno_seed{seed}",
                subject_id, subject, seed, ["ip_adapter_plus", "dreamo", "uno"], LEAKAGE_IDS,
                f"{subject_id}: Background-Leakage Review (Seed={seed})",
            )

    for subject_id, subject in subjects.items():
        for seed in SEEDS:
            comparison_grid(
                root,
                figure_root / "ip_adapter_scale03_background_review" /
                f"{subject_id}_ip_adapter_scale03_leakage_seed{seed}",
                subject_id, subject, seed, ["ip_adapter_plus"], LEAKAGE_IDS,
                f"{subject_id}: IP-Adapter Plus scale=0.3 Leakage Review (Seed={seed})",
            )

    if args.figures_only:
        print(f"MODERN_VISUALIZATION_COMPLETE records={len(records_global)}")
        return

    export = root / "results/exports/manual_annotation_scale03_dreamo_uno_all_subjects_20261013"
    archive = export.with_suffix(".zip")
    if export.exists() or archive.exists():
        raise FileExistsError(f"Refusing to overwrite existing export: {export} / {archive}")
    copy_export(root, export, records_global, subjects, figure_root)
    zip_directory(export, archive)
    print(f"MODERN_VISUALIZATION_EXPORT_COMPLETE records={len(records_global)} archive={archive}")


records_global = {}
if __name__ == "__main__":
    main()
