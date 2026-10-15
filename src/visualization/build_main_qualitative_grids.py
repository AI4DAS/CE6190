#!/usr/bin/env python3
"""Build seed-specific main-experiment qualitative review figures.

This script only reads existing images and sidecar JSON metadata.  It never
invokes an inference pipeline.  Column headers show semantic prompt classes;
the caption under each generated image shows the exact model input prompt.
"""

from __future__ import annotations

import argparse
import csv
import json
import textwrap
from pathlib import Path

import yaml
from PIL import Image, ImageDraw, ImageFont


METHODS = ["base_model", "dreambooth", "lora", "textual_inversion", "ip_adapter_plus"]
METHOD_LABELS = {
    "base_model": "Base SD1.5\n(no personalization)",
    "dreambooth": "DreamBooth",
    "lora": "DreamBooth-LoRA",
    "textual_inversion": "Textual Inversion",
    "ip_adapter_plus": "IP-Adapter Plus",
}
PROMPT_IDS = ["P0", "P1", "P2", "P3", "P4", "P5"]
PROMPT_LABELS = {
    "P0": "Identity",
    "P1": "Background change",
    "P2": "Context change",
    "P3": "Attribute change",
    "P4": "Composition change",
    "P5": "Style change",
}
LEAKAGE_PROMPT_IDS = ["P1", "P2", "P4"]
SEEDS = [42, 123, 456, 789]
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def load_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    path = Path("/usr/share/fonts/truetype/dejavu") / name
    return ImageFont.truetype(str(path), size=size)


def fit_image(path: Path, width: int, height: int) -> Image.Image:
    with Image.open(path) as source:
        image = source.convert("RGB")
    image.thumbnail((width, height), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (width, height), "white")
    canvas.paste(image, ((width - image.width) // 2, (height - image.height) // 2))
    return canvas


def centered_text(draw: ImageDraw.ImageDraw, box, text: str, font, fill="black") -> None:
    left, top, right, bottom = box
    bounds = draw.textbbox((0, 0), text, font=font)
    width = bounds[2] - bounds[0]
    height = bounds[3] - bounds[1]
    draw.text(
        (left + (right - left - width) / 2, top + (bottom - top - height) / 2),
        text,
        font=font,
        fill=fill,
    )


def wrapped_centered_text(
    draw: ImageDraw.ImageDraw,
    box,
    text: str,
    font,
    width_chars: int = 34,
) -> None:
    lines = textwrap.wrap(text, width=width_chars, break_long_words=False) or [""]
    left, top, right, bottom = box
    line_height = font.size + 3
    start_y = top + max(0, ((bottom - top) - line_height * len(lines)) // 2)
    for index, line in enumerate(lines):
        bounds = draw.textbbox((0, 0), line, font=font)
        line_width = bounds[2] - bounds[0]
        draw.text(
            (left + ((right - left) - line_width) / 2, start_y + index * line_height),
            line,
            font=font,
            fill="black",
        )


def metadata_path(root: Path, method: str, dataset: str, subject: str, prompt_id: str, seed: int) -> Path:
    return root / "outputs/main_evaluation" / method / dataset / subject / prompt_id / f"seed_{seed}.json"


def image_path(root: Path, method: str, dataset: str, subject: str, prompt_id: str, seed: int) -> Path:
    return root / "outputs/main_evaluation" / method / dataset / subject / prompt_id / f"seed_{seed}.png"


def load_and_validate_records(root: Path, subjects: dict) -> dict:
    records = {}
    missing = []
    errors = []
    for subject, subject_config in subjects.items():
        dataset = subject_config["dataset"]
        for seed in SEEDS:
            for method in METHODS:
                for prompt_id in PROMPT_IDS:
                    png = image_path(root, method, dataset, subject, prompt_id, seed)
                    sidecar = metadata_path(root, method, dataset, subject, prompt_id, seed)
                    if not png.is_file():
                        missing.append(str(png))
                        continue
                    if not sidecar.is_file():
                        missing.append(str(sidecar))
                        continue
                    record = json.loads(sidecar.read_text())
                    expected = {
                        "method": method,
                        "dataset": dataset,
                        "subject": subject,
                        "prompt_id": prompt_id,
                        "seed": seed,
                    }
                    for key, value in expected.items():
                        if record.get(key) != value:
                            errors.append(f"{sidecar}: {key}={record.get(key)!r}, expected {value!r}")
                    expected_category = PROMPT_LABELS[prompt_id].lower().replace(" ", "_")
                    if record.get("prompt_category") != expected_category:
                        errors.append(
                            f"{sidecar}: prompt_category={record.get('prompt_category')!r}, "
                            f"expected {expected_category!r}"
                        )
                    records[(subject, seed, method, prompt_id)] = {**record, "image_file": png}
    if missing:
        print("Missing required existing artifacts:")
        for path in missing:
            print(path)
        raise FileNotFoundError(f"Found {len(missing)} missing images/metadata files; no figures were created")
    if errors:
        print("Metadata validation errors:")
        for error in errors:
            print(error)
        raise ValueError(f"Found {len(errors)} metadata mismatches; no figures were created")
    return records


def reference_paths(root: Path, subject_config: dict) -> list[Path]:
    directory = root / subject_config["image_dir"]
    paths = sorted(path for path in directory.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES)
    expected = int(subject_config["expected_images"])
    if len(paths) != expected:
        raise RuntimeError(f"Expected {expected} references in {directory}, found {len(paths)}")
    return paths


def save_canvas(canvas: Image.Image, stem: Path, overwrite: bool = False) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    png = stem.with_suffix(".png")
    pdf = stem.with_suffix(".pdf")
    if (png.exists() or pdf.exists()) and not overwrite:
        raise FileExistsError(f"Refusing to overwrite an existing figure: {stem}")
    canvas.save(png, dpi=(300, 300))
    canvas.save(pdf, "PDF", resolution=300.0)


def build_comparison_grid(
    root: Path,
    output_stem: Path,
    subject: str,
    subject_config: dict,
    seed: int,
    records: dict,
    overwrite: bool = False,
) -> None:
    cell_width, image_height, caption_height = 286, 238, 78
    row_height = image_height + caption_height
    label_width, header_height, title_height, footer_height = 205, 74, 58, 44
    width = label_width + len(PROMPT_IDS) * cell_width
    height = title_height + header_height + (len(METHODS) + 1) * row_height + footer_height
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    title_font = load_font(24, bold=True)
    header_font = load_font(17, bold=True)
    row_font = load_font(18, bold=True)
    caption_font = load_font(13)
    note_font = load_font(13)

    centered_text(draw, (0, 0, width, title_height), f"{subject}: Five-Method Comparison (Seed={seed})", title_font)
    for column, prompt_id in enumerate(PROMPT_IDS):
        left = label_width + column * cell_width
        centered_text(
            draw,
            (left, title_height, left + cell_width, title_height + header_height),
            f"{prompt_id}  {PROMPT_LABELS[prompt_id]}",
            header_font,
        )

    refs = reference_paths(root, subject_config)
    row_labels = ["Reference"] + [METHOD_LABELS[method] for method in METHODS]
    for row, label in enumerate(row_labels):
        top = title_height + header_height + row * row_height
        centered_text(draw, (0, top, label_width - 8, top + image_height), label, row_font)
        for column, prompt_id in enumerate(PROMPT_IDS):
            left = label_width + column * cell_width
            if row == 0:
                source = refs[column % len(refs)]
                caption = source.name
            else:
                method = METHODS[row - 1]
                record = records[(subject, seed, method, prompt_id)]
                source = record["image_file"]
                caption = record["generation_prompt"]
            canvas.paste(fit_image(source, cell_width - 12, image_height - 10), (left + 6, top + 5))
            wrapped_centered_text(
                draw,
                (left + 5, top + image_height, left + cell_width - 5, top + row_height),
                caption,
                caption_font,
            )

    footer_top = title_height + header_height + (len(METHODS) + 1) * row_height
    centered_text(
        draw,
        (0, footer_top, width, height),
        "Headers = semantic prompt categories; captions = exact generation prompts from image metadata.",
        note_font,
        fill="#333333",
    )
    save_canvas(canvas, output_stem, overwrite=overwrite)


def build_leakage_grid(
    root: Path,
    output_stem: Path,
    subject: str,
    subject_config: dict,
    seed: int,
    records: dict,
    overwrite: bool = False,
) -> None:
    cell_width, image_height, caption_height = 370, 292, 80
    row_height = image_height + caption_height
    label_width, title_height, reference_height, header_height, footer_height = 210, 58, 225, 70, 52
    columns = len(LEAKAGE_PROMPT_IDS)
    width = label_width + columns * cell_width
    height = title_height + reference_height + header_height + len(METHODS) * row_height + footer_height
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    title_font = load_font(23, bold=True)
    header_font = load_font(17, bold=True)
    row_font = load_font(18, bold=True)
    caption_font = load_font(14)
    note_font = load_font(13)

    centered_text(
        draw,
        (0, 0, width, title_height),
        f"{subject}: Background-Leakage Human Review (Seed={seed})",
        title_font,
    )
    centered_text(draw, (0, title_height, label_width, title_height + reference_height), "Training\nreferences", row_font)
    refs = reference_paths(root, subject_config)
    strip_left = label_width
    strip_width = width - strip_left
    ref_width = strip_width // len(refs)
    for index, ref in enumerate(refs):
        left = strip_left + index * ref_width
        canvas.paste(fit_image(ref, ref_width - 8, reference_height - 33), (left + 4, title_height + 4))
        centered_text(
            draw,
            (left, title_height + reference_height - 29, left + ref_width, title_height + reference_height),
            ref.name,
            note_font,
        )

    header_top = title_height + reference_height
    for column, prompt_id in enumerate(LEAKAGE_PROMPT_IDS):
        left = label_width + column * cell_width
        centered_text(
            draw,
            (left, header_top, left + cell_width, header_top + header_height),
            f"{prompt_id}  {PROMPT_LABELS[prompt_id]}",
            header_font,
        )

    content_top = header_top + header_height
    for row, method in enumerate(METHODS):
        top = content_top + row * row_height
        centered_text(draw, (0, top, label_width - 8, top + image_height), METHOD_LABELS[method], row_font)
        for column, prompt_id in enumerate(LEAKAGE_PROMPT_IDS):
            left = label_width + column * cell_width
            record = records[(subject, seed, method, prompt_id)]
            canvas.paste(fit_image(record["image_file"], cell_width - 12, image_height - 10), (left + 6, top + 5))
            wrapped_centered_text(
                draw,
                (left + 6, top + image_height, left + cell_width - 6, top + row_height),
                record["generation_prompt"],
                caption_font,
                width_chars=43,
            )

    footer_top = content_top + len(METHODS) * row_height
    centered_text(
        draw,
        (0, footer_top, width, height),
        "For human inspection only: compare generated backgrounds/compositions with the original references; no leakage label is inferred.",
        note_font,
        fill="#333333",
    )
    save_canvas(canvas, output_stem, overwrite=overwrite)


def write_prompt_catalog(root: Path, subjects: dict, records: dict, overwrite: bool = False) -> None:
    rows = []
    for subject, subject_config in subjects.items():
        for prompt_id in PROMPT_IDS:
            for method in METHODS:
                seed_records = [records[(subject, seed, method, prompt_id)] for seed in SEEDS]
                generation_prompts = {record["generation_prompt"] for record in seed_records}
                evaluation_prompts = {record["evaluation_prompt"] for record in seed_records}
                if len(generation_prompts) != 1 or len(evaluation_prompts) != 1:
                    raise RuntimeError(f"Prompt text changes across seeds: {subject}/{method}/{prompt_id}")
                rows.append({
                    "dataset": subject_config["dataset"],
                    "subject": subject,
                    "class_name": subject_config["class_name"],
                    "prompt_id": prompt_id,
                    "prompt_category": seed_records[0]["prompt_category"],
                    "method": method,
                    "method_label": METHOD_LABELS[method],
                    "generation_prompt": next(iter(generation_prompts)),
                    "evaluation_prompt": next(iter(evaluation_prompts)),
                    "verified_seeds": "42;123;456;789",
                })

    output_dir = root / "results/report_tables"
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "main_generation_prompt_catalog.csv"
    md_path = output_dir / "main_generation_prompt_catalog.md"
    if (csv_path.exists() or md_path.exists()) and not overwrite:
        raise FileExistsError("Refusing to overwrite an existing main prompt catalog")
    fieldnames = list(rows[0])
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# Main Experiment Generation Prompt Catalog",
        "",
        "The figure column headers are semantic prompt categories. The `generation_prompt` below is the exact text passed to each method, recovered from the validated image metadata. The clean `evaluation_prompt` is used for CLIP scoring and intentionally excludes `sks` and learned Textual Inversion tokens.",
        "",
    ]
    for subject in subjects:
        lines.extend([f"## {subject}", ""])
        for prompt_id in PROMPT_IDS:
            selected = [row for row in rows if row["subject"] == subject and row["prompt_id"] == prompt_id]
            lines.extend([
                f"### {prompt_id}: {selected[0]['prompt_category']}",
                "",
                "| Method | Exact generation prompt | Clean evaluation prompt |",
                "|---|---|---|",
            ])
            for row in selected:
                lines.append(
                    f"| {row['method_label']} | `{row['generation_prompt']}` | `{row['evaluation_prompt']}` |"
                )
            lines.append("")
    md_path.write_text("\n".join(lines) + "\n")


def write_wide_prompt_and_conditioning_catalogs(
    root: Path, subjects: dict, records: dict, overwrite: bool = False
) -> None:
    """Write compact tables that put method-specific prompts side by side."""
    output_dir = root / "results/report_tables"
    output_dir.mkdir(parents=True, exist_ok=True)
    prompt_path = output_dir / "main_generation_prompt_catalog_wide.csv"
    conditioning_path = output_dir / "main_subject_conditioning_catalog.csv"
    if (prompt_path.exists() or conditioning_path.exists()) and not overwrite:
        raise FileExistsError("Refusing to overwrite an existing wide prompt/conditioning catalog")

    prompt_rows = []
    for subject, subject_config in subjects.items():
        for prompt_id in PROMPT_IDS:
            method_records = {
                method: records[(subject, SEEDS[0], method, prompt_id)] for method in METHODS
            }
            clean_prompts = {record["evaluation_prompt"] for record in method_records.values()}
            if len(clean_prompts) != 1:
                raise RuntimeError(f"Evaluation prompts differ by method: {subject}/{prompt_id}")
            prompt_rows.append({
                "dataset": subject_config["dataset"],
                "subject": subject,
                "class_name": subject_config["class_name"],
                "prompt_id": prompt_id,
                "prompt_category": method_records["dreambooth"]["prompt_category"],
                "dreambooth_generation_prompt": method_records["dreambooth"]["generation_prompt"],
                "lora_generation_prompt": method_records["lora"]["generation_prompt"],
                "textual_inversion_generation_prompt": method_records["textual_inversion"]["generation_prompt"],
                "ip_adapter_plus_generation_prompt": method_records["ip_adapter_plus"]["generation_prompt"],
                "clean_evaluation_prompt": next(iter(clean_prompts)),
            })
    with prompt_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(prompt_rows[0]))
        writer.writeheader()
        writer.writerows(prompt_rows)

    conditioning_rows = []
    for subject, subject_config in subjects.items():
        conditioning_rows.append({
            "dataset": subject_config["dataset"],
            "subject": subject,
            "class_name": subject_config["class_name"],
            "dreambooth_lora_instance_prompt": subject_config["instance_prompt"],
            "dreambooth_lora_identifier": subject_config["instance_token"],
            "prior_preservation_class_prompt": subject_config["class_prompt"],
            "textual_inversion_placeholder_token": subject_config["placeholder_token"],
            "textual_inversion_initializer_token": subject_config["initializer_token"],
            "ip_adapter_subject_training": "none",
        })
    with conditioning_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(conditioning_rows[0]))
        writer.writeheader()
        writer.writerows(conditioning_rows)


def write_manifests(root: Path, subjects: dict, records: dict, overwrite: bool = False) -> None:
    per_subject_dir = root / "results/figures/main_experiment/per_subject"
    leakage_dir = root / "results/figures/main_experiment/background_leakage_review"
    comparison_rows = []
    leakage_rows = []
    for subject, subject_config in subjects.items():
        for seed in SEEDS:
            for method in METHODS:
                for prompt_id in PROMPT_IDS:
                    record = records[(subject, seed, method, prompt_id)]
                    row = {
                        "subject": subject,
                        "dataset": subject_config["dataset"],
                        "seed": seed,
                        "method": method,
                        "prompt_id": prompt_id,
                        "prompt_category": record["prompt_category"],
                        "generation_prompt": record["generation_prompt"],
                        "evaluation_prompt": record["evaluation_prompt"],
                        "image_path": str(record["image_file"].relative_to(root)),
                        "status": "FOUND",
                    }
                    comparison_rows.append(row)
                    if prompt_id in LEAKAGE_PROMPT_IDS:
                        leakage_rows.append({**row, "annotation_status": "PENDING_HUMAN_REVIEW"})

    for path, rows in [
        (per_subject_dir / "grid_generation_summary.csv", comparison_rows),
        (leakage_dir / "background_leakage_review_manifest.csv", leakage_rows),
    ]:
        if path.exists() and not overwrite:
            raise FileExistsError(f"Refusing to overwrite existing manifest: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--wide-catalogs-only", action="store_true")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing grids, manifests, and prompt catalogs.",
    )
    args = parser.parse_args()
    root = args.project_root.resolve()
    subjects = yaml.safe_load((root / "configs/main_subjects.yaml").read_text())["subjects"]
    records = load_and_validate_records(root, subjects)

    if args.wide_catalogs_only:
        write_wide_prompt_and_conditioning_catalogs(
            root, subjects, records, overwrite=args.overwrite
        )
        print("Created wide prompt and subject-conditioning catalogs.")
        return

    per_subject_dir = root / "results/figures/main_experiment/per_subject"
    leakage_dir = root / "results/figures/main_experiment/background_leakage_review"
    for subject, subject_config in subjects.items():
        for seed in SEEDS:
            build_comparison_grid(
                root,
                per_subject_dir / f"{subject}_four_method_grid_seed{seed}",
                subject,
                subject_config,
                seed,
                records,
                overwrite=args.overwrite,
            )
            build_leakage_grid(
                root,
                leakage_dir / f"{subject}_background_leakage_seed{seed}",
                subject,
                subject_config,
                seed,
                records,
                overwrite=args.overwrite,
            )

    write_prompt_catalog(root, subjects, records, overwrite=args.overwrite)
    write_wide_prompt_and_conditioning_catalogs(
        root, subjects, records, overwrite=args.overwrite
    )
    write_manifests(root, subjects, records, overwrite=args.overwrite)
    print(f"Validated {len(records)} existing image/metadata pairs.")
    print(f"Created {len(subjects) * len(SEEDS)} seed-specific comparison grids (PNG + PDF).")
    print(f"Created {len(subjects) * len(SEEDS)} background-leakage review grids (PNG + PDF).")


if __name__ == "__main__":
    main()
