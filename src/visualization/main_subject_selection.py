#!/usr/bin/env python3
"""Create reference-only contact sheets for the proposed eight-subject audit."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps


SUBJECTS = [
    {
        "subject": "dog2",
        "dataset": "DreamBooth",
        "category": "Animal",
        "relative_dir": "data/dreambooth/upstream/dataset/dog2",
    },
    {
        "subject": "cat",
        "dataset": "DreamBooth",
        "category": "Animal",
        "relative_dir": "data/dreambooth/upstream/dataset/cat",
    },
    {
        "subject": "pet_dog1",
        "dataset": "CustomConcept101",
        "category": "Animal",
        "relative_dir": "data/customconcept101/dataset/benchmark_dataset/pet_dog1",
    },
    {
        "subject": "bear_plushie",
        "dataset": "DreamBooth",
        "category": "Soft texture object",
        "relative_dir": "data/dreambooth/upstream/dataset/bear_plushie",
    },
    {
        "subject": "robot_toy",
        "dataset": "DreamBooth",
        "category": "Rigid object",
        "relative_dir": "data/dreambooth/upstream/dataset/robot_toy",
    },
    {
        "subject": "teapot",
        "dataset": "DreamBooth",
        "category": "Rigid object",
        "relative_dir": "data/dreambooth/upstream/dataset/teapot",
    },
    {
        "subject": "colorful_sneaker",
        "dataset": "DreamBooth",
        "category": "Wearable/product",
        "relative_dir": "data/dreambooth/upstream/dataset/colorful_sneaker",
    },
    {
        "subject": "luggage_backpack1",
        "dataset": "CustomConcept101",
        "category": "Wearable/product",
        "relative_dir": "data/customconcept101/dataset/benchmark_dataset/luggage_backpack1",
    },
]

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
CATEGORY_COLORS = {
    "Animal": "#D9EAF7",
    "Soft texture object": "#F7E4C7",
    "Rigid object": "#DEEAD7",
    "Wearable/product": "#E8DDF2",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Explicitly replace existing audit figures.",
    )
    parser.add_argument(
        "--preserve-existing",
        action="store_true",
        help="Keep existing per-subject sheets and create only missing files.",
    )
    parser.add_argument(
        "--overview-name",
        default="all_subjects_overview.png",
        help="Filename for the overview PNG.",
    )
    return parser.parse_args()


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
        if bold
        else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for candidate in candidates:
        if Path(candidate).is_file():
            return ImageFont.truetype(candidate, size=size)
    return ImageFont.load_default()


def references(project_root: Path, subject: dict[str, str]) -> list[Path]:
    directory = project_root / subject["relative_dir"]
    if not directory.is_dir():
        raise FileNotFoundError(f"Missing subject directory: {directory}")
    paths = sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    if not paths:
        raise RuntimeError(f"No supported reference images in {directory}")
    for path in paths:
        if path.stat().st_size == 0:
            raise RuntimeError(f"Zero-byte reference image: {path}")
        with Image.open(path) as image:
            image.verify()
    return paths


def save_png(
    image: Image.Image,
    path: Path,
    overwrite: bool,
    preserve_existing: bool = False,
) -> None:
    if path.exists() and preserve_existing:
        print(f"Preserved existing {path}")
        return
    if path.exists() and not overwrite:
        raise FileExistsError(
            f"Refusing to overwrite {path}; pass --overwrite explicitly"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, format="PNG", dpi=(300, 300))
    print(f"Created {path}")


def contact_sheet(subject: dict[str, str], paths: list[Path]) -> Image.Image:
    columns = min(4, len(paths))
    rows = math.ceil(len(paths) / columns)
    cell_width, cell_height = 360, 330
    header_height = 105
    canvas = Image.new(
        "RGB",
        (columns * cell_width, header_height + rows * cell_height),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    draw.rectangle(
        (0, 0, canvas.width, header_height),
        fill=CATEGORY_COLORS[subject["category"]],
    )
    draw.text(
        (18, 14),
        f"{subject['subject']} — {subject['category']}",
        font=font(28, bold=True),
        fill="black",
    )
    draw.text(
        (18, 57),
        f"Dataset: {subject['dataset']}    References: {len(paths)}",
        font=font(20),
        fill="black",
    )
    for index, path in enumerate(paths):
        row, column = divmod(index, columns)
        x0 = column * cell_width
        y0 = header_height + row * cell_height
        draw.text((x0 + 10, y0 + 8), path.name, font=font(16), fill="black")
        with Image.open(path) as source:
            image = source.convert("RGB")
            image.thumbnail((cell_width - 20, cell_height - 48))
            x = x0 + (cell_width - image.width) // 2
            y = y0 + 38 + (cell_height - 42 - image.height) // 2
            canvas.paste(image, (x, y))
    return canvas


def overview(
    project_root: Path,
    subjects_with_paths: list[tuple[dict[str, str], list[Path]]],
) -> Image.Image:
    grouped: dict[str, list[tuple[dict[str, str], list[Path]]]] = {}
    for item in subjects_with_paths:
        grouped.setdefault(item[0]["category"], []).append(item)

    categories = list(CATEGORY_COLORS)
    columns = max(len(grouped[category]) for category in categories)
    label_width, cell_width, cell_height = 260, 330, 315
    title_height = 90
    canvas = Image.new(
        "RGB",
        (label_width + columns * cell_width, title_height + len(categories) * cell_height),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (18, 18),
        "Main Experiment: Eight-Subject Category Overview",
        font=font(30, bold=True),
        fill="black",
    )
    draw.text(
        (18, 57),
        "Representative reference shown; individual sheets contain every reference image.",
        font=font(16),
        fill="black",
    )

    for row, category in enumerate(categories):
        y0 = title_height + row * cell_height
        draw.rectangle(
            (0, y0, label_width, y0 + cell_height),
            fill=CATEGORY_COLORS[category],
        )
        draw.multiline_text(
            (18, y0 + 118),
            category.replace(" ", "\n", 1),
            font=font(23, bold=True),
            fill="black",
            spacing=5,
        )
        for column, (subject, paths) in enumerate(grouped[category]):
            x0 = label_width + column * cell_width
            draw.text(
                (x0 + 10, y0 + 8),
                subject["subject"],
                font=font(19, bold=True),
                fill="black",
            )
            draw.text(
                (x0 + 10, y0 + 35),
                f"{subject['dataset']} · n={len(paths)}",
                font=font(14),
                fill="black",
            )
            with Image.open(paths[0]) as source:
                image = ImageOps.contain(source.convert("RGB"), (cell_width - 20, 245))
                x = x0 + (cell_width - image.width) // 2
                y = y0 + 63 + (245 - image.height) // 2
                canvas.paste(image, (x, y))
    return canvas


def main() -> None:
    args = parse_args()
    project_root = args.project_root.resolve()
    output_dir = project_root / "results/figures/main_subject_selection"
    subjects_with_paths = [
        (subject, references(project_root, subject)) for subject in SUBJECTS
    ]

    for subject, paths in subjects_with_paths:
        save_png(
            contact_sheet(subject, paths),
            output_dir / f"{subject['subject']}_contact_sheet.png",
            args.overwrite,
            args.preserve_existing,
        )
    save_png(
        overview(project_root, subjects_with_paths),
        output_dir / args.overview_name,
        args.overwrite,
        args.preserve_existing,
    )


if __name__ == "__main__":
    main()
