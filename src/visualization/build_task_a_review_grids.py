#!/usr/bin/env python3
"""Create <=2000 px Task A review and four-method grids from existing images."""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

import yaml
from PIL import Image, ImageDraw, ImageFont


PROMPTS = ["P0", "P1", "P2", "P3", "P4", "P5"]
PROMPT_LABELS = {
    "P0": "Identity", "P1": "Background", "P2": "Context",
    "P3": "Attribute", "P4": "Composition", "P5": "Style",
}
SEEDS = [42, 123, 456, 789]
METHODS = ["dreambooth", "lora", "textual_inversion", "ip_adapter_plus"]
METHOD_LABELS = {
    "dreambooth": "DreamBooth", "lora": "DreamBooth-LoRA",
    "textual_inversion": "Textual Inversion",
    "ip_adapter_plus": "IP-Adapter Plus\n(scale=0.3)",
}


def font(size: int, bold: bool = False):
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    return ImageFont.truetype(f"/usr/share/fonts/truetype/dejavu/{name}", size)


def fit(path: Path, width: int, height: int) -> Image.Image:
    with Image.open(path) as source:
        image = source.convert("RGB")
    image.thumbnail((width, height), Image.Resampling.LANCZOS)
    result = Image.new("RGB", (width, height), "white")
    result.paste(image, ((width - image.width) // 2, (height - image.height) // 2))
    return result


def center(draw, box, value: str, value_font) -> None:
    left, top, right, bottom = box
    lines = value.splitlines()
    line_h = value_font.size + 3
    y = top + max(0, (bottom - top - len(lines) * line_h) // 2)
    for line in lines:
        bounds = draw.textbbox((0, 0), line, font=value_font)
        draw.text((left + (right - left - (bounds[2] - bounds[0])) / 2, y), line, font=value_font, fill="black")
        y += line_h


def caption(draw, box, value: str) -> None:
    left, top, right, bottom = box
    value_font = font(10)
    lines = textwrap.wrap(value, width=34, break_long_words=False)
    y = top + 3
    for line in lines[:3]:
        bounds = draw.textbbox((0, 0), line, font=value_font)
        draw.text((left + (right - left - (bounds[2] - bounds[0])) / 2, y), line, font=value_font, fill="black")
        y += value_font.size + 2


def refs(root: Path, cfg: dict) -> list[Path]:
    return sorted(path for path in (root / cfg["image_dir"]).iterdir() if path.is_file())


def read_pair(image: Path) -> dict:
    if not image.is_file() or not image.with_suffix(".json").is_file():
        raise FileNotFoundError(image)
    return json.loads(image.with_suffix(".json").read_text())


def review_grid(root: Path, subject: str, cfg: dict, destination: Path) -> None:
    cell_w, image_h, cap_h, label_w = 270, 210, 58, 245
    title_h, header_h = 48, 54
    row_h = image_h + cap_h
    width, height = label_w + 6 * cell_w, title_h + header_h + 5 * row_h
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    center(draw, (0, 0, width, title_h), f"IP-Adapter Plus scale=0.3 Review: {subject}", font(20, True))
    for col, prompt in enumerate(PROMPTS):
        left = label_w + col * cell_w
        center(draw, (left, title_h, left + cell_w, title_h + header_h), f"{prompt}  {PROMPT_LABELS[prompt]}", font(13, True))
    reference_paths = refs(root, cfg)
    rows = ["References"] + [f"Seed {seed}" for seed in SEEDS]
    for row, row_label in enumerate(rows):
        top = title_h + header_h + row * row_h
        center(draw, (0, top, label_w - 4, top + image_h), row_label, font(15, True))
        for col, prompt in enumerate(PROMPTS):
            left = label_w + col * cell_w
            if row == 0:
                path = reference_paths[col % len(reference_paths)]
                text = path.name
            else:
                seed = SEEDS[row - 1]
                path = root / "outputs/experiments/task_a/full_plus_s0.3" / cfg["dataset"] / subject / prompt / f"seed_{seed}.png"
                text = read_pair(path)["generation_prompt"]
            canvas.paste(fit(path, cell_w - 10, image_h - 8), (left + 5, top + 4))
            caption(draw, (left + 4, top + image_h, left + cell_w - 4, top + row_h), text)
    destination.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(destination, optimize=True, dpi=(200, 200))


def four_method_grid(root: Path, subject: str, cfg: dict, seed: int, destination: Path) -> None:
    cell_w, image_h, cap_h, label_w = 270, 205, 58, 245
    title_h, header_h = 48, 54
    row_h = image_h + cap_h
    width, height = label_w + 6 * cell_w, title_h + header_h + 5 * row_h
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    center(draw, (0, 0, width, title_h), f"{subject}: Four-Method Comparison (Seed={seed})", font(20, True))
    for col, prompt in enumerate(PROMPTS):
        left = label_w + col * cell_w
        center(draw, (left, title_h, left + cell_w, title_h + header_h), f"{prompt}  {PROMPT_LABELS[prompt]}", font(13, True))
    reference_paths = refs(root, cfg)
    rows = ["Reference"] + METHODS
    for row, method in enumerate(rows):
        top = title_h + header_h + row * row_h
        label = method if method == "Reference" else METHOD_LABELS[method]
        center(draw, (0, top, label_w - 4, top + image_h), label, font(14, True))
        for col, prompt in enumerate(PROMPTS):
            left = label_w + col * cell_w
            if method == "Reference":
                path = reference_paths[col % len(reference_paths)]
                text = path.name
            elif method == "ip_adapter_plus":
                path = root / "outputs/experiments/task_a/full_plus_s0.3" / cfg["dataset"] / subject / prompt / f"seed_{seed}.png"
                text = read_pair(path)["generation_prompt"]
            else:
                path = root / "outputs/main_evaluation" / method / cfg["dataset"] / subject / prompt / f"seed_{seed}.png"
                text = read_pair(path)["generation_prompt"]
            canvas.paste(fit(path, cell_w - 10, image_h - 8), (left + 5, top + 4))
            caption(draw, (left + 4, top + image_h, left + cell_w - 4, top + row_h), text)
    destination.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(destination, optimize=True, dpi=(200, 200))


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    subjects = yaml.safe_load((root / "configs/main_subjects.yaml").read_text())["subjects"]
    figure_root = root / "results/figures/ablation/task_a"
    for subject, cfg in subjects.items():
        review_grid(root, subject, cfg, figure_root / "ip_scale03_review" / f"ip_s0.3_review_grid_{subject}.png")
        for seed in SEEDS:
            four_method_grid(
                root, subject, cfg, seed,
                figure_root / "four_method_scale03" / f"{subject}_four_method_grid_seed{seed}.png",
            )
    print("TASK_A_GRIDS_COMPLETE review=8 four_method=32 max_width=1865")


if __name__ == "__main__":
    main()
