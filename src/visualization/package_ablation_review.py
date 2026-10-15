#!/usr/bin/env python3
"""Package review-only grids (never datasets or model weights) at <=2000 px width."""

from __future__ import annotations

import hashlib
import shutil
import zipfile
from pathlib import Path

from PIL import Image


def sha256(path: Path) -> str:
    digest = hashlib.sha256(path.read_bytes())
    return digest.hexdigest()


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    export_root = root / "results/exports/ablation_review"
    if export_root.exists():
        raise FileExistsError(f"Refusing to overwrite existing export: {export_root}")
    export_root.mkdir(parents=True)
    sources = [
        root / "results/figures/ablation/task_a/ip_scale03_review",
        root / "results/figures/ablation/task_a/four_method_scale03",
        root / "results/figures/ablation/task_c",
        root / "results/figures/ablation/task_d",
    ]
    copied = []
    for source in sources:
        if not source.is_dir():
            continue
        target = export_root / source.relative_to(root / "results/figures")
        target.mkdir(parents=True, exist_ok=True)
        for path in sorted(source.glob("*.png")):
            image = Image.open(path).convert("RGB")
            if image.width > 2000:
                height = round(image.height * 2000 / image.width)
                image = image.resize((2000, height), Image.Resampling.LANCZOS)
            destination = target / path.name
            image.save(destination, optimize=True)
            copied.append(destination)
    annotation = root / "results/metrics/manual_overfitting_annotation_ip_scale03_blank.csv"
    shutil.copy2(annotation, export_root / annotation.name)
    readme = export_root / "README.md"
    readme.write_text(
        "# Ablation review package\n\n"
        "This archive contains review grids only; it contains no dataset images as separate files and no model weights. "
        "All PNG files are at most 2000 pixels wide. Use the blank CSV for human scoring; score fields were not prefilled.\n"
    )
    manifest = export_root / "MANIFEST.sha256"
    manifest.write_text("\n".join(f"{sha256(path)}  {path.relative_to(export_root)}" for path in sorted(copied + [export_root / annotation.name, readme])) + "\n")
    archive = root / "results/exports/ablation_review_grids.zip"
    if archive.exists():
        raise FileExistsError(f"Refusing to overwrite existing archive: {archive}")
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as output:
        for path in sorted(export_root.rglob("*")):
            if path.is_file():
                output.write(path, path.relative_to(export_root.parent))
    (archive.with_suffix(archive.suffix + ".sha256")).write_text(f"{sha256(archive)}  {archive.name}\n")
    print(f"REVIEW_PACKAGE_COMPLETE png={len(copied)} archive={archive} bytes={archive.stat().st_size}")


if __name__ == "__main__":
    main()
