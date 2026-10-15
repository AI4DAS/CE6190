#!/usr/bin/env python3
"""Build Task C face-corruption inspection grids (seed 42)."""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


PROMPTS = ["P0", "P1", "P2", "P3", "P4", "P5"]
LABELS = ["Identity", "Background", "Context", "Attribute", "Composition", "Style"]
STEPS = {
    "dreambooth": list(range(100, 801, 100)),
    "lora": list(range(100, 1601, 100)),
    "textual_inversion": list(range(250, 3001, 250)),
}


def fnt(size, bold=False):
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    return ImageFont.truetype(f"/usr/share/fonts/truetype/dejavu/{name}", size)


def fit(path, width, height):
    image = Image.open(path).convert("RGB"); image.thumbnail((width, height), Image.Resampling.LANCZOS)
    out = Image.new("RGB", (width, height), "white")
    out.paste(image, ((width-image.width)//2, (height-image.height)//2)); return out


def center(draw, box, text, font):
    l,t,r,b=box; bounds=draw.textbbox((0,0),text,font=font)
    draw.text((l+(r-l-(bounds[2]-bounds[0]))/2,t+(b-t-(bounds[3]-bounds[1]))/2),text,font=font,fill="black")


def main():
    root = Path(__file__).resolve().parents[2]
    out_root = root / "results/figures/ablation/task_c"
    for method, steps in STEPS.items():
        for subject in ("dog2", "teapot", "bear_plushie"):
            cell_w, image_h, cap_h, label_w = 275, 165, 42, 190
            title_h, header_h = 46, 48
            row_h = image_h + cap_h
            width, height = label_w + 6*cell_w, title_h+header_h+len(steps)*row_h
            canvas=Image.new("RGB",(width,height),"white"); draw=ImageDraw.Draw(canvas)
            center(draw,(0,0,width,title_h),f"{method} / {subject}: checkpoint inspection (Seed=42)",fnt(19,True))
            for col,(prompt,label) in enumerate(zip(PROMPTS,LABELS)):
                left=label_w+col*cell_w; center(draw,(left,title_h,left+cell_w,title_h+header_h),f"{prompt} {label}",fnt(12,True))
            for row,step in enumerate(steps):
                top=title_h+header_h+row*row_h; center(draw,(0,top,label_w-4,top+image_h),f"Step {step}",fnt(13,True))
                for col,prompt in enumerate(PROMPTS):
                    left=label_w+col*cell_w
                    path=root/f"outputs/experiments/task_c_steps/{method}/{subject}/step_{step}/{prompt}/seed_42.png"
                    meta=path.with_suffix('.json')
                    if not path.is_file() or not meta.is_file(): raise FileNotFoundError(path)
                    text=json.loads(meta.read_text())["generation_prompt"]
                    canvas.paste(fit(path,cell_w-8,image_h-6),(left+4,top+3))
                    lines=textwrap.wrap(text,width=34,break_long_words=False)[:2]
                    y=top+image_h+2
                    for line in lines:
                        center(draw,(left,y,left+cell_w,y+18),line,fnt(9)); y+=17
            out_root.mkdir(parents=True,exist_ok=True)
            destination=out_root/f"steps_grid_{method}_{subject}.png"
            canvas.save(destination,optimize=True,dpi=(200,200))
    print("STEPS_GRIDS_COMPLETE count=9 max_width=1840")


if __name__ == "__main__": main()
