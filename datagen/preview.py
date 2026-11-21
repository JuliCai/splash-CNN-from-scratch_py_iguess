#!/usr/bin/env python3
"""
Step 5 - draw human-friendly preview sheets of the dataset into data/preview/:

  train_samples.png    16 random synthetic training images per class (32x32, shown 3x bigger)
  test_samples.png     all 150 real test images: original (left) and the 32x32 the network sees
  pipeline_steps.png   how a synthetic image is built, stage by stage, for a few examples

    python3 datagen/preview.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent))
import synthesize as S  # noqa: E402
from config import CLASSES, DATA  # noqa: E402
from imgutil import get_font, list_images, load_rgb  # noqa: E402

BG, INK = (24, 24, 27), (235, 235, 240)


def label_column(draw, rows, h, width, font):
    for r, c in enumerate(CLASSES):
        draw.text((8, r * h + h // 2 - 9), f"{r}  {c['name']}", fill=INK, font=font)


def train_samples(out: Path, per_row: int = 16, scale: int = 3):
    rng = np.random.default_rng(0)
    t = 32 * scale + 4
    lw = 210
    sheet = Image.new("RGB", (lw + per_row * t, len(CLASSES) * t), BG)
    d = ImageDraw.Draw(sheet)
    label_column(d, CLASSES, t, lw, get_font("arial_bold", 15))
    for r, c in enumerate(CLASSES):
        files = list_images(DATA / "train" / c["name"])
        for k, i in enumerate(rng.choice(len(files), per_row, replace=False)):
            im = Image.open(files[i]).convert("RGB").resize((32 * scale, 32 * scale), Image.NEAREST)
            sheet.paste(im, (lw + k * t + 2, r * t + 2))
    sheet.save(out)


def test_samples(out: Path, scale: int = 2):
    t_full, t_small = 96, 32 * scale
    cell = t_full + t_small + 10
    lw, per_row = 210, 15
    sheet = Image.new("RGB", (lw + per_row * cell, len(CLASSES) * (t_full + 6)), BG)
    d = ImageDraw.Draw(sheet)
    label_column(d, CLASSES, t_full + 6, lw, get_font("arial_bold", 15))
    for r, c in enumerate(CLASSES):
        for k, p in enumerate(list_images(DATA / "test_full" / c["name"])[:per_row]):
            x, y = lw + k * cell, r * (t_full + 6) + 3
            im = load_rgb(p)
            im.thumbnail((t_full, t_full))
            sheet.paste(im, (x + (t_full - im.width) // 2, y + (t_full - im.height) // 2))
            small = DATA / "test" / c["name"] / (p.stem + ".png")
            if small.exists():
                sheet.paste(Image.open(small).convert("RGB").resize((t_small, t_small), Image.NEAREST),
                            (x + t_full + 4, y + (t_full - t_small) // 2))
    sheet.save(out)


def pipeline_steps(out: Path, n: int = 6):
    seeds, bgs = S.load_pool(DATA / "raw" / "seeds", DATA / "raw" / "backgrounds")
    font = get_font("arial_bold", 13)
    t, lw = 150, 0
    rows = []
    rng0 = np.random.default_rng(7)
    for k in range(n):
        ci = int(rng0.integers(len(CLASSES)))
        name, seed, mask = seeds[ci][int(rng0.integers(len(seeds[ci])))]
        trace = [("real seed image", seed)]
        S.synthesize_one(np.random.default_rng([7, k]), seed, mask, bgs, 32, trace=trace)
        rows.append((CLASSES[ci]["name"], trace))
    cols = max(len(tr) for _, tr in rows)
    sheet = Image.new("RGB", (lw + cols * (t + 30), n * (t + 50) + 10), BG)
    d = ImageDraw.Draw(sheet)
    for r, (cname, trace) in enumerate(rows):
        y = r * (t + 50) + 26
        d.text((lw + 10, y + t + 4), f"class: {cname}", fill=(250, 200, 60), font=font)
        for k, (stage, im) in enumerate(trace):
            x = lw + k * (t + 30) + 10
            im = im.convert("RGB")
            im = im.resize((t, t), Image.NEAREST) if im.width <= 64 else im.resize(
                (t, int(t * im.height / im.width)) if im.width >= im.height else (int(t * im.width / im.height), t),
                Image.LANCZOS)
            sheet.paste(im, (x + (t - im.width) // 2, y + (t - im.height) // 2))
            d.text((x, y - 18), stage, fill=INK, font=font)
            if k < len(trace) - 1:
                d.text((x + t + 8, y + t // 2 - 8), "→", fill=(250, 200, 60), font=get_font("arial_bold", 18))
    sheet.save(out)


def main():
    out = DATA / "preview"
    out.mkdir(parents=True, exist_ok=True)
    train_samples(out / "train_samples.png")
    test_samples(out / "test_samples.png")
    pipeline_steps(out / "pipeline_steps.png")
    print("wrote", ", ".join(p.name for p in sorted(out.glob("*.png"))))


if __name__ == "__main__":
    main()
