#!/usr/bin/env python3
"""
Step 4 - turn the held-out REAL test images (data/test_full/<class>/*) into LeNet-sized inputs.

    centre square crop  ->  the exact same antialiased resize used for training (imgutil.to_lenet)

Writes data/test/<class>/<name>.png, data/test.bin (CIFAR-10 format) and data/test.csv.
The full-resolution originals stay in data/test_full/ for inference demos.

    python3 datagen/prepare_test.py                 # centre crop (default)
    python3 datagen/prepare_test.py --mode squash   # or: stretch the whole image to a square
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import CLASSES, DATA, IMAGE_SIZE  # noqa: E402
from imgutil import center_square, list_images, load_rgb, to_lenet, write_cifar_bin  # noqa: E402


def preprocess(im: Image.Image, size: int, mode: str) -> Image.Image:
    if mode == "crop":
        im = center_square(im)
    elif mode == "pad":
        s = max(im.size)
        canvas = Image.new("RGB", (s, s), (0, 0, 0))
        canvas.paste(im, ((s - im.width) // 2, (s - im.height) // 2))
        im = canvas
    return to_lenet(im, size)   # "squash" = resize the whole image straight to size x size


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, default=DATA / "test_full")
    ap.add_argument("--out", type=Path, default=DATA)
    ap.add_argument("--size", type=int, default=IMAGE_SIZE)
    ap.add_argument("--mode", choices=["crop", "squash", "pad"], default="crop")
    args = ap.parse_args()

    sources = {}
    src_csv = DATA / "raw" / "sources.csv"
    if src_csv.exists():
        for r in csv.DictReader(open(src_csv, newline="", encoding="utf-8")):
            sources[(r["class"], r["file"])] = r.get("source_url", "")

    images, labels, rows = [], [], []
    for ci, c in enumerate(CLASSES):
        out_dir = args.out / "test" / c["name"]
        out_dir.mkdir(parents=True, exist_ok=True)
        for old in out_dir.glob("*.png"):
            old.unlink()
        for p in list_images(args.src / c["name"]):
            im = preprocess(load_rgb(p), args.size, args.mode)
            dest = out_dir / (p.stem + ".png")
            im.save(dest)
            rows.append([len(rows), str(dest.relative_to(args.out)), ci, c["name"],
                         str(p.relative_to(args.out)), sources.get((c["name"], p.name), "")])
            images.append(np.asarray(im, dtype=np.uint8))
            labels.append(ci)
        print(f"  {c['name']:24s} {sum(1 for l in labels if l == ci):3d} test images")

    write_cifar_bin(args.out / "test.bin", np.stack(images), np.array(labels, dtype=np.uint8))
    with open(args.out / "test.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["index", "path", "label", "class", "original", "source_url"])
        w.writerows(rows)
    print(f"wrote {len(rows)} test images -> {args.out / 'test.bin'}  (mode={args.mode})")


if __name__ == "__main__":
    main()
