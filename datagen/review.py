#!/usr/bin/env python3
"""
Curation helper - turn a folder of candidate images into numbered contact sheets.

    python3 datagen/review.py <candidate_dir>/<class> --sheets <out_dir> [--cols 8 --rows 6]

Prints an index -> file table (and flags near-duplicates via dHash) so a human can look at the
sheets and decide which images become seeds / test images.  Used to hand-check every label.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent))
from imgutil import dhash, get_font, hamming, list_images, load_rgb  # noqa: E402


def contact_sheets(paths: list[Path], out_dir: Path, prefix: str, cols: int, rows: int, tile: int,
                   start_index: int = 0) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    font = get_font("arial_bold", 15)
    per = cols * rows
    sheets = []
    for s in range(0, len(paths), per):
        chunk = paths[s:s + per]
        sheet = Image.new("RGB", (cols * tile, ((len(chunk) - 1) // cols + 1) * tile), (40, 40, 40))
        draw = ImageDraw.Draw(sheet)
        for i, p in enumerate(chunk):
            x, y = (i % cols) * tile, (i // cols) * tile
            try:
                im = load_rgb(p)
                im.thumbnail((tile - 4, tile - 4), Image.LANCZOS)
                sheet.paste(im, (x + (tile - im.width) // 2, y + (tile - im.height) // 2))
            except Exception:
                draw.text((x + 10, y + tile // 2), "BROKEN", fill=(255, 0, 0), font=font)
            label = str(start_index + s + i)
            draw.rectangle([x, y, x + 9 * len(label) + 8, y + 19], fill=(255, 230, 0))
            draw.text((x + 3, y + 1), label, fill=(0, 0, 0), font=font)
        dest = out_dir / f"{prefix}_{s // per:02d}.png"
        sheet.save(dest)
        sheets.append(dest)
    return sheets


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("folder", type=Path)
    ap.add_argument("--sheets", type=Path, required=True, help="where to write contact sheet PNGs")
    ap.add_argument("--cols", type=int, default=8)
    ap.add_argument("--rows", type=int, default=6)
    ap.add_argument("--tile", type=int, default=150)
    ap.add_argument("--dup-threshold", type=int, default=8)
    args = ap.parse_args()

    paths = list_images(args.folder)
    hashes = {}
    for i, p in enumerate(paths):
        try:
            im = load_rgb(p)
            hashes[i] = dhash(im)
            print(f"{i:4d}  {p.name:32s} {im.width}x{im.height}")
        except Exception as exc:
            print(f"{i:4d}  {p.name:32s} BROKEN ({exc})")
    dups = [(a, b, hamming(hashes[a], hashes[b])) for a in hashes for b in hashes
            if a < b and hamming(hashes[a], hashes[b]) <= args.dup_threshold]
    if dups:
        print("near-duplicates (a, b, distance):", dups)
    for s in contact_sheets(paths, args.sheets, args.folder.name, args.cols, args.rows, args.tile):
        print("sheet:", s)


if __name__ == "__main__":
    main()
