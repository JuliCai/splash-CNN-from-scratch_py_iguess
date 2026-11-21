#!/usr/bin/env python3
"""
Step 2 - promote hand-picked candidates into the dataset.

After looking at the contact sheets from review.py, the chosen files are listed in a picks file
(JSON: {"<class>": {"seed": [file, ...], "test": [file, ...]}}).  This script

  * checks every picked image decodes,
  * refuses a test image that is the same picture as a seed image (imgutil.same_picture: a repost or
    re-encode; the same template with a different caption is allowed - that's how memes work),
  * copies the files to
        data/raw/seeds/<class>/    real images the synthesizer is allowed to look at
        data/test_full/<class>/    held-out real images (never seen by the synthesizer)
  * writes data/raw/sources.csv with where every image came from (URL + page).

    python3 datagen/promote.py --candidates /path/imgflip /path/commons --picks datagen/picks.json
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import CLASS_NAMES, DATA  # noqa: E402
from imgutil import fingerprint, load_rgb, same_fingerprint  # noqa: E402

DEST = {"seed": DATA / "raw" / "seeds", "test": DATA / "test_full"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", type=Path, nargs="+", required=True,
                    help="one or more folders, each with candidates.csv + <class>/ dirs")
    ap.add_argument("--picks", type=Path, required=True)
    ap.add_argument("--clean", action="store_true", help="empty the destination class folders first")
    args = ap.parse_args()

    picks = json.loads(args.picks.read_text())
    meta, where = {}, {}
    for cdir in args.candidates:
        for r in csv.DictReader(open(cdir / "candidates.csv", newline="", encoding="utf-8")):
            meta[(r["class"], r["file"])] = r
            where[(r["class"], r["file"])] = cdir / r["class"] / r["file"]

    unknown = set(picks) - set(CLASS_NAMES)
    if unknown:
        sys.exit(f"unknown classes in picks: {unknown}")

    rows, problems = [], []
    for cname in CLASS_NAMES:
        sel = picks.get(cname, {})
        imgs = {"seed": {}, "test": {}}
        for split in ("seed", "test"):
            for fname in sel.get(split, []):
                src = where.get((cname, fname))
                if src is None:
                    problems.append(f"{cname}/{fname}: not in any candidates.csv")
                    continue
                try:
                    imgs[split][fname] = fingerprint(load_rgb(src))
                except Exception as exc:
                    problems.append(f"{cname}/{fname}: cannot read ({exc})")
        both = set(sel.get("seed", [])) & set(sel.get("test", []))
        problems += [f"{cname}/{f}: picked as seed AND test" for f in both]
        for tf, ti in imgs["test"].items():
            for sf, si in imgs["seed"].items():
                if same_fingerprint(ti, si):
                    problems.append(f"{cname}: test {tf} is the same picture as seed {sf}")
        print(f"{cname:24s} seeds={len(imgs['seed']):3d}  test={len(imgs['test']):3d}")

        for split in ("seed", "test"):
            out = DEST[split] / cname
            if args.clean and out.exists():
                for old in out.iterdir():
                    if old.is_file():
                        old.unlink()
            out.mkdir(parents=True, exist_ok=True)
            for fname in sel.get(split, []):
                if (cname, fname) not in where:
                    continue
                shutil.copyfile(where[(cname, fname)], out / fname)
                m = meta[(cname, fname)]
                rows.append({"split": split, "class": cname, "file": fname,
                             "path": str((out / fname).relative_to(DATA)), "source_url": m.get("url", ""),
                             "source_page": m.get("page", ""), "origin": m.get("template", ""),
                             "kind": m.get("kind", ""),
                             "license": m.get("license") or "unspecified (user-uploaded meme, imgflip.com)",
                             "author": m.get("author", "")})

    if problems:
        print("\nPROBLEMS:\n  " + "\n  ".join(problems))
        sys.exit(1)

    out_csv = DATA / "raw" / "sources.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {out_csv} ({len(rows)} images)")


if __name__ == "__main__":
    main()
