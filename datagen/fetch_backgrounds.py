#!/usr/bin/env python3
"""
Download random real-world photos to use as backdrops in the synthesis pipeline
(the 'cutout' and 'screenshot' layouts paste memes onto them).

Photos come from picsum.photos, which serves Unsplash photos (free to use, Unsplash License).
They are class-agnostic: every class gets pasted onto the same pool of backgrounds.

    python3 datagen/fetch_backgrounds.py --n 300
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import DATA  # noqa: E402

UA = "Mozilla/5.0 (compatible; meme-dataset-builder/1.0; educational classroom project)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--size", type=int, default=320)
    ap.add_argument("--out", type=Path, default=DATA / "raw" / "backgrounds")
    ap.add_argument("--delay", type=float, default=0.3)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    rows = []
    for i in range(args.n):
        dest = args.out / f"bg_{i:04d}.jpg"
        url = f"https://picsum.photos/seed/meme-bg-{i}/{args.size}/{args.size}"
        for attempt in range(3):
            try:
                req = urllib.request.Request(url, headers={"User-Agent": UA})
                with urllib.request.urlopen(req, timeout=30) as r:
                    data, final = r.read(), r.geturl()
                m = re.search(r"/id/(\d+)/", final)
                pid = m.group(1) if m else ""
                break
            except Exception as exc:
                print(f"  retry {url}: {exc}", file=sys.stderr)
                time.sleep(2 + 2 * attempt)
        else:
            continue
        dest.write_bytes(data)
        rows.append([dest.name, pid, final])
        time.sleep(args.delay)
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{args.n}")
    with open(args.out.parent / "backgrounds.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["file", "picsum_id", "url"])
        w.writerows(rows)
    print(f"saved {len(rows)} backgrounds to {args.out}")


if __name__ == "__main__":
    main()
