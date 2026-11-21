#!/usr/bin/env python3
"""
Step 1b - collect CANDIDATE images from Wikimedia Commons (freely licensed: CC0 / CC BY / CC BY-SA ...).

Two kinds of entries per class (see COMMONS below):
  files    - exact File: titles we already know are good (e.g. the CC0 / CC BY-SA clips of people doing
             the 6-7 gesture).  For a video we take several still frames.
  searches - Commons searches whose result titles must match `include` and not match `exclude`.

Wikimedia rate-limits clients that hammer its file servers (HTTP 429), so this script only ever asks
for *standard-size thumbnails* (500 px), takes video frames with the thumbnailer's `seek=` option
(no full video downloads), waits between requests and honours Retry-After.

Licence + author of every file is logged to <out>/candidates.csv - CC BY / BY-SA images must be credited
(promote.py copies this into data/raw/sources.csv).

    python3 datagen/scrape_commons.py --out /tmp/commons
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

API = "https://commons.wikimedia.org/w/api.php"
UA = "meme-dataset-builder/1.0 (educational classroom project; python-urllib)"
THUMB_W = 500            # one of Wikimedia's standard thumbnail widths
FIELDS = ["class", "file", "kind", "url", "page", "template", "alt", "license", "author"]

COMMONS = {
    "six_seven": dict(
        files=["File:67 meme gesture.webm", "File:6-7 hand gesture 2025.webm",
               "File:67 Meme Performance (6-7, 6 7).webm", "File:67 dance.webm", "File:67 Brainrot Meme.webm",
               "File:67 (meme) Promoted on YouTube 9 times.webm", "File:Diosdado Cabello 67.webm",
               "File:6-7 Meme Washington University.jpg", "File:67 meme character.jpg",
               "File:Six seven на Ломоносове 20260523 112620.jpg"],
        frames=7),
    "tung_tung_tung_sahur": dict(searches=["Tung Tung Tung Sahur", "Tung Tung Sahur brainrot"],
                                 include=r"tung|sahur", exclude=r"copyright|\.webm|\.opus|\.ogg|\.svg|\.pdf"),
    "tralalero_tralala": dict(searches=["Tralalero Tralala", "Tralalero shark"],
                              include=r"tralal", exclude=r"\.opus|\.ogg|\.webm|music|\.svg|\.pdf"),
}


class Client:
    """Sequential HTTP with a minimum gap between requests and polite handling of 429s."""

    def __init__(self, gap: float):
        self.gap, self.last = gap, 0.0

    def get(self, url: str, timeout: int = 40) -> bytes | None:
        for attempt in range(4):
            wait = self.gap - (time.time() - self.last)
            if wait > 0:
                time.sleep(wait)
            self.last = time.time()
            try:
                req = urllib.request.Request(url, headers={"User-Agent": UA})
                with urllib.request.urlopen(req, timeout=timeout) as r:
                    return r.read()
            except urllib.error.HTTPError as e:
                if e.code == 429:
                    pause = int(e.headers.get("Retry-After") or 0) or 60 * (attempt + 1)
                    print(f"    429 rate-limited, sleeping {pause}s", file=sys.stderr)
                    time.sleep(pause)
                    continue
                print(f"    HTTP {e.code} for {url}", file=sys.stderr)
                return None
            except Exception as exc:
                print(f"    {exc} for {url}", file=sys.stderr)
                time.sleep(5 * (attempt + 1))
        return None


def api(client: Client, **params) -> dict:
    q = urllib.parse.urlencode({**params, "action": "query", "format": "json", "maxlag": "5"})
    raw = client.get(f"{API}?{q}")
    return json.loads(raw) if raw else {}


def strip_html(s: str) -> str:
    return html.unescape(re.sub(r"<[^>]+>", "", s or "")).strip()


def file_info(client: Client, titles: list[str]) -> list[dict]:
    data = api(client, titles="|".join(titles), prop="imageinfo",
               iiprop="url|mime|size|extmetadata|metadata", iiurlwidth=THUMB_W)
    return [p for p in data.get("query", {}).get("pages", {}).values() if "imageinfo" in p]


def search(client: Client, query: str) -> list[dict]:
    data = api(client, generator="search", gsrnamespace=6, gsrlimit=50, gsrsearch=query, prop="imageinfo",
               iiprop="url|mime|size|extmetadata|metadata", iiurlwidth=THUMB_W)
    return sorted(data.get("query", {}).get("pages", {}).values(), key=lambda p: p.get("index", 0))


def duration(ii: dict) -> float:
    for m in ii.get("metadata") or []:
        if m.get("name") in ("playtime_seconds", "length", "duration"):
            try:
                return float(m["value"])
            except (TypeError, ValueError):
                pass
    return 0.0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--classes", default="all")
    ap.add_argument("--gap", type=float, default=3.0, help="seconds between requests")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    client = Client(args.gap)
    rows: list[dict] = []

    for cname, cfg in COMMONS.items():
        if args.classes != "all" and cname not in args.classes.split(","):
            continue
        print(f"\n=== {cname}")
        out_dir = args.out / cname
        out_dir.mkdir(exist_ok=True)
        pages = file_info(client, cfg["files"]) if cfg.get("files") else []
        if cfg.get("searches"):
            inc, exc = re.compile(cfg["include"], re.I), re.compile(cfg["exclude"], re.I)
            seen = {p["title"] for p in pages}
            for q in cfg["searches"]:
                for p in search(client, q):
                    if p["title"] not in seen and inc.search(p["title"]) and not exc.search(p["title"]):
                        seen.add(p["title"])
                        pages.append(p)

        for p in pages:
            ii = p["imageinfo"][0]
            meta = ii.get("extmetadata", {})
            base = {"class": cname, "url": ii.get("url", ""), "page": ii.get("descriptionurl", ""),
                    "template": p["title"], "alt": "",
                    "license": strip_html(meta.get("LicenseShortName", {}).get("value", "")),
                    "author": strip_html(meta.get("Artist", {}).get("value", ""))[:120]}
            thumb = ii.get("thumburl")
            if not thumb:
                continue
            stem = "commons_" + re.sub(r"[^A-Za-z0-9]+", "_", p["title"][5:].rsplit(".", 1)[0])[:60].strip("_")
            if ii.get("mime", "").startswith("video/"):
                dur, n = duration(ii), cfg.get("frames", 6)
                if dur <= 0 or f"/{THUMB_W}px--" not in thumb:
                    continue
                got = 0
                for k in range(n):
                    t = dur * (k + 0.5) / n
                    url = thumb.replace(f"/{THUMB_W}px--", f"/{THUMB_W}px-seek%3D{t:.2f}-")
                    dest = out_dir / f"{stem}_t{t:05.2f}.jpg"
                    data = dest.read_bytes() if dest.exists() else client.get(url)
                    if data and len(data) > 2000:
                        dest.write_bytes(data)
                        rows.append({**base, "file": dest.name, "kind": f"video frame @{t:.2f}s"})
                        got += 1
                print(f"  video {p['title'][5:70]} ({dur:.1f}s) -> {got} frames  [{base['license']}]")
            elif ii.get("mime") in ("image/jpeg", "image/png", "image/webp"):
                dest = out_dir / f"{stem}.jpg"
                data = dest.read_bytes() if dest.exists() else client.get(thumb)
                if data and len(data) > 2000:
                    dest.write_bytes(data)
                    rows.append({**base, "file": dest.name, "kind": "image"})
                    print(f"  image {p['title'][5:70]}  [{base['license']}]")

        with open(args.out / "candidates.csv", "w", newline="", encoding="utf-8") as f:  # checkpoint
            w = csv.DictWriter(f, fieldnames=FIELDS)
            w.writeheader()
            w.writerows(rows)
    print(f"\nwrote {len(rows)} rows to {args.out / 'candidates.csv'}")


if __name__ == "__main__":
    main()
