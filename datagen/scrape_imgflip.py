#!/usr/bin/env python3
"""
Step 1 - collect CANDIDATE real meme images from imgflip.com.

For every class in config.CLASSES this script:
  1. searches imgflip's template index (/memesearch) with the class' queries,
  2. keeps templates whose title matches the class' include / exclude regexes,
  3. opens each template page (plus a few pages for big "official" templates) and grabs
       - the blank template image, and
       - the real, user-made memes shown on that page (captioned images),
  4. downloads them into <out>/<class>/ and logs every image to <out>/candidates.csv.

Candidates are NOT the dataset yet: a human reviews them (see review.py) and the good ones are
promoted to data/raw/seeds (sources for synthesis) or data/test_full (held-out real test images).

imgflip.com/robots.txt only disallows /orig/ and /browse/; we stay well under 1 request/second.
Standard library only, so it also runs under `python3 -I`.

    python3 datagen/scrape_imgflip.py --out /tmp/candidates            # all classes
    python3 datagen/scrape_imgflip.py --out /tmp/candidates --classes doge,trollface
"""

from __future__ import annotations

import argparse
import csv
import html
import re
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import CLASSES  # noqa: E402

BASE = "https://imgflip.com"
UA = "Mozilla/5.0 (compatible; meme-dataset-builder/1.0; educational classroom project)"

SEARCH_RE = re.compile(
    r'<h3 class="mt-title">\s*<a title="[^"]*" href="/meme/(\d+)/([^"]+)">([^<]+)</a>'
)
BLANK_RE = re.compile(r'alt="([^"]*) Blank Meme Template" src="([^"]+)"')
MEME_RE = re.compile(
    r"class='base-img' src='(//i\.imgflip\.com/[a-z0-9]+\.(?:jpg|jpeg|png))' alt='([^']*)'"
)

FIELDS = ["class", "file", "kind", "url", "page", "template", "alt"]


class Fetcher:
    """Tiny polite HTTP client: one request at a time, fixed delay, a few retries."""

    def __init__(self, delay: float):
        self.delay = delay
        self._last = 0.0

    def get(self, url: str, timeout: int = 30, retries: int = 3, delay: float | None = None) -> bytes:
        delay = self.delay if delay is None else delay
        for attempt in range(retries):
            wait = delay - (time.time() - self._last)
            if wait > 0:
                time.sleep(wait)
            self._last = time.time()
            try:
                req = urllib.request.Request(url, headers={"User-Agent": UA})
                with urllib.request.urlopen(req, timeout=timeout) as resp:
                    return resp.read()
            except Exception as exc:  # network hiccup / 5xx -> back off and retry
                if attempt == retries - 1:
                    raise
                print(f"    retry {attempt + 1} for {url}: {exc}", file=sys.stderr)
                time.sleep(3 * (attempt + 1))
        raise RuntimeError("unreachable")


def absolute(src: str) -> str:
    if src.startswith("//"):
        return "https:" + src
    if src.startswith("/"):
        return BASE + src
    return src


def search_templates(fetch: Fetcher, query: str, pages: int = 2) -> list[tuple[str, str, str]]:
    """Return [(template_id, slug, title)] from imgflip's template search."""
    found = []
    for page in range(1, pages + 1):
        url = f"{BASE}/memesearch?q={urllib.parse.quote_plus(query)}&page={page}"
        try:
            text = fetch.get(url).decode("utf-8", "replace")
        except Exception as exc:
            print(f"  ! search failed {url}: {exc}", file=sys.stderr)
            break
        hits = SEARCH_RE.findall(text)
        if not hits:
            break
        found += [(tid, slug, html.unescape(title).strip()) for tid, slug, title in hits]
    return found


def read_template_page(fetch: Fetcher, page_url: str):
    """Return (blank_image_url or None, [(meme_image_url, alt_text)])."""
    text = fetch.get(page_url).decode("utf-8", "replace")
    blank = BLANK_RE.search(text)
    blank_url = absolute(blank.group(2)) if blank else None
    memes = [(absolute(src), html.unescape(alt)) for src, alt in MEME_RE.findall(text)]
    return blank_url, memes


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, type=Path, help="directory for candidate images")
    ap.add_argument("--classes", default="all", help="comma separated class names, or 'all'")
    ap.add_argument("--max-templates", type=int, default=30, help="max user templates per class")
    ap.add_argument("--max-per-class", type=int, default=220, help="max candidate images per class")
    ap.add_argument("--delay", type=float, default=1.0, help="seconds between requests")
    args = ap.parse_args()

    wanted = None if args.classes == "all" else set(args.classes.split(","))
    classes = [c for c in CLASSES if wanted is None or c["name"] in wanted]
    args.out.mkdir(parents=True, exist_ok=True)
    csv_path = args.out / "candidates.csv"

    rows: dict[str, dict] = {}
    if csv_path.exists():  # resume: keep what an earlier run already logged
        with open(csv_path, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                rows[r["class"] + "/" + r["file"]] = r

    fetch = Fetcher(args.delay)
    for cls in classes:
        name, cfg = cls["name"], cls["imgflip"]
        print(f"\n=== {name}")
        inc = re.compile(cfg["include"], re.I)
        exc = re.compile(cfg["exclude"], re.I) if cfg.get("exclude") else None

        # (page_url, template_label) pairs to visit
        pages: list[tuple[str, str]] = []
        for slug in cfg.get("official", []):
            for p in range(1, cfg.get("pages", 1) + 1):
                pages.append((f"{BASE}/meme/{slug}" + (f"?page={p}" if p > 1 else ""), slug))

        seen_tpl = set()
        for q in cfg.get("queries", []):
            for tid, slug, title in search_templates(fetch, q):
                if tid in seen_tpl or not inc.search(title) or (exc and exc.search(title)):
                    continue
                seen_tpl.add(tid)
                if len(seen_tpl) <= args.max_templates:
                    pages.append((f"{BASE}/meme/{tid}/{slug}", f"{tid}:{title}"))
        print(f"  {len(pages)} template pages to read")

        out_dir = args.out / name
        out_dir.mkdir(exist_ok=True)
        n_class = sum(1 for k in rows if k.startswith(name + "/"))
        seen_urls = {r["url"] for r in rows.values()}

        for page_url, tpl in pages:
            if n_class >= args.max_per_class:
                break
            try:
                blank_url, memes = read_template_page(fetch, page_url)
            except Exception as exc_:
                print(f"  ! page failed {page_url}: {exc_}", file=sys.stderr)
                continue
            items = ([(blank_url, "blank template", "blank")] if blank_url else []) + [
                (u, alt, "meme") for u, alt in memes
            ]
            for url, alt, kind in items:
                if n_class >= args.max_per_class or url in seen_urls:
                    continue
                seen_urls.add(url)
                ext = Path(urllib.parse.urlparse(url).path).suffix.lower() or ".jpg"
                stem = Path(urllib.parse.urlparse(url).path).stem
                fname = f"imgflip_{stem}{ext}"
                dest = out_dir / fname
                if not dest.exists():
                    try:  # images come from the i.imgflip.com CDN -> shorter delay than HTML pages
                        data = fetch.get(url, delay=args.delay / 4)
                    except Exception as exc_:
                        print(f"  ! image failed {url}: {exc_}", file=sys.stderr)
                        continue
                    if len(data) < 2000:  # error pages / placeholders
                        continue
                    dest.write_bytes(data)
                rows[name + "/" + fname] = dict(
                    {"class": name, "file": fname, "kind": kind, "url": url, "page": page_url,
                     "template": tpl, "alt": alt[:300]}
                )
                n_class += 1
            print(f"  {tpl[:60]:60s} -> {len(items):3d} images (class total {n_class})")

        with open(csv_path, "w", newline="", encoding="utf-8") as f:  # checkpoint per class
            w = csv.DictWriter(f, fieldnames=FIELDS)
            w.writeheader()
            w.writerows(rows.values())

    print(f"\nwrote {csv_path} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
