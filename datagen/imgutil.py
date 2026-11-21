"""Small image helpers shared by the pipeline scripts (numpy + Pillow only)."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image, ImageFont, ImageOps

Image.MAX_IMAGE_PIXELS = 60_000_000  # refuse absurdly large (decompression-bomb) files

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp"}


def list_images(folder: Path) -> list[Path]:
    return sorted(p for p in Path(folder).iterdir() if p.suffix.lower() in IMAGE_EXTS)


def load_rgb(path: Path, background=(255, 255, 255)) -> Image.Image:
    """Open any image as RGB: honours EXIF rotation, takes frame 0 of GIFs,
    and flattens transparency onto `background`."""
    im = Image.open(path)
    im = ImageOps.exif_transpose(im)
    if getattr(im, "is_animated", False):
        im.seek(0)
    if im.mode in ("RGBA", "LA", "PA") or (im.mode == "P" and "transparency" in im.info):
        im = im.convert("RGBA")
        flat = Image.new("RGB", im.size, background)
        flat.paste(im, mask=im.getchannel("A"))
        return flat
    return im.convert("RGB")


def center_square(im: Image.Image) -> Image.Image:
    """Largest centred square crop."""
    s = min(im.size)
    x, y = (im.width - s) // 2, (im.height - s) // 2
    return im.crop((x, y, x + s, y + s))


def to_lenet(im: Image.Image, size: int) -> Image.Image:
    """THE final resize used for every image in the dataset (synthetic and real), so train and test
    images go through exactly the same antialiased downsampling."""
    return im.convert("RGB").resize((size, size), Image.LANCZOS, reducing_gap=3.0)


def write_cifar_bin(path: Path, images: np.ndarray, labels: np.ndarray) -> None:
    """CIFAR-10 binary format: one record per image = 1 label byte + R plane + G plane + B plane
    (each plane is size*size bytes, row-major).  For 32x32 that's 1 + 3072 = 3073 bytes/record."""
    n = len(labels)
    planar = images.transpose(0, 3, 1, 2).reshape(n, -1)
    np.concatenate([labels.astype(np.uint8).reshape(n, 1), planar], axis=1).tofile(path)


def read_cifar_bin(path: Path, size: int = 32) -> tuple[np.ndarray, np.ndarray]:
    rec = 1 + 3 * size * size
    raw = np.fromfile(path, dtype=np.uint8)
    assert raw.size % rec == 0, f"{path}: size {raw.size} is not a multiple of {rec}"
    raw = raw.reshape(-1, rec)
    return raw[:, 1:].reshape(-1, 3, size, size).transpose(0, 2, 3, 1), raw[:, 0]


def dhash(im: Image.Image, size: int = 8) -> int:
    """Difference hash: 64-bit fingerprint that survives resizing / recompression.
    Two images with hamming(dhash(a), dhash(b)) <= ~10 are near-duplicates."""
    g = np.asarray(im.convert("L").resize((size + 1, size), Image.LANCZOS), dtype=np.int16)
    bits = (g[:, 1:] > g[:, :-1]).flatten()
    return int("".join("1" if b else "0" for b in bits), 2)


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def fingerprint(im: Image.Image) -> tuple[int, np.ndarray]:
    """(256-bit dHash, 64x64 grey thumbnail) - compute once per image, compare with same_fingerprint."""
    return dhash(im, 16), np.asarray(im.convert("L").resize((64, 64), Image.BILINEAR), dtype=np.float32)


def max_block_diff(ga: np.ndarray, gb: np.ndarray, block: int = 8) -> float:
    """Largest mean |difference| over any block x block tile of two 64x64 grey thumbnails.
    Re-encodes/resizes of one picture stay < ~10; a changed caption or label pushes some tile >> 10."""
    k = ga.shape[0] // block
    return float(np.abs(ga - gb).reshape(k, block, k, block).mean(axis=(1, 3)).max())


def same_fingerprint(fa, fb) -> bool:
    """Same picture (repost / re-encode / resize), not merely the same meme template with different
    text: the 256-bit dHash must match AND no 8x8 tile of the thumbnails may differ."""
    return hamming(fa[0], fb[0]) <= 6 and max_block_diff(fa[1], fb[1]) < 12


def same_picture(a: Image.Image, b: Image.Image) -> bool:
    return same_fingerprint(fingerprint(a), fingerprint(b))


_FONT_FILES = {
    "impact": ["/System/Library/Fonts/Supplemental/Impact.ttf", "/Library/Fonts/Impact.ttf",
               "C:/Windows/Fonts/impact.ttf", "/usr/share/fonts/truetype/msttcorefonts/Impact.ttf"],
    "arial_bold": ["/System/Library/Fonts/Supplemental/Arial Bold.ttf", "C:/Windows/Fonts/arialbd.ttf",
                   "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"],
    "arial": ["/System/Library/Fonts/Supplemental/Arial.ttf", "C:/Windows/Fonts/arial.ttf",
              "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"],
    "comic": ["/System/Library/Fonts/Supplemental/Comic Sans MS Bold.ttf", "C:/Windows/Fonts/comicbd.ttf"],
    "times": ["/System/Library/Fonts/Supplemental/Times New Roman Bold.ttf", "C:/Windows/Fonts/timesbd.ttf",
              "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf"],
    "courier": ["/System/Library/Fonts/Supplemental/Courier New Bold.ttf", "C:/Windows/Fonts/courbd.ttf",
                "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf"],
}


@lru_cache(maxsize=None)
def font_path(kind: str) -> str | None:
    for p in _FONT_FILES.get(kind, []):
        if Path(p).exists():
            return p
    return None


@lru_cache(maxsize=512)
def get_font(kind: str, size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """TrueType font of the given family, falling back to any available family,
    then to Pillow's built-in font, so the pipeline runs on any OS."""
    size = max(6, int(size))
    for k in (kind, "arial_bold", "arial"):
        p = font_path(k)
        if p:
            return ImageFont.truetype(p, size)
    return ImageFont.load_default(size=size)


def available_fonts() -> list[str]:
    return [k for k in _FONT_FILES if font_path(k)] or ["default"]


# colour emoji fonts only render at fixed bitmap sizes
_EMOJI_FONTS = [("/System/Library/Fonts/Apple Color Emoji.ttc", 96),
                ("/usr/share/fonts/truetype/noto/NotoColorEmoji.ttf", 109),
                ("C:/Windows/Fonts/seguiemj.ttf", 96)]


@lru_cache(maxsize=None)
def render_emoji(char: str) -> Image.Image | None:
    """RGBA image of a colour emoji, or None if this machine has no colour emoji font."""
    from PIL import ImageDraw
    for path, size in _EMOJI_FONTS:
        if not Path(path).exists():
            continue
        try:
            font = ImageFont.truetype(path, size)
            im = Image.new("RGBA", (size * 2, size * 2), (0, 0, 0, 0))
            ImageDraw.Draw(im).text((size // 4, size // 4), char, font=font, embedded_color=True)
            box = im.getbbox()
            return im.crop(box) if box else None
        except Exception:
            continue
    return None
