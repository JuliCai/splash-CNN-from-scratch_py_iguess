#!/usr/bin/env python3
"""
Step 3 - the synthetic-data CV pipeline.

Turns a few hundred REAL meme images (data/raw/seeds/<class>/) into tens of thousands of NEW,
LeNet-sized training images.  Nothing here is learned: it is all hand-written image processing
(numpy + Pillow), i.e. classic "data augmentation" pushed far enough to manufacture a dataset.

    python3 datagen/synthesize.py                        # 3,500 train + 350 val images per class
    python3 datagen/synthesize.py --train-per-class 50 --val-per-class 0 --out /tmp/try   # quick test

Every image is produced by its own random recipe (fully reproducible from --seed):

    seed image -> random crop/flip -> LAYOUT -> meme text -> overlays -> tilt -> colour -> damage -> 32x32
                                        |
                                        +- fill        the crop fills the frame
                                        +- letterbox   whole picture + padding bars
                                        +- caption     "modern meme": text bar above/below the picture
                                        +- screenshot  picture inside a social-media post, or on a random photo
                                        +- collage     two-panel comparison meme next to an unrelated panel
                                        +- cutout      foreground (auto-masked) pasted on a brand-new background

Outputs (for each split):
    <out>/<split>/<class>/<class>_00000.png   individual images
    <out>/<split>.bin                         CIFAR-10 binary format (label byte + R,G,B planes), shuffled
    <out>/<split>.csv                         row i of the .bin -> png path, label, seed image, layout
"""

from __future__ import annotations

import argparse
import csv
import io
import os
import sys
import time
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageOps

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import CLASSES, DATA, IMAGE_SIZE  # noqa: E402
from imgutil import (get_font, list_images, load_rgb, render_emoji, to_lenet,  # noqa: E402
                     write_cifar_bin)

WORK = 128         # working canvas; the final image is an antialiased downsample of this
SEED_SIDE = 256    # seeds/backgrounds are pre-shrunk so their longer side is at most this

LAYOUTS = {"fill": 0.40, "letterbox": 0.08, "caption": 0.14, "screenshot": 0.18, "collage": 0.06,
           "cutout": 0.14}


# --------------------------------------------------------------------------------------------
# little random helpers (everything takes an explicit numpy Generator -> reproducible)
# --------------------------------------------------------------------------------------------
def U(rng, a, b):
    return float(rng.uniform(a, b))


def chance(rng, p):
    return rng.random() < p


def pick(rng, seq):
    return seq[int(rng.integers(len(seq)))]


def rand_color(rng, lo=0, hi=256):
    return tuple(int(c) for c in rng.integers(lo, hi, 3))


# --------------------------------------------------------------------------------------------
# random (class-agnostic!) meme captions.  The same text distribution is used for every class,
# so the network can't cheat by reading the caption - it has to look at the picture.
# --------------------------------------------------------------------------------------------
_WHO = ["me", "my teacher", "my mom", "the group chat", "bro", "my friend", "my dog", "nobody",
        "everyone", "my brain", "the wifi", "my sister", "the substitute", "gym class", "math class",
        "my code", "the neural network", "my little brother", "the bus driver", "the principal",
        "my cat", "the whole class", "chemistry", "my phone", "grandma", "the cafeteria"]
_DOING = ["seeing", "after", "trying", "explaining", "ignoring", "waiting for", "about to start",
          "realizing", "remembering", "finding", "skipping", "studying for", "pretending to do"]
_WHAT = ["homework", "monday", "the test", "the bell", "free food", "a pop quiz", "summer break",
         "the weekend", "lunch", "3am", "the final exam", "minecraft", "roblox", "fortnite", "the wifi",
         "a spider", "backprop", "my grades", "the loss curve", "dad jokes", "pizza", "the group project",
         "gradient descent", "a fire drill", "school photos", "the curve", "friday", "the sequel"]
_FORMS = ["{w} {d} {x}", "pov: {w} {d} {x}", "nobody:\n{w}:", "{w} be like", "when {x} hits",
          "{w} after {x}", "me explaining {x} to {w}", "not {x} again", "how {w} sees {x}",
          "{x} speedrun", "{w} vs {x}", "average {x} enjoyer", "it's {x} time", "bro really said {x}",
          "they don't know about {x}", "{x} >>>", "{w} when {x}", "me: {d} {x}\n{w}:", "{x}??",
          "real ones remember {x}", "{w} at {x}", "{x} be like", "no {x}?", "{w} {d} {x} fr"]


def random_caption(rng, upper=True):
    t = pick(rng, _FORMS).format(w=pick(rng, _WHO), d=pick(rng, _DOING), x=pick(rng, _WHAT))
    return t.upper() if upper else t


def wrap(text, font, max_w):
    lines = []
    for para in text.split("\n"):
        line = ""
        for word in para.split():
            trial = (line + " " + word).strip()
            if not line or font.getlength(trial) <= max_w:
                line = trial
            else:
                lines.append(line)
                line = word
        lines.append(line)
    return lines


def draw_lines(draw, lines, font, y, width, fill, stroke=0, stroke_fill=None, align="center", margin=4):
    step = int(font.size * 1.08) + 1
    for ln in lines:
        w = font.getlength(ln)
        x = (width - w) / 2 if align == "center" else margin
        draw.text((x, y), ln, font=font, fill=fill, stroke_width=stroke, stroke_fill=stroke_fill)
        y += step
    return y


# --------------------------------------------------------------------------------------------
# masks & backgrounds
# --------------------------------------------------------------------------------------------
def _fill_runs(seed, cand):
    """Grow `seed` along rows through `cand` pixels: any horizontal run of candidate pixels that
    touches a seed pixel becomes filled (vectorised flood-fill step)."""
    h, w = cand.shape
    starts = ~cand
    starts[:, 0] = True
    run = np.cumsum(starts.ravel())
    hit = np.zeros(run[-1] + 1, bool)
    hit[run[(seed & cand).ravel()]] = True
    return (hit[run] & cand.ravel()).reshape(h, w)


def auto_mask(im: Image.Image, tol: int = 45) -> Image.Image | None:
    """Foreground mask for images on a plain background (stickers, renders, clip-art).

    Classic CV: take the median colour of the image border, call every pixel within `tol` of it
    "background-coloured", flood-fill from the border through those pixels, and whatever the fill
    can't reach is the object.  Returns None when the border isn't a uniform colour."""
    a = np.asarray(im, dtype=np.int16)
    h, w, _ = a.shape
    border = np.concatenate([a[0], a[-1], a[:, 0], a[:, -1]])
    ref = np.median(border, axis=0)
    if np.mean(np.abs(border - ref).sum(1) < tol) < 0.92:
        return None
    cand = np.abs(a - ref).sum(2) < tol
    bg = np.zeros_like(cand)
    bg[0], bg[-1], bg[:, 0], bg[:, -1] = cand[0], cand[-1], cand[:, 0], cand[:, -1]
    for _ in range(50):                      # alternate row / column sweeps until stable
        nxt = _fill_runs(bg, cand)
        nxt = _fill_runs(nxt.T.copy(), cand.T.copy()).T
        if (nxt == bg).all():
            break
        bg = nxt
    fg = ~bg
    if not 0.08 < fg.mean() < 0.85:
        return None
    m = Image.fromarray(fg.astype(np.uint8) * 255)
    return m.filter(ImageFilter.MinFilter(3)).filter(ImageFilter.GaussianBlur(0.8))


def random_background(rng, size, bgs):
    r = rng.random()
    if bgs and r < 0.62:                                   # crop of a random real photo
        bg = pick(rng, bgs)
        s = U(rng, 0.45, 1.0)
        cw, ch = max(8, int(bg.width * s)), max(8, int(bg.height * s))
        x, y = int(rng.integers(0, bg.width - cw + 1)), int(rng.integers(0, bg.height - ch + 1))
        im = bg.crop((x, y, x + cw, y + ch)).resize((size, size), Image.BILINEAR)
        return ImageOps.mirror(im) if chance(rng, 0.5) else im
    if r < 0.76:                                           # flat colour
        c = pick(rng, [(255, 255, 255), (0, 0, 0), rand_color(rng)])
        return Image.new("RGB", (size, size), c)
    if r < 0.9:                                            # linear gradient
        c1, c2 = np.array(rand_color(rng), float), np.array(rand_color(rng), float)
        ang = U(rng, 0, 2 * np.pi)
        yy, xx = np.mgrid[0:size, 0:size] / max(1, size - 1)
        t = np.clip((xx * np.cos(ang) + yy * np.sin(ang) + 1) / 2, 0, 1)[..., None]
        return Image.fromarray((c1 * (1 - t) + c2 * t).astype(np.uint8))
    # smooth random blobs: upsampled low-res noise
    k = int(rng.integers(2, 7))
    small = rng.integers(0, 256, (k, k, 3), dtype=np.uint8)
    return Image.fromarray(small).resize((size, size), Image.BICUBIC)


# --------------------------------------------------------------------------------------------
# geometry helpers
# --------------------------------------------------------------------------------------------
def random_view(im, rng):
    """RandomResizedCrop + horizontal flip of the seed image."""
    if chance(rng, 0.3):
        im = ImageOps.mirror(im)
    if chance(rng, 0.25):
        return im
    area = im.width * im.height
    for _ in range(10):
        target = U(rng, 0.40, 1.0) * area
        ratio = np.exp(U(rng, np.log(3 / 4), np.log(4 / 3)))
        w, h = int(round(np.sqrt(target * ratio))), int(round(np.sqrt(target / ratio)))
        if 0 < w <= im.width and 0 < h <= im.height:
            x, y = int(rng.integers(0, im.width - w + 1)), int(rng.integers(0, im.height - h + 1))
            return im.crop((x, y, x + w, y + h))
    return im


def fit_contain(im, w, h):
    s = min(w / im.width, h / im.height)
    return im.resize((max(1, round(im.width * s)), max(1, round(im.height * s))), Image.BILINEAR)


def fit_cover(im, w, h):
    s = max(w / im.width, h / im.height)
    im = im.resize((max(w, round(im.width * s)), max(h, round(im.height * s))), Image.BILINEAR)
    x, y = (im.width - w) // 2, (im.height - h) // 2
    return im.crop((x, y, x + w, y + h))


def perspective_coeffs(src, dst):
    """Solve the 8 coefficients PIL's PERSPECTIVE transform needs (output point -> input point)."""
    A, b = [], []
    for (X, Y), (x, y) in zip(src, dst):
        A.append([x, y, 1, 0, 0, 0, -X * x, -X * y])
        A.append([0, 0, 0, x, y, 1, -Y * x, -Y * y])
        b += [X, Y]
    return np.linalg.solve(np.array(A, float), np.array(b, float)).tolist()


def paste_rotated(canvas, im, mask, xy, angle, shadow=False):
    if mask is None:
        mask = Image.new("L", im.size, 255)
    if angle:
        im = im.rotate(angle, Image.BILINEAR, expand=True)
        mask = mask.rotate(angle, Image.BILINEAR, expand=True)
    if shadow:
        sh = Image.new("RGB", im.size, (0, 0, 0))
        smask = mask.filter(ImageFilter.GaussianBlur(3)).point(lambda v: v * 0.55)
        canvas.paste(sh, (xy[0] + 3, xy[1] + 3), smask)
    canvas.paste(im, xy, mask)


# --------------------------------------------------------------------------------------------
# LAYOUTS - each returns a WORK x WORK RGB image
# --------------------------------------------------------------------------------------------
def layout_fill(view, rng, bgs):
    return view.resize((WORK, WORK), Image.BILINEAR)


def layout_letterbox(view, rng, bgs):
    pic = fit_contain(view, WORK, WORK)
    style = rng.random()
    if style < 0.45:
        canvas = Image.new("RGB", (WORK, WORK), (0, 0, 0))
    elif style < 0.75:
        canvas = Image.new("RGB", (WORK, WORK), (255, 255, 255))
    else:  # blurred, stretched copy of itself (how phones show non-square video)
        canvas = view.resize((WORK, WORK), Image.BILINEAR).filter(ImageFilter.GaussianBlur(6))
    canvas.paste(pic, ((WORK - pic.width) // 2, (WORK - pic.height) // 2))
    return canvas


def layout_caption(view, rng, bgs):
    """'Modern' meme: a plain text bar above (and sometimes below) the picture."""
    dark = chance(rng, 0.2)
    bg, fg = ((0, 0, 0), (255, 255, 255)) if dark else ((255, 255, 255), (0, 0, 0))
    top = int(WORK * U(rng, 0.16, 0.38))
    bottom = int(WORK * U(rng, 0.10, 0.22)) if chance(rng, 0.2) else 0
    canvas = Image.new("RGB", (WORK, WORK), bg)
    canvas.paste(fit_cover(view, WORK, WORK - top - bottom), (0, top))
    d = ImageDraw.Draw(canvas)
    font = get_font(pick(rng, ["arial", "arial_bold", "times", "comic"]), int(WORK * U(rng, 0.065, 0.1)))
    align = pick(rng, ["center", "left"])
    for y0, h in ((0, top), (WORK - bottom, bottom)):
        if h <= 0:
            continue
        lines = wrap(random_caption(rng, upper=chance(rng, 0.3)), font, WORK * 0.92)
        lines = lines[:max(1, (h - 2) // (int(font.size * 1.08) + 1))]
        y = y0 + (h - len(lines) * (int(font.size * 1.08) + 1)) / 2
        draw_lines(d, lines, font, y, WORK, fg, align=align)
    return canvas


def layout_screenshot(view, rng, bgs):
    if chance(rng, 0.5):
        return _social_post(view, rng)
    # the meme as a picture on top of some random photo / colour (repost, photo of a screen, sticker)
    canvas = random_background(rng, WORK, bgs)
    s = U(rng, 0.5, 0.92)
    pic = fit_contain(view, int(WORK * s), int(WORK * s))
    if chance(rng, 0.3):
        pic = ImageOps.expand(pic, border=int(rng.integers(2, 6)), fill=pick(rng, [(255, 255, 255), (0, 0, 0)]))
    angle = U(rng, -14, 14) if chance(rng, 0.45) else 0
    x = int(rng.integers(-pic.width // 8, WORK - pic.width * 7 // 8 + 1))
    y = int(rng.integers(-pic.height // 8, WORK - pic.height * 7 // 8 + 1))
    paste_rotated(canvas, pic, None, (x, y), angle, shadow=chance(rng, 0.4))
    return canvas


def _social_post(view, rng):
    """A fake social-media post screenshot: avatar + name, maybe a caption, the image, icon row."""
    dark = chance(rng, 0.35)
    bg = (21, 24, 28) if dark else (255, 255, 255)
    ink = (225, 228, 232) if dark else (20, 22, 25)
    grey = (110, 118, 125)
    canvas = Image.new("RGB", (WORK, WORK), bg)
    d = ImageDraw.Draw(canvas)
    head = int(WORK * U(rng, 0.10, 0.16))
    r = head // 3
    d.ellipse([4, head // 2 - r, 4 + 2 * r, head // 2 + r], fill=rand_color(rng, 40, 230))
    nx = 8 + 2 * r
    d.rectangle([nx, head // 2 - r + 1, nx + int(WORK * U(rng, 0.2, 0.45)), head // 2 - 1], fill=ink)
    d.rectangle([nx, head // 2 + 2, nx + int(WORK * U(rng, 0.15, 0.3)), head // 2 + r - 2], fill=grey)
    y = head
    if chance(rng, 0.6):  # post text
        font = get_font(pick(rng, ["arial", "arial_bold"]), int(WORK * U(rng, 0.055, 0.075)))
        lines = wrap(random_caption(rng, upper=False), font, WORK - 10)[:2]
        y = draw_lines(d, lines, font, y + 1, WORK, ink, align="left", margin=5) + 2
    foot = int(WORK * U(rng, 0.06, 0.12))
    m = int(rng.integers(0, 8))
    area_w, area_h = WORK - 2 * m, max(10, WORK - y - foot)
    pic = fit_cover(view, area_w, area_h) if chance(rng, 0.6) else fit_contain(view, area_w, area_h)
    canvas.paste(pic, (m + (area_w - pic.width) // 2, y + (area_h - pic.height) // 2))
    fy = WORK - foot // 2
    for i in range(4):  # like / comment / share icons
        cx = int(WORK * (0.12 + 0.25 * i))
        d.ellipse([cx - 3, fy - 3, cx + 3, fy + 3], outline=grey, width=1)
    return canvas


def layout_collage(view, rng, bgs):
    """Two-panel comparison meme ("2025 vs 2026", before/after): the class picture in one panel,
    an unrelated photo or a plain text panel in the other."""
    horizontal = chance(rng, 0.5)                  # side by side, or stacked
    a = int(WORK * U(rng, 0.5, 0.65))              # the class picture gets the bigger share
    first = chance(rng, 0.5)                       # class picture on the left/top or right/bottom
    if horizontal:
        main_box, other_box = ((0, 0, a, WORK), (a, 0, WORK, WORK)) if first else \
                              ((WORK - a, 0, WORK, WORK), (0, 0, WORK - a, WORK))
    else:
        main_box, other_box = ((0, 0, WORK, a), (0, a, WORK, WORK)) if first else \
                              ((0, WORK - a, WORK, WORK), (0, 0, WORK, WORK - a))
    canvas = Image.new("RGB", (WORK, WORK), (255, 255, 255))
    mw, mh = main_box[2] - main_box[0], main_box[3] - main_box[1]
    ow, oh = other_box[2] - other_box[0], other_box[3] - other_box[1]
    canvas.paste(fit_cover(view, mw, mh), main_box[:2])
    if chance(rng, 0.6):
        canvas.paste(random_background(rng, max(ow, oh), bgs).resize((ow, oh), Image.BILINEAR), other_box[:2])
    else:
        d = ImageDraw.Draw(canvas)
        d.rectangle(other_box, fill=pick(rng, [(255, 255, 255), (0, 0, 0), rand_color(rng)]))
        font = get_font(pick(rng, ["impact", "arial_bold"]), int(WORK * U(rng, 0.07, 0.11)))
        lines = wrap(random_caption(rng), font, ow * 0.9)[:3]
        y = other_box[1] + (oh - len(lines) * (int(font.size * 1.08) + 1)) / 2
        for ln in lines:
            x = other_box[0] + (ow - font.getlength(ln)) / 2
            d.text((x, y), ln, font=font, fill=(255, 255, 255), stroke_width=2, stroke_fill=(0, 0, 0))
            y += int(font.size * 1.08) + 1
    if chance(rng, 0.7):                           # divider line between the panels
        d = ImageDraw.Draw(canvas)
        if horizontal:
            x = main_box[2] if first else main_box[0]
            d.line([(x, 0), (x, WORK)], fill=(0, 0, 0), width=2)
        else:
            y = main_box[3] if first else main_box[1]
            d.line([(0, y), (WORK, y)], fill=(0, 0, 0), width=2)
    return canvas


def layout_cutout(seed, mask, rng, bgs):
    """Paste the auto-masked foreground (character / person) onto a brand new background."""
    canvas = random_background(rng, WORK, bgs)
    box = mask.getbbox()
    obj, m = seed.crop(box), mask.crop(box)
    if chance(rng, 0.3):
        obj, m = ImageOps.mirror(obj), ImageOps.mirror(m)
    s = min(U(rng, 0.55, 1.0) * WORK / obj.height, U(rng, 0.55, 1.05) * WORK / obj.width)
    size = (max(4, round(obj.width * s)), max(4, round(obj.height * s)))
    obj, m = obj.resize(size, Image.BILINEAR), m.resize(size, Image.BILINEAR)
    x = int(rng.integers(-obj.width // 6, WORK - obj.width * 5 // 6 + 1))
    y = int(rng.integers(min(0, WORK - obj.height), max(0, WORK - obj.height) + 1))
    paste_rotated(canvas, obj, m, (x, y), U(rng, -15, 15) if chance(rng, 0.35) else 0,
                  shadow=chance(rng, 0.3))
    return canvas


# --------------------------------------------------------------------------------------------
# text & overlays (also class-agnostic)
# --------------------------------------------------------------------------------------------
def impact_caption(im, rng):
    """Classic image macro: white Impact letters with a black outline at the top and/or bottom."""
    d = ImageDraw.Draw(im)
    size = int(WORK * U(rng, 0.09, 0.16))
    font = get_font("impact", size)
    stroke = max(1, size // 10)
    where = pick(rng, ["top", "bottom", "both", "both"])
    for pos in (("top", "bottom") if where == "both" else (where,)):
        lines = wrap(random_caption(rng), font, WORK * 0.94)[:2]
        h = len(lines) * (int(size * 1.08) + 1)
        y = int(WORK * 0.02) if pos == "top" else WORK - h - int(WORK * 0.03)
        draw_lines(d, lines, font, y, WORK, (255, 255, 255), stroke=stroke, stroke_fill=(0, 0, 0))
    return im


_EMOJI = ["\U0001F602", "\U0001F480", "\U0001F525", "\U0001F62D", "\U0001F5FF", "\U0001F923", "\U0001F633",
          "\U0001F976", "\U0001F4AF", "\U0001F440", "\U0001F92F", "\U0001F60E", "\U0001F64F", "✨",
          "\U0001F921", "\U0001F608", "\U0001F44D", "\U0001F9E2", "\U0001FAE1", "❤️"]
_HANDLES = ["imgflip.com", "@memes", "@daily.memes", "tiktok", "@user8231", "reddit", "ifunny.co",
            "@brainrot.hq", "made with mematic", "@schoolmemes"]


def overlays(im, rng):
    d = ImageDraw.Draw(im)
    if chance(rng, 0.12):                       # emoji stickers
        for _ in range(int(rng.integers(1, 4))):
            e = render_emoji(pick(rng, _EMOJI))
            if e is None:
                break
            s = int(WORK * U(rng, 0.12, 0.3))
            e = e.resize((s, s), Image.BILINEAR)
            im.paste(e, (int(rng.integers(-s // 4, WORK - s * 3 // 4)), int(rng.integers(-s // 4, WORK - s * 3 // 4))), e)
    if chance(rng, 0.2):                        # watermark / username
        font = get_font(pick(rng, ["arial", "arial_bold"]), int(WORK * U(rng, 0.045, 0.07)))
        text = pick(rng, _HANDLES)
        x = 2 if chance(rng, 0.6) else WORK - font.getlength(text) - 2
        y = WORK - font.size - 2 if chance(rng, 0.8) else 2
        col = pick(rng, [(255, 255, 255), (200, 200, 200), (0, 0, 0)])
        d.text((x, y), text, font=font, fill=col, stroke_width=1 if chance(rng, 0.5) else 0,
               stroke_fill=(0, 0, 0) if col != (0, 0, 0) else (255, 255, 255))
    if chance(rng, 0.06):                       # short-video UI: icon column + username + caption
        for i in range(4):
            cy = int(WORK * (0.45 + 0.12 * i))
            d.ellipse([WORK - 14, cy - 5, WORK - 4, cy + 5], fill=(240, 240, 240))
        font = get_font("arial_bold", int(WORK * 0.06))
        d.text((4, WORK - 2 * font.size - 6), pick(rng, _HANDLES), font=font, fill=(255, 255, 255))
        d.text((4, WORK - font.size - 4), random_caption(rng, upper=False)[:22], font=font, fill=(255, 255, 255))
    if chance(rng, 0.2):                        # random erasing / occluder
        area = U(rng, 0.02, 0.12) * WORK * WORK
        ratio = np.exp(U(rng, -1, 1))
        w, h = int(np.sqrt(area * ratio)), int(np.sqrt(area / ratio))
        x, y = int(rng.integers(0, max(1, WORK - w))), int(rng.integers(0, max(1, WORK - h)))
        if chance(rng, 0.5):
            d.rectangle([x, y, x + w, y + h], fill=rand_color(rng))
        else:
            noise = rng.integers(0, 256, (max(1, h), max(1, w), 3), dtype=np.uint8)
            im.paste(Image.fromarray(noise), (x, y))
    return im


# --------------------------------------------------------------------------------------------
# camera / colour / "internet damage"
# --------------------------------------------------------------------------------------------
def geometric_jitter(im, rng):
    if chance(rng, 0.2):                        # small tilt, then zoom so no corners show
        ang = U(rng, -10, 10)
        im = im.rotate(ang, Image.BILINEAR, fillcolor=pick(rng, [(0, 0, 0), (255, 255, 255)]))
        if chance(rng, 0.7):
            c = abs(np.cos(np.radians(ang))) + abs(np.sin(np.radians(ang)))
            k = int(WORK * (1 - 1 / c) / 2) + 1
            im = im.crop((k, k, WORK - k, WORK - k)).resize((WORK, WORK), Image.BILINEAR)
    if chance(rng, 0.1):                        # perspective: photo of a screen taken at an angle
        j = WORK * U(rng, 0.04, 0.14)
        dst = [(U(rng, 0, j), U(rng, 0, j)), (WORK - U(rng, 0, j), U(rng, 0, j)),
               (WORK - U(rng, 0, j), WORK - U(rng, 0, j)), (U(rng, 0, j), WORK - U(rng, 0, j))]
        src = [(0, 0), (WORK, 0), (WORK, WORK), (0, WORK)]
        im = im.transform((WORK, WORK), Image.PERSPECTIVE, perspective_coeffs(src, dst),
                          Image.BILINEAR, fillcolor=rand_color(rng))
    return im


def hue_shift(im, degrees):
    hsv = np.asarray(im.convert("HSV")).copy()
    hsv[..., 0] = (hsv[..., 0].astype(np.int16) + int(degrees / 360 * 256)) % 256
    return Image.fromarray(hsv, "HSV").convert("RGB")


def color_jitter(im, rng):
    if chance(rng, 0.8):
        im = ImageEnhance.Brightness(im).enhance(U(rng, 0.65, 1.35))
    if chance(rng, 0.8):
        im = ImageEnhance.Contrast(im).enhance(U(rng, 0.65, 1.4))
    if chance(rng, 0.7):
        im = ImageEnhance.Color(im).enhance(U(rng, 0.5, 1.5))
    if chance(rng, 0.3):
        im = hue_shift(im, U(rng, -12, 12))
    if chance(rng, 0.15):
        g = U(rng, 0.75, 1.35)
        im = im.point([int(255 * (i / 255) ** g) for i in range(256)] * 3)
    if chance(rng, 0.05):
        im = ImageOps.grayscale(im).convert("RGB")
    elif chance(rng, 0.05):                     # colour cast (warm/cool/sepia-ish filter)
        im = Image.blend(im, Image.new("RGB", im.size, rand_color(rng)), U(rng, 0.1, 0.3))
    return im


def motion_blur(im, rng):
    n = int(rng.integers(3, 10))
    th = U(rng, 0, np.pi)
    a = np.asarray(im, dtype=np.float32)
    p = n
    padded = np.pad(a, ((p, p), (p, p), (0, 0)), mode="edge")
    acc = np.zeros_like(a)
    for t in np.linspace(-(n - 1) / 2, (n - 1) / 2, n):
        dx, dy = int(round(t * np.cos(th))), int(round(t * np.sin(th)))
        acc += padded[p + dy:p + dy + a.shape[0], p + dx:p + dx + a.shape[1]]
    return Image.fromarray(np.clip(acc / n, 0, 255).astype(np.uint8))


def jpeg(im, quality):
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=int(quality))
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def add_noise(im, rng, sigma):
    a = np.asarray(im, dtype=np.float32)
    a = a + rng.normal(0, sigma, a.shape if chance(rng, 0.7) else a.shape[:2] + (1,))
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))


def deep_fry(im, rng):
    """The 'deep-fried meme' filter: crushed contrast, nuclear saturation, sharpen, awful JPEG."""
    im = ImageEnhance.Color(im).enhance(U(rng, 2.0, 3.5))
    im = ImageEnhance.Contrast(im).enhance(U(rng, 1.5, 2.2))
    im = ImageEnhance.Sharpness(im).enhance(U(rng, 2.0, 5.0))
    im = add_noise(im, rng, U(rng, 5, 15))
    return jpeg(im, rng.integers(4, 15))


def degrade(im, rng):
    if chance(rng, 0.04):
        return deep_fry(im, rng)
    if chance(rng, 0.25):
        im = im.filter(ImageFilter.GaussianBlur(U(rng, 0.3, 1.8)))
    elif chance(rng, 0.08):
        im = motion_blur(im, rng)
    elif chance(rng, 0.1):
        im = im.filter(ImageFilter.UnsharpMask(radius=2, percent=int(rng.integers(60, 200))))
    if chance(rng, 0.1):                        # low-res repost: shrink then blow back up
        k = U(rng, 2, 4.5)
        im = im.resize((int(WORK / k), int(WORK / k)), Image.BILINEAR).resize((WORK, WORK), Image.BILINEAR)
    if chance(rng, 0.3):
        im = add_noise(im, rng, U(rng, 2, 14))
    if chance(rng, 0.55):
        im = jpeg(im, rng.integers(12, 93))
    return im


# --------------------------------------------------------------------------------------------
# the recipe
# --------------------------------------------------------------------------------------------
_LAYOUT_FN = {"fill": layout_fill, "letterbox": layout_letterbox, "caption": layout_caption,
              "screenshot": layout_screenshot, "collage": layout_collage}
_LAYOUT_NAMES = list(LAYOUTS)
_LAYOUT_P = np.array(list(LAYOUTS.values())) / sum(LAYOUTS.values())


def synthesize_one(rng, seed, mask, bgs, size, trace=None):
    """One synthetic image from one seed.  If `trace` is a list, (stage name, image) pairs are
    appended to it after every stage (used by preview.py to draw the pipeline)."""
    def log(name, im):
        if trace is not None:
            trace.append((name, im.copy()))
        return im

    layout = _LAYOUT_NAMES[int(rng.choice(len(_LAYOUT_NAMES), p=_LAYOUT_P))]
    if layout == "cutout" and mask is None:
        layout = pick(rng, ["fill", "screenshot"])
    if layout == "cutout":
        im = layout_cutout(seed, mask, rng, bgs)
    else:
        im = _LAYOUT_FN[layout](random_view(seed, rng), rng, bgs)
    log(f"layout: {layout}", im)
    if layout != "caption" and chance(rng, 0.3):
        im = impact_caption(im, rng)
    im = log("meme text + overlays", overlays(im, rng))
    im = geometric_jitter(im, rng)
    im = log("tilt + colour", color_jitter(im, rng))
    im = log("blur / noise / JPEG", degrade(im, rng))
    return log(f"{size}x{size} (what LeNet sees)", to_lenet(im, size)), layout


# --------------------------------------------------------------------------------------------
# multiprocessing plumbing
# --------------------------------------------------------------------------------------------
_G: dict = {}


def _init_worker(seeds, backgrounds, size, base_seed, out_root):
    _G.update(seeds=seeds, bgs=backgrounds, size=size, base_seed=base_seed, out=out_root)


def _work(task):
    split_id, split, ci, i = task
    rng = np.random.default_rng([_G["base_seed"], split_id, ci, i])
    pool = _G["seeds"][ci]
    name, seed, mask = pool[int(rng.integers(len(pool)))]
    im, layout = synthesize_one(rng, seed, mask, _G["bgs"], _G["size"])
    cname = CLASSES[ci]["name"]
    rel = f"{split}/{cname}/{cname}_{i:05d}.png"
    im.save(Path(_G["out"]) / rel, optimize=False)
    return ci, i, rel, name, layout, np.asarray(im, dtype=np.uint8)


def load_pool(seed_dir: Path, bg_dir: Path, use_masks: bool = True):
    seeds = []
    for c in CLASSES:
        items = []
        for p in list_images(seed_dir / c["name"]):
            im = load_rgb(p)
            im.thumbnail((SEED_SIDE, SEED_SIDE), Image.LANCZOS)
            items.append((p.name, im, auto_mask(im) if use_masks else None))
        if not items:
            sys.exit(f"no seed images found in {seed_dir / c['name']}")
        seeds.append(items)
    bgs = []
    if bg_dir.exists():
        for p in list_images(bg_dir):
            im = load_rgb(p)
            im.thumbnail((SEED_SIDE, SEED_SIDE), Image.LANCZOS)
            bgs.append(im)
    return seeds, bgs


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=Path, default=DATA / "raw" / "seeds")
    ap.add_argument("--backgrounds", type=Path, default=DATA / "raw" / "backgrounds")
    ap.add_argument("--out", type=Path, default=DATA)
    ap.add_argument("--train-per-class", type=int, default=3500)
    ap.add_argument("--val-per-class", type=int, default=350)
    ap.add_argument("--size", type=int, default=IMAGE_SIZE, help="output resolution (LeNet-5: 32)")
    ap.add_argument("--seed", type=int, default=67, help="global random seed (reproducible output)")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--no-masks", action="store_true", help="disable the cutout layout")
    args = ap.parse_args()

    t0 = time.time()
    seeds, bgs = load_pool(args.seeds, args.backgrounds, not args.no_masks)
    for c, pool in zip(CLASSES, seeds):
        print(f"  {c['name']:24s} {len(pool):4d} seed images ({sum(m is not None for _, _, m in pool)} with auto-masks)")
    print(f"  {len(bgs)} background photos   [{time.time() - t0:.1f}s]")

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "classes.txt").write_text("\n".join(c["name"] for c in CLASSES) + "\n")
    splits = [("train", args.train_per_class), ("val", args.val_per_class)]
    with Pool(args.workers, initializer=_init_worker,
              initargs=(seeds, bgs, args.size, args.seed, str(args.out))) as pool:
        for split_id, (split, n) in enumerate(splits):
            if n <= 0:
                continue
            for c in CLASSES:
                d = args.out / split / c["name"]
                d.mkdir(parents=True, exist_ok=True)
                for old in d.glob(f"{c['name']}_*.png"):   # remove images from a previous run
                    old.unlink()
            tasks = [(split_id, split, ci, i) for ci in range(len(CLASSES)) for i in range(n)]
            t1 = time.time()
            results = []
            for k, r in enumerate(pool.imap_unordered(_work, tasks, chunksize=32)):
                results.append(r)
                if (k + 1) % 5000 == 0:
                    print(f"  {split}: {k + 1}/{len(tasks)}  ({time.time() - t1:.0f}s)")
            results.sort(key=lambda r: (r[0], r[1]))
            order = np.random.default_rng([args.seed, 999, split_id]).permutation(len(results))
            images = np.stack([results[j][5] for j in order])
            labels = np.array([results[j][0] for j in order], dtype=np.uint8)
            write_cifar_bin(args.out / f"{split}.bin", images, labels)
            with open(args.out / f"{split}.csv", "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["index", "path", "label", "class", "seed_image", "layout"])
                for row, j in enumerate(order):
                    ci, _, rel, seed_name, layout, _ = results[j]
                    w.writerow([row, rel, ci, CLASSES[ci]["name"], seed_name, layout])
            lay = Counter(r[4] for r in results)
            print(f"  {split}: wrote {len(results)} images in {time.time() - t1:.0f}s  layouts={dict(lay)}")
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
