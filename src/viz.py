"""
viz.py -- pictures for the lab, in pure Python (a tiny PNG encoder + inline HTML/SVG).

This file is NOT part of the neural network.  It only draws things in the notebook:
images, feature maps, layouts, charts, predictions.  No matplotlib or Pillow needed.
"""

import base64
import colorsys
import html
import math
import struct
import zlib
from pathlib import Path

CARD = ("background:#ffffff;color:#1d1d1f;border:1px solid #d9d9de;border-radius:10px;padding:10px 12px;"
        "margin:6px 0;display:inline-block;font:13px/1.35 -apple-system,Segoe UI,Helvetica,Arial,sans-serif;")
INK, MUTED, ACCENT = "#1d1d1f", "#6e6e73", "#2f6fdf"


def _show(markup):
    try:
        from IPython.display import HTML, display
        display(HTML(markup))
    except ImportError:  # running as a plain script
        print("(open the notebook in Jupyter to see this picture)")


# ----------------------------------------------------------------------------------------------- PNG
def png_bytes(rgb, width, height):
    """Encode raw RGB bytes (row by row, 3 bytes per pixel) as a PNG file."""
    raw = b"".join(b"\x00" + bytes(rgb[y * width * 3:(y + 1) * width * 3]) for y in range(height))

    def chunk(tag, data):
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b""))


def _to_rgb(pixels, C, H, W, normalize):
    """CHW values -> interleaved RGB bytes.  normalize=True stretches min..max to 0..255 (for floats)."""
    vals = list(pixels)
    if normalize:
        lo, hi = min(vals), max(vals)
        span = (hi - lo) or 1.0
        vals = [(v - lo) / span * 255 for v in vals]
    elif vals and isinstance(vals[0], float) and max(vals) <= 1.0001:
        vals = [v * 255 for v in vals]
    clamp = [max(0, min(255, int(round(v)))) for v in vals]
    out = bytearray()
    for y in range(H):
        for x in range(W):
            if C == 1:
                g = clamp[y * W + x]
                out += bytes((g, g, g))
            else:
                out += bytes(clamp[c * H * W + y * W + x] for c in range(3))
    return bytes(out)


def _img_tag(pixels, C=3, H=32, W=32, scale=4, normalize=False, title=""):
    data = base64.b64encode(png_bytes(_to_rgb(pixels, C, H, W, normalize), W, H)).decode()
    return (f'<img src="data:image/png;base64,{data}" width="{W * scale}" height="{H * scale}" '
            f'title="{html.escape(title)}" style="image-rendering:pixelated;border-radius:3px;display:block">')


def _figure(img, caption):
    cap = f'<div style="color:{MUTED};font-size:12px;margin-top:3px;text-align:center">{caption}</div>' if caption else ""
    return f'<div style="display:inline-block;margin:4px;vertical-align:top">{img}{cap}</div>'


def show_image(pixels, C=3, H=32, W=32, scale=6, caption=None, normalize=False):
    """Show one image given as flat CHW pixels (bytes 0..255, or floats)."""
    _show(f'<div style="{CARD}">' + _figure(_img_tag(pixels, C, H, W, scale, normalize), html.escape(caption or "")) + "</div>")


def show_images(items, scale=3, title=None, C=3, H=32, W=32, normalize=False):
    """Show a row/grid of images: items = [(pixels, caption), ...]."""
    head = f'<div style="font-weight:600;margin-bottom:4px">{html.escape(title)}</div>' if title else ""
    cells = "".join(_figure(_img_tag(p, C, H, W, scale, normalize), html.escape(str(c))) for p, c in items)
    _show(f'<div style="{CARD};max-width:100%">{head}{cells}</div>')


def show_maps(values, C, H, W, scale=4, title=None, labels=None):
    """Show C feature maps (a flat C*H*W list) as grey images, each stretched to its own min..max."""
    items = []
    for c in range(C):
        plane = values[c * H * W:(c + 1) * H * W]
        items.append((_img_tag(plane, 1, H, W, scale, normalize=True), labels[c] if labels else f"map {c}"))
    head = f'<div style="font-weight:600;margin-bottom:4px">{html.escape(title)}</div>' if title else ""
    _show(f'<div style="{CARD}">{head}' + "".join(_figure(i, html.escape(str(c))) for i, c in items) + "</div>")


def file_tag(path, height=120):
    """An <img> for an image file on disk (jpg / png / gif / webp)."""
    path = Path(path)
    mime = {"jpg": "jpeg", "jpeg": "jpeg", "png": "png", "gif": "gif", "webp": "webp"}.get(path.suffix[1:].lower(), "png")
    data = base64.b64encode(path.read_bytes()).decode()
    return f'<img src="data:image/{mime};base64,{data}" style="height:{height}px;border-radius:4px;display:block">'


def show_file(path, height=160, caption=None):
    _show(f'<div style="{CARD}">' + _figure(file_tag(path, height), html.escape(caption or Path(path).name)) + "</div>")


# ------------------------------------------------------------------------------------------- layouts
def _color(offset):
    h = (offset * 0.6180339887) % 1.0
    r, g, b = colorsys.hls_to_rgb(h, 0.80, 0.65)
    return f"#{int(r * 255):02x}{int(g * 255):02x}{int(b * 255):02x}"


def show_layout(layout, title=None, memory=True, max_cells=600):
    """Draw a rank-2 layout as a table of offsets.  Cells that point at the SAME memory get the SAME colour.
    memory=True also draws the memory it reaches (cosize cells), coloured the same way."""
    import babytorch as bt
    L = layout if len(layout) == 2 else bt.make_layout(bt.Layout(1, 0), layout)
    M, N = bt.size(L[0]), bt.size(L[1])
    if M * N > max_cells:
        raise ValueError(f"{M}x{N} is too big to draw; try a smaller example")
    rows_html = []
    for i in range(M):
        cells = "".join(f'<td style="background:{_color(L(i, j))};color:{INK};padding:2px 6px;text-align:right;'
                        f'border:1px solid #fff;font:12px ui-monospace,Menlo,monospace">{L(i, j)}</td>' for j in range(N))
        rows_html.append(f'<tr><th style="color:{MUTED};font:11px ui-monospace,Menlo,monospace;padding-right:6px">{i}</th>{cells}</tr>')
    head = "".join(f'<th style="color:{MUTED};font:11px ui-monospace,Menlo,monospace">{j}</th>' for j in range(N))
    table = f'<table style="border-collapse:collapse"><tr><th></th>{head}</tr>{"".join(rows_html)}</table>'
    mem = ""
    if memory:
        cos = bt.cosize(L)
        used = set(bt.offsets(L))
        boxes = "".join(f'<span style="display:inline-block;width:26px;padding:2px 0;margin:1px;text-align:center;'
                        f'background:{_color(o) if o in used else "#f2f2f5"};color:{INK if o in used else "#b0b0b8"};'
                        f'border-radius:3px;font:11px ui-monospace,Menlo,monospace">{o}</span>' for o in range(min(cos, 256)))
        more = f" … ({cos} cells)" if cos > 256 else ""
        mem = (f'<div style="margin-top:8px;color:{MUTED}">memory it reads (cosize = {cos}), size = {M * N} coordinates'
               f'</div><div style="max-width:760px">{boxes}{more}</div>')
    t = html.escape(title or repr(layout))
    _show(f'<div style="{CARD}"><div style="font:600 13px ui-monospace,Menlo,monospace;margin-bottom:6px">{t}</div>{table}{mem}</div>')


# -------------------------------------------------------------------------------------------- charts
PALETTE = ["#2f6fdf", "#e8710a", "#1e8e3e", "#d01884", "#9334e6", "#12a4af"]


def _nice_ticks(lo, hi, n=5):
    """About n round-numbered ticks (steps of 1, 2, 2.5 or 5 times a power of ten) covering lo..hi."""
    span = (hi - lo) or 1.0
    raw = span / (n - 1)
    mag = 10 ** math.floor(math.log10(raw))
    step = min((m * mag for m in (1, 2, 2.5, 5, 10)), key=lambda s: abs(s - raw))
    first = math.ceil(lo / step - 1e-9) * step
    ticks = [first + k * step for k in range(int((hi - first) / step + 1e-9) + 1)]
    return ticks, step


def _tick_label(v, step):
    if step >= 1:
        return f"{v:,.0f}"
    decimals = max(0, -math.floor(math.log10(step)) + (1 if step * 10 ** math.ceil(-math.log10(step)) % 1 else 0))
    return f"{v:.{decimals}f}"


def line_chart(series, title="", xlabel="", ylabel="", width=640, height=300, ylim=None, logy=False):
    """series = {name: [y0, y1, ...]} or {name: [(x, y), ...]}.  Draws an SVG line chart."""
    pts = {}
    for name, ys in series.items():
        ys = list(ys)
        pts[name] = ys if ys and isinstance(ys[0], (tuple, list)) else list(enumerate(ys))
    allx = [x for p in pts.values() for x, _ in p]
    ally = [y for p in pts.values() for _, y in p]
    f = (lambda v: math.log10(max(v, 1e-12))) if logy else (lambda v: v)
    x0, x1 = min(allx), max(allx)
    y0, y1 = ylim if ylim else (min(map(f, ally)), max(map(f, ally)))
    if y1 == y0:
        y1 = y0 + 1
    if x1 == x0:
        x1 = x0 + 1
    L, R, T, B = 70, 16, 52, 44
    sx = lambda x: L + (x - x0) / (x1 - x0) * (width - L - R)
    sy_raw = lambda v: T + (1 - (v - y0) / (y1 - y0)) * (height - T - B)
    sy = lambda y: sy_raw(f(y))
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" style="background:#fff">',
           f'<text x="{L}" y="20" font-size="14" font-weight="600" fill="{INK}">{html.escape(title)}</text>']
    if logy:  # ticks at powers of ten
        exps = list(range(math.ceil(y0), math.floor(y1) + 1)) or [round(y0)]
        every = max(1, -(-len(exps) // 6))
        yticks = [(e, f"{10.0 ** e:g}" if -3 <= e <= 4 else f"1e{e}") for e in exps[::every]]
    else:
        ticks, step = _nice_ticks(y0, y1)
        yticks = [(v, _tick_label(v, step)) for v in ticks]
    for yv, lab in yticks:
        y = sy_raw(yv)
        svg.append(f'<line x1="{L}" x2="{width - R}" y1="{y:.1f}" y2="{y:.1f}" stroke="#ececf0"/>'
                   f'<text x="{L - 6}" y="{y + 4:.1f}" font-size="11" fill="{MUTED}" text-anchor="end">{lab}</text>')
    xt, xstep = _nice_ticks(x0, x1)
    for xv in xt:
        svg.append(f'<text x="{sx(xv):.1f}" y="{height - B + 16}" font-size="11" fill="{MUTED}" '
                   f'text-anchor="middle">{_tick_label(xv, xstep)}</text>')
    svg.append(f'<text x="{(L + width - R) / 2}" y="{height - 8}" font-size="12" fill="{MUTED}" text-anchor="middle">{html.escape(xlabel)}</text>')
    svg.append(f'<text x="14" y="{(T + height - B) / 2}" font-size="12" fill="{MUTED}" text-anchor="middle" '
               f'transform="rotate(-90 14 {(T + height - B) / 2})">{html.escape(ylabel)}</text>')
    lx = L
    for i, (name, p) in enumerate(pts.items()):
        col = PALETTE[i % len(PALETTE)]
        d = " ".join(f"{'M' if k == 0 else 'L'}{sx(x):.1f},{sy(y):.1f}" for k, (x, y) in enumerate(p))
        svg.append(f'<path d="{d}" fill="none" stroke="{col}" stroke-width="2"/>')
        svg.append(f'<rect x="{lx}" y="30" width="10" height="10" fill="{col}" rx="2"/>'
                   f'<text x="{lx + 14}" y="39" font-size="12" fill="{INK}">{html.escape(name)}</text>')
        lx += 24 + 7 * len(name)
    svg.append("</svg>")
    _show(f'<div style="{CARD}">' + "".join(svg) + "</div>")


def bar_chart(values, labels, title="", highlight=None, width=420, fmt="{:.0%}"):
    """Horizontal bars (e.g. class probabilities).  highlight = index of the bar to colour."""
    h = 20 * len(values) + 34
    top = max(values) or 1
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{h}" style="background:#fff">',
           f'<text x="0" y="16" font-size="13" font-weight="600" fill="{INK}">{html.escape(title)}</text>']
    for i, (v, lab) in enumerate(zip(values, labels)):
        y = 26 + 20 * i
        col = ACCENT if i == highlight else "#c7cbd6"
        svg.append(f'<text x="150" y="{y + 12}" font-size="12" fill="{INK}" text-anchor="end">{html.escape(str(lab))}</text>'
                   f'<rect x="156" y="{y + 2}" width="{max(1, v / top * (width - 210)):.1f}" height="14" fill="{col}" rx="2"/>'
                   f'<text x="{160 + v / top * (width - 210):.1f}" y="{y + 13}" font-size="11" fill="{MUTED}">{fmt.format(v)}</text>')
    svg.append("</svg>")
    return "".join(svg)


def show_bars(values, labels, title="", highlight=None, fmt="{:.0%}"):
    _show(f'<div style="{CARD}">' + bar_chart(values, labels, title, highlight, fmt=fmt) + "</div>")


def show_confusion(confusion, classes, title="confusion matrix (rows = true class, columns = predicted)"):
    n = len(classes)
    top = max(max(r) for r in confusion) or 1
    head = "".join(f'<th style="writing-mode:vertical-rl;transform:rotate(180deg);font-weight:400;color:{MUTED};'
                   f'padding:2px;font-size:11px">{html.escape(c)}</th>' for c in classes)
    body = []
    for i in range(n):
        tot = sum(confusion[i]) or 1
        cells = []
        for j in range(n):
            v = confusion[i][j]
            a = v / top
            bg = f"rgba(47,111,223,{a:.2f})" if i == j else f"rgba(232,113,10,{a:.2f})"
            cells.append(f'<td style="background:{bg};color:{"#fff" if a > 0.55 else INK};text-align:center;'
                         f'min-width:26px;padding:3px;font-size:12px">{v or ""}</td>')
        body.append(f'<tr><th style="text-align:right;font-weight:400;padding-right:6px;font-size:12px">'
                    f'{html.escape(classes[i])} <span style="color:{MUTED}">{confusion[i][i] / tot:.0%}</span></th>{"".join(cells)}</tr>')
    _show(f'<div style="{CARD}"><div style="font-weight:600;margin-bottom:6px">{html.escape(title)}</div>'
          f'<table style="border-collapse:collapse"><tr><th></th>{head}</tr>{"".join(body)}</table></div>')


def show_predictions(rows_):
    """rows_ = [(original_path, pixels_32x32, probs, true_label, classes), ...]: original | what the net sees | bars."""
    out = []
    for path, pixels, probs, true, classes in rows_:
        guess = max(range(len(probs)), key=probs.__getitem__)
        ok = "right" if guess == true else "wrong"
        cap = f"{ok}, its really: <b>{html.escape(classes[true])}</b>"
        out.append(f'<div style="display:flex;gap:12px;align-items:center;margin:6px 0">'
                   f'<div>{file_tag(path, 130)}<div style="font-size:12px;margin-top:3px">{cap}</div></div>'
                   f'<div>{_img_tag(pixels, 3, 32, 32, 3)}<div style="font-size:11px;color:{MUTED}">32 x 32 input</div></div>'
                   f'<div>{bar_chart(probs, classes, "the network says", highlight=guess, width=400)}</div></div>')
    _show(f'<div style="{CARD}">' + "".join(out) + "</div>")


# ----------------------------------------------------------------------------------------- LeNet art
def lenet_diagram():
    """The LeNet-5 architecture with every tensor's shape."""
    stages = [("input", "3×32×32", "#9aa0a6"), ("conv 5×5", "6×28×28", "#2f6fdf"), ("max-pool 2", "6×14×14", "#1e8e3e"),
              ("conv 5×5", "16×10×10", "#2f6fdf"), ("max-pool 2", "16×5×5", "#1e8e3e"), ("flatten", "400", "#9334e6"),
              ("fully conn.", "120", "#e8710a"), ("fully conn.", "84", "#e8710a"), ("fully conn.", "10", "#e8710a"),
              ("softmax", "10 probs", "#d01884")]
    w, bw = 1010, 92
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="120" style="background:#fff">']
    for i, (name, shape, col) in enumerate(stages):
        x = 8 + i * 100
        svg.append(f'<rect x="{x}" y="30" width="{bw}" height="56" rx="8" fill="{col}" fill-opacity="0.12" stroke="{col}"/>'
                   f'<text x="{x + bw / 2}" y="54" font-size="12" font-weight="600" fill="{INK}" text-anchor="middle">{name}</text>'
                   f'<text x="{x + bw / 2}" y="74" font-size="12" fill="{INK}" text-anchor="middle">{shape}</text>')
        if i:
            svg.append(f'<path d="M{x - 8},58 L{x - 1},58" stroke="{MUTED}" stroke-width="1.5"/>')
    svg.append(f'<text x="8" y="18" font-size="12" fill="{MUTED}">relu after every conv and after the first two fully '
               f'connected layers · 62,006 numbers to learn</text>')
    svg.append("</svg>")
    _show(f'<div style="{CARD}">' + "".join(svg) + "</div>")
