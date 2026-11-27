"""Crisp visual PNG contact-sheet renderer for inspecting InkForge generated handwriting."""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def render_strokes_tile(
    strokes: Sequence[np.ndarray],
    label: str = "",
    *,
    width: int = 440,
    height: int = 160,
    header_h: int = 26,
    color_by_order: bool = False,
) -> Image.Image:
    """Render one handwritten sample onto a crisp off-white paper tile with its ground-truth LaTeX."""
    img = Image.new("RGB", (width, height), (252, 252, 250))
    draw = ImageDraw.Draw(img)

    # Header bar
    draw.rectangle([0, 0, width, header_h], fill=(241, 243, 247))
    draw.line([(0, header_h), (width, header_h)], fill=(218, 222, 230), width=1)
    short_lab = label if len(label) <= 58 else (label[:55] + "...")
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
    draw.text((8, 7), short_lab, fill=(35, 42, 56), font=font)

    clean = [np.asarray(s, dtype=np.float64)[:, :2] for s in strokes if len(s) > 0]
    if not clean:
        draw.text((12, header_h + 16), "(empty strokes)", fill=(200, 60, 60), font=font)
        return img

    all_pts = np.concatenate(clean, axis=0)
    finite = np.isfinite(all_pts[:, 0]) & np.isfinite(all_pts[:, 1])
    if not finite.any():
        draw.text((12, header_h + 16), "(non-finite)", fill=(200, 60, 60), font=font)
        return img

    pts_f = all_pts[finite]
    min_x, min_y = float(pts_f[:, 0].min()), float(pts_f[:, 1].min())
    max_x, max_y = float(pts_f[:, 0].max()), float(pts_f[:, 1].max())
    ext_x = max(max_x - min_x, 1e-4)
    ext_y = max(max_y - min_y, 1e-4)

    pad = 14.0
    canvas_w = width - 2 * pad
    canvas_h = (height - header_h) - 2 * pad
    scale = min(canvas_w / ext_x, canvas_h / ext_y)
    # Prevent single dots from blowing up
    scale = min(scale, 42.0)

    content_w = ext_x * scale
    content_h = ext_y * scale
    ox = (width - content_w) * 0.5 - min_x * scale
    oy = header_h + ((height - header_h) - content_h) * 0.5 - min_y * scale

    n_strokes = max(len(clean), 1)
    line_w = max(2, int(round(min(3.0, max(1.6, scale * 0.055)))))

    for idx, st in enumerate(clean):
        if color_by_order:
            t = idx / max(n_strokes - 1, 1)
            col = (int(20 + 185 * t), int(55 + 40 * (1 - t)), int(180 - 110 * t))
        else:
            col = (20, 26, 40)

        xy = [(float(p[0] * scale + ox), float(p[1] * scale + oy)) for p in st if np.isfinite(p[0]) and np.isfinite(p[1])]
        if len(xy) == 1:
            x0, y0 = xy[0]
            r = max(1.5, line_w * 0.7)
            draw.ellipse([x0 - r, y0 - r, x0 + r, y0 + r], fill=col)
        elif len(xy) >= 2:
            draw.line(xy, fill=col, width=line_w, joint="curve")
            for x0, y0 in (xy[0], xy[-1]):
                r = line_w * 0.45
                draw.ellipse([x0 - r, y0 - r, x0 + r, y0 + r], fill=col)

    draw.rectangle([0, 0, width - 1, height - 1], outline=(210, 214, 222), width=1)
    return img


def render_contact_sheet(
    samples: Sequence[tuple[str, Sequence[np.ndarray]]],
    out_path: str | Path,
    *,
    cols: int = 3,
    tile_w: int = 440,
    tile_h: int = 160,
    color_by_order: bool = False,
) -> Path:
    """Render a grid of `(label, strokes)` items into one PNG contact sheet."""
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    if not samples:
        Image.new("RGB", (tile_w, tile_h), (255, 255, 255)).save(out)
        return out

    rows = (len(samples) + cols - 1) // cols
    gap = 8
    sheet = Image.new(
        "RGB",
        (cols * tile_w + (cols + 1) * gap, rows * tile_h + (rows + 1) * gap),
        (232, 235, 240),
    )
    for idx, (label, strokes) in enumerate(samples):
        r = idx // cols
        c = idx % cols
        tile = render_strokes_tile(
            strokes,
            label,
            width=tile_w,
            height=tile_h,
            color_by_order=color_by_order,
        )
        sheet.paste(tile, (gap + c * (tile_w + gap), gap + r * (tile_h + gap)))
    sheet.save(out)
    return out
