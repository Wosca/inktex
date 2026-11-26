"""InkTeX v2 high-resolution strip rasterizer (H=192, static buckets W in {256, 512, 768, 1024}).

Key v2 upgrades over v1 (128x640):
  - Fixed height H=192 -> 24 vertical feature rows at stride-8 FPN (3x v1's 8 rows!)
  - Static CoreML-friendly width buckets {256, 512, 768, 1024}
  - Glyph-referenced scaling capped by both canvas dimensions:
        scale = min(target_glyph_px / g_hat, inner_h / h_box, inner_w / w_box)
  - Exposes exact `width_constrained` flag so UniMER squished-formula filtering checks true
    horizontal compression rather than penalizing short single-line expressions at 38px glyph height.
"""
from __future__ import annotations

from dataclasses import dataclass
import io
import math
from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from .ink_preprocess import clean_raw_strokes, estimate_glyph_scale

STRIP_HEIGHT_V2 = 192
WIDTH_BUCKETS_V2 = (256, 512, 768, 1024)
PADDING_V2 = 12
TARGET_GLYPH_PX_V2 = 38.0
SUPERSAMPLE_V2 = 3
WIDTH_MULTIPLE_V2 = 16


@dataclass(slots=True)
class RasterizedStripV2:
    image: Image.Image       # PIL LA image [W_bucket, 192]
    image_width: int         # valid content width (<= W_bucket)
    bucket_width: int        # static bucket in WIDTH_BUCKETS_V2
    scale: float             # source -> pixel scale
    offset: np.ndarray       # [2] float64 ink -> pixel offset (for online ink)
    width_constrained: bool  # True if max_width forced shrinking below desired height/glyph scale


def select_width_bucket(used_width: float, buckets: Sequence[int] = WIDTH_BUCKETS_V2) -> int:
    for b in buckets:
        if used_width <= b:
            return int(b)
    return int(buckets[-1])


def _blank_la(height: int, width: int) -> Image.Image:
    return Image.merge("LA", (Image.new("L", (width, height), 0), Image.new("L", (width, height), 0)))


def render_strokes_to_strip_v2(
    strokes: Sequence[np.ndarray],
    *,
    height: int = STRIP_HEIGHT_V2,
    width_buckets: Sequence[int] = WIDTH_BUCKETS_V2,
    padding: int = PADDING_V2,
    target_glyph_px: float = TARGET_GLYPH_PX_V2,
    scale_jitter: float = 1.0,
    stroke_width: float | None = None,
    bucket_override: int | None = None,
) -> RasterizedStripV2:
    clean, _ = clean_raw_strokes(strokes)
    max_width = int(bucket_override or width_buckets[-1])
    if not clean:
        b = int(bucket_override or width_buckets[0])
        return RasterizedStripV2(
            image=_blank_la(height, b),
            image_width=max(WIDTH_MULTIPLE_V2, 2 * padding),
            bucket_width=b,
            scale=1.0,
            offset=np.zeros(2, dtype=np.float64),
            width_constrained=False,
        )

    xy = np.concatenate(clean, axis=0)
    low = xy.min(axis=0)
    extent = xy.max(axis=0) - low
    inner_h = float(height - 2 * padding)
    inner_w = float(max_width - 2 * padding)

    g_hat = estimate_glyph_scale(clean)
    desired_glyph = target_glyph_px * float(scale_jitter)
    s_glyph = desired_glyph / max(g_hat, 1e-6)
    s_h = (inner_h / float(extent[1])) if extent[1] > 1e-9 else s_glyph
    s_w = (inner_w / float(extent[0])) if extent[0] > 1e-9 else s_glyph
    scale = float(min(s_glyph, s_h, s_w))
    width_constrained = bool(extent[0] > 1e-9 and s_w < min(s_glyph, s_h))

    content_w = float(extent[0]) * scale
    content_h = float(extent[1]) * scale
    used_w = int(math.ceil(content_w + 2 * padding))
    bucket_w = int(bucket_override) if bucket_override is not None else select_width_bucket(used_w, width_buckets)
    image_width = int(min(bucket_w, max(WIDTH_MULTIPLE_V2, used_w)))
    offset = np.array([padding - float(low[0]) * scale, (height - content_h) / 2.0 - float(low[1]) * scale],
                      dtype=np.float64)

    ss = SUPERSAMPLE_V2
    default_sw = max(1.6, min(3.6, desired_glyph / 14.0))
    line = max(1, int(round((stroke_width if stroke_width is not None else default_sw) * ss)))
    radius = line / 2.0

    ink = Image.new("L", (bucket_w * ss, height * ss), 0)
    order = Image.new("L", (bucket_w * ss, height * ss), 0)
    draw_ink, draw_order = ImageDraw.Draw(ink), ImageDraw.Draw(order)
    denominator = max(sum(map(len, clean)) - 1, 1)
    start = 0

    for stroke in clean:
        points = (stroke * scale + offset) * ss
        progress = 1 + np.rint(254.0 * (start + np.arange(len(points))) / denominator).astype(np.int64)
        start += len(points)
        keep = [0]
        for idx in range(1, len(points)):
            if idx == len(points) - 1 or np.abs(points[idx] - points[keep[-1]]).max() >= 1.0:
                keep.append(idx)
        previous = None
        for idx in keep:
            x, y = float(points[idx, 0]), float(points[idx, 1])
            val = int(progress[idx])
            if previous is not None:
                seg = [previous, (x, y)]
                draw_ink.line(seg, fill=255, width=line)
                draw_order.line(seg, fill=val, width=line)
            box = (x - radius, y - radius, x + radius, y + radius)
            draw_ink.ellipse(box, fill=255)
            draw_order.ellipse(box, fill=val)
            previous = (x, y)

    lum = np.asarray(ink.resize((bucket_w, height), Image.Resampling.BOX), dtype=np.uint8).copy()
    prog = np.asarray(order, dtype=np.uint8).reshape(height, ss, bucket_w, ss).max(axis=(1, 3))
    prog = np.where(lum > 0, np.maximum(prog, 1), 0).astype(np.uint8)
    img = Image.merge("LA", (Image.fromarray(lum, "L"), Image.fromarray(prog, "L")))
    return RasterizedStripV2(
        image=img,
        image_width=image_width,
        bucket_width=bucket_w,
        scale=scale,
        offset=offset,
        width_constrained=width_constrained,
    )


def _load_gray(img: Image.Image | bytes | bytearray | str | Path) -> np.ndarray:
    if isinstance(img, (bytes, bytearray)):
        img = Image.open(io.BytesIO(img))
    elif isinstance(img, (str, Path)):
        with Image.open(img) as handle:
            handle.load()
            img = handle.copy()
    if img.mode in ("RGBA", "LA", "PA") or (img.mode == "P" and "transparency" in img.info):
        rgba = img.convert("RGBA")
        canvas = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        img = Image.alpha_composite(canvas, rgba)
    return np.asarray(img.convert("L"), dtype=np.float32)


def _background(values: np.ndarray) -> np.ndarray:
    h, w = values.shape
    factor = max(1, int(round(min(h, w) / 32)))
    small = Image.fromarray(values.astype(np.uint8), "L").reduce(factor) if factor > 1 else \
        Image.fromarray(values.astype(np.uint8), "L")
    size = 5 if min(small.size) >= 10 else 3
    opened = small.filter(ImageFilter.MinFilter(size)).filter(ImageFilter.MaxFilter(size))
    smooth = opened.filter(ImageFilter.GaussianBlur(2)).resize((w, h), Image.Resampling.BILINEAR)
    return np.asarray(smooth, dtype=np.float32)


def _estimate_offline_glyph_height(mask: np.ndarray) -> float | None:
    """Zero-dependency deterministic 4-connected-component glyph height estimate.

    Downsamples large crops to <=160px on the vertical axis before BFS labeling so it runs
    in <1 ms with zero external (SciPy/OpenCV) dependency across Windows and Linux.
    """
    h, w = mask.shape
    if h < 4 or w < 4:
        return None
    ds = max(1, int(math.ceil(h / 128.0)))
    if ds > 1:
        # Block-OR downsample preserving thin strokes
        bh = (h // ds) * ds
        bw = (w // ds) * ds
        if bh < 4 or bw < 4:
            return None
        small = mask[:bh, :bw].reshape(bh // ds, ds, bw // ds, ds).any(axis=(1, 3))
    else:
        small = mask
    sh, sw = small.shape
    visited = np.zeros((sh, sw), dtype=np.bool_)
    ys, xs = np.nonzero(small)
    if len(ys) == 0:
        return None
    heights: list[float] = []
    min_dim = max(2, int(0.05 * sh))
    for sy, sx in zip(ys.tolist(), xs.tolist()):
        if visited[sy, sx]:
            continue
        stack = [(sy, sx)]
        visited[sy, sx] = True
        r0 = r1 = sy
        c0 = c1 = sx
        while stack:
            r, c = stack.pop()
            if r < r0:
                r0 = r
            elif r > r1:
                r1 = r
            if c < c0:
                c0 = c
            elif c > c1:
                c1 = c
            if r > 0 and small[r - 1, c] and not visited[r - 1, c]:
                visited[r - 1, c] = True
                stack.append((r - 1, c))
            if r + 1 < sh and small[r + 1, c] and not visited[r + 1, c]:
                visited[r + 1, c] = True
                stack.append((r + 1, c))
            if c > 0 and small[r, c - 1] and not visited[r, c - 1]:
                visited[r, c - 1] = True
                stack.append((r, c - 1))
            if c + 1 < sw and small[r, c + 1] and not visited[r, c + 1]:
                visited[r, c + 1] = True
                stack.append((r, c + 1))
        ch = r1 - r0 + 1
        cw = c1 - c0 + 1
        if max(ch, cw) >= min_dim:
            heights.append(float(max(ch, 0.5 * cw) * ds))
    if not heights:
        return None
    return float(np.median(heights))


def normalize_offline_image_to_strip_v2(
    img: Image.Image | bytes | Path,
    *,
    height: int = STRIP_HEIGHT_V2,
    width_buckets: Sequence[int] = WIDTH_BUCKETS_V2,
    padding: int = PADDING_V2,
    target_glyph_px: float = TARGET_GLYPH_PX_V2,
    scale_jitter: float = 1.0,
) -> RasterizedStripV2:
    max_width = int(width_buckets[-1])
    min_bucket = int(width_buckets[0])
    gray = _load_gray(img)
    if gray.size == 0:
        return RasterizedStripV2(_blank_la(height, min_bucket), min_bucket, min_bucket, 1.0, np.zeros(2), False)

    band = max(1, min(gray.shape) // 20)
    border = np.concatenate([gray[:band].ravel(), gray[-band:].ravel(), gray[:, :band].ravel(), gray[:, -band:].ravel()])
    p_low, median, p_high = np.percentile(gray, [0.5, 50, 99.5])
    if (median - p_low) - (p_high - median) + 0.25 * (float(np.median(border)) - 127.5) > 0:
        gray = 255.0 - gray
    if min(gray.shape) >= 24:
        gray = np.asarray(Image.fromarray(gray.astype(np.uint8), "L").filter(ImageFilter.GaussianBlur(0.6)),
                          dtype=np.float32)
    residual = np.clip(gray - np.minimum(_background(gray), gray), 0, 255)
    reference = float(np.percentile(residual, 99.5))
    if reference < 24:
        reference = float(residual.max())
    if reference < 8:
        return RasterizedStripV2(_blank_la(height, min_bucket), min_bucket, min_bucket, 1.0, np.zeros(2), False)

    med = float(np.median(residual))
    noise = med + 3.0 * 1.4826 * float(np.median(np.abs(residual - med)))
    low = max(6.0, 0.12 * reference, min(noise, 0.5 * reference))
    high = max(low + 8.0, 0.7 * reference)
    ink = np.clip((residual - low) / (high - low), 0.0, 1.0)

    mask = ink > 0.25
    rows, cols = np.flatnonzero(mask.any(axis=1)), np.flatnonzero(mask.any(axis=0))
    if not len(rows):
        return RasterizedStripV2(_blank_la(height, min_bucket), min_bucket, min_bucket, 1.0, np.zeros(2), False)

    margin = 2
    top, bottom = max(0, rows[0] - margin), min(ink.shape[0], rows[-1] + margin + 1)
    left, right = max(0, cols[0] - margin), min(ink.shape[1], cols[-1] + margin + 1)
    crop = ink[top:bottom, left:right]
    crop_h, crop_w = crop.shape

    inner_h, inner_w = float(height - 2 * padding), float(max_width - 2 * padding)
    s_h = inner_h / crop_h
    s_w = inner_w / crop_w
    g_hat = _estimate_offline_glyph_height(crop > 0.25)
    if g_hat is not None:
        s_glyph = (target_glyph_px * float(scale_jitter)) / max(g_hat, 1.0)
        scale = float(min(s_glyph, s_h, s_w))
        width_constrained = bool(s_w < min(s_glyph, s_h))
    else:
        scale = float(min(s_h, s_w))
        width_constrained = bool(s_w < s_h)

    target_w = max(1, int(round(crop_w * scale)))
    target_h = max(1, int(round(crop_h * scale)))
    resized = Image.fromarray(np.rint(crop * 255).astype(np.uint8), "L").resize(
        (target_w, target_h), Image.Resampling.LANCZOS)
    arr = np.asarray(resized, dtype=np.uint8).copy()
    arr[arr < 6] = 0

    used_w = target_w + 2 * padding
    bucket_w = select_width_bucket(used_w, width_buckets)
    image_width = int(min(bucket_w, max(WIDTH_MULTIPLE_V2, used_w)))
    canvas = np.zeros((height, bucket_w), dtype=np.uint8)
    y0 = max(0, (height - target_h) // 2)
    h_copy = min(height - y0, target_h)
    w_copy = min(bucket_w - padding, target_w)
    canvas[y0:y0 + h_copy, padding:padding + w_copy] = arr[:h_copy, :w_copy]
    image = Image.merge("LA", (Image.fromarray(canvas, "L"), Image.new("L", (bucket_w, height), 0)))
    return RasterizedStripV2(
        image=image,
        image_width=image_width,
        bucket_width=bucket_w,
        scale=scale,
        offset=np.array([float(padding), float(y0)], dtype=np.float64),
        width_constrained=width_constrained,
    )
