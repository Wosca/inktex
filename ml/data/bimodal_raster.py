"""Two-channel (ink, stroke order) strips shared by online ink and offline images.

Channel L: 0 = background, 255 = ink. Channel A: 0 = background/unknown order,
1..255 = monotonic progress from the first to the last recorded ink point.
"""
from __future__ import annotations

import io
import math
from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

STRIP_HEIGHT = 128
WIDTH_MULTIPLE = 16
SUPERSAMPLE = 3


def round_up(value: float, multiple: int = WIDTH_MULTIPLE) -> int:
    return int(math.ceil(max(float(value), 1.0) / multiple) * multiple)


def strip_width(content_width: float, min_width: int, max_width: int) -> int:
    return int(min(max_width, max(min_width, round_up(content_width))))


def _check_geometry(height: int, min_width: int, max_width: int, padding: int) -> None:
    if height <= 2 * padding or max_width <= 2 * padding:
        raise ValueError("Strip padding leaves no drawable area")
    if min_width % WIDTH_MULTIPLE or max_width % WIDTH_MULTIPLE or not 0 < min_width <= max_width:
        raise ValueError(f"Strip widths must be multiples of {WIDTH_MULTIPLE} with min <= max")


def _blank(height: int, width: int) -> Image.Image:
    return Image.merge("LA", (Image.new("L", (width, height), 0), Image.new("L", (width, height), 0)))


def _clean_strokes(strokes: Sequence[np.ndarray]) -> list[np.ndarray]:
    result = []
    for stroke in strokes:
        array = np.asarray(stroke, dtype=np.float64)
        if array.ndim == 1 and array.size >= 2:
            array = array.reshape(1, -1)
        if array.ndim != 2 or array.shape[1] < 2 or not len(array):
            continue
        array = array[:, :2]
        array = array[np.isfinite(array).all(axis=1)]
        if len(array):
            result.append(array)
    return result


def render_strokes_to_strip(
    strokes: Sequence[np.ndarray],
    height: int = STRIP_HEIGHT,
    min_width: int = 64,
    max_width: int = 640,
    padding: int = 8,
    stroke_width: float | None = None,
) -> tuple[Image.Image, int]:
    _check_geometry(height, min_width, max_width, padding)
    clean = _clean_strokes(strokes)
    if not clean:
        return _blank(height, min_width), max(WIDTH_MULTIPLE, min(min_width, 2 * padding))
    xy = np.concatenate(clean)
    low = xy.min(axis=0)
    extent = xy.max(axis=0) - low
    inner_h, inner_w = height - 2 * padding, max_width - 2 * padding
    scales = [limit / size for limit, size in ((inner_w, extent[0]), (inner_h, extent[1])) if size > 1e-9]
    scale = min(scales) if scales else 1.0
    content_w, content_h = extent[0] * scale, extent[1] * scale
    width = strip_width(content_w + 2 * padding, min_width, max_width)
    offset = np.asarray([padding - low[0] * scale, (height - content_h) / 2.0 - low[1] * scale])

    ss = SUPERSAMPLE
    line = max(1, int(round((stroke_width or max(1.0, height / 48.0)) * ss)))
    radius = line / 2.0
    ink = Image.new("L", (width * ss, height * ss), 0)
    order = Image.new("L", (width * ss, height * ss), 0)
    draw_ink, draw_order = ImageDraw.Draw(ink), ImageDraw.Draw(order)
    denominator = max(sum(map(len, clean)) - 1, 1)
    start = 0
    for stroke in clean:
        points = (stroke * scale + offset) * ss
        progress = 1 + np.rint(254.0 * (start + np.arange(len(points))) / denominator).astype(np.int64)
        start += len(points)
        # Drop sub-pixel steps; endpoints always survive so dots keep their cap.
        keep = [0]
        for index in range(1, len(points)):
            if index == len(points) - 1 or np.abs(points[index] - points[keep[-1]]).max() >= 1.0:
                keep.append(index)
        previous = None
        for index in keep:
            x, y = float(points[index, 0]), float(points[index, 1])
            value = int(progress[index])
            if previous is not None:
                segment = [previous, (x, y)]
                draw_ink.line(segment, fill=255, width=line)
                draw_order.line(segment, fill=value, width=line)
            box = (x - radius, y - radius, x + radius, y + radius)
            draw_ink.ellipse(box, fill=255)
            draw_order.ellipse(box, fill=value)
            previous = (x, y)

    luminance = np.asarray(ink.resize((width, height), Image.Resampling.BOX), dtype=np.uint8).copy()
    progress_map = np.asarray(order, dtype=np.uint8).reshape(height, ss, width, ss).max(axis=(1, 3))
    progress_map = np.where(luminance > 0, np.maximum(progress_map, 1), 0).astype(np.uint8)
    image = Image.merge("LA", (Image.fromarray(luminance, "L"), Image.fromarray(progress_map, "L")))
    used = int(math.ceil(content_w + 2 * padding))
    return image, int(min(width, max(WIDTH_MULTIPLE, used)))


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
    """Smooth low-frequency background estimate (ink is bright, thin and local)."""
    h, w = values.shape
    factor = max(1, int(round(min(h, w) / 32)))
    small = Image.fromarray(values.astype(np.uint8), "L").reduce(factor) if factor > 1 else \
        Image.fromarray(values.astype(np.uint8), "L")
    size = 5 if min(small.size) >= 10 else 3
    opened = small.filter(ImageFilter.MinFilter(size)).filter(ImageFilter.MaxFilter(size))
    smooth = opened.filter(ImageFilter.GaussianBlur(2)).resize((w, h), Image.Resampling.BILINEAR)
    return np.asarray(smooth, dtype=np.float32)


def normalize_offline_image_to_strip(
    img: Image.Image | bytes | Path,
    height: int = STRIP_HEIGHT,
    min_width: int = 64,
    max_width: int = 640,
    padding: int = 6,
) -> tuple[Image.Image, int]:
    _check_geometry(height, min_width, max_width, padding)
    gray = _load_gray(img)
    if gray.size == 0:
        return _blank(height, min_width), min_width
    band = max(1, min(gray.shape) // 20)
    border = np.concatenate([gray[:band].ravel(), gray[-band:].ravel(), gray[:, :band].ravel(), gray[:, -band:].ravel()])
    # Ink is the minority tail; dim photographs can have a dark (< 128) paper.
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
        return _blank(height, min_width), min_width
    median = float(np.median(residual))
    noise = median + 3.0 * 1.4826 * float(np.median(np.abs(residual - median)))
    low = max(6.0, 0.12 * reference, min(noise, 0.5 * reference))
    high = max(low + 8.0, 0.7 * reference)
    ink = np.clip((residual - low) / (high - low), 0.0, 1.0)

    mask = ink > 0.25
    rows, cols = np.flatnonzero(mask.any(axis=1)), np.flatnonzero(mask.any(axis=0))
    if not len(rows):
        return _blank(height, min_width), min_width
    margin = 2
    top, bottom = max(0, rows[0] - margin), min(ink.shape[0], rows[-1] + margin + 1)
    left, right = max(0, cols[0] - margin), min(ink.shape[1], cols[-1] + margin + 1)
    crop = ink[top:bottom, left:right]
    crop_h, crop_w = crop.shape
    inner_h, inner_w = height - 2 * padding, max_width - 2 * padding
    scale = min(inner_h / crop_h, inner_w / crop_w)
    target_w, target_h = max(1, int(round(crop_w * scale))), max(1, int(round(crop_h * scale)))
    resized = Image.fromarray(np.rint(crop * 255).astype(np.uint8), "L").resize(
        (target_w, target_h), Image.Resampling.LANCZOS)
    resized_array = np.asarray(resized, dtype=np.uint8).copy()
    resized_array[resized_array < 6] = 0

    used = target_w + 2 * padding
    width = strip_width(used, min_width, max_width)
    canvas = np.zeros((height, width), dtype=np.uint8)
    y0 = (height - target_h) // 2
    x_end = min(width, padding + target_w)
    canvas[y0:y0 + target_h, padding:x_end] = resized_array[:, :x_end - padding]
    image = Image.merge("LA", (Image.fromarray(canvas, "L"), Image.new("L", (width, height), 0)))
    return image, int(min(width, max(WIDTH_MULTIPLE, used)))


def compute_dhash64(img: Image.Image) -> str:
    """64-bit difference hash of channel L, cropped to visible ink first."""
    gray = img.getchannel("L") if img.mode in ("LA", "L") else img.convert("L")
    values = np.asarray(gray, dtype=np.uint8)
    rows, cols = np.flatnonzero((values > 32).any(axis=1)), np.flatnonzero((values > 32).any(axis=0))
    if len(rows) and len(cols):
        values = values[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1]
    small = np.asarray(Image.fromarray(values, "L").resize((9, 8), Image.Resampling.BOX), dtype=np.int16)
    bits = (small[:, 1:] > small[:, :-1]).ravel()
    return f"{int(''.join('1' if bit else '0' for bit in bits), 2):016x}"


def hamming64(left: str | int, right: str | int) -> int:
    a = int(left, 16) if isinstance(left, str) else left
    b = int(right, 16) if isinstance(right, str) else right
    return (a ^ b).bit_count()


def encode_png(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", compress_level=6)
    return buffer.getvalue()


def decode_png(data: bytes) -> np.ndarray:
    """Return uint8 [2, H, W] (L, A)."""
    with Image.open(io.BytesIO(data)) as image:
        array = np.asarray(image.convert("LA") if image.mode != "LA" else image, dtype=np.uint8)
    return np.ascontiguousarray(array.transpose(2, 0, 1))
