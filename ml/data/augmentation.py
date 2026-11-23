from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any

import numpy as np


@dataclass(slots=True)
class AugmentationConfig:
    enabled: bool = True
    rotation_degrees: float = 2.0
    scale_min: float = 0.92
    scale_max: float = 1.08
    anisotropic_scale_min: float = 1.0
    anisotropic_scale_max: float = 1.0
    translation: float = 0.025
    shear: float = 0.02
    jitter: float = 0.002
    point_dropout: float = 0.01
    resampling_variation: float = 0.0
    smooth_warp: float = 0.0
    time_scale_min: float = 0.8
    time_scale_max: float = 1.2
    pressure_jitter: float = 0.02
    timing_dropout: float = 0.0

    @classmethod
    def from_dict(cls, value: dict[str, Any] | None) -> "AugmentationConfig":
        value = dict(value or {})
        # Compatibility with the old symmetric configuration.
        if "scale" in value:
            magnitude = float(value.pop("scale"))
            value.setdefault("scale_min", 1.0 - magnitude)
            value.setdefault("scale_max", 1.0 + magnitude)
        if "time_scale" in value:
            magnitude = float(value.pop("time_scale"))
            value.setdefault("time_scale_min", 1.0 - magnitude)
            value.setdefault("time_scale_max", 1.0 + magnitude)
        fields = cls.__dataclass_fields__
        return cls(**{key: val for key, val in value.items() if key in fields})

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _drop_redundant_points(features: np.ndarray, probability: float, rng: np.random.Generator) -> np.ndarray:
    if probability <= 0 or len(features) <= 2:
        return features
    keep = np.ones(len(features), dtype=bool)
    starts = np.flatnonzero(features[:, 7] > 0.5)
    ends = np.flatnonzero(features[:, 8] > 0.5)
    interior = np.zeros(len(features), dtype=bool)
    for start, end in zip(starts, ends):
        # Short strokes (including dots) survive intact. Endpoints always survive.
        if end - start + 1 >= 4:
            interior[start + 1:end] = True
    before = features[1:-1, :2] - features[:-2, :2]
    after = features[2:, :2] - features[1:-1, :2]
    denominator = np.maximum((np.linalg.norm(before, axis=1) * np.linalg.norm(after, axis=1)).astype(np.float64), 1e-9)
    straightness = np.einsum('ij,ij->i', before, after).astype(np.float64) / denominator
    candidates = np.flatnonzero(interior[1:-1] & (straightness > .85)) + 1
    keep[candidates[rng.random(len(candidates)) < probability]] = False
    return features[keep]


def _recompute_deltas(features: np.ndarray) -> None:
    features[1:, 2:4] = features[1:, :2] - features[:-1, :2]
    features[0, 2:4] = 0
    features[features[:, 7] > .5, 2:4] = 0


def augment_features(
    features: np.ndarray,
    rng: np.random.Generator,
    config: AugmentationConfig | None = None,
) -> np.ndarray:
    """Conservative on-the-fly stroke-space augmentation preserving order and structure."""
    cfg = config or AugmentationConfig()
    if not 0 <= cfg.timing_dropout <= 1:
        raise ValueError('timing_dropout must be in [0, 1]')
    if not cfg.enabled:
        return features.copy()
    result = features.copy()
    angle = math.radians(float(rng.uniform(-cfg.rotation_degrees, cfg.rotation_degrees)))
    uniform_scale = float(rng.uniform(cfg.scale_min, cfg.scale_max))
    x_scale = float(rng.uniform(cfg.anisotropic_scale_min, cfg.anisotropic_scale_max))
    y_scale = float(rng.uniform(cfg.anisotropic_scale_min, cfg.anisotropic_scale_max))
    shear = float(rng.uniform(-cfg.shear, cfg.shear))
    rotation = np.asarray(
        [[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]], dtype=np.float32
    )
    affine = rotation @ np.asarray([[x_scale, shear], [0.0, y_scale]], dtype=np.float32) * uniform_scale
    translation = rng.uniform(-cfg.translation, cfg.translation, size=2).astype(np.float32)
    result[:, :2] = result[:, :2] @ affine.T + translation

    if cfg.smooth_warp > 0:
        # One low-frequency field over the complete expression; disabled by default.
        phase = float(rng.uniform(0, 2 * math.pi))
        direction = rng.normal(size=2).astype(np.float32)
        direction /= max(float(np.linalg.norm(direction)), 1e-9)
        displacement = np.sin(result[:, 0] * math.pi + phase)[:, None] * direction * cfg.smooth_warp
        result[:, :2] += displacement.astype(np.float32)
    if cfg.jitter > 0:
        result[:, :2] += rng.normal(0.0, cfg.jitter, result[:, :2].shape).astype(np.float32)

    result[:, 4] *= float(rng.uniform(cfg.time_scale_min, cfg.time_scale_max))
    # One availability decision per expression, preserving every other channel.
    # Do not consume a new RNG draw in existing recipes whose probability is 0.
    if cfg.timing_dropout > 0 and rng.random() < cfg.timing_dropout:
        result[:, 4] = 0
    pressure_mask = result[:, 6] > 0.5
    if cfg.pressure_jitter > 0 and pressure_mask.any():
        result[pressure_mask, 5] = np.clip(
            result[pressure_mask, 5] + rng.normal(0.0, cfg.pressure_jitter, pressure_mask.sum()), 0.0, 1.0
        )
    result = _drop_redundant_points(result, cfg.point_dropout, rng)
    _recompute_deltas(result)
    return result.astype(np.float32, copy=False)
