from __future__ import annotations

from pathlib import Path
import hashlib
from typing import Any

import torch

from data.preprocessing import FEATURE_NAMES, PreprocessingConfig
from model.stroke2latex import ModelConfig, Stroke2Latex
from tokenizer.latex_tokenizer import LatexTokenizer


def checkpoint_digest(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_checkpoint(
    path: str | Path,
    model: Stroke2Latex,
    tokenizer: LatexTokenizer,
    preprocessing: PreprocessingConfig,
    epoch: int,
    metrics: dict[str, Any],
    optimizer: torch.optim.Optimizer | None = None,
    dataset_info: dict[str, Any] | None = None,
    training_state: dict[str, Any] | None = None,
) -> None:
    payload: dict[str, Any] = {
        "format_version": 1,
        "model_config": model.config.to_dict(),
        "preprocessing_config": preprocessing.to_dict(),
        "feature_names": list(FEATURE_NAMES),
        "model_state": model.state_dict(),
        "vocabulary": tokenizer.to_dict(),
        "parameter_count": model.parameter_count(),
        "epoch": epoch,
        "metrics": metrics,
        "dataset_info": dataset_info or {},
    }
    if optimizer is not None:
        payload["optimizer_state"] = optimizer.state_dict()
    if training_state is not None:
        payload["training_state"] = training_state
    destination = Path(path)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(destination)


def load_checkpoint(path: str | Path, device: str | torch.device = "cpu") -> tuple[Stroke2Latex, LatexTokenizer, PreprocessingConfig, dict[str, Any]]:
    payload = torch.load(Path(path), map_location=device, weights_only=False)
    tokenizer = LatexTokenizer.from_dict(payload["vocabulary"])
    config = ModelConfig.from_dict(payload["model_config"])
    if config.vocab_size != len(tokenizer):
        raise ValueError("Checkpoint vocabulary and model vocabulary size are incompatible")
    model = Stroke2Latex(config)
    model.load_state_dict(payload["model_state"])
    model.to(device).eval()
    preprocessing = PreprocessingConfig.from_dict(payload["preprocessing_config"])
    if tuple(payload.get("feature_names", FEATURE_NAMES)) != FEATURE_NAMES:
        raise ValueError("Checkpoint feature layout is incompatible with this runtime")
    return model, tokenizer, preprocessing, payload
