"""InkForge v3 and checkpoint loading shared by training and inference."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from ..model import InkForgeConfig, InkForgeModel
from ..tokenizer import InkForgeTokenizer
from .model_v3 import InkForgeV3Config, InkForgeV3Model


CHECKPOINT_FORMAT = "inkforge-checkpoint-v3"

__all__ = ["InkForgeV3Config", "InkForgeV3Model", "load_inkforge_checkpoint"]


def load_inkforge_checkpoint(
    ckpt_path: str | Path,
    device: str | torch.device = "cpu",
    *,
    grad_checkpoint: bool = False,
    prefer_ema: bool = True,
) -> tuple[InkForgeModel, InkForgeTokenizer, dict[str, Any]]:
    """Load an eval model, its tokenizer, and the original CPU checkpoint."""
    payload = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    keys = ("ema_state_dict", "model_state_dict")
    if not prefer_ema:
        keys = keys[::-1]
    state = next((payload[key] for key in keys if payload.get(key) is not None), None)
    if state is None:
        raise ValueError(f"No model weights in checkpoint {ckpt_path}")
    tokenizer = InkForgeTokenizer.from_dict(payload["tokenizer"])
    if payload.get("format") == CHECKPOINT_FORMAT or "canvas_in.weight" in state:
        cfg = InkForgeV3Config.from_dict(payload["config"])
        if not cfg.token_roles:
            cfg.token_roles = cfg.roles_for_tokens(tokenizer.tokens)
        model = InkForgeV3Model(cfg, grad_checkpoint=grad_checkpoint)
    else:
        model = InkForgeModel(InkForgeConfig.from_dict(payload["config"]), grad_checkpoint=grad_checkpoint)
    if len(tokenizer) != model.cfg.vocab_size:
        raise ValueError(f"Vocabulary size does not match the checkpoint config in {ckpt_path}")
    model.load_state_dict(state)
    return model.to(device).eval(), tokenizer, payload

