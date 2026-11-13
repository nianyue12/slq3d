"""Strict condition loading and object encoding for View-Prompt."""

from __future__ import annotations

from pathlib import Path

import torch

from conditional_prompt.static_object_encoder import StaticPromptObjectEncoder
from conditional_prompt.view_prompt.model import ViewConditionedPrompt
from dataset.frozen_mean import load_object_cache


def load_view_condition(row: dict[str, str], view_cache_dir: Path) -> tuple[torch.Tensor, Path]:
    split = row.get("split")
    object_id = row.get("object_id")
    if not split or not object_id:
        raise ValueError("manifest row requires split and object_id")
    cache_path = Path(view_cache_dir) / split / f"{object_id}.pt"
    condition = load_object_cache(cache_path, row, split).float().detach()
    if tuple(condition.shape) != (24, 4096):
        raise ValueError(f"{cache_path}: expected [24,4096], found {tuple(condition.shape)}")
    if not torch.isfinite(condition).all():
        raise ValueError(f"{cache_path}: condition contains NaN or Inf")
    condition.requires_grad_(False)
    return condition, cache_path


class VCObjectEncoder:
    """Generate one four-token prompt and share it across an object's 24 views."""

    def __init__(self, generator: ViewConditionedPrompt, encoder: StaticPromptObjectEncoder):
        self.generator = generator
        self.encoder = encoder

    def __call__(self, row, condition, *, device, model_dtype):
        if tuple(condition.shape) != (24, 4096):
            raise ValueError("condition must have shape [24,4096]")
        generated = self.generator(condition.unsqueeze(0).to(device))
        encoded = self.encoder(
            row,
            device=device,
            model_dtype=model_dtype,
            prompt_embeddings=generated.dynamic_prompt[0],
        )
        return generated, encoded
