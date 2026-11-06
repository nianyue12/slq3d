"""Object-level inference for Frozen Mean-24 conditioned Dynamic Prompts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn

from conditional_prompt.dynamic_prompt import FrozenMeanConditionedPrompt
from conditional_prompt.static_object_encoder import (
    StaticObjectOutput,
    StaticPromptObjectEncoder,
)
from dataset.frozen_mean import aggregate_view_embeddings, load_object_cache


@dataclass(frozen=True)
class DynamicObjectOutput:
    """Condition, prompt, and 24-view representations for one object."""

    condition: torch.Tensor
    delta: torch.Tensor
    dynamic_prompt: torch.Tensor
    view_embeddings: torch.Tensor
    object_embedding: torch.Tensor
    view_columns: list[str]
    view_paths: list[Path]


def load_frozen_mean24_condition(
    row: dict[str, str],
    view_cache_dir: Path,
) -> tuple[torch.Tensor, Path]:
    """Load and strictly identify one cached Frozen Mean-24 condition."""

    split = row.get("split")
    if not split:
        raise ValueError("manifest row has no split")
    object_id = row.get("object_id")
    if not object_id:
        raise ValueError("manifest row has no object_id")
    cache_path = view_cache_dir / split / f"{object_id}.pt"
    # The shared loader checks object_id, label_name, split, and the exact
    # ordered list of 24 image paths against this Manifest row.
    view_embeddings = load_object_cache(cache_path, row, split)
    if tuple(view_embeddings.shape) != (24, 4096):
        raise ValueError(
            f"{cache_path}: expected [24, 4096], "
            f"found {tuple(view_embeddings.shape)}"
        )
    condition = aggregate_view_embeddings(view_embeddings).float().detach()
    condition.requires_grad_(False)
    return condition, cache_path


class DynamicPromptObjectEncoder(nn.Module):
    """Generate one prompt once, then share it across all 24 object views."""

    def __init__(
        self,
        prompt_generator: FrozenMeanConditionedPrompt,
        object_encoder: StaticPromptObjectEncoder,
    ) -> None:
        super().__init__()
        self.prompt_generator = prompt_generator
        self.object_encoder = object_encoder

    def forward(
        self,
        row: dict[str, str],
        condition: torch.Tensor,
        *,
        device: torch.device,
        model_dtype: torch.dtype,
        force_zero_delta: bool = False,
        delta_scale: float = 1.0,
    ) -> DynamicObjectOutput:
        if tuple(condition.shape) != (4096,):
            raise ValueError(
                f"condition must have shape [4096], found {tuple(condition.shape)}"
            )

        if not torch.isfinite(torch.tensor(delta_scale)) or delta_scale < 0:
            raise ValueError("delta_scale must be finite and non-negative")
        generated = self.prompt_generator(condition.unsqueeze(0))
        applied_scale = 0.0 if force_zero_delta else float(delta_scale)
        delta = generated.delta[0] * applied_scale
        # Construct from the same Shared Parameter so scale=0 is the exact
        # trained-Shared-Prompt ablation and scale=1 preserves prior behavior.
        dynamic_prompt = self.prompt_generator.shared_prompt + delta.unsqueeze(0)

        encoded: StaticObjectOutput = self.object_encoder(
            row,
            device=device,
            model_dtype=model_dtype,
            prompt_embeddings=dynamic_prompt,
        )
        return DynamicObjectOutput(
            condition=condition,
            delta=delta,
            dynamic_prompt=dynamic_prompt,
            view_embeddings=encoded.view_embeddings,
            object_embedding=encoded.object_embedding,
            view_columns=encoded.view_columns,
            view_paths=encoded.view_paths,
        )
