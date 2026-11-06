"""Object-level 24-view inference with a shared static soft prompt."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from conditional_prompt.static_prompt_encoder import Qwen3VLStaticPromptEncoder
from dataset.manifest import VIEW_COLUMNS
from utils.io import PROMPT, move_inputs


@dataclass
class StaticObjectOutput:
    """The per-view and fused representations for one 3D object."""

    view_embeddings: torch.Tensor
    object_embedding: torch.Tensor
    view_columns: list[str]
    view_paths: list[Path]


def object_view_paths(row: dict[str, str]) -> tuple[list[str], list[Path]]:
    """Return all 24 views in the existing manifest order."""

    view_columns = list(VIEW_COLUMNS)
    paths = [Path(row[column]) for column in view_columns]
    if len(paths) != 24:
        raise ValueError(f"expected 24 manifest views, found {len(paths)}")
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing view image: {missing[0]}")
    if len(set(paths)) != len(paths):
        raise ValueError("manifest contains duplicate view paths")
    return view_columns, paths


def build_independent_view_batch(
    processor,
    view_paths: Sequence[Path],
):
    """Build a batch whose rows remain independent single-view examples.

    This preserves the original Frozen input definition for every row:
    one image followed by the fixed ``PROMPT``. Images in the same chunk do
    not share a language sequence and cannot attend to one another.
    """

    images = []
    for path in view_paths:
        with Image.open(path) as image:
            images.append(image.convert("RGB").copy())

    promptings = []
    for image in images:
        message = {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": PROMPT},
            ],
        }
        promptings.append(processor.apply_chat_template([message]))

    return processor(
        text=promptings,
        images=[[image] for image in images],
        padding=True,
        return_tensors="pt",
    )


class StaticPromptObjectEncoder(nn.Module):
    """Fuse 24 independently encoded views into one object embedding."""

    def __init__(
        self,
        view_encoder: Qwen3VLStaticPromptEncoder,
        processor,
        chunk_size: int = 4,
    ) -> None:
        super().__init__()
        if chunk_size < 1:
            raise ValueError("chunk_size must be positive")
        self.view_encoder = view_encoder
        self.processor = processor
        self.chunk_size = chunk_size

    def forward(
        self,
        row: dict[str, str],
        *,
        device: torch.device,
        model_dtype: torch.dtype,
        prompt_embeddings: torch.Tensor | None = None,
    ) -> StaticObjectOutput:
        """Encode one object's 24 views without constructing gradient graphs."""

        view_columns, view_paths = object_view_paths(row)
        embeddings = []

        self.view_encoder.eval()
        with torch.no_grad():
            for start in range(0, len(view_paths), self.chunk_size):
                chunk_paths = view_paths[start : start + self.chunk_size]
                inputs = build_independent_view_batch(self.processor, chunk_paths)
                batch = move_inputs(inputs, device, model_dtype)
                output = self.view_encoder(
                    **batch,
                    prompt_embeddings=prompt_embeddings,
                )
                embeddings.append(output.embedding.detach().float().cpu())

        view_embeddings = torch.cat(embeddings, dim=0)
        if view_embeddings.shape != (24, 4096):
            raise RuntimeError(
                "expected 24 view embeddings with dimension 4096, "
                f"got {tuple(view_embeddings.shape)}"
            )
        if not torch.isfinite(view_embeddings).all():
            raise RuntimeError("view embeddings contain NaN or Inf")

        # Each view is already normalized by the stage-2 encoder. Preserve the
        # Frozen Mean-24 aggregation rule: FP32 mean followed by final L2.
        object_embedding = F.normalize(view_embeddings.mean(dim=0), dim=0)
        if not torch.isfinite(object_embedding).all():
            raise RuntimeError("object embedding contains NaN or Inf")

        return StaticObjectOutput(
            view_embeddings=view_embeddings,
            object_embedding=object_embedding,
            view_columns=view_columns,
            view_paths=view_paths,
        )
