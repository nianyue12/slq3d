#!/usr/bin/env python3
"""Minimal SLQ-style readout for a frozen Qwen3-VL backbone.

This module adapts the latent-query mechanism from the official SLQ code to
independently encoded multi-view 3D objects.  It deliberately contains no
retrieval loss, anchor loss, or task-specific aggregation module.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class Qwen3VLLatentQueryEncoder(nn.Module):
    """Append trainable queries after a Qwen3-VL multimodal sequence."""

    def __init__(self, backbone: nn.Module, num_queries: int = 16) -> None:
        super().__init__()
        if num_queries < 1:
            raise ValueError("num_queries must be positive")
        self.backbone = backbone
        self.num_queries = num_queries
        hidden_size = int(backbone.config.text_config.hidden_size)
        self.latent_queries = nn.Parameter(torch.empty(num_queries, hidden_size))
        nn.init.normal_(self.latent_queries, std=1.0 / math.sqrt(hidden_size))
        self.freeze_backbone()

    def freeze_backbone(self) -> None:
        self.backbone.eval()
        for parameter in self.backbone.parameters():
            parameter.requires_grad_(False)
            parameter.grad = None

    def train(self, mode: bool = True):
        super().train(mode)
        # The latent queries train, but dropout-like backbone behavior must stay off.
        self.backbone.eval()
        return self

    def forward(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pixel_values: torch.Tensor,
        image_grid_thw: torch.Tensor,
    ) -> torch.Tensor:
        qwen = self.backbone.model
        token_embedding = qwen.language_model.get_input_embeddings()
        inputs_embeds = token_embedding(input_ids)

        image_embeds, deepstack_visual_embeds = qwen.get_image_features(
            pixel_values, image_grid_thw
        )
        visual_features = torch.cat(image_embeds, dim=0).to(inputs_embeds.dtype)
        visual_mask = input_ids.eq(int(qwen.config.image_token_id))
        expected = int(visual_mask.sum().item())
        if expected != int(visual_features.shape[0]):
            raise ValueError(
                "image placeholder / visual feature mismatch: "
                f"{expected} placeholders vs {visual_features.shape[0]} features"
            )
        expanded_visual_mask = visual_mask.unsqueeze(-1).expand_as(inputs_embeds)
        inputs_embeds = inputs_embeds.masked_scatter(
            expanded_visual_mask, visual_features.reshape(-1)
        )

        batch_size = input_ids.shape[0]
        queries = self.latent_queries.unsqueeze(0).expand(batch_size, -1, -1)
        queries = queries.to(device=inputs_embeds.device, dtype=inputs_embeds.dtype)
        inputs_embeds = torch.cat((inputs_embeds, queries), dim=1)

        query_attention = torch.ones(
            batch_size,
            self.num_queries,
            dtype=attention_mask.dtype,
            device=attention_mask.device,
        )
        extended_attention_mask = torch.cat((attention_mask, query_attention), dim=1)
        query_visual_mask = torch.zeros(
            batch_size,
            self.num_queries,
            dtype=torch.bool,
            device=visual_mask.device,
        )
        visual_pos_masks = torch.cat((visual_mask, query_visual_mask), dim=1)

        # The local Qwen3VLModel config does not expose pad_token_id, while the
        # outer conditional-generation config may.  This ID is only a textual
        # placeholder for mRoPE indexing; the corresponding embedding is never
        # used because the latent-query vectors were already appended above.
        pad_token_id = getattr(self.backbone.config, "pad_token_id", None)
        if pad_token_id is None:
            pad_token_id = 151643  # Qwen3-VL official SLQ fallback.
        query_token_ids = torch.full(
            (batch_size, self.num_queries),
            int(pad_token_id),
            dtype=input_ids.dtype,
            device=input_ids.device,
        )
        extended_input_ids = torch.cat((input_ids, query_token_ids), dim=1)
        position_ids, _ = qwen.get_rope_index(
            input_ids=extended_input_ids,
            image_grid_thw=image_grid_thw,
            video_grid_thw=None,
            attention_mask=extended_attention_mask,
        )

        outputs = qwen.language_model(
            input_ids=None,
            inputs_embeds=inputs_embeds,
            attention_mask=extended_attention_mask,
            position_ids=position_ids,
            visual_pos_masks=visual_pos_masks,
            deepstack_visual_embeds=deepstack_visual_embeds,
            use_cache=False,
            return_dict=True,
        )
        query_states = outputs.last_hidden_state[:, -self.num_queries :, :]
        return F.normalize(query_states.float().mean(dim=1), dim=-1)


class Qwen3VLPrefixQueryEncoder(nn.Module):
    """Insert trainable prefix tokens before visual tokens and queries at the end."""

    def __init__(
        self, backbone: nn.Module, num_prefix_tokens: int = 4, num_queries: int = 16
    ) -> None:
        super().__init__()
        if num_prefix_tokens < 1 or num_queries < 1:
            raise ValueError("prefix-token and query counts must be positive")
        self.backbone = backbone
        self.num_prefix_tokens = num_prefix_tokens
        self.num_queries = num_queries
        hidden_size = int(backbone.config.text_config.hidden_size)
        self.latent_queries = nn.Parameter(torch.empty(num_queries, hidden_size))
        self.prefix_tokens = nn.Parameter(torch.empty(num_prefix_tokens, hidden_size))
        # Initialize terminal queries first so a fixed seed matches the SLQ-16
        # query initialization; prefix parameters then consume the extra draws.
        nn.init.normal_(self.latent_queries, std=1.0 / math.sqrt(hidden_size))
        nn.init.normal_(self.prefix_tokens, std=1.0 / math.sqrt(hidden_size))
        self.freeze_backbone()

    def freeze_backbone(self) -> None:
        self.backbone.eval()
        for parameter in self.backbone.parameters():
            parameter.requires_grad_(False)
            parameter.grad = None

    def train(self, mode: bool = True):
        super().train(mode)
        self.backbone.eval()
        return self

    @staticmethod
    def _insert_before_first_visual(
        tensor: torch.Tensor, inserts: torch.Tensor, visual_mask: torch.Tensor
    ) -> torch.Tensor:
        rows = []
        for batch_index in range(tensor.shape[0]):
            positions = visual_mask[batch_index].nonzero(as_tuple=False)
            if positions.numel() == 0:
                raise ValueError("input contains no image-token position")
            index = int(positions[0, 0].item())
            rows.append(
                torch.cat(
                    (tensor[batch_index, :index], inserts[batch_index], tensor[batch_index, index:]),
                    dim=0,
                )
            )
        return torch.stack(rows)

    def forward(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pixel_values: torch.Tensor,
        image_grid_thw: torch.Tensor,
    ) -> torch.Tensor:
        qwen = self.backbone.model
        token_embedding = qwen.language_model.get_input_embeddings()
        inputs_embeds = token_embedding(input_ids)

        image_embeds, deepstack_visual_embeds = qwen.get_image_features(
            pixel_values, image_grid_thw
        )
        visual_features = torch.cat(image_embeds, dim=0).to(inputs_embeds.dtype)
        original_visual_mask = input_ids.eq(int(qwen.config.image_token_id))
        expected = int(original_visual_mask.sum().item())
        if expected != int(visual_features.shape[0]):
            raise ValueError(
                "image placeholder / visual feature mismatch: "
                f"{expected} placeholders vs {visual_features.shape[0]} features"
            )
        expanded_visual_mask = original_visual_mask.unsqueeze(-1).expand_as(inputs_embeds)
        inputs_embeds = inputs_embeds.masked_scatter(
            expanded_visual_mask, visual_features.reshape(-1)
        )

        batch_size = input_ids.shape[0]
        prefixes = self.prefix_tokens.unsqueeze(0).expand(batch_size, -1, -1).to(
            device=inputs_embeds.device, dtype=inputs_embeds.dtype
        )
        inputs_embeds = self._insert_before_first_visual(
            inputs_embeds, prefixes, original_visual_mask
        )

        prefix_attention = torch.ones(
            batch_size,
            self.num_prefix_tokens,
            dtype=attention_mask.dtype,
            device=attention_mask.device,
        )
        attention_mask = self._insert_before_first_visual(
            attention_mask, prefix_attention, original_visual_mask
        )
        prefix_visual_mask = torch.zeros(
            batch_size,
            self.num_prefix_tokens,
            dtype=torch.bool,
            device=original_visual_mask.device,
        )
        visual_mask = self._insert_before_first_visual(
            original_visual_mask, prefix_visual_mask, original_visual_mask
        )

        pad_token_id = getattr(self.backbone.config, "pad_token_id", None)
        if pad_token_id is None:
            pad_token_id = 151643
        prefix_token_ids = torch.full(
            (batch_size, self.num_prefix_tokens),
            int(pad_token_id),
            dtype=input_ids.dtype,
            device=input_ids.device,
        )
        extended_input_ids = self._insert_before_first_visual(
            input_ids, prefix_token_ids, original_visual_mask
        )

        queries = self.latent_queries.unsqueeze(0).expand(batch_size, -1, -1).to(
            device=inputs_embeds.device, dtype=inputs_embeds.dtype
        )
        inputs_embeds = torch.cat((inputs_embeds, queries), dim=1)
        query_attention = torch.ones(
            batch_size,
            self.num_queries,
            dtype=attention_mask.dtype,
            device=attention_mask.device,
        )
        attention_mask = torch.cat((attention_mask, query_attention), dim=1)
        query_visual_mask = torch.zeros(
            batch_size,
            self.num_queries,
            dtype=torch.bool,
            device=visual_mask.device,
        )
        visual_pos_masks = torch.cat((visual_mask, query_visual_mask), dim=1)
        query_token_ids = torch.full(
            (batch_size, self.num_queries),
            int(pad_token_id),
            dtype=input_ids.dtype,
            device=input_ids.device,
        )
        extended_input_ids = torch.cat((extended_input_ids, query_token_ids), dim=1)
        position_ids, _ = qwen.get_rope_index(
            input_ids=extended_input_ids,
            image_grid_thw=image_grid_thw,
            video_grid_thw=None,
            attention_mask=attention_mask,
        )

        outputs = qwen.language_model(
            input_ids=None,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            visual_pos_masks=visual_pos_masks,
            deepstack_visual_embeds=deepstack_visual_embeds,
            use_cache=False,
            return_dict=True,
        )
        query_states = outputs.last_hidden_state[:, -self.num_queries :, :]
        return F.normalize(query_states.float().mean(dim=1), dim=-1)
