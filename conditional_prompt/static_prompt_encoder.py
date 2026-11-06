"""Single-view static soft-prompt encoder for frozen Qwen3-VL."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class StaticPromptOutput:
    """Outputs needed by the stage-2 forward and gradient smoke test."""

    embedding: torch.Tensor
    last_hidden_state: torch.Tensor
    attention_mask: torch.Tensor
    visual_pos_masks: torch.Tensor
    original_sequence_length: int
    extended_sequence_length: int
    original_first_visual_index: int
    extended_first_visual_index: int


class Qwen3VLStaticPromptEncoder(nn.Module):
    """Insert shared trainable tokens before the complete image-token block.

    The Qwen3-VL backbone is frozen. The output follows the existing Frozen
    baseline: mask-aware mean pooling over every valid final-layer position,
    followed by L2 normalization.
    """

    def __init__(
        self,
        backbone: nn.Module,
        num_prompt_tokens: int = 4,
        prompt_init_std: float = 0.02,
    ) -> None:
        super().__init__()
        if num_prompt_tokens < 1:
            raise ValueError("num_prompt_tokens must be positive")
        if prompt_init_std <= 0:
            raise ValueError("prompt_init_std must be positive")

        self.backbone = backbone
        self.num_prompt_tokens = num_prompt_tokens
        hidden_size = int(backbone.config.text_config.hidden_size)
        self.soft_prompt = nn.Parameter(torch.empty(num_prompt_tokens, hidden_size))
        nn.init.normal_(self.soft_prompt, mean=0.0, std=prompt_init_std)
        self.freeze_backbone()

    def freeze_backbone(self) -> None:
        self.backbone.eval()
        for parameter in self.backbone.parameters():
            parameter.requires_grad_(False)
            parameter.grad = None

    def train(self, mode: bool = True):
        super().train(mode)
        # The prompt can train, while the frozen backbone stays deterministic.
        self.backbone.eval()
        return self

    @staticmethod
    def _insert_before_first_visual(
        tensor: torch.Tensor,
        inserts: torch.Tensor,
        visual_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, list[int]]:
        rows = []
        insertion_indices = []
        for batch_index in range(tensor.shape[0]):
            positions = visual_mask[batch_index].nonzero(as_tuple=False)
            if positions.numel() == 0:
                raise ValueError("input contains no image-token position")
            index = int(positions[0, 0].item())
            insertion_indices.append(index)
            rows.append(
                torch.cat(
                    (
                        tensor[batch_index, :index],
                        inserts[batch_index],
                        tensor[batch_index, index:],
                    ),
                    dim=0,
                )
            )
        return torch.stack(rows), insertion_indices

    @staticmethod
    def _masked_mean(
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        expanded_mask = attention_mask.unsqueeze(-1).to(hidden_states.dtype)
        summed = (hidden_states * expanded_mask).float().sum(dim=1)
        denominator = expanded_mask.float().sum(dim=1).clamp_min(1e-9)
        return summed / denominator

    def forward(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pixel_values: torch.Tensor,
        image_grid_thw: torch.Tensor,
        prompt_embeddings: torch.Tensor | None = None,
    ) -> StaticPromptOutput:
        qwen = self.backbone.model
        token_embedding = qwen.language_model.get_input_embeddings()
        inputs_embeds = token_embedding(input_ids)

        # Run the frozen vision tower once and place its output at image-token
        # placeholder positions, matching the standard Qwen3-VL forward path.
        image_embeds, deepstack_visual_embeds = qwen.get_image_features(
            pixel_values,
            image_grid_thw,
        )
        visual_features = torch.cat(image_embeds, dim=0).to(inputs_embeds.dtype)
        original_visual_mask = input_ids.eq(int(qwen.config.image_token_id))
        expected_visual_tokens = int(original_visual_mask.sum().item())
        if expected_visual_tokens != int(visual_features.shape[0]):
            raise ValueError(
                "image placeholder / visual feature mismatch: "
                f"{expected_visual_tokens} placeholders vs "
                f"{visual_features.shape[0]} visual features"
            )
        inputs_embeds = inputs_embeds.masked_scatter(
            original_visual_mask.unsqueeze(-1).expand_as(inputs_embeds),
            visual_features.reshape(-1),
        )

        batch_size, original_sequence_length = input_ids.shape
        if prompt_embeddings is None:
            prompt = self.soft_prompt.unsqueeze(0).expand(batch_size, -1, -1)
        elif prompt_embeddings.ndim == 2:
            expected = (self.num_prompt_tokens, inputs_embeds.shape[-1])
            if tuple(prompt_embeddings.shape) != expected:
                raise ValueError(
                    f"prompt_embeddings must have shape {expected}, "
                    f"found {tuple(prompt_embeddings.shape)}"
                )
            prompt = prompt_embeddings.unsqueeze(0).expand(batch_size, -1, -1)
        elif prompt_embeddings.ndim == 3:
            expected = (
                batch_size,
                self.num_prompt_tokens,
                inputs_embeds.shape[-1],
            )
            if tuple(prompt_embeddings.shape) != expected:
                raise ValueError(
                    f"batched prompt_embeddings must have shape {expected}, "
                    f"found {tuple(prompt_embeddings.shape)}"
                )
            prompt = prompt_embeddings
        else:
            raise ValueError(
                "prompt_embeddings must be [tokens, hidden] or "
                "[batch, tokens, hidden]"
            )
        if not torch.isfinite(prompt_embeddings if prompt_embeddings is not None else self.soft_prompt).all():
            raise ValueError("prompt embeddings contain NaN or Inf")
        prompt = prompt.to(device=inputs_embeds.device, dtype=inputs_embeds.dtype)
        inputs_embeds, insertion_indices = self._insert_before_first_visual(
            inputs_embeds,
            prompt,
            original_visual_mask,
        )

        prompt_attention = torch.ones(
            batch_size,
            self.num_prompt_tokens,
            dtype=attention_mask.dtype,
            device=attention_mask.device,
        )
        extended_attention_mask, _ = self._insert_before_first_visual(
            attention_mask,
            prompt_attention,
            original_visual_mask,
        )

        prompt_visual_mask = torch.zeros(
            batch_size,
            self.num_prompt_tokens,
            dtype=torch.bool,
            device=original_visual_mask.device,
        )
        visual_pos_masks, _ = self._insert_before_first_visual(
            original_visual_mask,
            prompt_visual_mask,
            original_visual_mask,
        )

        # Soft prompts have embeddings but no natural token IDs. A fixed text
        # token acts only as an mRoPE position placeholder; its embedding is not
        # used because inputs_embeds has already been constructed above.
        pad_token_id = getattr(self.backbone.config, "pad_token_id", None)
        if pad_token_id is None:
            pad_token_id = 151643
        prompt_token_ids = torch.full(
            (batch_size, self.num_prompt_tokens),
            int(pad_token_id),
            dtype=input_ids.dtype,
            device=input_ids.device,
        )
        extended_input_ids, _ = self._insert_before_first_visual(
            input_ids,
            prompt_token_ids,
            original_visual_mask,
        )

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
        pooled = self._masked_mean(
            outputs.last_hidden_state,
            extended_attention_mask,
        )
        embedding = F.normalize(pooled, dim=-1)

        first_original_visual = insertion_indices[0]
        first_extended_visual = int(
            visual_pos_masks[0].nonzero(as_tuple=False)[0, 0].item()
        )
        return StaticPromptOutput(
            embedding=embedding,
            last_hidden_state=outputs.last_hidden_state,
            attention_mask=extended_attention_mask,
            visual_pos_masks=visual_pos_masks,
            original_sequence_length=original_sequence_length,
            extended_sequence_length=int(inputs_embeds.shape[1]),
            original_first_visual_index=first_original_visual,
            extended_first_visual_index=first_extended_visual,
        )
