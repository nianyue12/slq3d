"""Frozen-feature conditioned prompts for Dynamic Prompt scheme A."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass(frozen=True)
class DynamicPromptOutput:
    """Prompt tensors produced for a batch of frozen object conditions."""

    delta: torch.Tensor
    dynamic_prompt: torch.Tensor


class FrozenMeanConditionedPrompt(nn.Module):
    """Generate one object-conditioned offset shared by all prompt tokens.

    Frozen Mean-24 features are external constants.  They are converted to
    FP32 and detached before entering the Meta-Net, while gradients remain
    enabled for the shared prompt and every Meta-Net parameter.
    """

    def __init__(
        self,
        condition_dim: int = 4096,
        hidden_dim: int = 256,
        num_prompt_tokens: int = 4,
        prompt_init_std: float = 0.02,
        output_init_std: float = 1e-4,
    ) -> None:
        super().__init__()
        if min(condition_dim, hidden_dim, num_prompt_tokens) < 1:
            raise ValueError("dimensions and prompt-token count must be positive")
        if prompt_init_std <= 0 or output_init_std <= 0:
            raise ValueError("initialization standard deviations must be positive")

        self.condition_dim = condition_dim
        self.num_prompt_tokens = num_prompt_tokens
        self.shared_prompt = nn.Parameter(
            torch.empty(num_prompt_tokens, condition_dim, dtype=torch.float32)
        )
        self.meta_net = nn.Sequential(
            nn.Linear(condition_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, condition_dim),
        )

        # Match the Static Prompt initialization for the shared component.
        nn.init.normal_(self.shared_prompt, mean=0.0, std=prompt_init_std)
        # Begin close to the Static Prompt baseline without making delta
        # identically zero or independent of the input object.
        output_layer = self.meta_net[-1]
        assert isinstance(output_layer, nn.Linear)
        nn.init.normal_(output_layer.weight, mean=0.0, std=output_init_std)
        nn.init.zeros_(output_layer.bias)

    def forward(self, condition: torch.Tensor) -> DynamicPromptOutput:
        if condition.ndim != 2 or condition.shape[1] != self.condition_dim:
            raise ValueError(
                f"condition must have shape [batch, {self.condition_dim}], "
                f"found {tuple(condition.shape)}"
            )
        if not torch.isfinite(condition).all():
            raise ValueError("condition contains NaN or Inf")

        # The cached condition is never optimized. detach() also protects the
        # contract if a caller accidentally supplies a grad-enabled tensor.
        frozen_condition = condition.detach().to(
            device=self.shared_prompt.device,
            dtype=torch.float32,
        )
        delta = self.meta_net(frozen_condition)
        dynamic_prompt = (
            self.shared_prompt.unsqueeze(0) + delta.unsqueeze(1)
        )
        return DynamicPromptOutput(delta=delta, dynamic_prompt=dynamic_prompt)
