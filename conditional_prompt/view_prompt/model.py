"""View-conditioned Dynamic Soft Prompt used by View-Prompt.

This module intentionally contains no centering, norm control, auxiliary loss,
or later M1/M2 mechanisms.  It is the original Dynamic Prompt with only its
condition-to-prompt mapping replaced by a four-query view conditioner.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


VIEWS = 24
MODEL_DIM = 4096
CONDITION_DIM = 256
PROMPT_TOKENS = 4


@dataclass(frozen=True)
class ViewPromptOutput:
    dynamic_prompt: torch.Tensor
    delta: torch.Tensor
    attention: torch.Tensor
    projected_views: torch.Tensor
    attended_condition: torch.Tensor


class ViewConditionedPrompt(nn.Module):
    """Map 24 frozen per-view features to four instance prompt offsets."""

    def __init__(self) -> None:
        super().__init__()
        self.shared_prompt = nn.Parameter(torch.empty(PROMPT_TOKENS, MODEL_DIM))
        self.view_projection = nn.Linear(MODEL_DIM, CONDITION_DIM, bias=True)
        self.queries = nn.Parameter(torch.empty(PROMPT_TOKENS, CONDITION_DIM))
        self.output_projection = nn.Linear(CONDITION_DIM, MODEL_DIM, bias=True)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Match the initializations locked for the original Dynamic comparison.
        nn.init.normal_(self.shared_prompt, mean=0.0, std=0.02)
        # Keep PyTorch's standard Linear initialization for the first layer,
        # as in the original 4096->256 Meta-Net.
        self.view_projection.reset_parameters()
        nn.init.normal_(self.queries, mean=0.0, std=0.02)
        nn.init.normal_(self.output_projection.weight, mean=0.0, std=1e-4)
        nn.init.zeros_(self.output_projection.bias)

    def forward(self, condition: torch.Tensor) -> ViewPromptOutput:
        if condition.ndim != 3 or tuple(condition.shape[1:]) != (VIEWS, MODEL_DIM):
            raise ValueError(
                "VC condition must have shape [B,24,4096], found "
                f"{tuple(condition.shape)}"
            )
        if not torch.isfinite(condition).all():
            raise ValueError("VC condition contains NaN or Inf")

        # Frozen view conditions never receive gradients; all trainable mapping
        # operations below remain in the graph.
        frozen_condition = condition.detach().float()
        hidden = F.relu(self.view_projection(frozen_condition))
        logits = torch.einsum("md,bvd->bmv", self.queries, hidden) / 16.0
        attention = logits.softmax(dim=-1)
        attended = torch.einsum("bmv,bvd->bmd", attention, hidden)
        delta = self.output_projection(attended)
        dynamic_prompt = self.shared_prompt.unsqueeze(0) + delta
        return ViewPromptOutput(
            dynamic_prompt=dynamic_prompt,
            delta=delta,
            attention=attention,
            projected_views=hidden,
            attended_condition=attended,
        )

    @staticmethod
    def expected_trainable_parameters() -> int:
        return 2_118_912

    @classmethod
    def fair_original_initialization(cls, seed: int) -> "ViewConditionedPrompt":
        """Copy every shape-compatible tensor from the original Dynamic start.

        Queries are the sole extra tensor and use an isolated deterministic RNG,
        so adding them cannot perturb the original Shared/W1/W2 initialization.
        """
        from conditional_prompt.dynamic_prompt import FrozenMeanConditionedPrompt

        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(seed))
            original = FrozenMeanConditionedPrompt()
        instance = cls()
        query_rng = torch.Generator(device="cpu").manual_seed(int(seed) + 1)
        with torch.no_grad():
            instance.shared_prompt.copy_(original.shared_prompt)
            instance.view_projection.weight.copy_(original.meta_net[0].weight)
            instance.view_projection.bias.copy_(original.meta_net[0].bias)
            instance.output_projection.weight.copy_(original.meta_net[2].weight)
            instance.output_projection.bias.copy_(original.meta_net[2].bias)
            instance.queries.normal_(mean=0.0, std=0.02, generator=query_rng)
        return instance


def attention_diagnostics(attention: torch.Tensor) -> dict[str, float]:
    """Return the pre-registered attention diagnostics for a batch."""

    if attention.ndim != 3 or tuple(attention.shape[1:]) != (4, 24):
        raise ValueError("attention must have shape [B,4,24]")
    values = attention.float().clamp_min(torch.finfo(torch.float32).tiny)
    entropy = -(values * values.log()).sum(dim=-1)
    uniform = 1.0 / 24.0
    kl_uniform = (values * (values.log() - torch.log(values.new_tensor(uniform)))).sum(-1)
    query_pairs = []
    for left in range(4):
        for right in range(left + 1, 4):
            query_pairs.append((values[:, left] - values[:, right]).norm(dim=-1))
    return {
        "normalized_entropy_mean": float((entropy / torch.log(values.new_tensor(24.0))).mean().detach()),
        "kl_from_uniform_mean": float(kl_uniform.mean().detach()),
        "max_weight_mean": float(values.max(dim=-1).values.mean().detach()),
        "inter_query_l2_mean": float(torch.stack(query_pairs, dim=-1).mean().detach()),
    }
