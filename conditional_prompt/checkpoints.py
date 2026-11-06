"""Inference-time validation and loading for prompt-learning checkpoints."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from conditional_prompt.dynamic_prompt import FrozenMeanConditionedPrompt
from conditional_prompt.dynamic_conditions import validate_condition_state
from utils.io import PROMPT


STATIC_PROMPT_SHAPE = (4, 4096)
STATIC_INSERTION = "after_chat_prefix_before_complete_visual_block"
STATIC_VIEW_POOLING = "last_hidden_state_all_valid_tokens_attention_mask_mean_l2"
STATIC_OBJECT_FUSION = "24_view_embeddings_mean_then_l2"


def validate_static_checkpoint(
    payload: dict[str, Any], checkpoint_path: Path, model_path: Path
) -> tuple[torch.Tensor, int]:
    """Validate a step-based or epoch-based Static Prompt checkpoint."""

    stage = payload.get("stage")
    if stage not in {"4B-1", "static_prompt_epoch_training"}:
        raise ValueError(f"unsupported static-prompt checkpoint: {checkpoint_path}")
    if payload.get("format_version") != 1:
        raise ValueError(f"unsupported static-prompt checkpoint: {checkpoint_path}")
    if stage == "4B-1":
        checkpoint_step = int(payload.get("completed_step", -1))
    else:
        progress = payload.get("progress", {})
        checkpoint_step = int(progress.get("global_step", -1))
        sampler_state = payload.get("sampler_state", {})
        if int(progress.get("epoch_step", -1)) != int(
            sampler_state.get("batch_count", -2)
        ):
            raise ValueError("epoch checkpoint progress differs from sampler cursor")
    if checkpoint_step < 0:
        raise ValueError("checkpoint has an invalid training step")

    prompt = payload.get("soft_prompt")
    if not isinstance(prompt, torch.Tensor) or tuple(prompt.shape) != STATIC_PROMPT_SHAPE:
        raise ValueError("checkpoint soft_prompt must have shape [4, 4096]")
    if not torch.isfinite(prompt).all():
        raise ValueError("checkpoint soft_prompt contains NaN/Inf")

    configuration = payload.get("configuration", {})
    metadata = payload.get("metadata", {})
    expected_model = str(model_path.resolve())
    if configuration.get("model_path") != expected_model:
        raise ValueError("checkpoint configuration model_path does not match")
    if metadata.get("model_path") != expected_model:
        raise ValueError("checkpoint metadata model_path does not match")
    expected = {
        "fixed_instruction": PROMPT,
        "soft_prompt_tokens": STATIC_PROMPT_SHAPE[0],
        "soft_prompt_insertion": STATIC_INSERTION,
        "view_pooling": STATIC_VIEW_POOLING,
        "object_fusion": STATIC_OBJECT_FUSION,
        "embedding_dimension": STATIC_PROMPT_SHAPE[1],
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError(f"checkpoint metadata differs for {key}")

    latest_record = payload.get("latest_record", {})
    record_key = "step" if stage == "4B-1" else "global_step"
    if int(latest_record.get(record_key, -1)) != checkpoint_step:
        raise ValueError("checkpoint step differs from its latest training record")
    actual_norm = float(prompt.float().norm().item())
    if "soft_prompt_param_norm" in latest_record:
        recorded_norm = float(latest_record["soft_prompt_param_norm"])
        if not torch.isfinite(torch.tensor(recorded_norm)):
            raise ValueError("checkpoint recorded prompt norm is non-finite")
        if abs(recorded_norm - actual_norm) > 1e-6:
            raise ValueError("checkpoint prompt differs from latest training record")
    return prompt.detach().cpu(), checkpoint_step


def validate_dynamic_checkpoint(
    path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load and validate a completed dataset-specific Dynamic Epoch checkpoint."""

    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("stage") != "dynamic_prompt_epoch_training":
        raise ValueError("checkpoint is not a Dynamic Prompt Epoch checkpoint")
    if payload.get("format_version") != 1:
        raise ValueError("unsupported Dynamic Prompt checkpoint version")
    config = payload.get("configuration")
    metadata = payload.get("metadata")
    progress = payload.get("progress")
    if not all(isinstance(item, dict) for item in (config, metadata, progress)):
        raise ValueError("checkpoint configuration/metadata/progress is missing")

    training_mode = config.get("training_mode")
    condition_mode = config.get("condition_mode", "correct")
    expected_steps = {
        "internal_dev": 541,
        "full8_exploratory": 706,
        "esb_exploratory": 26,
        "esb_internal_dev": 20,
    }
    if training_mode not in expected_steps:
        raise ValueError(f"unsupported Dynamic training_mode: {training_mode!r}")
    expected_config = {
        "views_per_object": 24,
        "chunk_size": 4,
        "prompt_tokens": 4,
        "prompt_dimension": 4096,
        "meta_net_dimensions": [4096, 256, 4096],
        "condition_source": "frozen_mean_24_fp32_mean_l2",
    }
    for key, expected in expected_config.items():
        if config.get(key) != expected:
            raise ValueError(f"checkpoint configuration differs for {key}")
    condition_state = payload.get("condition_state", {"mode": "correct"})
    validate_condition_state(condition_state, condition_mode)
    expected_metadata = {
        "fixed_instruction": PROMPT,
        "prompt_insertion": STATIC_INSERTION,
        "view_pooling": STATIC_VIEW_POOLING,
        "object_fusion": "24_view_embeddings_fp32_mean_then_l2",
        "embedding_dimension": 4096,
        "condition_requires_grad": False,
        "qwen_frozen": True,
    }
    for key, expected in expected_metadata.items():
        if metadata.get(key) != expected:
            raise ValueError(f"checkpoint metadata differs for {key}")
    step = int(progress.get("global_step", -1))
    steps_per_epoch = expected_steps[training_mode]
    sampler_state = payload.get("sampler_state", {})
    batch_count = int(sampler_state.get("batch_count", -1))
    plan_cursor = int(sampler_state.get("plan_cursor", -1))
    batch_plan = sampler_state.get("batch_plan", [])
    if (
        step != int(progress.get("last_successful_optimizer_step", -2))
        or step < steps_per_epoch
        or step % steps_per_epoch != 0
        or int(progress.get("epoch_step", -1)) != steps_per_epoch
        or batch_count != steps_per_epoch
        or plan_cursor != len(batch_plan)
        or len(batch_plan) != steps_per_epoch
    ):
        raise ValueError(
            f"checkpoint is not a completed {training_mode} Epoch state "
            f"with a positive multiple of {steps_per_epoch} successful steps"
        )

    shared = payload.get("shared_prompt")
    meta_state = payload.get("meta_net_state")
    if not isinstance(shared, torch.Tensor) or tuple(shared.shape) != (4, 4096):
        raise ValueError("checkpoint Shared Prompt must have shape [4, 4096]")
    if not torch.isfinite(shared).all():
        raise ValueError("checkpoint Shared Prompt contains NaN/Inf")
    if not isinstance(meta_state, dict) or not meta_state:
        raise ValueError("checkpoint Meta-Net state is missing")
    if any(
        not isinstance(value, torch.Tensor) or not torch.isfinite(value).all()
        for value in meta_state.values()
    ):
        raise ValueError("checkpoint Meta-Net state contains invalid values")
    return payload, config


def load_dynamic_prompt_generator(
    payload: dict[str, Any], device: torch.device
) -> FrozenMeanConditionedPrompt:
    """Restore a Dynamic Prompt generator as a frozen inference module."""

    generator = FrozenMeanConditionedPrompt().to(device)
    with torch.no_grad():
        generator.shared_prompt.copy_(payload["shared_prompt"].to(device))
    generator.meta_net.load_state_dict(payload["meta_net_state"], strict=True)
    generator.eval()
    generator.requires_grad_(False)
    return generator
