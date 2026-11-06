"""Reusable cached-gradient training primitives for Dynamic Prompt scheme A."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from conditional_prompt.dynamic_object_encoder import load_frozen_mean24_condition
from conditional_prompt.dynamic_prompt import FrozenMeanConditionedPrompt
from conditional_prompt.static_object_encoder import (
    StaticPromptObjectEncoder,
    build_independent_view_batch,
    object_view_paths,
)
from conditional_prompt.static_prompt_encoder import Qwen3VLStaticPromptEncoder
from conditional_prompt.common import (
    capture_rng_state,
    restore_rng_state,
)
from loss.supervised_contrastive import supervised_contrastive_loss
from utils.io import move_inputs


VIEWS_PER_OBJECT = 24
EMBEDDING_DIMENSION = 4096
PARAMETER_NAMES = (
    "shared_prompt",
    "meta_net.0.weight",
    "meta_net.0.bias",
    "meta_net.2.weight",
    "meta_net.2.bias",
)


def named_dynamic_parameters(
    generator: FrozenMeanConditionedPrompt,
) -> dict[str, torch.nn.Parameter]:
    parameters = dict(generator.named_parameters())
    if tuple(parameters) != PARAMETER_NAMES:
        raise RuntimeError(f"unexpected Dynamic parameters: {tuple(parameters)}")
    return parameters


def cache_object_views(
    *,
    object_encoder: StaticPromptObjectEncoder,
    generator: FrozenMeanConditionedPrompt,
    row: dict[str, str],
    condition: torch.Tensor,
    device: torch.device,
    model_dtype: torch.dtype,
) -> torch.Tensor:
    with torch.no_grad():
        prompt = generator(condition.unsqueeze(0)).dynamic_prompt[0]
        output = object_encoder(
            row,
            device=device,
            model_dtype=model_dtype,
            prompt_embeddings=prompt,
        )
    return output.view_embeddings


def replay_batch_backward(
    *,
    generator: FrozenMeanConditionedPrompt,
    view_encoder: Qwen3VLStaticPromptEncoder,
    processor,
    batch_rows: list[dict[str, str]],
    conditions: list[torch.Tensor],
    view_upstream: torch.Tensor,
    device: torch.device,
    model_dtype: torch.dtype,
    chunk_size: int,
    rng_state,
    loss_scale: float,
) -> int:
    """Replay all object/view chunks and accumulate one scaled parameter VJP."""

    generator.zero_grad(set_to_none=True)
    view_encoder.zero_grad(set_to_none=True)
    restore_rng_state(rng_state, device)
    processed = 0
    for object_index, (row, condition) in enumerate(
        zip(batch_rows, conditions, strict=True)
    ):
        _, paths = object_view_paths(row)
        for start in range(0, len(paths), chunk_size):
            chunk_paths = paths[start : start + chunk_size]
            inputs = build_independent_view_batch(processor, chunk_paths)
            batch = move_inputs(inputs, device, model_dtype)
            dynamic = generator(condition.unsqueeze(0))
            output = view_encoder(
                **batch,
                prompt_embeddings=dynamic.dynamic_prompt[0],
            )
            upstream = view_upstream[
                object_index, start : start + len(chunk_paths)
            ]
            (torch.sum(output.embedding * upstream) * loss_scale).backward()
            processed += len(chunk_paths)
            del dynamic, output, batch, inputs

    for name, parameter in named_dynamic_parameters(generator).items():
        if parameter.grad is None:
            raise RuntimeError(f"{name} received no cached gradient")
        parameter.grad.div_(loss_scale)
    return processed


def grouped_tensor_norm(values: dict[str, torch.Tensor], prefix: str) -> float:
    tensors = [value.flatten() for name, value in values.items() if name.startswith(prefix)]
    if not tensors:
        raise RuntimeError(f"parameter group not found: {prefix}")
    return float(torch.cat(tensors).norm().item())


def train_dynamic_one_step(
    *,
    generator: FrozenMeanConditionedPrompt,
    view_encoder: Qwen3VLStaticPromptEncoder,
    object_encoder: StaticPromptObjectEncoder,
    processor,
    optimizer: torch.optim.Optimizer,
    batch_rows: list[dict[str, str]],
    labels: torch.Tensor,
    view_cache_dir,
    device: torch.device,
    model_dtype: torch.dtype,
    chunk_size: int,
    temperature: float,
    conditions: list[torch.Tensor] | None = None,
) -> dict[str, Any]:
    """Perform exactly one validated P=2,K=2 Dynamic optimizer update."""

    if chunk_size != 4:
        raise ValueError("Dynamic training requires the fixed chunk_size=4")
    if len(batch_rows) != 4 or len({row["object_id"] for row in batch_rows}) != 4:
        raise ValueError("Dynamic training requires four distinct objects")

    if conditions is None:
        conditions = [
            load_frozen_mean24_condition(row, view_cache_dir)[0]
            for row in batch_rows
        ]
    if len(conditions) != len(batch_rows):
        raise ValueError("condition count differs from Dynamic batch size")
    conditions = [condition.detach().float().to(device) for condition in conditions]
    if any(condition.requires_grad for condition in conditions):
        raise RuntimeError("Frozen Mean-24 condition unexpectedly requires grad")

    replay_rng_state = capture_rng_state(device)
    restore_rng_state(replay_rng_state, device)
    cached = [
        cache_object_views(
            object_encoder=object_encoder,
            generator=generator,
            row=row,
            condition=condition,
            device=device,
            model_dtype=model_dtype,
        )
        for row, condition in zip(batch_rows, conditions, strict=True)
    ]
    cached_values = torch.stack(cached).to(device).float().detach()
    expected = (4, VIEWS_PER_OBJECT, EMBEDDING_DIMENSION)
    if tuple(cached_values.shape) != expected:
        raise RuntimeError(f"cached shape {tuple(cached_values.shape)} != {expected}")

    leaves = cached_values.requires_grad_(True)
    object_embeddings = F.normalize(leaves.mean(dim=1), dim=-1)
    loss = supervised_contrastive_loss(
        object_embeddings, labels.to(device), temperature
    )
    object_upstream = torch.autograd.grad(
        loss, object_embeddings, retain_graph=True
    )[0].detach()
    view_upstream = torch.autograd.grad(
        object_embeddings, leaves, grad_outputs=object_upstream
    )[0].detach()
    upstream_norm = float(view_upstream.norm().item())
    if not torch.isfinite(loss) or not torch.isfinite(view_upstream).all():
        raise RuntimeError("Dynamic SupCon loss/upstream is non-finite")
    if upstream_norm <= 0:
        raise RuntimeError("Dynamic SupCon produced zero view upstream")

    # Late in training, an easy P-K Batch can produce gradients below FP16's
    # representable range while traversing frozen Qwen. Try progressively
    # larger proxy scales and always unscale at the FP32 Dynamic parameters.
    # This changes numerical representation only, not the intended gradient.
    loss_scale = 1.0
    processed = 0
    gradients: dict[str, torch.Tensor] = {}
    scale_attempts: list[dict[str, object]] = []
    for candidate_scale in (1.0, 1024.0, 16384.0, 65536.0):
        loss_scale = candidate_scale
        optimizer.zero_grad(set_to_none=True)
        processed = replay_batch_backward(
            generator=generator,
            view_encoder=view_encoder,
            processor=processor,
            batch_rows=batch_rows,
            conditions=conditions,
            view_upstream=view_upstream,
            device=device,
            model_dtype=model_dtype,
            chunk_size=chunk_size,
            rng_state=replay_rng_state,
            loss_scale=loss_scale,
        )
        current_gradients = {
            name: parameter.grad.detach().float().clone()
            for name, parameter in named_dynamic_parameters(generator).items()
        }
        status = {
            name: {
                "finite": bool(torch.isfinite(value).all().item()),
                "norm": float(value.norm().item()),
            }
            for name, value in current_gradients.items()
        }
        scale_attempts.append({"loss_scale": candidate_scale, "parameters": status})
        all_finite = all(item["finite"] for item in status.values())
        all_nonzero = all(item["norm"] > 0.0 for item in status.values())
        if all_finite and all_nonzero:
            gradients = current_gradients
            break
        # A larger scale cannot repair an overflow/non-finite gradient.
        if not all_finite:
            break
    if not gradients:
        raise RuntimeError(
            "Dynamic gradient has no finite nonzero loss scale; "
            f"loss={float(loss.detach().item()):.10g}, "
            f"view_upstream_norm={upstream_norm:.10g}, "
            f"attempts={scale_attempts}"
        )
    if processed != 96:
        raise RuntimeError(f"expected 96 replayed views, found {processed}")
    if any(parameter.grad is not None for parameter in view_encoder.backbone.parameters()):
        raise RuntimeError("frozen Qwen received a parameter gradient")

    before = {
        name: parameter.detach().float().clone()
        for name, parameter in named_dynamic_parameters(generator).items()
    }
    optimizer.step()
    updates = {
        name: parameter.detach().float() - before[name]
        for name, parameter in named_dynamic_parameters(generator).items()
    }
    if any(
        not torch.isfinite(value).all() or float(value.norm().item()) <= 0
        for value in updates.values()
    ):
        raise RuntimeError("Dynamic optimizer update is zero/non-finite")
    parameters_finite = all(
        torch.isfinite(parameter).all().item() for parameter in generator.parameters()
    )
    if not parameters_finite:
        raise RuntimeError("Dynamic parameter became non-finite")

    return {
        "supcon_loss": float(loss.detach().item()),
        "processed_backward_views": processed,
        "backward_loss_scale": loss_scale,
        "backward_loss_scale_attempts": scale_attempts,
        "view_embedding_upstream_grad_norm": upstream_norm,
        "shared_prompt_grad_norm": float(gradients["shared_prompt"].norm().item()),
        "meta_net_layer1_grad_norm": grouped_tensor_norm(gradients, "meta_net.0."),
        "meta_net_layer2_grad_norm": grouped_tensor_norm(gradients, "meta_net.2."),
        "shared_prompt_update_norm": float(updates["shared_prompt"].norm().item()),
        "meta_net_layer1_update_norm": grouped_tensor_norm(updates, "meta_net.0."),
        "meta_net_layer2_update_norm": grouped_tensor_norm(updates, "meta_net.2."),
        "gradient_finite": True,
        "update_finite": True,
        "parameters_finite": parameters_finite,
        "embedding_finite": bool(torch.isfinite(cached_values).all().item()),
        "loss_finite": bool(torch.isfinite(loss).item()),
        "backbone_parameters_with_grad": 0,
    }
