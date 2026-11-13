"""Gradient-caching update for View-Prompt."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from conditional_prompt.common import capture_rng_state, restore_rng_state
from conditional_prompt.static_object_encoder import (
    StaticPromptObjectEncoder,
    build_independent_view_batch,
    object_view_paths,
)
from conditional_prompt.static_prompt_encoder import Qwen3VLStaticPromptEncoder
from conditional_prompt.view_prompt.model import ViewConditionedPrompt, attention_diagnostics
from loss.supervised_contrastive import supervised_contrastive_loss
from utils.io import move_inputs


PARAMETER_NAMES = (
    "shared_prompt",
    "queries",
    "view_projection.weight",
    "view_projection.bias",
    "output_projection.weight",
    "output_projection.bias",
)


def named_view_prompt_parameters(
    generator: ViewConditionedPrompt,
) -> dict[str, torch.nn.Parameter]:
    parameters = dict(generator.named_parameters())
    if set(parameters) != set(PARAMETER_NAMES):
        raise RuntimeError(f"unexpected VC parameters: {tuple(parameters)}")
    return parameters


def _cache_views(*, generator, object_encoder, row, condition, device, model_dtype):
    with torch.no_grad():
        prompt = generator(condition.unsqueeze(0)).dynamic_prompt[0]
        return object_encoder(
            row, device=device, model_dtype=model_dtype, prompt_embeddings=prompt
        ).view_embeddings


def _replay_backward(
    *, generator, view_encoder, processor, rows, conditions, upstream,
    device, model_dtype, chunk_size, rng_state, loss_scale,
) -> int:
    generator.zero_grad(set_to_none=True)
    view_encoder.zero_grad(set_to_none=True)
    restore_rng_state(rng_state, device)
    processed = 0
    for object_index, (row, condition) in enumerate(zip(rows, conditions, strict=True)):
        _, paths = object_view_paths(row)
        # Generate the object prompt inside every replay graph. Its gradient is
        # accumulated across all 24 independent views.
        for start in range(0, 24, chunk_size):
            inputs = build_independent_view_batch(processor, paths[start:start + chunk_size])
            batch = move_inputs(inputs, device, model_dtype)
            prompt = generator(condition.unsqueeze(0)).dynamic_prompt[0]
            output = view_encoder(**batch, prompt_embeddings=prompt)
            proxy = (output.embedding * upstream[object_index, start:start + len(output.embedding)]).sum()
            (proxy * loss_scale).backward()
            processed += len(output.embedding)
            del output, prompt, batch, inputs
    for name, parameter in named_view_prompt_parameters(generator).items():
        if parameter.grad is None:
            raise RuntimeError(f"{name} received no cached gradient")
        parameter.grad.div_(loss_scale)
    return processed


def _delta_diagnostics(generator: ViewConditionedPrompt, conditions: list[torch.Tensor]):
    with torch.no_grad():
        output = generator(torch.stack(conditions))
        delta = output.delta.float()
        flat = delta.flatten(1)
        object_cosine = F.normalize(flat, dim=-1) @ F.normalize(flat, dim=-1).T
        upper = torch.triu_indices(len(flat), len(flat), offset=1, device=flat.device)
        token_cosines = []
        for item in delta:
            normalized = F.normalize(item, dim=-1)
            matrix = normalized @ normalized.T
            token_upper = torch.triu_indices(4, 4, offset=1, device=flat.device)
            token_cosines.append(matrix[token_upper[0], token_upper[1]].mean())
        norms = flat.norm(dim=-1)
        prompt_norm = generator.shared_prompt.float().norm()
        return {
            **attention_diagnostics(output.attention),
            "delta_norm_mean": float(norms.mean()),
            "delta_to_shared_norm_mean": float((norms / prompt_norm).mean()),
            "delta_object_pairwise_cosine_mean": float(object_cosine[upper[0], upper[1]].mean()),
            "delta_token_pairwise_cosine_mean": float(torch.stack(token_cosines).mean()),
        }


def train_view_prompt_one_step(
    *, generator: ViewConditionedPrompt, view_encoder: Qwen3VLStaticPromptEncoder,
    object_encoder: StaticPromptObjectEncoder, processor, optimizer,
    batch_rows: list[dict[str, str]], labels: torch.Tensor,
    conditions: list[torch.Tensor], device: torch.device, model_dtype: torch.dtype,
    chunk_size: int = 4, temperature: float = 0.05,
) -> dict[str, Any]:
    if len(batch_rows) != 4 or len({r["object_id"] for r in batch_rows}) != 4:
        raise ValueError("VC training requires four distinct P=2,K=2 objects")
    if chunk_size != 4:
        raise ValueError("the locked VC experiment requires chunk_size=4")
    conditions = [value.detach().float().to(device) for value in conditions]
    if any(tuple(value.shape) != (24, 4096) for value in conditions):
        raise ValueError("every VC condition must have shape [24,4096]")

    rng_state = capture_rng_state(device)
    cached = torch.stack([
        _cache_views(
            generator=generator, object_encoder=object_encoder, row=row,
            condition=condition, device=device, model_dtype=model_dtype,
        )
        for row, condition in zip(batch_rows, conditions, strict=True)
    ]).to(device).float().detach()
    if tuple(cached.shape) != (4, 24, 4096):
        raise RuntimeError(f"wrong cached shape: {tuple(cached.shape)}")
    leaves = cached.requires_grad_(True)
    objects = F.normalize(leaves.mean(dim=1), dim=-1)
    loss = supervised_contrastive_loss(objects, labels.to(device), temperature)
    upstream = torch.autograd.grad(loss, leaves)[0].detach()
    if not torch.isfinite(upstream).all() or upstream.norm() == 0:
        raise RuntimeError("invalid VC cached upstream")

    before = {
        name: p.detach().float().clone()
        for name, p in named_view_prompt_parameters(generator).items()
    }
    attempts = []
    selected_scale = None
    for scale in (1.0, 1024.0, 16384.0, 65536.0):
        optimizer.zero_grad(set_to_none=True)
        processed = _replay_backward(
            generator=generator, view_encoder=view_encoder, processor=processor,
            rows=batch_rows, conditions=conditions, upstream=upstream,
            device=device, model_dtype=model_dtype, chunk_size=chunk_size,
            rng_state=rng_state, loss_scale=scale,
        )
        status = {
            name: {"finite": bool(torch.isfinite(p.grad).all()), "norm": float(p.grad.float().norm())}
            for name, p in named_view_prompt_parameters(generator).items()
        }
        attempts.append({"loss_scale": scale, "parameters": status})
        if all(item["finite"] and item["norm"] > 0 for item in status.values()):
            selected_scale = scale
            break
    if selected_scale is None:
        raise RuntimeError(f"no finite nonzero VC gradient scale: {attempts}")
    if processed != 96:
        raise RuntimeError(f"expected 96 replayed views, found {processed}")
    if any(p.grad is not None for p in view_encoder.backbone.parameters()):
        raise RuntimeError("frozen Qwen received gradients")
    gradient_norms = {
        name: float(p.grad.float().norm())
        for name, p in named_view_prompt_parameters(generator).items()
    }
    optimizer.step()
    update_norms = {
        name: float((p.detach().float() - before[name]).norm())
        for name, p in named_view_prompt_parameters(generator).items()
    }
    if any(value <= 0 for value in update_norms.values()):
        raise RuntimeError(f"zero VC update: {update_norms}")
    return {
        "supcon_loss": float(loss.detach()),
        "view_upstream_norm": float(upstream.norm()),
        "selected_loss_scale": selected_scale,
        "gradient_attempts": attempts,
        "gradient_norms": gradient_norms,
        "update_norms": update_norms,
        "diagnostics_after_update": _delta_diagnostics(generator, conditions),
    }
