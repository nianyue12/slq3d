#!/usr/bin/env python3
"""Stage-4B-1 multi-step training for the shared static soft prompt only."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from conditional_prompt.static_object_encoder import (
    StaticPromptObjectEncoder,
    build_independent_view_batch,
    object_view_paths,
)
from conditional_prompt.static_prompt_encoder import Qwen3VLStaticPromptEncoder
from conditional_prompt.common import (
    DEFAULT_MODEL_PATH,
    DEFAULT_PROTOCOL,
    DEFAULT_QUERY_MANIFEST,
    DEFAULT_TARGET_MANIFEST,
    DEFAULT_TRAIN_MANIFEST,
    capture_rng_state,
    restore_rng_state,
    validate_training_source,
)
from dataset.multiview_inputs import read_rows
from loss.supervised_contrastive import supervised_contrastive_loss
from train import balanced_batch
from utils.io import PROMPT, move_inputs


VIEWS_PER_OBJECT = 24
PROMPT_TOKENS = 4
PROMPT_INIT_STD = 0.02
EMBEDDING_DIMENSION = 4096


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train only the four-token static prompt with gradient caching."
    )
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_TRAIN_MANIFEST)
    parser.add_argument("--split", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--query-manifest", type=Path, default=DEFAULT_QUERY_MANIFEST)
    parser.add_argument("--target-manifest", type=Path, default=DEFAULT_TARGET_MANIFEST)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--classes-per-batch", type=int, default=2)
    parser.add_argument("--objects-per-class", type=int, default=2)
    parser.add_argument("--chunk-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--temperature", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=20261020)
    parser.add_argument("--checkpoint-every", type=int, default=1)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    """Reject configurations that cannot form a valid object-level batch."""

    positive_integer_names = (
        "steps",
        "classes_per_batch",
        "objects_per_class",
        "chunk_size",
        "checkpoint_every",
    )
    for name in positive_integer_names:
        if getattr(args, name) < 1:
            raise ValueError(f"{name.replace('_', '-')} must be positive")
    if args.classes_per_batch < 2:
        raise ValueError("classes-per-batch must be at least two")
    if args.objects_per_class < 2:
        raise ValueError("objects-per-class must be at least two for SupCon positives")
    if args.learning_rate <= 0:
        raise ValueError("learning-rate must be positive")
    if args.temperature <= 0:
        raise ValueError("temperature must be positive")


def resolved_configuration(args: argparse.Namespace) -> dict[str, Any]:
    """Return the complete JSON-serializable experiment configuration."""

    return {
        "model_path": str(args.model_path.resolve()),
        "manifest": str(args.manifest.resolve()),
        "split": str(args.split.resolve()),
        "query_manifest": str(args.query_manifest.resolve()),
        "target_manifest": str(args.target_manifest.resolve()),
        "output_dir": str(args.output_dir.resolve()),
        "device": args.device,
        "steps": args.steps,
        "classes_per_batch": args.classes_per_batch,
        "objects_per_class": args.objects_per_class,
        "views_per_object": VIEWS_PER_OBJECT,
        "chunk_size": args.chunk_size,
        "learning_rate": args.learning_rate,
        "temperature": args.temperature,
        "seed": args.seed,
        "checkpoint_every": args.checkpoint_every,
        "prompt_tokens": PROMPT_TOKENS,
        "prompt_init_std": PROMPT_INIT_STD,
    }


def git_version() -> dict[str, Any]:
    """Return the repository commit and dirty flag when Git is available."""

    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPOSITORY_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=REPOSITORY_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        return {"commit": commit, "dirty": bool(status.strip())}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}


def training_metadata(args: argparse.Namespace, processor, backbone) -> dict[str, Any]:
    """Describe the exact model input, prompt placement, pooling, and fusion."""

    chat_template = processor.chat_template or ""
    return {
        "stage": "4B-1",
        "model_path": str(args.model_path.resolve()),
        "model_type": getattr(backbone.config, "model_type", None),
        "model_name_or_path": getattr(backbone.config, "_name_or_path", None),
        "fixed_instruction": PROMPT,
        "chat_template_sha256": hashlib.sha256(
            chat_template.encode("utf-8")
        ).hexdigest(),
        "soft_prompt_tokens": PROMPT_TOKENS,
        "soft_prompt_insertion": "after_chat_prefix_before_complete_visual_block",
        "view_pooling": "last_hidden_state_all_valid_tokens_attention_mask_mean_l2",
        "object_fusion": "24_view_embeddings_mean_then_l2",
        "embedding_dimension": EMBEDDING_DIMENSION,
        "loss": "object_level_supervised_contrastive",
        "optimizer": "AdamW",
        "git": git_version(),
    }


def checkpoint_rng_state(
    sampler: random.Random,
) -> dict[str, Any]:
    """Capture every RNG that can affect initialization, sampling, or replay."""

    return {
        "python_global": random.getstate(),
        "sampler": sampler.getstate(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_checkpoint_rng_state(state: dict[str, Any], sampler: random.Random) -> None:
    """Restore RNGs exactly so resumed batches match uninterrupted training."""

    random.setstate(state["python_global"])
    sampler.setstate(state["sampler"])
    # The checkpoint itself is loaded with ``map_location=device`` so that
    # optimizer tensors are restored on the training device. That also moves
    # serialized RNG ByteTensors to CUDA, but both CPU and CUDA RNG restore
    # APIs require their state arguments to be CPU ByteTensors.
    torch.set_rng_state(state["torch_cpu"].cpu())
    if state["torch_cuda_all"] is not None:
        torch.cuda.set_rng_state_all(
            [cuda_state.cpu() for cuda_state in state["torch_cuda_all"]]
        )


def save_checkpoint_atomic(payload: dict[str, Any], path: Path) -> None:
    """Atomically replace a checkpoint so interruption cannot leave a partial file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def checkpoint_payload(
    *,
    encoder: Qwen3VLStaticPromptEncoder,
    optimizer: torch.optim.Optimizer,
    completed_step: int,
    sampler: random.Random,
    configuration: dict[str, Any],
    metadata: dict[str, Any],
    latest_record: dict[str, Any],
) -> dict[str, Any]:
    """Build a self-contained resumable static-prompt checkpoint."""

    return {
        "format_version": 1,
        "stage": "4B-1",
        "completed_step": completed_step,
        "soft_prompt": encoder.soft_prompt.detach().cpu().clone(),
        "optimizer_state": optimizer.state_dict(),
        "rng_state": checkpoint_rng_state(sampler),
        "configuration": configuration,
        "metadata": metadata,
        "latest_record": latest_record,
    }


def compatibility_configuration(configuration: dict[str, Any]) -> dict[str, Any]:
    """Select settings that must not change across an exact resume."""

    ignored = {"steps", "checkpoint_every", "output_dir"}
    return {key: value for key, value in configuration.items() if key not in ignored}


def resume_from_checkpoint(
    path: Path,
    *,
    encoder: Qwen3VLStaticPromptEncoder,
    optimizer: torch.optim.Optimizer,
    sampler: random.Random,
    configuration: dict[str, Any],
    device: torch.device,
) -> int:
    """Restore prompt, optimizer, step, and all RNG/sampler states."""

    payload = torch.load(path, map_location=device, weights_only=False)
    if payload.get("stage") != "4B-1" or payload.get("format_version") != 1:
        raise ValueError(f"unsupported checkpoint format: {path}")
    saved_configuration = payload["configuration"]
    if compatibility_configuration(saved_configuration) != compatibility_configuration(
        configuration
    ):
        raise ValueError("resume configuration differs from the checkpoint")

    supplied_prompt = payload["soft_prompt"]
    if supplied_prompt.shape != encoder.soft_prompt.shape:
        raise ValueError(
            f"checkpoint prompt shape {tuple(supplied_prompt.shape)} does not match "
            f"{tuple(encoder.soft_prompt.shape)}"
        )
    encoder.soft_prompt.data.copy_(
        supplied_prompt.to(device=device, dtype=encoder.soft_prompt.dtype)
    )
    saved_prompt_norm = float(supplied_prompt.float().norm().item())
    recorded_prompt_norm = float(
        payload["latest_record"]["soft_prompt_param_norm"]
    )
    if abs(saved_prompt_norm - recorded_prompt_norm) > 1e-6:
        raise ValueError("checkpoint prompt does not match its latest training record")
    optimizer.load_state_dict(payload["optimizer_state"])
    restore_checkpoint_rng_state(payload["rng_state"], sampler)
    completed_step = int(payload["completed_step"])
    if completed_step < 0:
        raise ValueError("checkpoint completed_step is negative")
    return completed_step


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    """Append one durable, machine-readable training record."""

    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()


def train_one_step(
    *,
    encoder: Qwen3VLStaticPromptEncoder,
    object_encoder: StaticPromptObjectEncoder,
    processor,
    optimizer: torch.optim.Optimizer,
    batch_rows: list[dict[str, str]],
    labels: torch.Tensor,
    device: torch.device,
    model_dtype: torch.dtype,
    chunk_size: int,
    temperature: float,
) -> dict[str, Any]:
    """Run one verified cached-gradient SupCon update over a complete P-K batch."""

    batch_paths = [object_view_paths(row)[1] for row in batch_rows]
    if any(len(paths) != VIEWS_PER_OBJECT for paths in batch_paths):
        raise RuntimeError("a sampled training object does not have 24 valid views")

    replay_rng_state = capture_rng_state(device)
    cached_objects = []
    restore_rng_state(replay_rng_state, device)
    for row in batch_rows:
        output = object_encoder(
            row,
            device=device,
            model_dtype=model_dtype,
        )
        cached_objects.append(output.view_embeddings)

    cached_values = torch.stack(cached_objects).to(device).detach()
    cached_leaves = cached_values.requires_grad_(True)
    object_embeddings = F.normalize(cached_leaves.float().mean(dim=1), dim=-1)
    loss = supervised_contrastive_loss(
        object_embeddings,
        labels.to(device),
        temperature,
    )
    embedding_gradients = torch.autograd.grad(loss, cached_leaves)[0].detach()
    loss_finite = bool(torch.isfinite(loss).item())
    embedding_finite = bool(
        torch.isfinite(cached_values).all().item()
        and torch.isfinite(object_embeddings).all().item()
    )
    upstream_gradient_finite = bool(torch.isfinite(embedding_gradients).all().item())
    upstream_gradient_norm = float(embedding_gradients.float().norm().item())
    if not loss_finite or not embedding_finite:
        raise RuntimeError("loss or embeddings contain NaN/Inf before cached backward")
    if not upstream_gradient_finite:
        raise RuntimeError(
            "view-embedding upstream gradient is non-finite before cached backward; "
            f"loss={float(loss.detach().item()):.8g}, "
            f"upstream_gradient_norm={upstream_gradient_norm:.8g}"
        )

    encoder.train()

    def replay_cached_backward(loss_scale: float) -> tuple[torch.Tensor, int]:
        """Replay all views and return an unscaled FP32 prompt gradient."""

        optimizer.zero_grad(set_to_none=True)
        restore_rng_state(replay_rng_state, device)
        processed = 0
        for object_index, paths in enumerate(batch_paths):
            for start in range(0, len(paths), chunk_size):
                chunk_paths = paths[start : start + chunk_size]
                inputs = build_independent_view_batch(processor, chunk_paths)
                batch = move_inputs(inputs, device, model_dtype)
                replay = encoder(**batch)
                upstream = embedding_gradients[
                    object_index,
                    start : start + len(chunk_paths),
                ]
                (torch.sum(replay.embedding * upstream) * loss_scale).backward()
                processed += len(chunk_paths)

        if encoder.soft_prompt.grad is None:
            raise RuntimeError("cached backward produced no soft-prompt gradient")
        encoder.soft_prompt.grad.div_(loss_scale)
        return encoder.soft_prompt.grad, processed

    backward_loss_scale = 1.0
    prompt_gradient, processed_views = replay_cached_backward(backward_loss_scale)
    gradient_norm = float(prompt_gradient.float().norm().item())
    gradient_finite = bool(torch.isfinite(prompt_gradient).all().item())

    # A nonzero object-level upstream gradient can underflow while traversing
    # the frozen FP16 Qwen backbone. Retry only that case with loss scaling;
    # unscaling at the FP32 prompt parameter preserves the intended gradient.
    if gradient_finite and gradient_norm <= 0 and upstream_gradient_norm > 0:
        backward_loss_scale = 1024.0
        prompt_gradient, processed_views = replay_cached_backward(backward_loss_scale)
        gradient_norm = float(prompt_gradient.float().norm().item())
        gradient_finite = bool(torch.isfinite(prompt_gradient).all().item())
    if not gradient_finite:
        finite_count = int(torch.isfinite(prompt_gradient).sum().item())
        raise RuntimeError(
            "soft-prompt gradient is non-finite before optimizer.step; "
            f"finite_elements={finite_count}/{prompt_gradient.numel()}, "
            f"upstream_gradient_norm={upstream_gradient_norm:.8g}"
        )
    if gradient_norm <= 0:
        raise RuntimeError(
            "soft-prompt gradient is exactly zero before optimizer.step; "
            f"upstream_gradient_norm={upstream_gradient_norm:.8g}"
        )
    backbone_with_grad = sum(
        parameter.grad is not None for parameter in encoder.backbone.parameters()
    )
    if backbone_with_grad != 0:
        raise RuntimeError("frozen Qwen parameters received gradients")

    # Only a fully validated gradient may update the prompt or AdamW state.
    before_step = encoder.soft_prompt.detach().clone()
    optimizer.step()
    update = encoder.soft_prompt.detach() - before_step

    prompt_norm = float(encoder.soft_prompt.detach().float().norm().item())
    prompt_finite = bool(torch.isfinite(encoder.soft_prompt).all().item())
    update_norm = float(update.float().norm().item())
    update_finite = bool(torch.isfinite(update).all().item())
    expected_shape = (len(batch_rows), VIEWS_PER_OBJECT, EMBEDDING_DIMENSION)

    if tuple(cached_values.shape) != expected_shape:
        raise RuntimeError(
            f"expected cached shape {expected_shape}, got {tuple(cached_values.shape)}"
        )
    if processed_views != len(batch_rows) * VIEWS_PER_OBJECT:
        raise RuntimeError("not every sampled view participated in cached backward")
    if not prompt_finite:
        raise RuntimeError("soft-prompt parameters contain NaN/Inf")
    if not update_finite or update_norm <= 0:
        raise RuntimeError("soft-prompt update is zero or non-finite")

    return {
        "supcon_loss": float(loss.detach().item()),
        "soft_prompt_grad_norm": gradient_norm,
        "view_embedding_upstream_grad_norm": upstream_gradient_norm,
        "backward_loss_scale": backward_loss_scale,
        "soft_prompt_param_norm": prompt_norm,
        "soft_prompt_param_finite": prompt_finite,
        "soft_prompt_update_norm": update_norm,
        "loss_finite": loss_finite,
        "embedding_finite": embedding_finite,
        "gradient_finite": gradient_finite,
        "update_finite": update_finite,
        "processed_backward_views": processed_views,
        "backbone_parameters_with_grad": backbone_with_grad,
    }


def main() -> None:
    args = parse_args()
    validate_args(args)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    sampler = random.Random(args.seed)
    configuration = resolved_configuration(args)

    rows, protocol = read_rows(args.manifest, args.split)
    validate_training_source(
        rows,
        protocol,
        args.query_manifest,
        args.target_manifest,
    )
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["label_name"]].append(row)
    grouped = {
        label: objects
        for label, objects in grouped.items()
        if len(objects) >= args.objects_per_class
    }
    if len(grouped) < args.classes_per_batch:
        raise ValueError("classes-per-batch exceeds eligible official training classes")

    if args.resume is None and (args.output_dir / "latest.pt").exists():
        raise FileExistsError(
            f"{args.output_dir / 'latest.pt'} already exists; "
            "use --resume or choose a new output directory"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.output_dir / "train_log.jsonl"

    print("Stage-4B-1 multi-step static soft-prompt training")
    print(json.dumps(configuration, ensure_ascii=False, sort_keys=True))
    print("loading processor and frozen model ...", flush=True)
    processor = AutoProcessor.from_pretrained(args.model_path, local_files_only=True)
    processor.tokenizer.padding_side = "left"
    model_dtype = torch.float16 if device.type == "cuda" else torch.float32
    backbone = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_path,
        dtype=model_dtype,
        low_cpu_mem_usage=True,
        local_files_only=True,
    ).to(device)
    encoder = Qwen3VLStaticPromptEncoder(
        backbone,
        num_prompt_tokens=PROMPT_TOKENS,
        prompt_init_std=PROMPT_INIT_STD,
    ).to(device)
    optimizer = torch.optim.AdamW([encoder.soft_prompt], lr=args.learning_rate)
    object_encoder = StaticPromptObjectEncoder(
        encoder,
        processor,
        chunk_size=args.chunk_size,
    ).to(device)
    metadata = training_metadata(args, processor, backbone)

    backbone_trainable = sum(
        parameter.numel()
        for parameter in backbone.parameters()
        if parameter.requires_grad
    )
    if backbone_trainable != 0:
        raise RuntimeError("Qwen backbone contains trainable parameters")

    completed_step = 0
    if args.resume is not None:
        completed_step = resume_from_checkpoint(
            args.resume,
            encoder=encoder,
            optimizer=optimizer,
            sampler=sampler,
            configuration=configuration,
            device=device,
        )
        print(f"resumed_from: {args.resume.resolve()}")
        print(f"resumed_completed_step: {completed_step}")
    if completed_step >= args.steps:
        raise ValueError(
            f"checkpoint already completed step {completed_step}; "
            f"--steps must be greater than that"
        )

    run_record = {
        "event": "run_start",
        "resume": str(args.resume.resolve()) if args.resume is not None else None,
        "completed_step_before_run": completed_step,
        "configuration": configuration,
        "metadata": metadata,
        "backbone_trainable_parameters": backbone_trainable,
    }
    append_jsonl(log_path, run_record)

    for step in range(completed_step + 1, args.steps + 1):
        batch_rows, labels = balanced_batch(
            grouped,
            args.classes_per_batch,
            args.objects_per_class,
            sampler,
        )
        expected_objects = args.classes_per_batch * args.objects_per_class
        if len(batch_rows) != expected_objects:
            raise RuntimeError("Balanced P-K sampler returned the wrong batch size")
        if len({row["object_id"] for row in batch_rows}) != expected_objects:
            raise RuntimeError("Balanced P-K sampler returned duplicate objects")

        metrics = train_one_step(
            encoder=encoder,
            object_encoder=object_encoder,
            processor=processor,
            optimizer=optimizer,
            batch_rows=batch_rows,
            labels=labels,
            device=device,
            model_dtype=model_dtype,
            chunk_size=args.chunk_size,
            temperature=args.temperature,
        )
        record = {
            "event": "train_step",
            "step": step,
            **metrics,
            "batch_class_count": args.classes_per_batch,
            "batch_object_count": expected_objects,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "classes": [row["label_name"] for row in batch_rows],
            "object_ids": [row["object_id"] for row in batch_rows],
            "backbone_trainable_parameters": backbone_trainable,
        }
        append_jsonl(log_path, record)
        print(json.dumps(record, ensure_ascii=False, sort_keys=True), flush=True)

        if step % args.checkpoint_every == 0 or step == args.steps:
            payload = checkpoint_payload(
                encoder=encoder,
                optimizer=optimizer,
                completed_step=step,
                sampler=sampler,
                configuration=configuration,
                metadata=metadata,
                latest_record=record,
            )
            numbered_path = args.output_dir / f"checkpoint_step_{step:06d}.pt"
            latest_path = args.output_dir / "latest.pt"
            save_checkpoint_atomic(payload, numbered_path)
            save_checkpoint_atomic(payload, latest_path)
            print(f"checkpoint_saved: {numbered_path.resolve()}", flush=True)

    print(
        f"PASS: completed static-prompt training through step {args.steps}; "
        f"latest_checkpoint={args.output_dir.resolve() / 'latest.pt'}"
    )


if __name__ == "__main__":
    main()
