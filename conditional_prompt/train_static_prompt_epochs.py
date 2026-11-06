#!/usr/bin/env python3
"""Epoch training for the shared static prompt with complete-coverage P-K batches."""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from conditional_prompt.balanced_pk_epoch_sampler import BalancedPKEpochSampler
from conditional_prompt.static_object_encoder import StaticPromptObjectEncoder
from conditional_prompt.static_prompt_encoder import Qwen3VLStaticPromptEncoder
from conditional_prompt.common import (
    DEFAULT_MODEL_PATH,
    DEFAULT_PROTOCOL,
    DEFAULT_QUERY_MANIFEST,
    DEFAULT_TARGET_MANIFEST,
    DEFAULT_TRAIN_MANIFEST,
    initialize_prompt_from_seed,
    validate_training_source,
)
from conditional_prompt.train_static_prompt import (
    PROMPT_INIT_STD,
    PROMPT_TOKENS,
    append_jsonl,
    save_checkpoint_atomic,
    train_one_step,
    training_metadata,
)
from dataset.multiview_inputs import read_rows


CHECKPOINT_FORMAT_VERSION = 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the static prompt by complete-coverage P=2, K=2 epochs."
    )
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_TRAIN_MANIFEST)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--query-manifest", type=Path, default=DEFAULT_QUERY_MANIFEST)
    parser.add_argument("--target-manifest", type=Path, default=DEFAULT_TARGET_MANIFEST)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--chunk-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--temperature", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=20261103)
    parser.add_argument(
        "--shared-prompt-init-seed",
        type=int,
        help="optional independent seed for an exactly shared Static/Dynamic start",
    )
    parser.add_argument(
        "--max-batches",
        type=int,
        help="maximum additional batches in this invocation; omit for full epochs",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=50,
        help="numbered global-step checkpoint interval; latest.pt is saved every batch",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    for name in ("epochs", "chunk_size", "checkpoint_every"):
        if getattr(args, name) < 1:
            raise ValueError(f"{name.replace('_', '-')} must be positive")
    if args.max_batches is not None and args.max_batches < 1:
        raise ValueError("max-batches must be positive")
    if args.learning_rate <= 0 or args.temperature <= 0:
        raise ValueError("learning-rate and temperature must be positive")


def configuration(args: argparse.Namespace) -> dict[str, Any]:
    """Return all settings needed to reproduce or validate a resumed run."""

    return {
        "mode": "complete_coverage_balanced_pk_epoch",
        "model_path": str(args.model_path.resolve()),
        "manifest": str(args.manifest.resolve()),
        "protocol": str(args.protocol.resolve()),
        "query_manifest": str(args.query_manifest.resolve()),
        "target_manifest": str(args.target_manifest.resolve()),
        "output_dir": str(args.output_dir.resolve()),
        "device": args.device,
        "epochs": args.epochs,
        "classes_per_batch": 2,
        "objects_per_class": 2,
        "views_per_object": 24,
        "chunk_size": args.chunk_size,
        "learning_rate": args.learning_rate,
        "temperature": args.temperature,
        "seed": args.seed,
        "shared_prompt_init_seed": args.shared_prompt_init_seed,
        "max_batches": args.max_batches,
        "checkpoint_every": args.checkpoint_every,
        "prompt_tokens": PROMPT_TOKENS,
        "prompt_init_std": PROMPT_INIT_STD,
    }


def compatible_configuration(config: dict[str, Any]) -> dict[str, Any]:
    """Exclude invocation-only settings while requiring training semantics to match."""

    ignored = {"epochs", "max_batches", "checkpoint_every", "output_dir"}
    return {key: value for key, value in config.items() if key not in ignored}


def capture_global_rng_state() -> dict[str, Any]:
    """Capture non-sampler RNG state after a completed optimizer update."""

    return {
        "python": random.getstate(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_global_rng_state(state: dict[str, Any]) -> None:
    """Restore RNG ByteTensors on CPU even when checkpoint tensors use CUDA."""

    random.setstate(state["python"])
    torch.set_rng_state(state["torch_cpu"].cpu())
    if state["torch_cuda_all"] is not None:
        torch.cuda.set_rng_state_all(
            [cuda_state.cpu() for cuda_state in state["torch_cuda_all"]]
        )


def make_checkpoint(
    *,
    encoder: Qwen3VLStaticPromptEncoder,
    optimizer: torch.optim.Optimizer,
    sampler: BalancedPKEpochSampler,
    current_epoch: int,
    global_step: int,
    epoch_loss_sum: float,
    epoch_loss_count: int,
    config: dict[str, Any],
    metadata: dict[str, Any],
    latest_record: dict[str, Any],
) -> dict[str, Any]:
    """Build a checkpoint whose sampler cursor points to the next batch."""

    return {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "stage": "static_prompt_epoch_training",
        "soft_prompt": encoder.soft_prompt.detach().cpu().clone(),
        "optimizer_state": optimizer.state_dict(),
        "global_rng_state": capture_global_rng_state(),
        "sampler_state": sampler.state_dict(),
        "progress": {
            "current_epoch": current_epoch,
            "completed_epoch_count": current_epoch if not sampler.finished else current_epoch + 1,
            "epoch_step": sampler.batch_count,
            "global_step": global_step,
            "epoch_loss_sum": epoch_loss_sum,
            "epoch_loss_count": epoch_loss_count,
        },
        "configuration": config,
        "metadata": metadata,
        "latest_record": latest_record,
    }


def load_checkpoint(
    path: Path,
    *,
    encoder: Qwen3VLStaticPromptEncoder,
    optimizer: torch.optim.Optimizer,
    sampler: BalancedPKEpochSampler,
    config: dict[str, Any],
    device: torch.device,
) -> tuple[int, int, float, int]:
    """Restore parameters, optimizer, exact sampler cursor, progress, and RNGs."""

    payload = torch.load(path, map_location=device, weights_only=False)
    if payload.get("stage") != "static_prompt_epoch_training":
        raise ValueError("resume checkpoint is not an epoch-training checkpoint")
    if payload.get("format_version") != CHECKPOINT_FORMAT_VERSION:
        raise ValueError("unsupported epoch-training checkpoint version")
    if compatible_configuration(payload["configuration"]) != compatible_configuration(config):
        raise ValueError("resume configuration differs from checkpoint")

    prompt = payload["soft_prompt"]
    if tuple(prompt.shape) != tuple(encoder.soft_prompt.shape):
        raise ValueError("checkpoint Soft Prompt shape differs")
    if not torch.isfinite(prompt).all():
        raise ValueError("checkpoint Soft Prompt contains NaN/Inf")
    encoder.soft_prompt.data.copy_(
        prompt.to(device=device, dtype=encoder.soft_prompt.dtype)
    )
    optimizer.load_state_dict(payload["optimizer_state"])
    sampler.load_state_dict(payload["sampler_state"])
    restore_global_rng_state(payload["global_rng_state"])

    progress = payload["progress"]
    current_epoch = int(progress["current_epoch"])
    global_step = int(progress["global_step"])
    epoch_loss_sum = float(progress["epoch_loss_sum"])
    epoch_loss_count = int(progress["epoch_loss_count"])
    if sampler.epoch != current_epoch:
        raise ValueError("checkpoint sampler epoch differs from training progress")
    if sampler.batch_count != int(progress["epoch_step"]):
        raise ValueError("checkpoint sampler cursor differs from epoch_step")
    if epoch_loss_count != sampler.batch_count:
        raise ValueError("checkpoint epoch loss count differs from sampler progress")
    return current_epoch, global_step, epoch_loss_sum, epoch_loss_count


def main() -> None:
    args = parse_args()
    validate_args(args)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    if args.resume is None and (args.output_dir / "latest.pt").exists():
        raise FileExistsError("output latest.pt exists; use --resume or a new directory")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    config = configuration(args)
    rows, protocol = read_rows(args.manifest, args.protocol)
    validate_training_source(
        rows,
        protocol,
        args.query_manifest,
        args.target_manifest,
    )
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["label_name"]].append(row)
    sampler = BalancedPKEpochSampler(dict(grouped), seed=args.seed)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.output_dir / "train_log.jsonl"
    print("Static Soft Prompt complete-coverage Epoch training")
    print(json.dumps(config, ensure_ascii=False, sort_keys=True))
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
    if args.shared_prompt_init_seed is not None:
        initialize_prompt_from_seed(
            encoder.soft_prompt,
            seed=args.shared_prompt_init_seed,
            std=PROMPT_INIT_STD,
        )
    optimizer = torch.optim.AdamW([encoder.soft_prompt], lr=args.learning_rate)
    object_encoder = StaticPromptObjectEncoder(
        encoder,
        processor,
        chunk_size=args.chunk_size,
    ).to(device)
    metadata = training_metadata(args, processor, backbone)
    metadata["stage"] = "static_prompt_epoch_training"
    metadata["training_mode"] = "complete_coverage_balanced_pk_epoch"
    backbone_trainable = sum(
        parameter.numel() for parameter in backbone.parameters() if parameter.requires_grad
    )
    if backbone_trainable != 0:
        raise RuntimeError("Qwen backbone contains trainable parameters")

    current_epoch = 0
    global_step = 0
    epoch_loss_sum = 0.0
    epoch_loss_count = 0
    if args.resume is None:
        sampler.start_epoch(current_epoch)
    else:
        current_epoch, global_step, epoch_loss_sum, epoch_loss_count = load_checkpoint(
            args.resume,
            encoder=encoder,
            optimizer=optimizer,
            sampler=sampler,
            config=config,
            device=device,
        )
        print(f"resumed_from: {args.resume.resolve()}")
        print(f"resumed_epoch: {current_epoch + 1}")
        print(f"resumed_epoch_step: {sampler.batch_count}")
        print(f"resumed_global_step: {global_step}")

        # Epoch-end checkpoints contain a finished sampler. Advance here so a
        # resumed run neither repeats its summary nor restarts a completed batch.
        if sampler.finished:
            current_epoch += 1
            if current_epoch < args.epochs:
                sampler.start_epoch(current_epoch)
                epoch_loss_sum = 0.0
                epoch_loss_count = 0
        elif current_epoch >= args.epochs:
            raise ValueError(
                "resume checkpoint is inside an epoch outside the requested "
                "--epochs target"
            )

    run_record = {
        "event": "run_start",
        "configuration": config,
        "metadata": metadata,
        "resume": str(args.resume.resolve()) if args.resume else None,
        "current_epoch": current_epoch + 1,
        "epoch_step": sampler.batch_count,
        "global_step": global_step,
    }
    append_jsonl(log_path, run_record)
    invocation_batches = 0
    last_record: dict[str, Any] = run_record

    while current_epoch < args.epochs:
        if sampler.finished:
            summary = sampler.summary()
            average_loss = epoch_loss_sum / max(epoch_loss_count, 1)
            epoch_record = {
                "event": "epoch_end",
                "epoch": current_epoch + 1,
                "global_step": global_step,
                "actual_batch_count": summary["batch_count"],
                "unique_object_coverage": summary["unique_object_coverage"],
                "coverage_rate": summary["coverage_rate"],
                "repeat_sampling_count": summary["repeat_sampling_count"],
                "category_coverage": summary["category_coverage"],
                "average_supcon_loss": average_loss,
            }
            append_jsonl(log_path, epoch_record)
            print(json.dumps(epoch_record, ensure_ascii=False, sort_keys=True), flush=True)
            last_record = epoch_record
            epoch_path = args.output_dir / f"checkpoint_epoch_{current_epoch + 1:04d}.pt"
            save_checkpoint_atomic(
                make_checkpoint(
                    encoder=encoder,
                    optimizer=optimizer,
                    sampler=sampler,
                    current_epoch=current_epoch,
                    global_step=global_step,
                    epoch_loss_sum=epoch_loss_sum,
                    epoch_loss_count=epoch_loss_count,
                    config=config,
                    metadata=metadata,
                    latest_record=last_record,
                ),
                epoch_path,
            )
            current_epoch += 1
            if current_epoch >= args.epochs:
                break
            if args.max_batches is not None and invocation_batches >= args.max_batches:
                print(
                    "PARTIAL PASS: batch limit reached at an epoch boundary; "
                    f"completed_epochs={current_epoch}, global_step={global_step}, "
                    "the next epoch will be created after resume"
                )
                return
            sampler.start_epoch(current_epoch)
            epoch_loss_sum = 0.0
            epoch_loss_count = 0
            continue

        batch = sampler.next_batch()
        labels = torch.tensor(batch.labels, dtype=torch.long)
        try:
            metrics = train_one_step(
                encoder=encoder,
                object_encoder=object_encoder,
                processor=processor,
                optimizer=optimizer,
                batch_rows=batch.rows,
                labels=labels,
                device=device,
                model_dtype=model_dtype,
                chunk_size=args.chunk_size,
                temperature=args.temperature,
            )
        except Exception as error:
            failure_record = {
                "event": "train_batch_failure",
                "epoch": current_epoch + 1,
                "epoch_step": sampler.batch_count,
                "global_step_before_batch": global_step,
                "classes": batch.class_names,
                "object_ids": [row["object_id"] for row in batch.rows],
                "error_type": type(error).__name__,
                "error": str(error),
                "checkpoint_cursor_preserved_at_epoch_step": sampler.batch_count - 1,
            }
            append_jsonl(log_path, failure_record)
            print(json.dumps(failure_record, ensure_ascii=False, sort_keys=True), flush=True)
            raise
        global_step += 1
        invocation_batches += 1
        epoch_loss_sum += metrics["supcon_loss"]
        epoch_loss_count += 1
        sampler_progress = sampler.summary()
        record = {
            "event": "train_batch",
            "epoch": current_epoch + 1,
            "epoch_step": sampler.batch_count,
            "planned_epoch_batches": sampler_progress["planned_batch_count"],
            "global_step": global_step,
            **metrics,
            "classes": batch.class_names,
            "object_ids": [row["object_id"] for row in batch.rows],
            "batch_repeated_object_ids": batch.repeated_object_ids,
            "epoch_unique_object_coverage": sampler_progress["unique_object_coverage"],
            "epoch_coverage_rate": sampler_progress["coverage_rate"],
            "epoch_repeat_sampling_count": sampler_progress["repeat_sampling_count"],
            "learning_rate": optimizer.param_groups[0]["lr"],
            "backbone_trainable_parameters": backbone_trainable,
        }
        append_jsonl(log_path, record)
        print(json.dumps(record, ensure_ascii=False, sort_keys=True), flush=True)
        last_record = record

        payload = make_checkpoint(
            encoder=encoder,
            optimizer=optimizer,
            sampler=sampler,
            current_epoch=current_epoch,
            global_step=global_step,
            epoch_loss_sum=epoch_loss_sum,
            epoch_loss_count=epoch_loss_count,
            config=config,
            metadata=metadata,
            latest_record=last_record,
        )
        save_checkpoint_atomic(payload, args.output_dir / "latest.pt")
        if global_step % args.checkpoint_every == 0:
            save_checkpoint_atomic(
                payload,
                args.output_dir / f"checkpoint_step_{global_step:06d}.pt",
            )

        if (
            args.max_batches is not None
            and invocation_batches >= args.max_batches
            and not sampler.finished
        ):
            print(
                "PARTIAL PASS: batch limit reached; "
                f"epoch={current_epoch + 1}, epoch_step={sampler.batch_count}, "
                f"global_step={global_step}, next batch is preserved in latest.pt"
            )
            return

    final_payload = make_checkpoint(
        encoder=encoder,
        optimizer=optimizer,
        sampler=sampler,
        current_epoch=current_epoch - 1,
        global_step=global_step,
        epoch_loss_sum=epoch_loss_sum,
        epoch_loss_count=epoch_loss_count,
        config=config,
        metadata=metadata,
        latest_record=last_record,
    )
    save_checkpoint_atomic(final_payload, args.output_dir / "latest.pt")
    print(
        f"PASS: completed {args.epochs} epoch(s), global_step={global_step}, "
        f"latest_checkpoint={args.output_dir.resolve() / 'latest.pt'}"
    )


if __name__ == "__main__":
    main()
