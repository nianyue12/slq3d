#!/usr/bin/env python3
"""Complete-coverage Epoch training for Frozen Mean conditioned Dynamic Prompt."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from conditional_prompt.balanced_pk_epoch_sampler import BalancedPKEpochSampler
from conditional_prompt.dynamic_prompt import FrozenMeanConditionedPrompt
from conditional_prompt.dynamic_training import train_dynamic_one_step
from conditional_prompt.dynamic_conditions import (
    CONDITION_MODES,
    build_training_condition_state,
    condition_state_identity,
    resolve_training_conditions,
    validate_condition_state,
)
from conditional_prompt.static_object_encoder import StaticPromptObjectEncoder
from conditional_prompt.static_prompt_encoder import Qwen3VLStaticPromptEncoder
from conditional_prompt.common import (
    DEFAULT_MODEL_PATH,
    DEFAULT_QUERY_MANIFEST,
    DEFAULT_TARGET_MANIFEST,
    initialize_prompt_from_seed,
    validate_training_source,
)
from conditional_prompt.train_static_prompt import (
    append_jsonl,
    git_version,
    save_checkpoint_atomic,
)
from dataset.multiview_inputs import read_rows
from utils.io import PROMPT


INTERNAL_DIR = ROOT / "dataset/internal_dev"
PROTOCOL_STEM = "os_mn40_core_category_dev_seed20261103"
DEFAULT_MANIFEST = INTERNAL_DIR / f"{PROTOCOL_STEM}_train.csv"
DEFAULT_PROTOCOL = INTERNAL_DIR / f"{PROTOCOL_STEM}.json"
DEFAULT_DEV_QUERY = INTERNAL_DIR / f"{PROTOCOL_STEM}_dev_query.csv"
DEFAULT_DEV_GALLERY = INTERNAL_DIR / f"{PROTOCOL_STEM}_dev_gallery.csv"
DEFAULT_VIEW_CACHE = Path(
    "/home/jwjin/projects/realign_3d/outputs/"
    "os_mn40_core/embeddings/view_objects"
)
FULL8_MANIFEST = ROOT / "dataset/manifests/os_mn40_core_train.csv"
FULL8_PROTOCOL = ROOT / "dataset/protocols/os_mn40_core_full_train.json"
ESB_PROMPT_DIR = ROOT / "dataset/esb_prompt"
ESB_MANIFEST = ESB_PROMPT_DIR / "os_esb_core_pk_train.csv"
ESB_PROTOCOL = ESB_PROMPT_DIR / "os_esb_core_pk_train.json"
ESB_QUERY_MANIFEST = ROOT / "dataset/manifests/os_esb_core_query.csv"
ESB_TARGET_MANIFEST = ROOT / "dataset/manifests/os_esb_core_target.csv"
ESB_VIEW_CACHE = Path(
    "/mnt/data/jwjin/realign_3d/outputs/"
    "os_esb_core/embeddings/view_objects"
)
ESB_INTERNAL_DEV_DIR = ESB_PROMPT_DIR / "internal_dev"
ESB_INTERNAL_MANIFEST = ESB_INTERNAL_DEV_DIR / "train.csv"
ESB_INTERNAL_PROTOCOL = ESB_INTERNAL_DEV_DIR / "protocol.json"
ESB_INTERNAL_DEV_QUERY = ESB_INTERNAL_DEV_DIR / "dev_query.csv"
ESB_INTERNAL_DEV_GALLERY = ESB_INTERNAL_DEV_DIR / "dev_gallery.csv"
CHECKPOINT_FORMAT_VERSION = 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--training-mode",
        choices=(
            "internal_dev",
            "full8_exploratory",
            "esb_exploratory",
            "esb_internal_dev",
        ),
        default="internal_dev",
    )
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--protocol", type=Path)
    parser.add_argument("--dev-query-manifest", type=Path)
    parser.add_argument("--dev-gallery-manifest", type=Path)
    parser.add_argument("--query-manifest", type=Path)
    parser.add_argument("--target-manifest", type=Path)
    parser.add_argument("--view-cache-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--chunk-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--temperature", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=20261103)
    parser.add_argument(
        "--condition-mode",
        choices=CONDITION_MODES,
        default="correct",
        help="object-condition policy; correct preserves the existing behavior",
    )
    parser.add_argument(
        "--condition-seed",
        type=int,
        default=20261103,
        help="fixed seed used only by shuffled_fixed",
    )
    parser.add_argument(
        "--shared-prompt-init-seed",
        type=int,
        help="optional independent seed for an exactly shared Static/Dynamic start",
    )
    parser.add_argument("--max-batches", type=int)
    parser.add_argument("--checkpoint-every", type=int, default=50)
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="validate protocol, caches, and sampler without loading Qwen or using CUDA",
    )
    return parser.parse_args()


def resolve_training_paths(args) -> None:
    """Select mode defaults while still allowing explicit identical paths."""

    mode_paths = {
        "internal_dev": (
            DEFAULT_MANIFEST,
            DEFAULT_PROTOCOL,
            DEFAULT_QUERY_MANIFEST,
            DEFAULT_TARGET_MANIFEST,
            DEFAULT_VIEW_CACHE,
        ),
        "full8_exploratory": (
            FULL8_MANIFEST,
            FULL8_PROTOCOL,
            DEFAULT_QUERY_MANIFEST,
            DEFAULT_TARGET_MANIFEST,
            DEFAULT_VIEW_CACHE,
        ),
        "esb_exploratory": (
            ESB_MANIFEST,
            ESB_PROTOCOL,
            ESB_QUERY_MANIFEST,
            ESB_TARGET_MANIFEST,
            ESB_VIEW_CACHE,
        ),
        "esb_internal_dev": (
            ESB_INTERNAL_MANIFEST,
            ESB_INTERNAL_PROTOCOL,
            ESB_QUERY_MANIFEST,
            ESB_TARGET_MANIFEST,
            ESB_VIEW_CACHE,
        ),
    }
    (
        expected_manifest,
        expected_protocol,
        expected_query,
        expected_target,
        expected_cache,
    ) = mode_paths[args.training_mode]
    if args.manifest is None:
        args.manifest = expected_manifest
    if args.protocol is None:
        args.protocol = expected_protocol
    if args.query_manifest is None:
        args.query_manifest = expected_query
    if args.target_manifest is None:
        args.target_manifest = expected_target
    if args.view_cache_dir is None:
        args.view_cache_dir = expected_cache
    expected_dev_paths = {
        "internal_dev": (DEFAULT_DEV_QUERY, DEFAULT_DEV_GALLERY),
        "esb_internal_dev": (
            ESB_INTERNAL_DEV_QUERY,
            ESB_INTERNAL_DEV_GALLERY,
        ),
    }
    if args.training_mode in expected_dev_paths:
        expected_dev_query, expected_dev_gallery = expected_dev_paths[args.training_mode]
        if args.dev_query_manifest is None:
            args.dev_query_manifest = expected_dev_query
        if args.dev_gallery_manifest is None:
            args.dev_gallery_manifest = expected_dev_gallery
        if args.dev_query_manifest.resolve() != expected_dev_query.resolve():
            raise ValueError(
                f"training-mode={args.training_mode} requires dev-query-manifest "
                f"{expected_dev_query.resolve()}"
            )
        if args.dev_gallery_manifest.resolve() != expected_dev_gallery.resolve():
            raise ValueError(
                f"training-mode={args.training_mode} requires dev-gallery-manifest "
                f"{expected_dev_gallery.resolve()}"
            )
    if args.manifest.resolve() != expected_manifest.resolve():
        raise ValueError(
            f"training-mode={args.training_mode} requires manifest "
            f"{expected_manifest.resolve()}"
        )
    if args.protocol.resolve() != expected_protocol.resolve():
        raise ValueError(
            f"training-mode={args.training_mode} requires protocol "
            f"{expected_protocol.resolve()}"
        )
    for name, actual, expected in (
        ("query-manifest", args.query_manifest, expected_query),
        ("target-manifest", args.target_manifest, expected_target),
        ("view-cache-dir", args.view_cache_dir, expected_cache),
    ):
        if actual.resolve() != expected.resolve():
            raise ValueError(
                f"training-mode={args.training_mode} requires {name} "
                f"{expected.resolve()}"
            )


def validate_args(args) -> None:
    if args.epochs < 1 or args.checkpoint_every < 1:
        raise ValueError("epochs and checkpoint-every must be positive")
    if args.max_batches is not None and args.max_batches < 1:
        raise ValueError("max-batches must be positive")
    if args.chunk_size != 4:
        raise ValueError("Dynamic Epoch training fixes chunk-size at 4")
    if args.learning_rate <= 0 or args.temperature <= 0:
        raise ValueError("learning-rate and temperature must be positive")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def manifest_identities(path: Path) -> tuple[set[str], set[str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return {row["object_id"] for row in rows}, {row["label_name"] for row in rows}


def validate_internal_source(args, rows, protocol) -> None:
    validate_training_source(rows, protocol, args.query_manifest, args.target_manifest)
    train_ids = {row["object_id"] for row in rows}
    train_categories = {row["label_name"] for row in rows}
    dev_ids: set[str] = set()
    dev_categories: set[str] = set()
    for path in (args.dev_query_manifest, args.dev_gallery_manifest):
        ids, categories = manifest_identities(path)
        dev_ids.update(ids)
        dev_categories.update(categories)
    if train_ids & dev_ids or train_categories & dev_categories:
        raise ValueError("internal Train overlaps internal Dev")


def validate_condition_cache_identities(args, rows) -> None:
    """Check every condition cache against its exact Manifest identity."""

    from dataset.frozen_mean import load_object_cache

    for row in rows:
        cache_path = (
            args.view_cache_dir / row["split"] / f"{row['object_id']}.pt"
        )
        embeddings = load_object_cache(cache_path, row, row["split"])
        if tuple(embeddings.shape) != (24, 4096):
            raise ValueError(
                f"{cache_path}: expected condition source [24,4096], "
                f"found {tuple(embeddings.shape)}"
            )


def validate_training_protocol(args, rows, protocol) -> dict[str, Any]:
    """Validate mode policy, all caches, and one full deterministic plan."""

    validate_training_source(rows, protocol, args.query_manifest, args.target_manifest)
    if args.training_mode == "internal_dev":
        validate_internal_source(args, rows, protocol)
        expected_objects = 2160
        expected_categories = 6
        expected_batches = 541
        expected_repeats = 4
    elif args.training_mode == "esb_internal_dev":
        validate_internal_source(args, rows, protocol)
        expected_objects = 72
        expected_categories = 12
        expected_batches = 20
        expected_repeats = 8
    elif args.training_mode == "full8_exploratory":
        expected_objects = 2821
        expected_categories = 8
        expected_batches = 706
        expected_repeats = 3
    else:
        expected_objects = 96
        expected_categories = 15
        expected_batches = 26
        expected_repeats = 8

    object_ids = [row["object_id"] for row in rows]
    categories = {row["label_name"] for row in rows}
    if len(rows) != expected_objects or len(set(object_ids)) != expected_objects:
        raise ValueError(
            f"{args.training_mode}: expected {expected_objects} unique objects, "
            f"found {len(rows)}/{len(set(object_ids))}"
        )
    if len(categories) != expected_categories:
        raise ValueError(
            f"{args.training_mode}: expected {expected_categories} categories, "
            f"found {len(categories)}"
        )
    validate_condition_cache_identities(args, rows)

    grouped = defaultdict(list)
    for row in rows:
        grouped[row["label_name"]].append(row)
    audit_sampler = BalancedPKEpochSampler(dict(grouped), seed=args.seed)
    audit_sampler.start_epoch(0)
    while not audit_sampler.finished:
        audit_sampler.next_batch()
    summary = audit_sampler.summary()
    if summary["planned_batch_count"] != expected_batches:
        raise ValueError(
            f"{args.training_mode}: expected {expected_batches} batches, "
            f"found {summary['planned_batch_count']}"
        )
    if summary["unique_object_coverage"] != expected_objects:
        raise ValueError("Sampler did not cover every selected training object")
    if summary["repeat_sampling_count"] != expected_repeats:
        raise ValueError(
            f"{args.training_mode} sampler expected exactly "
            f"{expected_repeats} minimum repeats, found "
            f"{summary['repeat_sampling_count']}"
        )
    return {
        "object_count": expected_objects,
        "category_count": expected_categories,
        "category_names": sorted(categories),
        "planned_batch_count": summary["planned_batch_count"],
        "unique_object_coverage": summary["unique_object_coverage"],
        "repeat_sampling_count": summary["repeat_sampling_count"],
        "condition_cache_count": len(rows),
    }


def configuration(args) -> dict[str, Any]:
    config = {
        "stage": "5E_dynamic_prompt_epoch_training",
        "training_mode": args.training_mode,
        "model_path": str(args.model_path.resolve()),
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": sha256_file(args.manifest),
        "protocol": str(args.protocol.resolve()),
        "protocol_sha256": sha256_file(args.protocol),
        "query_manifest": str(args.query_manifest.resolve()),
        "target_manifest": str(args.target_manifest.resolve()),
        "view_cache_dir": str(args.view_cache_dir.resolve()),
        "condition_source": "frozen_mean_24_fp32_mean_l2",
        "condition_mode": args.condition_mode,
        "condition_seed": args.condition_seed,
        "output_dir": str(args.output_dir.resolve()),
        "device": args.device,
        "epochs": args.epochs,
        "max_batches": args.max_batches,
        "checkpoint_every": args.checkpoint_every,
        "preflight_only": args.preflight_only,
        "classes_per_batch": 2,
        "objects_per_class": 2,
        "views_per_object": 24,
        "chunk_size": args.chunk_size,
        "learning_rate": args.learning_rate,
        "temperature": args.temperature,
        "seed": args.seed,
        "shared_prompt_init_seed": args.shared_prompt_init_seed,
        "prompt_tokens": 4,
        "prompt_dimension": 4096,
        "prompt_init_std": 0.02,
        "meta_net_dimensions": [4096, 256, 4096],
        "meta_output_init_std": 1e-4,
        "optimizer": "AdamW",
        "loss": "object_level_supervised_contrastive",
    }
    if args.training_mode in {"internal_dev", "esb_internal_dev"}:
        config.update(
            {
                "dev_query_manifest": str(args.dev_query_manifest.resolve()),
                "dev_query_sha256": sha256_file(args.dev_query_manifest),
                "dev_gallery_manifest": str(args.dev_gallery_manifest.resolve()),
                "dev_gallery_sha256": sha256_file(args.dev_gallery_manifest),
            }
        )
    return config


def compatible_configuration(config: dict[str, Any]) -> dict[str, Any]:
    config = dict(config)
    config.setdefault("condition_mode", "correct")
    config.setdefault("condition_seed", 20261103)
    ignored = {
        "epochs", "max_batches", "checkpoint_every", "output_dir", "preflight_only"
    }
    if config.get("training_mode") not in {"internal_dev", "esb_internal_dev"}:
        ignored.update(
            {
                "dev_query_manifest",
                "dev_query_sha256",
                "dev_gallery_manifest",
                "dev_gallery_sha256",
            }
        )
    if config["condition_mode"] != "shuffled_fixed":
        ignored.add("condition_seed")
    return {key: value for key, value in config.items() if key not in ignored}


def metadata(args, processor, backbone) -> dict[str, Any]:
    template = processor.chat_template or ""
    return {
        "stage": "5E_dynamic_prompt_epoch_training",
        "model_type": getattr(backbone.config, "model_type", None),
        "model_name_or_path": getattr(backbone.config, "_name_or_path", None),
        "fixed_instruction": PROMPT,
        "chat_template_sha256": hashlib.sha256(template.encode()).hexdigest(),
        "prompt_insertion": "after_chat_prefix_before_complete_visual_block",
        "view_pooling": "last_hidden_state_all_valid_tokens_attention_mask_mean_l2",
        "object_fusion": "24_view_embeddings_fp32_mean_then_l2",
        "embedding_dimension": 4096,
        "condition_requires_grad": False,
        "qwen_frozen": True,
        "git": git_version(),
    }


def capture_global_rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_global_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"].cpu())
    if state["torch_cuda_all"] is not None:
        torch.cuda.set_rng_state_all(
            [item.cpu() for item in state["torch_cuda_all"]]
        )


def make_checkpoint(
    *, generator, optimizer, sampler, current_epoch, global_step,
    epoch_loss_sum, epoch_loss_count, config, meta, latest_record,
    condition_state,
) -> dict[str, Any]:
    return {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "stage": "dynamic_prompt_epoch_training",
        "shared_prompt": generator.shared_prompt.detach().cpu().clone(),
        "meta_net_state": {
            key: value.detach().cpu().clone()
            for key, value in generator.meta_net.state_dict().items()
        },
        "condition_state": condition_state,
        "optimizer_state": optimizer.state_dict(),
        "global_rng_state": capture_global_rng_state(),
        "sampler_state": sampler.state_dict(),
        "progress": {
            "current_epoch": current_epoch,
            "epoch_step": sampler.batch_count,
            "global_step": global_step,
            "epoch_loss_sum": epoch_loss_sum,
            "epoch_loss_count": epoch_loss_count,
            "last_successful_optimizer_step": global_step,
        },
        "configuration": config,
        "metadata": meta,
        "latest_record": latest_record,
    }


def load_checkpoint(
    path, *, generator, optimizer, sampler, config, device, condition_state
):
    payload = torch.load(path, map_location=device, weights_only=False)
    if payload.get("stage") != "dynamic_prompt_epoch_training":
        raise ValueError("checkpoint is not a Dynamic Epoch checkpoint")
    if payload.get("format_version") != CHECKPOINT_FORMAT_VERSION:
        raise ValueError("unsupported Dynamic checkpoint version")
    if compatible_configuration(payload["configuration"]) != compatible_configuration(config):
        raise ValueError("resume configuration/identity differs from checkpoint")
    saved_condition_state = payload.get("condition_state", {"mode": "correct"})
    validate_condition_state(saved_condition_state, config["condition_mode"])
    if condition_state_identity(saved_condition_state) != condition_state_identity(
        condition_state
    ):
        raise ValueError("resume condition state differs from reconstructed state")
    shared = payload["shared_prompt"]
    if tuple(shared.shape) != (4, 4096) or not torch.isfinite(shared).all():
        raise ValueError("checkpoint Shared Prompt is invalid")
    generator.shared_prompt.data.copy_(shared.to(device))
    meta_state = payload["meta_net_state"]
    if any(
        not isinstance(value, torch.Tensor) or not torch.isfinite(value).all()
        for value in meta_state.values()
    ):
        raise ValueError("checkpoint Meta-Net state is invalid")
    generator.meta_net.load_state_dict(meta_state, strict=True)
    optimizer.load_state_dict(payload["optimizer_state"])
    sampler.load_state_dict(payload["sampler_state"])
    restore_global_rng_state(payload["global_rng_state"])
    progress = payload["progress"]
    if int(progress["last_successful_optimizer_step"]) != int(progress["global_step"]):
        raise ValueError("checkpoint contains an incomplete optimizer step")
    if sampler.epoch != int(progress["current_epoch"]):
        raise ValueError("checkpoint epoch and sampler epoch differ")
    if sampler.batch_count != int(progress["epoch_step"]):
        raise ValueError("checkpoint sampler cursor and epoch_step differ")
    if int(progress["epoch_loss_count"]) != sampler.batch_count:
        raise ValueError("checkpoint loss count and sampler cursor differ")
    return (
        int(progress["current_epoch"]),
        int(progress["global_step"]),
        float(progress["epoch_loss_sum"]),
        int(progress["epoch_loss_count"]),
    )


def main() -> None:
    args = parse_args()
    resolve_training_paths(args)
    validate_args(args)
    device = torch.device(args.device)
    if args.resume is None and (args.output_dir / "latest.pt").exists():
        raise FileExistsError("Dynamic latest.pt exists; use --resume or a new directory")
    if (
        args.training_mode == "full8_exploratory"
        and "full8" not in str(args.output_dir.resolve()).lower()
    ):
        raise ValueError(
            "full8_exploratory requires a dedicated output path containing 'full8'"
        )
    if (
        args.training_mode == "esb_exploratory"
        and "esb" not in str(args.output_dir.resolve()).lower()
    ):
        raise ValueError(
            "esb_exploratory requires a dedicated output path containing 'esb'"
        )
    if (
        args.training_mode == "esb_internal_dev"
        and "esb" not in str(args.output_dir.resolve()).lower()
    ):
        raise ValueError(
            "esb_internal_dev requires a dedicated output path containing 'esb'"
        )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    rows, protocol = read_rows(args.manifest, args.protocol)
    preflight = validate_training_protocol(args, rows, protocol)
    condition_state = build_training_condition_state(
        args.condition_mode, rows, args.view_cache_dir, args.condition_seed
    )
    validate_condition_state(condition_state, args.condition_mode)
    preflight["condition_state_identity"] = condition_state_identity(condition_state)
    config = configuration(args)
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["label_name"]].append(row)
    sampler = BalancedPKEpochSampler(dict(grouped), seed=args.seed)

    if args.preflight_only:
        print("Dynamic Prompt training-mode CPU preflight")
        print(f"training_mode: {args.training_mode}")
        print(f"condition_mode: {args.condition_mode}")
        print("qwen_loaded: False")
        print("gpu_used: False")
        print("configuration: " + json.dumps(config, ensure_ascii=False, sort_keys=True))
        print("validation: " + json.dumps(preflight, ensure_ascii=False, sort_keys=True))
        print("PASS: protocol, condition caches, and Epoch sampler are valid.")
        return

    if args.condition_mode != "correct" and args.condition_mode not in str(
        args.output_dir.resolve()
    ).lower():
        raise ValueError(
            "ablation output directory must contain its condition-mode name"
        )

    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.output_dir / "train_log.jsonl"
    print("Dynamic Prompt complete-coverage Epoch training")
    print(json.dumps(config, ensure_ascii=False, sort_keys=True))
    print("preflight: " + json.dumps(preflight, ensure_ascii=False, sort_keys=True))
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
    backbone.eval()
    view_encoder = Qwen3VLStaticPromptEncoder(backbone).to(device)
    view_encoder.soft_prompt.requires_grad_(False)
    generator = FrozenMeanConditionedPrompt().to(device)
    if args.shared_prompt_init_seed is not None:
        initialize_prompt_from_seed(
            generator.shared_prompt,
            seed=args.shared_prompt_init_seed,
            std=0.02,
        )
    object_encoder = StaticPromptObjectEncoder(
        view_encoder, processor, chunk_size=args.chunk_size
    ).to(device)
    optimizer = torch.optim.AdamW(generator.parameters(), lr=args.learning_rate)
    meta = metadata(args, processor, backbone)
    backbone_trainable = sum(
        parameter.numel() for parameter in backbone.parameters() if parameter.requires_grad
    )
    if backbone_trainable:
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
            generator=generator,
            optimizer=optimizer,
            sampler=sampler,
            config=config,
            device=device,
            condition_state=condition_state,
        )
        print(f"resumed_from: {args.resume.resolve()}")
        print(f"resumed_epoch: {current_epoch + 1}")
        print(f"resumed_epoch_step: {sampler.batch_count}")
        print(f"resumed_global_step: {global_step}")
        if sampler.finished:
            current_epoch += 1
            if current_epoch < args.epochs:
                sampler.start_epoch(current_epoch)
                epoch_loss_sum = 0.0
                epoch_loss_count = 0
        elif current_epoch >= args.epochs:
            raise ValueError("resume checkpoint lies beyond --epochs target")

    run_record = {
        "event": "run_start",
        "resume": str(args.resume.resolve()) if args.resume else None,
        "epoch": current_epoch + 1,
        "epoch_step": sampler.batch_count,
        "global_step": global_step,
        "configuration": config,
    }
    append_jsonl(log_path, run_record)
    invocation_batches = 0
    last_record = run_record
    rows_by_id = {row["object_id"]: row for row in rows}

    while current_epoch < args.epochs:
        if sampler.finished:
            summary = sampler.summary()
            epoch_record = {
                "event": "epoch_end",
                "epoch": current_epoch + 1,
                "global_step": global_step,
                "actual_batch_count": summary["batch_count"],
                "unique_object_coverage": summary["unique_object_coverage"],
                "coverage_rate": summary["coverage_rate"],
                "repeat_sampling_count": summary["repeat_sampling_count"],
                "category_coverage": summary["category_coverage"],
                "average_supcon_loss": epoch_loss_sum / max(epoch_loss_count, 1),
            }
            append_jsonl(log_path, epoch_record)
            print(json.dumps(epoch_record, ensure_ascii=False, sort_keys=True), flush=True)
            last_record = epoch_record
            payload = make_checkpoint(
                generator=generator, optimizer=optimizer, sampler=sampler,
                current_epoch=current_epoch, global_step=global_step,
                epoch_loss_sum=epoch_loss_sum, epoch_loss_count=epoch_loss_count,
                config=config, meta=meta, latest_record=last_record,
                condition_state=condition_state,
            )
            save_checkpoint_atomic(
                payload,
                args.output_dir / f"checkpoint_epoch_{current_epoch + 1:04d}.pt",
            )
            save_checkpoint_atomic(payload, args.output_dir / "latest.pt")
            current_epoch += 1
            if current_epoch >= args.epochs:
                break
            if args.max_batches is not None and invocation_batches >= args.max_batches:
                print("PARTIAL PASS: batch limit reached at epoch boundary")
                return
            sampler.start_epoch(current_epoch)
            epoch_loss_sum = 0.0
            epoch_loss_count = 0
            continue

        batch = sampler.next_batch()
        labels = torch.tensor(batch.labels, dtype=torch.long)
        batch_conditions = resolve_training_conditions(
            batch.rows, rows_by_id, args.view_cache_dir, condition_state
        )
        if args.condition_mode == "constant_train_mean":
            batch_condition_ids = ["TRAIN_MEAN"] * len(batch.rows)
        elif args.condition_mode == "shuffled_fixed":
            batch_condition_ids = [
                condition_state["train_mapping"][row["object_id"]]
                for row in batch.rows
            ]
        else:
            batch_condition_ids = [row["object_id"] for row in batch.rows]
        try:
            metrics = train_dynamic_one_step(
                generator=generator,
                view_encoder=view_encoder,
                object_encoder=object_encoder,
                processor=processor,
                optimizer=optimizer,
                batch_rows=batch.rows,
                labels=labels,
                view_cache_dir=args.view_cache_dir,
                device=device,
                model_dtype=backbone.dtype,
                chunk_size=args.chunk_size,
                temperature=args.temperature,
                conditions=batch_conditions,
            )
        except Exception as error:
            failure = {
                "event": "train_batch_failure",
                "epoch": current_epoch + 1,
                "attempted_epoch_step": sampler.batch_count,
                "global_step_before_batch": global_step,
                "object_ids": [row["object_id"] for row in batch.rows],
                "error_type": type(error).__name__,
                "error": str(error),
                "latest_checkpoint_successful_epoch_step": sampler.batch_count - 1,
            }
            append_jsonl(log_path, failure)
            print(json.dumps(failure, ensure_ascii=False, sort_keys=True), flush=True)
            raise

        global_step += 1
        invocation_batches += 1
        epoch_loss_sum += metrics["supcon_loss"]
        epoch_loss_count += 1
        summary = sampler.summary()
        record = {
            "event": "train_batch",
            "epoch": current_epoch + 1,
            "epoch_step": sampler.batch_count,
            "global_step": global_step,
            "classes": batch.class_names,
            "object_ids": [row["object_id"] for row in batch.rows],
            "condition_mode": args.condition_mode,
            "condition_object_ids": batch_condition_ids,
            "batch_repeated_object_ids": batch.repeated_object_ids,
            "planned_epoch_batches": summary["planned_batch_count"],
            "epoch_unique_object_coverage": summary["unique_object_coverage"],
            "epoch_coverage_rate": summary["coverage_rate"],
            "epoch_repeat_sampling_count": summary["repeat_sampling_count"],
            "learning_rate": optimizer.param_groups[0]["lr"],
            "backbone_trainable_parameters": backbone_trainable,
            **metrics,
        }
        append_jsonl(log_path, record)
        print(json.dumps(record, ensure_ascii=False, sort_keys=True), flush=True)
        last_record = record
        payload = make_checkpoint(
            generator=generator, optimizer=optimizer, sampler=sampler,
            current_epoch=current_epoch, global_step=global_step,
            epoch_loss_sum=epoch_loss_sum, epoch_loss_count=epoch_loss_count,
            config=config, meta=meta, latest_record=last_record,
            condition_state=condition_state,
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

    print(
        f"PASS: completed target total epochs={args.epochs}, "
        f"global_step={global_step}, latest_checkpoint="
        f"{args.output_dir.resolve() / 'latest.pt'}"
    )


if __name__ == "__main__":
    main()
