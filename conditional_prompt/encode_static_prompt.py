#!/usr/bin/env python3
"""Resumably encode Query or Target with a trained static soft prompt."""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path
from typing import Any

import torch
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from conditional_prompt.static_object_encoder import StaticPromptObjectEncoder
from conditional_prompt.static_prompt_encoder import Qwen3VLStaticPromptEncoder
from conditional_prompt.checkpoints import (
    STATIC_OBJECT_FUSION,
    STATIC_PROMPT_SHAPE,
    STATIC_VIEW_POOLING,
    validate_static_checkpoint,
)
from conditional_prompt.common import DEFAULT_MODEL_PATH
from dataset.frozen_mean import read_manifest
from utils.io import PROMPT, save_atomic


DATASET_MANIFESTS = {
    "os_mn40_core": {
        "query": REPOSITORY_ROOT / "dataset/manifests/os_mn40_core_query.csv",
        "target": REPOSITORY_ROOT / "dataset/manifests/os_mn40_core_target.csv",
    },
    "os_esb_core": {
        "query": REPOSITORY_ROOT / "dataset/manifests/os_esb_core_query.csv",
        "target": REPOSITORY_ROOT / "dataset/manifests/os_esb_core_target.csv",
    },
}
DATASET_DISPLAY_NAMES = {
    "os_mn40_core": "OS-MN40-core",
    "os_esb_core": "OS-ESB-core",
}
VIEWS_PER_OBJECT = 24
EMBEDDING_DIMENSION = 4096
DEFAULT_CHECKPOINT_PATH = (
    REPOSITORY_ROOT / "outputs/conditional_prompt/stage4b1_step20/latest.pt"
)
EXPECTED_PROMPT_SHAPE = STATIC_PROMPT_SHAPE
EXPECTED_VIEW_POOLING = STATIC_VIEW_POOLING
EXPECTED_OBJECT_FUSION = STATIC_OBJECT_FUSION


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Encode one official split with the Step-20 static prompt."
    )
    parser.add_argument("--split", choices=("query", "target"), required=True)
    parser.add_argument(
        "--dataset",
        choices=tuple(DATASET_MANIFESTS),
        default="os_mn40_core",
    )
    parser.add_argument("--manifest", type=Path)
    parser.add_argument(
        "--manifest-row-split",
        choices=("train", "query", "target"),
        help="source split stored in a custom Manifest; output role still uses --split",
    )
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT_PATH)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--chunk-size", type=int, default=4)
    parser.add_argument(
        "--max-objects",
        type=int,
        help="target number of manifest-prefix objects present after this run",
    )
    parser.add_argument(
        "--allow-full-split",
        action="store_true",
        help="required when max-objects is omitted",
    )
    return parser.parse_args()


def default_manifest(dataset: str, split: str) -> Path:
    """Return the repository's official manifest for one evaluation split."""

    return DATASET_MANIFESTS[dataset][split]


def file_sha256(path: Path) -> str:
    """Return a stable identifier for the exact training checkpoint file."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_manifest_rows(path: Path, split: str) -> list[dict[str, str]]:
    """Read an official manifest and reject duplicate object IDs."""

    rows = read_manifest(path, split)
    object_ids = [row["object_id"] for row in rows]
    duplicates = sorted(
        object_id for object_id in set(object_ids) if object_ids.count(object_id) > 1
    )
    if duplicates:
        raise ValueError(f"manifest contains duplicate object IDs: {duplicates[:5]}")
    return rows


def cache_metadata(
    *,
    args: argparse.Namespace,
    manifest: Path,
    checkpoint_step: int,
    checkpoint_stage: str,
    checkpoint_sha256: str,
    chat_template_sha256: str,
) -> dict[str, Any]:
    """Describe all settings that must match across resumed encoding runs."""

    return {
        "format_version": 1,
        "method": (
            "Epoch Static Soft Prompt Mean-24"
            if checkpoint_stage == "static_prompt_epoch_training"
            else "Step-20 Static Soft Prompt Mean-24"
        ),
        "checkpoint_stage": checkpoint_stage,
        "checkpoint_path": str(args.checkpoint.resolve()),
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_step": checkpoint_step,
        "model_path": str(args.model_path.resolve()),
        "manifest_path": str(manifest.resolve()),
        "manifest_row_split": args.manifest_row_split or args.split,
        "dataset": DATASET_DISPLAY_NAMES[args.dataset],
        "split": args.split,
        "views": VIEWS_PER_OBJECT,
        "prompt_tokens": EXPECTED_PROMPT_SHAPE[0],
        "fixed_instruction": PROMPT,
        "chat_template_sha256": chat_template_sha256,
        "view_pooling": EXPECTED_VIEW_POOLING,
        "object_fusion": EXPECTED_OBJECT_FUSION,
        "embedding_dimension": EMBEDDING_DIMENSION,
    }


def validate_cached_object(
    path: Path,
    row: dict[str, str],
    expected_metadata: dict[str, Any],
) -> torch.Tensor:
    """Validate one atomic object cache before treating it as complete."""

    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("metadata") != expected_metadata:
        raise ValueError(
            f"{path}: metadata differs; refusing to mix checkpoints or settings"
        )
    if payload.get("object_id") != row["object_id"]:
        raise ValueError(f"{path}: cached object_id differs from manifest")
    if payload.get("label_name") != row["label_name"]:
        raise ValueError(f"{path}: cached label differs from manifest")
    if payload.get("split") != row["split"]:
        raise ValueError(f"{path}: cached split differs from manifest")
    if payload.get("view_count") != VIEWS_PER_OBJECT:
        raise ValueError(f"{path}: cached object was not encoded from 24 views")
    embedding = payload.get("object_embedding")
    if not isinstance(embedding, torch.Tensor) or embedding.shape != (4096,):
        raise ValueError(f"{path}: expected one [4096] object embedding")
    embedding = embedding.float()
    if not torch.isfinite(embedding).all():
        raise ValueError(f"{path}: embedding contains NaN/Inf")
    if abs(float(embedding.norm().item()) - 1.0) > 1e-3:
        raise ValueError(f"{path}: embedding is not L2-normalized")
    return embedding


def discover_completed(
    rows: list[dict[str, str]],
    object_dir: Path,
    metadata: dict[str, Any],
) -> dict[str, torch.Tensor]:
    """Load all valid caches and reject unknown or duplicate object records."""

    manifest_by_id = {row["object_id"]: row for row in rows}
    completed: dict[str, torch.Tensor] = {}
    if not object_dir.exists():
        return completed

    for path in sorted(object_dir.glob("*.pt")):
        object_id = path.stem
        if object_id not in manifest_by_id:
            raise ValueError(f"unknown object cache not present in manifest: {path}")
        if object_id in completed:
            raise ValueError(f"duplicate object cache: {object_id}")
        completed[object_id] = validate_cached_object(
            path,
            manifest_by_id[object_id],
            metadata,
        )
    return completed


def write_evaluate_payload(
    *,
    path: Path,
    rows: list[dict[str, str]],
    completed: dict[str, torch.Tensor],
    metadata: dict[str, Any],
) -> tuple[torch.Tensor, list[dict[str, str]]]:
    """Write an evaluate.py-compatible aggregate in strict manifest order."""

    completed_rows = [row for row in rows if row["object_id"] in completed]
    if not completed_rows:
        raise RuntimeError("no completed embeddings are available to aggregate")
    embeddings = torch.stack(
        [completed[row["object_id"]] for row in completed_rows]
    ).to(torch.float16)
    save_atomic(
        {
            "split": metadata["split"],
            "object_embeddings": embeddings,
            "object_ids": [row["object_id"] for row in completed_rows],
            "label_names": [row["label_name"] for row in completed_rows],
            "method": metadata["method"],
            "views": metadata["views"],
            "checkpoint": metadata["checkpoint_path"],
            "checkpoint_sha256": metadata["checkpoint_sha256"],
            "checkpoint_step": metadata["checkpoint_step"],
            "metadata": metadata,
            "complete": len(completed_rows) == len(rows),
            "expected_object_count": len(rows),
        },
        path,
    )
    return embeddings, completed_rows


def main() -> None:
    args = parse_args()
    manifest = args.manifest or default_manifest(args.dataset, args.split)
    if args.chunk_size < 1:
        raise ValueError("chunk-size must be positive")
    if args.max_objects is not None and args.max_objects < 1:
        raise ValueError("max-objects must be positive")
    if args.max_objects is None and not args.allow_full_split:
        raise SystemExit(
            "Refusing full-split encoding without --allow-full-split; "
            "use --max-objects for the smoke test"
        )
    if not manifest.is_file():
        raise FileNotFoundError(f"manifest does not exist: {manifest}")
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint does not exist: {args.checkpoint}")
    if not args.model_path.is_dir():
        raise FileNotFoundError(f"model directory does not exist: {args.model_path}")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")

    rows = read_manifest_rows(manifest, args.manifest_row_split or args.split)
    requested_count = (
        len(rows) if args.max_objects is None else min(args.max_objects, len(rows))
    )
    requested_rows = rows[:requested_count]

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    soft_prompt, checkpoint_step = validate_static_checkpoint(
        checkpoint,
        args.checkpoint,
        args.model_path,
    )
    checkpoint_digest = file_sha256(args.checkpoint)

    processor = AutoProcessor.from_pretrained(args.model_path, local_files_only=True)
    processor.tokenizer.padding_side = "left"
    chat_template = processor.chat_template or ""
    chat_template_digest = hashlib.sha256(chat_template.encode("utf-8")).hexdigest()
    if checkpoint["metadata"].get("chat_template_sha256") != chat_template_digest:
        raise ValueError("current processor chat template differs from the checkpoint")

    metadata = cache_metadata(
        args=args,
        manifest=manifest,
        checkpoint_step=checkpoint_step,
        checkpoint_stage=str(checkpoint["stage"]),
        checkpoint_sha256=checkpoint_digest,
        chat_template_sha256=chat_template_digest,
    )
    object_dir = args.output_dir / args.split / "objects"
    aggregate_path = args.output_dir / f"{args.split}.pt"
    object_dir.mkdir(parents=True, exist_ok=True)
    completed = discover_completed(rows, object_dir, metadata)

    selected_ids = {row["object_id"] for row in requested_rows}
    skipped_count = len(selected_ids & set(completed))
    processed_count = 0

    model_dtype = torch.float16 if device.type == "cuda" else torch.float32
    backbone = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_path,
        dtype=model_dtype,
        low_cpu_mem_usage=True,
        local_files_only=True,
    ).to(device)
    encoder = Qwen3VLStaticPromptEncoder(
        backbone,
        num_prompt_tokens=EXPECTED_PROMPT_SHAPE[0],
        prompt_init_std=0.02,
    ).to(device)
    encoder.soft_prompt.data.copy_(
        soft_prompt.to(device=device, dtype=encoder.soft_prompt.dtype)
    )
    encoder.soft_prompt.requires_grad_(False)
    encoder.eval()
    if any(parameter.requires_grad for parameter in encoder.parameters()):
        raise RuntimeError("inference encoder contains trainable parameters")
    object_encoder = StaticPromptObjectEncoder(
        encoder,
        processor,
        chunk_size=args.chunk_size,
    ).to(device)

    with torch.inference_mode():
        for index, row in enumerate(requested_rows, start=1):
            object_id = row["object_id"]
            if object_id in completed:
                print(
                    f"[{index}/{requested_count}] skip completed {object_id}",
                    flush=True,
                )
                continue
            output = object_encoder(
                row,
                device=device,
                model_dtype=model_dtype,
            )
            embedding = output.object_embedding.detach().float().cpu().clone()
            object_path = object_dir / f"{object_id}.pt"
            save_atomic(
                {
                    "object_id": object_id,
                    "label_name": row["label_name"],
                    "split": row["split"],
                    "object_embedding": embedding.to(torch.float16),
                    "view_count": len(output.view_paths),
                    "metadata": metadata,
                },
                object_path,
            )
            completed[object_id] = validate_cached_object(
                object_path,
                row,
                metadata,
            )
            processed_count += 1
            print(f"[{index}/{requested_count}] encoded {object_id}", flush=True)

    unexpected_ids = set(completed) - {row["object_id"] for row in rows}
    if unexpected_ids:
        raise RuntimeError(f"completed cache contains unknown IDs: {sorted(unexpected_ids)[:5]}")
    missing_requested = selected_ids - set(completed)
    if missing_requested:
        raise RuntimeError(
            f"requested objects were not completed: {sorted(missing_requested)[:5]}"
        )
    if args.max_objects is None and set(completed) != {row["object_id"] for row in rows}:
        raise RuntimeError("full-split completed ID set differs from the manifest")

    aggregate, aggregate_rows = write_evaluate_payload(
        path=aggregate_path,
        rows=rows,
        completed=completed,
        metadata=metadata,
    )
    norms = aggregate.float().norm(dim=-1)
    duplicate_ids = len(aggregate_rows) - len(
        {row["object_id"] for row in aggregate_rows}
    )

    print("\nStatic-prompt encoding summary")
    print(f"checkpoint_step: {checkpoint_step}")
    print(f"split: {args.split}")
    print(f"expected_object_count: {len(rows)}")
    print(f"requested_object_count: {requested_count}")
    print(f"processed_object_count: {processed_count}")
    print(f"skipped_object_count: {skipped_count}")
    print(f"completed_object_count: {len(aggregate_rows)}")
    print(f"view_count_per_object: {VIEWS_PER_OBJECT}")
    print(f"embedding_shape: {list(aggregate.shape)}")
    print(f"embedding_norm_min: {float(norms.min().item()):.8f}")
    print(f"embedding_norm_max: {float(norms.max().item()):.8f}")
    print(f"embedding_finite: {bool(torch.isfinite(aggregate).all().item())}")
    print(f"duplicate_object_ids: {duplicate_ids}")
    print(f"output_path: {aggregate_path.resolve()}")

    if aggregate.shape != (len(aggregate_rows), EMBEDDING_DIMENSION):
        raise RuntimeError("aggregate embedding shape is invalid")
    if not torch.isfinite(aggregate).all():
        raise RuntimeError("aggregate embeddings contain NaN/Inf")
    if duplicate_ids != 0:
        raise RuntimeError("aggregate output contains duplicate object IDs")
    if not torch.allclose(norms, torch.ones_like(norms), atol=1e-3):
        raise RuntimeError("one or more aggregate embeddings are not normalized")

    print("PASS: resumable static-prompt split encoding is valid.")


if __name__ == "__main__":
    main()
