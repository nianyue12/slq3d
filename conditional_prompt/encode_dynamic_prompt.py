#!/usr/bin/env python3
"""Resumably encode Query or Target with a trained Dynamic Prompt."""

from __future__ import annotations

import argparse
import hashlib
import math
import sys
from pathlib import Path
from typing import Any

import torch
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from conditional_prompt.dynamic_object_encoder import (
    DynamicPromptObjectEncoder,
    load_frozen_mean24_condition,
)
from conditional_prompt.dynamic_conditions import (
    build_inference_mapping,
    mapping_sha256,
)
from conditional_prompt.encode_static_prompt import (
    DATASET_DISPLAY_NAMES,
    DATASET_MANIFESTS,
    discover_completed,
    file_sha256,
    read_manifest_rows,
    validate_cached_object,
    write_evaluate_payload,
)
from conditional_prompt.static_object_encoder import StaticPromptObjectEncoder
from conditional_prompt.static_prompt_encoder import Qwen3VLStaticPromptEncoder
from conditional_prompt.checkpoints import (
    load_dynamic_prompt_generator,
    validate_dynamic_checkpoint,
)
from utils.io import PROMPT, save_atomic


VIEWS_PER_OBJECT = 24
EMBEDDING_DIMENSION = 4096
CHUNK_SIZE = 4
DEFAULT_CHECKPOINT = ROOT / "outputs/conditional_prompt/dynamic_full8_epoch_1/latest.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
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
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--delta-scale",
        type=float,
        default=1.0,
        help="inference-only multiplier for Meta-Net Delta; 1 preserves behavior",
    )
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
    return DATASET_MANIFESTS[dataset][split]


def validate_completed_condition_identities(
    object_dir: Path,
    completed: dict[str, torch.Tensor],
    condition_references: dict[str, str],
) -> None:
    """Reject resumed records that name a condition cache for another object."""

    for object_id in completed:
        expected = condition_references[object_id]
        payload = torch.load(
            object_dir / f"{object_id}.pt",
            map_location="cpu",
            weights_only=False,
        )
        recorded = payload.get("condition_cache")
        if recorded != expected:
            raise ValueError(
                f"{object_id}: resumed condition cache identity differs from Manifest"
            )


def cache_metadata(
    *,
    split: str,
    manifest: Path,
    checkpoint: Path,
    checkpoint_sha256: str,
    checkpoint_step: int,
    config: dict[str, Any],
    checkpoint_metadata: dict[str, Any],
    chat_template_sha256: str,
    dataset: str,
    manifest_row_split: str,
    delta_scale: float,
) -> dict[str, Any]:
    """Bind every resumable object record to one exact inference setup."""

    metadata = {
        "format_version": 1,
        "method": "Dynamic Prompt Frozen-Mean-24 Condition Epoch Checkpoint",
        "checkpoint_stage": "dynamic_prompt_epoch_training",
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_step": checkpoint_step,
        "training_mode": config["training_mode"],
        "model_path": str(Path(config["model_path"]).resolve()),
        "manifest_path": str(manifest.resolve()),
        "manifest_row_split": manifest_row_split,
        "dataset": DATASET_DISPLAY_NAMES[dataset],
        "split": split,
        "views": VIEWS_PER_OBJECT,
        "prompt_tokens": 4,
        "prompt_dimension": EMBEDDING_DIMENSION,
        "meta_net_dimensions": [4096, 256, 4096],
        "condition_source": config["condition_source"],
        "condition_mode": config.get("condition_mode", "correct"),
        "condition_cache_dir": str(Path(config["view_cache_dir"]).resolve()),
        "fixed_instruction": PROMPT,
        "chat_template_sha256": chat_template_sha256,
        "prompt_insertion": checkpoint_metadata["prompt_insertion"],
        "view_pooling": checkpoint_metadata["view_pooling"],
        "object_fusion": checkpoint_metadata["object_fusion"],
        "embedding_dimension": EMBEDDING_DIMENSION,
        "chunk_size": CHUNK_SIZE,
    }
    # Scale 1.0 is the historical behavior. Keep it implicit so existing
    # resumable outputs remain compatible after this option was introduced.
    if delta_scale != 1.0:
        metadata["delta_scale"] = delta_scale
    return metadata


def main() -> None:
    args = parse_args()
    manifest = (args.manifest or default_manifest(args.dataset, args.split)).resolve()
    checkpoint_path = args.checkpoint.resolve()
    output_dir = args.output_dir.resolve()
    if args.max_objects is not None and args.max_objects < 1:
        raise ValueError("max-objects must be positive")
    if not math.isfinite(args.delta_scale) or args.delta_scale < 0:
        raise ValueError("delta-scale must be finite and non-negative")
    if args.max_objects is None and not args.allow_full_split:
        raise SystemExit(
            "Refusing full-split encoding without --allow-full-split; "
            "use --max-objects for the smoke test"
        )
    if not manifest.is_file():
        raise FileNotFoundError(f"manifest does not exist: {manifest}")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"checkpoint does not exist: {checkpoint_path}")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    manifest_row_split = args.manifest_row_split or args.split
    rows = read_manifest_rows(manifest, manifest_row_split)
    requested_count = (
        len(rows) if args.max_objects is None else min(args.max_objects, len(rows))
    )
    requested_rows = rows[:requested_count]
    payload, config = validate_dynamic_checkpoint(checkpoint_path)
    condition_mode = config.get("condition_mode", "correct")
    condition_state = payload.get("condition_state", {"mode": "correct"})
    expected_mode = {
        "os_mn40_core": {"internal_dev", "full8_exploratory"},
        "os_esb_core": {"esb_exploratory", "esb_internal_dev"},
    }
    if config["training_mode"] not in expected_mode[args.dataset]:
        raise ValueError(
            f"dataset={args.dataset} is incompatible with Dynamic "
            f"training_mode={config['training_mode']}"
        )
    checkpoint_step = int(payload["progress"]["global_step"])
    checkpoint_digest = file_sha256(checkpoint_path)
    model_path = Path(config["model_path"])
    view_cache_dir = Path(config["view_cache_dir"])
    if not model_path.is_dir():
        raise FileNotFoundError(f"model directory does not exist: {model_path}")
    if not view_cache_dir.is_dir():
        raise FileNotFoundError(
            f"Frozen Mean-24 cache directory does not exist: {view_cache_dir}"
        )

    # Shuffled inference uses one deterministic, label-free derangement over
    # the complete Query+Gallery universe. Constant inference reuses the exact
    # Internal-Train mean stored in the checkpoint.
    if "dev_query_manifest" in config and manifest in {
        Path(config["dev_query_manifest"]).resolve(),
        Path(config["dev_gallery_manifest"]).resolve(),
    }:
        universe_rows = read_manifest_rows(Path(config["dev_query_manifest"]), "train")
        universe_rows += read_manifest_rows(Path(config["dev_gallery_manifest"]), "train")
    else:
        universe_rows = read_manifest_rows(Path(config["query_manifest"]), "query")
        universe_rows += read_manifest_rows(Path(config["target_manifest"]), "target")
    universe_by_id = {row["object_id"]: row for row in universe_rows}
    if len(universe_by_id) != len(universe_rows):
        raise ValueError("condition inference universe contains duplicate Object IDs")
    inference_mapping = None
    if condition_mode == "shuffled_fixed":
        inference_mapping = build_inference_mapping(
            universe_rows, int(condition_state["seed"])
        )
    condition_references: dict[str, str] = {}
    for row in rows:
        object_id = row["object_id"]
        if condition_mode == "constant_train_mean":
            condition_references[object_id] = (
                "checkpoint:train_mean:" + str(condition_state["train_mean_sha256"])
            )
        else:
            source_id = object_id if inference_mapping is None else inference_mapping[object_id]
            source_row = universe_by_id[source_id]
            condition_references[object_id] = str(
                (view_cache_dir / source_row["split"] / f"{source_id}.pt").resolve()
            )

    processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
    processor.tokenizer.padding_side = "left"
    chat_template_digest = hashlib.sha256(
        (processor.chat_template or "").encode("utf-8")
    ).hexdigest()
    if chat_template_digest != payload["metadata"]["chat_template_sha256"]:
        raise ValueError("current processor chat template differs from checkpoint")

    metadata = cache_metadata(
        split=args.split,
        manifest=manifest,
        checkpoint=checkpoint_path,
        checkpoint_sha256=checkpoint_digest,
        checkpoint_step=checkpoint_step,
        config=config,
        checkpoint_metadata=payload["metadata"],
        chat_template_sha256=chat_template_digest,
        dataset=args.dataset,
        manifest_row_split=manifest_row_split,
        delta_scale=args.delta_scale,
    )
    metadata["condition_state_identity"] = {
        "mode": condition_mode,
        "train_mean_sha256": condition_state.get("train_mean_sha256"),
        "train_mapping_sha256": condition_state.get("train_mapping_sha256"),
        "inference_mapping_sha256": (
            mapping_sha256(inference_mapping) if inference_mapping is not None else None
        ),
    }
    object_dir = output_dir / args.split / "objects"
    aggregate_path = output_dir / f"{args.split}.pt"
    object_dir.mkdir(parents=True, exist_ok=True)
    completed = discover_completed(rows, object_dir, metadata)
    validate_completed_condition_identities(
        object_dir, completed, condition_references
    )
    selected_ids = {row["object_id"] for row in requested_rows}
    skipped_count = len(selected_ids & set(completed))
    processed_count = 0

    print("Dynamic Prompt resumable split encoding")
    print(f"checkpoint_path: {checkpoint_path}")
    print(f"checkpoint_step: {checkpoint_step}")
    print(f"checkpoint_sha256: {checkpoint_digest}")
    print(f"split: {args.split}")
    print(f"manifest_path: {manifest}")
    print(f"condition_cache_dir: {view_cache_dir}")
    print(f"condition_mode: {condition_mode}")
    print(f"delta_scale: {args.delta_scale}")
    print("loading processor and frozen model ...", flush=True)

    model_dtype = torch.float16 if device.type == "cuda" else torch.float32
    backbone = Qwen3VLForConditionalGeneration.from_pretrained(
        model_path,
        dtype=model_dtype,
        low_cpu_mem_usage=True,
        local_files_only=True,
    ).to(device)
    backbone.eval()
    view_encoder = Qwen3VLStaticPromptEncoder(backbone).to(device)
    view_encoder.soft_prompt.requires_grad_(False)
    object_encoder = StaticPromptObjectEncoder(
        view_encoder, processor, chunk_size=CHUNK_SIZE
    ).to(device)
    prompt_generator = load_dynamic_prompt_generator(payload, device)
    encoder = DynamicPromptObjectEncoder(prompt_generator, object_encoder).eval()
    if any(parameter.requires_grad for parameter in encoder.parameters()):
        raise RuntimeError("Dynamic inference encoder contains trainable parameters")

    with torch.inference_mode():
        for index, row in enumerate(requested_rows, start=1):
            object_id = row["object_id"]
            if object_id in completed:
                print(
                    f"[{index}/{requested_count}] skip completed {object_id}",
                    flush=True,
                )
                continue

            # Identity, split, label and exact ordered view paths are checked by
            # the shared Frozen Mean cache loader before the condition is used.
            if condition_mode == "constant_train_mean":
                condition = condition_state["train_mean"].detach().float().clone()
                condition_reference = condition_references[object_id]
            else:
                source_id = object_id if inference_mapping is None else inference_mapping[object_id]
                source_row = universe_by_id[source_id]
                condition, condition_path = load_frozen_mean24_condition(
                    source_row, view_cache_dir
                )
                condition_reference = str(condition_path.resolve())
            output = encoder(
                row,
                condition.to(device),
                device=device,
                model_dtype=model_dtype,
                delta_scale=args.delta_scale,
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
                    "condition_cache": condition_reference,
                    "metadata": metadata,
                },
                object_path,
            )
            # Reload the atomic file before declaring this object complete.
            completed[object_id] = validate_cached_object(
                object_path, row, metadata
            )
            processed_count += 1
            print(f"[{index}/{requested_count}] encoded {object_id}", flush=True)

    manifest_ids = {row["object_id"] for row in rows}
    unexpected_ids = set(completed) - manifest_ids
    if unexpected_ids:
        raise RuntimeError(
            f"completed cache contains unknown IDs: {sorted(unexpected_ids)[:5]}"
        )
    missing_requested = selected_ids - set(completed)
    if missing_requested:
        raise RuntimeError(
            f"requested objects were not completed: {sorted(missing_requested)[:5]}"
        )
    if args.max_objects is None and set(completed) != manifest_ids:
        raise RuntimeError("full-split completed ID set differs from manifest")

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
    finite = bool(torch.isfinite(aggregate).all().item())

    print("\nDynamic-prompt encoding summary")
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
    print(f"embedding_finite: {finite}")
    print(f"duplicate_object_ids: {duplicate_ids}")
    print(f"output_path: {aggregate_path}")

    if aggregate.shape != (len(aggregate_rows), EMBEDDING_DIMENSION):
        raise RuntimeError("aggregate embedding shape is invalid")
    if not finite:
        raise RuntimeError("aggregate embeddings contain NaN/Inf")
    if duplicate_ids != 0:
        raise RuntimeError("aggregate output contains duplicate object IDs")
    if not torch.allclose(norms, torch.ones_like(norms), atol=1e-3):
        raise RuntimeError("one or more aggregate embeddings are not normalized")

    print("PASS: resumable Dynamic Prompt split encoding is valid.")


if __name__ == "__main__":
    main()
