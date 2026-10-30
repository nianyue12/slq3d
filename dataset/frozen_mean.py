#!/usr/bin/env python3
"""Aggregate cached view embeddings into Frozen Mean object embeddings."""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dataset.manifest import NUM_VIEWS, VIEW_COLUMNS


ANGLE_PATTERN = re.compile(r"^h_(-?\d+(?:\.\d+)?)$")


def angle_from_path(path: str) -> float:
    """Read the camera azimuth from a view filename such as ``h_90.jpg``."""
    match = ANGLE_PATTERN.match(Path(path).stem)
    if match is None:
        raise ValueError(f"cannot parse camera angle from {path!r}")
    return float(match.group(1))


def select_view_embeddings(
    view_embeddings: torch.Tensor,
    row: dict[str, str],
    num_views: int,
) -> tuple[torch.Tensor, list[int], list[float]]:
    """Select evenly spaced views after sorting them by numeric azimuth."""
    if view_embeddings.ndim != 2:
        raise ValueError(f"expected [views, dim], found {tuple(view_embeddings.shape)}")
    if view_embeddings.shape[0] != NUM_VIEWS:
        raise ValueError(f"expected {NUM_VIEWS} views, found {view_embeddings.shape[0]}")
    if num_views not in (4, 24):
        raise ValueError("num_views must be 4 or 24")

    indexed_angles = [
        (angle_from_path(row[column]), index)
        for index, column in enumerate(VIEW_COLUMNS)
    ]
    angles = [angle for angle, _ in indexed_angles]
    if len(set(angles)) != NUM_VIEWS:
        raise ValueError(f"{row['object_id']}: camera angles are not unique")

    numeric_order = [index for _, index in sorted(indexed_angles)]
    step = NUM_VIEWS // num_views
    numeric_positions = list(range(0, NUM_VIEWS, step))[:num_views]
    manifest_indices = [numeric_order[position] for position in numeric_positions]
    selected_angles = [
        sorted(indexed_angles)[position][0]
        for position in numeric_positions
    ]
    selected = view_embeddings[manifest_indices]
    return selected, manifest_indices, selected_angles


def aggregate_view_embeddings(view_embeddings: torch.Tensor) -> torch.Tensor:
    """FP32 mean pooling followed by L2 normalization."""
    if view_embeddings.ndim != 2:
        raise ValueError(f"expected [views, dim], found {tuple(view_embeddings.shape)}")
    features = view_embeddings.to(dtype=torch.float32)
    if not torch.isfinite(features).all():
        raise ValueError("view embeddings contain NaN or Inf")
    pooled = F.normalize(features.mean(dim=0), dim=0)
    if not torch.isfinite(pooled).all():
        raise ValueError("object embedding contains NaN or Inf")
    return pooled


def read_manifest(path: Path, split: str) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"{path}: empty manifest")
    if any(row["split"] != split for row in rows):
        raise ValueError(f"{path}: contains a split other than {split}")
    return rows


def load_object_cache(cache_path: Path, row: dict[str, str], split: str) -> torch.Tensor:
    if not cache_path.is_file():
        raise FileNotFoundError(cache_path)
    payload = torch.load(cache_path, map_location="cpu", weights_only=False)
    for key, expected in (
        ("object_id", row["object_id"]),
        ("label_name", row["label_name"]),
        ("split", split),
    ):
        if payload.get(key) != expected:
            raise ValueError(
                f"{cache_path}: {key}={payload.get(key)!r}, expected {expected!r}"
            )
    expected_paths = [row[column] for column in VIEW_COLUMNS]
    if payload.get("view_paths") != expected_paths:
        raise ValueError(f"{cache_path}: view path order differs from manifest")
    embeddings = payload.get("view_embeddings")
    if not isinstance(embeddings, torch.Tensor):
        raise ValueError(f"{cache_path}: missing tensor view_embeddings")
    return embeddings


def save_atomic(payload: dict[str, object], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    torch.save(payload, temporary_path)
    temporary_path.replace(output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--view-cache-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "query", "target"), required=True)
    parser.add_argument("--views", type=int, choices=(4, 24), required=True)
    parser.add_argument("--max-objects", type=int)
    parser.add_argument("--allow-full-split", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.max_objects is None and not args.allow_full_split:
        raise SystemExit(
            "Refusing full-split aggregation without --allow-full-split; "
            "use --max-objects 1 for the smoke test"
        )
    if args.max_objects is not None and args.max_objects < 1:
        raise SystemExit("--max-objects must be positive")

    rows = read_manifest(args.manifest, args.split)
    if args.max_objects is not None:
        rows = rows[: args.max_objects]

    object_embeddings = []
    object_ids = []
    label_names = []
    selected_manifest_indices: list[int] | None = None
    selected_angles: list[float] | None = None
    for index, row in enumerate(rows, start=1):
        cache_path = args.view_cache_dir / args.split / f"{row['object_id']}.pt"
        view_embeddings = load_object_cache(cache_path, row, args.split)
        selected, manifest_indices, angles = select_view_embeddings(
            view_embeddings, row, args.views
        )
        if selected_manifest_indices is None:
            selected_manifest_indices = manifest_indices
            selected_angles = angles
        elif manifest_indices != selected_manifest_indices or angles != selected_angles:
            raise ValueError(
                f"{row['object_id']}: selected view order differs from prior objects"
            )
        object_embeddings.append(aggregate_view_embeddings(selected))
        object_ids.append(row["object_id"])
        label_names.append(row["label_name"])
        print(f"[{index}/{len(rows)}] aggregated {row['object_id']}")

    stacked = torch.stack(object_embeddings, dim=0)
    save_atomic(
        {
            "split": args.split,
            "object_ids": object_ids,
            "label_names": label_names,
            "object_embeddings": stacked,
            "aggregation": f"fp32_{args.views}_view_mean_l2",
            "views": args.views,
            "selected_manifest_indices": selected_manifest_indices,
            "selected_angles_degrees": selected_angles,
            "manifest_path": str(args.manifest.resolve()),
        },
        args.output,
    )
    print(f"saved {args.output} shape={tuple(stacked.shape)} dtype={stacked.dtype}")


if __name__ == "__main__":
    main()
