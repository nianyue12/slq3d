#!/usr/bin/env python3
"""Evaluate each cached camera view under the DEC/DAC retrieval protocol."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from metrics.retrieval import anmrr_score, map_score, ndcg_score, recall_score
from dataset.manifest import VIEW_COLUMNS
from dataset.frozen_mean import load_object_cache, read_manifest


ANGLE_PATTERN = re.compile(r"^h_(-?\d+(?:\.\d+)?)$")


def angle_from_path(path: str) -> float:
    match = ANGLE_PATTERN.match(Path(path).stem)
    if match is None:
        raise ValueError(f"cannot parse camera angle from {path!r}")
    return float(match.group(1))


def validate_view_order(
    query_rows: list[dict[str, str]], target_rows: list[dict[str, str]]
) -> list[float]:
    reference_names = [Path(query_rows[0][column]).name for column in VIEW_COLUMNS]
    for row in [*query_rows, *target_rows]:
        names = [Path(row[column]).name for column in VIEW_COLUMNS]
        if names != reference_names:
            raise ValueError(
                f"{row['object_id']}: view filename order differs from the reference"
            )
    angles = sorted(angle_from_path(name) for name in reference_names)
    if len(set(angles)) != len(angles):
        raise ValueError(f"camera angles are not unique: {angles}")
    if len(angles) > 1:
        circular_gaps = [
            (angles[(index + 1) % len(angles)] - angle) % 360.0
            for index, angle in enumerate(angles)
        ]
        if not all(np.isclose(gap, circular_gaps[0]) for gap in circular_gaps):
            raise ValueError(f"camera angles are not uniformly spaced: {angles}")
    return angles


def numeric_angle_order(row: dict[str, str]) -> list[int]:
    """Indices that reorder manifest/cache views by numeric camera azimuth."""
    indexed = [
        (angle_from_path(row[column]), index)
        for index, column in enumerate(VIEW_COLUMNS)
    ]
    angles = [angle for angle, _ in indexed]
    if len(set(angles)) != len(angles):
        raise ValueError(f"{row['object_id']}: camera angles are not unique")
    return [index for _, index in sorted(indexed)]


def load_view_tensor(
    rows: list[dict[str, str]], cache_root: Path, split: str
) -> torch.Tensor:
    tensors = []
    expected_shape: tuple[int, int] | None = None
    for index, row in enumerate(rows, start=1):
        cache_path = cache_root / split / f"{row['object_id']}.pt"
        tensor = load_object_cache(cache_path, row, split)
        tensor = tensor[numeric_angle_order(row)]
        if tensor.ndim != 2:
            raise ValueError(f"{cache_path}: expected [views, dim], got {tuple(tensor.shape)}")
        shape = (int(tensor.shape[0]), int(tensor.shape[1]))
        if expected_shape is None:
            expected_shape = shape
        elif shape != expected_shape:
            raise ValueError(f"{cache_path}: shape {shape}, expected {expected_shape}")
        if not torch.isfinite(tensor).all():
            raise ValueError(f"{cache_path}: contains NaN or Inf")
        tensors.append(tensor.to(dtype=torch.float16))
        if index % 1000 == 0 or index == len(rows):
            print(f"loaded {split}: {index}/{len(rows)} objects", flush=True)
    return torch.stack(tensors, dim=0)


@torch.inference_mode()
def cosine_distances_by_view(
    query_views: torch.Tensor,
    target_views: torch.Tensor,
    device: torch.device,
    target_batch_size: int,
) -> np.ndarray:
    """Return [views, queries, targets] cosine distances using target chunks."""
    if query_views.ndim != 3 or target_views.ndim != 3:
        raise ValueError("query_views and target_views must have shape [objects, views, dim]")
    if query_views.shape[1:] != target_views.shape[1:]:
        raise ValueError("query and target view shapes differ")

    num_queries, num_views, _ = query_views.shape
    num_targets = target_views.shape[0]
    distances = np.empty((num_views, num_queries, num_targets), dtype=np.float32)
    query = F.normalize(query_views.float(), dim=-1).permute(1, 0, 2).to(device)

    for start in range(0, num_targets, target_batch_size):
        stop = min(start + target_batch_size, num_targets)
        target = F.normalize(target_views[start:stop].float(), dim=-1)
        target = target.permute(1, 0, 2).to(device)
        similarities = torch.bmm(query, target.transpose(1, 2))
        distances[:, :, start:stop] = (1.0 - similarities).cpu().numpy()
        print(f"scored targets: {stop}/{num_targets}", flush=True)
    return distances


def official_metrics_from_distances(
    distances: np.ndarray, query_labels: list[str], target_labels: list[str]
) -> dict[str, float]:
    query_array = np.asarray(query_labels)
    target_array = np.asarray(target_labels)
    return {
        "mAP": float(map_score(distances, query_array, target_array)),
        "NDCG@100": float(ndcg_score(distances, query_array, target_array)),
        "ANMRR": float(anmrr_score(distances, query_array, target_array)),
        "Recall@100": float(recall_score(distances, query_array, target_array)),
    }


def evaluate_views(
    distances: np.ndarray,
    angles: list[float],
    query_labels: list[str],
    target_labels: list[str],
) -> list[dict[str, float | int]]:
    if distances.shape[0] != len(angles):
        raise ValueError("distance view count and angle count differ")
    rows: list[dict[str, float | int]] = []
    for view_index, angle in enumerate(angles):
        metrics = official_metrics_from_distances(
            distances[view_index], query_labels, target_labels
        )
        row: dict[str, float | int] = {
            "view_index": view_index,
            "angle_degrees": angle,
            **metrics,
        }
        rows.append(row)
        print(
            f"view={view_index:02d} angle={angle:g} "
            f"mAP={metrics['mAP']:.5f} NDCG@100={metrics['NDCG@100']:.5f} "
            f"ANMRR={metrics['ANMRR']:.5f}",
            flush=True,
        )
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--query-manifest", type=Path, required=True)
    parser.add_argument("--target-manifest", type=Path, required=True)
    parser.add_argument("--view-cache-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--target-batch-size", type=int, default=512)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.target_batch_size < 1:
        raise SystemExit("--target-batch-size must be positive")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is not available")

    query_rows = read_manifest(args.query_manifest, "query")
    target_rows = read_manifest(args.target_manifest, "target")
    angles = validate_view_order(query_rows, target_rows)
    query_views = load_view_tensor(query_rows, args.view_cache_dir, "query")
    target_views = load_view_tensor(target_rows, args.view_cache_dir, "target")
    print(
        f"query tensor={tuple(query_views.shape)} {query_views.dtype}; "
        f"target tensor={tuple(target_views.shape)} {target_views.dtype}",
        flush=True,
    )

    distances = cosine_distances_by_view(
        query_views, target_views, device, args.target_batch_size
    )
    query_labels = [row["label_name"] for row in query_rows]
    target_labels = [row["label_name"] for row in target_rows]
    rows = evaluate_views(distances, angles, query_labels, target_labels)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "per_view_metrics.csv"
    json_path = args.output_dir / "per_view_metrics.json"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    payload = {
        "dataset": args.dataset,
        "metric_implementation": "DEC/DAC metric_tools.py",
        "query_objects": len(query_rows),
        "target_objects": len(target_rows),
        "embedding_shape": list(query_views.shape[1:]),
        "manifest_view_order_angles_degrees": angles,
        "results": rows,
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {csv_path}")
    print(f"wrote {json_path}")


if __name__ == "__main__":
    main()
