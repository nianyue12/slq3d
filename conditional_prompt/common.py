"""Shared paths, manifest validation, and RNG helpers for prompt learning."""

from __future__ import annotations

import csv
from pathlib import Path

import torch

from conditional_prompt.static_object_encoder import object_view_paths


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = Path(
    "/home/jwjin/projects/realign_3d/models/Qwen3-VL-8B-Instruct"
)
DEFAULT_TRAIN_MANIFEST = ROOT / "dataset/manifests/os_mn40_core_train.csv"
DEFAULT_PROTOCOL = ROOT / "dataset/protocols/os_mn40_core_full_train.json"
DEFAULT_QUERY_MANIFEST = ROOT / "dataset/manifests/os_mn40_core_query.csv"
DEFAULT_TARGET_MANIFEST = ROOT / "dataset/manifests/os_mn40_core_target.csv"


def initialize_prompt_from_seed(
    parameter: torch.nn.Parameter,
    *,
    seed: int,
    std: float,
) -> None:
    """Give Static and Dynamic runs an identical CPU-generated prompt start."""

    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    initial = torch.empty(parameter.shape, dtype=torch.float32, device="cpu")
    initial.normal_(mean=0.0, std=float(std), generator=generator)
    with torch.no_grad():
        parameter.copy_(initial.to(device=parameter.device, dtype=parameter.dtype))


def read_manifest_row(path: Path, row_index: int) -> dict[str, str]:
    """Read one zero-based row from a CSV Manifest."""

    if row_index < 0:
        raise ValueError("row-index must be non-negative")
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"{path}: manifest is empty")
    if row_index >= len(rows):
        raise IndexError(
            f"row-index {row_index} is outside a manifest with {len(rows)} rows"
        )
    return rows[row_index]


def read_manifest_identities(path: Path) -> tuple[set[str], set[str]]:
    """Read object IDs and category names without loading any images."""

    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return (
        {row["object_id"] for row in rows},
        {row["label_name"] for row in rows},
    )


def validate_training_source(
    rows: list[dict[str, str]],
    protocol: dict,
    query_manifest: Path,
    target_manifest: Path,
) -> None:
    """Require legal official-train objects and disjoint official test data."""

    if not rows:
        raise ValueError("the official optimization split contains no objects")
    train_ids = [row["object_id"] for row in rows]
    if len(train_ids) != len(set(train_ids)):
        raise ValueError("the training manifest contains duplicate optimization IDs")
    if any(row["split"] != "train" for row in rows):
        raise ValueError("an optimization object is not marked as split=train")
    if set(train_ids) != set(protocol["optimization_object_ids"]):
        raise ValueError("selected training IDs differ from optimization_object_ids")

    train_categories = {row["label_name"] for row in rows}
    if train_categories != set(protocol["train_category_names"]):
        raise ValueError("training categories differ from train_category_names")

    forbidden_ids: set[str] = set()
    forbidden_categories: set[str] = set()
    for manifest_path in (query_manifest, target_manifest):
        object_ids, categories = read_manifest_identities(manifest_path)
        forbidden_ids.update(object_ids)
        forbidden_categories.update(categories)
    overlap = set(train_ids) & forbidden_ids
    if overlap:
        raise ValueError(f"training IDs overlap Query/Target: {sorted(overlap)[:5]}")
    category_overlap = train_categories & forbidden_categories
    if category_overlap:
        raise ValueError(
            "official training and test categories overlap: "
            f"{sorted(category_overlap)}"
        )
    for row in rows:
        _, paths = object_view_paths(row)
        if any("train" not in path.parts for path in paths):
            raise ValueError(f"non-training view path found for {row['object_id']}")


def capture_rng_state(
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Capture CPU and selected CUDA RNG state for exact recomputation."""

    cpu_state = torch.get_rng_state()
    cuda_state = torch.cuda.get_rng_state(device) if device.type == "cuda" else None
    return cpu_state, cuda_state


def restore_rng_state(
    state: tuple[torch.Tensor, torch.Tensor | None],
    device: torch.device,
) -> None:
    """Restore a state captured immediately before cached forward."""

    cpu_state, cuda_state = state
    torch.set_rng_state(cpu_state)
    if cuda_state is not None:
        torch.cuda.set_rng_state(cuda_state, device)
