"""Deterministic condition policies for Dynamic Prompt ablations."""

from __future__ import annotations

import hashlib
import random
from pathlib import Path

import torch

from conditional_prompt.dynamic_object_encoder import load_frozen_mean24_condition


CONDITION_MODES = ("correct", "constant_train_mean", "shuffled_fixed")


def tensor_sha256(value: torch.Tensor) -> str:
    tensor = value.detach().cpu().contiguous().float()
    return hashlib.sha256(tensor.numpy().tobytes()).hexdigest()


def mapping_sha256(mapping: dict[str, str]) -> str:
    text = "\n".join(f"{key}\t{mapping[key]}" for key in sorted(mapping))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def fixed_derangement(object_ids: list[str], seed: int) -> dict[str, str]:
    """Return a deterministic permutation with no object mapped to itself."""

    ordered = sorted(object_ids)
    if len(ordered) < 2 or len(set(ordered)) != len(ordered):
        raise ValueError("shuffled_fixed requires at least two unique Object IDs")
    shuffled = ordered.copy()
    rng = random.Random(seed)
    for _ in range(10000):
        rng.shuffle(shuffled)
        if all(left != right for left, right in zip(ordered, shuffled, strict=True)):
            return dict(zip(ordered, shuffled, strict=True))
    raise RuntimeError("failed to construct deterministic condition derangement")


def build_training_condition_state(
    mode: str,
    rows: list[dict[str, str]],
    view_cache_dir: Path,
    seed: int,
) -> dict[str, object]:
    if mode not in CONDITION_MODES:
        raise ValueError(f"unsupported condition mode: {mode}")
    if mode == "correct":
        return {"mode": mode}
    if mode == "constant_train_mean":
        conditions = [
            load_frozen_mean24_condition(row, view_cache_dir)[0].float()
            for row in rows
        ]
        mean = torch.stack(conditions).mean(dim=0)
        if tuple(mean.shape) != (4096,) or not torch.isfinite(mean).all():
            raise RuntimeError("training-only mean condition is invalid")
        return {
            "mode": mode,
            "train_mean": mean.cpu(),
            "train_mean_sha256": tensor_sha256(mean),
            "source_object_count": len(rows),
        }
    mapping = fixed_derangement([row["object_id"] for row in rows], seed)
    return {
        "mode": mode,
        "seed": seed,
        "train_mapping": mapping,
        "train_mapping_sha256": mapping_sha256(mapping),
        "source_object_count": len(rows),
    }


def validate_condition_state(state: dict[str, object], expected_mode: str) -> None:
    if state.get("mode") != expected_mode or expected_mode not in CONDITION_MODES:
        raise ValueError("checkpoint condition state/mode mismatch")
    if expected_mode == "constant_train_mean":
        mean = state.get("train_mean")
        if not isinstance(mean, torch.Tensor) or tuple(mean.shape) != (4096,):
            raise ValueError("checkpoint training mean condition is invalid")
        if not torch.isfinite(mean).all() or tensor_sha256(mean) != state.get(
            "train_mean_sha256"
        ):
            raise ValueError("checkpoint training mean condition hash differs")
    if expected_mode == "shuffled_fixed":
        mapping = state.get("train_mapping")
        if not isinstance(mapping, dict) or any(key == value for key, value in mapping.items()):
            raise ValueError("checkpoint shuffled condition mapping is invalid")
        if mapping_sha256(mapping) != state.get("train_mapping_sha256"):
            raise ValueError("checkpoint shuffled condition mapping hash differs")


def condition_state_identity(state: dict[str, object]) -> dict[str, object]:
    """Return the non-tensor identity fields used for strict Resume checks."""

    mode = state.get("mode")
    identity = {"mode": mode, "source_object_count": state.get("source_object_count")}
    if mode == "constant_train_mean":
        identity["train_mean_sha256"] = state.get("train_mean_sha256")
    elif mode == "shuffled_fixed":
        identity.update(
            {
                "seed": state.get("seed"),
                "train_mapping_sha256": state.get("train_mapping_sha256"),
            }
        )
    return identity


def resolve_training_conditions(
    batch_rows: list[dict[str, str]],
    all_rows_by_id: dict[str, dict[str, str]],
    view_cache_dir: Path,
    state: dict[str, object],
) -> list[torch.Tensor]:
    mode = str(state["mode"])
    if mode == "constant_train_mean":
        return [state["train_mean"].detach().float().clone() for _ in batch_rows]
    if mode == "correct":
        source_rows = batch_rows
    else:
        mapping = state["train_mapping"]
        source_rows = [all_rows_by_id[mapping[row["object_id"]]] for row in batch_rows]
    return [
        load_frozen_mean24_condition(row, view_cache_dir)[0].float()
        for row in source_rows
    ]


def build_inference_mapping(rows: list[dict[str, str]], seed: int) -> dict[str, str]:
    """Build a label-free fixed mapping for one complete evaluation universe."""

    return fixed_derangement([row["object_id"] for row in rows], seed)
