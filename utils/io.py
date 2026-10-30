#!/usr/bin/env python3
"""Extract per-view Qwen3-VL embeddings with resumable object caches."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

from dataset.manifest import VIEW_COLUMNS


PROMPT = "\nSummarize the above image in one word: "


def read_manifest(path: Path, split: str) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"{path}: empty manifest")
    wrong = [row["object_id"] for row in rows if row["split"] != split]
    if wrong:
        raise ValueError(f"{path}: rows with unexpected split: {wrong[:5]}")
    return rows


def mean_pooling(hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Match the official ReAlign mask-aware mean pooling."""
    expanded_mask = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
    summed = torch.sum(hidden_states * expanded_mask, dim=1)
    denominator = torch.clamp(expanded_mask.sum(dim=1), min=1e-9)
    return summed / denominator


def build_inputs(processor, image: Image.Image):
    message = {
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": PROMPT},
        ],
    }
    prompting = processor.apply_chat_template([message])
    return processor(
        text=prompting,
        images=[[image]],
        padding=True,
        return_tensors="pt",
    )


def move_inputs(inputs, device: torch.device, model_dtype: torch.dtype) -> dict[str, torch.Tensor]:
    moved: dict[str, torch.Tensor] = {}
    for key, value in inputs.items():
        if not isinstance(value, torch.Tensor):
            continue
        if key == "pixel_values":
            moved[key] = value.to(device=device, dtype=model_dtype)
        else:
            moved[key] = value.to(device=device)
    return moved


@torch.inference_mode()
def encode_view(model, processor, image_path: Path, device: torch.device) -> torch.Tensor:
    with Image.open(image_path) as image:
        inputs = build_inputs(processor, image.convert("RGB"))
    batch = move_inputs(inputs, device, model.dtype)
    outputs = model(**batch, inference=True, output_hidden_states=True)
    pooled = mean_pooling(outputs.hidden_states[-1], batch["attention_mask"])
    embedding = F.normalize(pooled, dim=-1)[0]
    if not torch.isfinite(embedding).all():
        raise ValueError(f"{image_path}: embedding contains NaN or Inf")
    return embedding.detach().to(device="cpu", dtype=torch.float16)


def save_atomic(payload: dict[str, object], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    torch.save(payload, temporary_path)
    temporary_path.replace(output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "query", "target"), required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-objects", type=int)
    parser.add_argument("--allow-full-split", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.max_objects is None and not args.allow_full_split:
        raise SystemExit(
            "Refusing a full split without --allow-full-split; "
            "use --max-objects 1 for the smoke test"
        )
    if args.max_objects is not None and args.max_objects < 1:
        raise SystemExit("--max-objects must be positive")
    if not args.model_path.is_dir():
        raise SystemExit(f"Model directory does not exist: {args.model_path}")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available")

    device = torch.device(args.device)
    rows = read_manifest(args.manifest, args.split)
    if args.max_objects is not None:
        rows = rows[: args.max_objects]

    print(f"loading processor from {args.model_path} ...", flush=True)
    processor = AutoProcessor.from_pretrained(args.model_path, local_files_only=True)
    print(f"loading model on {device} ...", flush=True)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        local_files_only=True,
    ).to(device)
    model.eval()
    print(f"model loaded; processing {len(rows)} objects", flush=True)

    for object_index, row in enumerate(rows, start=1):
        output_path = args.output_dir / args.split / f"{row['object_id']}.pt"
        if output_path.exists() and not args.overwrite:
            print(
                f"[{object_index}/{len(rows)}] skip existing {row['object_id']}",
                flush=True,
            )
            continue

        view_paths = [Path(row[column]) for column in VIEW_COLUMNS]
        embeddings = []
        for view_index, view_path in enumerate(view_paths, start=1):
            embeddings.append(encode_view(model, processor, view_path, device))
            print(
                f"[{object_index}/{len(rows)}] {row['object_id']} "
                f"view {view_index}/{len(view_paths)}",
                flush=True,
            )
        stacked = torch.stack(embeddings, dim=0)
        save_atomic(
            {
                "object_id": row["object_id"],
                "label_name": row["label_name"],
                "split": args.split,
                "view_paths": [str(path) for path in view_paths],
                "view_embeddings": stacked,
                "model_path": str(args.model_path.resolve()),
                "prompt": PROMPT,
                "pooling": "last_hidden_state_attention_mask_mean_l2",
            },
            output_path,
        )
        print(
            f"[{object_index}/{len(rows)}] saved {output_path}", flush=True
        )
        print(f"saved {output_path} shape={tuple(stacked.shape)} dtype={stacked.dtype}")


if __name__ == "__main__":
    main()
