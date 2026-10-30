#!/usr/bin/env python3
"""Train latent queries for multi-view 3D retrieval with gradient caching."""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import torch
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

from models.qwen3_vl_query_encoder import Qwen3VLLatentQueryEncoder
from utils.io import move_inputs
from dataset.multiview_inputs import build_inputs, read_rows
from loss.supervised_contrastive import supervised_contrastive_loss

def balanced_batch(
    grouped: dict[str, list[dict[str, str]]], classes: int, objects: int,
    rng: random.Random,
) -> tuple[list[dict[str, str]], torch.Tensor]:
    chosen_labels = rng.sample(sorted(grouped), classes)
    rows, labels = [], []
    for label_index, label in enumerate(chosen_labels):
        rows.extend(rng.sample(grouped[label], objects))
        labels.extend([label_index] * objects)
    return rows, torch.tensor(labels, dtype=torch.long)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--views", type=int, choices=(1, 2, 4), default=4)
    parser.add_argument("--num-queries", type=int, default=16)
    parser.add_argument("--classes-per-batch", type=int, default=6)
    parser.add_argument("--objects-per-class", type=int, default=2)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--seed", type=int, default=20261020)
    parser.add_argument(
        "--initial-queries", type=Path,
        help="optional checkpoint supplying latent_queries instead of random initialization",
    )
    parser.add_argument(
        "--checkpoint-steps", type=int, nargs="*", default=[],
        help="training steps at which to save query checkpoints",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    rows, protocol = read_rows(args.manifest, args.split)

    #import pdb; pdb.set_trace()

    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["label_name"]].append(row)
    grouped = {
        label: objects
        for label, objects in grouped.items()
        if len(objects) >= args.objects_per_class
    }
    if args.classes_per_batch > len(grouped):
        raise ValueError("classes-per-batch exceeds optimization categories")
    if len(grouped) < 2:
        raise ValueError("fewer than two optimization categories can form positives")

    device = torch.device(args.device)
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    processor = AutoProcessor.from_pretrained(args.model_path, local_files_only=True)
    processor.tokenizer.padding_side = "left"
    backbone = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_path, dtype=dtype, low_cpu_mem_usage=True, local_files_only=True
    ).to(device)
    encoder = Qwen3VLLatentQueryEncoder(backbone, args.num_queries).to(device).train()
    if args.initial_queries is not None:
        initialization = torch.load(args.initial_queries, map_location="cpu", weights_only=True)
        supplied = initialization["latent_queries"]
        if supplied.shape != encoder.latent_queries.shape:
            raise ValueError(
                f"initial query shape {tuple(supplied.shape)} != {tuple(encoder.latent_queries.shape)}"
            )
        encoder.latent_queries.data.copy_(supplied.to(device))
    initial_latent_queries = encoder.latent_queries.detach().cpu().clone()
    optimizer = torch.optim.AdamW([encoder.latent_queries], lr=args.learning_rate)
    history = []

    for step in range(args.steps):
        batch_rows, labels = balanced_batch(
            grouped, args.classes_per_batch, args.objects_per_class, rng
        )

        # import pdb; pdb.set_trace()

        cached = []
        encoder.eval()
        with torch.no_grad():
            for row in batch_rows:
                inputs = move_inputs(build_inputs(processor, row, args.views), device, dtype)
                cached.append(encoder(**inputs)[0].cpu())
        cached_leaf = torch.stack(cached).to(device).detach().requires_grad_(True)
        loss = supervised_contrastive_loss(
            cached_leaf, labels.to(device), args.temperature
        )
        embedding_grads = torch.autograd.grad(loss, cached_leaf)[0].detach()

        optimizer.zero_grad(set_to_none=True)
        encoder.train()
        for row, embedding_grad in zip(batch_rows, embedding_grads):
            inputs = move_inputs(build_inputs(processor, row, args.views), device, dtype)
            embedding = encoder(**inputs)[0]
            torch.dot(embedding, embedding_grad).backward()
        grad_norm = float(encoder.latent_queries.grad.float().norm().item())
        before = encoder.latent_queries.detach().clone()
        optimizer.step()
        update_norm = float((encoder.latent_queries.detach() - before).float().norm().item())
        record = {
            "step": step + 1,
            "loss": float(loss.item()),
            "query_gradient_norm": grad_norm,
            "query_update_norm": update_norm,
            "labels": [row["label_name"] for row in batch_rows],
            "object_ids": [row["object_id"] for row in batch_rows],
        }
        history.append(record)
        print(json.dumps(record), flush=True)
        if step + 1 in set(args.checkpoint_steps):
            args.output_dir.mkdir(parents=True, exist_ok=True)
            current = encoder.latent_queries.detach().cpu()
            torch.save(
                {
                    "latent_queries": current,
                    "initial_latent_queries": initial_latent_queries,
                    "num_queries": args.num_queries,
                    "model_path": str(args.model_path.resolve()),
                    "protocol": str(args.split.resolve()),
                    "step": step + 1,
                    "relative_query_displacement": float(
                        (current.float() - initial_latent_queries.float()).norm()
                        / initial_latent_queries.float().norm()
                    ),
                    "initial_queries_source": (
                        str(args.initial_queries.resolve()) if args.initial_queries else None
                    ),
                },
                args.output_dir / f"queries_step{step + 1:03d}.pt",
            )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "latent_queries": encoder.latent_queries.detach().cpu(),
            "initial_latent_queries": initial_latent_queries,
            "num_queries": args.num_queries,
            "model_path": str(args.model_path.resolve()),
            "protocol": str(args.split.resolve()),
            "relative_query_displacement": float(
                (encoder.latent_queries.detach().cpu().float() - initial_latent_queries.float()).norm()
                / initial_latent_queries.float().norm()
            ),
            "initial_queries_source": (
                str(args.initial_queries.resolve()) if args.initial_queries else None
            ),
        },
        args.output_dir / "queries.pt",
    )
    summary = {
        "method": "multi-view SLQ baseline with exact gradient caching",
        "arguments": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "train_categories": protocol["train_category_names"],
        "validation_categories": protocol["validation_category_names"],
        "sampled_train_categories": sorted(grouped),
        "history": history,
        "trainable_parameters": encoder.latent_queries.numel(),
    }
    (args.output_dir / "history.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
