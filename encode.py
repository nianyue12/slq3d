#!/usr/bin/env python3
"""Encode official Probe and Gallery splits with a fixed SLQ checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

from dataset.multiview_inputs import build_inputs
from models.qwen3_vl_query_encoder import Qwen3VLLatentQueryEncoder
from utils.io import move_inputs, save_atomic
from dataset.frozen_mean import read_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--query-manifest", type=Path, required=True)
    parser.add_argument("--target-manifest", type=Path, required=True)
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--views", type=int, choices=(1, 2, 4), default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint = torch.load(args.queries, map_location="cpu", weights_only=True)
    device = torch.device(args.device)
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    processor = AutoProcessor.from_pretrained(args.model_path, local_files_only=True)
    processor.tokenizer.padding_side = "left"
    backbone = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_path, dtype=dtype, low_cpu_mem_usage=True, local_files_only=True
    ).to(device)
    encoder = Qwen3VLLatentQueryEncoder(backbone, int(checkpoint["num_queries"])).to(device).eval()
    encoder.latent_queries.data.copy_(checkpoint["latent_queries"].to(device))

    for split, manifest in (("query", args.query_manifest), ("target", args.target_manifest)):
        rows = read_manifest(manifest, split)
        embeddings = []
        with torch.no_grad():
            for index, row in enumerate(rows, start=1):
                inputs = move_inputs(build_inputs(processor, row, args.views), device, dtype)
                embeddings.append(encoder(**inputs)[0].cpu().to(torch.float16))
                if index % 50 == 0:
                    print(f"{split}: {index}/{len(rows)}", flush=True)
        save_atomic(
            {
                "split": split,
                "object_embeddings": torch.stack(embeddings),
                "object_ids": [row["object_id"] for row in rows],
                "label_names": [row["label_name"] for row in rows],
                "method": "fixed multi-view SLQ official open-set evaluation",
                "views": args.views,
                "queries": str(args.queries.resolve()),
            },
            args.output_dir / f"{split}.pt",
        )
        print(f"wrote {split}: {len(rows)} objects", flush=True)


if __name__ == "__main__":
    main()
