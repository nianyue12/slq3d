#!/usr/bin/env python3
"""Resumably encode official Query/Gallery with View-Prompt conditions."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

from conditional_prompt.dynamic_conditions import build_inference_mapping, mapping_sha256
from conditional_prompt.encode_static_prompt import DATASET_MANIFESTS, read_manifest_rows
from conditional_prompt.static_object_encoder import StaticPromptObjectEncoder
from conditional_prompt.static_prompt_encoder import Qwen3VLStaticPromptEncoder
from conditional_prompt.view_prompt.model import ViewConditionedPrompt
from conditional_prompt.view_prompt.conditions import load_view_condition
from utils.io import save_atomic


MODES = ("correct", "shuffled", "shared")
ROLES = ("query", "target")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=tuple(DATASET_MANIFESTS), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--view-cache-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--shuffle-seed", type=int, default=20261103)
    parser.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    return parser.parse_args()


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def load_checkpoint(path, device):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("format") != "vc_dprompt_v1":
        raise ValueError("not a compatible View-Prompt checkpoint")
    sampler = payload.get("sampler", {})
    if int(sampler.get("plan_cursor", -1)) != len(sampler.get("batch_plan", [])):
        raise ValueError("View-Prompt checkpoint training is incomplete")
    generator = ViewConditionedPrompt()
    generator.load_state_dict(payload["generator"], strict=True)
    generator.eval()
    for parameter in generator.parameters():
        parameter.requires_grad_(False)
    return payload, generator.to(device)


def load_cached(path, row, metadata):
    if not path.is_file():
        return None
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("metadata") != metadata or payload.get("object_id") != row["object_id"]:
        raise ValueError(f"{path}: incompatible resumable VC cache")
    value = payload.get("object_embedding")
    if not isinstance(value, torch.Tensor) or tuple(value.shape) != (4096,):
        raise ValueError(f"{path}: invalid VC embedding")
    return value.float()


def main():
    args = parse_args()
    output = args.output_dir.resolve()
    if not str(output).startswith("/mnt/data/"):
        raise ValueError("View-Prompt inference output must be on /mnt/data")
    device = torch.device(args.device)
    training, generator = load_checkpoint(args.checkpoint.resolve(), device)
    model_path = Path(training["configuration"]["model_path"])
    checkpoint_hash = digest(args.checkpoint)

    processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
    processor.tokenizer.padding_side = "left"
    backbone = Qwen3VLForConditionalGeneration.from_pretrained(
        model_path, dtype=torch.float16, low_cpu_mem_usage=True, local_files_only=True,
    ).to(device)
    view_encoder = Qwen3VLStaticPromptEncoder(backbone).to(device).eval()
    view_encoder.soft_prompt.requires_grad_(False)
    object_encoder = StaticPromptObjectEncoder(view_encoder, processor, chunk_size=4).to(device)

    for role in ROLES:
        manifest = DATASET_MANIFESTS[args.dataset][role]
        rows = read_manifest_rows(manifest, role)
        # Strict independent encoding: Query and Gallery each get their own
        # label-free derangement, constructed without looking at the other role.
        mapping = build_inference_mapping(rows, args.shuffle_seed)
        mapping_hash = mapping_sha256(mapping)
        by_id = {row["object_id"]: row for row in rows}
        conditions = {}
        if any(mode != "shared" for mode in args.modes):
            conditions = {
                object_id: load_view_condition(row, args.view_cache_dir)[0]
                for object_id, row in by_id.items()
            }
        for mode in args.modes:
            metadata = {
                "format": "vc_dprompt_v1_official", "dataset": args.dataset,
                "role": role, "mode": mode, "checkpoint_sha256": checkpoint_hash,
                "manifest": str(manifest.resolve()), "shuffle_seed": args.shuffle_seed,
                "role_specific_mapping_sha256": mapping_hash,
                "view_pooling": "last-layer all-valid-token mean then L2",
                "object_fusion": "24-view FP32 mean then L2",
            }
            object_dir = output / mode / role / "objects"
            object_dir.mkdir(parents=True, exist_ok=True)
            completed = {}
            for row in rows:
                cached = load_cached(object_dir / f"{row['object_id']}.pt", row, metadata)
                if cached is not None:
                    completed[row["object_id"]] = cached
            for index, row in enumerate(rows, 1):
                object_id = row["object_id"]
                if object_id in completed:
                    print(f"{args.dataset} {mode} {role}: {index}/{len(rows)} skip {object_id}", flush=True)
                    continue
                if mode == "shared":
                    prompt = generator.shared_prompt
                    source_id = None
                    delta = torch.zeros_like(prompt)
                else:
                    source_id = object_id if mode == "correct" else mapping[object_id]
                    generated = generator(conditions[source_id].unsqueeze(0).to(device))
                    prompt = generated.dynamic_prompt[0]
                    delta = generated.delta[0]
                encoded = object_encoder(
                    row, device=device, model_dtype=torch.float16,
                    prompt_embeddings=prompt,
                )
                embedding = encoded.object_embedding.detach().float()
                save_atomic({
                    "object_id": object_id, "label_name": row["label_name"],
                    "object_embedding": embedding.to(torch.float16),
                    "condition_source_object_id": source_id,
                    "prompt_delta": delta.detach().cpu().to(torch.float16),
                    "metadata": metadata,
                }, object_dir / f"{object_id}.pt")
                completed[object_id] = embedding
                print(f"{args.dataset} {mode} {role}: {index}/{len(rows)} encoded {object_id}", flush=True)
            matrix = torch.stack([completed[row["object_id"]] for row in rows])
            save_atomic({
                "object_embeddings": matrix.to(torch.float16),
                "object_ids": [row["object_id"] for row in rows],
                "label_names": [row["label_name"] for row in rows],
                "metadata": metadata,
            }, output / mode / f"{role}.pt")
            print(f"wrote {output / mode / f'{role}.pt'}", flush=True)


if __name__ == "__main__":
    main()
