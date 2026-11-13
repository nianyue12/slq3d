#!/usr/bin/env python3
"""Train View-Prompt for complete, resumable Balanced-PK epochs."""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path

import torch
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

from conditional_prompt.balanced_pk_epoch_sampler import BalancedPKEpochSampler
from conditional_prompt.common import DEFAULT_MODEL_PATH
from conditional_prompt.static_object_encoder import StaticPromptObjectEncoder
from conditional_prompt.static_prompt_encoder import Qwen3VLStaticPromptEncoder
from conditional_prompt.train_static_prompt import append_jsonl, save_checkpoint_atomic
from conditional_prompt.view_prompt.model import ViewConditionedPrompt
from conditional_prompt.view_prompt.conditions import load_view_condition
from conditional_prompt.view_prompt.training import train_view_prompt_one_step


ROOT = Path(__file__).resolve().parents[2]
DATASETS = {
    "os_esb_core": {
        "manifest": ROOT / "dataset/esb_prompt/os_esb_core_pk_train.csv",
        "view_cache": Path(
            "/mnt/data/jwjin/realign_3d/outputs/"
            "os_esb_core/embeddings/view_objects"
        ),
        "output": Path("/mnt/data/jwjin/slq3d_outputs/view_prompt/esb_epoch1"),
    },
    "os_mn40_core": {
        "manifest": ROOT / "dataset/manifests/os_mn40_core_train.csv",
        "view_cache": Path(
            "/mnt/data/jwjin/realign_3d/outputs/"
            "os_mn40_core/embeddings/view_objects"
        ),
        "output": Path("/mnt/data/jwjin/slq3d_outputs/view_prompt/mn40_epoch1"),
    },
}


def args_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=tuple(DATASETS), default="os_esb_core")
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--view-cache-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20261103)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--max-updates", type=int)
    return parser.parse_args()


def rng_state():
    return {
        "python": random.getstate(), "torch_cpu": torch.get_rng_state(),
        "torch_cuda_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_rng(state):
    random.setstate(state["python"])
    torch.set_rng_state(state["torch_cpu"].cpu())
    if state["torch_cuda_all"] is not None:
        torch.cuda.set_rng_state_all([item.cpu() for item in state["torch_cuda_all"]])


def main():
    args = args_parser()
    if args.epochs != 1:
        raise ValueError("the formal View-Prompt comparison uses exactly one epoch")
    dataset = DATASETS[args.dataset]
    manifest = (args.manifest or dataset["manifest"]).resolve()
    view_cache_dir = (args.view_cache_dir or dataset["view_cache"]).resolve()
    output = (args.output_dir or dataset["output"]).resolve()
    if not str(output).startswith("/mnt/data/"):
        raise ValueError("View-Prompt training output must be on /mnt/data")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("formal VC training requires CUDA")

    with manifest.open("r", encoding="utf-8", newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if row.get("split") == "train"]
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["label_name"]].append(row)
    sampler = BalancedPKEpochSampler(dict(grouped), seed=args.seed)
    generator = ViewConditionedPrompt.fair_original_initialization(args.seed).to(device)
    optimizer = torch.optim.AdamW(generator.parameters(), lr=1e-4)
    epoch_index = 0
    global_step = 0
    if args.resume:
        payload = torch.load(args.resume, map_location="cpu", weights_only=False)
        if payload.get("format") != "vc_dprompt_v1":
            raise ValueError("resume checkpoint is not compatible with View-Prompt")
        generator.load_state_dict(payload["generator"], strict=True)
        optimizer.load_state_dict(payload["optimizer"])
        for state in optimizer.state.values():
            for key, value in state.items():
                if isinstance(value, torch.Tensor):
                    state[key] = value.to(device)
        sampler.load_state_dict(payload["sampler"])
        epoch_index = int(payload["epoch_index"])
        global_step = int(payload["global_step"])
        restore_rng(payload["rng_state"])
    else:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        sampler.start_epoch(0)

    processor = AutoProcessor.from_pretrained(args.model_path, local_files_only=True)
    processor.tokenizer.padding_side = "left"
    dtype = torch.float16
    backbone = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_path, dtype=dtype, low_cpu_mem_usage=True, local_files_only=True,
    ).to(device)
    encoder = Qwen3VLStaticPromptEncoder(backbone).to(device)
    encoder.soft_prompt.requires_grad_(False)
    object_encoder = StaticPromptObjectEncoder(encoder, processor, chunk_size=4).to(device)
    if any(parameter.requires_grad for parameter in backbone.parameters()):
        raise RuntimeError("Qwen is not fully frozen")

    output.mkdir(parents=True, exist_ok=True)
    log_path = output / "train.jsonl"
    updates_this_run = 0
    while not sampler.finished:
        selected = sampler.next_batch()
        conditions = [load_view_condition(row, view_cache_dir)[0] for row in selected.rows]
        metrics = train_view_prompt_one_step(
            generator=generator, view_encoder=encoder, object_encoder=object_encoder,
            processor=processor, optimizer=optimizer, batch_rows=selected.rows,
            labels=torch.tensor(selected.labels), conditions=conditions,
            device=device, model_dtype=dtype, chunk_size=4, temperature=0.05,
        )
        global_step += 1
        updates_this_run += 1
        record = {
            "epoch_index": epoch_index, "epoch_step": sampler.plan_cursor,
            "epoch_steps": len(sampler.batch_plan), "global_step": global_step,
            "object_ids": [row["object_id"] for row in selected.rows],
            "class_names": selected.class_names,
            "repeated_object_ids": selected.repeated_object_ids,
            **metrics,
        }
        append_jsonl(log_path, record)
        payload = {
            "format": "vc_dprompt_v1", "generator": generator.state_dict(),
            "optimizer": optimizer.state_dict(), "sampler": sampler.state_dict(),
            "epoch_index": epoch_index, "global_step": global_step,
            "rng_state": rng_state(),
            "configuration": {
                "model_path": str(args.model_path.resolve()),
                "dataset": args.dataset,
                "manifest": str(manifest),
                "view_cache_dir": str(view_cache_dir),
                "epochs": 1, "seed": args.seed, "learning_rate": 1e-4,
                "temperature": 0.05, "P": 2, "K": 2, "views": 24,
                "prompt_tokens": 4, "trainable_parameters": 2_118_912,
                "condition": "24 last-layer full-valid-token view embeddings",
                "readout": "last-layer full-valid-token mean; view L2; FP32 Mean24; object L2",
            },
        }
        save_checkpoint_atomic(payload, output / "latest.pt")
        print(json.dumps(record, sort_keys=True), flush=True)
        if args.max_updates is not None and updates_this_run >= args.max_updates:
            break

    complete = sampler.finished
    print(json.dumps({
        "training_complete": complete, "global_step": global_step,
        "epoch_steps": len(sampler.batch_plan), "updates_this_run": updates_this_run,
        "output": str(output / "latest.pt"),
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
