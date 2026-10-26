#!/usr/bin/env python3
"""Initialize retrieval queries from fixed Qwen input-token embeddings."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from safetensors import safe_open
from transformers import AutoTokenizer


# Fixed before retrieval evaluation. The first 16 entries that tokenize to exactly
# one token (with a leading space) define the semantic-query set.
CANDIDATES = [
    "shape", "structure", "part", "function", "surface", "material",
    "geometry", "outline", "contour", "symmetry", "arrangement", "component",
    "feature", "appearance", "spatial", "visual", "detail", "form", "size",
    "proportion", "orientation", "view", "texture", "edge", "volume", "body",
    "base", "support", "opening", "connection",
]


def stats(x: torch.Tensor) -> dict[str, object]:
    norms = x.float().norm(dim=1)
    unit = torch.nn.functional.normalize(x.float(), dim=1)
    cosine = unit @ unit.T
    off_diagonal = cosine[~torch.eye(len(x), dtype=torch.bool)]
    return {
        "norms": [float(v) for v in norms],
        "mean_norm": float(norms.mean()),
        "std_norm": float(norms.std(unbiased=True)),
        "pairwise_cosine_mean": float(off_diagonal.mean()),
        "pairwise_cosine_std": float(off_diagonal.std(unbiased=True)),
        "pairwise_cosine_min": float(off_diagonal.min()),
        "pairwise_cosine_max": float(off_diagonal.max()),
    }


def save_new(payload: dict[str, object], path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=False)
    torch.save(payload, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--random-p4q16", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    if args.output_root.exists():
        raise FileExistsError(f"refusing to reuse {args.output_root}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    audit = []
    selected = []
    for word in CANDIDATES:
        text = " " + word
        ids = tokenizer.encode(text, add_special_tokens=False)
        entry = {
            "word": word,
            "encoded_text": text,
            "token_ids": ids,
            "decoded_tokens": [tokenizer.decode([token_id]) for token_id in ids],
            "single_token": len(ids) == 1,
        }
        audit.append(entry)
        if len(ids) == 1 and len(selected) < 16:
            selected.append(entry)
    if len(selected) != 16:
        raise RuntimeError(f"only found {len(selected)} single-token candidates")
    token_ids = [entry["token_ids"][0] for entry in selected]
    if len(set(token_ids)) != 16:
        raise RuntimeError("selected token IDs are not unique")

    index = json.loads((args.model_path / "model.safetensors.index.json").read_text())
    key = "model.language_model.embed_tokens.weight"
    shard = args.model_path / index["weight_map"][key]
    with safe_open(shard, framework="pt", device="cpu") as handle:
        embedding_table = handle.get_tensor(key)
        pretrained_raw = embedding_table[token_ids].float()

    random_checkpoint = torch.load(args.random_p4q16, map_location="cpu", weights_only=True)
    random_queries = random_checkpoint["latent_queries"].float()
    random_mean_norm = random_queries.norm(dim=1).mean()
    pretrained_mean_norm = pretrained_raw.norm(dim=1).mean()
    scale = random_mean_norm / pretrained_mean_norm
    pretrained = (pretrained_raw * scale).to(random_checkpoint["latent_queries"].dtype)
    prefix = random_checkpoint["prefix_tokens"].clone()

    common = {
        "latent_queries": pretrained,
        "initial_latent_queries": pretrained.clone(),
        "num_queries": 16,
        "model_path": str(args.model_path.resolve()),
        "initialization": "fixed_single_token_qwen_input_embeddings_mean_norm_matched",
        "selected_words": [entry["word"] for entry in selected],
        "selected_token_ids": token_ids,
        "uniform_scale": float(scale),
    }
    save_new(common, args.output_root / "slq16_pretrained" / "queries.pt")
    save_new(
        {
            **common,
            "prefix_tokens": prefix,
            "initial_prefix_tokens": prefix.clone(),
            "num_prefix_tokens": 4,
            "prefix_source": str(args.random_p4q16.resolve()),
        },
        args.output_root / "p4q16_pretrained" / "tokens.pt",
    )
    report = {
        "selection_rule": "first 16 unique candidates encoding to one token after one leading space",
        "candidate_audit": audit,
        "selected": selected,
        "uniform_scale": float(scale),
        "random_query_stats": stats(random_queries),
        "pretrained_raw_stats": stats(pretrained_raw),
        "pretrained_scaled_stats": stats(pretrained.float()),
        "query_equality_between_slq_and_p4q16": True,
    }
    (args.output_root / "initialization_audit.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"selected": report["selected"], "uniform_scale": float(scale)}, indent=2))


if __name__ == "__main__":
    main()
