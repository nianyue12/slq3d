"""Evaluate saved object embeddings with the supplied DEC/DAC metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import scipy.spatial
import torch

from metrics.retrieval import anmrr_score, map_score, ndcg_score, recall_score


def load_payload(path: Path) -> tuple[np.ndarray, np.ndarray]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    embeddings = payload["object_embeddings"].float().numpy()
    labels = np.asarray(payload["label_names"])
    return embeddings, labels


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--method", required=True)
    args = parser.parse_args()

    query, query_labels = load_payload(args.query)
    target, target_labels = load_payload(args.target)
    distances = scipy.spatial.distance.cdist(query, target, "cosine")
    cutoff = min(100, target.shape[0])
    result = {
        "method": args.method,
        "protocol": "official Probe/Gallery",
        "query_objects": int(query.shape[0]),
        "gallery_objects": int(target.shape[0]),
        "test_categories": int(np.unique(query_labels).size),
        "metric_implementation": "metrics/retrieval.py",
        "metrics": {
            "mAP": float(map_score(distances, query_labels, target_labels)),
            "NDCG@100": float(ndcg_score(distances, query_labels, target_labels, k=cutoff)),
            "ANMRR": float(anmrr_score(distances, query_labels, target_labels)),
            "Recall@100": float(recall_score(distances, query_labels, target_labels, top_k=cutoff)),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
