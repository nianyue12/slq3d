#!/usr/bin/env python3
"""Evaluate official View-Prompt Correct/Shuffled/Shared embeddings."""

import argparse
import json
from pathlib import Path

from conditional_prompt.evaluate_m1_dev import evaluate, load


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    args = parser.parse_args()
    results = {"dataset": args.dataset, "protocol": "official Probe/Gallery", "modes": {}}
    for mode in ("correct", "shuffled", "shared"):
        results["modes"][mode] = evaluate(
            load(args.input_dir / mode / "query.pt"),
            load(args.input_dir / mode / "target.pt"),
        )
    results["comparisons_mAP"] = {
        "correct_minus_shuffled": results["modes"]["correct"]["metrics"]["mAP"] - results["modes"]["shuffled"]["metrics"]["mAP"],
        "correct_minus_shared": results["modes"]["correct"]["metrics"]["mAP"] - results["modes"]["shared"]["metrics"]["mAP"],
    }
    path = args.input_dir / "evaluation.json"
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)
    print(json.dumps({
        "dataset": args.dataset,
        "metrics": {key: value["metrics"] for key, value in results["modes"].items()},
        "comparisons_mAP": results["comparisons_mAP"], "output": str(path),
    }, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
