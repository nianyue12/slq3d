#!/usr/bin/env python3
"""Generate manifests and the full-train protocol."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


NUM_VIEWS = 24
VIEW_COLUMNS = [
    f"view_{index:02d}"
    for index in range(NUM_VIEWS)
]
FIELDNAMES = [
    "object_id",
    "label_name",
    "split",
    *VIEW_COLUMNS,
]

DATASET_PREFIXES = {
    "esb": "os_esb_core",
    "mn40": "os_mn40_core",
}


def select_dac_views(
    object_dir: Path,
    num_views: int = NUM_VIEWS,
) -> list[Path]:
    """Select the same 24 views used by the old project."""
    all_views = sorted(
        object_dir.glob("image/h_*.jpg")
    )

    if len(all_views) < num_views:
        raise ValueError(
            f"{object_dir}: expected at least "
            f"{num_views} views, found {len(all_views)}"
        )

    selected = all_views[
        :: len(all_views) // num_views
    ][:num_views]

    if len(selected) != num_views:
        raise ValueError(
            f"{object_dir}: failed to select "
            f"{num_views} views"
        )

    return selected


def make_row(
    object_id: str,
    label_name: str,
    split: str,
    object_dir: Path,
) -> dict[str, str]:
    if not object_dir.is_dir():
        raise FileNotFoundError(object_dir)

    views = select_dac_views(object_dir)

    row = {
        "object_id": object_id,
        "label_name": label_name,
        "split": split,
    }
    row.update(
        {
            column: str(view.resolve())
            for column, view in zip(
                VIEW_COLUMNS,
                views,
                strict=True,
            )
        }
    )
    return row


def build_train_rows(
    dataset_root: Path,
) -> list[dict[str, str]]:
    train_root = dataset_root / "train"

    if not train_root.is_dir():
        raise FileNotFoundError(train_root)

    rows = []
    seen_ids = set()

    category_dirs = sorted(
        path
        for path in train_root.iterdir()
        if path.is_dir()
    )

    for category_dir in category_dirs:
        object_dirs = sorted(
            path
            for path in category_dir.iterdir()
            if path.is_dir()
        )

        for object_dir in object_dirs:
            object_id = object_dir.name

            if object_id in seen_ids:
                raise ValueError(
                    f"duplicate Train object ID: {object_id}"
                )
            seen_ids.add(object_id)

            rows.append(
                make_row(
                    object_id=object_id,
                    label_name=category_dir.name,
                    split="train",
                    object_dir=object_dir,
                )
            )

    if not rows:
        raise ValueError(
            f"{train_root}: no Train objects found"
        )

    return rows


def read_labels(
    label_path: Path,
) -> list[tuple[str, str]]:
    rows = []

    with label_path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as handle:
        for line_number, values in enumerate(
            csv.reader(handle),
            start=1,
        ):
            if len(values) != 2:
                raise ValueError(
                    f"{label_path}:{line_number}: "
                    "expected object_id,label_name"
                )

            object_id, label_name = (
                value.strip()
                for value in values
            )

            if not object_id or not label_name:
                raise ValueError(
                    f"{label_path}:{line_number}: "
                    "empty field"
                )

            rows.append((object_id, label_name))

    return rows


def build_evaluation_rows(
    dataset_root: Path,
    split: str,
) -> list[dict[str, str]]:
    label_path = dataset_root / f"{split}_label.txt"
    split_root = dataset_root / split

    if not label_path.is_file():
        raise FileNotFoundError(label_path)

    if not split_root.is_dir():
        raise FileNotFoundError(split_root)

    labels = read_labels(label_path)
    object_ids = [
        object_id
        for object_id, _ in labels
    ]

    if len(object_ids) != len(set(object_ids)):
        raise ValueError(
            f"{label_path}: duplicate object IDs"
        )

    return [
        make_row(
            object_id=object_id,
            label_name=label_name,
            split=split,
            object_dir=split_root / object_id,
        )
        for object_id, label_name in labels
    ]


def write_manifest(
    rows: list[dict[str, str]],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=FIELDNAMES,
        )
        writer.writeheader()
        writer.writerows(rows)


def write_full_train_protocol(
    train_rows: list[dict[str, str]],
    output_path: Path,
) -> None:
    labels = sorted(
        {
            row["label_name"]
            for row in train_rows
        }
    )

    payload = {
        "protocol": (
            "fixed SLQ baseline trained on "
            "all official Train objects"
        ),
        "selection_policy": (
            "configuration frozen before official "
            "Probe/Gallery evaluation"
        ),
        "optimization_object_ids": sorted(
            row["object_id"]
            for row in train_rows
        ),
        "validation_probe_object_ids": [],
        "validation_gallery_object_ids": [],
        "train_category_names": labels,
        "validation_category_names": [],
        "summary": {
            "official_train_objects": len(train_rows),
            "train_categories": len(labels),
        },
    }

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    output_path.write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        choices=tuple(DATASET_PREFIXES),
        required=True,
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root.resolve()

    dataset_dir = Path(__file__).resolve().parent
    manifest_dir = dataset_dir / "manifests"
    protocol_dir = dataset_dir / "protocols"
    prefix = DATASET_PREFIXES[args.dataset]

    split_rows = {
        "train": build_train_rows(dataset_root),
        "query": build_evaluation_rows(
            dataset_root,
            "query",
        ),
        "target": build_evaluation_rows(
            dataset_root,
            "target",
        ),
    }

    for split, rows in split_rows.items():
        output_path = (
            manifest_dir
            / f"{prefix}_{split}.csv"
        )
        write_manifest(rows, output_path)
        print(
            f"{split}: wrote {len(rows)} objects "
            f"to {output_path}"
        )

    protocol_path = (
        protocol_dir
        / f"{prefix}_full_train.json"
    )
    write_full_train_protocol(
        split_rows["train"],
        protocol_path,
    )
    print(f"wrote protocol to {protocol_path}")


if __name__ == "__main__":
    main()