"""Multi-view object loading and Qwen3-VL input construction."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from PIL import Image

from dataset.manifest import VIEW_COLUMNS
from dataset.prompts import OBJECT_RETRIEVAL_PROMPT
from dataset.views import numeric_angle_order


def read_rows(
    manifest: Path,
    split_path: Path,
) -> tuple[list[dict[str, str]], dict]:
    with manifest.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    protocol = json.loads(split_path.read_text(encoding="utf-8"))
    allowed = set(protocol["optimization_object_ids"])
    selected = [row for row in rows if row["object_id"] in allowed]

    if len(selected) != len(allowed):
        raise ValueError("protocol object IDs do not match the manifest")

    return selected, protocol


def load_views(
    row: dict[str, str],
    count: int,
) -> list[Image.Image]:
    order = numeric_angle_order(row)
    indices = order[:: 24 // count][:count]

    images = []
    for index in indices:
        with Image.open(row[VIEW_COLUMNS[index]]) as image:
            images.append(image.convert("RGB").copy())

    return images


def build_inputs(processor, row: dict[str, str], views: int):
    images = load_views(row, views)
    message = {
        "role": "user",
        "content": [
            *({"type": "image", "image": image} for image in images),
            {"type": "text", "text": OBJECT_RETRIEVAL_PROMPT},
        ],
    }
    prompt = processor.apply_chat_template(
        [message],
        tokenize=False,
        add_generation_prompt=True,
    )
    return processor(
        text=[prompt],
        images=[images],
        padding=True,
        return_tensors="pt",
    )
