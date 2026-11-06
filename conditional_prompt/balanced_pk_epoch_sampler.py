"""Minimum-repeat complete-coverage Balanced P-K epoch sampling."""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from typing import Any


CLASSES_PER_BATCH = 2
OBJECTS_PER_CLASS = 2
STATE_VERSION = 2


@dataclass(frozen=True)
class EpochBatch:
    rows: list[dict[str, str]]
    class_names: list[str]
    labels: list[int]
    repeated_object_ids: list[str]


class BalancedPKEpochSampler:
    """Pair shuffled two-object class groups into a minimum-repeat plan."""

    def __init__(
        self, grouped: dict[str, list[dict[str, str]]], *, seed: int
    ) -> None:
        if len(grouped) < CLASSES_PER_BATCH:
            raise ValueError("at least two categories are required for P=2")
        self.seed = int(seed)
        self.categories = sorted(grouped)
        self.rows_by_id: dict[str, dict[str, str]] = {}
        self.ids_by_category: dict[str, list[str]] = {}
        for category in self.categories:
            rows = grouped[category]
            if len(rows) < OBJECTS_PER_CLASS:
                raise ValueError(
                    f"category {category!r} has {len(rows)} objects; "
                    "P=2, K=2 requires at least 2 distinct objects per category"
                )
            category_ids = []
            for row in rows:
                object_id = row["object_id"]
                if row["label_name"] != category:
                    raise ValueError(
                        f"object {object_id}: label_name differs from group {category}"
                    )
                if object_id in self.rows_by_id:
                    raise ValueError(f"duplicate object ID across groups: {object_id}")
                self.rows_by_id[object_id] = row
                category_ids.append(object_id)
            self.ids_by_category[category] = sorted(category_ids)

        self.total_object_count = len(self.rows_by_id)
        self.dataset_signature = self._dataset_signature()
        self.epoch: int | None = None
        self.rng = random.Random()
        self.batch_plan: list[tuple[str, list[str], str, list[str]]] = []
        self.plan_cursor = 0
        self.visited: set[str] = set()
        self.batch_count = 0
        self.repeat_count = 0
        self.category_selection_count: dict[str, int] = {}
        self.category_repeat_count: dict[str, int] = {}
        self.base_group_count: dict[str, int] = {}
        self.added_balance_group_count: dict[str, int] = {}

    def _dataset_signature(self) -> str:
        serialized = json.dumps(
            self.ids_by_category,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    def _epoch_seed(self, epoch: int) -> int:
        digest = hashlib.sha256(f"{self.seed}:{epoch}".encode("utf-8")).digest()
        return int.from_bytes(digest[:8], byteorder="big", signed=False)

    def _make_base_groups(self) -> dict[str, list[list[str]]]:
        """Shuffle each class, form pairs, and fill only an odd final object."""

        groups: dict[str, list[list[str]]] = {}
        for category in self.categories:
            object_ids = list(self.ids_by_category[category])
            self.rng.shuffle(object_ids)
            category_groups = [
                object_ids[index : index + OBJECTS_PER_CLASS]
                for index in range(0, len(object_ids), OBJECTS_PER_CLASS)
            ]
            if len(category_groups[-1]) == 1:
                lone_id = category_groups[-1][0]
                candidates = [object_id for object_id in object_ids if object_id != lone_id]
                category_groups[-1].append(self.rng.choice(candidates))
            self.rng.shuffle(category_groups)
            groups[category] = category_groups
        return groups

    @staticmethod
    def _minimum_balance_groups(group_counts: dict[str, int]) -> int:
        """Minimum extra groups required for even, cross-class pairability."""

        total = sum(group_counts.values())
        maximum = max(group_counts.values())
        required = max(0, 2 * maximum - total)
        if (total + required) % 2:
            required += 1
        return required

    def _add_balance_groups(self, groups: dict[str, list[list[str]]]) -> None:
        """Correct parity/dominance only when a general distribution needs it."""

        counts = {category: len(items) for category, items in groups.items()}
        extra_count = self._minimum_balance_groups(counts)
        self.added_balance_group_count = {category: 0 for category in self.categories}
        for _ in range(extra_count):
            maximum = max(counts.values())
            dominant = self.rng.choice(
                [category for category, count in counts.items() if count == maximum]
            )
            eligible = [category for category in self.categories if category != dominant]
            minimum = min(counts[category] for category in eligible)
            recipient = self.rng.choice(
                [category for category in eligible if counts[category] == minimum]
            )
            groups[recipient].append(
                self.rng.sample(self.ids_by_category[recipient], OBJECTS_PER_CLASS)
            )
            counts[recipient] += 1
            self.added_balance_group_count[recipient] += 1

        total = sum(counts.values())
        if total % 2 or max(counts.values()) > total // 2:
            raise RuntimeError("failed to make class-group distribution pairable")

    def _choose_maximum_category(
        self,
        groups: dict[str, list[list[str]]],
        *,
        exclude: str | None = None,
    ) -> str:
        eligible = [
            category
            for category in self.categories
            if category != exclude and groups[category]
        ]
        if not eligible:
            raise RuntimeError("no different category remains for group pairing")
        maximum = max(len(groups[category]) for category in eligible)
        return self.rng.choice(
            [category for category in eligible if len(groups[category]) == maximum]
        )

    def _build_batch_plan(
        self, groups: dict[str, list[list[str]]]
    ) -> list[tuple[str, list[str], str, list[str]]]:
        """Pair groups from the two largest remaining different categories."""

        plan = []
        while any(groups.values()):
            first = self._choose_maximum_category(groups)
            second = self._choose_maximum_category(groups, exclude=first)
            first_group = groups[first].pop()
            second_group = groups[second].pop()
            if len(set(first_group)) != 2 or len(set(second_group)) != 2:
                raise RuntimeError("a class group contains duplicate Object IDs")
            plan.append((first, first_group, second, second_group))
        return plan

    def start_epoch(self, epoch: int) -> None:
        """Build a deterministic minimum-repeat plan and reset progress."""

        if epoch < 0:
            raise ValueError("epoch must be non-negative")
        self.epoch = int(epoch)
        self.rng = random.Random(self._epoch_seed(epoch))
        groups = self._make_base_groups()
        self.base_group_count = {
            category: len(items) for category, items in groups.items()
        }
        self._add_balance_groups(groups)
        self.batch_plan = self._build_batch_plan(groups)
        self.plan_cursor = 0
        self.visited = set()
        self.batch_count = 0
        self.repeat_count = 0
        self.category_selection_count = {category: 0 for category in self.categories}
        self.category_repeat_count = {category: 0 for category in self.categories}

    @property
    def finished(self) -> bool:
        self._require_started()
        return self.plan_cursor == len(self.batch_plan)

    def _require_started(self) -> None:
        if self.epoch is None:
            raise RuntimeError("start_epoch() or load_state_dict() must be called first")

    def next_batch(self) -> EpochBatch:
        """Return the next planned P=2, K=2 batch or raise StopIteration."""

        self._require_started()
        if self.finished:
            raise StopIteration
        first, first_ids, second, second_ids = self.batch_plan[self.plan_cursor]
        categories = [first, second]
        grouped_ids = [first_ids, second_ids]
        rows: list[dict[str, str]] = []
        repeated_ids: list[str] = []
        for category, object_ids in zip(categories, grouped_ids):
            for object_id in object_ids:
                if object_id in self.visited:
                    repeated_ids.append(object_id)
                    self.repeat_count += 1
                    self.category_repeat_count[category] += 1
                else:
                    self.visited.add(object_id)
                rows.append(self.rows_by_id[object_id])
                self.category_selection_count[category] += 1

        batch_ids = [row["object_id"] for row in rows]
        if len(batch_ids) != 4 or len(batch_ids) != len(set(batch_ids)):
            raise RuntimeError("a planned batch does not contain four distinct objects")
        self.plan_cursor += 1
        self.batch_count += 1
        return EpochBatch(
            rows=rows,
            class_names=categories,
            labels=[0, 0, 1, 1],
            repeated_object_ids=repeated_ids,
        )

    def state_dict(self) -> dict[str, Any]:
        """Serialize exact plan, cursor, statistics, and RNG state."""

        self._require_started()
        return {
            "version": STATE_VERSION,
            "seed": self.seed,
            "epoch": self.epoch,
            "classes_per_batch": CLASSES_PER_BATCH,
            "objects_per_class": OBJECTS_PER_CLASS,
            "dataset_signature": self.dataset_signature,
            "rng_state": self.rng.getstate(),
            "batch_plan": [
                (first, list(first_ids), second, list(second_ids))
                for first, first_ids, second, second_ids in self.batch_plan
            ],
            "plan_cursor": self.plan_cursor,
            "visited": sorted(self.visited),
            "batch_count": self.batch_count,
            "repeat_count": self.repeat_count,
            "category_selection_count": dict(self.category_selection_count),
            "category_repeat_count": dict(self.category_repeat_count),
            "base_group_count": dict(self.base_group_count),
            "added_balance_group_count": dict(self.added_balance_group_count),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore only a structurally valid state for this exact dataset."""

        expected = {
            "version": STATE_VERSION,
            "seed": self.seed,
            "classes_per_batch": CLASSES_PER_BATCH,
            "objects_per_class": OBJECTS_PER_CLASS,
            "dataset_signature": self.dataset_signature,
        }
        for key, value in expected.items():
            if state.get(key) != value:
                raise ValueError(
                    f"sampler checkpoint {key}={state.get(key)!r}, expected {value!r}"
                )
        plan = state.get("batch_plan")
        if not isinstance(plan, list) or not plan:
            raise ValueError("sampler checkpoint has no batch plan")
        normalized_plan = []
        for item in plan:
            if len(item) != 4:
                raise ValueError("sampler checkpoint contains a malformed batch")
            first, first_ids, second, second_ids = item
            if first == second or first not in self.ids_by_category or second not in self.ids_by_category:
                raise ValueError("sampler checkpoint contains invalid batch categories")
            if len(first_ids) != 2 or len(second_ids) != 2:
                raise ValueError("sampler checkpoint violates K=2")
            if not set(first_ids) <= set(self.ids_by_category[first]):
                raise ValueError("sampler checkpoint mislabels first-category objects")
            if not set(second_ids) <= set(self.ids_by_category[second]):
                raise ValueError("sampler checkpoint mislabels second-category objects")
            if len(set(first_ids + second_ids)) != 4:
                raise ValueError("sampler checkpoint repeats an Object ID in a batch")
            normalized_plan.append((first, list(first_ids), second, list(second_ids)))

        cursor = int(state["plan_cursor"])
        if cursor < 0 or cursor > len(normalized_plan):
            raise ValueError("sampler checkpoint cursor is outside its plan")
        planned_ids = {
            object_id
            for _, first_ids, _, second_ids in normalized_plan
            for object_id in first_ids + second_ids
        }
        if planned_ids != set(self.rows_by_id):
            raise ValueError("sampler checkpoint plan does not cover the dataset exactly")
        expected_visited = {
            object_id
            for _, first_ids, _, second_ids in normalized_plan[:cursor]
            for object_id in first_ids + second_ids
        }
        visited = set(state.get("visited", []))
        if visited != expected_visited:
            raise ValueError("sampler checkpoint visited set differs from plan prefix")
        if int(state["batch_count"]) != cursor:
            raise ValueError("sampler checkpoint batch count differs from cursor")

        seen: set[str] = set()
        expected_repeats = 0
        expected_selections = {category: 0 for category in self.categories}
        expected_category_repeats = {category: 0 for category in self.categories}
        for first, first_ids, second, second_ids in normalized_plan[:cursor]:
            for category, object_ids in ((first, first_ids), (second, second_ids)):
                for object_id in object_ids:
                    expected_selections[category] += 1
                    if object_id in seen:
                        expected_repeats += 1
                        expected_category_repeats[category] += 1
                    else:
                        seen.add(object_id)
        if int(state["repeat_count"]) != expected_repeats:
            raise ValueError("sampler checkpoint repeat count differs from plan prefix")
        if state["category_selection_count"] != expected_selections:
            raise ValueError("sampler checkpoint category selections differ from plan prefix")
        if state["category_repeat_count"] != expected_category_repeats:
            raise ValueError("sampler checkpoint category repeats differ from plan prefix")

        self.epoch = int(state["epoch"])
        self.rng = random.Random()
        self.rng.setstate(state["rng_state"])
        self.batch_plan = normalized_plan
        self.plan_cursor = cursor
        self.visited = visited
        self.batch_count = int(state["batch_count"])
        self.repeat_count = int(state["repeat_count"])
        self.category_selection_count = {
            category: int(state["category_selection_count"][category])
            for category in self.categories
        }
        self.category_repeat_count = {
            category: int(state["category_repeat_count"][category])
            for category in self.categories
        }
        self.base_group_count = {
            category: int(state["base_group_count"][category])
            for category in self.categories
        }
        self.added_balance_group_count = {
            category: int(state["added_balance_group_count"][category])
            for category in self.categories
        }

    def summary(self) -> dict[str, Any]:
        """Report plan size, exact coverage, repeats, and per-class statistics."""

        self._require_started()
        unique_coverage = len(self.visited)
        category_coverage = {}
        for category in self.categories:
            category_ids = set(self.ids_by_category[category])
            unique_count = len(category_ids & self.visited)
            total_count = len(category_ids)
            category_coverage[category] = {
                "unique_objects": unique_count,
                "total_objects": total_count,
                "coverage_rate": unique_count / total_count,
                "base_groups": self.base_group_count[category],
                "added_balance_groups": self.added_balance_group_count[category],
                "total_selections": self.category_selection_count[category],
                "repeat_selections": self.category_repeat_count[category],
            }
        return {
            "epoch": self.epoch,
            "finished": self.finished,
            "batch_count": self.batch_count,
            "planned_batch_count": len(self.batch_plan),
            "total_object_count": self.total_object_count,
            "unique_object_coverage": unique_coverage,
            "coverage_rate": unique_coverage / self.total_object_count,
            "repeat_sampling_count": self.repeat_count,
            "total_object_selections": self.batch_count * 4,
            "base_group_count": dict(self.base_group_count),
            "added_balance_group_count": dict(self.added_balance_group_count),
            "category_coverage": category_coverage,
        }
