"""Deterministic, transitive leakage-group partition manifests."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
import json
import logging
from pathlib import Path
import re

import polars as pl

from biominer.common.semantic_hash import canonical_semantic_fingerprint
from biominer.flickr_fetch.geographic_clustering import (
    GLOBAL_FALLBACK_CLUSTER_IDS,
)
from biominer.references.readiness import REFERENCE_ROUTES
from biominer.storage.parquet import write_parquet


DATASET_SPLIT_MANIFEST_SCHEMA_VERSION = "dataset-split-manifest-v1.0.0"
DATASET_SPLIT_MANIFEST_FILE = "dataset_split_manifest.parquet"
DATASET_SPLIT_GROUPING_POLICY_VERSION = (
    "transitive-multi-identity-leakage-groups-v1.0.0"
)
DATASET_SPLIT_ASSIGNMENT_POLICY_VERSION = (
    "deterministic-class-aware-component-allocation-v1.0.0"
)
DATASET_SPLITS = (
    "support_train",
    "model_selection",
    "calibration",
    "final_test",
)
DATASET_SPLIT_SET = frozenset(DATASET_SPLITS)

DEFAULT_SUPPORT_TRAIN_WEIGHT = 55
DEFAULT_MODEL_SELECTION_WEIGHT = 15
DEFAULT_CALIBRATION_WEIGHT = 15
DEFAULT_FINAL_TEST_WEIGHT = 15

_SHA256_PATTERN = re.compile(r"sha256:[0-9a-f]{64}\Z")
_UINT32_MAX = 2**32 - 1
_UINT64_MAX = 2**64 - 1
_LOGGER = logging.getLogger(__name__)

_GROUP_FIELDS = (
    "source_observation_id",
    "source_owner_id",
    "observer_id",
    "photographer_id",
    "flickr_owner_id",
    "duplicate_group_id",
    "exact_hash_group_id",
    "perceptual_duplicate_group_id",
    "burst_group_id",
    "provider_mirror_group_id",
    "geo_cluster_id",
)


@dataclass(frozen=True, slots=True)
class DatasetSplitConfig:
    """Versioned weights and deterministic seed for one four-way split."""

    split_version: str
    random_seed: int = 42
    support_train_weight: int = DEFAULT_SUPPORT_TRAIN_WEIGHT
    model_selection_weight: int = DEFAULT_MODEL_SELECTION_WEIGHT
    calibration_weight: int = DEFAULT_CALIBRATION_WEIGHT
    final_test_weight: int = DEFAULT_FINAL_TEST_WEIGHT
    require_class_coverage: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "split_version",
            _required_text(self.split_version, field="split_version"),
        )
        object.__setattr__(
            self,
            "random_seed",
            _bounded_nonnegative_integer(
                self.random_seed,
                maximum=_UINT64_MAX,
                field="random_seed",
            ),
        )
        for field in (
            "support_train_weight",
            "model_selection_weight",
            "calibration_weight",
            "final_test_weight",
        ):
            object.__setattr__(
                self,
                field,
                _bounded_positive_integer(
                    getattr(self, field),
                    maximum=_UINT32_MAX,
                    field=field,
                ),
            )
        if not isinstance(self.require_class_coverage, bool):
            raise TypeError("require_class_coverage must be boolean")

    @property
    def weights(self) -> tuple[tuple[str, int], ...]:
        return (
            ("support_train", self.support_train_weight),
            ("model_selection", self.model_selection_weight),
            ("calibration", self.calibration_weight),
            ("final_test", self.final_test_weight),
        )

    @property
    def total_weight(self) -> int:
        return sum(weight for _, weight in self.weights)

    @property
    def fingerprint(self) -> str:
        return canonical_semantic_fingerprint(
            {
                "schema_version": DATASET_SPLIT_MANIFEST_SCHEMA_VERSION,
                "grouping_policy_version": DATASET_SPLIT_GROUPING_POLICY_VERSION,
                "assignment_policy_version": (DATASET_SPLIT_ASSIGNMENT_POLICY_VERSION),
                "split_version": self.split_version,
                "random_seed": self.random_seed,
                "weights": [list(item) for item in self.weights],
                "require_class_coverage": self.require_class_coverage,
            }
        )


@dataclass(frozen=True, slots=True)
class DatasetSplitItem:
    """One reviewed item and every known identity that can cause leakage."""

    item_type: str
    item_id: str
    source: str
    route: str
    stratification_label: str
    source_artifact_fingerprint: str
    accepted_class_taxon_key: str | None = None
    source_observation_id: str | None = None
    source_owner_id: str | None = None
    observer_id: str | None = None
    photographer_id: str | None = None
    flickr_owner_id: str | None = None
    duplicate_group_id: str | None = None
    exact_hash_group_id: str | None = None
    perceptual_duplicate_group_id: str | None = None
    burst_group_id: str | None = None
    provider_mirror_group_id: str | None = None
    geo_cluster_id: str | None = None

    def __post_init__(self) -> None:
        for field in (
            "item_type",
            "item_id",
            "source",
            "stratification_label",
        ):
            object.__setattr__(
                self,
                field,
                _required_text(getattr(self, field), field=field),
            )
        route = _required_text(self.route, field="route")
        if route not in REFERENCE_ROUTES:
            raise ValueError(f"unsupported route: {route}")
        object.__setattr__(self, "route", route)
        object.__setattr__(
            self,
            "source_artifact_fingerprint",
            _sha256(
                self.source_artifact_fingerprint,
                field="source_artifact_fingerprint",
            ),
        )
        object.__setattr__(
            self,
            "accepted_class_taxon_key",
            _optional_text(
                self.accepted_class_taxon_key,
                field="accepted_class_taxon_key",
            ),
        )
        for field in _GROUP_FIELDS:
            object.__setattr__(
                self,
                field,
                _optional_text(getattr(self, field), field=field),
            )
        if self.source.casefold().startswith("flickr") and self.flickr_owner_id is None:
            raise ValueError("flickr items require flickr_owner_id for split isolation")

    @property
    def key(self) -> tuple[str, str]:
        return (self.item_type, self.item_id)

    @property
    def fingerprint(self) -> str:
        return canonical_semantic_fingerprint(
            {
                "schema_version": DATASET_SPLIT_MANIFEST_SCHEMA_VERSION,
                "item": {
                    "item_type": self.item_type,
                    "item_id": self.item_id,
                    "source": self.source,
                    "route": self.route,
                    "stratification_label": self.stratification_label,
                    "accepted_class_taxon_key": self.accepted_class_taxon_key,
                    "source_artifact_fingerprint": (self.source_artifact_fingerprint),
                    **{field: getattr(self, field) for field in _GROUP_FIELDS},
                },
            }
        )


@dataclass(frozen=True, slots=True)
class DatasetSplitBuild:
    """A validated manifest plus compact allocation audit counts."""

    manifest: pl.DataFrame
    split_fingerprint: str
    configuration_fingerprint: str
    component_count: int
    split_item_counts: tuple[tuple[str, int], ...]
    split_component_counts: tuple[tuple[str, int], ...]


@dataclass(frozen=True, slots=True)
class _LeakageComponent:
    component_id: str
    item_keys: tuple[tuple[str, str], ...]
    item_count: int
    class_item_counts: tuple[tuple[str, int], ...]

    @property
    def class_labels(self) -> tuple[str, ...]:
        return tuple(label for label, _ in self.class_item_counts)


class _DisjointSet:
    def __init__(self, keys: Sequence[tuple[str, str]]) -> None:
        self._parent = {key: key for key in keys}

    def find(self, key: tuple[str, str]) -> tuple[str, str]:
        parent = self._parent[key]
        if parent != key:
            self._parent[key] = self.find(parent)
        return self._parent[key]

    def union(self, left: tuple[str, str], right: tuple[str, str]) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        first, second = sorted((left_root, right_root))
        self._parent[second] = first


def dataset_split_manifest_schema() -> dict[str, pl.DataType]:
    return {
        "schema_version": pl.String,
        "split_version": pl.String,
        "split_fingerprint": pl.String,
        "configuration_fingerprint": pl.String,
        "grouping_policy_version": pl.String,
        "assignment_policy_version": pl.String,
        "random_seed": pl.UInt64,
        "support_train_weight": pl.UInt32,
        "model_selection_weight": pl.UInt32,
        "calibration_weight": pl.UInt32,
        "final_test_weight": pl.UInt32,
        "require_class_coverage": pl.Boolean,
        "item_type": pl.String,
        "item_id": pl.String,
        "item_fingerprint": pl.String,
        "source": pl.String,
        "route": pl.String,
        "stratification_label": pl.String,
        "accepted_class_taxon_key": pl.String,
        "source_artifact_fingerprint": pl.String,
        "source_observation_id": pl.String,
        "source_owner_id": pl.String,
        "observer_id": pl.String,
        "photographer_id": pl.String,
        "flickr_owner_id": pl.String,
        "duplicate_group_id": pl.String,
        "exact_hash_group_id": pl.String,
        "perceptual_duplicate_group_id": pl.String,
        "burst_group_id": pl.String,
        "provider_mirror_group_id": pl.String,
        "geo_cluster_id": pl.String,
        "leakage_component_id": pl.String,
        "leakage_component_size": pl.UInt32,
        "dataset_split": pl.String,
    }


def build_dataset_split_manifest(
    items: Sequence[DatasetSplitItem],
    config: DatasetSplitConfig,
) -> DatasetSplitBuild:
    """Build four partitions over transitive multi-identity components."""

    if not isinstance(config, DatasetSplitConfig):
        raise TypeError("config must be a DatasetSplitConfig")
    normalized = tuple(items)
    if not normalized:
        raise ValueError("dataset split input must not be empty")
    if any(not isinstance(item, DatasetSplitItem) for item in normalized):
        raise TypeError("items must contain DatasetSplitItem values")
    ordered = tuple(sorted(normalized, key=lambda item: item.key))
    keys = tuple(item.key for item in ordered)
    if len(keys) != len(set(keys)):
        raise ValueError("dataset split input contains duplicate item keys")

    _log_event(
        "dataset_split_build_start",
        split_version=config.split_version,
        item_count=len(ordered),
        random_seed=config.random_seed,
        configuration_fingerprint=config.fingerprint,
    )
    components, component_by_item = _build_components(ordered)
    assignments = _assign_components(components, config)
    component_size = {
        component.component_id: component.item_count for component in components
    }
    rows = [
        _manifest_row(
            item,
            config=config,
            component_id=component_by_item[item.key],
            component_size=component_size[component_by_item[item.key]],
            dataset_split=assignments[component_by_item[item.key]],
        )
        for item in ordered
    ]
    split_fingerprint = _split_fingerprint(config, rows)
    for row in rows:
        row["split_fingerprint"] = split_fingerprint
    frame = pl.DataFrame(rows, schema=dataset_split_manifest_schema()).sort(
        "split_version", "item_type", "item_id"
    )
    validate_dataset_split_manifest(frame)
    split_item_counts = _split_counts(frame, unique_component=False)
    split_component_counts = _split_counts(frame, unique_component=True)
    result = DatasetSplitBuild(
        manifest=frame,
        split_fingerprint=split_fingerprint,
        configuration_fingerprint=config.fingerprint,
        component_count=len(components),
        split_item_counts=split_item_counts,
        split_component_counts=split_component_counts,
    )
    _log_event(
        "dataset_split_build_complete",
        split_version=config.split_version,
        item_count=frame.height,
        component_count=len(components),
        split_item_counts=dict(split_item_counts),
        split_component_counts=dict(split_component_counts),
        split_fingerprint=split_fingerprint,
    )
    return result


def validate_dataset_split_manifest(
    frame: pl.DataFrame,
    *,
    expected_split_fingerprint: str | None = None,
    expected_configuration_fingerprint: str | None = None,
) -> None:
    """Recompute every item, component, assignment, and artifact identity."""

    if not isinstance(frame, pl.DataFrame):
        raise TypeError("dataset split manifest must be a Polars DataFrame")
    if dict(frame.schema) != dataset_split_manifest_schema():
        raise ValueError("dataset split manifest physical schema mismatch")
    if frame.is_empty():
        raise ValueError("dataset split manifest must not be empty")
    expected_sort = frame.sort("split_version", "item_type", "item_id")
    if not frame.equals(expected_sort):
        raise ValueError("dataset split manifest is not deterministically sorted")
    if frame.select("split_version", "item_type", "item_id").n_unique() != frame.height:
        raise ValueError("dataset split manifest contains duplicate primary keys")

    config = _config_from_manifest(frame)
    configuration_fingerprint = _single_text(frame, "configuration_fingerprint")
    if configuration_fingerprint != config.fingerprint:
        raise ValueError("dataset split configuration_fingerprint is invalid")
    _match_expected_fingerprint(
        configuration_fingerprint,
        expected_configuration_fingerprint,
        field="configuration_fingerprint",
    )
    if _single_text(frame, "schema_version") != DATASET_SPLIT_MANIFEST_SCHEMA_VERSION:
        raise ValueError("dataset split manifest schema version is incompatible")
    if (
        _single_text(frame, "grouping_policy_version")
        != DATASET_SPLIT_GROUPING_POLICY_VERSION
    ):
        raise ValueError("dataset split grouping policy is incompatible")
    if (
        _single_text(frame, "assignment_policy_version")
        != DATASET_SPLIT_ASSIGNMENT_POLICY_VERSION
    ):
        raise ValueError("dataset split assignment policy is incompatible")

    items: list[DatasetSplitItem] = []
    rows_by_key: dict[tuple[str, str], Mapping[str, object]] = {}
    for row in frame.iter_rows(named=True):
        item = _item_from_manifest_row(row)
        if row["item_fingerprint"] != item.fingerprint:
            raise ValueError(f"item_fingerprint is invalid for {item.key!r}")
        items.append(item)
        rows_by_key[item.key] = row
    components, component_by_item = _build_components(tuple(items))
    assignments = _assign_components(components, config)
    component_sizes = {
        component.component_id: component.item_count for component in components
    }
    splits_by_component: dict[str, set[str]] = defaultdict(set)
    for item in items:
        row = rows_by_key[item.key]
        component_id = component_by_item[item.key]
        if row["leakage_component_id"] != component_id:
            raise ValueError(f"leakage component is invalid for {item.key!r}")
        if int(row["leakage_component_size"]) != component_sizes[component_id]:
            raise ValueError(f"leakage component size is invalid for {item.key!r}")
        dataset_split = str(row["dataset_split"])
        if dataset_split not in DATASET_SPLIT_SET:
            raise ValueError(f"unsupported dataset_split: {dataset_split}")
        splits_by_component[component_id].add(dataset_split)
    crossing = sorted(
        component_id
        for component_id, splits in splits_by_component.items()
        if len(splits) != 1
    )
    if crossing:
        raise ValueError(f"leakage component crosses splits: {crossing[0]}")
    for component_id, expected_split in assignments.items():
        actual_split = next(iter(splits_by_component[component_id]))
        if actual_split != expected_split:
            raise ValueError(
                "dataset split assignment does not match the deterministic policy"
            )
    _validate_group_isolation(frame)

    split_fingerprint = _single_text(frame, "split_fingerprint")
    _sha256(split_fingerprint, field="split_fingerprint")
    _match_expected_fingerprint(
        split_fingerprint,
        expected_split_fingerprint,
        field="split_fingerprint",
    )
    rows_without_split_fingerprint = [
        {key: value for key, value in row.items() if key != "split_fingerprint"}
        for row in frame.iter_rows(named=True)
    ]
    if _split_fingerprint(config, rows_without_split_fingerprint) != split_fingerprint:
        raise ValueError("dataset split_fingerprint is invalid")


def write_dataset_split_manifest(
    result: DatasetSplitBuild | pl.DataFrame,
    output: str | Path,
    *,
    overwrite: bool = False,
) -> Path:
    """Publish an immutable Parquet manifest and verify its physical round trip."""

    frame = result.manifest if isinstance(result, DatasetSplitBuild) else result
    validate_dataset_split_manifest(frame)
    destination = Path(output)
    if destination.suffix.casefold() != ".parquet":
        destination /= DATASET_SPLIT_MANIFEST_FILE
    written = write_parquet(frame, destination, overwrite=overwrite)
    loaded = pl.read_parquet(written)
    validate_dataset_split_manifest(loaded)
    if not frame.equals(loaded):
        raise ValueError("dataset split manifest Parquet round-trip mismatch")
    _log_event(
        "dataset_split_manifest_written",
        path=str(written),
        row_count=frame.height,
        byte_count=written.stat().st_size,
        split_fingerprint=str(frame["split_fingerprint"][0]),
    )
    return written


def load_dataset_split_manifest(
    path: str | Path,
    *,
    expected_split_fingerprint: str | None = None,
    expected_configuration_fingerprint: str | None = None,
) -> pl.DataFrame:
    frame = pl.read_parquet(path)
    validate_dataset_split_manifest(
        frame,
        expected_split_fingerprint=expected_split_fingerprint,
        expected_configuration_fingerprint=expected_configuration_fingerprint,
    )
    return frame


def _build_components(
    items: Sequence[DatasetSplitItem],
) -> tuple[tuple[_LeakageComponent, ...], dict[tuple[str, str], str]]:
    keys = tuple(item.key for item in items)
    disjoint = _DisjointSet(keys)
    first_by_token: dict[tuple[str, ...], tuple[str, str]] = {}
    for item in items:
        for token in _group_tokens(item):
            previous = first_by_token.setdefault(token, item.key)
            disjoint.union(previous, item.key)
    members_by_root: dict[tuple[str, str], list[DatasetSplitItem]] = defaultdict(list)
    for item in items:
        members_by_root[disjoint.find(item.key)].append(item)

    components: list[_LeakageComponent] = []
    component_by_item: dict[tuple[str, str], str] = {}
    for members in members_by_root.values():
        ordered = tuple(sorted(members, key=lambda item: item.key))
        component_id = _component_id(ordered)
        class_counts = tuple(
            sorted(Counter(item.stratification_label for item in ordered).items())
        )
        component = _LeakageComponent(
            component_id=component_id,
            item_keys=tuple(item.key for item in ordered),
            item_count=len(ordered),
            class_item_counts=class_counts,
        )
        components.append(component)
        for item in ordered:
            component_by_item[item.key] = component_id
    return tuple(
        sorted(components, key=lambda item: item.component_id)
    ), component_by_item


def _group_tokens(item: DatasetSplitItem) -> tuple[tuple[str, ...], ...]:
    source = item.source.casefold()
    tokens: list[tuple[str, ...]] = []
    if item.source_observation_id is not None:
        tokens.append(("observation", source, item.source_observation_id))
    for field in (
        "source_owner_id",
        "observer_id",
        "photographer_id",
        "flickr_owner_id",
    ):
        value = getattr(item, field)
        if value is not None:
            tokens.append(("person_or_owner", source, value))
    for field in (
        "duplicate_group_id",
        "exact_hash_group_id",
        "perceptual_duplicate_group_id",
    ):
        value = getattr(item, field)
        if value is not None:
            tokens.append(("duplicate", value))
    if item.burst_group_id is not None:
        tokens.append(("burst", source, item.burst_group_id))
    if item.provider_mirror_group_id is not None:
        tokens.append(("provider_mirror", item.provider_mirror_group_id))
    if item.geo_cluster_id is not None and (
        item.geo_cluster_id not in GLOBAL_FALLBACK_CLUSTER_IDS
    ):
        assert item.geo_cluster_id is not None
        tokens.append(("geo_cluster", item.geo_cluster_id))
    return tuple(tokens)


def _component_id(items: Sequence[DatasetSplitItem]) -> str:
    digest = canonical_semantic_fingerprint(
        {
            "schema_version": DATASET_SPLIT_GROUPING_POLICY_VERSION,
            "items": [
                {
                    "item_type": item.item_type,
                    "item_id": item.item_id,
                    "item_fingerprint": item.fingerprint,
                }
                for item in items
            ],
        }
    ).removeprefix("sha256:")
    return f"leakage-component:{digest}"


def _assign_components(
    components: Sequence[_LeakageComponent],
    config: DatasetSplitConfig,
) -> dict[str, str]:
    if len(components) < len(DATASET_SPLITS):
        raise ValueError(
            "dataset requires at least four independent components for four partitions"
        )
    component_counts_by_class: Counter[str] = Counter()
    item_counts_by_class: Counter[str] = Counter()
    for component in components:
        for class_label, count in component.class_item_counts:
            component_counts_by_class[class_label] += 1
            item_counts_by_class[class_label] += count
    if config.require_class_coverage:
        insufficient = {
            label: count
            for label, count in sorted(component_counts_by_class.items())
            if count < len(DATASET_SPLITS)
        }
        if insufficient:
            raise ValueError(
                "class independent components cannot cover all four partitions: "
                f"{sorted(insufficient.items())}"
            )

    ordered = sorted(
        components,
        key=lambda component: _component_assignment_order_key(
            component,
            component_counts_by_class=component_counts_by_class,
            random_seed=config.random_seed,
        ),
    )
    remaining_by_class = Counter(component_counts_by_class)
    uncovered_global = set(DATASET_SPLITS)
    uncovered_by_class = {
        label: set(DATASET_SPLITS) for label in component_counts_by_class
    }
    assigned_items = Counter({name: 0 for name in DATASET_SPLITS})
    assigned_components = Counter({name: 0 for name in DATASET_SPLITS})
    assigned_class_items: dict[str, Counter[str]] = {
        name: Counter() for name in DATASET_SPLITS
    }
    assignments: dict[str, str] = {}
    total_items = sum(component.item_count for component in components)

    for index, component in enumerate(ordered):
        remaining_components_after = len(ordered) - index - 1
        feasible: list[tuple[tuple[object, ...], str]] = []
        for split_name in DATASET_SPLITS:
            if not _coverage_reservations_hold(
                component,
                split_name,
                remaining_components_after=remaining_components_after,
                remaining_by_class=remaining_by_class,
                uncovered_global=uncovered_global,
                uncovered_by_class=uncovered_by_class,
                require_class_coverage=config.require_class_coverage,
            ):
                continue
            class_coverage_gain = sum(
                split_name in uncovered_by_class[class_label]
                for class_label in component.class_labels
            )
            global_coverage_gain = int(split_name in uncovered_global)
            balance_delta = _assignment_balance_delta(
                component,
                split_name,
                config=config,
                total_items=total_items,
                item_counts_by_class=item_counts_by_class,
                assigned_items=assigned_items,
                assigned_components=assigned_components,
                assigned_class_items=assigned_class_items,
                total_components=len(components),
            )
            tie_break = canonical_semantic_fingerprint(
                {
                    "schema_version": DATASET_SPLIT_ASSIGNMENT_POLICY_VERSION,
                    "random_seed": config.random_seed,
                    "component_id": component.component_id,
                    "split": split_name,
                }
            )
            score: tuple[object, ...] = (
                -class_coverage_gain,
                -global_coverage_gain,
                balance_delta,
                assigned_components[split_name],
                tie_break,
                DATASET_SPLITS.index(split_name),
            )
            feasible.append((score, split_name))
        if not feasible:
            raise ValueError(
                "unable to assign leakage components without violating coverage "
                f"reservations at {component.component_id}"
            )
        _, selected = min(feasible, key=lambda item: item[0])
        assignments[component.component_id] = selected
        assigned_items[selected] += component.item_count
        assigned_components[selected] += 1
        uncovered_global.discard(selected)
        for class_label, count in component.class_item_counts:
            assigned_class_items[selected][class_label] += count
            remaining_by_class[class_label] -= 1
            uncovered_by_class[class_label].discard(selected)

    if uncovered_global:
        raise ValueError(f"dataset partitions are empty: {sorted(uncovered_global)}")
    if config.require_class_coverage:
        missing = {
            label: sorted(splits)
            for label, splits in uncovered_by_class.items()
            if splits
        }
        if missing:
            raise ValueError(f"classes are missing from dataset partitions: {missing}")
    return assignments


def _component_assignment_order_key(
    component: _LeakageComponent,
    *,
    component_counts_by_class: Mapping[str, int],
    random_seed: int,
) -> tuple[object, ...]:
    rarity = min(component_counts_by_class[label] for label in component.class_labels)
    tie_break = canonical_semantic_fingerprint(
        {
            "schema_version": DATASET_SPLIT_ASSIGNMENT_POLICY_VERSION,
            "random_seed": random_seed,
            "component_id": component.component_id,
        }
    )
    return (
        rarity,
        -len(component.class_labels),
        -component.item_count,
        tie_break,
        component.component_id,
    )


def _coverage_reservations_hold(
    component: _LeakageComponent,
    split_name: str,
    *,
    remaining_components_after: int,
    remaining_by_class: Mapping[str, int],
    uncovered_global: set[str],
    uncovered_by_class: Mapping[str, set[str]],
    require_class_coverage: bool,
) -> bool:
    global_after = len(uncovered_global - {split_name})
    if global_after > remaining_components_after:
        return False
    if not require_class_coverage:
        return True
    for class_label in component.class_labels:
        class_remaining_after = remaining_by_class[class_label] - 1
        class_uncovered_after = len(uncovered_by_class[class_label] - {split_name})
        if class_uncovered_after > class_remaining_after:
            return False
    return True


def _assignment_balance_delta(
    component: _LeakageComponent,
    split_name: str,
    *,
    config: DatasetSplitConfig,
    total_items: int,
    item_counts_by_class: Mapping[str, int],
    assigned_items: Mapping[str, int],
    assigned_components: Mapping[str, int],
    assigned_class_items: Mapping[str, Mapping[str, int]],
    total_components: int,
) -> Fraction:
    weights = dict(config.weights)
    weight = weights[split_name]
    before = _relative_error(
        assigned_items[split_name],
        total=total_items,
        weight=weight,
        total_weight=config.total_weight,
    )
    after = _relative_error(
        assigned_items[split_name] + component.item_count,
        total=total_items,
        weight=weight,
        total_weight=config.total_weight,
    )
    delta = after - before
    component_before = _relative_error(
        assigned_components[split_name],
        total=total_components,
        weight=weight,
        total_weight=config.total_weight,
    )
    component_after = _relative_error(
        assigned_components[split_name] + 1,
        total=total_components,
        weight=weight,
        total_weight=config.total_weight,
    )
    delta += component_after - component_before
    for class_label, count in component.class_item_counts:
        class_total = item_counts_by_class[class_label]
        class_before = _relative_error(
            assigned_class_items[split_name].get(class_label, 0),
            total=class_total,
            weight=weight,
            total_weight=config.total_weight,
        )
        class_after = _relative_error(
            assigned_class_items[split_name].get(class_label, 0) + count,
            total=class_total,
            weight=weight,
            total_weight=config.total_weight,
        )
        delta += class_after - class_before
    return delta


def _relative_error(
    count: int,
    *,
    total: int,
    weight: int,
    total_weight: int,
) -> Fraction:
    difference = count * total_weight - total * weight
    denominator = max(total * weight, 1)
    return Fraction(difference * difference, denominator * denominator)


def _manifest_row(
    item: DatasetSplitItem,
    *,
    config: DatasetSplitConfig,
    component_id: str,
    component_size: int,
    dataset_split: str,
) -> dict[str, object]:
    return {
        "schema_version": DATASET_SPLIT_MANIFEST_SCHEMA_VERSION,
        "split_version": config.split_version,
        "split_fingerprint": "",
        "configuration_fingerprint": config.fingerprint,
        "grouping_policy_version": DATASET_SPLIT_GROUPING_POLICY_VERSION,
        "assignment_policy_version": DATASET_SPLIT_ASSIGNMENT_POLICY_VERSION,
        "random_seed": config.random_seed,
        "support_train_weight": config.support_train_weight,
        "model_selection_weight": config.model_selection_weight,
        "calibration_weight": config.calibration_weight,
        "final_test_weight": config.final_test_weight,
        "require_class_coverage": config.require_class_coverage,
        "item_type": item.item_type,
        "item_id": item.item_id,
        "item_fingerprint": item.fingerprint,
        "source": item.source,
        "route": item.route,
        "stratification_label": item.stratification_label,
        "accepted_class_taxon_key": item.accepted_class_taxon_key,
        "source_artifact_fingerprint": item.source_artifact_fingerprint,
        **{field: getattr(item, field) for field in _GROUP_FIELDS},
        "leakage_component_id": component_id,
        "leakage_component_size": component_size,
        "dataset_split": dataset_split,
    }


def _item_from_manifest_row(row: Mapping[str, object]) -> DatasetSplitItem:
    return DatasetSplitItem(
        item_type=str(row["item_type"]),
        item_id=str(row["item_id"]),
        source=str(row["source"]),
        route=str(row["route"]),
        stratification_label=str(row["stratification_label"]),
        accepted_class_taxon_key=_optional_row_text(row["accepted_class_taxon_key"]),
        source_artifact_fingerprint=str(row["source_artifact_fingerprint"]),
        **{field: _optional_row_text(row[field]) for field in _GROUP_FIELDS},
    )


def _config_from_manifest(frame: pl.DataFrame) -> DatasetSplitConfig:
    return DatasetSplitConfig(
        split_version=_single_text(frame, "split_version"),
        random_seed=_single_integer(frame, "random_seed"),
        support_train_weight=_single_integer(frame, "support_train_weight"),
        model_selection_weight=_single_integer(frame, "model_selection_weight"),
        calibration_weight=_single_integer(frame, "calibration_weight"),
        final_test_weight=_single_integer(frame, "final_test_weight"),
        require_class_coverage=_single_boolean(frame, "require_class_coverage"),
    )


def _split_fingerprint(
    config: DatasetSplitConfig,
    rows: Sequence[Mapping[str, object]],
) -> str:
    assignments = [
        {
            "item_type": str(row["item_type"]),
            "item_id": str(row["item_id"]),
            "item_fingerprint": str(row["item_fingerprint"]),
            "leakage_component_id": str(row["leakage_component_id"]),
            "dataset_split": str(row["dataset_split"]),
        }
        for row in sorted(
            rows,
            key=lambda item: (str(item["item_type"]), str(item["item_id"])),
        )
    ]
    return canonical_semantic_fingerprint(
        {
            "schema_version": DATASET_SPLIT_MANIFEST_SCHEMA_VERSION,
            "configuration_fingerprint": config.fingerprint,
            "assignments": assignments,
        }
    )


def _validate_group_isolation(frame: pl.DataFrame) -> None:
    field_namespaces = {
        "source_observation_id": ("source",),
        "source_owner_id": ("source",),
        "observer_id": ("source",),
        "photographer_id": ("source",),
        "flickr_owner_id": ("source",),
        "duplicate_group_id": (),
        "exact_hash_group_id": (),
        "perceptual_duplicate_group_id": (),
        "burst_group_id": ("source",),
        "provider_mirror_group_id": (),
        "geo_cluster_id": (),
    }
    for field, namespace_fields in field_namespaces.items():
        assignments: dict[tuple[str, ...], str] = {}
        selected = (*namespace_fields, field, "dataset_split")
        for values in frame.select(*selected).iter_rows():
            *namespace, raw_value, split = values
            if raw_value is None or (
                field == "geo_cluster_id"
                and raw_value in GLOBAL_FALLBACK_CLUSTER_IDS
            ):
                continue
            key = tuple(str(value).casefold() for value in namespace) + (
                str(raw_value),
            )
            previous = assignments.setdefault(key, str(split))
            if previous != str(split):
                raise ValueError(f"{field} {raw_value!r} crosses dataset splits")


def _split_counts(
    frame: pl.DataFrame,
    *,
    unique_component: bool,
) -> tuple[tuple[str, int], ...]:
    if unique_component:
        grouped = frame.group_by("dataset_split").agg(
            pl.col("leakage_component_id").n_unique().alias("count")
        )
    else:
        grouped = frame.group_by("dataset_split").agg(pl.len().alias("count"))
    counts = {
        str(row["dataset_split"]): int(row["count"])
        for row in grouped.iter_rows(named=True)
    }
    return tuple((name, counts.get(name, 0)) for name in DATASET_SPLITS)


def _single_text(frame: pl.DataFrame, field: str) -> str:
    values = frame[field].unique().to_list()
    if len(values) != 1:
        raise ValueError(f"dataset split manifest has multiple {field} values")
    return _required_text(values[0], field=field)


def _single_integer(frame: pl.DataFrame, field: str) -> int:
    values = frame[field].unique().to_list()
    if len(values) != 1:
        raise ValueError(f"dataset split manifest has multiple {field} values")
    value = values[0]
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field} must be an integer")
    return value


def _single_boolean(frame: pl.DataFrame, field: str) -> bool:
    values = frame[field].unique().to_list()
    if len(values) != 1 or not isinstance(values[0], bool):
        raise ValueError(f"dataset split manifest has multiple {field} values")
    return values[0]


def _required_text(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ValueError(f"{field} must be non-empty canonical text")
    return value


def _optional_text(value: object, *, field: str) -> str | None:
    return None if value is None else _required_text(value, field=field)


def _optional_row_text(value: object) -> str | None:
    return None if value is None else str(value)


def _sha256(value: object, *, field: str) -> str:
    text = _required_text(value, field=field)
    if _SHA256_PATTERN.fullmatch(text) is None:
        raise ValueError(f"{field} must be a full lowercase sha256 fingerprint")
    return text


def _bounded_nonnegative_integer(
    value: object,
    *,
    maximum: int,
    field: str,
) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < 0
        or value > maximum
    ):
        raise ValueError(f"{field} must be an integer between zero and {maximum}")
    return value


def _bounded_positive_integer(
    value: object,
    *,
    maximum: int,
    field: str,
) -> int:
    result = _bounded_nonnegative_integer(value, maximum=maximum, field=field)
    if result == 0:
        raise ValueError(f"{field} must be positive")
    return result


def _match_expected_fingerprint(
    actual: str,
    expected: str | None,
    *,
    field: str,
) -> None:
    if expected is None:
        return
    validated = _sha256(expected, field=f"expected_{field}")
    if actual != validated:
        raise ValueError(f"{field} does not match the expected fingerprint")


def _log_event(event: str, **fields: object) -> None:
    _LOGGER.info(
        "%s",
        json.dumps(
            {"event": event, **fields},
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ),
    )


__all__ = [
    "DATASET_SPLIT_ASSIGNMENT_POLICY_VERSION",
    "DATASET_SPLIT_GROUPING_POLICY_VERSION",
    "DATASET_SPLIT_MANIFEST_FILE",
    "DATASET_SPLIT_MANIFEST_SCHEMA_VERSION",
    "DATASET_SPLITS",
    "DatasetSplitBuild",
    "DatasetSplitConfig",
    "DatasetSplitItem",
    "build_dataset_split_manifest",
    "dataset_split_manifest_schema",
    "load_dataset_split_manifest",
    "validate_dataset_split_manifest",
    "write_dataset_split_manifest",
]
