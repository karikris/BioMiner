"""Leakage-safe reviewed Flickr components and frozen evaluation splits."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math

import polars as pl

from biominer.common.semantic_hash import canonical_semantic_fingerprint


REVIEWED_FLICKR_COMPONENT_SCHEMA_VERSION = (
    "reviewed-flickr-independence-component-v1.0.0"
)
REVIEWED_FLICKR_COMPONENT_FILE = "reviewed_flickr_independence_components.parquet"
DYNAMIC_POOL_EVALUATION_SPLIT_SCHEMA_VERSION = "dynamic-pool-evaluation-split-v1.0.0"
DYNAMIC_POOL_EVALUATION_SPLIT_FILE = "dynamic_pool_evaluation_splits.parquet"
DYNAMIC_POOL_EVALUATION_SPLITS = ("calibration", "validation", "final_test")
DYNAMIC_POOL_EVALUATION_SPLIT_SET = frozenset(DYNAMIC_POOL_EVALUATION_SPLITS)
DYNAMIC_POOL_SPLIT_ASSIGNMENT_POLICY_VERSION = (
    "deterministic-outcome-aware-component-allocation-v1.0.0"
)

REVIEWED_FLICKR_COMPONENT_SCHEMA: dict[str, pl.DataType] = {
    "schema_version": pl.String,
    "register_fingerprint": pl.String,
    "item_fingerprint": pl.String,
    "item_id": pl.String,
    "source_record_hash": pl.String,
    "source_artifact_fingerprint": pl.String,
    "review_decision_fingerprint": pl.String,
    "flickr_photo_id": pl.String,
    "owner_group_id": pl.String,
    "duplicate_group_id": pl.String,
    "observation_group_id": pl.String,
    "geographic_cluster_id": pl.String,
    "no_geo": pl.Boolean,
    "source_mirror_group_id": pl.String,
    "stratum_id": pl.String,
    "candidate_species_key": pl.String,
    "human_supported": pl.Boolean,
    "sampling_weight": pl.Float64,
    "independence_component_id": pl.String,
    "independence_component_size": pl.UInt32,
}

DYNAMIC_POOL_EVALUATION_SPLIT_SCHEMA: dict[str, pl.DataType] = {
    **REVIEWED_FLICKR_COMPONENT_SCHEMA,
    "split_schema_version": pl.String,
    "split_version": pl.String,
    "split_policy_fingerprint": pl.String,
    "split_fingerprint": pl.String,
    "assignment_policy_version": pl.String,
    "random_seed": pl.UInt64,
    "calibration_weight": pl.UInt32,
    "validation_weight": pl.UInt32,
    "final_test_weight": pl.UInt32,
    "require_outcome_coverage": pl.Boolean,
    "evaluation_split": pl.String,
}


@dataclass(frozen=True, slots=True)
class ReviewedFlickrSplitItem:
    """One decisive reviewed candidate and every source-independence identity."""

    item_id: str
    source_record_hash: str
    source_artifact_fingerprint: str
    review_decision_fingerprint: str
    flickr_photo_id: str
    owner_group_id: str
    duplicate_group_id: str
    observation_group_id: str
    geographic_cluster_id: str | None
    no_geo: bool
    source_mirror_group_id: str
    stratum_id: str
    candidate_species_key: str
    human_supported: bool
    sampling_weight: float

    def __post_init__(self) -> None:
        for field in (
            "item_id",
            "flickr_photo_id",
            "owner_group_id",
            "duplicate_group_id",
            "observation_group_id",
            "source_mirror_group_id",
            "stratum_id",
            "candidate_species_key",
        ):
            object.__setattr__(
                self, field, _required_text(getattr(self, field), field=field)
            )
        for field in (
            "source_record_hash",
            "source_artifact_fingerprint",
            "review_decision_fingerprint",
        ):
            object.__setattr__(self, field, _sha256(getattr(self, field), field=field))
        if not isinstance(self.no_geo, bool):
            raise TypeError("no_geo must be a boolean")
        cluster = _optional_text(
            self.geographic_cluster_id, field="geographic_cluster_id"
        )
        if self.no_geo and cluster is not None:
            raise ValueError("no_geo reviewed items cannot claim a geographic cluster")
        if not self.no_geo and cluster is None:
            raise ValueError(
                "georeferenced reviewed items require geographic_cluster_id"
            )
        object.__setattr__(self, "geographic_cluster_id", cluster)
        if not isinstance(self.human_supported, bool):
            raise TypeError("human_supported must be a decisive boolean")
        object.__setattr__(
            self,
            "sampling_weight",
            _positive_float(self.sampling_weight, field="sampling_weight"),
        )

    @property
    def fingerprint(self) -> str:
        return canonical_semantic_fingerprint(
            {
                "schema_version": REVIEWED_FLICKR_COMPONENT_SCHEMA_VERSION,
                "item_id": self.item_id,
                "source_record_hash": self.source_record_hash,
                "source_artifact_fingerprint": self.source_artifact_fingerprint,
                "review_decision_fingerprint": self.review_decision_fingerprint,
                "flickr_photo_id": self.flickr_photo_id,
                "owner_group_id": self.owner_group_id,
                "duplicate_group_id": self.duplicate_group_id,
                "observation_group_id": self.observation_group_id,
                "geographic_cluster_id": self.geographic_cluster_id,
                "no_geo": self.no_geo,
                "source_mirror_group_id": self.source_mirror_group_id,
                "stratum_id": self.stratum_id,
                "candidate_species_key": self.candidate_species_key,
                "human_supported": self.human_supported,
                "sampling_weight": self.sampling_weight,
            }
        )


@dataclass(frozen=True, slots=True)
class ReviewedFlickrComponentBuild:
    register: pl.DataFrame
    register_fingerprint: str
    item_count: int
    component_count: int
    maximum_component_size: int


@dataclass(frozen=True, slots=True)
class DynamicPoolEvaluationSplitPolicy:
    """Immutable three-way component allocation policy."""

    split_version: str
    random_seed: int = 42
    calibration_weight: int = 40
    validation_weight: int = 30
    final_test_weight: int = 30
    require_outcome_coverage: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "split_version",
            _required_text(self.split_version, field="split_version"),
        )
        if (
            not isinstance(self.random_seed, int)
            or isinstance(self.random_seed, bool)
            or not 0 <= self.random_seed < 2**64
        ):
            raise ValueError("random_seed must be an unsigned 64-bit integer")
        for field in (
            "calibration_weight",
            "validation_weight",
            "final_test_weight",
        ):
            value = getattr(self, field)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{field} must be a positive integer")
            if value >= 2**32:
                raise ValueError(f"{field} must fit UInt32")
        if not isinstance(self.require_outcome_coverage, bool):
            raise TypeError("require_outcome_coverage must be a boolean")

    @property
    def weights(self) -> tuple[tuple[str, int], ...]:
        return (
            ("calibration", self.calibration_weight),
            ("validation", self.validation_weight),
            ("final_test", self.final_test_weight),
        )

    @property
    def total_weight(self) -> int:
        return sum(weight for _, weight in self.weights)

    @property
    def fingerprint(self) -> str:
        return canonical_semantic_fingerprint(
            {
                "schema_version": DYNAMIC_POOL_EVALUATION_SPLIT_SCHEMA_VERSION,
                "assignment_policy_version": (
                    DYNAMIC_POOL_SPLIT_ASSIGNMENT_POLICY_VERSION
                ),
                "split_version": self.split_version,
                "random_seed": self.random_seed,
                "weights": self.weights,
                "require_outcome_coverage": self.require_outcome_coverage,
            }
        )


@dataclass(frozen=True, slots=True)
class DynamicPoolEvaluationSplitBuild:
    manifest: pl.DataFrame
    split_fingerprint: str
    split_policy_fingerprint: str
    component_count: int
    split_item_counts: tuple[tuple[str, int], ...]
    split_component_counts: tuple[tuple[str, int], ...]


@dataclass(frozen=True, slots=True)
class _SplitComponent:
    component_id: str
    item_count: int
    outcome_item_counts: tuple[tuple[str, int], ...]

    @property
    def outcomes(self) -> tuple[str, ...]:
        return tuple(outcome for outcome, _ in self.outcome_item_counts)


class _DisjointSet:
    def __init__(self, item_ids: Sequence[str]) -> None:
        self.parent = {item_id: item_id for item_id in item_ids}

    def find(self, item_id: str) -> str:
        parent = self.parent[item_id]
        if parent != item_id:
            self.parent[item_id] = self.find(parent)
        return self.parent[item_id]

    def union(self, left: str, right: str) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        first, second = sorted((left_root, right_root))
        self.parent[second] = first


def build_reviewed_flickr_components(
    items: Sequence[ReviewedFlickrSplitItem],
) -> ReviewedFlickrComponentBuild:
    """Build transitive components before any calibration/test assignment."""

    normalized = tuple(items)
    if not normalized:
        raise ValueError("reviewed Flickr split input must not be empty")
    if any(not isinstance(item, ReviewedFlickrSplitItem) for item in normalized):
        raise TypeError("items must contain ReviewedFlickrSplitItem values")
    ordered = tuple(sorted(normalized, key=lambda item: item.item_id))
    item_ids = tuple(item.item_id for item in ordered)
    if len(item_ids) != len(set(item_ids)):
        raise ValueError("reviewed Flickr item_id must be unique")
    components = _component_members(ordered)
    component_by_item: dict[str, str] = {}
    component_sizes: dict[str, int] = {}
    for members in components:
        component_id = _component_id(members)
        component_sizes[component_id] = len(members)
        for item in members:
            component_by_item[item.item_id] = component_id
    semantic_rows = [
        {
            "item_fingerprint": item.fingerprint,
            "item_id": item.item_id,
            "independence_component_id": component_by_item[item.item_id],
            "independence_component_size": component_sizes[
                component_by_item[item.item_id]
            ],
        }
        for item in ordered
    ]
    register_fingerprint = canonical_semantic_fingerprint(
        {
            "schema_version": REVIEWED_FLICKR_COMPONENT_SCHEMA_VERSION,
            "items": semantic_rows,
        }
    )
    rows = [
        {
            "schema_version": REVIEWED_FLICKR_COMPONENT_SCHEMA_VERSION,
            "register_fingerprint": register_fingerprint,
            "item_fingerprint": item.fingerprint,
            **_item_values(item),
            "independence_component_id": component_by_item[item.item_id],
            "independence_component_size": component_sizes[
                component_by_item[item.item_id]
            ],
        }
        for item in ordered
    ]
    register = pl.DataFrame(
        rows,
        schema=REVIEWED_FLICKR_COMPONENT_SCHEMA,
        strict=True,
    ).sort("item_id")
    validate_reviewed_flickr_components(register)
    return ReviewedFlickrComponentBuild(
        register=register,
        register_fingerprint=register_fingerprint,
        item_count=register.height,
        component_count=len(components),
        maximum_component_size=max(component_sizes.values()),
    )


def validate_reviewed_flickr_components(register: pl.DataFrame) -> None:
    """Recompute identities, components and complete-register fingerprint."""

    if not isinstance(register, pl.DataFrame):
        raise TypeError("register must be a Polars DataFrame")
    if register.schema != REVIEWED_FLICKR_COMPONENT_SCHEMA:
        raise ValueError("reviewed Flickr component schema does not match contract")
    if not register.height:
        raise ValueError("reviewed Flickr component register must not be empty")
    if not register.equals(register.sort("item_id")):
        raise ValueError("reviewed Flickr component register is not sorted")
    if register["item_id"].n_unique() != register.height:
        raise ValueError("reviewed Flickr item_id must be unique")
    if set(register["schema_version"].to_list()) != {
        REVIEWED_FLICKR_COMPONENT_SCHEMA_VERSION
    }:
        raise ValueError("unsupported reviewed Flickr component schema version")
    items = tuple(_item_from_row(row) for row in register.to_dicts())
    for item, row in zip(items, register.to_dicts(), strict=True):
        if row["item_fingerprint"] != item.fingerprint:
            raise ValueError(f"item fingerprint mismatch for {item.item_id}")
    components = _component_members(items)
    expected_by_item: dict[str, tuple[str, int]] = {}
    for members in components:
        component_id = _component_id(members)
        for item in members:
            expected_by_item[item.item_id] = (component_id, len(members))
    for row in register.to_dicts():
        expected_id, expected_size = expected_by_item[str(row["item_id"])]
        if row["independence_component_id"] != expected_id:
            raise ValueError("reviewed Flickr independence component mismatch")
        if row["independence_component_size"] != expected_size:
            raise ValueError("reviewed Flickr independence component size mismatch")
    expected_fingerprint = canonical_semantic_fingerprint(
        {
            "schema_version": REVIEWED_FLICKR_COMPONENT_SCHEMA_VERSION,
            "items": [
                {
                    "item_fingerprint": item.fingerprint,
                    "item_id": item.item_id,
                    "independence_component_id": expected_by_item[item.item_id][0],
                    "independence_component_size": expected_by_item[item.item_id][1],
                }
                for item in items
            ],
        }
    )
    if set(register["register_fingerprint"].to_list()) != {expected_fingerprint}:
        raise ValueError("reviewed Flickr register fingerprint mismatch")


def build_dynamic_pool_evaluation_splits(
    components: pl.DataFrame,
    policy: DynamicPoolEvaluationSplitPolicy,
) -> DynamicPoolEvaluationSplitBuild:
    """Freeze component-level calibration, validation and final-test roles."""

    validate_reviewed_flickr_components(components)
    if not isinstance(policy, DynamicPoolEvaluationSplitPolicy):
        raise TypeError("policy must be a DynamicPoolEvaluationSplitPolicy")
    summaries = _split_components(components)
    assignments = _assign_split_components(summaries, policy=policy)
    register_fingerprint = components["register_fingerprint"].item(0)
    split_fingerprint = canonical_semantic_fingerprint(
        {
            "schema_version": DYNAMIC_POOL_EVALUATION_SPLIT_SCHEMA_VERSION,
            "component_register_fingerprint": register_fingerprint,
            "split_policy_fingerprint": policy.fingerprint,
            "component_assignments": sorted(assignments.items()),
        }
    )
    rows = []
    for row in components.to_dicts():
        rows.append(
            {
                **row,
                "split_schema_version": (DYNAMIC_POOL_EVALUATION_SPLIT_SCHEMA_VERSION),
                "split_version": policy.split_version,
                "split_policy_fingerprint": policy.fingerprint,
                "split_fingerprint": split_fingerprint,
                "assignment_policy_version": (
                    DYNAMIC_POOL_SPLIT_ASSIGNMENT_POLICY_VERSION
                ),
                "random_seed": policy.random_seed,
                "calibration_weight": policy.calibration_weight,
                "validation_weight": policy.validation_weight,
                "final_test_weight": policy.final_test_weight,
                "require_outcome_coverage": policy.require_outcome_coverage,
                "evaluation_split": assignments[str(row["independence_component_id"])],
            }
        )
    manifest = pl.DataFrame(
        rows,
        schema=DYNAMIC_POOL_EVALUATION_SPLIT_SCHEMA,
        strict=True,
    ).sort("item_id")
    validate_dynamic_pool_evaluation_splits(manifest)
    item_counts = tuple(
        (split, manifest.filter(pl.col("evaluation_split") == split).height)
        for split in DYNAMIC_POOL_EVALUATION_SPLITS
    )
    component_counts = tuple(
        (
            split,
            manifest.filter(pl.col("evaluation_split") == split)[
                "independence_component_id"
            ].n_unique(),
        )
        for split in DYNAMIC_POOL_EVALUATION_SPLITS
    )
    return DynamicPoolEvaluationSplitBuild(
        manifest=manifest,
        split_fingerprint=split_fingerprint,
        split_policy_fingerprint=policy.fingerprint,
        component_count=len(summaries),
        split_item_counts=item_counts,
        split_component_counts=component_counts,
    )


def validate_dynamic_pool_evaluation_splits(manifest: pl.DataFrame) -> None:
    """Recompute the frozen allocation and reject identity crossover."""

    if not isinstance(manifest, pl.DataFrame):
        raise TypeError("manifest must be a Polars DataFrame")
    if manifest.schema != DYNAMIC_POOL_EVALUATION_SPLIT_SCHEMA:
        raise ValueError("dynamic-pool evaluation split schema does not match contract")
    if not manifest.height:
        raise ValueError("dynamic-pool evaluation split manifest must not be empty")
    if not manifest.equals(manifest.sort("item_id")):
        raise ValueError("dynamic-pool evaluation split manifest is not sorted")
    component_register = manifest.select(*REVIEWED_FLICKR_COMPONENT_SCHEMA)
    validate_reviewed_flickr_components(component_register)
    if set(manifest["split_schema_version"].to_list()) != {
        DYNAMIC_POOL_EVALUATION_SPLIT_SCHEMA_VERSION
    }:
        raise ValueError("unsupported dynamic-pool evaluation split schema version")
    if set(manifest["assignment_policy_version"].to_list()) != {
        DYNAMIC_POOL_SPLIT_ASSIGNMENT_POLICY_VERSION
    }:
        raise ValueError("unsupported dynamic-pool split assignment policy")
    policy = DynamicPoolEvaluationSplitPolicy(
        split_version=_single_value(manifest, "split_version"),
        random_seed=int(_single_value(manifest, "random_seed")),
        calibration_weight=int(_single_value(manifest, "calibration_weight")),
        validation_weight=int(_single_value(manifest, "validation_weight")),
        final_test_weight=int(_single_value(manifest, "final_test_weight")),
        require_outcome_coverage=bool(
            _single_value(manifest, "require_outcome_coverage")
        ),
    )
    if set(manifest["split_policy_fingerprint"].to_list()) != {policy.fingerprint}:
        raise ValueError("dynamic-pool split policy fingerprint mismatch")
    summaries = _split_components(component_register)
    expected_assignments = _assign_split_components(summaries, policy=policy)
    actual_by_component: dict[str, str] = {}
    for component_id in manifest["independence_component_id"].unique().to_list():
        rows = manifest.filter(pl.col("independence_component_id") == component_id)
        splits = set(rows["evaluation_split"].to_list())
        if len(splits) != 1:
            raise ValueError("reviewed Flickr independence component crosses splits")
        split = next(iter(splits))
        if split not in DYNAMIC_POOL_EVALUATION_SPLIT_SET:
            raise ValueError(f"unsupported evaluation split: {split}")
        actual_by_component[str(component_id)] = str(split)
    if actual_by_component != expected_assignments:
        raise ValueError("dynamic-pool evaluation split assignment mismatch")
    if set(actual_by_component.values()) != DYNAMIC_POOL_EVALUATION_SPLIT_SET:
        raise ValueError("dynamic-pool evaluation split contains an empty partition")
    expected_fingerprint = canonical_semantic_fingerprint(
        {
            "schema_version": DYNAMIC_POOL_EVALUATION_SPLIT_SCHEMA_VERSION,
            "component_register_fingerprint": component_register[
                "register_fingerprint"
            ].item(0),
            "split_policy_fingerprint": policy.fingerprint,
            "component_assignments": sorted(expected_assignments.items()),
        }
    )
    if set(manifest["split_fingerprint"].to_list()) != {expected_fingerprint}:
        raise ValueError("dynamic-pool evaluation split fingerprint mismatch")


def _split_components(register: pl.DataFrame) -> tuple[_SplitComponent, ...]:
    summaries = []
    for component_id in register["independence_component_id"].unique().to_list():
        rows = register.filter(pl.col("independence_component_id") == component_id)
        outcomes = Counter(
            "supported" if value else "error"
            for value in rows["human_supported"].to_list()
        )
        summaries.append(
            _SplitComponent(
                component_id=str(component_id),
                item_count=rows.height,
                outcome_item_counts=tuple(sorted(outcomes.items())),
            )
        )
    return tuple(sorted(summaries, key=lambda item: item.component_id))


def _assign_split_components(
    components: Sequence[_SplitComponent],
    *,
    policy: DynamicPoolEvaluationSplitPolicy,
) -> dict[str, str]:
    if len(components) < len(DYNAMIC_POOL_EVALUATION_SPLITS):
        raise ValueError(
            "reviewed evidence requires at least three independent components"
        )
    components_by_outcome = Counter()
    items_by_outcome = Counter()
    for component in components:
        for outcome, count in component.outcome_item_counts:
            components_by_outcome[outcome] += 1
            items_by_outcome[outcome] += count
    if policy.require_outcome_coverage:
        insufficient = {
            outcome: count
            for outcome, count in sorted(components_by_outcome.items())
            if count < len(DYNAMIC_POOL_EVALUATION_SPLITS)
        }
        if set(components_by_outcome) != {"error", "supported"}:
            insufficient["both_outcomes_required"] = 0
        if insufficient:
            raise ValueError(
                "independent outcome components cannot cover every partition: "
                f"{sorted(insufficient.items())}"
            )
    ordered = sorted(
        components,
        key=lambda component: (
            min(components_by_outcome[outcome] for outcome in component.outcomes),
            -component.item_count,
            canonical_semantic_fingerprint(
                {
                    "random_seed": policy.random_seed,
                    "component_id": component.component_id,
                }
            ),
        ),
    )
    remaining_by_outcome = Counter(components_by_outcome)
    uncovered_by_outcome = {
        outcome: set(DYNAMIC_POOL_EVALUATION_SPLITS)
        for outcome in components_by_outcome
    }
    uncovered_global = set(DYNAMIC_POOL_EVALUATION_SPLITS)
    assigned_items = Counter({split: 0 for split in DYNAMIC_POOL_EVALUATION_SPLITS})
    assigned_outcomes = {split: Counter() for split in DYNAMIC_POOL_EVALUATION_SPLITS}
    assignments: dict[str, str] = {}
    total_items = sum(component.item_count for component in components)
    for index, component in enumerate(ordered):
        candidates = []
        remaining_components_after = len(ordered) - index - 1
        for split in DYNAMIC_POOL_EVALUATION_SPLITS:
            global_after = uncovered_global - {split}
            if len(global_after) > remaining_components_after:
                continue
            feasible = True
            for outcome in components_by_outcome:
                remaining_after = remaining_by_outcome[outcome] - (
                    outcome in component.outcomes
                )
                uncovered_after = uncovered_by_outcome[outcome] - (
                    {split} if outcome in component.outcomes else set()
                )
                if (
                    policy.require_outcome_coverage
                    and len(uncovered_after) > remaining_after
                ):
                    feasible = False
                    break
            if not feasible:
                continue
            coverage_gain = int(split in uncovered_global) + sum(
                split in uncovered_by_outcome[outcome] for outcome in component.outcomes
            )
            target = total_items * dict(policy.weights)[split] / policy.total_weight
            projected_item_fill = (
                assigned_items[split] + component.item_count
            ) / target
            projected_outcome_fill = 0.0
            for outcome, count in component.outcome_item_counts:
                outcome_target = (
                    items_by_outcome[outcome]
                    * dict(policy.weights)[split]
                    / policy.total_weight
                )
                projected_outcome_fill += (
                    assigned_outcomes[split][outcome] + count
                ) / outcome_target
            tie = canonical_semantic_fingerprint(
                {
                    "random_seed": policy.random_seed,
                    "component_id": component.component_id,
                    "split": split,
                }
            )
            candidates.append(
                (
                    (
                        -coverage_gain,
                        projected_item_fill,
                        projected_outcome_fill,
                        tie,
                    ),
                    split,
                )
            )
        if not candidates:
            raise ValueError("unable to allocate components with required coverage")
        _, selected = min(candidates, key=lambda item: item[0])
        assignments[component.component_id] = selected
        assigned_items[selected] += component.item_count
        uncovered_global.discard(selected)
        for outcome, count in component.outcome_item_counts:
            assigned_outcomes[selected][outcome] += count
            uncovered_by_outcome[outcome].discard(selected)
            remaining_by_outcome[outcome] -= 1
    if uncovered_global or (
        policy.require_outcome_coverage
        and any(uncovered for uncovered in uncovered_by_outcome.values())
    ):
        raise ValueError("component allocation did not cover required partitions")
    return assignments


def _single_value(frame: pl.DataFrame, field: str) -> object:
    values = frame[field].unique().to_list()
    if len(values) != 1:
        raise ValueError(f"{field} must have exactly one value")
    return values[0]


def _component_members(
    items: Sequence[ReviewedFlickrSplitItem],
) -> tuple[tuple[ReviewedFlickrSplitItem, ...], ...]:
    disjoint = _DisjointSet([item.item_id for item in items])
    first_by_identity: dict[tuple[str, str], str] = {}
    for item in items:
        for identity in _independence_identities(item):
            previous = first_by_identity.setdefault(identity, item.item_id)
            disjoint.union(previous, item.item_id)
    members: dict[str, list[ReviewedFlickrSplitItem]] = defaultdict(list)
    for item in items:
        members[disjoint.find(item.item_id)].append(item)
    components = [
        tuple(sorted(values, key=lambda item: item.item_id))
        for values in members.values()
    ]
    return tuple(sorted(components, key=lambda values: values[0].item_id))


def _independence_identities(
    item: ReviewedFlickrSplitItem,
) -> tuple[tuple[str, str], ...]:
    identities = [
        ("flickr_photo", item.flickr_photo_id),
        ("owner", item.owner_group_id),
        ("duplicate", item.duplicate_group_id),
        ("observation", item.observation_group_id),
        ("source_mirror", item.source_mirror_group_id),
    ]
    if not item.no_geo:
        assert item.geographic_cluster_id is not None
        identities.append(("geographic_cluster", item.geographic_cluster_id))
    return tuple(identities)


def _component_id(items: Sequence[ReviewedFlickrSplitItem]) -> str:
    fingerprint = canonical_semantic_fingerprint(
        {
            "schema_version": REVIEWED_FLICKR_COMPONENT_SCHEMA_VERSION,
            "member_item_fingerprints": sorted(item.fingerprint for item in items),
        }
    )
    return f"reviewed-flickr-component:{fingerprint.removeprefix('sha256:')}"


def _item_values(item: ReviewedFlickrSplitItem) -> dict[str, object]:
    return {
        "item_id": item.item_id,
        "source_record_hash": item.source_record_hash,
        "source_artifact_fingerprint": item.source_artifact_fingerprint,
        "review_decision_fingerprint": item.review_decision_fingerprint,
        "flickr_photo_id": item.flickr_photo_id,
        "owner_group_id": item.owner_group_id,
        "duplicate_group_id": item.duplicate_group_id,
        "observation_group_id": item.observation_group_id,
        "geographic_cluster_id": item.geographic_cluster_id,
        "no_geo": item.no_geo,
        "source_mirror_group_id": item.source_mirror_group_id,
        "stratum_id": item.stratum_id,
        "candidate_species_key": item.candidate_species_key,
        "human_supported": item.human_supported,
        "sampling_weight": item.sampling_weight,
    }


def _item_from_row(row: Mapping[str, object]) -> ReviewedFlickrSplitItem:
    return ReviewedFlickrSplitItem(
        **{field: row[field] for field in ReviewedFlickrSplitItem.__dataclass_fields__}
    )


def _required_text(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be nonempty text")
    return value.strip()


def _optional_text(value: object, *, field: str) -> str | None:
    if value is None:
        return None
    return _required_text(value, field=field)


def _sha256(value: object, *, field: str) -> str:
    text = _required_text(value, field=field).casefold()
    if not text.startswith("sha256:") or len(text) != 71:
        raise ValueError(f"{field} must be a sha256 fingerprint")
    try:
        int(text.removeprefix("sha256:"), 16)
    except ValueError as exc:
        raise ValueError(f"{field} must be a sha256 fingerprint") from exc
    return text


def _positive_float(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field} must be numeric")
    normalized = float(value)
    if not math.isfinite(normalized) or normalized <= 0.0:
        raise ValueError(f"{field} must be finite and positive")
    return normalized


__all__ = [
    "DYNAMIC_POOL_EVALUATION_SPLIT_FILE",
    "DYNAMIC_POOL_EVALUATION_SPLIT_SCHEMA",
    "DYNAMIC_POOL_EVALUATION_SPLIT_SCHEMA_VERSION",
    "DYNAMIC_POOL_EVALUATION_SPLITS",
    "DYNAMIC_POOL_SPLIT_ASSIGNMENT_POLICY_VERSION",
    "REVIEWED_FLICKR_COMPONENT_FILE",
    "REVIEWED_FLICKR_COMPONENT_SCHEMA",
    "REVIEWED_FLICKR_COMPONENT_SCHEMA_VERSION",
    "ReviewedFlickrComponentBuild",
    "ReviewedFlickrSplitItem",
    "DynamicPoolEvaluationSplitBuild",
    "DynamicPoolEvaluationSplitPolicy",
    "build_dynamic_pool_evaluation_splits",
    "build_reviewed_flickr_components",
    "validate_reviewed_flickr_components",
    "validate_dynamic_pool_evaluation_splits",
]
