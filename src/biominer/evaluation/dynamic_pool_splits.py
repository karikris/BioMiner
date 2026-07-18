"""Leakage-safe reviewed Flickr components and frozen evaluation splits."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math

import polars as pl

from biominer.common.semantic_hash import canonical_semantic_fingerprint


REVIEWED_FLICKR_COMPONENT_SCHEMA_VERSION = (
    "reviewed-flickr-independence-component-v1.0.0"
)
REVIEWED_FLICKR_COMPONENT_FILE = "reviewed_flickr_independence_components.parquet"

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
    "REVIEWED_FLICKR_COMPONENT_FILE",
    "REVIEWED_FLICKR_COMPONENT_SCHEMA",
    "REVIEWED_FLICKR_COMPONENT_SCHEMA_VERSION",
    "ReviewedFlickrComponentBuild",
    "ReviewedFlickrSplitItem",
    "build_reviewed_flickr_components",
    "validate_reviewed_flickr_components",
]
