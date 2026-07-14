from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
from typing import Any
from uuid import uuid4

from PIL import ExifTags, Image
import polars as pl

from biominer.references.licensing import (
    canonicalise_creative_commons_licence,
    canonicalise_creative_commons_licence_identity,
)
from biominer.references.schemas import (
    DUPLICATE_RELATIONSHIP_TYPES,
    REFERENCE_MEDIA_DUPLICATE_RELATIONSHIPS_SCHEMA_VERSION,
    reference_media_candidates_frame,
    reference_media_duplicate_relationships_frame,
    reference_media_objects_frame,
    reference_observations_frame,
    validate_reference_media_candidates,
    validate_reference_media_duplicate_relationships,
    validate_reference_media_objects,
    validate_reference_observations,
    write_reference_media_duplicate_relationships,
    write_reference_media_objects,
)
from biominer.storage.cloud import CloudStorage
from biominer.storage.paths import build_report_uri, safe_path_component
from biominer.storage.uri import join_uri


REFERENCE_MEDIA_DEDUPLICATOR_VERSION = "reference-media-deduplicator-v1"
REFERENCE_MEDIA_DEDUPLICATION_REPORT_VERSION = "reference-media-deduplication-report-v1"
REFERENCE_MEDIA_DEDUPLICATION_REPORT_FILE = "reference_media_deduplication_report.json"
REFERENCE_MEDIA_DEDUPLICATION_SUMMARY_FILE = "reference_media_deduplication_summary.md"
REFERENCE_PERCEPTUAL_HASH_VERSION = "dhash128-v1"

LOGGER = logging.getLogger(__name__)

_PERCEPTUAL_HASH_PATTERN = re.compile(r"dhash128-v1:([0-9a-f]{32})\Z")
_INATURALIST_OBSERVATION_PATTERN = re.compile(
    r"(?:inaturalist(?:\.org)?/observations/|observations/)([0-9]+)(?:\D|\Z)",
    re.IGNORECASE,
)
_INATURALIST_PHOTO_PATTERN = re.compile(
    r"(?:inaturalist[^/]*/photos/|/photos/)([0-9]+)(?:/|\D|\Z)",
    re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class ReferenceMediaDeduplicationConfig:
    same_observation_distance_threshold: int = 8
    cross_observation_distance_threshold: int = 4
    max_aspect_ratio_delta: float = 0.05
    minimum_informative_bits: int = 8
    max_perceptual_hash_neighbors: int = 64
    max_perceptual_search_nodes: int = 4_096
    policy_version: str = "reference-media-deduplication-policy-v1"
    source_priority: tuple[str, ...] = ("iNaturalist", "GBIF")

    def __post_init__(self) -> None:
        for field_name in (
            "same_observation_distance_threshold",
            "cross_observation_distance_threshold",
            "minimum_informative_bits",
            "max_perceptual_hash_neighbors",
            "max_perceptual_search_nodes",
        ):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{field_name} must be an integer")
            if value < 0:
                raise ValueError(f"{field_name} must be nonnegative")
        if self.same_observation_distance_threshold > 128:
            raise ValueError("same-observation distance threshold cannot exceed 128")
        if self.cross_observation_distance_threshold > 128:
            raise ValueError("cross-observation distance threshold cannot exceed 128")
        if (
            self.cross_observation_distance_threshold
            > self.same_observation_distance_threshold
        ):
            raise ValueError(
                "cross-observation distance threshold cannot exceed the "
                "same-observation threshold"
            )
        if not 0 <= self.minimum_informative_bits <= 64:
            raise ValueError("minimum_informative_bits must be between 0 and 64")
        if self.max_perceptual_hash_neighbors == 0:
            raise ValueError("max_perceptual_hash_neighbors must be positive")
        if self.max_perceptual_search_nodes == 0:
            raise ValueError("max_perceptual_search_nodes must be positive")
        ratio_delta = float(self.max_aspect_ratio_delta)
        if not math.isfinite(ratio_delta) or not 0.0 <= ratio_delta <= 1.0:
            raise ValueError("max_aspect_ratio_delta must be between 0 and 1")
        object.__setattr__(self, "max_aspect_ratio_delta", ratio_delta)
        policy_version = str(self.policy_version or "").strip()
        if not policy_version:
            raise ValueError("policy_version must be nonblank")
        object.__setattr__(self, "policy_version", policy_version)
        if not isinstance(self.source_priority, tuple) or not all(
            isinstance(value, str) for value in self.source_priority
        ):
            raise TypeError("source_priority must be a tuple of strings")
        priorities = tuple(value.strip() for value in self.source_priority)
        if any(not value for value in priorities):
            raise ValueError("source_priority values must be nonblank")
        if len({value.casefold() for value in priorities}) != len(priorities):
            raise ValueError("source_priority values must be unique")
        object.__setattr__(self, "source_priority", priorities)

    @property
    def payload(self) -> dict[str, object]:
        return {
            "deduplicator_version": REFERENCE_MEDIA_DEDUPLICATOR_VERSION,
            "perceptual_hash_version": REFERENCE_PERCEPTUAL_HASH_VERSION,
            "policy_version": self.policy_version,
            "same_observation_distance_threshold": (
                self.same_observation_distance_threshold
            ),
            "cross_observation_distance_threshold": (
                self.cross_observation_distance_threshold
            ),
            "max_aspect_ratio_delta": self.max_aspect_ratio_delta,
            "minimum_informative_bits": self.minimum_informative_bits,
            "max_perceptual_hash_neighbors": self.max_perceptual_hash_neighbors,
            "max_perceptual_search_nodes": self.max_perceptual_search_nodes,
            "source_priority": list(self.source_priority),
        }

    @property
    def fingerprint(self) -> str:
        return _fingerprint(self.payload)


def _config_from_report_settings(
    settings: Mapping[str, object],
) -> ReferenceMediaDeduplicationConfig:
    try:
        config = ReferenceMediaDeduplicationConfig(
            same_observation_distance_threshold=settings[
                "same_observation_distance_threshold"
            ],
            cross_observation_distance_threshold=settings[
                "cross_observation_distance_threshold"
            ],
            max_aspect_ratio_delta=settings["max_aspect_ratio_delta"],
            minimum_informative_bits=settings["minimum_informative_bits"],
            max_perceptual_hash_neighbors=settings["max_perceptual_hash_neighbors"],
            max_perceptual_search_nodes=settings["max_perceptual_search_nodes"],
            policy_version=settings["policy_version"],
            source_priority=tuple(settings["source_priority"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("reference media deduplication settings are invalid") from exc
    if dict(settings) != config.payload:
        raise ValueError("reference media deduplication settings are incompatible")
    return config


@dataclass(frozen=True, slots=True)
class ReferenceMediaDeduplicationResult:
    media_objects: pl.DataFrame
    relationships: pl.DataFrame
    media_candidates: pl.DataFrame
    observations: pl.DataFrame
    report: dict[str, Any]
    markdown: str


@dataclass(slots=True)
class _RelationshipEvidence:
    left_id: str
    right_id: str
    evidence_types: set[str]
    sha256_equal: bool = False
    perceptual_hash_distance: int | None = None
    same_observation: bool = False
    provider_mirror: bool = False
    relationship_type: str = "perceptual_candidate"
    resolution_status: str = "resolved"

    def merge(
        self,
        *,
        evidence_types: Iterable[str] = (),
        sha256_equal: bool = False,
        perceptual_hash_distance: int | None = None,
        same_observation: bool = False,
        provider_mirror: bool = False,
        relationship_type: str | None = None,
        resolution_status: str | None = None,
    ) -> None:
        self.evidence_types.update(evidence_types)
        self.sha256_equal = self.sha256_equal or sha256_equal
        self.same_observation = self.same_observation or same_observation
        self.provider_mirror = self.provider_mirror or provider_mirror
        if perceptual_hash_distance is not None:
            if self.perceptual_hash_distance is None:
                self.perceptual_hash_distance = perceptual_hash_distance
            elif self.perceptual_hash_distance != perceptual_hash_distance:
                raise ValueError("one media pair has inconsistent perceptual distances")
        if relationship_type is not None:
            self.relationship_type = _stronger_relationship_type(
                self.relationship_type,
                relationship_type,
            )
        if resolution_status is not None:
            self.resolution_status = _stronger_resolution_status(
                self.resolution_status,
                resolution_status,
            )


class _DisjointSet:
    def __init__(self, values: Iterable[str]) -> None:
        self.parent = {value: value for value in values}

    def add(self, value: str) -> None:
        self.parent.setdefault(value, value)

    def find(self, value: str) -> str:
        parent = self.parent[value]
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(self, left: str, right: str) -> None:
        self.add(left)
        self.add(right)
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        if left_root < right_root:
            self.parent[right_root] = left_root
        else:
            self.parent[left_root] = right_root


@dataclass(slots=True)
class _BKNode:
    value: int
    children: dict[int, _BKNode]


class _HammingBKTree:
    def __init__(self) -> None:
        self.root: _BKNode | None = None

    def add(self, value: int) -> None:
        if self.root is None:
            self.root = _BKNode(value=value, children={})
            return
        node = self.root
        while True:
            distance = (value ^ node.value).bit_count()
            if distance == 0:
                return
            child = node.children.get(distance)
            if child is None:
                node.children[distance] = _BKNode(value=value, children={})
                return
            node = child

    def query(
        self,
        value: int,
        radius: int,
        *,
        max_results: int,
        max_visited: int,
    ) -> list[int]:
        if self.root is None:
            return []
        matches: list[int] = []
        pending = [self.root]
        visited = 0
        while pending:
            node = pending.pop()
            visited += 1
            if visited > max_visited:
                raise ValueError(
                    "perceptual hash search exceeds the configured node limit"
                )
            distance = (value ^ node.value).bit_count()
            if distance <= radius:
                matches.append(node.value)
                if len(matches) > max_results:
                    raise ValueError(
                        "perceptual hash neighborhood exceeds the configured limit"
                    )
            minimum = distance - radius
            maximum = distance + radius
            pending.extend(
                child
                for edge_distance, child in node.children.items()
                if minimum <= edge_distance <= maximum
            )
        return sorted(matches)


def compute_reference_perceptual_hash(image: Image.Image) -> str:
    if not isinstance(image, Image.Image):
        raise TypeError("image must be a Pillow Image")
    orientation = image.getexif().get(ExifTags.Base.Orientation, 1)
    resample_source = image
    converted: Image.Image | None = None
    if image.mode == "P":
        converted = image.convert("RGBA" if "transparency" in image.info else "RGB")
        resample_source = converted
    elif image.mode == "1":
        converted = image.convert("L")
        resample_source = converted
    try:
        resized = resample_source.resize((9, 9), Image.Resampling.LANCZOS)
    finally:
        if converted is not None:
            converted.close()
    transpose = {
        2: Image.Transpose.FLIP_LEFT_RIGHT,
        3: Image.Transpose.ROTATE_180,
        4: Image.Transpose.FLIP_TOP_BOTTOM,
        5: Image.Transpose.TRANSPOSE,
        6: Image.Transpose.ROTATE_270,
        7: Image.Transpose.TRANSVERSE,
        8: Image.Transpose.ROTATE_90,
    }.get(orientation)
    oriented = resized.transpose(transpose) if transpose is not None else resized
    if oriented is not resized:
        resized.close()
    rgba: Image.Image | None = None
    background: Image.Image | None = None
    grayscale: Image.Image | None = None
    try:
        if oriented.mode in {"RGBA", "LA"} or (
            oriented.mode == "P" and "transparency" in oriented.info
        ):
            rgba = oriented.convert("RGBA")
            background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
            background.alpha_composite(rgba)
            grayscale = background.convert("L")
        else:
            grayscale = oriented.convert("L")
        values = tuple(int(value) for value in grayscale.get_flattened_data())
    finally:
        if grayscale is not None:
            grayscale.close()
        if background is not None:
            background.close()
        if rgba is not None:
            rgba.close()
        oriented.close()
    row_hash = 0
    column_hash = 0
    for y in range(8):
        for x in range(8):
            offset = y * 9 + x
            row_hash = (row_hash << 1) | int(values[offset] < values[offset + 1])
            column_hash = (column_hash << 1) | int(values[offset] < values[offset + 9])
    combined = (row_hash << 64) | column_hash
    return f"{REFERENCE_PERCEPTUAL_HASH_VERSION}:{combined:032x}"


def perceptual_hash_distance(left: str, right: str) -> int:
    return (_parse_perceptual_hash(left) ^ _parse_perceptual_hash(right)).bit_count()


def deduplicate_reference_media(
    media_objects: pl.DataFrame,
    media_candidates: pl.DataFrame,
    observations: pl.DataFrame,
    *,
    config: ReferenceMediaDeduplicationConfig | None = None,
    generated_at: str | datetime | None = None,
) -> ReferenceMediaDeduplicationResult:
    supplied_generated_at = (
        generated_at if generated_at is not None else datetime.now(UTC)
    )
    event_context = {
        "command": "references.deduplicate_media",
        "generated_at": (
            supplied_generated_at.isoformat()
            if isinstance(supplied_generated_at, datetime)
            else str(supplied_generated_at)
        ),
        "media_object_rows": getattr(media_objects, "height", None),
        "media_candidate_rows": getattr(media_candidates, "height", None),
        "observation_rows": getattr(observations, "height", None),
    }
    _log_event("reference_media_deduplication_started", **event_context)
    try:
        timestamp = _utc_datetime(supplied_generated_at, field="generated_at")
        result = _deduplicate_reference_media_impl(
            media_objects,
            media_candidates,
            observations,
            config=config,
            generated_at=timestamp,
        )
    except Exception as exc:
        _log_event(
            "reference_media_deduplication_failed",
            **event_context,
            error_type=type(exc).__name__,
            error=str(exc),
        )
        raise
    _log_event(
        "reference_media_deduplication_completed",
        **event_context,
        counts=result.report["counts"],
        outputs=result.report["outputs"],
    )
    return result


def _deduplicate_reference_media_impl(
    media_objects: pl.DataFrame,
    media_candidates: pl.DataFrame,
    observations: pl.DataFrame,
    *,
    config: ReferenceMediaDeduplicationConfig | None = None,
    generated_at: str | datetime | None = None,
) -> ReferenceMediaDeduplicationResult:
    effective_config = config or ReferenceMediaDeduplicationConfig()
    if not isinstance(effective_config, ReferenceMediaDeduplicationConfig):
        raise TypeError("config must be a ReferenceMediaDeduplicationConfig")
    media_objects = reference_media_objects_frame(media_objects.to_dicts())
    media_candidates = reference_media_candidates_frame(media_candidates.to_dicts())
    observations = reference_observations_frame(observations.to_dicts())
    timestamp = _utc_datetime(generated_at or datetime.now(UTC), field="generated_at")

    object_rows = {
        str(row["reference_media_id"]): dict(row)
        for row in media_objects.iter_rows(named=True)
    }
    candidate_rows = {
        str(row["reference_media_id"]): dict(row)
        for row in media_candidates.iter_rows(named=True)
    }
    observation_rows = {
        str(row["reference_observation_id"]): dict(row)
        for row in observations.iter_rows(named=True)
    }
    missing_candidates = sorted(set(object_rows) - set(candidate_rows))
    if missing_candidates:
        raise ValueError(
            f"reference media objects lack candidate provenance: {missing_candidates}"
        )
    missing_observations = sorted(
        {str(row["reference_observation_id"]) for row in candidate_rows.values()}
        - set(observation_rows)
    )
    if missing_observations:
        raise ValueError(
            "reference media candidates lack observation provenance: "
            f"{missing_observations}"
        )

    valid_rows = {
        media_id: row
        for media_id, row in object_rows.items()
        if row["decode_status"] == "valid"
    }
    for media_id, row in valid_rows.items():
        _parse_perceptual_hash(str(row["perceptual_hash"]))
        if str(row["sha256"]) != str(row["sha256"]).casefold():
            raise ValueError(f"reference media {media_id} has a noncanonical SHA-256")

    edges: dict[tuple[str, str], _RelationshipEvidence] = {}
    provider_pairs = _provider_mirror_pairs(candidate_rows, observation_rows)
    for left_id, right_id in provider_pairs:
        if left_id not in valid_rows and right_id not in valid_rows:
            continue
        edge = _edge(edges, left_id, right_id)
        edge.merge(
            evidence_types={"provider_identifier"},
            provider_mirror=True,
            relationship_type="provider_mirror",
            resolution_status="resolved",
        )

    by_sha256: dict[str, list[str]] = defaultdict(list)
    for media_id, row in valid_rows.items():
        by_sha256[str(row["sha256"])].append(media_id)
    for media_ids in by_sha256.values():
        ordered = sorted(media_ids)
        if not ordered:
            continue
        anchor_id = ordered[0]
        for right_id in ordered[1:]:
            edge = _edge(edges, anchor_id, right_id)
            edge.merge(
                evidence_types={"exact_sha256"},
                sha256_equal=True,
                relationship_type="exact",
                resolution_status="resolved",
            )

    for left_id, right_id in _perceptual_candidate_pairs(
        valid_rows,
        candidate_rows,
        config=effective_config,
    ):
        left_row = valid_rows[left_id]
        right_row = valid_rows[right_id]
        left_candidate = candidate_rows[left_id]
        right_candidate = candidate_rows[right_id]
        same_observation = (
            left_candidate["reference_observation_id"]
            == right_candidate["reference_observation_id"]
        )
        threshold = (
            effective_config.same_observation_distance_threshold
            if same_observation
            else effective_config.cross_observation_distance_threshold
        )
        distance = perceptual_hash_distance(
            str(left_row["perceptual_hash"]),
            str(right_row["perceptual_hash"]),
        )
        provider_mirror = (left_id, right_id) in provider_pairs
        if distance > threshold and not provider_mirror:
            continue
        aspect_compatible = _aspect_ratios_compatible(
            left_row,
            right_row,
            maximum_delta=effective_config.max_aspect_ratio_delta,
        )
        informative = _perceptual_hash_is_informative(
            str(left_row["perceptual_hash"]),
            minimum_bits=effective_config.minimum_informative_bits,
        ) and _perceptual_hash_is_informative(
            str(right_row["perceptual_hash"]),
            minimum_bits=effective_config.minimum_informative_bits,
        )
        edge = _edge(edges, left_id, right_id)
        evidence = {"perceptual_hash"}
        if same_observation:
            evidence.add("same_observation")
        if edge.sha256_equal:
            relationship_type = "exact"
            resolution_status = "resolved"
        elif (
            provider_mirror
            and distance <= threshold
            and aspect_compatible
            and informative
        ):
            relationship_type = "provider_mirror"
            resolution_status = "resolved"
        elif provider_mirror:
            relationship_type = "provider_mirror"
            resolution_status = "conflict"
        elif same_observation and aspect_compatible and informative:
            dimensions_differ = (
                left_row["decoded_width"],
                left_row["decoded_height"],
            ) != (
                right_row["decoded_width"],
                right_row["decoded_height"],
            )
            relationship_type = (
                "resized_copy" if dimensions_differ else "near_identical_burst"
            )
            resolution_status = "resolved"
        else:
            relationship_type = "perceptual_candidate"
            resolution_status = "review_required"
        edge.merge(
            evidence_types=evidence,
            perceptual_hash_distance=distance,
            same_observation=same_observation,
            relationship_type=relationship_type,
            resolution_status=resolution_status,
        )

    for edge in edges.values():
        if edge.left_id not in valid_rows or edge.right_id not in valid_rows:
            continue
        left_row = valid_rows[edge.left_id]
        right_row = valid_rows[edge.right_id]
        distance = perceptual_hash_distance(
            str(left_row["perceptual_hash"]),
            str(right_row["perceptual_hash"]),
        )
        edge.merge(
            evidence_types={"perceptual_hash"},
            perceptual_hash_distance=distance,
        )
        if left_row["sha256"] == right_row["sha256"]:
            edge.merge(
                evidence_types={"exact_sha256"},
                sha256_equal=True,
                relationship_type="exact",
                resolution_status="resolved",
            )

    for left_id, right_id in provider_pairs:
        edge = edges.get((left_id, right_id))
        if edge is None:
            continue
        if left_id in valid_rows and right_id in valid_rows:
            distance = perceptual_hash_distance(
                str(valid_rows[left_id]["perceptual_hash"]),
                str(valid_rows[right_id]["perceptual_hash"]),
            )
            edge.merge(
                evidence_types={"perceptual_hash"},
                perceptual_hash_distance=distance,
            )
            same_observation = (
                candidate_rows[left_id]["reference_observation_id"]
                == candidate_rows[right_id]["reference_observation_id"]
            )
            threshold = (
                effective_config.same_observation_distance_threshold
                if same_observation
                else effective_config.cross_observation_distance_threshold
            )
            informative = _perceptual_hash_is_informative(
                str(valid_rows[left_id]["perceptual_hash"]),
                minimum_bits=effective_config.minimum_informative_bits,
            ) and _perceptual_hash_is_informative(
                str(valid_rows[right_id]["perceptual_hash"]),
                minimum_bits=effective_config.minimum_informative_bits,
            )
            if not edge.sha256_equal and (
                distance > threshold
                or not informative
                or not _aspect_ratios_compatible(
                    valid_rows[left_id],
                    valid_rows[right_id],
                    maximum_delta=effective_config.max_aspect_ratio_delta,
                )
            ):
                edge.merge(resolution_status="conflict")

    for edge in edges.values():
        left_observation_id = candidate_rows[edge.left_id]["reference_observation_id"]
        right_observation_id = candidate_rows[edge.right_id]["reference_observation_id"]
        if left_observation_id == right_observation_id:
            edge.merge(
                evidence_types={"same_observation"},
                same_observation=True,
            )
        if _metadata_conflict(
            candidate_rows[edge.left_id],
            candidate_rows[edge.right_id],
            observation_rows,
        ):
            edge.merge(
                evidence_types={"metadata_conflict"},
                resolution_status="conflict",
            )

    disjoint = _DisjointSet(valid_rows)
    for edge in edges.values():
        disjoint.union(edge.left_id, edge.right_id)
    component_members: dict[str, set[str]] = defaultdict(set)
    for media_id in disjoint.parent:
        component_members[disjoint.find(media_id)].add(media_id)
    components = sorted(
        (sorted(values) for values in component_members.values()),
        key=lambda values: values[0],
    )

    edge_by_component: dict[str, list[_RelationshipEvidence]] = defaultdict(list)
    component_key_by_media: dict[str, str] = {}
    for members in components:
        key = members[0]
        for media_id in members:
            component_key_by_media[media_id] = key
    for edge in edges.values():
        key = component_key_by_media[edge.left_id]
        edge_by_component[key].append(edge)

    for members in components:
        key = members[0]
        if _component_metadata_conflict(
            members,
            candidate_rows=candidate_rows,
            observation_rows=observation_rows,
        ):
            for edge in edge_by_component.get(key, []):
                edge.merge(
                    evidence_types={"component_metadata_conflict"},
                    resolution_status="conflict",
                )

    group_context: dict[str, dict[str, object]] = {}
    for members in components:
        key = members[0]
        valid_members = sorted(
            media_id for media_id in members if media_id in valid_rows
        )
        if not valid_members:
            continue
        canonical_id = min(
            valid_members,
            key=lambda media_id: _canonical_sort_key(
                media_id,
                object_row=valid_rows[media_id],
                candidate_row=candidate_rows[media_id],
                observation_row=observation_rows[
                    str(candidate_rows[media_id]["reference_observation_id"])
                ],
                config=effective_config,
            ),
        )
        group_edges = sorted(
            edge_by_component.get(key, []),
            key=lambda edge: (edge.left_id, edge.right_id),
        )
        distance_upper_bound = _component_perceptual_distance_upper_bound(
            valid_members,
            valid_rows,
        )
        group_type = _component_duplicate_type(
            valid_members,
            group_edges,
            distance_upper_bound=distance_upper_bound,
            config=effective_config,
        )
        group_id = _duplicate_group_id(members)
        group_context[key] = {
            "members": members,
            "valid_members": valid_members,
            "canonical_id": canonical_id,
            "duplicate_group_id": group_id,
            "duplicate_type": group_type,
            "perceptual_distance_upper_bound": distance_upper_bound,
        }

    provider_peers: dict[str, set[str]] = defaultdict(set)
    for edge in edges.values():
        if edge.provider_mirror:
            provider_peers[edge.left_id].add(edge.right_id)
            provider_peers[edge.right_id].add(edge.left_id)

    annotated_rows: list[dict[str, object]] = []
    for media_id, source_row in object_rows.items():
        row = dict(source_row)
        if row["decode_status"] != "valid":
            row["perceptual_hash"] = None
            row["duplicate_group_id"] = None
            row["duplicate_type"] = None
            row["canonical_reference_media_id"] = None
            row["provider_mirror_ids"] = []
        else:
            context = group_context[component_key_by_media[media_id]]
            row["duplicate_group_id"] = context["duplicate_group_id"]
            row["duplicate_type"] = context["duplicate_type"]
            row["canonical_reference_media_id"] = context["canonical_id"]
            row["provider_mirror_ids"] = sorted(provider_peers.get(media_id, set()))
        annotated_rows.append(row)
    annotated = reference_media_objects_frame(annotated_rows)

    relationship_rows: list[dict[str, object]] = []
    for edge in sorted(
        edges.values(), key=lambda value: (value.left_id, value.right_id)
    ):
        key = component_key_by_media[edge.left_id]
        context = group_context[key]
        left_candidate = candidate_rows[edge.left_id]
        right_candidate = candidate_rows[edge.right_id]
        evidence_types = sorted(edge.evidence_types)
        relationship_rows.append(
            {
                "schema_version": (
                    REFERENCE_MEDIA_DUPLICATE_RELATIONSHIPS_SCHEMA_VERSION
                ),
                "duplicate_relationship_id": _duplicate_relationship_id(
                    edge.left_id,
                    edge.right_id,
                    evidence_types=evidence_types,
                ),
                "duplicate_group_id": context["duplicate_group_id"],
                "canonical_reference_media_id": context["canonical_id"],
                "left_reference_media_id": edge.left_id,
                "right_reference_media_id": edge.right_id,
                "left_reference_observation_id": left_candidate[
                    "reference_observation_id"
                ],
                "right_reference_observation_id": right_candidate[
                    "reference_observation_id"
                ],
                "left_source": left_candidate["source"],
                "right_source": right_candidate["source"],
                "left_provider_media_id": left_candidate["provider_media_id"],
                "right_provider_media_id": right_candidate["provider_media_id"],
                "relationship_type": edge.relationship_type,
                "evidence_types": evidence_types,
                "sha256_equal": edge.sha256_equal,
                "perceptual_hash_distance": edge.perceptual_hash_distance,
                "same_observation": edge.same_observation,
                "provider_mirror": edge.provider_mirror,
                "resolution_status": edge.resolution_status,
                "policy_version": effective_config.policy_version,
                "policy_fingerprint": effective_config.fingerprint,
            }
        )
    relationships = reference_media_duplicate_relationships_frame(relationship_rows)
    _validate_deduplication_graph(
        annotated,
        relationships,
        media_candidates=media_candidates,
        observations=observations,
        config=effective_config,
    )

    report = _deduplication_report(
        media_objects=annotated,
        relationships=relationships,
        media_candidates=media_candidates,
        observations=observations,
        group_context=group_context,
        generated_at=timestamp,
        config=effective_config,
    )
    return ReferenceMediaDeduplicationResult(
        media_objects=annotated,
        relationships=relationships,
        media_candidates=media_candidates,
        observations=observations,
        report=report,
        markdown=_deduplication_markdown(report),
    )


def validate_reference_media_deduplication_result(
    result: ReferenceMediaDeduplicationResult,
) -> None:
    if not isinstance(result, ReferenceMediaDeduplicationResult):
        raise TypeError("result must be a ReferenceMediaDeduplicationResult")
    validate_reference_media_objects(result.media_objects)
    validate_reference_media_duplicate_relationships(result.relationships)
    validate_reference_media_candidates(result.media_candidates)
    validate_reference_observations(result.observations)
    if result.report.get("schema_version") != (
        REFERENCE_MEDIA_DEDUPLICATION_REPORT_VERSION
    ):
        raise ValueError("reference media deduplication report schema is incompatible")
    if result.report.get("command") != "references.deduplicate_media":
        raise ValueError("reference media deduplication report command is invalid")
    if result.report.get("status") != "complete":
        raise ValueError("reference media deduplication result is not complete")

    inputs = result.report.get("inputs")
    counts = result.report.get("counts")
    outputs = result.report.get("outputs")
    settings = result.report.get("settings")
    if (
        not isinstance(inputs, Mapping)
        or not isinstance(counts, Mapping)
        or not isinstance(outputs, Mapping)
        or not isinstance(settings, Mapping)
    ):
        raise ValueError("reference media deduplication report shape is invalid")
    config = _config_from_report_settings(settings)
    if result.report.get("policy_fingerprint") != config.fingerprint:
        raise ValueError("reference media deduplication policy fingerprint is invalid")
    for relationship in result.relationships.iter_rows(named=True):
        if (
            relationship["policy_version"] != config.policy_version
            or relationship["policy_fingerprint"] != config.fingerprint
        ):
            raise ValueError(
                "duplicate relationship policy conflicts with the run policy"
            )
    _validate_deduplication_graph(
        result.media_objects,
        result.relationships,
        media_candidates=result.media_candidates,
        observations=result.observations,
        config=config,
    )
    expected_inputs = {
        "media_object_rows": result.media_objects.height,
        "media_candidate_rows": result.media_candidates.height,
        "observation_rows": result.observations.height,
        "media_objects_fingerprint": _intrinsic_media_objects_fingerprint(
            result.media_objects
        ),
        "media_candidates_fingerprint": _frame_fingerprint(result.media_candidates),
        "observations_fingerprint": _frame_fingerprint(result.observations),
    }
    if any(inputs.get(key) != value for key, value in expected_inputs.items()):
        raise ValueError("reference media deduplication input fingerprint is invalid")
    valid_count = result.media_objects.filter(pl.col("decode_status") == "valid").height
    expected_counts = {
        "valid_media": valid_count,
        "invalid_media": result.media_objects.height - valid_count,
        "duplicate_groups": result.media_objects.filter(
            pl.col("duplicate_group_id").is_not_null()
        )["duplicate_group_id"].n_unique(),
        "canonical_media": result.media_objects.filter(
            pl.col("canonical_reference_media_id").is_not_null()
        )["canonical_reference_media_id"].n_unique(),
        "relationships": result.relationships.height,
        "provider_mirror_relationships": result.relationships.filter(
            pl.col("provider_mirror")
        ).height,
        "review_required_relationships": result.relationships.filter(
            pl.col("resolution_status") == "review_required"
        ).height,
        "conflicting_relationships": result.relationships.filter(
            pl.col("resolution_status") == "conflict"
        ).height,
    }
    if any(counts.get(key) != value for key, value in expected_counts.items()):
        raise ValueError("reference media deduplication report counts are inconsistent")
    expected_distributions = {
        "duplicate_type_counts": Counter(
            str(value)
            for value in result.media_objects.filter(
                pl.col("decode_status") == "valid"
            )["duplicate_type"].to_list()
        ),
        "relationship_type_counts": Counter(
            str(value) for value in result.relationships["relationship_type"].to_list()
        ),
        "resolution_status_counts": Counter(
            str(value) for value in result.relationships["resolution_status"].to_list()
        ),
    }
    if any(
        result.report.get(key) != dict(sorted(values.items()))
        for key, values in expected_distributions.items()
    ):
        raise ValueError(
            "reference media deduplication report distributions are inconsistent"
        )
    if outputs.get("media_objects_fingerprint") != _frame_fingerprint(
        result.media_objects
    ) or outputs.get("relationships_fingerprint") != _frame_fingerprint(
        result.relationships
    ):
        raise ValueError("reference media deduplication output fingerprint is invalid")
    if result.markdown != _deduplication_markdown(result.report):
        raise ValueError("reference media deduplication summary is inconsistent")

    object_rows = {
        str(row["reference_media_id"]): row
        for row in result.media_objects.iter_rows(named=True)
    }
    candidate_rows = {
        str(row["reference_media_id"]): row
        for row in result.media_candidates.iter_rows(named=True)
    }
    observation_rows = {
        str(row["reference_observation_id"]): row
        for row in result.observations.iter_rows(named=True)
    }
    provider_peers: dict[str, set[str]] = defaultdict(set)
    adjacency: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    component_nodes: dict[str, set[str]] = defaultdict(set)
    relationships_by_group: dict[str, list[dict[str, object]]] = defaultdict(list)
    endpoint_group: dict[str, str] = {}
    for relationship in result.relationships.iter_rows(named=True):
        group_id = str(relationship["duplicate_group_id"])
        canonical_id = str(relationship["canonical_reference_media_id"])
        canonical = object_rows.get(canonical_id)
        if canonical is None or canonical["decode_status"] != "valid":
            raise ValueError("duplicate relationship canonical object is unavailable")
        if canonical["duplicate_group_id"] != group_id:
            raise ValueError(
                "duplicate relationship group conflicts with its canonical"
            )
        left_id = str(relationship["left_reference_media_id"])
        right_id = str(relationship["right_reference_media_id"])
        relationships_by_group[group_id].append(relationship)
        for media_id in (left_id, right_id):
            previous_group = endpoint_group.setdefault(media_id, group_id)
            if previous_group != group_id:
                raise ValueError(
                    "duplicate relationship endpoint belongs to multiple groups"
                )
            object_row = object_rows.get(media_id)
            if (
                object_row is not None
                and object_row["decode_status"] == "valid"
                and object_row["duplicate_group_id"] != group_id
            ):
                raise ValueError(
                    "duplicate relationship group conflicts with a media object"
                )
        adjacency[group_id][left_id].add(right_id)
        adjacency[group_id][right_id].add(left_id)
        component_nodes[group_id].update((left_id, right_id))
        if relationship["provider_mirror"]:
            provider_peers[left_id].add(right_id)
            provider_peers[right_id].add(left_id)
    for media_id, row in object_rows.items():
        if row["decode_status"] == "valid" and row["provider_mirror_ids"] != sorted(
            provider_peers.get(media_id, set())
        ):
            raise ValueError(
                "provider mirror IDs do not match direct relationship evidence"
            )
    valid_groups: dict[str, set[str]] = defaultdict(set)
    canonical_by_group: dict[str, str] = {}
    for media_id, row in object_rows.items():
        if row["decode_status"] != "valid":
            continue
        group_id = str(row["duplicate_group_id"])
        valid_groups[group_id].add(media_id)
        component_nodes[group_id].add(media_id)
        canonical_id = str(row["canonical_reference_media_id"])
        previous_canonical = canonical_by_group.setdefault(group_id, canonical_id)
        if previous_canonical != canonical_id:
            raise ValueError("duplicate group has multiple canonical media IDs")
    for group_id in valid_groups:
        if group_id != _duplicate_group_id(sorted(component_nodes[group_id])):
            raise ValueError("duplicate group ID does not match its evidence component")
        canonical_id = canonical_by_group[group_id]
        valid_members = sorted(valid_groups[group_id])
        expected_canonical_id = min(
            valid_members,
            key=lambda media_id: _canonical_sort_key(
                media_id,
                object_row=object_rows[media_id],
                candidate_row=candidate_rows[media_id],
                observation_row=observation_rows[
                    str(candidate_rows[media_id]["reference_observation_id"])
                ],
                config=config,
            ),
        )
        if canonical_id != expected_canonical_id:
            raise ValueError("duplicate group canonical selection is inconsistent")
        group_relationships = relationships_by_group.get(group_id, [])
        has_component_conflict = _component_metadata_conflict(
            sorted(component_nodes[group_id]),
            candidate_rows=candidate_rows,
            observation_rows=observation_rows,
        )
        if any(
            ("component_metadata_conflict" in row["evidence_types"])
            != has_component_conflict
            for row in group_relationships
        ):
            raise ValueError("component metadata conflict evidence is inconsistent")
        relationship_evidence = [
            _RelationshipEvidence(
                left_id=str(row["left_reference_media_id"]),
                right_id=str(row["right_reference_media_id"]),
                evidence_types=set(row["evidence_types"]),
                sha256_equal=bool(row["sha256_equal"]),
                perceptual_hash_distance=row["perceptual_hash_distance"],
                same_observation=bool(row["same_observation"]),
                provider_mirror=bool(row["provider_mirror"]),
                relationship_type=str(row["relationship_type"]),
                resolution_status=str(row["resolution_status"]),
            )
            for row in group_relationships
        ]
        distance_upper_bound = _component_perceptual_distance_upper_bound(
            valid_members,
            object_rows,
        )
        expected_duplicate_type = _component_duplicate_type(
            valid_members,
            relationship_evidence,
            distance_upper_bound=distance_upper_bound,
            config=config,
        )
        if any(
            object_rows[media_id]["duplicate_type"] != expected_duplicate_type
            for media_id in valid_members
        ):
            raise ValueError("duplicate group type is inconsistent with its evidence")
        reachable = {canonical_id}
        pending = [canonical_id]
        while pending:
            media_id = pending.pop()
            for peer_id in adjacency[group_id].get(media_id, set()):
                if peer_id not in reachable:
                    reachable.add(peer_id)
                    pending.append(peer_id)
        if not component_nodes[group_id] <= reachable:
            raise ValueError("duplicate group lacks a connected evidence graph")
    distance_upper_bounds = sorted(
        _component_perceptual_distance_upper_bound(
            sorted(media_ids),
            object_rows,
        )
        for media_ids in valid_groups.values()
    )
    expected_distance_summary = {
        "max": max(distance_upper_bounds, default=0),
        "p50": _nearest_rank(distance_upper_bounds, 0.50),
        "p95": _nearest_rank(distance_upper_bounds, 0.95),
    }
    if result.report.get("perceptual_component_distance_upper_bound") != (
        expected_distance_summary
    ):
        raise ValueError(
            "reference media deduplication distance summary is inconsistent"
        )
    expected_result = _deduplicate_reference_media_impl(
        result.media_objects,
        result.media_candidates,
        result.observations,
        config=config,
        generated_at=result.report.get("generated_at"),
    )
    if not result.relationships.equals(expected_result.relationships):
        raise ValueError(
            "duplicate relationship ledger is not the deterministic sparse result"
        )
    if not result.media_objects.equals(expected_result.media_objects):
        raise ValueError(
            "reference media annotations are not the deterministic deduplication result"
        )


def validate_reference_media_deduplication_artifacts(
    *,
    media_objects: pl.DataFrame,
    relationships: pl.DataFrame,
    media_candidates: pl.DataFrame,
    observations: pl.DataFrame,
    report: Mapping[str, object],
) -> None:
    """Validate a loose artifact set against its committed deduplication report."""

    normalized_report = dict(report)
    validate_reference_media_deduplication_result(
        ReferenceMediaDeduplicationResult(
            media_objects=media_objects,
            relationships=relationships,
            media_candidates=media_candidates,
            observations=observations,
            report=normalized_report,
            markdown=_deduplication_markdown(normalized_report),
        )
    )


def write_reference_media_deduplication_result(
    result: ReferenceMediaDeduplicationResult,
    output: str | Path,
) -> dict[str, Path]:
    validate_reference_media_deduplication_result(result)
    directory = Path(output)
    if directory.suffix:
        raise ValueError("reference media deduplication output must be a directory")
    if directory.exists():
        raise FileExistsError(directory)
    directory.parent.mkdir(parents=True, exist_ok=True)
    staging = directory.parent / f".{directory.name}.{uuid4().hex}.tmp"
    try:
        media_objects_path = write_reference_media_objects(
            result.media_objects,
            staging,
        )
        relationships_path = write_reference_media_duplicate_relationships(
            result.relationships,
            staging,
        )
        summary_path = staging / REFERENCE_MEDIA_DEDUPLICATION_SUMMARY_FILE
        report_path = staging / REFERENCE_MEDIA_DEDUPLICATION_REPORT_FILE
        _write_text_atomic(result.markdown, summary_path)
        _write_text_atomic(
            json.dumps(result.report, indent=2, sort_keys=True) + "\n",
            report_path,
        )
        staging.replace(directory)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return {
        "media_objects": directory / media_objects_path.name,
        "relationships": directory / relationships_path.name,
        "report": directory / report_path.name,
        "summary": directory / summary_path.name,
    }


def publish_reference_media_deduplication_result(
    result: ReferenceMediaDeduplicationResult,
    *,
    storage: CloudStorage,
    output_prefix: str,
    run_id: str | None = None,
    now: Callable[[], datetime] | None = None,
) -> dict[str, str]:
    validate_reference_media_deduplication_result(result)
    output_prefix = str(output_prefix or "").rstrip("/")
    if not output_prefix:
        raise ValueError("output_prefix must be nonblank")
    clock = now or (lambda: datetime.now(UTC))
    started_at = _utc_datetime(clock(), field="started_at")
    effective_run_id = str(run_id or "").strip() or (
        "reference-dedup-" + started_at.strftime("%Y%m%dT%H%M%S%fZ-") + uuid4().hex[:12]
    )
    run_component = _run_component(effective_run_id)
    artifact_prefix = join_uri(
        output_prefix,
        "deduplication",
        f"run_id={run_component}",
    )
    uris = {
        "media_objects": join_uri(artifact_prefix, "reference_media_objects.parquet"),
        "relationships": join_uri(
            artifact_prefix,
            "reference_media_duplicate_relationships.parquet",
        ),
        "summary": build_report_uri(
            output_prefix,
            run_id=run_component,
            report_name=REFERENCE_MEDIA_DEDUPLICATION_SUMMARY_FILE.removesuffix(".md"),
            suffix="md",
        ),
        "report": build_report_uri(
            output_prefix,
            run_id=run_component,
            report_name=REFERENCE_MEDIA_DEDUPLICATION_REPORT_FILE.removesuffix(".json"),
        ),
    }
    if any(storage.exists(uri) for uri in uris.values()):
        raise FileExistsError("reference media deduplication run already exists")
    _log_event(
        "reference_media_deduplication_publication_started",
        command="references.deduplicate_media",
        run_id=effective_run_id,
        started_at=started_at.isoformat(),
        output_prefix=artifact_prefix,
        artifacts=uris,
    )
    try:
        storage.write_parquet_shard(
            uris["media_objects"],
            result.media_objects,
            overwrite=False,
        )
        storage.write_parquet_shard(
            uris["relationships"],
            result.relationships,
            overwrite=False,
        )
        ended_at = _utc_datetime(clock(), field="ended_at")
        publication_report = _publication_report(
            result.report,
            run_id=effective_run_id,
            started_at=started_at,
            ended_at=ended_at,
            status="complete",
            uris=uris,
            storage=storage,
        )
        storage.write_text(
            uris["summary"],
            _deduplication_markdown(publication_report),
        )
        publication_report["artifacts"] = _publication_artifact_records(
            storage,
            uris,
        )
        storage.write_json(uris["report"], publication_report)
    except Exception as exc:
        ended_at = _utc_datetime(clock(), field="ended_at")
        failed_report = _publication_report(
            result.report,
            run_id=effective_run_id,
            started_at=started_at,
            ended_at=ended_at,
            status="failed",
            uris=uris,
            storage=storage,
            error=exc,
        )
        report_exists = False
        try:
            report_exists = storage.exists(uris["report"])
        except Exception:  # noqa: BLE001 - preserve the primary publication failure.
            pass
        if not report_exists:
            try:
                storage.write_text(
                    uris["summary"],
                    _deduplication_markdown(failed_report),
                )
            except Exception:  # noqa: BLE001 - JSON audit remains independently useful.
                pass
            failed_report["artifacts"] = _publication_artifact_records(
                storage,
                uris,
            )
            try:
                storage.write_json(uris["report"], failed_report)
            except Exception:  # noqa: BLE001 - preserve the primary publication failure.
                pass
        _log_event(
            "reference_media_deduplication_publication_failed",
            command="references.deduplicate_media",
            run_id=effective_run_id,
            ended_at=ended_at.isoformat(),
            error_type=type(exc).__name__,
            error=str(exc),
            artifacts=failed_report["artifacts"],
        )
        raise
    _log_event(
        "reference_media_deduplication_publication_completed",
        command="references.deduplicate_media",
        run_id=effective_run_id,
        ended_at=publication_report["ended_at"],
        elapsed_seconds=publication_report["elapsed_seconds"],
        artifacts=publication_report["artifacts"],
    )
    return uris


def _publication_report(
    base_report: Mapping[str, object],
    *,
    run_id: str,
    started_at: datetime,
    ended_at: datetime,
    status: str,
    uris: Mapping[str, str],
    storage: CloudStorage,
    error: Exception | None = None,
) -> dict[str, Any]:
    report = json.loads(json.dumps(base_report))
    outputs = dict(report.get("outputs") or {})
    outputs["artifact_uris"] = dict(sorted(uris.items()))
    report.update(
        {
            "run_id": run_id,
            "status": status,
            "started_at": started_at.isoformat(),
            "ended_at": ended_at.isoformat(),
            "elapsed_seconds": max(0.0, (ended_at - started_at).total_seconds()),
            "outputs": outputs,
            "artifacts": _publication_artifact_records(storage, uris),
        }
    )
    if error is not None:
        report["error_type"] = type(error).__name__
        report["error"] = str(error)
    return report


def _publication_artifact_records(
    storage: CloudStorage,
    uris: Mapping[str, str],
) -> dict[str, dict[str, object]]:
    records: dict[str, dict[str, object]] = {}
    for key, uri in sorted(uris.items()):
        if key == "report":
            continue
        try:
            committed: bool | None = storage.exists(uri)
            byte_count = storage.file_size(uri) if committed else None
            sha256 = storage.file_sha256(uri) if committed else None
        except Exception:  # noqa: BLE001 - audit collection must preserve primary errors.
            committed = None
            byte_count = None
            sha256 = None
        records[key] = {
            "uri": uri,
            "committed": committed,
            "byte_count": byte_count,
            "sha256": sha256,
        }
    return records


def _edge(
    edges: dict[tuple[str, str], _RelationshipEvidence],
    left_id: str,
    right_id: str,
) -> _RelationshipEvidence:
    left, right = sorted((left_id, right_id))
    if left == right:
        raise ValueError("duplicate relationship cannot link a media row to itself")
    return edges.setdefault(
        (left, right),
        _RelationshipEvidence(left_id=left, right_id=right, evidence_types=set()),
    )


def _perceptual_candidate_pairs(
    valid_rows: Mapping[str, Mapping[str, object]],
    candidate_rows: Mapping[str, Mapping[str, object]],
    *,
    config: ReferenceMediaDeduplicationConfig,
) -> list[tuple[str, str]]:
    ids_by_hash: dict[int, list[str]] = defaultdict(list)
    for media_id, row in valid_rows.items():
        ids_by_hash[_parse_perceptual_hash(str(row["perceptual_hash"]))].append(
            media_id
        )
    tree = _HammingBKTree()
    pairs: set[tuple[str, str]] = set()
    for hash_value in sorted(ids_by_hash):
        current_ids = sorted(ids_by_hash[hash_value])
        matching_hashes = tree.query(
            hash_value,
            config.same_observation_distance_threshold,
            max_results=config.max_perceptual_hash_neighbors,
            max_visited=config.max_perceptual_search_nodes,
        )
        for matching_hash in matching_hashes:
            matching_ids = sorted(ids_by_hash[matching_hash])
            distance = (hash_value ^ matching_hash).bit_count()
            if distance <= config.cross_observation_distance_threshold:
                pairs.add(tuple(sorted((matching_ids[0], current_ids[0]))))
                continue
            matching_by_observation = {
                str(candidate_rows[media_id]["reference_observation_id"]): media_id
                for media_id in matching_ids
            }
            current_by_observation = {
                str(candidate_rows[media_id]["reference_observation_id"]): media_id
                for media_id in current_ids
            }
            shared_observations = sorted(
                set(matching_by_observation) & set(current_by_observation)
            )
            if shared_observations:
                observation_id = shared_observations[0]
                pairs.add(
                    tuple(
                        sorted(
                            (
                                matching_by_observation[observation_id],
                                current_by_observation[observation_id],
                            )
                        )
                    )
                )
        if current_ids:
            anchor_id = current_ids[0]
            pairs.update((anchor_id, right_id) for right_id in current_ids[1:])
        tree.add(hash_value)
    return sorted(pairs)


def _provider_mirror_pairs(
    candidate_rows: Mapping[str, Mapping[str, object]],
    observation_rows: Mapping[str, Mapping[str, object]],
) -> set[tuple[str, str]]:
    inaturalist_observation_ids: dict[str, str] = {}
    gbif_observation_ids: dict[str, str] = {}
    for observation_id, row in observation_rows.items():
        source = str(row["source"]).casefold()
        if source == "inaturalist":
            inaturalist_observation_ids[observation_id] = str(
                row["source_observation_id"]
            )
        elif source == "gbif":
            parsed = _inaturalist_observation_id(row.get("source_record_url"))
            if parsed is not None:
                gbif_observation_ids[observation_id] = parsed

    direct: dict[tuple[str, str], set[str]] = defaultdict(set)
    mirrored: dict[tuple[str, str], set[str]] = defaultdict(set)
    for media_id, row in candidate_rows.items():
        observation_id = str(row["reference_observation_id"])
        source = str(row["source"]).casefold()
        if source == "inaturalist":
            provider_observation_id = inaturalist_observation_ids.get(observation_id)
            destination = direct
        elif source == "gbif":
            provider_observation_id = gbif_observation_ids.get(observation_id)
            destination = mirrored
        else:
            continue
        if provider_observation_id is None:
            continue
        for identity_key in _media_identity_keys(row):
            destination[(provider_observation_id, identity_key)].add(media_id)

    pairs: set[tuple[str, str]] = set()
    for key in sorted(set(direct) & set(mirrored)):
        left_ids = sorted(direct[key])
        right_ids = sorted(mirrored[key])
        left_anchor = left_ids[0]
        right_anchor = right_ids[0]
        pairs.update(
            tuple(sorted((left_anchor, right_id)))
            for right_id in right_ids
            if left_anchor != right_id
        )
        pairs.update(
            tuple(sorted((left_id, right_anchor)))
            for left_id in left_ids[1:]
            if left_id != right_anchor
        )
    return pairs


def _media_identity_keys(row: Mapping[str, object]) -> set[str]:
    keys: set[str] = set()
    photo_id = _inaturalist_photo_id(row.get("media_identifier"))
    if photo_id is not None:
        keys.add(f"photo:{photo_id}")
    if str(row.get("source") or "").casefold() == "inaturalist":
        provider_id = str(row.get("provider_media_id") or "").strip()
        if provider_id.isdigit() and int(provider_id) > 0:
            keys.add(f"photo:{int(provider_id)}")
    identifier = str(row.get("media_identifier") or "").strip()
    if identifier:
        keys.add(f"url:{identifier}")
    return keys


def _inaturalist_observation_id(value: object) -> str | None:
    match = _INATURALIST_OBSERVATION_PATTERN.search(str(value or ""))
    return match.group(1) if match else None


def _inaturalist_photo_id(value: object) -> str | None:
    match = _INATURALIST_PHOTO_PATTERN.search(str(value or ""))
    return match.group(1) if match else None


def _metadata_conflict(
    left_candidate: Mapping[str, object],
    right_candidate: Mapping[str, object],
    observation_rows: Mapping[str, Mapping[str, object]],
) -> bool:
    left_observation = observation_rows[str(left_candidate["reference_observation_id"])]
    right_observation = observation_rows[
        str(right_candidate["reference_observation_id"])
    ]
    left_taxon = str(left_observation.get("accepted_taxon_key") or "").strip()
    right_taxon = str(right_observation.get("accepted_taxon_key") or "").strip()
    if left_taxon and right_taxon and left_taxon != right_taxon:
        return True
    left_licence = _licence_identity(left_candidate)
    right_licence = _licence_identity(right_candidate)
    return bool(
        left_licence
        and left_licence.startswith("conflict:")
        or right_licence
        and right_licence.startswith("conflict:")
        or left_licence
        and right_licence
        and left_licence != right_licence
    )


def _component_metadata_conflict(
    media_ids: Sequence[str],
    *,
    candidate_rows: Mapping[str, Mapping[str, object]],
    observation_rows: Mapping[str, Mapping[str, object]],
) -> bool:
    taxa: set[str] = set()
    licences: set[str] = set()
    for media_id in media_ids:
        candidate = candidate_rows[media_id]
        observation = observation_rows[str(candidate["reference_observation_id"])]
        if taxon := str(observation.get("accepted_taxon_key") or "").strip():
            taxa.add(taxon)
        licence = _licence_identity(candidate)
        if licence is not None:
            if licence.startswith("conflict:"):
                return True
            licences.add(licence)
    return len(taxa) > 1 or len(licences) > 1


def _licence_identity(candidate: Mapping[str, object]) -> str | None:
    supplied = [
        value
        for value in (candidate.get("licence"), candidate.get("licence_uri"))
        if str(value or "").strip()
    ]
    if not supplied:
        return None
    resolved: list[tuple[str, str]] = []
    raw_identities: set[str] = set()
    for value in supplied:
        suite = canonicalise_creative_commons_licence(value)
        identity = canonicalise_creative_commons_licence_identity(value)
        if suite is None or identity is None:
            raw_identities.add(str(value).strip().casefold())
        else:
            resolved.append((suite, identity))
    suites = {suite for suite, _identity in resolved}
    explicit_identities = {
        identity for suite, identity in resolved if identity != suite
    }
    if raw_identities or len(suites) > 1 or len(explicit_identities) > 1:
        identities = raw_identities | {identity for _suite, identity in resolved}
        return "conflict:" + "+".join(sorted(identities))
    if explicit_identities:
        return next(iter(explicit_identities))
    if suites:
        return next(iter(suites))
    return None


def _canonical_sort_key(
    media_id: str,
    *,
    object_row: Mapping[str, object],
    candidate_row: Mapping[str, object],
    observation_row: Mapping[str, object],
    config: ReferenceMediaDeduplicationConfig,
) -> tuple[object, ...]:
    licence_rank = {
        "allowed": 0,
        "research_only": 1,
    }.get(str(object_row["licence_policy_status"]), 2)
    verification_rank = 0 if candidate_row["verification_status"] == "accepted" else 1
    identity_rank = (
        0
        if observation_row["taxon_reconciliation_status"]
        in {
            "accepted_key_exact",
            "accepted_name_synonym",
        }
        else 1
    )
    quality_rank = (
        0
        if str(observation_row.get("identification_quality") or "").casefold()
        in {"research", "research_grade"}
        else 1
    )
    source_ranks = {
        source.casefold(): index for index, source in enumerate(config.source_priority)
    }
    source_rank = source_ranks.get(
        str(candidate_row["source"]).casefold(),
        len(source_ranks),
    )
    pixel_area = int(object_row["decoded_width"]) * int(object_row["decoded_height"])
    byte_count = int(object_row["source_byte_count"])
    return (
        licence_rank,
        verification_rank,
        identity_rank,
        quality_rank,
        source_rank,
        -pixel_area,
        -byte_count,
        media_id,
    )


def _component_perceptual_distance_upper_bound(
    media_ids: Sequence[str],
    valid_rows: Mapping[str, Mapping[str, object]],
) -> int:
    values = [
        _parse_perceptual_hash(str(valid_rows[media_id]["perceptual_hash"]))
        for media_id in media_ids
    ]
    if not values:
        return 0
    union = 0
    intersection = (1 << 128) - 1
    for value in values:
        union |= value
        intersection &= value
    return (union ^ intersection).bit_count()


def _component_duplicate_type(
    valid_members: Sequence[str],
    edges: Sequence[_RelationshipEvidence],
    *,
    distance_upper_bound: int,
    config: ReferenceMediaDeduplicationConfig,
) -> str:
    if not edges:
        return "unique"
    if any(edge.resolution_status != "resolved" for edge in edges):
        return "unresolved_perceptual_candidate"
    if distance_upper_bound > config.same_observation_distance_threshold:
        return "unresolved_perceptual_candidate"
    types = {edge.relationship_type for edge in edges}
    if not types <= DUPLICATE_RELATIONSHIP_TYPES:
        raise ValueError("component has an unsupported relationship type")
    if len(types) == 1:
        return next(iter(types))
    if len(valid_members) == 1 and types == {"provider_mirror"}:
        return "provider_mirror"
    return "mixed"


def _validate_deduplication_graph(
    media_objects: pl.DataFrame,
    relationships: pl.DataFrame,
    *,
    media_candidates: pl.DataFrame,
    observations: pl.DataFrame,
    config: ReferenceMediaDeduplicationConfig,
) -> None:
    validate_reference_media_objects(media_objects)
    validate_reference_media_duplicate_relationships(relationships)
    validate_reference_media_candidates(media_candidates)
    validate_reference_observations(observations)
    candidate_rows = {
        str(row["reference_media_id"]): row
        for row in media_candidates.iter_rows(named=True)
    }
    observation_rows = {
        str(row["reference_observation_id"]): row
        for row in observations.iter_rows(named=True)
    }
    object_rows = {
        str(row["reference_media_id"]): row
        for row in media_objects.iter_rows(named=True)
    }
    if unknown_objects := sorted(set(object_rows) - set(candidate_rows)):
        raise ValueError(
            f"reference media objects lack candidate provenance: {unknown_objects}"
        )
    if missing_observations := sorted(
        {
            str(candidate["reference_observation_id"])
            for candidate in candidate_rows.values()
        }
        - set(observation_rows)
    ):
        raise ValueError(
            "reference media candidates lack observation provenance: "
            f"{missing_observations}"
        )
    for candidate in candidate_rows.values():
        observation = observation_rows[str(candidate["reference_observation_id"])]
        if candidate["source"] != observation["source"]:
            raise ValueError(
                "reference media candidate source conflicts with observation provenance"
            )
    provider_pairs = _provider_mirror_pairs(candidate_rows, observation_rows)
    relationships_by_group: dict[str, list[dict[str, object]]] = defaultdict(list)
    provider_peers: dict[str, set[str]] = defaultdict(set)
    endpoint_groups: dict[str, str] = {}
    for row in relationships.iter_rows(named=True):
        left_id = str(row["left_reference_media_id"])
        right_id = str(row["right_reference_media_id"])
        if left_id not in candidate_rows or right_id not in candidate_rows:
            raise ValueError(
                "duplicate relationship references unknown media provenance"
            )
        left_candidate = candidate_rows[left_id]
        right_candidate = candidate_rows[right_id]
        for side, candidate in (
            ("left", left_candidate),
            ("right", right_candidate),
        ):
            if any(
                row[f"{side}_{field}"] != candidate[candidate_field]
                for field, candidate_field in (
                    ("reference_observation_id", "reference_observation_id"),
                    ("source", "source"),
                    ("provider_media_id", "provider_media_id"),
                )
            ):
                raise ValueError(
                    "duplicate relationship endpoint conflicts with candidate provenance"
                )
        group_id = str(row["duplicate_group_id"])
        for media_id in (left_id, right_id):
            previous_group = endpoint_groups.setdefault(media_id, group_id)
            if previous_group != group_id:
                raise ValueError(
                    "duplicate relationship endpoint belongs to multiple groups"
                )
        same_observation = (
            left_candidate["reference_observation_id"]
            == right_candidate["reference_observation_id"]
        )
        if bool(row["same_observation"]) != same_observation:
            raise ValueError(
                "same-observation evidence conflicts with candidate provenance"
            )
        provider_mirror = (left_id, right_id) in provider_pairs
        if bool(row["provider_mirror"]) != provider_mirror:
            raise ValueError(
                "provider-mirror evidence conflicts with candidate provenance"
            )
        left_object = object_rows.get(left_id)
        right_object = object_rows.get(right_id)
        left_valid = left_object is not None and left_object["decode_status"] == "valid"
        right_valid = (
            right_object is not None and right_object["decode_status"] == "valid"
        )
        if not left_valid and not right_valid:
            raise ValueError("duplicate relationship has no valid endpoint object")
        if left_valid != right_valid and not provider_mirror:
            raise ValueError(
                "content duplicate evidence requires two valid endpoint objects"
            )
        both_valid = left_valid and right_valid
        expected_sha256_equal = bool(
            both_valid and left_object["sha256"] == right_object["sha256"]
        )
        if bool(row["sha256_equal"]) != expected_sha256_equal:
            raise ValueError("exact evidence conflicts with endpoint object hashes")
        expected_distance: int | None = None
        if both_valid:
            expected_distance = perceptual_hash_distance(
                str(left_object["perceptual_hash"]),
                str(right_object["perceptual_hash"]),
            )
        if row["perceptual_hash_distance"] != expected_distance:
            raise ValueError(
                "perceptual evidence conflicts with endpoint object hashes"
            )
        evidence_types = set(row["evidence_types"])
        if ("perceptual_hash" in evidence_types) != both_valid:
            raise ValueError(
                "perceptual evidence presence conflicts with endpoint objects"
            )
        direct_metadata_conflict = _metadata_conflict(
            left_candidate,
            right_candidate,
            observation_rows,
        )
        if ("metadata_conflict" in evidence_types) != direct_metadata_conflict:
            raise ValueError("metadata conflict evidence is inconsistent")
        if expected_sha256_equal:
            expected_relationship_type = "exact"
            expected_resolution_status = "resolved"
        elif provider_mirror:
            expected_relationship_type = "provider_mirror"
            if both_valid:
                threshold = (
                    config.same_observation_distance_threshold
                    if same_observation
                    else config.cross_observation_distance_threshold
                )
                informative = _perceptual_hash_is_informative(
                    str(left_object["perceptual_hash"]),
                    minimum_bits=config.minimum_informative_bits,
                ) and _perceptual_hash_is_informative(
                    str(right_object["perceptual_hash"]),
                    minimum_bits=config.minimum_informative_bits,
                )
                compatible = _aspect_ratios_compatible(
                    left_object,
                    right_object,
                    maximum_delta=config.max_aspect_ratio_delta,
                )
                expected_resolution_status = (
                    "resolved"
                    if expected_distance <= threshold and informative and compatible
                    else "conflict"
                )
            else:
                expected_resolution_status = "resolved"
        else:
            assert both_valid and expected_distance is not None
            threshold = (
                config.same_observation_distance_threshold
                if same_observation
                else config.cross_observation_distance_threshold
            )
            if expected_distance > threshold:
                raise ValueError(
                    "perceptual relationship exceeds the configured threshold"
                )
            informative = _perceptual_hash_is_informative(
                str(left_object["perceptual_hash"]),
                minimum_bits=config.minimum_informative_bits,
            ) and _perceptual_hash_is_informative(
                str(right_object["perceptual_hash"]),
                minimum_bits=config.minimum_informative_bits,
            )
            compatible = _aspect_ratios_compatible(
                left_object,
                right_object,
                maximum_delta=config.max_aspect_ratio_delta,
            )
            if same_observation and informative and compatible:
                dimensions_differ = (
                    left_object["decoded_width"],
                    left_object["decoded_height"],
                ) != (
                    right_object["decoded_width"],
                    right_object["decoded_height"],
                )
                expected_relationship_type = (
                    "resized_copy" if dimensions_differ else "near_identical_burst"
                )
                expected_resolution_status = "resolved"
            else:
                expected_relationship_type = "perceptual_candidate"
                expected_resolution_status = "review_required"
        if direct_metadata_conflict or "component_metadata_conflict" in evidence_types:
            expected_resolution_status = "conflict"
        if row["relationship_type"] != expected_relationship_type:
            raise ValueError(
                "duplicate relationship type conflicts with endpoint evidence"
            )
        if row["resolution_status"] != expected_resolution_status:
            raise ValueError(
                "duplicate resolution status conflicts with endpoint evidence"
            )
        canonical_id = str(row["canonical_reference_media_id"])
        if (
            canonical_id not in object_rows
            or object_rows[canonical_id]["decode_status"] != "valid"
        ):
            raise ValueError("duplicate relationship canonical object is unavailable")
        relationships_by_group[group_id].append(row)
        if row["provider_mirror"]:
            provider_peers[left_id].add(right_id)
            provider_peers[right_id].add(left_id)
    for media_id, row in object_rows.items():
        if row["decode_status"] != "valid":
            continue
        if row["provider_mirror_ids"] != sorted(provider_peers.get(media_id, set())):
            raise ValueError(
                "provider mirror IDs do not match direct relationship evidence"
            )
        group_id = str(row["duplicate_group_id"])
        for relationship in relationships_by_group.get(group_id, []):
            if (
                relationship["canonical_reference_media_id"]
                != row["canonical_reference_media_id"]
            ):
                raise ValueError("duplicate relationship canonical ID is inconsistent")


def _deduplication_report(
    *,
    media_objects: pl.DataFrame,
    relationships: pl.DataFrame,
    media_candidates: pl.DataFrame,
    observations: pl.DataFrame,
    group_context: Mapping[str, Mapping[str, object]],
    generated_at: datetime,
    config: ReferenceMediaDeduplicationConfig,
) -> dict[str, Any]:
    valid = media_objects.filter(pl.col("decode_status") == "valid")
    duplicate_type_counts = Counter(
        str(value) for value in valid["duplicate_type"].to_list()
    )
    relationship_type_counts = Counter(
        str(value) for value in relationships["relationship_type"].to_list()
    )
    resolution_counts = Counter(
        str(value) for value in relationships["resolution_status"].to_list()
    )
    distance_upper_bounds = sorted(
        int(context["perceptual_distance_upper_bound"])
        for context in group_context.values()
    )
    return {
        "schema_version": REFERENCE_MEDIA_DEDUPLICATION_REPORT_VERSION,
        "command": "references.deduplicate_media",
        "pid": os.getpid(),
        "git_sha": _git_sha(),
        "status": "complete",
        "generated_at": generated_at.isoformat(),
        "inputs": {
            "media_object_rows": media_objects.height,
            "media_candidate_rows": media_candidates.height,
            "observation_rows": observations.height,
            "artifact_uris": {
                "media_objects": "not_instrumented",
                "media_candidates": "not_instrumented",
                "observations": "not_instrumented",
            },
            "media_objects_fingerprint": _intrinsic_media_objects_fingerprint(
                media_objects
            ),
            "media_candidates_fingerprint": _frame_fingerprint(media_candidates),
            "observations_fingerprint": _frame_fingerprint(observations),
        },
        "settings": config.payload,
        "policy_fingerprint": config.fingerprint,
        "counts": {
            "valid_media": valid.height,
            "invalid_media": media_objects.height - valid.height,
            "duplicate_groups": len(group_context),
            "canonical_media": len(group_context),
            "relationships": relationships.height,
            "provider_mirror_relationships": relationships.filter(
                pl.col("provider_mirror")
            ).height,
            "review_required_relationships": resolution_counts["review_required"],
            "conflicting_relationships": resolution_counts["conflict"],
        },
        "duplicate_type_counts": dict(sorted(duplicate_type_counts.items())),
        "relationship_type_counts": dict(sorted(relationship_type_counts.items())),
        "resolution_status_counts": dict(sorted(resolution_counts.items())),
        "perceptual_component_distance_upper_bound": {
            "max": max(distance_upper_bounds, default=0),
            "p50": _nearest_rank(distance_upper_bounds, 0.50),
            "p95": _nearest_rank(distance_upper_bounds, 0.95),
        },
        "outputs": {
            "media_objects_fingerprint": _frame_fingerprint(media_objects),
            "relationships_fingerprint": _frame_fingerprint(relationships),
            "artifact_uris": {
                "media_objects": "not_instrumented",
                "relationships": "not_instrumented",
                "report": "not_instrumented",
                "summary": "not_instrumented",
            },
        },
    }


def _deduplication_markdown(report: Mapping[str, object]) -> str:
    counts = report["counts"]
    assert isinstance(counts, Mapping)
    inputs = report.get("inputs")
    outputs = report.get("outputs")
    artifacts = report.get("artifacts")
    lines = [
        "# Reference media deduplication",
        "",
        f"- Command: `{report.get('command', 'not_instrumented')}`",
        f"- Run ID: `{report.get('run_id', 'not_instrumented')}`",
        f"- PID: `{report.get('pid', 'not_instrumented')}`",
        f"- Git SHA: `{report.get('git_sha') or 'not_instrumented'}`",
        f"- Status: `{report['status']}`",
        f"- Started: `{report.get('started_at', report.get('generated_at', 'not_instrumented'))}`",
        f"- Ended: `{report.get('ended_at', 'not_instrumented')}`",
        f"- Elapsed seconds: `{report.get('elapsed_seconds', 'not_instrumented')}`",
        f"- Valid media: {counts['valid_media']}",
        f"- Duplicate groups: {counts['duplicate_groups']}",
        f"- Direct relationships: {counts['relationships']}",
        f"- Review required: {counts['review_required_relationships']}",
        f"- Conflicts: {counts['conflicting_relationships']}",
    ]
    if report.get("error_type"):
        lines.extend(
            [
                f"- Error type: `{report['error_type']}`",
                f"- Error: `{report.get('error', 'not_instrumented')}`",
            ]
        )
    for title, values in (("Inputs", inputs), ("Outputs", outputs)):
        lines.extend(["", f"## {title}"])
        if isinstance(values, Mapping):
            lines.extend(
                f"- `{key}`: `{_audit_value(value)}`"
                for key, value in sorted(values.items())
            )
        else:
            lines.append("- `not_instrumented`")
    lines.extend(["", "## Artifacts"])
    if isinstance(artifacts, Mapping) and artifacts:
        for key, value in sorted(artifacts.items()):
            uri = value.get("uri") if isinstance(value, Mapping) else None
            lines.append(f"- `{key}`: `{uri or 'not_instrumented'}`")
    else:
        lines.append("- `not_instrumented`")
    lines.append("")
    return "\n".join(lines)


def _audit_value(value: object) -> str:
    if isinstance(value, Mapping) or isinstance(value, (list, tuple)):
        return json.dumps(value, sort_keys=True, separators=(",", ":"))
    return str(value) if value is not None else "null"


def _parse_perceptual_hash(value: object) -> int:
    match = _PERCEPTUAL_HASH_PATTERN.fullmatch(str(value or ""))
    if match is None:
        raise ValueError(
            "perceptual hash must use dhash128-v1 with 32 lowercase hex digits"
        )
    return int(match.group(1), 16)


def _perceptual_hash_is_informative(value: str, *, minimum_bits: int) -> bool:
    bit_count = _parse_perceptual_hash(value).bit_count()
    return minimum_bits <= bit_count <= 128 - minimum_bits


def _aspect_ratios_compatible(
    left: Mapping[str, object],
    right: Mapping[str, object],
    *,
    maximum_delta: float,
) -> bool:
    left_width = int(left["decoded_width"])
    left_height = int(left["decoded_height"])
    right_width = int(right["decoded_width"])
    right_height = int(right["decoded_height"])
    left_cross = max(left_width, left_height) * min(right_width, right_height)
    right_cross = max(right_width, right_height) * min(left_width, left_height)
    denominator = max(left_cross, right_cross)
    return (
        denominator > 0 and abs(left_cross - right_cross) / denominator <= maximum_delta
    )


def _stronger_relationship_type(left: str, right: str) -> str:
    priority = {
        "perceptual_candidate": 0,
        "near_identical_burst": 1,
        "resized_copy": 2,
        "provider_mirror": 3,
        "exact": 4,
    }
    return max((left, right), key=lambda value: priority[value])


def _stronger_resolution_status(left: str, right: str) -> str:
    priority = {"resolved": 0, "review_required": 1, "conflict": 2}
    return max((left, right), key=lambda value: priority[value])


def _duplicate_group_id(media_ids: Sequence[str]) -> str:
    digest = _fingerprint({"reference_media_ids": sorted(media_ids)}).removeprefix(
        "sha256:"
    )
    return f"reference-duplicate-group:{digest[:32]}"


def _duplicate_relationship_id(
    left_id: str,
    right_id: str,
    *,
    evidence_types: Sequence[str],
) -> str:
    digest = _fingerprint(
        {
            "left_reference_media_id": left_id,
            "right_reference_media_id": right_id,
            "evidence_types": list(evidence_types),
        }
    ).removeprefix("sha256:")
    return f"reference-duplicate-relationship:{digest[:32]}"


def _normalised_optional_text(value: object) -> str | None:
    text = str(value or "").strip().casefold()
    return text or None


def _utc_datetime(value: str | datetime, *, field: str) -> datetime:
    if isinstance(value, str):
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    elif isinstance(value, datetime):
        parsed = value
    else:
        raise TypeError(f"{field} must be a datetime or ISO timestamp")
    if parsed.tzinfo is None:
        raise ValueError(f"{field} must be timezone-aware")
    return parsed.astimezone(UTC)


def _fingerprint(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        default=str,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _frame_fingerprint(frame: pl.DataFrame) -> str:
    return _fingerprint(frame.to_dicts())


def _intrinsic_media_objects_fingerprint(frame: pl.DataFrame) -> str:
    rows: list[dict[str, object]] = []
    for source_row in frame.iter_rows(named=True):
        row = dict(source_row)
        row["duplicate_group_id"] = None
        row["duplicate_type"] = None
        row["canonical_reference_media_id"] = None
        row["provider_mirror_ids"] = []
        rows.append(row)
    return _fingerprint(rows)


def _nearest_rank(values: Sequence[int], quantile: float) -> int | None:
    if not values:
        return None
    index = max(0, math.ceil(quantile * len(values)) - 1)
    return int(values[index])


def _git_sha() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except OSError, subprocess.CalledProcessError:
        return None


def _run_component(run_id: str) -> str:
    readable = safe_path_component(run_id)[:80].rstrip("_") or "run"
    digest = hashlib.sha256(run_id.encode("utf-8")).hexdigest()[:12]
    return f"{readable}-{digest}"


def _write_text_atomic(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        temporary.write_text(text, encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _log_event(event: str, **values: object) -> None:
    payload = {
        "event": event,
        "pid": os.getpid(),
        "git_sha": _git_sha(),
        **values,
    }
    LOGGER.info("%s", json.dumps(payload, sort_keys=True, default=str))


__all__ = [
    "REFERENCE_MEDIA_DEDUPLICATION_REPORT_FILE",
    "REFERENCE_MEDIA_DEDUPLICATION_REPORT_VERSION",
    "REFERENCE_MEDIA_DEDUPLICATION_SUMMARY_FILE",
    "REFERENCE_MEDIA_DEDUPLICATOR_VERSION",
    "REFERENCE_PERCEPTUAL_HASH_VERSION",
    "ReferenceMediaDeduplicationConfig",
    "ReferenceMediaDeduplicationResult",
    "compute_reference_perceptual_hash",
    "deduplicate_reference_media",
    "perceptual_hash_distance",
    "publish_reference_media_deduplication_result",
    "validate_reference_media_deduplication_artifacts",
    "validate_reference_media_deduplication_result",
    "write_reference_media_deduplication_result",
]
