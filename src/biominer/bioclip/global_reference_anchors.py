"""Deterministic global safety anchors over the reference geography index."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path
import re

import polars as pl

from biominer.bioclip.reference_geography_index import (
    reference_geography_index_artifact_fingerprint,
    validate_reference_geography_index,
)
from biominer.common.semantic_hash import canonical_semantic_fingerprint
from biominer.references.admission import REFERENCE_ADMISSION_MODES
from biominer.references.schemas import REFERENCE_VIEWS
from biominer.storage.parquet import write_parquet


GLOBAL_REFERENCE_ANCHORS_SCHEMA_VERSION = "global-reference-anchors-v1.0.0"
GLOBAL_REFERENCE_ANCHORS_FILE = "global_reference_anchors.parquet"
GLOBAL_REFERENCE_ANCHOR_POLICY_VERSION = "global-reference-diversity-v1.0.0"

_SHA256_PATTERN = re.compile(r"sha256:[0-9a-f]{64}\Z")
_DIVERSITY_DIMENSION_ORDER = (
    "route",
    "visual_domain",
    "photographer",
    "country",
    "bioregion",
    "coarse_cell",
    "regional_cell",
    "local_cell",
    "observation_month",
    "visual_view",
    "visual_input_kind",
)
_DIVERSITY_DIMENSIONS = frozenset(_DIVERSITY_DIMENSION_ORDER)
_SORT = (
    "accepted_taxon_key",
    "route",
    "group_selection_rank",
    "reference_observation_id",
    "reference_media_id",
    "embedding_fingerprint",
)


@dataclass(frozen=True, slots=True)
class GlobalReferenceAnchorPolicy:
    """Bounded, observation-independent global anchor selection policy."""

    anchors_per_taxon_route: int = 12
    version: str = GLOBAL_REFERENCE_ANCHOR_POLICY_VERSION

    def __post_init__(self) -> None:
        if (
            isinstance(self.anchors_per_taxon_route, bool)
            or not isinstance(self.anchors_per_taxon_route, int)
            or self.anchors_per_taxon_route < 1
        ):
            raise ValueError("anchors_per_taxon_route must be a positive integer")
        if not isinstance(self.version, str) or not self.version.strip():
            raise ValueError("global anchor policy version must be nonblank text")
        object.__setattr__(self, "version", self.version.strip())

    @property
    def fingerprint(self) -> str:
        """Return the immutable semantic identity of this policy."""

        return canonical_semantic_fingerprint(
            {
                "schema_version": GLOBAL_REFERENCE_ANCHORS_SCHEMA_VERSION,
                "version": self.version,
                "anchors_per_taxon_route": self.anchors_per_taxon_route,
                "maximum_per_observation": 1,
                "maximum_per_duplicate_group": 1,
                "photographer_reuse": "only_after_unused_known_photographers",
                "date_bucket": "calendar_month",
                "diversity_dimensions": list(_DIVERSITY_DIMENSION_ORDER),
                "quality_order": "fewer_explicit_reference_quality_flags",
            }
        )


def global_reference_anchors_schema() -> dict[str, pl.DataType]:
    """Return the closed artifact schema at independent-observation grain."""

    return {
        "schema_version": pl.String,
        "registry_version": pl.String,
        "reference_bank_version": pl.String,
        "accepted_taxon_key": pl.String,
        "scientific_name": pl.String,
        "route": pl.String,
        "visual_domain": pl.String,
        "reference_media_id": pl.String,
        "reference_observation_id": pl.String,
        "duplicate_group_id": pl.String,
        "observer_id_hash": pl.String,
        "source": pl.String,
        "country_code": pl.String,
        "bioregion": pl.String,
        "coarse_cell_id": pl.String,
        "regional_cell_id": pl.String,
        "local_cell_id": pl.String,
        "coordinate_quality": pl.String,
        "observation_date": pl.Date,
        "visual_input_kind": pl.String,
        "view": pl.String,
        "admission_mode": pl.String,
        "reference_quality_flags": pl.List(pl.String),
        "embedding_fingerprint": pl.String,
        "reference_geography_row_fingerprint": pl.String,
        "anchor_group_id": pl.String,
        "selection_rank": pl.UInt32,
        "group_selection_rank": pl.UInt32,
        "selection_round": pl.String,
        "observer_reused": pl.Boolean,
        "geography_diversity_key": pl.String,
        "date_diversity_key": pl.String,
        "diversity_dimensions_added": pl.List(pl.String),
        "quality_flag_count": pl.UInt16,
        "eligible_observation_count": pl.UInt32,
        "requested_anchor_count": pl.UInt32,
        "selected_anchor_count": pl.UInt32,
        "anchor_shortfall": pl.UInt32,
        "selection_policy_fingerprint": pl.String,
        "reference_geography_index_fingerprint": pl.String,
        "row_fingerprint": pl.String,
    }


def select_global_reference_anchors(
    reference_geography_index: pl.DataFrame,
    *,
    views_by_media: Mapping[str, str] | None = None,
    policy: GlobalReferenceAnchorPolicy | None = None,
) -> pl.DataFrame:
    """Select a bounded global core without counting duplicate observations.

    The reference index deliberately does not contain biological-view labels.
    Callers may provide those labels by media identity. Missing labels remain
    explicit as ``unknown`` and never earn a view-diversity gain.
    """

    validate_reference_geography_index(reference_geography_index)
    selected_policy = policy or GlobalReferenceAnchorPolicy()
    if not isinstance(selected_policy, GlobalReferenceAnchorPolicy):
        raise TypeError("policy must be GlobalReferenceAnchorPolicy")
    views = _normalized_views(
        views_by_media,
        known_media=set(reference_geography_index["reference_media_id"].to_list()),
    )
    candidates = [
        _candidate(row, view=views.get(str(row["reference_media_id"]), "unknown"))
        for row in reference_geography_index.iter_rows(named=True)
        if row["global_anchor_eligible"]
    ]
    _validate_candidate_identity(candidates)
    index_fingerprint = reference_geography_index_artifact_fingerprint(
        reference_geography_index
    )
    grouped: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for candidate in candidates:
        grouped[(str(candidate["accepted_taxon_key"]), str(candidate["route"]))].append(
            candidate
        )

    selected: list[dict[str, object]] = []
    for group_key in sorted(grouped):
        group_rows = grouped[group_key]
        selected.extend(
            _select_group(
                group_rows,
                group_key=group_key,
                policy=selected_policy,
                index_fingerprint=index_fingerprint,
            )
        )
    for rank, row in enumerate(selected, start=1):
        row["selection_rank"] = rank
        row["row_fingerprint"] = canonical_semantic_fingerprint(row)
    schema = global_reference_anchors_schema()
    frame = (
        pl.DataFrame(selected, schema=schema, orient="row", strict=True).sort(*_SORT)
        if selected
        else pl.DataFrame(schema=schema)
    )
    validate_global_reference_anchors(frame)
    return frame


def validate_global_reference_anchors(frame: pl.DataFrame) -> None:
    """Reject schema, independence, ordering, rank, or provenance drift."""

    if not isinstance(frame, pl.DataFrame):
        raise TypeError("global reference anchors must be a Polars DataFrame")
    if frame.schema != global_reference_anchors_schema():
        raise ValueError("global reference anchors schema mismatch")
    if frame.is_empty():
        return
    if frame["reference_observation_id"].n_unique() != frame.height:
        raise ValueError("global anchors inflate a biological observation")
    if frame["duplicate_group_id"].n_unique() != frame.height:
        raise ValueError("global anchors inflate a duplicate group")
    if not frame.equals(frame.sort(*_SORT)):
        raise ValueError("global reference anchors are not canonically sorted")
    if sorted(frame["selection_rank"].to_list()) != list(range(1, frame.height + 1)):
        raise ValueError("global anchor selection_rank is not contiguous")
    for row in frame.iter_rows(named=True):
        _validate_anchor_row(row)
    for group in frame.partition_by(
        ["accepted_taxon_key", "route"], maintain_order=True
    ):
        _validate_anchor_group(group)


def global_reference_anchors_artifact_fingerprint(frame: pl.DataFrame) -> str:
    """Fingerprint selected global-core semantics independently of path."""

    validate_global_reference_anchors(frame)
    return canonical_semantic_fingerprint(
        {
            "schema_version": GLOBAL_REFERENCE_ANCHORS_SCHEMA_VERSION,
            "selection_policy_fingerprints": sorted(
                frame["selection_policy_fingerprint"].unique().to_list()
            ),
            "reference_geography_index_fingerprints": sorted(
                frame["reference_geography_index_fingerprint"].unique().to_list()
            ),
            "row_fingerprints": sorted(frame["row_fingerprint"].to_list()),
        }
    )


def write_global_reference_anchors(
    frame: pl.DataFrame,
    output: str | Path,
) -> Path:
    """Validate and write ``global_reference_anchors.parquet`` atomically."""

    validate_global_reference_anchors(frame)
    destination = Path(output)
    if destination.suffix.casefold() != ".parquet":
        destination /= GLOBAL_REFERENCE_ANCHORS_FILE
    return write_parquet(frame, destination)


def _select_group(
    candidates: Sequence[dict[str, object]],
    *,
    group_key: tuple[str, str],
    policy: GlobalReferenceAnchorPolicy,
    index_fingerprint: str,
) -> list[dict[str, object]]:
    observation_count = len(
        {str(row["reference_observation_id"]) for row in candidates}
    )
    requested = policy.anchors_per_taxon_route
    target = min(requested, observation_count)
    selected: list[dict[str, object]] = []
    used_observations: set[str] = set()
    used_duplicates: set[str] = set()
    used_observers: set[str] = set()
    covered: dict[str, set[str]] = defaultdict(set)
    while len(selected) < target:
        available = [
            row
            for row in candidates
            if str(row["reference_observation_id"]) not in used_observations
            and str(row["duplicate_group_id"]) not in used_duplicates
        ]
        if not available:
            break
        unused_observer = [
            row
            for row in available
            if row["observer_id_hash"] is not None
            and str(row["observer_id_hash"]) not in used_observers
        ]
        pool = unused_observer or available
        winner = min(pool, key=lambda row: _selection_key(row, covered=covered))
        additions = _diversity_additions(winner, covered=covered)
        observer = winner["observer_id_hash"]
        observer_reused = observer is not None and str(observer) in used_observers
        selection_round = (
            "independent_photographer"
            if unused_observer
            else "photographer_unavailable"
            if observer is None
            else "photographer_reuse"
        )
        selected.append(
            _anchor_row(
                winner,
                group_key=group_key,
                group_rank=len(selected) + 1,
                selection_round=selection_round,
                observer_reused=observer_reused,
                additions=additions,
                eligible_observation_count=observation_count,
                requested_anchor_count=requested,
                selected_anchor_count=0,
                policy=policy,
                index_fingerprint=index_fingerprint,
            )
        )
        used_observations.add(str(winner["reference_observation_id"]))
        used_duplicates.add(str(winner["duplicate_group_id"]))
        if observer is not None:
            used_observers.add(str(observer))
        _record_coverage(winner, covered=covered)

    selected_count = len(selected)
    shortfall = requested - selected_count
    for row in selected:
        row["selected_anchor_count"] = selected_count
        row["anchor_shortfall"] = shortfall
    return selected


def _candidate(row: Mapping[str, object], *, view: str) -> dict[str, object]:
    candidate = dict(row)
    candidate["view"] = view
    candidate["quality_flag_count"] = len(row["reference_quality_flags"])
    candidate["geography_diversity_key"] = _geography_key(row)
    candidate["date_diversity_key"] = _date_key(row["observation_date"])
    return candidate


def _selection_key(
    row: Mapping[str, object],
    *,
    covered: Mapping[str, set[str]],
) -> tuple[object, ...]:
    additions = _diversity_additions(row, covered=covered)
    known_dimensions = sum(
        value is not None
        for value in (
            row["observer_id_hash"],
            row["country_code"],
            row["bioregion"],
            row["coarse_cell_id"],
            row["observation_date"],
        )
    ) + (row["view"] != "unknown")
    return (
        -len(additions),
        -known_dimensions,
        int(row["quality_flag_count"]),
        str(row["reference_observation_id"]),
        str(row["reference_media_id"]),
        str(row["embedding_fingerprint"]),
    )


def _diversity_values(row: Mapping[str, object]) -> dict[str, str | None]:
    observed = row["observation_date"]
    return {
        "route": str(row["route"]),
        "visual_domain": str(row["visual_domain"]),
        "photographer": _optional_string(row["observer_id_hash"]),
        "country": _optional_string(row["country_code"]),
        "bioregion": _optional_string(row["bioregion"]),
        "coarse_cell": _optional_string(row["coarse_cell_id"]),
        "regional_cell": _optional_string(row["regional_cell_id"]),
        "local_cell": _optional_string(row["local_cell_id"]),
        "observation_month": (
            f"{observed.year:04d}-{observed.month:02d}"
            if isinstance(observed, date)
            else None
        ),
        "visual_view": str(row["view"]) if row["view"] != "unknown" else None,
        "visual_input_kind": str(row["visual_input_kind"]),
    }


def _diversity_additions(
    row: Mapping[str, object],
    *,
    covered: Mapping[str, set[str]],
) -> list[str]:
    values = _diversity_values(row)
    return [
        dimension
        for dimension in _DIVERSITY_DIMENSION_ORDER
        if values[dimension] is not None
        and str(values[dimension]) not in covered.get(dimension, set())
    ]


def _record_coverage(
    row: Mapping[str, object],
    *,
    covered: dict[str, set[str]],
) -> None:
    for dimension, value in _diversity_values(row).items():
        if value is not None:
            covered[dimension].add(value)


def _anchor_row(
    candidate: Mapping[str, object],
    *,
    group_key: tuple[str, str],
    group_rank: int,
    selection_round: str,
    observer_reused: bool,
    additions: list[str],
    eligible_observation_count: int,
    requested_anchor_count: int,
    selected_anchor_count: int,
    policy: GlobalReferenceAnchorPolicy,
    index_fingerprint: str,
) -> dict[str, object]:
    copied_fields = (
        "registry_version",
        "reference_bank_version",
        "accepted_taxon_key",
        "scientific_name",
        "route",
        "visual_domain",
        "reference_media_id",
        "reference_observation_id",
        "duplicate_group_id",
        "observer_id_hash",
        "source",
        "country_code",
        "bioregion",
        "coarse_cell_id",
        "regional_cell_id",
        "local_cell_id",
        "coordinate_quality",
        "observation_date",
        "visual_input_kind",
        "view",
        "admission_mode",
        "reference_quality_flags",
        "embedding_fingerprint",
    )
    return {
        "schema_version": GLOBAL_REFERENCE_ANCHORS_SCHEMA_VERSION,
        **{field: candidate[field] for field in copied_fields},
        "reference_geography_row_fingerprint": candidate["row_fingerprint"],
        "anchor_group_id": canonical_semantic_fingerprint(
            {
                "accepted_taxon_key": group_key[0],
                "route": group_key[1],
                "selection_policy_fingerprint": policy.fingerprint,
                "reference_geography_index_fingerprint": index_fingerprint,
            }
        ),
        "selection_rank": 0,
        "group_selection_rank": group_rank,
        "selection_round": selection_round,
        "observer_reused": observer_reused,
        "geography_diversity_key": candidate["geography_diversity_key"],
        "date_diversity_key": candidate["date_diversity_key"],
        "diversity_dimensions_added": additions,
        "quality_flag_count": candidate["quality_flag_count"],
        "eligible_observation_count": eligible_observation_count,
        "requested_anchor_count": requested_anchor_count,
        "selected_anchor_count": selected_anchor_count,
        "anchor_shortfall": 0,
        "selection_policy_fingerprint": policy.fingerprint,
        "reference_geography_index_fingerprint": index_fingerprint,
    }


def _validate_candidate_identity(candidates: Sequence[Mapping[str, object]]) -> None:
    observations: dict[str, tuple[str, str, str]] = {}
    duplicates: dict[str, tuple[str, str]] = {}
    for row in candidates:
        observation_id = str(row["reference_observation_id"])
        observation_semantics = (
            str(row["accepted_taxon_key"]),
            str(row["route"]),
            str(row["visual_domain"]),
        )
        previous_observation = observations.setdefault(
            observation_id, observation_semantics
        )
        if previous_observation != observation_semantics:
            raise ValueError(
                "global-anchor observation spans conflicting taxon or route"
            )
        duplicate_id = str(row["duplicate_group_id"])
        duplicate_semantics = observation_semantics[:2]
        previous_duplicate = duplicates.setdefault(duplicate_id, duplicate_semantics)
        if previous_duplicate != duplicate_semantics:
            raise ValueError(
                "global-anchor duplicate group spans conflicting taxon or route"
            )


def _validate_anchor_row(row: Mapping[str, object]) -> None:
    if row["schema_version"] != GLOBAL_REFERENCE_ANCHORS_SCHEMA_VERSION:
        raise ValueError("unsupported global reference anchors schema version")
    for field in (
        "reference_geography_row_fingerprint",
        "anchor_group_id",
        "selection_policy_fingerprint",
        "reference_geography_index_fingerprint",
        "embedding_fingerprint",
        "row_fingerprint",
    ):
        if not _SHA256_PATTERN.fullmatch(str(row[field])):
            raise ValueError(f"{field} is not a canonical SHA-256 fingerprint")
    if row["admission_mode"] not in REFERENCE_ADMISSION_MODES:
        raise ValueError("unsupported global-anchor admission mode")
    if row["view"] not in REFERENCE_VIEWS:
        raise ValueError("unsupported global-anchor view")
    if row["quality_flag_count"] != len(row["reference_quality_flags"]):
        raise ValueError("global-anchor quality flag count is inconsistent")
    additions = row["diversity_dimensions_added"]
    if (
        not isinstance(additions, list)
        or len(additions) != len(set(additions))
        or any(item not in _DIVERSITY_DIMENSIONS for item in additions)
        or additions
        != [item for item in _DIVERSITY_DIMENSION_ORDER if item in additions]
    ):
        raise ValueError("global-anchor diversity additions are not canonical")
    if row["selection_round"] not in {
        "independent_photographer",
        "photographer_reuse",
        "photographer_unavailable",
    }:
        raise ValueError("unsupported global-anchor selection round")
    if row["observer_reused"] != (row["selection_round"] == "photographer_reuse"):
        raise ValueError("global-anchor photographer reuse provenance conflicts")
    payload = dict(row)
    fingerprint = payload.pop("row_fingerprint")
    if fingerprint != canonical_semantic_fingerprint(payload):
        raise ValueError("global reference anchor row fingerprint mismatch")


def _validate_anchor_group(group: pl.DataFrame) -> None:
    ranks = group["group_selection_rank"].to_list()
    if ranks != list(range(1, group.height + 1)):
        raise ValueError("global anchor group ranks are not contiguous")
    for field in (
        "anchor_group_id",
        "selection_policy_fingerprint",
        "reference_geography_index_fingerprint",
        "eligible_observation_count",
        "requested_anchor_count",
        "selected_anchor_count",
        "anchor_shortfall",
    ):
        if group[field].n_unique() != 1:
            raise ValueError(f"global anchor group has conflicting {field}")
    selected = int(group["selected_anchor_count"][0])
    requested = int(group["requested_anchor_count"][0])
    eligible = int(group["eligible_observation_count"][0])
    shortfall = int(group["anchor_shortfall"][0])
    if selected != group.height or selected > eligible or selected > requested:
        raise ValueError("global anchor group counts are inconsistent")
    if shortfall != requested - selected:
        raise ValueError("global anchor group shortfall is inconsistent")


def _normalized_views(
    views_by_media: Mapping[str, str] | None,
    *,
    known_media: set[str],
) -> dict[str, str]:
    if views_by_media is None:
        return {}
    if not isinstance(views_by_media, Mapping):
        raise TypeError("views_by_media must be a mapping")
    unknown_media = set(views_by_media) - known_media
    if unknown_media:
        raise ValueError(
            "global-anchor views contain unknown reference media: "
            f"{sorted(unknown_media)}"
        )
    normalized: dict[str, str] = {}
    for media_id, value in views_by_media.items():
        if not isinstance(media_id, str) or not media_id.strip():
            raise ValueError("global-anchor view media IDs must be nonblank text")
        if not isinstance(value, str) or value.strip() not in REFERENCE_VIEWS:
            raise ValueError(f"unsupported global-anchor view: {value}")
        normalized[media_id.strip()] = value.strip()
    return normalized


def _geography_key(row: Mapping[str, object]) -> str:
    for scope, field in (
        ("local_cell", "local_cell_id"),
        ("regional_cell", "regional_cell_id"),
        ("coarse_cell", "coarse_cell_id"),
        ("bioregion", "bioregion"),
        ("country", "country_code"),
    ):
        value = row[field]
        if value is not None:
            return f"{scope}:{value}"
    return "no_geo"


def _date_key(value: object) -> str:
    if isinstance(value, date):
        return f"month:{value.year:04d}-{value.month:02d}"
    return "date:unknown"


def _optional_string(value: object) -> str | None:
    return str(value) if value is not None else None


__all__ = [
    "GLOBAL_REFERENCE_ANCHORS_FILE",
    "GLOBAL_REFERENCE_ANCHORS_SCHEMA_VERSION",
    "GLOBAL_REFERENCE_ANCHOR_POLICY_VERSION",
    "GlobalReferenceAnchorPolicy",
    "global_reference_anchors_artifact_fingerprint",
    "global_reference_anchors_schema",
    "select_global_reference_anchors",
    "validate_global_reference_anchors",
    "write_global_reference_anchors",
]
