"""Target-preserving family/geography candidate-set evidence contract."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from math import isfinite
from pathlib import Path
import re

import polars as pl

from biominer.common.semantic_hash import canonical_semantic_fingerprint
from biominer.storage.parquet import write_parquet


FAMILY_GEO_CANDIDATE_SCHEMA_VERSION = "family-geo-candidate-set-v1.0.0"
FAMILY_GEO_CANDIDATE_FILE = "family_geo_candidate_sets.parquet"
EVIDENCE_AVAILABILITY_STATES = frozenset({"available", "unavailable", "not_applicable"})

_SHA256_PATTERN = re.compile(r"sha256:[0-9a-f]{64}\Z")
_SET_ID_PATTERN = re.compile(r"family-geo-candidate-set:[0-9a-f]{64}\Z")
_GROUP_FIELDS = (
    "run_id",
    "flickr_query_id",
    "flickr_photo_id",
    "organism_unit_id",
    "scoring_stage",
    "registry_version",
    "target_accepted_taxon_key",
    "target_scientific_name",
    "query_geo_cluster_id",
    "query_coordinate_quality",
)
_INPUT_FIELDS = (
    *_GROUP_FIELDS,
    "candidate_accepted_taxon_key",
    "candidate_scientific_name",
    "family_key",
    "family_name",
    "genus_key",
    "genus_name",
    "candidate_priority",
    "candidate_reasons",
    "family_evidence_status",
    "family_evidence_reason",
    "family_evidence_rank",
    "family_evidence_raw_score",
    "family_priority_match",
    "family_changed_membership",
    "geographic_evidence_status",
    "geographic_evidence_reason",
    "geographic_scopes",
    "geographic_evidence_score",
    "occurrence_support",
    "query_evidence_status",
    "query_evidence_reason",
    "query_evidence_ids",
    "query_associated",
    "visual_neighbour_evidence_status",
    "visual_neighbour_evidence_reason",
    "visual_neighbour_graph_fingerprint",
    "visual_neighbour_rank",
    "visual_neighbour_raw_similarity",
    "visual_neighbour",
    "safety_union_membership",
    "safety_union_reasons",
    "target_candidate",
    "target_preserved",
    "included_in_complete_union",
    "source_versions",
)
_SORT = (
    "run_id",
    "flickr_photo_id",
    "organism_unit_id",
    "scoring_stage",
    "candidate_priority",
    "candidate_accepted_taxon_key",
)


def family_geo_candidate_schema() -> dict[str, pl.DataType]:
    return {
        "schema_version": pl.String,
        "candidate_set_id": pl.String,
        "run_id": pl.String,
        "flickr_query_id": pl.String,
        "flickr_photo_id": pl.String,
        "organism_unit_id": pl.String,
        "scoring_stage": pl.String,
        "registry_version": pl.String,
        "target_accepted_taxon_key": pl.String,
        "target_scientific_name": pl.String,
        "query_geo_cluster_id": pl.String,
        "query_coordinate_quality": pl.String,
        "candidate_accepted_taxon_key": pl.String,
        "candidate_scientific_name": pl.String,
        "family_key": pl.String,
        "family_name": pl.String,
        "genus_key": pl.String,
        "genus_name": pl.String,
        "candidate_priority": pl.UInt32,
        "candidate_reasons": pl.List(pl.String),
        "family_evidence_status": pl.String,
        "family_evidence_reason": pl.String,
        "family_evidence_rank": pl.UInt32,
        "family_evidence_raw_score": pl.Float64,
        "family_priority_match": pl.Boolean,
        "family_changed_membership": pl.Boolean,
        "geographic_evidence_status": pl.String,
        "geographic_evidence_reason": pl.String,
        "geographic_scopes": pl.List(pl.String),
        "geographic_evidence_score": pl.Float64,
        "occurrence_support": pl.UInt64,
        "query_evidence_status": pl.String,
        "query_evidence_reason": pl.String,
        "query_evidence_ids": pl.List(pl.String),
        "query_associated": pl.Boolean,
        "visual_neighbour_evidence_status": pl.String,
        "visual_neighbour_evidence_reason": pl.String,
        "visual_neighbour_graph_fingerprint": pl.String,
        "visual_neighbour_rank": pl.UInt32,
        "visual_neighbour_raw_similarity": pl.Float64,
        "visual_neighbour": pl.Boolean,
        "safety_union_membership": pl.Boolean,
        "safety_union_reasons": pl.List(pl.String),
        "target_candidate": pl.Boolean,
        "target_preserved": pl.Boolean,
        "included_in_complete_union": pl.Boolean,
        "source_versions": pl.List(pl.String),
        "candidate_row_fingerprint": pl.String,
        "candidate_set_fingerprint": pl.String,
    }


def build_family_geo_candidate_sets(
    rows: Sequence[Mapping[str, object]],
) -> pl.DataFrame:
    """Build one canonical row per complete-union candidate."""

    if isinstance(rows, str | bytes) or not isinstance(rows, Sequence):
        raise TypeError("family/geography candidate rows must be a sequence")
    normalized: list[dict[str, object]] = []
    for row in rows:
        if not isinstance(row, Mapping):
            raise TypeError("family/geography candidate rows must contain mappings")
        _require_exact_fields(row, set(_INPUT_FIELDS))
        base = _normalized_row(row)
        complete = {
            "schema_version": FAMILY_GEO_CANDIDATE_SCHEMA_VERSION,
            **base,
        }
        complete["candidate_row_fingerprint"] = canonical_semantic_fingerprint(complete)
        normalized.append(complete)
    if not normalized:
        return pl.DataFrame(schema=family_geo_candidate_schema())

    groups: dict[tuple[object, ...], list[dict[str, object]]] = {}
    for row in normalized:
        groups.setdefault(tuple(row[field] for field in _GROUP_FIELDS), []).append(row)
    output: list[dict[str, object]] = []
    for group_key in sorted(groups, key=lambda key: tuple(str(item) for item in key)):
        group = sorted(
            groups[group_key],
            key=lambda row: (
                int(row["candidate_priority"]),
                str(row["candidate_accepted_taxon_key"]),
            ),
        )
        _validate_set_semantics(group)
        set_fingerprint = canonical_semantic_fingerprint(
            {
                "schema_version": FAMILY_GEO_CANDIDATE_SCHEMA_VERSION,
                "group": {field: group[0][field] for field in _GROUP_FIELDS},
                "candidate_row_fingerprints": [
                    row["candidate_row_fingerprint"] for row in group
                ],
            }
        )
        set_id = "family-geo-candidate-set:" + set_fingerprint.removeprefix("sha256:")
        for row in group:
            output.append(
                {
                    "candidate_set_id": set_id,
                    **row,
                    "candidate_set_fingerprint": set_fingerprint,
                }
            )
    frame = pl.DataFrame(
        output,
        schema=family_geo_candidate_schema(),
        orient="row",
        strict=True,
    ).sort(*_SORT)
    validate_family_geo_candidate_sets(frame)
    return frame


def validate_family_geo_candidate_sets(frame: pl.DataFrame) -> None:
    if not isinstance(frame, pl.DataFrame):
        raise TypeError("family/geography candidates must be a Polars DataFrame")
    if frame.schema != family_geo_candidate_schema():
        raise ValueError("family/geography candidate schema mismatch")
    if frame.is_empty():
        return
    if not frame.equals(frame.sort(*_SORT)):
        raise ValueError("family/geography candidate rows are not canonically sorted")
    grain = frame.select("candidate_set_id", "candidate_accepted_taxon_key")
    if grain.n_unique() != frame.height:
        raise ValueError("family/geography candidate grain is not unique")

    for row in frame.iter_rows(named=True):
        if row["schema_version"] != FAMILY_GEO_CANDIDATE_SCHEMA_VERSION:
            raise ValueError("unsupported family/geography candidate schema version")
        if not _SET_ID_PATTERN.fullmatch(str(row["candidate_set_id"])):
            raise ValueError("family/geography candidate_set_id is invalid")
        normalized = _normalized_row(row)
        if any(normalized[field] != row[field] for field in _INPUT_FIELDS):
            raise ValueError("family/geography candidate fields are not canonical")
        payload = {
            "schema_version": FAMILY_GEO_CANDIDATE_SCHEMA_VERSION,
            **normalized,
        }
        if row["candidate_row_fingerprint"] != canonical_semantic_fingerprint(payload):
            raise ValueError("family/geography candidate row fingerprint mismatch")
        _sha256(row["candidate_set_fingerprint"], field="candidate_set_fingerprint")

    for set_id in sorted(frame["candidate_set_id"].unique().to_list()):
        group_frame = frame.filter(pl.col("candidate_set_id") == set_id).sort(
            "candidate_priority", "candidate_accepted_taxon_key"
        )
        group = group_frame.to_dicts()
        _validate_set_semantics(group)
        expected = canonical_semantic_fingerprint(
            {
                "schema_version": FAMILY_GEO_CANDIDATE_SCHEMA_VERSION,
                "group": {field: group[0][field] for field in _GROUP_FIELDS},
                "candidate_row_fingerprints": [
                    row["candidate_row_fingerprint"] for row in group
                ],
            }
        )
        if set(group_frame["candidate_set_fingerprint"].to_list()) != {expected}:
            raise ValueError("family/geography candidate set fingerprint mismatch")
        expected_id = "family-geo-candidate-set:" + expected.removeprefix("sha256:")
        if set_id != expected_id:
            raise ValueError("family/geography candidate set identity mismatch")


def write_family_geo_candidate_sets(frame: pl.DataFrame, output: str | Path) -> Path:
    validate_family_geo_candidate_sets(frame)
    destination = Path(output)
    if destination.suffix.casefold() != ".parquet":
        destination /= FAMILY_GEO_CANDIDATE_FILE
    return write_parquet(frame, destination)


def _normalized_row(values: Mapping[str, object]) -> dict[str, object]:
    required_text = (
        "run_id",
        "flickr_query_id",
        "flickr_photo_id",
        "organism_unit_id",
        "scoring_stage",
        "registry_version",
        "target_accepted_taxon_key",
        "target_scientific_name",
        "query_coordinate_quality",
        "candidate_accepted_taxon_key",
        "candidate_scientific_name",
        "family_key",
        "family_name",
        "genus_key",
        "genus_name",
        "family_evidence_status",
        "geographic_evidence_status",
        "query_evidence_status",
        "visual_neighbour_evidence_status",
    )
    row: dict[str, object] = {
        field: _required_text(values[field], field=field) for field in required_text
    }
    optional_text = (
        "query_geo_cluster_id",
        "family_evidence_reason",
        "geographic_evidence_reason",
        "query_evidence_reason",
        "visual_neighbour_evidence_reason",
        "visual_neighbour_graph_fingerprint",
    )
    row.update(
        {field: _optional_text(values[field], field=field) for field in optional_text}
    )
    row.update(
        {
            "candidate_priority": _nonnegative_int(
                values["candidate_priority"],
                field="candidate_priority",
                maximum=2**32 - 1,
            ),
            "candidate_reasons": _canonical_strings(
                values["candidate_reasons"],
                field="candidate_reasons",
                allow_empty=False,
            ),
            "family_evidence_rank": _optional_positive_int(
                values["family_evidence_rank"], field="family_evidence_rank"
            ),
            "family_evidence_raw_score": _optional_score(
                values["family_evidence_raw_score"],
                field="family_evidence_raw_score",
                lower=-1,
                upper=1,
            ),
            "family_priority_match": _optional_boolean(
                values["family_priority_match"], field="family_priority_match"
            ),
            "family_changed_membership": _boolean(
                values["family_changed_membership"],
                field="family_changed_membership",
            ),
            "geographic_scopes": _canonical_strings(
                values["geographic_scopes"], field="geographic_scopes"
            ),
            "geographic_evidence_score": _optional_score(
                values["geographic_evidence_score"],
                field="geographic_evidence_score",
                lower=0,
                upper=1,
            ),
            "occurrence_support": _nonnegative_int(
                values["occurrence_support"],
                field="occurrence_support",
                maximum=2**64 - 1,
            ),
            "query_evidence_ids": _canonical_strings(
                values["query_evidence_ids"], field="query_evidence_ids"
            ),
            "query_associated": _boolean(
                values["query_associated"], field="query_associated"
            ),
            "visual_neighbour_rank": _optional_positive_int(
                values["visual_neighbour_rank"], field="visual_neighbour_rank"
            ),
            "visual_neighbour_raw_similarity": _optional_score(
                values["visual_neighbour_raw_similarity"],
                field="visual_neighbour_raw_similarity",
                lower=-1,
                upper=1,
            ),
            "visual_neighbour": _boolean(
                values["visual_neighbour"], field="visual_neighbour"
            ),
            "safety_union_membership": _boolean(
                values["safety_union_membership"],
                field="safety_union_membership",
            ),
            "safety_union_reasons": _canonical_strings(
                values["safety_union_reasons"], field="safety_union_reasons"
            ),
            "target_candidate": _boolean(
                values["target_candidate"], field="target_candidate"
            ),
            "target_preserved": _boolean(
                values["target_preserved"], field="target_preserved"
            ),
            "included_in_complete_union": _boolean(
                values["included_in_complete_union"],
                field="included_in_complete_union",
            ),
            "source_versions": _canonical_strings(
                values["source_versions"], field="source_versions", allow_empty=False
            ),
        }
    )
    for field in (
        "family_evidence_status",
        "geographic_evidence_status",
        "query_evidence_status",
        "visual_neighbour_evidence_status",
    ):
        if row[field] not in EVIDENCE_AVAILABILITY_STATES:
            raise ValueError(f"unsupported {field}")
    _validate_family_evidence(row)
    _validate_geographic_evidence(row)
    _validate_query_evidence(row)
    _validate_visual_evidence(row)
    if row["family_changed_membership"]:
        raise ValueError("family evidence must not change complete-union membership")
    if not row["included_in_complete_union"]:
        raise ValueError("every candidate row must belong to the complete union")
    if row["safety_union_membership"] != bool(row["safety_union_reasons"]):
        raise ValueError("safety-union membership and reasons are inconsistent")
    return row


def _validate_family_evidence(row: Mapping[str, object]) -> None:
    status = row["family_evidence_status"]
    values = (
        row["family_evidence_rank"],
        row["family_evidence_raw_score"],
        row["family_priority_match"],
    )
    reason = row["family_evidence_reason"]
    if status == "available":
        if any(value is None for value in values) or reason is not None:
            raise ValueError("available family evidence requires rank, score and match")
    elif any(value is not None for value in values) or reason is None:
        raise ValueError("unavailable family evidence requires null values and reason")


def _validate_geographic_evidence(row: Mapping[str, object]) -> None:
    status = row["geographic_evidence_status"]
    scopes = row["geographic_scopes"]
    score = row["geographic_evidence_score"]
    reason = row["geographic_evidence_reason"]
    if status == "available":
        if not scopes or score is None or reason is not None:
            raise ValueError("available geographic evidence requires scopes and score")
    elif (
        scopes or score is not None or reason is None or row["occurrence_support"] != 0
    ):
        raise ValueError(
            "unavailable geographic evidence requires empty values and reason"
        )


def _validate_query_evidence(row: Mapping[str, object]) -> None:
    status = row["query_evidence_status"]
    evidence_ids = row["query_evidence_ids"]
    associated = row["query_associated"]
    reason = row["query_evidence_reason"]
    if status == "available":
        if not evidence_ids or not associated or reason is not None:
            raise ValueError("available query evidence requires IDs and association")
    elif evidence_ids or associated or reason is None:
        raise ValueError("unavailable query evidence requires empty values and reason")


def _validate_visual_evidence(row: Mapping[str, object]) -> None:
    status = row["visual_neighbour_evidence_status"]
    values = (
        row["visual_neighbour_graph_fingerprint"],
        row["visual_neighbour_rank"],
        row["visual_neighbour_raw_similarity"],
    )
    reason = row["visual_neighbour_evidence_reason"]
    if status == "available":
        if any(value is None for value in values) or not row["visual_neighbour"]:
            raise ValueError("available visual-neighbour evidence is incomplete")
        if reason is not None:
            raise ValueError("available visual-neighbour evidence cannot have a reason")
        _sha256(values[0], field="visual_neighbour_graph_fingerprint")
    elif (
        any(value is not None for value in values)
        or row["visual_neighbour"]
        or reason is None
    ):
        raise ValueError("unavailable visual-neighbour evidence requires null values")


def _validate_set_semantics(group: Sequence[Mapping[str, object]]) -> None:
    priorities = [int(row["candidate_priority"]) for row in group]
    if priorities != list(range(len(group))):
        raise ValueError("candidate priorities must be contiguous from zero")
    keys = [str(row["candidate_accepted_taxon_key"]) for row in group]
    if len(keys) != len(set(keys)):
        raise ValueError("complete candidate union contains duplicate taxa")
    targets = [row for row in group if row["target_candidate"]]
    if len(targets) != 1:
        raise ValueError("candidate set must contain exactly one target candidate")
    target = targets[0]
    if target["candidate_accepted_taxon_key"] != target["target_accepted_taxon_key"]:
        raise ValueError("target candidate does not match target taxon")
    if not target["safety_union_membership"]:
        raise ValueError("target candidate must belong to the safety union")
    if not all(row["target_preserved"] for row in group):
        raise ValueError("candidate set must record target preservation")
    if not all(row["included_in_complete_union"] for row in group):
        raise ValueError("candidate set is not a complete candidate union")


def _require_exact_fields(values: Mapping[str, object], expected: set[str]) -> None:
    missing = expected - set(values)
    unexpected = set(values) - expected
    if missing or unexpected:
        raise ValueError(
            "family/geography candidate fields mismatch: "
            f"missing={sorted(missing)}, unexpected={sorted(unexpected)}"
        )


def _required_text(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value.strip()


def _optional_text(value: object, *, field: str) -> str | None:
    if value is None:
        return None
    return _required_text(value, field=field)


def _boolean(value: object, *, field: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{field} must be Boolean")
    return value


def _optional_boolean(value: object, *, field: str) -> bool | None:
    if value is None:
        return None
    return _boolean(value, field=field)


def _nonnegative_int(value: object, *, field: str, maximum: int) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or not 0 <= value <= maximum
    ):
        raise ValueError(f"{field} must be an integer in [0, {maximum}]")
    return value


def _optional_positive_int(value: object, *, field: str) -> int | None:
    if value is None:
        return None
    result = _nonnegative_int(value, field=field, maximum=2**32 - 1)
    if result == 0:
        raise ValueError(f"{field} must be positive")
    return result


def _optional_score(
    value: object, *, field: str, lower: float, upper: float
) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{field} must be numeric or null")
    result = float(value)
    if not isfinite(result) or not lower <= result <= upper:
        raise ValueError(f"{field} must be finite and in [{lower}, {upper}]")
    return result


def _canonical_strings(
    value: object, *, field: str, allow_empty: bool = True
) -> list[str]:
    if isinstance(value, str | bytes) or not isinstance(value, Sequence):
        raise ValueError(f"{field} must be a sequence of strings")
    items = sorted({_required_text(item, field=field) for item in value})
    if not allow_empty and not items:
        raise ValueError(f"{field} must not be empty")
    return items


def _sha256(value: object, *, field: str) -> str:
    text = _required_text(value, field=field)
    if not _SHA256_PATTERN.fullmatch(text):
        raise ValueError(f"{field} must be a canonical SHA-256 fingerprint")
    return text


__all__ = [
    "EVIDENCE_AVAILABILITY_STATES",
    "FAMILY_GEO_CANDIDATE_FILE",
    "FAMILY_GEO_CANDIDATE_SCHEMA_VERSION",
    "build_family_geo_candidate_sets",
    "family_geo_candidate_schema",
    "validate_family_geo_candidate_sets",
    "write_family_geo_candidate_sets",
]
