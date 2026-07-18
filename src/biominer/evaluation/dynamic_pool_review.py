"""Sampling and review contracts for dynamic-pool Flickr candidates.

The audit frame describes provisional model evidence.  Its taxonomic fields
are candidate strata, never reviewed identity, and its raw scores are never
probabilities.  Representative sampling, targeted failure discovery and
release review are separate downstream contracts.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math
import re

import polars as pl

from biominer.common.semantic_hash import canonical_semantic_fingerprint


DYNAMIC_POOL_AUDIT_FRAME_SCHEMA_VERSION = "dynamic-pool-audit-frame-v1.0.0"
DYNAMIC_POOL_AUDIT_FRAME_FILE = "dynamic_pool_audit_frame.parquet"

QUERY_TIERS = frozenset({"T1", "T2", "T3", "T4", "T5"})
RAW_SCORE_SEMANTICS = "raw_model_evidence_not_probability"

_SHA256_PATTERN = re.compile(r"sha256:[0-9a-f]{64}\Z")
_REQUIRED_INPUT_FIELDS = frozenset(
    {
        "sampling_unit_id",
        "source_record_hash",
        "source_artifact_fingerprint",
        "flickr_photo_id",
        "organism_unit_id",
        "candidate_family_accepted_taxon_key",
        "candidate_family_scientific_name",
        "candidate_genus_accepted_taxon_key",
        "candidate_genus_scientific_name",
        "candidate_species_accepted_taxon_key",
        "candidate_species_scientific_name",
        "geographic_cluster_id",
        "no_geo",
        "primary_query_tier",
        "raw_fusion_score",
        "raw_competitor_margin",
        "pool_disagreement",
        "route",
        "visual_domain",
        "subject_area_ratio",
        "owner_group_id",
        "duplicate_group_id",
        "observation_group_id",
        "final_release_candidate",
    }
)

DYNAMIC_POOL_AUDIT_FRAME_SCHEMA: dict[str, pl.DataType] = {
    "schema_version": pl.String,
    "strata_policy_fingerprint": pl.String,
    "frame_fingerprint": pl.String,
    "audit_unit_fingerprint": pl.String,
    "sampling_unit_id": pl.String,
    "source_record_hash": pl.String,
    "source_artifact_fingerprint": pl.String,
    "flickr_photo_id": pl.String,
    "organism_unit_id": pl.String,
    "candidate_family_accepted_taxon_key": pl.String,
    "candidate_family_scientific_name": pl.String,
    "candidate_genus_accepted_taxon_key": pl.String,
    "candidate_genus_scientific_name": pl.String,
    "candidate_species_accepted_taxon_key": pl.String,
    "candidate_species_scientific_name": pl.String,
    "geographic_cluster_id": pl.String,
    "no_geo": pl.Boolean,
    "geography_stratum": pl.String,
    "primary_query_tier": pl.String,
    "raw_fusion_score": pl.Float64,
    "raw_score_band": pl.String,
    "raw_competitor_margin": pl.Float64,
    "raw_margin_band": pl.String,
    "pool_disagreement": pl.Float64,
    "pool_disagreement_band": pl.String,
    "route": pl.String,
    "visual_domain": pl.String,
    "route_domain_stratum": pl.String,
    "subject_area_ratio": pl.Float64,
    "subject_size_band": pl.String,
    "owner_group_id": pl.String,
    "duplicate_group_id": pl.String,
    "observation_group_id": pl.String,
    "independence_group_fingerprint": pl.String,
    "analysis_stratum_id": pl.String,
    "analysis_stratum_fingerprint": pl.String,
    "score_semantics": pl.String,
    "probability_available": pl.Boolean,
    "final_release_candidate": pl.Boolean,
}


@dataclass(frozen=True, slots=True)
class DynamicPoolAuditStrataPolicy:
    """Immutable cut points for audit analysis, not decision thresholds."""

    schema_version: str = "dynamic-pool-audit-strata-policy-v1.0.0"
    score_cutpoints: tuple[float, ...] = (0.25, 0.50, 0.75)
    margin_cutpoints: tuple[float, ...] = (0.0, 0.05, 0.15)
    pool_disagreement_cutpoints: tuple[float, ...] = (0.05, 0.15)
    subject_area_cutpoints: tuple[float, ...] = (0.02, 0.10, 0.30)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "schema_version",
            _required_text(self.schema_version, field="schema_version"),
        )
        for field in (
            "score_cutpoints",
            "margin_cutpoints",
            "pool_disagreement_cutpoints",
            "subject_area_cutpoints",
        ):
            values = _cutpoints(getattr(self, field), field=field)
            if field in {"pool_disagreement_cutpoints", "subject_area_cutpoints"} and (
                values and values[0] < 0.0
            ):
                raise ValueError(f"{field} cannot contain negative values")
            if field == "subject_area_cutpoints" and values and values[-1] > 1.0:
                raise ValueError("subject_area_cutpoints cannot exceed one")
            object.__setattr__(self, field, values)

    @property
    def fingerprint(self) -> str:
        return canonical_semantic_fingerprint(
            {
                "schema_version": self.schema_version,
                "score_cutpoints": self.score_cutpoints,
                "margin_cutpoints": self.margin_cutpoints,
                "pool_disagreement_cutpoints": self.pool_disagreement_cutpoints,
                "subject_area_cutpoints": self.subject_area_cutpoints,
            }
        )


def empty_dynamic_pool_audit_frame() -> pl.DataFrame:
    return pl.DataFrame(schema=DYNAMIC_POOL_AUDIT_FRAME_SCHEMA)


def build_dynamic_pool_audit_frame(
    candidates: Sequence[Mapping[str, object]],
    *,
    policy: DynamicPoolAuditStrataPolicy | None = None,
) -> pl.DataFrame:
    """Build deterministic audit strata without creating reviewed labels."""

    selected_policy = policy or DynamicPoolAuditStrataPolicy()
    if not isinstance(selected_policy, DynamicPoolAuditStrataPolicy):
        raise TypeError("policy must be a DynamicPoolAuditStrataPolicy")
    normalized = [
        _normalize_candidate(candidate, policy=selected_policy)
        for candidate in candidates
    ]
    unit_ids = [row["sampling_unit_id"] for row in normalized]
    if len(unit_ids) != len(set(unit_ids)):
        raise ValueError("sampling_unit_id must be unique")
    normalized.sort(key=lambda row: str(row["sampling_unit_id"]))
    frame_fingerprint = canonical_semantic_fingerprint(
        {
            "schema_version": DYNAMIC_POOL_AUDIT_FRAME_SCHEMA_VERSION,
            "strata_policy_fingerprint": selected_policy.fingerprint,
            "audit_unit_fingerprints": [
                row["audit_unit_fingerprint"] for row in normalized
            ],
        }
    )
    for row in normalized:
        row["frame_fingerprint"] = frame_fingerprint
    frame = pl.DataFrame(
        normalized,
        schema=DYNAMIC_POOL_AUDIT_FRAME_SCHEMA,
        strict=True,
    )
    validate_dynamic_pool_audit_frame(frame)
    return frame


def validate_dynamic_pool_audit_frame(frame: pl.DataFrame) -> None:
    """Validate schema, identities and semantic fingerprints fail closed."""

    if not isinstance(frame, pl.DataFrame):
        raise TypeError("frame must be a Polars DataFrame")
    if frame.schema != DYNAMIC_POOL_AUDIT_FRAME_SCHEMA:
        raise ValueError("dynamic-pool audit frame schema does not match contract")
    if not frame.height:
        return
    if set(frame["schema_version"].to_list()) != {
        DYNAMIC_POOL_AUDIT_FRAME_SCHEMA_VERSION
    }:
        raise ValueError("unsupported dynamic-pool audit frame schema version")
    if frame["sampling_unit_id"].n_unique() != frame.height:
        raise ValueError("sampling_unit_id must be unique")
    if frame.filter(
        pl.any_horizontal(
            pl.col(field).is_null() | (pl.col(field).str.strip_chars() == "")
            for field in (
                "owner_group_id",
                "duplicate_group_id",
                "observation_group_id",
            )
        )
    ).height:
        raise ValueError("owner, duplicate and observation groups must be complete")
    if frame.filter(pl.col("score_semantics") != RAW_SCORE_SEMANTICS).height:
        raise ValueError("raw score semantics must remain explicit")
    if frame.filter(pl.col("probability_available")).height:
        raise ValueError("raw audit scores cannot be marked as probabilities")
    fingerprints = set(frame["frame_fingerprint"].to_list())
    policy_fingerprints = set(frame["strata_policy_fingerprint"].to_list())
    if len(fingerprints) != 1 or len(policy_fingerprints) != 1:
        raise ValueError("audit frame must have one frame and policy fingerprint")
    rows = frame.sort("sampling_unit_id").to_dicts()
    expected = canonical_semantic_fingerprint(
        {
            "schema_version": DYNAMIC_POOL_AUDIT_FRAME_SCHEMA_VERSION,
            "strata_policy_fingerprint": next(iter(policy_fingerprints)),
            "audit_unit_fingerprints": [row["audit_unit_fingerprint"] for row in rows],
        }
    )
    if fingerprints != {expected}:
        raise ValueError("dynamic-pool audit frame fingerprint mismatch")


def _normalize_candidate(
    candidate: Mapping[str, object],
    *,
    policy: DynamicPoolAuditStrataPolicy,
) -> dict[str, object]:
    if not isinstance(candidate, Mapping):
        raise TypeError("each candidate must be a mapping")
    missing = sorted(_REQUIRED_INPUT_FIELDS - set(candidate))
    if missing:
        raise ValueError(f"dynamic-pool audit candidate missing fields: {missing}")
    text_fields = (
        "sampling_unit_id",
        "flickr_photo_id",
        "organism_unit_id",
        "candidate_family_accepted_taxon_key",
        "candidate_family_scientific_name",
        "candidate_genus_accepted_taxon_key",
        "candidate_genus_scientific_name",
        "candidate_species_accepted_taxon_key",
        "candidate_species_scientific_name",
        "route",
        "visual_domain",
        "owner_group_id",
        "duplicate_group_id",
        "observation_group_id",
    )
    values = {
        field: _required_text(candidate[field], field=field) for field in text_fields
    }
    source_record_hash = _sha256(
        candidate["source_record_hash"], field="source_record_hash"
    )
    source_artifact_fingerprint = _sha256(
        candidate["source_artifact_fingerprint"],
        field="source_artifact_fingerprint",
    )
    no_geo = _required_bool(candidate["no_geo"], field="no_geo")
    geographic_cluster_id = _optional_text(
        candidate["geographic_cluster_id"], field="geographic_cluster_id"
    )
    if no_geo and geographic_cluster_id is not None:
        raise ValueError("no_geo candidates cannot claim a geographic cluster")
    if not no_geo and geographic_cluster_id is None:
        raise ValueError("georeferenced candidates require geographic_cluster_id")
    geography_stratum = "no_geo" if no_geo else f"geo:{geographic_cluster_id}"
    query_tier = _required_text(
        candidate["primary_query_tier"], field="primary_query_tier"
    ).upper()
    if query_tier not in QUERY_TIERS:
        raise ValueError(f"unsupported primary_query_tier: {query_tier}")
    raw_score = _finite_float(candidate["raw_fusion_score"], field="raw_fusion_score")
    margin = _finite_float(
        candidate["raw_competitor_margin"], field="raw_competitor_margin"
    )
    disagreement = _optional_finite_float(
        candidate["pool_disagreement"], field="pool_disagreement"
    )
    if disagreement is not None and disagreement < 0.0:
        raise ValueError("pool_disagreement cannot be negative")
    subject_area = _finite_float(
        candidate["subject_area_ratio"], field="subject_area_ratio"
    )
    if not 0.0 <= subject_area <= 1.0:
        raise ValueError("subject_area_ratio must be within [0, 1]")
    final_release_candidate = _required_bool(
        candidate["final_release_candidate"], field="final_release_candidate"
    )
    route_domain = f"{values['route']}|{values['visual_domain']}"
    stratum_values = {
        "candidate_family_accepted_taxon_key": values[
            "candidate_family_accepted_taxon_key"
        ],
        "candidate_genus_accepted_taxon_key": values[
            "candidate_genus_accepted_taxon_key"
        ],
        "candidate_species_accepted_taxon_key": values[
            "candidate_species_accepted_taxon_key"
        ],
        "geography_stratum": geography_stratum,
        "primary_query_tier": query_tier,
        "raw_score_band": _band(raw_score, policy.score_cutpoints),
        "raw_margin_band": _band(margin, policy.margin_cutpoints),
        "pool_disagreement_band": (
            "unavailable"
            if disagreement is None
            else _band(disagreement, policy.pool_disagreement_cutpoints)
        ),
        "route_domain_stratum": route_domain,
        "subject_size_band": _band(subject_area, policy.subject_area_cutpoints),
    }
    analysis_stratum_fingerprint = canonical_semantic_fingerprint(stratum_values)
    independence_group_fingerprint = canonical_semantic_fingerprint(
        {
            "owner_group_id": values["owner_group_id"],
            "duplicate_group_id": values["duplicate_group_id"],
            "observation_group_id": values["observation_group_id"],
        }
    )
    semantic_values: dict[str, object] = {
        **values,
        "source_record_hash": source_record_hash,
        "source_artifact_fingerprint": source_artifact_fingerprint,
        "geographic_cluster_id": geographic_cluster_id,
        "no_geo": no_geo,
        "primary_query_tier": query_tier,
        "raw_fusion_score": raw_score,
        "raw_competitor_margin": margin,
        "pool_disagreement": disagreement,
        "subject_area_ratio": subject_area,
        "final_release_candidate": final_release_candidate,
        **stratum_values,
        "independence_group_fingerprint": independence_group_fingerprint,
        "analysis_stratum_fingerprint": analysis_stratum_fingerprint,
        "strata_policy_fingerprint": policy.fingerprint,
    }
    audit_unit_fingerprint = canonical_semantic_fingerprint(
        {
            "schema_version": DYNAMIC_POOL_AUDIT_FRAME_SCHEMA_VERSION,
            "candidate": semantic_values,
        }
    )
    return {
        "schema_version": DYNAMIC_POOL_AUDIT_FRAME_SCHEMA_VERSION,
        "strata_policy_fingerprint": policy.fingerprint,
        "frame_fingerprint": "",
        "audit_unit_fingerprint": audit_unit_fingerprint,
        **values,
        "source_record_hash": source_record_hash,
        "source_artifact_fingerprint": source_artifact_fingerprint,
        "geographic_cluster_id": geographic_cluster_id,
        "no_geo": no_geo,
        "geography_stratum": geography_stratum,
        "primary_query_tier": query_tier,
        "raw_fusion_score": raw_score,
        "raw_score_band": stratum_values["raw_score_band"],
        "raw_competitor_margin": margin,
        "raw_margin_band": stratum_values["raw_margin_band"],
        "pool_disagreement": disagreement,
        "pool_disagreement_band": stratum_values["pool_disagreement_band"],
        "route_domain_stratum": route_domain,
        "subject_area_ratio": subject_area,
        "subject_size_band": stratum_values["subject_size_band"],
        "independence_group_fingerprint": independence_group_fingerprint,
        "analysis_stratum_id": (
            f"dynamic-pool-audit-stratum:{analysis_stratum_fingerprint.removeprefix('sha256:')}"
        ),
        "analysis_stratum_fingerprint": analysis_stratum_fingerprint,
        "score_semantics": RAW_SCORE_SEMANTICS,
        "probability_available": False,
        "final_release_candidate": final_release_candidate,
    }


def _required_text(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be nonempty text")
    return value.strip()


def _optional_text(value: object, *, field: str) -> str | None:
    if value is None:
        return None
    return _required_text(value, field=field)


def _sha256(value: object, *, field: str) -> str:
    text = _required_text(value, field=field)
    if not _SHA256_PATTERN.fullmatch(text):
        raise ValueError(f"{field} must be a canonical sha256 fingerprint")
    return text


def _required_bool(value: object, *, field: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{field} must be a boolean")
    return value


def _finite_float(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field} must be finite")
    return result


def _optional_finite_float(value: object, *, field: str) -> float | None:
    if value is None:
        return None
    return _finite_float(value, field=field)


def _cutpoints(values: object, *, field: str) -> tuple[float, ...]:
    if not isinstance(values, tuple):
        raise TypeError(f"{field} must be a tuple")
    normalized = tuple(_finite_float(value, field=field) for value in values)
    if tuple(sorted(set(normalized))) != normalized:
        raise ValueError(f"{field} must be unique and strictly increasing")
    return normalized


def _band(value: float, cutpoints: tuple[float, ...]) -> str:
    for index, upper in enumerate(cutpoints):
        if value < upper:
            return f"band_{index:02d}_lt_{upper:g}"
    return f"band_{len(cutpoints):02d}_gte_{cutpoints[-1]:g}" if cutpoints else "all"


__all__ = [
    "DYNAMIC_POOL_AUDIT_FRAME_FILE",
    "DYNAMIC_POOL_AUDIT_FRAME_SCHEMA",
    "DYNAMIC_POOL_AUDIT_FRAME_SCHEMA_VERSION",
    "RAW_SCORE_SEMANTICS",
    "DynamicPoolAuditStrataPolicy",
    "build_dynamic_pool_audit_frame",
    "empty_dynamic_pool_audit_frame",
    "validate_dynamic_pool_audit_frame",
]
