from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import polars as pl

from biominer.references.schemas import (
    validate_reference_media_candidates,
    validate_reference_observations,
)


PROTOTYPE_REFERENCE_PLANNER_VERSION = "trust-first-layered-regional-v1.0.0"
PROTOTYPE_REFERENCE_EVIDENCE_SCHEMA_VERSION = (
    "prototype-reference-planner-evidence-v1.0.0"
)
PROTOTYPE_REFERENCE_REPORT_SCHEMA_VERSION = "prototype-reference-planner-report-v1.0.0"
PROTOTYPE_REFERENCE_EVIDENCE_FILE = "prototype_reference_planner_evidence.parquet"
PROTOTYPE_REFERENCE_REPORT_FILE = "prototype_reference_planner_report.json"
PROTOTYPE_REFERENCE_REPORT_MARKDOWN_FILE = "prototype_reference_planner_report.md"

TRUST_LEVELS = ("R1", "R2", "R3", "R4", "R5")
GEOGRAPHIC_LAYERS = ("A", "B", "C", "D", "E")
PROTOTYPE_VERIFICATION_STATUSES = (
    "human_verified",
    "provider_high_trust",
    "provider_supported",
    "provisional",
    "excluded",
)
PROTOTYPE_REFERENCE_ROUTES = ("adult_field", "larval", "pinned_specimen")
PLANNER_SCORE_SEMANTICS = "ordinal_priority_not_probability"
_LAYER_BUCKETS = ("AB", "C", "D", "E")
_TRUST_PRIORITY = {value: index for index, value in enumerate(TRUST_LEVELS, start=1)}
_GEOGRAPHIC_PRIORITY = {
    value: index for index, value in enumerate(GEOGRAPHIC_LAYERS, start=1)
}


@dataclass(frozen=True, slots=True)
class PrototypeReferenceQuota:
    reference_group: str
    accepted_taxon_key: str
    scientific_name: str
    requested_count: int
    route: str = "adult_field"
    life_stage: str = "adult"
    visual_domain: str = "live_field"

    def __post_init__(self) -> None:
        for field_name in (
            "reference_group",
            "accepted_taxon_key",
            "scientific_name",
            "route",
            "life_stage",
            "visual_domain",
        ):
            object.__setattr__(
                self,
                field_name,
                _required_text(getattr(self, field_name), field=field_name),
            )
        if self.route not in PROTOTYPE_REFERENCE_ROUTES:
            raise ValueError(f"unsupported prototype reference route: {self.route}")
        if isinstance(self.requested_count, bool) or not isinstance(
            self.requested_count, int
        ):
            raise TypeError("requested_count must be an integer")
        if self.requested_count <= 0:
            raise ValueError("requested_count must be positive")


@dataclass(frozen=True, slots=True)
class PrototypeReferencePlannerConfig:
    layer_mix: tuple[tuple[str, float], ...] = (
        ("AB", 0.35),
        ("C", 0.25),
        ("D", 0.30),
        ("E", 0.10),
    )
    eligible_trust_levels: tuple[str, ...] = ("R1", "R2", "R3", "R4")
    acceptable_licence_statuses: tuple[str, ...] = ("allowed", "research_only")
    eligible_download_statuses: tuple[str, ...] = ("pending", "complete")
    minimum_image_quality_score: float = 0.0
    selection_seed: int = 20260715
    policy_version: str = PROTOTYPE_REFERENCE_PLANNER_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.layer_mix, tuple):
            raise TypeError("layer_mix must be a tuple")
        keys = tuple(key for key, _ in self.layer_mix)
        if keys != _LAYER_BUCKETS:
            raise ValueError(f"layer_mix keys must be {_LAYER_BUCKETS}")
        weights = tuple(float(value) for _, value in self.layer_mix)
        if any(not math.isfinite(value) or value < 0.0 for value in weights):
            raise ValueError("layer_mix values must be finite and nonnegative")
        if not math.isclose(sum(weights), 1.0, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("layer_mix values must sum to one")
        object.__setattr__(
            self,
            "layer_mix",
            tuple(zip(keys, weights, strict=True)),
        )
        object.__setattr__(
            self,
            "eligible_trust_levels",
            _closed_tuple(
                self.eligible_trust_levels,
                allowed=TRUST_LEVELS,
                field="eligible_trust_levels",
            ),
        )
        object.__setattr__(
            self,
            "acceptable_licence_statuses",
            _text_tuple(
                self.acceptable_licence_statuses,
                field="acceptable_licence_statuses",
            ),
        )
        object.__setattr__(
            self,
            "eligible_download_statuses",
            _text_tuple(
                self.eligible_download_statuses,
                field="eligible_download_statuses",
            ),
        )
        quality = float(self.minimum_image_quality_score)
        if not math.isfinite(quality) or not 0.0 <= quality <= 1.0:
            raise ValueError("minimum_image_quality_score must be in [0, 1]")
        object.__setattr__(self, "minimum_image_quality_score", quality)
        if isinstance(self.selection_seed, bool) or not isinstance(
            self.selection_seed, int
        ):
            raise TypeError("selection_seed must be an integer")
        if not 0 <= self.selection_seed <= 2**64 - 1:
            raise ValueError("selection_seed must fit an unsigned 64-bit integer")
        object.__setattr__(
            self,
            "policy_version",
            _required_text(self.policy_version, field="policy_version"),
        )

    def payload(self) -> dict[str, object]:
        return {
            "policy_version": self.policy_version,
            "layer_mix": dict(self.layer_mix),
            "eligible_trust_levels": list(self.eligible_trust_levels),
            "acceptable_licence_statuses": list(self.acceptable_licence_statuses),
            "eligible_download_statuses": list(self.eligible_download_statuses),
            "minimum_image_quality_score": self.minimum_image_quality_score,
            "selection_seed": self.selection_seed,
            "score_semantics": PLANNER_SCORE_SEMANTICS,
        }


@dataclass(frozen=True, slots=True)
class PrototypeReferencePlanResult:
    evidence: pl.DataFrame
    selected: pl.DataFrame
    report: dict[str, Any]
    markdown: str


def prototype_reference_planner_evidence_schema() -> dict[str, pl.DataType]:
    return {
        "schema_version": pl.String,
        "planner_version": pl.String,
        "reference_group": pl.String,
        "reference_media_id": pl.String,
        "reference_observation_id": pl.String,
        "accepted_taxon_key": pl.String,
        "scientific_name": pl.String,
        "source": pl.String,
        "provider_media_id": pl.String,
        "route": pl.String,
        "life_stage": pl.String,
        "visual_domain": pl.String,
        "verification_status": pl.String,
        "verification_actor": pl.String,
        "trust_level": pl.String,
        "trust_priority": pl.UInt8,
        "trust_component": pl.Float64,
        "geographic_layer": pl.String,
        "geographic_priority": pl.UInt8,
        "geographic_component": pl.Float64,
        "no_geo_fallback": pl.Boolean,
        "image_quality_score": pl.Float64,
        "morphology_tags": pl.List(pl.String),
        "licence_policy_status": pl.String,
        "attribution_complete": pl.Boolean,
        "exact_duplicate": pl.Boolean,
        "canonical_media_id": pl.String,
        "provider_mirror_resolved": pl.Boolean,
        "eligible": pl.Boolean,
        "exclusion_reasons": pl.List(pl.String),
        "layer_target_count": pl.UInt32,
        "selected": pl.Boolean,
        "selection_rank": pl.UInt32,
        "planner_priority_score": pl.UInt64,
        "score_semantics": pl.String,
        "evidence_fingerprint": pl.String,
    }


def plan_trust_first_layered_references(
    *,
    observations: pl.DataFrame,
    media_candidates: pl.DataFrame,
    quotas: Sequence[PrototypeReferenceQuota],
    qualification_metadata: pl.DataFrame | None = None,
    config: PrototypeReferencePlannerConfig | None = None,
) -> PrototypeReferencePlanResult:
    validate_reference_observations(observations)
    validate_reference_media_candidates(media_candidates)
    effective_config = config or PrototypeReferencePlannerConfig()
    if not isinstance(effective_config, PrototypeReferencePlannerConfig):
        raise TypeError("config must be a PrototypeReferencePlannerConfig")
    quota_rows = _validate_quotas(quotas)
    qualification = _qualification_rows(qualification_metadata)
    observations_by_id = {
        str(row["reference_observation_id"]): row
        for row in observations.iter_rows(named=True)
    }
    missing_observations = sorted(
        set(media_candidates["reference_observation_id"].to_list())
        - set(observations_by_id)
    )
    if missing_observations:
        raise ValueError(
            f"prototype reference media lack observation rows: {missing_observations}"
        )

    candidates: list[dict[str, object]] = []
    for media in media_candidates.iter_rows(named=True):
        observation = observations_by_id[str(media["reference_observation_id"])]
        taxon_key = _optional_text(observation.get("accepted_taxon_key"))
        if taxon_key is None:
            continue
        metadata = qualification.get(str(media["reference_media_id"]), {})
        for quota in quota_rows:
            if quota.accepted_taxon_key != taxon_key:
                continue
            candidate = _candidate_evidence(
                quota=quota,
                observation=observation,
                media=media,
                qualification=metadata,
                config=effective_config,
            )
            if candidate["route"] == quota.route:
                candidates.append(candidate)

    _resolve_exact_duplicates(candidates, seed=effective_config.selection_seed)
    rows: list[dict[str, object]] = []
    quota_reports: list[dict[str, object]] = []
    for quota in quota_rows:
        group_rows = [
            row for row in candidates if row["reference_group"] == quota.reference_group
        ]
        targets = _allocate_layer_targets(
            quota.requested_count,
            dict(effective_config.layer_mix),
        )
        selected_rows = _select_layered_candidates(
            group_rows,
            requested=quota.requested_count,
            targets=targets,
            seed=effective_config.selection_seed,
        )
        selected_ids = {id(row) for row in selected_rows}
        ranked = sorted(
            selected_rows,
            key=lambda row: _priority_key(row, effective_config.selection_seed),
        )
        ranks = {id(row): rank for rank, row in enumerate(ranked, start=1)}
        for row in group_rows:
            bucket = _layer_bucket(str(row["geographic_layer"]))
            row["layer_target_count"] = targets[bucket]
            row["selected"] = id(row) in selected_ids
            row["selection_rank"] = ranks.get(id(row))
            row["planner_priority_score"] = _priority_score(row)
            public = {
                key: value for key, value in row.items() if not key.startswith("_")
            }
            public["evidence_fingerprint"] = ""
            rows.append(public)
        actual = Counter(
            _layer_bucket(str(row["geographic_layer"])) for row in selected_rows
        )
        quota_reports.append(
            {
                "reference_group": quota.reference_group,
                "accepted_taxon_key": quota.accepted_taxon_key,
                "route": quota.route,
                "requested_count": quota.requested_count,
                "eligible_count": sum(bool(row["eligible"]) for row in group_rows),
                "selected_count": len(selected_rows),
                "shortfall_count": quota.requested_count - len(selected_rows),
                "target_counts_by_layer_bucket": targets,
                "actual_counts_by_layer_bucket": {
                    key: actual.get(key, 0) for key in _LAYER_BUCKETS
                },
                "quota_deviations": {
                    key: actual.get(key, 0) - targets[key] for key in _LAYER_BUCKETS
                },
            }
        )

    schema = prototype_reference_planner_evidence_schema()
    evidence = pl.DataFrame(rows, schema=schema, strict=True).sort(
        ["reference_group", "selected", "selection_rank", "reference_media_id"],
        descending=[False, True, False, False],
        nulls_last=True,
    )
    evidence = evidence.with_columns(
        pl.Series(
            "evidence_fingerprint",
            [
                _sha256_json(
                    {
                        key: value
                        for key, value in row.items()
                        if key != "evidence_fingerprint"
                    }
                )
                for row in evidence.iter_rows(named=True)
            ],
            dtype=pl.String,
        )
    )
    selected = evidence.filter(pl.col("selected")).sort(
        ["reference_group", "selection_rank"]
    )
    planner_fingerprint = _sha256_json(
        {
            "configuration": effective_config.payload(),
            "quotas": [_quota_payload(quota) for quota in quota_rows],
            "evidence_fingerprints": evidence["evidence_fingerprint"].to_list(),
        }
    )
    report = _build_report(
        evidence=evidence,
        selected=selected,
        quota_reports=quota_reports,
        config=effective_config,
        planner_fingerprint=planner_fingerprint,
    )
    result = PrototypeReferencePlanResult(
        evidence=evidence,
        selected=selected,
        report=report,
        markdown=_report_markdown(report),
    )
    validate_prototype_reference_plan_result(result)
    return result


def validate_prototype_reference_plan_result(
    result: PrototypeReferencePlanResult,
) -> None:
    if not isinstance(result, PrototypeReferencePlanResult):
        raise TypeError("result must be a PrototypeReferencePlanResult")
    expected_schema = prototype_reference_planner_evidence_schema()
    if result.evidence.schema != expected_schema:
        raise ValueError("prototype planner evidence has an unexpected schema")
    if not result.selected.equals(
        result.evidence.filter(pl.col("selected")).sort(
            ["reference_group", "selection_rank"]
        )
    ):
        raise ValueError("prototype planner selected projection is inconsistent")
    for row in result.evidence.iter_rows(named=True):
        if row["schema_version"] != PROTOTYPE_REFERENCE_EVIDENCE_SCHEMA_VERSION:
            raise ValueError("prototype planner evidence schema version is invalid")
        if row["planner_version"] != PROTOTYPE_REFERENCE_PLANNER_VERSION:
            raise ValueError("prototype planner version is invalid")
        if row["trust_level"] not in TRUST_LEVELS:
            raise ValueError("prototype planner trust level is invalid")
        if row["geographic_layer"] not in GEOGRAPHIC_LAYERS:
            raise ValueError("prototype planner geographic layer is invalid")
        if row["verification_status"] not in PROTOTYPE_VERIFICATION_STATUSES:
            raise ValueError("prototype verification status is invalid")
        if row["score_semantics"] != PLANNER_SCORE_SEMANTICS:
            raise ValueError("prototype planner score semantics are invalid")
        if (
            row["verification_status"] == "human_verified"
            and not row["verification_actor"]
        ):
            raise ValueError("human_verified evidence requires an actor")
        if row["selected"] and not row["eligible"]:
            raise ValueError("ineligible prototype reference was selected")
        if row["selected"] and row["exact_duplicate"]:
            raise ValueError("exact duplicate prototype reference was selected")
        if row["selected"] and row["trust_level"] == "R5":
            raise ValueError("R5 prototype reference was selected")
        expected_fingerprint = _sha256_json(
            {key: value for key, value in row.items() if key != "evidence_fingerprint"}
        )
        if row["evidence_fingerprint"] != expected_fingerprint:
            raise ValueError("prototype planner evidence fingerprint mismatch")
    for (_,), group in result.selected.group_by("reference_group"):
        ranks = group["selection_rank"].to_list()
        if ranks != list(range(1, group.height + 1)):
            raise ValueError("prototype planner selection ranks are not contiguous")
        if group["route"].n_unique() != 1:
            raise ValueError("prototype planner mixed routes within a reference group")
        if group["reference_observation_id"].n_unique() != group.height:
            raise ValueError(
                "prototype planner selected multiple media from one observation"
            )
    summary = _mapping(result.report.get("summary"))
    if summary.get("candidate_count") != result.evidence.height:
        raise ValueError("prototype planner report candidate count is inconsistent")
    if summary.get("selected_count") != result.selected.height:
        raise ValueError("prototype planner report selected count is inconsistent")


def write_prototype_reference_plan_result(
    result: PrototypeReferencePlanResult,
    output_dir: str | Path,
) -> dict[str, Path]:
    validate_prototype_reference_plan_result(result)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    evidence_path = destination / PROTOTYPE_REFERENCE_EVIDENCE_FILE
    report_path = destination / PROTOTYPE_REFERENCE_REPORT_FILE
    markdown_path = destination / PROTOTYPE_REFERENCE_REPORT_MARKDOWN_FILE
    _write_parquet_atomic(result.evidence, evidence_path)
    _write_text_atomic(
        json.dumps(result.report, indent=2, sort_keys=True) + "\n",
        report_path,
    )
    _write_text_atomic(result.markdown, markdown_path)
    return {
        "evidence": evidence_path,
        "report": report_path,
        "markdown": markdown_path,
    }


def _candidate_evidence(
    *,
    quota: PrototypeReferenceQuota,
    observation: Mapping[str, object],
    media: Mapping[str, object],
    qualification: Mapping[str, object],
    config: PrototypeReferencePlannerConfig,
) -> dict[str, object]:
    route = _candidate_route(observation, qualification)
    life_stage = (
        _optional_text(qualification.get("life_stage"))
        or _required_text(observation.get("life_stage"), field="life_stage")
    ).casefold()
    visual_domain = (
        _optional_text(qualification.get("visual_domain"))
        or ("pinned_specimen" if route == "pinned_specimen" else "live_field")
    ).casefold()
    verification_status, trust_level, actor = _trust_evidence(
        observation=observation,
        media=media,
        qualification=qualification,
    )
    geographic_layer, no_geo_fallback = _geographic_layer(
        observation,
        qualification,
    )
    quality = _image_quality_score(media, qualification)
    attribution_complete = all(
        _optional_text(media.get(field_name)) is not None
        for field_name in (
            "creator",
            "rights_holder",
            "licence",
            "licence_uri",
            "attribution",
        )
    )
    provider_mirror_status = (
        _optional_text(qualification.get("provider_mirror_status")) or "resolved"
    ).casefold()
    provider_mirror_resolved = provider_mirror_status in {
        "resolved",
        "not_detected",
        "canonical",
    }
    reasons: list[str] = []
    if str(media["licence_policy_status"]) not in config.acceptable_licence_statuses:
        reasons.append("licence_ineligible")
    if not attribution_complete:
        reasons.append("attribution_incomplete")
    if str(media["download_status"]) not in config.eligible_download_statuses:
        reasons.append("download_status_ineligible")
    if _optional_text(media.get("media_identifier")) is None:
        reasons.append("missing_media_identifier")
    if str(media.get("media_type") or "").casefold() not in {
        "stillimage",
        "image",
    }:
        reasons.append("not_still_image")
    if media.get("exclusion_reason") is not None:
        reasons.append("provider_excluded")
    if observation.get("accepted_taxon_key") != quota.accepted_taxon_key:
        reasons.append("taxon_conflict")
    if str(observation.get("taxon_reconciliation_status")) not in {
        "accepted_key_exact",
        "accepted",
        "exact",
    }:
        reasons.append("taxon_reconciliation_inexact")
    if bool(observation.get("uncertain_taxon_match")):
        reasons.append("taxon_conflict")
    if bool(observation.get("identification_disagreement")) or bool(
        qualification.get("known_identification_disagreement")
    ):
        reasons.append("known_identification_disagreement")
    if bool(observation.get("occurrence_absent")):
        reasons.append("occurrence_absent")
    if bool(observation.get("fossil")):
        reasons.append("fossil")
    if bool(observation.get("geospatial_issue")):
        reasons.append("fatal_geospatial_issue")
    if not bool(observation.get("basis_of_record_suitable")):
        reasons.append("basis_of_record_ineligible")
    if route != quota.route:
        reasons.append("wrong_life_stage_or_visual_route")
    if visual_domain != quota.visual_domain.casefold():
        reasons.append("wrong_visual_domain")
    if trust_level not in config.eligible_trust_levels:
        reasons.append("trust_level_ineligible")
    if verification_status in {"provisional", "excluded"}:
        reasons.append(verification_status)
    if quality < config.minimum_image_quality_score:
        reasons.append("fatal_quality_issue")
    if bool(qualification.get("fatal_quality_issue")):
        reasons.append("fatal_quality_issue")
    if not provider_mirror_resolved:
        reasons.append("unresolved_provider_mirror")
    morphology_tags = _morphology_tags(observation, qualification)
    duplicate_key = _duplicate_key(media, qualification)
    trust_priority = _TRUST_PRIORITY[trust_level]
    geographic_priority = _GEOGRAPHIC_PRIORITY[geographic_layer]
    return {
        "schema_version": PROTOTYPE_REFERENCE_EVIDENCE_SCHEMA_VERSION,
        "planner_version": PROTOTYPE_REFERENCE_PLANNER_VERSION,
        "reference_group": quota.reference_group,
        "reference_media_id": str(media["reference_media_id"]),
        "reference_observation_id": str(media["reference_observation_id"]),
        "accepted_taxon_key": quota.accepted_taxon_key,
        "scientific_name": quota.scientific_name,
        "source": str(media["source"]),
        "provider_media_id": str(media["provider_media_id"]),
        "route": route,
        "life_stage": life_stage,
        "visual_domain": visual_domain,
        "verification_status": verification_status,
        "verification_actor": actor,
        "trust_level": trust_level,
        "trust_priority": trust_priority,
        "trust_component": float(6 - trust_priority),
        "geographic_layer": geographic_layer,
        "geographic_priority": geographic_priority,
        "geographic_component": float(6 - geographic_priority),
        "no_geo_fallback": no_geo_fallback,
        "image_quality_score": quality,
        "morphology_tags": morphology_tags,
        "licence_policy_status": str(media["licence_policy_status"]),
        "attribution_complete": attribution_complete,
        "exact_duplicate": False,
        "canonical_media_id": None,
        "provider_mirror_resolved": provider_mirror_resolved,
        "eligible": not reasons,
        "exclusion_reasons": sorted(set(reasons)),
        "layer_target_count": 0,
        "selected": False,
        "selection_rank": None,
        "planner_priority_score": 0,
        "score_semantics": PLANNER_SCORE_SEMANTICS,
        "evidence_fingerprint": "",
        "_duplicate_key": duplicate_key,
        "_observer_id": _optional_text(observation.get("observer_id")),
    }


def _trust_evidence(
    *,
    observation: Mapping[str, object],
    media: Mapping[str, object],
    qualification: Mapping[str, object],
) -> tuple[str, str, str | None]:
    explicit_status = _optional_text(qualification.get("verification_status"))
    actor = _optional_text(qualification.get("verified_by"))
    explicit_trust = _optional_text(qualification.get("trust_level"))
    if (
        explicit_status is not None
        and explicit_status not in PROTOTYPE_VERIFICATION_STATUSES
    ):
        raise ValueError(f"unknown prototype verification status: {explicit_status}")
    if explicit_trust is not None and explicit_trust not in TRUST_LEVELS:
        raise ValueError(f"unknown prototype trust level: {explicit_trust}")
    if explicit_status == "human_verified":
        if actor is None:
            raise ValueError("human_verified qualification requires verified_by")
        if explicit_trust not in {None, "R1"}:
            raise ValueError("human_verified qualification must use R1")
        return "human_verified", "R1", actor
    if explicit_trust == "R1":
        raise ValueError("R1 requires explicit attributable human verification")
    if explicit_status == "provider_high_trust":
        return "provider_high_trust", explicit_trust or "R2", actor
    if explicit_status == "provider_supported":
        return (
            "provider_supported",
            explicit_trust or _provider_supported_trust(observation, media),
            actor,
        )
    if explicit_status in {"provisional", "excluded"}:
        return explicit_status, explicit_trust or "R5", actor
    source = str(media.get("source") or "").casefold()
    exact_taxon = str(observation.get("taxon_reconciliation_status")) in {
        "accepted_key_exact",
        "accepted",
        "exact",
    }
    if exact_taxon and source in {"inaturalist", "gbif"}:
        return (
            "provider_supported",
            explicit_trust or _provider_supported_trust(observation, media),
            actor,
        )
    return "provisional", explicit_trust or "R5", actor


def _provider_supported_trust(
    observation: Mapping[str, object],
    media: Mapping[str, object],
) -> str:
    source = str(media.get("source") or "").casefold()
    quality = str(observation.get("identification_quality") or "").casefold()
    community = str(observation.get("community_taxon_status") or "").casefold()
    if (
        source == "inaturalist"
        and quality in {"research", "research_grade"}
        and community in {"species", "exact_species"}
        and not bool(observation.get("identification_disagreement"))
        and not bool(observation.get("captive_or_cultivated"))
    ):
        return "R3"
    return "R4"


def _candidate_route(
    observation: Mapping[str, object],
    qualification: Mapping[str, object],
) -> str:
    explicit = _optional_text(qualification.get("route"))
    if explicit is not None:
        if explicit not in PROTOTYPE_REFERENCE_ROUTES:
            raise ValueError(f"unknown prototype reference route: {explicit}")
        return explicit
    if bool(observation.get("preserved_specimen")):
        return "pinned_specimen"
    life_stage = (
        _optional_text(qualification.get("life_stage"))
        or _optional_text(observation.get("life_stage"))
        or "unknown"
    ).casefold()
    if life_stage in {"larva", "caterpillar"}:
        return "larval"
    return "adult_field" if life_stage == "adult" else "unsupported"


def _geographic_layer(
    observation: Mapping[str, object],
    qualification: Mapping[str, object],
) -> tuple[str, bool]:
    explicit = _optional_text(qualification.get("geographic_layer"))
    if explicit is not None:
        explicit = explicit.upper()
        if explicit not in GEOGRAPHIC_LAYERS:
            raise ValueError(f"unknown geographic layer: {explicit}")
        return explicit, explicit == "D" and _is_no_geo(observation)
    if bool(qualification.get("global_hard_negative")):
        return "E", _is_no_geo(observation)
    if _is_no_geo(observation):
        return "D", True
    fallback = int(observation.get("fallback_level") or 0)
    return ({0: "A", 1: "B", 2: "C"}.get(fallback, "D"), False)


def _is_no_geo(observation: Mapping[str, object]) -> bool:
    cluster = str(observation.get("geo_cluster_id") or "").casefold()
    return cluster in {"", "no_geo", "unassigned_geo"}


def _image_quality_score(
    media: Mapping[str, object],
    qualification: Mapping[str, object],
) -> float:
    explicit = qualification.get("image_quality_score")
    if explicit is not None:
        value = float(explicit)
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError("image_quality_score must be finite and in [0, 1]")
        return value
    width = int(media.get("width") or 0)
    height = int(media.get("height") or 0)
    short_side = min(width, height)
    megapixels = width * height / 1_000_000.0
    return round(
        0.5 * min(1.0, short_side / 1024.0) + 0.5 * min(1.0, megapixels / 2.0),
        6,
    )


def _morphology_tags(
    observation: Mapping[str, object],
    qualification: Mapping[str, object],
) -> list[str]:
    raw = qualification.get("morphology_tags")
    if raw is None:
        values: list[object] = []
    elif isinstance(raw, str):
        values = [raw]
    elif isinstance(raw, Sequence):
        values = list(raw)
    else:
        raise TypeError("morphology_tags must be a sequence of strings")
    for field_name in ("view", "sex"):
        value = qualification.get(field_name, observation.get(field_name))
        if _optional_text(value) is not None:
            values.append(str(value))
    return sorted({_required_text(value, field="morphology_tag") for value in values})


def _duplicate_key(
    media: Mapping[str, object],
    qualification: Mapping[str, object],
) -> str | None:
    explicit = _optional_text(qualification.get("exact_duplicate_key"))
    if explicit is not None:
        return f"explicit:{explicit}"
    checksum = _optional_text(media.get("source_checksum"))
    algorithm = _optional_text(media.get("source_checksum_algorithm"))
    if checksum is not None and algorithm is not None:
        return f"checksum:{algorithm.casefold()}:{checksum.casefold()}"
    return None


def _resolve_exact_duplicates(rows: list[dict[str, object]], *, seed: int) -> None:
    groups: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        duplicate_key = row.get("_duplicate_key")
        if duplicate_key is not None:
            groups[(str(row["reference_group"]), str(duplicate_key))].append(row)
    for group in groups.values():
        if len(group) < 2:
            continue
        canonical = min(group, key=lambda row: _priority_key(row, seed))
        canonical_id = str(canonical["reference_media_id"])
        canonical["canonical_media_id"] = canonical_id
        for row in group:
            row["canonical_media_id"] = canonical_id
            if row is canonical:
                continue
            row["exact_duplicate"] = True
            row["eligible"] = False
            row["exclusion_reasons"] = sorted(
                set(row["exclusion_reasons"]) | {"exact_duplicate"}
            )


def _allocate_layer_targets(
    requested: int,
    mix: Mapping[str, float],
) -> dict[str, int]:
    exact = {key: requested * mix[key] for key in _LAYER_BUCKETS}
    targets = {key: math.floor(exact[key]) for key in _LAYER_BUCKETS}
    remaining = requested - sum(targets.values())
    order = sorted(
        _LAYER_BUCKETS,
        key=lambda key: (-(exact[key] - targets[key]), _LAYER_BUCKETS.index(key)),
    )
    for key in order[:remaining]:
        targets[key] += 1
    return targets


def _select_layered_candidates(
    rows: list[dict[str, object]],
    *,
    requested: int,
    targets: Mapping[str, int],
    seed: int,
) -> list[dict[str, object]]:
    selected: list[dict[str, object]] = []
    selected_ids: set[int] = set()
    selected_observation_ids: set[str] = set()
    morphology: Counter[str] = Counter()
    observers: Counter[str] = Counter()
    sources: Counter[str] = Counter()
    for bucket in _LAYER_BUCKETS:
        for _ in range(targets[bucket]):
            available = [
                row
                for row in rows
                if bool(row["eligible"])
                and id(row) not in selected_ids
                and str(row["reference_observation_id"])
                not in selected_observation_ids
                and _layer_bucket(str(row["geographic_layer"])) == bucket
            ]
            if not available:
                break
            chosen = min(
                available,
                key=lambda row: _dynamic_priority_key(
                    row,
                    morphology=morphology,
                    observers=observers,
                    sources=sources,
                    seed=seed,
                ),
            )
            _record_selection(
                chosen,
                selected,
                selected_ids,
                selected_observation_ids,
                morphology,
                observers,
                sources,
            )
    while len(selected) < requested:
        available = [
            row
            for row in rows
            if bool(row["eligible"])
            and id(row) not in selected_ids
            and str(row["reference_observation_id"])
            not in selected_observation_ids
        ]
        if not available:
            break
        chosen = min(
            available,
            key=lambda row: _dynamic_priority_key(
                row,
                morphology=morphology,
                observers=observers,
                sources=sources,
                seed=seed,
            ),
        )
        _record_selection(
            chosen,
            selected,
            selected_ids,
            selected_observation_ids,
            morphology,
            observers,
            sources,
        )
    return selected


def _record_selection(
    row: dict[str, object],
    selected: list[dict[str, object]],
    selected_ids: set[int],
    selected_observation_ids: set[str],
    morphology: Counter[str],
    observers: Counter[str],
    sources: Counter[str],
) -> None:
    selected.append(row)
    selected_ids.add(id(row))
    selected_observation_ids.add(str(row["reference_observation_id"]))
    morphology.update(str(value) for value in row["morphology_tags"])
    observer = row.get("_observer_id")
    if observer:
        observers[str(observer)] += 1
    sources[str(row["source"])] += 1


def _dynamic_priority_key(
    row: Mapping[str, object],
    *,
    morphology: Counter[str],
    observers: Counter[str],
    sources: Counter[str],
    seed: int,
) -> tuple[object, ...]:
    tags = [str(value) for value in row["morphology_tags"]]
    observer = row.get("_observer_id")
    return (
        int(row["trust_priority"]),
        -float(row["image_quality_score"]),
        int(row["geographic_priority"]),
        sum(morphology[tag] for tag in tags),
        1 if not tags else 0,
        observers[str(observer)] if observer else 2**31,
        sources[str(row["source"])],
        _seeded_tiebreak(seed, str(row["reference_media_id"])),
        str(row["reference_media_id"]),
    )


def _priority_key(row: Mapping[str, object], seed: int) -> tuple[object, ...]:
    return (
        int(row["trust_priority"]),
        -float(row["image_quality_score"]),
        int(row["geographic_priority"]),
        _seeded_tiebreak(seed, str(row["reference_media_id"])),
        str(row["reference_media_id"]),
    )


def _priority_score(row: Mapping[str, object]) -> int:
    quality_penalty = int(round((1.0 - float(row["image_quality_score"])) * 1_000_000))
    return (
        int(row["trust_priority"]) * 10**15
        + quality_penalty * 10**8
        + int(row["geographic_priority"]) * 10**6
    )


def _layer_bucket(layer: str) -> str:
    return "AB" if layer in {"A", "B"} else layer


def _build_report(
    *,
    evidence: pl.DataFrame,
    selected: pl.DataFrame,
    quota_reports: list[dict[str, object]],
    config: PrototypeReferencePlannerConfig,
    planner_fingerprint: str,
) -> dict[str, Any]:
    selected_rows = selected.to_dicts()
    eligible = evidence.filter(pl.col("eligible"))
    layer_counts = Counter(str(row["geographic_layer"]) for row in selected_rows)
    bucket_counts = Counter(
        _layer_bucket(key) for key, value in layer_counts.items() for _ in range(value)
    )
    selected_count = len(selected_rows)
    return {
        "schema_version": PROTOTYPE_REFERENCE_REPORT_SCHEMA_VERSION,
        "planner_version": config.policy_version,
        "planner_fingerprint": planner_fingerprint,
        "bank_status": "prototype_only",
        "score_semantics": PLANNER_SCORE_SEMANTICS,
        "score_is_probability": False,
        "provider_supported_is_human_verified": False,
        "configuration": config.payload(),
        "summary": {
            "candidate_count": evidence.height,
            "eligible_count": eligible.height,
            "selected_count": selected_count,
            "requested_count": sum(
                int(row["requested_count"]) for row in quota_reports
            ),
            "shortfall_count": sum(
                int(row["shortfall_count"]) for row in quota_reports
            ),
            "exact_duplicate_exclusion_count": evidence.filter(
                pl.col("exact_duplicate")
            ).height,
            "no_geo_fallback_selected_count": selected.filter(
                pl.col("no_geo_fallback")
            ).height,
        },
        "eligible_trust_distribution": _frame_counter(eligible, "trust_level"),
        "selected_trust_distribution": _frame_counter(selected, "trust_level"),
        "selected_verification_distribution": _frame_counter(
            selected, "verification_status"
        ),
        "selected_geographic_layer_distribution": {
            key: layer_counts.get(key, 0) for key in GEOGRAPHIC_LAYERS
        },
        "selected_layer_bucket_distribution": {
            key: bucket_counts.get(key, 0) for key in _LAYER_BUCKETS
        },
        "actual_layer_bucket_proportions": {
            key: (bucket_counts.get(key, 0) / selected_count if selected_count else 0.0)
            for key in _LAYER_BUCKETS
        },
        "target_layer_bucket_proportions": dict(config.layer_mix),
        "quota_results": quota_reports,
        "unresolved_risks": [
            "provider-supported labels have not all received independent human taxonomic verification",
            "pre-download exact-duplicate detection is limited to provider checksums and explicit metadata",
            "provider mirror detection remains incomplete until media acquisition and deduplication",
        ],
    }


def _report_markdown(report: Mapping[str, object]) -> str:
    summary = _mapping(report.get("summary"))
    lines = [
        "# Trust-first layered prototype reference plan",
        "",
        "This is prototype support evidence. Provider-supported references are not human verified unless an attributable human actor is recorded.",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| Candidates | {summary.get('candidate_count', 0)} |",
        f"| Eligible | {summary.get('eligible_count', 0)} |",
        f"| Selected | {summary.get('selected_count', 0)} |",
        f"| Shortfall | {summary.get('shortfall_count', 0)} |",
        "",
        "Scores are ordinal planner priorities, not probabilities.",
    ]
    return "\n".join(lines) + "\n"


def _validate_quotas(
    quotas: Sequence[PrototypeReferenceQuota],
) -> tuple[PrototypeReferenceQuota, ...]:
    if not isinstance(quotas, Sequence) or isinstance(quotas, (str, bytes)):
        raise TypeError("quotas must be a sequence")
    values = tuple(quotas)
    if not values:
        raise ValueError("quotas must not be empty")
    if not all(isinstance(value, PrototypeReferenceQuota) for value in values):
        raise TypeError("quotas must contain PrototypeReferenceQuota values")
    groups = [value.reference_group for value in values]
    if len(set(groups)) != len(groups):
        raise ValueError("prototype reference groups must be unique")
    return tuple(sorted(values, key=lambda value: value.reference_group))


def _qualification_rows(
    frame: pl.DataFrame | None,
) -> dict[str, Mapping[str, object]]:
    if frame is None:
        return {}
    if not isinstance(frame, pl.DataFrame):
        raise TypeError("qualification_metadata must be a Polars DataFrame")
    if "reference_media_id" not in frame.columns:
        raise ValueError("qualification_metadata is missing reference_media_id")
    if frame["reference_media_id"].n_unique() != frame.height:
        raise ValueError("qualification_metadata contains duplicate media IDs")
    return {str(row["reference_media_id"]): row for row in frame.iter_rows(named=True)}


def _quota_payload(quota: PrototypeReferenceQuota) -> dict[str, object]:
    return {
        "reference_group": quota.reference_group,
        "accepted_taxon_key": quota.accepted_taxon_key,
        "scientific_name": quota.scientific_name,
        "requested_count": quota.requested_count,
        "route": quota.route,
        "life_stage": quota.life_stage,
        "visual_domain": quota.visual_domain,
    }


def _frame_counter(frame: pl.DataFrame, field: str) -> dict[str, int]:
    return dict(sorted(Counter(str(value) for value in frame[field]).items()))


def _seeded_tiebreak(seed: int, media_id: str) -> str:
    return hashlib.sha256(f"{seed}:{media_id}".encode()).hexdigest()


def _sha256_json(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        default=_json_default,
    ).encode()
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _json_default(value: object) -> str:
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    raise TypeError(f"unsupported JSON value: {type(value).__name__}")


def _closed_tuple(
    values: tuple[str, ...],
    *,
    allowed: tuple[str, ...],
    field: str,
) -> tuple[str, ...]:
    result = _text_tuple(values, field=field)
    unknown = sorted(set(result) - set(allowed))
    if unknown:
        raise ValueError(f"{field} contains unsupported values: {unknown}")
    return result


def _text_tuple(values: tuple[str, ...], *, field: str) -> tuple[str, ...]:
    if not isinstance(values, tuple):
        raise TypeError(f"{field} must be a tuple")
    result = tuple(
        dict.fromkeys(_required_text(value, field=field) for value in values)
    )
    if not result:
        raise ValueError(f"{field} must not be empty")
    return result


def _required_text(value: object, *, field: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{field} must be nonblank")
    return text


def _optional_text(value: object) -> str | None:
    text = str(value).strip() if value is not None else ""
    return text or None


def _mapping(value: object) -> Mapping[str, object]:
    return value if isinstance(value, Mapping) else {}


def _write_parquet_atomic(frame: pl.DataFrame, path: Path) -> None:
    temporary = path.with_name(path.name + ".tmp")
    try:
        frame.write_parquet(temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_text_atomic(text: str, path: Path) -> None:
    temporary = path.with_name(path.name + ".tmp")
    try:
        temporary.write_text(text, encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


__all__ = [
    "GEOGRAPHIC_LAYERS",
    "PLANNER_SCORE_SEMANTICS",
    "PROTOTYPE_REFERENCE_EVIDENCE_FILE",
    "PROTOTYPE_REFERENCE_EVIDENCE_SCHEMA_VERSION",
    "PROTOTYPE_REFERENCE_PLANNER_VERSION",
    "PROTOTYPE_REFERENCE_REPORT_FILE",
    "PROTOTYPE_REFERENCE_REPORT_MARKDOWN_FILE",
    "PROTOTYPE_REFERENCE_REPORT_SCHEMA_VERSION",
    "PROTOTYPE_REFERENCE_ROUTES",
    "PROTOTYPE_VERIFICATION_STATUSES",
    "TRUST_LEVELS",
    "PrototypeReferencePlanResult",
    "PrototypeReferencePlannerConfig",
    "PrototypeReferenceQuota",
    "plan_trust_first_layered_references",
    "prototype_reference_planner_evidence_schema",
    "validate_prototype_reference_plan_result",
    "write_prototype_reference_plan_result",
]
