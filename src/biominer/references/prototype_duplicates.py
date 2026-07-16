from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
import hashlib
import json
import re
from typing import Any
from uuid import uuid4

import polars as pl

from biominer.references.deduplication import (
    ReferenceMediaDeduplicationConfig,
    ReferenceMediaDeduplicationResult,
    deduplicate_reference_media,
    validate_reference_media_deduplication_result,
)
from biominer.references.prototype_acquisition import (
    validate_prototype_reference_selections,
)
from biominer.references.prototype_download import validate_prototype_download_inputs
from biominer.references.schemas import (
    REFERENCE_OBSERVATIONS_SCHEMA_VERSION,
    reference_observations_frame,
    validate_reference_media_objects,
    validate_reference_observations,
)
from biominer.storage.cloud import CloudStorage
from biominer.storage.paths import build_report_uri, safe_path_component
from biominer.storage.uri import join_uri


PROTOTYPE_DUPLICATE_RESOLUTION_VERSION = "prototype-duplicate-resolution-v1.0.0"
PROTOTYPE_IDENTITY_GROUPS_SCHEMA_VERSION = "prototype-identity-groups-v1.0.0"
PROTOTYPE_IDENTITY_GROUPS_FILE = "prototype_reference_identity_groups.parquet"
PROTOTYPE_DUPLICATE_RESOLUTION_REPORT_FILE = (
    "prototype_duplicate_resolution_report.json"
)
PROTOTYPE_DUPLICATE_RESOLUTION_SUMMARY_FILE = (
    "prototype_duplicate_resolution_summary.md"
)

_GROUP_ID_PATTERN = re.compile(r"prototype-[a-z-]+-group:[0-9a-f]{32}\Z")


@dataclass(frozen=True, slots=True)
class PrototypeDuplicateResolutionResult:
    deduplication: ReferenceMediaDeduplicationResult
    identity_groups: pl.DataFrame
    observations: pl.DataFrame
    report: dict[str, Any]
    markdown: str


def prototype_identity_group_schema() -> dict[str, pl.DataType]:
    return {
        "schema_version": pl.String,
        "reference_media_id": pl.String,
        "reference_observation_id": pl.String,
        "source": pl.String,
        "provider_media_id": pl.String,
        "duplicate_group_id": pl.String,
        "duplicate_type": pl.String,
        "canonical_reference_media_id": pl.String,
        "is_canonical": pl.Boolean,
        "resolution_status": pl.String,
        "support_disposition": pl.String,
        "exact_hash_group_id": pl.String,
        "perceptual_duplicate_group_id": pl.String,
        "observation_group_id": pl.String,
        "burst_group_id": pl.String,
        "owner_group_id": pl.String,
        "photographer_group_id": pl.String,
        "provider_mirror_group_id": pl.String,
        "owner_evidence_available": pl.Boolean,
        "photographer_evidence_available": pl.Boolean,
        "identity_fingerprint": pl.String,
    }


def compile_prototype_deduplication_observations(
    *,
    selections: pl.DataFrame,
    media_candidates: pl.DataFrame,
    biological_observations: Sequence[pl.DataFrame],
    visual_domain_manifest: Mapping[str, object],
) -> pl.DataFrame:
    validate_prototype_download_inputs(selections, media_candidates)
    if not biological_observations:
        raise ValueError("biological observation inputs must not be empty")
    for frame in biological_observations:
        validate_reference_observations(frame)
    biological = pl.concat(list(biological_observations), how="vertical").sort(
        ["source", "source_observation_id"]
    )
    if biological["reference_observation_id"].n_unique() != biological.height:
        raise ValueError("biological observation inputs contain duplicate identities")

    selected_ids = set(selections["reference_observation_id"].to_list())
    selected_biological_ids = set(
        selections.filter(pl.col("candidate_scope_type") == "accepted_taxon")[
            "reference_observation_id"
        ].to_list()
    )
    selected_biological = biological.filter(
        pl.col("reference_observation_id").is_in(selected_biological_ids)
    )
    if set(selected_biological["reference_observation_id"].to_list()) != (
        selected_biological_ids
    ):
        missing = sorted(
            selected_biological_ids
            - set(selected_biological["reference_observation_id"].to_list())
        )
        raise ValueError(
            f"prototype selections lack biological observation metadata: {missing}"
        )

    visual_candidates = _visual_candidates_by_media_id(visual_domain_manifest)
    rows = selected_biological.to_dicts()
    for selection in selections.filter(
        pl.col("candidate_scope_type") == "visual_domain"
    ).iter_rows(named=True):
        media_id = str(selection["reference_media_id"])
        raw = visual_candidates.get(media_id)
        if raw is None:
            raise ValueError(
                f"prototype visual selection lacks observation metadata: {media_id}"
            )
        rows.append(_visual_observation(selection, raw, visual_domain_manifest))
    observations = reference_observations_frame(rows)
    if set(observations["reference_observation_id"].to_list()) != selected_ids:
        raise ValueError("prototype deduplication observations differ from selections")
    return observations


def resolve_prototype_duplicates(
    *,
    selections: pl.DataFrame,
    media_objects: pl.DataFrame,
    media_candidates: pl.DataFrame,
    biological_observations: Sequence[pl.DataFrame],
    visual_domain_manifest: Mapping[str, object],
    config: ReferenceMediaDeduplicationConfig | None = None,
    generated_at: str | datetime | None = None,
) -> PrototypeDuplicateResolutionResult:
    validate_prototype_reference_selections(selections)
    validate_reference_media_objects(media_objects)
    observations = compile_prototype_deduplication_observations(
        selections=selections,
        media_candidates=media_candidates,
        biological_observations=biological_observations,
        visual_domain_manifest=visual_domain_manifest,
    )
    timestamp = _utc_datetime(generated_at or datetime.now(UTC))
    deduplication = deduplicate_reference_media(
        media_objects,
        media_candidates,
        observations,
        config=config,
        generated_at=timestamp,
    )
    identity_groups = _build_identity_groups(
        selections=selections,
        candidates=media_candidates,
        deduplication=deduplication,
    )
    report = _build_report(
        selections=selections,
        media_objects=media_objects,
        media_candidates=media_candidates,
        observations=observations,
        deduplication=deduplication,
        identity_groups=identity_groups,
        generated_at=timestamp,
    )
    result = PrototypeDuplicateResolutionResult(
        deduplication=deduplication,
        identity_groups=identity_groups,
        observations=observations,
        report=report,
        markdown=_markdown(report),
    )
    validate_prototype_duplicate_resolution_result(result)
    return result


def validate_prototype_duplicate_resolution_result(
    result: PrototypeDuplicateResolutionResult,
) -> None:
    if not isinstance(result, PrototypeDuplicateResolutionResult):
        raise TypeError("result must be a PrototypeDuplicateResolutionResult")
    validate_reference_media_deduplication_result(result.deduplication)
    validate_reference_observations(result.observations)
    _validate_identity_groups(result.identity_groups, result.deduplication)
    if result.report.get("schema_version") != PROTOTYPE_DUPLICATE_RESOLUTION_VERSION:
        raise ValueError("prototype duplicate resolution report schema is invalid")
    if result.report.get("status") != "complete":
        raise ValueError("prototype duplicate resolution report is incomplete")
    if result.report.get("prototype_only") is not True:
        raise ValueError("prototype duplicate resolution report lacks prototype marker")
    if result.report.get("counts", {}).get("selected") != (
        result.identity_groups.height
    ):
        raise ValueError("prototype duplicate resolution report count is stale")


def publish_prototype_duplicate_resolution_result(
    result: PrototypeDuplicateResolutionResult,
    *,
    storage: CloudStorage,
    output_prefix: str,
    run_id: str | None = None,
    settings_fingerprint: str | None = None,
) -> dict[str, str]:
    validate_prototype_duplicate_resolution_result(result)
    prefix = str(output_prefix or "").strip().rstrip("/")
    if not prefix:
        raise ValueError("prototype duplicate output_prefix must be nonblank")
    effective_run_id = str(run_id or "").strip() or (
        "prototype-duplicates-"
        + datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ-")
        + uuid4().hex[:12]
    )
    component = _run_component(effective_run_id)
    artifact_prefix = join_uri(
        prefix,
        "duplicate-resolution",
        f"run_id={component}",
    )
    uris = {
        "media_objects": join_uri(artifact_prefix, "reference_media_objects.parquet"),
        "relationships": join_uri(
            artifact_prefix,
            "reference_media_duplicate_relationships.parquet",
        ),
        "identity_groups": join_uri(artifact_prefix, PROTOTYPE_IDENTITY_GROUPS_FILE),
        "summary": build_report_uri(
            prefix,
            run_id=component,
            report_name=PROTOTYPE_DUPLICATE_RESOLUTION_SUMMARY_FILE.removesuffix(".md"),
            suffix="md",
        ),
        "report": build_report_uri(
            prefix,
            run_id=component,
            report_name=PROTOTYPE_DUPLICATE_RESOLUTION_REPORT_FILE.removesuffix(".json"),
        ),
    }
    if any(storage.exists(uri) for uri in uris.values()):
        raise FileExistsError("prototype duplicate resolution run already exists")
    storage.write_parquet_shard(
        uris["media_objects"],
        result.deduplication.media_objects,
        overwrite=False,
    )
    storage.write_parquet_shard(
        uris["relationships"],
        result.deduplication.relationships,
        overwrite=False,
    )
    storage.write_parquet_shard(
        uris["identity_groups"],
        result.identity_groups,
        overwrite=False,
    )
    storage.write_text(uris["summary"], result.markdown)
    report = json.loads(json.dumps(result.report))
    report["run_id"] = effective_run_id
    report["settings_fingerprint"] = settings_fingerprint
    report["artifacts"] = {
        name: {
            "uri": uri,
            "byte_count": storage.file_size(uri),
            "sha256": storage.file_sha256(uri),
        }
        for name, uri in sorted(uris.items())
        if name != "report"
    }
    storage.write_json(uris["report"], report)
    return uris


def _build_identity_groups(
    *,
    selections: pl.DataFrame,
    candidates: pl.DataFrame,
    deduplication: ReferenceMediaDeduplicationResult,
) -> pl.DataFrame:
    selections_by_id = {
        str(row["reference_media_id"]): row
        for row in selections.iter_rows(named=True)
    }
    candidates_by_id = {
        str(row["reference_media_id"]): row
        for row in candidates.iter_rows(named=True)
    }
    relationships_by_group: dict[str, list[Mapping[str, object]]] = {}
    for relationship in deduplication.relationships.iter_rows(named=True):
        relationships_by_group.setdefault(
            str(relationship["duplicate_group_id"]), []
        ).append(relationship)

    rows: list[dict[str, object]] = []
    for obj in deduplication.media_objects.iter_rows(named=True):
        media_id = str(obj["reference_media_id"])
        selection = selections_by_id[media_id]
        candidate = candidates_by_id[media_id]
        valid = obj["decode_status"] == "valid"
        group_id = str(obj["duplicate_group_id"]) if valid else None
        relationships = relationships_by_group.get(group_id or "", [])
        statuses = {str(row["resolution_status"]) for row in relationships}
        if not valid:
            resolution_status = "operational_failure"
            disposition = "operational_failure"
        elif "conflict" in statuses:
            resolution_status = "conflict"
            disposition = "duplicate_conflict"
        elif "review_required" in statuses:
            resolution_status = "review_required"
            disposition = "unresolved_duplicate"
        else:
            resolution_status = "resolved"
            disposition = (
                "eligible"
                if media_id == obj["canonical_reference_media_id"]
                else "noncanonical_duplicate"
            )
        duplicate_type = str(obj["duplicate_type"]) if valid else None
        candidate_creator = _optional_identity(candidate.get("creator"))
        candidate_owner = _optional_identity(candidate.get("rights_holder"))
        observer = _optional_identity(selection.get("observer_id"))
        owner_identity = observer or candidate_owner
        canonical_id = str(obj["canonical_reference_media_id"]) if valid else None
        provider_mirror = bool(obj["provider_mirror_ids"]) if valid else False
        row = {
            "schema_version": PROTOTYPE_IDENTITY_GROUPS_SCHEMA_VERSION,
            "reference_media_id": media_id,
            "reference_observation_id": selection["reference_observation_id"],
            "source": selection["source"],
            "provider_media_id": selection["provider_media_id"],
            "duplicate_group_id": group_id,
            "duplicate_type": duplicate_type,
            "canonical_reference_media_id": canonical_id,
            "is_canonical": valid and media_id == canonical_id,
            "resolution_status": resolution_status,
            "support_disposition": disposition,
            "exact_hash_group_id": (
                _group_id("exact-hash", str(obj["sha256"])) if valid else None
            ),
            "perceptual_duplicate_group_id": (
                _group_id("perceptual-duplicate", group_id)
                if valid and duplicate_type != "unique" and group_id is not None
                else None
            ),
            "observation_group_id": _group_id(
                "observation",
                f"{selection['source']}:{selection['reference_observation_id']}",
            ),
            "burst_group_id": (
                _group_id("burst", group_id)
                if valid
                and duplicate_type in {"near_identical_burst", "resized_copy"}
                and group_id is not None
                else None
            ),
            "owner_group_id": (
                _group_id("owner", f"{selection['source']}:{owner_identity}")
                if owner_identity
                else None
            ),
            "photographer_group_id": (
                _group_id(
                    "photographer",
                    f"{selection['source']}:{candidate_creator}",
                )
                if candidate_creator
                else None
            ),
            "provider_mirror_group_id": (
                _group_id("provider-mirror", group_id)
                if provider_mirror and group_id is not None
                else None
            ),
            "owner_evidence_available": owner_identity is not None,
            "photographer_evidence_available": candidate_creator is not None,
            "identity_fingerprint": "",
        }
        row["identity_fingerprint"] = _fingerprint(
            {key: value for key, value in row.items() if key != "identity_fingerprint"}
        )
        rows.append(row)
    frame = pl.DataFrame(
        rows,
        schema=prototype_identity_group_schema(),
        strict=True,
    ).sort("reference_media_id")
    _validate_identity_groups(frame, deduplication)
    return frame


def _validate_identity_groups(
    frame: pl.DataFrame,
    deduplication: ReferenceMediaDeduplicationResult,
) -> None:
    if frame.schema != prototype_identity_group_schema():
        raise ValueError("prototype identity groups have an incompatible schema")
    if frame["reference_media_id"].n_unique() != frame.height:
        raise ValueError("prototype identity groups contain duplicate media IDs")
    expected_ids = set(deduplication.media_objects["reference_media_id"].to_list())
    if set(frame["reference_media_id"].to_list()) != expected_ids:
        raise ValueError("prototype identity groups differ from deduplicated media")
    if frame.sort("reference_media_id")["reference_media_id"].to_list() != frame[
        "reference_media_id"
    ].to_list():
        raise ValueError("prototype identity groups are not deterministically sorted")
    for row in frame.iter_rows(named=True):
        for field in (
            "duplicate_group_id",
            "exact_hash_group_id",
            "observation_group_id",
            "owner_group_id",
            "photographer_group_id",
            "perceptual_duplicate_group_id",
            "burst_group_id",
            "provider_mirror_group_id",
        ):
            value = row[field]
            if value is not None and field != "duplicate_group_id":
                if _GROUP_ID_PATTERN.fullmatch(str(value)) is None:
                    raise ValueError(f"{field} has an unsupported namespace")
        if row["resolution_status"] not in {
            "resolved",
            "review_required",
            "conflict",
            "operational_failure",
        }:
            raise ValueError("prototype identity group resolution status is invalid")
        if row["support_disposition"] not in {
            "eligible",
            "noncanonical_duplicate",
            "unresolved_duplicate",
            "duplicate_conflict",
            "operational_failure",
        }:
            raise ValueError("prototype identity support disposition is invalid")
        expected = _fingerprint(
            {key: value for key, value in row.items() if key != "identity_fingerprint"}
        )
        if row["identity_fingerprint"] != expected:
            raise ValueError("prototype identity group fingerprint mismatch")


def _build_report(
    *,
    selections: pl.DataFrame,
    media_objects: pl.DataFrame,
    media_candidates: pl.DataFrame,
    observations: pl.DataFrame,
    deduplication: ReferenceMediaDeduplicationResult,
    identity_groups: pl.DataFrame,
    generated_at: datetime,
) -> dict[str, Any]:
    relationships = deduplication.relationships
    repeated_owner = _repeated_group_counts(identity_groups, "owner_group_id")
    repeated_photographer = _repeated_group_counts(
        identity_groups, "photographer_group_id"
    )
    repeated_observation = _repeated_group_counts(
        identity_groups, "observation_group_id"
    )
    return {
        "schema_version": PROTOTYPE_DUPLICATE_RESOLUTION_VERSION,
        "command": "references resolve-prototype-duplicates",
        "status": "complete",
        "prototype_only": True,
        "generated_at": generated_at.isoformat(),
        "counts": {
            "selected": selections.height,
            "valid_media": media_objects.filter(
                pl.col("decode_status") == "valid"
            ).height,
            "relationships": relationships.height,
            "exact_relationships": relationships.filter(
                pl.col("relationship_type") == "exact"
            ).height,
            "perceptual_candidate_relationships": relationships.filter(
                pl.col("relationship_type") == "perceptual_candidate"
            ).height,
            "observation_duplicate_groups": repeated_observation[0],
            "burst_groups": identity_groups["burst_group_id"].drop_nulls().n_unique(),
            "repeated_owner_groups": repeated_owner[0],
            "repeated_owner_members": repeated_owner[1],
            "repeated_photographer_groups": repeated_photographer[0],
            "repeated_photographer_members": repeated_photographer[1],
            "provider_mirror_relationships": relationships.filter(
                pl.col("provider_mirror")
            ).height,
            "gbif_inaturalist_mirror_relationships": relationships.filter(
                pl.col("provider_mirror")
                & pl.col("left_source").str.to_lowercase().is_in(["gbif", "inaturalist"])
                & pl.col("right_source").str.to_lowercase().is_in(
                    ["gbif", "inaturalist"]
                )
            ).height,
            "eligible": identity_groups.filter(
                pl.col("support_disposition") == "eligible"
            ).height,
            "noncanonical_duplicates": identity_groups.filter(
                pl.col("support_disposition") == "noncanonical_duplicate"
            ).height,
            "unresolved_duplicates": identity_groups.filter(
                pl.col("support_disposition") == "unresolved_duplicate"
            ).height,
            "duplicate_conflicts": identity_groups.filter(
                pl.col("support_disposition") == "duplicate_conflict"
            ).height,
            "operational_failures": identity_groups.filter(
                pl.col("support_disposition") == "operational_failure"
            ).height,
            "missing_owner_evidence": identity_groups.filter(
                ~pl.col("owner_evidence_available")
            ).height,
            "missing_photographer_evidence": identity_groups.filter(
                ~pl.col("photographer_evidence_available")
            ).height,
        },
        "duplicate_type_counts": dict(
            sorted(
                Counter(
                    str(value) if value is not None else "not_evaluated"
                    for value in identity_groups["duplicate_type"].to_list()
                ).items()
            )
        ),
        "relationship_type_counts": dict(
            sorted(Counter(relationships["relationship_type"].to_list()).items())
        ),
        "resolution_status_counts": dict(
            sorted(Counter(relationships["resolution_status"].to_list()).items())
        ),
        "inputs": {
            "selections_fingerprint": _frame_fingerprint(selections),
            "media_objects_fingerprint": _frame_fingerprint(media_objects),
            "media_candidates_fingerprint": _frame_fingerprint(media_candidates),
            "observations_fingerprint": _frame_fingerprint(observations),
        },
        "outputs": {
            "annotated_media_objects_fingerprint": _frame_fingerprint(
                deduplication.media_objects
            ),
            "relationships_fingerprint": _frame_fingerprint(relationships),
            "identity_groups_fingerprint": _frame_fingerprint(identity_groups),
        },
        "limitations": [
            "Owner and photographer identifiers are provider metadata used only for leakage grouping, not taxonomic validation.",
            "Missing owner or photographer metadata remains explicit and is not represented as a shared group.",
            "Perceptual candidates without independently resolved evidence remain review-required.",
        ],
    }


def _visual_candidates_by_media_id(
    manifest: Mapping[str, object],
) -> dict[str, Mapping[str, object]]:
    from biominer.references.prototype_acquisition import (
        make_prototype_visual_reference_ids,
    )

    raw = manifest.get("candidates")
    if not isinstance(raw, list):
        raise ValueError("visual-domain manifest candidates must be an array")
    result: dict[str, Mapping[str, object]] = {}
    for item in raw:
        if not isinstance(item, Mapping):
            raise ValueError("visual-domain candidate must be an object")
        _observation_id, media_id = make_prototype_visual_reference_ids(item)
        if media_id in result:
            raise ValueError(f"duplicate visual-domain media identity: {media_id}")
        result[media_id] = item
    return result


def _visual_observation(
    selection: Mapping[str, object],
    raw: Mapping[str, object],
    manifest: Mapping[str, object],
) -> dict[str, object]:
    snapshot = str(manifest.get("source_snapshot_version") or "").strip()
    if not snapshot:
        raise ValueError("visual-domain manifest lacks source_snapshot_version")
    source_record_id = str(raw.get("source_record_id") or "").strip()
    payload = json.dumps(raw, sort_keys=True, separators=(",", ":"), default=str)
    selected_at = selection["selected_at"]
    if not isinstance(selected_at, datetime):
        raise ValueError("prototype visual selected_at is invalid")
    return {
        "schema_version": REFERENCE_OBSERVATIONS_SCHEMA_VERSION,
        "reference_observation_id": selection["reference_observation_id"],
        "source": selection["source"],
        "source_observation_id": source_record_id,
        "source_taxon_id": None,
        "supplied_scientific_name": None,
        "accepted_taxon_key": None,
        "reconciled_scientific_name": None,
        "registry_version": "prototype-visual-domain-v1",
        "taxon_reconciliation_status": "unresolved",
        "identification_quality": "visual_domain_negative",
        "community_taxon_status": None,
        "identification_disagreement": False,
        "captive_or_cultivated": False,
        "observer_id": None,
        "locality": None,
        "life_stage": "not_applicable",
        "sex": None,
        "observed_at": None,
        "latitude": None,
        "longitude": None,
        "coordinate_uncertainty": None,
        "coordinates_obscured": False,
        "country": None,
        "country_code": None,
        "geo_cluster_id": None,
        "distance_to_cluster_medoid_km": None,
        "source_dataset_key": None,
        "source_dataset_doi": None,
        "source_record_url": selection["source_record_uri"],
        "source_record_hash": _fingerprint(payload),
        "retrieved_at": selected_at,
        "source_snapshot_version": snapshot,
        "source_query_fingerprint": _fingerprint(snapshot),
        "fallback_level": 0,
        "geospatial_issue": False,
        "preserved_specimen": False,
        "fossil": False,
        "occurrence_absent": False,
        "uncertain_taxon_match": True,
        "basis_of_record_suitable": False,
    }


def _repeated_group_counts(frame: pl.DataFrame, field: str) -> tuple[int, int]:
    repeated = (
        frame.filter(pl.col(field).is_not_null())
        .group_by(field)
        .len()
        .filter(pl.col("len") > 1)
    )
    return repeated.height, int(repeated["len"].sum() or 0)


def _optional_identity(value: object) -> str | None:
    normalized = " ".join(str(value or "").casefold().split())
    return normalized or None


def _group_id(kind: str, value: str) -> str:
    digest = hashlib.sha256(value.encode()).hexdigest()[:32]
    return f"prototype-{kind}-group:{digest}"


def _frame_fingerprint(frame: pl.DataFrame) -> str:
    return _fingerprint(frame.to_dicts())


def _fingerprint(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode()
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _run_component(run_id: str) -> str:
    safe = safe_path_component(run_id)[:96]
    digest = hashlib.sha256(run_id.encode()).hexdigest()[:12]
    return f"{safe}-{digest}"


def _utc_datetime(value: str | datetime) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00")) if isinstance(value, str) else value
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("generated_at must include a timezone")
    return parsed.astimezone(UTC)


def _markdown(report: Mapping[str, object]) -> str:
    counts = report["counts"]
    assert isinstance(counts, Mapping)
    return "\n".join(
        [
            "# Prototype duplicate resolution",
            "",
            f"- Status: {report['status']}",
            f"- Prototype only: {str(report['prototype_only']).lower()}",
            f"- Selected media: {counts['selected']}",
            f"- Direct relationships: {counts['relationships']}",
            f"- Eligible canonical media: {counts['eligible']}",
            f"- Noncanonical duplicates: {counts['noncanonical_duplicates']}",
            f"- Unresolved duplicates: {counts['unresolved_duplicates']}",
            f"- Duplicate conflicts: {counts['duplicate_conflicts']}",
            f"- Retryable operational failures: {counts['operational_failures']}",
            f"- Repeated owner groups: {counts['repeated_owner_groups']}",
            f"- Repeated photographer groups: {counts['repeated_photographer_groups']}",
            "",
            "Provider metadata and perceptual matches remain evidence, not taxonomic validation.",
            "",
        ]
    )


__all__ = [
    "PROTOTYPE_DUPLICATE_RESOLUTION_REPORT_FILE",
    "PROTOTYPE_DUPLICATE_RESOLUTION_SUMMARY_FILE",
    "PROTOTYPE_DUPLICATE_RESOLUTION_VERSION",
    "PROTOTYPE_IDENTITY_GROUPS_FILE",
    "PROTOTYPE_IDENTITY_GROUPS_SCHEMA_VERSION",
    "PrototypeDuplicateResolutionResult",
    "compile_prototype_deduplication_observations",
    "prototype_identity_group_schema",
    "publish_prototype_duplicate_resolution_result",
    "resolve_prototype_duplicates",
    "validate_prototype_duplicate_resolution_result",
]
