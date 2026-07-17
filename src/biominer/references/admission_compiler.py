"""Compile provenance-complete provisional GBIF admission artifacts."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
import json
from pathlib import Path
from uuid import uuid4

import polars as pl

from biominer.common.semantic_hash import canonical_semantic_fingerprint
from biominer.references.admission import (
    DEFAULT_REFERENCE_ADMISSION_MODE,
    ReferenceAdmissionPolicy,
)
from biominer.references.admission_eligibility import (
    GBIFEligibilityEvidence,
    GBIFEligibilityResult,
    evaluate_gbif_provisional_eligibility,
)
from biominer.references.provisional_selection import (
    PROVISIONAL_SELECTION_DECISION_SCHEMA,
)
from biominer.references.schemas import (
    reference_media_duplicate_relationship_schema,
    reference_media_candidate_schema,
    reference_media_object_schema,
    reference_observation_schema,
    validate_reference_media_candidates,
    validate_reference_media_duplicate_relationships,
    validate_reference_media_objects,
    validate_reference_observations,
)
from biominer.references.yoloe_routing import REFERENCE_YOLOE_ROUTE_SCHEMA
from biominer.storage.parquet import write_parquet


REFERENCE_ADMISSION_COMPILER_SCHEMA_VERSION = (
    "reference-admission-compiler-v1.0.0"
)
REFERENCE_ADMISSION_DECISIONS_FILE = "reference_admission_decisions.parquet"
REFERENCE_PROVISIONAL_SUPPORT_FILE = "reference_provisional_support.parquet"
REFERENCE_ADMISSION_SUMMARY_FILE = "reference_admission_summary.parquet"
REFERENCE_ADMISSION_REPORT_FILE = "reference_admission_report.json"
REFERENCE_ADMISSION_SUMMARY_MD_FILE = "reference_admission_summary.md"

REFERENCE_ADMISSION_DECISION_SCHEMA: dict[str, pl.DataType] = {
    "schema_version": pl.String,
    "reference_media_id": pl.String,
    "reference_observation_id": pl.String,
    "source": pl.String,
    "provider_media_id": pl.String,
    "accepted_taxon_key": pl.String,
    "scientific_name": pl.String,
    "provider_assertion_status": pl.String,
    "provider_assertion_identity_basis": pl.String,
    "human_verified": pl.Boolean,
    "route": pl.String,
    "admission_decision": pl.String,
    "admission_reason_codes": pl.List(pl.String),
    "automated_gate_ids": pl.List(pl.String),
    "automated_gate_dispositions": pl.List(pl.String),
    "automated_gate_reason_codes": pl.List(pl.String),
    "selection_decision": pl.String,
    "selection_reason": pl.String,
    "provisional_support": pl.Boolean,
    "provisional_status": pl.String,
    "duplicate_group_id": pl.String,
    "duplicate_type": pl.String,
    "canonical_reference_media_id": pl.String,
    "duplicate_resolution_status": pl.String,
    "route_evidence_fingerprint": pl.String,
    "selection_policy_fingerprint": pl.String,
    "reference_admission_mode": pl.String,
    "reference_admission_policy_version": pl.String,
    "reference_admission_policy_fingerprint": pl.String,
    "eligibility_evidence_fingerprint": pl.String,
    "eligibility_result_fingerprint": pl.String,
    "decision_fingerprint": pl.String,
}

REFERENCE_PROVISIONAL_SUPPORT_SCHEMA: dict[str, pl.DataType] = {
    "schema_version": pl.String,
    "reference_media_id": pl.String,
    "reference_observation_id": pl.String,
    "source": pl.String,
    "provider_media_id": pl.String,
    "accepted_taxon_key": pl.String,
    "scientific_name": pl.String,
    "provider_assertion_status": pl.String,
    "provider_assertion_identity_basis": pl.String,
    "human_verified": pl.Boolean,
    "provisional_support": pl.Boolean,
    "provisional_status": pl.String,
    "admission_decision": pl.String,
    "admission_reason_codes": pl.List(pl.String),
    "automated_gate_ids": pl.List(pl.String),
    "automated_gate_dispositions": pl.List(pl.String),
    "automated_gate_reason_codes": pl.List(pl.String),
    "route": pl.String,
    "provisional_life_stage": pl.String,
    "provisional_visual_domain": pl.String,
    "subject_area_ratio": pl.Float64,
    "observer_id": pl.String,
    "geo_cluster_id": pl.String,
    "latitude": pl.Float64,
    "longitude": pl.Float64,
    "source_record_url": pl.String,
    "source_record_hash": pl.String,
    "source_snapshot_version": pl.String,
    "source_object_uri": pl.String,
    "image_sha256": pl.String,
    "content_type": pl.String,
    "decoded_width": pl.UInt32,
    "decoded_height": pl.UInt32,
    "creator": pl.String,
    "rights_holder": pl.String,
    "licence": pl.String,
    "licence_uri": pl.String,
    "attribution": pl.String,
    "duplicate_group_id": pl.String,
    "duplicate_type": pl.String,
    "canonical_reference_media_id": pl.String,
    "duplicate_resolution_status": pl.String,
    "detector_model_id": pl.String,
    "detector_model_version": pl.String,
    "detector_checkpoint": pl.String,
    "routing_policy_fingerprint": pl.String,
    "route_evidence_fingerprint": pl.String,
    "reference_admission_mode": pl.String,
    "reference_admission_policy_version": pl.String,
    "reference_admission_policy_fingerprint": pl.String,
    "eligibility_evidence_fingerprint": pl.String,
    "eligibility_result_fingerprint": pl.String,
    "selection_policy_fingerprint": pl.String,
    "selection_decision": pl.String,
    "selection_reason": pl.String,
    "selection_decision_fingerprint": pl.String,
    "support_row_fingerprint": pl.String,
}

REFERENCE_ADMISSION_SUMMARY_SCHEMA: dict[str, pl.DataType] = {
    "schema_version": pl.String,
    "accepted_taxon_key": pl.String,
    "route": pl.String,
    "candidate_count": pl.UInt32,
    "admitted_count": pl.UInt32,
    "review_required_count": pl.UInt32,
    "excluded_count": pl.UInt32,
    "selected_count": pl.UInt32,
    "provisional_support_count": pl.UInt32,
    "species_quota": pl.UInt32,
    "support_shortfall_count": pl.UInt32,
    "reference_admission_policy_fingerprint": pl.String,
    "selection_policy_fingerprint": pl.String,
    "compiler_fingerprint": pl.String,
}


@dataclass(frozen=True, slots=True)
class ReferenceAdmissionCompilationResult:
    """All compiled frames, reports, and their durable paths."""

    decisions: pl.DataFrame
    provisional_support: pl.DataFrame
    summary: pl.DataFrame
    report: Mapping[str, object]
    markdown: str
    decisions_path: Path
    provisional_support_path: Path
    summary_path: Path
    report_path: Path
    markdown_path: Path
    compiler_fingerprint: str


def validate_reference_provisional_support(frame: pl.DataFrame) -> None:
    """Validate the exact, provenance-bound provisional support projection."""

    if not isinstance(frame, pl.DataFrame):
        raise TypeError("provisional support must be a Polars DataFrame")
    if frame.schema != REFERENCE_PROVISIONAL_SUPPORT_SCHEMA:
        raise ValueError("provisional support schema mismatch")
    if frame["reference_media_id"].n_unique() != frame.height:
        raise ValueError("provisional support contains duplicate media IDs")
    for row in frame.iter_rows(named=True):
        if (
            row["schema_version"] != REFERENCE_ADMISSION_COMPILER_SCHEMA_VERSION
            or row["source"] != "gbif"
            or row["provider_assertion_status"]
            != "provider_asserted_unreviewed"
            or row["provider_assertion_identity_basis"]
            != "gbif_provider_asserted"
            or row["human_verified"]
            or not row["provisional_support"]
            or row["provisional_status"]
            != "provisional_admitted_selected"
            or row["admission_decision"] != "admitted"
            or row["selection_decision"] != "selected"
            or row["reference_admission_mode"]
            != DEFAULT_REFERENCE_ADMISSION_MODE
        ):
            raise ValueError("provisional support row semantics are inconsistent")
        if (
            not row["automated_gate_ids"]
            or len(row["automated_gate_ids"])
            != len(row["automated_gate_dispositions"])
            or len(row["automated_gate_ids"])
            != len(row["automated_gate_reason_codes"])
            or set(row["automated_gate_dispositions"]) != {"passed"}
        ):
            raise ValueError("provisional support automated gates are incomplete")
        payload = dict(row)
        fingerprint = payload.pop("support_row_fingerprint")
        if fingerprint != canonical_semantic_fingerprint(payload):
            raise ValueError("provisional support row fingerprint mismatch")


def compile_provisional_gbif_support_bank(
    *,
    observations: pl.DataFrame,
    media_candidates: pl.DataFrame,
    media_objects: pl.DataFrame,
    duplicate_relationships: pl.DataFrame,
    yoloe_routes: pl.DataFrame,
    selection_decisions: pl.DataFrame,
    admission_policy: ReferenceAdmissionPolicy,
    prototype_scope: str,
    output_dir: str | Path,
    created_at: datetime | None = None,
) -> ReferenceAdmissionCompilationResult:
    """Recompute admission and emit the complete provisional support bank."""

    if admission_policy.mode != DEFAULT_REFERENCE_ADMISSION_MODE:
        raise ValueError("provisional compiler requires the adaptive admission policy")
    if prototype_scope not in {"global", "local", "regional"}:
        raise ValueError("prototype_scope must be global, local, or regional")
    created = created_at or datetime.now(UTC)
    if created.tzinfo is None or created.utcoffset() is None:
        raise ValueError("created_at must be timezone-aware")
    _validate_inputs(
        observations=observations,
        media_candidates=media_candidates,
        media_objects=media_objects,
        duplicate_relationships=duplicate_relationships,
        yoloe_routes=yoloe_routes,
        selection_decisions=selection_decisions,
    )
    input_fingerprints = {
        "observations": _frame_fingerprint(observations),
        "media_candidates": _frame_fingerprint(media_candidates),
        "media_objects": _frame_fingerprint(media_objects),
        "duplicate_relationships": _frame_fingerprint(duplicate_relationships),
        "yoloe_routes": _frame_fingerprint(yoloe_routes),
        "selection_decisions": _frame_fingerprint(selection_decisions),
    }
    selection_policy_fingerprint = _single_frame_text(
        selection_decisions, "selection_policy_fingerprint"
    )
    compiler_fingerprint = canonical_semantic_fingerprint(
        {
            "schema_version": REFERENCE_ADMISSION_COMPILER_SCHEMA_VERSION,
            "admission_policy_fingerprint": admission_policy.fingerprint,
            "selection_policy_fingerprint": selection_policy_fingerprint,
            "prototype_scope": prototype_scope,
            "inputs": input_fingerprints,
        }
    )
    joined = _indexed_inputs(
        observations=observations,
        media_candidates=media_candidates,
        media_objects=media_objects,
        duplicate_relationships=duplicate_relationships,
        yoloe_routes=yoloe_routes,
        selection_decisions=selection_decisions,
    )
    decision_rows: list[dict[str, object]] = []
    support_rows: list[dict[str, object]] = []
    for media_id in sorted(joined["candidates"]):
        candidate = joined["candidates"][media_id]
        observation = joined["observations"][
            str(candidate["reference_observation_id"])
        ]
        media_object = joined["objects"][media_id]
        selection = joined["selections"][media_id]
        route = _route_for_selection(
            joined["routes"][media_id], requested_route=str(selection["route"])
        )
        _validate_joined_row(
            observation=observation,
            candidate=candidate,
            media_object=media_object,
            route=route,
            selection=selection,
        )
        duplicate_status = _duplicate_resolution_status(
            media_object, joined["relationships"]
        )
        evidence = _eligibility_evidence(
            observation=observation,
            candidate=candidate,
            media_object=media_object,
            route=route,
            selection=selection,
            duplicate_status=duplicate_status,
            prototype_scope=prototype_scope,
        )
        eligibility = evaluate_gbif_provisional_eligibility(
            evidence, admission_policy
        )
        if selection["admission_decision"] != eligibility.decision.value:
            raise ValueError(
                "selection admission decision is stale for "
                f"{media_id}: persisted={selection['admission_decision']}, "
                f"recomputed={eligibility.decision.value}"
            )
        decision = _admission_decision_row(
            observation=observation,
            candidate=candidate,
            media_object=media_object,
            route=route,
            selection=selection,
            eligibility=eligibility,
            duplicate_status=duplicate_status,
            admission_policy=admission_policy,
        )
        decision_rows.append(decision)
        if decision["provisional_support"]:
            support_rows.append(
                _support_row(
                    observation=observation,
                    candidate=candidate,
                    media_object=media_object,
                    route=route,
                    selection=selection,
                    eligibility=eligibility,
                    duplicate_status=duplicate_status,
                    admission_policy=admission_policy,
                )
            )

    decisions = pl.DataFrame(
        decision_rows, schema=REFERENCE_ADMISSION_DECISION_SCHEMA
    ).sort("reference_media_id")
    support = pl.DataFrame(
        support_rows, schema=REFERENCE_PROVISIONAL_SUPPORT_SCHEMA
    ).sort("reference_media_id")
    summary = _summary_frame(
        decisions,
        selection_decisions=selection_decisions,
        admission_policy=admission_policy,
        selection_policy_fingerprint=selection_policy_fingerprint,
        compiler_fingerprint=compiler_fingerprint,
    )
    report = _report(
        decisions=decisions,
        support=support,
        summary=summary,
        admission_policy=admission_policy,
        prototype_scope=prototype_scope,
        input_fingerprints=input_fingerprints,
        compiler_fingerprint=compiler_fingerprint,
        created_at=created,
    )
    markdown = _markdown(report, summary)
    output = Path(output_dir)
    paths = {
        "decisions": output / REFERENCE_ADMISSION_DECISIONS_FILE,
        "support": output / REFERENCE_PROVISIONAL_SUPPORT_FILE,
        "summary": output / REFERENCE_ADMISSION_SUMMARY_FILE,
        "report": output / REFERENCE_ADMISSION_REPORT_FILE,
        "markdown": output / REFERENCE_ADMISSION_SUMMARY_MD_FILE,
    }
    write_parquet(decisions, paths["decisions"])
    write_parquet(support, paths["support"])
    write_parquet(summary, paths["summary"])
    _write_text_atomic(
        paths["report"], json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    _write_text_atomic(paths["markdown"], markdown)
    return ReferenceAdmissionCompilationResult(
        decisions=decisions,
        provisional_support=support,
        summary=summary,
        report=report,
        markdown=markdown,
        decisions_path=paths["decisions"],
        provisional_support_path=paths["support"],
        summary_path=paths["summary"],
        report_path=paths["report"],
        markdown_path=paths["markdown"],
        compiler_fingerprint=compiler_fingerprint,
    )


def _validate_inputs(
    *,
    observations: pl.DataFrame,
    media_candidates: pl.DataFrame,
    media_objects: pl.DataFrame,
    duplicate_relationships: pl.DataFrame,
    yoloe_routes: pl.DataFrame,
    selection_decisions: pl.DataFrame,
) -> None:
    for frame, schema, name in (
        (observations, reference_observation_schema(), "observations"),
        (media_candidates, reference_media_candidate_schema(), "media candidates"),
        (media_objects, reference_media_object_schema(), "media objects"),
        (
            duplicate_relationships,
            reference_media_duplicate_relationship_schema(),
            "duplicate relationships",
        ),
        (yoloe_routes, REFERENCE_YOLOE_ROUTE_SCHEMA, "YOLOE routes"),
        (
            selection_decisions,
            PROVISIONAL_SELECTION_DECISION_SCHEMA,
            "selection decisions",
        ),
    ):
        if not isinstance(frame, pl.DataFrame) or frame.schema != schema:
            raise ValueError(f"{name} schema mismatch")
    validate_reference_observations(observations)
    validate_reference_media_candidates(media_candidates)
    validate_reference_media_objects(media_objects)
    validate_reference_media_duplicate_relationships(duplicate_relationships)
    candidate_ids = set(media_candidates["reference_media_id"].to_list())
    for frame, name in (
        (media_objects, "media objects"),
        (selection_decisions, "selection decisions"),
    ):
        if set(frame["reference_media_id"].to_list()) != candidate_ids:
            raise ValueError(f"{name} must cover every media candidate exactly")
    route_ids = set(yoloe_routes["reference_media_id"].to_list())
    if route_ids != candidate_ids:
        raise ValueError("YOLOE routes must cover every media candidate")
    if selection_decisions["reference_media_id"].n_unique() != (
        selection_decisions.height
    ):
        raise ValueError("selection decisions contain duplicate media IDs")
    for row in yoloe_routes.iter_rows(named=True):
        expected = row["route_evidence_fingerprint"]
        payload = dict(row)
        payload.pop("route_evidence_fingerprint")
        if expected != canonical_semantic_fingerprint(payload):
            raise ValueError("YOLOE route evidence fingerprint mismatch")
        if row["species_identity_decision"] != "not_assessed_by_yoloe":
            raise ValueError("YOLOE must not make a species identity decision")
    for row in selection_decisions.iter_rows(named=True):
        expected = row["decision_fingerprint"]
        payload = dict(row)
        payload.pop("decision_fingerprint")
        if expected != canonical_semantic_fingerprint(payload):
            raise ValueError("selection decision fingerprint mismatch")


def _indexed_inputs(**frames: pl.DataFrame) -> dict[str, object]:
    candidates = _index(frames["media_candidates"], "reference_media_id")
    observations = _index(frames["observations"], "reference_observation_id")
    objects = _index(frames["media_objects"], "reference_media_id")
    selections = _index(frames["selection_decisions"], "reference_media_id")
    routes: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in frames["yoloe_routes"].iter_rows(named=True):
        routes[str(row["reference_media_id"])].append(row)
    relationships: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in frames["duplicate_relationships"].iter_rows(named=True):
        relationships[str(row["duplicate_group_id"])].append(row)
    observation_ids = set(observations)
    if any(
        str(row["reference_observation_id"]) not in observation_ids
        for row in candidates.values()
    ):
        raise ValueError("media candidate references an unknown observation")
    return {
        "candidates": candidates,
        "observations": observations,
        "objects": objects,
        "selections": selections,
        "routes": routes,
        "relationships": relationships,
    }


def _index(frame: pl.DataFrame, field: str) -> dict[str, dict[str, object]]:
    return {str(row[field]): row for row in frame.iter_rows(named=True)}


def _route_for_selection(
    routes: list[dict[str, object]], *, requested_route: str
) -> dict[str, object]:
    exact = [row for row in routes if row["route"] == requested_route]
    if len(exact) == 1:
        return exact[0]
    if not exact and len(routes) == 1:
        return routes[0]
    raise ValueError(
        f"expected one YOLOE route for requested bank route {requested_route!r}"
    )


def _validate_joined_row(
    *,
    observation: Mapping[str, object],
    candidate: Mapping[str, object],
    media_object: Mapping[str, object],
    route: Mapping[str, object],
    selection: Mapping[str, object],
) -> None:
    media_id = candidate["reference_media_id"]
    if any(
        row["reference_media_id"] != media_id
        for row in (media_object, route, selection)
    ):
        raise ValueError("joined reference media identities disagree")
    observation_id = candidate["reference_observation_id"]
    if (
        observation["reference_observation_id"] != observation_id
        or selection["reference_observation_id"] != observation_id
    ):
        raise ValueError("joined reference observation identities disagree")
    if candidate["source"] != observation["source"] or route["source"] != candidate[
        "source"
    ]:
        raise ValueError("joined provider sources disagree")
    if route["source_record_hash"] != observation["source_record_hash"]:
        raise ValueError("YOLOE route source evidence does not match the observation")
    if selection["accepted_taxon_key"] != observation["accepted_taxon_key"]:
        raise ValueError("selection accepted taxon does not match the observation")
    if (
        selection["duplicate_group_id"] != media_object["duplicate_group_id"]
        or selection["canonical_reference_media_id"]
        != media_object["canonical_reference_media_id"]
    ):
        raise ValueError("selection duplicate evidence does not match media evidence")
    if candidate["licence_policy_status"] != media_object[
        "licence_policy_status"
    ]:
        raise ValueError("candidate and downloaded-object licence decisions disagree")


def _duplicate_resolution_status(
    media_object: Mapping[str, object],
    relationships: Mapping[str, list[dict[str, object]]],
) -> str | None:
    group_id = media_object["duplicate_group_id"]
    if group_id is None:
        return None
    group = relationships.get(str(group_id), [])
    statuses = {str(row["resolution_status"]) for row in group}
    if "conflict" in statuses:
        return "conflict"
    if "review_required" in statuses:
        return "review_required"
    return "resolved"


def _eligibility_evidence(
    *,
    observation: Mapping[str, object],
    candidate: Mapping[str, object],
    media_object: Mapping[str, object],
    route: Mapping[str, object],
    selection: Mapping[str, object],
    duplicate_status: str | None,
    prototype_scope: str,
) -> GBIFEligibilityEvidence:
    accepted_key = observation["accepted_taxon_key"]
    candidate_key = selection["accepted_taxon_key"]
    usable_geography = (
        observation["latitude"] is not None
        and observation["longitude"] is not None
        and observation["geo_cluster_id"] is not None
        and observation["geospatial_issue"] is False
    )
    return GBIFEligibilityEvidence(
        source=str(candidate["source"]),
        taxon_reconciliation_status=str(
            observation["taxon_reconciliation_status"]
        ),
        resolves_to_candidate_accepted_taxon_key=(
            accepted_key is not None and accepted_key == candidate_key
        ),
        uncertain_taxon_match=bool(observation["uncertain_taxon_match"]),
        occurrence_absent=bool(observation["occurrence_absent"]),
        fossil=bool(observation["fossil"]),
        media_type=str(candidate["media_type"]),
        download_status=str(candidate["download_status"]),
        content_type=_optional_text(media_object["content_type"]),
        decode_status=str(media_object["decode_status"]),
        decoded_width=media_object["decoded_width"],  # type: ignore[arg-type]
        decoded_height=media_object["decoded_height"],  # type: ignore[arg-type]
        image_sha256=_optional_text(media_object["sha256"]),
        licence_policy_status=str(candidate["licence_policy_status"]),
        creator=_optional_text(candidate["creator"]),
        source_url=_optional_text(observation["source_record_url"]),
        attribution=_optional_text(candidate["attribution"]),
        duplicate_processing_completed=(
            media_object["duplicate_group_id"] is not None
            and media_object["canonical_reference_media_id"] is not None
        ),
        canonical_media=(
            media_object["canonical_reference_media_id"]
            == candidate["reference_media_id"]
        ),
        duplicate_resolution_status=duplicate_status,
        duplicate_conflict_targeted_review=(
            duplicate_status == "review_required"
        ),
        provider_identity_matches_candidate_taxon=(
            accepted_key is not None and accepted_key == candidate_key
        ),
        independence_processing_completed=True,
        selected_images_from_observation=int(
            selection["observation_selection_ordinal"] or 1
        ),
        observer_identity_available=observation["observer_id"] is not None,
        observer_image_ordinal_before_reuse=int(
            selection["observer_selection_ordinal"] or 1
        ),
        observer_reuse_justified=bool(selection["observer_reuse_justified"]),
        near_identical_view=False,
        distinct_additional_view_justified=bool(
            selection["distinct_additional_view_justified"]
        ),
        yoloe_routing_completed=(
            route["routing_action"] in {"score", "review", "exclude"}
            and "image_load_failed" not in route["domain_flags"]
        ),
        yoloe_route=_optional_text(route["route"]),
        requested_bank_route=str(selection["route"]),
        visual_domain=str(route["provisional_visual_domain"]),
        subject_present=route["subject_present"],  # type: ignore[arg-type]
        ambiguous_domain_targeted_review=route["routing_action"] == "review",
        subject_area_ratio=route["subject_area_ratio"],  # type: ignore[arg-type]
        full_frame_input_generation_succeeded=bool(
            route["full_frame_input_generation_succeeded"]
        ),
        prototype_scope=prototype_scope,
        usable_geography=usable_geography,
    )


def _admission_decision_row(
    *,
    observation: Mapping[str, object],
    candidate: Mapping[str, object],
    media_object: Mapping[str, object],
    route: Mapping[str, object],
    selection: Mapping[str, object],
    eligibility: GBIFEligibilityResult,
    duplicate_status: str | None,
    admission_policy: ReferenceAdmissionPolicy,
) -> dict[str, object]:
    support = (
        eligibility.decision.value == "admitted"
        and selection["decision"] == "selected"
    )
    if support:
        provisional_status = "provisional_admitted_selected"
    elif eligibility.decision.value == "admitted":
        provisional_status = "provisional_admitted_not_selected"
    else:
        provisional_status = eligibility.decision.value
    row: dict[str, object] = {
        "schema_version": REFERENCE_ADMISSION_COMPILER_SCHEMA_VERSION,
        "reference_media_id": candidate["reference_media_id"],
        "reference_observation_id": candidate["reference_observation_id"],
        "source": candidate["source"],
        "provider_media_id": candidate["provider_media_id"],
        "accepted_taxon_key": observation["accepted_taxon_key"],
        "scientific_name": observation["reconciled_scientific_name"],
        "provider_assertion_status": "provider_asserted_unreviewed",
        "provider_assertion_identity_basis": eligibility.identity_basis,
        "human_verified": eligibility.human_verified,
        "route": selection["route"],
        "admission_decision": eligibility.decision.value,
        "admission_reason_codes": list(eligibility.reason_codes),
        "automated_gate_ids": [gate.gate_id for gate in eligibility.gate_results],
        "automated_gate_dispositions": [
            gate.disposition.value for gate in eligibility.gate_results
        ],
        "automated_gate_reason_codes": [
            gate.reason_code for gate in eligibility.gate_results
        ],
        "selection_decision": selection["decision"],
        "selection_reason": selection["decision_reason"],
        "provisional_support": support,
        "provisional_status": provisional_status,
        "duplicate_group_id": media_object["duplicate_group_id"],
        "duplicate_type": media_object["duplicate_type"],
        "canonical_reference_media_id": media_object[
            "canonical_reference_media_id"
        ],
        "duplicate_resolution_status": duplicate_status,
        "route_evidence_fingerprint": route["route_evidence_fingerprint"],
        "selection_policy_fingerprint": selection[
            "selection_policy_fingerprint"
        ],
        "reference_admission_mode": admission_policy.mode,
        "reference_admission_policy_version": admission_policy.policy_version,
        "reference_admission_policy_fingerprint": admission_policy.fingerprint,
        "eligibility_evidence_fingerprint": eligibility.evidence_fingerprint,
        "eligibility_result_fingerprint": eligibility.fingerprint,
    }
    row["decision_fingerprint"] = canonical_semantic_fingerprint(row)
    return row


def _support_row(
    *,
    observation: Mapping[str, object],
    candidate: Mapping[str, object],
    media_object: Mapping[str, object],
    route: Mapping[str, object],
    selection: Mapping[str, object],
    eligibility: GBIFEligibilityResult,
    duplicate_status: str | None,
    admission_policy: ReferenceAdmissionPolicy,
) -> dict[str, object]:
    row: dict[str, object] = {
        "schema_version": REFERENCE_ADMISSION_COMPILER_SCHEMA_VERSION,
        "reference_media_id": candidate["reference_media_id"],
        "reference_observation_id": candidate["reference_observation_id"],
        "source": candidate["source"],
        "provider_media_id": candidate["provider_media_id"],
        "accepted_taxon_key": observation["accepted_taxon_key"],
        "scientific_name": observation["reconciled_scientific_name"],
        "provider_assertion_status": "provider_asserted_unreviewed",
        "provider_assertion_identity_basis": eligibility.identity_basis,
        "human_verified": False,
        "provisional_support": True,
        "provisional_status": "provisional_admitted_selected",
        "admission_decision": eligibility.decision.value,
        "admission_reason_codes": list(eligibility.reason_codes),
        "automated_gate_ids": [gate.gate_id for gate in eligibility.gate_results],
        "automated_gate_dispositions": [
            gate.disposition.value for gate in eligibility.gate_results
        ],
        "automated_gate_reason_codes": [
            gate.reason_code for gate in eligibility.gate_results
        ],
        "route": selection["route"],
        "provisional_life_stage": route["provisional_life_stage"],
        "provisional_visual_domain": route["provisional_visual_domain"],
        "subject_area_ratio": route["subject_area_ratio"],
        "observer_id": observation["observer_id"],
        "geo_cluster_id": observation["geo_cluster_id"],
        "latitude": observation["latitude"],
        "longitude": observation["longitude"],
        "source_record_url": observation["source_record_url"],
        "source_record_hash": observation["source_record_hash"],
        "source_snapshot_version": observation["source_snapshot_version"],
        "source_object_uri": media_object["source_object_uri"],
        "image_sha256": media_object["sha256"],
        "content_type": media_object["content_type"],
        "decoded_width": media_object["decoded_width"],
        "decoded_height": media_object["decoded_height"],
        "creator": candidate["creator"],
        "rights_holder": candidate["rights_holder"],
        "licence": candidate["licence"],
        "licence_uri": candidate["licence_uri"],
        "attribution": candidate["attribution"],
        "duplicate_group_id": media_object["duplicate_group_id"],
        "duplicate_type": media_object["duplicate_type"],
        "canonical_reference_media_id": media_object[
            "canonical_reference_media_id"
        ],
        "duplicate_resolution_status": duplicate_status,
        "detector_model_id": route["detector_model_id"],
        "detector_model_version": route["detector_model_version"],
        "detector_checkpoint": route["detector_checkpoint"],
        "routing_policy_fingerprint": route["routing_policy_fingerprint"],
        "route_evidence_fingerprint": route["route_evidence_fingerprint"],
        "reference_admission_mode": admission_policy.mode,
        "reference_admission_policy_version": admission_policy.policy_version,
        "reference_admission_policy_fingerprint": admission_policy.fingerprint,
        "eligibility_evidence_fingerprint": eligibility.evidence_fingerprint,
        "eligibility_result_fingerprint": eligibility.fingerprint,
        "selection_policy_fingerprint": selection[
            "selection_policy_fingerprint"
        ],
        "selection_decision": selection["decision"],
        "selection_reason": selection["decision_reason"],
        "selection_decision_fingerprint": selection["decision_fingerprint"],
    }
    row["support_row_fingerprint"] = canonical_semantic_fingerprint(row)
    return row


def _summary_frame(
    decisions: pl.DataFrame,
    *,
    selection_decisions: pl.DataFrame,
    admission_policy: ReferenceAdmissionPolicy,
    selection_policy_fingerprint: str,
    compiler_fingerprint: str,
) -> pl.DataFrame:
    rows: list[dict[str, object]] = []
    selection_by_id = _index(selection_decisions, "reference_media_id")
    for keys, group in decisions.group_by(
        ["accepted_taxon_key", "route"], maintain_order=True
    ):
        taxon, route = keys
        selection = selection_by_id[str(group["reference_media_id"].item(0))]
        support_count = int(group["provisional_support"].sum())
        quota = int(selection["species_quota"])
        rows.append(
            {
                "schema_version": REFERENCE_ADMISSION_COMPILER_SCHEMA_VERSION,
                "accepted_taxon_key": taxon,
                "route": route,
                "candidate_count": group.height,
                "admitted_count": _count(group, "admitted"),
                "review_required_count": _count(group, "review_required"),
                "excluded_count": _count(group, "excluded"),
                "selected_count": int(
                    group.filter(pl.col("selection_decision") == "selected").height
                ),
                "provisional_support_count": support_count,
                "species_quota": quota,
                "support_shortfall_count": max(0, quota - support_count),
                "reference_admission_policy_fingerprint": (
                    admission_policy.fingerprint
                ),
                "selection_policy_fingerprint": selection_policy_fingerprint,
                "compiler_fingerprint": compiler_fingerprint,
            }
        )
    return pl.DataFrame(rows, schema=REFERENCE_ADMISSION_SUMMARY_SCHEMA).sort(
        ["accepted_taxon_key", "route"]
    )


def _count(frame: pl.DataFrame, decision: str) -> int:
    return frame.filter(pl.col("admission_decision") == decision).height


def _report(
    *,
    decisions: pl.DataFrame,
    support: pl.DataFrame,
    summary: pl.DataFrame,
    admission_policy: ReferenceAdmissionPolicy,
    prototype_scope: str,
    input_fingerprints: Mapping[str, str],
    compiler_fingerprint: str,
    created_at: datetime,
) -> dict[str, object]:
    decision_counts = Counter(decisions["admission_decision"].to_list())
    reason_counts = Counter(
        reason
        for reasons in decisions["admission_reason_codes"].to_list()
        for reason in reasons
    )
    return {
        "schema_version": REFERENCE_ADMISSION_COMPILER_SCHEMA_VERSION,
        "created_at": created_at.astimezone(UTC).isoformat(),
        "compiler_fingerprint": compiler_fingerprint,
        "prototype_scope": prototype_scope,
        "admission_policy": admission_policy.to_dict(),
        "input_fingerprints": dict(sorted(input_fingerprints.items())),
        "counts": {
            "candidates": decisions.height,
            "admitted": decision_counts["admitted"],
            "review_required": decision_counts["review_required"],
            "excluded": decision_counts["excluded"],
            "selected": decisions.filter(
                pl.col("selection_decision") == "selected"
            ).height,
            "provisional_support": support.height,
        },
        "admission_reason_counts": dict(sorted(reason_counts.items())),
        "summary_rows": summary.to_dicts(),
        "evidence_semantics": {
            "provider_assertion": "GBIF provider metadata; not human verification",
            "automated_qa": "24 policy gates with complete ordered outcomes",
            "human_verified": False,
            "provisional_support": (
                "admitted automated evidence selected under independence policy"
            ),
        },
        "non_claims": [
            "YOLOE does not decide species identity",
            "provider assertion is not human verification",
            "provisional support is not calibrated or release-ready evidence",
        ],
        "artifacts": {
            "decisions": REFERENCE_ADMISSION_DECISIONS_FILE,
            "provisional_support": REFERENCE_PROVISIONAL_SUPPORT_FILE,
            "summary": REFERENCE_ADMISSION_SUMMARY_FILE,
            "report": REFERENCE_ADMISSION_REPORT_FILE,
            "markdown": REFERENCE_ADMISSION_SUMMARY_MD_FILE,
        },
    }


def _markdown(
    report: Mapping[str, object], summary: pl.DataFrame
) -> str:
    counts = report["counts"]
    if not isinstance(counts, Mapping):
        raise TypeError("report counts must be a mapping")
    lines = [
        "# Provisional GBIF reference admission summary",
        "",
        f"- Compiler fingerprint: `{report['compiler_fingerprint']}`",
        f"- Candidates: **{counts['candidates']}**",
        f"- Admitted: **{counts['admitted']}**",
        f"- Review required: **{counts['review_required']}**",
        f"- Excluded: **{counts['excluded']}**",
        f"- Provisional support: **{counts['provisional_support']}**",
        "",
        "GBIF identity is provider-asserted and unreviewed. YOLOE supplies only "
        "quality, domain, and life-stage routing evidence.",
        "",
        "| Accepted taxon | Route | Candidates | Admitted | Review | Excluded | "
        "Support | Shortfall |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary.iter_rows(named=True):
        lines.append(
            f"| {row['accepted_taxon_key']} | {row['route']} | "
            f"{row['candidate_count']} | {row['admitted_count']} | "
            f"{row['review_required_count']} | {row['excluded_count']} | "
            f"{row['provisional_support_count']} | "
            f"{row['support_shortfall_count']} |"
        )
    return "\n".join(lines) + "\n"


def _frame_fingerprint(frame: pl.DataFrame) -> str:
    return canonical_semantic_fingerprint(
        {
            "schema": {
                name: str(dtype) for name, dtype in frame.schema.items()
            },
            "rows": frame.to_dicts(),
        }
    )


def _single_frame_text(frame: pl.DataFrame, field: str) -> str:
    values = frame[field].unique().to_list()
    if len(values) != 1 or not isinstance(values[0], str) or not values[0]:
        raise ValueError(f"{field} must have one nonblank value")
    return values[0]


def _optional_text(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        temporary.write_text(text, encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


__all__ = [
    "REFERENCE_ADMISSION_DECISION_SCHEMA",
    "REFERENCE_ADMISSION_DECISIONS_FILE",
    "REFERENCE_ADMISSION_REPORT_FILE",
    "REFERENCE_ADMISSION_SUMMARY_FILE",
    "REFERENCE_ADMISSION_SUMMARY_MD_FILE",
    "REFERENCE_PROVISIONAL_SUPPORT_FILE",
    "REFERENCE_PROVISIONAL_SUPPORT_SCHEMA",
    "ReferenceAdmissionCompilationResult",
    "compile_provisional_gbif_support_bank",
    "validate_reference_provisional_support",
]
