"""Blind review inputs and fail-closed ButterflyLens maturity projections."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
import json
import os
from pathlib import Path
import shutil
from tempfile import NamedTemporaryFile, mkdtemp

import polars as pl

from biominer.common.semantic_hash import canonical_semantic_fingerprint
from biominer.evaluation.dynamic_pool_review import (
    ProbabilityAuditSamplingPolicy,
    ProbabilityAuditSelection,
    validate_probability_audit_selection,
)
from biominer.integration.butterflylens_model_export import (
    ButterflyLensModelLayer,
    validate_butterflylens_model_layer,
    validate_butterflylens_project_projection,
    validate_butterflylens_run_projection,
)
from biominer.integration.butterflylens_pool_handoff import (
    BUTTERFLYLENS_ROLE_DEFAULTS,
)
from biominer.integration.product_handoff import (
    normalize_product_artifacts,
    validate_fingerprint,
)
from biominer.integration.taxalens_quality_export import (
    build_taxalens_review_sampling_frame,
)
from biominer.storage.content_address import sha256_file
from biominer.storage.parquet import write_parquet


BUTTERFLYLENS_REVIEW_ROLES = (
    "review_campaign_inputs",
    "review_assignment_inputs",
    "classification_maturity",
    "release_state",
)
BUTTERFLYLENS_REVIEW_CAMPAIGN_VERSION = (
    "biominer-butterflylens-review-campaign-input-v1.0.0"
)
BUTTERFLYLENS_REVIEW_ASSIGNMENT_VERSION = (
    "biominer-butterflylens-review-assignment-input-v1.0.0"
)
BUTTERFLYLENS_CLASSIFICATION_MATURITY_VERSION = (
    "biominer-butterflylens-classification-maturity-projection-v1.0.0"
)
BUTTERFLYLENS_RELEASE_STATE_VERSION = "biominer-butterflylens-release-state-v1.0.0"
TARGET_CAMPAIGN_VERSION = "butterflylens-verification-campaign:v1.0.0"
TARGET_ASSIGNMENT_VERSION = "butterflylens-verification-assignment:v1.0.0"
TARGET_MATURITY_VERSION = "butterflylens-classification-maturity:v1.0.0"
ASSIGNMENT_POLICY_VERSION = "repeated-independent-v1"
_MATURITY_NAMES = (
    "butterfly_detected",
    "species_candidate_available",
    "community_reviewed",
    "quality_estimate_available",
    "expert_reviewed",
    "release_ready",
)
_TARGET_FIELDS = frozenset({"accepted_taxon_key", "scientific_name", "rank"})
_TARGET_RANKS = frozenset(
    {
        "superfamily",
        "family",
        "subfamily",
        "tribe",
        "genus",
        "species",
        "subspecies",
        "other",
    }
)


BUTTERFLYLENS_REVIEW_ASSIGNMENT_SCHEMA: dict[str, pl.DataType] = {
    "schema_version": pl.String,
    "target_schema_version": pl.String,
    "project_id": pl.String,
    "run_id": pl.String,
    "campaign_id": pl.String,
    "item_id": pl.String,
    "media_object_id": pl.String,
    "flickr_record_id": pl.String,
    "source_sampling_unit_id": pl.String,
    "source_record_fingerprint": pl.String,
    "source_artifact_fingerprint": pl.String,
    "source_frame_fingerprint": pl.String,
    "candidate_accepted_taxon_key": pl.String,
    "candidate_scientific_name": pl.String,
    "review_round": pl.UInt32,
    "assignment_reason": pl.String,
    "assignment_status": pl.String,
    "blind": pl.Boolean,
    "independence_group_key": pl.String,
    "assignment_policy_version": pl.String,
    "required_independent_reviewers": pl.UInt8,
    "inclusion_probability": pl.Float64,
    "sampling_weight": pl.Float64,
    "sampling_design": pl.String,
    "representative": pl.Boolean,
    "geographic_cluster_id": pl.String,
    "no_geo": pl.Boolean,
    "raw_score_is_probability": pl.Boolean,
    "model_evidence_fingerprints": pl.List(pl.String),
    "reviewer_identity_included": pl.Boolean,
    "assignment_created": pl.Boolean,
    "database_primary_key_included": pl.Boolean,
    "occurrence_release_authorized": pl.Boolean,
    "visibility": pl.String,
    "assignment_input_fingerprint": pl.String,
}


def _maturity_schema() -> dict[str, pl.DataType]:
    schema: dict[str, pl.DataType] = {
        "schema_version": pl.String,
        "target_schema_version": pl.String,
        "project_id": pl.String,
        "run_id": pl.String,
        "image_id": pl.String,
        "source_record_fingerprint": pl.String,
        "observed_at": pl.String,
    }
    for name in _MATURITY_NAMES:
        schema[f"{name}_status"] = pl.String
        schema[f"{name}_value"] = pl.Boolean
        schema[f"{name}_reason"] = pl.String
        schema[f"{name}_evidence_fingerprints"] = pl.List(pl.String)
    schema.update(
        {
            "database_primary_key_included": pl.Boolean,
            "scientific_claim_allowed": pl.Boolean,
            "projection_fingerprint": pl.String,
        }
    )
    return schema


BUTTERFLYLENS_CLASSIFICATION_MATURITY_SCHEMA = _maturity_schema()


@dataclass(frozen=True, slots=True)
class ButterflyLensReviewLayer:
    campaign: dict[str, object]
    assignment_inputs: pl.DataFrame
    classification_maturity: pl.DataFrame
    release_state: dict[str, object]


@dataclass(frozen=True, slots=True)
class ButterflyLensReviewExport:
    root: Path
    review_directory: Path
    artifacts: tuple[dict[str, object], ...]


def build_butterflylens_review_layer(
    *,
    project: Mapping[str, object],
    run: Mapping[str, object],
    model_layer: ButterflyLensModelLayer,
    selection: ProbabilityAuditSelection,
    sampling_policy: ProbabilityAuditSamplingPolicy,
    target: Mapping[str, object],
    observed_at: str | datetime,
) -> ButterflyLensReviewLayer:
    """Build a campaign and work items without creating reviewer assignments."""

    validate_butterflylens_project_projection(project)
    validate_butterflylens_run_projection(run)
    validate_butterflylens_model_layer(model_layer)
    if (
        run["project_id"] != project["project_id"]
        or run["project_fingerprint"] != project["project_fingerprint"]
    ):
        raise ValueError("ButterflyLens review project and run differ")
    _validate_layer_scope(
        model_layer, project_id=project["project_id"], run_id=run["run_id"]
    )
    if not isinstance(selection, ProbabilityAuditSelection):
        raise TypeError("selection must be a ProbabilityAuditSelection")
    if not isinstance(sampling_policy, ProbabilityAuditSamplingPolicy):
        raise TypeError("sampling_policy must be a ProbabilityAuditSamplingPolicy")
    validate_probability_audit_selection(selection.register, selection.sample)
    if (
        selection.population_count != selection.register.height
        or selection.selected_count != selection.sample.height
        or selection.sample.is_empty()
    ):
        raise ValueError("ButterflyLens review selection counts differ")
    if set(selection.sample["sample_policy_fingerprint"].to_list()) != {
        sampling_policy.fingerprint
    }:
        raise ValueError("ButterflyLens review selection policy differs")
    review_frame = build_taxalens_review_sampling_frame(
        selection, policy=sampling_policy
    )
    normalized_target = _normalize_target(target)
    instant = _utc(observed_at)
    campaign = _build_campaign(
        project_id=str(project["project_id"]),
        run_id=str(run["run_id"]),
        target=normalized_target,
        selection=selection,
        review_frame=review_frame,
        observed_at=instant,
    )
    assignments = _build_assignments(
        project_id=str(project["project_id"]),
        run_id=str(run["run_id"]),
        campaign_id=str(campaign["campaign_id"]),
        review_frame=review_frame,
        model_layer=model_layer,
    )
    maturity = _build_maturity(model_layer=model_layer, observed_at=instant)
    release_state = _build_release_state(
        project_id=str(project["project_id"]),
        run_id=str(run["run_id"]),
        campaign_fingerprint=str(campaign["campaign_fingerprint"]),
        assignment_fingerprints=assignments["assignment_input_fingerprint"].to_list(),
        maturity_fingerprints=maturity["projection_fingerprint"].to_list(),
    )
    layer = ButterflyLensReviewLayer(campaign, assignments, maturity, release_state)
    validate_butterflylens_review_layer(layer)
    return layer


def validate_butterflylens_review_layer(layer: ButterflyLensReviewLayer) -> None:
    if not isinstance(layer, ButterflyLensReviewLayer):
        raise TypeError("layer must be a ButterflyLensReviewLayer")
    _validate_campaign(layer.campaign)
    _validate_assignments(layer.assignment_inputs)
    _validate_maturity(layer.classification_maturity)
    _validate_release_state(layer.release_state)
    campaign_id = layer.campaign["campaign_id"]
    project_id = layer.campaign["project_id"]
    run_id = layer.campaign["run_id"]
    if set(layer.assignment_inputs["campaign_id"].to_list()) != {campaign_id}:
        raise ValueError("ButterflyLens review campaign relationship differs")
    for frame in (layer.assignment_inputs, layer.classification_maturity):
        if set(frame["project_id"].to_list()) != {project_id} or set(
            frame["run_id"].to_list()
        ) != {run_id}:
            raise ValueError("ButterflyLens review layer scope differs")
    if (
        layer.release_state["project_id"] != project_id
        or layer.release_state["run_id"] != run_id
        or layer.release_state["campaign_fingerprint"]
        != layer.campaign["campaign_fingerprint"]
    ):
        raise ValueError("ButterflyLens release projection scope differs")
    if layer.release_state["assignment_input_fingerprints"] != (
        sorted(layer.assignment_inputs["assignment_input_fingerprint"].to_list())
    ) or layer.release_state["maturity_fingerprints"] != (
        sorted(layer.classification_maturity["projection_fingerprint"].to_list())
    ):
        raise ValueError("ButterflyLens release projection lineage differs")


def export_butterflylens_review_evidence(
    *, layer: ButterflyLensReviewLayer, output_root: str | Path
) -> ButterflyLensReviewExport:
    """Stage, re-read, and atomically publish all review-layer roles."""

    validate_butterflylens_review_layer(layer)
    root = Path(output_root)
    if root.is_symlink():
        raise ValueError("ButterflyLens handoff root must not be a symlink")
    root.mkdir(parents=True, exist_ok=True)
    root = root.resolve()
    artifacts_directory = root / "artifacts"
    if artifacts_directory.is_symlink():
        raise ValueError("ButterflyLens artifacts directory must not be a symlink")
    artifacts_directory.mkdir(exist_ok=True)
    review_directory = artifacts_directory / "review"
    if review_directory.exists():
        raise FileExistsError(
            f"ButterflyLens review directory is create-only: {review_directory}"
        )
    staging_root = Path(
        mkdtemp(dir=artifacts_directory, prefix=".butterflylens-review-")
    )
    staging_review = staging_root / "artifacts" / "review"
    staging_review.mkdir(parents=True)
    try:
        descriptors = (
            _write_json_role("review_campaign_inputs", layer.campaign, staging_review),
            _write_frame_role(
                "review_assignment_inputs",
                layer.assignment_inputs,
                "assignment_input_fingerprint",
                staging_review,
            ),
            _write_frame_role(
                "classification_maturity",
                layer.classification_maturity,
                "projection_fingerprint",
                staging_review,
            ),
            _write_json_role("release_state", layer.release_state, staging_review),
        )
        validate_butterflylens_review_export(staging_root, descriptors)
        staging_review.replace(review_directory)
    finally:
        shutil.rmtree(staging_root, ignore_errors=True)
    result = ButterflyLensReviewExport(root, review_directory, descriptors)
    validate_butterflylens_review_export(result.root, result.artifacts)
    return result


def validate_butterflylens_review_export(
    root: str | Path, descriptors: Sequence[Mapping[str, object]]
) -> None:
    normalized = normalize_product_artifacts(
        descriptors,
        required_roles=BUTTERFLYLENS_REVIEW_ROLES,
        producer_repository="karikris/BioMiner",
        producer_commit="0" * 40,
    )
    by_role = {str(row["role"]): row for row in normalized}
    root_path = Path(root)
    if root_path.is_symlink():
        raise ValueError("ButterflyLens handoff root must not be a symlink")
    directory = root_path / "artifacts" / "review"
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError("ButterflyLens review directory is unavailable")
    expected_paths: set[Path] = set()
    values: dict[str, object] = {}
    frame_roles = {
        "review_assignment_inputs": (
            BUTTERFLYLENS_REVIEW_ASSIGNMENT_SCHEMA,
            "assignment_input_fingerprint",
        ),
        "classification_maturity": (
            BUTTERFLYLENS_CLASSIFICATION_MATURITY_SCHEMA,
            "projection_fingerprint",
        ),
    }
    for role in BUTTERFLYLENS_REVIEW_ROLES:
        descriptor = by_role[role]
        filename, schema_version = BUTTERFLYLENS_ROLE_DEFAULTS[role]
        relative = f"artifacts/review/{filename}"
        if (
            descriptor["availability"] != "available"
            or descriptor["relative_path"] != relative
            or descriptor["schema_version"] != schema_version
            or descriptor["evidence_maturity_label"] is not None
        ):
            raise ValueError(f"ButterflyLens review role {role!r} contract differs")
        path = root_path / relative
        expected_paths.add(path.resolve())
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"ButterflyLens review role {role!r} is unavailable")
        if (
            path.stat().st_size != descriptor["byte_count"]
            or sha256_file(path) != descriptor["sha256"]
        ):
            raise ValueError(f"ButterflyLens review role {role!r} identity differs")
        if role in frame_roles:
            frame = pl.read_parquet(path)
            schema, fingerprint_field = frame_roles[role]
            if frame.schema != schema:
                raise ValueError(f"ButterflyLens review role {role!r} schema differs")
            values[role] = frame
            expected_semantic = _frame_semantic(role, frame, fingerprint_field)
            expected_parents = _frame_parents(frame, fingerprint_field)
            row_count = frame.height
        else:
            payload = json.loads(path.read_text(encoding="utf-8"))
            values[role] = payload
            expected_semantic = str(payload[f"{_json_prefix(role)}_fingerprint"])
            expected_parents = _json_parents(role, payload)
            row_count = 1
        if (
            descriptor["row_count"] != row_count
            or descriptor["semantic_fingerprint"] != expected_semantic
            or descriptor["parent_fingerprints"] != expected_parents
        ):
            raise ValueError(f"ButterflyLens review role {role!r} semantics differ")
    layer = ButterflyLensReviewLayer(
        campaign=values["review_campaign_inputs"],
        assignment_inputs=values["review_assignment_inputs"],
        classification_maturity=values["classification_maturity"],
        release_state=values["release_state"],
    )
    validate_butterflylens_review_layer(layer)
    entries = tuple(directory.iterdir())
    if any(path.is_symlink() or not path.is_file() for path in entries):
        raise ValueError("ButterflyLens review directory has unsafe entries")
    if {path.resolve() for path in entries} != expected_paths:
        raise ValueError("ButterflyLens review directory file set differs")


def _build_campaign(
    *,
    project_id: str,
    run_id: str,
    target: Mapping[str, object],
    selection: ProbabilityAuditSelection,
    review_frame: pl.DataFrame,
    observed_at: str,
) -> dict[str, object]:
    population = selection.register.height
    strata: list[dict[str, object]] = []
    for stratum_id in sorted(selection.register["analysis_stratum_id"].unique()):
        rows = selection.register.filter(pl.col("analysis_stratum_id") == stratum_id)
        selected_count = rows.filter(pl.col("selected")).height
        strata.append(
            {
                "stratum_id": stratum_id,
                "label": stratum_id,
                "population_count": rows.height,
                "target_sample_count": selected_count,
                "population_weight": rows.height / population,
            }
        )
    question_fingerprint = canonical_semantic_fingerprint(
        {
            "question": "Does this image provide human-supported evidence for the target taxon?",
            "target": target,
            "blind": True,
            "outcomes": ["yes", "no", "uncertain", "media_failure", "deferred"],
        }
    )
    base: dict[str, object] = {
        "schema_version": BUTTERFLYLENS_REVIEW_CAMPAIGN_VERSION,
        "target_schema_version": TARGET_CAMPAIGN_VERSION,
        "project_id": project_id,
        "run_id": run_id,
        "kind": "flickr_target_verification",
        "status": "draft",
        "target": dict(target),
        "source_providers": ["flickr"],
        "question_fingerprint": question_fingerprint,
        "manifest_fingerprint": selection.register_fingerprint,
        "sampling_plan": {
            "plan_id": review_frame["sampling_plan_id"][0],
            "purpose": "quality_estimation",
            "design": "stratified_random",
            "representative": True,
            "blind": True,
            "inclusion_probabilities_recorded": True,
            "grouping_keys": sorted(review_frame["grouping_keys"][0].to_list()),
            "strata": strata,
            "quality_estimation_eligible": True,
            "quality_estimation_blockers": [],
        },
        "review_requirement": {
            "required_independent_reviewers": 2,
            "second_review_policy": "always",
            "adjudication_required_on_conflict": True,
            "expert_gate_required_for_release": True,
        },
        "blind_policy": {
            "enabled": True,
            "hidden_fields": [
                "control_identity",
                "flickr_query_term",
                "model_label",
                "model_score",
                "other_reviews",
                "source_comments",
            ],
        },
        "public_replay": False,
        "scientific_claim_allowed": False,
        "created_at": observed_at,
        "updated_at": observed_at,
        "assignment_policy_version": ASSIGNMENT_POLICY_VERSION,
        "assignment_authority": False,
        "reviewer_identity_included": False,
        "database_primary_key_included": False,
    }
    digest = canonical_semantic_fingerprint(base).removeprefix("sha256:")
    identified = {**base, "campaign_id": f"biominer-campaign:{digest}"}
    return {
        **identified,
        "campaign_fingerprint": canonical_semantic_fingerprint(identified),
    }


def _build_assignments(
    *,
    project_id: str,
    run_id: str,
    campaign_id: str,
    review_frame: pl.DataFrame,
    model_layer: ButterflyLensModelLayer,
) -> pl.DataFrame:
    source_by_fingerprint: dict[str, dict[str, object]] = {}
    for source in model_layer.flickr_source_records.iter_rows(named=True):
        fingerprint = str(source["source_record_fingerprint"])
        if fingerprint in source_by_fingerprint:
            raise ValueError("ButterflyLens source record fingerprint repeats")
        source_by_fingerprint[fingerprint] = source
    media_by_source = {
        str(row["flickr_record_id"]): row
        for row in model_layer.media_objects.iter_rows(named=True)
    }
    evidence_by_media: dict[str, list[dict[str, object]]] = {}
    for evidence in model_layer.model_evidence.iter_rows(named=True):
        evidence_by_media.setdefault(str(evidence["media_object_id"]), []).append(
            evidence
        )
    rows: list[dict[str, object]] = []
    for sample in review_frame.iter_rows(named=True):
        source = source_by_fingerprint.get(str(sample["source_record_hash"]))
        if source is None:
            raise ValueError("ButterflyLens selected review source is unavailable")
        media = media_by_source[str(source["flickr_record_id"])]
        evidence_rows = evidence_by_media[str(media["media_object_id"])]
        matching = [
            row
            for row in evidence_rows
            if row["candidate_accepted_taxon_key"]
            == sample["candidate_species_accepted_taxon_key"]
        ]
        if len(matching) != 1:
            raise ValueError("ButterflyLens selected review model candidate differs")
        base = {
            "schema_version": BUTTERFLYLENS_REVIEW_ASSIGNMENT_VERSION,
            "target_schema_version": TARGET_ASSIGNMENT_VERSION,
            "project_id": project_id,
            "run_id": run_id,
            "campaign_id": campaign_id,
            "item_id": media["media_object_id"],
            "media_object_id": media["media_object_id"],
            "flickr_record_id": source["flickr_record_id"],
            "source_sampling_unit_id": sample["source_sampling_unit_id"],
            "source_record_fingerprint": source["source_record_fingerprint"],
            "source_artifact_fingerprint": sample["source_artifact_fingerprint"],
            "source_frame_fingerprint": sample["source_frame_fingerprint"],
            "candidate_accepted_taxon_key": sample[
                "candidate_species_accepted_taxon_key"
            ],
            "candidate_scientific_name": sample["candidate_species_scientific_name"],
            "review_round": 1,
            "assignment_reason": "ordinary",
            "assignment_status": "unassigned",
            "blind": True,
            "independence_group_key": sample["independent_unit"],
            "assignment_policy_version": ASSIGNMENT_POLICY_VERSION,
            "required_independent_reviewers": 2,
            "inclusion_probability": sample["inclusion_probability"],
            "sampling_weight": sample["sampling_weight"],
            "sampling_design": sample["sampling_design"],
            "representative": True,
            "geographic_cluster_id": sample["geographic_cluster_id"],
            "no_geo": sample["no_geo"],
            "raw_score_is_probability": False,
            "model_evidence_fingerprints": sorted(
                str(row["evidence_fingerprint"]) for row in matching
            ),
            "reviewer_identity_included": False,
            "assignment_created": False,
            "database_primary_key_included": False,
            "occurrence_release_authorized": False,
            "visibility": "private",
        }
        rows.append(
            {
                **base,
                "assignment_input_fingerprint": canonical_semantic_fingerprint(base),
            }
        )
    frame = pl.DataFrame(
        rows, schema=BUTTERFLYLENS_REVIEW_ASSIGNMENT_SCHEMA, strict=True
    ).sort("item_id", "source_sampling_unit_id")
    _validate_assignments(frame)
    return frame


def _build_maturity(
    *, model_layer: ButterflyLensModelLayer, observed_at: str
) -> pl.DataFrame:
    sources = {
        str(row["flickr_record_id"]): row
        for row in model_layer.flickr_source_records.iter_rows(named=True)
    }
    evidence_by_media: dict[str, list[dict[str, object]]] = {}
    for evidence in model_layer.model_evidence.iter_rows(named=True):
        evidence_by_media.setdefault(str(evidence["media_object_id"]), []).append(
            evidence
        )
    rows: list[dict[str, object]] = []
    for media in model_layer.media_objects.iter_rows(named=True):
        source = sources[str(media["flickr_record_id"])]
        evidence = evidence_by_media.get(str(media["media_object_id"]), [])
        if not evidence:
            raise ValueError("ButterflyLens maturity model evidence is unavailable")
        model_fingerprints = sorted(
            str(row["evidence_fingerprint"]) for row in evidence
        )
        base: dict[str, object] = {
            "schema_version": BUTTERFLYLENS_CLASSIFICATION_MATURITY_VERSION,
            "target_schema_version": TARGET_MATURITY_VERSION,
            "project_id": media["project_id"],
            "run_id": media["run_id"],
            "image_id": media["media_object_id"],
            "source_record_fingerprint": source["source_row_fingerprint"],
            "observed_at": observed_at,
            **_unavailable_maturity(
                "butterfly_detected", "YOLOE detection evidence was not joined"
            ),
            **_available_maturity(
                "species_candidate_available", True, model_fingerprints
            ),
            **_unavailable_maturity(
                "community_reviewed", "community review has not occurred"
            ),
            **_unavailable_maturity(
                "quality_estimate_available", "reviewed quality is unavailable"
            ),
            **_unavailable_maturity(
                "expert_reviewed", "expert review has not occurred"
            ),
            **_unavailable_maturity(
                "release_ready", "release gates are downstream and unresolved"
            ),
            "database_primary_key_included": False,
            "scientific_claim_allowed": False,
        }
        rows.append(
            {
                **base,
                "projection_fingerprint": canonical_semantic_fingerprint(base),
            }
        )
    frame = pl.DataFrame(
        rows, schema=BUTTERFLYLENS_CLASSIFICATION_MATURITY_SCHEMA, strict=True
    ).sort("image_id")
    _validate_maturity(frame)
    return frame


def _build_release_state(
    *,
    project_id: str,
    run_id: str,
    campaign_fingerprint: str,
    assignment_fingerprints: Sequence[str],
    maturity_fingerprints: Sequence[str],
) -> dict[str, object]:
    blockers = [
        "community_review_incomplete",
        "duplicate_independence_unverified",
        "expert_review_incomplete",
        "human_supported_identity_unavailable",
        "quality_threshold_unavailable",
        "release_evidence_packet_incomplete",
        "rights_provenance_unverified",
    ]
    body: dict[str, object] = {
        "schema_version": BUTTERFLYLENS_RELEASE_STATE_VERSION,
        "project_id": project_id,
        "run_id": run_id,
        "candidate_state": "blocked",
        "release_ready": False,
        "all_release_gates_passed": False,
        "release_blockers": blockers,
        "gate_states": {
            "human_supported_identity": False,
            "qualified_consensus_passed": False,
            "expert_review_passed": False,
            "coordinate_valid": False,
            "date_valid": False,
            "duplicate_independence_passed": False,
            "rights_provenance_passed": False,
            "quality_threshold_passed": False,
            "no_unresolved_conflict": False,
            "evidence_packet_complete": False,
        },
        "campaign_fingerprint": campaign_fingerprint,
        "assignment_input_fingerprints": sorted(
            validate_fingerprint(value, field="assignment_fingerprints")
            for value in assignment_fingerprints
        ),
        "maturity_fingerprints": sorted(
            validate_fingerprint(value, field="maturity_fingerprints")
            for value in maturity_fingerprints
        ),
        "review_event_count": 0,
        "reviewer_identity_included": False,
        "authorization_included": False,
        "downstream_authorization_required": True,
        "database_primary_key_included": False,
        "scientific_claim_allowed": False,
    }
    return {**body, "release_fingerprint": canonical_semantic_fingerprint(body)}


def _validate_campaign(campaign: Mapping[str, object]) -> None:
    if not isinstance(campaign, Mapping):
        raise TypeError("campaign must be a mapping")
    body = dict(campaign)
    fingerprint = body.pop("campaign_fingerprint", None)
    if fingerprint != canonical_semantic_fingerprint(body):
        raise ValueError("ButterflyLens review campaign fingerprint differs")
    campaign_id = body.pop("campaign_id", None)
    expected_id = (
        "biominer-campaign:"
        f"{canonical_semantic_fingerprint(body).removeprefix('sha256:')}"
    )
    if campaign_id != expected_id:
        raise ValueError("ButterflyLens review campaign identity differs")
    if (
        campaign.get("schema_version") != BUTTERFLYLENS_REVIEW_CAMPAIGN_VERSION
        or campaign.get("target_schema_version") != TARGET_CAMPAIGN_VERSION
        or campaign.get("status") != "draft"
        or campaign.get("scientific_claim_allowed") is not False
        or campaign.get("assignment_authority") is not False
        or campaign.get("reviewer_identity_included") is not False
        or campaign.get("database_primary_key_included") is not False
        or campaign.get("public_replay") is not False
    ):
        raise ValueError("ButterflyLens review campaign authority differs")
    if campaign.get("source_providers") != ["flickr"]:
        raise ValueError("ButterflyLens review campaign provider differs")
    _normalize_target(campaign.get("target"))
    for field in ("question_fingerprint", "manifest_fingerprint"):
        _sha(campaign.get(field), field=field)
    created = _canonical_utc(campaign.get("created_at"), field="created_at")
    updated = _canonical_utc(campaign.get("updated_at"), field="updated_at")
    if updated < created:
        raise ValueError("ButterflyLens review campaign timestamps differ")
    plan = campaign.get("sampling_plan")
    if not isinstance(plan, Mapping) or (
        plan.get("purpose") != "quality_estimation"
        or plan.get("design") != "stratified_random"
        or plan.get("representative") is not True
        or plan.get("blind") is not True
        or plan.get("inclusion_probabilities_recorded") is not True
        or plan.get("quality_estimation_eligible") is not True
        or plan.get("quality_estimation_blockers") != []
    ):
        raise ValueError("ButterflyLens review sampling plan differs")
    _text(plan.get("plan_id"), field="sampling_plan.plan_id")
    grouping_keys = plan.get("grouping_keys")
    if (
        not isinstance(grouping_keys, list)
        or grouping_keys != sorted(set(grouping_keys))
        or not grouping_keys
    ):
        raise ValueError("ButterflyLens review grouping keys are not canonical")
    strata = plan.get("strata")
    if not isinstance(strata, list) or not strata:
        raise ValueError("ButterflyLens review strata are unavailable")
    stratum_ids: list[str] = []
    population_weight = 0.0
    for stratum in strata:
        if not isinstance(stratum, Mapping) or set(stratum) != {
            "stratum_id",
            "label",
            "population_count",
            "target_sample_count",
            "population_weight",
        }:
            raise ValueError("ButterflyLens review stratum fields differ")
        stratum_ids.append(_text(stratum["stratum_id"], field="stratum_id"))
        _text(stratum["label"], field="stratum.label", maximum=160)
        population_count = _nonnegative_int(
            stratum["population_count"], field="stratum.population_count"
        )
        sample_count = _nonnegative_int(
            stratum["target_sample_count"], field="stratum.target_sample_count"
        )
        weight = _proportion(stratum["population_weight"], field="population_weight")
        if sample_count > population_count:
            raise ValueError("ButterflyLens review stratum sample exceeds population")
        population_weight += weight
    if stratum_ids != sorted(set(stratum_ids)) or abs(population_weight - 1.0) > 1e-12:
        raise ValueError("ButterflyLens review strata are not canonical")
    requirement = campaign.get("review_requirement")
    if requirement != {
        "required_independent_reviewers": 2,
        "second_review_policy": "always",
        "adjudication_required_on_conflict": True,
        "expert_gate_required_for_release": True,
    }:
        raise ValueError("ButterflyLens review requirement differs")
    blind = campaign.get("blind_policy")
    if not isinstance(blind, Mapping) or blind.get("enabled") is not True:
        raise ValueError("ButterflyLens blind policy differs")
    hidden = blind.get("hidden_fields")
    if not isinstance(hidden, list) or hidden != sorted(set(hidden)):
        raise ValueError("ButterflyLens blind fields are not canonical")
    _sha(fingerprint, field="campaign_fingerprint")


def _validate_assignments(frame: pl.DataFrame) -> None:
    if (
        frame.schema != BUTTERFLYLENS_REVIEW_ASSIGNMENT_SCHEMA
        or frame.is_empty()
        or not frame.equals(frame.sort("item_id", "source_sampling_unit_id"))
    ):
        raise ValueError("ButterflyLens review assignment schema or rows differ")
    if (
        frame["assignment_input_fingerprint"].n_unique() != frame.height
        or frame["source_sampling_unit_id"].n_unique() != frame.height
    ):
        raise ValueError("ButterflyLens review assignment identities repeat")
    if frame.filter(
        (pl.col("schema_version") != BUTTERFLYLENS_REVIEW_ASSIGNMENT_VERSION)
        | (pl.col("target_schema_version") != TARGET_ASSIGNMENT_VERSION)
        | (pl.col("review_round") != 1)
        | (pl.col("assignment_reason") != "ordinary")
        | (pl.col("assignment_status") != "unassigned")
        | ~pl.col("blind")
        | (pl.col("assignment_policy_version") != ASSIGNMENT_POLICY_VERSION)
        | (pl.col("required_independent_reviewers") != 2)
        | ~pl.col("representative")
        | pl.col("raw_score_is_probability")
        | pl.col("reviewer_identity_included")
        | pl.col("assignment_created")
        | pl.col("database_primary_key_included")
        | pl.col("occurrence_release_authorized")
        | (pl.col("visibility") != "private")
        | ~pl.col("inclusion_probability").is_between(0.0, 1.0, closed="right")
        | (
            (pl.col("inclusion_probability") * pl.col("sampling_weight") - 1.0)
            .abs()
            .gt(1e-12)
        )
        | (pl.col("no_geo") & pl.col("geographic_cluster_id").is_not_null())
    ).height:
        raise ValueError("ButterflyLens review assignment authority differs")
    for row in frame.iter_rows(named=True):
        if row["item_id"] != row["media_object_id"]:
            raise ValueError("ButterflyLens review item identity differs")
        for field in (
            "source_record_fingerprint",
            "source_artifact_fingerprint",
            "source_frame_fingerprint",
        ):
            _sha(row[field], field=field)
        fingerprints = _fingerprints(row["model_evidence_fingerprints"])
        if list(row["model_evidence_fingerprints"]) != fingerprints:
            raise ValueError("ButterflyLens review model lineage is not canonical")
        payload = dict(row)
        fingerprint = payload.pop("assignment_input_fingerprint")
        if fingerprint != canonical_semantic_fingerprint(payload):
            raise ValueError("ButterflyLens review assignment fingerprint differs")
        _sha(fingerprint, field="assignment_input_fingerprint")


def _validate_maturity(frame: pl.DataFrame) -> None:
    if (
        frame.schema != BUTTERFLYLENS_CLASSIFICATION_MATURITY_SCHEMA
        or frame.is_empty()
        or not frame.equals(frame.sort("image_id"))
    ):
        raise ValueError("ButterflyLens classification maturity schema or rows differ")
    if frame["projection_fingerprint"].n_unique() != frame.height:
        raise ValueError("ButterflyLens maturity fingerprints repeat")
    if frame.filter(
        (pl.col("schema_version") != BUTTERFLYLENS_CLASSIFICATION_MATURITY_VERSION)
        | (pl.col("target_schema_version") != TARGET_MATURITY_VERSION)
        | pl.col("database_primary_key_included")
        | pl.col("scientific_claim_allowed")
    ).height:
        raise ValueError("ButterflyLens maturity authority differs")
    for row in frame.iter_rows(named=True):
        _sha(row["source_record_fingerprint"], field="source_record_fingerprint")
        _canonical_utc(row["observed_at"], field="observed_at")
        for name in _MATURITY_NAMES:
            status = row[f"{name}_status"]
            value = row[f"{name}_value"]
            reason = row[f"{name}_reason"]
            fingerprints = _fingerprints(
                row[f"{name}_evidence_fingerprints"], allow_empty=True
            )
            if status == "available":
                if value is not True or reason is not None or not fingerprints:
                    raise ValueError(
                        "ButterflyLens available maturity evidence differs"
                    )
                if name != "species_candidate_available":
                    raise ValueError("ButterflyLens human maturity was fabricated")
            elif status != "unavailable" or value is not None or not reason:
                raise ValueError("ButterflyLens unavailable maturity evidence differs")
            elif fingerprints:
                raise ValueError("ButterflyLens unavailable maturity has evidence")
        payload = dict(row)
        fingerprint = payload.pop("projection_fingerprint")
        if fingerprint != canonical_semantic_fingerprint(payload):
            raise ValueError("ButterflyLens maturity fingerprint differs")
        _sha(fingerprint, field="projection_fingerprint")


def _validate_release_state(release: Mapping[str, object]) -> None:
    if not isinstance(release, Mapping):
        raise TypeError("release state must be a mapping")
    body = dict(release)
    fingerprint = body.pop("release_fingerprint", None)
    if fingerprint != canonical_semantic_fingerprint(body):
        raise ValueError("ButterflyLens release fingerprint differs")
    gates = release.get("gate_states")
    if (
        release.get("schema_version") != BUTTERFLYLENS_RELEASE_STATE_VERSION
        or release.get("candidate_state") != "blocked"
        or release.get("release_ready") is not False
        or release.get("all_release_gates_passed") is not False
        or not release.get("release_blockers")
        or not isinstance(gates, Mapping)
        or any(value is not False for value in gates.values())
        or release.get("review_event_count") != 0
        or release.get("reviewer_identity_included") is not False
        or release.get("authorization_included") is not False
        or release.get("downstream_authorization_required") is not True
        or release.get("database_primary_key_included") is not False
        or release.get("scientific_claim_allowed") is not False
    ):
        raise ValueError("ButterflyLens release authority differs")
    blockers = release.get("release_blockers")
    if not isinstance(blockers, list) or blockers != sorted(set(blockers)):
        raise ValueError("ButterflyLens release blockers are not canonical")
    expected_gates = {
        "human_supported_identity",
        "qualified_consensus_passed",
        "expert_review_passed",
        "coordinate_valid",
        "date_valid",
        "duplicate_independence_passed",
        "rights_provenance_passed",
        "quality_threshold_passed",
        "no_unresolved_conflict",
        "evidence_packet_complete",
    }
    if set(gates) != expected_gates:
        raise ValueError("ButterflyLens release gate fields differ")
    assignments = _fingerprints(release.get("assignment_input_fingerprints"))
    maturity = _fingerprints(release.get("maturity_fingerprints"))
    if (
        release.get("assignment_input_fingerprints") != assignments
        or release.get("maturity_fingerprints") != maturity
    ):
        raise ValueError("ButterflyLens release lineage is not canonical")
    _sha(release.get("campaign_fingerprint"), field="campaign_fingerprint")
    _sha(fingerprint, field="release_fingerprint")


def _available_maturity(
    name: str, value: bool, fingerprints: Sequence[str]
) -> dict[str, object]:
    return {
        f"{name}_status": "available",
        f"{name}_value": value,
        f"{name}_reason": None,
        f"{name}_evidence_fingerprints": sorted(fingerprints),
    }


def _unavailable_maturity(name: str, reason: str) -> dict[str, object]:
    return {
        f"{name}_status": "unavailable",
        f"{name}_value": None,
        f"{name}_reason": reason,
        f"{name}_evidence_fingerprints": [],
    }


def _write_json_role(
    role: str, payload: Mapping[str, object], directory: Path
) -> dict[str, object]:
    filename, schema_version = BUTTERFLYLENS_ROLE_DEFAULTS[role]
    path = directory / filename
    _write_json_create_only(path, payload)
    prefix = _json_prefix(role)
    return _descriptor(
        role,
        path,
        schema_version,
        str(payload[f"{prefix}_fingerprint"]),
        1,
        _json_parents(role, payload),
    )


def _write_frame_role(
    role: str, frame: pl.DataFrame, fingerprint_field: str, directory: Path
) -> dict[str, object]:
    filename, schema_version = BUTTERFLYLENS_ROLE_DEFAULTS[role]
    path = write_parquet(frame, directory / filename, overwrite=False)
    return _descriptor(
        role,
        path,
        schema_version,
        _frame_semantic(role, frame, fingerprint_field),
        frame.height,
        _frame_parents(frame, fingerprint_field),
    )


def _descriptor(
    role: str,
    path: Path,
    schema_version: str,
    semantic_fingerprint: str,
    row_count: int,
    parents: Sequence[str],
) -> dict[str, object]:
    return {
        "role": role,
        "availability": "available",
        "unavailable_reason": None,
        "relative_path": f"artifacts/review/{path.name}",
        "media_type": (
            "application/json"
            if path.suffix == ".json"
            else "application/vnd.apache.parquet"
        ),
        "schema_version": schema_version,
        "semantic_fingerprint": semantic_fingerprint,
        "sha256": sha256_file(path),
        "byte_count": path.stat().st_size,
        "row_count": row_count,
        "parent_fingerprints": sorted(set(parents)),
        "evidence_maturity_label": None,
    }


def _frame_semantic(role: str, frame: pl.DataFrame, field: str) -> str:
    return canonical_semantic_fingerprint(
        {
            "role": role,
            "schema_version": BUTTERFLYLENS_ROLE_DEFAULTS[role][1],
            "row_fingerprints": frame[field].to_list(),
        }
    )


def _frame_parents(frame: pl.DataFrame, fingerprint_field: str) -> list[str]:
    values: set[str] = set()
    for field in frame.columns:
        if not (field.endswith("fingerprint") or field.endswith("fingerprints")):
            continue
        for item in frame[field].to_list():
            candidates = item if isinstance(item, list) else [item]
            values.update(
                value
                for value in candidates
                if isinstance(value, str) and value.startswith("sha256:")
            )
    return sorted(values - set(frame[fingerprint_field].to_list()))


def _json_prefix(role: str) -> str:
    return "campaign" if role == "review_campaign_inputs" else "release"


def _json_parents(role: str, payload: Mapping[str, object]) -> list[str]:
    if role == "review_campaign_inputs":
        return sorted(
            {
                str(payload["question_fingerprint"]),
                str(payload["manifest_fingerprint"]),
            }
        )
    return sorted(
        {
            str(payload["campaign_fingerprint"]),
            *payload["assignment_input_fingerprints"],
            *payload["maturity_fingerprints"],
        }
    )


def _write_json_create_only(path: Path, payload: Mapping[str, object]) -> None:
    data = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    temporary: Path | None = None
    try:
        with NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as output:
            temporary = Path(output.name)
            output.write(data)
        os.link(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _normalize_target(target: Mapping[str, object]) -> dict[str, object]:
    if not isinstance(target, Mapping) or set(target) != _TARGET_FIELDS:
        raise ValueError("ButterflyLens review target fields differ")
    rank = _text(target["rank"], field="rank")
    if rank not in _TARGET_RANKS:
        raise ValueError("ButterflyLens review target rank differs")
    return {
        "accepted_taxon_key": _text(
            target["accepted_taxon_key"], field="accepted_taxon_key"
        ),
        "scientific_name": _text(
            target["scientific_name"], field="scientific_name", maximum=240
        ),
        "rank": rank,
    }


def _validate_layer_scope(
    layer: ButterflyLensModelLayer, *, project_id: object, run_id: object
) -> None:
    for frame in (
        layer.flickr_source_records,
        layer.media_objects,
        layer.model_evidence,
    ):
        if set(frame["project_id"].to_list()) != {project_id} or set(
            frame["run_id"].to_list()
        ) != {run_id}:
            raise ValueError("ButterflyLens review model-layer scope differs")


def _fingerprints(value: object, *, allow_empty: bool = False) -> list[str]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError("fingerprints must be a sequence")
    normalized = sorted(
        {validate_fingerprint(item, field="fingerprints") for item in value}
    )
    if not normalized and not allow_empty:
        raise ValueError("fingerprints must be nonempty")
    return normalized


def _sha(value: object, *, field: str) -> str:
    return validate_fingerprint(value, field=field)


def _text(value: object, *, field: str, maximum: int = 500) -> str:
    if not isinstance(value, str) or not value.strip() or len(value.strip()) > maximum:
        raise ValueError(f"{field} must be nonblank text")
    return value.strip()


def _nonnegative_int(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field} must be a nonnegative integer")
    return value


def _proportion(value: object, *, field: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not 0.0 <= float(value) <= 1.0
    ):
        raise ValueError(f"{field} must be a proportion")
    return float(value)


def _utc(value: str | datetime) -> str:
    parsed = (
        value
        if isinstance(value, datetime)
        else datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    )
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("observed_at must include a timezone")
    return parsed.astimezone(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def _canonical_utc(value: object, *, field: str) -> str:
    if not isinstance(value, str) or value != _utc(value):
        raise ValueError(f"{field} must be a canonical UTC instant")
    return value


__all__ = [
    "BUTTERFLYLENS_CLASSIFICATION_MATURITY_SCHEMA",
    "BUTTERFLYLENS_REVIEW_ASSIGNMENT_SCHEMA",
    "BUTTERFLYLENS_REVIEW_ROLES",
    "ButterflyLensReviewExport",
    "ButterflyLensReviewLayer",
    "build_butterflylens_review_layer",
    "export_butterflylens_review_evidence",
    "validate_butterflylens_review_export",
    "validate_butterflylens_review_layer",
]
