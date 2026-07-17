from __future__ import annotations

from datetime import UTC, datetime, timedelta
import hashlib
import json

import polars as pl
import pytest

from biominer.bioclip.reference_quality_diagnostics import (
    REFERENCE_OUTLIER_SCORE_VERSION,
    REFERENCE_QUALITY_DIAGNOSTICS_SCHEMA_VERSION,
    reference_quality_diagnostics_schema,
)
from biominer.common.semantic_hash import canonical_semantic_fingerprint
from biominer.evaluation.reference_escalation import (
    flag_species_for_reference_review,
)
from biominer.references.schemas import (
    REFERENCE_REVIEW_QUEUE_SCHEMA_VERSION,
    make_reference_media_id,
    make_reference_review_request_id,
    reference_review_decisions_frame,
    reference_review_queue_frame,
    reference_review_queue_schema,
)
from biominer.references.review import (
    REFERENCE_REVIEW_QUEUE_PROVENANCE_SCHEMA_VERSION,
    reference_review_queue_provenance_schema,
)
from biominer.references.targeted_review import (
    TARGETED_REFERENCE_REVIEW_QUEUE_FILE,
    TargetedReferenceReviewPolicy,
    build_targeted_reference_review_queue,
    reference_error_involvement_frame,
    validate_targeted_reference_review_queue,
    write_targeted_reference_review_queue,
)
from biominer.references.targeted_review_decisions import (
    review_statistically_flagged_support,
    targeted_reference_review_decisions_frame,
    write_targeted_reference_review_result,
)


NOW = datetime(2026, 7, 17, 1, 2, 3, tzinfo=UTC)
SHA_A = "sha256:" + "a" * 64
SHA_B = "sha256:" + "b" * 64
SHA_C = "sha256:" + "c" * 64


def _media_id(seed: str) -> tuple[str, str]:
    observation_id = "reference-observation:" + seed * 64
    return (
        make_reference_media_id("gbif", f"provider-{seed}", observation_id),
        observation_id,
    )


def _queue_row(seed: str, species: str) -> dict[str, object]:
    media_id, observation_id = _media_id(seed)
    row: dict[str, object] = {
        "schema_version": REFERENCE_REVIEW_QUEUE_SCHEMA_VERSION,
        "reference_media_id": media_id,
        "reference_observation_id": observation_id,
        "canonical_reference_media_id": media_id,
        "accepted_taxon_key": f"gbif:{seed}",
        "scientific_name": species,
        "durable_preview_uri": f"s3://reference/{seed}.jpg",
        "media_object_fingerprint": SHA_A,
        "duplicate_group_id": "reference-duplicate-group:" + seed * 32,
        "source": "gbif",
        "provider_media_id": f"provider-{seed}",
        "provider_verification_status": "accepted",
        "creator": "Observer",
        "rights_holder": "Observer",
        "licence": "CC-BY-4.0",
        "licence_uri": "https://creativecommons.org/licenses/by/4.0/",
        "licence_policy_status": "allowed",
        "attribution": "Observer / CC-BY-4.0",
        "life_stage": "adult",
        "visual_domain": "live_field",
        "view": "dorsal",
        "review_reason": "manual_identity_review",
        "review_priority": 100,
        "required_review_count": 1,
        "review_status": "pending",
        "created_at": NOW,
        "reference_bank_version": "reference-bank-v1",
    }
    semantics = _review_payload_hash(
        {
            "domain": "biominer.reference-review.queue-semantics.v1",
            "queue": {
                field: row[field]
                for field in reference_review_queue_schema()
                if field
                not in {
                    "created_at",
                    "durable_preview_uri",
                    "review_priority",
                    "review_reason",
                    "review_request_id",
                    "review_status",
                    "input_fingerprint",
                }
            },
        }
    )
    source_binding = _review_payload_hash(
        {
            "domain": "biominer.reference-review.source-set.v1",
            "source_leaf_fingerprints": [SHA_A],
        }
    )
    input_fingerprint = _review_payload_hash(
        {
            "domain": "biominer.reference-review.request-input.v1",
            "queue_semantics_fingerprint": semantics,
            "source_binding_fingerprint": source_binding,
        }
    )
    row["input_fingerprint"] = input_fingerprint
    row["review_request_id"] = make_reference_review_request_id(
        reference_media_id=media_id,
        media_object_fingerprint=SHA_A,
        reference_bank_version="reference-bank-v1",
        input_fingerprint=input_fingerprint,
    )
    return row


def _review_payload_hash(value: object) -> str:
    def jsonable(item: object) -> object:
        if isinstance(item, datetime):
            return item.isoformat()
        if isinstance(item, dict):
            return {str(key): jsonable(child) for key, child in sorted(item.items())}
        if isinstance(item, (list, tuple)):
            return [jsonable(child) for child in item]
        return item

    encoded = json.dumps(
        jsonable(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode()
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _queue_provenance(queue: pl.DataFrame) -> pl.DataFrame:
    excluded = {
        "created_at",
        "durable_preview_uri",
        "review_priority",
        "review_reason",
        "review_request_id",
        "review_status",
        "input_fingerprint",
    }
    rows = []
    for row in queue.iter_rows(named=True):
        semantics = _review_payload_hash(
            {
                "domain": "biominer.reference-review.queue-semantics.v1",
                "queue": {
                    field: row[field]
                    for field in reference_review_queue_schema()
                    if field not in excluded
                },
            }
        )
        source_binding = _review_payload_hash(
            {
                "domain": "biominer.reference-review.source-set.v1",
                "source_leaf_fingerprints": [SHA_A],
            }
        )
        rows.append(
            {
                "schema_version": (
                    REFERENCE_REVIEW_QUEUE_PROVENANCE_SCHEMA_VERSION
                ),
                "review_request_id": row["review_request_id"],
                "reference_media_id": row["reference_media_id"],
                "source_binding_fingerprint": source_binding,
                "source_leaf_fingerprints": [SHA_A],
                "queue_semantics_fingerprint": semantics,
                "queue_row_fingerprint": _review_payload_hash(
                    {
                        "domain": "biominer.reference-review.queue-row.v1",
                        "queue": {
                            field: row[field]
                            for field in reference_review_queue_schema()
                        },
                    }
                ),
                "input_fingerprint": row["input_fingerprint"],
            }
        )
    return pl.DataFrame(
        rows,
        schema=reference_review_queue_provenance_schema(),
        orient="row",
        strict=True,
    ).sort("reference_media_id", "review_request_id")


def _diagnostic(
    seed: str,
    species: str,
    *,
    outlier: float,
    competitor_similarity: float,
    influence: float,
    mismatch: bool,
) -> dict[str, object]:
    media_id, observation_id = _media_id(seed)
    row: dict[str, object] = {
        "schema_version": REFERENCE_QUALITY_DIAGNOSTICS_SCHEMA_VERSION,
        "reference_media_id": media_id,
        "reference_observation_id": observation_id,
        "accepted_taxon_key": f"gbif:{seed}",
        "species": species,
        "route": "adult_field",
        "visual_domain": "live_field",
        "identity_evidence_basis": "gbif_provider_asserted",
        "reference_admission_mode": "provisional_provider_asserted",
        "admission_policy_fingerprint": SHA_A,
        "similarity_to_class_centroid": 0.8,
        "leave_one_out_centroid_similarity": 0.7,
        "nearest_same_species_reference_media_id": media_id,
        "nearest_same_species_similarity": 0.7,
        "nearest_competing_reference_media_id": media_id,
        "nearest_competing_taxon_key": "gbif:competitor",
        "nearest_competing_species_similarity": competitor_similarity,
        "same_minus_competitor_margin": 0.7 - competitor_similarity,
        "prototype_influence": influence,
        "route_domain_mismatch": mismatch,
        "embedding_outlier_score": outlier,
        "outlier_score_version": REFERENCE_OUTLIER_SCORE_VERSION,
        "review_threshold": 0.35,
        "diagnostic_state": "reference_quality_review_candidate",
        "taxon_misidentification_conclusion": "not_assessed",
        "policy_fingerprint": SHA_B,
        "model_fingerprint": SHA_C,
        "reference_embedding_fingerprint": SHA_A,
        "support_manifest_fingerprint": SHA_B,
        "diagnostic_fingerprint": "",
    }
    payload = dict(row)
    payload.pop("diagnostic_fingerprint")
    row["diagnostic_fingerprint"] = canonical_semantic_fingerprint(payload)
    return row


def _flagged_escalation(region: str = "geo:qld") -> pl.DataFrame:
    return flag_species_for_reference_review(
        pl.DataFrame(
            [
                {
                    "target_species": "Papilio demoleus",
                    "competitor_species": "Papilio polytes",
                    "region": region,
                    "route": "adult_field",
                    "metric_status": "complete",
                    "reviewed_record_count": 40,
                    "precision_ci_lower": 0.4,
                    "recall": 0.9,
                    "false_positive_rate": 0.1,
                    "competitor_confusion_rate": 0.1,
                }
            ]
        ),
        pl.DataFrame(
            [
                {
                    "target_species": "Papilio demoleus",
                    "competitor_species": "Papilio polytes",
                    "region": region,
                    "route": "adult_field",
                    "prototype_dispersion_max": 0.1,
                    "high_influence_reference_count": 0,
                    "reference_outlier_count": 0,
                    "route_imbalance_ratio": 0.0,
                    "target_reference_count": 10,
                    "reference_identity_conclusion": "not_assessed",
                }
            ]
        ),
    )


def _escalations() -> pl.DataFrame:
    flagged = _flagged_escalation()
    unflagged = flag_species_for_reference_review(
        pl.DataFrame(
            [
                {
                    "target_species": "Papilio machaon",
                    "competitor_species": "Papilio zelicaon",
                    "region": "geo:qld",
                    "route": "adult_field",
                    "metric_status": "complete",
                    "reviewed_record_count": 40,
                    "precision_ci_lower": 0.95,
                    "recall": 0.95,
                    "false_positive_rate": 0.05,
                    "competitor_confusion_rate": 0.05,
                }
            ]
        ),
        pl.DataFrame(
            [
                {
                    "target_species": "Papilio machaon",
                    "competitor_species": "Papilio zelicaon",
                    "region": "geo:qld",
                    "route": "adult_field",
                    "prototype_dispersion_max": 0.1,
                    "high_influence_reference_count": 0,
                    "reference_outlier_count": 0,
                    "route_imbalance_ratio": 0.0,
                    "target_reference_count": 10,
                    "reference_identity_conclusion": "not_assessed",
                }
            ]
        ),
    )
    return pl.concat([flagged, unflagged])


def _inputs() -> tuple[pl.DataFrame, ...]:
    species = "Papilio demoleus"
    other = "Papilio machaon"
    queue = reference_review_queue_frame(
        [_queue_row("a", species), _queue_row("b", species), _queue_row("c", other)]
    )
    diagnostics = pl.DataFrame(
        [
            _diagnostic(
                "a",
                species,
                outlier=1.8,
                competitor_similarity=0.9,
                influence=1.2,
                mismatch=True,
            ),
            _diagnostic(
                "b",
                species,
                outlier=0.1,
                competitor_similarity=-0.5,
                influence=0.0,
                mismatch=False,
            ),
            _diagnostic(
                "c",
                other,
                outlier=2.0,
                competitor_similarity=1.0,
                influence=2.0,
                mismatch=True,
            ),
        ],
        schema=reference_quality_diagnostics_schema(),
        orient="row",
        strict=True,
    ).sort("accepted_taxon_key", "route", "reference_media_id")
    media_a, _ = _media_id("a")
    media_b, _ = _media_id("b")
    media_c, _ = _media_id("c")
    support = pl.DataFrame(
        [
            {
                "reference_media_id": media_a,
                "scientific_name": species,
                "route": "adult_field",
                "support_eligible": True,
                "provider_dataset_key": "dataset:dominant",
            },
            {
                "reference_media_id": media_b,
                "scientific_name": species,
                "route": "adult_field",
                "support_eligible": True,
                "provider_dataset_key": "dataset:other",
            },
            {
                "reference_media_id": media_c,
                "scientific_name": other,
                "route": "adult_field",
                "support_eligible": True,
                "provider_dataset_key": "dataset:other",
            },
        ]
    )
    qa = pl.DataFrame(
        [
            {"reference_media_id": media_a, "subject_area_ratio": 0.05},
            {"reference_media_id": media_b, "subject_area_ratio": 0.9},
            {"reference_media_id": media_c, "subject_area_ratio": 0.01},
        ]
    )
    identities = pl.DataFrame(
        [
            {
                "reference_media_id": media_a,
                "resolution_status": "review_required",
                "support_disposition": "unresolved_duplicate",
            },
            {
                "reference_media_id": media_b,
                "resolution_status": "resolved",
                "support_disposition": "eligible",
            },
            {
                "reference_media_id": media_c,
                "resolution_status": "conflict",
                "support_disposition": "duplicate_conflict",
            },
        ]
    )
    errors = reference_error_involvement_frame(
        [
            {"reference_media_id": media_a, "error_involvement_count": 4},
            {"reference_media_id": media_b, "error_involvement_count": 0},
        ]
    )
    return queue, _escalations(), diagnostics, support, qa, identities, errors


def test_queue_contains_only_flagged_species_and_all_priority_evidence(
    tmp_path,
) -> None:
    result = build_targeted_reference_review_queue(*_inputs())

    assert result.height == 2
    assert result["scientific_name"].unique().to_list() == ["Papilio demoleus"]
    assert result["target_review_rank"].to_list() == [1, 2]
    assert result.row(0, named=True)["duplicate_ambiguity"] is True
    assert {
        component["component"]
        for component in result.row(0, named=True)["priority_components"]
    } == {
        "embedding_outlier_score",
        "nearest_competing_species_similarity",
        "prototype_influence",
        "route_domain_mismatch",
        "provider_dataset_concentration",
        "repeated_error_involvement",
        "low_subject_area_ratio",
        "duplicate_ambiguity",
    }
    assert set(result["taxon_misidentification_conclusion"].to_list()) == {
        "not_assessed"
    }
    path = write_targeted_reference_review_queue(result, tmp_path)
    assert path.name == TARGETED_REFERENCE_REVIEW_QUEUE_FILE
    assert pl.read_parquet(path).equals(result)


def test_multiple_flag_contexts_collapse_to_one_reference_row() -> None:
    inputs = list(_inputs())
    escalations = inputs[1]
    second = _flagged_escalation("geo:nsw")
    inputs[1] = pl.concat([escalations, second])

    result = build_targeted_reference_review_queue(*inputs)

    assert result.height == 2
    assert len(result.row(0, named=True)["flagged_contexts"]) == 2


def test_targeting_is_deterministic_and_policy_fingerprinted() -> None:
    first = build_targeted_reference_review_queue(*_inputs())
    second = build_targeted_reference_review_queue(*_inputs())

    assert first.equals(second)
    assert first["targeting_fingerprint"].to_list() == second[
        "targeting_fingerprint"
    ].to_list()
    assert TargetedReferenceReviewPolicy().fingerprint.startswith("sha256:")
    with pytest.raises(ValueError, match="outlier_weight"):
        TargetedReferenceReviewPolicy(outlier_weight=0)


def test_missing_targeted_reference_evidence_fails_closed() -> None:
    inputs = list(_inputs())
    inputs[4] = inputs[4].head(1)

    with pytest.raises(ValueError, match="reference QA missing targeted reference"):
        build_targeted_reference_review_queue(*inputs)


def test_statistical_identity_claim_is_rejected() -> None:
    inputs = list(_inputs())
    inputs[1] = inputs[1].with_columns(
        pl.when(pl.col("flagged_for_reference_review"))
        .then(pl.lit("misidentified"))
        .otherwise(pl.col("statistical_identity_conclusion"))
        .alias("statistical_identity_conclusion")
    )

    with pytest.raises(ValueError, match="must not claim identity error"):
        build_targeted_reference_review_queue(*inputs)


def test_targeted_queue_validator_rejects_tampering() -> None:
    result = build_targeted_reference_review_queue(*_inputs()).with_columns(
        pl.lit(0.0).alias("target_review_priority_score")
    )

    with pytest.raises(ValueError, match="fingerprint mismatch"):
        validate_targeted_reference_review_queue(result)


def _targeted_decision(
    targeted_row: dict[str, object],
    *,
    action: str,
    notes: str | None = None,
    exclusion_reason: str | None = None,
    alternative_species: str | None = None,
) -> pl.DataFrame:
    return targeted_reference_review_decisions_frame(
        [
            {
                "review_request_id": targeted_row["review_request_id"],
                "reference_media_id": targeted_row["reference_media_id"],
                "targeting_fingerprint": targeted_row["targeting_fingerprint"],
                "review_action": action,
                "review_round": 1,
                "verified_by": "reviewer-1",
                "reviewed_at": NOW + timedelta(days=1),
                "life_stage": "larva",
                "visual_domain": "pinned_specimen",
                "view": "ventral",
                "review_confidence": "high",
                "review_notes": notes,
                "exclusion_reason": exclusion_reason,
                "alternative_species": alternative_species,
                "conflicts_with_decision_id": None,
            }
        ]
    )


def test_targeted_verification_reuses_review_resolver_and_corrections(
    tmp_path,
) -> None:
    inputs = _inputs()
    targeted = build_targeted_reference_review_queue(*inputs)
    source_queue = inputs[0]
    decision = _targeted_decision(
        targeted.row(0, named=True),
        action="verify",
        notes="Identity verified; metadata fields corrected.",
    )

    result = review_statistically_flagged_support(
        targeted,
        _queue_provenance(source_queue),
        decision,
        resolved_at=NOW + timedelta(days=2),
    )

    assert result.imported_decision_count == 1
    assert result.idempotent_replay_count == 0
    assert result.workflow.verified.height == 1
    verified = result.workflow.verified.row(0, named=True)
    assert verified["resolved_life_stage"] == "larva"
    assert verified["resolved_visual_domain"] == "pinned_specimen"
    assert verified["resolved_view"] == "ventral"
    assert result.decision_bindings.height == 1
    artifacts = write_targeted_reference_review_result(result, tmp_path)
    assert all(path.exists() for path in artifacts.values())


def test_exclusion_preserves_alternative_species_in_strict_binding() -> None:
    inputs = _inputs()
    targeted = build_targeted_reference_review_queue(*inputs)
    decision = _targeted_decision(
        targeted.row(0, named=True),
        action="exclude",
        notes="Wing pattern supports a different taxon.",
        exclusion_reason="alternative_species",
        alternative_species="Papilio polytes",
    )

    result = review_statistically_flagged_support(
        targeted,
        _queue_provenance(inputs[0]),
        decision,
    )

    assert result.workflow.excluded.height == 1
    assert result.decision_bindings["alternative_species"].item() == (
        "Papilio polytes"
    )
    assert "Alternative species noted: Papilio polytes." in result.workflow.decisions[
        "review_notes"
    ].item()


@pytest.mark.parametrize(
    ("action", "notes"),
    [
        ("uncertain", "Visible evidence is insufficient."),
        ("request_second_review", None),
    ],
)
def test_non_decisive_actions_request_a_second_review(
    action: str,
    notes: str | None,
) -> None:
    inputs = _inputs()
    targeted = build_targeted_reference_review_queue(*inputs)
    decision = _targeted_decision(
        targeted.row(0, named=True),
        action=action,
        notes=notes,
    )

    result = review_statistically_flagged_support(
        targeted,
        _queue_provenance(inputs[0]),
        decision,
    )

    outcome = result.workflow.outcomes.filter(
        pl.col("review_request_id") == decision["review_request_id"].item()
    ).row(0, named=True)
    assert outcome["review_status"] == "second_review_required"
    assert outcome["second_review_required"] is True


def test_targeted_review_is_append_only_and_idempotent() -> None:
    inputs = _inputs()
    targeted = build_targeted_reference_review_queue(*inputs)
    provenance = _queue_provenance(inputs[0])
    decision = _targeted_decision(
        targeted.row(0, named=True),
        action="verify",
        notes="Verified.",
    )
    first = review_statistically_flagged_support(
        targeted,
        provenance,
        decision,
    )

    replay = review_statistically_flagged_support(
        targeted,
        provenance,
        decision,
        existing_decisions=first.workflow.decisions,
    )

    assert replay.imported_decision_count == 0
    assert replay.idempotent_replay_count == 1
    assert replay.workflow.decisions.equals(first.workflow.decisions)

    replacement = _targeted_decision(
        targeted.row(0, named=True),
        action="exclude",
        notes="Changed decision.",
        exclusion_reason="identity_not_supported",
    )
    with pytest.raises(ValueError, match="one decision per request round"):
        review_statistically_flagged_support(
            targeted,
            provenance,
            replacement,
            existing_decisions=first.workflow.decisions,
        )


def test_stale_targeting_fingerprint_is_rejected() -> None:
    inputs = _inputs()
    targeted = build_targeted_reference_review_queue(*inputs)
    row = targeted.row(0, named=True)
    row["targeting_fingerprint"] = SHA_A
    decision = _targeted_decision(row, action="verify", notes="Verified.")

    with pytest.raises(ValueError, match="stale targeting fingerprint"):
        review_statistically_flagged_support(
            targeted,
            _queue_provenance(inputs[0]),
            decision,
        )
