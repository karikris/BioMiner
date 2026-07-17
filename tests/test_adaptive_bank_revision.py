from __future__ import annotations

from dataclasses import replace

import polars as pl
import pytest

from biominer.references.adaptive_bank_revision import (
    reference_downstream_dependencies_frame,
    revise_adaptive_support_bank,
    validate_adaptive_support_bank_revision,
    write_adaptive_support_bank_revision,
)
from biominer.references.readiness import (
    REFERENCE_SUPPORT_MANIFEST_SCHEMA_VERSION,
    reference_support_manifest_frame,
)
from biominer.references.targeted_review import (
    build_targeted_reference_review_queue,
)
from biominer.references.targeted_review_decisions import (
    review_statistically_flagged_support,
    targeted_reference_review_decisions_frame,
)
from test_targeted_reference_review import (
    NOW,
    SHA_A,
    SHA_B,
    SHA_C,
    _inputs,
    _queue_provenance,
    _targeted_decision,
)


def _support_manifest(queue: pl.DataFrame) -> pl.DataFrame:
    rows: list[dict[str, object]] = []
    for index, queue_row in enumerate(queue.iter_rows(named=True), start=1):
        seed = chr(96 + index)
        rows.append(
            {
                "schema_version": REFERENCE_SUPPORT_MANIFEST_SCHEMA_VERSION,
                "reference_bank_version": "reference-bank-v1",
                "registry_version": "registry-v1",
                "reference_media_id": queue_row["reference_media_id"],
                "canonical_reference_media_id": queue_row["reference_media_id"],
                "reference_observation_id": queue_row[
                    "reference_observation_id"
                ],
                "reference_admission_mode": "adaptive_gbif_fast_start",
                "reference_admission_policy_version": "adaptive-v1",
                "reference_admission_policy_fingerprint": SHA_A,
                "identity_evidence_basis": "gbif_provider_asserted",
                "provider_asserted_identity": True,
                "provider_asserted_taxon_key": queue_row["accepted_taxon_key"],
                "provider_asserted_scientific_name": queue_row[
                    "scientific_name"
                ],
                "provider_dataset_key": f"dataset:{seed}",
                "provider_quality_status": "accepted",
                "human_review_status": "pending",
                "human_verified_identity": False,
                "provisional_support": True,
                "statistical_audit_required": True,
                "admission_status": "admitted",
                "admission_reasons": [
                    "gbif_provider_asserted_provisional_support"
                ],
                "reference_quality_flags": [],
                "route_evidence_basis": "yoloe",
                "geographic_prototype_eligible": True,
                "review_request_id": queue_row["review_request_id"],
                "review_decision_ids": [],
                "reviewer_ids": [],
                "source": "gbif",
                "source_observation_id": f"provider-observation-{seed}",
                "provider_media_id": queue_row["provider_media_id"],
                "source_record_url": f"https://example.test/record/{seed}",
                "source_snapshot_version": "gbif-2026-07-17",
                "source_dataset_key": f"dataset:{seed}",
                "accepted_taxon_key": queue_row["accepted_taxon_key"],
                "scientific_name": queue_row["scientific_name"],
                "target_candidate": True,
                "geo_cluster_id": "geo:qld",
                "observer_id": f"observer-{seed}",
                "observed_at": NOW,
                "latitude": -27.5,
                "longitude": 153.0,
                "source_object_uri": queue_row["durable_preview_uri"],
                "image_sha256": SHA_A,
                "perceptual_hash": f"dhash128-v1:{seed * 32}",
                "object_fingerprint": SHA_B,
                "duplicate_group_id": queue_row["duplicate_group_id"],
                "duplicate_type": "unique",
                "creator": queue_row["creator"],
                "rights_holder": queue_row["rights_holder"],
                "licence": queue_row["licence"],
                "licence_uri": queue_row["licence_uri"],
                "licence_policy_status": "allowed",
                "attribution": queue_row["attribution"],
                "review_status": "pending",
                "verification_status": "unreviewed",
                "target_identity_verified": False,
                "life_stage": "adult",
                "visual_domain": "live_field",
                "view": "dorsal",
                "route": "adult_field",
                "support_split": "support_train",
                "support_eligible": True,
                "exclusion_reasons": [],
                "split_assignment_fingerprint": SHA_C,
                "support_row_fingerprint": "",
                "reference_bank_fingerprint": SHA_C,
            }
        )
    return reference_support_manifest_frame(
        rows,
        reference_bank_version="reference-bank-v1",
        reference_bank_fingerprint=SHA_C,
    )


def _dependencies(queue: pl.DataFrame) -> pl.DataFrame:
    media_ids = queue["reference_media_id"].to_list()
    return reference_downstream_dependencies_frame(
        [
            {
                "artifact_id": "embedding:a",
                "artifact_type": "reference_embedding",
                "artifact_fingerprint": SHA_A,
                "reference_bank_fingerprint": SHA_C,
                "reference_media_ids": [media_ids[0]],
            },
            {
                "artifact_id": "embedding:unflagged",
                "artifact_type": "reference_embedding",
                "artifact_fingerprint": SHA_B,
                "reference_bank_fingerprint": SHA_C,
                "reference_media_ids": [media_ids[2]],
            },
            {
                "artifact_id": "prototype:target",
                "artifact_type": "reference_prototype",
                "artifact_fingerprint": SHA_C,
                "reference_bank_fingerprint": SHA_C,
                "reference_media_ids": sorted(media_ids[:2]),
            },
        ]
    )


def _review_with_verify_and_exclude():  # noqa: ANN202
    inputs = _inputs()
    targeted = build_targeted_reference_review_queue(*inputs)
    verify = _targeted_decision(
        targeted.row(0, named=True),
        action="verify",
        notes="Human verification supports the target.",
    )
    exclude = _targeted_decision(
        targeted.row(1, named=True),
        action="exclude",
        notes="Human verification rejects the target.",
        exclusion_reason="alternative_species",
        alternative_species="Papilio polytes",
    )
    decisions = targeted_reference_review_decisions_frame(
        [*verify.to_dicts(), *exclude.to_dicts()]
    )
    review = review_statistically_flagged_support(
        targeted,
        _queue_provenance(inputs[0]),
        decisions,
    )
    return inputs, review


def test_revision_promotes_excludes_preserves_and_invalidates_selectively(
    tmp_path,
) -> None:
    inputs, review = _review_with_verify_and_exclude()
    current = _support_manifest(inputs[0])

    revision = revise_adaptive_support_bank(
        current,
        review,
        _dependencies(inputs[0]),
    )

    assert revision.old_reference_bank_version == "reference-bank-v1"
    assert revision.new_reference_bank_version == "reference-bank-v2"
    revised = {
        str(row["reference_media_id"]): row
        for row in revision.revised_support_manifest.iter_rows(named=True)
    }
    changes = {
        str(row["reference_media_id"]): row
        for row in revision.change_manifest.iter_rows(named=True)
    }
    verified_id = review.workflow.verified["reference_media_id"].item()
    excluded_id = review.workflow.excluded["reference_media_id"].item()
    unflagged_id = next(
        media_id
        for media_id in revised
        if media_id not in {verified_id, excluded_id}
    )
    assert revised[verified_id]["identity_evidence_basis"] == "human_verified"
    assert revised[verified_id]["provisional_support"] is False
    assert revised[excluded_id]["support_eligible"] is False
    assert revised[excluded_id]["identity_evidence_basis"] == "none"
    assert revised[unflagged_id]["provisional_support"] is True
    assert revised[unflagged_id]["support_eligible"] is True
    assert changes[verified_id]["change_type"] == "promoted_verified"
    assert changes[excluded_id]["change_type"] == "excluded_after_review"
    assert changes[unflagged_id]["change_type"] == "unchanged_provisional"
    assert changes[unflagged_id]["changed_fields"] == []

    invalidations = {
        str(row["artifact_id"]): row
        for row in revision.invalidation_manifest.iter_rows(named=True)
    }
    assert invalidations["embedding:a"]["invalidated"] is True
    assert invalidations["prototype:target"]["invalidated"] is True
    assert invalidations["embedding:unflagged"]["invalidated"] is False
    assert invalidations["embedding:unflagged"][
        "affected_reference_media_ids"
    ] == []
    artifacts = write_adaptive_support_bank_revision(revision, tmp_path)
    assert all(path.exists() for path in artifacts.values())


def test_flagged_but_unresolved_references_are_quarantined() -> None:
    inputs = _inputs()
    targeted = build_targeted_reference_review_queue(*inputs)
    review = review_statistically_flagged_support(
        targeted,
        _queue_provenance(inputs[0]),
        targeted_reference_review_decisions_frame([]),
    )

    revision = revise_adaptive_support_bank(
        _support_manifest(inputs[0]),
        review,
        _dependencies(inputs[0]),
    )

    pending = revision.change_manifest.filter(
        pl.col("change_type") == "flagged_review_pending"
    )
    assert pending.height == 2
    pending_ids = set(pending["reference_media_id"])
    assert revision.revised_support_manifest.filter(
        pl.col("reference_media_id").is_in(pending_ids)
        & pl.col("support_eligible")
    ).is_empty()
    unchanged = revision.change_manifest.filter(
        pl.col("change_type") == "unchanged_provisional"
    )
    assert unchanged.height == 1


def test_bank_version_must_increment_exactly_once() -> None:
    inputs, review = _review_with_verify_and_exclude()

    with pytest.raises(ValueError, match="must increment to reference-bank-v2"):
        revise_adaptive_support_bank(
            _support_manifest(inputs[0]),
            review,
            _dependencies(inputs[0]),
            new_reference_bank_version="reference-bank-v3",
        )


def test_revision_validator_rejects_change_manifest_tampering() -> None:
    inputs, review = _review_with_verify_and_exclude()
    revision = revise_adaptive_support_bank(
        _support_manifest(inputs[0]),
        review,
        _dependencies(inputs[0]),
    )
    tampered = replace(
        revision,
        change_manifest=revision.change_manifest.with_columns(
            pl.lit("unchanged").alias("change_type")
        ),
    )

    with pytest.raises(ValueError, match="change fingerprint mismatch"):
        validate_adaptive_support_bank_revision(tampered)
