from __future__ import annotations

import hashlib
import json
from pathlib import Path

import polars as pl
import pytest

import biominer.bioclip.prototype_support as support_module
from biominer.bioclip.prototype_mode import BuildWeekPrototypeConfig
from biominer.bioclip.prototype_support import (
    PROVIDER_SUPPORT_GOAL,
    PROVIDER_SUPPORT_GOAL_VERIFICATION_SCHEMA_VERSION,
    PROTOTYPE_SCORE_SEMANTICS,
    validate_metadata_qualified_prototype_support,
)
from biominer.common.semantic_hash import canonical_semantic_fingerprint
from biominer.references.prototype_freeze import (
    PROTOTYPE_READINESS_SCHEMA_VERSION,
    PROTOTYPE_SUPPORT_SCHEMA_VERSION,
    prototype_support_schema,
)


TARGET_KEY = "gbif:1938069"
TARGET_NAME = "Papilio demoleus"


def test_metadata_qualified_prototype_support_allows_zero_human_labels(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _fixture(tmp_path)
    monkeypatch.setattr(
        support_module,
        "validate_prototype_reference_embeddings",
        lambda _frame: None,
    )

    permit = validate_metadata_qualified_prototype_support(config)

    assert permit.readiness.human_verified_count == 0
    assert permit.readiness.human_verification_complete is False
    assert permit.readiness.classification_authorised is True
    assert permit.goal_verification.verified_record_count == 1
    assert permit.goal_verification.records_meeting_goal_count == 1
    assert (
        permit.goal_verification.independent_human_taxonomic_verification_claimed
        is False
    )
    assert permit.calibration_fingerprint is None
    assert permit.classifier_fingerprint == config.classifier_fingerprint
    assert (
        permit.support_qualification
        == "user_goal_verified_metadata_qualified_prototype_only"
    )
    assert permit.readiness.score_semantics == PROTOTYPE_SCORE_SEMANTICS


def test_metadata_qualified_prototype_support_keeps_licensing_mandatory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _fixture(tmp_path, licence_policy_status="blocked")
    monkeypatch.setattr(
        support_module,
        "validate_prototype_reference_embeddings",
        lambda _frame: None,
    )

    with pytest.raises(ValueError, match="ineligible licensing"):
        validate_metadata_qualified_prototype_support(config)


def test_metadata_qualified_prototype_support_rejects_incomplete_goal_verification(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _fixture(tmp_path, verified_record_count=0)
    monkeypatch.setattr(
        support_module,
        "validate_prototype_reference_embeddings",
        lambda _frame: None,
    )

    with pytest.raises(ValueError, match="verified_record_count"):
        validate_metadata_qualified_prototype_support(config)


def test_metadata_qualified_prototype_support_rejects_taxonomic_overclaim(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _fixture(
        tmp_path,
        independent_human_taxonomic_verification_claimed=True,
    )
    monkeypatch.setattr(
        support_module,
        "validate_prototype_reference_embeddings",
        lambda _frame: None,
    )

    with pytest.raises(
        ValueError,
        match="independent_human_taxonomic_verification_claimed",
    ):
        validate_metadata_qualified_prototype_support(config)


def _fixture(
    tmp_path: Path,
    *,
    licence_policy_status: str = "research_only",
    verified_record_count: int = 1,
    independent_human_taxonomic_verification_claimed: bool = False,
) -> BuildWeekPrototypeConfig:
    support_path = tmp_path / "support.parquet"
    support = _support_frame(licence_policy_status=licence_policy_status)
    support.write_parquet(support_path)
    support_fingerprint = canonical_semantic_fingerprint(support.to_dicts())

    readiness_path = tmp_path / "readiness.json"
    readiness_path.write_text(
        json.dumps(
            {
                "schema_version": PROTOTYPE_READINESS_SCHEMA_VERSION,
                "prototype_readiness_status": "prototype_ready_with_shortfalls",
                "classification_authorised": True,
                "bank_status": "prototype_only",
                "human_verification_complete": False,
                "target_accepted_taxon_key": TARGET_KEY,
                "target_scientific_name": TARGET_NAME,
                "support_manifest_fingerprint": support_fingerprint,
                "counts": {
                    "prototype_support_count": 1,
                    "human_verified_count": 0,
                },
            }
        ),
        encoding="utf-8",
    )
    goal_verification_path = tmp_path / "goal-verification.json"
    goal_verification_path.write_text(
        json.dumps(
            {
                "schema_version": (
                    PROVIDER_SUPPORT_GOAL_VERIFICATION_SCHEMA_VERSION
                ),
                "status": "verified_complete",
                "assertion_source": "direct_user_confirmation",
                "asserted_by": "fixture-reviewer",
                "verification_completed_on": "2026-07-16",
                "recorded_at": "2026-07-16T11:57:54Z",
                "reference_bank_version": "prototype-bank-v1",
                "support_manifest_sha256": _sha(support_path),
                "support_manifest_fingerprint": support_fingerprint,
                "goal": PROVIDER_SUPPORT_GOAL,
                "provider_supported_record_count": 1,
                "verified_record_count": verified_record_count,
                "records_meeting_goal_count": 1,
                "all_provider_supported_records_verified": True,
                "all_verified_records_meet_goal": True,
                "semantics": {
                    "verification_is_user_goal_suitability_confirmation": True,
                    "provider_provenance_is_preserved": True,
                    "independent_human_taxonomic_verification_claimed": (
                        independent_human_taxonomic_verification_claimed
                    ),
                    "classification_accuracy_authorized": False,
                    "scientific_release_authorized": False,
                    "production_default_change_authorized": False,
                },
            }
        ),
        encoding="utf-8",
    )
    embeddings_path = tmp_path / "embeddings.parquet"
    pl.DataFrame(
        {
            "reference_media_id": ["media-1"],
            "support_row_fingerprint": ["sha256:" + "5" * 64],
            "support_manifest_fingerprint": [support_fingerprint],
            "model_revision": ["model-revision"],
            "preprocessing_version": ["preprocess-v1"],
        }
    ).write_parquet(embeddings_path)
    candidates_path = tmp_path / "candidate-scores.parquet"
    pl.DataFrame(
        {
            "flickr_photo_id": ["photo-1", "photo-1"],
            "class_kind": ["species", "known_negative"],
            "class_id": [TARGET_KEY, "artwork"],
            "accepted_taxon_key": [TARGET_KEY, None],
            "target_candidate": [True, False],
            "score_semantics": [
                PROTOTYPE_SCORE_SEMANTICS,
                PROTOTYPE_SCORE_SEMANTICS,
            ],
            "experimental_screening_evidence": [True, True],
        }
    ).write_parquet(candidates_path)
    policy_path = tmp_path / "policy.json"
    classifier_fingerprint = "sha256:" + "a" * 64
    policy_path.write_text(
        json.dumps(
            {
                "deployment_status": "prototype",
                "policy_status": "prototype_uncalibrated",
                "score_semantics": PROTOTYPE_SCORE_SEMANTICS,
                "experimental_screening_evidence_only": True,
                "target": {
                    "accepted_taxon_key": TARGET_KEY,
                    "scientific_name": TARGET_NAME,
                },
                "frozen_identity": {
                    "classifier_fingerprint": classifier_fingerprint,
                    "model_revision": "model-revision",
                    "preprocessing_version": "preprocess-v1",
                    "visual_input_version": "visual-input-v1",
                    "support_manifest_fingerprint": support_fingerprint,
                },
                "calibration": {
                    "calibrator_fingerprint": None,
                    "probabilities_emitted": False,
                },
                "selected_policy": {
                    "higher_rank_pruning_permitted": False,
                    "spatial_crop_permitted": False,
                    "target_always_scored": True,
                    "visual_input": "raw_full_image",
                },
            }
        ),
        encoding="utf-8",
    )
    return BuildWeekPrototypeConfig(
        target_accepted_taxon_key=TARGET_KEY,
        target_scientific_name=TARGET_NAME,
        reference_bank_readiness=readiness_path,
        reference_bank_readiness_sha256=_sha(readiness_path),
        support_manifest=support_path,
        support_manifest_sha256=_sha(support_path),
        provider_support_goal_verification=goal_verification_path,
        provider_support_goal_verification_sha256=_sha(goal_verification_path),
        reference_embeddings=embeddings_path,
        reference_embeddings_sha256=_sha(embeddings_path),
        candidate_score_evidence=candidates_path,
        candidate_score_evidence_sha256=_sha(candidates_path),
        prototype_policy=policy_path,
        prototype_policy_sha256=_sha(policy_path),
        model_revision="model-revision",
        preprocessing_version="preprocess-v1",
        visual_input_version="visual-input-v1",
        classifier_fingerprint=classifier_fingerprint,
        margin_policy_version="margin-policy-v1",
        limitations=("No independent human taxonomic verification.",),
    )


def _support_frame(*, licence_policy_status: str) -> pl.DataFrame:
    row: dict[str, object] = {
        field: None for field in prototype_support_schema()
    }
    row.update(
        {
            "schema_version": PROTOTYPE_SUPPORT_SCHEMA_VERSION,
            "reference_bank_version": "prototype-bank-v1",
            "reference_media_id": "media-1",
            "reference_observation_id": "observation-1",
            "candidate_scope_type": "accepted_taxon",
            "accepted_taxon_key": TARGET_KEY,
            "scientific_name": TARGET_NAME,
            "source": "GBIF",
            "source_snapshot_version": "snapshot-v1",
            "provider_media_id": "provider-media-1",
            "trust_level": "R4",
            "verification_status": "provider_supported",
            "human_verified": False,
            "geographic_layer": "A",
            "geo_cluster_id": "cluster-1",
            "route": "adult_field",
            "life_stage": "adult",
            "visual_domain": "live_field",
            "reference_group": f"target:{TARGET_KEY}",
            "licence": "CC-BY-NC-4.0",
            "licence_uri": "https://creativecommons.org/licenses/by-nc/4.0/",
            "licence_policy_status": licence_policy_status,
            "attribution": "Photographer / CC-BY-NC-4.0",
            "attribution_complete": True,
            "source_object_uri": "local://object",
            "source_image_sha256": "sha256:" + "1" * 64,
            "source_object_fingerprint": "sha256:" + "2" * 64,
            "duplicate_group_id": "duplicate-1",
            "exact_hash_group_id": "exact-1",
            "observation_group_id": "observation-group-1",
            "owner_group_id": "owner-1",
            "photographer_group_id": "photographer-1",
            "qa_disposition": "needs_review",
            "image_quality_check": "pass",
            "subject_presence_check": "review",
            "subject_size_check": "review",
            "detector_evidence_status": "not_run",
            "dataset_split": "support_train",
            "leakage_component_id": "leakage-1",
            "leakage_component_size": 1,
            "split_fingerprint": "sha256:" + "4" * 64,
            "prototype_only": True,
            "support_row_fingerprint": "sha256:" + "5" * 64,
        }
    )
    return pl.DataFrame(
        [row],
        schema=prototype_support_schema(),
        orient="row",
        strict=True,
    )


def _sha(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
