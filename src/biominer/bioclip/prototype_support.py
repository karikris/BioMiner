"""Metadata-qualified support permit for the explicit Build Week prototype."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import polars as pl

from biominer.benchmarks.prototype_support_embeddings import (
    validate_prototype_reference_embeddings,
)
from biominer.bioclip.prototype_mode import BuildWeekPrototypeConfig
from biominer.common.semantic_hash import canonical_semantic_fingerprint
from biominer.references.prototype_freeze import (
    PROTOTYPE_READINESS_SCHEMA_VERSION,
    PROTOTYPE_SUPPORT_SCHEMA_VERSION,
    prototype_support_schema,
)


PROTOTYPE_SCORE_SEMANTICS = (
    "experimental_screening_evidence_uncalibrated_not_probability"
)
_ALLOWED_READINESS = {"prototype_ready", "prototype_ready_with_shortfalls"}
_ALLOWED_VERIFICATION = {
    "human_verified",
    "provider_high_trust",
    "provider_supported",
}
_ROUTE_DIMENSIONS = {
    "adult_field": {("adult", "live_field")},
    "larval": {("larva", "live_field")},
    "pinned_specimen": {
        ("adult", "pinned_specimen"),
        ("unknown", "pinned_specimen"),
    },
}


@dataclass(frozen=True, slots=True)
class PrototypeReadinessPermit:
    status: str
    readiness_sha256: str
    deployment_status: str
    bank_status: str
    classification_authorised: bool
    human_verification_complete: bool
    target_accepted_taxon_key: str
    target_scientific_name: str
    support_manifest_fingerprint: str
    prototype_support_count: int
    human_verified_count: int
    score_semantics: str


@dataclass(frozen=True, slots=True)
class MetadataQualifiedPrototypePermit:
    readiness: PrototypeReadinessPermit
    candidate_set_fingerprints: tuple[str, ...]
    reference_embedding_fingerprint: str
    model_fingerprint: str
    classifier_fingerprint: str
    calibration_fingerprint: None
    support_qualification: str


def validate_metadata_qualified_prototype_support(
    config: BuildWeekPrototypeConfig,
) -> MetadataQualifiedPrototypePermit:
    """Validate the prototype support chain without claiming scientific readiness."""

    config.verify_artifacts()
    support = pl.read_parquet(config.support_manifest)
    _validate_support(support, config=config)
    readiness_payload = _read_json(config.reference_bank_readiness)
    readiness = _validate_readiness(
        readiness_payload,
        config=config,
        support=support,
    )
    embeddings = pl.read_parquet(config.reference_embeddings)
    validate_prototype_reference_embeddings(embeddings)
    _validate_embedding_bindings(embeddings, config=config, support=support)
    policy = _read_json(config.prototype_policy)
    _validate_policy(policy, config=config, support=support)
    candidates = pl.read_parquet(config.candidate_score_evidence)
    _validate_candidate_scores(candidates, config=config)
    model_fingerprint = canonical_semantic_fingerprint(
        {
            "model_revision": config.model_revision,
            "preprocessing_version": config.preprocessing_version,
            "visual_input_version": config.visual_input_version,
            "reference_embeddings_sha256": config.reference_embeddings_sha256,
        }
    )
    return MetadataQualifiedPrototypePermit(
        readiness=readiness,
        candidate_set_fingerprints=(config.candidate_score_evidence_sha256,),
        reference_embedding_fingerprint=config.reference_embeddings_sha256,
        model_fingerprint=model_fingerprint,
        classifier_fingerprint=config.classifier_fingerprint,
        calibration_fingerprint=None,
        support_qualification="metadata_qualified_prototype_only",
    )


def _validate_support(
    frame: pl.DataFrame,
    *,
    config: BuildWeekPrototypeConfig,
) -> None:
    if dict(frame.schema) != prototype_support_schema():
        raise ValueError("prototype support manifest physical schema mismatch")
    if frame.is_empty():
        raise ValueError("prototype support manifest must not be empty")
    if set(frame["schema_version"].to_list()) != {PROTOTYPE_SUPPORT_SCHEMA_VERSION}:
        raise ValueError("unsupported prototype support schema")
    if not frame.equals(frame.sort("reference_media_id")):
        raise ValueError("prototype support manifest is not deterministically sorted")
    for field in (
        "reference_media_id",
        "reference_observation_id",
        "source_image_sha256",
        "duplicate_group_id",
        "support_row_fingerprint",
    ):
        if frame[field].n_unique() != frame.height:
            raise ValueError(f"prototype support contains duplicate {field}")
    if frame.filter(~pl.col("prototype_only")).height:
        raise ValueError("prototype support contains non-prototype rows")
    if frame.filter(~pl.col("attribution_complete")).height:
        raise ValueError("prototype support has incomplete attribution")
    if frame.filter(pl.col("attribution").str.len_chars() == 0).height:
        raise ValueError("prototype support has blank attribution")
    if frame.filter(
        ~pl.col("licence_policy_status").is_in(["allowed", "research_only"])
    ).height:
        raise ValueError("prototype support has ineligible licensing")
    if frame.filter(
        (pl.col("licence").str.len_chars() == 0)
        | (pl.col("licence_uri").str.len_chars() == 0)
    ).height:
        raise ValueError("prototype support has incomplete licensing")
    if frame.filter(
        ~pl.col("verification_status").is_in(sorted(_ALLOWED_VERIFICATION))
    ).height:
        raise ValueError("prototype support lacks metadata-qualified evidence")
    if frame.filter(
        pl.col("human_verified")
        & (pl.col("verification_status") != "human_verified")
    ).height:
        raise ValueError("prototype support overstates human verification")
    if frame.filter(
        pl.col("qa_disposition").is_in(["excluded", "operational_failure"])
        | (pl.col("image_quality_check") == "exclude")
    ).height:
        raise ValueError("prototype support contains an invalid image")
    target = frame.filter(
        pl.col("accepted_taxon_key") == config.target_accepted_taxon_key
    )
    if target.is_empty() or set(target["scientific_name"].to_list()) != {
        config.target_scientific_name
    }:
        raise ValueError("prototype support target reconciliation failed")
    for row in frame.select("route", "life_stage", "visual_domain").iter_rows():
        route, life_stage, visual_domain = (str(value) for value in row)
        if (life_stage, visual_domain) not in _ROUTE_DIMENSIONS.get(route, set()):
            raise ValueError("prototype support route separation is invalid")


def _validate_readiness(
    payload: dict[str, Any],
    *,
    config: BuildWeekPrototypeConfig,
    support: pl.DataFrame,
) -> PrototypeReadinessPermit:
    if payload.get("schema_version") != PROTOTYPE_READINESS_SCHEMA_VERSION:
        raise ValueError("unsupported prototype readiness schema")
    status = str(payload.get("prototype_readiness_status") or "")
    if status not in _ALLOWED_READINESS:
        raise ValueError("prototype readiness does not authorize classification")
    if payload.get("classification_authorised") is not True:
        raise ValueError("prototype readiness does not authorize classification")
    if payload.get("bank_status") != "prototype_only":
        raise ValueError("prototype readiness is not prototype-only")
    if payload.get("human_verification_complete") is not False:
        raise ValueError("prototype readiness overstates human verification")
    if (
        payload.get("target_accepted_taxon_key")
        != config.target_accepted_taxon_key
        or payload.get("target_scientific_name") != config.target_scientific_name
    ):
        raise ValueError("prototype readiness target reconciliation failed")
    support_fingerprint = canonical_semantic_fingerprint(support.to_dicts())
    if payload.get("support_manifest_fingerprint") != support_fingerprint:
        raise ValueError("prototype readiness support fingerprint mismatch")
    counts = dict(payload.get("counts") or {})
    return PrototypeReadinessPermit(
        status=status,
        readiness_sha256=config.reference_bank_readiness_sha256,
        deployment_status="prototype",
        bank_status="prototype_only",
        classification_authorised=True,
        human_verification_complete=False,
        target_accepted_taxon_key=config.target_accepted_taxon_key,
        target_scientific_name=config.target_scientific_name,
        support_manifest_fingerprint=support_fingerprint,
        prototype_support_count=int(counts.get("prototype_support_count") or 0),
        human_verified_count=int(counts.get("human_verified_count") or 0),
        score_semantics=PROTOTYPE_SCORE_SEMANTICS,
    )


def _validate_embedding_bindings(
    frame: pl.DataFrame,
    *,
    config: BuildWeekPrototypeConfig,
    support: pl.DataFrame,
) -> None:
    expected_support = canonical_semantic_fingerprint(support.to_dicts())
    expected = {
        "support_manifest_fingerprint": expected_support,
        "model_revision": config.model_revision,
        "preprocessing_version": config.preprocessing_version,
    }
    for field, value in expected.items():
        values = frame[field].unique().to_list()
        if values != [value]:
            raise ValueError(
                f"prototype reference embeddings {field} does not match "
                "the frozen configuration"
            )
    support_ids = set(support["reference_media_id"].to_list())
    embedding_ids = set(frame["reference_media_id"].to_list())
    if embedding_ids != support_ids:
        raise ValueError("prototype reference embeddings do not cover support exactly")
    support_rows = set(support["support_row_fingerprint"].to_list())
    if set(frame["support_row_fingerprint"].to_list()) != support_rows:
        raise ValueError("prototype embedding support identities are stale")


def _validate_policy(
    payload: dict[str, Any],
    *,
    config: BuildWeekPrototypeConfig,
    support: pl.DataFrame,
) -> None:
    if payload.get("deployment_status") != "prototype":
        raise ValueError("prototype policy deployment status is invalid")
    if payload.get("policy_status") != "prototype_uncalibrated":
        raise ValueError("prototype policy must remain explicitly uncalibrated")
    if payload.get("score_semantics") != PROTOTYPE_SCORE_SEMANTICS:
        raise ValueError("prototype policy score semantics are invalid")
    if payload.get("experimental_screening_evidence_only") is not True:
        raise ValueError("prototype policy overstates evidence semantics")
    target = dict(payload.get("target") or {})
    if (
        target.get("accepted_taxon_key") != config.target_accepted_taxon_key
        or target.get("scientific_name") != config.target_scientific_name
    ):
        raise ValueError("prototype policy target reconciliation failed")
    identity = dict(payload.get("frozen_identity") or {})
    expected = {
        "classifier_fingerprint": config.classifier_fingerprint,
        "model_revision": config.model_revision,
        "preprocessing_version": config.preprocessing_version,
        "visual_input_version": config.visual_input_version,
        "support_manifest_fingerprint": canonical_semantic_fingerprint(
            support.to_dicts()
        ),
    }
    for field, value in expected.items():
        if identity.get(field) != value:
            raise ValueError(f"prototype policy {field} is stale")
    calibration = dict(payload.get("calibration") or {})
    if (
        calibration.get("calibrator_fingerprint") is not None
        or calibration.get("probabilities_emitted") is not False
    ):
        raise ValueError("prototype policy falsely claims calibrated probabilities")
    selected = dict(payload.get("selected_policy") or {})
    required = {
        "higher_rank_pruning_permitted": False,
        "spatial_crop_permitted": False,
        "target_always_scored": True,
        "visual_input": "raw_full_image",
    }
    for field, value in required.items():
        if selected.get(field) != value:
            raise ValueError(f"prototype policy invariant failed: {field}")


def _validate_candidate_scores(
    frame: pl.DataFrame,
    *,
    config: BuildWeekPrototypeConfig,
) -> None:
    required = {
        "flickr_photo_id",
        "class_kind",
        "class_id",
        "accepted_taxon_key",
        "target_candidate",
        "score_semantics",
        "experimental_screening_evidence",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"prototype candidate scores omit columns: {missing}")
    if frame.is_empty():
        raise ValueError("prototype candidate score evidence is empty")
    if frame.filter(
        (pl.col("score_semantics") != PROTOTYPE_SCORE_SEMANTICS)
        | ~pl.col("experimental_screening_evidence")
    ).height:
        raise ValueError("prototype candidate scores overstate score semantics")
    target_rows = frame.filter(pl.col("target_candidate"))
    if target_rows.filter(
        pl.col("accepted_taxon_key") != config.target_accepted_taxon_key
    ).height:
        raise ValueError("prototype candidate target identity is inconsistent")
    target_counts = target_rows.group_by("flickr_photo_id").len()
    if (
        target_rows["flickr_photo_id"].n_unique()
        != frame["flickr_photo_id"].n_unique()
        or target_counts.filter(pl.col("len") != 1).height
    ):
        raise ValueError("prototype candidate evidence does not score target once")
    if (
        frame.select("flickr_photo_id", "class_kind", "class_id").unique().height
        != frame.height
    ):
        raise ValueError("prototype candidate evidence contains duplicate classes")


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"prototype artifact must be a JSON object: {path}")
    return payload


__all__ = [
    "MetadataQualifiedPrototypePermit",
    "PROTOTYPE_SCORE_SEMANTICS",
    "PrototypeReadinessPermit",
    "validate_metadata_qualified_prototype_support",
]
