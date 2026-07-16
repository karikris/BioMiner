from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
import json
from typing import Any

import polars as pl

from biominer.common.semantic_hash import canonical_semantic_fingerprint
from biominer.evaluation.splits import (
    DATASET_SPLITS,
    DatasetSplitBuild,
    DatasetSplitConfig,
    DatasetSplitItem,
    build_dataset_split_manifest,
    validate_dataset_split_manifest,
)
from biominer.references.prototype_acquisition import (
    validate_prototype_reference_selections,
)
from biominer.references.prototype_qa import validate_prototype_qa_result
from biominer.references.schemas import validate_reference_media_objects
from biominer.storage.cloud import CloudStorage
from biominer.storage.uri import join_uri


PROTOTYPE_FREEZE_VERSION = "prototype-support-freeze-v1.0.0"
PROTOTYPE_SUPPORT_SCHEMA_VERSION = "prototype-support-manifest-v1.0.0"
PROTOTYPE_EXCLUDED_SCHEMA_VERSION = "prototype-excluded-manifest-v1.0.0"
PROTOTYPE_DUPLICATE_GROUPS_SCHEMA_VERSION = "prototype-duplicate-groups-v1.0.0"
PROTOTYPE_READINESS_SCHEMA_VERSION = "reference-bank-prototype-readiness-v1.0.0"

PROTOTYPE_SUPPORT_FILE = "prototype_support_manifest.parquet"
PROTOTYPE_EXCLUDED_FILE = "prototype_excluded_manifest.parquet"
PROTOTYPE_DUPLICATE_GROUPS_FILE = "prototype_duplicate_groups.parquet"
PROTOTYPE_SPLIT_FILE = "prototype_dataset_split_manifest.parquet"
PROTOTYPE_READINESS_FILE = "reference_bank_prototype_readiness.json"

_ALLOWED_VERIFICATION = frozenset(
    {"human_verified", "provider_high_trust", "provider_supported"}
)
_ALLOWED_TRUST = frozenset({"R1", "R2", "R3", "R4"})
_ROUTE_LIFE_STAGES = {
    "adult_field": frozenset({"adult"}),
    "larval": frozenset({"larva", "caterpillar"}),
    "pinned_specimen": frozenset({"adult", "unknown"}),
}


@dataclass(frozen=True, slots=True)
class PrototypeFreezeConfig:
    reference_bank_version: str
    split_version: str
    target_accepted_taxon_key: str
    target_scientific_name: str
    random_seed: int = 20260715
    support_weight: int = 55
    model_selection_weight: int = 15
    calibration_weight: int = 15
    final_test_weight: int = 15
    minimum_target_adult_support_train: int = 5
    minimum_regional_competitor_species_support_train: int = 1
    generated_at: str | datetime = "2026-07-16T04:00:00Z"

    def __post_init__(self) -> None:
        for field in (
            "reference_bank_version",
            "split_version",
            "target_accepted_taxon_key",
            "target_scientific_name",
        ):
            value = str(getattr(self, field)).strip()
            if not value:
                raise ValueError(f"{field} must be nonblank")
            object.__setattr__(self, field, value)
        for field in (
            "random_seed",
            "support_weight",
            "model_selection_weight",
            "calibration_weight",
            "final_test_weight",
            "minimum_target_adult_support_train",
            "minimum_regional_competitor_species_support_train",
        ):
            value = getattr(self, field)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{field} must be a nonnegative integer")
        if (
            min(
                self.support_weight,
                self.model_selection_weight,
                self.calibration_weight,
                self.final_test_weight,
            )
            <= 0
        ):
            raise ValueError("prototype split weights must be positive")
        object.__setattr__(self, "generated_at", _utc_datetime(self.generated_at))

    @property
    def split_config(self) -> DatasetSplitConfig:
        return DatasetSplitConfig(
            split_version=self.split_version,
            random_seed=self.random_seed,
            support_train_weight=self.support_weight,
            model_selection_weight=self.model_selection_weight,
            calibration_weight=self.calibration_weight,
            final_test_weight=self.final_test_weight,
            require_class_coverage=False,
        )

    @property
    def fingerprint(self) -> str:
        return canonical_semantic_fingerprint(
            {
                "schema_version": PROTOTYPE_FREEZE_VERSION,
                "reference_bank_version": self.reference_bank_version,
                "split_version": self.split_version,
                "target_accepted_taxon_key": self.target_accepted_taxon_key,
                "target_scientific_name": self.target_scientific_name,
                "random_seed": self.random_seed,
                "weights": [
                    self.support_weight,
                    self.model_selection_weight,
                    self.calibration_weight,
                    self.final_test_weight,
                ],
                "minimum_target_adult_support_train": (
                    self.minimum_target_adult_support_train
                ),
                "minimum_regional_competitor_species_support_train": (
                    self.minimum_regional_competitor_species_support_train
                ),
            }
        )


@dataclass(frozen=True, slots=True)
class PrototypeFreezeResult:
    support: pl.DataFrame
    excluded: pl.DataFrame
    duplicate_groups: pl.DataFrame
    split: DatasetSplitBuild
    readiness: dict[str, Any]


def prototype_support_schema() -> dict[str, pl.DataType]:
    return {
        "schema_version": pl.String,
        "reference_bank_version": pl.String,
        "reference_media_id": pl.String,
        "reference_observation_id": pl.String,
        "candidate_scope_type": pl.String,
        "accepted_taxon_key": pl.String,
        "scientific_name": pl.String,
        "source": pl.String,
        "source_snapshot_version": pl.String,
        "provider_media_id": pl.String,
        "trust_level": pl.String,
        "verification_status": pl.String,
        "human_verified": pl.Boolean,
        "geographic_layer": pl.String,
        "geo_cluster_id": pl.String,
        "route": pl.String,
        "life_stage": pl.String,
        "visual_domain": pl.String,
        "reference_group": pl.String,
        "licence": pl.String,
        "licence_uri": pl.String,
        "licence_policy_status": pl.String,
        "attribution": pl.String,
        "attribution_complete": pl.Boolean,
        "source_object_uri": pl.String,
        "source_image_sha256": pl.String,
        "source_object_fingerprint": pl.String,
        "duplicate_group_id": pl.String,
        "exact_hash_group_id": pl.String,
        "perceptual_duplicate_group_id": pl.String,
        "observation_group_id": pl.String,
        "burst_group_id": pl.String,
        "owner_group_id": pl.String,
        "photographer_group_id": pl.String,
        "provider_mirror_group_id": pl.String,
        "qa_disposition": pl.String,
        "image_quality_check": pl.String,
        "subject_presence_check": pl.String,
        "subject_size_check": pl.String,
        "detector_evidence_status": pl.String,
        "dataset_split": pl.String,
        "leakage_component_id": pl.String,
        "leakage_component_size": pl.UInt32,
        "split_fingerprint": pl.String,
        "prototype_only": pl.Boolean,
        "support_row_fingerprint": pl.String,
    }


def prototype_excluded_schema() -> dict[str, pl.DataType]:
    return {
        "schema_version": pl.String,
        "reference_bank_version": pl.String,
        "reference_media_id": pl.String,
        "reference_observation_id": pl.String,
        "candidate_scope_type": pl.String,
        "candidate_scope_id": pl.String,
        "source": pl.String,
        "route": pl.String,
        "verification_status": pl.String,
        "qa_disposition": pl.String,
        "exclusion_reason": pl.String,
        "retryable_operational_failure": pl.Boolean,
        "human_verified": pl.Boolean,
        "prototype_only": pl.Boolean,
        "exclusion_fingerprint": pl.String,
    }


def prototype_duplicate_groups_schema() -> dict[str, pl.DataType]:
    return {
        "schema_version": pl.String,
        "reference_media_id": pl.String,
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


def freeze_prototype_support_bank(
    *,
    selections: pl.DataFrame,
    media_candidates: pl.DataFrame,
    media_objects: pl.DataFrame,
    identity_groups: pl.DataFrame,
    qualifications: pl.DataFrame,
    biological_observations: Sequence[pl.DataFrame],
    regional_competitor_keys: Sequence[str],
    false_winner_keys: Sequence[str],
    config: PrototypeFreezeConfig,
) -> PrototypeFreezeResult:
    validate_prototype_reference_selections(selections)
    validate_reference_media_objects(media_objects)
    validate_prototype_qa_result(_qa_result_for_validation(qualifications))
    _validate_input_membership(
        selections, media_candidates, media_objects, identity_groups, qualifications
    )
    observations = _observation_lookup(biological_observations)
    candidates = _by_media_id(media_candidates)
    objects = _by_media_id(media_objects)
    identities = _by_media_id(identity_groups)
    qa = _by_media_id(qualifications)

    eligible_rows: list[dict[str, object]] = []
    excluded_rows: list[dict[str, object]] = []
    for selection in selections.sort("reference_media_id").iter_rows(named=True):
        media_id = str(selection["reference_media_id"])
        reasons = _exclusion_reasons(
            selection=selection,
            candidate=candidates[media_id],
            media_object=objects[media_id],
            identity=identities[media_id],
            qa=qa[media_id],
        )
        if reasons:
            excluded_rows.append(
                _excluded_row(selection, qa[media_id], reasons, config)
            )
            continue
        observation = observations.get(str(selection["reference_observation_id"]))
        if observation is None:
            raise ValueError(f"eligible prototype row lacks observation: {media_id}")
        eligible_rows.append(
            _support_base_row(
                selection,
                candidates[media_id],
                objects[media_id],
                identities[media_id],
                qa[media_id],
                observation,
                config,
            )
        )

    if not eligible_rows:
        raise ValueError("prototype freeze has no eligible support rows")
    split = build_dataset_split_manifest(
        tuple(_split_item(row, config) for row in eligible_rows),
        config.split_config,
    )
    split_by_media = {
        str(row["item_id"]): row for row in split.manifest.iter_rows(named=True)
    }
    support_rows = []
    for row in eligible_rows:
        assignment = split_by_media[str(row["reference_media_id"])]
        combined = {
            **row,
            "dataset_split": assignment["dataset_split"],
            "leakage_component_id": assignment["leakage_component_id"],
            "leakage_component_size": assignment["leakage_component_size"],
            "split_fingerprint": assignment["split_fingerprint"],
            "prototype_only": True,
        }
        combined["support_row_fingerprint"] = _row_fingerprint(
            combined, field="support_row_fingerprint"
        )
        support_rows.append(combined)
    support = pl.DataFrame(support_rows, schema=prototype_support_schema()).sort(
        "reference_media_id"
    )
    excluded = pl.DataFrame(excluded_rows, schema=prototype_excluded_schema()).sort(
        "reference_media_id"
    )
    duplicate_groups = _duplicate_groups(identity_groups)
    readiness = _readiness(
        support=support,
        excluded=excluded,
        duplicate_groups=duplicate_groups,
        split=split,
        regional_competitor_keys=tuple(regional_competitor_keys),
        false_winner_keys=tuple(false_winner_keys),
        config=config,
    )
    result = PrototypeFreezeResult(
        support=support,
        excluded=excluded,
        duplicate_groups=duplicate_groups,
        split=split,
        readiness=readiness,
    )
    validate_prototype_freeze_result(result)
    return result


def validate_prototype_freeze_result(result: PrototypeFreezeResult) -> None:
    _validate_frame(result.support, prototype_support_schema(), "support")
    _validate_frame(result.excluded, prototype_excluded_schema(), "excluded")
    _validate_frame(
        result.duplicate_groups,
        prototype_duplicate_groups_schema(),
        "duplicate groups",
    )
    validate_dataset_split_manifest(result.split.manifest)
    support_ids = set(result.support["reference_media_id"].to_list())
    excluded_ids = set(result.excluded["reference_media_id"].to_list())
    duplicate_ids = set(result.duplicate_groups["reference_media_id"].to_list())
    if support_ids & excluded_ids or support_ids | excluded_ids != duplicate_ids:
        raise ValueError("prototype freeze support/exclusion partition is incomplete")
    if result.support.filter(~pl.col("prototype_only")).height:
        raise ValueError("prototype support contains a non-prototype row")
    if result.support.filter(
        pl.col("human_verified") & (pl.col("verification_status") != "human_verified")
    ).height:
        raise ValueError("prototype support human verification flag is inconsistent")
    payload = result.readiness
    if payload.get("schema_version") != PROTOTYPE_READINESS_SCHEMA_VERSION:
        raise ValueError("prototype readiness schema is invalid")
    if payload.get("prototype_readiness_status") not in {
        "prototype_ready",
        "prototype_ready_with_shortfalls",
        "blocked",
        "invalid",
    }:
        raise ValueError("prototype readiness status is invalid")
    if payload.get("bank_status") != "prototype_only":
        raise ValueError("prototype readiness lacks prototype-only status")
    if payload.get("human_verification_complete") is not False:
        raise ValueError("prototype readiness overstates human verification")


def publish_prototype_freeze_result(
    result: PrototypeFreezeResult,
    *,
    storage: CloudStorage,
    output_prefix: str,
    settings_fingerprint: str | None = None,
) -> dict[str, str]:
    validate_prototype_freeze_result(result)
    prefix = str(output_prefix).strip().rstrip("/")
    if not prefix:
        raise ValueError("prototype freeze output_prefix must be nonblank")
    uris = {
        "support": join_uri(prefix, PROTOTYPE_SUPPORT_FILE),
        "excluded": join_uri(prefix, PROTOTYPE_EXCLUDED_FILE),
        "duplicate_groups": join_uri(prefix, PROTOTYPE_DUPLICATE_GROUPS_FILE),
        "split": join_uri(prefix, PROTOTYPE_SPLIT_FILE),
        "readiness": join_uri(prefix, PROTOTYPE_READINESS_FILE),
    }
    if any(storage.exists(uri) for uri in uris.values()):
        raise FileExistsError("prototype support freeze already exists")
    storage.write_parquet_shard(uris["support"], result.support, overwrite=False)
    storage.write_parquet_shard(uris["excluded"], result.excluded, overwrite=False)
    storage.write_parquet_shard(
        uris["duplicate_groups"], result.duplicate_groups, overwrite=False
    )
    storage.write_parquet_shard(uris["split"], result.split.manifest, overwrite=False)
    payload = json.loads(json.dumps(result.readiness))
    payload["settings_fingerprint"] = settings_fingerprint
    payload["artifacts"] = {
        name: {
            "uri": uri,
            "byte_count": storage.file_size(uri),
            "sha256": storage.file_sha256(uri),
            "row_count": _artifact_rows(result, name),
        }
        for name, uri in uris.items()
        if name != "readiness"
    }
    storage.write_json(uris["readiness"], payload)
    return uris


def _support_base_row(
    selection, candidate, media_object, identity, qa, observation, config
):  # noqa: ANN001
    verification = str(selection["verification_status"])
    return {
        "schema_version": PROTOTYPE_SUPPORT_SCHEMA_VERSION,
        "reference_bank_version": config.reference_bank_version,
        "reference_media_id": selection["reference_media_id"],
        "reference_observation_id": selection["reference_observation_id"],
        "candidate_scope_type": selection["candidate_scope_type"],
        "accepted_taxon_key": selection["candidate_scope_id"],
        "scientific_name": selection["candidate_name"],
        "source": selection["source"],
        "source_snapshot_version": candidate["source_snapshot_version"],
        "provider_media_id": selection["provider_media_id"],
        "trust_level": selection["trust_level"],
        "verification_status": verification,
        "human_verified": verification == "human_verified",
        "geographic_layer": selection["geographic_layer"],
        "geo_cluster_id": observation.get("geo_cluster_id"),
        "route": selection["route"],
        "life_stage": selection["life_stage"],
        "visual_domain": selection["visual_domain"],
        "reference_group": selection["reference_group"],
        "licence": selection["licence"],
        "licence_uri": selection["licence_uri"],
        "licence_policy_status": selection["licence_policy_status"],
        "attribution": selection["attribution"],
        "attribution_complete": selection["attribution_complete"],
        "source_object_uri": media_object["source_object_uri"],
        "source_image_sha256": media_object["sha256"],
        "source_object_fingerprint": media_object["object_fingerprint"],
        "duplicate_group_id": identity["duplicate_group_id"],
        "exact_hash_group_id": identity["exact_hash_group_id"],
        "perceptual_duplicate_group_id": identity["perceptual_duplicate_group_id"],
        "observation_group_id": identity["observation_group_id"],
        "burst_group_id": identity["burst_group_id"],
        "owner_group_id": identity["owner_group_id"],
        "photographer_group_id": identity["photographer_group_id"],
        "provider_mirror_group_id": identity["provider_mirror_group_id"],
        "qa_disposition": qa["qa_disposition"],
        "image_quality_check": qa["image_quality_check"],
        "subject_presence_check": qa["subject_presence_check"],
        "subject_size_check": qa["subject_size_check"],
        "detector_evidence_status": qa["detector_evidence_status"],
    }


def _split_item(
    row: Mapping[str, object], config: PrototypeFreezeConfig
) -> DatasetSplitItem:
    return DatasetSplitItem(
        item_type="prototype_support_media",
        item_id=str(row["reference_media_id"]),
        source=str(row["source"]),
        route=str(row["route"]),
        stratification_label=str(row["accepted_taxon_key"]),
        accepted_class_taxon_key=str(row["accepted_taxon_key"]),
        source_artifact_fingerprint=config.fingerprint,
        source_observation_id=_optional_text(row["observation_group_id"]),
        source_owner_id=_optional_text(row["owner_group_id"]),
        observer_id=_optional_text(row["owner_group_id"]),
        photographer_id=_optional_text(row["photographer_group_id"]),
        duplicate_group_id=_optional_text(row["duplicate_group_id"]),
        exact_hash_group_id=_optional_text(row["exact_hash_group_id"]),
        perceptual_duplicate_group_id=_optional_text(
            row["perceptual_duplicate_group_id"]
        ),
        burst_group_id=_optional_text(row["burst_group_id"]),
        provider_mirror_group_id=_optional_text(row["provider_mirror_group_id"]),
        geo_cluster_id=None,
    )


def _exclusion_reasons(*, selection, candidate, media_object, identity, qa):  # noqa: ANN001
    reasons = []
    if selection.get("candidate_scope_type") != "accepted_taxon":
        reasons.append("non_taxonomic_visual_domain_scope")
    if selection.get("trust_level") not in _ALLOWED_TRUST:
        reasons.append("trust_level_ineligible")
    if selection.get("verification_status") not in _ALLOWED_VERIFICATION:
        reasons.append("verification_status_ineligible")
    if media_object.get("decode_status") != "valid":
        reasons.append(str(media_object.get("quarantine_reason") or "decode_failed"))
    if identity.get("support_disposition") != "eligible":
        reasons.append(
            str(identity.get("support_disposition") or "duplicate_unresolved")
        )
    if qa.get("qa_disposition") in {"excluded", "operational_failure"}:
        reasons.append(str(qa.get("qa_reason") or qa.get("qa_disposition")))
    if qa.get("image_quality_check") == "exclude":
        reasons.append("fatal_image_quality")
    if qa.get("metadata_disagreement_check") != "pass":
        reasons.append("known_metadata_disagreement")
    if qa.get("licence_completeness_check") != "pass":
        reasons.append("licence_incomplete")
    if qa.get("attribution_completeness_check") != "pass":
        reasons.append("attribution_incomplete")
    route = str(selection.get("route") or "")
    stage = str(selection.get("life_stage") or "")
    if route not in _ROUTE_LIFE_STAGES or stage not in _ROUTE_LIFE_STAGES.get(
        route, ()
    ):
        reasons.append("wrong_life_stage_or_route")
    if candidate.get("licence_policy_status") not in {"allowed", "research_only"}:
        reasons.append("licence_ineligible")
    return tuple(sorted(set(reasons)))


def _excluded_row(selection, qa, reasons, config):  # noqa: ANN001
    retryable = qa.get("qa_disposition") == "operational_failure"
    row = {
        "schema_version": PROTOTYPE_EXCLUDED_SCHEMA_VERSION,
        "reference_bank_version": config.reference_bank_version,
        "reference_media_id": selection["reference_media_id"],
        "reference_observation_id": selection["reference_observation_id"],
        "candidate_scope_type": selection["candidate_scope_type"],
        "candidate_scope_id": selection["candidate_scope_id"],
        "source": selection["source"],
        "route": selection["route"],
        "verification_status": selection["verification_status"],
        "qa_disposition": qa["qa_disposition"],
        "exclusion_reason": ",".join(reasons),
        "retryable_operational_failure": retryable,
        "human_verified": selection["verification_status"] == "human_verified",
        "prototype_only": True,
    }
    row["exclusion_fingerprint"] = _row_fingerprint(row, field="exclusion_fingerprint")
    return row


def _duplicate_groups(identity_groups: pl.DataFrame) -> pl.DataFrame:
    columns = [
        name for name in prototype_duplicate_groups_schema() if name != "schema_version"
    ]
    return (
        identity_groups.select(columns)
        .with_columns(
            pl.lit(PROTOTYPE_DUPLICATE_GROUPS_SCHEMA_VERSION).alias("schema_version")
        )
        .select(*prototype_duplicate_groups_schema())
        .sort("reference_media_id")
    )


def _readiness(
    *,
    support,
    excluded,
    duplicate_groups,
    split,
    regional_competitor_keys,
    false_winner_keys,
    config,
):  # noqa: ANN001
    support_train = support.filter(pl.col("dataset_split") == "support_train")
    target_adult = support_train.filter(
        (pl.col("accepted_taxon_key") == config.target_accepted_taxon_key)
        & (pl.col("route") == "adult_field")
    ).height
    competitor_count = support_train.filter(
        pl.col("accepted_taxon_key").is_in(regional_competitor_keys)
    )["accepted_taxon_key"].n_unique()
    false_winner_count = support_train.filter(
        pl.col("accepted_taxon_key").is_in(false_winner_keys)
    )["accepted_taxon_key"].n_unique()
    support_ids = set(support["reference_media_id"].to_list())
    duplicate_support = duplicate_groups.filter(
        pl.col("reference_media_id").is_in(support_ids)
    )
    route_crossings = (
        support.filter(pl.col("observation_group_id").is_not_null())
        .group_by("observation_group_id")
        .agg(pl.col("route").n_unique().alias("route_count"))
        .filter(pl.col("route_count") > 1)
        .height
    )
    cross_split_groups = _cross_split_group_count(support)
    checks = [
        _check(
            "target_support_minimum",
            target_adult >= config.minimum_target_adult_support_train,
            target_adult,
            config.minimum_target_adult_support_train,
        ),
        _check(
            "regional_competitor_support",
            competitor_count
            >= config.minimum_regional_competitor_species_support_train,
            competitor_count,
            config.minimum_regional_competitor_species_support_train,
        ),
        _check("false_winner_support", false_winner_count > 0, false_winner_count, 1),
        _check(
            "biological_hard_negatives",
            support.filter(pl.col("reference_group").str.contains("negative")).height
            > 0,
            support.filter(pl.col("reference_group").str.contains("negative")).height,
            1,
        ),
        _check(
            "adult_larval_specimen_separation",
            route_crossings == 0,
            route_crossings,
            0,
        ),
        _check(
            "licence_complete",
            support.filter(
                pl.col("licence").is_null() | pl.col("licence_uri").is_null()
            ).height
            == 0,
            0,
            0,
        ),
        _check(
            "attribution_complete",
            support.filter(~pl.col("attribution_complete")).height == 0,
            0,
            0,
        ),
        _check(
            "exact_duplicates_removed",
            duplicate_support.filter(~pl.col("is_canonical")).height == 0,
            duplicate_support.filter(~pl.col("is_canonical")).height,
            0,
        ),
        _check(
            "provider_mirrors_resolved",
            duplicate_support.filter(pl.col("resolution_status") != "resolved").height
            == 0,
            duplicate_support.filter(pl.col("resolution_status") != "resolved").height,
            0,
        ),
        _check(
            "split_leakage_components_atomic",
            cross_split_groups == 0,
            cross_split_groups,
            0,
        ),
        _check(
            "source_trust_persisted",
            support.filter(pl.col("trust_level").is_null()).height == 0,
            support.height,
            support.height,
        ),
        _check(
            "geographic_layers_persisted",
            support.filter(pl.col("geographic_layer").is_null()).height == 0,
            support.height,
            support.height,
        ),
        _check(
            "prototype_only",
            support.filter(~pl.col("prototype_only")).height == 0,
            True,
            True,
        ),
        _check(
            "human_verification_claims",
            support.filter(
                pl.col("human_verified")
                & (pl.col("verification_status") != "human_verified")
            ).height
            == 0,
            0,
            0,
        ),
    ]
    fatal = [check for check in checks if check["status"] == "failed"]
    shortfalls = _shortfalls(support, excluded, split, config)
    status = (
        "blocked"
        if fatal
        else ("prototype_ready_with_shortfalls" if shortfalls else "prototype_ready")
    )
    return {
        "schema_version": PROTOTYPE_READINESS_SCHEMA_VERSION,
        "prototype_readiness_status": status,
        "classification_authorised": status
        in {"prototype_ready", "prototype_ready_with_shortfalls"},
        "bank_status": "prototype_only",
        "reference_bank_version": config.reference_bank_version,
        "target_accepted_taxon_key": config.target_accepted_taxon_key,
        "target_scientific_name": config.target_scientific_name,
        "generated_at": config.generated_at.isoformat(),
        "policy_version": PROTOTYPE_FREEZE_VERSION,
        "policy_fingerprint": config.fingerprint,
        "support_manifest_fingerprint": canonical_semantic_fingerprint(
            support.to_dicts()
        ),
        "split_fingerprint": split.split_fingerprint,
        "human_verification_complete": False,
        "human_verification_required_for_scientific_release": True,
        "counts": _counts(support, excluded, duplicate_groups, split),
        "checks": checks,
        "documented_shortfalls": shortfalls,
        "unresolved_risks": _unresolved_risks(support, excluded, duplicate_groups),
        "semantics": {
            "prototype_only": True,
            "provider_supported_is_human_verified": False,
            "experimental_screening_evidence_only": True,
            "raw_scores_are_probabilities": False,
            "missing_identity_metadata_means_complete_leakage_protection": False,
            "operational_failures_are_biological_negatives": False,
        },
    }


def _counts(
    support: pl.DataFrame,
    excluded: pl.DataFrame,
    duplicate_groups: pl.DataFrame,
    split: DatasetSplitBuild,
) -> dict[str, object]:
    def distribution(frame: pl.DataFrame, column: str) -> dict[str, int]:
        return dict(
            sorted(Counter(str(value) for value in frame[column].to_list()).items())
        )

    return {
        "selected": support.height + excluded.height,
        "prototype_support_count": support.height,
        "excluded_count": excluded.height,
        "retryable_operational_failure_count": excluded.filter(
            pl.col("retryable_operational_failure")
        ).height,
        "human_verified_count": support.filter(
            pl.col("verification_status") == "human_verified"
        ).height,
        "provider_high_trust_count": support.filter(
            pl.col("verification_status") == "provider_high_trust"
        ).height,
        "provider_supported_count": support.filter(
            pl.col("verification_status") == "provider_supported"
        ).height,
        "missing_owner_evidence_count": support.filter(
            pl.col("owner_group_id").is_null()
        ).height,
        "missing_photographer_evidence_count": support.filter(
            pl.col("photographer_group_id").is_null()
        ).height,
        "selected_missing_owner_evidence_count": duplicate_groups.filter(
            ~pl.col("owner_evidence_available")
        ).height,
        "selected_missing_photographer_evidence_count": duplicate_groups.filter(
            ~pl.col("photographer_evidence_available")
        ).height,
        "leakage_component_count": split.component_count,
        "split_counts": distribution(support, "dataset_split"),
        "route_counts": distribution(support, "route"),
        "trust_counts": distribution(support, "trust_level"),
        "geographic_layer_counts": distribution(support, "geographic_layer"),
        "source_counts": distribution(support, "source"),
        "licence_policy_counts": distribution(support, "licence_policy_status"),
    }


def _shortfalls(support, excluded, split, config):  # noqa: ANN001
    counts = dict(split.split_item_counts)
    total = support.height
    weights = dict(config.split_config.weights)
    shortfalls = []
    detector_missing = support.filter(
        pl.col("detector_evidence_status") != "available"
    ).height
    if detector_missing:
        shortfalls.append(
            {
                "shortfall_id": "automated_subject_evidence",
                "observed": support.height - detector_missing,
                "required": "full-bank detector evidence",
                "blocking": False,
            }
        )
    retryable = excluded.filter(pl.col("retryable_operational_failure")).height
    if retryable:
        shortfalls.append(
            {
                "shortfall_id": "retryable_media",
                "observed": retryable,
                "required": 0,
                "blocking": False,
            }
        )
    larval_support_train = support.filter(
        (pl.col("route") == "larval") & (pl.col("dataset_split") == "support_train")
    ).height
    if larval_support_train < 1:
        shortfalls.append(
            {
                "shortfall_id": "larval_support_train",
                "observed": larval_support_train,
                "required": 1,
                "blocking": False,
            }
        )
    visual_domain_count = support.filter(
        pl.col("candidate_scope_type") == "visual_domain"
    ).height
    if visual_domain_count < 1:
        shortfalls.append(
            {
                "shortfall_id": "visual_domain_negative_support",
                "observed": visual_domain_count,
                "required": 1,
                "blocking": False,
                "reason": (
                    "the only decoded visual-domain negative failed subject-presence QA; "
                    "operational failures remain retryable and are not relabelled"
                ),
            }
        )
    pinned_count = support.filter(pl.col("route") == "pinned_specimen").height
    if pinned_count < 1:
        shortfalls.append(
            {
                "shortfall_id": "pinned_specimen_support",
                "observed": pinned_count,
                "required": 1,
                "blocking": False,
            }
        )
    for split_name in DATASET_SPLITS:
        observed = counts.get(split_name, 0)
        target = total * weights[split_name] / config.split_config.total_weight
        if abs(observed - target) >= 1:
            shortfalls.append(
                {
                    "shortfall_id": f"split_weight:{split_name}",
                    "observed": observed,
                    "required": f"approximately {target:.2f}",
                    "blocking": False,
                    "reason": "transitive leakage components are atomic",
                }
            )
    return shortfalls


def _cross_split_group_count(support: pl.DataFrame) -> int:
    fields = (
        "observation_group_id",
        "owner_group_id",
        "photographer_group_id",
        "duplicate_group_id",
        "exact_hash_group_id",
        "perceptual_duplicate_group_id",
        "burst_group_id",
        "provider_mirror_group_id",
    )
    return sum(
        support.filter(pl.col(field).is_not_null())
        .group_by(field)
        .agg(pl.col("dataset_split").n_unique().alias("split_count"))
        .filter(pl.col("split_count") > 1)
        .height
        for field in fields
    )


def _unresolved_risks(
    support: pl.DataFrame,
    excluded: pl.DataFrame,
    duplicate_groups: pl.DataFrame,
) -> list[str]:
    risks = []
    detector_missing = support.filter(
        pl.col("detector_evidence_status") != "available"
    ).height
    if detector_missing:
        risks.append(
            f"Detector evidence is unavailable for {detector_missing} support rows; "
            "subject presence and size remain review-routed."
        )
    retryable = excluded.filter(pl.col("retryable_operational_failure")).height
    if retryable:
        risks.append(
            f"{retryable} selected media records remain retryable operational failures."
        )
    missing_owner = duplicate_groups.filter(~pl.col("owner_evidence_available")).height
    missing_photographer = duplicate_groups.filter(
        ~pl.col("photographer_evidence_available")
    ).height
    if missing_owner or missing_photographer:
        risks.append(
            "Owner evidence is missing for "
            f"{missing_owner} selected rows and photographer evidence for "
            f"{missing_photographer}; leakage protection uses every available "
            "identity without claiming completeness."
        )
    if support.filter(
        (pl.col("route") == "larval") & (pl.col("dataset_split") == "support_train")
    ).is_empty():
        risks.append("No larval reference is assigned to support_train.")
    unverified = support.filter(~pl.col("human_verified")).height
    if unverified:
        risks.append(
            f"{unverified} support labels lack independent human taxonomic verification."
        )
    return risks


def _check(
    check_id: str, passed: bool, observed: object, required: object
) -> dict[str, object]:
    return {
        "check_id": check_id,
        "status": "passed" if passed else "failed",
        "observed": observed,
        "required": required,
    }


def _validate_input_membership(
    selections, media_candidates, media_objects, identities, qa
):  # noqa: ANN001
    expected = set(selections["reference_media_id"].to_list())
    for name, frame in (
        ("media candidates", media_candidates),
        ("media objects", media_objects),
        ("identity groups", identities),
        ("qualifications", qa),
    ):
        if (
            "reference_media_id" not in frame.columns
            or set(frame["reference_media_id"].to_list()) != expected
        ):
            raise ValueError(f"prototype freeze {name} differ from selections")


def _observation_lookup(
    frames: Sequence[pl.DataFrame],
) -> dict[str, Mapping[str, object]]:
    result = {}
    for frame in frames:
        for row in frame.iter_rows(named=True):
            key = str(row["reference_observation_id"])
            if key in result:
                raise ValueError(f"duplicate observation identity: {key}")
            result[key] = row
    return result


def _by_media_id(frame: pl.DataFrame) -> dict[str, Mapping[str, object]]:
    return {str(row["reference_media_id"]): row for row in frame.iter_rows(named=True)}


def _qa_result_for_validation(frame: pl.DataFrame):
    from biominer.references.prototype_qa import PrototypeQAResult

    report = {
        "schema_version": "prototype-support-qa-v1.0.0",
        "semantics": {"human_taxonomic_verification": False},
    }
    return PrototypeQAResult(frame, report, "")


def _validate_frame(
    frame: pl.DataFrame, schema: Mapping[str, pl.DataType], artifact: str
) -> None:
    if frame.schema != schema:
        raise ValueError(f"prototype {artifact} physical schema mismatch")
    if frame["reference_media_id"].n_unique() != frame.height:
        raise ValueError(f"prototype {artifact} contains duplicate media IDs")
    if not frame.equals(frame.sort("reference_media_id")):
        raise ValueError(f"prototype {artifact} is not deterministically sorted")


def _row_fingerprint(row: Mapping[str, object], *, field: str) -> str:
    return canonical_semantic_fingerprint(
        {key: value for key, value in row.items() if key != field}
    )


def _optional_text(value: object) -> str | None:
    return None if value is None else str(value)


def _utc_datetime(value: str | datetime) -> datetime:
    parsed = (
        datetime.fromisoformat(value.replace("Z", "+00:00"))
        if isinstance(value, str)
        else value
    )
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _artifact_rows(result: PrototypeFreezeResult, name: str) -> int:
    return {
        "support": result.support.height,
        "excluded": result.excluded.height,
        "duplicate_groups": result.duplicate_groups.height,
        "split": result.split.manifest.height,
    }[name]


__all__ = [
    "PROTOTYPE_DUPLICATE_GROUPS_FILE",
    "PROTOTYPE_EXCLUDED_FILE",
    "PROTOTYPE_READINESS_FILE",
    "PROTOTYPE_SPLIT_FILE",
    "PROTOTYPE_SUPPORT_FILE",
    "PrototypeFreezeConfig",
    "PrototypeFreezeResult",
    "freeze_prototype_support_bank",
    "prototype_duplicate_groups_schema",
    "prototype_excluded_schema",
    "prototype_support_schema",
    "publish_prototype_freeze_result",
    "validate_prototype_freeze_result",
]
