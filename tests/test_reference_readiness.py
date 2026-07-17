from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import FrozenInstanceError, dataclass, replace
from datetime import UTC, datetime, timedelta
import hashlib
import json
from pathlib import Path

import polars as pl
import pytest

import biominer.references.readiness as readiness_module
from biominer.candidates.regional_union import (
    REGIONAL_CANDIDATE_SPECIES_SCHEMA_VERSION,
    regional_candidate_species_schema,
)
from biominer.references.admission import default_reference_admission_policy
from biominer.references.deduplication import deduplicate_reference_media
from biominer.references.readiness import (
    LEGACY_REFERENCE_SUPPORT_MANIFEST_SCHEMA_VERSION,
    REFERENCE_BANK_READINESS_FILE,
    REFERENCE_BANK_SPLIT_ASSIGNMENTS_SCHEMA_VERSION,
    REFERENCE_BANK_SUMMARY_FILE,
    REFERENCE_SUPPORT_MANIFEST_FILE,
    DocumentedReferenceShortfall,
    ReferenceBankReadinessPolicy,
    ReferenceBankRequirement,
    ReferenceModelInputIdentity,
    build_reference_bank_readiness,
    legacy_reference_support_manifest_v2_schema,
    load_reference_bank_readiness,
    make_reference_split_assignment_fingerprint,
    migrate_strict_reference_support_manifest_v2,
    publish_reference_bank_readiness,
    reference_bank_split_assignments_frame,
    reference_bank_split_assignments_schema,
    reference_bank_summary_schema,
    reference_readiness_allows_vision,
    reference_support_manifest_fingerprint,
    reference_support_manifest_schema,
    validate_reference_bank_readiness,
    validate_reference_bank_split_assignments,
    validate_reference_bank_summary,
    validate_reference_support_manifest,
)
from biominer.references.planner import make_reference_candidate_union_id
from biominer.references.review import (
    build_reference_review_queue,
    import_reference_review_decisions,
)
from biominer.references.schemas import (
    REFERENCE_ACQUISITION_PLAN_SCHEMA_VERSION,
    REFERENCE_ACQUISITION_SELECTIONS_SCHEMA_VERSION,
    REFERENCE_MEDIA_CANDIDATES_SCHEMA_VERSION,
    REFERENCE_MEDIA_OBJECTS_SCHEMA_VERSION,
    REFERENCE_OBSERVATIONS_SCHEMA_VERSION,
    make_acquisition_plan_id,
    make_reference_media_id,
    make_reference_observation_id,
    make_reference_selection_id,
    reference_acquisition_plan_frame,
    reference_acquisition_selections_frame,
    reference_media_candidates_frame,
    reference_media_objects_frame,
    reference_observations_frame,
    reference_review_decisions_frame,
)


NOW = datetime(2026, 7, 14, 6, 0, tzinfo=UTC)
REGISTRY_VERSION = "butterflies-v2-20260714"
BANK_VERSION = "reference-bank-v1"
TARGET = "gbif:1938069"
COMPETITOR = "gbif:888"
CANDIDATE_SET_ID = "regional:cluster-a"


def _sha(value: str) -> str:
    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


def _report_sha256(report: object) -> str:
    encoded = json.dumps(
        report,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _candidate_species() -> pl.DataFrame:
    fingerprint = _sha("candidate-set:cluster-a")
    rows = []
    for priority, (taxon_key, name, target) in enumerate(
        (
            (TARGET, "Papilio demoleus", True),
            (COMPETITOR, "Papilio polytes", False),
        )
    ):
        rows.append(
            {
                "schema_version": REGIONAL_CANDIDATE_SPECIES_SCHEMA_VERSION,
                "candidate_set_id": CANDIDATE_SET_ID,
                "target_accepted_taxon_key": TARGET,
                "geo_cluster_id": "cluster-a",
                "candidate_accepted_taxon_key": taxon_key,
                "scientific_name": name,
                "family": "Papilionidae",
                "genus": "Papilio",
                "candidate_reason": ["target" if target else "close_congener"],
                "geographic_evidence_score": 1.0,
                "occurrence_support": 10,
                "same_genus": True,
                "same_family": True,
                "known_mimic": False,
                "historical_false_positive": False,
                "visually_nearest": False,
                "target_candidate": target,
                "candidate_priority": priority,
                "source_versions": ["fixture-v1"],
                "candidate_set_fingerprint": fingerprint,
            }
        )
    return pl.DataFrame(
        rows,
        schema=regional_candidate_species_schema(),
        strict=True,
    ).sort(["candidate_set_id", "candidate_priority", "candidate_accepted_taxon_key"])


def _observation(
    *,
    index: int,
    taxon_key: str,
    name: str,
    life_stage: str = "adult",
    observer_id: str | None = None,
) -> dict[str, object]:
    source = "GBIF"
    source_id = f"observation-{index}"
    observation_id = make_reference_observation_id(source, source_id)
    return {
        "schema_version": REFERENCE_OBSERVATIONS_SCHEMA_VERSION,
        "reference_observation_id": observation_id,
        "source": source,
        "source_observation_id": source_id,
        "source_taxon_id": taxon_key.removeprefix("gbif:"),
        "supplied_scientific_name": name,
        "accepted_taxon_key": taxon_key,
        "reconciled_scientific_name": name,
        "registry_version": REGISTRY_VERSION,
        "taxon_reconciliation_status": "accepted_key_exact",
        "identification_quality": "research_grade",
        "community_taxon_status": "species",
        "identification_disagreement": False,
        "captive_or_cultivated": False,
        "observer_id": observer_id or f"observer-{index}",
        "locality": f"locality-{index}",
        "life_stage": life_stage,
        "sex": None,
        "observed_at": datetime(2025, 1, index + 1, 3, 4, tzinfo=UTC),
        "latitude": -33.87 + index / 100,
        "longitude": 151.21 + index / 100,
        "coordinate_uncertainty": 50.0,
        "coordinates_obscured": False,
        "country": "Australia",
        "country_code": "AU",
        "geo_cluster_id": "cluster-a",
        "distance_to_cluster_medoid_km": float(index + 1),
        "source_dataset_key": f"dataset-{index}",
        "source_dataset_doi": None,
        "source_record_url": f"https://example.test/gbif/{source_id}",
        "source_record_hash": _sha(f"record:{source_id}"),
        "retrieved_at": NOW,
        "source_snapshot_version": "gbif-2026-07-14",
        "source_query_fingerprint": _sha(f"query:{taxon_key}"),
        "fallback_level": 0,
        "geospatial_issue": False,
        "preserved_specimen": False,
        "fossil": False,
        "occurrence_absent": False,
        "uncertain_taxon_match": False,
        "basis_of_record_suitable": True,
    }


def _candidate(
    observation: dict[str, object],
    *,
    index: int,
    licence_status: str = "allowed",
) -> dict[str, object]:
    provider_id = f"media-{index}"
    observation_id = str(observation["reference_observation_id"])
    research_only = licence_status == "research_only"
    licence = "CC-BY-NC-4.0" if research_only else "CC-BY-4.0"
    licence_uri = (
        "https://creativecommons.org/licenses/by-nc/4.0/"
        if research_only
        else "https://creativecommons.org/licenses/by/4.0/"
    )
    return {
        "schema_version": REFERENCE_MEDIA_CANDIDATES_SCHEMA_VERSION,
        "reference_media_id": make_reference_media_id(
            "GBIF",
            provider_id,
            observation_id,
        ),
        "reference_observation_id": observation_id,
        "provider_media_id": provider_id,
        "source": "GBIF",
        "media_identifier": f"https://media.example.test/{provider_id}.jpg",
        "media_type": "StillImage",
        "width": 96,
        "height": 72,
        "creator": f"Creator {index}",
        "rights_holder": f"Creator {index}",
        "licence": licence,
        "licence_uri": licence_uri,
        "attribution": f"Creator {index} / {licence}",
        "occurrence_licence": "CC0-1.0",
        "original_provider": "GBIF",
        "media_position": 0,
        "source_checksum": None,
        "source_checksum_algorithm": None,
        "download_status": "complete",
        "verification_status": "unreviewed",
        "exclusion_reason": None,
        "licence_policy_status": licence_status,
        "retrieved_at": NOW,
        "source_snapshot_version": observation["source_snapshot_version"],
    }


def _object(
    candidate: dict[str, object],
    *,
    index: int,
    perceptual_hash: str | None = None,
) -> dict[str, object]:
    media_id = str(candidate["reference_media_id"])
    sha = _sha(f"image-{index}")
    perceptual = hashlib.md5(f"perceptual-{index}".encode()).hexdigest()  # noqa: S324
    return {
        "schema_version": REFERENCE_MEDIA_OBJECTS_SCHEMA_VERSION,
        "reference_media_id": media_id,
        "source_object_uri": f"s3://reference-objects/{sha.removeprefix('sha256:')}.jpg",
        "content_type": "image/jpeg",
        "source_byte_count": 10_000,
        "decoded_width": 96,
        "decoded_height": 72,
        "sha256": sha,
        "perceptual_hash": perceptual_hash or f"dhash128-v1:{perceptual}",
        "duplicate_group_id": None,
        "duplicate_type": None,
        "canonical_reference_media_id": None,
        "provider_mirror_ids": [],
        "downloaded_at": NOW,
        "download_attempt_count": 1,
        "licence_policy_status": candidate["licence_policy_status"],
        "decode_status": "valid",
        "quarantine_reason": None,
        "object_fingerprint": _sha(f"object:{media_id}:{sha}"),
    }


@dataclass(frozen=True)
class _Spec:
    taxon_key: str
    scientific_name: str
    life_stage: str = "adult"
    visual_domain: str = "live_field"
    licence_status: str = "allowed"
    support_split: str = "support_train"
    included: bool = True
    reviewed: bool = True
    observer_id: str | None = None
    background_group_id: str | None = None
    perceptual_hash: str | None = None
    missing_attribution: bool = False


@dataclass(frozen=True)
class _Fixture:
    candidate_species: pl.DataFrame
    acquisition_plan: pl.DataFrame
    selections: pl.DataFrame
    observations: pl.DataFrame
    candidates: pl.DataFrame
    objects: pl.DataFrame
    relationships: pl.DataFrame
    deduplication_report: dict[str, object]
    queue: pl.DataFrame
    provenance: pl.DataFrame
    decisions: pl.DataFrame
    split_assignments: pl.DataFrame
    plan_fingerprints: dict[tuple[str, str, str], str]


def _default_specs() -> tuple[_Spec, ...]:
    return (
        _Spec(TARGET, "Papilio demoleus"),
        _Spec(COMPETITOR, "Papilio polytes"),
    )


def _make_fixture(specs: tuple[_Spec, ...] | None = None) -> _Fixture:
    effective_specs = specs or _default_specs()
    candidate_species = _candidate_species()
    candidate_union_id = make_reference_candidate_union_id(candidate_species)
    observations_rows = [
        _observation(
            index=index,
            taxon_key=spec.taxon_key,
            name=spec.scientific_name,
            life_stage=spec.life_stage,
            observer_id=spec.observer_id,
        )
        for index, spec in enumerate(effective_specs)
    ]
    candidate_rows = [
        _candidate(
            observation,
            index=index,
            licence_status=spec.licence_status,
        )
        for index, (spec, observation) in enumerate(
            zip(effective_specs, observations_rows, strict=True)
        )
    ]
    for candidate, spec in zip(candidate_rows, effective_specs, strict=True):
        if spec.missing_attribution:
            candidate["attribution"] = None
    object_rows = [
        _object(
            candidate,
            index=index,
            perceptual_hash=spec.perceptual_hash,
        )
        for index, (candidate, spec) in enumerate(
            zip(candidate_rows, effective_specs, strict=True)
        )
    ]
    deduplicated = deduplicate_reference_media(
        reference_media_objects_frame(object_rows),
        reference_media_candidates_frame(candidate_rows),
        reference_observations_frame(observations_rows),
        generated_at=NOW,
    )

    grouped_indexes: dict[tuple[str, str, str], list[int]] = {}
    for index, spec in enumerate(effective_specs):
        grouped_indexes.setdefault(
            (spec.taxon_key, spec.life_stage, spec.visual_domain),
            [],
        ).append(index)
    plan_rows: list[dict[str, object]] = []
    selection_rows: list[dict[str, object]] = []
    plan_fingerprints: dict[tuple[str, str, str], str] = {}
    for group_key, indexes in sorted(grouped_indexes.items()):
        taxon_key, life_stage, visual_domain = group_key
        fingerprint = _sha(f"plan:{taxon_key}:{life_stage}:{visual_domain}")
        plan_fingerprints[group_key] = fingerprint
        plan_id = make_acquisition_plan_id(
            target_accepted_taxon_key=TARGET,
            candidate_set_id=candidate_union_id,
            plan_configuration_fingerprint=fingerprint,
        )
        count = len(indexes)
        plan_rows.append(
            {
                "schema_version": REFERENCE_ACQUISITION_PLAN_SCHEMA_VERSION,
                "acquisition_plan_id": plan_id,
                "target_accepted_taxon_key": TARGET,
                "candidate_set_id": candidate_union_id,
                "candidate_accepted_taxon_key": taxon_key,
                "scientific_name": effective_specs[indexes[0]].scientific_name,
                "geo_cluster_id": "cluster-a",
                "life_stage": life_stage,
                "visual_domain": visual_domain,
                "source": "GBIF",
                "requested_count": count,
                "existing_support_count": 0,
                "available_candidate_count": count,
                "selected_candidate_count": count,
                "shortfall_count": 0,
                "fallback_level": 0,
                "selection_strategy": "test-selection-v1",
                "selection_seed": 42,
                "max_distance_km": float(max(indexes) + 1),
                "licence_policy_version": "reference-licences-v1",
                "source_snapshot_version": "gbif-2026-07-14",
                "plan_configuration_fingerprint": fingerprint,
                "created_at": NOW,
            }
        )
        for rank, index in enumerate(indexes, start=1):
            observation = observations_rows[index]
            candidate = candidate_rows[index]
            media_id = str(candidate["reference_media_id"])
            selection_rows.append(
                {
                    "schema_version": REFERENCE_ACQUISITION_SELECTIONS_SCHEMA_VERSION,
                    "reference_selection_id": make_reference_selection_id(
                        acquisition_plan_id=plan_id,
                        reference_media_id=media_id,
                        candidate_accepted_taxon_key=taxon_key,
                        geo_cluster_id="cluster-a",
                        life_stage=life_stage,
                        visual_domain=visual_domain,
                    ),
                    "acquisition_plan_id": plan_id,
                    "target_accepted_taxon_key": TARGET,
                    "candidate_set_id": candidate_union_id,
                    "source_candidate_set_id": CANDIDATE_SET_ID,
                    "candidate_accepted_taxon_key": taxon_key,
                    "scientific_name": effective_specs[index].scientific_name,
                    "geo_cluster_id": "cluster-a",
                    "life_stage": life_stage,
                    "visual_domain": visual_domain,
                    "reference_media_id": media_id,
                    "reference_observation_id": observation["reference_observation_id"],
                    "source": "GBIF",
                    "fallback_level": 0,
                    "selection_rank": rank,
                    "selection_round": "independent_observation",
                    "distance_to_cluster_medoid_km": float(index + 1),
                    "observer_id": observation["observer_id"],
                    "observed_date": observation["observed_at"].date(),
                    "locality": observation["locality"],
                    "background_group_id": effective_specs[index].background_group_id
                    or f"background-{index}",
                    "licence": candidate["licence"],
                    "source_snapshot_version": observation["source_snapshot_version"],
                    "selection_strategy": "test-selection-v1",
                    "selection_seed": 42,
                    "plan_configuration_fingerprint": fingerprint,
                    "selected_at": NOW,
                }
            )

    plan = reference_acquisition_plan_frame(plan_rows)
    selections = reference_acquisition_selections_frame(selection_rows)
    queue_result = build_reference_review_queue(
        selections,
        deduplicated.media_objects,
        deduplicated.media_candidates,
        deduplicated.observations,
        deduplicated.relationships,
        deduplication_report=deduplicated.report,
        reference_bank_version=BANK_VERSION,
        created_at=NOW + timedelta(hours=1),
        include_research_only=any(
            spec.licence_status == "research_only" for spec in effective_specs
        ),
    )
    spec_by_media = {
        str(candidate["reference_media_id"]): spec
        for candidate, spec in zip(candidate_rows, effective_specs, strict=True)
    }
    reviewed_ids = {
        media_id for media_id, spec in spec_by_media.items() if spec.reviewed
    }
    template = queue_result.decision_template.filter(
        pl.col("reference_media_id").is_in(reviewed_ids)
    )
    if template.is_empty():
        decisions = reference_review_decisions_frame([])
    else:
        ordered_specs = [
            spec_by_media[str(media_id)]
            for media_id in template["reference_media_id"].to_list()
        ]
        raw = template.with_columns(
            pl.lit(1, dtype=pl.UInt16).alias("review_round"),
            pl.lit("reviewer-a").alias("verified_by"),
            pl.lit(NOW + timedelta(hours=2))
            .cast(pl.Datetime("us", "UTC"))
            .alias("reviewed_at"),
            pl.lit(True, dtype=pl.Boolean).alias("target_identity_verified"),
            pl.lit("verified").alias("verification_status"),
            pl.Series(
                "life_stage",
                [spec.life_stage for spec in ordered_specs],
                dtype=pl.String,
            ),
            pl.Series(
                "visual_domain",
                [spec.visual_domain for spec in ordered_specs],
                dtype=pl.String,
            ),
            pl.lit("dorsal").alias("view"),
            pl.lit("high").alias("review_confidence"),
            pl.lit("Diagnostic markings are visible.").alias("review_notes"),
            pl.lit(None, dtype=pl.String).alias("exclusion_reason"),
        )
        imported = import_reference_review_decisions(
            raw,
            queue=queue_result.queue,
            queue_provenance=queue_result.provenance,
            existing_decisions=reference_review_decisions_frame([]),
            prior_report=queue_result.report,
            prior_report_sha256=_report_sha256(queue_result.report),
        )
        decisions = imported.decisions

    assigned_at = NOW + timedelta(hours=3)
    assignments = []
    for candidate, spec in zip(candidate_rows, effective_specs, strict=True):
        media_id = str(candidate["reference_media_id"])
        assignments.append(
            {
                "schema_version": REFERENCE_BANK_SPLIT_ASSIGNMENTS_SCHEMA_VERSION,
                "reference_media_id": media_id,
                "split_version": "reference-splits-v1",
                "support_split": spec.support_split,
                "included": spec.included,
                "exclusion_reason": None
                if spec.included
                else "excluded_from_support_bank",
                "assigned_by": "fixture-builder",
                "assigned_at": assigned_at,
                "assignment_fingerprint": make_reference_split_assignment_fingerprint(
                    reference_media_id=media_id,
                    split_version="reference-splits-v1",
                    support_split=spec.support_split,
                    included=spec.included,
                    exclusion_reason=None
                    if spec.included
                    else "excluded_from_support_bank",
                    assigned_by="fixture-builder",
                    assigned_at=assigned_at,
                ),
            }
        )
    return _Fixture(
        candidate_species=candidate_species,
        acquisition_plan=plan,
        selections=selections,
        observations=deduplicated.observations,
        candidates=deduplicated.media_candidates,
        objects=deduplicated.media_objects,
        relationships=deduplicated.relationships,
        deduplication_report=deduplicated.report,
        queue=queue_result.queue,
        provenance=queue_result.provenance,
        decisions=decisions,
        split_assignments=reference_bank_split_assignments_frame(assignments),
        plan_fingerprints=plan_fingerprints,
    )


def _policy(
    *,
    target_minimum: int = 1,
    competitor_minimum: int = 1,
    documented_shortfalls: tuple[DocumentedReferenceShortfall, ...] = (),
) -> ReferenceBankReadinessPolicy:
    return ReferenceBankReadinessPolicy(
        policy_version="reference-readiness-policy-v1",
        target_accepted_taxon_key=TARGET,
        requirements=(
            ReferenceBankRequirement(
                TARGET,
                "adult_field",
                target_minimum,
                geo_cluster_id="cluster-a",
            ),
            ReferenceBankRequirement(
                COMPETITOR,
                "adult_field",
                competitor_minimum,
            ),
        ),
        documented_shortfalls=documented_shortfalls,
    )


def _model() -> ReferenceModelInputIdentity:
    return ReferenceModelInputIdentity(
        model_name="bioclip-2.5-huge",
        model_version="2.5.0",
        model_revision="191d741545e4c741cdef4b22c6eb69c945c1e592",
        checkpoint_uri="s3://models/bioclip-2.5-huge/model.safetensors",
        checkpoint_sha256=_sha("bioclip-model"),
        open_clip_version="3.3.0",
        open_clip_config_sha256=_sha("openclip-config"),
        preprocessing_version="bioclip-preprocess-v2",
        preprocessing_contract_fingerprint=_sha("preprocessing-contract"),
        preprocessing_attestation_fingerprint=_sha("preprocessing-attestation"),
        input_contract_version="target-aware-reference-input-v1",
    )


def _build(
    fixture: _Fixture,
    *,
    policy: ReferenceBankReadinessPolicy | None = None,
    created_at: datetime = NOW + timedelta(hours=4),
):
    return build_reference_bank_readiness(
        candidate_species=fixture.candidate_species,
        acquisition_plan=fixture.acquisition_plan,
        acquisition_selections=fixture.selections,
        observations=fixture.observations,
        media_candidates=fixture.candidates,
        media_objects=fixture.objects,
        duplicate_relationships=fixture.relationships,
        deduplication_report=fixture.deduplication_report,
        review_queue=fixture.queue,
        queue_provenance=fixture.provenance,
        review_decisions=fixture.decisions,
        split_assignments=fixture.split_assignments,
        policy=policy or _policy(),
        registry_version=REGISTRY_VERSION,
        reference_bank_version=BANK_VERSION,
        model_identity=_model(),
        created_at=created_at,
    )


def _replace_observations(
    fixture: _Fixture,
    observations: pl.DataFrame,
) -> _Fixture:
    deduplicated = deduplicate_reference_media(
        fixture.objects,
        fixture.candidates,
        observations,
        generated_at=NOW,
    )
    return replace(
        fixture,
        observations=deduplicated.observations,
        candidates=deduplicated.media_candidates,
        objects=deduplicated.media_objects,
        relationships=deduplicated.relationships,
        deduplication_report=deduplicated.report,
    )


def _relocate_media_objects(fixture: _Fixture) -> _Fixture:
    relocated_objects = fixture.objects.with_columns(
        (
            pl.lit("s3://relocated/reference-media/")
            + pl.col("sha256").str.strip_prefix("sha256:")
            + pl.lit(".jpg")
        ).alias("source_object_uri")
    )
    deduplicated = deduplicate_reference_media(
        relocated_objects,
        fixture.candidates,
        fixture.observations,
        generated_at=NOW,
    )
    queue_result = build_reference_review_queue(
        fixture.selections,
        deduplicated.media_objects,
        deduplicated.media_candidates,
        deduplicated.observations,
        deduplicated.relationships,
        deduplication_report=deduplicated.report,
        reference_bank_version=BANK_VERSION,
        created_at=NOW + timedelta(hours=1),
    )
    return replace(
        fixture,
        observations=deduplicated.observations,
        candidates=deduplicated.media_candidates,
        objects=deduplicated.media_objects,
        relationships=deduplicated.relationships,
        deduplication_report=deduplicated.report,
        queue=queue_result.queue,
        provenance=queue_result.provenance,
    )


def _rebuild_fixture_with_operational_churn(fixture: _Fixture) -> _Fixture:
    rebuilt_at = NOW + timedelta(days=7)
    plan = fixture.acquisition_plan.with_columns(
        pl.lit(rebuilt_at).cast(pl.Datetime("us", "UTC")).alias("created_at")
    )
    selections = fixture.selections.with_columns(
        pl.lit(rebuilt_at).cast(pl.Datetime("us", "UTC")).alias("selected_at")
    )
    observations = fixture.observations.with_columns(
        (
            pl.lit("https://republished.example.test/records/")
            + pl.col("source_observation_id")
        ).alias("source_record_url"),
        pl.lit(rebuilt_at).cast(pl.Datetime("us", "UTC")).alias("retrieved_at"),
    )
    candidates = fixture.candidates.with_columns(
        (
            pl.lit("https://republished.example.test/media/")
            + pl.col("provider_media_id")
        ).alias("media_identifier"),
        pl.lit(rebuilt_at).cast(pl.Datetime("us", "UTC")).alias("retrieved_at"),
    )
    objects = fixture.objects.with_columns(
        (
            pl.lit("s3://republished-reference-media/")
            + pl.col("sha256").str.strip_prefix("sha256:")
            + pl.lit(".jpg")
        ).alias("source_object_uri"),
        pl.lit(rebuilt_at).cast(pl.Datetime("us", "UTC")).alias("downloaded_at"),
        pl.lit(9, dtype=pl.UInt16).alias("download_attempt_count"),
        pl.col("reference_media_id")
        .map_elements(
            lambda value: _sha(f"republished-object:{value}"),
            return_dtype=pl.String,
        )
        .alias("object_fingerprint"),
    )
    deduplicated = deduplicate_reference_media(
        objects,
        candidates,
        observations,
        generated_at=rebuilt_at,
    )
    report = json.loads(json.dumps(deduplicated.report))
    report["pid"] = 987654
    report["git_sha"] = "a" * 40
    report["generated_at"] = rebuilt_at.isoformat()
    report["inputs"]["artifact_uris"] = {
        "media_objects": "s3://republished/inputs/media-objects.parquet",
        "media_candidates": "s3://republished/inputs/media-candidates.parquet",
        "observations": "s3://republished/inputs/observations.parquet",
    }
    report["outputs"]["artifact_uris"] = {
        "media_objects": "s3://republished/outputs/media-objects.parquet",
        "relationships": "s3://republished/outputs/relationships.parquet",
        "report": "s3://republished/reports/report.json",
        "summary": "s3://republished/reports/summary.md",
    }
    queue_result = build_reference_review_queue(
        selections,
        deduplicated.media_objects,
        deduplicated.media_candidates,
        deduplicated.observations,
        deduplicated.relationships,
        deduplication_report=report,
        reference_bank_version=BANK_VERSION,
        created_at=rebuilt_at + timedelta(hours=1),
    )
    raw = queue_result.decision_template.with_columns(
        pl.lit(1, dtype=pl.UInt16).alias("review_round"),
        pl.lit("reviewer-after-republish").alias("verified_by"),
        pl.lit(rebuilt_at + timedelta(hours=2))
        .cast(pl.Datetime("us", "UTC"))
        .alias("reviewed_at"),
        pl.lit(True, dtype=pl.Boolean).alias("target_identity_verified"),
        pl.lit("verified").alias("verification_status"),
        pl.lit("adult").alias("life_stage"),
        pl.lit("live_field").alias("visual_domain"),
        pl.lit("dorsal").alias("view"),
        pl.lit("high").alias("review_confidence"),
        pl.lit("Diagnostic markings are visible.").alias("review_notes"),
        pl.lit(None, dtype=pl.String).alias("exclusion_reason"),
    )
    decisions = import_reference_review_decisions(
        raw,
        queue=queue_result.queue,
        queue_provenance=queue_result.provenance,
        existing_decisions=reference_review_decisions_frame([]),
        prior_report=queue_result.report,
        prior_report_sha256=_report_sha256(queue_result.report),
    ).decisions
    assigned_at = rebuilt_at + timedelta(hours=3)
    assignment_rows: list[dict[str, object]] = []
    for prior in fixture.split_assignments.iter_rows(named=True):
        row = dict(prior)
        row["assigned_by"] = "split-republisher"
        row["assigned_at"] = assigned_at
        row["assignment_fingerprint"] = make_reference_split_assignment_fingerprint(
            reference_media_id=str(row["reference_media_id"]),
            split_version=str(row["split_version"]),
            support_split=str(row["support_split"]),
            included=bool(row["included"]),
            exclusion_reason=row["exclusion_reason"],
            assigned_by=str(row["assigned_by"]),
            assigned_at=assigned_at,
        )
        assignment_rows.append(row)
    return replace(
        fixture,
        acquisition_plan=plan,
        selections=selections,
        observations=deduplicated.observations,
        candidates=deduplicated.media_candidates,
        objects=deduplicated.media_objects,
        relationships=deduplicated.relationships,
        deduplication_report=report,
        queue=queue_result.queue,
        provenance=queue_result.provenance,
        decisions=decisions,
        split_assignments=reference_bank_split_assignments_frame(assignment_rows),
    )


def test_ready_build_publish_and_strict_load(tmp_path: Path) -> None:
    result = _build(_make_fixture())

    assert result.readiness["status"] == "ready"
    assert result.readiness["permits_vision"] is True
    assert result.readiness["permits_reference_embedding"] is True
    assert result.readiness["permits_prototype_creation"] is True
    assert result.readiness["permits_provisional_scoring"] is True
    assert result.readiness["permits_calibrated_scoring"] is True
    assert result.readiness["permits_scientific_release"] is True
    assert result.readiness["requires_downstream_flickr_review"] is True
    assert result.readiness["reference_admission_mode"] == "human_verified_strict"
    assert result.readiness["provisional_support_count"] == 0
    assert result.readiness["human_verified_support_count"] == 2
    assert result.readiness["statistical_audit_required"] is False
    assert reference_readiness_allows_vision(result.readiness)
    assert result.support_manifest.height == 2
    assert result.support_manifest["support_eligible"].to_list() == [True, True]
    assert set(result.support_manifest["route"]) == {"adult_field"}
    assert set(result.support_manifest["reference_admission_mode"]) == {
        "human_verified_strict"
    }
    assert set(result.support_manifest["identity_evidence_basis"]) == {
        "human_verified"
    }
    assert result.support_manifest["provider_asserted_identity"].to_list() == [
        True,
        True,
    ]
    assert result.support_manifest["human_verified_identity"].to_list() == [
        True,
        True,
    ]
    assert result.support_manifest["provisional_support"].to_list() == [False, False]
    assert result.support_manifest["statistical_audit_required"].to_list() == [
        False,
        False,
    ]
    assert result.summary["provider_asserted_count"].to_list() == [1, 1, 1]
    assert result.summary["provider_asserted_eligible_count"].to_list() == [1, 1, 1]
    assert result.summary["human_verified_count"].to_list() == [1, 1, 1]
    assert result.summary["human_verified_eligible_count"].to_list() == [1, 1, 1]
    assert result.summary["provisional_support_count"].to_list() == [0, 0, 0]
    assert result.summary["strict_support_count"].to_list() == [1, 1, 1]
    assert result.summary["flagged_for_review_count"].to_list() == [0, 0, 0]
    assert result.summary["excluded_by_automated_qa_count"].to_list() == [0, 0, 0]
    assert result.summary["excluded_by_human_review_count"].to_list() == [0, 0, 0]
    assert [item["check_id"] for item in result.readiness["checks"]] == [
        "artifact_integrity",
        "target_adult_minimum",
        "competitor_minima",
        "geographic_cluster_coverage",
        "larval_route_separation",
        "pinned_specimen_separation",
        "support_admission_policy_satisfied",
        "provider_assertion_integrity",
        "automated_reference_qa_passed",
        "reference_routes_separated",
        "provisional_support_declared",
        "statistical_audit_plan_available",
        "human_rejections_respected",
        "strict_support_only",
        "duplicate_groups_resolved",
        "licences_accepted",
        "source_attribution_complete",
        "split_group_separation",
        "model_building_inputs_available",
    ]

    paths = publish_reference_bank_readiness(
        result,
        tmp_path / "readiness",
        run_id="readiness-test",
    )
    assert set(paths) == {"support_manifest", "summary", "readiness"}
    permit = load_reference_bank_readiness(
        tmp_path / "readiness",
        expected_registry_version=REGISTRY_VERSION,
        expected_target_accepted_taxon_key=TARGET,
        expected_model_name="bioclip-2.5-huge",
        expected_preprocessing_version="bioclip-preprocess-v2",
        expected_model_input_fingerprint=_model().fingerprint,
    )
    assert permit.status == "ready"
    assert permit.registry_version == REGISTRY_VERSION
    assert permit.target_accepted_taxon_key == TARGET
    assert permit.model_input_fingerprint == _model().fingerprint
    assert permit.model_revision == _model().model_revision
    assert permit.checkpoint_uri == _model().checkpoint_uri
    assert permit.open_clip_version == _model().open_clip_version
    assert permit.open_clip_config_sha256 == _model().open_clip_config_sha256
    assert (
        permit.preprocessing_contract_fingerprint
        == _model().preprocessing_contract_fingerprint
    )
    assert (
        permit.preprocessing_attestation_fingerprint
        == _model().preprocessing_attestation_fingerprint
    )
    assert permit.support_manifest_sha256.startswith("sha256:")
    assert permit.summary_sha256.startswith("sha256:")
    assert permit.readiness_sha256.startswith("sha256:")
    assert permit.permits_reference_embedding is True
    assert permit.permits_prototype_creation is True
    assert permit.permits_provisional_scoring is True
    assert permit.permits_calibrated_scoring is True
    assert permit.permits_scientific_release is True
    assert permit.requires_downstream_flickr_review is True
    assert permit.reference_admission_mode == "human_verified_strict"
    assert permit.provisional_support_count == 0
    assert permit.human_verified_support_count == 2
    assert permit.statistical_audit_required is False
    assert permit.candidate_set_fingerprints == tuple(
        result.readiness["candidate_set_fingerprints"]
    )
    assert len(permit.target_adult_requirements) == 1
    target_requirement = permit.target_adult_requirements[0]
    assert target_requirement.accepted_taxon_key == TARGET
    assert target_requirement.route == "adult_field"
    assert target_requirement.observed_count == target_requirement.minimum_count == 1
    assert (
        load_reference_bank_readiness(
            tmp_path / "readiness",
            expected_model_name="hf-hub:bioclip-2.5-huge",
        ).model_name
        == "bioclip-2.5-huge"
    )


def test_support_manifest_accepts_provider_assertion_only_as_provisional() -> None:
    strict = _build(_make_fixture()).support_manifest
    row = dict(strict.row(0, named=True))
    policy = default_reference_admission_policy()
    row.update(
        {
            "reference_admission_mode": policy.mode,
            "reference_admission_policy_version": policy.policy_version,
            "reference_admission_policy_fingerprint": policy.fingerprint,
            "identity_evidence_basis": "gbif_provider_asserted",
            "human_review_status": "not_requested",
            "human_verified_identity": False,
            "provisional_support": True,
            "statistical_audit_required": True,
            "admission_status": "admitted",
            "admission_reasons": ["automated_gbif_quality_gates_passed"],
            "reference_quality_flags": [],
            "route_evidence_basis": "yoloe",
            "review_status": "pending",
            "verification_status": "unreviewed",
            "target_identity_verified": False,
            "review_decision_ids": [],
            "reviewer_ids": [],
        }
    )
    row["support_row_fingerprint"] = readiness_module._support_row_fingerprint(  # noqa: SLF001 - fixture mirrors the persisted contract.
        row
    )
    provisional = pl.DataFrame(
        [row],
        schema=reference_support_manifest_schema(),
        orient="row",
        strict=True,
    )

    validate_reference_support_manifest(provisional)
    assert provisional["identity_evidence_basis"].item() == "gbif_provider_asserted"
    assert provisional["human_verified_identity"].item() is False
    assert provisional["provisional_support"].item() is True


def test_provider_assertion_cannot_be_encoded_as_human_verified() -> None:
    strict = _build(_make_fixture()).support_manifest
    row = dict(strict.row(0, named=True))
    policy = default_reference_admission_policy()
    row.update(
        {
            "reference_admission_mode": policy.mode,
            "reference_admission_policy_version": policy.policy_version,
            "reference_admission_policy_fingerprint": policy.fingerprint,
            "identity_evidence_basis": "gbif_provider_asserted",
            "provisional_support": True,
            "statistical_audit_required": True,
            "route_evidence_basis": "yoloe",
        }
    )
    row["support_row_fingerprint"] = readiness_module._support_row_fingerprint(  # noqa: SLF001 - adversarial fixture preserves the row identity.
        row
    )
    invalid = pl.DataFrame(
        [row],
        schema=reference_support_manifest_schema(),
        orient="row",
        strict=True,
    )

    with pytest.raises(ValueError, match="provider assertion is inconsistent"):
        validate_reference_support_manifest(invalid)


def test_ready_provisional_has_narrow_fail_closed_capabilities() -> None:
    payload = json.loads(json.dumps(_build(_make_fixture()).readiness))
    policy = default_reference_admission_policy()
    payload.update(
        {
            "status": "ready_provisional",
            "permits_vision": True,
            "permits_reference_embedding": True,
            "permits_provisional_scoring": True,
            "permits_calibrated_scoring": False,
            "permits_scientific_release": False,
            "reference_admission_mode": policy.mode,
            "admission_policy_fingerprint": policy.fingerprint,
            "provisional_support_count": 2,
            "human_verified_support_count": 0,
            "statistical_audit_required": True,
        }
    )
    payload["counts"]["provisional_support_count"] = 2
    payload["counts"]["human_verified_support_count"] = 0
    payload["checks"] = [
        check
        for check in payload["checks"]
        if check["check_id"] != "strict_support_only"
    ]
    payload["bank_fingerprint"] = readiness_module.canonical_semantic_fingerprint(
        {
            "schema_version": payload["schema_version"],
            "reference_bank_version": payload["reference_bank_version"],
            "registry_version": payload["registry_version"],
            "target_accepted_taxon_key": payload["target_accepted_taxon_key"],
            "policy_fingerprint": payload["policy_fingerprint"],
            "reference_admission_mode": payload["reference_admission_mode"],
            "admission_policy_fingerprint": payload[
                "admission_policy_fingerprint"
            ],
            "model_input_fingerprint": payload["model_input_fingerprint"],
            "candidate_set_ids": payload["candidate_set_ids"],
            "candidate_set_fingerprints": payload["candidate_set_fingerprints"],
            "inputs": payload["inputs"],
        }
    )
    for check in payload["checks"]:
        check["evidence"]["reference_bank_fingerprint"] = payload[
            "bank_fingerprint"
        ]

    readiness_module._validate_readiness_payload(payload, published=False)  # noqa: SLF001 - validates the persisted provisional contract directly.
    assert reference_readiness_allows_vision(payload)
    assert payload["permits_prototype_creation"] is True
    assert payload["requires_downstream_flickr_review"] is True
    assert payload["statistical_audit_required"] is True
    assert payload["permits_scientific_release"] is False

    payload["permits_calibrated_scoring"] = True
    with pytest.raises(ValueError, match="capabilities"):
        readiness_module._validate_readiness_payload(payload, published=False)  # noqa: SLF001 - adversarial persisted-contract validation.


def test_legacy_v2_support_requires_explicit_strict_migration() -> None:
    current = _build(_make_fixture()).support_manifest
    legacy_rows = []
    for current_row in current.iter_rows(named=True):
        row = {
            field: value
            for field, value in current_row.items()
            if field in legacy_reference_support_manifest_v2_schema()
        }
        row["schema_version"] = LEGACY_REFERENCE_SUPPORT_MANIFEST_SCHEMA_VERSION
        row["support_row_fingerprint"] = (
            readiness_module._legacy_support_row_fingerprint_v2(row)  # noqa: SLF001 - fixture recreates the exact persisted v2 identity.
        )
        legacy_rows.append(row)
    legacy = pl.DataFrame(
        legacy_rows,
        schema=legacy_reference_support_manifest_v2_schema(),
        orient="row",
        strict=True,
    ).sort(readiness_module._SUPPORT_SORT)  # noqa: SLF001 - persisted contract sort.

    with pytest.raises(ValueError, match="physical schema"):
        validate_reference_support_manifest(legacy)

    migration = migrate_strict_reference_support_manifest_v2(legacy)

    validate_reference_support_manifest(migration.manifest)
    assert set(migration.manifest["reference_admission_mode"]) == {
        "human_verified_strict"
    }
    assert migration.manifest["human_verified_identity"].all()
    assert not migration.manifest["provisional_support"].any()
    assert not migration.manifest["provider_asserted_identity"].any()
    assert migration.report["source_schema_version"] == (
        LEGACY_REFERENCE_SUPPORT_MANIFEST_SCHEMA_VERSION
    )
    assert migration.report["migration_mode"] == "explicit_human_verified_strict"
    assert migration.report["provider_assertion_backfilled"] is False
    assert migration.report["requires_downstream_rebuild"] == [
        "readiness",
        "reference_embeddings",
        "reference_prototypes",
        "models",
        "scores",
    ]


def test_admission_mode_change_invalidates_support_and_readiness_identity() -> None:
    result = _build(_make_fixture())
    strict_fingerprint = reference_support_manifest_fingerprint(
        result.support_manifest
    )
    adaptive = default_reference_admission_policy()
    changed_rows = []
    for source_row in result.support_manifest.iter_rows(named=True):
        row = dict(source_row)
        row["reference_admission_mode"] = adaptive.mode
        row["reference_admission_policy_version"] = adaptive.policy_version
        row["reference_admission_policy_fingerprint"] = adaptive.fingerprint
        row["support_row_fingerprint"] = readiness_module._support_row_fingerprint(row)  # noqa: SLF001 - recomputes a deliberate policy-identity change.
        changed_rows.append(row)
    changed = pl.DataFrame(
        changed_rows,
        schema=reference_support_manifest_schema(),
        orient="row",
        strict=True,
    ).sort(readiness_module._SUPPORT_SORT)  # noqa: SLF001 - persisted contract sort.

    validate_reference_support_manifest(changed)
    assert reference_support_manifest_fingerprint(changed) != strict_fingerprint
    with pytest.raises(ValueError, match="support manifest fingerprint mismatch"):
        validate_reference_bank_readiness(replace(result, support_manifest=changed))


def test_model_and_support_semantic_fingerprints_ignore_object_relocation() -> None:
    model = _model()
    assert replace(
        model,
        checkpoint_uri="s3://relocated/models/model.safetensors",
    ).fingerprint == model.fingerprint

    result = _build(_make_fixture())
    original = result.support_manifest
    relocated = original.with_columns(
        (
            pl.lit("s3://relocated/reference-media/")
            + pl.col("reference_media_id")
            + pl.lit(".jpg")
        ).alias("source_object_uri"),
        (
            pl.lit("https://relocated.example.test/records/")
            + pl.col("reference_media_id")
        ).alias("source_record_url"),
        pl.lit("https://relocated.example.test/licences/cc-by-4.0").alias(
            "licence_uri"
        ),
        pl.lit(_sha("relocated-object")).alias("object_fingerprint"),
        pl.lit(_sha("relocated-split-assignment")).alias(
            "split_assignment_fingerprint"
        ),
        (pl.lit("review:relocated:") + pl.col("reference_media_id")).alias(
            "review_request_id"
        ),
        pl.lit(["decision:relocated"], dtype=pl.List(pl.String)).alias(
            "review_decision_ids"
        ),
        pl.lit(["reviewer:relocated"], dtype=pl.List(pl.String)).alias(
            "reviewer_ids"
        ),
    )

    validate_reference_support_manifest(relocated)
    assert reference_support_manifest_fingerprint(relocated) == (
        reference_support_manifest_fingerprint(original)
    )

    changed_rows: list[dict[str, object]] = []
    for source_row in original.iter_rows(named=True):
        row = dict(source_row)
        row["image_sha256"] = _sha(
            "scientifically-different-image:" + str(row["reference_media_id"])
        )
        row["support_row_fingerprint"] = readiness_module._support_row_fingerprint(  # noqa: SLF001 - verifies that direct image evidence remains semantic.
            row
        )
        changed_rows.append(row)
    scientifically_changed = pl.DataFrame(
        changed_rows,
        schema=reference_support_manifest_schema(),
        orient="row",
        strict=True,
    )
    validate_reference_support_manifest(scientifically_changed)
    assert reference_support_manifest_fingerprint(scientifically_changed) != (
        reference_support_manifest_fingerprint(original)
    )


def test_rebuilt_support_bank_ignores_media_object_relocation() -> None:
    fixture = _make_fixture()
    relocated_fixture = _relocate_media_objects(fixture)

    assert fixture.queue["review_request_id"].to_list() == relocated_fixture.queue[
        "review_request_id"
    ].to_list()
    assert fixture.objects["source_object_uri"].to_list() != relocated_fixture.objects[
        "source_object_uri"
    ].to_list()

    original = _build(fixture)
    relocated = _build(relocated_fixture)

    assert original.readiness["inputs"] == relocated.readiness["inputs"]
    assert original.readiness["bank_fingerprint"] == relocated.readiness[
        "bank_fingerprint"
    ]
    assert original.readiness["support_manifest_fingerprint"] == relocated.readiness[
        "support_manifest_fingerprint"
    ]
    assert original.summary.equals(relocated.summary)
    assert original.support_manifest.drop("source_object_uri").equals(
        relocated.support_manifest.drop("source_object_uri")
    )


def test_regenerated_upstream_audit_provenance_does_not_change_bank_identity() -> None:
    fixture = _make_fixture()
    regenerated_fixture = _rebuild_fixture_with_operational_churn(fixture)

    assert fixture.objects["object_fingerprint"].to_list() != (
        regenerated_fixture.objects["object_fingerprint"].to_list()
    )
    assert fixture.queue["review_request_id"].to_list() != (
        regenerated_fixture.queue["review_request_id"].to_list()
    )
    assert fixture.decisions["review_decision_id"].to_list() != (
        regenerated_fixture.decisions["review_decision_id"].to_list()
    )
    assert fixture.split_assignments["assignment_fingerprint"].to_list() != (
        regenerated_fixture.split_assignments["assignment_fingerprint"].to_list()
    )
    assert fixture.observations["source_record_hash"].to_list() == (
        regenerated_fixture.observations["source_record_hash"].to_list()
    )
    assert fixture.objects["sha256"].to_list() == (
        regenerated_fixture.objects["sha256"].to_list()
    )

    original = _build(fixture)
    regenerated = _build(
        regenerated_fixture,
        created_at=NOW + timedelta(days=8),
    )

    assert original.readiness["inputs"] == regenerated.readiness["inputs"]
    assert original.readiness["bank_fingerprint"] == regenerated.readiness[
        "bank_fingerprint"
    ]
    assert original.readiness["support_manifest_fingerprint"] == (
        regenerated.readiness["support_manifest_fingerprint"]
    )
    assert original.readiness["split_assignments_fingerprint"] == (
        regenerated.readiness["split_assignments_fingerprint"]
    )
    assert original.support_manifest["support_row_fingerprint"].to_list() == (
        regenerated.support_manifest["support_row_fingerprint"].to_list()
    )
    assert original.summary.equals(regenerated.summary)
    assert original.support_manifest["review_request_id"].to_list() != (
        regenerated.support_manifest["review_request_id"].to_list()
    )
    assert original.support_manifest["reviewer_ids"].to_list() != (
        regenerated.support_manifest["reviewer_ids"].to_list()
    )


def test_policy_requires_explicit_coverage_for_every_candidate_cluster() -> None:
    policy = ReferenceBankReadinessPolicy(
        policy_version="reference-readiness-policy-v1",
        target_accepted_taxon_key=TARGET,
        requirements=(
            ReferenceBankRequirement(TARGET, "adult_field", 1),
            ReferenceBankRequirement(COMPETITOR, "adult_field", 1),
        ),
    )

    with pytest.raises(ValueError, match="cluster-scoped geographic requirements"):
        _build(_make_fixture(), policy=policy)


def test_observation_registry_version_cannot_be_relabelled_by_readiness() -> None:
    fixture = _make_fixture()
    observations = fixture.observations.with_columns(
        pl.lit("stale-registry").alias("registry_version")
    )
    fixture = _replace_observations(fixture, observations)

    with pytest.raises(ValueError, match="observation registry version mismatch"):
        _build(fixture)


def test_observation_name_must_match_candidate_union_identity() -> None:
    fixture = _make_fixture()
    target_observation_id = fixture.observations.filter(
        pl.col("accepted_taxon_key") == TARGET
    )["reference_observation_id"].item()
    observations = fixture.observations.with_columns(
        pl.when(pl.col("reference_observation_id") == target_observation_id)
        .then(pl.lit("Completely wrong species"))
        .otherwise(pl.col("reconciled_scientific_name"))
        .alias("reconciled_scientific_name")
    )
    fixture = _replace_observations(fixture, observations)

    with pytest.raises(ValueError, match="scientific name conflicts"):
        _build(fixture)


def test_review_decision_cannot_authorize_replaced_object_bytes() -> None:
    fixture = _make_fixture()
    target_observation_id = fixture.observations.filter(
        pl.col("accepted_taxon_key") == TARGET
    )["reference_observation_id"].item()
    target_media_id = fixture.candidates.filter(
        pl.col("reference_observation_id") == target_observation_id
    )["reference_media_id"].item()
    replacement_sha = _sha("replacement-image-bytes")
    objects = fixture.objects.with_columns(
        pl.when(pl.col("reference_media_id") == target_media_id)
        .then(pl.lit(replacement_sha))
        .otherwise(pl.col("sha256"))
        .alias("sha256"),
        pl.when(pl.col("reference_media_id") == target_media_id)
        .then(
            pl.lit(
                "s3://reference-bank/"
                + replacement_sha.removeprefix("sha256:")
                + ".ppm"
            )
        )
        .otherwise(pl.col("source_object_uri"))
        .alias("source_object_uri"),
    )
    deduplicated = deduplicate_reference_media(
        objects,
        fixture.candidates,
        fixture.observations,
        generated_at=NOW,
    )
    fixture = replace(
        fixture,
        objects=deduplicated.media_objects,
        candidates=deduplicated.media_candidates,
        observations=deduplicated.observations,
        relationships=deduplicated.relationships,
        deduplication_report=deduplicated.report,
    )

    with pytest.raises(ValueError, match="review queue provenance conflicts"):
        _build(fixture)


def test_build_is_semantically_deterministic_and_sorted() -> None:
    fixture = _make_fixture()
    first = _build(fixture, created_at=NOW + timedelta(hours=4))
    second = _build(fixture, created_at=NOW + timedelta(days=1))

    assert first.support_manifest.equals(second.support_manifest)
    assert first.summary.equals(second.summary)
    for field in (
        "bank_fingerprint",
        "support_manifest_fingerprint",
        "summary_fingerprint",
        "split_assignments_fingerprint",
        "model_input_fingerprint",
    ):
        assert first.readiness[field] == second.readiness[field]
    assert first.readiness["created_at"] != second.readiness["created_at"]
    assert first.support_manifest.equals(
        first.support_manifest.sort(
            [
                "accepted_taxon_key",
                "geo_cluster_id",
                "route",
                "support_split",
                "reference_media_id",
            ]
        )
    )


def test_blocked_missing_target_support_is_not_loadable(tmp_path: Path) -> None:
    result = _build(_make_fixture(), policy=_policy(target_minimum=2))

    assert result.readiness["status"] == "blocked_missing_target_support"
    assert not reference_readiness_allows_vision(result.readiness)
    assert result.readiness["counts"]["target_minimum_shortfall_count"] == 1

    publish_reference_bank_readiness(result, tmp_path / "blocked")
    with pytest.raises(ValueError, match="does not permit vision"):
        load_reference_bank_readiness(tmp_path / "blocked")


def test_blocked_licence_takes_precedence_over_target_shortfall() -> None:
    fixture = _make_fixture(
        (
            _Spec(
                TARGET,
                "Papilio demoleus",
                licence_status="research_only",
            ),
            _Spec(COMPETITOR, "Papilio polytes"),
        )
    )

    result = _build(fixture)

    assert result.readiness["status"] == "blocked_licence"
    assert result.readiness["counts"]["licence_blocker_count"] == 1
    assert result.readiness["counts"]["target_minimum_shortfall_count"] == 1
    assert result.readiness["counts"]["unverified_support_count"] == 0
    licence_check = next(
        item
        for item in result.readiness["checks"]
        if item["check_id"] == "licences_accepted"
    )
    assert licence_check["status"] == "failed"
    verified_check = next(
        item
        for item in result.readiness["checks"]
        if item["check_id"] == "strict_support_only"
    )
    assert verified_check["status"] == "passed"


def test_pending_included_media_awaits_manual_review() -> None:
    fixture = _make_fixture(
        (
            _Spec(TARGET, "Papilio demoleus", reviewed=False),
            _Spec(COMPETITOR, "Papilio polytes"),
        )
    )

    result = _build(fixture)

    assert result.readiness["status"] == "awaiting_manual_review"
    assert result.readiness["counts"]["pending_review_count"] == 1
    assert result.readiness["counts"]["unverified_support_count"] == 1
    check = next(
        item
        for item in result.readiness["checks"]
        if item["check_id"] == "strict_support_only"
    )
    assert check["status"] == "pending"


def test_pending_excluded_media_does_not_block_ready_support() -> None:
    fixture = _make_fixture(
        (
            _Spec(TARGET, "Papilio demoleus"),
            _Spec(COMPETITOR, "Papilio polytes"),
            _Spec(
                COMPETITOR,
                "Papilio polytes",
                reviewed=False,
                included=False,
            ),
        )
    )

    result = _build(fixture)

    assert result.readiness["status"] == "ready"
    assert result.readiness["counts"]["pending_review_count"] == 0
    assert result.readiness["counts"]["pending_target_review_count"] == 0


def test_unrelated_pending_media_does_not_mask_missing_target_support() -> None:
    fixture = _make_fixture(
        (
            _Spec(TARGET, "Papilio demoleus"),
            _Spec(COMPETITOR, "Papilio polytes"),
            _Spec(
                COMPETITOR,
                "Papilio polytes",
                reviewed=False,
                included=False,
            ),
        )
    )

    result = _build(fixture, policy=_policy(target_minimum=2))

    assert result.readiness["status"] == "blocked_missing_target_support"
    assert result.readiness["counts"]["pending_review_count"] == 0
    assert result.readiness["counts"]["pending_target_review_count"] == 0


def test_unresolved_perceptual_duplicate_awaits_manual_review() -> None:
    fixture = _make_fixture(
        (
            _Spec(
                TARGET,
                "Papilio demoleus",
                perceptual_hash="dhash128-v1:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            ),
            _Spec(
                TARGET,
                "Papilio demoleus",
                perceptual_hash="dhash128-v1:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaab",
            ),
            _Spec(
                COMPETITOR,
                "Papilio polytes",
                perceptual_hash="dhash128-v1:55555555555555555555555555555555",
            ),
        )
    )
    assert fixture.relationships.height == 1
    assert fixture.relationships["resolution_status"].item() == "review_required"

    result = _build(fixture)

    assert result.readiness["status"] == "awaiting_manual_review"
    assert result.readiness["counts"]["unresolved_duplicate_count"] == 1
    duplicate_check = next(
        item
        for item in result.readiness["checks"]
        if item["check_id"] == "duplicate_groups_resolved"
    )
    assert duplicate_check["status"] == "pending"


def test_split_leakage_is_invalid_and_detects_unverified_assignments() -> None:
    fixture = _make_fixture(
        (
            _Spec(
                TARGET,
                "Papilio demoleus",
                support_split="support_train",
                observer_id="shared-observer",
            ),
            _Spec(
                COMPETITOR,
                "Papilio polytes",
                support_split="final_test",
                reviewed=False,
                observer_id="shared-observer",
            ),
        )
    )

    result = _build(fixture)

    assert result.readiness["status"] == "invalid"
    assert result.readiness["counts"]["split_leakage_count"] >= 1
    assert result.readiness["counts"]["unverified_support_count"] == 1
    leakage_check = next(
        item
        for item in result.readiness["checks"]
        if item["check_id"] == "split_group_separation"
    )
    assert leakage_check["status"] == "failed"
    assert any(
        item["group_type"] == "observer_id"
        for item in leakage_check["evidence"]["leakage"]
    )


def test_split_leakage_detects_shared_acquisition_background_groups() -> None:
    fixture = _make_fixture(
        (
            _Spec(
                TARGET,
                "Papilio demoleus",
                background_group_id="shared-background",
            ),
            _Spec(
                COMPETITOR,
                "Papilio polytes",
                support_split="final_test",
                background_group_id="shared-background",
            ),
        )
    )

    result = _build(fixture)

    assert result.readiness["status"] == "invalid"
    leakage_check = next(
        item
        for item in result.readiness["checks"]
        if item["check_id"] == "split_group_separation"
    )
    assert {
        (item["group_type"], item["group_value"])
        for item in leakage_check["evidence"]["leakage"]
    } >= {("background_group_id", "shared-background")}


def test_included_verified_prohibited_domain_is_invalid() -> None:
    fixture = _make_fixture(
        (
            _Spec(
                TARGET,
                "Papilio demoleus",
                visual_domain="artwork",
            ),
            _Spec(COMPETITOR, "Papilio polytes"),
        )
    )

    result = _build(fixture)

    assert result.readiness["status"] == "invalid"
    integrity = next(
        item
        for item in result.readiness["checks"]
        if item["check_id"] == "artifact_integrity"
    )
    assert integrity["status"] == "failed"
    assert "no supported reference route" in integrity["evidence"]["issues"][0]


def test_included_media_with_incomplete_attribution_is_invalid() -> None:
    fixture = _make_fixture(
        (
            _Spec(
                TARGET,
                "Papilio demoleus",
                missing_attribution=True,
            ),
            _Spec(COMPETITOR, "Papilio polytes"),
        )
    )

    result = _build(fixture)

    assert result.readiness["status"] == "invalid"
    assert result.readiness["counts"]["attribution_blocker_count"] == 1
    check = next(
        item
        for item in result.readiness["checks"]
        if item["check_id"] == "source_attribution_complete"
    )
    assert check["status"] == "failed"
    assert len(check["evidence"]["media_ids"]) == 1


def test_exact_documented_competitor_shortfall_permits_vision() -> None:
    fixture = _make_fixture()
    approval = DocumentedReferenceShortfall(
        shortfall_id="approved-shortfall-1",
        accepted_taxon_key=COMPETITOR,
        route="adult_field",
        approved_minimum_count=1,
        reason="Only one independent licensed observation is currently available.",
        approved_by="taxonomy-lead",
        approved_at=NOW,
        plan_configuration_fingerprint=fixture.plan_fingerprints[
            (COMPETITOR, "adult", "live_field")
        ],
    )

    result = _build(
        fixture,
        policy=_policy(
            competitor_minimum=2,
            documented_shortfalls=(approval,),
        ),
    )

    assert result.readiness["status"] == "ready_with_documented_shortfalls"
    assert reference_readiness_allows_vision(result.readiness)
    assert result.readiness["counts"]["documented_shortfall_count"] == 1
    assert result.readiness["documented_shortfalls"][0]["shortfall_id"] == (
        "approved-shortfall-1"
    )


def test_stale_documented_shortfall_makes_readiness_invalid() -> None:
    fixture = _make_fixture()
    approval = DocumentedReferenceShortfall(
        shortfall_id="stale-shortfall",
        accepted_taxon_key=COMPETITOR,
        route="adult_field",
        approved_minimum_count=1,
        reason="Approval was issued against an older plan.",
        approved_by="taxonomy-lead",
        approved_at=NOW,
        plan_configuration_fingerprint=_sha("obsolete-plan"),
    )

    result = _build(
        fixture,
        policy=_policy(
            competitor_minimum=2,
            documented_shortfalls=(approval,),
        ),
    )

    assert result.readiness["status"] == "invalid"
    assert result.readiness["counts"]["structural_issue_count"] == 1
    integrity = result.readiness["checks"][0]
    assert integrity["status"] == "failed"
    assert "stale" in integrity["evidence"]["issues"][0]


def test_target_shortfall_approval_is_rejected_at_policy_boundary() -> None:
    target_approval = DocumentedReferenceShortfall(
        shortfall_id="invalid-target-shortfall",
        accepted_taxon_key=TARGET,
        route="adult_field",
        approved_minimum_count=0,
        reason="Target minimum cannot be waived.",
        approved_by="taxonomy-lead",
        approved_at=NOW,
        plan_configuration_fingerprint=_sha("target-plan"),
    )

    with pytest.raises(ValueError, match="target support shortfalls"):
        ReferenceBankReadinessPolicy(
            policy_version="reference-readiness-policy-v1",
            target_accepted_taxon_key=TARGET,
            requirements=(ReferenceBankRequirement(TARGET, "adult_field", 1),),
            documented_shortfalls=(target_approval,),
        )


def test_larval_references_are_routed_separately_from_adult_support() -> None:
    fixture = _make_fixture(
        (
            _Spec(TARGET, "Papilio demoleus"),
            _Spec(COMPETITOR, "Papilio polytes"),
            _Spec(
                TARGET,
                "Papilio demoleus",
                life_stage="larva",
            ),
        )
    )

    result = _build(fixture)

    assert result.readiness["status"] == "ready"
    by_stage = {
        str(row["life_stage"]): str(row["route"])
        for row in result.support_manifest.iter_rows(named=True)
        if row["accepted_taxon_key"] == TARGET
    }
    assert by_stage == {"adult": "adult_field", "larva": "larval"}
    check = next(
        item
        for item in result.readiness["checks"]
        if item["check_id"] == "larval_route_separation"
    )
    assert check["status"] == "passed"


def test_pinned_specimens_are_routed_separately_from_live_support() -> None:
    fixture = _make_fixture(
        (
            _Spec(TARGET, "Papilio demoleus"),
            _Spec(COMPETITOR, "Papilio polytes"),
            _Spec(
                TARGET,
                "Papilio demoleus",
                life_stage="unknown",
                visual_domain="pinned_specimen",
            ),
        )
    )

    result = _build(fixture)

    assert result.readiness["status"] == "ready"
    target_routes = set(
        result.support_manifest.filter(pl.col("accepted_taxon_key") == TARGET)["route"]
    )
    assert target_routes == {"adult_field", "pinned_specimen"}
    check = next(
        item
        for item in result.readiness["checks"]
        if item["check_id"] == "pinned_specimen_separation"
    )
    assert check["status"] == "passed"


def test_empty_frames_keep_exact_physical_schemas() -> None:
    assignments = reference_bank_split_assignments_frame([])
    support = pl.DataFrame(schema=reference_support_manifest_schema())
    summary = pl.DataFrame(schema=reference_bank_summary_schema())

    validate_reference_bank_split_assignments(assignments)
    validate_reference_support_manifest(support)
    validate_reference_bank_summary(summary)
    assert assignments.schema == reference_bank_split_assignments_schema()
    assert support.schema == reference_support_manifest_schema()
    assert summary.schema == reference_bank_summary_schema()

    with pytest.raises(ValueError, match="physical schema"):
        validate_reference_bank_split_assignments(
            assignments.with_columns(pl.lit("unexpected").alias("extra"))
        )


@pytest.mark.parametrize(
    ("loader_kwargs", "message"),
    [
        ({"expected_registry_version": "registry-other"}, "registry_version"),
        (
            {"expected_target_accepted_taxon_key": "gbif:other"},
            "target_accepted_taxon_key",
        ),
        ({"expected_model_name": "other-model"}, "model_name"),
        (
            {"expected_preprocessing_version": "other-preprocessing"},
            "preprocessing_version",
        ),
        (
            {"expected_model_input_fingerprint": _sha("other-model-input")},
            "model_input_fingerprint",
        ),
    ],
)
def test_loader_rejects_runtime_identity_mismatches(
    tmp_path: Path,
    loader_kwargs: dict[str, str],
    message: str,
) -> None:
    publish_reference_bank_readiness(_build(_make_fixture()), tmp_path / "ready")

    with pytest.raises(ValueError, match=message):
        load_reference_bank_readiness(tmp_path / "ready", **loader_kwargs)


def test_loader_returns_an_immutable_identity_bound_permit(tmp_path: Path) -> None:
    publish_reference_bank_readiness(_build(_make_fixture()), tmp_path / "ready")
    permit = load_reference_bank_readiness(tmp_path / "ready")

    with pytest.raises(FrozenInstanceError):
        permit.status = "invalid"  # type: ignore[misc]


def test_loader_rejects_readiness_manifest_outside_trusted_digest_pin(
    tmp_path: Path,
) -> None:
    publish_reference_bank_readiness(_build(_make_fixture()), tmp_path / "ready")

    with pytest.raises(ValueError, match="does not match its pin"):
        load_reference_bank_readiness(
            tmp_path / "ready",
            expected_readiness_sha256=_sha("different-readiness-manifest"),
        )


def test_loader_rejects_parquet_byte_tampering(tmp_path: Path) -> None:
    paths = publish_reference_bank_readiness(
        _build(_make_fixture()),
        tmp_path / "ready",
    )
    with paths["support_manifest"].open("ab") as stream:
        stream.write(b"tampered")

    with pytest.raises(ValueError, match="checksum mismatch"):
        load_reference_bank_readiness(tmp_path / "ready")


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda payload: payload.__setitem__(
                "target_accepted_taxon_key", "gbif:tampered"
            ),
            "bank fingerprint",
        ),
        (
            lambda payload: payload["model_input_identity"].__setitem__(
                "model_name", "tampered-model"
            ),
            "model input fingerprint",
        ),
        (
            lambda payload: payload.__setitem__("status", "invalid"),
            "permit flag",
        ),
    ],
)
def test_loader_rejects_semantically_tampered_readiness_json(
    tmp_path: Path,
    mutate,
    message: str,
) -> None:
    paths = publish_reference_bank_readiness(
        _build(_make_fixture()),
        tmp_path / "ready",
    )
    payload = json.loads(paths["readiness"].read_text(encoding="utf-8"))
    mutate(payload)
    paths["readiness"].write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=message):
        load_reference_bank_readiness(tmp_path / "ready")


def test_status_only_json_cannot_act_as_a_readiness_permit(tmp_path: Path) -> None:
    source = publish_reference_bank_readiness(
        _build(_make_fixture()),
        tmp_path / "source",
    )
    status_only = tmp_path / "status-only"
    status_only.mkdir()
    (status_only / REFERENCE_BANK_READINESS_FILE).write_bytes(
        source["readiness"].read_bytes()
    )

    with pytest.raises(ValueError, match="artifact is missing"):
        load_reference_bank_readiness(status_only)


def test_publication_is_create_only_under_concurrency(tmp_path: Path) -> None:
    result = _build(_make_fixture())
    destination = tmp_path / "ready"

    def publish(index: int):
        try:
            return publish_reference_bank_readiness(
                result,
                destination,
                run_id=f"publisher-{index}",
            )
        except Exception as exc:  # noqa: BLE001 - compare both race outcomes.
            return exc

    with ThreadPoolExecutor(max_workers=2) as executor:
        outcomes = list(executor.map(publish, range(2)))

    successes = [item for item in outcomes if isinstance(item, dict)]
    failures = [item for item in outcomes if isinstance(item, Exception)]
    assert len(successes) == 1
    assert len(failures) == 1
    assert isinstance(failures[0], FileExistsError)
    assert load_reference_bank_readiness(destination).status == "ready"
    assert not list(tmp_path.glob(".ready.*.tmp"))
    failed_audits = list(tmp_path.glob(".ready.*.failed.json"))
    assert len(failed_audits) == 1
    failed = json.loads(failed_audits[0].read_text(encoding="utf-8"))
    assert failed["status"] == "failed"
    assert failed["artifact"] == "not_committed"


def test_failed_publication_leaves_audit_and_no_partial_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "ready"

    def fail_write(*args: object, **kwargs: object) -> None:
        raise OSError("simulated storage failure")

    monkeypatch.setattr(readiness_module, "write_parquet", fail_write)
    with pytest.raises(OSError, match="simulated storage failure"):
        publish_reference_bank_readiness(
            _build(_make_fixture()),
            destination,
            run_id="failed-publication",
        )

    assert not destination.exists()
    assert not list(tmp_path.glob(".ready.*.tmp"))
    audits = list(tmp_path.glob(".ready.*.failed.json"))
    assert len(audits) == 1
    report = json.loads(audits[0].read_text(encoding="utf-8"))
    assert report["status"] == "failed"
    assert report["run_id"] == "failed-publication"
    assert report["error_type"] == "OSError"
    assert report["artifact"] == "not_committed"


def test_published_files_use_fixed_names_and_readiness_json_is_last(
    tmp_path: Path,
) -> None:
    paths = publish_reference_bank_readiness(
        _build(_make_fixture()),
        tmp_path / "ready",
    )

    assert paths["support_manifest"].name == REFERENCE_SUPPORT_MANIFEST_FILE
    assert paths["summary"].name == REFERENCE_BANK_SUMMARY_FILE
    assert paths["readiness"].name == REFERENCE_BANK_READINESS_FILE
    assert (
        paths["readiness"].stat().st_mtime_ns
        >= paths["support_manifest"].stat().st_mtime_ns
    )
    assert paths["readiness"].stat().st_mtime_ns >= paths["summary"].stat().st_mtime_ns
