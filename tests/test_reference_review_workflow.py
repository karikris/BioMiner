from __future__ import annotations

from dataclasses import dataclass, replace
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
import hashlib
import json
from pathlib import Path
import socket
from types import SimpleNamespace

import polars as pl
import pytest

from biominer.cli import build_parser, run
from biominer.references.deduplication import deduplicate_reference_media
from biominer.references.review import (
    REFERENCE_REVIEW_DECISION_IMPORT_SCHEMA_VERSION,
    _frame_fingerprint,
    _queue_row_fingerprint,
    _review_markdown,
    advance_reference_review_history_head,
    build_reference_review_queue,
    import_reference_review_decisions,
    initialize_reference_review_history_head,
    reference_review_decision_import_schema,
    resolve_reference_review_statuses,
    select_verified_reference_media,
    validate_reference_review_history_head,
    validate_reference_review_packet_artifact,
    write_reference_review_export,
    write_reference_review_import,
)
from biominer.references.schemas import (
    REFERENCE_ACQUISITION_SELECTIONS_SCHEMA_VERSION,
    REFERENCE_MEDIA_CANDIDATES_SCHEMA_VERSION,
    REFERENCE_MEDIA_OBJECTS_SCHEMA_VERSION,
    REFERENCE_OBSERVATIONS_SCHEMA_VERSION,
    make_reference_media_id,
    make_reference_observation_id,
    make_reference_selection_id,
    reference_acquisition_selections_frame,
    reference_media_candidates_frame,
    reference_media_duplicate_relationships_frame,
    reference_media_objects_frame,
    reference_observations_frame,
    reference_review_decisions_frame,
    reference_review_queue_frame,
)


NOW = datetime(2026, 7, 14, 6, 0, tzinfo=UTC)


@dataclass(frozen=True)
class _Inputs:
    selections: pl.DataFrame
    objects: pl.DataFrame
    candidates: pl.DataFrame
    observations: pl.DataFrame
    relationships: pl.DataFrame
    deduplication_report: dict[str, object]


def _sha(seed: str) -> str:
    return "sha256:" + hashlib.sha256(seed.encode()).hexdigest()


def _observation(source: str, source_id: str) -> dict[str, object]:
    observation_id = make_reference_observation_id(source, source_id)
    return {
        "schema_version": REFERENCE_OBSERVATIONS_SCHEMA_VERSION,
        "reference_observation_id": observation_id,
        "source": source,
        "source_observation_id": source_id,
        "source_taxon_id": "1938069",
        "supplied_scientific_name": "Papilio demoleus",
        "accepted_taxon_key": "gbif:1938069",
        "reconciled_scientific_name": "Papilio demoleus",
        "registry_version": "butterflies-v2-20260714",
        "taxon_reconciliation_status": "accepted_key_exact",
        "identification_quality": "research_grade",
        "community_taxon_status": "species",
        "identification_disagreement": False,
        "captive_or_cultivated": False,
        "observer_id": f"observer-{source_id}",
        "locality": "Sydney",
        "life_stage": "adult",
        "sex": None,
        "observed_at": datetime(2025, 1, 2, 3, 4, tzinfo=UTC),
        "latitude": -33.87,
        "longitude": 151.21,
        "coordinate_uncertainty": 50.0,
        "coordinates_obscured": False,
        "country": "Australia",
        "country_code": "AU",
        "geo_cluster_id": "cluster-au",
        "distance_to_cluster_medoid_km": 4.2,
        "source_dataset_key": f"dataset-{source}",
        "source_dataset_doi": None,
        "source_record_url": f"https://example.test/{source}/{source_id}",
        "source_record_hash": _sha(f"record:{source}:{source_id}"),
        "retrieved_at": NOW,
        "source_snapshot_version": f"{source}-2026-07-14",
        "source_query_fingerprint": _sha(f"query:{source}"),
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
    provider_id: str,
    *,
    licence_policy_status: str = "allowed",
) -> dict[str, object]:
    source = str(observation["source"])
    observation_id = str(observation["reference_observation_id"])
    licence = (
        "CC-BY-NC-4.0" if licence_policy_status == "research_only" else "CC-BY-4.0"
    )
    licence_uri = (
        "https://creativecommons.org/licenses/by-nc/4.0/"
        if licence_policy_status == "research_only"
        else "https://creativecommons.org/licenses/by/4.0/"
    )
    return {
        "schema_version": REFERENCE_MEDIA_CANDIDATES_SCHEMA_VERSION,
        "reference_media_id": make_reference_media_id(
            source,
            provider_id,
            observation_id,
        ),
        "reference_observation_id": observation_id,
        "provider_media_id": provider_id,
        "source": source,
        "media_identifier": f"https://media.example.test/{provider_id}.jpg",
        "media_type": "StillImage",
        "width": 96,
        "height": 72,
        "creator": "Example Observer",
        "rights_holder": "Example Observer",
        "licence": licence,
        "licence_uri": licence_uri,
        "attribution": f"Example Observer / {licence}",
        "occurrence_licence": "CC0-1.0",
        "original_provider": source,
        "media_position": 0,
        "source_checksum": None,
        "source_checksum_algorithm": None,
        "download_status": "complete",
        "verification_status": "unreviewed",
        "exclusion_reason": None,
        "licence_policy_status": licence_policy_status,
        "retrieved_at": NOW,
        "source_snapshot_version": observation["source_snapshot_version"],
    }


def _object(
    candidate: dict[str, object],
    *,
    sha_seed: str,
    perceptual_hash: str = "dhash128-v1:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
) -> dict[str, object]:
    sha = _sha(sha_seed)
    media_id = str(candidate["reference_media_id"])
    return {
        "schema_version": REFERENCE_MEDIA_OBJECTS_SCHEMA_VERSION,
        "reference_media_id": media_id,
        "source_object_uri": (
            f"s3://reference-objects/{sha.removeprefix('sha256:')}.jpg"
        ),
        "content_type": "image/jpeg",
        "source_byte_count": 10_000,
        "decoded_width": 96,
        "decoded_height": 72,
        "sha256": sha,
        "perceptual_hash": perceptual_hash,
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


def _selection(
    observation: dict[str, object],
    candidate: dict[str, object],
    *,
    plan_id: str,
    visual_domain: str = "field",
) -> dict[str, object]:
    media_id = str(candidate["reference_media_id"])
    taxon_key = str(observation["accepted_taxon_key"])
    cluster_id = str(observation["geo_cluster_id"])
    return {
        "schema_version": REFERENCE_ACQUISITION_SELECTIONS_SCHEMA_VERSION,
        "reference_selection_id": make_reference_selection_id(
            acquisition_plan_id=plan_id,
            reference_media_id=media_id,
            candidate_accepted_taxon_key=taxon_key,
            geo_cluster_id=cluster_id,
            life_stage="adult",
            visual_domain=visual_domain,
        ),
        "acquisition_plan_id": plan_id,
        "target_accepted_taxon_key": "gbif:1938069",
        "candidate_set_id": "candidate-set-v1",
        "source_candidate_set_id": "source-set-v1",
        "candidate_accepted_taxon_key": taxon_key,
        "scientific_name": observation["reconciled_scientific_name"],
        "geo_cluster_id": cluster_id,
        "life_stage": "adult",
        "visual_domain": visual_domain,
        "reference_media_id": media_id,
        "reference_observation_id": observation["reference_observation_id"],
        "source": observation["source"],
        "fallback_level": 0,
        "selection_rank": 1,
        "selection_round": "independent_observation",
        "distance_to_cluster_medoid_km": 1.0,
        "observer_id": observation["observer_id"],
        "observed_date": datetime(2025, 1, 2, tzinfo=UTC).date(),
        "locality": observation["locality"],
        "background_group_id": None,
        "licence": candidate["licence"],
        "source_snapshot_version": observation["source_snapshot_version"],
        "selection_strategy": "test-selection-v1",
        "selection_seed": 42,
        "plan_configuration_fingerprint": _sha(f"plan:{plan_id}"),
        "selected_at": NOW,
    }


def _inputs(
    *,
    item_count: int = 1,
    selected_indexes: tuple[int, ...] = (0,),
    same_sha: bool = False,
    licence_policy_status: str = "allowed",
    perceptual_hashes: tuple[str, ...] | None = None,
    candidate_download_status: str | None = None,
    object_licence_policy_status: str | None = None,
    observation_taxa: tuple[tuple[str, str], ...] | None = None,
    object_downloaded_at: tuple[datetime, ...] | None = None,
    plan_fingerprints: tuple[str, ...] | None = None,
) -> _Inputs:
    observations = [
        _observation("GBIF", f"observation-{index}") for index in range(item_count)
    ]
    if observation_taxa is not None:
        assert len(observation_taxa) == item_count
        for observation, (taxon_key, scientific_name) in zip(
            observations, observation_taxa, strict=True
        ):
            observation["accepted_taxon_key"] = taxon_key
            observation["source_taxon_id"] = taxon_key.removeprefix("gbif:")
            observation["supplied_scientific_name"] = scientific_name
            observation["reconciled_scientific_name"] = scientific_name
    candidates = [
        _candidate(
            observation,
            f"media-{index}",
            licence_policy_status=licence_policy_status,
        )
        for index, observation in enumerate(observations)
    ]
    if candidate_download_status is not None:
        for candidate in candidates:
            candidate["download_status"] = candidate_download_status
    objects = [
        _object(
            candidate,
            sha_seed="same-image" if same_sha else f"image-{index}",
            perceptual_hash=(
                perceptual_hashes[index]
                if perceptual_hashes is not None
                else f"dhash128-v1:{(0xAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA + index * 0x11111111111111111111111111111111) & ((1 << 128) - 1):032x}"
            ),
        )
        for index, candidate in enumerate(candidates)
    ]
    if object_licence_policy_status is not None:
        for obj in objects:
            obj["licence_policy_status"] = object_licence_policy_status
    if object_downloaded_at is not None:
        assert len(object_downloaded_at) == item_count
        for obj, downloaded_at in zip(objects, object_downloaded_at, strict=True):
            obj["downloaded_at"] = downloaded_at
    deduplicated = deduplicate_reference_media(
        reference_media_objects_frame(objects),
        reference_media_candidates_frame(candidates),
        reference_observations_frame(observations),
        generated_at=NOW,
    )
    selection_rows = [
        _selection(
            observations[index],
            candidates[index],
            plan_id=f"plan-{index}",
        )
        for index in selected_indexes
    ]
    if plan_fingerprints is not None:
        assert len(plan_fingerprints) == len(selected_indexes)
        for selection, fingerprint in zip(
            selection_rows, plan_fingerprints, strict=True
        ):
            selection["plan_configuration_fingerprint"] = fingerprint
    selections = reference_acquisition_selections_frame(selection_rows)
    return _Inputs(
        selections=selections,
        objects=deduplicated.media_objects,
        candidates=deduplicated.media_candidates,
        observations=deduplicated.observations,
        relationships=deduplicated.relationships,
        deduplication_report=deduplicated.report,
    )


def _queue(inputs: _Inputs, *, include_research_only: bool = False):
    return build_reference_review_queue(
        inputs.selections,
        inputs.objects,
        inputs.candidates,
        inputs.observations,
        inputs.relationships,
        deduplication_report=inputs.deduplication_report,
        reference_bank_version="reference-bank-v1",
        created_at=NOW + timedelta(hours=1),
        include_research_only=include_research_only,
    )


def _empty_decisions() -> pl.DataFrame:
    return reference_review_decisions_frame([])


def _report_sha256(report: object) -> str:
    canonical = json.dumps(
        report,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(canonical).hexdigest()


def _raw_decision(
    queue_result,
    *,
    actor: str,
    reviewed_at: datetime,
    status: str = "verified",
    review_round: int = 1,
    identity: bool | None = True,
    life_stage: str = "adult",
    visual_domain: str = "live_field",
    view: str = "dorsal",
    notes: str | None = "Diagnostic markings are visible.",
    exclusion_reason: str | None = None,
) -> pl.DataFrame:
    return queue_result.decision_template.with_columns(
        pl.lit(review_round, dtype=pl.UInt16).alias("review_round"),
        pl.lit(actor).alias("verified_by"),
        pl.lit(reviewed_at).cast(pl.Datetime("us", "UTC")).alias("reviewed_at"),
        pl.lit(identity, dtype=pl.Boolean).alias("target_identity_verified"),
        pl.lit(status).alias("verification_status"),
        pl.lit(life_stage).alias("life_stage"),
        pl.lit(visual_domain).alias("visual_domain"),
        pl.lit(view).alias("view"),
        pl.lit("high").alias("review_confidence"),
        pl.lit(notes, dtype=pl.String).alias("review_notes"),
        pl.lit(exclusion_reason, dtype=pl.String).alias("exclusion_reason"),
    )


def _published_successor(
    tmp_path: Path,
) -> tuple[Path, dict[str, Path], dict[str, Path]]:
    queue_result = _queue(_inputs())
    root_paths = write_reference_review_export(
        queue_result,
        tmp_path / "root",
        run_id="root-export",
    )
    history_head = tmp_path / "review-history-head.json"
    initialize_reference_review_history_head(history_head, root_paths["report"])
    prior_report, prior_sha256 = validate_reference_review_history_head(
        history_head,
        root_paths["report"],
    )
    imported = import_reference_review_decisions(
        _raw_decision(
            queue_result,
            actor="reviewer-a",
            reviewed_at=NOW + timedelta(hours=2),
        ),
        queue=queue_result.queue,
        queue_provenance=queue_result.provenance,
        existing_decisions=_empty_decisions(),
        prior_report=prior_report,
        prior_report_sha256=prior_sha256,
    )
    revision_paths = write_reference_review_import(
        imported,
        tmp_path / "revision-1",
        run_id="revision-1",
    )
    return history_head, root_paths, revision_paths


def _rewrite_bound_summary(
    report: dict[str, object],
    summary_path: Path,
) -> None:
    summary_report = json.loads(json.dumps(report))
    del summary_report["artifacts"]["summary"]
    del summary_report["outputs"]["artifact_uris"]["summary"]
    title = (
        "Reference review queue export"
        if report["command"] == "references.export_review_queue"
        else "Reference review decision import"
    )
    content = _review_markdown(summary_report, title=title).encode()
    summary_path.write_bytes(content)
    summary_record = report["artifacts"]["summary"]
    summary_record["byte_count"] = len(content)
    summary_record["sha256"] = "sha256:" + hashlib.sha256(content).hexdigest()


def test_queue_uses_current_selections_and_preserves_unreviewed_domain() -> None:
    inputs = _inputs(item_count=2, selected_indexes=(0,))

    result = _queue(inputs)

    assert result.queue.height == 1
    assert (
        result.queue["reference_media_id"].item()
        == inputs.selections["reference_media_id"].item()
    )
    assert result.queue["life_stage"].item() == "adult"
    assert result.queue["visual_domain"].item() is None
    assert result.queue["view"].item() is None
    assert result.report["counts"]["selected_rows"] == 1


def test_queue_accepts_immutable_pending_candidate_with_committed_allowed_object() -> (
    None
):
    production_inputs = _inputs(
        candidate_download_status="pending",
        licence_policy_status="unreviewed",
        object_licence_policy_status="allowed",
    )

    result = _queue(production_inputs)

    assert result.queue.height == 1
    assert result.queue["licence_policy_status"].item() == "allowed"


def test_resolved_duplicate_group_collapses_to_one_canonical_request() -> None:
    inputs = _inputs(item_count=2, selected_indexes=(0, 1), same_sha=True)

    result = _queue(inputs)

    assert result.queue.height == 1
    assert (
        result.queue["reference_media_id"].item()
        == result.queue["canonical_reference_media_id"].item()
    )
    assert result.report["counts"]["collapsed_selected_rows"] == 1


def test_unresolved_duplicate_group_keeps_selected_media_decisions_separate() -> None:
    inputs = _inputs(
        item_count=2,
        selected_indexes=(0, 1),
        perceptual_hashes=(
            "dhash128-v1:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            "dhash128-v1:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaab",
        ),
    )

    result = _queue(inputs)

    assert set(inputs.relationships["resolution_status"]) == {"review_required"}
    assert result.queue.height == 2
    assert set(result.queue["reference_media_id"]) == set(
        inputs.selections["reference_media_id"]
    )
    assert set(result.queue["required_review_count"]) == {2}
    assert all(
        "duplicate_resolution_pending" in reason
        for reason in result.queue["review_reason"]
    )


def test_queue_rejects_truncated_multi_object_duplicate_ledger() -> None:
    inputs = _inputs(item_count=2, selected_indexes=(0, 1), same_sha=True)

    with pytest.raises(
        ValueError,
        match="deduplication report counts are inconsistent",
    ):
        _queue(
            _Inputs(
                selections=inputs.selections,
                objects=inputs.objects,
                candidates=inputs.candidates,
                observations=inputs.observations,
                relationships=reference_media_duplicate_relationships_frame([]),
                deduplication_report=inputs.deduplication_report,
            )
        )


def test_queue_rejects_aligned_duplicate_truncation_with_original_report() -> None:
    inputs = _inputs(item_count=2, selected_indexes=(0, 1), same_sha=True)
    selected_media_id = str(inputs.selections["reference_media_id"][0])
    candidate = inputs.candidates.filter(
        pl.col("reference_media_id") == selected_media_id
    )
    observation_id = str(candidate["reference_observation_id"].item())
    object_rows = inputs.objects.filter(
        pl.col("reference_media_id") == selected_media_id
    ).to_dicts()
    object_rows[0].update(
        {
            "duplicate_group_id": None,
            "duplicate_type": None,
            "canonical_reference_media_id": None,
            "provider_mirror_ids": [],
        }
    )
    truncated = deduplicate_reference_media(
        reference_media_objects_frame(object_rows),
        candidate,
        inputs.observations.filter(
            pl.col("reference_observation_id") == observation_id
        ),
        generated_at=NOW,
    )

    with pytest.raises(ValueError, match="input fingerprint is invalid"):
        _queue(
            _Inputs(
                selections=inputs.selections.filter(
                    pl.col("reference_media_id") == selected_media_id
                ),
                objects=truncated.media_objects,
                candidates=truncated.media_candidates,
                observations=truncated.observations,
                relationships=truncated.relationships,
                deduplication_report=inputs.deduplication_report,
            )
        )


def test_queue_is_deterministic_after_source_row_reordering() -> None:
    inputs = _inputs(item_count=2, selected_indexes=(0, 1))
    reordered = _Inputs(
        selections=inputs.selections.reverse(),
        objects=inputs.objects.reverse(),
        candidates=inputs.candidates.reverse(),
        observations=inputs.observations.reverse(),
        relationships=inputs.relationships.reverse(),
        deduplication_report=inputs.deduplication_report,
    )

    assert _queue(reordered).queue.equals(_queue(inputs).queue)


def test_request_identity_binds_relevant_plan_but_not_unselected_timestamp() -> None:
    plan_v1 = _inputs(plan_fingerprints=(_sha("review-plan-v1"),))
    plan_v2 = _inputs(plan_fingerprints=(_sha("review-plan-v2"),))

    first_plan_queue = _queue(plan_v1)
    second_plan_queue = _queue(plan_v2)

    assert (
        first_plan_queue.queue["review_request_id"].item()
        != second_plan_queue.queue["review_request_id"].item()
    )

    first_inventory = _inputs(
        item_count=2,
        selected_indexes=(0,),
        object_downloaded_at=(NOW, NOW + timedelta(days=1)),
    )
    second_inventory = _inputs(
        item_count=2,
        selected_indexes=(0,),
        object_downloaded_at=(NOW, NOW + timedelta(days=2)),
    )
    first_inventory_queue = build_reference_review_queue(
        first_inventory.selections,
        first_inventory.objects,
        first_inventory.candidates,
        first_inventory.observations,
        first_inventory.relationships,
        deduplication_report=first_inventory.deduplication_report,
        reference_bank_version="reference-bank-v1",
    )
    second_inventory_queue = build_reference_review_queue(
        second_inventory.selections,
        second_inventory.objects,
        second_inventory.candidates,
        second_inventory.observations,
        second_inventory.relationships,
        deduplication_report=second_inventory.deduplication_report,
        reference_bank_version="reference-bank-v1",
    )

    assert first_inventory_queue.queue["created_at"].item() == NOW
    assert first_inventory_queue.queue["review_request_id"].item() == (
        second_inventory_queue.queue["review_request_id"].item()
    )


def test_queue_rejects_stale_selection_licence() -> None:
    inputs = _inputs()
    selection_rows = inputs.selections.to_dicts()
    selection_rows[0]["licence"] = "CC0-1.0"

    with pytest.raises(ValueError, match="selected media has stale licence"):
        _queue(
            _Inputs(
                selections=reference_acquisition_selections_frame(selection_rows),
                objects=inputs.objects,
                candidates=inputs.candidates,
                observations=inputs.observations,
                relationships=inputs.relationships,
                deduplication_report=inputs.deduplication_report,
            )
        )


def test_queue_rejects_cross_taxon_duplicate_collapse() -> None:
    inputs = _inputs(
        item_count=2,
        selected_indexes=(0, 1),
        same_sha=True,
        observation_taxa=(
            ("gbif:1938069", "Papilio demoleus"),
            ("gbif:1938070", "Papilio polytes"),
        ),
    )

    assert set(inputs.relationships["resolution_status"]) == {"conflict"}
    with pytest.raises(ValueError, match="conflicting taxon provenance"):
        _queue(inputs)


def test_default_queue_timestamp_is_stable_and_prior_decisions_resume() -> None:
    inputs = _inputs()
    first_queue = build_reference_review_queue(
        inputs.selections,
        inputs.objects,
        inputs.candidates,
        inputs.observations,
        inputs.relationships,
        deduplication_report=inputs.deduplication_report,
        reference_bank_version="reference-bank-v1",
    )
    second_queue = build_reference_review_queue(
        inputs.selections,
        inputs.objects,
        inputs.candidates,
        inputs.observations,
        inputs.relationships,
        deduplication_report=inputs.deduplication_report,
        reference_bank_version="reference-bank-v1",
    )
    first = import_reference_review_decisions(
        _raw_decision(
            first_queue,
            actor="reviewer-a",
            reviewed_at=NOW + timedelta(hours=2),
        ),
        queue=first_queue.queue,
        queue_provenance=first_queue.provenance,
        existing_decisions=_empty_decisions(),
        prior_report=first_queue.report,
        prior_report_sha256=_report_sha256(first_queue.report),
    )

    resumed = resolve_reference_review_statuses(
        second_queue.queue,
        first.decisions,
        queue_provenance=second_queue.provenance,
    )

    assert first_queue.queue.equals(second_queue.queue)
    assert resumed.verified.height == 1


def test_import_requires_complete_ledger_projection_and_rejects_reset() -> None:
    queue_result = _queue(_inputs())
    first = import_reference_review_decisions(
        _raw_decision(
            queue_result,
            actor="reviewer-a",
            reviewed_at=NOW + timedelta(hours=2),
        ),
        queue=queue_result.queue,
        queue_provenance=queue_result.provenance,
        existing_decisions=_empty_decisions(),
        prior_report=queue_result.report,
        prior_report_sha256=_report_sha256(queue_result.report),
    )
    correction = _raw_decision(
        queue_result,
        actor="reviewer-a",
        reviewed_at=NOW + timedelta(hours=3),
        review_round=2,
    )

    with pytest.raises(ValueError, match="complete existing decision ledger"):
        import_reference_review_decisions(
            correction,
            queue=first.queue,
            queue_provenance=first.provenance,
            existing_decisions=_empty_decisions(),
            prior_report=queue_result.report,
            prior_report_sha256=_report_sha256(queue_result.report),
        )
    with pytest.raises(ValueError, match="complete existing decision ledger"):
        import_reference_review_decisions(
            correction,
            queue=queue_result.queue,
            queue_provenance=queue_result.provenance,
            existing_decisions=first.decisions,
            prior_report=first.report,
            prior_report_sha256=_report_sha256(first.report),
        )

    corrected = import_reference_review_decisions(
        correction,
        queue=first.queue,
        queue_provenance=first.provenance,
        existing_decisions=first.decisions,
        prior_report=first.report,
        prior_report_sha256=_report_sha256(first.report),
    )
    assert corrected.decisions.height == 2


def test_queue_rejects_cross_artifact_provenance_drift() -> None:
    inputs = _inputs()
    rows = inputs.selections.to_dicts()
    rows[0]["scientific_name"] = "Papilio polytes"

    with pytest.raises(ValueError, match="scientific name"):
        _queue(
            _Inputs(
                selections=reference_acquisition_selections_frame(rows),
                objects=inputs.objects,
                candidates=inputs.candidates,
                observations=inputs.observations,
                relationships=inputs.relationships,
                deduplication_report=inputs.deduplication_report,
            )
        )


def test_verified_and_excluded_decisions_have_disjoint_projections() -> None:
    queue_result = _queue(_inputs())
    verified = import_reference_review_decisions(
        _raw_decision(
            queue_result,
            actor="reviewer-a",
            reviewed_at=NOW + timedelta(hours=2),
            life_stage="larva",
        ),
        queue=queue_result.queue,
        queue_provenance=queue_result.provenance,
        existing_decisions=_empty_decisions(),
        prior_report=queue_result.report,
        prior_report_sha256=_report_sha256(queue_result.report),
    )
    excluded = import_reference_review_decisions(
        _raw_decision(
            queue_result,
            actor="reviewer-a",
            reviewed_at=NOW + timedelta(hours=2),
            status="excluded",
            identity=False,
            life_stage="pupa",
            visual_domain="unsuitable",
            view="ventral",
            exclusion_reason="Incorrect species identity.",
        ),
        queue=queue_result.queue,
        queue_provenance=queue_result.provenance,
        existing_decisions=_empty_decisions(),
        prior_report=queue_result.report,
        prior_report_sha256=_report_sha256(queue_result.report),
    )

    selected = select_verified_reference_media(
        queue_result.queue,
        verified.decisions,
        queue_provenance=queue_result.provenance,
    )
    assert verified.queue["review_status"].item() == "completed"
    assert verified.verified.height == 1
    assert verified.verified.select("life_stage", "visual_domain", "view").row(0) == (
        "adult",
        None,
        None,
    )
    assert verified.verified.select(
        "resolved_life_stage",
        "resolved_visual_domain",
        "resolved_view",
    ).row(0) == ("larva", "live_field", "dorsal")
    assert verified.excluded.is_empty()
    assert selected.select("life_stage", "visual_domain", "view").row(0) == (
        "adult",
        None,
        None,
    )
    assert selected.select(
        "resolved_life_stage",
        "resolved_visual_domain",
        "resolved_view",
    ).row(0) == (
        "larva",
        "live_field",
        "dorsal",
    )
    assert excluded.excluded.height == 1
    assert excluded.excluded.select("life_stage", "visual_domain", "view").row(0) == (
        "adult",
        None,
        None,
    )
    assert excluded.excluded.select(
        "resolved_life_stage",
        "resolved_visual_domain",
        "resolved_view",
    ).row(0) == ("pupa", "unsuitable", "ventral")
    assert excluded.verified.is_empty()
    assert select_verified_reference_media(
        queue_result.queue,
        excluded.decisions,
        queue_provenance=queue_result.provenance,
    ).is_empty()


def test_uncertainty_requires_a_later_distinct_reviewer() -> None:
    queue_result = _queue(_inputs())
    uncertain = import_reference_review_decisions(
        _raw_decision(
            queue_result,
            actor="reviewer-a",
            reviewed_at=NOW + timedelta(hours=2),
            status="uncertain",
            identity=None,
            notes="The diagnostic marks are occluded.",
        ),
        queue=queue_result.queue,
        queue_provenance=queue_result.provenance,
        existing_decisions=_empty_decisions(),
        prior_report=queue_result.report,
        prior_report_sha256=_report_sha256(queue_result.report),
    )
    same_actor_correction = import_reference_review_decisions(
        _raw_decision(
            queue_result,
            actor="reviewer-a",
            reviewed_at=NOW + timedelta(hours=3),
            review_round=2,
        ),
        queue=uncertain.queue,
        queue_provenance=uncertain.provenance,
        existing_decisions=uncertain.decisions,
        prior_report=uncertain.report,
        prior_report_sha256=_report_sha256(uncertain.report),
    )
    distinct_review = import_reference_review_decisions(
        _raw_decision(
            queue_result,
            actor="reviewer-b",
            reviewed_at=NOW + timedelta(hours=4),
        ),
        queue=same_actor_correction.queue,
        queue_provenance=same_actor_correction.provenance,
        existing_decisions=same_actor_correction.decisions,
        prior_report=same_actor_correction.report,
        prior_report_sha256=_report_sha256(same_actor_correction.report),
    )

    assert uncertain.queue["review_status"].item() == "second_review_required"
    assert (
        same_actor_correction.queue["review_status"].item() == "second_review_required"
    )
    assert distinct_review.queue["review_status"].item() == "completed"
    assert distinct_review.verified.height == 1


def test_dissent_is_not_majority_overwritten_and_correction_clears_it() -> None:
    queue_result = _queue(_inputs())
    first = _raw_decision(
        queue_result,
        actor="reviewer-a",
        reviewed_at=NOW + timedelta(hours=2),
    )
    dissent = _raw_decision(
        queue_result,
        actor="reviewer-b",
        reviewed_at=NOW + timedelta(hours=3),
        status="excluded",
        identity=False,
        exclusion_reason="Wrong identity.",
    )
    majority = _raw_decision(
        queue_result,
        actor="reviewer-c",
        reviewed_at=NOW + timedelta(hours=4),
    )
    conflicted = import_reference_review_decisions(
        pl.concat([majority, dissent, first]),
        queue=queue_result.queue,
        queue_provenance=queue_result.provenance,
        existing_decisions=_empty_decisions(),
        prior_report=queue_result.report,
        prior_report_sha256=_report_sha256(queue_result.report),
    )
    persisted = import_reference_review_decisions(
        _raw_decision(
            queue_result,
            actor="reviewer-b",
            reviewed_at=NOW + timedelta(hours=5),
            review_round=2,
            status="excluded",
            identity=False,
            view="ventral",
            exclusion_reason="Wrong identity remains.",
        ),
        queue=conflicted.queue,
        queue_provenance=conflicted.provenance,
        existing_decisions=conflicted.decisions,
        prior_report=conflicted.report,
        prior_report_sha256=_report_sha256(conflicted.report),
    )
    corrected = import_reference_review_decisions(
        _raw_decision(
            queue_result,
            actor="reviewer-b",
            reviewed_at=NOW + timedelta(hours=6),
            review_round=3,
        ),
        queue=persisted.queue,
        queue_provenance=persisted.provenance,
        existing_decisions=persisted.decisions,
        prior_report=persisted.report,
        prior_report_sha256=_report_sha256(persisted.report),
    )

    assert conflicted.queue["review_status"].item() == "conflict"
    assert conflicted.conflicts.height == 2
    assert (
        conflicted.conflicts.filter(pl.col("resolution_status") == "resolved").height
        == 1
    )
    assert (
        conflicted.conflicts.filter(pl.col("resolution_status") == "open").height == 1
    )
    assert persisted.queue["review_status"].item() == "conflict"
    assert (
        persisted.conflicts.filter(pl.col("resolution_status") == "resolved").height
        == 2
    )
    assert persisted.conflicts.filter(pl.col("resolution_status") == "open").height == 1
    assert conflicted.verified.is_empty()
    assert corrected.queue["review_status"].item() == "completed"
    assert set(corrected.conflicts["resolution_status"]) == {"resolved"}
    assert corrected.verified.height == 1


def test_conflict_lineage_opens_only_the_current_decision_id_group() -> None:
    queue_result = _queue(_inputs())
    first_conflict = import_reference_review_decisions(
        pl.concat(
            [
                _raw_decision(
                    queue_result,
                    actor="reviewer-a",
                    reviewed_at=NOW + timedelta(hours=2),
                ),
                _raw_decision(
                    queue_result,
                    actor="reviewer-b",
                    reviewed_at=NOW + timedelta(hours=3),
                    status="excluded",
                    identity=False,
                    exclusion_reason="Wrong identity.",
                ),
            ]
        ),
        queue=queue_result.queue,
        queue_provenance=queue_result.provenance,
        existing_decisions=_empty_decisions(),
        prior_report=queue_result.report,
        prior_report_sha256=_report_sha256(queue_result.report),
    )
    reconciled = import_reference_review_decisions(
        _raw_decision(
            queue_result,
            actor="reviewer-b",
            reviewed_at=NOW + timedelta(hours=4),
            review_round=2,
        ),
        queue=first_conflict.queue,
        queue_provenance=first_conflict.provenance,
        existing_decisions=first_conflict.decisions,
        prior_report=first_conflict.report,
        prior_report_sha256=_report_sha256(first_conflict.report),
    )
    current_conflict = import_reference_review_decisions(
        _raw_decision(
            queue_result,
            actor="reviewer-b",
            reviewed_at=NOW + timedelta(hours=5),
            review_round=3,
            status="excluded",
            identity=False,
            exclusion_reason="Wrong identity again.",
        ),
        queue=reconciled.queue,
        queue_provenance=reconciled.provenance,
        existing_decisions=reconciled.decisions,
        prior_report=reconciled.report,
        prior_report_sha256=_report_sha256(reconciled.report),
    )
    ids = {
        (str(row["verified_by"]), int(row["review_round"])): str(
            row["review_decision_id"]
        )
        for row in current_conflict.decisions.iter_rows(named=True)
    }
    open_conflict = current_conflict.conflicts.filter(
        pl.col("resolution_status") == "open"
    )
    resolved_conflict = current_conflict.conflicts.filter(
        pl.col("resolution_status") == "resolved"
    )

    assert reconciled.queue["review_status"].item() == "completed"
    assert current_conflict.queue["review_status"].item() == "conflict"
    assert open_conflict.height == 1
    assert resolved_conflict.height == 1
    assert open_conflict["effective_decision_ids"].to_list()[0] == sorted(
        [ids[("reviewer-a", 1)], ids[("reviewer-b", 3)]]
    )
    assert resolved_conflict["effective_decision_ids"].to_list()[0] == sorted(
        [ids[("reviewer-a", 1)], ids[("reviewer-b", 1)]]
    )
    assert ids[("reviewer-b", 2)] not in set(
        open_conflict["effective_decision_ids"].to_list()[0]
        + resolved_conflict["effective_decision_ids"].to_list()[0]
    )


def test_import_is_idempotent_and_rejects_gapped_actor_revisions() -> None:
    queue_result = _queue(_inputs())
    raw = _raw_decision(
        queue_result,
        actor="reviewer-a",
        reviewed_at=NOW + timedelta(hours=2),
    )
    first = import_reference_review_decisions(
        raw,
        queue=queue_result.queue,
        queue_provenance=queue_result.provenance,
        existing_decisions=_empty_decisions(),
        prior_report=queue_result.report,
        prior_report_sha256=_report_sha256(queue_result.report),
    )
    duplicated_batch = import_reference_review_decisions(
        pl.concat([raw, raw]),
        queue=queue_result.queue,
        queue_provenance=queue_result.provenance,
        existing_decisions=_empty_decisions(),
        prior_report=queue_result.report,
        prior_report_sha256=_report_sha256(queue_result.report),
    )
    replay = import_reference_review_decisions(
        raw,
        queue=first.queue,
        queue_provenance=first.provenance,
        existing_decisions=first.decisions,
        prior_report=first.report,
        prior_report_sha256=_report_sha256(first.report),
    )

    assert replay.decisions.equals(first.decisions)
    assert replay.report["counts"]["idempotent_replay_rows"] == 1
    assert duplicated_batch.decisions.equals(first.decisions)
    assert duplicated_batch.report["counts"]["idempotent_replay_rows"] == 1
    with pytest.raises(ValueError, match="contiguous"):
        import_reference_review_decisions(
            _raw_decision(
                queue_result,
                actor="reviewer-a",
                reviewed_at=NOW + timedelta(hours=3),
                review_round=3,
            ),
            queue=first.queue,
            queue_provenance=first.provenance,
            existing_decisions=first.decisions,
            prior_report=first.report,
            prior_report_sha256=_report_sha256(first.report),
        )


def test_import_rejects_semantic_id_to_source_hash_collision() -> None:
    queue_result = _queue(_inputs())
    raw = _raw_decision(
        queue_result,
        actor="reviewer-a",
        reviewed_at=NOW + timedelta(hours=2),
    )
    first = import_reference_review_decisions(
        raw,
        queue=queue_result.queue,
        queue_provenance=queue_result.provenance,
        existing_decisions=_empty_decisions(),
        prior_report=queue_result.report,
        prior_report_sha256=_report_sha256(queue_result.report),
    )
    tampered_rows = first.decisions.to_dicts()
    tampered_rows[0]["decision_source_hash"] = _sha("different-source-record")
    tampered = reference_review_decisions_frame(tampered_rows)

    with pytest.raises(ValueError, match="source hash does not match"):
        import_reference_review_decisions(
            raw,
            queue=first.queue,
            queue_provenance=first.provenance,
            existing_decisions=tampered,
            prior_report=first.report,
            prior_report_sha256=_report_sha256(first.report),
        )


@pytest.mark.parametrize(
    "actor",
    ["Reviewer-A", "reviewer-a\u200b", "reviewer-\u00e4"],
)
def test_import_rejects_unicode_or_noncanonical_reviewer_id(actor: str) -> None:
    queue_result = _queue(_inputs())

    with pytest.raises(
        ValueError,
        match="canonical lowercase ASCII reviewer identifier",
    ):
        import_reference_review_decisions(
            _raw_decision(
                queue_result,
                actor=actor,
                reviewed_at=NOW + timedelta(hours=2),
            ),
            queue=queue_result.queue,
            queue_provenance=queue_result.provenance,
            existing_decisions=_empty_decisions(),
            prior_report=queue_result.report,
            prior_report_sha256=_report_sha256(queue_result.report),
        )


def test_import_requires_exact_physical_schema_and_queue_binding() -> None:
    queue_result = _queue(_inputs())
    raw = _raw_decision(
        queue_result,
        actor="reviewer-a",
        reviewed_at=NOW + timedelta(hours=2),
    )

    with pytest.raises(ValueError, match=r"unknown=\['unexpected'\]"):
        import_reference_review_decisions(
            raw.with_columns(pl.lit("x").alias("unexpected")),
            queue=queue_result.queue,
            queue_provenance=queue_result.provenance,
            existing_decisions=_empty_decisions(),
            prior_report=queue_result.report,
            prior_report_sha256=_report_sha256(queue_result.report),
        )
    assert raw.schema == reference_review_decision_import_schema()
    assert (
        raw["import_schema_version"].item()
        == REFERENCE_REVIEW_DECISION_IMPORT_SCHEMA_VERSION
    )
    foreign = raw.with_columns(
        pl.lit("reference-review-request:" + "0" * 64).alias("review_request_id")
    )
    with pytest.raises(ValueError, match="unknown review request"):
        import_reference_review_decisions(
            foreign,
            queue=queue_result.queue,
            queue_provenance=queue_result.provenance,
            existing_decisions=_empty_decisions(),
            prior_report=queue_result.report,
            prior_report_sha256=_report_sha256(queue_result.report),
        )
    wrong_media = raw.with_columns(
        pl.lit("reference-media:" + "0" * 64).alias("reference_media_id")
    )
    with pytest.raises(ValueError, match="media does not match"):
        import_reference_review_decisions(
            wrong_media,
            queue=queue_result.queue,
            queue_provenance=queue_result.provenance,
            existing_decisions=_empty_decisions(),
            prior_report=queue_result.report,
            prior_report_sha256=_report_sha256(queue_result.report),
        )
    tampered_queue_rows = queue_result.queue.to_dicts()
    tampered_queue_rows[0]["accepted_taxon_key"] = "gbif:999"
    tampered_queue_rows[0]["scientific_name"] = "Papilio falsus"
    tampered_queue = reference_review_queue_frame(tampered_queue_rows)
    with pytest.raises(ValueError, match="queue semantics binding is inconsistent"):
        import_reference_review_decisions(
            raw,
            queue=tampered_queue,
            queue_provenance=queue_result.provenance,
            existing_decisions=_empty_decisions(),
            prior_report=queue_result.report,
            prior_report_sha256=_report_sha256(queue_result.report),
        )


def test_cancelled_queue_request_is_not_reopened() -> None:
    queue_result = _queue(_inputs())
    rows = queue_result.queue.to_dicts()
    rows[0]["review_status"] = "cancelled"
    cancelled = reference_review_queue_frame(rows)
    provenance_rows = queue_result.provenance.to_dicts()
    provenance_rows[0]["queue_row_fingerprint"] = _queue_row_fingerprint(rows[0])
    cancelled_provenance = pl.DataFrame(
        provenance_rows,
        schema=queue_result.provenance.schema,
        strict=True,
    )

    with pytest.raises(ValueError, match="attributable cancellation record"):
        resolve_reference_review_statuses(
            cancelled,
            _empty_decisions(),
            queue_provenance=cancelled_provenance,
        )


def test_research_only_verified_reference_remains_support_blocked() -> None:
    inputs = _inputs(licence_policy_status="research_only")
    assert _queue(inputs).queue.is_empty()
    queue_result = _queue(inputs, include_research_only=True)
    result = import_reference_review_decisions(
        _raw_decision(
            queue_result,
            actor="reviewer-a",
            reviewed_at=NOW + timedelta(hours=2),
        ),
        queue=queue_result.queue,
        queue_provenance=queue_result.provenance,
        existing_decisions=_empty_decisions(),
        prior_report=queue_result.report,
        prior_report_sha256=_report_sha256(queue_result.report),
    )

    assert result.verified.height == 1
    assert result.outcomes["support_eligible"].item() is False
    assert "licence_not_allowed" in result.outcomes["blocker_reasons"].item()
    assert select_verified_reference_media(
        queue_result.queue,
        result.decisions,
        queue_provenance=queue_result.provenance,
    ).is_empty()


def test_review_packets_are_atomic_create_only_and_report_last(tmp_path: Path) -> None:
    queue_result = _queue(_inputs())
    export_dir = tmp_path / "export"
    export_paths = write_reference_review_export(
        queue_result,
        export_dir,
        run_id="export-run",
    )
    result = import_reference_review_decisions(
        _raw_decision(
            queue_result,
            actor="reviewer-a",
            reviewed_at=NOW + timedelta(hours=2),
        ),
        queue=queue_result.queue,
        queue_provenance=queue_result.provenance,
        existing_decisions=_empty_decisions(),
        prior_report=queue_result.report,
        prior_report_sha256=_report_sha256(queue_result.report),
    )
    import_dir = tmp_path / "import"
    import_paths = write_reference_review_import(
        result,
        import_dir,
        run_id="import-run",
    )

    assert set(export_paths) == {
        "decisions",
        "decision_template",
        "queue",
        "queue_provenance",
        "report",
        "summary",
    }
    assert set(import_paths) == {
        "conflicts",
        "decision_import",
        "decisions",
        "excluded",
        "outcomes",
        "queue",
        "queue_provenance",
        "report",
        "summary",
        "verified",
    }
    report = json.loads(import_paths["report"].read_text())
    assert report["status"] == "complete"
    assert report["run_id"] == "import-run"
    assert all(record["committed"] for record in report["artifacts"].values())
    with pytest.raises(FileExistsError):
        write_reference_review_import(result, import_dir, run_id="retry")
    assert not list(tmp_path.glob(".*.tmp"))


def test_relative_packet_artifacts_survive_working_directory_change(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    origin = tmp_path / "origin"
    elsewhere = tmp_path / "elsewhere"
    origin.mkdir()
    elsewhere.mkdir()
    monkeypatch.chdir(origin)
    root_paths = write_reference_review_export(
        _queue(_inputs()),
        Path("relative-root"),
        run_id="relative-root",
    )
    report_path = root_paths["report"].resolve()
    history_head = tmp_path / "review-history-head.json"

    monkeypatch.chdir(elsewhere)
    initialize_reference_review_history_head(history_head, report_path)
    report, _ = validate_reference_review_history_head(history_head, report_path)

    for logical_name, record in report["artifacts"].items():
        artifact_path = Path(str(record["uri"]))
        assert artifact_path.is_absolute()
        validate_reference_review_packet_artifact(
            report,
            str(logical_name),
            artifact_path,
        )


def test_history_head_initializes_validates_and_advances_one_packet(
    tmp_path: Path,
) -> None:
    queue_result = _queue(_inputs())
    root_paths = write_reference_review_export(
        queue_result,
        tmp_path / "root",
        run_id="root-export",
    )
    history_head = tmp_path / "review-history-head.json"

    initialize_reference_review_history_head(history_head, root_paths["report"])
    prior_report, prior_sha256 = validate_reference_review_history_head(
        history_head,
        root_paths["report"],
    )
    assert prior_report["history"]["revision"] == 0

    imported = import_reference_review_decisions(
        _raw_decision(
            queue_result,
            actor="reviewer-a",
            reviewed_at=NOW + timedelta(hours=2),
        ),
        queue=queue_result.queue,
        queue_provenance=queue_result.provenance,
        existing_decisions=_empty_decisions(),
        prior_report=prior_report,
        prior_report_sha256=prior_sha256,
    )
    revision_paths = write_reference_review_import(
        imported,
        tmp_path / "revision-1",
        run_id="revision-1",
    )
    advance_reference_review_history_head(
        history_head,
        prior_report_path=root_paths["report"],
        next_report_path=revision_paths["report"],
    )
    current_report, current_sha256 = validate_reference_review_history_head(
        history_head,
        revision_paths["report"],
    )

    assert current_report["history"]["revision"] == 1
    assert current_report["history"]["parent_report_sha256"] == prior_sha256
    assert current_sha256 != prior_sha256
    with pytest.raises(ValueError, match="authoritative history head"):
        validate_reference_review_history_head(history_head, root_paths["report"])


@pytest.mark.parametrize(
    "successor_fingerprint",
    ["malformed", _sha("mutated-successor-history")],
)
def test_history_head_rejects_invalid_successor_history_fingerprint(
    tmp_path: Path,
    successor_fingerprint: str,
) -> None:
    queue_result = _queue(_inputs())
    root_paths = write_reference_review_export(
        queue_result,
        tmp_path / "root",
        run_id="root-export",
    )
    history_head = tmp_path / "review-history-head.json"
    initialize_reference_review_history_head(history_head, root_paths["report"])
    prior_report, prior_sha256 = validate_reference_review_history_head(
        history_head,
        root_paths["report"],
    )
    imported = import_reference_review_decisions(
        _raw_decision(
            queue_result,
            actor="reviewer-a",
            reviewed_at=NOW + timedelta(hours=2),
        ),
        queue=queue_result.queue,
        queue_provenance=queue_result.provenance,
        existing_decisions=_empty_decisions(),
        prior_report=prior_report,
        prior_report_sha256=prior_sha256,
    )
    revision_paths = write_reference_review_import(
        imported,
        tmp_path / "revision-1",
        run_id="revision-1",
    )
    successor_report = json.loads(revision_paths["report"].read_text())
    successor_report["history"]["decision_ledger_fingerprint"] = successor_fingerprint
    revision_paths["report"].write_text(
        json.dumps(successor_report, indent=2, sort_keys=True) + "\n"
    )

    with pytest.raises(ValueError, match="history binding"):
        advance_reference_review_history_head(
            history_head,
            prior_report_path=root_paths["report"],
            next_report_path=revision_paths["report"],
        )
    validate_reference_review_history_head(history_head, root_paths["report"])


def test_history_head_rejects_modified_artifact_and_report_bytes(
    tmp_path: Path,
) -> None:
    queue_result = _queue(_inputs())
    root_paths = write_reference_review_export(
        queue_result,
        tmp_path / "root",
        run_id="root-export",
    )
    history_head = tmp_path / "review-history-head.json"
    initialize_reference_review_history_head(history_head, root_paths["report"])
    report, _ = validate_reference_review_history_head(
        history_head,
        root_paths["report"],
    )

    original_queue_bytes = root_paths["queue"].read_bytes()
    root_paths["queue"].write_bytes(original_queue_bytes + b"tampered")
    with pytest.raises(ValueError, match="artifact binding is invalid: queue"):
        validate_reference_review_packet_artifact(
            report,
            "queue",
            root_paths["queue"],
        )
    root_paths["queue"].write_bytes(original_queue_bytes)

    modified_report = dict(report)
    modified_report["run_id"] = "modified-root"
    root_paths["report"].write_text(
        json.dumps(modified_report, indent=2, sort_keys=True) + "\n"
    )
    with pytest.raises(ValueError, match="summary is not derived from its report"):
        validate_reference_review_history_head(history_head, root_paths["report"])


def test_history_head_rejects_tampered_decision_import_bytes(
    tmp_path: Path,
) -> None:
    history_head, root_paths, revision_paths = _published_successor(tmp_path)
    advance_reference_review_history_head(
        history_head,
        prior_report_path=root_paths["report"],
        next_report_path=revision_paths["report"],
    )
    decision_import_path = revision_paths["decision_import"]
    decision_import_path.write_bytes(decision_import_path.read_bytes() + b"tampered")

    with pytest.raises(
        ValueError,
        match="artifact binding is invalid: decision_import",
    ):
        validate_reference_review_history_head(
            history_head,
            revision_paths["report"],
        )


@pytest.mark.parametrize(
    ("section", "field", "value", "message"),
    [
        (
            "history",
            "parent_report_sha256",
            _sha("wrong-parent-report"),
            "does not extend the history head",
        ),
        (
            "inputs",
            "raw_decisions_fingerprint",
            _sha("wrong-decision-import"),
            "report inputs are inconsistent",
        ),
        (
            "counts",
            "idempotent_replay_rows",
            1,
            "import counts are inconsistent",
        ),
    ],
)
def test_history_head_rejects_inconsistent_successor_audit(
    tmp_path: Path,
    section: str,
    field: str,
    value: object,
    message: str,
) -> None:
    history_head, root_paths, revision_paths = _published_successor(tmp_path)
    successor_report = json.loads(revision_paths["report"].read_text())
    successor_report[section][field] = value
    _rewrite_bound_summary(successor_report, revision_paths["summary"])
    revision_paths["report"].write_text(
        json.dumps(successor_report, indent=2, sort_keys=True) + "\n"
    )

    with pytest.raises(ValueError, match=message):
        advance_reference_review_history_head(
            history_head,
            prior_report_path=root_paths["report"],
            next_report_path=revision_paths["report"],
        )
    validate_reference_review_history_head(history_head, root_paths["report"])


@pytest.mark.parametrize(
    ("section", "field", "value", "message"),
    [
        (
            None,
            "resolved_at",
            (NOW + timedelta(days=1)).isoformat(),
            "resolution timestamp is inconsistent",
        ),
        (
            "inputs",
            "source_queue_provenance_fingerprint",
            _sha("forged-source-provenance"),
            "parent ledger audit is inconsistent",
        ),
    ],
)
def test_history_head_rejects_forged_successor_resolution_binding(
    tmp_path: Path,
    section: str | None,
    field: str,
    value: object,
    message: str,
) -> None:
    history_head, root_paths, revision_paths = _published_successor(tmp_path)
    successor_report = json.loads(revision_paths["report"].read_text())
    target = successor_report if section is None else successor_report[section]
    target[field] = value
    revision_paths["report"].write_text(
        json.dumps(successor_report, indent=2, sort_keys=True) + "\n"
    )

    with pytest.raises(ValueError, match=message):
        advance_reference_review_history_head(
            history_head,
            prior_report_path=root_paths["report"],
            next_report_path=revision_paths["report"],
        )
    validate_reference_review_history_head(history_head, root_paths["report"])


def test_history_head_rejects_byte_bound_but_forged_summary(
    tmp_path: Path,
) -> None:
    history_head, root_paths, revision_paths = _published_successor(tmp_path)
    forged_summary = b"# Forged review summary\n"
    revision_paths["summary"].write_bytes(forged_summary)
    successor_report = json.loads(revision_paths["report"].read_text())
    summary_record = successor_report["artifacts"]["summary"]
    summary_record["byte_count"] = len(forged_summary)
    summary_record["sha256"] = "sha256:" + hashlib.sha256(forged_summary).hexdigest()
    revision_paths["report"].write_text(
        json.dumps(successor_report, indent=2, sort_keys=True) + "\n"
    )

    with pytest.raises(ValueError, match="summary is not derived from its report"):
        advance_reference_review_history_head(
            history_head,
            prior_report_path=root_paths["report"],
            next_report_path=revision_paths["report"],
        )


def test_history_head_rejects_semantically_mutated_decision_import(
    tmp_path: Path,
) -> None:
    history_head, root_paths, revision_paths = _published_successor(tmp_path)
    decision_import_path = revision_paths["decision_import"]
    mutated_import = pl.read_parquet(decision_import_path).with_columns(
        pl.lit("Mutated after review.").alias("review_notes")
    )
    mutated_import.write_parquet(decision_import_path)
    successor_report = json.loads(revision_paths["report"].read_text())
    import_record = successor_report["artifacts"]["decision_import"]
    content = decision_import_path.read_bytes()
    import_record["byte_count"] = len(content)
    import_record["sha256"] = "sha256:" + hashlib.sha256(content).hexdigest()
    successor_report["inputs"]["raw_decisions_fingerprint"] = _frame_fingerprint(
        mutated_import
    )
    revision_paths["report"].write_text(
        json.dumps(successor_report, indent=2, sort_keys=True) + "\n"
    )

    with pytest.raises(ValueError, match="decision import artifact is inconsistent"):
        advance_reference_review_history_head(
            history_head,
            prior_report_path=root_paths["report"],
            next_report_path=revision_paths["report"],
        )


def test_history_head_enforces_fixed_logical_artifact_filenames(
    tmp_path: Path,
) -> None:
    history_head, root_paths, revision_paths = _published_successor(tmp_path)
    renamed_import = revision_paths["decision_import"].with_name(
        "renamed-decision-import.parquet"
    )
    renamed_import.write_bytes(revision_paths["decision_import"].read_bytes())
    successor_report = json.loads(revision_paths["report"].read_text())
    renamed_uri = str(renamed_import.resolve())
    successor_report["artifacts"]["decision_import"]["uri"] = renamed_uri
    successor_report["outputs"]["artifact_uris"]["decision_import"] = renamed_uri
    revision_paths["report"].write_text(
        json.dumps(successor_report, indent=2, sort_keys=True) + "\n"
    )

    with pytest.raises(
        ValueError,
        match="artifact path is invalid: decision_import",
    ):
        advance_reference_review_history_head(
            history_head,
            prior_report_path=root_paths["report"],
            next_report_path=revision_paths["report"],
        )


def test_failed_packet_publication_leaves_no_visible_or_staged_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    queue_result = _queue(_inputs())
    output = tmp_path / "failed-export"
    real_write_text = Path.write_text

    def fail_report(
        path: Path,
        data: str,
        *args: object,
        **kwargs: object,
    ) -> int:
        if path.name == "reference_review_export_report.json":
            raise OSError("injected report failure")
        return real_write_text(path, data, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", fail_report)

    with pytest.raises(OSError, match="injected report failure"):
        write_reference_review_export(queue_result, output, run_id="failed-run")
    assert not output.exists()
    assert not list(tmp_path.glob(".*.tmp"))


def test_concurrent_packet_publication_has_one_create_only_winner(
    tmp_path: Path,
) -> None:
    queue_result = _queue(_inputs())
    output = tmp_path / "concurrent-export"

    def publish(index: int) -> str:
        try:
            write_reference_review_export(
                queue_result,
                output,
                run_id=f"writer-{index}",
            )
        except FileExistsError:
            return "exists"
        return "complete"

    with ThreadPoolExecutor(max_workers=8) as executor:
        statuses = list(executor.map(publish, range(8)))

    assert statuses.count("complete") == 1
    assert statuses.count("exists") == 7
    assert (output / "reference_review_export_report.json").is_file()
    assert not list(tmp_path.glob(".*.tmp"))
    lock = tmp_path / ".concurrent-export.publish.lock"
    assert list(tmp_path.glob(".*.publish.lock")) == [lock]
    assert lock.is_file()
    second_output = tmp_path / "second-export"
    write_reference_review_export(queue_result, second_output, run_id="later-writer")
    assert (second_output / "reference_review_export_report.json").is_file()


def test_import_writer_rejects_tampered_derived_conflict_projection(
    tmp_path: Path,
) -> None:
    queue_result = _queue(_inputs())
    verified = _raw_decision(
        queue_result,
        actor="reviewer-a",
        reviewed_at=NOW + timedelta(hours=2),
    )
    excluded = _raw_decision(
        queue_result,
        actor="reviewer-b",
        reviewed_at=NOW + timedelta(hours=3),
        status="excluded",
        identity=False,
        exclusion_reason="Wrong identity.",
    )
    result = import_reference_review_decisions(
        pl.concat([verified, excluded]),
        queue=queue_result.queue,
        queue_provenance=queue_result.provenance,
        existing_decisions=_empty_decisions(),
        prior_report=queue_result.report,
        prior_report_sha256=_report_sha256(queue_result.report),
    )
    tampered = replace(result, conflicts=result.conflicts.head(0))

    with pytest.raises(ValueError, match="conflicts is not the derived projection"):
        write_reference_review_import(tampered, tmp_path / "tampered")
    assert not (tmp_path / "tampered").exists()


def test_reference_review_commands_run_end_to_end_without_network(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    inputs = _inputs()
    paths = {
        "selections": tmp_path / "selections.parquet",
        "objects": tmp_path / "objects.parquet",
        "candidates": tmp_path / "candidates.parquet",
        "observations": tmp_path / "observations.parquet",
        "relationships": tmp_path / "relationships.parquet",
        "deduplication_report": tmp_path / "deduplication-report.json",
    }
    inputs.selections.write_parquet(paths["selections"])
    inputs.objects.write_parquet(paths["objects"])
    inputs.candidates.write_parquet(paths["candidates"])
    inputs.observations.write_parquet(paths["observations"])
    inputs.relationships.write_parquet(paths["relationships"])
    paths["deduplication_report"].write_text(
        json.dumps(inputs.deduplication_report, indent=2, sort_keys=True) + "\n"
    )
    monkeypatch.setattr(
        socket,
        "create_connection",
        lambda *_args, **_kwargs: pytest.fail("review CLI accessed the network"),
    )
    export_dir = tmp_path / "export"
    history_head = tmp_path / "review-history-head.json"
    export_inputs = [
        "--acquisition-selections",
        str(paths["selections"]),
        "--observations",
        str(paths["observations"]),
        "--media-candidates",
        str(paths["candidates"]),
        "--media-objects",
        str(paths["objects"]),
        "--duplicate-relationships",
        str(paths["relationships"]),
        "--deduplication-report",
        str(paths["deduplication_report"]),
        "--reference-bank-version",
        "reference-bank-v1",
    ]
    rejected_dir = tmp_path / "rejected-export"
    rejected_rc = run(
        build_parser().parse_args(
            [
                "references",
                "export-review-queue",
                *export_inputs,
                "--output-dir",
                str(rejected_dir),
                "--history-head",
                str(rejected_dir / "history-head.json"),
            ]
        )
    )
    rejected_payload = json.loads(capsys.readouterr().out)

    assert rejected_rc == 2
    assert "outside immutable packet directories" in rejected_payload["error"]
    assert not rejected_dir.exists()

    export_rc = run(
        build_parser().parse_args(
            [
                "references",
                "export-review-queue",
                *export_inputs,
                "--output-dir",
                str(export_dir),
                "--history-head",
                str(history_head),
                "--run-id",
                "cli-export",
            ]
        )
    )
    export_payload = json.loads(capsys.readouterr().out)
    template = pl.read_parquet(
        export_dir / "reference_review_decision_template.parquet"
    )
    raw = _raw_decision(
        SimpleNamespace(decision_template=template),
        actor="reviewer-a",
        reviewed_at=NOW + timedelta(hours=2),
    )
    raw_path = tmp_path / "completed-decisions.parquet"
    raw.write_parquet(raw_path)
    import_dir = tmp_path / "import"

    import_rc = run(
        build_parser().parse_args(
            [
                "references",
                "import-review-decisions",
                "--review-queue",
                str(export_dir / "reference_review_queue.parquet"),
                "--queue-provenance",
                str(export_dir / "reference_review_queue_provenance.parquet"),
                "--decisions",
                str(raw_path),
                "--existing-decisions",
                str(export_dir / "reference_review_decisions.parquet"),
                "--prior-review-report",
                str(export_dir / "reference_review_export_report.json"),
                "--history-head",
                str(history_head),
                "--output-dir",
                str(import_dir),
                "--run-id",
                "cli-import",
            ]
        )
    )
    import_payload = json.loads(capsys.readouterr().out)

    assert export_rc == 0
    assert export_payload["status"] == "complete"
    assert import_rc == 0
    assert import_payload["status"] == "complete"
    assert pl.read_parquet(import_dir / "verified_reference_media.parquet").height == 1
