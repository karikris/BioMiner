"""Deterministic ButterflyLens handoff fixture inputs."""

from __future__ import annotations

from biominer.integration.butterflylens_model_export import (
    build_butterflylens_model_layer,
    build_butterflylens_project_projection,
    build_butterflylens_run_projection,
)
from helpers.dynamic_pool_handoff_fixture import build_dynamic_pool_handoff_fixture


def sha(character: str) -> str:
    """Return a recognizable valid semantic fingerprint."""

    return f"sha256:{character * 64}"


def build_butterflylens_model_fixture() -> dict[str, object]:
    """Build a complete project/run/source/media/model projection fixture."""

    dynamic = build_dynamic_pool_handoff_fixture()
    project = build_butterflylens_project_projection(
        project_id="project:australian-butterflies",
        slug="australian-butterflies",
        name="Australian Butterflies",
        description="Pinned production-boundary fixture.",
        status="active",
        boundary_id="boundary:australia",
        boundary_version="asgs-2021",
        boundary_sha256=sha("c"),
        sensitive_coordinate_policy_version="sensitive-coordinates-v1",
        root_taxon_keys=["gbif:6953"],
        taxonomy_fingerprint=sha("d"),
        search_plan_fingerprint=sha("e"),
        data_policy_version="data-policy-v1",
        consent_policy_version="consent-policy-v1",
        created_at="2026-07-18T12:00:00+10:00",
        updated_at="2026-07-18T12:00:00+10:00",
    )
    run = build_butterflylens_run_projection(
        run_id="run-tx-handoff-1",
        project=project,
        run_kind="full_pipeline",
        mode="replay",
        status="succeeded",
        requested_at="2026-07-18T12:00:00+10:00",
        started_at="2026-07-18T12:01:00+10:00",
        finished_at="2026-07-18T12:02:00+10:00",
        updated_at="2026-07-18T12:02:00+10:00",
        producer_commit="1" * 40,
        engine_interface_version="butterflylens-handoff-v1",
        engine_command="biominer export butterflylens",
        input_fingerprints=[sha("c"), sha("d"), sha("e"), sha("9")],
    )
    source_media_records = [
        {
            "flickr_photo_id": "flickr-photo-1",
            "organism_unit_id": "organism-unit-1",
            "source_record_hash": sha("8"),
            "source_snapshot_fingerprint": sha("9"),
            "media_content_sha256": sha("a"),
            "media_byte_count": 1234,
            "media_type": "image/jpeg",
            "decode_status": "valid",
            "rights_fingerprint": sha("b"),
            "rights_status": "allowed",
            "duplicate_group_id": "duplicate-1",
            "owner_group_id": "owner-1",
            "observation_group_id": "observation-1",
        }
    ]
    layer = build_butterflylens_model_layer(
        project=project,
        run=run,
        source_media_records=source_media_records,
        candidate_scores=dynamic["candidate_scores"],
        pool_plans=dynamic["pool_plans"],
        source_score_artifact_sha256=sha("f"),
    )
    return {
        "project": project,
        "run": run,
        "source_media_records": source_media_records,
        "layer": layer,
        **dynamic,
    }


__all__ = ["build_butterflylens_model_fixture", "sha"]
