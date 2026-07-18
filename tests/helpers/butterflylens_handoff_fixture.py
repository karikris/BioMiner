"""Deterministic ButterflyLens handoff fixture inputs."""

from __future__ import annotations

from biominer.integration.butterflylens_model_export import (
    build_butterflylens_model_layer,
    build_butterflylens_project_projection,
    build_butterflylens_run_projection,
)
from biominer.integration.butterflylens_geographic_export import (
    build_butterflylens_geographic_impact,
)
from biominer.integration.butterflylens_review_export import (
    build_butterflylens_review_layer,
)
from biominer.evaluation.dynamic_pool_review import (
    ProbabilityAuditSamplingPolicy,
    build_dynamic_pool_audit_frame,
    build_probability_audit_sample,
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


def build_butterflylens_review_fixture() -> dict[str, object]:
    """Build a one-unit representative review selection linked to the model layer."""

    fixture = build_butterflylens_model_fixture()
    layer = fixture["layer"]
    source = layer.flickr_source_records.row(0, named=True)
    evidence = layer.model_evidence.sort("candidate_rank").row(0, named=True)
    audit = build_dynamic_pool_audit_frame(
        [
            {
                "sampling_unit_id": "review-unit-flickr-photo-1",
                "source_record_hash": source["source_record_fingerprint"],
                "source_artifact_fingerprint": source["source_snapshot_fingerprint"],
                "flickr_photo_id": source["flickr_photo_id"],
                "organism_unit_id": source["organism_unit_id"],
                "candidate_family_accepted_taxon_key": "gbif:9417",
                "candidate_family_scientific_name": "Papilionidae",
                "candidate_genus_accepted_taxon_key": "gbif:1920494",
                "candidate_genus_scientific_name": "Papilio",
                "candidate_species_accepted_taxon_key": evidence[
                    "candidate_accepted_taxon_key"
                ],
                "candidate_species_scientific_name": evidence[
                    "candidate_scientific_name"
                ],
                "geographic_cluster_id": "geo-au-qld",
                "no_geo": False,
                "primary_query_tier": "T2",
                "raw_fusion_score": evidence["fused_raw_score"],
                "raw_competitor_margin": evidence["margin_to_next_raw"],
                "pool_disagreement": abs(
                    evidence["global_raw_component_score"]
                    - evidence["local_raw_component_score"]
                ),
                "route": "adult_field",
                "visual_domain": "field_photo",
                "subject_area_ratio": 0.08,
                "owner_group_id": source["owner_group_id"],
                "duplicate_group_id": source["duplicate_group_id"],
                "observation_group_id": source["observation_group_id"],
                "final_release_candidate": False,
            }
        ]
    )
    policy = ProbabilityAuditSamplingPolicy(review_budget=1, random_seed=17)
    selection = build_probability_audit_sample(audit, policy=policy)
    return {**fixture, "selection": selection, "sampling_policy": policy}


def build_butterflylens_complete_fixture() -> dict[str, object]:
    """Build all ten role inputs for package and consumer-contract tests."""

    fixture = build_butterflylens_review_fixture()
    source_id = fixture["layer"].flickr_source_records["flickr_record_id"][0]
    geographic = build_butterflylens_geographic_impact(
        model_layer=fixture["layer"],
        geographic_records=[
            {
                "flickr_record_id": source_id,
                "geography_availability": "h3",
                "h3_cell": "8928308280fffff",
                "h3_version": "4.3.0",
                "h3_resolution": 9,
                "source_precision_metres": 20.0,
                "published_h3_resolution": 9,
                "public_geometry_status": "available",
                "public_geometry_reason": None,
                "latest_flickr_event_date": "2026-07-17",
                "geographic_evidence_fingerprint": sha("1"),
            }
        ],
        source_commit="1" * 40,
    )
    review = build_butterflylens_review_layer(
        project=fixture["project"],
        run=fixture["run"],
        model_layer=fixture["layer"],
        selection=fixture["selection"],
        sampling_policy=fixture["sampling_policy"],
        target={
            "accepted_taxon_key": "gbif:5131359",
            "scientific_name": "Papilio demoleus",
            "rank": "species",
        },
        observed_at="2026-07-18T12:03:00+10:00",
    )
    return {**fixture, "geographic": geographic, "review": review}


__all__ = [
    "build_butterflylens_model_fixture",
    "build_butterflylens_review_fixture",
    "build_butterflylens_complete_fixture",
    "sha",
]
