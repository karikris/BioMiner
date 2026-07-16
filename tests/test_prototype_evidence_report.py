from __future__ import annotations

import hashlib
import json
from pathlib import Path

import polars as pl
import pytest

from biominer.cli import build_parser
from biominer.reports.prototype_evidence import (
    PROTOTYPE_EVIDENCE_CONFIG_VERSION,
    PrototypeEvidenceConfig,
    build_prototype_evidence_outputs,
)


TARGET_KEY = "gbif:1938069"
TARGET_NAME = "Papilio demoleus"
SCORE_SEMANTICS = "experimental_screening_evidence_uncalibrated_not_probability"


def test_prototype_evidence_outputs_build_dashboard_and_explanation(
    tmp_path: Path,
) -> None:
    config = _fixture(tmp_path)

    result = build_prototype_evidence_outputs(config)

    dashboard = pl.read_parquet(result.dashboard_path)
    references = pl.read_parquet(result.references_path)
    competitors = pl.read_parquet(result.competitors_path)
    row = dashboard.row(0, named=True)
    panel = json.loads(row["why_this_image_was_ranked_json"])

    assert result.report["status"] == "passed"
    assert result.report["deployment_status"] == "prototype"
    assert result.report["counts"]["dashboard_rows"] == 1
    assert row["target_species"] == TARGET_NAME
    assert row["target_reference_trust_levels"] == ["R4"]
    assert row["geographic_layer"] == "A"
    assert row["reference_layer"] == f"target_adult:{TARGET_KEY}"
    assert row["yoloe_route"] == "adult_butterfly_field"
    assert row["abstention"] is True
    assert panel["target_similarity"] == pytest.approx(0.72)
    assert panel["best_competitor_similarity"] == pytest.approx(0.70)
    assert panel["competitor_margin"] == pytest.approx(0.02)
    assert panel["closest_target_references"][0]["trust_level"] == "R4"
    assert (
        panel["closest_competitor_references"][0]["scientific_name"]
        == "Graphium agamemnon"
    )
    assert competitors.height == 1
    assert references.height == 2
    assert "Why this image was ranked" in result.summary_path.read_text(
        encoding="utf-8"
    )
    for frame in (dashboard, references, competitors):
        assert "image_url" not in frame.columns
        assert "source_object_uri" not in frame.columns


def test_prototype_evidence_rejects_query_hits_as_labels(tmp_path: Path) -> None:
    config = _fixture(tmp_path, query_match_is_label=True)

    with pytest.raises(ValueError, match="classification evidence invariants"):
        build_prototype_evidence_outputs(config)


def test_prototype_evidence_config_is_local_only(tmp_path: Path) -> None:
    config_path = _config_path(
        tmp_path,
        overrides={"classifications": "s3://bucket/classifications.parquet"},
    )

    with pytest.raises(ValueError, match="classifications must be a local path"):
        PrototypeEvidenceConfig.read_json(config_path)


def test_prototype_evidence_cli_is_publicly_selectable() -> None:
    args = build_parser().parse_args(
        [
            "bioclip",
            "prototype-evidence",
            "--config",
            "config/pilot/papilio_demoleus_prototype_evidence.json",
        ]
    )

    assert args.bioclip_command == "prototype-evidence"
    assert args.config.endswith("papilio_demoleus_prototype_evidence.json")


def _fixture(
    tmp_path: Path,
    *,
    query_match_is_label: bool = False,
) -> PrototypeEvidenceConfig:
    classifications_path = tmp_path / "classifications.parquet"
    candidates_path = tmp_path / "candidates.parquet"
    support_path = tmp_path / "support.parquet"
    nearest = [
        {
            "accepted_taxon_key": TARGET_KEY,
            "reference_group": f"target_adult:{TARGET_KEY}",
            "reference_media_id": "reference-target",
            "scientific_name": TARGET_NAME,
            "similarity": 0.72,
        },
        {
            "accepted_taxon_key": "gbif:5141187",
            "reference_group": "competitor:gbif:5141187",
            "reference_media_id": "reference-competitor",
            "scientific_name": "Graphium agamemnon",
            "similarity": 0.70,
        },
    ]
    pl.DataFrame(
        {
            "order_index": [1],
            "flickr_photo_id": ["photo-1"],
            "target_scientific_name": [TARGET_NAME],
            "target_accepted_taxon_key": [TARGET_KEY],
            "target_scored": [True],
            "regional_candidate_count": [2],
            "regional_scored_count": [2],
            "geo_cluster_id": ["cluster-1"],
            "coordinate_quality": ["flickr_city"],
            "detection_route": ["adult_butterfly_field"],
            "bioclip_route": ["adult_field"],
            "reference_route_used": ["adult_field"],
            "visual_input": ["raw_full_image"],
            "complete_canvas_retained": [True],
            "spatial_crop_applied": [False],
            "higher_rank_pruning_applied": [False],
            "nearest_target_reference_similarity": [0.72],
            "best_competitor_reference_similarity": [0.70],
            "best_competitor_reference_key": ["gbif:5141187"],
            "best_competitor_reference_name": ["Graphium agamemnon"],
            "target_competitor_reference_margin": [0.02],
            "target_text_similarity": [0.4],
            "prototype_score": [0.61],
            "uncalibrated_margin": [0.01],
            "abstain": [True],
            "abstention_reason": ["raw_margin_below_threshold"],
            "prototype_status": ["prototype_only"],
            "score_semantics": [SCORE_SEMANTICS],
            "experimental_screening_evidence": [True],
            "flickr_query_match_is_label": [query_match_is_label],
            "nearest_references_json": [json.dumps(nearest)],
        }
    ).write_parquet(classifications_path)
    pl.DataFrame(
        {
            "order_index": [1, 1],
            "flickr_photo_id": ["photo-1", "photo-1"],
            "class_kind": ["species", "species"],
            "class_id": [TARGET_KEY, "gbif:5141187"],
            "accepted_taxon_key": [TARGET_KEY, "gbif:5141187"],
            "display_name": [TARGET_NAME, "Graphium agamemnon"],
            "candidate_reason": ["target", "regional_same_family"],
            "target_candidate": [True, False],
            "text_similarity": [0.4, 0.42],
            "reference_prototype_similarity": [0.7, 0.69],
            "prototype_score": [0.61, 0.60],
            "rank": [1, 2],
            "score_semantics": [SCORE_SEMANTICS, SCORE_SEMANTICS],
            "experimental_screening_evidence": [True, True],
        }
    ).write_parquet(candidates_path)
    pl.DataFrame(
        {
            "reference_media_id": ["reference-target", "reference-competitor"],
            "provider_media_id": ["provider-target", "provider-competitor"],
            "accepted_taxon_key": [TARGET_KEY, "gbif:5141187"],
            "scientific_name": [TARGET_NAME, "Graphium agamemnon"],
            "trust_level": ["R4", "R4"],
            "verification_status": ["provider_supported", "provider_supported"],
            "human_verified": [False, False],
            "geographic_layer": ["A", "D"],
            "geo_cluster_id": ["cluster-1", "cluster-2"],
            "route": ["adult_field", "adult_field"],
            "reference_group": [
                f"target_adult:{TARGET_KEY}",
                "competitor:gbif:5141187",
            ],
            "licence": ["CC-BY-NC-4.0", "CC-BY-NC-4.0"],
            "licence_policy_status": ["research_only", "research_only"],
            "attribution": ["Target photographer", "Competitor photographer"],
        }
    ).write_parquet(support_path)
    return PrototypeEvidenceConfig(
        target_accepted_taxon_key=TARGET_KEY,
        target_scientific_name=TARGET_NAME,
        classifications=classifications_path,
        classifications_sha256=_sha(classifications_path),
        candidate_scores=candidates_path,
        candidate_scores_sha256=_sha(candidates_path),
        support_manifest=support_path,
        support_manifest_sha256=_sha(support_path),
        output_dir=tmp_path / "output",
        verification_limitations=("Prototype evidence only.",),
    )


def _config_path(
    tmp_path: Path,
    *,
    overrides: dict[str, object],
) -> Path:
    config = _fixture(tmp_path)
    payload = {
        "schema_version": PROTOTYPE_EVIDENCE_CONFIG_VERSION,
        "target_accepted_taxon_key": config.target_accepted_taxon_key,
        "target_scientific_name": config.target_scientific_name,
        "classifications": str(config.classifications),
        "classifications_sha256": config.classifications_sha256,
        "candidate_scores": str(config.candidate_scores),
        "candidate_scores_sha256": config.candidate_scores_sha256,
        "support_manifest": str(config.support_manifest),
        "support_manifest_sha256": config.support_manifest_sha256,
        "output_dir": str(config.output_dir),
        "verification_limitations": list(config.verification_limitations),
    }
    payload.update(overrides)
    path = tmp_path / "config.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _sha(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
