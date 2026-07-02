from __future__ import annotations

import json
from pathlib import Path

import polars as pl

from biominer.bioclip.ablation import build_ablation_report, run_object_ablations
from biominer.bioclip.candidate_sets import build_candidate_set
from biominer.bioclip.object_runner import (
    FakeObjectBioClipScorer,
    apply_geospatial_soft_prior,
    screen_object_detections,
    write_object_evidence_outputs,
)
from biominer.species.context import CommonName, RegionHint, SpeciesContext


def _context() -> SpeciesContext:
    return SpeciesContext(
        scientific_name="Danaus plexippus",
        accepted_taxon_key="gbif:5131654",
        canonical_name="Danaus plexippus",
        family="Nymphalidae",
        genus="Danaus",
        family_key="gbif:7017",
        genus_key="gbif:1927164",
        species_key="gbif:5131654",
        registry_version="registry-v1",
        synonyms=("Anosia plexippus",),
        common_names=(CommonName(name="monarch butterfly", language="en", source="gbif"),),
        regions=(RegionHint(region="North America", bbox="-170.0,5.0,-50.0,75.0", source="fixture"),),
    )


def _canonical_records() -> pl.DataFrame:
    return pl.DataFrame(
        [
            {
                "source": "flickr",
                "flickr_photo_id": "photo-1",
                "source_record_hash": "sha256:source-1",
                "image_url": "https://live.staticflickr.com/photo-1.jpg",
                "photo_page_url": "https://www.flickr.com/photos/u/photo-1",
                "title": "monarch butterfly on milkweed",
                "raw_tags": "monarch Danaus plexippus",
                "latitude": 45.0,
                "longitude": -93.0,
                "date_taken": "2024-07-01",
            }
        ]
    )


def _detections() -> pl.DataFrame:
    return pl.DataFrame(
        [
            {
                "source": "flickr",
                "flickr_photo_id": "photo-1",
                "source_record_hash": "sha256:source-1",
                "image_url": "https://live.staticflickr.com/photo-1.jpg",
                "photo_page_url": "https://www.flickr.com/photos/u/photo-1",
                "detection_id": "det-1",
                "detector_backend": "fake",
                "detector_model_id": "fake-detector",
                "detector_model_version": "v1",
                "detector_checkpoint": "checkpoint-a",
                "detected_at": "2026-01-01T00:00:00+00:00",
                "bbox_xyxy": [0.0, 0.0, 10.0, 10.0],
                "bbox_xyxyn": [0.0, 0.0, 0.5, 0.5],
                "bbox_xywhn": [0.25, 0.25, 0.5, 0.5],
                "box_area_ratio": 0.25,
                "detector_label": "butterfly",
                "detector_score": 0.9,
                "objectness_score": 0.9,
                "nms_group_id": None,
                "crop_padding_ratio": 0.12,
                "crop_hash": "sha256:crop-1",
                "crop_width": 336,
                "crop_height": 336,
                "crop_storage_policy": "ephemeral",
                "detection_status": "detected",
                "failure_reason": None,
            },
            {
                "source": "flickr",
                "flickr_photo_id": "photo-1",
                "source_record_hash": "sha256:source-1",
                "image_url": "https://live.staticflickr.com/photo-1.jpg",
                "photo_page_url": "https://www.flickr.com/photos/u/photo-1",
                "detection_id": "det-2",
                "detector_backend": "fake",
                "detector_model_id": "fake-detector",
                "detector_model_version": "v1",
                "detector_checkpoint": "checkpoint-a",
                "detected_at": "2026-01-01T00:00:00+00:00",
                "bbox_xyxy": [10.0, 10.0, 20.0, 20.0],
                "bbox_xyxyn": [0.5, 0.5, 1.0, 1.0],
                "bbox_xywhn": [0.75, 0.75, 0.5, 0.5],
                "box_area_ratio": 0.25,
                "detector_label": "butterfly",
                "detector_score": 0.6,
                "objectness_score": 0.6,
                "nms_group_id": None,
                "crop_padding_ratio": 0.12,
                "crop_hash": "sha256:crop-2",
                "crop_width": 336,
                "crop_height": 336,
                "crop_storage_policy": "ephemeral",
                "detection_status": "detected",
                "failure_reason": None,
            },
        ]
    )


def test_candidate_set_uses_species_context_and_same_genus_family_candidates(tmp_path) -> None:
    candidates = tmp_path / "species_candidates.parquet"
    pl.DataFrame(
        [
            {"scientific_name": "Danaus plexippus", "accepted_taxon_key": "gbif:5131654", "family": "Nymphalidae", "genus": "Danaus"},
            {"scientific_name": "Danaus gilippus", "accepted_taxon_key": "gbif:5131655", "family": "Nymphalidae", "genus": "Danaus"},
            {"scientific_name": "Limenitis archippus", "accepted_taxon_key": "gbif:1900000", "family": "Nymphalidae", "genus": "Limenitis"},
            {"scientific_name": "Papilio polyxenes", "accepted_taxon_key": "gbif:1900001", "family": "Papilionidae", "genus": "Papilio"},
        ]
    ).write_parquet(candidates)

    candidate_set = build_candidate_set(_context(), species_candidate_path=candidates)

    assert candidate_set.target_scientific_name == "Danaus plexippus"
    assert candidate_set.target_accepted_taxon_key == "gbif:5131654"
    assert candidate_set.candidate_set_id.startswith("sha256:")
    assert [candidate.scientific_name for candidate in candidate_set.species_candidates] == [
        "Danaus plexippus",
        "Danaus gilippus",
        "Limenitis archippus",
    ]
    assert [candidate.scientific_name for candidate in candidate_set.genus_candidates] == ["Danaus plexippus", "Danaus gilippus"]
    assert "a photo of Danaus plexippus" in candidate_set.prompt_labels("species")
    assert "monarch butterfly" in candidate_set.prompt_labels("species")


def test_object_bioclip_scores_detection_crops_with_join_keys(tmp_path) -> None:
    candidate_set = build_candidate_set(_context())
    scorer = FakeObjectBioClipScorer(
        scores_by_crop={
            "sha256:crop-1": {
                "a photo of Danaus plexippus": 0.82,
                "Danaus plexippus": 0.75,
                "monarch butterfly": 0.7,
            },
            "sha256:crop-2": {"a photo of Danaus plexippus": 0.31},
        }
    )
    output = tmp_path / "object_bioclip_scores.parquet"

    result = screen_object_detections(
        canonical_records=_canonical_records(),
        detections=_detections(),
        species_context=_context(),
        candidate_set=candidate_set,
        scorer=scorer,
        output_path=output,
        ablation_mode="detector_crop",
    )

    assert output.exists()
    assert result.frame.height == 2
    row = result.frame.sort("detection_id").to_dicts()[0]
    assert row["source"] == "flickr"
    assert row["flickr_photo_id"] == "photo-1"
    assert row["detection_id"] == "det-1"
    assert row["crop_hash"] == "sha256:crop-1"
    assert row["candidate_set_id"] == candidate_set.candidate_set_id
    assert row["ablation_mode"] == "detector_crop"
    assert row["species_top5"][0] == "Danaus plexippus"
    assert row["target_species_rank"] == 1
    assert row["target_species_score"] == 0.82
    assert row["occurrence_bin"] == "gold"
    assert row["is_target_positive"] is True


def test_geography_soft_prior_routes_conflict_to_review_without_discarding() -> None:
    context = _context()
    in_range = apply_geospatial_soft_prior({"latitude": 45.0, "longitude": -93.0}, context, visual_score=0.8)
    out_of_range = apply_geospatial_soft_prior({"latitude": -35.0, "longitude": 149.0}, context, visual_score=0.8)
    missing = apply_geospatial_soft_prior({"latitude": None, "longitude": None}, context, visual_score=0.8)

    assert in_range.score > 0
    assert in_range.reason == "within_context_region"
    assert out_of_range.score < 0
    assert out_of_range.reason == "geospatial_conflict"
    assert out_of_range.route_to_review is True
    assert out_of_range.hard_discard is False
    assert missing.reason == "missing_geo"
    assert missing.hard_discard is False


def test_ablation_modes_write_rows_with_shared_photo_join_keys(tmp_path) -> None:
    candidate_set = build_candidate_set(_context())
    report = run_object_ablations(
        canonical_records=_canonical_records(),
        detections=_detections().head(1),
        species_context=_context(),
        candidate_set=candidate_set,
        scorer=FakeObjectBioClipScorer({"sha256:crop-1": {"a photo of Danaus plexippus": 0.82}}),
        output_dir=tmp_path,
        modes=("whole_image", "detector_crop", "detector_crop_segmentation"),
    )

    frames = [pl.read_parquet(tmp_path / f"object_bioclip_scores_{mode}.parquet") for mode in report.modes]
    rows = pl.concat(frames).sort("ablation_mode").to_dicts()

    assert {row["ablation_mode"] for row in rows} == {"whole_image", "detector_crop", "detector_crop_segmentation"}
    assert {row["source"] for row in rows} == {"flickr"}
    assert {row["flickr_photo_id"] for row in rows} == {"photo-1"}
    assert build_ablation_report(pl.concat(frames))["crops_scored"] == 3
    assert build_ablation_report(pl.concat(frames))["gold_count"] == 3


def test_object_evidence_join_and_photo_summary_outputs(tmp_path) -> None:
    candidate_set = build_candidate_set(_context())
    scores = screen_object_detections(
        canonical_records=_canonical_records(),
        detections=_detections(),
        species_context=_context(),
        candidate_set=candidate_set,
        scorer=FakeObjectBioClipScorer(
            {
                "sha256:crop-1": {"a photo of Danaus plexippus": 0.82},
                "sha256:crop-2": {"a photo of Danaus plexippus": 0.44},
            }
        ),
        output_path=tmp_path / "scores.parquet",
        ablation_mode="detector_crop",
    )
    canonical_path = tmp_path / "canonical.parquet"
    detections_path = tmp_path / "detections.parquet"
    scores_path = tmp_path / "scores.parquet"
    joined_path = tmp_path / "object_evidence_joined.parquet"
    summary_path = tmp_path / "photo_evidence_summary.parquet"
    _canonical_records().write_parquet(canonical_path)
    _detections().write_parquet(detections_path)
    scores.frame.write_parquet(scores_path)

    outputs = write_object_evidence_outputs(
        canonical_records_path=canonical_path,
        detections_path=detections_path,
        scores_path=scores_path,
        joined_output_path=joined_path,
        photo_summary_output_path=summary_path,
    )

    joined = pl.read_parquet(outputs.object_evidence_joined)
    summary = pl.read_parquet(outputs.photo_evidence_summary).to_dicts()[0]
    assert joined.height == 2
    assert {"source", "flickr_photo_id", "detection_id", "crop_hash"}.issubset(joined.columns)
    assert summary["best_detection_id"] == "det-1"
    assert summary["detection_count"] == 2
    assert summary["photo_occurrence_bin"] == "gold"
    assert summary["all_detection_ids"] == ["det-1", "det-2"]
    assert summary["all_candidate_species"] == ["Danaus plexippus"]


def test_detection_object_pipeline_has_no_hardcoded_species_labels() -> None:
    root = Path("src/biominer")
    forbidden = ("Papilio demoleus", "TARGET_SPECIES", "PAPILIO_DEMOLEUS", "monarch butterfly")
    allowed = {
        Path("src/biominer/species/context.py"),
    }
    offenders: list[str] = []
    for path in [*root.glob("detection/*.py"), root / "bioclip" / "object_runner.py", root / "bioclip" / "candidate_sets.py"]:
        if path in allowed or not path.exists():
            continue
        text = path.read_text(encoding="utf-8")
        offenders.extend(f"{path}: {token}" for token in forbidden if token in text)

    assert offenders == []


def test_species_context_round_trip_for_object_pipeline(tmp_path) -> None:
    path = tmp_path / "species_context.json"
    _context().write_json(path)

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["scientific_name"] == "Danaus plexippus"
    assert SpeciesContext.read_json(path).target_terms()[0] == "Danaus plexippus"
