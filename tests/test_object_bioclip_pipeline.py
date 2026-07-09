from __future__ import annotations

import json
from pathlib import Path

import polars as pl
import pytest

from biominer.bioclip.ablation import build_ablation_report, run_object_ablations
from biominer.bioclip.candidate_sets import CandidateTaxon, build_candidate_set, build_candidate_set_for_taxon_scope
from biominer.bioclip.object_runner import (
    CachedObjectEmbeddingScorer,
    EphemeralCropBioClipScorer,
    FakeObjectBioClipScorer,
    OBJECT_VISUAL_MODES,
    OBJECT_SCORE_OUTPUT_SCHEMA,
    PRIMARY_VISUAL_CLASSIFIER,
    TARGET_SCOPE_CANDIDATE_SELECTION_MODE,
    TARGET_SCOPE_SPECIES_RERANK_STRATEGY,
    apply_geospatial_soft_prior,
    empty_object_score_frame,
    iter_materialized_detector_crop_batches,
    materialize_detector_crop_inputs,
    screen_object_detections,
    write_object_evidence_outputs,
)
from biominer.detection.detector_base import DecodedImage
from biominer.detection.policy import DetectionPolicy
from biominer.detection.segmentation import make_segmenter
from biominer.detection.schema import DETECTION_OUTPUT_SCHEMA, empty_detection_frame
from biominer.run.taxon_scope import TaxonScope
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


def _fixture_candidate_set():
    return build_candidate_set(_context(), allow_single_target_fixture=True)


def test_object_visual_modes_are_segmentation_not_enhancement() -> None:
    assert OBJECT_VISUAL_MODES == ("whole_image", "detector_crop", "detector_crop_segmentation")

    source_paths = (
        Path("src/biominer/bioclip/object_runner.py"),
        Path("src/biominer/bioclip/ablation.py"),
        Path("src/biominer/detection/segmentation.py"),
        Path("src/biominer/cli.py"),
    )
    forbidden_terms = ("enhancement", "enhance", "superres", "super-res", "super_resolution", "sharpen")

    violations: dict[str, list[str]] = {}
    for path in source_paths:
        source = path.read_text(encoding="utf-8").casefold()
        matches = [term for term in forbidden_terms if term in source]
        if matches:
            violations[str(path)] = matches

    assert violations == {}


def _species_context(
    scientific_name: str,
    accepted_taxon_key: str,
    *,
    family: str = "Nymphalidae",
    genus: str = "Danaus",
    family_key: str = "gbif:7017",
    genus_key: str = "gbif:1927164",
) -> SpeciesContext:
    return SpeciesContext(
        scientific_name=scientific_name,
        accepted_taxon_key=accepted_taxon_key,
        canonical_name=scientific_name,
        family=family,
        genus=genus,
        family_key=family_key,
        genus_key=genus_key,
        species_key=accepted_taxon_key,
        registry_version="registry-v1",
        synonyms=(),
        common_names=(),
        regions=(),
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
                "detector_label": "butterfly_like",
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
                "detector_label": "butterfly_like",
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


def _decoded_image() -> DecodedImage:
    pixels = bytes(
        value
        for y in range(4)
        for x in range(4)
        for value in ((x * 40) % 256, (y * 40) % 256, ((x + y) * 20) % 256)
    )
    return DecodedImage(width=4, height=4, mode="RGB", data=pixels, source_uri="memory://photo-1")


def test_make_segmenter_defaults_to_optional_noop_backend() -> None:
    segmenter = make_segmenter("none")

    assert segmenter.backend == "none"
    assert segmenter.segment_crop(None) is None  # type: ignore[arg-type]


def test_detector_masks_are_not_persisted_in_detection_schema() -> None:
    assert not any("mask" in column or "segmentation" in column for column in DETECTION_OUTPUT_SCHEMA)


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


def test_candidate_set_requires_expansion_unless_fixture_mode() -> None:
    with pytest.raises(ValueError, match="registry-derived"):
        build_candidate_set(_context())

    candidate_set = build_candidate_set(_context(), allow_single_target_fixture=True)

    assert [candidate.scientific_name for candidate in candidate_set.species_candidates] == ["Danaus plexippus"]
    assert "single_target_fixture" in candidate_set.source_evidence


def test_candidate_set_for_family_scope_uses_all_scope_species(tmp_path) -> None:
    candidates = tmp_path / "species_candidates.parquet"
    pl.DataFrame(
        [
            {
                "scientific_name": "Limenitis arthemis",
                "accepted_taxon_key": "gbif:1900002",
                "family": "Nymphalidae",
                "family_key": "gbif:7017",
                "genus": "Limenitis",
                "genus_key": "gbif:1910000",
            },
            {
                "scientific_name": "Papilio polyxenes",
                "accepted_taxon_key": "gbif:1900001",
                "family": "Papilionidae",
                "family_key": "gbif:5506",
                "genus": "Papilio",
                "genus_key": "gbif:1920000",
            },
        ]
    ).write_parquet(candidates)
    scope = TaxonScope(
        input_name="Nymphalidae",
        input_rank="family",
        accepted_taxon_key="gbif:7017",
        accepted_scientific_name="Nymphalidae",
        accepted_rank="family",
        registry_version="registry-v1",
        species_contexts=(
            _context(),
            _species_context("Limenitis archippus", "gbif:1900000", genus="Limenitis", genus_key="gbif:1910000"),
        ),
    )

    candidate_set = build_candidate_set_for_taxon_scope(scope, species_candidate_path=candidates)

    assert candidate_set.target_scientific_name == "Nymphalidae"
    assert candidate_set.target_accepted_taxon_key == "gbif:7017"
    assert [candidate.scientific_name for candidate in candidate_set.species_candidates] == [
        "Danaus plexippus",
        "Limenitis archippus",
        "Limenitis arthemis",
    ]
    assert "Papilio polyxenes" not in [candidate.scientific_name for candidate in candidate_set.species_candidates]
    assert "taxon_scope:family" in candidate_set.source_evidence


def test_candidate_set_for_genus_scope_uses_all_scope_species_and_genus_rows(tmp_path) -> None:
    candidates = tmp_path / "species_candidates.parquet"
    pl.DataFrame(
        [
            {
                "scientific_name": "Danaus eresimus",
                "accepted_taxon_key": "gbif:1900003",
                "family": "Nymphalidae",
                "family_key": "gbif:7017",
                "genus": "Danaus",
                "genus_key": "gbif:1927164",
            },
            {
                "scientific_name": "Limenitis archippus",
                "accepted_taxon_key": "gbif:1900000",
                "family": "Nymphalidae",
                "family_key": "gbif:7017",
                "genus": "Limenitis",
                "genus_key": "gbif:1910000",
            },
        ]
    ).write_parquet(candidates)
    scope = TaxonScope(
        input_name="Danaus",
        input_rank="genus",
        accepted_taxon_key="gbif:1927164",
        accepted_scientific_name="Danaus",
        accepted_rank="genus",
        registry_version="registry-v1",
        species_contexts=(
            _context(),
            _species_context("Danaus gilippus", "gbif:5131655"),
        ),
    )

    candidate_set = build_candidate_set_for_taxon_scope(scope, species_candidate_path=candidates)

    assert candidate_set.target_scientific_name == "Danaus"
    assert [candidate.scientific_name for candidate in candidate_set.species_candidates] == [
        "Danaus plexippus",
        "Danaus gilippus",
        "Danaus eresimus",
    ]
    assert "Limenitis archippus" not in [candidate.scientific_name for candidate in candidate_set.species_candidates]
    assert "taxon_scope:genus" in candidate_set.source_evidence


def test_candidate_set_for_species_scope_requires_registry_expansion_unless_fixture_mode(tmp_path) -> None:
    scope = TaxonScope.from_species_context(_context())

    with pytest.raises(ValueError, match="registry-derived"):
        build_candidate_set_for_taxon_scope(scope)

    fixture = build_candidate_set_for_taxon_scope(scope, allow_single_target_fixture=True)
    assert [candidate.scientific_name for candidate in fixture.species_candidates] == ["Danaus plexippus"]

    candidates = tmp_path / "species_candidates.parquet"
    pl.DataFrame(
        [
            {
                "scientific_name": "Danaus gilippus",
                "accepted_taxon_key": "gbif:5131655",
                "family": "Nymphalidae",
                "genus": "Danaus",
            },
            {
                "scientific_name": "Limenitis archippus",
                "accepted_taxon_key": "gbif:1900000",
                "family": "Nymphalidae",
                "genus": "Limenitis",
            },
        ]
    ).write_parquet(candidates)

    candidate_set = build_candidate_set_for_taxon_scope(scope, species_candidate_path=candidates)

    assert [candidate.scientific_name for candidate in candidate_set.species_candidates] == [
        "Danaus plexippus",
        "Danaus gilippus",
        "Limenitis archippus",
    ]
    assert "taxon_scope:species" in candidate_set.source_evidence


def test_candidate_set_reads_list_valued_common_names_from_candidate_parquet(tmp_path) -> None:
    candidates = tmp_path / "species_candidates.parquet"
    pl.DataFrame(
        [
            {
                "scientific_name": "Danaus gilippus",
                "accepted_taxon_key": "gbif:5131655",
                "family": "Nymphalidae",
                "genus": "Danaus",
                "common_names": ["queen butterfly", "queen"],
            },
        ],
        schema={
            "scientific_name": pl.String,
            "accepted_taxon_key": pl.String,
            "family": pl.String,
            "genus": pl.String,
            "common_names": pl.List(pl.String),
        },
    ).write_parquet(candidates)

    labels = build_candidate_set(_context(), species_candidate_path=candidates).prompt_labels("species")

    assert "queen butterfly" in labels
    assert "queen" in labels
    assert "['queen butterfly', 'queen']" not in labels


def test_candidate_set_id_changes_when_prompt_common_names_change(tmp_path) -> None:
    first_candidates = tmp_path / "species_candidates_first.parquet"
    second_candidates = tmp_path / "species_candidates_second.parquet"
    base_row = {
        "scientific_name": "Danaus gilippus",
        "accepted_taxon_key": "gbif:5131655",
        "family": "Nymphalidae",
        "genus": "Danaus",
    }
    pl.DataFrame([{**base_row, "common_names": ["queen butterfly"]}]).write_parquet(first_candidates)
    pl.DataFrame([{**base_row, "common_names": ["southern queen"]}]).write_parquet(second_candidates)

    first = build_candidate_set(_context(), species_candidate_path=first_candidates)
    second = build_candidate_set(_context(), species_candidate_path=second_candidates)

    assert "queen butterfly" in first.prompt_labels("species")
    assert "southern queen" in second.prompt_labels("species")
    assert first.candidate_set_id != second.candidate_set_id


def test_candidate_set_records_geospatial_scope_source_evidence() -> None:
    candidate_set = build_candidate_set(_context(), geospatial_scope="geo_prior.parquet", allow_single_target_fixture=True)

    assert candidate_set.geospatial_scope == "geo_prior.parquet"
    assert "geospatial_scope:geo_prior.parquet" in candidate_set.source_evidence


def test_candidate_set_uses_geospatial_prior_table_candidates() -> None:
    candidate_set = build_candidate_set(
        _context(),
        geospatial_scope="geo_prior.parquet",
        geo_prior_table=pl.DataFrame(
            [
                {
                    "scientific_name": "Danaus erippus",
                    "accepted_taxon_key": "gbif:1901234",
                    "family": "Nymphalidae",
                    "genus": "Danaus",
                    "bbox": "-75.0,-56.0,-34.0,-20.0",
                    "source": "fixture",
                },
                {
                    "scientific_name": "Papilio polyxenes",
                    "accepted_taxon_key": "gbif:1900001",
                    "family": "Papilionidae",
                    "genus": "Papilio",
                    "bbox": "-170.0,5.0,-50.0,75.0",
                    "source": "fixture",
                },
            ]
        ),
    )

    names = [candidate.scientific_name for candidate in candidate_set.species_candidates]
    assert names == ["Danaus plexippus", "Danaus erippus"]
    assert "geospatial_prior_table" in candidate_set.source_evidence


def test_candidate_set_uses_query_provenance_accepted_taxon_keys() -> None:
    candidate_set = build_candidate_set(
        _context(),
        records=[
            {
                "discovery_accepted_taxon_keys": ["gbif:999001"],
                "scientific_names_detected": ["Danaus erippus"],
            }
        ],
    )

    by_name = {candidate.scientific_name: candidate for candidate in candidate_set.species_candidates}
    assert by_name["Danaus erippus"].accepted_taxon_key == "gbif:999001"
    assert "query_provenance" in candidate_set.source_evidence
    assert "a photo of Danaus erippus" in candidate_set.prompt_labels("species")


def test_candidate_set_resolves_query_provenance_keys_from_candidate_parquet(tmp_path) -> None:
    candidates = tmp_path / "species_candidates.parquet"
    pl.DataFrame(
        [
            {"scientific_name": "Danaus plexippus", "accepted_taxon_key": "gbif:5131654", "family": "Nymphalidae", "genus": "Danaus"},
            {"scientific_name": "Pieris rapae", "accepted_taxon_key": "gbif:1005738", "family": "Pieridae", "genus": "Pieris"},
        ]
    ).write_parquet(candidates)

    candidate_set = build_candidate_set(
        _context(),
        species_candidate_path=candidates,
        records=[{"discovery_species_keys": ["gbif:1005738"]}],
    )

    by_name = {candidate.scientific_name: candidate for candidate in candidate_set.species_candidates}
    assert by_name["Pieris rapae"].accepted_taxon_key == "gbif:1005738"
    assert "query_provenance" in candidate_set.source_evidence
    assert "a photo of Pieris rapae" in candidate_set.prompt_labels("species")


def test_candidate_set_expands_query_provenance_genus_keys_from_candidate_parquet(tmp_path) -> None:
    candidates = tmp_path / "species_candidates.parquet"
    pl.DataFrame(
        [
            {
                "scientific_name": "Danaus plexippus",
                "accepted_taxon_key": "gbif:5131654",
                "family": "Nymphalidae",
                "family_key": "gbif:7017",
                "genus": "Danaus",
                "genus_key": "gbif:1927164",
            },
            {
                "scientific_name": "Pieris rapae",
                "accepted_taxon_key": "gbif:1005738",
                "family": "Pieridae",
                "family_key": "gbif:5481",
                "genus": "Pieris",
                "genus_key": "gbif:1924554",
            },
        ]
    ).write_parquet(candidates)

    candidate_set = build_candidate_set(
        _context(),
        species_candidate_path=candidates,
        records=[{"discovery_genus_keys": ["gbif:1924554"]}],
    )

    assert "Pieris rapae" in [candidate.scientific_name for candidate in candidate_set.species_candidates]
    assert "query_provenance" in candidate_set.source_evidence
    assert "Pieridae" in [candidate.family for candidate in candidate_set.family_candidates]
    assert "Pieris" in [candidate.genus for candidate in candidate_set.species_candidates]


def test_candidate_set_uses_metadata_scientific_names_without_query_keys() -> None:
    candidate_set = build_candidate_set(
        _context(),
        records=[
            {
                "scientific_names_detected": ["Danaus gilippus"],
            }
        ],
    )

    by_name = {candidate.scientific_name: candidate for candidate in candidate_set.species_candidates}
    assert by_name["Danaus gilippus"].accepted_taxon_key is None
    assert "metadata_text" in candidate_set.source_evidence
    assert "a photo of Danaus gilippus" in candidate_set.prompt_labels("species")


def test_candidate_set_uses_comment_species_candidates() -> None:
    candidate_set = build_candidate_set(
        _context(),
        records=[
            {
                "comments_text": "Local reviewer says this looks like Danaus eresimus, not monarch.",
                "comment_species_candidate": "Danaus eresimus",
            }
        ],
    )

    by_name = {candidate.scientific_name: candidate for candidate in candidate_set.species_candidates}
    assert by_name["Danaus eresimus"].genus == "Danaus"
    assert "comments" in candidate_set.source_evidence
    assert "a photo of Danaus eresimus" in candidate_set.prompt_labels("species")


def test_ephemeral_crop_bioclip_scorer_scores_temp_crop_and_deletes_file(tmp_path) -> None:
    seen: dict[str, object] = {}

    def scorer(path: Path, labels: tuple[str, ...]) -> dict[str, float]:
        data = path.read_bytes()
        seen["exists_during_score"] = path.exists()
        seen["suffix"] = path.suffix
        seen["header"] = data.split(b"\n", 3)[:3]
        seen["labels"] = labels
        return {label: (0.9 if label == "a photo of Danaus plexippus" else 0.1) for label in labels}

    crop_scorer = EphemeralCropBioClipScorer(
        scorer=scorer,
        image_loader=lambda item: _decoded_image(),
        temp_dir=tmp_path,
        crop_padding_ratio=0.25,
        crop_target_px=3,
        model_id="bioclip2_5",
        model_version="bioclip2_5_huge",
        model_checkpoint="checkpoint-a",
    )

    scores = crop_scorer.score(
        {
            "source": "flickr",
            "flickr_photo_id": "photo-1",
            "detection_id": "det-1",
            "bbox_xyxy": [0.0, 0.0, 3.0, 3.0],
        },
        ("a photo of Danaus plexippus", "a photo of Danaus gilippus"),
    )

    assert scores["a photo of Danaus plexippus"] == 0.9
    assert seen["exists_during_score"] is True
    assert seen["suffix"] == ".ppm"
    assert seen["header"] == [b"P6", b"3 3", b"255"]
    assert seen["labels"] == ("a photo of Danaus plexippus", "a photo of Danaus gilippus")
    assert list(tmp_path.iterdir()) == []


def test_ephemeral_crop_bioclip_scorer_batches_label_sets_and_deletes_files(tmp_path) -> None:
    calls: list[dict[str, object]] = []

    class BatchPathScorer:
        def score_label_sets_batch(self, image_paths, label_sets):  # noqa: ANN001, ANN202 - mirrors persistent scorer API.
            paths = [Path(path) for path in image_paths]
            calls.append(
                {
                    "paths": paths,
                    "existing": [path.exists() for path in paths],
                    "label_sets": {name: tuple(labels) for name, labels in label_sets.items()},
                }
            )
            return {
                name: [
                    {label: (0.9 if label == "a photo of Danaus plexippus" else 0.1) for label in labels}
                    for _path in paths
                ]
                for name, labels in label_sets.items()
            }

    crop_scorer = EphemeralCropBioClipScorer(
        scorer=BatchPathScorer(),
        image_loader=lambda item: _decoded_image(),
        temp_dir=tmp_path,
        crop_target_px=3,
        model_id="bioclip2_5",
        model_version="bioclip2_5_huge",
        model_checkpoint="checkpoint-a",
    )
    items = [
        {
            "source": "flickr",
            "flickr_photo_id": "photo-1",
            "detection_id": "det-1",
            "bbox_xyxy": [0.0, 0.0, 3.0, 3.0],
        },
        {
            "source": "flickr",
            "flickr_photo_id": "photo-1",
            "detection_id": "det-2",
            "bbox_xyxy": [1.0, 1.0, 4.0, 4.0],
        },
    ]

    scores = crop_scorer.score_label_sets_batch(
        items,
        {"species": ("a photo of Danaus plexippus", "a photo of Danaus gilippus")},
    )

    assert scores["species"][0]["a photo of Danaus plexippus"] == 0.9
    assert len(calls) == 1
    assert calls[0]["existing"] == [True, True]
    assert calls[0]["label_sets"] == {"species": ("a photo of Danaus plexippus", "a photo of Danaus gilippus")}
    assert list(tmp_path.iterdir()) == []


def test_cached_object_embedding_scorer_scores_from_text_and_crop_embeddings() -> None:
    scorer = CachedObjectEmbeddingScorer(
        text_embeddings=pl.DataFrame(
            [
                {
                    "candidate_set_id": "candidate-set-1",
                    "label": "Danaus plexippus",
                    "model_id": "bioclip",
                    "model_checkpoint": "checkpoint-a",
                    "embedding": [1.0, 0.0],
                },
                {
                    "candidate_set_id": "candidate-set-1",
                    "label": "Danaus gilippus",
                    "model_id": "bioclip",
                    "model_checkpoint": "checkpoint-a",
                    "embedding": [0.0, 1.0],
                },
            ]
        ),
        image_embeddings=pl.DataFrame(
            [
                {
                    "source": "flickr",
                    "flickr_photo_id": "photo-1",
                    "detection_id": "det-1",
                    "crop_hash": "sha256:crop-1",
                    "model_id": "bioclip",
                    "model_checkpoint": "checkpoint-a",
                    "embedding": [0.8, 0.2],
                }
            ]
        ),
        candidate_set_id="candidate-set-1",
        model_id="bioclip",
        model_version="test",
        model_checkpoint="checkpoint-a",
    )

    scores = scorer.score({"crop_hash": "sha256:crop-1"}, ("Danaus plexippus", "Danaus gilippus"))

    assert scores["Danaus plexippus"] > scores["Danaus gilippus"]
    assert round(scores["Danaus plexippus"], 6) == round(0.8 / ((0.8**2 + 0.2**2) ** 0.5), 6)


def test_materialized_object_embedding_crops_are_cleaned_when_image_loading_fails(tmp_path) -> None:
    calls = 0

    def flaky_loader(item: dict[str, object]) -> DecodedImage:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError(f"decode failed for {item['detection_id']}")
        return _decoded_image()

    with pytest.raises(RuntimeError, match="decode failed"):
        materialize_detector_crop_inputs(
            canonical_records=_canonical_records(),
            detections=_detections(),
            image_loader=flaky_loader,
            temp_dir=tmp_path,
            crop_target_px=3,
        )

    assert not (tmp_path / ".object_image_embedding_cache.tmp").exists()


def test_materialized_detector_crop_batches_default_to_24_and_clean_between_batches(tmp_path) -> None:
    base_detection = _detections().to_dicts()[0]
    detections = pl.DataFrame(
        [
            {
                **base_detection,
                "detection_id": f"det-{index:02d}",
                "crop_hash": f"sha256:crop-{index:02d}",
            }
            for index in range(25)
        ]
    )
    loaded_for: list[str] = []

    def image_loader(item: dict[str, object]) -> DecodedImage:
        loaded_for.append(str(item["detection_id"]))
        return _decoded_image()

    batch_sizes: list[int] = []
    batch_dirs: list[Path] = []
    previous_dir: Path | None = None
    for batch in iter_materialized_detector_crop_batches(
        canonical_records=_canonical_records(),
        detections=detections,
        image_loader=image_loader,
        temp_dir=tmp_path,
        crop_target_px=3,
    ):
        if previous_dir is not None:
            assert not previous_dir.exists()
        assert batch.temp_dir.exists()
        assert all(path.exists() for path in batch.crop_paths)
        assert all(item["crop_path"].exists() for item in batch.items)
        assert all(Path(row["crop_path"]).exists() for row in batch.rows)
        assert all(item["title"] == "monarch butterfly on milkweed" for item in batch.items)
        assert all(item["crop_padding_ratio"] == 0.08 for item in batch.items)
        assert all(item["crop_width"] == 3 and item["crop_height"] == 3 for item in batch.items)
        batch_sizes.append(len(batch.items))
        batch_dirs.append(batch.temp_dir)
        previous_dir = batch.temp_dir

    assert batch_sizes == [24, 1]
    assert loaded_for == ["det-00", "det-24"]
    assert all(not path.exists() for path in batch_dirs)


def test_materialized_detector_crop_batches_skip_noneligible_without_image_load(tmp_path) -> None:
    eligible = _detections().to_dicts()[0]
    noneligible = {
        **eligible,
        "flickr_photo_id": "photo-without-canonical-record",
        "detection_id": "moth-1",
        "detector_label": "moth_like",
        "bbox_xyxy": [],
    }
    loaded_for: list[str] = []
    seen_batches: list[list[str]] = []

    def image_loader(item: dict[str, object]) -> DecodedImage:
        loaded_for.append(str(item["detection_id"]))
        return _decoded_image()

    for batch in iter_materialized_detector_crop_batches(
        canonical_records=_canonical_records(),
        detections=pl.DataFrame([noneligible, eligible]),
        image_loader=image_loader,
        temp_dir=tmp_path,
        detection_policy=DetectionPolicy(bioclip_eligible_labels=("butterfly_like",)),
        crop_target_px=3,
    ):
        seen_batches.append([str(item["detection_id"]) for item in batch.items])

    assert loaded_for == ["det-1"]
    assert seen_batches == [["det-1"]]


def test_materialized_detector_crop_batches_retain_debug_crops_when_requested(tmp_path) -> None:
    retained_batches = list(
        iter_materialized_detector_crop_batches(
            canonical_records=_canonical_records(),
            detections=_detections().head(1),
            image_loader=lambda item: _decoded_image(),
            temp_dir=tmp_path,
            crop_batch_size=1,
            crop_target_px=3,
            retain_debug_crops=True,
        )
    )

    assert len(retained_batches) == 1
    batch = retained_batches[0]
    assert batch.temp_dir.exists()
    assert all(path.exists() for path in batch.crop_paths)

    batch.cleanup(force=True)

    assert not batch.temp_dir.exists()


def test_screen_object_detections_passes_ablation_mode_to_scorer(tmp_path) -> None:
    class ModeRecordingScorer:
        model_id = "fake-bioclip"
        model_version = "test"
        model_checkpoint = "fake-checkpoint"

        def __init__(self) -> None:
            self.modes: list[str | None] = []

        def score(self, item: dict[str, object], labels: tuple[str, ...]) -> dict[str, float]:
            self.modes.append(item.get("ablation_mode"))  # type: ignore[arg-type]
            return {label: (0.8 if label == "a photo of Danaus plexippus" else 0.0) for label in labels}

    scorer = ModeRecordingScorer()

    screen_object_detections(
        canonical_records=_canonical_records(),
        detections=_detections().head(1),
        species_context=_context(),
        candidate_set=_fixture_candidate_set(),
        scorer=scorer,
        output_path=tmp_path / "scores.parquet",
        ablation_mode="whole_image",
    )

    assert scorer.modes == ["whole_image", "whole_image", "whole_image", "whole_image"]


def test_ephemeral_scorer_uses_distinct_visual_inputs_for_ablation_modes(tmp_path) -> None:
    seen: list[tuple[str, bytes, bytes]] = []

    class WhiteMaskSegmenter:
        backend = "fake-mask"

        def segment_crop(self, crop) -> bytes:
            return b"\xff" * len(crop.encoded_bytes)

    def scorer(path: Path, labels: tuple[str, ...]) -> dict[str, float]:
        header1, header2, header3, body = path.read_bytes().split(b"\n", 3)
        seen.append((header2.decode("ascii"), header3, body))
        return {label: 0.5 for label in labels}

    crop_scorer = EphemeralCropBioClipScorer(
        scorer=scorer,
        image_loader=lambda item: _decoded_image(),
        temp_dir=tmp_path,
        crop_target_px=3,
        model_id="bioclip2_5",
        model_version="bioclip2_5_huge",
        model_checkpoint="checkpoint-a",
        segmenter=WhiteMaskSegmenter(),
    )
    item = {
        "source": "flickr",
        "flickr_photo_id": "photo-1",
        "detection_id": "det-1",
        "bbox_xyxy": [0.0, 0.0, 3.0, 3.0],
    }

    for mode in ("whole_image", "detector_crop", "detector_crop_segmentation"):
        crop_scorer.score({**item, "ablation_mode": mode}, ("a photo of Danaus plexippus",))

    whole_image, crop, segmented = seen
    assert whole_image[0] == "4 4"
    assert len(whole_image[2]) == 4 * 4 * 3
    assert crop[0] == "3 3"
    assert len(crop[2]) == 3 * 3 * 3
    assert segmented[0] == "3 3"
    assert segmented[2] == b"\xff" * (3 * 3 * 3)
    assert segmented[2] != crop[2]
    assert list(tmp_path.iterdir()) == []


def test_ephemeral_scorer_uses_detector_crop_mask_without_persisting_artifacts(tmp_path) -> None:
    seen: list[bytes] = []

    def scorer(path: Path, labels: tuple[str, ...]) -> dict[str, float]:
        _header1, _header2, _header3, body = path.read_bytes().split(b"\n", 3)
        seen.append(body)
        return {label: 0.5 for label in labels}

    crop_scorer = EphemeralCropBioClipScorer(
        scorer=scorer,
        image_loader=lambda item: _decoded_image(),
        temp_dir=tmp_path,
        crop_target_px=3,
        model_id="bioclip2_5",
        model_version="bioclip2_5_huge",
        model_checkpoint="checkpoint-a",
    )
    item = {
        "source": "flickr",
        "flickr_photo_id": "photo-1",
        "detection_id": "det-1",
        "bbox_xyxy": [0.0, 0.0, 3.0, 3.0],
        "ablation_mode": "detector_crop_segmentation",
        "detector_crop_mask": [1, 0, 1, 0, 1, 0, 1, 0, 1],
    }

    crop_scorer.score(item, ("a photo of Danaus plexippus",))

    assert seen
    assert seen[0][3:6] == b"\x00\x00\x00"
    assert any(byte != 0 for byte in seen[0][12:15])
    assert list(tmp_path.iterdir()) == []


def test_object_bioclip_runner_can_score_detector_crops_with_ephemeral_scorer(tmp_path) -> None:
    candidate_set = _fixture_candidate_set()

    crop_scorer = EphemeralCropBioClipScorer(
        scorer=lambda path, labels: {label: (0.83 if label == "a photo of Danaus plexippus" else 0.05) for label in labels},
        image_loader=lambda item: _decoded_image(),
        temp_dir=tmp_path / "crops",
        crop_target_px=3,
        model_id="bioclip2_5",
        model_version="bioclip2_5_huge",
        model_checkpoint="checkpoint-a",
    )

    result = screen_object_detections(
        canonical_records=_canonical_records(),
        detections=_detections().head(1),
        species_context=_context(),
        candidate_set=candidate_set,
        scorer=crop_scorer,
        output_path=tmp_path / "object_scores.parquet",
        ablation_mode="detector_crop",
    )

    row = result.frame.to_dicts()[0]
    assert row["model_id"] == "bioclip2_5"
    assert row["model_checkpoint"] == "checkpoint-a"
    assert result.visual_classifier == PRIMARY_VISUAL_CLASSIFIER
    assert result.visual_mode == "detector_crop"
    assert result.visual_mode_status == "available"
    assert row["species_top1"] == "Danaus plexippus"
    assert row["species_top1_scientific_name"] == "Danaus plexippus"
    assert row["target_species_score"] == 0.83
    assert row["occurrence_bin"] == "gold"
    assert not (tmp_path / "crops").exists() or list((tmp_path / "crops").iterdir()) == []


def test_object_bioclip_scores_detection_crops_with_join_keys(tmp_path) -> None:
    candidate_set = _fixture_candidate_set()
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
    assert row["classification_mode"] == "target_scope_object_screening"
    assert row["candidate_selection_mode"] == TARGET_SCOPE_CANDIDATE_SELECTION_MODE
    assert row["candidate_source"] == "species_context,single_target_fixture"
    assert row["ablation_mode"] == "detector_crop"
    assert row["species_first_pass_top_k"] == 20
    assert row["species_rerank_top_k"] == 5
    assert row["species_rerank_strategy"] == TARGET_SCOPE_SPECIES_RERANK_STRATEGY
    assert row["species_top5"][0] == "Danaus plexippus"
    assert row["species_top5_accepted_taxon_keys"][0] == "gbif:5131654"
    assert row["accepted_taxon_key"] == "gbif:5131654"
    assert row["species_top1_accepted_taxon_key"] == "gbif:5131654"
    assert row["target_accepted_taxon_key"] == "gbif:5131654"
    assert row["target_species_rank"] == 1
    assert row["target_species_score"] == 0.82
    assert row["occurrence_bin"] == "gold"
    assert row["is_target_positive"] is True


def test_object_bioclip_batches_label_set_scoring_by_crop_batch_size(tmp_path) -> None:
    class BatchOnlyScorer:
        model_id = "fake-bioclip"
        model_version = "test"
        model_checkpoint = "fake-checkpoint"

        def __init__(self) -> None:
            self.calls: list[tuple[tuple[str, ...], tuple[str, ...]]] = []

        def score(self, item, labels):  # noqa: ANN001, ANN202 - proves the batch path is used.
            raise AssertionError("object scoring should use score_label_sets_batch")

        def score_label_sets_batch(self, items, label_sets):  # noqa: ANN001, ANN202 - mirrors object batch scorer API.
            detection_ids = tuple(str(item["detection_id"]) for item in items)
            self.calls.append((detection_ids, tuple(label_sets)))
            return {
                name: [
                    {
                        label: (
                            0.82
                            if label == "a photo of Danaus plexippus" and item["detection_id"] == "det-1"
                            else 0.44
                            if label == "a photo of Danaus plexippus"
                            else 0.1
                        )
                        for label in labels
                    }
                    for item in items
                ]
                for name, labels in label_sets.items()
            }

    scorer = BatchOnlyScorer()

    result = screen_object_detections(
        canonical_records=_canonical_records(),
        detections=_detections(),
        species_context=_context(),
        candidate_set=_fixture_candidate_set(),
        scorer=scorer,
        output_path=tmp_path / "object_scores.parquet",
        ablation_mode="detector_crop",
        bioclip_batch_size=2,
    )

    assert result.crops_scored == 2
    assert result.frame.height == 2
    assert scorer.calls == [
        (("det-1", "det-2"), ("family", "genus", "species")),
        (("det-1", "det-2"), ("rerank",)),
    ]


def test_object_bioclip_respects_bioclip_batch_size_for_label_set_scoring(tmp_path) -> None:
    class BatchRecordingScorer(FakeObjectBioClipScorer):
        def __init__(self) -> None:
            super().__init__({"sha256:crop-1": {"a photo of Danaus plexippus": 0.82}})
            self.initial_batches: list[tuple[str, ...]] = []

        def score_label_sets_batch(self, items, label_sets):  # noqa: ANN001, ANN202 - mirrors object batch scorer API.
            if "species" in label_sets:
                self.initial_batches.append(tuple(str(item["detection_id"]) for item in items))
            return super().score_label_sets_batch(items, label_sets)

    scorer = BatchRecordingScorer()

    screen_object_detections(
        canonical_records=_canonical_records(),
        detections=_detections(),
        species_context=_context(),
        candidate_set=_fixture_candidate_set(),
        scorer=scorer,
        output_path=tmp_path / "object_scores.parquet",
        ablation_mode="detector_crop",
        bioclip_batch_size=1,
    )

    assert scorer.initial_batches == [("det-1",), ("det-2",)]


def test_object_bioclip_production_gate_scores_only_detected_butterflies(tmp_path) -> None:
    base_detection = _detections().to_dicts()[0]
    cases = [
        ("det-butterfly", "photo-1", "butterfly_like", "detected"),
        ("det-moth", "photo-moth", "moth_like", "detected"),
        ("det-caterpillar", "photo-caterpillar", "caterpillar", "detected"),
        ("det-pupa", "photo-pupa", "pupa", "detected"),
        ("det-insect", "photo-insect", "insect_like", "detected"),
        ("det-hard-negative", "photo-hard-negative", "hard_negative", "detected"),
        ("det-no-detection", "photo-no-detection", "butterfly_like", "no_detection"),
        ("det-failed-image", "photo-failed-image", "butterfly_like", "failed_image_load"),
    ]
    detections = pl.DataFrame(
        [
            {
                **base_detection,
                "flickr_photo_id": photo_id,
                "detection_id": detection_id,
                "detector_label": label,
                "detection_status": status,
                "crop_hash": f"sha256:{detection_id}",
                "failure_reason": None if status == "detected" else status,
            }
            for detection_id, photo_id, label, status in cases
        ]
    )

    class GateRecordingScorer:
        model_id = "fake-bioclip"
        model_version = "test"
        model_checkpoint = "fake-checkpoint"

        def __init__(self) -> None:
            self.calls: list[tuple[str, ...]] = []

        def score(self, item, labels):  # noqa: ANN001, ANN202 - proves the batch path is used.
            raise AssertionError(f"unexpected single-item BioCLIP score for {item.get('detection_id')}")

        def score_label_sets_batch(self, items, label_sets):  # noqa: ANN001, ANN202 - mirrors object batch scorer API.
            self.calls.append(tuple(str(item["detection_id"]) for item in items))
            return {
                name: [
                    {label: (0.84 if label == "a photo of Danaus plexippus" else 0.1) for label in labels}
                    for _item in items
                ]
                for name, labels in label_sets.items()
            }

    scorer = GateRecordingScorer()

    result = screen_object_detections(
        canonical_records=_canonical_records(),
        detections=detections,
        species_context=_context(),
        candidate_set=_fixture_candidate_set(),
        scorer=scorer,
        output_path=tmp_path / "object_scores.parquet",
        ablation_mode="detector_crop",
    )

    assert result.crops_scored == 1
    assert result.frame["detection_id"].to_list() == ["det-butterfly"]
    assert scorer.calls == [("det-butterfly",), ("det-butterfly",)]


def test_object_bioclip_skips_non_butterfly_detector_labels(tmp_path) -> None:
    class FailingScorer:
        model_id = "fake-bioclip"
        model_version = "test"
        model_checkpoint = "fake-checkpoint"

        def score(self, item, labels):  # noqa: ANN001, ANN202 - mirrors scorer protocol.
            raise AssertionError(f"non-butterfly detection was sent to BioCLIP: {item.get('detector_label')}")

    detections = _detections().with_columns(pl.lit("moth_like").alias("detector_label"))
    result = screen_object_detections(
        canonical_records=_canonical_records(),
        detections=detections,
        species_context=_context(),
        candidate_set=_fixture_candidate_set(),
        scorer=FailingScorer(),
        output_path=tmp_path / "object_scores.parquet",
        ablation_mode="detector_crop",
        detection_policy=DetectionPolicy(bioclip_eligible_labels=("butterfly_like",)),
    )

    assert result.crops_scored == 0
    assert result.frame.is_empty()
    assert pl.read_parquet(tmp_path / "object_scores.parquet").is_empty()


def test_object_bioclip_empty_scores_write_stable_schema(tmp_path) -> None:
    output = tmp_path / "object_bioclip_scores.parquet"
    detections = pl.DataFrame(
        [
            {
                "source": "flickr",
                "flickr_photo_id": "photo-1",
                "detection_id": "no-det-1",
                "crop_hash": None,
                "detection_status": "no_detection",
                "failure_reason": "no_butterfly_like_object",
            }
        ]
    )

    result = screen_object_detections(
        canonical_records=_canonical_records(),
        detections=detections,
        species_context=_context(),
        candidate_set=_fixture_candidate_set(),
        scorer=FakeObjectBioClipScorer({}),
        output_path=output,
        ablation_mode="detector_crop",
    )

    frame = pl.read_parquet(output)
    assert result.crops_scored == 0
    assert frame.height == 0
    assert {
        "source",
        "flickr_photo_id",
        "detection_id",
        "crop_hash",
        "model_id",
        "model_version",
        "model_checkpoint",
        "candidate_set_id",
        "classified_at",
        "classification_mode",
        "candidate_selection_mode",
        "candidate_source",
        "ablation_mode",
        "species_first_pass_top_k",
        "species_rerank_top_k",
        "species_rerank_strategy",
        "triage_group_top",
        "triage_group_scores",
        "family_top3",
        "family_top1",
        "family_top1_score",
        "family_margin",
        "genus_top8",
        "genus_top1",
        "genus_top1_score",
        "genus_margin",
        "species_top20",
        "species_top5",
        "species_top1",
        "species_top1_scientific_name",
        "species_top1_score",
        "species_top1_margin",
        "target_species_score",
        "target_species_rank",
        "geospatial_prior_score",
        "geospatial_prior_reason",
        "text_evidence_score",
        "comment_evidence_score",
        "is_target_positive",
        "is_negative_material",
        "occurrence_bin",
        "bin_reason",
    }.issubset(frame.columns)


def test_object_bioclip_scores_flush_to_parquet_batches(tmp_path) -> None:
    output = tmp_path / "object_bioclip_scores.parquet"
    result = screen_object_detections(
        canonical_records=_canonical_records(),
        detections=_detections(),
        species_context=_context(),
        candidate_set=_fixture_candidate_set(),
        scorer=FakeObjectBioClipScorer(
            {
                "sha256:crop-1": {"a photo of Danaus plexippus": 0.82},
                "sha256:crop-2": {"a photo of Danaus plexippus": 0.42},
            }
        ),
        output_path=output,
        ablation_mode="detector_crop",
        parquet_batch_rows=1,
    )

    frame = pl.read_parquet(output)
    assert result.score_batches_written == 2
    assert result.crops_scored == 2
    assert frame.height == 2
    assert sorted(frame["detection_id"].to_list()) == ["det-1", "det-2"]
    assert not (tmp_path / ".object_bioclip_scores.parquet.batches.tmp").exists()


def test_object_bioclip_score_keeps_metadata_negative_hint_as_review_context(tmp_path) -> None:
    candidate_set = _fixture_candidate_set()
    canonical = _canonical_records().with_columns(
        pl.lit("artwork").alias("image_category"),
        pl.lit("artwork").alias("negative_filter_reason"),
        pl.lit(True).alias("is_negative_material"),
    )

    result = screen_object_detections(
        canonical_records=canonical,
        detections=_detections().head(1),
        species_context=_context(),
        candidate_set=candidate_set,
        scorer=FakeObjectBioClipScorer({"sha256:crop-1": {"a photo of Danaus plexippus": 0.82}}),
        output_path=tmp_path / "object_scores.parquet",
        ablation_mode="detector_crop",
    )

    row = result.frame.to_dicts()[0]
    assert row["occurrence_bin"] == "gold"
    assert row["bin_reason"] == "target_species_score_ge_070"
    assert row["is_negative_material"] is False
    assert row["is_target_positive"] is True


def test_object_bioclip_score_bins_visual_hard_negative_object(tmp_path) -> None:
    detection = _detections().head(1).with_columns(pl.lit("hard_negative").alias("detector_label"))
    scores_path = tmp_path / "object_scores.parquet"

    result = screen_object_detections(
        canonical_records=_canonical_records(),
        detections=detection,
        species_context=_context(),
        candidate_set=_fixture_candidate_set(),
        scorer=FakeObjectBioClipScorer({"sha256:crop-1": {"a photo of Danaus plexippus": 0.82}}),
        output_path=scores_path,
        ablation_mode="detector_crop",
    )

    assert result.frame.is_empty()
    _canonical_records().write_parquet(tmp_path / "canonical.parquet")
    detection.write_parquet(tmp_path / "detections.parquet")
    outputs = write_object_evidence_outputs(
        canonical_records_path=tmp_path / "canonical.parquet",
        detections_path=tmp_path / "detections.parquet",
        scores_path=scores_path,
        joined_output_path=tmp_path / "joined.parquet",
        photo_summary_output_path=tmp_path / "summary.parquet",
        species_context=_context(),
    )
    summary = pl.read_parquet(outputs.photo_evidence_summary).to_dicts()[0]
    assert summary["photo_occurrence_bin"] == "bin"
    assert summary["photo_bin_reason"] == "negative_material_hard_negative_object"


def test_object_bioclip_routes_non_top1_target_species_to_review(tmp_path) -> None:
    candidates = tmp_path / "species_candidates.parquet"
    pl.DataFrame(
        [
            {"scientific_name": "Danaus plexippus", "accepted_taxon_key": "gbif:5131654", "family": "Nymphalidae", "genus": "Danaus"},
            {"scientific_name": "Danaus gilippus", "accepted_taxon_key": "gbif:5131655", "family": "Nymphalidae", "genus": "Danaus"},
        ]
    ).write_parquet(candidates)
    candidate_set = build_candidate_set(_context(), species_candidate_path=candidates)

    result = screen_object_detections(
        canonical_records=_canonical_records(),
        detections=_detections().head(1),
        species_context=_context(),
        candidate_set=candidate_set,
        scorer=FakeObjectBioClipScorer(
            {
                "sha256:crop-1": {
                    "a photo of Danaus plexippus": 0.72,
                    "a photo of Danaus gilippus": 0.91,
                }
            }
        ),
        output_path=tmp_path / "object_scores.parquet",
        ablation_mode="detector_crop",
    )

    row = result.frame.to_dicts()[0]
    assert row["species_top1_scientific_name"] == "Danaus gilippus"
    assert row["target_species_rank"] == 2
    assert row["target_species_score"] == 0.72
    assert row["occurrence_bin"] == "in_review"
    assert row["bin_reason"] == "species_conflict"
    assert row["is_target_positive"] is False


def test_object_bioclip_rejects_detections_without_canonical_source_record(tmp_path) -> None:
    candidate_set = _fixture_candidate_set()
    detections = _detections().head(1).with_columns(pl.lit("photo-missing").alias("flickr_photo_id"))

    with pytest.raises(ValueError, match="no canonical source record"):
        screen_object_detections(
            canonical_records=_canonical_records(),
            detections=detections,
            species_context=_context(),
            candidate_set=candidate_set,
            scorer=FakeObjectBioClipScorer({"sha256:crop-1": {"a photo of Danaus plexippus": 0.82}}),
            output_path=tmp_path / "object_scores.parquet",
            ablation_mode="detector_crop",
        )

    assert not (tmp_path / "object_scores.parquet").exists()


def test_object_bioclip_scores_family_genus_and_species_stages_separately(tmp_path) -> None:
    candidates = tmp_path / "species_candidates.parquet"
    pl.DataFrame(
        [
            {"scientific_name": "Danaus plexippus", "accepted_taxon_key": "gbif:5131654", "family": "Nymphalidae", "genus": "Danaus"},
            {"scientific_name": "Danaus gilippus", "accepted_taxon_key": "gbif:5131655", "family": "Nymphalidae", "genus": "Danaus"},
            {"scientific_name": "Limenitis archippus", "accepted_taxon_key": "gbif:1900000", "family": "Nymphalidae", "genus": "Limenitis"},
        ]
    ).write_parquet(candidates)
    candidate_set = build_candidate_set(_context(), species_candidate_path=candidates)

    class StageRecordingScorer:
        model_id = "fake-bioclip"
        model_version = "test"
        model_checkpoint = "fake-checkpoint"

        def __init__(self) -> None:
            self.calls: list[tuple[str, ...]] = []

        def score(self, item: dict[str, object], labels: tuple[str, ...]) -> dict[str, float]:
            self.calls.append(labels)
            scores = {label: 0.0 for label in labels}
            if labels == ("Nymphalidae",):
                scores["Nymphalidae"] = 0.61
            elif labels == ("Danaus", "Limenitis"):
                scores["Danaus"] = 0.72
                scores["Limenitis"] = 0.21
            else:
                scores["a photo of Danaus plexippus"] = 0.83
                scores["a photo of Danaus gilippus"] = 0.44
                scores["a photo of Limenitis archippus"] = 0.12
            return scores

    scorer = StageRecordingScorer()

    result = screen_object_detections(
        canonical_records=_canonical_records(),
        detections=_detections().head(1),
        species_context=_context(),
        candidate_set=candidate_set,
        scorer=scorer,
        output_path=tmp_path / "object_scores.parquet",
        ablation_mode="detector_crop",
    )

    row = result.frame.to_dicts()[0]
    assert scorer.calls == [
        ("Nymphalidae",),
        ("Danaus", "Limenitis"),
        candidate_set.prompt_labels("species"),
        candidate_set.prompt_labels("species"),
    ]
    assert row["family_top3"] == ["Nymphalidae"]
    assert row["family_top1_score"] == 0.61
    assert row["genus_top8"] == ["Danaus", "Limenitis"]
    assert row["genus_top1_score"] == 0.72
    assert row["species_top5"] == ["Danaus plexippus", "Danaus gilippus", "Limenitis archippus"]
    assert row["target_species_score"] == 0.83


def test_object_bioclip_reranks_species_top20_into_top5(tmp_path) -> None:
    candidates = tmp_path / "species_candidates.parquet"
    pl.DataFrame(
        [
            {"scientific_name": "Danaus plexippus", "accepted_taxon_key": "gbif:5131654", "family": "Nymphalidae", "genus": "Danaus"},
            {"scientific_name": "Danaus gilippus", "accepted_taxon_key": "gbif:5131655", "family": "Nymphalidae", "genus": "Danaus"},
            {"scientific_name": "Danaus erippus", "accepted_taxon_key": "gbif:5131656", "family": "Nymphalidae", "genus": "Danaus"},
            {"scientific_name": "Danaus eresimus", "accepted_taxon_key": "gbif:5131657", "family": "Nymphalidae", "genus": "Danaus"},
            {"scientific_name": "Limenitis archippus", "accepted_taxon_key": "gbif:1900000", "family": "Nymphalidae", "genus": "Limenitis"},
            {"scientific_name": "Limenitis arthemis", "accepted_taxon_key": "gbif:1900001", "family": "Nymphalidae", "genus": "Limenitis"},
        ]
    ).write_parquet(candidates)
    candidate_set = build_candidate_set(_context(), species_candidate_path=candidates)

    class RerankRecordingScorer:
        model_id = "fake-bioclip"
        model_version = "test"
        model_checkpoint = "fake-checkpoint"

        def __init__(self) -> None:
            self.species_calls: list[tuple[str, ...]] = []

        def score(self, item: dict[str, object], labels: tuple[str, ...]) -> dict[str, float]:
            scores = {label: 0.0 for label in labels}
            if labels == ("Nymphalidae",):
                scores["Nymphalidae"] = 0.8
                return scores
            if labels == ("Danaus", "Limenitis"):
                scores["Danaus"] = 0.8
                scores["Limenitis"] = 0.2
                return scores
            self.species_calls.append(labels)
            if len(self.species_calls) == 1:
                for label, score in {
                    "a photo of Danaus plexippus": 0.61,
                    "a photo of Danaus gilippus": 0.93,
                    "a photo of Danaus erippus": 0.72,
                    "a photo of Danaus eresimus": 0.71,
                    "a photo of Limenitis archippus": 0.70,
                    "a photo of Limenitis arthemis": 0.20,
                }.items():
                    if label in scores:
                        scores[label] = score
                return scores
            for label, score in {
                "a photo of Danaus plexippus": 0.95,
                "a photo of Danaus gilippus": 0.44,
                "a photo of Danaus erippus": 0.43,
                "a photo of Danaus eresimus": 0.42,
                "a photo of Limenitis archippus": 0.41,
            }.items():
                if label in scores:
                    scores[label] = score
            return scores

    scorer = RerankRecordingScorer()

    result = screen_object_detections(
        canonical_records=_canonical_records(),
        detections=_detections().head(1),
        species_context=_context(),
        candidate_set=candidate_set,
        scorer=scorer,
        output_path=tmp_path / "object_scores.parquet",
        ablation_mode="detector_crop",
    )

    row = result.frame.to_dicts()[0]
    assert len(scorer.species_calls) == 2
    assert scorer.species_calls[0] == candidate_set.prompt_labels("species")
    assert "a photo of Limenitis arthemis" not in scorer.species_calls[1]
    assert row["species_top20"][:6] == [
        "Danaus gilippus",
        "Danaus erippus",
        "Danaus eresimus",
        "Limenitis archippus",
        "Danaus plexippus",
        "Limenitis arthemis",
    ]
    assert row["species_top5"] == [
        "Danaus plexippus",
        "Danaus gilippus",
        "Danaus erippus",
        "Danaus eresimus",
        "Limenitis archippus",
    ]
    assert row["species_top1_scientific_name"] == "Danaus plexippus"
    assert row["target_species_score"] == 0.95
    assert row["target_species_rank"] == 1


def test_object_bioclip_routes_ambiguous_species_margin_to_review(tmp_path) -> None:
    candidates = tmp_path / "species_candidates.parquet"
    pl.DataFrame(
        [
            {"scientific_name": "Danaus plexippus", "accepted_taxon_key": "gbif:5131654", "family": "Nymphalidae", "genus": "Danaus"},
            {"scientific_name": "Danaus gilippus", "accepted_taxon_key": "gbif:5131655", "family": "Nymphalidae", "genus": "Danaus"},
        ]
    ).write_parquet(candidates)
    candidate_set = build_candidate_set(_context(), species_candidate_path=candidates)

    result = screen_object_detections(
        canonical_records=_canonical_records(),
        detections=_detections().head(1),
        species_context=_context(),
        candidate_set=candidate_set,
        scorer=FakeObjectBioClipScorer(
            {
                "sha256:crop-1": {
                    "a photo of Danaus plexippus": 0.82,
                    "a photo of Danaus gilippus": 0.80,
                }
            }
        ),
        output_path=tmp_path / "object_scores.parquet",
        ablation_mode="detector_crop",
    )

    row = result.frame.to_dicts()[0]
    assert row["species_top1_margin"] == pytest.approx(0.02)
    assert row["occurrence_bin"] == "in_review"
    assert row["bin_reason"] == "ambiguous_species_margin"
    assert row["is_target_positive"] is False


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


def test_geography_soft_prior_uses_target_geo_prior_table() -> None:
    context = _context()
    geo_prior_table = pl.DataFrame(
        [
            {
                "accepted_taxon_key": "gbif:5131654",
                "scientific_name": "Danaus plexippus",
                "bbox": "140.0,-40.0,155.0,-25.0",
                "source": "fixture",
            }
        ]
    )

    prior = apply_geospatial_soft_prior(
        {"latitude": -35.0, "longitude": 149.0},
        context,
        visual_score=0.8,
        geo_prior_table=geo_prior_table,
    )

    assert prior.score > 0
    assert prior.reason == "within_geo_prior_table"
    assert prior.route_to_review is False
    assert prior.hard_discard is False


def test_geography_soft_prior_accepts_candidate_specific_geo_prior_table() -> None:
    context = _context()
    candidate = CandidateTaxon(
        scientific_name="Danaus erippus",
        accepted_taxon_key="gbif:1901234",
        family="Nymphalidae",
        genus="Danaus",
    )
    geo_prior_table = pl.DataFrame(
        [
            {
                "accepted_taxon_key": "gbif:1901234",
                "scientific_name": "Danaus erippus",
                "bbox": "140.0,-40.0,155.0,-25.0",
                "source": "fixture",
            }
        ]
    )

    prior = apply_geospatial_soft_prior(
        {"latitude": -35.0, "longitude": 149.0},
        candidate,
        context,
        visual_score=0.8,
        geo_prior_table=geo_prior_table,
    )

    assert prior.score > 0
    assert prior.reason == "within_geo_prior_table"
    assert prior.route_to_review is False
    assert prior.hard_discard is False


def test_ablation_modes_write_rows_with_shared_photo_join_keys(tmp_path) -> None:
    candidate_set = _fixture_candidate_set()
    report = run_object_ablations(
        canonical_records=_canonical_records(),
        detections=_detections().head(1),
        species_context=_context(),
        candidate_set=candidate_set,
        scorer=FakeObjectBioClipScorer({"sha256:crop-1": {"a photo of Danaus plexippus": 0.82}}),
        output_dir=tmp_path,
        modes=("whole_image", "detector_crop", "detector_crop_segmentation"),
        parquet_batch_rows=1,
    )

    frames = [pl.read_parquet(tmp_path / f"object_bioclip_scores_{mode}.parquet") for mode in report.modes]
    combined = pl.concat(frames, how="diagonal_relaxed")
    rows = combined.sort("ablation_mode").to_dicts()

    assert {row["ablation_mode"] for row in rows} == {"whole_image", "detector_crop"}
    assert {row["source"] for row in rows} == {"flickr"}
    assert {row["flickr_photo_id"] for row in rows} == {"photo-1"}
    assert report.report["score_batches_written_by_mode"] == {
        "detector_crop": 1,
        "detector_crop_segmentation": 0,
        "whole_image": 1,
    }
    assert report.report["primary_visual_classifier"] == PRIMARY_VISUAL_CLASSIFIER
    assert report.report["visual_modes_requested"] == ["whole_image", "detector_crop", "detector_crop_segmentation"]
    assert report.report["visual_modes_scored"] == ["detector_crop", "whole_image"]
    assert report.report["visual_mode_status_by_mode"] == {
        "detector_crop": "available",
        "detector_crop_segmentation": "unavailable",
        "whole_image": "available",
    }
    assert report.report["segmentation_status_by_mode"]["detector_crop_segmentation"] == "unavailable"
    assert report.report["segmentation_unavailable_count_by_mode"]["detector_crop_segmentation"] == 1
    assert report.report["segmentation_unavailable_reason_by_mode"]["detector_crop_segmentation"] == "detector_masks_missing"
    assert build_ablation_report(combined)["crops_scored"] == 2
    assert build_ablation_report(combined)["gold_count"] == 2


def test_ablation_report_uses_objective_disagreement_field_names() -> None:
    report = build_ablation_report(
        pl.DataFrame(
            [
                {
                    "source": "flickr",
                    "flickr_photo_id": "photo-1",
                    "detection_id": "det-1",
                    "ablation_mode": "whole_image",
                    "occurrence_bin": "gold",
                    "species_top1_scientific_name": "Danaus gilippus",
                },
                {
                    "source": "flickr",
                    "flickr_photo_id": "photo-1",
                    "detection_id": "det-1",
                    "ablation_mode": "detector_crop",
                    "occurrence_bin": "gold",
                    "species_top1_scientific_name": "Danaus plexippus",
                },
                {
                    "source": "flickr",
                    "flickr_photo_id": "photo-1",
                    "detection_id": "det-1",
                    "ablation_mode": "detector_crop_segmentation",
                    "occurrence_bin": "gold",
                    "species_top1_scientific_name": "Danaus plexippus",
                },
            ]
        )
    )

    assert report["whole_image_vs_crop"] == 1
    assert report["crop_vs_segmentation"] == 0
    assert report["whole_image_vs_crop_disagreements"] == 1
    assert report["crop_vs_segmentation_disagreements"] == 0


def test_ablation_report_counts_no_detection_records(tmp_path) -> None:
    detections = pl.DataFrame(
        [
            {
                "source": "flickr",
                "flickr_photo_id": "photo-1",
                "detection_id": "no-detection-photo-1",
                "crop_hash": None,
                "bbox_xyxy": [],
                "detection_status": "no_detection",
                "failure_reason": "no_butterfly_like_object",
            }
        ]
    )

    report = run_object_ablations(
        canonical_records=_canonical_records(),
        detections=detections,
        species_context=_context(),
        candidate_set=_fixture_candidate_set(),
        scorer=FakeObjectBioClipScorer({}),
        output_dir=tmp_path,
        modes=("detector_crop",),
    )

    assert report.report["records_seen"] == 1
    assert report.report["detections_seen"] == 1
    assert report.report["crops_scored"] == 0
    assert report.report["no_detection_records"] == 1
    assert report.report["ablation_mode"] == ["detector_crop"]
    persisted = json.loads((tmp_path / "ablation_report.json").read_text(encoding="utf-8"))
    assert persisted["no_detection_records"] == 1
    assert persisted["ablation_mode"] == ["detector_crop"]


def test_object_evidence_join_and_photo_summary_outputs(tmp_path) -> None:
    candidate_set = _fixture_candidate_set()
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


def test_photo_summary_counts_unscored_detections_from_detection_table(tmp_path) -> None:
    canonical_path = tmp_path / "canonical.parquet"
    detections_path = tmp_path / "detections.parquet"
    scores_path = tmp_path / "scores.parquet"
    _canonical_records().write_parquet(canonical_path)
    _detections().write_parquet(detections_path)
    pl.DataFrame(
        [
            {
                "source": "flickr",
                "flickr_photo_id": "photo-1",
                "detection_id": "det-1",
                "crop_hash": "sha256:crop-1",
                "target_species_score": 0.82,
                "occurrence_bin": "gold",
                "species_top1_scientific_name": "Danaus plexippus",
                "bin_reason": "target_species_score_ge_070",
            }
        ]
    ).write_parquet(scores_path)

    outputs = write_object_evidence_outputs(
        canonical_records_path=canonical_path,
        detections_path=detections_path,
        scores_path=scores_path,
        joined_output_path=tmp_path / "object_evidence_joined.parquet",
        photo_summary_output_path=tmp_path / "photo_evidence_summary.parquet",
    )

    joined = pl.read_parquet(outputs.object_evidence_joined)
    summary = pl.read_parquet(outputs.photo_evidence_summary).to_dicts()[0]
    assert joined.height == 2
    assert summary["best_detection_id"] == "det-1"
    assert summary["detection_count"] == 2
    assert summary["all_detection_ids"] == ["det-1", "det-2"]


def test_photo_summary_retains_topk_candidate_species(tmp_path) -> None:
    canonical_path = tmp_path / "canonical.parquet"
    detections_path = tmp_path / "detections.parquet"
    scores_path = tmp_path / "scores.parquet"
    _canonical_records().write_parquet(canonical_path)
    _detections().head(1).write_parquet(detections_path)
    pl.DataFrame(
        [
            {
                "source": "flickr",
                "flickr_photo_id": "photo-1",
                "detection_id": "det-1",
                "crop_hash": "sha256:crop-1",
                "target_species_score": 0.82,
                "occurrence_bin": "gold",
                "species_top1_scientific_name": "Danaus plexippus",
                "species_top5": ["Danaus plexippus", "Danaus gilippus"],
                "species_top20": ["Danaus plexippus", "Danaus gilippus", "Limenitis archippus"],
                "bin_reason": "target_species_score_ge_070",
            }
        ]
    ).write_parquet(scores_path)

    outputs = write_object_evidence_outputs(
        canonical_records_path=canonical_path,
        detections_path=detections_path,
        scores_path=scores_path,
        joined_output_path=tmp_path / "object_evidence_joined.parquet",
        photo_summary_output_path=tmp_path / "photo_evidence_summary.parquet",
    )

    summary = pl.read_parquet(outputs.photo_evidence_summary).to_dicts()[0]
    assert summary["all_candidate_species"] == ["Danaus plexippus", "Danaus gilippus", "Limenitis archippus"]


def test_empty_object_evidence_outputs_keep_stable_join_table_schemas(tmp_path) -> None:
    canonical_path = tmp_path / "canonical.parquet"
    detections_path = tmp_path / "detections.parquet"
    scores_path = tmp_path / "scores.parquet"
    joined_path = tmp_path / "object_evidence_joined.parquet"
    summary_path = tmp_path / "photo_evidence_summary.parquet"
    pl.DataFrame(schema={"source": pl.String, "flickr_photo_id": pl.String}).write_parquet(canonical_path)
    empty_detection_frame().write_parquet(detections_path)
    empty_object_score_frame().write_parquet(scores_path)

    outputs = write_object_evidence_outputs(
        canonical_records_path=canonical_path,
        detections_path=detections_path,
        scores_path=scores_path,
        joined_output_path=joined_path,
        photo_summary_output_path=summary_path,
    )

    joined = pl.read_parquet(outputs.object_evidence_joined)
    summary = pl.read_parquet(outputs.photo_evidence_summary)
    assert joined.height == 0
    assert summary.height == 0
    assert set(OBJECT_SCORE_OUTPUT_SCHEMA).issubset(joined.columns)
    assert {
        "source",
        "flickr_photo_id",
        "detection_id",
        "crop_hash",
        "source_record_hash",
        "prediction_source",
        "bbox_xyxy",
        "bbox_xyxyn",
        "bbox_xywhn",
        "schema_version",
        "model_id",
        "model_checkpoint",
        "candidate_set_id",
        "species_top1_scientific_name",
        "species_top1_accepted_taxon_key",
        "accepted_taxon_key",
        "target_accepted_taxon_key",
        "geospatial_prior_score",
        "geospatial_prior_reason",
        "text_evidence_score",
        "comment_evidence_score",
        "is_target_positive",
        "is_negative_material",
        "comment_review_decision",
        "comment_review_reason",
        "comment_species_candidate",
        "comment_resolves_conflict",
        "geo_evidence_from_comments",
    }.issubset(joined.columns)
    assert set(DETECTION_OUTPUT_SCHEMA).issubset(joined.columns)
    assert {
        "source",
        "flickr_photo_id",
        "best_detection_id",
        "detection_count",
        "best_object_occurrence_bin",
        "best_object_species_top1",
        "best_object_score",
        "photo_occurrence_bin",
        "photo_bin_reason",
        "all_detection_ids",
        "all_candidate_species",
    }.issubset(summary.columns)


def test_photo_summary_routes_geospatial_conflict_to_review(tmp_path) -> None:
    canonical_path = tmp_path / "canonical.parquet"
    detections_path = tmp_path / "detections.parquet"
    scores_path = tmp_path / "scores.parquet"
    _canonical_records().write_parquet(canonical_path)
    _detections().write_parquet(detections_path)
    pl.DataFrame(
        [
            {
                "source": "flickr",
                "flickr_photo_id": "photo-1",
                "detection_id": "det-1",
                "crop_hash": "sha256:crop-1",
                "target_species_score": 0.20,
                "occurrence_bin": "bronze",
                "species_top1_scientific_name": "Danaus plexippus",
                "bin_reason": "weak_species_score",
            },
            {
                "source": "flickr",
                "flickr_photo_id": "photo-1",
                "detection_id": "det-2",
                "crop_hash": "sha256:crop-2",
                "target_species_score": 0.75,
                "occurrence_bin": "in_review",
                "species_top1_scientific_name": "Danaus plexippus",
                "bin_reason": "geospatial_conflict",
            },
        ]
    ).write_parquet(scores_path)

    outputs = write_object_evidence_outputs(
        canonical_records_path=canonical_path,
        detections_path=detections_path,
        scores_path=scores_path,
        joined_output_path=tmp_path / "object_evidence_joined.parquet",
        photo_summary_output_path=tmp_path / "photo_evidence_summary.parquet",
    )

    summary = pl.read_parquet(outputs.photo_evidence_summary).to_dicts()[0]
    assert summary["best_detection_id"] == "det-2"
    assert summary["best_object_occurrence_bin"] == "in_review"
    assert summary["photo_occurrence_bin"] == "in_review"
    assert summary["photo_bin_reason"] == "geospatial_conflict"


def test_photo_summary_metadata_negative_hint_does_not_override_gold_object(tmp_path) -> None:
    canonical_path = tmp_path / "canonical.parquet"
    detections_path = tmp_path / "detections.parquet"
    scores_path = tmp_path / "scores.parquet"
    _canonical_records().with_columns(
        pl.lit("artwork").alias("image_category"),
        pl.lit("artwork").alias("negative_filter_reason"),
        pl.lit(True).alias("is_negative_material"),
    ).write_parquet(canonical_path)
    _detections().head(1).write_parquet(detections_path)
    pl.DataFrame(
        [
            {
                "source": "flickr",
                "flickr_photo_id": "photo-1",
                "detection_id": "det-1",
                "crop_hash": "sha256:crop-1",
                "target_species_score": 0.82,
                "occurrence_bin": "gold",
                "species_top1_scientific_name": "Danaus plexippus",
                "bin_reason": "target_species_score_ge_070",
            },
        ]
    ).write_parquet(scores_path)

    outputs = write_object_evidence_outputs(
        canonical_records_path=canonical_path,
        detections_path=detections_path,
        scores_path=scores_path,
        joined_output_path=tmp_path / "object_evidence_joined.parquet",
        photo_summary_output_path=tmp_path / "photo_evidence_summary.parquet",
    )

    summary = pl.read_parquet(outputs.photo_evidence_summary).to_dicts()[0]
    assert summary["best_object_occurrence_bin"] == "gold"
    assert summary["photo_occurrence_bin"] == "gold"
    assert summary["photo_bin_reason"] == "target_species_score_ge_070"


def test_no_detection_with_strong_text_evidence_routes_photo_to_review(tmp_path) -> None:
    canonical = pl.DataFrame(
        [
            {
                "source": "flickr",
                "flickr_photo_id": "photo-no-det",
                "source_record_hash": "sha256:source-no-det",
                "image_url": "https://live.staticflickr.com/photo-no-det.jpg",
                "photo_page_url": "https://www.flickr.com/photos/u/photo-no-det",
                "title": "Danaus plexippus on milkweed",
                "raw_tags": "monarch butterfly",
                "latitude": 45.0,
                "longitude": -93.0,
                "date_taken": "2024-07-01",
            }
        ]
    )
    detections = pl.DataFrame(
        [
            {
                "source": "flickr",
                "flickr_photo_id": "photo-no-det",
                "source_record_hash": "sha256:source-no-det",
                "image_url": "https://live.staticflickr.com/photo-no-det.jpg",
                "photo_page_url": "https://www.flickr.com/photos/u/photo-no-det",
                "detection_id": "no-detection-photo-no-det",
                "detector_backend": "fake",
                "detector_model_id": "fake-detector",
                "detector_model_version": "v1",
                "detector_checkpoint": "checkpoint-a",
                "detected_at": "2026-01-01T00:00:00+00:00",
                "bbox_xyxy": [],
                "bbox_xyxyn": [],
                "bbox_xywhn": [],
                "box_area_ratio": 0.0,
                "detector_label": "no_detection",
                "detector_score": 0.0,
                "objectness_score": None,
                "nms_group_id": None,
                "crop_padding_ratio": 0.12,
                "crop_hash": None,
                "crop_width": None,
                "crop_height": None,
                "crop_storage_policy": "ephemeral",
                "detection_status": "no_detection",
                "failure_reason": "no_butterfly_like_object",
            }
        ]
    )
    scores = pl.DataFrame(
        [],
        schema={
            "source": pl.String,
            "flickr_photo_id": pl.String,
            "detection_id": pl.String,
            "crop_hash": pl.String,
            "target_species_score": pl.Float64,
            "occurrence_bin": pl.String,
            "species_top1_scientific_name": pl.String,
            "bin_reason": pl.String,
        },
    )
    canonical_path = tmp_path / "canonical.parquet"
    detections_path = tmp_path / "detections.parquet"
    scores_path = tmp_path / "scores.parquet"
    canonical.write_parquet(canonical_path)
    detections.write_parquet(detections_path)
    scores.write_parquet(scores_path)

    outputs = write_object_evidence_outputs(
        canonical_records_path=canonical_path,
        detections_path=detections_path,
        scores_path=scores_path,
        joined_output_path=tmp_path / "object_evidence_joined.parquet",
        photo_summary_output_path=tmp_path / "photo_evidence_summary.parquet",
        species_context=_context(),
    )

    joined = pl.read_parquet(outputs.object_evidence_joined).to_dicts()
    summary = pl.read_parquet(outputs.photo_evidence_summary).to_dicts()
    assert len(joined) == 1
    assert joined[0]["source"] == "flickr"
    assert joined[0]["flickr_photo_id"] == "photo-no-det"
    assert joined[0]["detection_status"] == "no_detection"
    assert summary == [
        {
            "source": "flickr",
            "flickr_photo_id": "photo-no-det",
            "best_detection_id": None,
            "detection_count": 0,
            "best_object_occurrence_bin": None,
            "best_object_species_top1": None,
            "best_object_score": None,
            "photo_occurrence_bin": "in_review",
            "photo_bin_reason": "no_detection_strong_text_evidence",
            "all_detection_ids": [],
            "all_candidate_species": ["Danaus plexippus"],
        }
    ]


def test_no_detection_without_species_text_bins_photo_as_no_butterfly(tmp_path) -> None:
    canonical = pl.DataFrame(
        [
            {
                "source": "flickr",
                "flickr_photo_id": "photo-no-butterfly",
                "source_record_hash": "sha256:source-no-butterfly",
                "image_url": "https://live.staticflickr.com/photo-no-butterfly.jpg",
                "photo_page_url": "https://www.flickr.com/photos/u/photo-no-butterfly",
                "title": "empty garden path",
                "raw_tags": "garden path",
                "latitude": 45.0,
                "longitude": -93.0,
                "date_taken": "2024-07-01",
            }
        ]
    )
    detections = pl.DataFrame(
        [
            {
                "source": "flickr",
                "flickr_photo_id": "photo-no-butterfly",
                "source_record_hash": "sha256:source-no-butterfly",
                "image_url": "https://live.staticflickr.com/photo-no-butterfly.jpg",
                "photo_page_url": "https://www.flickr.com/photos/u/photo-no-butterfly",
                "detection_id": "no-detection-photo-no-butterfly",
                "detector_backend": "fake",
                "detector_model_id": "fake-detector",
                "detector_model_version": "v1",
                "detector_checkpoint": "checkpoint-a",
                "detected_at": "2026-01-01T00:00:00+00:00",
                "bbox_xyxy": [],
                "bbox_xyxyn": [],
                "bbox_xywhn": [],
                "box_area_ratio": 0.0,
                "detector_label": "no_detection",
                "detector_score": 0.0,
                "objectness_score": None,
                "nms_group_id": None,
                "crop_padding_ratio": 0.12,
                "crop_hash": None,
                "crop_width": None,
                "crop_height": None,
                "crop_storage_policy": "ephemeral",
                "detection_status": "no_detection",
                "failure_reason": "no_butterfly_like_object",
            }
        ]
    )
    scores = empty_object_score_frame()
    canonical_path = tmp_path / "canonical.parquet"
    detections_path = tmp_path / "detections.parquet"
    scores_path = tmp_path / "scores.parquet"
    canonical.write_parquet(canonical_path)
    detections.write_parquet(detections_path)
    scores.write_parquet(scores_path)

    outputs = write_object_evidence_outputs(
        canonical_records_path=canonical_path,
        detections_path=detections_path,
        scores_path=scores_path,
        joined_output_path=tmp_path / "object_evidence_joined.parquet",
        photo_summary_output_path=tmp_path / "photo_evidence_summary.parquet",
        species_context=_context(),
    )

    summary = pl.read_parquet(outputs.photo_evidence_summary).to_dicts()[0]
    assert summary["photo_occurrence_bin"] == "bin"
    assert summary["photo_bin_reason"] == "no_butterfly_like_object"
    assert summary["all_candidate_species"] == []


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
