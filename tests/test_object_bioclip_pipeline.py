from __future__ import annotations

import json
from pathlib import Path

import polars as pl
import pytest

from biominer.bioclip.candidate_sets import CandidateSet, CandidateTaxon, build_candidate_set, build_candidate_set_for_taxon_scope
from biominer.bioclip.classification_modes import HIERARCHICAL_BUTTERFLY_CLASSIFICATION
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
from biominer.bioclip.path_cascade_output import PATH_CASCADE_OUTPUT_SCHEMA
from biominer.detection.detector_base import DecodedImage
from biominer.detection.policy import DetectionPolicy
from biominer.detection.segmentation import make_segmenter
from biominer.detection.schema import DETECTION_OUTPUT_SCHEMA, empty_detection_frame
from biominer.evidence.join import build_photo_summary_from_joined_evidence
from biominer.run.taxon_scope import TaxonScope
from biominer.species.context import CommonName, RegionHint, SpeciesContext
from biominer.vision.gates import BioClipGateMode, BioClipGatePolicy
from factories import canonical_records, object_detection_row, object_detections


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


def _papilio_context() -> SpeciesContext:
    return SpeciesContext(
        scientific_name="Papilio demoleus",
        accepted_taxon_key="gbif:1938069",
        canonical_name="Papilio demoleus",
        family="Papilionidae",
        genus="Papilio",
        family_key="gbif:9417",
        genus_key="gbif:1938068",
        species_key="gbif:1938069",
        registry_version="registry-v1",
    )


def _fixture_candidate_set():
    return build_candidate_set(_context(), allow_single_target_fixture=True)


def test_object_visual_modes_are_segmentation_not_enhancement() -> None:
    assert OBJECT_VISUAL_MODES == ("whole_image", "detector_crop", "detector_crop_segmentation")

    source_paths = (
        Path("src/biominer/bioclip/object_runner.py"),
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


def test_object_score_schema_preserves_exact_versioned_cascade_dtypes() -> None:
    assert {
        field: OBJECT_SCORE_OUTPUT_SCHEMA[field]
        for field in PATH_CASCADE_OUTPUT_SCHEMA
    } == PATH_CASCADE_OUTPUT_SCHEMA
    for legacy_field in (
        "family_top3_accepted_taxon_keys",
        "genus_top8",
        "selected_family_key",
        "species_top20_scores",
        "species_top5_scores",
        "taxonomy_fingerprint",
        "classification_path_json",
        "rank_candidates_json",
        "candidate_counts_json",
        "pruning_decisions_json",
        "skipped_level_reasons_json",
        "rerank_mode",
    ):
        assert legacy_field not in OBJECT_SCORE_OUTPUT_SCHEMA


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
    return canonical_records()


def _detections() -> pl.DataFrame:
    return object_detections(
        object_detection_row(detection_id="det-1", score=0.9, crop_hash="sha256:crop-1"),
        object_detection_row(
            detection_id="det-2",
            score=0.6,
            bbox_xyxy=[10.0, 10.0, 20.0, 20.0],
            crop_hash="sha256:crop-2",
        ),
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


def test_candidate_set_preserves_regional_union_identity_and_provenance(
    tmp_path,
) -> None:
    path = tmp_path / "regional_candidate_species.parquet"
    fingerprint = "sha256:" + "a" * 64
    candidates = [
        (
            "gbif:5131654",
            "Danaus plexippus",
            "Nymphalidae",
            "Danaus",
            ["target"],
            True,
        ),
        (
            "gbif:5131655",
            "Danaus gilippus",
            "Nymphalidae",
            "Danaus",
            ["same_genus_range_overlap"],
            False,
        ),
        (
            "gbif:1902000",
            "Papilio machaon",
            "Papilionidae",
            "Papilio",
            ["known_mimic"],
            False,
        ),
    ]
    pl.DataFrame(
        [
            {
                "schema_version": "regional-candidate-species-v1.0.0",
                "candidate_set_id": "regional:test-set",
                "target_accepted_taxon_key": "gbif:5131654",
                "geo_cluster_id": "no_geo",
                "candidate_accepted_taxon_key": key,
                "scientific_name": name,
                "family": family,
                "genus": genus,
                "candidate_reason": reasons,
                "target_candidate": target,
                "candidate_priority": priority,
                "source_versions": ["registry:registry-v1", "occurrence:test-v1"],
                "candidate_set_fingerprint": fingerprint,
            }
            for priority, (key, name, family, genus, reasons, target) in enumerate(
                candidates
            )
        ]
    ).write_parquet(path)

    candidate_set = build_candidate_set(
        _context(),
        species_candidate_path=path,
        geospatial_scope="no_geo",
    )

    assert candidate_set.candidate_set_id == "regional:test-set"
    assert [candidate.accepted_taxon_key for candidate in candidate_set.species_candidates] == [
        "gbif:5131654",
        "gbif:5131655",
        "gbif:1902000",
    ]
    assert candidate_set.species_candidates[0].target_candidate is True
    assert candidate_set.species_candidates[2].candidate_reasons == ("known_mimic",)
    assert candidate_set.species_candidates[2].source_versions == (
        "registry:registry-v1",
        "occurrence:test-v1",
    )
    assert {candidate.genus for candidate in candidate_set.genus_candidates} == {
        "Danaus",
        "Papilio",
    }


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


def test_materialized_detector_crop_batches_reuse_duplicate_crop_hash_within_batch(tmp_path) -> None:
    base_detection = _detections().to_dicts()[0]
    detections = pl.DataFrame(
        [
            {**base_detection, "detection_id": "det-a", "crop_hash": "sha256:same-crop"},
            {**base_detection, "detection_id": "det-b", "crop_hash": "sha256:same-crop"},
        ]
    )
    batch_paths: list[Path] = []

    for batch in iter_materialized_detector_crop_batches(
        canonical_records=_canonical_records(),
        detections=detections,
        image_loader=lambda item: _decoded_image(),
        temp_dir=tmp_path,
        crop_batch_size=2,
        crop_target_px=3,
    ):
        item_paths = [Path(item["crop_path"]) for item in batch.items]
        assert len(batch.crop_path_by_hash) == 1
        assert item_paths[0] == item_paths[1]
        assert item_paths[0].exists()
        batch_paths.extend(item_paths)

    assert batch_paths
    assert all(not path.exists() for path in batch_paths)


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


def test_screen_object_detections_reuses_materialized_detector_crop_paths_and_cleans_after_success(tmp_path) -> None:
    calls: list[dict[str, object]] = []

    class PathBatchScorer:
        def score_label_sets_batch(self, image_paths, label_sets):  # noqa: ANN001, ANN202 - mirrors persistent scorer API.
            paths = tuple(Path(path) for path in image_paths)
            calls.append(
                {
                    "paths": paths,
                    "existing": tuple(path.exists() for path in paths),
                    "label_sets": tuple(label_sets),
                }
            )
            return {
                name: [
                    {
                        label: (
                            0.86
                            if "Danaus plexippus" in label
                            else 0.90
                            if label == "Nymphalidae"
                            else 0.1
                        )
                        for label in labels
                    }
                    for _path in paths
                ]
                for name, labels in label_sets.items()
            }

    crop_root = tmp_path / "crops"
    crop_scorer = EphemeralCropBioClipScorer(
        scorer=PathBatchScorer(),
        image_loader=lambda item: _decoded_image(),
        temp_dir=crop_root,
        crop_target_px=3,
        model_id="bioclip2_5",
        model_version="bioclip2_5_huge",
        model_checkpoint="checkpoint-a",
    )

    result = screen_object_detections(
        canonical_records=_canonical_records(),
        detections=_detections().head(1),
        species_context=_context(),
        candidate_set=_fixture_candidate_set(),
        scorer=crop_scorer,
        output_path=tmp_path / "object_scores.parquet",
        ablation_mode="detector_crop",
        bioclip_batch_size=1,
    )

    assert result.crops_scored == 1
    assert [call["existing"] for call in calls] == [(True,), (True,)]
    assert calls[0]["paths"] == calls[1]["paths"]
    assert calls[0]["label_sets"] == ("family", "genus", "species")
    assert calls[1]["label_sets"] == ("rerank",)
    assert all(not path.exists() for call in calls for path in call["paths"])
    assert not list(crop_root.rglob("*.ppm"))


def test_screen_object_detections_keeps_materialized_detector_crops_after_scorer_error(tmp_path) -> None:
    seen_paths: list[Path] = []

    class FailingPathBatchScorer:
        def score_label_sets_batch(self, image_paths, label_sets):  # noqa: ANN001, ANN202 - mirrors persistent scorer API.
            seen_paths.extend(Path(path) for path in image_paths)
            assert all(path.exists() for path in seen_paths)
            raise RuntimeError("bioclip boom")

    crop_root = tmp_path / "crops"
    crop_scorer = EphemeralCropBioClipScorer(
        scorer=FailingPathBatchScorer(),
        image_loader=lambda item: _decoded_image(),
        temp_dir=crop_root,
        crop_target_px=3,
        model_id="bioclip2_5",
        model_version="bioclip2_5_huge",
        model_checkpoint="checkpoint-a",
    )

    with pytest.raises(RuntimeError, match="bioclip boom"):
        screen_object_detections(
            canonical_records=_canonical_records(),
            detections=_detections().head(1),
            species_context=_context(),
            candidate_set=_fixture_candidate_set(),
            scorer=crop_scorer,
            output_path=tmp_path / "object_scores.parquet",
            ablation_mode="detector_crop",
            bioclip_batch_size=1,
        )

    assert seen_paths
    assert all(path.exists() for path in seen_paths)
    assert list(crop_root.rglob("*.ppm"))


def test_screen_object_detections_keeps_materialized_detector_crops_after_parquet_commit_failure(tmp_path, monkeypatch) -> None:
    seen_paths: list[Path] = []

    class PathBatchScorer:
        def score_label_sets_batch(self, image_paths, label_sets):  # noqa: ANN001, ANN202 - mirrors persistent scorer API.
            paths = tuple(Path(path) for path in image_paths)
            seen_paths.extend(paths)
            assert all(path.exists() for path in paths)
            return {
                name: [
                    {
                        label: (
                            0.86
                            if "Danaus plexippus" in label
                            else 0.90
                            if label == "Nymphalidae"
                            else 0.1
                        )
                        for label in labels
                    }
                    for _path in paths
                ]
                for name, labels in label_sets.items()
            }

    import biominer.bioclip.object_runner as object_runner_module

    original_write_parquet_batches = object_runner_module.write_parquet_batches

    def failing_final_write(batches, path, **kwargs):  # noqa: ANN001, ANN003, ANN202 - mirrors batch writer.
        target = Path(path)
        if target.name == "object_scores.parquet":
            raise RuntimeError("parquet commit failed")
        return original_write_parquet_batches(batches, target, **kwargs)

    monkeypatch.setattr(object_runner_module, "write_parquet_batches", failing_final_write)
    crop_root = tmp_path / "crops"
    crop_scorer = EphemeralCropBioClipScorer(
        scorer=PathBatchScorer(),
        image_loader=lambda item: _decoded_image(),
        temp_dir=crop_root,
        crop_target_px=3,
        model_id="bioclip2_5",
        model_version="bioclip2_5_huge",
        model_checkpoint="checkpoint-a",
    )

    with pytest.raises(RuntimeError, match="parquet commit failed"):
        screen_object_detections(
            canonical_records=_canonical_records(),
            detections=_detections().head(1),
            species_context=_context(),
            candidate_set=_fixture_candidate_set(),
            scorer=crop_scorer,
            output_path=tmp_path / "object_scores.parquet",
            ablation_mode="detector_crop",
            bioclip_batch_size=1,
        )

    assert seen_paths
    assert all(path.exists() for path in seen_paths)
    assert list(crop_root.rglob("*.ppm"))
    assert not (tmp_path / "object_scores.parquet").exists()


def test_screen_object_detections_retains_materialized_detector_crops_when_debug_requested(tmp_path) -> None:
    class PathBatchScorer:
        def score_label_sets_batch(self, image_paths, label_sets):  # noqa: ANN001, ANN202 - mirrors persistent scorer API.
            paths = [Path(path) for path in image_paths]
            assert all(path.exists() for path in paths)
            return {
                name: [
                    {label: (0.86 if "Danaus plexippus" in label else 0.1) for label in labels}
                    for _path in paths
                ]
                for name, labels in label_sets.items()
            }

    crop_root = tmp_path / "debug-crops"
    crop_scorer = EphemeralCropBioClipScorer(
        scorer=PathBatchScorer(),
        image_loader=lambda item: _decoded_image(),
        temp_dir=crop_root,
        crop_target_px=3,
        model_id="bioclip2_5",
        model_version="bioclip2_5_huge",
        model_checkpoint="checkpoint-a",
        retain_debug_crops=True,
    )

    screen_object_detections(
        canonical_records=_canonical_records(),
        detections=_detections().head(1),
        species_context=_context(),
        candidate_set=_fixture_candidate_set(),
        scorer=crop_scorer,
        output_path=tmp_path / "object_scores.parquet",
        ablation_mode="detector_crop",
        bioclip_batch_size=1,
    )

    assert list(crop_root.rglob("*.ppm"))


def test_screen_object_detections_materialized_path_skips_legacy_ineligible_without_image_load(tmp_path) -> None:
    image_loads: list[str] = []

    class FailingScorer:
        def score_label_sets_batch(self, image_paths, label_sets):  # noqa: ANN001, ANN202 - should not be called.
            raise AssertionError("legacy-ineligible detection should not be scored")

    def image_loader(item: dict[str, object]) -> DecodedImage:
        image_loads.append(str(item["detection_id"]))
        return _decoded_image()

    crop_scorer = EphemeralCropBioClipScorer(
        scorer=FailingScorer(),
        image_loader=image_loader,
        temp_dir=tmp_path / "crops",
        crop_target_px=3,
        model_id="bioclip2_5",
        model_version="bioclip2_5_huge",
        model_checkpoint="checkpoint-a",
    )
    detections = _detections().with_columns(pl.lit("moth_like").alias("detector_label"))

    result = screen_object_detections(
        canonical_records=_canonical_records(),
        detections=detections,
        species_context=_context(),
        candidate_set=_fixture_candidate_set(),
        scorer=crop_scorer,
        output_path=tmp_path / "object_scores.parquet",
        ablation_mode="detector_crop",
        detection_policy=DetectionPolicy(bioclip_eligible_labels=("butterfly_like",)),
    )

    assert result.crops_scored == 0
    assert image_loads == []
    assert not list((tmp_path / "crops").rglob("*.ppm"))


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


def test_object_bioclip_adaptive_batching_retries_memory_error_at_smaller_batch(tmp_path) -> None:
    class AdaptiveScorer(FakeObjectBioClipScorer):
        def __init__(self) -> None:
            super().__init__(
                {
                    "sha256:crop-1": {"a photo of Danaus plexippus": 0.82},
                    "sha256:crop-2": {"a photo of Danaus plexippus": 0.81},
                }
            )
            self.initial_batches: list[tuple[str, ...]] = []

        def score_label_sets_batch(self, items, label_sets):  # noqa: ANN001, ANN202 - mirrors object batch scorer API.
            if "species" in label_sets:
                self.initial_batches.append(tuple(str(item["detection_id"]) for item in items))
                if len(items) > 1:
                    raise RuntimeError("MPS memory allocation failed while scoring BioCLIP crops")
            return super().score_label_sets_batch(items, label_sets)

    scorer = AdaptiveScorer()

    result = screen_object_detections(
        canonical_records=_canonical_records(),
        detections=_detections(),
        species_context=_context(),
        candidate_set=_fixture_candidate_set(),
        scorer=scorer,
        output_path=tmp_path / "object_scores.parquet",
        ablation_mode="detector_crop",
        bioclip_batch_size=2,
        adaptive_batching=True,
        min_bioclip_batch_size=1,
    )

    assert result.crops_scored == 2
    assert result.adaptive_batching_enabled is True
    assert result.bioclip_batch_retries == 1
    assert result.bioclip_batch_size_initial == 2
    assert result.bioclip_batch_size_final == 1
    assert result.bioclip_batch_size_min == 1
    assert scorer.initial_batches == [("det-1", "det-2"), ("det-1",), ("det-2",)]


def test_object_bioclip_adaptive_batching_does_not_retry_non_memory_error(tmp_path) -> None:
    class NonMemoryScorer(FakeObjectBioClipScorer):
        def score_label_sets_batch(self, items, label_sets):  # noqa: ANN001, ANN202 - mirrors object batch scorer API.
            if "species" in label_sets:
                raise RuntimeError("invalid BioCLIP label tensor shape")
            return super().score_label_sets_batch(items, label_sets)

    with pytest.raises(RuntimeError, match="invalid BioCLIP label tensor shape"):
        screen_object_detections(
            canonical_records=_canonical_records(),
            detections=_detections(),
            species_context=_context(),
            candidate_set=_fixture_candidate_set(),
            scorer=NonMemoryScorer({}),
            output_path=tmp_path / "object_scores.parquet",
            ablation_mode="detector_crop",
            bioclip_batch_size=2,
            adaptive_batching=True,
            min_bioclip_batch_size=1,
        )


def test_object_bioclip_adaptive_batching_reports_min_batch_memory_failure(tmp_path) -> None:
    class AlwaysMemoryScorer(FakeObjectBioClipScorer):
        def score_label_sets_batch(self, items, label_sets):  # noqa: ANN001, ANN202 - mirrors object batch scorer API.
            if "species" in label_sets:
                raise RuntimeError(f"CUDA out of memory at batch size {len(items)}")
            return super().score_label_sets_batch(items, label_sets)

    with pytest.raises(RuntimeError, match="CUDA out of memory at batch size 1"):
        screen_object_detections(
            canonical_records=_canonical_records(),
            detections=_detections(),
            species_context=_context(),
            candidate_set=_fixture_candidate_set(),
            scorer=AlwaysMemoryScorer({}),
            output_path=tmp_path / "object_scores.parquet",
            ablation_mode="detector_crop",
            bioclip_batch_size=2,
            adaptive_batching=True,
            min_bioclip_batch_size=1,
        )


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


def test_object_bioclip_exclude_hard_negative_gate_scores_mixed_non_hard_labels(tmp_path) -> None:
    base_detection = _detections().to_dicts()[0]
    cases = [
        ("det-butterfly", "butterfly_like"),
        ("det-moth", "moth_like"),
        ("det-caterpillar", "caterpillar"),
        ("det-pupa", "pupa"),
        ("det-insect", "insect_like"),
        ("det-hard-negative", "hard_negative"),
    ]
    detections = pl.DataFrame(
        [
            {
                **base_detection,
                "detection_id": detection_id,
                "detector_label": label,
                "detection_status": "detected",
                "crop_hash": f"sha256:{detection_id}",
            }
            for detection_id, label in cases
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
        bioclip_gate_policy=BioClipGatePolicy(
            mode=BioClipGateMode.EXCLUDE_HARD_NEGATIVE,
            score_no_detection_whole_image=False,
        ),
    )

    expected_scored = ["det-butterfly", "det-moth", "det-caterpillar", "det-pupa", "det-insect"]
    assert result.crops_scored == len(expected_scored)
    assert result.frame["detection_id"].to_list() == expected_scored
    assert scorer.calls == [tuple(expected_scored), tuple(expected_scored)]


def test_object_evidence_join_preserves_non_scored_detection_rows(tmp_path) -> None:
    base_detection = _detections().to_dicts()[0]
    cases = [
        ("det-butterfly", "photo-1", "butterfly_like", "detected", "sha256:det-butterfly"),
        ("det-moth", "photo-moth", "moth_like", "detected", None),
        ("det-hard-negative", "photo-hard-negative", "hard_negative", "detected", None),
        ("det-no-detection", "photo-no-detection", "butterfly_like", "no_detection", None),
    ]
    detections = pl.DataFrame(
        [
            {
                **base_detection,
                "flickr_photo_id": photo_id,
                "detection_id": detection_id,
                "detector_label": label,
                "detection_status": status,
                "crop_hash": crop_hash,
                "crop_storage_policy": "ephemeral" if crop_hash else "not_created",
                "failure_reason": None if status == "detected" else "no_butterfly_like_object",
            }
            for detection_id, photo_id, label, status, crop_hash in cases
        ]
    )
    canonical = pl.DataFrame(
        [
            {**_canonical_records().to_dicts()[0], "flickr_photo_id": photo_id, "source_record_hash": f"sha256:{photo_id}"}
            for _detection_id, photo_id, _label, _status, _crop_hash in cases
        ]
    )
    scores_path = tmp_path / "scores.parquet"

    score_result = screen_object_detections(
        canonical_records=canonical,
        detections=detections,
        species_context=_context(),
        candidate_set=_fixture_candidate_set(),
        scorer=FakeObjectBioClipScorer({"sha256:det-butterfly": {"a photo of Danaus plexippus": 0.84}}),
        output_path=scores_path,
        ablation_mode="detector_crop",
    )

    canonical.write_parquet(tmp_path / "canonical.parquet")
    detections.write_parquet(tmp_path / "detections.parquet")
    outputs = write_object_evidence_outputs(
        canonical_records_path=tmp_path / "canonical.parquet",
        detections_path=tmp_path / "detections.parquet",
        scores_path=scores_path,
        joined_output_path=tmp_path / "joined.parquet",
        photo_summary_output_path=tmp_path / "summary.parquet",
        species_context=_context(),
    )
    joined = pl.read_parquet(outputs.object_evidence_joined).sort("detection_id")
    by_detection = {row["detection_id"]: row for row in joined.to_dicts()}

    assert score_result.frame.select("detection_id").to_series().to_list() == ["det-butterfly"]
    assert set(by_detection) == {"det-butterfly", "det-hard-negative", "det-moth", "det-no-detection"}
    assert by_detection["det-butterfly"]["classification_mode"] == "target_scope_object_screening"
    assert by_detection["det-moth"]["classification_mode"] is None
    assert by_detection["det-hard-negative"]["crop_storage_policy"] == "not_created"
    assert by_detection["det-no-detection"]["detection_status"] == "no_detection"


def test_object_bioclip_hierarchical_mode_requires_v3_store_and_cache(tmp_path) -> None:
    class FailingScorer:
        model_id = "fake-bioclip"
        model_version = "test"
        model_checkpoint = "fake-checkpoint"

        def score(self, item, labels):  # noqa: ANN001, ANN202 - should not be reached.
            raise AssertionError("hierarchical mode must not run target-scope object scoring")

    with pytest.raises(ValueError, match="classification-v3.*required"):
        screen_object_detections(
            canonical_records=_canonical_records(),
            detections=_detections().head(1),
            species_context=_context(),
            candidate_set=_fixture_candidate_set(),
            scorer=FailingScorer(),
            output_path=tmp_path / "object_scores.parquet",
            ablation_mode="detector_crop",
            classification_mode="hierarchical_butterfly_classification",
        )

    assert not (tmp_path / "object_scores.parquet").exists()




def test_object_bioclip_skips_legacy_ineligible_detector_labels(tmp_path) -> None:
    class FailingScorer:
        model_id = "fake-bioclip"
        model_version = "test"
        model_checkpoint = "fake-checkpoint"

        def score(self, item, labels):  # noqa: ANN001, ANN202 - mirrors scorer protocol.
            raise AssertionError(f"legacy-ineligible detection was sent to BioCLIP: {item.get('detector_label')}")

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
        "genus_top3",
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
    assert summary["photo_species_top1"] is None
    assert summary["photo_species_top1_key"] is None
    assert summary["photo_species_confidence_score"] is None
    assert summary["photo_multi_object_conflict"] is False
    assert summary["photo_review_reason"] == "negative_material_hard_negative_object"


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
    assert row["family_top1_score"] == pytest.approx(0.61)
    assert row["genus_top3"] == ["Danaus", "Limenitis"]
    assert row["genus_top1_score"] == pytest.approx(0.72)
    assert row["species_top5"] == ["Danaus plexippus", "Danaus gilippus", "Limenitis archippus"]
    assert row["target_species_score"] == 0.83


def test_object_bioclip_target_screening_reranks_only_first_pass_candidates(tmp_path) -> None:
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
    assert "a photo of Limenitis arthemis" in scorer.species_calls[1]
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
    assert row["target_species_score"] == 0.61
    assert row["target_species_rank"] == 5
    assert row["species_rerank_strategy"] == TARGET_SCOPE_SPECIES_RERANK_STRATEGY


def test_object_bioclip_top_k_settings_control_first_pass_and_rerank_candidates(tmp_path) -> None:
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

    class TopKRecordingScorer:
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
                    "a photo of Danaus plexippus": 0.50,
                    "a photo of Danaus gilippus": 0.93,
                    "a photo of Danaus erippus": 0.82,
                    "a photo of Danaus eresimus": 0.71,
                    "a photo of Limenitis archippus": 0.60,
                    "a photo of Limenitis arthemis": 0.40,
                }.items():
                    if label in scores:
                        scores[label] = score
                return scores
            for label, score in {
                "a photo of Danaus plexippus": 0.95,
                "a photo of Danaus gilippus": 0.44,
                "a photo of Danaus erippus": 0.43,
            }.items():
                if label in scores:
                    scores[label] = score
            return scores

    scorer = TopKRecordingScorer()

    result = screen_object_detections(
        canonical_records=_canonical_records(),
        detections=_detections().head(1),
        species_context=_context(),
        candidate_set=candidate_set,
        scorer=scorer,
        output_path=tmp_path / "object_scores.parquet",
        ablation_mode="detector_crop",
        species_first_pass_top_k=4,
        species_rerank_top_k=2,
    )

    row = result.frame.to_dicts()[0]
    assert len(scorer.species_calls) == 2
    assert scorer.species_calls[0] == candidate_set.prompt_labels("species")
    assert "a photo of Danaus gilippus" in scorer.species_calls[1]
    assert "a photo of Danaus erippus" in scorer.species_calls[1]
    assert "a photo of Danaus eresimus" in scorer.species_calls[1]
    assert "a photo of Limenitis archippus" in scorer.species_calls[1]
    assert "a photo of Danaus plexippus" in scorer.species_calls[1]
    assert row["family_top3"] == ["Nymphalidae"]
    assert row["species_top20"] == [
        "Danaus gilippus",
        "Danaus erippus",
        "Danaus eresimus",
        "Limenitis archippus",
        "Danaus plexippus",
    ]
    assert row["species_top5"] == ["Danaus plexippus", "Danaus gilippus"]
    assert row["species_first_pass_top_k"] == 4
    assert row["species_rerank_top_k"] == 2
    assert row["species_rerank_strategy"] == "complete_first_pass_top4_target_required"


def test_object_bioclip_family_rank_prioritizes_without_deleting_candidates(
    tmp_path,
) -> None:
    context = _context()
    candidate_set = CandidateSet(
        candidate_set_id="test_family_filtered",
        registry_version=context.registry_version,
        target_accepted_taxon_key=context.accepted_taxon_key,
        target_scientific_name=context.scientific_name,
        family_candidates=(
            CandidateTaxon("Danaus plexippus", family="Nymphalidae", genus="Danaus", accepted_taxon_key="gbif:5131654"),
            CandidateTaxon("Danaus gilippus", family="Nymphalidae", genus="Danaus", accepted_taxon_key="gbif:5131655"),
            CandidateTaxon("Danaus eresimus", family="Nymphalidae", genus="Danaus", accepted_taxon_key="gbif:5131657"),
            CandidateTaxon("Limenitis archippus", family="Nymphalidae", genus="Limenitis", accepted_taxon_key="gbif:1900000"),
            CandidateTaxon("Limenitis arthemis", family="Nymphalidae", genus="Limenitis", accepted_taxon_key="gbif:1900001"),
            CandidateTaxon("Papilio machaon", family="Papilionidae", genus="Papilio", accepted_taxon_key="gbif:1902000"),
        ),
        genus_candidates=(
            CandidateTaxon("Danaus", genus="Danaus"),
            CandidateTaxon("Limenitis", genus="Limenitis"),
            CandidateTaxon("Papilio", genus="Papilio"),
        ),
        species_candidates=(
            CandidateTaxon("Danaus plexippus", family="Nymphalidae", genus="Danaus", accepted_taxon_key="gbif:5131654"),
            CandidateTaxon("Danaus gilippus", family="Nymphalidae", genus="Danaus", accepted_taxon_key="gbif:5131655"),
            CandidateTaxon("Limenitis archippus", family="Nymphalidae", genus="Limenitis", accepted_taxon_key="gbif:1900000"),
            CandidateTaxon("Limenitis arthemis", family="Nymphalidae", genus="Limenitis", accepted_taxon_key="gbif:1900001"),
            CandidateTaxon("Danaus eresimus", family="Nymphalidae", genus="Danaus", accepted_taxon_key="gbif:5131657"),
            CandidateTaxon("Papilio machaon", family="Papilionidae", genus="Papilio", accepted_taxon_key="gbif:1902000"),
        ),
        prompt_variant_version="object-bioclip-prompts-v1",
        geospatial_scope=None,
        source_evidence=("test",),
    )

    class FamilyFilterScorer:
        model_id = "fake-bioclip"
        model_version = "test"
        model_checkpoint = "fake-checkpoint"

        def __init__(self) -> None:
            self.species_calls: list[tuple[str, ...]] = []

        def score(self, item: dict[str, object], labels: tuple[str, ...]) -> dict[str, float]:
            scores = {label: 0.0 for label in labels}
            if set(labels) == {"Nymphalidae", "Papilionidae"}:
                scores["Nymphalidae"] = 0.8
                scores["Papilionidae"] = 0.2
                return scores
            if set(labels) == {"Danaus", "Limenitis", "Papilio"}:
                scores["Danaus"] = 0.7
                scores["Papilio"] = 0.2
                return scores
            self.species_calls.append(labels)
            if len(self.species_calls) == 1:
                scores["a photo of Papilio machaon"] = 0.99
                scores["a photo of Danaus gilippus"] = 0.93
                scores["a photo of Limenitis archippus"] = 0.90
                scores["a photo of Danaus eresimus"] = 0.20
                scores["a photo of Danaus plexippus"] = 0.50
                return scores
            for label, score in {
                "a photo of Danaus gilippus": 0.90,
                "a photo of Danaus plexippus": 0.05,
                "a photo of Limenitis archippus": 0.40,
                "a photo of Danaus eresimus": 0.60,
            }.items():
                if label in scores:
                    scores[label] = score
            return scores

    scorer = FamilyFilterScorer()

    result = screen_object_detections(
        canonical_records=_canonical_records(),
        detections=_detections().head(1),
        species_context=_context(),
        candidate_set=candidate_set,
        scorer=scorer,
        output_path=tmp_path / "object_scores.parquet",
        ablation_mode="detector_crop",
        species_first_pass_top_k=4,
        species_rerank_top_k=2,
    )

    row = result.frame.to_dicts()[0]
    assert row["species_top20"] == [
        "Danaus gilippus",
        "Limenitis archippus",
        "Danaus plexippus",
        "Papilio machaon",
    ]
    assert len(scorer.species_calls) == 2
    assert "Nymphalidae" not in scorer.species_calls[0]
    assert "Papilionidae" not in scorer.species_calls[0]
    assert len(scorer.species_calls) == 2
    assert "a photo of Papilio machaon" in scorer.species_calls[1]
    assert "a photo of Danaus gilippus" in scorer.species_calls[1]
    assert "a photo of Limenitis archippus" in scorer.species_calls[1]
    assert "a photo of Danaus eresimus" not in scorer.species_calls[1]
    assert row["species_top5"][0] == "Danaus gilippus"
    assert row["species_top5"][1] == "Limenitis archippus"
    assert row["target_species_rank"] == 4
    assert row["target_species_score"] == 0.50
    assert row["species_rerank_strategy"] == "complete_first_pass_top4_target_required"
    provenance = {
        candidate["scientific_name"]: candidate
        for candidate in row["species_candidate_provenance"]
    }
    assert set(provenance) == {
        candidate.scientific_name for candidate in candidate_set.species_candidates
    }
    assert provenance["Papilio machaon"]["included_in_reference_comparison"] is True
    assert provenance["Papilio machaon"]["family_text_priority_match"] is False


def test_papilio_ranked_twentieth_stays_in_species_reference_comparison(
    tmp_path,
) -> None:
    context = _papilio_context()
    genera = [f"Genus{index:02d}" for index in range(1, 20)] + ["Papilio"]
    species = tuple(
        CandidateTaxon(
            scientific_name=(
                "Papilio demoleus" if genus == "Papilio" else f"{genus} species"
            ),
            accepted_taxon_key=(
                "gbif:1938069" if genus == "Papilio" else f"gbif:test-{index:02d}"
            ),
            family="Papilionidae",
            genus=genus,
            candidate_reasons=(
                ("target",) if genus == "Papilio" else ("regional_same_family",)
            ),
            source_versions=("regional-candidate-test-v1",),
            target_candidate=genus == "Papilio",
            candidate_priority=index,
        )
        for index, genus in enumerate(genera)
    )
    candidate_set = CandidateSet(
        candidate_set_id="regional:papilio-rank20",
        registry_version=context.registry_version,
        target_accepted_taxon_key=context.accepted_taxon_key,
        target_scientific_name=context.scientific_name,
        family_candidates=species,
        genus_candidates=species,
        species_candidates=species,
        prompt_variant_version="object-bioclip-prompts-v1",
        geospatial_scope="cluster-papilio",
        source_evidence=("regional-candidate-test-v1",),
    )

    class RankTwentyScorer:
        model_id = "fake-bioclip"
        model_version = "test"
        model_checkpoint = "fake-checkpoint"

        def __init__(self) -> None:
            self.species_calls: list[tuple[str, ...]] = []

        def score(
            self,
            item: dict[str, object],
            labels: tuple[str, ...],
        ) -> dict[str, float]:
            del item
            scores = {label: 0.0 for label in labels}
            if labels == ("Papilionidae",):
                scores["Papilionidae"] = 1.0
                return scores
            if "Papilio" in labels and "Papilio demoleus" not in labels:
                for rank, genus in enumerate(genera, start=1):
                    scores[genus] = float(len(genera) - rank)
                return scores
            self.species_calls.append(labels)
            for index, candidate in enumerate(species):
                scores[f"a photo of {candidate.scientific_name}"] = 1.0 - index / 100.0
            return scores

    scorer = RankTwentyScorer()
    result = screen_object_detections(
        canonical_records=_canonical_records(),
        detections=_detections().head(1),
        species_context=context,
        candidate_set=candidate_set,
        scorer=scorer,
        output_path=tmp_path / "object_scores.parquet",
        species_first_pass_top_k=20,
        species_rerank_top_k=5,
    )

    row = result.frame.row(0, named=True)
    assert row["genus_top3"] == ["Genus01", "Genus02", "Genus03"]
    assert "Papilio" not in row["genus_top3"]
    assert "a photo of Papilio demoleus" in scorer.species_calls[1]
    target_provenance = next(
        item
        for item in row["species_candidate_provenance"]
        if item["target_candidate"]
    )
    assert target_provenance["scientific_name"] == "Papilio demoleus"
    assert target_provenance["included_in_reference_comparison"] is True
    assert target_provenance["rerank_score"] is not None


def test_object_bioclip_rejects_incoherent_top_k_settings(tmp_path) -> None:
    with pytest.raises(ValueError, match="species_rerank_top_k must be <= species_first_pass_top_k"):
        screen_object_detections(
            canonical_records=_canonical_records(),
            detections=_detections().head(1),
            species_context=_context(),
            candidate_set=_fixture_candidate_set(),
            scorer=FakeObjectBioClipScorer({}),
            output_path=tmp_path / "object_scores.parquet",
            species_first_pass_top_k=2,
            species_rerank_top_k=3,
        )


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


def test_photo_summary_from_joined_evidence_keeps_open_classification_rows() -> None:
    joined = pl.DataFrame(
        [
            {
                "source": "flickr",
                "flickr_photo_id": "photo-1",
                "detection_id": "det-1",
                "classification_mode": HIERARCHICAL_BUTTERFLY_CLASSIFICATION,
                "selected_family": "Nymphalidae",
                "occurrence_bin": "in_review",
                "bin_reason": "hierarchical_open_classification_requires_review",
                "target_species_score": None,
                "species_top1_score": 0.91,
                "species_top1_margin": 0.24,
                "species_top1_scientific_name": "Danaus plexippus",
                "species_top1_accepted_taxon_key": "gbif:5131654",
                "species_top5": ["Danaus plexippus"],
                "species_top20": ["Danaus plexippus"],
                "detection_status": "detected",
                "detector_label": "butterfly_like",
            }
        ]
    )

    summary = build_photo_summary_from_joined_evidence(joined).to_dicts()[0]

    assert summary["best_object_score"] == 0.91
    assert summary["photo_occurrence_bin"] == "in_review"
    assert summary["photo_bin_reason"] == "hierarchical_open_classification_requires_review"
    assert summary["all_candidate_species"] == ["Danaus plexippus"]
    assert summary["all_selected_families"] == ["Nymphalidae"]
    assert summary["photo_selected_family"] == "Nymphalidae"
    assert summary["photo_species_top1"] == "Danaus plexippus"
    assert summary["photo_species_top1_key"] == "gbif:5131654"
    assert summary["photo_species_confidence_score"] == pytest.approx(0.91)
    assert summary["photo_species_margin"] == pytest.approx(0.24)
    assert summary["photo_multi_object_conflict"] is False
    assert summary["photo_review_reason"] == "hierarchical_open_classification_requires_review"


def test_photo_summary_from_joined_evidence_routes_hierarchical_multi_object_conflict_to_review() -> None:
    joined = pl.DataFrame(
        [
            {
                "source": "flickr",
                "flickr_photo_id": "photo-conflict",
                "detection_id": "det-weak-margin",
                "classification_mode": HIERARCHICAL_BUTTERFLY_CLASSIFICATION,
                "selected_family": "Nymphalidae",
                "occurrence_bin": "gold",
                "bin_reason": "target_species_score_ge_070",
                "target_species_score": 0.99,
                "species_top1_score": 0.88,
                "species_top1_margin": 0.04,
                "species_top1_scientific_name": "Danaus plexippus",
                "species_top1_accepted_taxon_key": "gbif:5131654",
                "species_top5": ["Danaus plexippus"],
                "species_top20": ["Danaus plexippus"],
                "detection_status": "detected",
                "detector_label": "butterfly_like",
            },
            {
                "source": "flickr",
                "flickr_photo_id": "photo-conflict",
                "detection_id": "det-strong-margin",
                "classification_mode": HIERARCHICAL_BUTTERFLY_CLASSIFICATION,
                "selected_family": "Papilionidae",
                "occurrence_bin": "gold",
                "bin_reason": "target_species_score_ge_070",
                "target_species_score": 0.10,
                "species_top1_score": 0.88,
                "species_top1_margin": 0.31,
                "species_top1_scientific_name": "Papilio demoleus",
                "species_top1_accepted_taxon_key": "gbif:9417001",
                "species_top5": ["Papilio demoleus"],
                "species_top20": ["Papilio demoleus"],
                "detection_status": "detected",
                "detector_label": "butterfly_like",
            },
        ]
    )

    summary = build_photo_summary_from_joined_evidence(joined).to_dicts()[0]

    assert summary["best_detection_id"] == "det-strong-margin"
    assert summary["photo_occurrence_bin"] == "in_review"
    assert summary["photo_bin_reason"] == "multiple_species"
    assert summary["photo_review_reason"] == "multiple_species"
    assert summary["photo_multi_object_conflict"] is True
    assert summary["all_candidate_species"] == ["Papilio demoleus", "Danaus plexippus"]
    assert summary["all_selected_families"] == ["Papilionidae", "Nymphalidae"]
    assert summary["photo_selected_family"] == "Papilionidae"
    assert summary["photo_species_top1"] == "Papilio demoleus"
    assert summary["photo_species_top1_key"] == "gbif:9417001"
    assert summary["photo_species_confidence_score"] == pytest.approx(0.88)
    assert summary["photo_species_margin"] == pytest.approx(0.31)


def test_photo_summary_from_joined_evidence_keeps_target_scope_bucket_behavior() -> None:
    joined = pl.DataFrame(
        [
            {
                "source": "flickr",
                "flickr_photo_id": "photo-target",
                "detection_id": "det-target",
                "classification_mode": "target_scope_object_screening",
                "occurrence_bin": "gold",
                "bin_reason": "target_species_score_ge_070",
                "target_species_score": 0.82,
                "species_top1_score": 0.82,
                "species_top1_margin": 0.22,
                "species_top1_scientific_name": "Danaus plexippus",
                "species_top1_accepted_taxon_key": "gbif:5131654",
                "species_top5": ["Danaus plexippus"],
                "species_top20": ["Danaus plexippus"],
                "detection_status": "detected",
                "detector_label": "butterfly_like",
            }
        ]
    )

    summary = build_photo_summary_from_joined_evidence(joined).to_dicts()[0]

    assert summary["photo_occurrence_bin"] == "gold"
    assert summary["photo_bin_reason"] == "target_species_score_ge_070"
    assert summary["all_selected_families"] == []
    assert summary["photo_selected_family"] is None
    assert summary["photo_species_top1"] == "Danaus plexippus"
    assert summary["photo_species_confidence_score"] == pytest.approx(0.82)
    assert summary["photo_species_margin"] == pytest.approx(0.22)
    assert summary["photo_multi_object_conflict"] is False
    assert summary["photo_review_reason"] == ""


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
        "all_selected_families",
        "photo_selected_family",
        "photo_species_top1",
        "photo_species_top1_key",
        "photo_species_confidence_score",
        "photo_species_margin",
        "photo_multi_object_conflict",
        "photo_review_reason",
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
            "all_selected_families": [],
            "all_selected_genera": [],
            "photo_selected_family": None,
            "photo_selected_family_node_id": None,
            "photo_selected_genus": None,
            "photo_selected_genus_node_id": None,
            "photo_species_top1": None,
            "photo_species_top1_key": None,
            "photo_species_confidence_score": None,
            "photo_species_margin": None,
            "photo_multi_object_conflict": False,
            "photo_review_reason": "no_detection_strong_text_evidence",
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
