from __future__ import annotations

from pathlib import Path

import polars as pl

from biominer.bioclip.candidate_sets import CandidateSet, CandidateTaxon
from biominer.bioclip.embedding_cache import (
    candidate_text_embedding_rows,
    prepare_candidate_text_embedding_cache,
    prepare_object_image_embedding_cache,
    upsert_image_embedding_cache,
    upsert_text_embedding_cache,
    write_image_embedding_cache,
    write_text_embedding_cache,
)


def test_text_embedding_cache_computes_only_missing_labels(tmp_path: Path) -> None:
    cache_path = tmp_path / "candidate_text_embeddings.parquet"
    write_text_embedding_cache(
        [
            {
                "candidate_set_id": "candidate-set-1",
                "label": "Danaus plexippus",
                "accepted_taxon_key": "gbif:5131654",
                "rank": "species",
                "model_id": "bioclip",
                "model_checkpoint": "checkpoint-a",
                "embedding_dim": 2,
                "embedding": [1.0, 0.0],
                "created_at": "2026-01-01T00:00:00+00:00",
            }
        ],
        cache_path,
    )
    requested = [
        {
            "candidate_set_id": "candidate-set-1",
            "label": "Danaus plexippus",
            "accepted_taxon_key": "gbif:5131654",
            "rank": "species",
            "model_id": "bioclip",
            "model_checkpoint": "checkpoint-a",
        },
        {
            "candidate_set_id": "candidate-set-1",
            "label": "a photo of Danaus plexippus",
            "accepted_taxon_key": "gbif:5131654",
            "rank": "species",
            "model_id": "bioclip",
            "model_checkpoint": "checkpoint-a",
        },
        {
            "candidate_set_id": "candidate-set-1",
            "label": "a photo of Danaus plexippus",
            "accepted_taxon_key": "gbif:5131654",
            "rank": "species",
            "model_id": "bioclip",
            "model_checkpoint": "checkpoint-a",
        },
    ]
    calls: list[list[str]] = []

    def embed(labels: list[str]) -> list[list[float]]:
        calls.append(labels)
        return [[0.5, 0.5] for _label in labels]

    result = upsert_text_embedding_cache(requested, cache_path, embed_labels=embed, created_at="2026-01-02T00:00:00+00:00")
    second = upsert_text_embedding_cache(requested, cache_path, embed_labels=embed, created_at="2026-01-03T00:00:00+00:00")

    frame = pl.read_parquet(cache_path).sort("label")
    assert calls == [["a photo of Danaus plexippus"]]
    assert result.embeddings_computed == 1
    assert result.rows_added == 1
    assert result.rows_reused == 1
    assert second.embeddings_computed == 0
    assert second.rows_added == 0
    assert second.rows_reused == 2
    assert frame.get_column("label").to_list() == ["Danaus plexippus", "a photo of Danaus plexippus"]
    assert frame.filter(pl.col("label") == "a photo of Danaus plexippus").to_dicts()[0]["embedding"] == [0.5, 0.5]


def test_text_embedding_cache_batches_missing_labels(tmp_path: Path) -> None:
    cache_path = tmp_path / "candidate_text_embeddings.parquet"
    requested = [
        {
            "candidate_set_id": "candidate-set-1",
            "label": f"label-{index}",
            "accepted_taxon_key": f"gbif:{index}",
            "rank": "species",
            "model_id": "bioclip",
            "model_checkpoint": "checkpoint-a",
        }
        for index in range(5)
    ]
    calls: list[list[str]] = []

    def embed(labels: list[str]) -> list[list[float]]:
        calls.append(labels)
        return [[float(index), float(index + 1)] for index, _label in enumerate(labels)]

    result = upsert_text_embedding_cache(
        requested,
        cache_path,
        embed_labels=embed,
        created_at="2026-01-02T00:00:00+00:00",
        batch_size=2,
    )

    assert calls == [["label-0", "label-1"], ["label-2", "label-3"], ["label-4"]]
    assert result.embeddings_computed == 5
    assert result.rows_added == 5
    assert pl.read_parquet(cache_path).height == 5


def test_candidate_text_embedding_rows_derive_stage_labels_from_candidate_set() -> None:
    candidate_set = CandidateSet(
        candidate_set_id="candidate-set-1",
        registry_version="registry-v1",
        target_accepted_taxon_key="gbif:5131654",
        target_scientific_name="Danaus plexippus",
        family_candidates=(CandidateTaxon(scientific_name="Danaus plexippus", accepted_taxon_key="gbif:5131654", family="Nymphalidae"),),
        genus_candidates=(CandidateTaxon(scientific_name="Danaus plexippus", accepted_taxon_key="gbif:5131654", genus="Danaus"),),
        species_candidates=(
            CandidateTaxon(
                scientific_name="Danaus plexippus",
                accepted_taxon_key="gbif:5131654",
                family="Nymphalidae",
                genus="Danaus",
                common_names=("monarch butterfly", "monarch butterfly"),
            ),
            CandidateTaxon(scientific_name="Danaus gilippus", accepted_taxon_key="gbif:5131660", family="Nymphalidae", genus="Danaus"),
        ),
        prompt_variant_version="object-bioclip-prompts-v1",
        geospatial_scope=None,
        source_evidence=("fixture",),
    )

    rows = candidate_text_embedding_rows(candidate_set, model_id="bioclip", model_checkpoint="checkpoint-a")

    assert [(row["label"], row["rank"], row["accepted_taxon_key"]) for row in rows] == [
        ("Nymphalidae", "family", None),
        ("Danaus", "genus", None),
        ("Danaus plexippus", "species", "gbif:5131654"),
        ("a photo of Danaus plexippus", "species", "gbif:5131654"),
        ("monarch butterfly", "species", "gbif:5131654"),
        ("Danaus gilippus", "species", "gbif:5131660"),
        ("a photo of Danaus gilippus", "species", "gbif:5131660"),
    ]
    assert {row["candidate_set_id"] for row in rows} == {"candidate-set-1"}
    assert {row["model_id"] for row in rows} == {"bioclip"}
    assert {row["model_checkpoint"] for row in rows} == {"checkpoint-a"}


def test_candidate_text_embedding_rows_include_family_genus_and_species_stages() -> None:
    candidate_set = CandidateSet(
        candidate_set_id="candidate-set-staged",
        registry_version="registry-v1",
        target_accepted_taxon_key="gbif:5131654",
        target_scientific_name="Danaus plexippus",
        family_candidates=(CandidateTaxon(scientific_name="Nymphalidae", accepted_taxon_key="gbif:7017", rank="family"),),
        genus_candidates=(CandidateTaxon(scientific_name="Danaus", accepted_taxon_key="gbif:5131645", rank="genus"),),
        species_candidates=(CandidateTaxon(scientific_name="Danaus plexippus", accepted_taxon_key="gbif:5131654", rank="species"),),
        prompt_variant_version="object-bioclip-prompts-v1",
        geospatial_scope="global",
        source_evidence=("fixture",),
    )

    rows = candidate_text_embedding_rows(candidate_set, model_id="bioclip", model_checkpoint="checkpoint-a")

    assert [(row["label"], row["rank"], row["accepted_taxon_key"]) for row in rows] == [
        ("Nymphalidae", "family", "gbif:7017"),
        ("Danaus", "genus", "gbif:5131645"),
        ("Danaus plexippus", "species", "gbif:5131654"),
        ("a photo of Danaus plexippus", "species", "gbif:5131654"),
    ]

def test_candidate_text_embedding_rows_include_same_family_genus_stage_labels() -> None:
    candidate_set = CandidateSet(
        candidate_set_id="candidate-set-family-genera",
        registry_version="registry-v1",
        target_accepted_taxon_key="gbif:5131654",
        target_scientific_name="Danaus plexippus",
        family_candidates=(
            CandidateTaxon(scientific_name="Danaus plexippus", accepted_taxon_key="gbif:5131654", family="Nymphalidae", genus="Danaus"),
            CandidateTaxon(scientific_name="Limenitis archippus", accepted_taxon_key="gbif:1900000", family="Nymphalidae", genus="Limenitis"),
        ),
        genus_candidates=(CandidateTaxon(scientific_name="Danaus plexippus", accepted_taxon_key="gbif:5131654", family="Nymphalidae", genus="Danaus"),),
        species_candidates=(CandidateTaxon(scientific_name="Danaus plexippus", accepted_taxon_key="gbif:5131654", family="Nymphalidae", genus="Danaus"),),
        prompt_variant_version="object-bioclip-prompts-v1",
        geospatial_scope="global",
        source_evidence=("fixture",),
    )

    rows = candidate_text_embedding_rows(candidate_set, model_id="bioclip", model_checkpoint="checkpoint-a")

    assert [(row["label"], row["rank"], row["accepted_taxon_key"]) for row in rows if row["rank"] == "genus"] == [
        ("Danaus", "genus", None),
        ("Limenitis", "genus", None),
    ]


def test_prepare_candidate_text_embedding_cache_uses_candidate_set_and_reuses_cached_labels(tmp_path: Path) -> None:
    cache_path = tmp_path / "candidate_text_embeddings.parquet"
    candidate_set = CandidateSet(
        candidate_set_id="candidate-set-cache",
        registry_version="registry-v1",
        target_accepted_taxon_key="gbif:5131654",
        target_scientific_name="Danaus plexippus",
        family_candidates=(CandidateTaxon(scientific_name="Nymphalidae", accepted_taxon_key="gbif:7017", rank="family"),),
        genus_candidates=(CandidateTaxon(scientific_name="Danaus", accepted_taxon_key="gbif:5131645", rank="genus"),),
        species_candidates=(CandidateTaxon(scientific_name="Danaus plexippus", accepted_taxon_key="gbif:5131654", rank="species"),),
        prompt_variant_version="object-bioclip-prompts-v1",
        geospatial_scope="global",
        source_evidence=("fixture",),
    )
    calls: list[list[str]] = []

    def embed(labels: list[str]) -> list[list[float]]:
        calls.append(labels)
        return [[float(index), float(index + 1)] for index, _label in enumerate(labels)]

    first = prepare_candidate_text_embedding_cache(
        candidate_set,
        cache_path,
        model_id="bioclip",
        model_checkpoint="checkpoint-a",
        embed_labels=embed,
        created_at="2026-01-02T00:00:00+00:00",
    )
    second = prepare_candidate_text_embedding_cache(
        candidate_set,
        cache_path,
        model_id="bioclip",
        model_checkpoint="checkpoint-a",
        embed_labels=embed,
        created_at="2026-01-03T00:00:00+00:00",
    )

    assert calls == [["Nymphalidae", "Danaus", "Danaus plexippus", "a photo of Danaus plexippus"]]
    assert first.embeddings_computed == 4
    assert first.rows_added == 4
    assert second.embeddings_computed == 0
    assert second.rows_added == 0
    assert pl.read_parquet(cache_path).sort("rank", "label").select(["label", "rank"]).to_dicts() == [
        {"label": "Nymphalidae", "rank": "family"},
        {"label": "Danaus", "rank": "genus"},
        {"label": "Danaus plexippus", "rank": "species"},
        {"label": "a photo of Danaus plexippus", "rank": "species"},
    ]




def test_image_embedding_cache_reuses_existing_crop_hash_without_recomputing(tmp_path: Path) -> None:
    cache_path = tmp_path / "object_image_embeddings.parquet"
    write_image_embedding_cache(
        [
            {
                "source": "flickr",
                "flickr_photo_id": "photo-1",
                "detection_id": "det-1",
                "crop_hash": "sha256:crop-a",
                "model_id": "bioclip",
                "model_checkpoint": "checkpoint-a",
                "embedding_dim": 2,
                "embedding": [0.9, 0.1],
                "created_at": "2026-01-01T00:00:00+00:00",
            }
        ],
        cache_path,
    )
    requested = [
        {
            "source": "flickr",
            "flickr_photo_id": "photo-1",
            "detection_id": "det-1",
            "crop_hash": "sha256:crop-a",
            "model_id": "bioclip",
            "model_checkpoint": "checkpoint-a",
        },
        {
            "source": "flickr",
            "flickr_photo_id": "photo-2",
            "detection_id": "det-2",
            "crop_hash": "sha256:crop-a",
            "model_id": "bioclip",
            "model_checkpoint": "checkpoint-a",
        },
        {
            "source": "flickr",
            "flickr_photo_id": "photo-3",
            "detection_id": "det-3",
            "crop_hash": "sha256:crop-b",
            "model_id": "bioclip",
            "model_checkpoint": "checkpoint-a",
        },
    ]
    calls: list[list[str]] = []

    def embed(rows: list[dict[str, object]]) -> list[list[float]]:
        calls.append([str(row["crop_hash"]) for row in rows])
        return [[0.2, 0.8] for _row in rows]

    result = upsert_image_embedding_cache(requested, cache_path, embed_images=embed, created_at="2026-01-02T00:00:00+00:00")
    second = upsert_image_embedding_cache(requested, cache_path, embed_images=embed, created_at="2026-01-03T00:00:00+00:00")

    rows = pl.read_parquet(cache_path).sort("detection_id").to_dicts()
    assert calls == [["sha256:crop-b"]]
    assert result.embeddings_computed == 1
    assert result.rows_reused == 1
    assert result.rows_added == 2
    assert second.embeddings_computed == 0
    assert second.rows_added == 0
    assert [(row["detection_id"], row["embedding"]) for row in rows] == [
        ("det-1", [0.9, 0.1]),
        ("det-2", [0.9, 0.1]),
        ("det-3", [0.2, 0.8]),
    ]


def test_prepare_object_image_embedding_cache_uses_crop_paths_without_persisting_them(tmp_path: Path) -> None:
    cache_path = tmp_path / "object_image_embeddings.parquet"
    crop_a = tmp_path / "crop-a.ppm"
    crop_b = tmp_path / "crop-b.ppm"
    crop_a.write_bytes(b"P6\n1 1\n255\nabc")
    crop_b.write_bytes(b"P6\n1 1\n255\ndef")
    write_image_embedding_cache(
        [
            {
                "source": "flickr",
                "flickr_photo_id": "photo-1",
                "detection_id": "det-1",
                "crop_hash": "sha256:crop-a",
                "model_id": "bioclip",
                "model_checkpoint": "checkpoint-a",
                "embedding_dim": 2,
                "embedding": [0.9, 0.1],
                "created_at": "2026-01-01T00:00:00+00:00",
            }
        ],
        cache_path,
    )
    rows = [
        {"source": "flickr", "flickr_photo_id": "photo-1", "detection_id": "det-1", "crop_hash": "sha256:crop-a"},
        {"source": "flickr", "flickr_photo_id": "photo-2", "detection_id": "det-2", "crop_hash": "sha256:crop-a"},
        {"source": "flickr", "flickr_photo_id": "photo-3", "detection_id": "det-3", "crop_hash": "sha256:crop-b"},
    ]
    calls: list[list[Path]] = []

    def embed(paths: list[Path]) -> list[list[float]]:
        calls.append(paths)
        return [[0.2, 0.8] for _path in paths]

    result = prepare_object_image_embedding_cache(
        rows,
        cache_path,
        model_id="bioclip",
        model_checkpoint="checkpoint-a",
        crop_path_by_hash={"sha256:crop-a": crop_a, "sha256:crop-b": crop_b},
        embed_image_paths=embed,
        created_at="2026-01-02T00:00:00+00:00",
    )

    frame = pl.read_parquet(cache_path).sort("detection_id")
    assert calls == [[crop_b]]
    assert result.embeddings_computed == 1
    assert result.rows_reused == 1
    assert result.rows_added == 2
    assert "image_path" not in frame.columns
    assert frame.select(["detection_id", "embedding"]).to_dicts() == [
        {"detection_id": "det-1", "embedding": [0.9, 0.1]},
        {"detection_id": "det-2", "embedding": [0.9, 0.1]},
        {"detection_id": "det-3", "embedding": [0.2, 0.8]},
    ]
