from __future__ import annotations

from pathlib import Path

import polars as pl

from biominer.bioclip.candidate_sets import CandidateSet, CandidateTaxon
from biominer.bioclip.embedding_cache import (
    candidate_text_embedding_rows,
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
    assert second.embeddings_computed == 0
    assert second.rows_added == 0
    assert frame.get_column("label").to_list() == ["Danaus plexippus", "a photo of Danaus plexippus"]
    assert frame.filter(pl.col("label") == "a photo of Danaus plexippus").to_dicts()[0]["embedding"] == [0.5, 0.5]


def test_candidate_text_embedding_rows_are_derived_from_candidate_set_prompts() -> None:
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

    labels = [row["label"] for row in rows]
    assert labels == [
        "Danaus plexippus",
        "a photo of Danaus plexippus",
        "monarch butterfly",
        "Danaus gilippus",
        "a photo of Danaus gilippus",
    ]
    assert {row["candidate_set_id"] for row in rows} == {"candidate-set-1"}
    assert {row["model_id"] for row in rows} == {"bioclip"}
    assert {row["model_checkpoint"] for row in rows} == {"checkpoint-a"}
    assert {row["rank"] for row in rows} == {"species"}
    assert rows[0]["accepted_taxon_key"] == "gbif:5131654"
    assert rows[-1]["accepted_taxon_key"] == "gbif:5131660"


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
