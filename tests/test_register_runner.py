from __future__ import annotations

from pathlib import Path

from biominer.bioclip.image_cache import CachedImage
from biominer.bioclip.register_runner import process_records_with_registers
from biominer.bioclip.species_candidates import SpeciesCandidate


def _record(index: int) -> dict[str, object]:
    return {
        "source_record_id": str(index),
        "flickr_photo_id": str(index),
        "image_url": f"https://live.staticflickr.com/{index}.jpg",
        "image_url_kind": "url_l",
        "latitude": "-27.0",
        "longitude": "153.0",
        "date_taken": "2024-05-06 10:30:00",
        "title": "Papilio demoleus",
    }


def _danaus_record(index: int) -> dict[str, object]:
    record = _record(index)
    record["title"] = "Danaus plexippus"
    record["resolved_scientific_name"] = "Danaus plexippus"
    return record


def _candidates() -> list[SpeciesCandidate]:
    return [
        SpeciesCandidate(
            scientific_name="Danaus plexippus",
            canonical_name="Danaus plexippus",
            rank="species",
            family="Nymphalidae",
            genus="Danaus",
            source="test",
            source_taxon_id="3",
            is_target_species=False,
        ),
        SpeciesCandidate(
            scientific_name="Papilio demoleus",
            canonical_name="Papilio demoleus",
            rank="species",
            family="Papilionidae",
            genus="Papilio",
            source="test",
            source_taxon_id="1",
            is_target_species=True,
        ),
        SpeciesCandidate(
            scientific_name="Papilio machaon",
            canonical_name="Papilio machaon",
            rank="species",
            family="Papilionidae",
            genus="Papilio",
            source="test",
            source_taxon_id="2",
            is_target_species=False,
        ),
    ]


def test_register_runner_uses_four_twenty_image_registers_and_deletes_images(tmp_path) -> None:
    classifier = FakeRegisterClassifier()
    cache_calls: list[Path] = []

    result = process_records_with_registers(
        [_record(index) for index in range(85)],
        classifier=classifier,
        species_candidates=_candidates(),
        output_path=tmp_path / "triage.parquet",
        cache_root=tmp_path / "cache",
        cache_image=fake_cache(cache_calls),
        register_count=4,
        register_size=20,
        download_workers=4,
    )

    assert result.records_classified == 85
    assert result.images_deleted_after_classification == 85
    assert result.register_count == 4
    assert result.register_size == 20
    assert result.candidate_set_count == 1
    assert result.max_records_per_candidate_set == 85
    assert result.text_embedding_cache_hit_proxy == 84
    assert result.max_staged_images <= 80
    assert result.max_staged_images >= 20
    assert max(classifier.batch_sizes) == 20
    assert sorted(classifier.batch_sizes) == [5, 20, 20, 20, 20]
    assert {path.parent.name for path in cache_calls} <= {"register_0", "register_1", "register_2", "register_3"}
    assert not any(path.exists() for path in cache_calls)
    assert result.frame.filter(result.frame["occurrence_bin"] == "gold").height == 85
    assert (tmp_path / "gold_records.parquet").exists()
    assert (tmp_path / "silver_records.parquet").exists()
    assert (tmp_path / "bronze_records.parquet").exists()
    assert (tmp_path / "bin_records.parquet").exists()


def test_register_runner_routes_other_species_to_bronze(tmp_path) -> None:
    classifier = FakeRegisterClassifier(species_label="a photo of Papilio machaon")

    result = process_records_with_registers(
        [_record(1)],
        classifier=classifier,
        species_candidates=_candidates(),
        output_path=tmp_path / "triage.parquet",
        cache_root=tmp_path / "cache",
        cache_image=fake_cache([]),
    )

    row = result.frame.to_dicts()[0]
    assert row["species_top1_scientific_name"] == "Papilio machaon"
    assert row["occurrence_bin"] == "bronze"
    assert row["bin_reason"] == "below_50"


def test_register_runner_preserves_non_papilio_resolved_payload_species(tmp_path) -> None:
    classifier = FakeRegisterClassifier(species_label="a photo of Danaus plexippus", species_name="Danaus plexippus")

    result = process_records_with_registers(
        [_danaus_record(1)],
        classifier=classifier,
        species_candidates=_candidates(),
        output_path=tmp_path / "triage.parquet",
        cache_root=tmp_path / "cache",
        cache_image=fake_cache([]),
        candidate_strategy="metadata",
    )

    assert classifier.images_seen[0]["resolved_scientific_name"] == "Danaus plexippus"
    assert result.frame.to_dicts()[0]["species_top1_scientific_name"] == "Danaus plexippus"


def test_register_runner_preserves_scientific_name_for_common_name_prompt(tmp_path) -> None:
    classifier = FakeRegisterClassifier(
        species_label="a photo of lime butterfly",
        species_name="Papilio demoleus",
    )

    result = process_records_with_registers(
        [_record(1)],
        classifier=classifier,
        species_candidates=_candidates(),
        output_path=tmp_path / "triage.parquet",
        cache_root=tmp_path / "cache",
        cache_image=fake_cache([]),
    )

    row = result.frame.to_dicts()[0]
    assert classifier.received_species_prompt_variants is True
    assert row["species_top1_label"] == "a photo of lime butterfly"
    assert row["species_top1_scientific_name"] == "Papilio demoleus"


def test_register_runner_persists_species_taxonomy_metadata(tmp_path) -> None:
    classifier = FakeRegisterClassifier(species_name="Papilio demoleus")

    result = process_records_with_registers(
        [_record(1)],
        classifier=classifier,
        species_candidates=_candidates(),
        output_path=tmp_path / "triage.parquet",
        cache_root=tmp_path / "cache",
        cache_image=fake_cache([]),
    )

    row = result.frame.to_dicts()[0]
    assert row["species_top1_genus"] == "Papilio"
    assert row["species_top1_family"] == "Papilionidae"


def test_register_runner_groups_records_by_candidate_signature(tmp_path) -> None:
    classifier = FakeRegisterClassifier(species_label="a photo of Danaus plexippus", species_name="Danaus plexippus")
    records = [_danaus_record(1), _danaus_record(2), _record(3)]

    result = process_records_with_registers(
        records,
        classifier=classifier,
        species_candidates=_candidates(),
        output_path=tmp_path / "triage.parquet",
        cache_root=tmp_path / "cache",
        cache_image=fake_cache([]),
        candidate_strategy="metadata",
        register_count=1,
        register_size=10,
    )

    assert result.candidate_set_count == 2
    assert result.max_records_per_candidate_set == 2
    assert result.text_embedding_cache_hit_proxy == 1
    assert sorted(classifier.batch_sizes) == [1, 2]


def test_register_runner_writes_embedding_output_schema(tmp_path) -> None:
    classifier = FakeRegisterClassifier(
        species_label="a photo of Danaus plexippus",
        species_name="Danaus plexippus",
        image_embedding=[0.6, 0.8],
    )
    embedding_output = tmp_path / "embeddings.parquet"

    result = process_records_with_registers(
        [_danaus_record(1)],
        classifier=classifier,
        species_candidates=_candidates(),
        output_path=tmp_path / "triage.parquet",
        cache_root=tmp_path / "cache",
        cache_image=fake_cache([]),
        emit_image_embeddings=True,
        embedding_output=embedding_output,
    )

    embeddings = __import__("polars").read_parquet(embedding_output)
    row = embeddings.to_dicts()[0]
    assert result.embeddings_written == 1
    assert row["image_hash"] == "sha256:1"
    assert row["embedding_dimension"] == 2
    assert row["embedding"] == [0.6, 0.8]


def test_failed_download_remains_retryable(tmp_path) -> None:
    def failing_cache(_image_url: str, *, cache_root: str | Path) -> CachedImage:
        raise RuntimeError("temporary network failure")

    result = process_records_with_registers(
        [_record(1)],
        classifier=FakeRegisterClassifier(),
        species_candidates=_candidates(),
        output_path=tmp_path / "triage.parquet",
        cache_root=tmp_path / "cache",
        cache_image=failing_cache,
    )

    row = result.frame.to_dicts()[0]
    assert row["classification_status"] == "failed_download"
    assert row["retry_eligible"] is True
    assert row["occurrence_bin"] == "in_review"


def test_failed_bioclip_remains_retryable(tmp_path) -> None:
    result = process_records_with_registers(
        [_record(1)],
        classifier=FailingRegisterClassifier(),
        species_candidates=_candidates(),
        output_path=tmp_path / "triage.parquet",
        cache_root=tmp_path / "cache",
        cache_image=fake_cache([]),
    )

    row = result.frame.to_dicts()[0]
    assert row["classification_status"] == "failed_bioclip"
    assert row["retry_eligible"] is True
    assert row["occurrence_bin"] == "in_review"


def test_successful_records_are_skipped_idempotently(tmp_path) -> None:
    output = tmp_path / "triage.parquet"

    first = process_records_with_registers(
        [_record(1)],
        classifier=FakeRegisterClassifier(),
        species_candidates=_candidates(),
        output_path=output,
        cache_root=tmp_path / "cache",
        cache_image=fake_cache([]),
    )
    second = process_records_with_registers(
        [_record(1)],
        classifier=FakeRegisterClassifier(),
        species_candidates=_candidates(),
        output_path=output,
        cache_root=tmp_path / "cache",
        cache_image=fake_cache([]),
    )

    assert first.records_classified == 1
    assert second.records_classified == 0
    assert second.records_skipped_existing == 1
    assert second.frame.filter(second.frame["classification_status"] == "success").height == 1


class FakeRegisterClassifier:
    def __init__(
        self,
        *,
        species_label: str = "a photo of Papilio demoleus",
        species_name: str | None = None,
        image_embedding: list[float] | None = None,
    ) -> None:
        self.species_label = species_label
        self.species_name = species_name
        self.image_embedding = image_embedding
        self.batch_sizes: list[int] = []
        self.images_seen: list[dict[str, object]] = []
        self.received_species_prompt_variants = False

    def classify_images_with_label_sets(self, images, *, label_sets, species_prompt_variants=None, top_k=10, return_image_embeddings=False):  # noqa: ANN001 - test fake.
        self.batch_sizes.append(len(images))
        self.images_seen.extend(images)
        self.received_species_prompt_variants = species_prompt_variants is not None
        assert "species" in label_sets
        assert "triage" in label_sets
        rows = [
            {
                "species_top1_label": self.species_label,
                "species_top1_scientific_name": self.species_name,
                "species_top1_score": 0.91,
                "species_topk_json": [{"label": self.species_label, "score": 0.91}],
                "triage_top1_label": "a photo of an adult butterfly",
                "triage_top1_score": 0.88,
                "triage_topk_json": [{"label": "a photo of an adult butterfly", "score": 0.88}],
                "top1_label": self.species_label,
                "top1_score": 0.91,
                "topk_json": [{"label": self.species_label, "score": 0.91}],
            }
            for _image in images
        ]
        if return_image_embeddings and self.image_embedding is not None:
            for row in rows:
                row["image_embedding"] = self.image_embedding
        return rows


class FailingRegisterClassifier:
    def classify_images_with_label_sets(self, images, *, label_sets, species_prompt_variants=None, top_k=10, return_image_embeddings=False):  # noqa: ANN001 - test fake.
        raise RuntimeError("worker unavailable")


def fake_cache(paths: list[Path]):
    def cache_image(image_url: str, *, cache_root: str | Path) -> CachedImage:
        digest = image_url.rsplit("/", 1)[-1].removesuffix(".jpg")
        path = Path(cache_root) / f"{digest}.jpg"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"jpeg")
        paths.append(path)
        return CachedImage(
            source_url=image_url,
            path=path,
            image_hash=f"sha256:{digest}",
            content_type="image/jpeg",
            byte_size=4,
        )

    return cache_image
