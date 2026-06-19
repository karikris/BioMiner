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


def _candidates() -> list[SpeciesCandidate]:
    return [
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


class FakeRegisterClassifier:
    def __init__(self, *, species_label: str = "a photo of Papilio demoleus", species_name: str | None = None) -> None:
        self.species_label = species_label
        self.species_name = species_name
        self.batch_sizes: list[int] = []
        self.received_species_prompt_variants = False

    def classify_images_with_label_sets(self, images, *, label_sets, species_prompt_variants=None, top_k=10):  # noqa: ANN001 - test fake.
        self.batch_sizes.append(len(images))
        self.received_species_prompt_variants = species_prompt_variants is not None
        assert "species" in label_sets
        assert "triage" in label_sets
        return [
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
