from __future__ import annotations

import polars as pl

from flickr_bio_occurrence.benchmark.vision_checkpoint import build_checkpointed_vision_predictions


def _bronze(rows: int) -> pl.DataFrame:
    return pl.DataFrame(
        [
            {
                "flickr_photo_id": str(index),
                "image_url": f"https://live.staticflickr.com/{index}.jpg",
            }
            for index in range(rows)
        ]
    )


def _prediction(photo_id: str) -> dict[str, object]:
    return {
        "flickr_photo_id": photo_id,
        "model_family": "bioclip",
        "model_name": "imageomics/bioclip-2",
        "model_version": "bioclip2_5_huge",
        "model_checkpoint": "checkpoint",
        "model_hash": "sha256:test",
        "image_hash": f"sha256:image-{photo_id}",
        "image_url_used": f"https://live.staticflickr.com/{photo_id}.jpg",
        "top1_label": "a photo of Papilio demoleus",
        "top1_score": 0.9,
        "topk_json": [{"label": "a photo of Papilio demoleus", "score": 0.9}],
        "species_agreement_status": "exact_species_agreement",
        "vision_review_required": False,
        "created_at": "2026-06-03T00:00:00+00:00",
    }


def test_checkpoint_writes_partitioned_shards_instead_of_one_file_per_photo(tmp_path) -> None:
    def fake_classifier(row: dict[str, object]) -> dict[str, object]:
        return _prediction(str(row["flickr_photo_id"]))

    result = build_checkpointed_vision_predictions(
        _bronze(13_015),
        fake_classifier,
        tmp_path / "silver_vision_prediction",
        shard_size=1000,
    )

    assert result.completed == 13_015
    assert result.newly_completed == 13_015
    assert len(result.paths) == 14
    assert len(result.paths) < 13_015
    assert all("model_version=bioclip2_5_huge" in str(path) for path in result.paths)
    assert all(path.name == "part-00000.parquet" for path in result.paths)
    assert sum(result.rows_per_file.values()) == 13_015


def test_checkpoint_uses_batch_classifier_when_available(tmp_path) -> None:
    class BatchClassifier:
        def __init__(self) -> None:
            self.batch_sizes: list[int] = []

        def __call__(self, row: dict[str, object]) -> dict[str, object]:
            raise AssertionError("single-row classifier should not be called")

        def classify_rows(self, rows: list[dict[str, object]]) -> list[dict[str, object]]:
            self.batch_sizes.append(len(rows))
            return [_prediction(str(row["flickr_photo_id"])) for row in rows]

    classifier = BatchClassifier()

    result = build_checkpointed_vision_predictions(
        _bronze(3),
        classifier,
        tmp_path / "silver_vision_prediction",
        shard_size=2,
    )

    assert classifier.batch_sizes == [3]
    assert result.completed == 3
    assert len(result.paths) == 2


def test_checkpoint_skips_duplicate_prediction_keys_on_resume(tmp_path) -> None:
    calls: list[str] = []

    def fake_classifier(row: dict[str, object]) -> dict[str, object]:
        photo_id = str(row["flickr_photo_id"])
        calls.append(photo_id)
        return _prediction(photo_id)

    checkpoint_dir = tmp_path / "silver_vision_prediction"
    first = build_checkpointed_vision_predictions(_bronze(2), fake_classifier, checkpoint_dir)
    second = build_checkpointed_vision_predictions(_bronze(2), fake_classifier, checkpoint_dir)

    assert first.completed == 2
    assert first.skipped_existing == 0
    assert second.completed == 2
    assert second.newly_completed == 0
    assert second.skipped_existing == 2
    assert pl.read_parquet(second.paths).height == 2
    assert calls == ["0", "1", "0", "1"]
