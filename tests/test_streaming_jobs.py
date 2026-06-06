from __future__ import annotations

from datetime import UTC, datetime, timedelta

import polars as pl

from flickr_bio_occurrence.pipeline.job_queue import (
    CLAIMED,
    COMPLETED,
    FAILED,
    PENDING,
    ClassificationJobQueue,
)
from flickr_bio_occurrence.pipeline.sharded_fetch import write_evidence_shard_and_enqueue
from flickr_bio_occurrence.vision.service import BioClipJobService


def test_fetch_like_flow_writes_evidence_shard_and_enqueues_job(tmp_path) -> None:
    queue = ClassificationJobQueue(tmp_path / "queue.sqlite")

    result = write_evidence_shard_and_enqueue(
        [
            {
                "photos": {
                    "photo": [
                        {
                            "id": "1",
                            "title": "Papilio demoleus",
                            "url_l": "https://live.staticflickr.com/large.jpg",
                        }
                    ]
                }
            }
        ],
        species_query="Papilio demoleus",
        evidence_parquet_path=tmp_path / "evidence" / "part-00000.parquet",
        queue=queue,
        model_version="bioclip2_5_huge",
    )

    frame = pl.read_parquet(result.evidence_parquet_path)
    jobs = queue.list_jobs()
    assert frame["flickr_photo_id"][0] == "1"
    assert result.job.status == PENDING
    assert jobs == [result.job]
    assert result.job.evidence_parquet_path == result.evidence_parquet_path
    assert result.job.model_version == "bioclip2_5_huge"


def test_classifier_service_claims_processes_and_completes_job(tmp_path) -> None:
    evidence_path = _write_evidence(tmp_path, "1")
    queue = ClassificationJobQueue(tmp_path / "queue.sqlite")
    queue.enqueue_evidence_shard(evidence_path, model_version="bioclip2_5_huge")
    classifier = FakeBatchClassifier()
    service = BioClipJobService(
        queue=queue,
        classifier=classifier,
        prediction_output_dir=tmp_path / "predictions",
    )

    result = service.process_next_job()

    assert result is not None
    assert result.job.status == COMPLETED
    assert result.job.attempts == 1
    assert result.prediction_rows == 1
    assert len(result.prediction_parquet_paths) == 1
    predictions = pl.read_parquet(result.prediction_parquet_paths)
    assert predictions["flickr_photo_id"][0] == "1"
    assert classifier.batch_sizes == [1]


def test_completed_jobs_are_skipped_on_rerun(tmp_path) -> None:
    evidence_path = _write_evidence(tmp_path, "1")
    queue = ClassificationJobQueue(tmp_path / "queue.sqlite")
    queue.enqueue_evidence_shard(evidence_path, model_version="bioclip2_5_huge")
    classifier = FakeBatchClassifier()
    service = BioClipJobService(
        queue=queue,
        classifier=classifier,
        prediction_output_dir=tmp_path / "predictions",
    )

    first = service.process_pending_jobs()
    second = service.process_pending_jobs()
    queue.enqueue_evidence_shard(evidence_path, model_version="bioclip2_5_huge")
    third = service.process_pending_jobs()

    assert len(first) == 1
    assert second == []
    assert third == []
    assert queue.list_jobs()[0].status == COMPLETED
    assert classifier.batch_sizes == [1]


def test_service_reuses_one_externally_owned_classifier_across_jobs(tmp_path) -> None:
    queue = ClassificationJobQueue(tmp_path / "queue.sqlite")
    first_path = _write_evidence(tmp_path / "first", "1")
    second_path = _write_evidence(tmp_path / "second", "2")
    queue.enqueue_evidence_shard(first_path, model_version="bioclip2_5_huge")
    queue.enqueue_evidence_shard(second_path, model_version="bioclip2_5_huge")
    classifier = FakeBatchClassifier()
    service = BioClipJobService(
        queue=queue,
        classifier=classifier,
        prediction_output_dir=tmp_path / "predictions",
    )

    processed = service.process_pending_jobs()

    assert len(processed) == 2
    assert classifier.batch_sizes == [1, 1]
    assert classifier.instance_ids == [id(classifier), id(classifier)]
    assert [job.status for job in queue.list_jobs()] == [COMPLETED, COMPLETED]


def test_stale_claimed_jobs_can_be_retried_or_failed(tmp_path) -> None:
    queue = ClassificationJobQueue(tmp_path / "queue.sqlite")
    retry_path = _write_evidence(tmp_path / "retry", "1")
    fail_path = _write_evidence(tmp_path / "fail", "2")
    retry_job = queue.enqueue_evidence_shard(retry_path, model_version="bioclip2_5_huge")
    fail_job = queue.enqueue_evidence_shard(fail_path, model_version="bioclip2_5_huge")
    old = datetime(2026, 6, 1, tzinfo=UTC)
    queue.claim_next(now=old)
    queue.claim_next(now=old)

    retried = queue.retry_stale_claimed(
        stale_after=timedelta(hours=1),
        now=old + timedelta(hours=2),
        max_attempts=3,
    )
    retry_statuses = {job.job_id: job.status for job in retried}
    assert retry_statuses[retry_job.job_id] == PENDING
    assert retry_statuses[fail_job.job_id] == PENDING

    queue.claim_next(now=old)
    queue.claim_next(now=old)
    failed = queue.retry_stale_claimed(
        stale_after=timedelta(hours=1),
        now=old + timedelta(hours=2),
        max_attempts=2,
    )

    failed_statuses = {job.job_id: job.status for job in failed}
    assert failed_statuses[retry_job.job_id] == FAILED
    assert failed_statuses[fail_job.job_id] == FAILED
    assert {job.status for job in queue.list_jobs()} == {FAILED}


def test_claim_next_marks_pending_job_as_claimed(tmp_path) -> None:
    evidence_path = _write_evidence(tmp_path, "1")
    queue = ClassificationJobQueue(tmp_path / "queue.sqlite")
    queue.enqueue_evidence_shard(evidence_path, model_version="bioclip2_5_huge")

    claimed = queue.claim_next()

    assert claimed is not None
    assert claimed.status == CLAIMED
    assert claimed.claimed_at is not None
    assert claimed.attempts == 1


class FakeBatchClassifier:
    def __init__(self) -> None:
        self.batch_sizes: list[int] = []
        self.instance_ids: list[int] = []

    def classify_evidence_rows(self, rows: list[dict[str, object]]) -> list[dict[str, object]]:
        self.batch_sizes.append(len(rows))
        self.instance_ids.append(id(self))
        return [_prediction(row) for row in rows]


def _write_evidence(tmp_path, photo_id: str):
    tmp_path.mkdir(parents=True, exist_ok=True)
    path = tmp_path / "part-00000.parquet"
    pl.DataFrame(
        [
            {
                "flickr_photo_id": photo_id,
                "image_url": f"https://live.staticflickr.com/{photo_id}.jpg",
                "species_query": "Papilio demoleus",
                "species_text_match": True,
            }
        ]
    ).write_parquet(path)
    return path


def _prediction(row: dict[str, object]) -> dict[str, object]:
    return {
        "flickr_photo_id": row["flickr_photo_id"],
        "model_version": "bioclip2_5_huge",
        "model_checkpoint": "checkpoint",
        "image_hash": f"sha256:{row['flickr_photo_id']}",
        "image_url_used": row["image_url"],
        "top1_label": "a photo of Papilio demoleus",
        "top1_score": 0.9,
        "species_agreement_status": "exact_species_agreement",
    }
