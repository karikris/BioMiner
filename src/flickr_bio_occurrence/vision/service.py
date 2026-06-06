from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import polars as pl

from flickr_bio_occurrence.pipeline.job_queue import ClassificationJob, ClassificationJobQueue, COMPLETED
from flickr_bio_occurrence.storage.parquet_io import write_parquet_dataset


class EvidenceClassifier(Protocol):
    def __call__(self, row: dict[str, Any]) -> dict[str, Any]:
        ...


@dataclass(frozen=True)
class ProcessedVisionJob:
    job: ClassificationJob
    prediction_parquet_paths: list[Path]
    prediction_rows: int


class BioClipJobService:
    def __init__(
        self,
        *,
        queue: ClassificationJobQueue,
        classifier: EvidenceClassifier,
        prediction_output_dir: str | Path,
    ) -> None:
        self.queue = queue
        self.classifier = classifier
        self.prediction_output_dir = Path(prediction_output_dir)

    def process_next_job(self) -> ProcessedVisionJob | None:
        job = self.queue.claim_next()
        if job is None:
            return None
        try:
            result = self._process_claimed_job(job)
        except Exception as exc:
            self.queue.mark_failed(job.job_id, error=str(exc))
            raise
        completed = self.queue.mark_complete(job.job_id)
        return ProcessedVisionJob(
            job=completed,
            prediction_parquet_paths=result.prediction_parquet_paths,
            prediction_rows=result.prediction_rows,
        )

    def process_pending_jobs(self, *, limit: int | None = None) -> list[ProcessedVisionJob]:
        processed: list[ProcessedVisionJob] = []
        while limit is None or len(processed) < limit:
            result = self.process_next_job()
            if result is None:
                break
            processed.append(result)
        return processed

    def _process_claimed_job(self, job: ClassificationJob) -> ProcessedVisionJob:
        if job.status == COMPLETED:
            return ProcessedVisionJob(job=job, prediction_parquet_paths=[], prediction_rows=0)
        evidence = pl.read_parquet(job.evidence_parquet_path)
        rows = evidence.to_dicts()
        predictions = _classify_rows(self.classifier, rows)
        prediction_frame = pl.DataFrame(predictions) if predictions else pl.DataFrame()
        output_dir = self.prediction_output_dir / f"model_version={job.model_version}" / f"job_id={job.job_id}"
        paths = write_parquet_dataset(prediction_frame, output_dir) if prediction_frame.height else []
        return ProcessedVisionJob(
            job=job,
            prediction_parquet_paths=paths,
            prediction_rows=prediction_frame.height,
        )


def _classify_rows(classifier: EvidenceClassifier, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    classify_evidence_rows = getattr(classifier, "classify_evidence_rows", None)
    if callable(classify_evidence_rows):
        return list(classify_evidence_rows(rows))
    classify_rows = getattr(classifier, "classify_rows", None)
    if callable(classify_rows):
        return list(classify_rows(rows))
    return [classifier(row) for row in rows]
