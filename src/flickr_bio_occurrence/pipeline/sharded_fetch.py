from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from flickr_bio_occurrence.evidence.extractor import write_staging_evidence
from flickr_bio_occurrence.pipeline.job_queue import ClassificationJob, ClassificationJobQueue


@dataclass(frozen=True)
class EvidenceShardJob:
    evidence_parquet_path: Path
    job: ClassificationJob


def write_evidence_shard_and_enqueue(
    payloads: Iterable[dict[str, Any]],
    *,
    species_query: str,
    evidence_parquet_path: str | Path,
    queue: ClassificationJobQueue,
    model_version: str,
) -> EvidenceShardJob:
    path = write_staging_evidence(
        payloads,
        species_query=species_query,
        output_path=evidence_parquet_path,
    )
    job = queue.enqueue_evidence_shard(path, model_version=model_version)
    return EvidenceShardJob(evidence_parquet_path=path, job=job)
