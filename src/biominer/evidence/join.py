from __future__ import annotations

from pathlib import Path

import polars as pl

from biominer.bioclip.object_runner import (
    OBJECT_EVIDENCE_JOINED_SCHEMA,
    PHOTO_EVIDENCE_SUMMARY_SCHEMA,
    ObjectEvidenceOutputs,
    _object_evidence_joined,
    _photo_summary,
    empty_photo_summary_frame,
)
from biominer.species.context import SpeciesContext
from biominer.storage.parquet import write_parquet


def build_object_evidence_frames(
    *,
    canonical_source_records: pl.DataFrame,
    object_detections: pl.DataFrame,
    object_scores: pl.DataFrame,
    species_context: SpeciesContext | None = None,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Return joined object evidence and photo summary frames."""

    joined = _object_evidence_joined(
        canonical=canonical_source_records,
        detections=object_detections,
        scores=object_scores,
    )
    summary = _photo_summary(
        object_scores,
        canonical=canonical_source_records,
        detections=object_detections,
        species_context=species_context,
    )
    return joined, summary


def write_object_evidence_outputs(
    *,
    canonical_source_records: pl.DataFrame,
    object_detections: pl.DataFrame,
    object_scores: pl.DataFrame,
    joined_output_path: str | Path,
    photo_summary_output_path: str | Path,
    species_context: SpeciesContext | None = None,
) -> ObjectEvidenceOutputs:
    """Write joined object evidence through the production evidence package.

    This package-level boundary accepts already-loaded frames so the run
    orchestrator can compose local artifact stages without coupling directly to
    the BioCLIP object runner's path-oriented helper.
    """

    joined, summary = build_object_evidence_frames(
        canonical_source_records=canonical_source_records,
        object_detections=object_detections,
        object_scores=object_scores,
        species_context=species_context,
    )
    joined_path = write_parquet(joined, joined_output_path)
    summary_path = write_parquet(summary, photo_summary_output_path)
    return ObjectEvidenceOutputs(object_evidence_joined=joined_path, photo_evidence_summary=summary_path)


__all__ = [
    "OBJECT_EVIDENCE_JOINED_SCHEMA",
    "PHOTO_EVIDENCE_SUMMARY_SCHEMA",
    "ObjectEvidenceOutputs",
    "build_object_evidence_frames",
    "empty_photo_summary_frame",
    "write_object_evidence_outputs",
]
