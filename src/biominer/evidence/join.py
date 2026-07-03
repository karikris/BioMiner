from __future__ import annotations

from pathlib import Path

import polars as pl

from biominer.bioclip.object_runner import (
    OBJECT_EVIDENCE_JOINED_SCHEMA,
    PHOTO_EVIDENCE_SUMMARY_SCHEMA,
    ObjectEvidenceOutputs,
    empty_photo_summary_frame,
    write_object_evidence_outputs as _write_object_evidence_outputs,
)
from biominer.species.context import SpeciesContext


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

    The current implementation delegates to ``biominer.bioclip.object_runner``.
    Later cleanup phases can move the underlying implementation here without
    changing callers that already adopt the production package boundary.
    """

    return _write_object_evidence_outputs(
        canonical_source_records=canonical_source_records,
        object_detections=object_detections,
        object_scores=object_scores,
        joined_output_path=joined_output_path,
        photo_summary_output_path=photo_summary_output_path,
        species_context=species_context,
    )


__all__ = [
    "OBJECT_EVIDENCE_JOINED_SCHEMA",
    "PHOTO_EVIDENCE_SUMMARY_SCHEMA",
    "ObjectEvidenceOutputs",
    "empty_photo_summary_frame",
    "write_object_evidence_outputs",
]
