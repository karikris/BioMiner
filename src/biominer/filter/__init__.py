"""Metadata text extraction, metadata flags, and evidence classification."""

from biominer.filter.extractor import build_evidence_frame, extract_evidence_rows, write_staging_evidence
from biominer.filter.metadata_flags import flag_metadata_parquet, flag_metadata_records, load_metadata_keyword_groups
from biominer.filter.rules import classify_evidence_frame, classify_evidence_row, classify_evidence_rows

__all__ = [
    "build_evidence_frame",
    "classify_evidence_frame",
    "classify_evidence_row",
    "classify_evidence_rows",
    "extract_evidence_rows",
    "flag_metadata_parquet",
    "flag_metadata_records",
    "load_metadata_keyword_groups",
    "write_staging_evidence",
]
