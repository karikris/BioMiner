"""Final, user-authorized GBIF legacy-lineage consolidation."""

from biominer.gbif_final.bounded import (
    assemble_parts,
    preflight_assembly,
    seal_record_batches,
    seal_part,
    validate_part_receipt,
)
from biominer.gbif_final.pipeline import build_final_parquet, build_species_enrichments
from biominer.gbif_final.spine import build_source_spine

__all__ = [
    "assemble_parts",
    "build_final_parquet",
    "build_source_spine",
    "build_species_enrichments",
    "preflight_assembly",
    "seal_record_batches",
    "seal_part",
    "validate_part_receipt",
]
