"""Final, user-authorized GBIF legacy-lineage consolidation."""

from biominer.gbif_final.bounded import (
    assemble_parts,
    preflight_assembly,
    seal_record_batches,
    seal_part,
    validate_part_receipt,
)
from biominer.gbif_final.dimensions import build_derived_assertion_dimension
from biominer.gbif_final.pipeline import build_final_parquet, build_species_enrichments
from biominer.gbif_final.spine import build_source_spine
from biominer.gbif_final.windowed import (
    seal_keyed_dimension_window,
    seal_null_safe_composite_dimension_window,
    seal_ordinal_aligned_window,
)

__all__ = [
    "assemble_parts",
    "build_final_parquet",
    "build_derived_assertion_dimension",
    "build_source_spine",
    "build_species_enrichments",
    "preflight_assembly",
    "seal_record_batches",
    "seal_part",
    "seal_keyed_dimension_window",
    "seal_null_safe_composite_dimension_window",
    "seal_ordinal_aligned_window",
    "validate_part_receipt",
]
