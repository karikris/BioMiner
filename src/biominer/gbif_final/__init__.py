"""Final, user-authorized GBIF legacy-lineage consolidation."""

from biominer.gbif_final.bounded import (
    assemble_parts,
    cleanup_bounded_state,
    preflight_assembly,
    seal_record_batches,
    seal_part,
    validate_assembled_output,
    validate_part_receipt,
)
from biominer.gbif_final.bounded_pipeline import (
    build_bounded_final_from_spine,
)
from biominer.gbif_final.dimensions import (
    build_derived_assertion_dimension,
    build_species_enrichment_dimension,
)
from biominer.gbif_final.global_sidecar import (
    seal_global_keyed_dimension,
    seal_global_sidecar_window,
)
from biominer.gbif_final.materialize import seal_temporal_enriched_window
from biominer.gbif_final.locator_index import (
    build_final_locator_index,
    validate_final_locator_index,
)
from biominer.gbif_final.pipeline import build_final_parquet, build_species_enrichments
from biominer.gbif_final.publication_audit import (
    audit_final_publication,
    validate_publication_audit,
)
from biominer.gbif_final.resolution_enrichment import (
    enrich_final_with_resolutions,
    validate_resolution_enriched_publication,
)
from biominer.gbif_final.spine import (
    build_source_spine,
    validate_source_spine,
)
from biominer.gbif_final.superseded_cleanup import (
    execute_superseded_cleanup,
    plan_superseded_cleanup,
    prepare_superseded_cleanup,
    validate_superseded_cleanup,
)
from biominer.gbif_final.windowed import (
    seal_keyed_dimension_window,
    seal_null_safe_composite_dimension_window,
    seal_ordinal_aligned_window,
)

__all__ = [
    "assemble_parts",
    "audit_final_publication",
    "build_final_parquet",
    "build_final_locator_index",
    "build_bounded_final_from_spine",
    "build_derived_assertion_dimension",
    "build_source_spine",
    "build_species_enrichment_dimension",
    "build_species_enrichments",
    "cleanup_bounded_state",
    "enrich_final_with_resolutions",
    "execute_superseded_cleanup",
    "plan_superseded_cleanup",
    "preflight_assembly",
    "prepare_superseded_cleanup",
    "seal_record_batches",
    "seal_global_keyed_dimension",
    "seal_global_sidecar_window",
    "seal_part",
    "seal_keyed_dimension_window",
    "seal_null_safe_composite_dimension_window",
    "seal_ordinal_aligned_window",
    "seal_temporal_enriched_window",
    "validate_assembled_output",
    "validate_final_locator_index",
    "validate_publication_audit",
    "validate_resolution_enriched_publication",
    "validate_source_spine",
    "validate_part_receipt",
    "validate_superseded_cleanup",
]
