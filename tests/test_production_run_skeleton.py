from __future__ import annotations

import json

import polars as pl
import pytest

from biominer.evidence import evidence_count_metrics
from biominer.evidence.join import write_object_evidence_outputs
from biominer.registry.trust_policy import (
    TrustTier,
    decide_name_trust,
    disabled_reason_for_candidate,
    should_enable_name_by_default,
    source_default_trust_tier,
)
from biominer.run import ProductionRunOrchestrator, ProductionRunRequest, RunManifest, RunPaths, RunStage, TaxonScope
from biominer.species.context import CommonName, SpeciesContext


def test_taxon_scope_construction_and_roundtrip() -> None:
    context = _species_context()
    scope = TaxonScope.from_species_context(context)

    assert scope.input_name == "Danaus plexippus"
    assert scope.input_rank == "species"
    assert scope.accepted_rank == "species"
    assert scope.species_count == 1
    assert scope.species_names == ("Danaus plexippus",)
    assert TaxonScope.from_dict(scope.to_dict()) == scope


def test_taxon_scope_validates_rank_and_species_contexts() -> None:
    with pytest.raises(ValueError, match="input_rank"):
        TaxonScope(
            input_name="Danaus",
            input_rank="subgenus",  # type: ignore[arg-type]
            accepted_taxon_key="gbif:5131",
            accepted_scientific_name="Danaus",
            accepted_rank="genus",
            registry_version="test-v1",
            species_contexts=(_species_context(),),
        )
    with pytest.raises(ValueError, match="species_contexts"):
        TaxonScope(
            input_name="Danaus",
            input_rank="genus",
            accepted_taxon_key="gbif:5131",
            accepted_scientific_name="Danaus",
            accepted_rank="genus",
            registry_version="test-v1",
            species_contexts=(),
        )


def test_run_paths_and_dry_run_manifest(tmp_path) -> None:
    scope = TaxonScope.from_species_context(_species_context())
    request = ProductionRunRequest(taxon="Danaus plexippus", rank="species", output_root=tmp_path, dry_run=True)
    orchestrator = ProductionRunOrchestrator(request, taxon_scope=scope)

    manifest_path = orchestrator.write_dry_run_manifest()
    manifest = RunManifest.read_json(manifest_path)

    assert manifest_path == tmp_path / "run_id=species_danaus_plexippus" / "run_manifest.json"
    assert manifest.storage_backend == "s3"
    assert manifest.workstore_backend == "postgres"
    assert manifest.taxon_scope == scope
    assert [stage.stage for stage in manifest.stages][:3] == [
        RunStage.RESOLVE_TAXON_SCOPE,
        RunStage.BUILD_REGISTRY,
        RunStage.COMPILE_QUERIES,
    ]
    assert json.loads(manifest_path.read_text(encoding="utf-8"))["species_count"] == 1


def test_run_paths_are_stable(tmp_path) -> None:
    paths = RunPaths.from_root(tmp_path, run_id="Family: Papilionidae")

    assert paths.run_root == tmp_path / "run_id=family_papilionidae"
    assert paths.query_definitions_path.name == "flickr_query_definitions.parquet"
    assert paths.object_evidence_path.name == "object_evidence_joined.parquet"
    assert paths.photo_summary_path.name == "photo_evidence_summary.parquet"
    assert paths.species_dir("Danaus plexippus") == paths.run_root / "species" / "danaus_plexippus"


def test_evidence_package_imports_and_metrics(tmp_path) -> None:
    joined = pl.DataFrame([{"occurrence_bin": "gold"}, {"occurrence_bin": "in_review"}, {"occurrence_bin": "gold"}])
    summary = pl.DataFrame([{"photo_occurrence_bin": "gold"}])

    assert evidence_count_metrics(joined, summary) == {
        "object_evidence_rows": 3,
        "photo_summary_rows": 1,
        "object_occurrence_bin_counts": {"gold": 2, "in_review": 1},
        "photo_occurrence_bin_counts": {"gold": 1},
    }
    assert callable(write_object_evidence_outputs)


def test_trust_policy_default_tiers_and_enablement() -> None:
    assert source_default_trust_tier("GBIF", "vernacular") is TrustTier.T2
    assert source_default_trust_tier("CoL", "scientific_synonym") is TrustTier.T1
    assert source_default_trust_tier("Wikidata", "vernacular") is TrustTier.T3
    assert source_default_trust_tier("iNaturalist", "vernacular_alias") is TrustTier.T4
    assert source_default_trust_tier("LibreTranslate", "generated_translation") is TrustTier.T5

    assert should_enable_name_by_default(TrustTier.T1, "low", "collision") is True
    assert should_enable_name_by_default(TrustTier.T2, "medium", "collision") is True
    assert should_enable_name_by_default(TrustTier.T3, "high", "none", external_taxon_link_confident=True) is True
    assert should_enable_name_by_default(TrustTier.T3, "high", "none") is False
    assert should_enable_name_by_default(TrustTier.T4, "medium", "none") is True
    assert should_enable_name_by_default(TrustTier.T4, "low", "none") is False
    assert should_enable_name_by_default(TrustTier.T5, "high", "none") is False
    assert should_enable_name_by_default(TrustTier.T5, "high", "none", review_state="accepted") is True


def test_trust_policy_disabled_reasons() -> None:
    assert disabled_reason_for_candidate(TrustTier.T3, "high", "none") == "wikidata_name_requires_confident_taxon_link"
    assert disabled_reason_for_candidate(TrustTier.T4, "medium", "ambiguous") == "name_collision_requires_review"
    assert disabled_reason_for_candidate(TrustTier.T5, "high", "none") == "generated_translation_requires_review"
    assert decide_name_trust(source="Wikidata", name_class="vernacular", confidence="high", collision_status="none").enabled is False


def _species_context() -> SpeciesContext:
    return SpeciesContext(
        scientific_name="Danaus plexippus",
        accepted_taxon_key="gbif:5130",
        canonical_name="Danaus plexippus",
        family="Nymphalidae",
        genus="Danaus",
        family_key="gbif:7017",
        genus_key="gbif:5131",
        species_key="gbif:5130",
        registry_version="test-v1",
        common_names=(CommonName(name="Monarch", language="en", source="GBIF", trust_tier="T2"),),
    )
