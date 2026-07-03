from __future__ import annotations

import json
from pathlib import Path

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
from biominer.run import (
    ProductionRunOrchestrator,
    ProductionRunRequest,
    RunArtifactUris,
    RunManifest,
    RunPaths,
    RunStage,
    StageRecord,
    StageStatus,
    TaxonScope,
    resolve_taxon_scope_from_registry,
)
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


def test_resolve_taxon_scope_from_registry_expands_species_genus_and_family(tmp_path) -> None:
    registry = _write_rank_registry(tmp_path / "registry")

    species_scope = resolve_taxon_scope_from_registry(registry_dir=registry, input_name="Papilio demoleus", input_rank="species")
    genus_scope = resolve_taxon_scope_from_registry(registry_dir=registry, input_name="Papilio", input_rank="genus")
    family_scope = resolve_taxon_scope_from_registry(registry_dir=registry, input_name="Papilionidae", input_rank="family")
    auto_scope = resolve_taxon_scope_from_registry(registry_dir=registry, input_name="Papilio", input_rank="auto")

    assert species_scope.accepted_rank == "species"
    assert species_scope.species_names == ("Papilio demoleus",)
    assert species_scope.species_contexts[0].common_names[0].name == "Lime butterfly"
    assert genus_scope.accepted_rank == "genus"
    assert genus_scope.accepted_taxon_key == "gbif:90"
    assert genus_scope.species_names == ("Papilio demoleus", "Papilio machaon")
    assert family_scope.accepted_rank == "family"
    assert family_scope.species_names == ("Papilio demoleus", "Papilio machaon", "Shared name")
    assert auto_scope.accepted_rank == "genus"
    assert auto_scope.species_count == 2


def test_resolve_taxon_scope_reports_ambiguous_or_empty_registry_matches(tmp_path) -> None:
    registry = _write_rank_registry(tmp_path / "registry")

    with pytest.raises(ValueError, match="ambiguous taxon match"):
        resolve_taxon_scope_from_registry(registry_dir=registry, input_name="Shared name", input_rank="auto")

    with pytest.raises(ValueError, match="no species found under genus"):
        resolve_taxon_scope_from_registry(registry_dir=registry, input_name="Emptygenus", input_rank="genus")

    with pytest.raises(ValueError, match="species not found"):
        resolve_taxon_scope_from_registry(registry_dir=registry, input_name="Missing species", input_rank="species")


def test_run_paths_and_dry_run_manifest(tmp_path) -> None:
    scope = TaxonScope.from_species_context(_species_context())
    request = ProductionRunRequest(taxon="Danaus plexippus", rank="species", output_root=tmp_path, dry_run=True)
    orchestrator = ProductionRunOrchestrator(request, taxon_scope=scope)
    plan = orchestrator.plan()

    manifest_path = orchestrator.write_dry_run_manifest()
    manifest = RunManifest.read_json(manifest_path)

    assert manifest_path == tmp_path / "run_id=species_danaus_plexippus" / "run_manifest.json"
    assert plan.artifact_uris.query_definitions_uri.endswith("/registry/flickr_query_definitions.parquet")
    assert manifest.storage_backend == "s3"
    assert manifest.workstore_backend == "postgres"
    assert manifest.taxon_scope == scope
    assert manifest.query_counts == {"compiled_definitions": 0, "enqueued_work_items": 0}
    assert manifest.detection_counts == {"images_seen": 0, "detections": 0, "crops_created": 0}
    assert manifest.bioclip_counts == {"objects_scored": 0, "whole_images_scored": 0}
    assert manifest.evidence_counts == {"object_evidence_rows": 0, "photo_summary_rows": 0}
    assert manifest.outputs["manifest"].endswith("/run_manifest.json")
    assert [stage.stage for stage in manifest.stages][:3] == [
        RunStage.RESOLVE_TAXON_SCOPE,
        RunStage.BUILD_REGISTRY,
        RunStage.COMPILE_QUERIES,
    ]
    assert json.loads(manifest_path.read_text(encoding="utf-8"))["species_count"] == 1


def test_run_artifact_uris_are_s3_safe_and_species_scoped() -> None:
    uris = RunArtifactUris.from_prefix("s3://biominer/runs", run_id="Family: Papilionidae")

    assert uris.run_root_uri == "s3://biominer/runs/run_id=family_papilionidae"
    assert uris.manifest_uri == "s3://biominer/runs/run_id=family_papilionidae/run_manifest.json"
    assert uris.query_definitions_uri == "s3://biominer/runs/run_id=family_papilionidae/registry/flickr_query_definitions.parquet"
    assert uris.object_detections_uri == "s3://biominer/runs/run_id=family_papilionidae/staging/object_detections.parquet"
    assert uris.species_uri("Papilio demoleus") == "s3://biominer/runs/run_id=family_papilionidae/species/papilio_demoleus"
    assert uris.species_context_uri("Papilio demoleus").endswith("/species/papilio_demoleus/species_context.json")
    assert uris.to_dict()["photo_summary"].endswith("/staging/photo_evidence_summary.parquet")


def test_run_manifest_stage_status_and_count_roundtrip() -> None:
    scope = TaxonScope.from_species_context(_species_context())
    manifest = RunManifest(
        run_id="run-1",
        taxon_scope=scope,
        stages=(StageRecord(stage=RunStage.COMPILE_QUERIES),),
        query_counts={"compiled_definitions": 0},
    )

    manifest = manifest.with_stage_status(
        RunStage.COMPILE_QUERIES,
        StageStatus.COMPLETE,
        started_at="2026-01-01T00:00:00Z",
        ended_at="2026-01-01T00:00:01Z",
        metrics={"compiled_definitions": 12},
        outputs={"query_definitions": "s3://biominer/runs/run_id=run-1/registry/flickr_query_definitions.parquet"},
    )
    payload = manifest.to_dict()
    roundtrip = RunManifest.from_dict(payload)

    assert roundtrip.query_counts == {"compiled_definitions": 0}
    assert roundtrip.stages[0].status is StageStatus.COMPLETE
    assert roundtrip.stages[0].metrics == {"compiled_definitions": 12}
    assert roundtrip.stages[0].outputs["query_definitions"].startswith("s3://")


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


def _write_rank_registry(registry: Path) -> Path:
    registry.mkdir(parents=True, exist_ok=True)
    taxa_rows = [
        _taxon_row("gbif:10", "Papilionidae", "FAMILY", family_key="gbif:10", family="Papilionidae"),
        _taxon_row("gbif:90", "Papilio", "GENUS", parent_key="gbif:10", family_key="gbif:10", family="Papilionidae", genus_key="gbif:90", genus="Papilio"),
        _taxon_row("gbif:91", "Emptygenus", "GENUS", parent_key="gbif:10", family_key="gbif:10", family="Papilionidae", genus_key="gbif:91", genus="Emptygenus"),
        _taxon_row("gbif:100", "Papilio demoleus", "SPECIES", parent_key="gbif:90", family_key="gbif:10", family="Papilionidae", genus_key="gbif:90", genus="Papilio", species_key="gbif:100", species="Papilio demoleus"),
        _taxon_row("gbif:101", "Papilio machaon", "SPECIES", parent_key="gbif:90", family_key="gbif:10", family="Papilionidae", genus_key="gbif:90", genus="Papilio", species_key="gbif:101", species="Papilio machaon"),
        _taxon_row("gbif:20", "Nymphalidae", "FAMILY", family_key="gbif:20", family="Nymphalidae"),
        _taxon_row("gbif:190", "Danaus", "GENUS", parent_key="gbif:20", family_key="gbif:20", family="Nymphalidae", genus_key="gbif:190", genus="Danaus"),
        _taxon_row("gbif:200", "Danaus plexippus", "SPECIES", parent_key="gbif:190", family_key="gbif:20", family="Nymphalidae", genus_key="gbif:190", genus="Danaus", species_key="gbif:200", species="Danaus plexippus"),
        _taxon_row("gbif:300", "Shared name", "GENUS", parent_key="gbif:10", family_key="gbif:10", family="Papilionidae", genus_key="gbif:300", genus="Shared name"),
        _taxon_row("gbif:301", "Shared name", "SPECIES", parent_key="gbif:300", family_key="gbif:10", family="Papilionidae", genus_key="gbif:300", genus="Shared name", species_key="gbif:301", species="Shared name"),
    ]
    pl.DataFrame(taxa_rows).write_parquet(registry / "taxa.parquet")
    pl.DataFrame(
        [
            _name_row("gbif:100", "Papilio demoleus", "accepted_scientific", "la", "T1"),
            _name_row("gbif:100", "Lime butterfly", "vernacular", "eng", "T2"),
            _name_row("gbif:101", "Papilio machaon", "accepted_scientific", "la", "T1"),
            _name_row("gbif:200", "Danaus plexippus", "accepted_scientific", "la", "T1"),
            _name_row("gbif:301", "Shared name", "accepted_scientific", "la", "T1"),
        ]
    ).write_parquet(registry / "names.parquet")
    pl.DataFrame([{"source": "GBIF", "source_version": "fixture", "retrieved_at": "2026-01-01T00:00:00Z"}]).write_parquet(registry / "source_snapshots.parquet")
    (registry / "manifest.json").write_text(json.dumps({"registry_version": "rank-registry-v1"}), encoding="utf-8")
    return registry


def _taxon_row(
    accepted_taxon_key: str,
    scientific_name: str,
    rank: str,
    *,
    parent_key: str = "",
    family_key: str = "",
    family: str = "",
    genus_key: str = "",
    genus: str = "",
    species_key: str = "",
    species: str = "",
) -> dict[str, str]:
    return {
        "accepted_taxon_key": accepted_taxon_key,
        "scientific_name": scientific_name,
        "rank": rank,
        "parent_key": parent_key,
        "family_key": family_key,
        "family": family,
        "genus_key": genus_key,
        "genus": genus,
        "species_key": species_key,
        "species": species,
    }


def _name_row(accepted_taxon_key: str, display_name: str, name_class: str, language: str, trust_tier: str) -> dict[str, object]:
    return {
        "name_id": f"name:{accepted_taxon_key}:{display_name}",
        "registry_version": "rank-registry-v1",
        "accepted_taxon_key": accepted_taxon_key,
        "verbatim_name": display_name,
        "display_name": display_name,
        "language": language,
        "script": "Latn",
        "region": "",
        "bbox": "",
        "name_class": name_class,
        "source": "GBIF",
        "source_record_id": accepted_taxon_key,
        "trust_tier": trust_tier,
        "precision_tier": "high",
        "confidence": "high",
        "enabled": True,
        "disabled_reason": "",
    }
