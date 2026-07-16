from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

import biominer.run.orchestrator as orchestrator_module
from biominer.bioclip.classification_modes import (
    BUILD_WEEK_TARGET_AWARE_PROTOTYPE,
    TARGET_SCOPE_OBJECT_SCREENING,
)
from biominer.bioclip.prototype_mode import (
    BUILD_WEEK_PROTOTYPE_CONFIG_VERSION,
    BuildWeekPrototypeConfig,
)
from biominer.bioclip.prototype_support import (
    MetadataQualifiedPrototypePermit,
    PrototypeReadinessPermit,
)
from biominer.run import (
    ProductionRunOrchestrator,
    ProductionRunRequest,
    RunStage,
    StageExecutionResult,
    TaxonScope,
)
from biominer.species.context import SpeciesContext


TARGET_KEY = "gbif:1938069"
TARGET_NAME = "Papilio demoleus"
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _write_config(
    tmp_path: Path,
    *,
    overrides: dict[str, object] | None = None,
) -> tuple[Path, BuildWeekPrototypeConfig]:
    artifacts = {}
    for name in (
        "reference_bank_readiness",
        "support_manifest",
        "reference_embeddings",
        "candidate_score_evidence",
        "prototype_policy",
    ):
        path = tmp_path / f"{name}.artifact"
        path.write_bytes(f"{name}-frozen".encode())
        artifacts[name] = path
    payload: dict[str, object] = {
        "schema_version": BUILD_WEEK_PROTOTYPE_CONFIG_VERSION,
        "classification_mode": BUILD_WEEK_TARGET_AWARE_PROTOTYPE,
        "deployment_status": "prototype",
        "workflow": "reference-first",
        "storage_backend": "local",
        "s3_permitted": False,
        "target_accepted_taxon_key": TARGET_KEY,
        "target_scientific_name": TARGET_NAME,
        "model_revision": "model-revision-1",
        "preprocessing_version": "preprocessing-v1",
        "visual_input_version": "visual-input-v1",
        "classifier_fingerprint": "sha256:" + "a" * 64,
        "margin_policy_version": "margin-policy-v1",
        "target_always_scored": True,
        "complete_regional_candidate_union_scored": True,
        "hierarchy_pruning_permitted": False,
        "spatial_crop_permitted": False,
        "visual_input": "raw_full_image",
        "prototype_readiness_required": True,
        "prototype_support_bank_required": True,
        "silent_fallback_permitted": False,
        "output_status": "prototype",
        "legacy_classifier_available": True,
        "b0_baseline_available": True,
        "limitations": [
            "Prototype labels are not independent taxonomic validation.",
            "The prototype policy is not probability-calibrated.",
        ],
    }
    for name, path in artifacts.items():
        payload[name] = str(path)
        payload[f"{name}_sha256"] = _sha256(path)
    payload.update(overrides or {})
    config_path = tmp_path / "prototype.json"
    config_path.write_text(json.dumps(payload), encoding="utf-8")
    return config_path, BuildWeekPrototypeConfig.read_json(config_path)


def _target_scope() -> TaxonScope:
    context = SpeciesContext(
        scientific_name=TARGET_NAME,
        accepted_taxon_key=TARGET_KEY,
        canonical_name=TARGET_NAME,
        family="Papilionidae",
        genus="Papilio",
        family_key="gbif:5481",
        genus_key="gbif:1937907",
        species_key=TARGET_KEY,
        registry_version="pilot-registry-v1",
    )
    return TaxonScope.from_species_context(context)


def test_build_week_prototype_config_verifies_local_frozen_artifacts(
    tmp_path: Path,
) -> None:
    config_path, config = _write_config(tmp_path)

    config.verify_artifacts()

    manifest = config.to_manifest()
    assert config_path.is_file()
    assert config.classification_mode == BUILD_WEEK_TARGET_AWARE_PROTOTYPE
    assert config.storage_backend == "local"
    assert config.s3_permitted is False
    assert manifest["deployment_status"] == "prototype"
    assert manifest["invariants"]["visual_input"] == "raw_full_image"
    assert manifest["invariants"]["silent_fallback_permitted"] is False
    assert config.fingerprint.startswith("sha256:")


def test_tracked_papilio_build_week_config_is_local_and_prototype_only() -> None:
    config = BuildWeekPrototypeConfig.read_json(
        PROJECT_ROOT
        / "config/pilot/papilio_demoleus_build_week_target_aware_prototype.json"
    )

    assert config.target_accepted_taxon_key == TARGET_KEY
    assert config.target_scientific_name == TARGET_NAME
    assert config.classification_mode == BUILD_WEEK_TARGET_AWARE_PROTOTYPE
    assert config.storage_backend == "local"
    assert config.s3_permitted is False
    assert config.output_status == "prototype"
    assert config.reference_embeddings.name == (
        "prototype_reference_embeddings.parquet"
    )
    assert config.support_manifest.name == "prototype_support_manifest.parquet"
    assert all("://" not in str(path) for path, _sha in config.artifact_pins().values())


def test_build_week_prototype_config_rejects_cloud_paths(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="support_manifest must be a local path"):
        _write_config(
            tmp_path,
            overrides={"support_manifest": "s3://bucket/support.parquet"},
        )


def test_build_week_prototype_config_detects_artifact_tampering(
    tmp_path: Path,
) -> None:
    _config_path, config = _write_config(tmp_path)
    config.support_manifest.write_bytes(b"tampered")

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        config.verify_artifacts()


def test_build_week_prototype_plan_persists_explicit_contract(
    tmp_path: Path,
) -> None:
    config_path, config = _write_config(tmp_path)
    request = ProductionRunRequest(
        taxon=TARGET_NAME,
        rank="species",
        output_root=tmp_path / "runs",
        storage_backend="local",
        workstore_backend="sqlite",
        classification_mode=BUILD_WEEK_TARGET_AWARE_PROTOTYPE,
        classification_config_path=config_path,
        build_week_prototype_config=config,
        stages=(RunStage.TARGET_AWARE_SCORING,),
        dry_run=True,
    )

    plan = ProductionRunOrchestrator(
        request,
        taxon_scope=_target_scope(),
    ).plan()
    model_configs = plan.manifest.model_configs

    assert request.reference_bank_readiness == config.reference_bank_readiness
    assert request.reference_embeddings == config.reference_embeddings
    assert (
        model_configs["classification_mode"]
        == BUILD_WEEK_TARGET_AWARE_PROTOTYPE
    )
    assert model_configs["deployment_status"] == "prototype"
    assert model_configs["output_status"] == "prototype"
    assert (
        model_configs["classification_config"]["fingerprint"]
        == config.fingerprint
    )
    assert model_configs["classification_contract"]["target_always_scored"] is True
    assert (
        model_configs["classification_contract"][
            "complete_regional_candidate_union_required"
        ]
        is True
    )
    assert (
        model_configs["classification_contract"]["hierarchy_pruning_permitted"]
        is False
    )
    assert model_configs["classification_contract"]["visual_input"] == (
        "raw_full_image"
    )
    assert model_configs["classification_contract"]["silent_fallback_permitted"] is (
        False
    )
    assert model_configs["limitations"] == list(config.limitations)
    assert (
        model_configs["classification_config"]["configuration"]["diagnostic_baselines"][
            "b0_baseline_available"
        ]
        is True
    )


def test_build_week_prototype_request_fails_closed_without_matching_config(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="requires an explicit"):
        ProductionRunRequest(
            taxon=TARGET_NAME,
            classification_mode=BUILD_WEEK_TARGET_AWARE_PROTOTYPE,
            storage_backend="local",
            workstore_backend="sqlite",
            stages=(RunStage.TARGET_AWARE_SCORING,),
        )

    config_path, config = _write_config(tmp_path)
    with pytest.raises(ValueError, match="requires target_aware_scoring"):
        ProductionRunRequest(
            taxon=TARGET_NAME,
            classification_mode=BUILD_WEEK_TARGET_AWARE_PROTOTYPE,
            classification_config_path=config_path,
            build_week_prototype_config=config,
            storage_backend="local",
            workstore_backend="sqlite",
            stages=(RunStage.SCORE_BIOCLIP,),
        )

    with pytest.raises(ValueError, match="only valid"):
        ProductionRunRequest(
            taxon=TARGET_NAME,
            classification_mode=TARGET_SCOPE_OBJECT_SCREENING,
            classification_config_path=config_path,
            build_week_prototype_config=config,
        )


def test_build_week_prototype_plan_rejects_another_target(
    tmp_path: Path,
) -> None:
    config_path, config = _write_config(tmp_path)
    request = ProductionRunRequest(
        taxon=TARGET_NAME,
        rank="species",
        output_root=tmp_path / "runs",
        storage_backend="local",
        workstore_backend="sqlite",
        classification_mode=BUILD_WEEK_TARGET_AWARE_PROTOTYPE,
        classification_config_path=config_path,
        build_week_prototype_config=config,
        stages=(RunStage.TARGET_AWARE_SCORING,),
        dry_run=True,
    )
    wrong_context = SpeciesContext(
        scientific_name="Papilio machaon",
        accepted_taxon_key="gbif:1938082",
        canonical_name="Papilio machaon",
        family="Papilionidae",
        genus="Papilio",
        family_key="gbif:5481",
        genus_key="gbif:1937907",
        species_key="gbif:1938082",
        registry_version="pilot-registry-v1",
    )

    with pytest.raises(ValueError, match="does not match"):
        ProductionRunOrchestrator(
            request,
            taxon_scope=TaxonScope.from_species_context(wrong_context),
        ).plan()


def test_build_week_prototype_execution_uses_metadata_qualified_support(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path, config = _write_config(tmp_path)
    readiness = PrototypeReadinessPermit(
        status="prototype_ready_with_shortfalls",
        readiness_sha256=config.reference_bank_readiness_sha256,
        deployment_status="prototype",
        bank_status="prototype_only",
        classification_authorised=True,
        human_verification_complete=False,
        target_accepted_taxon_key=TARGET_KEY,
        target_scientific_name=TARGET_NAME,
        support_manifest_fingerprint="sha256:" + "b" * 64,
        prototype_support_count=81,
        human_verified_count=0,
        score_semantics=(
            "experimental_screening_evidence_uncalibrated_not_probability"
        ),
    )
    permit = MetadataQualifiedPrototypePermit(
        readiness=readiness,
        candidate_set_fingerprints=(config.candidate_score_evidence_sha256,),
        reference_embedding_fingerprint=config.reference_embeddings_sha256,
        model_fingerprint="sha256:" + "c" * 64,
        classifier_fingerprint=config.classifier_fingerprint,
        calibration_fingerprint=None,
        support_qualification="metadata_qualified_prototype_only",
    )
    monkeypatch.setattr(
        orchestrator_module,
        "validate_metadata_qualified_prototype_support",
        lambda _config: permit,
    )
    called = False

    def handler(_plan: object) -> StageExecutionResult:
        nonlocal called
        called = True
        return StageExecutionResult()

    request = ProductionRunRequest(
        taxon=TARGET_NAME,
        rank="species",
        output_root=tmp_path / "runs",
        storage_backend="local",
        workstore_backend="sqlite",
        classification_mode=BUILD_WEEK_TARGET_AWARE_PROTOTYPE,
        classification_config_path=config_path,
        build_week_prototype_config=config,
        stages=(RunStage.TARGET_AWARE_SCORING,),
        dry_run=False,
    )

    result = ProductionRunOrchestrator(
        request,
        taxon_scope=_target_scope(),
        stage_handlers={RunStage.TARGET_AWARE_SCORING: handler},
    ).run()

    assert called is True
    assert request.classifier_artifact is None
    assert request.calibrator_artifact is None
    assert result.manifest.status == "complete"
    assert result.manifest.metrics["calibration_fingerprint"] is None
    assert (
        result.manifest.metrics["support_qualification"]
        == "metadata_qualified_prototype_only"
    )
    assert (
        result.manifest.metrics["reference_bank_readiness_status"]
        == "prototype_ready_with_shortfalls"
    )
