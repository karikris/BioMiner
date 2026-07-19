from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace

import polars as pl
import pytest

import biominer.run.orchestrator as orchestrator_module
import biominer.run.support_dependencies as dependency_module
from biominer.run import (
    ProductionRunOrchestrator,
    ProductionRunRequest,
    RunStage,
    StageExecutionResult,
    TaxonScope,
)
from biominer.run.support_dependencies import (
    SupportDependencyError,
    validate_support_readiness_dependencies,
)
from biominer.species.context import SpeciesContext


@dataclass(frozen=True)
class _TargetRequirementFixture:
    accepted_taxon_key: str
    route: str
    geo_cluster_id: str | None
    minimum_count: int
    observed_count: int


@dataclass(frozen=True)
class _ReadinessPermitFixture:
    status: str
    registry_version: str
    reference_bank_version: str
    target_accepted_taxon_key: str
    candidate_set_fingerprints: tuple[str, ...]
    target_adult_requirements: tuple[_TargetRequirementFixture, ...]
    bank_fingerprint: str
    support_manifest_fingerprint: str
    model_input_fingerprint: str
    model_name: str
    model_revision: str
    checkpoint_sha256: str
    preprocessing_version: str
    preprocessing_attestation_fingerprint: str
    input_contract_version: str
    readiness_sha256: str
    reference_admission_mode: str
    provisional_support_count: int
    human_verified_support_count: int
    permits_provisional_scoring: bool
    permits_calibrated_scoring: bool


def test_support_dependency_validation_accepts_one_consistent_chain(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    permit = _readiness_permit()
    candidates = pl.DataFrame(
        {
            "accepted_taxon_key": [permit.target_accepted_taxon_key],
            "candidate_set_fingerprint": [_sha("candidate-set")],
        }
    )
    embeddings = _embedding_frame(permit)
    classifier = _classifier(permit)
    calibration = SimpleNamespace(
        calibrator=SimpleNamespace(
            calibration_fingerprint=_sha("calibration"),
            classifier_fingerprint=classifier.classifier_fingerprint,
        )
    )
    monkeypatch.setattr(dependency_module.pl, "read_parquet", lambda _path: candidates)
    monkeypatch.setattr(
        dependency_module,
        "validate_regional_candidate_species",
        lambda _frame: None,
    )
    monkeypatch.setattr(
        dependency_module,
        "load_reference_bank_readiness",
        lambda _path, **_expected: permit,
    )
    monkeypatch.setattr(
        dependency_module,
        "load_reference_embeddings",
        lambda _path, **_expected: embeddings,
    )
    monkeypatch.setattr(
        dependency_module,
        "reference_embeddings_artifact_fingerprint",
        lambda _frame: _sha("reference-embeddings"),
    )
    monkeypatch.setattr(
        dependency_module,
        "load_frozen_classifier",
        lambda _path, **_expected: classifier,
    )
    monkeypatch.setattr(
        dependency_module,
        "load_probability_calibrator",
        lambda _path, **_expected: calibration,
    )

    result = validate_support_readiness_dependencies(
        stage=RunStage.FLICKR_DETECTION,
        regional_candidates=tmp_path / "regional_candidate_species.parquet",
        reference_bank_readiness=tmp_path / "readiness",
        reference_bank_readiness_sha256=permit.readiness_sha256,
        reference_embeddings=tmp_path / "reference_embeddings.parquet",
        classifier_artifact=tmp_path / "classifier",
        calibrator_artifact=tmp_path / "calibrator",
        expected_registry_version=permit.registry_version,
        expected_target_accepted_taxon_key=permit.target_accepted_taxon_key,
        expected_model_name=permit.model_name,
    )

    assert result.readiness is permit
    assert result.candidate_set_fingerprints == (_sha("candidate-set"),)
    assert result.reference_embedding_fingerprint == _sha("reference-embeddings")
    assert result.model_fingerprint == _sha("foundation-model")
    assert result.classifier_fingerprint == _sha("classifier")
    assert result.calibration_fingerprint == _sha("calibration")
    assert result.scoring_mode == "calibrated"
    assert result.score_semantics == "independently_calibrated_probability"


def test_provisional_nonparametric_mode_does_not_require_trained_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    permit = replace(
        _readiness_permit(),
        status="ready_provisional",
        reference_admission_mode="adaptive_gbif_fast_start",
        provisional_support_count=5,
        human_verified_support_count=0,
        permits_calibrated_scoring=False,
    )
    candidates = pl.DataFrame(
        {
            "accepted_taxon_key": [permit.target_accepted_taxon_key],
            "candidate_set_fingerprint": [_sha("candidate-set")],
        }
    )
    monkeypatch.setattr(dependency_module.pl, "read_parquet", lambda _path: candidates)
    monkeypatch.setattr(
        dependency_module,
        "validate_regional_candidate_species",
        lambda _frame: None,
    )
    monkeypatch.setattr(
        dependency_module,
        "load_reference_bank_readiness",
        lambda _path, **_expected: permit,
    )
    monkeypatch.setattr(
        dependency_module,
        "load_reference_embeddings",
        lambda _path, **_expected: _embedding_frame(permit),
    )
    monkeypatch.setattr(
        dependency_module,
        "reference_embeddings_artifact_fingerprint",
        lambda _frame: _sha("reference-embeddings"),
    )
    monkeypatch.setattr(
        dependency_module,
        "load_frozen_classifier",
        lambda *_args, **_kwargs: pytest.fail("classifier must not be loaded"),
    )
    monkeypatch.setattr(
        dependency_module,
        "load_probability_calibrator",
        lambda *_args, **_kwargs: pytest.fail("calibrator must not be loaded"),
    )

    result = validate_support_readiness_dependencies(
        stage=RunStage.DYNAMIC_POOL_SCORING,
        regional_candidates=tmp_path / "regional_candidate_species.parquet",
        reference_bank_readiness=tmp_path / "readiness",
        reference_bank_readiness_sha256=permit.readiness_sha256,
        reference_embeddings=tmp_path / "reference_embeddings.parquet",
        classifier_artifact=None,
        calibrator_artifact=None,
        expected_registry_version=permit.registry_version,
        expected_target_accepted_taxon_key=permit.target_accepted_taxon_key,
        expected_model_name=permit.model_name,
        scoring_mode="provisional_nonparametric",
    )

    assert result.classifier_fingerprint is None
    assert result.calibration_fingerprint is None
    assert result.score_semantics == (
        "uncalibrated_similarity_and_margin_not_probability"
    )


def test_support_dependency_validation_reports_all_missing_configuration() -> None:
    with pytest.raises(SupportDependencyError) as captured:
        validate_support_readiness_dependencies(
            stage=RunStage.DYNAMIC_POOL_SCORING,
            regional_candidates=None,
            reference_bank_readiness=None,
            reference_bank_readiness_sha256=None,
            reference_embeddings=None,
            classifier_artifact=None,
            calibrator_artifact=None,
            expected_registry_version="registry-v1",
            expected_target_accepted_taxon_key="gbif:1941315",
            expected_model_name="imageomics/bioclip-2.5-vith14",
        )

    message = str(captured.value)
    assert "cannot start dynamic_pool_scoring" in message
    assert "regional candidates are not configured" in message
    assert "reference readiness is not configured" in message
    assert "reference embeddings are not configured" in message
    assert "classifier artifact is not configured" in message
    assert "calibrator artifact is not configured" in message


def test_support_dependency_validation_requires_readiness_digest_pin(
    tmp_path: Path,
) -> None:
    with pytest.raises(
        SupportDependencyError,
        match="reference readiness fingerprint is not configured",
    ):
        validate_support_readiness_dependencies(
            stage=RunStage.FLICKR_EMBEDDING,
            regional_candidates=tmp_path / "regional_candidate_species.parquet",
            reference_bank_readiness=tmp_path / "readiness",
            reference_bank_readiness_sha256=None,
            reference_embeddings=tmp_path / "reference_embeddings.parquet",
            classifier_artifact=tmp_path / "classifier",
            calibrator_artifact=tmp_path / "calibrator",
            expected_registry_version="registry-v1",
            expected_target_accepted_taxon_key="gbif:1941315",
            expected_model_name="imageomics/bioclip-2.5-vith14",
        )


@pytest.mark.parametrize(
    ("mutation", "expected_message"),
    (
        ("blocked_readiness", "reference readiness is blocked or invalid"),
        ("target_shortfall", "human-verified target support is below its configured minimum"),
        ("candidate_mismatch", "regional candidate fingerprints do not match reference readiness"),
        ("stale_embeddings", "reference embeddings are stale"),
        ("missing_classifier", "classifier artifact is missing or incompatible"),
        ("missing_calibrator", "calibrator artifact is missing or incompatible"),
        ("model_version_mismatch", "reference and model versions disagree"),
    ),
)
def test_support_dependency_validation_rejects_broken_chain(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
    expected_message: str,
) -> None:
    permit = _readiness_permit(
        observed_target_support=2 if mutation == "target_shortfall" else 5
    )
    candidate_fingerprint = (
        _sha("different-candidates")
        if mutation == "candidate_mismatch"
        else _sha("candidate-set")
    )
    candidates = pl.DataFrame(
        {
            "accepted_taxon_key": [permit.target_accepted_taxon_key],
            "candidate_set_fingerprint": [candidate_fingerprint],
        }
    )
    embeddings = _embedding_frame(
        permit,
        bank_fingerprint=(
            _sha("stale-bank")
            if mutation == "stale_embeddings"
            else permit.bank_fingerprint
        ),
    )
    classifier = _classifier(
        permit,
        model_fingerprint=(
            _sha("different-model")
            if mutation == "model_version_mismatch"
            else _sha("foundation-model")
        ),
    )
    calibration = SimpleNamespace(
        calibrator=SimpleNamespace(
            calibration_fingerprint=_sha("calibration"),
            classifier_fingerprint=classifier.classifier_fingerprint,
        )
    )
    monkeypatch.setattr(dependency_module.pl, "read_parquet", lambda _path: candidates)
    monkeypatch.setattr(
        dependency_module,
        "validate_regional_candidate_species",
        lambda _frame: None,
    )

    def load_readiness(_path: object, **_expected: object) -> object:
        if mutation == "blocked_readiness":
            raise ValueError("status=blocked_missing_target_support")
        return permit

    def load_classifier(_path: object, **_expected: object) -> object:
        if mutation == "missing_classifier":
            raise ValueError("classifier artifact path must be a real directory")
        return classifier

    def load_calibrator(_path: object, **_expected: object) -> object:
        if mutation == "missing_calibrator":
            raise ValueError("calibration artifact path must be a real directory")
        return calibration

    monkeypatch.setattr(
        dependency_module,
        "load_reference_bank_readiness",
        load_readiness,
    )
    monkeypatch.setattr(
        dependency_module,
        "load_reference_embeddings",
        lambda _path, **_expected: embeddings,
    )
    monkeypatch.setattr(
        dependency_module,
        "reference_embeddings_artifact_fingerprint",
        lambda _frame: _sha("reference-embeddings"),
    )
    monkeypatch.setattr(dependency_module, "load_frozen_classifier", load_classifier)
    monkeypatch.setattr(
        dependency_module,
        "load_probability_calibrator",
        load_calibrator,
    )

    with pytest.raises(SupportDependencyError, match=expected_message):
        validate_support_readiness_dependencies(
            stage=RunStage.FLICKR_DETECTION,
            regional_candidates=tmp_path / "regional_candidate_species.parquet",
            reference_bank_readiness=tmp_path / "readiness",
            reference_bank_readiness_sha256=permit.readiness_sha256,
            reference_embeddings=tmp_path / "reference_embeddings.parquet",
            classifier_artifact=tmp_path / "classifier",
            calibrator_artifact=tmp_path / "calibrator",
            expected_registry_version=permit.registry_version,
            expected_target_accepted_taxon_key=permit.target_accepted_taxon_key,
            expected_model_name=permit.model_name,
        )


def test_orchestrator_refuses_support_dependent_stage_before_handler(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    called = False

    def fail_validation(**kwargs: object) -> object:
        assert kwargs["stage"] is RunStage.FLICKR_DETECTION
        raise SupportDependencyError(
            RunStage.FLICKR_DETECTION,
            ("regional candidates are missing; run regional_candidate_generation",),
        )

    def handler(_plan: object) -> StageExecutionResult:
        nonlocal called
        called = True
        return StageExecutionResult()

    monkeypatch.setattr(
        orchestrator_module,
        "validate_support_readiness_dependencies",
        fail_validation,
    )
    request = _request(tmp_path, stages=(RunStage.FLICKR_DETECTION,))

    with pytest.raises(SupportDependencyError, match="run regional_candidate_generation"):
        ProductionRunOrchestrator(
            request,
            taxon_scope=_taxon_scope(),
            stage_handlers={RunStage.FLICKR_DETECTION: handler},
        ).run()
    assert called is False


def test_orchestrator_reuses_one_dependency_preflight_for_scoring_stages(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    validations = 0
    calls: list[RunStage] = []
    dependency_permit = SimpleNamespace(
        readiness=_readiness_permit(),
        candidate_set_fingerprints=(_sha("candidate-set"),),
        reference_embedding_fingerprint=_sha("reference-embeddings"),
        model_fingerprint=_sha("foundation-model"),
        classifier_fingerprint=_sha("classifier"),
        calibration_fingerprint=_sha("calibration"),
    )

    def validate(**_kwargs: object) -> object:
        nonlocal validations
        validations += 1
        return dependency_permit

    def handler(stage: RunStage) -> object:
        def run(_plan: object) -> StageExecutionResult:
            calls.append(stage)
            return StageExecutionResult()

        return run

    monkeypatch.setattr(
        orchestrator_module,
        "validate_support_readiness_dependencies",
        validate,
    )
    stages = (
        RunStage.FLICKR_DETECTION,
        RunStage.FLICKR_EMBEDDING,
        RunStage.DYNAMIC_POOL_SCORING,
    )
    result = ProductionRunOrchestrator(
        _request(tmp_path, stages=stages),
        taxon_scope=_taxon_scope(),
        stage_handlers={stage: handler(stage) for stage in stages},
    ).run()

    assert result.manifest.status == "complete"
    assert validations == 1
    assert calls == list(stages)
    assert (
        result.manifest.model_configs["support_dependencies"]["validation_status"]
        == "validated"
    )
    assert result.manifest.metrics["classifier_fingerprint"] == _sha("classifier")


def _request(tmp_path: Path, *, stages: tuple[RunStage, ...]) -> ProductionRunRequest:
    return ProductionRunRequest(
        taxon="Papilio demoleus",
        rank="species",
        output_root=tmp_path,
        run_id="support-dependency-test",
        stages=stages,
        regional_candidates=tmp_path / "regional_candidate_species.parquet",
        reference_bank_readiness=tmp_path / "readiness",
        reference_bank_readiness_sha256=_sha("readiness"),
        reference_embeddings=tmp_path / "reference_embeddings.parquet",
        classifier_artifact=tmp_path / "classifier",
        calibrator_artifact=tmp_path / "calibrator",
    )


def _readiness_permit(
    *,
    observed_target_support: int = 5,
) -> _ReadinessPermitFixture:
    return _ReadinessPermitFixture(
        status="ready",
        registry_version="registry-v1",
        reference_bank_version="reference-bank-v1",
        target_accepted_taxon_key="gbif:1941315",
        candidate_set_fingerprints=(_sha("candidate-set"),),
        target_adult_requirements=(
            _TargetRequirementFixture(
                accepted_taxon_key="gbif:1941315",
                route="adult_field",
                geo_cluster_id=None,
                minimum_count=5,
                observed_count=observed_target_support,
            ),
        ),
        bank_fingerprint=_sha("reference-bank"),
        support_manifest_fingerprint=_sha("support-manifest"),
        model_input_fingerprint=_sha("model-input"),
        model_name="imageomics/bioclip-2.5-vith14",
        model_revision="revision-v1",
        checkpoint_sha256=_sha("weights"),
        preprocessing_version="preprocessing-v1",
        preprocessing_attestation_fingerprint=_sha("preprocessing-attestation"),
        input_contract_version="input-contract-v1",
        readiness_sha256=_sha("readiness"),
        reference_admission_mode="human_verified_strict",
        provisional_support_count=0,
        human_verified_support_count=5,
        permits_provisional_scoring=True,
        permits_calibrated_scoring=True,
    )


def _embedding_frame(
    permit: _ReadinessPermitFixture,
    *,
    bank_fingerprint: str | None = None,
) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "registry_version": [permit.registry_version],
            "reference_bank_version": [permit.reference_bank_version],
            "readiness_sha256": [permit.readiness_sha256],
            "reference_bank_fingerprint": [
                bank_fingerprint or permit.bank_fingerprint
            ],
            "support_manifest_fingerprint": [permit.support_manifest_fingerprint],
            "model_input_fingerprint": [permit.model_input_fingerprint],
            "model_fingerprint": [_sha("foundation-model")],
            "preprocessing_fingerprint": [_sha("preprocessing")],
        }
    )


def _classifier(
    permit: _ReadinessPermitFixture,
    *,
    model_fingerprint: str = "",
) -> SimpleNamespace:
    return SimpleNamespace(
        classifier_fingerprint=_sha("classifier"),
        reference_bank_version=permit.reference_bank_version,
        reference_bank_fingerprint=permit.bank_fingerprint,
        support_manifest_fingerprint=permit.support_manifest_fingerprint,
        reference_embedding_fingerprint=_sha("reference-embeddings"),
        model_fingerprint=model_fingerprint or _sha("foundation-model"),
        preprocessing_fingerprint=_sha("preprocessing"),
    )


def _taxon_scope() -> TaxonScope:
    context = SpeciesContext(
        scientific_name="Papilio demoleus",
        accepted_taxon_key="gbif:1941315",
        canonical_name="Papilio demoleus",
        family="Papilionidae",
        genus="Papilio",
        family_key="gbif:9417",
        genus_key="gbif:1920490",
        species_key="gbif:1941315",
        registry_version="registry-v1",
    )
    return TaxonScope.from_species_context(context)


def _sha(seed: str) -> str:
    import hashlib

    return "sha256:" + hashlib.sha256(seed.encode("utf-8")).hexdigest()
