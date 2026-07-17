from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import polars as pl

from biominer.bioclip.reference_embeddings import (
    load_reference_embeddings,
    reference_embeddings_artifact_fingerprint,
)
from biominer.candidates import (
    REGIONAL_CANDIDATE_SPECIES_FILE,
    validate_regional_candidate_species,
)
from biominer.ml.calibration import load_probability_calibrator
from biominer.ml.persistence import load_frozen_classifier
from biominer.references.readiness import (
    ReferenceBankReadinessPermit,
    load_reference_bank_readiness,
)
from biominer.run.stages import RunStage


SUPPORT_DEPENDENT_STAGES: frozenset[RunStage] = frozenset(
    {
        RunStage.FLICKR_DETECTION,
        RunStage.FLICKR_EMBEDDING,
        RunStage.TARGET_AWARE_SCORING,
    }
)
SUPPORT_SCORING_MODES = frozenset(
    {"provisional_nonparametric", "calibrated"}
)


class SupportDependencyError(ValueError):
    """Raised before Flickr vision when the comparison system is not ready."""

    def __init__(self, stage: RunStage, issues: tuple[str, ...]) -> None:
        if not issues:
            raise ValueError("support dependency errors require at least one issue")
        self.stage = stage
        self.issues = tuple(issues)
        details = "\n".join(f"- {issue}" for issue in self.issues)
        super().__init__(
            f"cannot start {stage.value}; support readiness dependencies failed:\n"
            f"{details}"
        )


@dataclass(frozen=True, slots=True)
class SupportDependencyPermit:
    readiness: ReferenceBankReadinessPermit
    candidate_set_fingerprints: tuple[str, ...]
    reference_embedding_fingerprint: str
    model_fingerprint: str
    classifier_fingerprint: str | None
    calibration_fingerprint: str | None
    scoring_mode: str
    score_semantics: str


def validate_support_readiness_dependencies(
    *,
    stage: RunStage,
    regional_candidates: str | Path | None,
    reference_bank_readiness: str | Path | None,
    reference_bank_readiness_sha256: str | None,
    reference_embeddings: str | Path | None,
    classifier_artifact: str | Path | None,
    calibrator_artifact: str | Path | None,
    expected_registry_version: str,
    expected_target_accepted_taxon_key: str,
    expected_model_name: str,
    scoring_mode: str = "calibrated",
) -> SupportDependencyPermit:
    """Validate the complete immutable support chain before Flickr vision."""

    if stage not in SUPPORT_DEPENDENT_STAGES:
        raise ValueError(f"stage does not require support preflight: {stage.value}")
    if scoring_mode not in SUPPORT_SCORING_MODES:
        raise ValueError(
            f"unsupported support scoring mode: {scoring_mode!r}"
        )
    calibrated = scoring_mode == "calibrated"
    issues = _missing_configuration_issues(
        regional_candidates=regional_candidates,
        reference_bank_readiness=reference_bank_readiness,
        reference_bank_readiness_sha256=reference_bank_readiness_sha256,
        reference_embeddings=reference_embeddings,
        classifier_artifact=classifier_artifact,
        calibrator_artifact=calibrator_artifact,
        calibrated=calibrated,
    )
    if issues:
        raise SupportDependencyError(stage, tuple(issues))

    assert regional_candidates is not None
    assert reference_bank_readiness is not None
    assert reference_embeddings is not None

    readiness = _load_readiness(
        reference_bank_readiness,
        expected_registry_version=expected_registry_version,
        expected_target_accepted_taxon_key=expected_target_accepted_taxon_key,
        expected_model_name=expected_model_name,
        expected_readiness_sha256=reference_bank_readiness_sha256,
        issues=issues,
    )
    _validate_scoring_permit(
        readiness,
        scoring_mode=scoring_mode,
        issues=issues,
    )
    candidates = _load_regional_candidates(regional_candidates, issues=issues)
    candidate_fingerprints = _validate_candidate_bindings(
        candidates,
        readiness=readiness,
        expected_target_accepted_taxon_key=expected_target_accepted_taxon_key,
        issues=issues,
    )
    _validate_target_support(
        readiness,
        scoring_mode=scoring_mode,
        issues=issues,
    )

    embeddings = _load_embeddings(
        reference_embeddings,
        readiness=readiness,
        issues=issues,
    )
    embedding_fingerprint, model_fingerprint, preprocessing_fingerprint = (
        _validate_embedding_bindings(
            embeddings,
            readiness=readiness,
            issues=issues,
        )
    )
    classifier_fingerprint: str | None = None
    calibration_fingerprint: str | None = None
    if calibrated:
        assert classifier_artifact is not None
        assert calibrator_artifact is not None
        classifier = _load_classifier(
            classifier_artifact,
            model_fingerprint=model_fingerprint,
            preprocessing_fingerprint=preprocessing_fingerprint,
            readiness=readiness,
            issues=issues,
        )
        classifier_fingerprint = _validate_classifier_bindings(
            classifier,
            readiness=readiness,
            reference_embedding_fingerprint=embedding_fingerprint,
            model_fingerprint=model_fingerprint,
            preprocessing_fingerprint=preprocessing_fingerprint,
            issues=issues,
        )
        calibration = _load_calibrator(
            calibrator_artifact,
            classifier_fingerprint=classifier_fingerprint,
            issues=issues,
        )
        calibration_fingerprint = _validate_calibrator_binding(
            calibration,
            classifier_fingerprint=classifier_fingerprint,
            issues=issues,
        )

    if issues:
        raise SupportDependencyError(stage, tuple(issues))
    assert readiness is not None
    assert embedding_fingerprint is not None
    assert model_fingerprint is not None
    return SupportDependencyPermit(
        readiness=readiness,
        candidate_set_fingerprints=candidate_fingerprints,
        reference_embedding_fingerprint=embedding_fingerprint,
        model_fingerprint=model_fingerprint,
        classifier_fingerprint=classifier_fingerprint,
        calibration_fingerprint=calibration_fingerprint,
        scoring_mode=scoring_mode,
        score_semantics=(
            "independently_calibrated_probability"
            if calibrated
            else "uncalibrated_similarity_and_margin_not_probability"
        ),
    )


def _missing_configuration_issues(
    *,
    regional_candidates: object,
    reference_bank_readiness: object,
    reference_bank_readiness_sha256: object,
    reference_embeddings: object,
    classifier_artifact: object,
    calibrator_artifact: object,
    calibrated: bool,
) -> list[str]:
    issues: list[str] = []
    if regional_candidates is None:
        issues.append(
            "regional candidates are not configured; run "
            "regional_candidate_generation and pass its artifact"
        )
    if reference_bank_readiness is None:
        issues.append(
            "reference readiness is not configured; run reference_readiness and "
            "pass the published readiness directory"
        )
    elif reference_bank_readiness_sha256 is None:
        issues.append(
            "reference readiness fingerprint is not configured; pin "
            "reference_bank_readiness_sha256 from the reviewed manifest"
        )
    if reference_embeddings is None:
        issues.append(
            "reference embeddings are not configured; run reference_embeddings "
            "against the current readiness permit"
        )
    if calibrated and classifier_artifact is None:
        issues.append(
            "classifier artifact is not configured; run classifier_training and "
            "pass the immutable classifier directory"
        )
    if calibrated and calibrator_artifact is None:
        issues.append(
            "calibrator artifact is not configured; run classifier_calibration and "
            "pass the immutable calibrator directory"
        )
    return issues


def _load_readiness(
    path: str | Path,
    *,
    expected_registry_version: str,
    expected_target_accepted_taxon_key: str,
    expected_model_name: str,
    expected_readiness_sha256: str | None,
    issues: list[str],
) -> ReferenceBankReadinessPermit | None:
    try:
        return load_reference_bank_readiness(
            path,
            expected_registry_version=expected_registry_version,
            expected_target_accepted_taxon_key=(
                expected_target_accepted_taxon_key
            ),
            expected_model_name=expected_model_name,
            expected_readiness_sha256=expected_readiness_sha256,
        )
    except (OSError, ValueError) as exc:
        issues.append(
            "reference readiness is blocked or invalid: "
            f"{exc}; resolve review/support blockers and republish reference_readiness"
        )
        return None


def _load_regional_candidates(
    path: str | Path,
    *,
    issues: list[str],
) -> pl.DataFrame | None:
    source = Path(path)
    if source.suffix.casefold() != ".parquet":
        source /= REGIONAL_CANDIDATE_SPECIES_FILE
    try:
        frame = pl.read_parquet(source)
        validate_regional_candidate_species(frame)
        return frame
    except (OSError, ValueError, pl.exceptions.PolarsError) as exc:
        issues.append(
            "regional candidates are missing or invalid: "
            f"{exc}; rerun regional_candidate_generation"
        )
        return None


def _validate_candidate_bindings(
    frame: pl.DataFrame | None,
    *,
    readiness: ReferenceBankReadinessPermit | None,
    expected_target_accepted_taxon_key: str,
    issues: list[str],
) -> tuple[str, ...]:
    if frame is None:
        return ()
    target_keys = set(str(value) for value in frame["accepted_taxon_key"].to_list())
    if expected_target_accepted_taxon_key not in target_keys:
        issues.append(
            "regional candidates omit the target taxon "
            f"{expected_target_accepted_taxon_key}; regenerate the candidate set"
        )
    fingerprints = tuple(
        sorted(
            set(
                str(value)
                for value in frame["candidate_set_fingerprint"].to_list()
            )
        )
    )
    if readiness is not None:
        expected = tuple(sorted(readiness.candidate_set_fingerprints))
        if fingerprints != expected:
            issues.append(
                "regional candidate fingerprints do not match reference readiness: "
                f"candidates={fingerprints!r}, readiness={expected!r}; rebuild "
                "reference acquisition and readiness from the current candidates"
            )
    return fingerprints


def _validate_target_support(
    readiness: ReferenceBankReadinessPermit | None,
    *,
    scoring_mode: str,
    issues: list[str],
) -> None:
    if readiness is None:
        return
    requirements = tuple(readiness.target_adult_requirements)
    if not requirements:
        issues.append(
            "reference readiness does not expose admitted target support minima; "
            "republish reference_readiness with the current schema"
        )
        return
    strict = readiness.reference_admission_mode == "human_verified_strict"
    support_label = "human-verified" if strict else "admitted provisional"
    for requirement in requirements:
        if requirement.observed_count >= requirement.minimum_count:
            continue
        cluster = requirement.geo_cluster_id or "all_clusters"
        issues.append(
            f"{support_label} target support is below its configured minimum: "
            f"route={requirement.route}, cluster={cluster}, "
            f"observed={requirement.observed_count}, "
            f"required={requirement.minimum_count}, "
            f"provisional={readiness.provisional_support_count}, "
            f"human_verified={readiness.human_verified_support_count}, "
            f"scoring_mode={scoring_mode}; add admitted target references before "
            "Flickr vision"
        )


def _validate_scoring_permit(
    readiness: ReferenceBankReadinessPermit | None,
    *,
    scoring_mode: str,
    issues: list[str],
) -> None:
    if readiness is None:
        return
    if scoring_mode == "provisional_nonparametric":
        if not readiness.permits_provisional_scoring:
            issues.append(
                "reference readiness does not permit provisional nonparametric "
                f"scoring: status={readiness.status}"
            )
        return
    if not readiness.permits_calibrated_scoring:
        issues.append(
            "reference readiness does not permit calibrated scoring: "
            f"status={readiness.status}, provisional="
            f"{readiness.provisional_support_count}, human_verified="
            f"{readiness.human_verified_support_count}"
        )


def _load_embeddings(
    path: str | Path,
    *,
    readiness: ReferenceBankReadinessPermit | None,
    issues: list[str],
) -> pl.DataFrame | None:
    expected: dict[str, str] = {}
    if readiness is not None:
        expected = {
            "expected_model_revision": readiness.model_revision,
            "expected_model_weights_sha256": readiness.checkpoint_sha256,
            "expected_preprocessing_version": readiness.preprocessing_version,
            "expected_preprocessing_attestation_fingerprint": (
                readiness.preprocessing_attestation_fingerprint
            ),
            "expected_model_input_fingerprint": readiness.model_input_fingerprint,
            "expected_input_contract_version": readiness.input_contract_version,
        }
    try:
        return load_reference_embeddings(path, **expected)
    except (OSError, ValueError, pl.exceptions.PolarsError) as exc:
        issues.append(
            "reference embeddings are stale, missing, or invalid: "
            f"{exc}; rebuild reference_embeddings from the current readiness permit"
        )
        return None


def _validate_embedding_bindings(
    frame: pl.DataFrame | None,
    *,
    readiness: ReferenceBankReadinessPermit | None,
    issues: list[str],
) -> tuple[str | None, str | None, str | None]:
    if frame is None:
        return None, None, None
    try:
        embedding_fingerprint = reference_embeddings_artifact_fingerprint(frame)
        model_fingerprint = _single_frame_value(frame, "model_fingerprint")
        preprocessing_fingerprint = _single_frame_value(
            frame,
            "preprocessing_fingerprint",
        )
    except (TypeError, ValueError) as exc:
        issues.append(
            "reference embeddings are stale or internally inconsistent: "
            f"{exc}; rebuild reference_embeddings"
        )
        return None, None, None
    if readiness is not None:
        expected = {
            "registry_version": readiness.registry_version,
            "reference_bank_version": readiness.reference_bank_version,
            "readiness_sha256": readiness.readiness_sha256,
            "reference_bank_fingerprint": readiness.bank_fingerprint,
            "support_manifest_fingerprint": readiness.support_manifest_fingerprint,
            "model_input_fingerprint": readiness.model_input_fingerprint,
        }
        for field, expected_value in expected.items():
            try:
                actual = _single_frame_value(frame, field)
            except ValueError as exc:
                issues.append(
                    "reference embeddings are stale or internally inconsistent: "
                    f"{exc}; rebuild reference_embeddings"
                )
                continue
            if actual != expected_value:
                issues.append(
                    "reference embeddings are stale: "
                    f"{field}={actual!r}, expected={expected_value!r}; rebuild "
                    "reference_embeddings from the current readiness permit"
                )
    return embedding_fingerprint, model_fingerprint, preprocessing_fingerprint


def _load_classifier(
    path: str | Path,
    *,
    model_fingerprint: str | None,
    preprocessing_fingerprint: str | None,
    readiness: ReferenceBankReadinessPermit | None,
    issues: list[str],
) -> Any | None:
    expected: dict[str, str] = {}
    if model_fingerprint is not None:
        expected["expected_model_fingerprint"] = model_fingerprint
    if preprocessing_fingerprint is not None:
        expected["expected_preprocessing_fingerprint"] = preprocessing_fingerprint
    if readiness is not None:
        expected["expected_reference_bank_fingerprint"] = readiness.bank_fingerprint
    try:
        return load_frozen_classifier(path, **expected)
    except (OSError, ValueError) as exc:
        issues.append(
            "classifier artifact is missing or incompatible, or reference and model "
            f"versions disagree: {exc}; rerun classifier_training"
        )
        return None


def _validate_classifier_bindings(
    classifier: Any | None,
    *,
    readiness: ReferenceBankReadinessPermit | None,
    reference_embedding_fingerprint: str | None,
    model_fingerprint: str | None,
    preprocessing_fingerprint: str | None,
    issues: list[str],
) -> str | None:
    if classifier is None:
        return None
    classifier_fingerprint = str(classifier.classifier_fingerprint)
    if readiness is not None:
        expected = {
            "reference_bank_version": readiness.reference_bank_version,
            "reference_bank_fingerprint": readiness.bank_fingerprint,
            "support_manifest_fingerprint": readiness.support_manifest_fingerprint,
        }
        for field, expected_value in expected.items():
            actual = getattr(classifier, field, None)
            if actual != expected_value:
                issues.append(
                    "classifier reference version disagrees with readiness: "
                    f"{field}={actual!r}, expected={expected_value!r}; rerun "
                    "classifier_training"
                )
    if (
        reference_embedding_fingerprint is not None
        and classifier.reference_embedding_fingerprint
        != reference_embedding_fingerprint
    ):
        issues.append(
            "classifier reference embedding fingerprint is stale: "
            f"classifier={classifier.reference_embedding_fingerprint!r}, "
            f"current={reference_embedding_fingerprint!r}; rerun classifier_training"
        )
    for field, expected_value in (
        ("model_fingerprint", model_fingerprint),
        ("preprocessing_fingerprint", preprocessing_fingerprint),
    ):
        if expected_value is None:
            continue
        actual = getattr(classifier, field, None)
        if actual != expected_value:
            issues.append(
                "reference and model versions disagree: "
                f"classifier {field}={actual!r}, embeddings={expected_value!r}; "
                "rebuild embeddings and classifier with one pinned model identity"
            )
    return classifier_fingerprint


def _load_calibrator(
    path: str | Path,
    *,
    classifier_fingerprint: str | None,
    issues: list[str],
) -> Any | None:
    expected = (
        {"expected_classifier_fingerprint": classifier_fingerprint}
        if classifier_fingerprint is not None
        else {}
    )
    try:
        return load_probability_calibrator(path, **expected)
    except (OSError, ValueError) as exc:
        issues.append(
            "calibrator artifact is missing or incompatible: "
            f"{exc}; rerun classifier_calibration from the current classifier"
        )
        return None


def _validate_calibrator_binding(
    calibration: Any | None,
    *,
    classifier_fingerprint: str | None,
    issues: list[str],
) -> str | None:
    if calibration is None:
        return None
    calibrator = calibration.calibrator
    if (
        classifier_fingerprint is not None
        and calibrator.classifier_fingerprint != classifier_fingerprint
    ):
        issues.append(
            "calibrator classifier fingerprint is stale: "
            f"calibrator={calibrator.classifier_fingerprint!r}, "
            f"classifier={classifier_fingerprint!r}; rerun classifier_calibration"
        )
    return str(calibrator.calibration_fingerprint)


def _single_frame_value(frame: pl.DataFrame, field: str) -> str:
    if field not in frame.columns:
        raise ValueError(f"reference embeddings omit {field}")
    values = frame[field].unique().to_list()
    if len(values) != 1 or values[0] is None or not str(values[0]).strip():
        raise ValueError(f"reference embeddings contain mixed or blank {field}")
    return str(values[0])


__all__ = [
    "SUPPORT_DEPENDENT_STAGES",
    "SUPPORT_SCORING_MODES",
    "SupportDependencyError",
    "SupportDependencyPermit",
    "validate_support_readiness_dependencies",
]
